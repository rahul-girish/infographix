"""Offline PDF content extraction and document-structure identity features for elements.parquet.

Extracts leak-free content features locally from SciPostLayout v2 poster PDFs using PyMuPDF:
  - word_count, char_count, has_text_length: spatially matched PDF words (>= 1 word, capturing 1-word headings).
  - intrinsic_aspect_ratio, has_aspect_ratio: spatially matched embedded raster image pixel aspect ratio
    (width / height of the embedded image asset itself, never the annotation bounding box).
  - in_cat_rank_norm, section_group_norm: leak-free document-order identity within each category and section,
    matching the order/grouping a user provides in a PosterRequest.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None


PDF_CONTENT_COLUMNS: tuple[str, ...] = (
    "word_count",
    "char_count",
    "has_text_length",
    "intrinsic_aspect_ratio",
    "has_aspect_ratio",
    "in_cat_rank_norm",
    "section_group_norm",
)


def build_poster_id_to_pdf_map(
    pdf_dir: Path,
    png_dir: Path | None = None,
) -> dict[int, Path]:
    """Build a mapping from poster_id to local poster PDF path using COCO split JSONs."""
    pdf_dir = Path(pdf_dir)
    coco_dir = Path(png_dir) if png_dir is not None else pdf_dir.parent / "png"
    id_to_pdf: dict[int, Path] = {}

    if not pdf_dir.exists() or not coco_dir.exists():
        return id_to_pdf

    for split in ("train", "dev", "test"):
        json_path = coco_dir / f"{split}.json"
        if not json_path.exists():
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        for img in meta.get("images", []):
            pid = int(img["id"])
            stem = Path(str(img.get("file_name", ""))).stem
            cand = pdf_dir / split / f"{stem}.pdf"
            if cand.exists():
                id_to_pdf[pid] = cand

    return id_to_pdf


def _extract_poster_worker(
    task: tuple[str | None, list[tuple[int, str, float, float, float, float]]],
) -> list[tuple[int, float, float, bool, float, bool]]:
    """Extract word/char counts and intrinsic figure aspect ratios for one poster PDF."""
    pdf_path_str, elem_specs = task
    if pymupdf is None or not pdf_path_str:
        return [(eid, 0.0, 0.0, False, 1.0, False) for eid, *_ in elem_specs]

    pymupdf.TOOLS.mupdf_display_errors(False)
    words_norm: list[tuple[float, float, str]] = []
    imgs_norm: list[tuple[float, float, float, float, float, float]] = []

    try:
        doc = pymupdf.open(pdf_path_str)
        if len(doc) > 0:
            page = doc[0]
            pw = float(page.rect.width)
            ph = float(page.rect.height)
            if pw > 0 and ph > 0:
                for w in page.get_text("words"):
                    txt = str(w[4]).strip()
                    if txt:
                        wcx = (float(w[0]) + float(w[2])) / (2.0 * pw)
                        wcy = (float(w[1]) + float(w[3])) / (2.0 * ph)
                        words_norm.append((wcx, wcy, txt))

                for entry_type, bbox in page.get_bboxlog():
                    if "image" not in str(entry_type):
                        continue
                    bx0, by0, bx1, by1 = bbox
                    bw = float(bx1) - float(bx0)
                    bh = float(by1) - float(by0)
                    if bw <= 5.0 or bh <= 5.0:
                        continue
                    nx0, ny0, nx1, ny1 = float(bx0) / pw, float(by0) / ph, float(bx1) / pw, float(by1) / ph
                    r_area = max(0.0, nx1 - nx0) * max(0.0, ny1 - ny0)
                    # Ignore decorative full-page backgrounds and tiny icons
                    if 0.0005 < r_area < 0.85:
                        imgs_norm.append((nx0, ny0, nx1, ny1, bw, bh))
        doc.close()
    except Exception:
        return [(eid, 0.0, 0.0, False, 1.0, False) for eid, *_ in elem_specs]

    results: list[tuple[int, float, float, bool, float, bool]] = []
    tol = 0.004

    for eid, cslug, ex0, ey0, ex1, ey1 in elem_specs:
        if cslug in ("figure", "logo"):
            best_ov = 0.0
            best_aspect = 1.0
            ea = max(1e-6, (ex1 - ex0) * (ey1 - ey0))
            for nx0, ny0, nx1, ny1, bw, bh in imgs_norm:
                ix0 = max(ex0, nx0)
                iy0 = max(ey0, ny0)
                ix1 = min(ex1, nx1)
                iy1 = min(ey1, ny1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    ia = max(1e-6, (nx1 - nx0) * (ny1 - ny0))
                    ov = inter / min(ea, ia)
                    if ov > best_ov:
                        best_ov = ov
                        best_aspect = float(np.clip(bw / max(1e-3, bh), 0.25, 4.0))
            if best_ov > 0.25:
                results.append((eid, 0.0, 0.0, False, best_aspect, True))
            else:
                results.append((eid, 0.0, 0.0, False, 1.0, False))
        else:
            matched = [
                txt
                for wcx, wcy, txt in words_norm
                if (ex0 - tol <= wcx <= ex1 + tol and ey0 - tol <= wcy <= ey1 + tol)
            ]
            if matched:
                w_cnt = float(len(matched))
                c_cnt = float(sum(len(t) for t in matched) + max(0, len(matched) - 1))
                results.append((eid, w_cnt, c_cnt, True, 1.0, False))
            else:
                results.append((eid, 0.0, 0.0, False, 1.0, False))

    return results


def compute_document_structure_features(
    elements_df: pd.DataFrame,
    posters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute leak-free document-order identity features (`in_cat_rank_norm`, `section_group_norm`)."""
    col_lookup: dict[int, int] = {}
    if "estimated_column_count" in posters_df.columns:
        for pid, k_raw in zip(posters_df["poster_id"], posters_df["estimated_column_count"]):
            k = 3 if pd.isna(k_raw) else max(1, min(5, int(round(float(k_raw)))))
            col_lookup[int(pid)] = k

    in_cat_rank = np.full(len(elements_df), 0.5, dtype=np.float32)
    section_group = np.zeros(len(elements_df), dtype=np.float32)

    for pid, grp in elements_df.groupby("poster_id", sort=False):
        k_cols = col_lookup.get(int(pid), 3)
        idxs = grp.index.to_numpy()
        slugs = grp["category_slug"].astype(str).to_numpy()
        cx = grp["cx"].to_numpy(dtype=float)
        cy = grp["cy"].to_numpy(dtype=float)
        y0 = grp["y0"].to_numpy(dtype=float)
        x0 = grp["x0"].to_numpy(dtype=float)

        # Header elements (title / author_info near top) come first; body ordered column-major
        is_header = np.isin(slugs, ["title", "author_info"]) & (cy < 0.30)
        col_idx = np.where(is_header, -1, np.clip((cx * k_cols).astype(int), 0, k_cols - 1))
        order = np.lexsort((x0, y0, col_idx))

        # Section group assignment in document reading order
        total_sections = int(np.sum(slugs == "section"))
        sec_denom = float(max(1, total_sections))
        cur_sec = 0
        cat_positions: dict[str, list[int]] = {}

        for pos_in_grp in order:
            s = slugs[pos_in_grp]
            if s == "section":
                cur_sec += 1
            global_idx = idxs[pos_in_grp]
            section_group[global_idx] = 0.0 if is_header[pos_in_grp] else float(cur_sec) / sec_denom
            cat_positions.setdefault(s, []).append(global_idx)

        for cat_idxs in cat_positions.values():
            n_c = len(cat_idxs)
            if n_c == 1:
                in_cat_rank[cat_idxs[0]] = 0.5
            else:
                for r_i, global_idx in enumerate(cat_idxs):
                    in_cat_rank[global_idx] = float(r_i) / float(n_c - 1)

    out = elements_df.copy()
    out["in_cat_rank_norm"] = in_cat_rank
    out["section_group_norm"] = section_group
    return out


def _pipe_worker_loop(conn: Any) -> None:
    """Dedicated single-pipe worker: signals READY after imports, then serves tasks."""
    try:
        conn.send("READY")
        while True:
            task = conn.recv()
            if task is None:
                break
            res = _extract_poster_worker(task)
            conn.send(res)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_timeout_safe_pool(
    tasks: list[tuple[str | None, list[tuple[int, str, float, float, float, float]]]],
    n_workers: int,
    per_pdf_timeout_s: float = 4.0,
) -> tuple[list[tuple[int, float, float, bool, float, bool]], int]:
    """Run _extract_poster_worker using isolated per-worker Pipes (lock-free, kill-safe)."""
    ctx = mp.get_context("spawn")
    total = len(tasks)
    next_task_idx = 0
    completed = 0
    timed_out = 0
    records: list[tuple[int, float, float, bool, float, bool]] = []
    t0 = time.perf_counter()

    # Each slot: {"proc": Process, "conn": Connection, "task_idx": int | None, "start": float, "done_cnt": int}
    slots: list[dict[str, Any]] = []

    def _make_slot() -> dict[str, Any]:
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=_pipe_worker_loop, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        # Wait for worker imports to finish before starting per-PDF timeout clock
        if mp_wait([parent_conn], timeout=30.0):
            try:
                _ = parent_conn.recv()
            except Exception:
                pass
        return {"proc": proc, "conn": parent_conn, "task_idx": None, "start": 0.0, "done_cnt": 0}

    def _close_slot(slot: dict[str, Any], force_kill: bool = False) -> None:
        try:
            if not force_kill and slot["proc"].is_alive():
                slot["conn"].send(None)
        except Exception:
            pass
        try:
            slot["conn"].close()
        except Exception:
            pass
        if force_kill and slot["proc"].is_alive():
            slot["proc"].kill()
        slot["proc"].join(timeout=1.0)
        if slot["proc"].is_alive():
            slot["proc"].kill()
            slot["proc"].join(timeout=0.5)

    def _assign(slot: dict[str, Any]) -> None:
        nonlocal next_task_idx
        if next_task_idx < total:
            slot["task_idx"] = next_task_idx
            slot["start"] = time.monotonic()
            slot["conn"].send(tasks[next_task_idx])
            next_task_idx += 1
        else:
            slot["task_idx"] = None

    def _log_progress() -> None:
        if completed % 500 == 0 or completed == total:
            elapsed = time.perf_counter() - t0
            rate = completed / max(elapsed, 1e-6)
            print(
                f"  Processed {completed:,}/{total:,} poster PDFs "
                f"({elapsed:.1f}s, {rate:.1f} pdf/s, timeouts={timed_out})...",
                flush=True,
            )

    try:
        for _ in range(n_workers):
            s = _make_slot()
            _assign(s)
            slots.append(s)

        while completed < total:
            busy_slots = [s for s in slots if s["task_idx"] is not None]
            conns = [s["conn"] for s in busy_slots]
            ready = set(mp_wait(conns, timeout=0.25)) if conns else set()

            now = time.monotonic()
            for i, s in enumerate(slots):
                t_idx = s["task_idx"]
                if t_idx is None:
                    continue

                if s["conn"] in ready:
                    try:
                        res = s["conn"].recv()
                        records.extend(res)
                    except (EOFError, BrokenPipeError, OSError):
                        _, elem_specs = tasks[t_idx]
                        records.extend([(eid, 0.0, 0.0, False, 1.0, False) for eid, *_ in elem_specs])
                        timed_out += 1
                        _close_slot(s, force_kill=True)
                        s = _make_slot()
                        slots[i] = s
                    completed += 1
                    s["done_cnt"] += 1
                    _log_progress()
                    # Recycle worker periodically to keep MuPDF memory flat
                    if s["done_cnt"] >= 400 and next_task_idx < total:
                        _close_slot(s, force_kill=False)
                        s = _make_slot()
                        slots[i] = s
                    _assign(s)
                elif (now - s["start"] > per_pdf_timeout_s) or (not s["proc"].is_alive()):
                    _close_slot(s, force_kill=True)
                    _, elem_specs = tasks[t_idx]
                    records.extend([(eid, 0.0, 0.0, False, 1.0, False) for eid, *_ in elem_specs])
                    completed += 1
                    timed_out += 1
                    _log_progress()
                    s = _make_slot()
                    slots[i] = s
                    _assign(s)
    finally:
        for s in slots:
            _close_slot(s, force_kill=True)

    return records, timed_out


def enrich_elements_with_pdf_content(
    elements_df: pd.DataFrame,
    posters_df: pd.DataFrame,
    pdf_dir: Path,
    png_dir: Path | None = None,
    max_workers: int | None = None,
    per_pdf_timeout_s: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run parallel PyMuPDF extraction across all posters and attach content + structure columns."""
    if pymupdf is None:
        raise ImportError("pymupdf is required for offline PDF content extraction.")

    id_to_pdf = build_poster_id_to_pdf_map(pdf_dir=pdf_dir, png_dir=png_dir)
    tasks: list[tuple[str | None, list[tuple[int, str, float, float, float, float]]]] = []

    for pid, grp in elements_df.groupby("poster_id", sort=False):
        pdf_p = id_to_pdf.get(int(pid))
        specs = [
            (
                int(eid),
                str(slug).lower(),
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            )
            for eid, slug, x0, y0, x1, y1 in zip(
                grp["element_id"],
                grp["category_slug"],
                grp["x0"],
                grp["y0"],
                grp["x1"],
                grp["y1"],
            )
        ]
        tasks.append((str(pdf_p) if pdf_p is not None else None, specs))

    n_workers = max_workers or max(1, os.cpu_count() or 1)
    records: list[tuple[int, float, float, bool, float, bool]] = []
    timed_out_pdfs = 0

    if n_workers == 1 or len(tasks) <= 8:
        for task in tasks:
            records.extend(_extract_poster_worker(task))
    else:
        records, timed_out_pdfs = _run_timeout_safe_pool(
            tasks=tasks,
            n_workers=n_workers,
            per_pdf_timeout_s=per_pdf_timeout_s,
        )

    pdf_df = pd.DataFrame(
        records,
        columns=[
            "element_id",
            "word_count",
            "char_count",
            "has_text_length",
            "intrinsic_aspect_ratio",
            "has_aspect_ratio",
        ],
    )

    drop_existing = [c for c in PDF_CONTENT_COLUMNS if c in elements_df.columns]
    base_df = elements_df.drop(columns=drop_existing)
    merged = base_df.merge(pdf_df, on="element_id", how="left")
    merged["word_count"] = merged["word_count"].fillna(0.0).astype(np.float32)
    merged["char_count"] = merged["char_count"].fillna(0.0).astype(np.float32)
    merged["has_text_length"] = merged["has_text_length"].fillna(False).astype(bool)
    merged["intrinsic_aspect_ratio"] = merged["intrinsic_aspect_ratio"].fillna(1.0).astype(np.float32)
    merged["has_aspect_ratio"] = merged["has_aspect_ratio"].fillna(False).astype(bool)

    enriched = compute_document_structure_features(merged, posters_df)

    non_fig = enriched[~enriched["category_slug"].isin(["figure", "logo"])]
    figs = enriched[enriched["category_slug"].isin(["figure", "logo"])]
    stats = {
        "posters_total": int(elements_df["poster_id"].nunique()),
        "pdfs_found": int(sum(1 for t in tasks if t[0] is not None)),
        "pdfs_timed_out": int(timed_out_pdfs),
        "elements_total": int(len(enriched)),
        "non_figure_elements": int(len(non_fig)),
        "non_figure_with_text": int(non_fig["has_text_length"].sum()),
        "non_figure_text_coverage": round(float(non_fig["has_text_length"].mean()), 4) if len(non_fig) else 0.0,
        "figure_elements": int(len(figs)),
        "figure_with_intrinsic_aspect": int(figs["has_aspect_ratio"].sum()),
        "figure_aspect_coverage": round(float(figs["has_aspect_ratio"].mean()), 4) if len(figs) else 0.0,
    }
    return enriched, stats

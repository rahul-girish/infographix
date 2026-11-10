"""Dataset Construction and Preprocessing for Neural Layout Generation (Stage A).

Implements strict anti-leakage synthetic training pair construction from SciPostLayout v2:
  1. Genuinely extracted PDF text via `pdftotext -bbox-layout` mapped spatially to element bounding boxes.
  2. When extraction is unavailable, content length is explicitly marked unavailable:
     `has_text_length = False`, `word_count = 0.0`, `text_source = "unavailable"`.
     NO estimation of word/character count from ground-truth bounding box area or dimensions.
  3. Figure aspect ratios come exclusively from intrinsic image assets via `pdfimages -list`.
     If intrinsic dimensions cannot be recovered, `has_aspect_ratio = False`, `aspect_source = "unavailable"`.
     NO derivation of aspect ratio from annotation bounding boxes.
  4. Input sequence order is decoupled from target reading order via deterministic pseudo-random shuffling.
  5. Target column count, target occupancy, target bboxes, and target reading order
     are strictly isolated as supervision targets and never enter content demand inputs.
  6. Candidate template retrieval excludes the target poster ID (`exclude_poster_id=target_pid`).
"""

from __future__ import annotations

import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.request_schema import ContentBlockDemand, PosterRequest
from src.models.layout_retrieval import LayoutRetriever


KNOWN_CATEGORIES = [
    "title",
    "author_info",
    "section",
    "text",
    "figure",
    "table",
    "caption",
    "list",
    "logo",
]
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(KNOWN_CATEGORIES)}
NUM_CATEGORIES = len(KNOWN_CATEGORIES)
CONTENT_FEAT_DIM = 18
TEMPLATE_FEAT_DIM = 16


@dataclass
class UserContentItem:
    """User content demand specification with strict anti-leakage and explicit missingness."""

    item_id: str
    category_slug: str
    category_idx: int
    word_count: float = 0.0
    char_count: float = 0.0
    has_text_length: bool = False
    text_source: str = "unavailable"  # 'pdf_extracted', 'metadata', or 'unavailable'
    aspect_ratio: float = 1.0
    has_aspect_ratio: bool = False
    aspect_source: str = "unavailable"  # 'intrinsic_asset' or 'unavailable'
    table_rows: int = 0
    table_cols: int = 0
    priority: float = 1.0
    shuffled_order_idx: int = 0  # Decoupled from ground-truth reading order
    sample_text: str = ""

    def to_feature_vector(
        self,
        total_items: int = 1,
        include_order_index: bool = True,
    ) -> np.ndarray:
        """18-dimensional leak-free content representation vector.

        Structure:
          - Dims 0..8: One-hot category embedding (9 dims)
          - Dim 9: log1p(word_count) if has_text_length else 0.0
          - Dim 10: log1p(char_count) if has_text_length else 0.0
          - Dim 11: has_text_length binary flag (1.0 or 0.0)
          - Dim 12: aspect_ratio if has_aspect_ratio else 1.0
          - Dim 13: has_aspect_ratio binary flag (1.0 or 0.0)
          - Dim 14: table_rows / 10.0
          - Dim 15: table_cols / 10.0
          - Dim 16: priority (0.1 to 1.0)
          - Dim 17: shuffled_order_idx / total_items if include_order_index else 0.0
        """
        one_hot = np.zeros(NUM_CATEGORIES, dtype=np.float32)
        if 0 <= self.category_idx < NUM_CATEGORIES:
            one_hot[self.category_idx] = 1.0

        w_val = math.log1p(max(0.0, self.word_count)) if self.has_text_length else 0.0
        c_val = math.log1p(max(0.0, self.char_count)) if self.has_text_length else 0.0
        has_txt = 1.0 if self.has_text_length else 0.0

        asp_val = float(self.aspect_ratio) if self.has_aspect_ratio else 1.0
        has_asp = 1.0 if self.has_aspect_ratio else 0.0
        order_val = (
            float(self.shuffled_order_idx / max(1, total_items))
            if include_order_index
            else 0.0
        )

        extra_feats = np.array([
            w_val,
            c_val,
            has_txt,
            asp_val,
            has_asp,
            float(self.table_rows / 10.0),
            float(self.table_cols / 10.0),
            float(self.priority),
            order_val,
        ], dtype=np.float32)

        return np.concatenate([one_hot, extra_feats], axis=0)  # Total: 9 + 9 = 18 dims


@dataclass
class TemplateSlotItem:
    """A slot in the retrieved conditioning template."""

    slot_id: str
    category_slug: str
    category_idx: int
    bbox_norm: list[float]  # [x0, y0, w, h] in [0, 1]
    center: tuple[float, float]
    area_norm: float
    column_idx: int = 0

    def to_feature_vector(self) -> np.ndarray:
        """16-dimensional template slot representation vector."""
        one_hot = np.zeros(NUM_CATEGORIES, dtype=np.float32)
        if 0 <= self.category_idx < NUM_CATEGORIES:
            one_hot[self.category_idx] = 1.0

        geom = np.array([
            float(self.bbox_norm[0]),
            float(self.bbox_norm[1]),
            float(self.bbox_norm[2]),
            float(self.bbox_norm[3]),
            float(self.center[0]),
            float(self.center[1]),
            float(self.area_norm),
        ], dtype=np.float32)

        return np.concatenate([one_hot, geom], axis=0)  # Total 9 + 7 = 16 dims


@dataclass
class NeuralLayoutSample:
    """Complete paired training sample for neural layout generation."""

    target_poster_id: int
    template_poster_id: int
    split: str
    canvas_aspect_ratio: float
    target_column_count: int
    content_items: list[UserContentItem]
    template_slots: list[TemplateSlotItem]
    target_boxes: np.ndarray  # [M, 4] normalized [x0, y0, w, h]
    target_columns: np.ndarray  # [M] integer column index
    target_reading_orders: np.ndarray  # [M] integer reading order index
    element_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_poster_id": self.target_poster_id,
            "template_poster_id": self.template_poster_id,
            "split": self.split,
            "canvas_aspect_ratio": round(self.canvas_aspect_ratio, 4),
            "target_column_count": self.target_column_count,
            "num_content_items": len(self.content_items),
            "num_template_slots": len(self.template_slots),
            "content_summary": [
                {
                    "id": item.item_id,
                    "category": item.category_slug,
                    "words": int(item.word_count) if item.has_text_length else "N/A",
                    "text_source": item.text_source,
                    "has_text_length": item.has_text_length,
                    "aspect_ratio": round(item.aspect_ratio, 2) if item.has_aspect_ratio else "N/A",
                    "has_aspect_ratio": item.has_aspect_ratio,
                    "aspect_source": item.aspect_source,
                    "shuffled_order_idx": item.shuffled_order_idx,
                    "snippet": item.sample_text[:40] if item.sample_text else "None",
                }
                for item in self.content_items[:8]
            ],
            "sample_target_boxes": [
                [round(float(v), 3) for v in b] for b in self.target_boxes[:4]
            ],
        }


_POSTER_ID_TO_PDF_CACHE: dict[int, Path] = {}
_SCANNED_COCO_META_DIRS: set[Path] = set()


def get_pdf_path_for_poster_id(
    poster_id: int,
    split: str = "train",
    pdf_base_dir: Path | None = None,
) -> Path | None:
    """Resolve poster ID to vector PDF file path using cached COCO filename mapping."""
    if pdf_base_dir is None or not pdf_base_dir.exists():
        return None

    if poster_id in _POSTER_ID_TO_PDF_CACHE:
        p = _POSTER_ID_TO_PDF_CACHE[poster_id]
        if p.exists():
            return p

    # Check direct filename first
    for cand in [pdf_base_dir / split / f"{poster_id}.pdf", pdf_base_dir / f"{poster_id}.pdf"]:
        if cand.exists():
            _POSTER_ID_TO_PDF_CACHE[poster_id] = cand
            return cand

    # Load mapping from png/{split}.json once per coco_meta_dir
    coco_meta_dir = pdf_base_dir.parent / "png"
    if coco_meta_dir.exists() and coco_meta_dir not in _SCANNED_COCO_META_DIRS:
        _SCANNED_COCO_META_DIRS.add(coco_meta_dir)
        for s in ["train", "dev", "test"]:
            json_path = coco_meta_dir / f"{s}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    for img in meta.get("images", []):
                        iid = int(img["id"])
                        fname = img.get("file_name", "")
                        stem = Path(fname).stem
                        target_pdf = pdf_base_dir / s / f"{stem}.pdf"
                        if target_pdf.exists():
                            _POSTER_ID_TO_PDF_CACHE[iid] = target_pdf
                except Exception:
                    pass

    return _POSTER_ID_TO_PDF_CACHE.get(poster_id)


def extract_pdf_blocks_via_pdftotext(pdf_path: Path) -> list[dict[str, Any]]:
    """Run pdftotext -bbox-layout and parse word blocks with spatial coordinates."""
    if not pdf_path.exists():
        return []

    try:
        cmd = ["/usr/bin/pdftotext", "-bbox-layout", str(pdf_path), "-"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        xml_text = res.stdout
        if "<doc>" not in xml_text:
            return []

        root = ET.fromstring(xml_text)
        ns_prefix = ""
        if root.tag.startswith("{"):
            ns_url = root.tag.split("}")[0].strip("{")
            ns_prefix = f"{{{ns_url}}}"

        page = root.find(f".//{ns_prefix}page")
        if page is None:
            return []

        page_w = float(page.get("width", 1000.0))
        page_h = float(page.get("height", 1000.0))

        blocks: list[dict[str, Any]] = []
        for blk in page.findall(f".//{ns_prefix}block"):
            x0 = float(blk.get("xMin", 0.0)) / page_w
            y0 = float(blk.get("yMin", 0.0)) / page_h
            x1 = float(blk.get("xMax", 0.0)) / page_w
            y1 = float(blk.get("yMax", 0.0)) / page_h

            words: list[str] = []
            for word_el in blk.findall(f".//{ns_prefix}word"):
                txt = (word_el.text or "").strip()
                if txt:
                    words.append(txt)

            if words:
                full_text = " ".join(words)
                blocks.append({
                    "bbox_norm": [x0, y0, max(0.01, x1 - x0), max(0.01, y1 - y0)],
                    "words": words,
                    "word_count": len(words),
                    "char_count": len(full_text),
                    "text": full_text,
                })

        return blocks
    except Exception:
        return []


def extract_intrinsic_image_aspect_ratios(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract intrinsic image dimensions via pdfimages -list from original PDF asset."""
    if not pdf_path.exists():
        return []

    try:
        cmd = ["/usr/bin/pdfimages", "-list", str(pdf_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return []

        images: list[dict[str, Any]] = []
        for l in lines[2:]:
            parts = l.split()
            if len(parts) >= 5 and parts[2] == "image":
                w, h = int(parts[3]), int(parts[4])
                if w > 20 and h > 20:  # Ignore tiny icon decorations
                    images.append({
                        "intrinsic_width": w,
                        "intrinsic_height": h,
                        "aspect_ratio": float(w / max(1, h)),
                    })
        return images
    except Exception:
        return []


def match_pdf_text_to_elements(
    pdf_blocks: list[dict[str, Any]],
    elements_df_for_poster: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    """Spatially match PDF text blocks to annotated elements by intersection area."""
    matched_results: dict[int, dict[str, Any]] = {}

    for _, el in elements_df_for_poster.iterrows():
        eid = int(el["element_id"])
        ex0 = float(el["x0"])
        ey0 = float(el["y0"])
        ew = float(el["w_norm"])
        eh = float(el["h_norm"])
        ex1 = ex0 + ew
        ey1 = ey0 + eh
        cslug = str(el.get("category_slug", "text")).lower()

        # Non-textual categories don't search for text
        if cslug in ("figure", "logo"):
            continue

        matched_words: list[str] = []
        total_chars = 0

        for blk in pdf_blocks:
            bx0, by0, bw, bh = blk["bbox_norm"]
            bx1 = bx0 + bw
            by1 = by0 + bh

            ix0 = max(ex0, bx0)
            iy0 = max(ey0, by0)
            ix1 = min(ex1, bx1)
            iy1 = min(ey1, by1)

            if ix1 > ix0 and iy1 > iy0:
                inter_area = (ix1 - ix0) * (iy1 - iy0)
                blk_area = max(1e-5, bw * bh)
                overlap_ratio = inter_area / blk_area
                if overlap_ratio > 0.40 or (ix1 - ix0) / max(1e-5, ew) > 0.40:
                    matched_words.extend(blk["words"])
                    total_chars += blk["char_count"]

        if len(matched_words) >= 2:
            matched_results[eid] = {
                "word_count": float(len(matched_words)),
                "char_count": float(total_chars),
                "has_text_length": True,
                "text_source": "pdf_extracted",
                "sample_text": " ".join(matched_words[:12]),
            }

    return matched_results


def compute_target_overlap_exempt_mask(
    target_boxes: np.ndarray,
    iou_thresh: float = 0.02,
) -> np.ndarray:
    """Identify ground-truth element pairs that genuinely overlap/contain each other."""
    m = len(target_boxes)
    if m <= 1:
        return np.zeros((m, m), dtype=bool)

    x0 = target_boxes[:, 0]
    y0 = target_boxes[:, 1]
    w = target_boxes[:, 2]
    h = target_boxes[:, 3]
    x1 = x0 + w
    y1 = y0 + h
    area = w * h

    ix0 = np.maximum(x0[:, None], x0[None, :])
    iy0 = np.maximum(y0[:, None], y0[None, :])
    ix1 = np.minimum(x1[:, None], x1[None, :])
    iy1 = np.minimum(y1[:, None], y1[None, :])

    inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
    union = np.maximum(1e-6, area[:, None] + area[None, :] - inter)
    iou = inter / union
    np.fill_diagonal(iou, 0.0)
    return iou > iou_thresh


def build_synthetic_request_and_sample(
    target_poster_row: pd.Series | dict[str, Any],
    target_elements_df: pd.DataFrame,
    retriever_model: LayoutRetriever,
    pdf_base_dir: Path | None = None,
    elements_by_poster: dict[int, pd.DataFrame] | None = None,
    use_pdf_extraction: bool = True,
) -> NeuralLayoutSample:
    """Construct one strictly leak-free synthetic training sample from a ground-truth poster.

    Guarantees:
      - Content items have zero target coordinates or bounding-box area derived features.
      - Figures use intrinsic image dimensions if available; otherwise marked unavailable.
      - Elements without text extraction are marked unavailable (`has_text_length=False`).
      - Input sequence order is randomly/deterministically shuffled.
      - Target column count is excluded from content input vector.
      - Conditioning template is retrieved strictly excluding target_poster_id.
    """
    pid = int(target_poster_row["poster_id"])
    split = str(target_poster_row.get("split", "train"))
    source_split = str(target_poster_row.get("source_split", split))
    c_aspect = float(target_poster_row.get("aspect_ratio", 1.414))
    raw_k = target_poster_row.get("estimated_column_count", 3)
    k_cols = 3 if pd.isna(raw_k) else int(round(float(raw_k)))
    k_cols = max(1, min(5, k_cols))

    if elements_by_poster is not None:
        p_els = elements_by_poster.get(pid, target_elements_df.iloc[0:0]).copy()
    else:
        p_els = target_elements_df[target_elements_df["poster_id"] == pid].copy()

    if len(p_els) == 0:
        raise ValueError(f"Poster {pid} has 0 valid elements and cannot form a layout sample.")

    # Sort ground-truth target elements in true visual reading order (top-to-bottom, left-to-right)
    p_els = p_els.sort_values(by=["y0", "x0"]).reset_index(drop=True)

    # 1. Attempt PDF Text & Intrinsic Image Extraction from PDF assets
    matched_text_lookup: dict[int, dict[str, Any]] = {}
    intrinsic_images: list[dict[str, Any]] = []

    if use_pdf_extraction and pdf_base_dir is not None:
        pdf_path = get_pdf_path_for_poster_id(pid, split=source_split, pdf_base_dir=pdf_base_dir)
        if pdf_path is not None and pdf_path.exists():
            pdf_blocks = extract_pdf_blocks_via_pdftotext(pdf_path)
            if pdf_blocks:
                matched_text_lookup = match_pdf_text_to_elements(pdf_blocks, p_els)
            intrinsic_images = extract_intrinsic_image_aspect_ratios(pdf_path)

    # 2. Build Unordered Content Demands & Supervision Targets
    raw_content_items: list[UserContentItem] = []
    raw_element_ids: list[int] = []
    target_boxes_list: list[list[float]] = []
    target_columns_list: list[int] = []
    target_reading_orders_list: list[int] = []

    image_idx_counter = 0

    for true_reading_idx, (_, el) in enumerate(p_els.iterrows()):
        eid = int(el["element_id"])
        raw_element_ids.append(eid)
        cslug = str(el.get("category_slug", "text")).lower()
        cidx = CATEGORY_TO_IDX.get(cslug, CATEGORY_TO_IDX["text"])

        # Ground-truth targets (What model must predict)
        bx = float(el["x0"])
        by = float(el["y0"])
        bw = float(el["w_norm"])
        bh = float(el["h_norm"])
        target_boxes_list.append([bx, by, bw, bh])

        cx = float(el.get("cx", bx + bw / 2.0))
        target_col = int(min(4, max(0, int(cx * k_cols))))
        target_columns_list.append(target_col)
        target_reading_orders_list.append(true_reading_idx)

        # Content Features (Zero leakage from target coordinates or box area)
        if eid in matched_text_lookup:
            tinfo = matched_text_lookup[eid]
            w_cnt = float(tinfo["word_count"])
            c_cnt = float(tinfo["char_count"])
            has_txt = True
            txt_src = "pdf_extracted"
            snippet = tinfo["sample_text"]
        else:
            # Explicit missing value: NO geometric box area estimation!
            w_cnt = 0.0
            c_cnt = 0.0
            has_txt = False
            txt_src = "unavailable"
            snippet = ""

        # Figures: Check intrinsic image assets from PDF
        if cslug in ("figure", "logo"):
            if image_idx_counter < len(intrinsic_images):
                img_data = intrinsic_images[image_idx_counter]
                asp_val = float(np.clip(img_data["aspect_ratio"], 0.25, 4.0))
                has_asp = True
                asp_src = "intrinsic_asset"
                image_idx_counter += 1
            else:
                # Intrinsic asset unavailable: mark explicitly missing
                asp_val = 1.0
                has_asp = False
                asp_src = "unavailable"
        else:
            asp_val = 1.0
            has_asp = False
            asp_src = "unavailable"

        # Tables: user metadata specification (independent of target box)
        t_rows = 4 if cslug == "table" else 0
        t_cols = 3 if cslug == "table" else 0

        # Initial content item with dummy index (will be decoupled via shuffle)
        item = UserContentItem(
            item_id=f"item_{eid}",
            category_slug=cslug,
            category_idx=cidx,
            word_count=w_cnt,
            char_count=c_cnt,
            has_text_length=has_txt,
            text_source=txt_src,
            aspect_ratio=asp_val,
            has_aspect_ratio=has_asp,
            aspect_source=asp_src,
            table_rows=t_rows,
            table_cols=t_cols,
            priority=1.0 if cslug in ("title", "section") else 0.8,
            shuffled_order_idx=0,
            sample_text=snippet,
        )
        raw_content_items.append(item)

    # 3. Decouple Input Sequence from Target Reading Order via Deterministic Pseudo-Random Shuffle
    num_items = len(raw_content_items)
    rng = np.random.RandomState(seed=(pid * 10007) & 0x7FFFFFFF)
    perm = rng.permutation(num_items)

    shuffled_content_items: list[UserContentItem] = []
    shuffled_element_ids: list[int] = []
    shuffled_target_boxes: list[list[float]] = []
    shuffled_target_columns: list[int] = []
    shuffled_target_reading_orders: list[int] = []

    for new_input_pos, orig_idx in enumerate(perm):
        orig_item = raw_content_items[orig_idx]
        # Assign decoupled shuffled order index
        orig_item.shuffled_order_idx = new_input_pos

        shuffled_content_items.append(orig_item)
        shuffled_element_ids.append(raw_element_ids[orig_idx])
        shuffled_target_boxes.append(target_boxes_list[orig_idx])
        shuffled_target_columns.append(target_columns_list[orig_idx])
        shuffled_target_reading_orders.append(target_reading_orders_list[orig_idx])

    # 4. Retrieve Candidate Template (STRICT ANTI-LEAKAGE: Exclude target_poster_id)
    req_demands = [
        ContentBlockDemand(
            block_id=item.item_id,
            semantic_role=item.category_slug,
            visual_type=item.category_slug,
            word_count=item.word_count if item.has_text_length else 50,
            aspect_ratio=item.aspect_ratio if item.has_aspect_ratio else None,
        )
        for item in shuffled_content_items
    ]

    sim_request = PosterRequest(
        request_id=f"sim_req_{pid}",
        requested_aspect_ratio=c_aspect,
        requested_orientation="landscape" if c_aspect >= 1.05 else "portrait",
        preferred_column_count=None,  # Do not leak target column count
        blocks=req_demands,
    )

    retrieved_df = retriever_model.retrieve_for_request(
        request=sim_request,
        top_k=5,
        diversify=True,
        exclude_poster_id=pid,  # Target poster strictly excluded!
    )

    chosen_template_row = retrieved_df.iloc[0]
    template_pid = int(chosen_template_row["poster_id"])

    # Extract template slots from elements_df
    if elements_by_poster is not None:
        t_els = elements_by_poster.get(template_pid, target_elements_df.iloc[0:0]).copy()
    else:
        t_els = target_elements_df[target_elements_df["poster_id"] == template_pid].copy()
    t_els = t_els.sort_values(by=["y0", "x0"]).reset_index(drop=True)

    template_slots: list[TemplateSlotItem] = []
    raw_tk = chosen_template_row.get("estimated_column_count", 3)
    t_k_cols = 3 if pd.isna(raw_tk) else max(1, min(5, int(round(float(raw_tk)))))

    for _, tel in t_els.iterrows():
        tcslug = str(tel.get("category_slug", "text")).lower()
        tcidx = CATEGORY_TO_IDX.get(tcslug, CATEGORY_TO_IDX["text"])
        tbx = float(tel["x0"])
        tby = float(tel["y0"])
        tbw = float(tel["w_norm"])
        tbh = float(tel["h_norm"])
        tcx = float(tel.get("cx", tbx + tbw / 2.0))
        tcy = float(tel.get("cy", tby + tbh / 2.0))
        col_i = int(min(t_k_cols - 1, max(0, int(tcx * t_k_cols))))

        template_slots.append(
            TemplateSlotItem(
                slot_id=f"tslot_{tel['element_id']}",
                category_slug=tcslug,
                category_idx=tcidx,
                bbox_norm=[tbx, tby, tbw, tbh],
                center=(tcx, tcy),
                area_norm=float(tel.get("area_norm", tbw * tbh)),
                column_idx=col_i,
            )
        )

    return NeuralLayoutSample(
        target_poster_id=pid,
        template_poster_id=template_pid,
        split=split,
        canvas_aspect_ratio=c_aspect,
        target_column_count=k_cols,
        content_items=shuffled_content_items,
        template_slots=template_slots,
        target_boxes=np.array(shuffled_target_boxes, dtype=np.float32),
        target_columns=np.array(shuffled_target_columns, dtype=np.int64),
        target_reading_orders=np.array(shuffled_target_reading_orders, dtype=np.int64),
        element_ids=shuffled_element_ids,
    )


class PosterLayoutDataset(Dataset):
    """PyTorch Dataset for Template-Conditioned Neural Poster Layout Generation."""

    def __init__(
        self,
        posters_df: pd.DataFrame,
        elements_df: pd.DataFrame,
        retriever_model: LayoutRetriever,
        split: str = "train",
        pdf_base_dir: Path | None = None,
        max_samples: int | None = None,
        include_order_index: bool = True,
        use_pdf_extraction: bool = True,
        elements_by_poster: dict[int, pd.DataFrame] | None = None,
    ):
        self.split = split
        mask = posters_df["split"] == split
        if "total_element_count" in posters_df.columns:
            mask = mask & (posters_df["total_element_count"] > 0)
        self.split_posters = posters_df[mask].copy().reset_index(drop=True)
        if max_samples is not None:
            self.split_posters = self.split_posters.iloc[:max_samples].copy().reset_index(drop=True)

        self.elements_df = elements_df
        self.elements_by_poster = (
            elements_by_poster
            if elements_by_poster is not None
            else {int(pid): grp for pid, grp in elements_df.groupby("poster_id", sort=False)}
        )
        self.retriever_model = retriever_model
        self.pdf_base_dir = pdf_base_dir
        self.include_order_index = include_order_index
        self.use_pdf_extraction = use_pdf_extraction
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.split_posters)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]

        row = self.split_posters.iloc[idx]
        sample = build_synthetic_request_and_sample(
            target_poster_row=row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=self.pdf_base_dir,
            elements_by_poster=self.elements_by_poster,
            use_pdf_extraction=self.use_pdf_extraction,
        )

        total_items = len(sample.content_items)
        content_feats = np.stack(
            [
                item.to_feature_vector(
                    total_items=total_items,
                    include_order_index=self.include_order_index,
                )
                for item in sample.content_items
            ],
            axis=0,
        )
        template_feats = np.stack([slot.to_feature_vector() for slot in sample.template_slots], axis=0)
        exempt_mask = compute_target_overlap_exempt_mask(sample.target_boxes)

        res = {
            "content_feats": torch.from_numpy(content_feats),          # [M, 18]
            "template_feats": torch.from_numpy(template_feats),        # [K, 16]
            "target_boxes": torch.from_numpy(sample.target_boxes),      # [M, 4]
            "target_columns": torch.from_numpy(sample.target_columns),  # [M]
            "target_orders": torch.from_numpy(sample.target_reading_orders),  # [M]
            "target_overlap_exempt_mask": torch.from_numpy(exempt_mask),      # [M, M]
            "element_ids": torch.tensor(sample.element_ids, dtype=torch.long),  # [M]
            "macro_target": torch.tensor(max(0, min(4, sample.target_column_count - 1)), dtype=torch.long),  # 0..4
            "num_content": torch.tensor(len(sample.content_items), dtype=torch.long),
            "num_template": torch.tensor(len(sample.template_slots), dtype=torch.long),
            "target_poster_id": torch.tensor(sample.target_poster_id, dtype=torch.long),
            "template_poster_id": torch.tensor(sample.template_poster_id, dtype=torch.long),
        }
        self._cache[idx] = res
        return res


def collate_layout_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable M content items and variable K template slots with attention masks."""
    batch_size = len(batch)
    max_m = max(item["content_feats"].shape[0] for item in batch)
    max_k = max(item["template_feats"].shape[0] for item in batch)

    feat_dim_c = batch[0]["content_feats"].shape[1]
    feat_dim_t = batch[0]["template_feats"].shape[1]

    padded_content = torch.zeros(batch_size, max_m, feat_dim_c, dtype=torch.float32)
    padded_template = torch.zeros(batch_size, max_k, feat_dim_t, dtype=torch.float32)
    padded_target_boxes = torch.zeros(batch_size, max_m, 4, dtype=torch.float32)
    padded_target_cols = torch.zeros(batch_size, max_m, dtype=torch.long)
    padded_target_orders = torch.zeros(batch_size, max_m, dtype=torch.long)
    padded_element_ids = torch.full((batch_size, max_m), -1, dtype=torch.long)
    padded_exempt_mask = torch.zeros(batch_size, max_m, max_m, dtype=torch.bool)

    content_mask = torch.zeros(batch_size, max_m, dtype=torch.bool)
    template_mask = torch.zeros(batch_size, max_k, dtype=torch.bool)

    macro_targets = torch.stack([item["macro_target"] for item in batch], dim=0)
    target_poster_ids = torch.stack([item["target_poster_id"] for item in batch], dim=0)
    template_poster_ids = torch.stack([item["template_poster_id"] for item in batch], dim=0)

    for i, item in enumerate(batch):
        m_len = item["content_feats"].shape[0]
        k_len = item["template_feats"].shape[0]

        padded_content[i, :m_len] = item["content_feats"]
        padded_template[i, :k_len] = item["template_feats"]
        padded_target_boxes[i, :m_len] = item["target_boxes"]
        padded_target_cols[i, :m_len] = item["target_columns"]
        padded_target_orders[i, :m_len] = item["target_orders"]
        if "element_ids" in item:
            padded_element_ids[i, :m_len] = item["element_ids"]
        if "target_overlap_exempt_mask" in item:
            padded_exempt_mask[i, :m_len, :m_len] = item["target_overlap_exempt_mask"]

        content_mask[i, :m_len] = True
        template_mask[i, :k_len] = True

    return {
        "content_feats": padded_content,             # [B, max_M, 18]
        "template_feats": padded_template,           # [B, max_K, 16]
        "content_mask": content_mask,               # [B, max_M] (True where valid)
        "template_mask": template_mask,             # [B, max_K] (True where valid)
        "target_boxes": padded_target_boxes,         # [B, max_M, 4]
        "target_columns": padded_target_cols,       # [B, max_M]
        "target_orders": padded_target_orders,       # [B, max_M]
        "target_overlap_exempt_mask": padded_exempt_mask,  # [B, max_M, max_M]
        "element_ids": padded_element_ids,           # [B, max_M] (-1 where padded)
        "macro_targets": macro_targets,             # [B]
        "target_poster_ids": target_poster_ids,     # [B]
        "template_poster_ids": template_poster_ids, # [B]
    }

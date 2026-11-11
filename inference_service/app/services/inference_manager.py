from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

from inference_service.app.config import Settings
from inference_service.app.schemas.layout import (
    ContentBlockInput,
    GenerateLayoutRequest,
    PosterRequestPayload,
)
from inference_service.app.services.data_loader import (
    DataDependencyStatus,
    has_configured_adaptation_source,
    load_elements_dataframe,
)
from inference_service.app.services.memory_profiler import (
    MemoryTracker,
    trim_process_memory,
)
from inference_service.app.services.model_loader import (
    LoadedModelMetadata,
    configure_clearml_credentials,
    load_clearml_neural_model,
    load_clearml_retriever,
)
from src.adaptation.adaptation_evaluation import compute_pairwise_overlap_area
from src.features.request_schema import ContentBlockDemand, PosterRequest
from src.models.layout_retrieval import LayoutRetriever
from src.neural_layout.dataset import CATEGORY_TO_IDX, TemplateSlotItem, UserContentItem
from src.neural_layout.layout_model import TemplateConditionedLayoutModel

logger = logging.getLogger(__name__)


class ServiceNotReadyError(RuntimeError):
    """Raised when an engine or the service is not ready to serve requests."""


class InferenceExecutionError(RuntimeError):
    """Raised when model inference or geometry validation fails."""


def _word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def _safe_aspect_ratio(val: float | None, default: float = 1.33) -> float:
    if val is None or val <= 0:
        return default
    return float(np.clip(val, 0.25, 4.0))


def _sanitize_error_message(exc: Exception, settings: Settings | None = None) -> str:
    msg = str(exc)
    if settings is not None:
        for secret in (
            settings.CLEARML_API_SECRET_KEY,
            settings.CLEARML_API_ACCESS_KEY,
            settings.AWS_SECRET_ACCESS_KEY or "",
            settings.AWS_ACCESS_KEY_ID or "",
            settings.MINIDVC_CLIENT_SECRET or "",
            settings.MINIDVC_REFRESH_TOKEN or "",
        ):
            if secret and secret in msg:
                msg = msg.replace(secret, "[REDACTED]")
    return msg


def validate_and_normalize_geometry(
    raw_elements: list[dict[str, Any]],
    canvas_w: int,
    canvas_h: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate finite coordinates, enforce canvas bounds/min dimensions, and compute metrics."""
    if not raw_elements:
        raise InferenceExecutionError("Model produced zero layout elements.")

    min_w_norm = 12.0 / max(1, canvas_w)
    min_h_norm = 12.0 / max(1, canvas_h)

    validated: list[dict[str, Any]] = []
    overflow_before_clamp = 0
    min_size_violations = 0
    clamped_count = 0
    bboxes_norm: list[list[float]] = []

    for el in raw_elements:
        bx_n = el.get("bbox_norm")
        if not isinstance(bx_n, (list, tuple)) or len(bx_n) != 4:
            raise InferenceExecutionError(f"Element '{el.get('id')}' missing valid 4-tuple bbox_norm.")

        x0, y0, wn, hn = [float(v) for v in bx_n]
        if not all(math.isfinite(v) for v in (x0, y0, wn, hn)):
            raise InferenceExecutionError(
                f"Element '{el.get('id')}' contains non-finite coordinates."
            )

        if x0 < 0.0 or y0 < 0.0 or x0 + wn > 1.001 or y0 + hn > 1.001:
            overflow_before_clamp += 1

        if wn < min_w_norm or hn < min_h_norm:
            min_size_violations += 1

        cx0 = float(np.clip(x0, 0.01, 0.96))
        cy0 = float(np.clip(y0, 0.01, 0.96))
        cwn = float(np.clip(wn, min_w_norm, 0.99 - cx0))
        chn = float(np.clip(hn, min_h_norm, 0.99 - cy0))

        if (
            abs(cx0 - x0) > 1e-4
            or abs(cy0 - y0) > 1e-4
            or abs(cwn - wn) > 1e-4
            or abs(chn - hn) > 1e-4
        ):
            clamped_count += 1

        norm_box = [round(cx0, 4), round(cy0, 4), round(cwn, 4), round(chn, 4)]
        bboxes_norm.append(norm_box)

        updated = dict(el)
        updated["bbox_norm"] = norm_box
        updated["bbox"] = [
            cx0 * canvas_w,
            cy0 * canvas_h,
            cwn * canvas_w,
            chn * canvas_h,
        ]
        validated.append(updated)

    overlap_area = compute_pairwise_overlap_area(bboxes_norm)
    n_el = max(1, len(validated))
    geometry_metrics = {
        "num_elements": len(validated),
        "raw_overflow_rate": round(float(overflow_before_clamp / n_el), 4),
        "clamped_boxes_count": clamped_count,
        "min_size_violation_rate": round(float(min_size_violations / n_el), 4),
        "validated_overlap_area": round(float(overlap_area), 6),
    }
    return validated, geometry_metrics


def _build_template_slots_from_poster_row(poster_row: pd.Series) -> list[TemplateSlotItem]:
    """Construct conditioning TemplateSlotItems from LayoutRetriever's in-memory poster metadata
    so the neural engine does not require loading elements.parquet."""
    raw_cols = poster_row.get("estimated_column_count", 3)
    k_cols = 3 if pd.isna(raw_cols) else max(1, min(5, int(round(float(raw_cols)))))

    slots: list[TemplateSlotItem] = []
    slot_idx = 0

    # Header band slots (title + author_info)
    title_cnt = max(1, int(round(float(poster_row.get("count_title", 1) or 1))))
    auth_cnt = max(1, int(round(float(poster_row.get("count_author_info", 1) or 1))))
    for t_i in range(min(title_cnt, 2)):
        y0 = 0.04 + t_i * 0.07
        slots.append(
            TemplateSlotItem(
                slot_id=f"synth_title_{slot_idx}",
                category_slug="title",
                category_idx=CATEGORY_TO_IDX["title"],
                bbox_norm=[0.06, y0, 0.88, 0.065],
                center=(0.50, y0 + 0.0325),
                area_norm=0.88 * 0.065,
                column_idx=0,
            )
        )
        slot_idx += 1

    for a_i in range(min(auth_cnt, 2)):
        y0 = 0.12 + a_i * 0.045
        slots.append(
            TemplateSlotItem(
                slot_id=f"synth_auth_{slot_idx}",
                category_slug="author_info",
                category_idx=CATEGORY_TO_IDX["author_info"],
                bbox_norm=[0.10, y0, 0.80, 0.04],
                center=(0.50, y0 + 0.02),
                area_norm=0.80 * 0.04,
                column_idx=0,
            )
        )
        slot_idx += 1

    # Body category inventory from poster summary counts
    body_specs = [
        ("section", max(0, int(round(float(poster_row.get("count_section", 3) or 0))))),
        ("text", max(0, int(round(float(poster_row.get("count_text", 3) or 0))))),
        ("list", max(0, int(round(float(poster_row.get("count_list", 0) or 0))))),
        ("figure", max(0, int(round(float(poster_row.get("count_figure", 1) or 0))))),
        ("table", max(0, int(round(float(poster_row.get("count_table", 0) or 0))))),
        ("caption", max(0, int(round(float(poster_row.get("count_caption", 1) or 0))))),
    ]
    body_cats: list[str] = []
    for cat, count in body_specs:
        body_cats.extend([cat] * min(count, 12))
    if not body_cats:
        body_cats = ["section", "text", "figure", "caption"]

    # Order items across columns cleanly (sections paired before text/media)
    col_buckets: list[list[str]] = [[] for _ in range(k_cols)]
    for idx, cat in enumerate(body_cats):
        col_buckets[idx % k_cols].append(cat)

    margin_x = 0.04
    gap_x = 0.025
    col_w = (1.0 - 2.0 * margin_x - max(0, k_cols - 1) * gap_x) / k_cols
    body_top = 0.19
    body_height = 0.77
    gap_y = 0.015

    for c_idx, items in enumerate(col_buckets):
        if not items:
            continue
        x0 = margin_x + c_idx * (col_w + gap_x)
        weights = [
            0.04 if cat == "section" else 0.05 if cat == "caption" else 0.16 if cat in ("figure", "table") else 0.14
            for cat in items
        ]
        avail_h = max(0.20, body_height - max(0, len(items) - 1) * gap_y)
        total_w = sum(weights) or 1.0
        cur_y = body_top
        for cat, wt in zip(items, weights):
            slot_h = max(0.03, min(0.35, avail_h * (wt / total_w)))
            if cur_y + slot_h > 0.97:
                slot_h = max(0.02, 0.97 - cur_y)
            slots.append(
                TemplateSlotItem(
                    slot_id=f"synth_{cat}_{slot_idx}",
                    category_slug=cat,
                    category_idx=CATEGORY_TO_IDX[cat],
                    bbox_norm=[x0, cur_y, col_w, slot_h],
                    center=(x0 + col_w / 2.0, cur_y + slot_h / 2.0),
                    area_norm=col_w * slot_h,
                    column_idx=c_idx,
                )
            )
            cur_y = min(0.95, cur_y + slot_h + gap_y)
            slot_idx += 1

    return slots


class InferenceManager:
    """Thread-safe lifecycle and lazy-engine manager for ClearML-registered layout models."""

    def __init__(self, memory_tracker: MemoryTracker | None = None) -> None:
        self._init_lock = threading.Lock()
        self._adaptation_lock = threading.Lock()
        self.memory_tracker = memory_tracker or MemoryTracker()

        self._settings: Settings | None = None
        self.ready: bool = False
        self.init_error: str | None = None
        self.default_engine: str = "neural"
        self.torch_device: str = "cpu"

        self.retriever: LayoutRetriever | None = None
        self.retriever_info: LoadedModelMetadata | None = None
        self.neural_model: TemplateConditionedLayoutModel | None = None
        self.neural_info: LoadedModelMetadata | None = None

        self.elements_df: pd.DataFrame | None = None
        self._layout_adapter_cls: Any = None
        self.data_status: DataDependencyStatus = DataDependencyStatus(
            ready=False,
            source="not_loaded",
        )

        self._first_neural_profiled = False
        self._first_adaptation_profiled = False

    def initialize(self, settings: Settings) -> None:
        """Load ClearML retriever and neural model at startup; defer adaptation data unless configured eager."""
        with self._init_lock:
            self._settings = settings
            self.ready = False
            self.init_error = None
            self.default_engine = settings.DEFAULT_ENGINE
            self.torch_device = settings.TORCH_DEVICE
            self.elements_df = None
            self._layout_adapter_cls = None
            self.data_status = DataDependencyStatus(ready=False, source="not_loaded")
            self._first_neural_profiled = False
            self._first_adaptation_profiled = False

            try:
                self.memory_tracker.record("dependencies_imported")
                configure_clearml_credentials(settings)

                retriever, retriever_info = load_clearml_retriever(settings)
                self.retriever = retriever
                self.retriever_info = retriever_info
                self.memory_tracker.record("retriever_loaded")

                neural_model, neural_info = load_clearml_neural_model(settings)
                self.neural_model = neural_model
                self.neural_info = neural_info
                self.memory_tracker.record("neural_model_loaded")

                if settings.DEFAULT_ENGINE == "adaptation" or settings.EAGER_LOAD_ADAPTATION_DATA:
                    self._ensure_adaptation_ready_locked()

                self.ready = True
                logger.info(
                    "InferenceManager ready (default_engine=%s, adaptation_loaded=%s): retriever=%s (%s), neural=%s (%s)",
                    self.default_engine,
                    self.adaptation_initialized,
                    retriever_info.model_name,
                    retriever_info.clearml_model_id,
                    neural_info.model_name,
                    neural_info.clearml_model_id,
                )
            except Exception as exc:
                self.ready = False
                self.init_error = _sanitize_error_message(exc, settings)
                if self.elements_df is None:
                    self.data_status = DataDependencyStatus(
                        ready=False,
                        source=self.data_status.source,
                        error=self.init_error,
                    )
                logger.error("InferenceManager initialization failed: %s", self.init_error)

    def _ensure_adaptation_ready_locked(self) -> None:
        if self.elements_df is not None and self._layout_adapter_cls is not None:
            return
        if self._settings is None:
            raise ServiceNotReadyError("InferenceManager settings not initialized.")

        valid_poster_ids: set[int] | None = None
        if self.retriever is not None and self.retriever.train_posters_ is not None:
            valid_poster_ids = {
                int(pid) for pid in self.retriever.train_posters_["poster_id"].tolist()
            }

        try:
            elements_df, data_status = load_elements_dataframe(
                self._settings,
                valid_poster_ids=valid_poster_ids,
                memory_tracker=self.memory_tracker,
            )
            from src.adaptation.layout_adapter import LayoutAdapter

            self.elements_df = elements_df
            self.data_status = data_status
            self._layout_adapter_cls = LayoutAdapter
            trim_process_memory()
            self.memory_tracker.record("layout_adapter_initialized")
        except Exception as exc:
            sanitized = _sanitize_error_message(exc, self._settings)
            self.elements_df = None
            self._layout_adapter_cls = None
            self.data_status = DataDependencyStatus(
                ready=False,
                source="error",
                error=sanitized,
            )
            raise ServiceNotReadyError(f"Adaptation engine unavailable: {sanitized}") from exc

    def ensure_adaptation_ready(self) -> None:
        """Thread-safe lazy initialization of elements.parquet and LayoutAdapter on first adaptation use."""
        if self.elements_df is not None and self._layout_adapter_cls is not None:
            return
        with self._adaptation_lock:
            self._ensure_adaptation_ready_locked()

    def shutdown(self) -> None:
        with self._init_lock:
            self.ready = False
            self.retriever = None
            self.neural_model = None
            self.elements_df = None
            self._layout_adapter_cls = None
            trim_process_memory()

    @property
    def adaptation_initialized(self) -> bool:
        return self.elements_df is not None and self._layout_adapter_cls is not None

    @property
    def engine_states(self) -> dict[str, str]:
        neural_state = (
            "ready" if (self.retriever is not None and self.neural_model is not None) else "unavailable"
        )
        if self.retriever is None:
            adaptation_state = "unavailable"
        elif self.adaptation_initialized:
            adaptation_state = "ready"
        elif self.data_status.error:
            adaptation_state = "unavailable"
        elif self._settings is not None and has_configured_adaptation_source(self._settings):
            adaptation_state = "lazy_uninitialized"
        else:
            adaptation_state = "unavailable"
        return {"neural": neural_state, "adaptation": adaptation_state}

    @property
    def supported_engines(self) -> list[str]:
        engines: list[str] = []
        if self.retriever is not None and self.neural_model is not None:
            engines.append("neural")
        if self.retriever is not None and self.adaptation_initialized:
            engines.append("adaptation")
        return engines

    def get_readiness_snapshot(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "model_initialized": self.neural_model is not None,
            "retriever_initialized": self.retriever is not None,
            "data_dependency_ready": self.data_status.ready and self.elements_df is not None,
            "adaptation_initialized": self.adaptation_initialized,
            "selected_models": {
                "retriever": {
                    "model_name": self.retriever_info.model_name if self.retriever_info else None,
                    "clearml_model_id": (
                        self.retriever_info.clearml_model_id if self.retriever_info else None
                    ),
                    "semantic_version": (
                        self.retriever_info.semantic_version if self.retriever_info else None
                    ),
                },
                "neural_generator": {
                    "model_name": self.neural_info.model_name if self.neural_info else None,
                    "clearml_model_id": (
                        self.neural_info.clearml_model_id if self.neural_info else None
                    ),
                    "semantic_version": (
                        self.neural_info.semantic_version if self.neural_info else None
                    ),
                },
            },
            "supported_engines": self.supported_engines,
            "engine_states": self.engine_states,
            "error": self.init_error,
        }

    def get_loaded_models_snapshot(self) -> dict[str, Any]:
        models: list[dict[str, Any]] = []
        for info in (self.retriever_info, self.neural_info):
            if info is None:
                continue
            models.append({
                "role": info.role,
                "model_name": info.model_name,
                "model_family": info.model_family,
                "clearml_model_id": info.clearml_model_id,
                "semantic_version": info.semantic_version,
                "architecture": info.architecture,
                "checkpoint_sha256": info.checkpoint_sha256,
                "lifecycle_status": info.lifecycle_status,
                "dataset_identifier": info.dataset_identifier,
                "supported_engines": info.supported_engines,
            })

        return {
            "status": "ready" if self.ready else "not_ready",
            "default_engine": self.default_engine,
            "supported_engines": self.supported_engines,
            "models": models,
            "data_dependency": {
                "ready": self.data_status.ready,
                "source": self.data_status.source,
                "sha256": self.data_status.sha256,
                "row_count": self.data_status.row_count,
                "poster_count": self.data_status.poster_count,
                "artifact_size_mib": self.data_status.artifact_size_mib,
                "dataset_mem_mib": self.data_status.dataset_mem_mib,
                "error": self.data_status.error,
            },
        }

    def _build_poster_request(
        self, payload: PosterRequestPayload
    ) -> tuple[PosterRequest, dict[str, dict[str, Any]]]:
        aspect_ratio = 1.414 if payload.orientation == "landscape" else 0.707
        blocks_meta: dict[str, dict[str, Any]] = {}
        demands: list[ContentBlockDemand] = []

        title_text = payload.title.strip()
        title_blk = ContentBlockDemand(
            block_id="b_title",
            semantic_role="title",
            visual_type="title",
            priority=1.5,
            order_index=0,
            word_count=max(2, _word_count(title_text)),
            char_count=max(10, len(title_text)),
        )
        setattr(title_blk, "content", title_text)
        demands.append(title_blk)
        blocks_meta["b_title"] = {"role": "title", "content": title_text}

        authors_str = str(payload.authors or "").strip()
        affiliations_str = str(payload.affiliations or "").strip()
        author_line = " — ".join(p for p in [authors_str, affiliations_str] if p) or "Author Name"
        auth_blk = ContentBlockDemand(
            block_id="b_auth",
            semantic_role="author_info",
            visual_type="author_info",
            priority=1.3,
            order_index=1,
            word_count=max(2, _word_count(author_line)),
            char_count=max(10, len(author_line)),
        )
        setattr(auth_blk, "content", author_line)
        demands.append(auth_blk)
        blocks_meta["b_auth"] = {"role": "author_info", "content": author_line}

        order = 2
        source_blocks = payload.blocks or [
            ContentBlockInput(
                id="intro",
                type="section",
                heading="Introduction",
                text="Summarize your research problem, motivation, and key contributions.",
            )
        ]

        for blk in source_blocks:
            b_type = blk.type
            if b_type in ("section", "text", "list"):
                if blk.heading and blk.heading.strip():
                    hid = f"{blk.id}_h"
                    h_txt = blk.heading.strip()
                    h_demand = ContentBlockDemand(
                        block_id=hid,
                        semantic_role=blk.id,
                        visual_type="section",
                        priority=blk.priority,
                        order_index=order,
                        word_count=max(1, _word_count(h_txt)),
                        char_count=max(4, len(h_txt)),
                    )
                    setattr(h_demand, "content", h_txt)
                    demands.append(h_demand)
                    blocks_meta[hid] = {
                        "role": "section",
                        "content": h_txt,
                        "sourceBlockId": blk.id,
                    }
                    order += 1

                body_txt = (blk.text or "").strip() or (blk.heading or "Content")
                pid = f"{blk.id}_p"
                vis_type = "list" if b_type == "list" else "text"
                p_demand = ContentBlockDemand(
                    block_id=pid,
                    semantic_role=blk.id,
                    visual_type=vis_type,
                    priority=blk.priority,
                    order_index=order,
                    word_count=max(5, _word_count(body_txt)),
                    char_count=max(20, len(body_txt)),
                )
                setattr(p_demand, "content", body_txt)
                demands.append(p_demand)
                blocks_meta[pid] = {
                    "role": vis_type,
                    "content": body_txt,
                    "sourceBlockId": blk.id,
                }
                order += 1

            elif b_type == "figure":
                fid = blk.id
                asp = _safe_aspect_ratio(blk.aspectRatio, 1.33)
                f_demand = ContentBlockDemand(
                    block_id=fid,
                    semantic_role="figure",
                    visual_type="figure",
                    priority=blk.priority,
                    order_index=order,
                    aspect_ratio=asp,
                )
                demands.append(f_demand)
                blocks_meta[fid] = {
                    "role": "figure",
                    "content": blk.heading or "Figure",
                    "imageDataUrl": blk.imageDataUrl,
                    "sourceBlockId": blk.id,
                }
                order += 1

                if blk.caption and blk.caption.strip():
                    cid = f"{blk.id}_cap"
                    c_txt = blk.caption.strip()
                    c_demand = ContentBlockDemand(
                        block_id=cid,
                        semantic_role="caption",
                        visual_type="caption",
                        priority=blk.priority * 0.9,
                        order_index=order,
                        word_count=max(3, _word_count(c_txt)),
                        char_count=max(12, len(c_txt)),
                        must_be_adjacent_to=fid,
                    )
                    setattr(c_demand, "content", c_txt)
                    demands.append(c_demand)
                    blocks_meta[cid] = {
                        "role": "caption",
                        "content": c_txt,
                        "sourceBlockId": blk.id,
                    }
                    order += 1

            elif b_type == "table":
                tid = blk.id
                rows = blk.tableRows or 4
                cols = blk.tableCols or 3
                t_label = (blk.heading or blk.text or f"Table ({rows}×{cols})").strip()
                t_demand = ContentBlockDemand(
                    block_id=tid,
                    semantic_role="table",
                    visual_type="table",
                    priority=blk.priority,
                    order_index=order,
                    table_rows=rows,
                    table_cols=cols,
                )
                setattr(t_demand, "content", t_label)
                demands.append(t_demand)
                blocks_meta[tid] = {
                    "role": "table",
                    "content": t_label,
                    "tableRows": rows,
                    "tableCols": cols,
                    "sourceBlockId": blk.id,
                }
                order += 1

                if blk.caption and blk.caption.strip():
                    cid = f"{blk.id}_cap"
                    c_txt = blk.caption.strip()
                    c_demand = ContentBlockDemand(
                        block_id=cid,
                        semantic_role="caption",
                        visual_type="caption",
                        priority=blk.priority * 0.9,
                        order_index=order,
                        word_count=max(3, _word_count(c_txt)),
                        char_count=max(12, len(c_txt)),
                        must_be_adjacent_to=tid,
                    )
                    setattr(c_demand, "content", c_txt)
                    demands.append(c_demand)
                    blocks_meta[cid] = {
                        "role": "caption",
                        "content": c_txt,
                        "sourceBlockId": blk.id,
                    }
                    order += 1

            elif b_type == "caption":
                cid = blk.id
                c_txt = (blk.text or blk.caption or "Caption").strip()
                c_demand = ContentBlockDemand(
                    block_id=cid,
                    semantic_role="caption",
                    visual_type="caption",
                    priority=blk.priority,
                    order_index=order,
                    word_count=max(3, _word_count(c_txt)),
                    char_count=max(12, len(c_txt)),
                )
                setattr(c_demand, "content", c_txt)
                demands.append(c_demand)
                blocks_meta[cid] = {
                    "role": "caption",
                    "content": c_txt,
                    "sourceBlockId": blk.id,
                }
                order += 1

        req = PosterRequest(
            request_id=f"req_{uuid.uuid4().hex[:10]}",
            requested_aspect_ratio=aspect_ratio,
            requested_orientation=payload.orientation,
            preferred_column_count=payload.preferredColumns,
            blocks=demands,
            author_count=max(1, len([a for a in authors_str.split(",") if a.strip()])),
            affiliation_count=max(1, len([a for a in affiliations_str.split(",") if a.strip()])),
        )
        return req, blocks_meta

    def _get_neural_template_slots(self, template_poster_id: int) -> tuple[list[np.ndarray], str]:
        assert self.retriever is not None and self.retriever.train_posters_ is not None

        if self.elements_df is not None:
            tmpl_rows = self.elements_df[self.elements_df["poster_id"] == template_poster_id]
            tmpl_vecs: list[np.ndarray] = []
            for _, row in tmpl_rows.iterrows():
                cat = str(row.get("category_slug", "text"))
                if cat not in CATEGORY_TO_IDX:
                    continue
                slot = TemplateSlotItem(
                    slot_id=str(row["element_id"]),
                    category_slug=cat,
                    category_idx=CATEGORY_TO_IDX[cat],
                    bbox_norm=[float(row["x0"]), float(row["y0"]), float(row["w_norm"]), float(row["h_norm"])],
                    center=(float(row["cx"]), float(row["cy"])),
                    area_norm=float(row["area_norm"]),
                )
                tmpl_vecs.append(slot.to_feature_vector())
            if tmpl_vecs:
                return tmpl_vecs, "elements_df"

        # Synthesize conditioning slots from retriever's in-memory train_posters_ metadata
        # so the neural engine never triggers loading elements.parquet.
        matched = self.retriever.train_posters_[
            self.retriever.train_posters_["poster_id"] == template_poster_id
        ]
        if len(matched) == 0:
            raise InferenceExecutionError(
                f"Retrieved template {template_poster_id} missing from retriever train_posters_."
            )
        synth_slots = _build_template_slots_from_poster_row(matched.iloc[0])
        return [slot.to_feature_vector() for slot in synth_slots], "retriever_metadata"

    def _run_neural_engine(
        self,
        req: PosterRequest,
        blocks_meta: dict[str, dict[str, Any]],
        template_poster_id: int,
        canvas_w: int,
        canvas_h: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert self.neural_model is not None

        total_items = max(1, len(req.blocks))
        content_vecs: list[np.ndarray] = []
        for idx, blk in enumerate(req.blocks):
            cat = blk.scipost_category if blk.scipost_category in CATEGORY_TO_IDX else "text"
            raw_txt = blocks_meta.get(blk.block_id, {}).get("content")
            has_txt = (
                isinstance(raw_txt, str)
                and len(raw_txt.strip()) > 0
                and cat not in ("figure", "table")
            )
            item = UserContentItem(
                item_id=blk.block_id,
                category_slug=cat,
                category_idx=CATEGORY_TO_IDX[cat],
                word_count=float(blk.word_count or 0.0),
                char_count=float(blk.char_count or 0.0),
                has_text_length=has_txt,
                text_source="metadata" if has_txt else "unavailable",
                aspect_ratio=float(blk.aspect_ratio or 1.0),
                has_aspect_ratio=(blk.aspect_ratio is not None),
                aspect_source="metadata" if blk.aspect_ratio is not None else "unavailable",
                table_rows=int(blk.table_rows or 0),
                table_cols=int(blk.table_cols or 0),
                priority=float(blk.priority),
                shuffled_order_idx=idx,
            )
            content_vecs.append(item.to_feature_vector(total_items=total_items, include_order_index=True))

        tmpl_vecs, slot_source = self._get_neural_template_slots(template_poster_id)

        device = torch.device(self.torch_device)
        content_tensor = torch.tensor(np.stack(content_vecs), dtype=torch.float32, device=device).unsqueeze(0)
        template_tensor = torch.tensor(np.stack(tmpl_vecs), dtype=torch.float32, device=device).unsqueeze(0)
        content_mask = torch.ones((1, len(content_vecs)), dtype=torch.bool, device=device)
        template_mask = torch.ones((1, len(tmpl_vecs)), dtype=torch.bool, device=device)

        with torch.inference_mode():
            preds = self.neural_model(content_tensor, template_tensor, content_mask, template_mask)

        pred_boxes = preds["pred_boxes"][0].cpu().numpy()
        pred_macro = int(torch.argmax(preds["pred_macro_column_logits"][0]).item()) + 1

        raw_elements: list[dict[str, Any]] = []
        for idx, blk in enumerate(req.blocks):
            x0, y0, wn, hn = [float(v) for v in pred_boxes[idx]]
            raw_elements.append({
                "id": blk.block_id,
                "type": blk.scipost_category,
                "bbox": [x0 * canvas_w, y0 * canvas_h, wn * canvas_w, hn * canvas_h],
                "bbox_norm": [x0, y0, wn, hn],
            })

        metrics = {
            "predicted_macro_columns": pred_macro,
            "num_content_blocks": len(req.blocks),
            "num_template_slots": len(tmpl_vecs),
            "template_slot_source": slot_source,
        }
        return raw_elements, metrics

    def _run_adaptation_engine(
        self,
        req: PosterRequest,
        cands_df: pd.DataFrame,
        canvas_w: int,
        canvas_h: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str, int]:
        self.ensure_adaptation_ready()
        assert self.elements_df is not None and self._layout_adapter_cls is not None

        adapter = self._layout_adapter_cls(
            elements_df=self.elements_df,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )
        best, all_results = adapter.adapt_and_select_best(req, cands_df)
        retrieved_template_id = str(best.template_poster_id)
        columns_used = len(best.hierarchical_template.columns) or (req.preferred_column_count or 3)

        metrics = {
            "is_feasible": bool(best.is_feasible),
            "composite_quality_score": round(
                float(best.evaluation_metrics.get("composite_quality_score", 0.0)), 3
            ),
            "overflow_rate": round(float(best.evaluation_metrics.get("overflow_rate", 0.0)), 4),
            "overlap_area": round(float(best.evaluation_metrics.get("overlap_area", 0.0)), 6),
            "whitespace_utilization": round(
                float(best.evaluation_metrics.get("whitespace_utilization", 0.0)), 4
            ),
            "candidates_evaluated": len(all_results),
        }

        raw_elements: list[dict[str, Any]] = []
        for el in best.poster_document.elements:
            bx_n = el.bbox_norm
            raw_elements.append({
                "id": el.id,
                "type": el.type,
                "bbox": [
                    float(bx_n[0] * canvas_w),
                    float(bx_n[1] * canvas_h),
                    float(bx_n[2] * canvas_w),
                    float(bx_n[3] * canvas_h),
                ],
                "bbox_norm": [float(v) for v in bx_n],
                "content": el.content,
            })

        return raw_elements, metrics, retrieved_template_id, columns_used

    def _to_konva_elements(
        self,
        raw_elements: list[dict[str, Any]],
        blocks_meta: dict[str, dict[str, Any]],
        canvas_w: int,
        canvas_h: int,
        accent_color: str,
    ) -> list[dict[str, Any]]:
        konva_elements: list[dict[str, Any]] = []
        layer_idx = 0

        header_boxes = [
            el["bbox"] for el in raw_elements if el.get("type") in ("title", "author_info")
        ]
        if header_boxes:
            min_x = max(16.0, min(b[0] for b in header_boxes) - 20.0)
            min_y = max(16.0, min(b[1] for b in header_boxes) - 14.0)
            max_x = min(canvas_w - 16.0, max(b[0] + b[2] for b in header_boxes) + 20.0)
            max_y = min(canvas_h * 0.30, max(b[1] + b[3] for b in header_boxes) + 14.0)
            konva_elements.append({
                "id": f"el_header_bg_{uuid.uuid4().hex[:6]}",
                "type": "rect",
                "name": "Header background",
                "x": round(min_x, 2),
                "y": round(min_y, 2),
                "width": round(max(120.0, max_x - min_x), 2),
                "height": round(max(64.0, max_y - min_y), 2),
                "rotation": 0,
                "layer": layer_idx,
                "visible": True,
                "locked": False,
                "fill": accent_color,
                "opacity": 1.0,
                "semanticRole": "header_background",
            })
            layer_idx += 1

        for el in raw_elements:
            bid = el["id"]
            etype = el["type"]
            x, y, w, h = [float(v) for v in el["bbox"]]
            bbox_norm = [round(float(v), 4) for v in el["bbox_norm"]]
            meta = blocks_meta.get(bid, {})
            content = meta.get("content") or el.get("content") or ""
            source_block_id = meta.get("sourceBlockId", bid)

            if etype == "title":
                font_size = float(np.clip(round(h * 0.44), 20, 38))
                konva_elements.append({
                    "id": bid,
                    "type": "text",
                    "name": "Poster title",
                    "text": str(content),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(max(80.0, w), 2),
                    "height": round(max(28.0, h), 2),
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": "#ffffff",
                    "fontFamily": "Arial",
                    "fontSize": font_size,
                    "fontWeight": "bold",
                    "align": "center",
                    "lineHeight": 1.15,
                    "sourceBlockId": source_block_id,
                    "semanticRole": "title",
                    "bboxNorm": bbox_norm,
                })
                layer_idx += 1

            elif etype == "author_info":
                font_size = float(np.clip(round(h * 0.40), 12, 18))
                konva_elements.append({
                    "id": bid,
                    "type": "text",
                    "name": "Authors",
                    "text": str(content),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(max(80.0, w), 2),
                    "height": round(max(22.0, h), 2),
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": "#e8eeff",
                    "fontFamily": "Arial",
                    "fontSize": font_size,
                    "fontWeight": "normal",
                    "align": "center",
                    "lineHeight": 1.2,
                    "sourceBlockId": source_block_id,
                    "semanticRole": "author_info",
                    "bboxNorm": bbox_norm,
                })
                layer_idx += 1

            elif etype == "section":
                font_size = float(np.clip(round(h * 0.55), 14, 22))
                konva_elements.append({
                    "id": bid,
                    "type": "text",
                    "name": f"Heading: {str(content)[:24]}",
                    "text": str(content).upper(),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(max(60.0, w), 2),
                    "height": round(max(20.0, h), 2),
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": accent_color,
                    "fontFamily": "Arial",
                    "fontSize": font_size,
                    "fontWeight": "bold",
                    "align": "left",
                    "lineHeight": 1.2,
                    "sourceBlockId": source_block_id,
                    "semanticRole": "section",
                    "bboxNorm": bbox_norm,
                })
                layer_idx += 1

            elif etype in ("text", "list"):
                konva_elements.append({
                    "id": bid,
                    "type": "text",
                    "name": f"Body: {str(content)[:24]}",
                    "text": str(content),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(max(60.0, w), 2),
                    "height": round(max(24.0, h), 2),
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": "#1e293b",
                    "fontFamily": "Arial",
                    "fontSize": 13,
                    "fontWeight": "normal",
                    "align": "left",
                    "lineHeight": 1.38,
                    "sourceBlockId": source_block_id,
                    "semanticRole": etype,
                    "bboxNorm": bbox_norm,
                })
                layer_idx += 1

            elif etype == "caption":
                konva_elements.append({
                    "id": bid,
                    "type": "text",
                    "name": "Caption",
                    "text": str(content),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(max(60.0, w), 2),
                    "height": round(max(18.0, h), 2),
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": "#475569",
                    "fontFamily": "Arial",
                    "fontSize": 11,
                    "fontStyle": "italic",
                    "align": "left",
                    "lineHeight": 1.3,
                    "sourceBlockId": source_block_id,
                    "semanticRole": "caption",
                    "bboxNorm": bbox_norm,
                })
                layer_idx += 1

            elif etype == "figure":
                img_src = meta.get("imageDataUrl")
                if img_src:
                    konva_elements.append({
                        "id": bid,
                        "type": "image",
                        "name": str(content) or "Figure",
                        "src": img_src,
                        "imageFit": "contain",
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "width": round(max(40.0, w), 2),
                        "height": round(max(40.0, h), 2),
                        "rotation": 0,
                        "layer": layer_idx,
                        "visible": True,
                        "locked": False,
                        "opacity": 1.0,
                        "sourceBlockId": source_block_id,
                        "semanticRole": "figure",
                        "bboxNorm": bbox_norm,
                    })
                    layer_idx += 1
                else:
                    konva_elements.append({
                        "id": bid,
                        "type": "rect",
                        "name": f"Figure box: {str(content)[:20]}",
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "width": round(max(40.0, w), 2),
                        "height": round(max(40.0, h), 2),
                        "rotation": 0,
                        "layer": layer_idx,
                        "visible": True,
                        "locked": False,
                        "fill": "#eef2ff",
                        "stroke": accent_color,
                        "strokeWidth": 1.5,
                        "opacity": 1.0,
                        "sourceBlockId": source_block_id,
                        "semanticRole": "figure",
                        "bboxNorm": bbox_norm,
                    })
                    layer_idx += 1
                    konva_elements.append({
                        "id": f"{bid}_lbl",
                        "type": "text",
                        "name": "Figure placeholder label",
                        "text": f"[Figure: {content}]\nUpload or replace image in editor",
                        "x": round(x + 8, 2),
                        "y": round(y + max(8.0, h / 2 - 16), 2),
                        "width": round(max(40.0, w - 16), 2),
                        "height": 36,
                        "rotation": 0,
                        "layer": layer_idx,
                        "visible": True,
                        "locked": False,
                        "fill": "#475569",
                        "fontFamily": "Arial",
                        "fontSize": 11,
                        "align": "center",
                        "lineHeight": 1.3,
                        "sourceBlockId": source_block_id,
                        "semanticRole": "figure_label",
                    })
                    layer_idx += 1

            elif etype == "table":
                konva_elements.append({
                    "id": bid,
                    "type": "rect",
                    "name": f"Table box: {str(content)[:20]}",
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(max(50.0, w), 2),
                    "height": round(max(40.0, h), 2),
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": "#f8fafc",
                    "stroke": "#94a3b8",
                    "strokeWidth": 1.5,
                    "opacity": 1.0,
                    "sourceBlockId": source_block_id,
                    "semanticRole": "table",
                    "bboxNorm": bbox_norm,
                })
                layer_idx += 1
                konva_elements.append({
                    "id": f"{bid}_lbl",
                    "type": "text",
                    "name": "Table label",
                    "text": str(content),
                    "x": round(x + 8, 2),
                    "y": round(y + max(8.0, h / 2 - 12), 2),
                    "width": round(max(40.0, w - 16), 2),
                    "height": 28,
                    "rotation": 0,
                    "layer": layer_idx,
                    "visible": True,
                    "locked": False,
                    "fill": "#334155",
                    "fontFamily": "Arial",
                    "fontSize": 12,
                    "fontWeight": "bold",
                    "align": "center",
                    "sourceBlockId": source_block_id,
                    "semanticRole": "table_label",
                })
                layer_idx += 1

        return konva_elements

    def generate(self, payload: GenerateLayoutRequest) -> dict[str, Any]:
        """Execute retrieval + selected layout generation engine and return structured layout + provenance."""
        if not self.ready or self.retriever is None or self.retriever_info is None:
            raise ServiceNotReadyError(
                self.init_error or "Inference service has not completed model initialization."
            )

        engine = payload.engine or self.default_engine
        if engine not in ("neural", "adaptation"):
            raise InferenceExecutionError(f"Unsupported engine '{engine}'.")
        if engine == "neural" and (self.neural_model is None or self.neural_info is None):
            raise ServiceNotReadyError("Neural layout model is not initialized.")

        t0 = time.perf_counter()
        poster_in = payload.request
        canvas_w, canvas_h = (1000, 700) if poster_in.orientation == "landscape" else (700, 1000)

        req, blocks_meta = self._build_poster_request(poster_in)
        request_summary = req.to_summary_dict()

        cands_df = self.retriever.retrieve_for_request(req, top_k=5, diversify=True)
        if len(cands_df) == 0:
            raise InferenceExecutionError("LayoutRetriever returned zero candidate templates.")

        columns_used = poster_in.preferredColumns or (3 if poster_in.orientation == "landscape" else 2)
        candidates_meta = [
            {
                "rank": int(idx) + 1,
                "poster_id": str(int(row["poster_id"])),
                "orientation": str(row.get("orientation", "")),
                "estimated_column_count": int(row.get("estimated_column_count", columns_used)),
                "relevance_score": round(float(row.get("relevance_score", 0.0)), 4),
            }
            for idx, row in cands_df.iterrows()
        ]

        if engine == "neural":
            assert self.neural_info is not None
            top_pid = int(cands_df.iloc[0]["poster_id"])
            retrieved_template_id = str(top_pid)
            raw_elements, engine_metrics = self._run_neural_engine(
                req=req,
                blocks_meta=blocks_meta,
                template_poster_id=top_pid,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            columns_used = int(engine_metrics.get("predicted_macro_columns", columns_used))
            model_name = self.neural_info.model_family
            model_version = self.neural_info.semantic_version
            model_id = self.neural_info.clearml_model_id
        else:
            raw_elements, engine_metrics, retrieved_template_id, columns_used = (
                self._run_adaptation_engine(
                    req=req,
                    cands_df=cands_df,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )
            )
            model_name = f"{self.retriever_info.model_family}+layout_adapter"
            model_version = self.retriever_info.semantic_version
            model_id = self.retriever_info.clearml_model_id

        validated_elements, geom_metrics = validate_and_normalize_geometry(
            raw_elements=raw_elements,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        generation_metrics = {
            **engine_metrics,
            **geom_metrics,
            "latency_ms": latency_ms,
        }

        konva_elements = self._to_konva_elements(
            raw_elements=validated_elements,
            blocks_meta=blocks_meta,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            accent_color=poster_in.accentColor,
        )

        if engine == "neural" and not self._first_neural_profiled:
            self._first_neural_profiled = True
            self.memory_tracker.record("first_neural_inference")
        elif engine == "adaptation" and not self._first_adaptation_profiled:
            self._first_adaptation_profiled = True
            self.memory_tracker.record("first_adaptation_inference")

        now_iso = datetime.now(timezone.utc).isoformat()
        layout_doc = {
            "id": str(uuid.uuid4()),
            "title": poster_in.title.strip(),
            "width": canvas_w,
            "height": canvas_h,
            "orientation": poster_in.orientation,
            "background": poster_in.background,
            "accentColor": poster_in.accentColor,
            "columns": columns_used,
            "elements": konva_elements,
            "updatedAt": now_iso,
        }

        return {
            "engine": engine,
            "model_name": model_name,
            "model_version": model_version,
            "model_id": model_id,
            "retriever_version": self.retriever_info.semantic_version,
            "retriever_model_id": self.retriever_info.clearml_model_id,
            "retrieved_template_id": retrieved_template_id,
            "candidate_templates": candidates_meta,
            "generation_metrics": generation_metrics,
            "request_features": {
                "summary": request_summary,
                "blocks": [
                    {
                        "block_id": b.block_id,
                        "semantic_role": b.semantic_role,
                        "visual_type": b.visual_type,
                        "word_count": b.word_count,
                        "char_count": b.char_count,
                        "aspect_ratio": b.aspect_ratio,
                        "table_rows": b.table_rows,
                        "table_cols": b.table_cols,
                        "order_index": b.order_index,
                    }
                    for b in req.blocks
                ],
            },
            "layout": layout_doc,
        }

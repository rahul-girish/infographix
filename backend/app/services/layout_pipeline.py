from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.poster import ContentBlockInput, PosterGenerateRequest

logger = logging.getLogger(__name__)


def _word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def _safe_aspect_ratio(val: float | None, default: float = 1.33) -> float:
    if val is None or val <= 0:
        return default
    return float(np.clip(val, 0.25, 4.0))


class LayoutPipelineService:
    """Lazy-loading inference wrapper around Infographix retrieval, HiGHS adaptation, and neural layout models."""

    def __init__(self) -> None:
        self._initialized_retrieval = False
        self._initialized_neural = False
        self.retriever: Any = None
        self.retriever_manifest: Any = None
        self.elements_df: pd.DataFrame | None = None
        self.adapter: Any = None
        self.neural_model: Any = None
        self.neural_device: str = "cpu"

    def _ensure_retrieval_loaded(self) -> None:
        if self._initialized_retrieval:
            return
        self._initialized_retrieval = True

        try:
            from src.adaptation.layout_adapter import LayoutAdapter
            from src.registry.model_registry import ModelRegistry

            registry = ModelRegistry(registry_root=settings.RETRIEVER_REGISTRY_ROOT)
            self.retriever, _, self.retriever_manifest = registry.load_production_model(
                "poster_layout_retriever"
            )

            parquet_path = settings.ELEMENTS_PARQUET_PATH
            if not parquet_path.exists():
                alt = settings.PROJECT_ROOT_DIR / "data" / "processed" / "features" / "v2" / "elements.parquet"
                if alt.exists():
                    parquet_path = alt

            if parquet_path.exists():
                self.elements_df = pd.read_parquet(parquet_path)
                self.adapter = LayoutAdapter(elements_df=self.elements_df)
            else:
                logger.warning("elements.parquet not found at %s; adaptation disabled.", parquet_path)
        except Exception as exc:
            logger.exception("Failed to initialize retrieval/adaptation pipeline: %s", exc)
            self.retriever = None
            self.adapter = None

    def _ensure_neural_loaded(self) -> None:
        if self._initialized_neural:
            return
        self._initialized_neural = True

        ckpt_path = settings.NEURAL_CHECKPOINT_PATH
        if not ckpt_path.exists():
            logger.warning("Neural checkpoint not found at %s", ckpt_path)
            return

        try:
            import torch
            from src.neural_layout.layout_model import TemplateConditionedLayoutModel

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config")
            model = TemplateConditionedLayoutModel(
                d_model=getattr(cfg, "d_model", 128),
                num_heads=getattr(cfg, "num_heads", 4),
                encoder_layers=getattr(cfg, "encoder_layers", 3),
                decoder_layers=getattr(cfg, "decoder_layers", 3),
                ffn_dim=getattr(cfg, "ffn_dim", 256),
                dropout=0.0,
                bbox_parameterization=getattr(cfg, "bbox_parameterization", "independent"),
                use_anchor_conditioning=getattr(cfg, "use_anchor_conditioning", False),
                min_box_width=getattr(cfg, "min_box_width", 0.01),
                min_box_height=getattr(cfg, "min_box_height", 0.005),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.neural_model = model
        except Exception as exc:
            logger.exception("Failed to load neural layout model: %s", exc)
            self.neural_model = None

    def status(self) -> dict[str, Any]:
        self._ensure_retrieval_loaded()
        return {
            "retrieverLoaded": self.retriever is not None,
            "retrieverVersion": getattr(self.retriever_manifest, "version", None),
            "adapterReady": self.adapter is not None,
            "neuralCheckpointExists": settings.NEURAL_CHECKPOINT_PATH.exists(),
            "neuralLoaded": self.neural_model is not None,
        }

    def _build_poster_request(
        self, payload: PosterGenerateRequest
    ) -> tuple[Any, dict[str, dict[str, Any]]]:
        from src.features.request_schema import ContentBlockDemand, PosterRequest

        aspect_ratio = 1.414 if payload.orientation == "landscape" else 0.707
        blocks_meta: dict[str, dict[str, Any]] = {}
        demands: list[ContentBlockDemand] = []

        # 1. Title & Authors
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
        blocks_meta["b_title"] = {"role": "title", "content": title_text, "source": payload}

        author_line = " — ".join(
            part for part in [payload.authors.strip(), payload.affiliations.strip()] if part
        ) or "Author Name"
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

        # 2. User content blocks
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
                    blocks_meta[hid] = {"role": "section", "content": h_txt, "sourceBlockId": blk.id}
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
                blocks_meta[pid] = {"role": vis_type, "content": body_txt, "sourceBlockId": blk.id}
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
                    blocks_meta[cid] = {"role": "caption", "content": c_txt, "sourceBlockId": blk.id}
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
                    blocks_meta[cid] = {"role": "caption", "content": c_txt, "sourceBlockId": blk.id}
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
                blocks_meta[cid] = {"role": "caption", "content": c_txt, "sourceBlockId": blk.id}
                order += 1

        req = PosterRequest(
            request_id=f"req_{uuid.uuid4().hex[:10]}",
            requested_aspect_ratio=aspect_ratio,
            requested_orientation=payload.orientation,
            preferred_column_count=payload.preferredColumns,
            blocks=demands,
            author_count=max(1, len([a for a in payload.authors.split(",") if a.strip()])),
            affiliation_count=max(1, len([a for a in payload.affiliations.split(",") if a.strip()])),
        )
        return req, blocks_meta

    def _to_konva_elements(
        self,
        raw_elements: list[dict[str, Any]],
        blocks_meta: dict[str, dict[str, Any]],
        canvas_w: int,
        canvas_h: int,
        accent_color: str,
    ) -> list[dict[str, Any]]:
        """Convert normalized/pixel layout elements into styled Konva PosterElements."""
        konva_elements: list[dict[str, Any]] = []
        layer_idx = 0

        # Identify header region (title + author_info) to place a clean header banner behind them
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
            bbox_norm = [round(float(v), 4) for v in el.get("bbox_norm", [x / canvas_w, y / canvas_h, w / canvas_w, h / canvas_h])]
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

    def _run_neural_inference(
        self,
        req: Any,
        blocks_meta: dict[str, dict[str, Any]],
        template_poster_id: int,
        canvas_w: int,
        canvas_h: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import torch
        from src.neural_layout.dataset import CATEGORY_TO_IDX, TemplateSlotItem, UserContentItem

        self._ensure_neural_loaded()
        if self.neural_model is None or self.elements_df is None:
            raise RuntimeError("Neural layout model or elements_df is not available.")

        # Build content_feats [1, M, 18]
        total_items = max(1, len(req.blocks))
        content_vecs: list[np.ndarray] = []
        for idx, blk in enumerate(req.blocks):
            cat = blk.scipost_category if blk.scipost_category in CATEGORY_TO_IDX else "text"
            raw_txt = blocks_meta.get(blk.block_id, {}).get("content")
            has_txt = isinstance(raw_txt, str) and len(raw_txt.strip()) > 0 and cat not in ("figure", "table")
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

        # Build template_feats [1, K, 16] from retrieved template elements
        tmpl_rows = self.elements_df[self.elements_df["poster_id"] == template_poster_id]
        tmpl_vecs: list[np.ndarray] = []
        for _, row in tmpl_rows.iterrows():
            cat = str(row.get("category_slug", "text"))
            if cat not in CATEGORY_TO_IDX:
                continue
            x0 = float(row.get("x0", 0.0))
            y0 = float(row.get("y0", 0.0))
            wn = float(row.get("w_norm", 0.2))
            hn = float(row.get("h_norm", 0.1))
            cx = float(row.get("cx", x0 + 0.5 * wn))
            cy = float(row.get("cy", y0 + 0.5 * hn))
            area = float(row.get("area_norm", wn * hn))
            slot = TemplateSlotItem(
                slot_id=str(row.get("element_id", "0")),
                category_slug=cat,
                category_idx=CATEGORY_TO_IDX[cat],
                bbox_norm=[x0, y0, wn, hn],
                center=(cx, cy),
                area_norm=area,
            )
            tmpl_vecs.append(slot.to_feature_vector())

        if not tmpl_vecs:
            raise RuntimeError(f"No valid template slots found for poster_id={template_poster_id}")

        content_tensor = torch.tensor(np.stack(content_vecs), dtype=torch.float32).unsqueeze(0)
        template_tensor = torch.tensor(np.stack(tmpl_vecs), dtype=torch.float32).unsqueeze(0)
        content_mask = torch.ones((1, len(content_vecs)), dtype=torch.bool)
        template_mask = torch.ones((1, len(tmpl_vecs)), dtype=torch.bool)

        with torch.no_grad():
            preds = self.neural_model(content_tensor, template_tensor, content_mask, template_mask)

        pred_boxes = preds["pred_boxes"][0].cpu().numpy()
        pred_macro = int(torch.argmax(preds["pred_macro_column_logits"][0]).item()) + 1

        raw_elements: list[dict[str, Any]] = []
        for idx, blk in enumerate(req.blocks):
            x0, y0, wn, hn = [float(v) for v in pred_boxes[idx]]
            x0 = float(np.clip(x0, 0.02, 0.90))
            y0 = float(np.clip(y0, 0.02, 0.90))
            wn = float(np.clip(wn, 0.08, 0.96 - x0))
            hn = float(np.clip(hn, 0.03, 0.96 - y0))
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
        }
        return raw_elements, metrics

    def _fallback_grid_layout(
        self,
        req: Any,
        canvas_w: int,
        canvas_h: int,
    ) -> list[dict[str, Any]]:
        """Simple deterministic fallback if model artifacts are unavailable on disk."""
        cols = req.preferred_column_count or (3 if canvas_w >= canvas_h else 2)
        margin = 36.0
        header_h = 128.0
        col_gap = 22.0
        col_w = (canvas_w - 2 * margin - (cols - 1) * col_gap) / cols
        col_y = [margin + header_h + 24.0 for _ in range(cols)]

        raw: list[dict[str, Any]] = []
        body_idx = 0
        for blk in req.blocks:
            cat = blk.scipost_category
            if cat == "title":
                bbox = [margin + 20, margin + 14, canvas_w - 2 * margin - 40, 56.0]
            elif cat == "author_info":
                bbox = [margin + 24, margin + 78, canvas_w - 2 * margin - 48, 34.0]
            else:
                c = body_idx % cols
                if cat == "section":
                    h = 28.0
                elif cat == "figure":
                    h = col_w / _safe_aspect_ratio(blk.aspect_ratio, 1.33)
                elif cat == "table":
                    h = 130.0
                elif cat == "caption":
                    h = 26.0
                else:
                    words = blk.word_count or 40
                    h = max(64.0, min(220.0, words * 1.9))
                x = margin + c * (col_w + col_gap)
                y = col_y[c]
                bbox = [x, y, col_w, h]
                col_y[c] += h + 14.0
                if cat != "section":
                    body_idx += 1

            raw.append({
                "id": blk.block_id,
                "type": cat,
                "bbox": bbox,
                "bbox_norm": [
                    bbox[0] / canvas_w,
                    bbox[1] / canvas_h,
                    bbox[2] / canvas_w,
                    bbox[3] / canvas_h,
                ],
            })
        return raw

    def generate_poster(self, payload: PosterGenerateRequest) -> dict[str, Any]:
        """Run the AI layout pipeline and return the generated PosterDocument + MLOps telemetry."""
        t0 = time.perf_counter()
        self._ensure_retrieval_loaded()

        canvas_w, canvas_h = (1000, 700) if payload.orientation == "landscape" else (700, 1000)
        req, blocks_meta = self._build_poster_request(payload)
        request_summary = req.to_summary_dict()

        engine = payload.engine or settings.DEFAULT_ENGINE
        model_name = "poster_layout_retriever+layout_adapter"
        model_version = getattr(self.retriever_manifest, "version", "v2.0.0") or "v2.0.0"
        retrieved_template_id: str | None = None
        candidates_meta: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}
        columns_used = payload.preferredColumns or (3 if payload.orientation == "landscape" else 2)
        raw_elements: list[dict[str, Any]] = []

        if self.retriever is not None and self.elements_df is not None:
            cands_df = self.retriever.retrieve_for_request(req, top_k=5, diversify=True)
            for idx, row in cands_df.iterrows():
                candidates_meta.append({
                    "rank": int(idx) + 1,
                    "poster_id": str(int(row["poster_id"])),
                    "orientation": str(row.get("orientation", "")),
                    "estimated_column_count": int(row.get("estimated_column_count", columns_used)),
                    "relevance_score": round(float(row.get("relevance_score", 0.0)), 4),
                })

            if engine == "neural":
                try:
                    top_pid = int(cands_df.iloc[0]["poster_id"])
                    retrieved_template_id = str(top_pid)
                    model_name = "TemplateConditionedLayoutModel"
                    model_version = "v0.1.0"
                    raw_elements, metrics = self._run_neural_inference(
                        req=req,
                        blocks_meta=blocks_meta,
                        template_poster_id=top_pid,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                    )
                    columns_used = int(metrics.get("predicted_macro_columns", columns_used))
                except Exception as exc:
                    logger.warning("Neural inference failed (%s); falling back to adaptation.", exc)
                    engine = "adaptation"

            if engine == "adaptation" and self.adapter is not None:
                self.adapter.default_width = canvas_w
                self.adapter.default_height = canvas_h
                best, all_results = self.adapter.adapt_and_select_best(req, cands_df)
                retrieved_template_id = str(best.template_poster_id)
                columns_used = len(best.hierarchical_template.columns) or columns_used
                metrics = {
                    "is_feasible": bool(best.is_feasible),
                    "composite_quality_score": round(float(best.evaluation_metrics.get("composite_quality_score", 0.0)), 3),
                    "overflow_rate": round(float(best.evaluation_metrics.get("overflow_rate", 0.0)), 4),
                    "overlap_area": round(float(best.evaluation_metrics.get("overlap_area", 0.0)), 6),
                    "whitespace_utilization": round(float(best.evaluation_metrics.get("whitespace_utilization", 0.0)), 4),
                    "candidates_evaluated": len(all_results),
                }
                for el in best.poster_document.elements:
                    # Rescale to exact canvas_w x canvas_h
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

        if not raw_elements:
            model_name = "deterministic_grid_fallback"
            model_version = "v0.0.0"
            raw_elements = self._fallback_grid_layout(req, canvas_w, canvas_h)
            metrics["fallback_used"] = True

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        metrics["latency_ms"] = latency_ms

        konva_elements = self._to_konva_elements(
            raw_elements=raw_elements,
            blocks_meta=blocks_meta,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            accent_color=payload.accentColor,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        poster_doc = {
            "id": str(uuid.uuid4()),
            "title": payload.title.strip(),
            "width": canvas_w,
            "height": canvas_h,
            "orientation": payload.orientation,
            "background": payload.background,
            "accentColor": payload.accentColor,
            "columns": columns_used,
            "elements": konva_elements,
            "updatedAt": now_iso,
        }

        return {
            "engine": engine,
            "modelName": model_name,
            "modelVersion": model_version,
            "retrievedTemplateId": retrieved_template_id,
            "candidateTemplates": candidates_meta,
            "generationMetrics": metrics,
            "requestFeatures": {
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
            "poster": poster_doc,
        }


layout_pipeline = LayoutPipelineService()

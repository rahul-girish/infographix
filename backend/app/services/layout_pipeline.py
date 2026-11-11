from __future__ import annotations

import logging
from typing import Any

import requests
from fastapi import HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.poster import PosterGenerateRequest

logger = logging.getLogger(__name__)


class LayoutPipelineService:
    """HTTP client delegating poster layout generation to the standalone inference_service."""

    def _base_url(self) -> str:
        url = (settings.INFERENCE_SERVICE_URL or "").strip().rstrip("/")
        if not url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="INFERENCE_SERVICE_URL is not configured on the application backend.",
            )
        return url

    def status(self) -> dict[str, Any]:
        url = (settings.INFERENCE_SERVICE_URL or "").strip().rstrip("/")
        if not url:
            return {"configured": False, "status": "unconfigured"}
        try:
            resp = requests.get(f"{url}/health/ready", timeout=5.0)
            data = resp.json() if resp.content else {}
            return {
                "configured": True,
                "url": url,
                "http_status": resp.status_code,
                **data,
            }
        except Exception as exc:
            return {
                "configured": True,
                "url": url,
                "status": "unreachable",
                "error": exc.__class__.__name__,
            }

    def generate_poster(self, payload: PosterGenerateRequest) -> dict[str, Any]:
        """Forward user poster request to inference_service and re-attach client images."""
        base_url = self._base_url()

        # Strip inline base64 image blobs before remote inference; re-attach locally to figure boxes
        client_images: dict[str, str] = {}
        sanitized_blocks = []
        for blk in payload.blocks:
            b_dict = blk.model_dump()
            if b_dict.get("imageDataUrl"):
                client_images[blk.id] = b_dict["imageDataUrl"]
                b_dict["imageDataUrl"] = None
            sanitized_blocks.append(b_dict)

        inference_body = {
            "request": {
                "title": payload.title,
                "authors": payload.authors,
                "affiliations": payload.affiliations,
                "orientation": payload.orientation,
                "preferredColumns": payload.preferredColumns,
                "accentColor": payload.accentColor,
                "background": payload.background,
                "blocks": sanitized_blocks,
            },
            "engine": payload.engine or settings.DEFAULT_ENGINE,
        }

        try:
            resp = requests.post(
                f"{base_url}/v1/layouts/generate",
                json=inference_body,
                timeout=settings.INFERENCE_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference service timed out generating the poster layout.",
            ) from exc
        except requests.RequestException as exc:
            logger.error("Failed reaching inference service: %s", exc.__class__.__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not connect to the inference service.",
            ) from exc

        if resp.status_code >= 400:
            detail = "Inference service failed to generate layout."
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and isinstance(err_json.get("detail"), str):
                    detail = err_json["detail"]
            except Exception:
                pass
            mapped_code = (
                resp.status_code
                if resp.status_code in (400, 422, 503)
                else status.HTTP_502_BAD_GATEWAY
            )
            raise HTTPException(status_code=mapped_code, detail=detail)

        data = resp.json()
        poster_doc = dict(data["layout"])

        # Re-attach client-side figure images onto matching figure slots
        if client_images:
            updated_elements = []
            for el in poster_doc.get("elements", []):
                src_id = el.get("sourceBlockId") or el.get("id")
                role = el.get("semanticRole")
                if src_id in client_images and role == "figure":
                    updated_elements.append({
                        **el,
                        "type": "image",
                        "name": el.get("name", "Figure").replace("Figure box: ", ""),
                        "src": client_images[src_id],
                        "imageFit": "contain",
                        "fill": None,
                        "stroke": None,
                        "strokeWidth": None,
                    })
                    continue
                if src_id in client_images and role == "figure_label":
                    continue
                updated_elements.append(el)
            poster_doc["elements"] = updated_elements

        metrics = dict(data.get("generation_metrics") or {})
        if data.get("model_id"):
            metrics["clearml_model_id"] = data["model_id"]
        if data.get("retriever_model_id"):
            metrics["retriever_model_id"] = data["retriever_model_id"]
        if data.get("retriever_version"):
            metrics["retriever_version"] = data["retriever_version"]

        return {
            "engine": data["engine"],
            "modelName": data["model_name"],
            "modelVersion": data["model_version"],
            "retrievedTemplateId": data.get("retrieved_template_id"),
            "candidateTemplates": data.get("candidate_templates", []),
            "generationMetrics": metrics,
            "requestFeatures": data.get("request_features", {}),
            "poster": poster_doc,
        }


layout_pipeline = LayoutPipelineService()

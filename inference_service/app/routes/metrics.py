"""Prometheus /metrics scrape endpoint for the Infographix Inference Service."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response, status

router = APIRouter(tags=["Observability"])


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics(request: Request) -> Response:
    """Expose Prometheus metrics in standard text exposition format."""
    settings = request.app.state.settings
    expected_token = (settings.METRICS_AUTH_TOKEN or "").strip()
    if expected_token:
        auth_header = request.headers.get("authorization", "")
        prefix = "Bearer "
        provided = auth_header[len(prefix) :].strip() if auth_header.startswith(prefix) else ""
        if not provided or not hmac.compare_digest(provided, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing metrics bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    metrics = request.app.state.metrics
    manager = request.app.state.inference_manager
    metrics.update_runtime_gauges(is_ready=bool(getattr(manager, "is_ready", False)))
    payload, content_type = metrics.render_prometheus()
    return Response(content=payload, media_type=content_type)

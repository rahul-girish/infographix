import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.generation_run import GenerationRun
from backend.app.schemas.poster import (
    GenerationRunOut,
    GenerationRunSummaryOut,
    PdfExportRequest,
    PosterDocumentSchema,
    PosterGenerateRequest,
    RunUpdateRequest,
)
from backend.app.services.layout_pipeline import layout_pipeline
from backend.app.services.pdf_export_service import export_canvas_png_to_pdf

router = APIRouter(tags=["Generation & MLOps"])


def _sanitize_input_for_training(payload: PosterGenerateRequest) -> dict[str, Any]:
    """Strip large inline base64 image blobs before persisting training telemetry."""
    raw = payload.model_dump()
    for blk in raw.get("blocks", []):
        if blk.get("imageDataUrl"):
            blk["hasImage"] = True
            blk["imageDataUrl"] = None
    return raw


def _sanitize_layout_for_training(layout_doc: dict[str, Any]) -> dict[str, Any]:
    """Keep geometric bounding boxes and text/metadata while stripping large inline image payloads."""
    cleaned = dict(layout_doc)
    elements = []
    canvas_w = max(1.0, float(cleaned.get("width", 1000)))
    canvas_h = max(1.0, float(cleaned.get("height", 700)))
    for el in cleaned.get("elements", []):
        item = dict(el)
        if isinstance(item.get("src"), str) and item["src"].startswith("data:"):
            item["src"] = "[inline-image-omitted]"
        # Always keep updated normalized bbox for downstream training
        item["bboxNorm"] = [
            round(float(item.get("x", 0.0)) / canvas_w, 4),
            round(float(item.get("y", 0.0)) / canvas_h, 4),
            round(float(item.get("width", 0.0)) / canvas_w, 4),
            round(float(item.get("height", 0.0)) / canvas_h, 4),
        ]
        elements.append(item)
    cleaned["elements"] = elements
    return cleaned


def _format_run_out(run: GenerationRun, poster_override: Optional[dict[str, Any]] = None) -> GenerationRunOut:
    active_layout = poster_override or run.edited_layout or run.generated_layout
    return GenerationRunOut(
        id=str(run.id),
        createdAt=run.created_at.isoformat(),
        updatedAt=run.updated_at.isoformat(),
        title=run.title,
        authors=run.authors,
        orientation=run.orientation,
        engine=run.engine,
        modelName=run.model_name,
        modelVersion=run.model_version,
        retrievedTemplateId=run.retrieved_template_id,
        candidateTemplates=run.candidate_templates or [],
        generationMetrics=run.generation_metrics or {},
        poster=PosterDocumentSchema(**active_layout),
        wasEdited=run.was_edited,
        exportedFormats=run.exported_formats or [],
        feedbackRating=run.feedback_rating,
        feedbackNotes=run.feedback_notes,
    )


@router.post("/generate", response_model=GenerationRunOut, status_code=status.HTTP_201_CREATED)
def generate_poster(payload: PosterGenerateRequest, db: Session = Depends(get_db)):
    """Run the AI layout pipeline on user content and persist the run to Supabase."""
    result = layout_pipeline.generate_poster(payload)
    poster_doc = result["poster"]

    run = GenerationRun(
        title=payload.title.strip(),
        authors=payload.authors.strip(),
        orientation=payload.orientation,
        engine=result["engine"],
        model_name=result["modelName"],
        model_version=result["modelVersion"],
        retrieved_template_id=result["retrievedTemplateId"],
        candidate_templates=result["candidateTemplates"],
        generation_metrics=result["generationMetrics"],
        input_payload=_sanitize_input_for_training(payload),
        request_features=result["requestFeatures"],
        generated_layout=_sanitize_layout_for_training(poster_doc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return _format_run_out(run, poster_override=poster_doc)


@router.get("/runs", response_model=List[GenerationRunSummaryOut])
def list_runs(
    limit: int = Query(default=25, ge=1, le=200),
    edited_only: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """List recent inference runs stored in Supabase."""
    query = db.query(GenerationRun)
    if edited_only:
        query = query.filter(GenerationRun.was_edited.is_(True))
    runs = query.order_by(desc(GenerationRun.created_at)).limit(limit).all()
    return [
        GenerationRunSummaryOut(
            id=str(r.id),
            createdAt=r.created_at.isoformat(),
            title=r.title,
            orientation=r.orientation,
            engine=r.engine,
            modelName=r.model_name,
            modelVersion=r.model_version,
            retrievedTemplateId=r.retrieved_template_id,
            wasEdited=r.was_edited,
            exportedFormats=r.exported_formats or [],
            feedbackRating=r.feedback_rating,
            generationMetrics=r.generation_metrics or {},
        )
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=GenerationRunOut)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    """Fetch a single generation run and its latest layout."""
    run = db.query(GenerationRun).filter(GenerationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found.")
    return _format_run_out(run)


@router.patch("/runs/{run_id}", response_model=GenerationRunOut)
def update_run(run_id: uuid.UUID, payload: RunUpdateRequest, db: Session = Depends(get_db)):
    """Record user edits, export events, or quality feedback for MLOps training curation."""
    run = db.query(GenerationRun).filter(GenerationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found.")

    if payload.editedLayout is not None:
        run.edited_layout = _sanitize_layout_for_training(payload.editedLayout)
        run.was_edited = True

    if payload.exportedFormat is not None:
        formats = list(run.exported_formats or [])
        if payload.exportedFormat not in formats:
            formats.append(payload.exportedFormat)
            run.exported_formats = formats

    if payload.feedbackRating is not None:
        run.feedback_rating = payload.feedbackRating

    if payload.feedbackNotes is not None:
        run.feedback_notes = payload.feedbackNotes

    run.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return _format_run_out(run)


@router.get("/runs/export/training-data")
def export_training_data(
    format: str = Query(default="jsonl", pattern="^(jsonl|json)$"),
    edited_only: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    """Export collected (input, model output, human-edited output) pairs for model retraining."""
    query = db.query(GenerationRun)
    if edited_only:
        query = query.filter(GenerationRun.was_edited.is_(True))
    runs = query.order_by(desc(GenerationRun.created_at)).limit(limit).all()

    records = [
        {
            "run_id": str(r.id),
            "created_at": r.created_at.isoformat(),
            "engine": r.engine,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "retrieved_template_id": r.retrieved_template_id,
            "generation_metrics": r.generation_metrics,
            "input_payload": r.input_payload,
            "request_features": r.request_features,
            "generated_layout": r.generated_layout,
            "edited_layout": r.edited_layout,
            "was_edited": r.was_edited,
            "exported_formats": r.exported_formats,
            "feedback_rating": r.feedback_rating,
            "feedback_notes": r.feedback_notes,
        }
        for r in runs
    ]

    if format == "json":
        return {"count": len(records), "runs": records}

    jsonl_body = "\n".join(json.dumps(rec) for rec in records)
    return Response(
        content=jsonl_body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="infographix_training_runs.jsonl"'},
    )


@router.post("/export/pdf")
def export_pdf(payload: PdfExportRequest, db: Session = Depends(get_db)):
    """Convert the edited Konva canvas snapshot into a downloadable PDF and log the export."""
    if not payload.imageDataUrl:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="imageDataUrl is required to export PDF.",
        )

    pdf_bytes = export_canvas_png_to_pdf(payload.imageDataUrl)

    if payload.runId:
        try:
            run_uuid = uuid.UUID(payload.runId)
            run = db.query(GenerationRun).filter(GenerationRun.id == run_uuid).first()
            if run:
                formats = list(run.exported_formats or [])
                if "pdf" not in formats:
                    formats.append("pdf")
                    run.exported_formats = formats
                    run.updated_at = datetime.now(timezone.utc)
                    db.commit()
        except ValueError:
            pass

    safe_name = re.sub(r"[^a-z0-9]+", "-", payload.title.lower()).strip("-") or "infographix"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}-poster.pdf"'},
    )

import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base

JSONVariant = JSON().with_variant(JSONB, "postgresql")


class GenerationRun(Base):
    """Stores every poster layout inference run and subsequent human edits in Supabase
    for MLOps observability and future training dataset curation."""

    __tablename__ = "generation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Core poster metadata
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    authors: Mapped[str] = mapped_column(String(500), default="")
    orientation: Mapped[str] = mapped_column(String(32), default="landscape")
    engine: Mapped[str] = mapped_column(String(64), default="adaptation")

    # Model & provenance metadata
    model_name: Mapped[str] = mapped_column(String(128), default="poster_layout_retriever+layout_adapter")
    model_version: Mapped[str] = mapped_column(String(64), default="v2.0.0")
    retrieved_template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_templates: Mapped[list] = mapped_column(JSONVariant, default=list)
    generation_metrics: Mapped[dict] = mapped_column(JSONVariant, default=dict)

    # Raw inputs & engineered features
    input_payload: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    request_features: Mapped[dict] = mapped_column(JSONVariant, default=dict)

    # Raw model output vs. human-edited output (HITL training pair)
    generated_layout: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    edited_layout: Mapped[dict | None] = mapped_column(JSONVariant, nullable=True)
    was_edited: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Downstream usage / feedback signals
    exported_formats: Mapped[list] = mapped_column(JSONVariant, default=list)
    feedback_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

import uuid
from datetime import datetime, timezone
from typing import Any, List
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from backend.app.db.base import Base

class PosterDocument(Base):
    __tablename__ = "poster_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=700, nullable=False)
    orientation: Mapped[str] = mapped_column(String(20), default="landscape", nullable=False)  # "landscape" | "portrait"
    background: Mapped[str] = mapped_column(String(50), default="#ffffff", nullable=False)
    template_id: Mapped[str] = mapped_column(String(50), default="research", nullable=False)
    columns: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    elements: Mapped[List[Any]] = mapped_column(JSONB, default=list, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    project = relationship("Project", back_populates="poster")


"""Append-only governance audit logging service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session, sessionmaker

from backend.app.governance.models import AuditEvent


class AuditService:
    """Append-only audit trail writer and read-only query service.

    Exposes only INSERT (record/record_isolated) and SELECT (query) operations.
    """

    @staticmethod
    def record(
        db: Session,
        *,
        actor_id: str,
        actor_role: str,
        action: str,
        entity_type: str,
        entity_id: str,
        correlation_id: str,
        outcome: str = "success",
        previous_state: Optional[dict[str, Any]] = None,
        new_state: Optional[dict[str, Any]] = None,
        reason: Optional[str] = None,
        related_model_version: Optional[str] = None,
        related_dataset_version: Optional[str] = None,
    ) -> AuditEvent:
        """Append an audit event inside the caller's active DB transaction."""
        event = AuditEvent(
            actor_id=actor_id,
            actor_role=actor_role,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id),
            related_model_version=related_model_version,
            related_dataset_version=related_dataset_version,
            previous_state=previous_state,
            new_state=new_state,
            reason=reason,
            outcome=outcome,
            correlation_id=correlation_id,
        )
        db.add(event)
        db.flush()
        return event

    @staticmethod
    def record_isolated(
        db: Session,
        *,
        actor_id: str,
        actor_role: str,
        action: str,
        entity_type: str,
        entity_id: str,
        correlation_id: str,
        outcome: str = "denied",
        previous_state: Optional[dict[str, Any]] = None,
        new_state: Optional[dict[str, Any]] = None,
        reason: Optional[str] = None,
        related_model_version: Optional[str] = None,
        related_dataset_version: Optional[str] = None,
    ) -> None:
        """Persist a denied/failed audit event in an independent session so rollbacks never erase it."""
        bind = db.get_bind()
        if bind is None:
            return
        if db.in_transaction():
            db.rollback()
        iso_factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
        iso_db = iso_factory()
        try:
            event = AuditEvent(
                actor_id=actor_id,
                actor_role=actor_role,
                action=action,
                entity_type=entity_type,
                entity_id=str(entity_id),
                related_model_version=related_model_version,
                related_dataset_version=related_dataset_version,
                previous_state=previous_state,
                new_state=new_state,
                reason=reason,
                outcome=outcome,
                correlation_id=correlation_id,
            )
            iso_db.add(event)
            iso_db.commit()
        finally:
            iso_db.close()

    @staticmethod
    def query(
        db: Session,
        *,
        action: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        outcome: Optional[str] = None,
        correlation_id: Optional[str] = None,
        related_model_version: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Read-only filtered and paginated audit event query."""
        q = db.query(AuditEvent)
        if action:
            q = q.filter(AuditEvent.action == action)
        if entity_type:
            q = q.filter(AuditEvent.entity_type == entity_type)
        if entity_id:
            q = q.filter(AuditEvent.entity_id == entity_id)
        if actor_id:
            q = q.filter(AuditEvent.actor_id == actor_id)
        if outcome:
            q = q.filter(AuditEvent.outcome == outcome)
        if correlation_id:
            q = q.filter(AuditEvent.correlation_id == correlation_id)
        if related_model_version:
            q = q.filter(AuditEvent.related_model_version == related_model_version)
        if since:
            q = q.filter(AuditEvent.timestamp >= since)
        if until:
            q = q.filter(AuditEvent.timestamp <= until)

        return (
            q.order_by(desc(AuditEvent.timestamp))
            .offset(max(0, offset))
            .limit(min(200, max(1, limit)))
            .all()
        )

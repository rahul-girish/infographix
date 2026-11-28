"""ClearML Native Model Registry adapter with deterministic offline mock fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.app.governance.config import governance_settings


@dataclass
class ClearMLVerificationResult:
    verified: bool
    is_mock: bool
    clearml_model_id: str
    lifecycle_status: str
    details: dict[str, Any]


class ClearMLRegistryProtocol(Protocol):
    """Interface decoupling governance services from ClearML network SDK calls."""

    def verify_model_artifact(
        self, *, clearml_model_id: str, expected_sha256: str
    ) -> ClearMLVerificationResult: ...

    def sync_lifecycle_status(
        self,
        *,
        clearml_model_id: str,
        target_status: str,
        model_family: str,
    ) -> ClearMLVerificationResult: ...


class MockClearMLRegistryAdapter:
    """Offline deterministic adapter clearly marked as mock for dev & automated tests."""

    def __init__(self) -> None:
        self._synced_stages: dict[str, str] = {}

    def verify_model_artifact(
        self, *, clearml_model_id: str, expected_sha256: str
    ) -> ClearMLVerificationResult:
        return ClearMLVerificationResult(
            verified=bool(clearml_model_id and len(expected_sha256) == 64),
            is_mock=True,
            clearml_model_id=clearml_model_id,
            lifecycle_status=self._synced_stages.get(clearml_model_id, "unknown"),
            details={
                "mock_adapter": True,
                "checked_sha256": expected_sha256,
            },
        )

    def sync_lifecycle_status(
        self,
        *,
        clearml_model_id: str,
        target_status: str,
        model_family: str,
    ) -> ClearMLVerificationResult:
        self._synced_stages[clearml_model_id] = target_status
        return ClearMLVerificationResult(
            verified=True,
            is_mock=True,
            clearml_model_id=clearml_model_id,
            lifecycle_status=target_status,
            details={"mock_adapter": True, "model_family": model_family},
        )


class LiveClearMLRegistryAdapter:
    """Adapter delegating to src.registry.clearml_registry when ClearML mode is enabled."""

    def verify_model_artifact(
        self, *, clearml_model_id: str, expected_sha256: str
    ) -> ClearMLVerificationResult:
        from src.registry.clearml_registry import query_registered_models

        # Query ClearML metadata by model ID via InputModel without downloading large weights on every check
        from clearml import InputModel

        model_obj = InputModel(model_id=clearml_model_id)
        meta = model_obj.get_all_metadata() or {}
        sha_entry = meta.get("checkpoint_sha256")
        recorded_sha = (
            sha_entry.get("value") if isinstance(sha_entry, dict) else sha_entry
        ) or ""
        verified = bool(recorded_sha) and recorded_sha.lower() == expected_sha256.lower()
        stage_entry = meta.get("lifecycle_status")
        stage = (stage_entry.get("value") if isinstance(stage_entry, dict) else stage_entry) or "unknown"
        return ClearMLVerificationResult(
            verified=verified,
            is_mock=False,
            clearml_model_id=clearml_model_id,
            lifecycle_status=str(stage),
            details={"clearml_recorded_sha256": recorded_sha, "queried": bool(query_registered_models)},
        )

    def sync_lifecycle_status(
        self,
        *,
        clearml_model_id: str,
        target_status: str,
        model_family: str,
    ) -> ClearMLVerificationResult:
        from src.registry.clearml_registry import set_model_lifecycle_status

        res = set_model_lifecycle_status(
            target_status=target_status,
            model_id=clearml_model_id,
            project_name=governance_settings.GOV_CLEARML_PROJECT_NAME,
            model_family=model_family,
        )
        return ClearMLVerificationResult(
            verified=True,
            is_mock=False,
            clearml_model_id=clearml_model_id,
            lifecycle_status=str(res.get("lifecycle_status", target_status)),
            details={"tags": res.get("tags", [])},
        )


_default_mock_adapter = MockClearMLRegistryAdapter()


def get_clearml_adapter() -> ClearMLRegistryProtocol:
    """Return live ClearML adapter or deterministic mock adapter based on GOV_CLEARML_MODE."""
    if governance_settings.GOV_CLEARML_MODE == "clearml":
        return LiveClearMLRegistryAdapter()
    return _default_mock_adapter

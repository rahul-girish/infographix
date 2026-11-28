"""Governance configuration loaded from environment variables."""

from __future__ import annotations

import json
from typing import Dict, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app.core.config import BACKEND_DIR


class GovernanceSettings(BaseSettings):
    """Environment-driven settings for the MLOps Governance module."""

    # Authentication mode:
    # - "strict": requires valid Bearer JWT or scoped Service Token (fail-closed default)
    # - "dev": also permits X-Actor-Id / X-Actor-Role headers for local development/testing
    GOV_AUTH_MODE: Literal["strict", "dev"] = "strict"

    # HS256 JWT verification secret (compatible with Supabase Auth or custom JWT issuer)
    GOV_JWT_SECRET: str = ""
    GOV_JWT_AUDIENCE: str = ""

    # JSON mapping of static/service tokens to actor metadata:
    # e.g. '{"ci-promote-token": {"actor_id": "ci-cd-promote", "role": "admin"}}'
    GOV_SERVICE_TOKENS: str = "{}"

    # ClearML registry adapter mode:
    # - "mock": offline deterministic adapter (safe default when ClearML keys are unset)
    # - "clearml": delegates to src.registry.clearml_registry
    GOV_CLEARML_MODE: Literal["mock", "clearml"] = "mock"
    GOV_CLEARML_PROJECT_NAME: str = "Infographix"

    # Auto-seed default governance policies on FastAPI lifespan startup if missing
    GOV_SEED_DEFAULT_POLICIES: bool = True

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def parsed_service_tokens(self) -> Dict[str, dict]:
        """Parse GOV_SERVICE_TOKENS JSON into a token -> actor descriptor dict."""
        raw = (self.GOV_SERVICE_TOKENS or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        result: Dict[str, dict] = {}
        for token, meta in data.items():
            if not isinstance(token, str) or not token.strip():
                continue
            if isinstance(meta, dict) and meta.get("actor_id") and meta.get("role"):
                result[token.strip()] = {
                    "actor_id": str(meta["actor_id"]),
                    "role": str(meta["role"]),
                }
        return result


governance_settings = GovernanceSettings()

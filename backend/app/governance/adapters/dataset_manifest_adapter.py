"""Adapter for reading .mydvc/manifests/*.manifest and feature_report.json metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from backend.app.governance.exceptions import ResourceNotFoundError, ValidationError


def parse_minidvc_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Parse a MINIDVC-MANIFEST file into deterministic summary metadata and checksum."""
    path = Path(manifest_path)
    if not path.exists():
        raise ResourceNotFoundError(f"minidvc manifest not found at '{path}'.")

    raw_text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("MINIDVC-MANIFEST"):
        raise ValidationError(f"File '{path}' is not a valid MINIDVC-MANIFEST.")

    header_meta: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    for line in lines[1:]:
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 3:
                files.append(
                    {
                        "sha256": parts[0].strip(),
                        "size_bytes": int(parts[1].strip()),
                        "path": parts[2].strip(),
                    }
                )
        elif ":" in line:
            k, v = line.split(":", 1)
            header_meta[k.strip()] = v.strip()

    if not files:
        raise ValidationError(f"minidvc manifest '{path}' contains zero tracked files.")

    manifest_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    total_size = int(header_meta.get("size", sum(f["size_bytes"] for f in files)))
    file_count = int(header_meta.get("files", len(files)))

    return {
        "manifest_version": path.stem,
        "project_id": header_meta.get("project_id", ""),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "manifest_sha256": manifest_sha256,
        "tracked_paths_preview": [f["path"] for f in files[:10]],
    }


def parse_feature_report(report_path: Optional[str | Path]) -> dict[str, Any]:
    """Read feature_report.json produced by src/features/exporters.py if provided."""
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        raise ResourceNotFoundError(f"feature_report.json not found at '{path}'.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in feature_report at '{path}'.") from exc
    return {
        "pipeline_version": str(data.get("pipeline_version", "2.0.0")),
        "quality_summary": data.get("quality_summary", {}),
        "orientation_distribution": data.get("orientation_distribution", {}),
        "estimated_columns_distribution": data.get("estimated_columns_distribution", {}),
    }

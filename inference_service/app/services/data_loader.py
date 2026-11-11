from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from inference_service.app.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

REQUIRED_ELEMENTS_COLUMNS = [
    "poster_id",
    "element_id",
    "category_slug",
    "x0",
    "y0",
    "w_norm",
    "h_norm",
    "cx",
    "cy",
    "area_norm",
    "raw_x",
    "raw_y",
    "raw_w",
    "raw_h",
]


@dataclass
class DataDependencyStatus:
    """Status metadata for the template elements parquet dependency."""

    ready: bool
    source: str
    sha256: Optional[str] = None
    row_count: int = 0
    poster_count: int = 0
    error: Optional[str] = None


def _compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_from_url(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        shutil.move(str(tmp_path), str(dest_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _download_from_clearml_dataset(settings: Settings, dest_path: Path) -> None:
    from clearml import Dataset

    dataset = Dataset.get(
        dataset_id=settings.CLEARML_DATASET_ID or None,
        dataset_project=settings.CLEARML_PROJECT_NAME,
        dataset_name=settings.CLEARML_DATASET_NAME or None,
        dataset_version=settings.CLEARML_DATASET_VERSION or None,
    )
    local_dir = Path(dataset.get_local_copy())
    matches = list(local_dir.rglob("elements.parquet"))
    if not matches:
        raise FileNotFoundError("ClearML dataset does not contain 'elements.parquet'.")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dest_path)


def _download_from_minidvc(settings: Settings, dest_path: Path) -> None:
    snapshot = settings.MINIDVC_SNAPSHOT
    if not snapshot:
        raise RuntimeError("MINIDVC_SNAPSHOT is not configured.")
    minidvc_bin = shutil.which("minidvc")
    if not minidvc_bin:
        raise FileNotFoundError("minidvc binary is not installed in PATH.")

    work_dir = PROJECT_ROOT if (PROJECT_ROOT / ".mydvc").exists() else dest_path.parent
    if settings.MINIDVC_CLIENT_ID and settings.MINIDVC_CLIENT_SECRET:
        auth_cmd = [
            minidvc_bin,
            "auth",
            settings.MINIDVC_CLIENT_ID,
            settings.MINIDVC_CLIENT_SECRET,
        ]
        if settings.MINIDVC_REFRESH_TOKEN:
            auth_cmd.append(settings.MINIDVC_REFRESH_TOKEN)
        subprocess.run(auth_cmd, cwd=str(work_dir), check=True, capture_output=True, text=True)

    subprocess.run(
        [minidvc_bin, "pull", snapshot],
        cwd=str(work_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    with tempfile.TemporaryDirectory() as tmp_out:
        subprocess.run(
            [minidvc_bin, "checkout", snapshot, tmp_out],
            cwd=str(work_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        matches = list(Path(tmp_out).rglob("elements.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"Minidvc snapshot '{snapshot}' does not contain 'elements.parquet'."
            )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], dest_path)


def _validate_elements_dataframe(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_ELEMENTS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"elements.parquet missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("elements.parquet is empty (0 rows).")


def load_elements_dataframe(settings: Settings) -> tuple[pd.DataFrame, DataDependencyStatus]:
    """Ensure `elements.parquet` is available, verified, and loaded into memory.

    Resolution order:
      1. Existing file at `ELEMENTS_PARQUET_PATH` (e.g. mounted volume / persistent disk).
      2. Explicit download from `ELEMENTS_PARQUET_URL`.
      3. ClearML Dataset (`CLEARML_DATASET_ID` or `CLEARML_DATASET_NAME`).
      4. Minidvc snapshot (`MINIDVC_SNAPSHOT`).

    Never silently guesses arbitrary fallback files.
    """
    target_path = Path(settings.ELEMENTS_PARQUET_PATH)
    source = "local_cache"

    if not target_path.exists():
        if settings.ELEMENTS_PARQUET_URL:
            logger.info("Downloading elements.parquet from configured URL...")
            _download_from_url(settings.ELEMENTS_PARQUET_URL, target_path)
            source = "http_url"
        elif settings.CLEARML_DATASET_ID or settings.CLEARML_DATASET_NAME:
            logger.info("Downloading elements.parquet from ClearML Dataset...")
            _download_from_clearml_dataset(settings, target_path)
            source = "clearml_dataset"
        elif settings.MINIDVC_SNAPSHOT:
            logger.info("Pulling elements.parquet via minidvc snapshot '%s'...", settings.MINIDVC_SNAPSHOT)
            _download_from_minidvc(settings, target_path)
            source = f"minidvc:{settings.MINIDVC_SNAPSHOT}"
        else:
            raise FileNotFoundError(
                "Required template geometry dataset 'elements.parquet' was not found at "
                "ELEMENTS_PARQUET_PATH, and no remote source (ELEMENTS_PARQUET_URL, "
                "CLEARML_DATASET_ID/NAME, or MINIDVC_SNAPSHOT) is configured."
            )

    actual_sha256 = _compute_file_sha256(target_path)
    if settings.ELEMENTS_PARQUET_SHA256:
        expected = settings.ELEMENTS_PARQUET_SHA256.strip().lower()
        if actual_sha256.lower() != expected:
            raise ValueError(
                f"Checksum mismatch for elements.parquet: expected {expected}, got {actual_sha256}."
            )

    df = pd.read_parquet(target_path, columns=REQUIRED_ELEMENTS_COLUMNS)
    _validate_elements_dataframe(df)

    status = DataDependencyStatus(
        ready=True,
        source=source,
        sha256=actual_sha256,
        row_count=int(len(df)),
        poster_count=int(df["poster_id"].nunique()),
    )
    return df, status

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

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from inference_service.app.config import PROJECT_ROOT, Settings
from inference_service.app.services.memory_profiler import (
    MemoryTracker,
    trim_process_memory,
)

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

FLOAT32_COLUMNS = [
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
    artifact_size_mib: Optional[float] = None
    dataset_mem_mib: Optional[float] = None
    error: Optional[str] = None


def _compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(file_path: Path, expected_sha256: Optional[str]) -> str:
    actual_sha256 = _compute_file_sha256(file_path)
    if expected_sha256 and expected_sha256.strip():
        expected = expected_sha256.strip().lower()
        if actual_sha256.lower() != expected:
            raise ValueError(
                f"Checksum mismatch for elements.parquet: expected {expected}, got {actual_sha256}."
            )
    return actual_sha256


def _atomic_promote_temp_file(
    tmp_path: Path,
    dest_path: Path,
    expected_sha256: Optional[str],
) -> str:
    """Verify SHA-256 on temp file before atomically replacing dest_path."""
    actual_sha256 = _verify_sha256(tmp_path, expected_sha256)
    os.replace(str(tmp_path), str(dest_path))
    return actual_sha256


def _download_from_s3(settings: Settings, dest_path: Path) -> str:
    """Stream elements.parquet from a private AWS S3 bucket to disk, verify SHA-256, and atomically rename."""
    bucket = (settings.S3_BUCKET_NAME or "").strip()
    key = (settings.S3_OBJECT_KEY or "").strip()
    if not bucket or not key:
        raise ValueError(
            "S3 configuration incomplete: both S3_BUCKET_NAME and S3_OBJECT_KEY are required."
        )

    access_key = (settings.AWS_ACCESS_KEY_ID or "").strip()
    secret_key = (settings.AWS_SECRET_ACCESS_KEY or "").strip()
    if not access_key or not secret_key:
        raise RuntimeError(
            "AWS S3 credentials missing: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required."
        )

    region = (settings.AWS_DEFAULT_REGION or "").strip() or None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".s3.tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    # Cap transfer concurrency and chunk size to avoid high RAM spikes in 512 MiB containers
    transfer_cfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=4 * 1024 * 1024,
        max_concurrency=2,
        max_io_queue=4,
        use_threads=True,
    )

    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(connect_timeout=15, read_timeout=120, retries={"max_attempts": 2}),
        )
        client.download_file(bucket, key, str(tmp_path), Config=transfer_cfg)
        return _atomic_promote_temp_file(tmp_path, dest_path, settings.ELEMENTS_PARQUET_SHA256)
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise RuntimeError(
            "AWS S3 credentials missing or incomplete for private bucket download."
        ) from exc
    except ClientError as exc:
        err = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
        meta = exc.response.get("ResponseMetadata", {}) if isinstance(exc.response, dict) else {}
        code = str(err.get("Code", ""))
        http_status = meta.get("HTTPStatusCode")

        if code in ("404", "NoSuchKey", "NoSuchBucket", "NotFound") or http_status == 404:
            raise FileNotFoundError(
                f"S3 object not found: s3://{bucket}/{key} ({code or '404'})."
            ) from exc
        if (
            code in ("403", "AccessDenied", "Forbidden", "InvalidAccessKeyId", "SignatureDoesNotMatch")
            or http_status == 403
        ):
            raise PermissionError(
                f"Access denied downloading S3 object s3://{bucket}/{key} ({code or '403'})."
            ) from exc
        raise RuntimeError(
            f"AWS S3 client error ({code or 'unknown'}) downloading s3://{bucket}/{key}."
        ) from exc
    except (BotoCoreError, OSError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ConnectionError(
            f"Network or transport error downloading S3 object s3://{bucket}/{key}: {exc.__class__.__name__}."
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_from_url(url: str, dest_path: Path, expected_sha256: Optional[str]) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".url.tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        return _atomic_promote_temp_file(tmp_path, dest_path, expected_sha256)
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_from_clearml_dataset(settings: Settings, dest_path: Path) -> str:
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
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".clearml.tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(matches[0], tmp_path)
        return _atomic_promote_temp_file(tmp_path, dest_path, settings.ELEMENTS_PARQUET_SHA256)
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_from_minidvc(settings: Settings, dest_path: Path) -> str:
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
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".minidvc.tmp")
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            shutil.copy2(matches[0], tmp_path)
            return _atomic_promote_temp_file(tmp_path, dest_path, settings.ELEMENTS_PARQUET_SHA256)
        finally:
            tmp_path.unlink(missing_ok=True)


def _read_elements_parquet_streaming(
    parquet_path: Path,
    valid_poster_ids: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Stream-load only required columns from Parquet in bounded batches with float32 downcasting."""
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    pf = pq.ParquetFile(parquet_path, memory_map=True)
    schema_names = set(pf.schema_arrow.names)
    missing = [col for col in REQUIRED_ELEMENTS_COLUMNS if col not in schema_names]
    if missing:
        raise ValueError(f"elements.parquet missing required columns: {missing}")

    chunks: list[pd.DataFrame] = []
    for batch in pf.iter_batches(
        batch_size=32768,
        columns=REQUIRED_ELEMENTS_COLUMNS,
        use_threads=False,
    ):
        pdf = batch.to_pandas()
        del batch
        if valid_poster_ids is not None and len(pdf) > 0:
            pdf = pdf[pdf["poster_id"].isin(valid_poster_ids)]
        if len(pdf) == 0:
            continue
        for col in FLOAT32_COLUMNS:
            pdf[col] = pdf[col].astype("float32")
        chunks.append(pdf)

    del pf
    if not chunks:
        raise ValueError("elements.parquet is empty (0 rows).")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    df["category_slug"] = df["category_slug"].astype("category")
    trim_process_memory()
    return df


def _validate_elements_dataframe(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_ELEMENTS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"elements.parquet missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("elements.parquet is empty (0 rows).")


def has_configured_adaptation_source(settings: Settings) -> bool:
    """Return True if local elements.parquet exists or a remote download source is configured."""
    return bool(
        Path(settings.ELEMENTS_PARQUET_PATH).exists()
        or (settings.S3_BUCKET_NAME or "").strip()
        or (settings.S3_OBJECT_KEY or "").strip()
        or settings.ELEMENTS_PARQUET_URL
        or settings.CLEARML_DATASET_ID
        or settings.CLEARML_DATASET_NAME
        or settings.MINIDVC_SNAPSHOT
    )


def load_elements_dataframe(
    settings: Settings,
    *,
    valid_poster_ids: Optional[set[int]] = None,
    memory_tracker: Optional[MemoryTracker] = None,
) -> tuple[pd.DataFrame, DataDependencyStatus]:
    """Ensure `elements.parquet` is available, verified, and loaded into memory.

    Explicit source-selection precedence:
      1. Existing cached/local file at `ELEMENTS_PARQUET_PATH`.
      2. Private AWS S3 object (`S3_BUCKET_NAME` + `S3_OBJECT_KEY`; both required if either is set).
      3. Remote HTTPS URL (`ELEMENTS_PARQUET_URL`).
      4. ClearML Dataset (`CLEARML_DATASET_ID` or `CLEARML_DATASET_NAME`).
      5. Minidvc snapshot (`MINIDVC_SNAPSHOT`).

    Does not silently fall back to another remote source if the selected source fails.
    """
    target_path = Path(settings.ELEMENTS_PARQUET_PATH)
    s3_bucket_set = bool((settings.S3_BUCKET_NAME or "").strip())
    s3_key_set = bool((settings.S3_OBJECT_KEY or "").strip())

    if target_path.exists():
        source = "local_cache"
        actual_sha256 = _verify_sha256(target_path, settings.ELEMENTS_PARQUET_SHA256)
    elif s3_bucket_set or s3_key_set:
        if not (s3_bucket_set and s3_key_set):
            raise ValueError(
                "S3 configuration incomplete: both S3_BUCKET_NAME and S3_OBJECT_KEY are required."
            )
        logger.info(
            "Downloading elements.parquet from private S3 bucket '%s' (key='%s')...",
            settings.S3_BUCKET_NAME,
            settings.S3_OBJECT_KEY,
        )
        actual_sha256 = _download_from_s3(settings, target_path)
        source = "s3"
        trim_process_memory()
        if memory_tracker is not None:
            size_mib = round(target_path.stat().st_size / (1024.0 * 1024.0), 2)
            memory_tracker.record("s3_dataset_downloaded", artifact_size_mib=size_mib)
    elif settings.ELEMENTS_PARQUET_URL:
        logger.info("Downloading elements.parquet from configured URL...")
        actual_sha256 = _download_from_url(
            settings.ELEMENTS_PARQUET_URL, target_path, settings.ELEMENTS_PARQUET_SHA256
        )
        source = "http_url"
    elif settings.CLEARML_DATASET_ID or settings.CLEARML_DATASET_NAME:
        logger.info("Downloading elements.parquet from ClearML Dataset...")
        actual_sha256 = _download_from_clearml_dataset(settings, target_path)
        source = "clearml_dataset"
    elif settings.MINIDVC_SNAPSHOT:
        logger.info(
            "Pulling elements.parquet via minidvc snapshot '%s'...", settings.MINIDVC_SNAPSHOT
        )
        actual_sha256 = _download_from_minidvc(settings, target_path)
        source = f"minidvc:{settings.MINIDVC_SNAPSHOT}"
    else:
        raise FileNotFoundError(
            "Required template geometry dataset 'elements.parquet' was not found at "
            "ELEMENTS_PARQUET_PATH, and no remote source (S3_BUCKET_NAME/S3_OBJECT_KEY, "
            "ELEMENTS_PARQUET_URL, CLEARML_DATASET_ID/NAME, or MINIDVC_SNAPSHOT) is configured."
        )

    artifact_size_mib = round(target_path.stat().st_size / (1024.0 * 1024.0), 2)
    df = _read_elements_parquet_streaming(target_path, valid_poster_ids=valid_poster_ids)
    _validate_elements_dataframe(df)
    dataset_mem_mib = round(float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0), 2)

    if memory_tracker is not None:
        memory_tracker.record(
            "parquet_loaded",
            artifact_size_mib=artifact_size_mib,
            dataset_mem_mib=dataset_mem_mib,
        )

    status = DataDependencyStatus(
        ready=True,
        source=source,
        sha256=actual_sha256,
        row_count=int(len(df)),
        poster_count=int(df["poster_id"].nunique()),
        artifact_size_mib=artifact_size_mib,
        dataset_mem_mib=dataset_mem_mib,
    )
    return df, status

"""CLI script to extract offline PDF text/aspect and structure features into elements.parquet."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.pdf_content import enrich_elements_with_pdf_content


def _default_features_dir() -> Path:
    for sub in ("v2", "v1"):
        cand = PROJECT_ROOT / "data" / "processed" / "features" / sub
        if (cand / "elements.parquet").exists():
            return cand
    return PROJECT_ROOT / "data" / "processed" / "features" / "v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich elements.parquet with offline PDF word counts, intrinsic figure aspect ratios, and document-order identity."
    )
    parser.add_argument("--features-dir", type=Path, default=_default_features_dir())
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "pdf",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "png",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    posters_path = args.features_dir / "posters.parquet"
    elements_path = args.features_dir / "elements.parquet"
    posters_df = pd.read_parquet(posters_path)
    elements_df = pd.read_parquet(elements_path)

    print(f"Enriching {len(elements_df):,} elements across {elements_df['poster_id'].nunique():,} posters...")
    enriched_df, stats = enrich_elements_with_pdf_content(
        elements_df=elements_df,
        posters_df=posters_df,
        pdf_dir=args.pdf_dir,
        png_dir=args.png_dir,
        max_workers=args.workers,
    )

    enriched_df.to_parquet(elements_path, index=False)

    schema_path = args.features_dir / "feature_schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["tables"]["elements"]["columns"] = list(enriched_df.columns)
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    report_path = args.features_dir / "feature_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["pdf_content_extraction"] = stats
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = round(time.time() - t0, 2)
    print(f"Completed in {elapsed}s! Updated {elements_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

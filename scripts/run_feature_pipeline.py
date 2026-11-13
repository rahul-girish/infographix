"""CLI entrypoint to run the modular Infographix v2 feature engineering pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.config import DEFAULT_PIPELINE_CONFIG_PATH, load_pipeline_config
from src.features.pipeline import run_pipeline_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Infographix v2 multi-table feature engineering pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG_PATH,
        help="Path to pipeline YAML configuration file.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Override raw COCO JSON dir.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override v2 output dir.")
    parser.add_argument("--random-state", type=int, default=None, help="Override split RNG seed.")
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable orientation stratification during pooled 70/15/15 split assignment.",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Skip exporting legacy v1-compatible split parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config(args.config)

    overrides = {}
    if args.raw_dir is not None:
        overrides["raw_dir"] = args.raw_dir
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.random_state is not None:
        overrides["random_state"] = args.random_state
    if args.no_stratify:
        overrides["stratify_by_orientation"] = False
    if args.no_legacy:
        overrides["export_legacy"] = False

    if overrides:
        cfg = replace(cfg, **overrides)

    run_pipeline_from_config(cfg)


if __name__ == "__main__":
    main()

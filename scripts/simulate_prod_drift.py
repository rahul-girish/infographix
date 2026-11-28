"""Send normal or shifted/OOD poster requests to the deployed Render inference service
to verify live Prometheus metrics and Grafana Cloud data drift dashboards.

Usage:
  # 1. Send normal SciPostLayout-like requests (keeps PSI low / green)
  .venv/bin/python scripts/simulate_prod_drift.py --url https://<your-app>.onrender.com --mode normal --count 15

  # 2. Send drifted / OOD requests (spikes PSI >= 0.20 and triggers drift alert)
  .venv/bin/python scripts/simulate_prod_drift.py --url https://<your-app>.onrender.com --mode drift --count 20
"""

from __future__ import annotations

import argparse
import json
import random
import time

import requests


def _words(n: int) -> str:
    vocab = [
        "poster", "layout", "neural", "transformer", "scientific", "evaluation",
        "method", "dataset", "experiment", "metric", "figure", "analysis",
        "model", "architecture", "results", "alignment", "column", "structure",
    ]
    return " ".join(random.choices(vocab, k=max(1, n)))


def build_normal_payload(rng: random.Random) -> dict:
    """In-distribution poster matching SciPostLayout v2 train statistics (~20-26 blocks, ~600 words, 3-5 figures)."""
    orientation = "landscape" if rng.random() < 0.31 else "portrait"
    n_sections = rng.randint(5, 8)
    n_figures = rng.randint(2, 5)
    blocks = []
    for i in range(n_sections):
        blocks.append(
            {
                "id": f"sec_{i}",
                "type": "section",
                "heading": f"Section {i + 1}: {_words(2)}",
                "text": _words(rng.randint(50, 110)),
            }
        )
    for j in range(n_figures):
        blocks.append(
            {
                "id": f"fig_{j}",
                "type": "figure",
                "heading": f"Figure {j + 1}",
                "aspectRatio": round(rng.uniform(1.0, 2.0), 2),
                "caption": f"Figure {j + 1}: {_words(rng.randint(8, 18))}",
            }
        )
    return {
        "engine": "neural",
        "request": {
            "title": f"In-Distribution Research Poster: {_words(5)}",
            "authors": "A. Author, B. Collaborator",
            "affiliations": "Infographix Research Lab",
            "orientation": orientation,
            "preferredColumns": 3 if orientation == "landscape" else 2,
            "blocks": blocks,
        },
    }


def build_drifted_payload(rng: random.Random) -> dict:
    """Heavily shifted OOD poster (100% landscape, 45+ blocks, 12+ figures, ultra-wide ratios, 2000+ words)."""
    blocks = []
    for i in range(12):
        blocks.append(
            {
                "id": f"dsec_{i}",
                "type": "section",
                "heading": f"Dense Section {i + 1}",
                "text": _words(rng.randint(150, 220)),
            }
        )
    for j in range(12):
        blocks.append(
            {
                "id": f"dfig_{j}",
                "type": "figure",
                "heading": f"Wide Chart {j + 1}",
                "aspectRatio": round(rng.uniform(3.2, 3.9), 2),
                "caption": f"Chart {j + 1}: {_words(rng.randint(15, 25))}",
            }
        )
    return {
        "engine": "neural",
        "request": {
            "title": f"Shifted Ultra-Dense Landscape Poster: {_words(6)}",
            "authors": "Dr. Drift, Prof. Shift",
            "affiliations": "OOD Stress Lab",
            "orientation": "landscape",
            "preferredColumns": 4,
            "blocks": blocks,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate normal or drifted traffic against Infographix API.")
    parser.add_argument("--url", required=True, help="Base URL of deployed Render inference service.")
    parser.add_argument("--mode", choices=("normal", "drift"), default="normal")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between requests.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    rng = random.Random(args.seed)
    ok = 0
    for i in range(1, args.count + 1):
        payload = build_normal_payload(rng) if args.mode == "normal" else build_drifted_payload(rng)
        t0 = time.perf_counter()
        resp = requests.post(f"{base}/v1/layouts/generate", json=payload, timeout=30)
        ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            ok += 1
            data = resp.json()
            gm = data.get("generation_metrics", {})
            print(f"[{i:02d}/{args.count}] 200 OK ({ms:.0f}ms) | elems={gm.get('num_elements')} | overlap={gm.get('validated_overlap_area')}")
        else:
            print(f"[{i:02d}/{args.count}] HTTP {resp.status_code}: {resp.text[:160]}")
        if i < args.count and args.delay > 0:
            time.sleep(args.delay)

    print(f"\nCompleted {ok}/{args.count} requests in '{args.mode}' mode. Fetching /v1/monitoring/drift:")
    drift_resp = requests.get(f"{base}/v1/monitoring/drift", timeout=15)
    print(json.dumps(drift_resp.json(), indent=2))


if __name__ == "__main__":
    main()

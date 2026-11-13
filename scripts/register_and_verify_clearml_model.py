"""First ClearML Model Registration & End-to-End Reproducibility Demonstration.

Executes Requirements 7 & 8:
  1. Inspects both candidate local checkpoints (`best_model.pt` and `latest_checkpoint.pt`)
     and verifies provenance against ClearML Task `8cd6cb755a084535a7bba3ae551e3e20`.
  2. Loads original local `best_model.pt` and runs inference on a fixed held-out validation sample.
  3. Registers `poster_layout_generator-v0.1.0` (`candidate`) in ClearML Native Model Registry.
  4. Queries ClearML Model Registry and prints recorded tags, provenance, and metadata.
  5. Temporarily isolates/hides local checkpoint directory during retrieval test to prove zero
     dependence on local training artifacts.
  6. Downloads & verifies the artifact bundle freshly from ClearML Cloud (`force_download=True`)
     using both `model_id` and `semantic_version="0.1.0"`.
  7. Runs inference with reconstructed model on identical inputs and verifies floating-point equivalence.
  8. Verifies corrupted bundle rejection and outputs a JSON verification report.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from clearml import InputModel, Task

from src.neural_layout.dataset import PosterLayoutDataset
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.registry.clearml_registry import (
    extract_model_config_from_checkpoint,
    load_registered_model,
    query_registered_models,
    register_model,
    verify_bundle_checksums,
    verify_checkpoint_provenance,
)
from src.registry.model_registry import ModelRegistry


TARGET_TASK_ID = "8cd6cb755a084535a7bba3ae551e3e20"
SEMANTIC_VERSION = "0.1.0"


def main():
    print("=" * 80)
    print("CLEARML MODEL REGISTRY: REGISTRATION & REPRODUCIBILITY DEMONSTRATION")
    print("=" * 80)

    # 1. Inspect both candidate checkpoints and verify provenance
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints" / "neural_layout"
    best_ckpt_path = ckpt_dir / "best_model.pt"
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"

    orig_task = Task.get_task(task_id=TARGET_TASK_ID)
    print(f"\n[1/7] Provenance Inspection against ClearML Task: {orig_task.id} ('{orig_task.name}')")

    for cand in [best_ckpt_path, latest_ckpt_path]:
        ckpt_data = torch.load(cand, map_location="cpu", weights_only=False)
        prov = verify_checkpoint_provenance(ckpt_data, orig_task)
        val_iou = ckpt_data.get("metrics", {}).get("val_iou_mean", 0.0)
        val_loss = ckpt_data.get("metrics", {}).get("val_loss_total", 0.0)
        print(
            f"  Candidate: {cand.name:<22} | Epoch: {ckpt_data['epoch']} (iter {ckpt_data['epoch']+1}) "
            f"| Val IoU: {val_iou:.6f} | Val Loss: {val_loss:.6f} | Provenance: {prov['provenance_method']} OK"
        )

    # Select best_model.pt as canonical candidate checkpoint
    chosen_ckpt_path = best_ckpt_path
    ckpt_data = torch.load(chosen_ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = extract_model_config_from_checkpoint(ckpt_data)

    # 2. Load original local model and run inference on a fixed real validation sample
    print("\n[2/7] Running inference with original local checkpoint on fixed validation sample...")
    orig_model = TemplateConditionedLayoutModel(
        content_dim=model_cfg["content_dim"],
        template_dim=model_cfg["template_dim"],
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        encoder_layers=model_cfg["encoder_layers"],
        decoder_layers=model_cfg["decoder_layers"],
        ffn_dim=model_cfg["ffn_dim"],
        num_columns=model_cfg["num_columns"],
        dropout=model_cfg["dropout"],
    )
    orig_model.load_state_dict(ckpt_data["model_state_dict"])
    orig_model.eval()

    data_dir = PROJECT_ROOT / "data" / "processed" / "features" / "v2"
    posters_df = pd.read_parquet(data_dir / "posters.parquet")
    elements_df = pd.read_parquet(data_dir / "elements.parquet")
    pdf_dir = PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "pdf"

    reg = ModelRegistry(registry_root=PROJECT_ROOT / "models" / "registry")
    retriever_model, _, _ = reg.load_production_model("poster_layout_retriever")
    train_posters = posters_df[posters_df["split"] == "train"].copy().reset_index(drop=True)
    retriever_model.fit(train_posters)

    dev_dataset = PosterLayoutDataset(
        posters_df=posters_df,
        elements_df=elements_df,
        retriever_model=retriever_model,
        split="dev",
        pdf_base_dir=pdf_dir,
        max_samples=5,
    )
    sample = dev_dataset[0]
    c_feats = sample["content_feats"].unsqueeze(0)
    t_feats = sample["template_feats"].unsqueeze(0)
    c_mask = torch.ones(1, c_feats.shape[1], dtype=torch.bool)
    t_mask = torch.ones(1, t_feats.shape[1], dtype=torch.bool)

    with torch.no_grad():
        orig_outputs = orig_model(c_feats, t_feats, c_mask, t_mask)
    print(f"  Sample Poster ID: {int(sample['target_poster_id'].item())} | Elements: {c_feats.shape[1]} | Template Slots: {t_feats.shape[1]}")
    print(f"  Original Model First Box Prediction: {[round(float(x), 5) for x in orig_outputs['pred_boxes'][0, 0]]}")

    # 3. Register model in ClearML Native Model Registry
    print(f"\n[3/7] Registering poster_layout_generator-v{SEMANTIC_VERSION} in ClearML Cloud...")
    model_id = register_model(
        checkpoint_path=chosen_ckpt_path,
        training_task_id=TARGET_TASK_ID,
        semantic_version=SEMANTIC_VERSION,
        dataset_identifier="SciPostLayout v2.0.0",
        model_config=model_cfg,
        validation_metrics=ckpt_data.get("metrics", {}),
        lifecycle_status="candidate",
        comment="First development candidate registered from Stage B ClearML validation run.",
        verify_provenance=True,
    )
    print(f"  Registered ClearML Model ID: {model_id}")

    # 4. Query ClearML Model Registry & inspect recorded metadata
    print("\n[4/7] Querying ClearML Model Registry...")
    catalog = query_registered_models(semantic_version=SEMANTIC_VERSION)
    assert len(catalog) == 1, f"Expected 1 match, got {len(catalog)}"
    entry = catalog[0]
    print(f"  Name: {entry['name']} | ID: {entry['model_id']}")
    print(f"  Originating Task ID: {entry['task_id']}")
    print(f"  Artifact URI: {entry['uri']}")
    print(f"  Tags: {entry['tags']}")
    print(f"  Published Flag: {entry['published']} | Lifecycle Status: {entry['lifecycle_status']}")
    print(f"  SHA-256 Checksum: {entry['checkpoint_sha256']}")

    # 5. Retrieve by Model ID with original checkpoint temporarily hidden to prove independence
    print("\n[5/7] Isolating local checkpoint & downloading freshly from ClearML Cloud...")
    hidden_ckpt_path = ckpt_dir / "best_model.pt.hidden_for_test"
    shutil.move(str(chosen_ckpt_path), str(hidden_ckpt_path))
    try:
        retrieved_by_id, manifest_id = load_registered_model(
            model_id=model_id,
            device="cpu",
            force_download=True,
            return_manifest=True,
        )
        retrieved_by_ver, manifest_ver = load_registered_model(
            semantic_version=SEMANTIC_VERSION,
            device="cpu",
            force_download=False,
            return_manifest=True,
        )
    finally:
        shutil.move(str(hidden_ckpt_path), str(chosen_ckpt_path))

    print(f"  Retrieved & verified SHA-256: {manifest_id['checkpoint_sha256']}")
    print(f"  Reconstructed Model Class: {retrieved_by_id.__class__.__name__}")

    # 6. Verify inference equivalence
    print("\n[6/7] Comparing original vs. ClearML-reconstructed model predictions...")
    with torch.no_grad():
        reconstructed_out_id = retrieved_by_id(c_feats, t_feats, c_mask, t_mask)
        reconstructed_out_ver = retrieved_by_ver(c_feats, t_feats, c_mask, t_mask)

    max_diffs: dict[str, float] = {}
    for key in ["pred_boxes", "pred_column_logits", "pred_order_scores", "pred_macro_column_logits"]:
        diff_id = float(np.max(np.abs(orig_outputs[key].numpy() - reconstructed_out_id[key].numpy())))
        diff_ver = float(np.max(np.abs(orig_outputs[key].numpy() - reconstructed_out_ver[key].numpy())))
        max_diffs[key] = max(diff_id, diff_ver)
        np.testing.assert_allclose(orig_outputs[key].numpy(), reconstructed_out_id[key].numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(orig_outputs[key].numpy(), reconstructed_out_ver[key].numpy(), rtol=1e-6, atol=1e-6)
        print(f"  {key:<26}: max_abs_diff = {max_diffs[key]:.2e} (MATCH)")

    # 7. Verify corruption rejection on a copy of the downloaded bundle
    print("\n[7/7] Verifying corruption detection on downloaded bundle copy...")
    dl_path = Path(InputModel(model_id=model_id).get_local_copy(extract_archive=True))
    with tempfile.TemporaryDirectory() as tmp_corrupt:
        corrupt_dir = Path(tmp_corrupt) / "corrupt_bundle"
        shutil.copytree(dl_path, corrupt_dir)
        with open(corrupt_dir / "model_weights.pt", "ab") as f:
            f.write(b"\x00\xffcorrupt")
        try:
            verify_bundle_checksums(corrupt_dir)
            raise AssertionError("Corruption check failed to catch tampered weights!")
        except ValueError as e:
            print(f"  Corruption successfully rejected: {e}")

    # Save report summary JSON
    report_dir = PROJECT_ROOT / "outputs" / "clearml_registry_verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "registration_verification_v0.1.0.json"
    report_data = {
        "clearml_model_id": model_id,
        "model_name": entry["name"],
        "originating_training_task_id": TARGET_TASK_ID,
        "artifact_uri": entry["uri"],
        "tags": entry["tags"],
        "lifecycle_status": entry["lifecycle_status"],
        "published": entry["published"],
        "checkpoint_sha256": manifest_id["checkpoint_sha256"],
        "manifest": manifest_id,
        "reproducibility_max_abs_diffs": max_diffs,
        "test_poster_id": int(sample["target_poster_id"].item()),
    }
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"SUCCESS! Registered Model ID: {model_id}")
    print(f"Model Name: {entry['name']} | URI: {entry['uri']}")
    print(f"Verification Report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

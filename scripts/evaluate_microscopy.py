"""
General fluorescence-microscopy isotropic-reconstruction evaluation.

Compares two methods against ground-truth isotropic volumes:
  1. CRIS          — CRIS_volume.pt from the model output directory
  2. Interpolation — bicubic upsampling from the gap-sampled (LR) input

Usage
-----
Edit the CONFIGURATION block below (marked with TODO), then run from the repo root:

    conda activate cris
    python scripts/evaluate_microscopy.py

Outputs (all under output_dir):
    evaluation_results.csv
    case_level_results.csv
    table_<plane>.png
    images/<plane>/<method>.png   (mid-slice visual)

CSV format expected
-------------------
| Column      | Description                                       |
|-------------|---------------------------------------------------|
| index       | Unique integer row identifier                     |
| dataset     | Split label; only 'test' rows are evaluated       |
| axial_gt    | Path to the isotropic GT .pt tensor (Z, H, W)     |
| axial       | Path to the gap-sampled (LR) input .pt tensor     |
|             | (optional; needed only for the interpolation      |
|             | baseline — omit or leave blank to skip it)        |
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import clear_cuda_cache, set_cuda_device, volume_to_plane_stack
from eval_helper import (
    evaluate_volumes,
    normalize_volume_to_01,
    save_evaluation_results,
    save_mid_slice_image,
    SEED,
)

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------

# Path to the dataset CSV.
CSV_PATH = "/path/to/microscopy_dataset_metadata.csv"  # TODO

# Directory that contains the CRIS isotropic_volumes output, structured as:
#   <CRIS_DIR>/<case_index>/CRIS_volume.pt
CRIS_DIR = "/path/to/isotropic_volumes/test/axial/axial/"  # TODO

# Where to write evaluation outputs.
OUTPUT_DIR = "/path/to/evaluation_output/"  # TODO

# Gap factor used during data preparation (used for interpolation upsampling).
GAP = 8

# Planes to evaluate metrics over.
PLANES = ["axial", "coronal", "sagittal"]

# GPU to run metrics on.
DEVICE_STR = "cuda:3"

# Set to True to compute FID and KID (requires more GPU memory; uses InceptionV3).
COMPUTE_FID_KID = True

# Batch size for 2D per-slice metrics.
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_gt_volumes(csv_path):
    """
    Load isotropic ground-truth tensors from axial_gt paths in the CSV.

    Returns
    -------
    gt_volumes : list of torch.Tensor  (Z, H, W)  normalised to [0, 1]
    cases      : list of pd.Series    one row per case
    """
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == "test"].reset_index(drop=True)

    gt_volumes, cases = [], []
    for _, row in df.iterrows():
        gt_path = row["axial_gt"]
        if not os.path.exists(gt_path):
            print(f"  Missing GT for case {row['index']}: {gt_path}")
            continue
        gt_vol = torch.load(gt_path)
        gt_vol = normalize_volume_to_01(gt_vol)
        gt_volumes.append(gt_vol)
        cases.append(row)

    print(f"Loaded {len(gt_volumes)} GT volumes.")
    return gt_volumes, cases


def load_cris_volumes(cases, cris_dir):
    """
    Load CRIS_volume.pt files from cris_dir/<case_index>/CRIS_volume.pt.

    Returns a list aligned with `cases`; entries are None for missing files.
    """
    pred_volumes = []
    for row in cases:
        path = os.path.join(cris_dir, str(row["index"]), "CRIS_volume.pt")
        if not os.path.exists(path):
            path = os.path.join(cris_dir, f"case_{row['index']}", "CRIS_volume.pt")
        if os.path.exists(path):
            vol = torch.load(path)
            pred_volumes.append(normalize_volume_to_01(vol))
        else:
            print(f"  Missing CRIS volume for case {row['index']}: {path}")
            pred_volumes.append(None)
    return pred_volumes


def generate_interpolation_volumes(cases, gt_volumes, gap):
    """
    Generate bicubic interpolation baseline.

    Strategy
    --------
    If the CSV row contains an 'axial' column pointing to a saved LR .pt file,
    load it and upsample along the depth axis to match the GT depth.

    Otherwise synthesise the LR input by taking every `gap`-th slice of the GT
    and upsampling back — this exactly replicates the degradation used in training.

    Returns a list aligned with `cases` (same length as gt_volumes).
    """
    has_lr_col = "axial" in cases[0].index if cases else False
    interp_volumes = []

    for row, gt_vol in zip(cases, gt_volumes):
        z_gt = gt_vol.shape[0]

        lr_vol = None
        if has_lr_col:
            lr_path = row.get("axial", None)
            if lr_path and os.path.exists(lr_path):
                lr_vol = torch.load(lr_path)
                lr_vol = normalize_volume_to_01(lr_vol)

        if lr_vol is None:
            # synthesise LR from GT
            lr_vol = gt_vol[::gap].clone()

        z_lr, h, w = lr_vol.shape

        # bicubic along the Z axis: treat H as batch dimension
        lr_4d = lr_vol.permute(1, 0, 2).unsqueeze(1)  # (H, 1, Z_lr, W)
        target_z = z_gt
        interp_4d = F.interpolate(
            lr_4d, size=(target_z, w), mode="bicubic", align_corners=False
        )
        interp_vol = interp_4d.squeeze(1).permute(1, 0, 2)  # (Z_gt, H, W)
        interp_vol = torch.clamp(interp_vol, 0.0, 1.0)

        interp_volumes.append(interp_vol)

    return interp_volumes


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    csv_path,
    cris_dir,
    output_dir,
    gap=GAP,
    planes=None,
    device_str=DEVICE_STR,
    compute_fid_kid=COMPUTE_FID_KID,
    batch_size=BATCH_SIZE,
):
    if planes is None:
        planes = PLANES

    device = set_cuda_device(
        torch.device(device_str if torch.cuda.is_available() else "cpu")
    )

    print("=" * 60)
    print("Microscopy Evaluation  (CRIS vs Bicubic Interpolation)")
    print(f"  CSV:      {csv_path}")
    print(f"  CRIS dir: {cris_dir}")
    print(f"  Output:   {output_dir}")
    print("=" * 60)

    gt_volumes, cases = load_gt_volumes(csv_path)
    if not gt_volumes:
        print("No GT volumes found — aborting.")
        return

    case_ids = [int(row["index"]) for row in cases]
    case_names = [str(row.get("name", row["index"])) for row in cases]

    cris_volumes = load_cris_volumes(cases, cris_dir)
    interp_volumes = generate_interpolation_volumes(cases, gt_volumes, gap)

    method_map = {
        "CRIS": cris_volumes,
        "Interpolation": interp_volumes,
    }

    results = {}
    case_level_results = {}

    for method_name, pred_list in method_map.items():
        print(f"\n--- Evaluating: {method_name} ---")

        valid = [
            (gt, pred, cid, cname)
            for gt, pred, cid, cname in zip(gt_volumes, pred_list, case_ids, case_names)
            if pred is not None
        ]
        if not valid:
            print(f"  No valid volumes for {method_name} — skipping.")
            continue

        v_gt, v_pred, v_ids, v_names = zip(*valid)

        results[method_name] = {}
        case_level_results[method_name] = {}

        for plane in planes:
            print(f"  Plane: {plane}")

            gt_plane = [volume_to_plane_stack(v, plane) for v in v_gt]
            pred_plane = [volume_to_plane_stack(v, plane) for v in v_pred]

            if gt_plane:
                save_mid_slice_image(gt_plane[0], "Ground Truth", plane, output_dir)
                save_mid_slice_image(pred_plane[0], method_name, plane, output_dir)

            metrics, case_df = evaluate_volumes(
                gt_plane, pred_plane,
                method_name=method_name,
                plane=plane,
                device=device,
                batch_size=batch_size,
                case_ids=list(v_ids),
                case_names=list(v_names),
                compute_fid_kid=compute_fid_kid,
                fid_kid_resize_mode="pad",
            )

            results[method_name][plane] = metrics
            case_level_results[method_name][plane] = case_df

            clear_cuda_cache(device)

    save_evaluation_results(results, case_level_results, output_dir)
    print("\nEvaluation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(
        csv_path=CSV_PATH,
        cris_dir=CRIS_DIR,
        output_dir=OUTPUT_DIR,
        gap=GAP,
        planes=PLANES,
        device_str=DEVICE_STR,
        compute_fid_kid=COMPUTE_FID_KID,
        batch_size=BATCH_SIZE,
    )

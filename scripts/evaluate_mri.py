"""
Brain MRI isotropic-reconstruction evaluation.

Compares two methods against ground-truth isotropic volumes:
  1. CRIS     — predicted CRIS_volume.pt from the model output directory
  2. Interpolation — isotropic linear resampling of the original NIfTI (SimpleITK baseline)

Usage
-----
Edit the CONFIGURATION block below (marked with TODO), then run from the repo root:

    conda activate cris
    python scripts/evaluate_mri.py

Outputs (all under output_dir):
    evaluation_results.csv
    case_level_results.csv
    table_<plane>.png
    images/<plane>/<method>.png   (mid-slice visual)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from preprocess import read_volume
from utils import clear_cuda_cache, resample_isotropic, set_cuda_device, volume_to_plane_stack
from eval_helper import (
    evaluate_volumes,
    normalize_volume_to_01,
    save_evaluation_results,
    save_mid_slice_image,
    SEED,
)


# ---------------------------------------------------------------------------
# Black-slice helpers
# ---------------------------------------------------------------------------

def _fix_slices_to_black(gt_vol, pred_vol):
    """
    For every slice along Z, Y, and X where the GT is entirely background (-1),
    set the prediction to background as well.  Operates on [-1, 1] tensors.
    Shapes must match; returns unchanged tensors if they differ.
    """
    if gt_vol.shape != pred_vol.shape:
        return gt_vol, pred_vol
    for i in range(gt_vol.shape[0]):
        if torch.all(gt_vol[i] == -1):
            pred_vol[i] = torch.zeros_like(pred_vol[i]) - 1
    for i in range(gt_vol.shape[1]):
        if torch.all(gt_vol[:, i, :] == -1):
            pred_vol[:, i, :] = torch.zeros_like(pred_vol[:, i, :]) - 1
    for i in range(gt_vol.shape[2]):
        if torch.all(gt_vol[:, :, i] == -1):
            pred_vol[:, :, i] = torch.zeros_like(pred_vol[:, :, i]) - 1
    return gt_vol, pred_vol


def _remove_black_slices(gt_plane, pred_plane, threshold_pct=98, eps=0.002):
    """
    Remove slices where ≥ threshold_pct% of GT pixels are background (≤ -1 + eps).
    Inputs: (N, H, W) tensors in [-1, 1].
    """
    n = min(gt_plane.shape[0], pred_plane.shape[0])
    gt_plane   = gt_plane[:n]
    pred_plane = pred_plane[:n]
    keep = [
        i for i in range(n)
        if 100.0 * (gt_plane[i] <= -1.0 + eps).float().mean().item() < threshold_pct
    ]
    if len(keep) == n:
        return gt_plane, pred_plane
    idx = torch.tensor(keep, dtype=torch.long)
    return gt_plane[idx], pred_plane[idx]

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------

# Path to the dataset CSV (must contain columns: index, dataset, <default_plane>,
# global_min, global_max; only rows with dataset == 'test' are evaluated).
CSV_PATH = "/path/to/your/dataset.csv"  # TODO

# Directory that contains the CRIS isotropic_volumes output, structured as:
#   <CRIS_DIR>/<case_index>/CRIS_volume.pt
CRIS_DIR = "/path/to/isotropic_volumes/test/<default_plane>/<default_plane>/"  # TODO

# Where to write evaluation outputs.
OUTPUT_DIR = "/path/to/evaluation_output/"  # TODO

# Anatomical plane that CRIS was trained/tested on.
DEFAULT_PLANE = "coronal"

# Planes to evaluate metrics over (can be a subset of axial/coronal/sagittal).
PLANES = ["coronal", "axial", "sagittal"]

# GPU to run metrics on (e.g. "cuda:3" on a multi-GPU server; "cuda" for default).
DEVICE_STR = "cuda:3"

# Set to True to compute FID and KID (requires more GPU memory; uses InceptionV3).
COMPUTE_FID_KID = True

# Batch size for 2D per-slice metrics.
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_PLANE_TO_ARRAY_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}
_PLANE_TO_SITK_AXIS  = {"axial": 2, "coronal": 1, "sagittal": 0}


def _remove_slices_sitk(img_itk, n_remove, axis):
    """Remove n_remove slices from the right along a SimpleITK axis (X=0,Y=1,Z=2)."""
    if n_remove <= 0:
        return img_itk
    size  = list(img_itk.GetSize())
    index = [0] * 3
    size[axis] -= int(n_remove)
    ex = sitk.ExtractImageFilter()
    ex.SetSize(size)
    ex.SetIndex(index)
    return ex.Execute(img_itk)


def _load_gt_unstretch(nifti_path, global_min, global_max, gap, plane):
    """
    Load a NIfTI volume and normalise it to [-1, 1]:
      1. Crop the trailing edge along the acquisition plane so the slice count
         satisfies (n - 1) % gap == 0.
      2. Resample to isotropic voxel spacing.
      3. Clip to [global_min, global_max] and scale linearly to [-1, 1].

    Returns a float32 torch.Tensor of shape (Z, Y, X).
    """
    img = read_volume(nifti_path)
    arr = sitk.GetArrayFromImage(img)

    plane_axis_np   = _PLANE_TO_ARRAY_AXIS[plane.lower()]
    plane_axis_sitk = _PLANE_TO_SITK_AXIS[plane.lower()]
    n = int(arr.shape[plane_axis_np])
    n_remove = (n - 1) % gap
    if n_remove > 0:
        img = _remove_slices_sitk(img, n_remove, plane_axis_sitk)

    img_iso = resample_isotropic(img)
    arr = sitk.GetArrayFromImage(img_iso).astype(np.float32)

    # unstretch normalisation (no percentile stretch, just clip + linear)
    arr = np.clip(arr, global_min, global_max)
    arr = (arr - global_min) / (global_max - global_min)  # → [0, 1]
    arr = arr * 2.0 - 1.0                                  # → [-1, 1]
    return torch.tensor(arr, dtype=torch.float32)


def load_gt_and_cases(csv_path, plane, gap=5):
    """
    Load isotropic GT volumes for all test cases in the CSV.

    GT NIfTI path is read from the ``{plane}_gt`` column (falls back to
    the ``{plane}`` column if the ``_gt`` variant is absent).

    Returns
    -------
    gt_volumes : list of torch.Tensor  (Z, Y, X) in [-1, 1]
    cases      : list of pd.Series    one row per case
    """
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == "test"].reset_index(drop=True)
    gt_col = f"{plane}_gt" if f"{plane}_gt" in df.columns else plane
    df = df.dropna(subset=[gt_col]).reset_index(drop=True)

    gt_volumes, cases = [], []
    for _, row in df.iterrows():
        nifti_path = row[gt_col]
        if not os.path.exists(str(nifti_path)):
            print(f"  Missing GT NIfTI for case {row['index']}: {nifti_path}")
            continue
        gt_vol = _load_gt_unstretch(
            str(nifti_path), row["global_min"], row["global_max"], gap=gap, plane=plane
        )
        gt_volumes.append(gt_vol)
        cases.append(row)

    print(f"Loaded {len(gt_volumes)} GT volumes (plane={plane}, gt_col={gt_col}).")
    return gt_volumes, cases


def load_cris_volumes(cases, cris_dir):
    """
    Load CRIS prediction volumes from cris_dir.

    File search order (first match wins):
      1. ``<cris_dir>/<index>/CRIS_volume.pt``
      2. ``<cris_dir>/case_<index>/CRIS_volume.pt``
      3. ``<cris_dir>/<index>/imputed_volume.pt``       (legacy format)
      4. ``<cris_dir>/case_<index>/imputed_volume.pt``  (legacy format)

    Returns a list aligned with `cases`; entries are None for missing files.
    """
    pred_volumes = []
    for row in cases:
        idx = str(row["index"])
        candidates = [
            os.path.join(cris_dir, idx, "CRIS_volume.pt"),
            os.path.join(cris_dir, f"case_{idx}", "CRIS_volume.pt"),
            os.path.join(cris_dir, idx, "imputed_volume.pt"),
            os.path.join(cris_dir, f"case_{idx}", "imputed_volume.pt"),
        ]
        found = next((p for p in candidates if os.path.exists(p)), None)
        if found:
            vol = torch.load(found)
            if isinstance(vol, dict) and "volume" in vol:
                vol = vol["volume"]
            pred_volumes.append(vol.float())
        else:
            print(f"  Missing prediction for case {idx} in {cris_dir}")
            pred_volumes.append(None)
    return pred_volumes


def load_interpolation_volumes(cases, gt_volumes, plane, gap=5):
    """
    Build the isotropic linear-interpolation baseline.

      1. Load the original thick-slice (anisotropic) NIfTI from ``row[plane]``.
      2. Resample it to isotropic voxel spacing using SimpleITK's linear interpolator.
      3. Clip to [global_min, global_max] and scale linearly to [-1, 1].
      4. Trim the resampled depth to match the corresponding GT volume.

    Falls back to ``row[{plane}_gt]`` if ``row[plane]`` is absent.
    Returns a list aligned with `cases`; entries are None for missing files.
    """
    plane_axis = _PLANE_TO_ARRAY_AXIS[plane.lower()]
    interp_volumes = []

    for row, gt_vol in zip(cases, gt_volumes):
        nifti_path = row.get(plane, None)
        if not nifti_path or not os.path.exists(str(nifti_path)):
            # No separate interpolation NIfTI → fall back to GT path (same as GT)
            nifti_path = row.get(f"{plane}_gt", None)
        if not nifti_path or not os.path.exists(str(nifti_path)):
            print(f"  Missing NIfTI for interpolation baseline, case {row['index']}")
            interp_volumes.append(None)
            continue

        global_min = float(row["global_min"])
        global_max = float(row["global_max"])

        img = read_volume(str(nifti_path))
        img_iso = resample_isotropic(img)
        arr = sitk.GetArrayFromImage(img_iso).astype(np.float32)

        arr = np.clip(arr, global_min, global_max)
        arr = (arr - global_min) / (global_max - global_min) * 2.0 - 1.0
        pred_vol = torch.tensor(arr, dtype=torch.float32)

        gt_depth  = gt_vol.shape[plane_axis]
        pred_depth = pred_vol.shape[plane_axis]
        if pred_depth > gt_depth:
            slices = [slice(None)] * pred_vol.ndim
            slices[plane_axis] = slice(0, gt_depth)
            pred_vol = pred_vol[tuple(slices)].contiguous()
        elif pred_depth < gt_depth:
            print(
                f"  [WARN] Interpolation depth {pred_depth} < GT depth {gt_depth} "
                f"for case {row['index']} — keeping as-is."
            )

        interp_volumes.append(pred_vol)

    return interp_volumes


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    csv_path,
    cris_dir,
    output_dir,
    default_plane=DEFAULT_PLANE,
    planes=None,
    device_str=DEVICE_STR,
    compute_fid_kid=COMPUTE_FID_KID,
    batch_size=BATCH_SIZE,
    gap=5,
):
    if planes is None:
        planes = PLANES

    device = set_cuda_device(
        torch.device(device_str if torch.cuda.is_available() else "cpu")
    )

    print("=" * 60)
    print("Brain MRI Evaluation  (CRIS vs Interpolation)")
    print(f"  CSV:      {csv_path}")
    print(f"  CRIS dir: {cris_dir}")
    print(f"  Gap:      {gap}")
    print(f"  Output:   {output_dir}")
    print("=" * 60)

    gt_volumes, cases = load_gt_and_cases(csv_path, default_plane, gap=gap)
    if not gt_volumes:
        print("No GT volumes found — aborting.")
        return

    case_ids = [int(row["index"]) for row in cases]
    case_names = [str(row.get("name", row["index"])) for row in cases]

    # Load prediction volumes
    cris_volumes = load_cris_volumes(cases, cris_dir)
    interp_volumes = load_interpolation_volumes(cases, gt_volumes, default_plane, gap=gap)

    # Build method dict: name → list of volumes (None entries excluded below)
    method_map = {
        "CRIS": cris_volumes,
        "Interpolation": interp_volumes,
    }

    results = {}
    case_level_results = {}

    for method_name, pred_list in method_map.items():
        print(f"\n--- Evaluating: {method_name} ---")

        # Filter out cases with missing predictions
        valid = [
            (gt, pred, cid, cname)
            for gt, pred, cid, cname in zip(gt_volumes, pred_list, case_ids, case_names)
            if pred is not None
        ]
        if not valid:
            print(f"  No valid volumes for {method_name} — skipping.")
            continue

        v_gt, v_pred, v_ids, v_names = zip(*valid)

        # Zero-out prediction slices where GT is all-background, then decompose
        # into plane stacks and discard predominantly-background slices.
        fixed_pairs = [_fix_slices_to_black(gt.clone(), pred.clone()) for gt, pred in zip(v_gt, v_pred)]
        v_gt_fixed   = [p[0] for p in fixed_pairs]
        v_pred_fixed = [p[1] for p in fixed_pairs]

        results[method_name] = {}
        case_level_results[method_name] = {}

        for plane in planes:
            print(f"  Plane: {plane}")

            # Orient both GT and predictions to the evaluation plane
            gt_plane   = [volume_to_plane_stack(v, plane) for v in v_gt_fixed]
            pred_plane = [volume_to_plane_stack(v, plane) for v in v_pred_fixed]

            # Discard predominantly-background slices before computing metrics.
            gt_plane, pred_plane = zip(*[
                _remove_black_slices(g, p)
                for g, p in zip(gt_plane, pred_plane)
            ])
            gt_plane   = list(gt_plane)
            pred_plane = list(pred_plane)

            # Save mid-slice visual for the first case
            if gt_plane:
                save_mid_slice_image(
                    normalize_volume_to_01(gt_plane[0]), "Ground Truth", plane, output_dir
                )
                save_mid_slice_image(
                    normalize_volume_to_01(pred_plane[0]), method_name, plane, output_dir
                )

            metrics, case_df = evaluate_volumes(
                gt_plane, pred_plane,
                method_name=method_name,
                plane=plane,
                device=device,
                batch_size=batch_size,
                case_ids=list(v_ids),
                case_names=list(v_names),
                compute_fid_kid=compute_fid_kid,
                compute_edges=True,
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
        default_plane=DEFAULT_PLANE,
        planes=PLANES,
        device_str=DEVICE_STR,
        compute_fid_kid=COMPUTE_FID_KID,
        batch_size=BATCH_SIZE,
    )

"""
Shared evaluation utilities for CRIS evaluation scripts.

Provides metric computation (PSNR, CF-PSNR, SSIM 3D, GMSD, FID, KID),
volume normalisation, result serialisation, and table visualisation.
All evaluation scripts (evaluate_mri.py, evaluate_microscopy.py, evaluate_epfl.py)
import from this module.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import piq
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from torchvision.models import inception_v3, Inception_V3_Weights

from utils import clear_cuda_cache, set_cuda_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

down_arrow = "\u2193"
up_arrow = "\u2191"

CASE_LEVEL_METRICS = ["PSNR", "CF-PSNR", "SSIM 3D", "GMSD", "Edges 3D"]
LOWER_IS_BETTER = ["FID", "KID", "GMSD"]

SEED = 42


# ---------------------------------------------------------------------------
# Low-pass filter and Clipped Fourier PSNR
# ---------------------------------------------------------------------------

class LowPassFilter(torch.nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.rows = rows
        self.cols = cols

    def forward(self, radius, device):
        crow, ccol = self.rows // 2, self.cols // 2
        low_pass = torch.zeros((self.rows, self.cols), dtype=torch.uint8, device=device)
        x = torch.arange(0, self.cols, device=device).unsqueeze(0).expand(self.rows, self.cols)
        y = torch.arange(0, self.rows, device=device).unsqueeze(1).expand(self.rows, self.cols)
        low_pass[(x - ccol) ** 2 + (y - crow) ** 2 <= radius ** 2] = 1
        return low_pass


class ClippedFourierPSNR(torch.nn.Module):
    def __init__(self, im_max=1.0, im_size=128):
        super().__init__()
        self.im_max = torch.tensor(im_max)
        self.low_pass_filter = LowPassFilter(im_size, im_size)

    def forward(self, prediction, gt, threshold=25):
        device = prediction.device
        n_pixels = prediction.shape[-1] * prediction.shape[-2]
        low_pass = self.low_pass_filter(threshold, device=device)

        pred_fourier = fftshift(fft2(prediction)) * low_pass
        gt_fourier = fftshift(fft2(gt)) * low_pass
        sum_sq_error = torch.sum(torch.abs(pred_fourier - gt_fourier) ** 2, dim=[-3, -2, -1])

        self.im_max = self.im_max.to(device)
        return 20 * torch.log10(self.im_max * n_pixels) - 10 * torch.log10(sum_sq_error + 1e-8)


# ---------------------------------------------------------------------------
# GMSD
# ---------------------------------------------------------------------------

def _prewitt_kernels_2d(device, dtype):
    kx = torch.tensor(
        [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kx, ky


@torch.no_grad()
def gmsd2D_per_image(gt_batch, pred_batch, c=0.0026, downsample=2, eps=1e-12):
    """GMSD per 2D slice. Inputs: [N, 1, H, W] in [0, 1]. Returns numpy array of length N."""
    assert gt_batch.shape == pred_batch.shape
    device, dtype = gt_batch.device, gt_batch.dtype

    if downsample and downsample > 1:
        gt = F.avg_pool2d(gt_batch, kernel_size=downsample, stride=downsample)
        pr = F.avg_pool2d(pred_batch, kernel_size=downsample, stride=downsample)
    else:
        gt, pr = gt_batch, pred_batch

    kx, ky = _prewitt_kernels_2d(device, dtype)

    gm_r = torch.sqrt(F.conv2d(gt, kx, padding=1) ** 2 + F.conv2d(gt, ky, padding=1) ** 2 + eps)
    gm_t = torch.sqrt(F.conv2d(pr, kx, padding=1) ** 2 + F.conv2d(pr, ky, padding=1) ** 2 + eps)

    gms = (2 * gm_r * gm_t + c) / (gm_r ** 2 + gm_t ** 2 + c)
    return torch.std(gms, dim=(2, 3), unbiased=False).squeeze(1).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3D SSIM
# ---------------------------------------------------------------------------

def _gaussian_1d(window_size, sigma):
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window_3d(window_size, channel):
    w1d = _gaussian_1d(window_size, 1.5).unsqueeze(1)
    w2d = w1d.mm(w1d.t())
    w3d = (
        w1d.mm(w2d.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return torch.autograd.Variable(
        w3d.expand(channel, 1, window_size, window_size, window_size).contiguous()
    )


def ssim3D(img1, img2, window_size=11, size_average=True):
    """3D SSIM. Inputs: [1, 1, D, H, W] in [0, 1]."""
    _, channel, _, _, _ = img1.size()
    window = _create_window_3d(window_size, channel).to(img1.device).type_as(img1)
    g = window_size // 2

    mu1 = F.conv3d(img1, window, padding=g, groups=channel)
    mu2 = F.conv3d(img2, window, padding=g, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=g, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=g, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=g, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


@torch.no_grad()
def ssim_edges3D(gt_vol_3d, pred_vol_3d, eps=1e-8):
    """
    SSIM computed on 3D Sobel gradient magnitudes (Edges 3D metric).

    Inputs: [1, 1, D, H, W] in [0, 1]. Returns: float.
    """
    device, dtype = gt_vol_3d.device, gt_vol_3d.dtype
    dx = torch.tensor([1., 0., -1.], device=device, dtype=dtype)
    s  = torch.tensor([1., 2.,  1.], device=device, dtype=dtype)
    Gx = (s[:, None, None] * s[None, :, None] * dx[None, None, :]).view(1, 1, 3, 3, 3)
    Gy = (s[:, None, None] * dx[None, :, None] * s[None, None, :]).view(1, 1, 3, 3, 3)
    Gz = (dx[:, None, None] * s[None, :, None] * s[None, None, :]).view(1, 1, 3, 3, 3)

    def _mag(vol):
        gx = F.conv3d(vol, Gx, padding=1)
        gy = F.conv3d(vol, Gy, padding=1)
        gz = F.conv3d(vol, Gz, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + eps)

    mag_r = _mag(gt_vol_3d)
    mag_t = _mag(pred_vol_3d)
    dims = (2, 3, 4)
    mag_r = (mag_r / (mag_r.amax(dim=dims, keepdim=True) + eps)).clamp(0, 1)
    mag_t = (mag_t / (mag_t.amax(dim=dims, keepdim=True) + eps)).clamp(0, 1)
    return ssim3D(mag_t, mag_r, window_size=11, size_average=True).item()


# ---------------------------------------------------------------------------
# Per-slice PSNR and CF-PSNR
# ---------------------------------------------------------------------------

@torch.no_grad()
def psnr2D_per_image(gt_batch, pred_batch, data_range=1.0, eps=1e-12):
    """PSNR per 2D slice. Inputs: [N, 1, H, W] in [0, 1]. Returns numpy array of length N."""
    mse = torch.mean((gt_batch - pred_batch) ** 2, dim=(1, 2, 3))
    return (10.0 * torch.log10((data_range ** 2) / torch.clamp(mse, min=eps))).detach().cpu().numpy()


@torch.no_grad()
def cf_psnr2D_per_image(cf_calculator, pred_batch, gt_batch, threshold=25):
    """Clipped Fourier PSNR per 2D slice. Returns numpy array of length N."""
    return cf_calculator(pred_batch, gt_batch, threshold=threshold).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Inception features (for FID / KID)
# ---------------------------------------------------------------------------

def extract_features(images, model, device, batch_size=64, resize_mode="bilinear"):
    """
    Extract InceptionV3 features from [N, 1, H, W] slices.

    resize_mode : str
        ``"bilinear"`` — bilinear interpolation to 299×299 (EPFL electron microscopy)
        ``"pad"``       — zero-pad to 299×299 (fluorescence microscopy / MRI)
    """
    with torch.no_grad():
        features = []
        for i in range(0, images.size(0), batch_size):
            batch = images[i: i + batch_size].to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            if resize_mode == "pad":
                padded = []
                for img in batch:
                    _, h, w = img.shape
                    pad_h, pad_w = (299 - h) // 2, (299 - w) // 2
                    p = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
                    padded.append(p[:, :299, :299])
                batch = torch.stack(padded)
            else:
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)

            features.append(model(batch).detach().cpu())
            del batch
            clear_cuda_cache(device)
        return torch.cat(features, dim=0)


# ---------------------------------------------------------------------------
# Volume normalisation
# ---------------------------------------------------------------------------

def normalize_volume_to_01(volume):
    """Map any volume in [-1, 1] or [0, 255] to [0, 1]."""
    if isinstance(volume, dict) and "volume" in volume:
        volume = volume["volume"]
    volume = volume.float()
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmin < -0.05:
        volume = (volume + 1.0) / 2.0
    elif vmax > 1.5:
        volume = volume / 255.0
    return torch.clamp(volume, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core per-volume evaluation loop
# ---------------------------------------------------------------------------

def evaluate_volumes(
    gt_volumes,
    pred_volumes,
    method_name,
    plane,
    device,
    batch_size=32,
    case_ids=None,
    case_names=None,
    compute_fid_kid=True,
    compute_edges=False,
    fid_kid_resize_mode="bilinear",
):
    """
    Compute PSNR, CF-PSNR, SSIM 3D, GMSD (per-case) and optionally FID/KID (global).

    Parameters
    ----------
    gt_volumes, pred_volumes : list of torch.Tensor  shape (D, H, W) in any range
    method_name : str  label for the "Test Name" column
    plane : str  label for the "Plane" column
    device : torch.device
    batch_size : int  slices processed at once for 2D metrics
    case_ids : list[int] | None
    case_names : list[str] | None
    compute_fid_kid : bool
    compute_edges : bool
        If True, also compute Edges 3D (SSIM on Sobel gradient maps) — used by evaluate_mri.py.
    fid_kid_resize_mode : str
        How to resize slices for InceptionV3: ``"bilinear"`` (EPFL) or ``"pad"`` (MRI / microscopy).

    Returns
    -------
    metrics : dict   global FID/KID (empty if compute_fid_kid=False)
    case_df : pd.DataFrame  one row per case
    """
    if case_ids is None:
        case_ids = list(range(len(gt_volumes)))
    if case_names is None:
        case_names = [str(c) for c in case_ids]

    if compute_fid_kid:
        inception_net = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=False
        ).to(device)
        inception_net.fc = torch.nn.Identity()
        inception_net.eval()
        pred_features, gt_features = [], []
    else:
        inception_net = pred_features = gt_features = None

    # CF-PSNR calculator is built lazily per-case so the filter always matches
    # the actual (post-crop) slice dimensions.
    _cf_h = _cf_w = None
    cf_calculator = None

    case_rows = []

    for case_id, case_name, gt_vol, pred_vol in zip(
        case_ids, case_names, gt_volumes, pred_volumes
    ):
        gt_vol = normalize_volume_to_01(gt_vol).contiguous()
        pred_vol = normalize_volume_to_01(pred_vol).contiguous()

        if gt_vol.shape != pred_vol.shape:
            min_shape = tuple(min(a, b) for a, b in zip(gt_vol.shape, pred_vol.shape))
            print(
                f"  [WARN] Shape mismatch for {method_name} | plane={plane} | case={case_id}: "
                f"GT {tuple(gt_vol.shape)} vs pred {tuple(pred_vol.shape)} "
                f"— cropping both to {min_shape}"
            )
            gt_vol = gt_vol[:min_shape[0], :min_shape[1], :min_shape[2]].contiguous()
            pred_vol = pred_vol[:min_shape[0], :min_shape[1], :min_shape[2]].contiguous()

        # Rebuild the CF-PSNR calculator if slice dimensions changed.
        h, w = int(gt_vol.shape[-2]), int(gt_vol.shape[-1])
        if h != _cf_h or w != _cf_w:
            _cf_h, _cf_w = h, w
            cf_calculator = ClippedFourierPSNR(im_max=1.0, im_size=128)
            cf_calculator.low_pass_filter = LowPassFilter(_cf_h, _cf_w)
            cf_calculator = cf_calculator.to(device)

        n_slices = int(gt_vol.size(0))

        gt_3d = gt_vol.unsqueeze(0).unsqueeze(0).to(device).contiguous()
        pred_3d = pred_vol.unsqueeze(0).unsqueeze(0).to(device).contiguous()

        ssim_val = ssim3D(pred_3d, gt_3d, window_size=11, size_average=True).item()

        psnr_vals, cf_psnr_vals, gmsd_vals = [], [], []

        for i in range(0, n_slices, batch_size):
            gt_b = gt_vol[i: i + batch_size].unsqueeze(1).to(device).contiguous()
            pred_b = pred_vol[i: i + batch_size].unsqueeze(1).to(device).contiguous()

            psnr_vals.extend(psnr2D_per_image(gt_b, pred_b).tolist())
            cf_psnr_vals.extend(cf_psnr2D_per_image(cf_calculator, pred_b, gt_b).tolist())
            gmsd_vals.extend(gmsd2D_per_image(gt_b, pred_b, c=0.5, downsample=2).tolist())

            if compute_fid_kid:
                pred_features.append(
                    extract_features(pred_b, inception_net, device,
                                     resize_mode=fid_kid_resize_mode)
                )
                gt_features.append(
                    extract_features(gt_b, inception_net, device,
                                     resize_mode=fid_kid_resize_mode)
                )

            del gt_b, pred_b
            clear_cuda_cache(device)

        row = {
            "Test Name": method_name,
            "Plane": plane,
            "case_id": int(case_id),
            "case_name": str(case_name),
            "n_slices": n_slices,
            "PSNR": float(np.mean(psnr_vals)),
            "CF-PSNR": float(np.mean(cf_psnr_vals)),
            "SSIM 3D": float(ssim_val),
            "GMSD": float(np.mean(gmsd_vals)),
        }
        if compute_edges:
            row["Edges 3D"] = ssim_edges3D(gt_3d, pred_3d)
        case_rows.append(row)

    case_df = pd.DataFrame(case_rows)

    metrics = {}
    if compute_fid_kid and pred_features:
        all_pred = torch.cat(pred_features, dim=0)
        all_gt = torch.cat(gt_features, dim=0)
        metrics["FID"] = piq.FID()(all_pred, all_gt).item()
        metrics["KID"] = piq.KID()(all_pred, all_gt).item()

    if compute_fid_kid and inception_net is not None:
        del inception_net
        clear_cuda_cache(device)

    return metrics, case_df


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def save_evaluation_results(results, case_level_results, output_dir):
    """
    Save evaluation outputs:

    * case_level_results.csv  — one row per method / plane / case (+ avg plane)
    * evaluation_results.csv  — one row per method / plane (averaged over cases)

    Parameters
    ----------
    results : dict  {method_name: {plane: metrics_dict}}
        metrics_dict contains global FID/KID (may be empty)
    case_level_results : dict  {method_name: {plane: case_df}}
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- build flat case-level dataframe ---
    case_dfs = []
    for method_name, plane_dfs in case_level_results.items():
        for plane, df in plane_dfs.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            if "Test Name" not in df.columns:
                df["Test Name"] = method_name
            if "Plane" not in df.columns:
                df["Plane"] = plane
            case_dfs.append(df)

    if not case_dfs:
        print("No case-level results to save.")
        return pd.DataFrame(), pd.DataFrame()

    all_case_df = pd.concat(case_dfs, ignore_index=True)
    metric_cols = [m for m in CASE_LEVEL_METRICS if m in all_case_df.columns]

    # add per-case "avg" plane rows
    avg_rows = []
    for method, mdf in all_case_df.groupby("Test Name"):
        mdf_no_avg = mdf[mdf["Plane"] != "avg"]
        if mdf_no_avg.empty:
            continue
        avg_df = mdf_no_avg.groupby("case_id", as_index=False)[metric_cols].mean()
        if "case_name" in mdf_no_avg.columns:
            avg_df = avg_df.merge(
                mdf_no_avg.groupby("case_id", as_index=False)["case_name"].first(),
                on="case_id", how="left",
            )
        avg_df["Test Name"] = method
        avg_df["Plane"] = "avg"
        avg_df["n_slices"] = float("nan")
        avg_rows.append(avg_df)

    if avg_rows:
        all_case_df = pd.concat([all_case_df] + avg_rows, ignore_index=True)

    preferred_case = ["Test Name", "Plane", "case_id", "case_name", "n_slices"] + CASE_LEVEL_METRICS
    all_case_df = _reorder_df(all_case_df, preferred_case)
    all_case_df.to_csv(os.path.join(output_dir, "case_level_results.csv"), index=False)
    print(f"Case-level results → {os.path.join(output_dir, 'case_level_results.csv')}")

    # --- build summary dataframe ---
    summary_df = (
        all_case_df.groupby(["Test Name", "Plane"], as_index=False)[metric_cols].mean()
    )
    n_cases = (
        all_case_df.groupby(["Test Name", "Plane"], as_index=False)["case_id"]
        .nunique()
        .rename(columns={"case_id": "n_cases"})
    )
    summary_df = n_cases.merge(summary_df, on=["Test Name", "Plane"], how="left")

    # attach FID / KID from global results
    fid_rows = []
    for method, plane_metrics in results.items():
        for plane, m in plane_metrics.items():
            row = {"Test Name": method, "Plane": plane}
            row.update({k: v for k, v in m.items() if k in ("FID", "KID")})
            if len(row) > 2:
                fid_rows.append(row)

    if fid_rows:
        fid_df = pd.DataFrame(fid_rows)
        fid_metric_cols = [c for c in fid_df.columns if c not in ("Test Name", "Plane")]

        fid_no_avg = fid_df[fid_df["Plane"] != "avg"]
        if not fid_no_avg.empty:
            fid_avg = fid_no_avg.groupby("Test Name", as_index=False)[fid_metric_cols].mean()
            fid_avg["Plane"] = "avg"
            fid_df = pd.concat([fid_df, fid_avg], ignore_index=True)

        summary_df = summary_df.merge(fid_df, on=["Test Name", "Plane"], how="left")

    preferred = ["Test Name", "Plane", "n_cases"] + CASE_LEVEL_METRICS + ["FID", "KID"]
    summary_df = _reorder_df(summary_df, preferred)

    plane_order = {"axial": 0, "coronal": 1, "sagittal": 2, "avg": 3}
    method_order = {m: i for i, m in enumerate(results)}
    summary_df["_po"] = summary_df["Plane"].map(plane_order).fillna(9)
    summary_df["_mo"] = summary_df["Test Name"].map(method_order).fillna(9)
    summary_df = (
        summary_df.sort_values(["_po", "_mo"]).drop(columns=["_po", "_mo"]).reset_index(drop=True)
    )

    summary_path = os.path.join(output_dir, "evaluation_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results → {summary_path}")

    save_table_images(summary_df.drop(columns=["n_cases"], errors="ignore"), output_dir)

    return all_case_df, summary_df


def _reorder_df(df, preferred):
    existing = [c for c in preferred if c in df.columns]
    others = [c for c in df.columns if c not in existing]
    return df[existing + others]


# ---------------------------------------------------------------------------
# Table visualisation
# ---------------------------------------------------------------------------

def underline_text(text):
    return "".join(c + "\u0333" for c in text)


def save_table_images(df, output_dir, title_prefix="Evaluation"):
    """Save one PNG results table per unique plane in df."""
    os.makedirs(output_dir, exist_ok=True)

    for plane in df["Plane"].unique():
        df_plane = df[df["Plane"] == plane].copy().drop(columns=["Plane"], errors="ignore")
        for metric in df_plane.columns[1:]:
            df_plane[metric] = pd.to_numeric(df_plane[metric], errors="coerce").round(5)

        fig, ax = plt.subplots(figsize=(max(10, len(df_plane.columns) * 1.2), len(df_plane) * 0.6 + 1))
        ax.axis("tight")
        ax.axis("off")
        ax.set_title(
            f"{title_prefix} — {plane.capitalize()} Plane", fontsize=14, fontweight="bold"
        )

        table_data = [df_plane.columns.tolist()] + df_plane.astype(str).values.tolist()
        table_data[0] = [
            h if h == "Test Name"
            else (f"{h} {down_arrow}" if h in LOWER_IS_BETTER else f"{h} {up_arrow}")
            for h in table_data[0]
        ]

        col_widths = [0.3] + [0.12] * (len(df_plane.columns) - 1)
        table = ax.table(
            cellText=table_data, loc="center", cellLoc="center", colWidths=col_widths
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for metric in df_plane.columns[1:]:
            values = pd.to_numeric(df_plane[metric], errors="coerce")
            if values.isnull().any() or len(values) < 1:
                continue
            best_idx = values.idxmin() if metric in LOWER_IS_BETTER else values.idxmax()
            row_idx = df_plane.index.get_loc(best_idx) + 1
            col_idx = df_plane.columns.get_loc(metric)
            if (row_idx, col_idx) in table._cells:
                table[row_idx, col_idx].set_text_props(fontweight="bold")
            if len(values) > 1:
                second_idx = (
                    values.nsmallest(2).idxmax()
                    if metric in LOWER_IS_BETTER
                    else values.nlargest(2).idxmin()
                )
                row2 = df_plane.index.get_loc(second_idx) + 1
                if (row2, col_idx) in table._cells:
                    cell = table[row2, col_idx]
                    cell.get_text().set_text(underline_text(cell.get_text().get_text()))

        img_path = os.path.join(output_dir, f"table_{plane}.png")
        plt.savefig(img_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  Saved table image → {img_path}")


def save_mid_slice_image(volume, label, plane, output_dir):
    """Save the middle 2D slice of a (D, H, W) volume as a PNG."""
    save_dir = os.path.join(output_dir, "images", plane)
    os.makedirs(save_dir, exist_ok=True)
    mid_idx = volume.shape[0] // 2
    slice_2d = volume[mid_idx].cpu().numpy()
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    filepath = os.path.join(save_dir, f"{safe_label}.png")
    plt.imsave(filepath, slice_2d, cmap="gray", vmin=0.0, vmax=1.0)
    print(f"  Saved mid-slice → {filepath}")

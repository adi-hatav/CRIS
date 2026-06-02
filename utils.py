"""Training, inference, and I/O helpers for the CRIS core package."""

import json
import os
import random

import matplotlib
import numpy as np
import pytorch_msssim
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.fft import fft2, fftshift

from constants import MASK_VALUE

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Losses (training)
# ---------------------------------------------------------------------------


def sobel_loss_calc(pred, target, eps=1e-5):
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=pred.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    pred_grad = torch.sqrt(F.conv2d(pred, sobel_x, padding=1) ** 2 + F.conv2d(pred, sobel_y, padding=1) ** 2 + eps)
    tgt_grad = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + eps)
    return F.l1_loss(pred_grad, tgt_grad)


def ssim_loss(predicted, target):
    return 1 - pytorch_msssim.ssim(predicted, target, data_range=2.0)


def focal_frequency_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    diff = torch.clamp((torch.abs(pred_fft) - torch.abs(target_fft)).pow(2), max=50.0)
    weight = diff.detach() / (diff.max() + 1e-8)
    return torch.mean(weight * diff) * 10.0


def loss_flags_for_epoch(opt):
    epoch = getattr(opt, "epoch", 0)
    domain = getattr(opt, "domain", "MRI")
    return {
        "ssim": epoch >= 3,
        "focal": epoch >= 5 and domain != "microscopy",
        "sobel": epoch >= 10 and domain != "microscopy",
    }


def compute_training_loss(predicted, target, criterion, opt, flags):
    """Build weighted training loss from enabled terms."""
    l2_loss = criterion(predicted, target) * 10
    total = l2_loss
    parts = {"l2_loss": l2_loss.item()}

    if flags["ssim"]:
        ssim_term = ssim_loss(predicted, target) * (5 if opt.domain == "MRI" else 0.5)
        total = total + ssim_term
        parts["ssim_loss"] = ssim_term.item()

    if flags["sobel"]:
        sobel_term = sobel_loss_calc(predicted, target) * 5
        total = total + sobel_term
        parts["sobel_loss"] = sobel_term.item()

    if flags["focal"]:
        focal_term = focal_frequency_loss(predicted, target) * 10
        total = total + focal_term
        parts["focal_loss"] = focal_term.item()

    return total, parts


# ---------------------------------------------------------------------------
# Inference slice helpers
# ---------------------------------------------------------------------------


def center_pad_slice_for_inference(input_slice, target_size, fill_value):
    """Center-crop/pad a 2D slice and extend periodic gap mask into padding."""
    h, w = input_slice.shape
    top_crop = max((h - target_size) // 2, 0)
    left_crop = max((w - target_size) // 2, 0)
    crop = input_slice[
        top_crop : top_crop + min(h, target_size),
        left_crop : left_crop + min(w, target_size),
    ]
    ch, cw = crop.shape

    canvas = torch.full((target_size, target_size), fill_value, dtype=input_slice.dtype)
    top = (target_size - ch) // 2
    left = (target_size - cw) // 2
    canvas[top : top + ch, left : left + cw] = crop

    row_mask = (input_slice == MASK_VALUE).all(dim=1)
    col_mask = (input_slice == MASK_VALUE).all(dim=0)
    use_rows = row_mask.sum() >= col_mask.sum()
    m = row_mask if use_rows else col_mask
    orig_idx = (~m).nonzero(as_tuple=False).flatten()

    period = None
    if orig_idx.numel() >= 2:
        mode_val = torch.mode(orig_idx[1:] - orig_idx[:-1]).values.item()
        if mode_val >= 2:
            period = int(mode_val)
            phase = int(orig_idx[0].item() % period)

    if period is not None:
        if use_rows:
            rows = torch.arange(target_size)
            pattern = ((rows - (top + phase)) % period != 0)
            outside = torch.ones(target_size, dtype=torch.bool)
            outside[top : top + ch] = False
            canvas[pattern & outside, :] = MASK_VALUE
        else:
            cols = torch.arange(target_size)
            pattern = ((cols - (left + phase)) % period != 0)
            outside = torch.ones(target_size, dtype=torch.bool)
            outside[left : left + cw] = False
            canvas[:, pattern & outside] = MASK_VALUE

    return canvas, h, w


def map_imputed_slice_to_original(imputed_slice, original_shape, target_size, orig_h, orig_w, dtype):
    h0, w0 = original_shape
    if h0 != target_size or w0 != target_size:
        if h0 > target_size or w0 > target_size:
            temp = torch.full(original_shape, -1, dtype=dtype)
            top = max((h0 - target_size) // 2, 0)
            left = max((w0 - target_size) // 2, 0)
            temp[top : top + target_size, left : left + target_size] = imputed_slice
            return temp
        top = (target_size - orig_h) // 2
        left = (target_size - orig_w) // 2
        return imputed_slice[top : top + orig_h, left : left + orig_w]
    return imputed_slice


def build_model_input_from_slice(slice_2d, device):
    missing = slice_2d == MASK_VALUE
    intensity = slice_2d.clone()
    mask = (~missing).float()
    return torch.stack([intensity, mask], dim=0).unsqueeze(0).float().to(device)


# ---------------------------------------------------------------------------
# Volume / metrics utilities
# ---------------------------------------------------------------------------


def volume_to_plane_stack(volume_zyx, plane):
    plane = plane.lower()
    if plane == "axial":
        return volume_zyx.contiguous()
    if plane == "coronal":
        return volume_zyx.permute(1, 0, 2).contiguous()
    if plane == "sagittal":
        return volume_zyx.permute(2, 0, 1).contiguous()
    raise ValueError(f"Invalid plane: {plane}")


def resample_isotropic(volume):
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    if len(set(round(s, 3) for s in original_spacing)) == 1:
        print("Image is already isotropic. Skipping resampling.")
        return volume

    isotropic_spacing = min(original_spacing)
    new_size = [
        int(round((original_size[i] * original_spacing[i]) / isotropic_spacing))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([isotropic_spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1)
    resampler.SetOutputOrigin(volume.GetOrigin())
    resampler.SetOutputDirection(volume.GetDirection())
    return resampler.Execute(volume)


def save_random_predictions(epoch, model, loader, dataset_dir, device, n_samples=1):
    fig_save_dir = os.path.join(dataset_dir, "figures")
    os.makedirs(fig_save_dir, exist_ok=True)

    n_total = len(loader.dataset)
    n_samples = min(n_samples, n_total)
    indices = random.sample(range(n_total), n_samples)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = loader.dataset[idx]
            x_t = torch.tensor(sample[0], dtype=torch.float32) if isinstance(sample[0], np.ndarray) else sample[0]
            gt_slice = sample[1].squeeze()
            info = sample[2] if len(sample) > 2 else {}

            input_tensor = x_t.unsqueeze(0).to(device)
            data_slice = x_t[0].cpu().numpy()
            prediction_slice = model(input_tensor).squeeze().cpu().numpy()
            if isinstance(gt_slice, torch.Tensor):
                gt_slice = gt_slice.cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            vmin, vmax = -1, 1
            axes[0].imshow(data_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[0].set_title("Input")
            axes[1].imshow(gt_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[1].set_title("Target")
            axes[2].imshow(prediction_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[2].set_title("Prediction")
            diff = prediction_slice - gt_slice
            axes[3].imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
            axes[3].set_title("Difference")
            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"Epoch {epoch} - Case {info.get('case', '?')}, "
                f"Plane: {info.get('plane', '?')}, Slice: {info.get('slice', '?')}",
                fontsize=16,
            )
            fig_path = os.path.join(fig_save_dir, f"epoch_{epoch}_sample_{i}.png")
            fig.savefig(fig_path)
            plt.close(fig)


def save_networks(model, epoch, save_dir):
    models_dir = os.path.join(save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    save_path = os.path.join(models_dir, f"{epoch}_model.pth")
    torch.save(state_dict, save_path)
    print(f"Saved model to {save_path}")


def save_metrics_to_json(metrics, output_dir):
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def plot_metrics(metrics, output_dir):
    save_dir = os.path.join(output_dir, "figures", "losses plots")
    os.makedirs(save_dir, exist_ok=True)
    for loss_type in metrics["train"]:
        plt.figure()
        plt.plot(metrics["train"][loss_type], label=f"Train {loss_type}")
        plt.xlabel("Epochs")
        plt.ylabel(loss_type)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"train_{loss_type}_loss.png"))
        plt.close()


def validate(model, criterion, val_loader, device, name):
    model.eval()
    total_val_loss, total_psnr = 0.0, 0.0
    with torch.no_grad():
        for data_masked, data_gt, _ in val_loader:
            data_masked, data_gt = data_masked.to(device), data_gt.to(device)
            prediction = model(data_masked)
            total_val_loss += criterion(prediction, data_gt).item()
            total_psnr += psnr(
                data_gt.cpu().numpy(), prediction.cpu().numpy(), data_range=2.0
            )
    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    print(f"{name} Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.4f}")
    return avg_val_loss, avg_psnr


def log_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}, Total trainable parameters: {total_params:,}")


def create_circular_mask(h, w, radius, device):
    crow, ccol = h // 2, w // 2
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    return (((x - ccol) ** 2 + (y - crow) ** 2) <= radius ** 2).float()


def clipped_fourier_psnr(prediction, gt, threshold=25, data_range=1.0):
    h, w = prediction.shape[-2:]
    n_pixels = h * w
    low_pass = create_circular_mask(h, w, threshold, prediction.device)
    pred_fourier = fftshift(fft2(prediction)) * low_pass
    gt_fourier = fftshift(fft2(gt)) * low_pass
    sum_sq_error = torch.sum(torch.abs(pred_fourier - gt_fourier) ** 2, dim=[-2, -1])
    return (
        20 * torch.log10(torch.tensor(data_range * n_pixels, device=prediction.device))
        - 10 * torch.log10(sum_sq_error)
    ).mean().item()


def set_cuda_device(device):
    """Set PyTorch's active CUDA device (avoids accidental use of cuda:0)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


def clear_cuda_cache(device=None):
    """Release cached GPU memory on the given device, not the default cuda:0."""
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.empty_cache()
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()

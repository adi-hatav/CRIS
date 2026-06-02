import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import random
import matplotlib
import torch.nn.functional as F
from torch.utils.data import Dataset

from constants import ARRAY_AXIS_TO_PLANE, MASK_VALUE, PLANE_TO_ARRAY_AXIS

matplotlib.use('Agg')


def choose_training_slice_axis(
    volume_zyx,
    spacing_xyz,
    csv_plane,
    spacing_tol=1.2,
    size_separation_ratio=2.0,
    strict_plane=False,
):
    """
    Choose the native acquired slice-stack axis for CRIS 2D training.

    Primary rule:
        If one dimension is clearly much smaller than the other two, use it.
        This is the most reliable signal for anisotropic MRI stacks.

    Secondary rule:
        If dimensions are not clearly separated, use largest spacing.

    csv_plane is used for warning/validation, not blindly trusted.
    """
    csv_plane = csv_plane.lower()
    expected_axis = PLANE_TO_ARRAY_AXIS[csv_plane]

    shape_zyx = np.array(volume_zyx.shape)
    spacing_zyx = np.array(spacing_xyz)[::-1]

    smallest_size_axis = int(np.argmin(shape_zyx))
    sorted_sizes = np.sort(shape_zyx)

    # Example: (81, 512, 512) or (512, 31, 512)
    # Here the smallest axis is clearly the slice stack.
    has_clear_lowres_axis = sorted_sizes[1] / max(sorted_sizes[0], 1) >= size_separation_ratio

    min_spacing = float(np.min(spacing_zyx))
    max_spacing = float(np.max(spacing_zyx))
    spacing_ratio = max_spacing / max(min_spacing, 1e-8)
    largest_spacing_axis = int(np.argmax(spacing_zyx))

    if has_clear_lowres_axis:
        chosen_axis = smallest_size_axis
        reason = "smallest_size_clear_lowres_axis"
    elif spacing_ratio >= spacing_tol:
        chosen_axis = largest_spacing_axis
        reason = "largest_spacing_no_clear_size_axis"
    else:
        chosen_axis = expected_axis
        reason = "fallback_csv_plane"

    if chosen_axis != expected_axis:
        message = (
            "⚠️ Plane-axis mismatch detected. "
            f"csv_plane={csv_plane}, expected_axis={expected_axis} "
            f"({ARRAY_AXIS_TO_PLANE[expected_axis]}), "
            f"chosen_axis={chosen_axis} ({ARRAY_AXIS_TO_PLANE[chosen_axis]}), "
            f"reason={reason}, shape_zyx={tuple(shape_zyx)}, "
            f"spacing_xyz={tuple(spacing_xyz)}, spacing_zyx={tuple(spacing_zyx)}"
        )

        if strict_plane:
            raise ValueError(message)
        else:
            print(message)

    if has_clear_lowres_axis and largest_spacing_axis != chosen_axis:
        print(
            "⚠️ Spacing/shape disagreement. "
            f"Using smallest-size axis={chosen_axis} ({ARRAY_AXIS_TO_PLANE[chosen_axis]}) "
            f"instead of largest-spacing axis={largest_spacing_axis} "
            f"({ARRAY_AXIS_TO_PLANE[largest_spacing_axis]}). "
            f"shape_zyx={tuple(shape_zyx)}, spacing_zyx={tuple(spacing_zyx)}. "
            "This likely indicates incorrect NIfTI spacing/header metadata."
        )

    return chosen_axis

def extract_oriented_slice_by_axis(volume_zyx, axis, slice_index):
    """
    Extract a 2D slice from canonical LPS volume_zyx=(Z,Y,X)
    and store it in consistent display orientation.

    axis=0: axial
    axis=1: coronal
    axis=2: sagittal
    """
    if axis == 0:
        # axial: rows A->P, columns R->L
        slice_2d = volume_zyx[slice_index, :, :]
        return np.ascontiguousarray(slice_2d)

    if axis == 1:
        # coronal: rows should be S->I, columns R->L
        slice_2d = volume_zyx[:, slice_index, :]
        return np.ascontiguousarray(np.flip(slice_2d, axis=0))

    if axis == 2:
        # sagittal: rows should be S->I, columns A->P
        slice_2d = volume_zyx[:, :, slice_index]
        return np.ascontiguousarray(np.flip(slice_2d, axis=0))

    raise ValueError(f"Invalid axis: {axis}")


def normalize_and_scale(volume, min_val, max_val, stretch_percentage=10):
    """Clip to [min_val, max_val], linearly stretch the top stretch_percentage% of
    intensities to fill the full range, then map to [-1, 1].

    Kept as a standalone helper for backward compatibility with evaluation scripts.
    New code should call apply_intensity_normalization() instead.
    """
    max_volume_val = np.max(volume)
    range_width = max_val - min_val
    stretch_threshold = max_val - (range_width * (stretch_percentage / 100))
    mask = volume > stretch_threshold
    stretched_volume = volume.copy()
    if np.any(mask):
        original_values = volume[mask]
        stretched_values = np.interp(
            original_values,
            [stretch_threshold, max_volume_val],
            [stretch_threshold, max_val],
        )
        stretched_volume[mask] = stretched_values

    stretched_volume = np.clip(stretched_volume, min_val, max_val)
    stretched_volume = (stretched_volume - min_val) / (max_val - min_val)
    stretched_volume = (stretched_volume * 2) - 1
    return stretched_volume


def apply_intensity_normalization(
    volume,
    mode,
    intensity_min=None,
    intensity_max=None,
    stretch_percentage=10,
):
    """Normalize a numpy volume to [-1, 1] using the specified mode.

    Args:
        volume:           numpy array of raw voxel intensities.
        mode:             One of:
            'stretch'     – Auto-compute per-volume bounds (0.05th / 99.9th percentile),
                            then stretch the top ``stretch_percentage``% of intensities
                            and map to [-1, 1].  Good default for MRI with unknown range.
            'clip'        – Clip to [intensity_min, intensity_max] (per-case values, usually
                            from the CSV), then linearly map to [-1, 1].
            'fixed_range' – Same linear clip as 'clip' but uses a single cohort-wide
                            [intensity_min, intensity_max] for every volume.
            'minmax'      – No clipping; pure per-volume min-max rescale to [-1, 1].
                            Equivalent to the previous microscopy default.
        intensity_min:    Lower bound (required for 'clip' and 'fixed_range').
        intensity_max:    Upper bound (required for 'clip' and 'fixed_range').
        stretch_percentage: Top-end stretch width as a % of the total range (only used
                            in 'stretch' mode, default 10).

    Returns:
        numpy float32 array in [-1, 1].

    Raises:
        ValueError: For unknown mode or missing bounds in 'clip'/'fixed_range'.
    """
    volume = np.asarray(volume, dtype=np.float64)

    if mode == 'stretch':
        auto_min = float(np.percentile(volume, 0.05))
        auto_max = float(np.percentile(volume, 99.9))
        return normalize_and_scale(volume, auto_min, auto_max, stretch_percentage).astype(np.float32)

    if mode in ('clip', 'fixed_range'):
        if intensity_min is None or intensity_max is None:
            raise ValueError(
                f"intensity_min and intensity_max must be provided for mode='{mode}'. "
                "Pass them explicitly or set --intensity_min / --intensity_max."
            )
        volume = np.clip(volume, intensity_min, intensity_max)
        denom = float(intensity_max) - float(intensity_min)
        if denom == 0:
            return np.zeros_like(volume, dtype=np.float32)
        volume = (volume - intensity_min) / denom
        return ((volume * 2.0) - 1.0).astype(np.float32)

    if mode == 'minmax':
        v_min = float(np.min(volume))
        v_max = float(np.max(volume))
        if v_max <= v_min:
            return np.zeros_like(volume, dtype=np.float32)
        volume = (volume - v_min) / (v_max - v_min)
        return ((volume * 2.0) - 1.0).astype(np.float32)

    raise ValueError(
        f"Unknown intensity_norm_mode: '{mode}'. "
        "Choose from 'stretch', 'clip', 'fixed_range', 'minmax'."
    )


def load_volume_from_path(volume_path, opt, scale=True):
    # **Load Volume (Microscopy .npy or .pt)**
    if volume_path.endswith('.npy') or volume_path.endswith('.pt'):

        # 1. Load the data
        if volume_path.endswith('.pt'):
            # Load PyTorch tensor and convert to numpy
            volume = torch.load(volume_path).float().cpu().numpy()
        else:
            volume = np.load(volume_path)

        # 2. Provide dummy metadata
        size = list(volume.shape)
        origin = [0.0, 0.0, 0.0]
        direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # 3. Dynamically determine spacing based on orientation
        # Numpy shape is (Z, Y, X). SimpleITK spacing expects (X, Y, Z).
        squashed_dim = size.index(min(size))  # Find the axis with size 16

        spacing = [1.0, 1.0, 1.0]
        if squashed_dim == 0:  # Z is squashed
            spacing[2] = float(opt.gap)
        elif squashed_dim == 1:  # Y is squashed
            spacing[1] = float(opt.gap)
        elif squashed_dim == 2:  # X is squashed
            spacing[0] = float(opt.gap)

        # 4. Normalize to [-1, 1]
        if scale:
            mode = getattr(opt, 'intensity_norm_mode', 'minmax')
            i_min = getattr(opt, 'intensity_min', None)
            i_max = getattr(opt, 'intensity_max', None)
            volume = apply_intensity_normalization(volume, mode, intensity_min=i_min, intensity_max=i_max)

        return volume, origin, spacing, size, direction

    # **Load Volume (DICOM or NIfTI)**
    volume_sitk = read_volume(volume_path, from_dicom=os.path.isdir(volume_path))

    if volume_sitk is None:
        raise ValueError(f"Failed to load volume from path: {volume_path}")
    origin = list(volume_sitk.GetOrigin())
    spacing = list(volume_sitk.GetSpacing())
    size = list(volume_sitk.GetSize())
    direction = list(volume_sitk.GetDirection())

    volume = sitk.GetArrayFromImage(volume_sitk)
    if scale:
        mode = getattr(opt, 'intensity_norm_mode', 'clip')
        i_min = getattr(opt, 'intensity_min', None)
        i_max = getattr(opt, 'intensity_max', None)
        volume = apply_intensity_normalization(volume, mode, intensity_min=i_min, intensity_max=i_max)

    return volume, origin, spacing, size, direction


def read_volume(case_path, from_dicom=False, legacy_orientation=False):
    if from_dicom:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(case_path)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
    elif case_path.endswith('.nii') or case_path.endswith('.nii.gz'):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(case_path)
        img = reader.Execute()
    else:
        print(f'Format {case_path} does not exist or is unsupported!')
        return None

    # THE NEW ADDITION: Force the image into standard RAS anatomical orientation
    if img is not None and not legacy_orientation:
        img = sitk.DICOMOrient(img, 'LPS')

    return img


def generate_random_figures(dataset, output_dir, split_name, num_examples=2):
    """
    Randomly select a few samples from the new preprocessed dataset and save a figure.
    """
    figures_dir = os.path.join(output_dir, "figures", split_name)
    os.makedirs(figures_dir, exist_ok=True)

    n_total = len(dataset)
    num_examples = min(num_examples, n_total)

    indices = random.sample(range(n_total), num_examples)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        # sample is (input_tensor, GT, info)
        # input_tensor shape: (2, H, W) -> We need channel 0 for visualization

        # FIX: Select channel 0 (Intensity) explicitly
        if isinstance(sample[0], torch.Tensor):
            data_slice = sample[0][0].cpu().numpy()
        else:
            data_slice = sample[0][0]

        # GT is usually (1, H, W), so squeeze is fine here
        gt_slice = sample[1].squeeze(0).cpu().numpy() if isinstance(sample[1], torch.Tensor) else sample[1]

        info = sample[2]
        VMIN, VMAX = -1, 1
        # Adjusted to 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(data_slice, cmap='gray', vmin=VMIN, vmax=VMAX)
        axes[0].set_title("Input (Intensity Channel)")
        axes[0].axis('off')

        axes[1].imshow(gt_slice, cmap='gray', vmin=VMIN, vmax=VMAX)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        case_num = info.get("case", "unknown")
        slice_idx = info.get("slice", "unknown")
        plane_name = info.get("plane", "unknown")

        fig.suptitle(f"Case {case_num}, Plane: {plane_name}, Slice: {slice_idx}", fontsize=16)
        fig_path = os.path.join(figures_dir, f'example_{i}.png')
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved random figure {i} to {fig_path}")


def build_slice_cache(opt, plane):
    """Build per-slice .pt training caches for one plane under main_dir/CRIS_Dataset/."""
    output_dir = os.path.join(opt.main_dir, 'CRIS_Dataset', f"{plane}")

    save_preprocessed_data(opt, output_dir, plane)

    # Load one sample from each split and generate figures
    train_dataset = PreprocessedDataset(os.path.join(output_dir, 'train'), opt)
    val_dataset = PreprocessedDataset(os.path.join(output_dir, 'val'), opt, random_normalization=False)

    print(f"loaded {len(train_dataset)} train samples")
    print(f"loaded {len(val_dataset)} val samples")

    generate_random_figures(train_dataset, output_dir, 'train', num_examples=3)
    if not opt.use_only_train:
        generate_random_figures(val_dataset, output_dir, 'val')

def safe_reflection_pad_2d(data, pad_left, pad_right, pad_top, pad_bottom):
    """
    Reflection-pad a [C, H, W] tensor safely.

    PyTorch reflection padding requires each pad amount to be smaller than
    the current spatial size. This function applies reflection padding in
    chunks so it also works when the required padding is large.

    If a dimension has size 1, true reflection is mathematically impossible
    along that dimension, so it falls back to replicate padding only for the
    remaining impossible case.
    """
    while pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        _, H, W = data.shape

        if (H <= 1 and (pad_top > 0 or pad_bottom > 0)) or (W <= 1 and (pad_left > 0 or pad_right > 0)):
            data = F.pad(
                data,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )
            return data

        step_left = min(pad_left, W - 1) if pad_left > 0 else 0
        step_right = min(pad_right, W - 1) if pad_right > 0 else 0
        step_top = min(pad_top, H - 1) if pad_top > 0 else 0
        step_bottom = min(pad_bottom, H - 1) if pad_bottom > 0 else 0

        data = F.pad(
            data.unsqueeze(0),
            (step_left, step_right, step_top, step_bottom),
            mode="reflect",
        ).squeeze(0)

        pad_left -= step_left
        pad_right -= step_right
        pad_top -= step_top
        pad_bottom -= step_bottom

    return data


def crop_or_pad_to_patch_size(data, crop_size, domain, random_crop):
    """
    Convert [C, H, W] slice to [C, crop_size, crop_size].

    If larger than crop_size:
        - random crop during training
        - center crop during validation

    If smaller than crop_size:
        - microscopy: reflection padding
        - MRI: constant background padding with -1.0
    """
    _, H, W = data.shape

    # 1. Crop height if needed
    if H > crop_size:
        if random_crop:
            top = random.randint(0, H - crop_size)
        else:
            top = (H - crop_size) // 2
        data = data[:, top:top + crop_size, :]

    # 2. Crop width if needed
    _, H, W = data.shape
    if W > crop_size:
        if random_crop:
            left = random.randint(0, W - crop_size)
        else:
            left = (W - crop_size) // 2
        data = data[:, :, left:left + crop_size]

    # 3. Pad if needed
    _, H, W = data.shape
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)

    if pad_h == 0 and pad_w == 0:
        return data

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if domain.lower() == "microscopy":
        data = safe_reflection_pad_2d(
            data,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
        )
    else:
        data = F.pad(
            data,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=-1.0,
        )

    return data


def save_preprocessed_data(opt, output_dir, plane):
    """
    Create and save a PyTorch training dataset from volumes in a given plane.
    This version saves the original padded slices (one per slice) without
    applying any gap mask. At training time, the gap mask augmentation
    will be applied on-the-fly.

    The CSV (opt.csv_path) must contain columns "axial"/"coronal"/"sagittal",
    "dataset", and "index". The columns "intensity_min" and "intensity_max"
    are optional; they are only used when opt.intensity_norm_mode == 'clip'.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_types = ['train', 'val'] if not opt.use_only_train else ['train']
    for d in dataset_types:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    df = pd.read_csv(opt.csv_path)
    df = df.dropna(subset=[plane]).reset_index(drop=True)

    for _, row in df.iterrows():
        plane_path = row[plane]
        dataset_type = row['dataset'] if not opt.use_only_train else 'train'
        if dataset_type == 'test':
            continue  # Skip test samples when creating datasets
        case_number = row['index']
        print(f"Processing case {case_number} from {plane_path} for plane {plane}")

        norm_mode = getattr(opt, 'intensity_norm_mode', 'minmax')

        if plane_path.endswith(".pt"):
            loaded = torch.load(plane_path)

            if isinstance(loaded, dict):
                vol_tensor = loaded["volume"].float()
            else:
                vol_tensor = loaded.float()

            raw_volume = vol_tensor.cpu().numpy()

            i_min = getattr(opt, 'intensity_min', None)
            i_max = getattr(opt, 'intensity_max', None)
            volume_norm = apply_intensity_normalization(raw_volume, norm_mode, intensity_min=i_min, intensity_max=i_max)

            origin = [0.0, 0.0, 0.0]

            plane_axis = PLANE_TO_ARRAY_AXIS[plane.lower()]
            spacing = [1.0, 1.0, 1.0]
            if plane_axis == 0:
                spacing[2] = float(opt.gap)
            elif plane_axis == 1:
                spacing[1] = float(opt.gap)
            elif plane_axis == 2:
                spacing[0] = float(opt.gap)

        elif plane_path.endswith(".npy"):
            raw_volume = np.load(plane_path)

            i_min = getattr(opt, 'intensity_min', None)
            i_max = getattr(opt, 'intensity_max', None)
            volume_norm = apply_intensity_normalization(raw_volume, norm_mode, intensity_min=i_min, intensity_max=i_max)

            origin = [0.0, 0.0, 0.0]

            plane_axis = PLANE_TO_ARRAY_AXIS[plane.lower()]
            spacing = [1.0, 1.0, 1.0]
            if plane_axis == 0:
                spacing[2] = float(opt.gap)
            elif plane_axis == 1:
                spacing[1] = float(opt.gap)
            elif plane_axis == 2:
                spacing[0] = float(opt.gap)

        else:
            from_dicom = os.path.isdir(plane_path)
            image = read_volume(plane_path, from_dicom=from_dicom)

            if image is None:
                raise ValueError(f"Failed to load image: {plane_path}")

            image_array = sitk.GetArrayFromImage(image)

            if norm_mode == 'clip':
                i_min = row.get("intensity_min", np.percentile(image_array, 0.05))
                i_max = row.get("intensity_max", np.percentile(image_array, 99.9))
            elif norm_mode == 'fixed_range':
                i_min = opt.intensity_min
                i_max = opt.intensity_max
            else:
                i_min = getattr(opt, 'intensity_min', None)
                i_max = getattr(opt, 'intensity_max', None)

            volume_norm = apply_intensity_normalization(
                image_array,
                norm_mode,
                intensity_min=i_min,
                intensity_max=i_max,
                stretch_percentage=10,
            )

            origin = list(image.GetOrigin())
            spacing = list(image.GetSpacing())

        print(
            f"[DEBUG] case={case_number}, plane={plane}, "
            f"shape_zyx={volume_norm.shape}, "
            f"spacing_xyz={tuple(spacing)}, "
            f"spacing_zyx={tuple(np.array(spacing)[::-1])}"
        )


        if plane_path.endswith(".pt") or plane_path.endswith(".npy"):
            # Raw microscopy tensors do not contain reliable orientation metadata.
            # For EPFL 3-plane data, the CSV/file plane defines the degraded axis.
            axis = PLANE_TO_ARRAY_AXIS[plane.lower()]
        else:
            axis = choose_training_slice_axis(
                volume_zyx=volume_norm,
                spacing_xyz=spacing,
                csv_plane=plane,
            )

        depth = volume_norm.shape[axis]
        start_idx, end_idx = 0, depth

        # 3. Dynamically extract the slice from the correct axis
        for i in range(start_idx, end_idx):
            slice_2d = extract_oriented_slice_by_axis(
                volume_zyx=volume_norm,
                axis=axis,
                slice_index=i,
            )

            slice_tensor = torch.tensor(slice_2d.copy(), dtype=torch.float32).unsqueeze(0)

            information = {
                "case": case_number,
                "slice": i,
                "csv_plane": plane,
                "actual_slice_axis": int(axis),
                "actual_slice_plane": ARRAY_AXIS_TO_PLANE[int(axis)],
                "origin": origin,
                "spacing": spacing,
                "shape_zyx": tuple(volume_norm.shape),
            }

            save_path = os.path.join(output_dir, dataset_type, f"case_{case_number}_slice_{i}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"slice": slice_tensor, "information": information}, save_path)

        print(f"Processed and saved case {case_number} ({dataset_type}), total samples: {end_idx - start_idx}")


def apply_random_gap_mask(slice_tensor, orig_gap, row_or_col=None, offset=None, val=False, domain='MRI', no_degradation=False):
    """
    Simulates Thick-Slice MRI acquisition physics via 1D Gaussian Blur,
    then applies the gap mask.

    When *no_degradation* is True the blur step is skipped entirely (sigma=0):
    known slices are kept at their original resolution.  Use this only for
    qualitative evaluation without GT; it will hurt quantitative scores because
    the model was trained with the degradation applied.
    """
    # Determine Gap Size (with optional jitter during training)
    gap = orig_gap# + random.randint(-1, 1)# TODO Optionally make the gap dynamic during training for augmentation.
    if val:
        gap = orig_gap  # Enforce fixed gap for validation

    H, W = slice_tensor.shape
    device = slice_tensor.device

    # 1. Decide Orientation (Row vs Col)
    # True = Keep Rows (Horizontal stripes), False = Keep Cols (Vertical stripes)
    if row_or_col is None:
        is_row_mode = random.random() < 0.5
    else:
        is_row_mode = row_or_col

    # 2. Apply 1D Gaussian Filter (Physics Simulation)
    sigma = gap / 3.0 if domain == 'MRI' else gap
    axis = 0 if is_row_mode else 1
    if no_degradation:
        slice_blurred = slice_tensor
    elif domain == 'MRI':
        slice_blurred = pytorch_gaussian_filter1d(slice_tensor, sigma=sigma, axis=axis)
    else:
        slice_blurred = pytorch_average_filter1d(slice_tensor, gap=sigma, axis=axis)

    # 3. Create the Mask
    offset = random.randint(0, gap - 1) if offset is None else offset

    if is_row_mode:
        # Keep specific rows
        indices = torch.arange(H, device=device)
        # 1.0 where (index % gap) == offset, else 0.0
        mask = ((indices % gap) == offset).float().unsqueeze(1).expand(H, W)
    else:
        # Keep specific columns
        indices = torch.arange(W, device=device)
        mask = ((indices % gap) == offset).float().unsqueeze(0).expand(H, W)

    # 4. Apply Mask
    # Where mask is 1 (Known): Keep the BLURRED value
    # Where mask is 0 (Gap): Set to MASK_VALUE
    masked_slice = slice_blurred * mask + (MASK_VALUE) * (1 - mask)

    # Metadata for debugging or consistency
    meta = {
        "row_or_col": "row" if is_row_mode else "col",
        "gap": int(gap),
        "offset": int(offset)
    }

    return masked_slice, meta

class PreprocessedDataset(Dataset):
    def __init__(self, data_dirs, opt, random_normalization=True, val = False):
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        self.data_paths = []
        for data_dir in data_dirs:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.pt'):
                        self.data_paths.append(os.path.join(root, file))
        self.gap = opt.gap
        self.random_normalization = random_normalization
        self.val = val
        self.domain = opt.domain
        self.patch_size = opt.patch_size
        self.alpha_stretch = 0.4
        self.no_degradation = getattr(opt, 'no_degradation', False)
    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        sample = torch.load(self.data_paths[idx])
        info = sample['information']

        # 1. Start with the RAW high-res slice
        data = sample['slice']  # Shape: (1, H, W)

        crop_size = self.patch_size
        data = crop_or_pad_to_patch_size(
            data=data,
            crop_size=crop_size,
            domain=self.domain,
            random_crop=not self.val,
        )

        # 2. Apply Random Normalization (Augmentation)
        if self.random_normalization:
            data = random_normalization(data)

        if not self.val and self.domain == 'microscopy':
            k_rotations = random.randint(0, 3)
            if k_rotations > 0:
                # Rotate the tensor k_rotations * 90 degrees along the spatial dimensions (H, W)
                data = torch.rot90(data, k_rotations, dims=[1, 2])

        # 3. Apply 1D Gaussian Blur & Masking
        # The 1D blur simulates the "thick slice" physics dynamically.
        data_masked_raw, meta = apply_random_gap_mask(
            data.squeeze(0),  # Pass raw data (H, W)
            orig_gap=self.gap,
            row_or_col=None,  # Randomize orientation
            val=self.val,
            domain=self.domain,
            no_degradation=self.no_degradation,
        )
        info['mask_meta'] = meta

        # 4. Prepare Channels
        # Identify where the gaps are MASK_VALUE
        missing = (data_masked_raw == MASK_VALUE)

        # Channel 0: Intensity
        intensity = data_masked_raw.clone()

        # Channel 1: Mask
        # 1.0 = Real Data, 0.0 = Gap
        mask = (~missing).float()

        # Stack into (2, H, W)
        input_tensor = torch.stack([intensity, mask], dim=0)

        if self.domain == 'microscopy':
            target_gt = data.clone()
            direction = meta['row_or_col']
            if not self.val:
                # Stretch the target GT using the inverse degradation calculation
                target_gt = apply_inverse_degradation_stretch(
                    target_gt, direction=direction, gap=self.gap, alpha=self.alpha_stretch
                )
        else:
            target_gt = data.clone()

        return input_tensor, target_gt, info


def random_normalization(data):
    """
    Randomly clips the lower intensities (black crushing) and stretches
    the remaining histogram to [-1, 1].
    Acts as data augmentation for contrast/brightness.
    """
    # 1. Identify valid pixels (exclude background -1)
    # We use a small epsilon because values might not be exactly -1.0
    valid_mask = data > -0.99
    valid_pixels = data[valid_mask]

    # Safety check: if image is mostly empty/background, return as is
    if valid_pixels.numel() == 0 or (valid_pixels.numel() / data.numel() < 0.05):
        return data

    # 2. Pick a random percentile (0% to 12%)
    # This determines how much of the "dark" signals we crush to black
    x = random.uniform(0, 12)
    min_val = torch.quantile(valid_pixels, x / 100.0)
    max_val = torch.max(valid_pixels)

    # Safety check: avoid division by zero if image is flat
    if max_val <= min_val:
        return data

    # 3. Apply Contrast Stretching
    # Clamp: Everything below min_val becomes min_val
    data_clamped = torch.clamp(data, min=min_val, max=max_val)

    # Normalize to [0, 1]
    data_norm = (data_clamped - min_val) / (max_val - min_val)

    # Scale to [-1, 1]
    data_scaled = (data_norm * 2) - 1

    return data_scaled

def pytorch_gaussian_filter1d(tensor, sigma, axis, truncate=4.0):
    """
    Native PyTorch exact mathematical equivalent to scipy.ndimage.gaussian_filter1d.
    Keeps everything on the same device (CPU or GPU) without Numpy conversions.
    """
    if sigma <= 0.0:
        return tensor

    # Scipy calculates radius like this by default:
    radius = int(truncate * sigma + 0.5)
    kernel_size = 2 * radius + 1

    # Build the 1D Gaussian kernel
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor.device)
    kernel1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel1d = kernel1d / kernel1d.sum() # Normalize

    # F.conv2d expects shape (Batch, Channels, Height, Width)
    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)

    if axis == 0:  # Blur vertically (along H)
        kernel2d = kernel1d.view(1, 1, kernel_size, 1)
        # Change 'reflect' to 'replicate' to match SciPy
        padded = F.pad(tensor_4d, pad=(0, 0, radius, radius), mode='replicate')
    elif axis == 1:  # Blur horizontally (along W)
        kernel2d = kernel1d.view(1, 1, 1, kernel_size)
        # Change 'reflect' to 'replicate' to match SciPy
        padded = F.pad(tensor_4d, pad=(radius, radius, 0, 0), mode='replicate')
    else:
        raise ValueError("Only axis 0 or 1 supported")

    # Apply the convolution natively
    blurred = F.conv2d(padded, kernel2d)

    # Return to original (H, W) shape
    return blurred.squeeze(0).squeeze(0)


def pytorch_average_filter1d(tensor, gap, axis):
    """
    Native PyTorch 1D Average Pooling.
    Simulates thick-slice physics exactly like F.avg_pool1d with stride=gap.
    Creates a piecewise-constant (blocky) image so that random mask offsets
    always sample the true block-average.
    """
    if gap <= 1:
        return tensor

    gap = int(round(gap))
    H, W = tensor.shape

    # Shape: (1, 1, H, W)
    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)

    if axis == 0:  # Block average vertically (along H)
        # 1. Pad H to be a perfect multiple of gap to prevent dropping the edge
        pad_h = (gap - (H % gap)) % gap
        if pad_h > 0:
            padded = F.pad(tensor_4d, pad=(0, 0, 0, pad_h), mode='replicate')
        else:
            padded = tensor_4d

        # 2. Block Average (stride=gap matches efpl_create_dataset2 exactly)
        pooled = F.avg_pool2d(padded, kernel_size=(gap, 1), stride=(gap, 1))

        # 3. Stretch the averaged pixels back out into chunks
        upsampled = torch.repeat_interleave(pooled, gap, dim=2)

        # 4. Crop away the padding
        blurred = upsampled[:, :, :H, :]

    elif axis == 1:  # Block average horizontally (along W)
        # 1. Pad W to be a perfect multiple of gap
        pad_w = (gap - (W % gap)) % gap
        if pad_w > 0:
            padded = F.pad(tensor_4d, pad=(0, pad_w, 0, 0), mode='replicate')
        else:
            padded = tensor_4d

        # 2. Block Average
        pooled = F.avg_pool2d(padded, kernel_size=(1, gap), stride=(1, gap))

        # 3. Stretch the averaged pixels back out into chunks
        upsampled = torch.repeat_interleave(pooled, gap, dim=3)

        # 4. Crop away the padding
        blurred = upsampled[:, :, :, :W]
    else:
        raise ValueError("Only axis 0 or 1 supported")

    return blurred.squeeze(0).squeeze(0)


def torch_interp(x, xp, fp):
    """
    Pure PyTorch 1D linear interpolation.
    x: target x values to interpolate
    xp: known x values (must be sorted)
    fp: known y values
    """
    # Clamp indices to ensure we don't go out of bounds
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)

    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]

    dx = x1 - x0
    # Prevent division by zero if there are duplicate quantiles
    dx = torch.where(dx == 0, torch.tensor(1e-6, device=x.device, dtype=x.dtype), dx)

    return y0 + (y1 - y0) * (x - x0) / dx


def apply_inverse_degradation_stretch(tensor, direction, gap, alpha=1.0):
    """
    Dynamically learns how block-averaging squashes the intensity distribution,
    and applies the inverse stretch to correct the Ground Truth.
    Supports [H, W], [C, H, W], or batched [B, C, H, W] tensors.
    """
    if alpha <= 0.0:
        return tensor

    device = tensor.device
    dtype = tensor.dtype

    # 1. Handle shapes seamlessly
    original_shape = tensor.shape
    if tensor.dim() == 2:
        tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor_4d = tensor.unsqueeze(0)
    else:
        tensor_4d = tensor

    B, C, H, W = tensor_4d.shape

    # 2. Create the LL-R (Low-Low-Res) version using exact block averaging
    if direction == 'row':  # Average vertically (along H)
        pad_h = (gap - (H % gap)) % gap
        padded = F.pad(tensor_4d, pad=(0, 0, 0, pad_h), mode='replicate') if pad_h > 0 else tensor_4d
        pooled = F.avg_pool2d(padded, kernel_size=(gap, 1), stride=(gap, 1))
        upsampled = torch.repeat_interleave(pooled, gap, dim=2)
        llr_tensor = upsampled[:, :, :H, :W]
    else:  # Average horizontally (along W)
        pad_w = (gap - (W % gap)) % gap
        padded = F.pad(tensor_4d, pad=(0, pad_w, 0, 0), mode='replicate') if pad_w > 0 else tensor_4d
        pooled = F.avg_pool2d(padded, kernel_size=(1, gap), stride=(1, gap))
        upsampled = torch.repeat_interleave(pooled, gap, dim=3)
        llr_tensor = upsampled[:, :, :H, :W]

    # 3. Apply stretch per image in the batch
    corrected_tensor = tensor_4d.clone()
    q_steps = torch.linspace(0, 1, steps=200, device=device, dtype=dtype)

    for b in range(B):
        for c in range(C):
            orig_slice = tensor_4d[b, c]
            llr_slice = llr_tensor[b, c]

            # Isolate valid tissue (ignore black padding)
            valid_mask = orig_slice > -0.99
            if not valid_mask.any():
                continue

            orig_valid = orig_slice[valid_mask]
            llr_valid = llr_slice[valid_mask]

            # Compute quantiles entirely in PyTorch
            orig_q = torch.quantile(orig_valid, q_steps)
            llr_q = torch.quantile(llr_valid, q_steps)

            # Calculate shift and extrapolate target distribution
            shift = orig_q - llr_q
            target_q = orig_q + (alpha * shift)

            # Keep strictly within [-1.0, 1.0] image bounds
            target_q = torch.clamp(target_q, -1.0, 1.0)

            # Map original pixels to the new stretched distribution
            matched_pixels = torch_interp(orig_valid, orig_q, target_q)

            # Assign back to the corrected tensor
            corrected_tensor[b, c][valid_mask] = matched_pixels

    # Return exactly the shape that was passed in
    return corrected_tensor.view(original_shape)
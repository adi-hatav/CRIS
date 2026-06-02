import os

import matplotlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from constants import INPUT_NC, MASK_VALUE, PLANE_TO_ARRAY_AXIS, PLANE_TO_SITK_AXIS
from models import UNetWithAttention
from preprocess import load_volume_from_path
from utils import (
    build_model_input_from_slice,
    center_pad_slice_for_inference,
    map_imputed_slice_to_original,
    resample_isotropic,
)

matplotlib.use("Agg")


def load_model(model_path, opt):
    device = opt.device
    model = UNetWithAttention(
        in_channels=INPUT_NC,
        out_channels=opt.output_nc,
        base_filters=opt.base_filters,
        window_size=opt.window_size,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def save_volume_as_torch(volume, spacing, origin, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    volume = np.clip(volume, -1, 1).astype(np.float16)
    torch.save(
        {
            "volume": torch.tensor(volume, dtype=torch.float16),
            "spacing": spacing,
            "origin": origin,
        },
        save_path,
    )


def save_volume_as_nifti(volume, spacing, origin, direction, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    volume = np.clip(volume, -1, 1).astype(np.float32)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(origin))
    image.SetDirection(tuple(direction))
    sitk.WriteImage(image, save_path)


def insert_blank_slices_plane(volume, gap, plane, opt):
    plane = plane.lower()
    axis = PLANE_TO_ARRAY_AXIS[plane]
    slices_list = []
    num_slices = volume.shape[axis]

    if axis == 0:
        ref_slice = volume[0, :, :]
    elif axis == 1:
        ref_slice = volume[:, 0, :]
    else:
        ref_slice = volume[:, :, 0]

    blank_slice = torch.full_like(ref_slice, MASK_VALUE)
    is_microscopy = opt.domain == "microscopy"

    for i in range(num_slices):
        if axis == 0:
            slice_i = volume[i, :, :]
        elif axis == 1:
            slice_i = volume[:, i, :]
        else:
            slice_i = volume[:, :, i]

        slices_list.append(slice_i)
        if is_microscopy or i < num_slices - 1:
            slices_list.extend([blank_slice] * (gap - 1))

    return torch.stack(slices_list, dim=axis)


def get_prediction(model, slice_2d, opt):
    if torch.all((slice_2d == -1) | (slice_2d == MASK_VALUE)):
        return torch.full_like(slice_2d, -1)

    original_shape = slice_2d.shape
    slice_resized, orig_h, orig_w = center_pad_slice_for_inference(
        slice_2d, opt.patch_size, fill_value=-1
    )
    with torch.no_grad():
        input_tensor = build_model_input_from_slice(slice_resized, opt.device)
        imputed_slice = model(input_tensor).squeeze(0).squeeze(0).cpu()
    return map_imputed_slice_to_original(
        imputed_slice, original_shape, opt.patch_size, orig_h, orig_w, slice_2d.dtype
    )


def remove_remains(volume_array, gap, dicom_series=False):
    gap = int(gap)
    if gap <= 1:
        return volume_array

    if dicom_series:
        z_end = volume_array.shape[0] - (volume_array.shape[0] - 1) % gap
        return volume_array[:z_end, :, :]

    z_end = volume_array.shape[0] - (volume_array.shape[0] - 1) % gap
    y_end = volume_array.shape[1] - (volume_array.shape[1] - 1) % gap
    x_end = volume_array.shape[2] - (volume_array.shape[2] - 1) % gap
    return volume_array[:z_end, :y_end, :x_end]


def validate_plane_matches_spacing(spacing, plane, tolerance=1.2):
    plane = plane.lower()
    spacing_zyx = np.array(spacing)[::-1]
    min_spacing = float(np.min(spacing_zyx))
    max_spacing = float(np.max(spacing_zyx))
    if max_spacing / min_spacing < tolerance:
        return

    expected_axis = PLANE_TO_ARRAY_AXIS[plane]
    actual_axis = int(np.argmax(spacing_zyx))
    if actual_axis != expected_axis:
        raise ValueError(
            f"Plane/spacing mismatch: plane={plane}, expected axis={expected_axis}, "
            f"actual axis={actual_axis}, spacing_xyz={tuple(spacing)}"
        )


def process_and_save_volume(
    volume, spacing, origin, direction, save_dir, dataset_type, plane, case_index, opt, view, dicom_series=False
):
    save_path = os.path.join(save_dir, "isotropic_volumes", dataset_type, plane, view, str(case_index))
    os.makedirs(save_path, exist_ok=True)

    if opt.domain == "microscopy":
        volume_array = volume.numpy() if isinstance(volume, torch.Tensor) else np.asarray(volume)
        save_volume_as_torch(volume_array, spacing, origin, os.path.join(save_path, "CRIS_volume.pt"))
        return volume_array

    volume_array = volume.detach().cpu().numpy() if isinstance(volume, torch.Tensor) else np.asarray(volume)
    volume_sitk = sitk.GetImageFromArray(volume_array)
    volume_sitk.SetSpacing(spacing)
    volume_sitk.SetOrigin(origin)
    volume_sitk.SetDirection(direction)
    volume_sitk = resample_isotropic(volume_sitk)

    new_spacing = volume_sitk.GetSpacing()
    new_origin = volume_sitk.GetOrigin()
    new_direction = volume_sitk.GetDirection()
    volume_array = remove_remains(sitk.GetArrayFromImage(volume_sitk), opt.gap, dicom_series=dicom_series)

    save_volume_as_nifti(
        volume_array,
        new_spacing,
        new_origin,
        new_direction,
        os.path.join(save_path, "CRIS_volume.nii.gz"),
    )
    save_volume_as_torch(volume_array, new_spacing, new_origin, os.path.join(save_path, "CRIS_volume.pt"))
    return volume_array


def _impute_plane_stack(volume, model, opt, axis_index, empty_volume):
    result = empty_volume.clone()
    for i in range(volume.shape[axis_index]):
        if axis_index == 0:
            slice_2d = volume[i]
            result[i] = get_prediction(model, slice_2d, opt)
        elif axis_index == 1:
            slice_2d = volume[:, i, :]
            result[:, i, :] = get_prediction(model, slice_2d, opt)
        else:
            slice_2d = volume[:, :, i]
            result[:, :, i] = get_prediction(model, slice_2d, opt)
        torch.cuda.empty_cache()
    return torch.flip(result, dims=(0, 2))


def gap_imputation(opt, model, input_dir, save_dir, case_index, plane, dataset_type):
    dicom_series = os.path.isdir(input_dir)
    volume, origin, spacing, size, direction = load_volume_from_path(input_dir, opt, scale=True)
    plane = plane.lower()
    array_axis = PLANE_TO_ARRAY_AXIS[plane]

    spacing_zyx = np.array(spacing)[::-1]
    gap = max(int(round(spacing_zyx[array_axis] / np.min(spacing_zyx))), 1)

    if not isinstance(volume, torch.Tensor):
        volume = torch.from_numpy(volume)

    volume = insert_blank_slices_plane(volume, gap, plane, opt)
    volume = torch.flip(volume, dims=(0, 2))

    sitk_axis = PLANE_TO_SITK_AXIS[plane]
    spacing[sitk_axis] = spacing[sitk_axis] / gap
    if opt.domain == "microscopy":
        size[sitk_axis] = size[sitk_axis] * gap
    else:
        size[sitk_axis] = size[sitk_axis] + (size[sitk_axis] - 1) * (gap - 1)

    empty_volume = torch.full_like(volume, -1)
    views = {}

    if plane != "axial":
        views["axial"] = process_and_save_volume(
            _impute_plane_stack(volume, model, opt, 0, empty_volume),
            spacing, origin, direction, save_dir, dataset_type, plane, case_index, opt, "axial",
            dicom_series,
        )
    if plane != "coronal":
        views["coronal"] = process_and_save_volume(
            _impute_plane_stack(volume, model, opt, 1, empty_volume),
            spacing, origin, direction, save_dir, dataset_type, plane, case_index, opt, "coronal",
            dicom_series,
        )
    if plane != "sagittal":
        views["sagittal"] = process_and_save_volume(
            _impute_plane_stack(volume, model, opt, 2, empty_volume),
            spacing, origin, direction, save_dir, dataset_type, plane, case_index, opt, "sagittal",
            dicom_series,
        )

    if plane == "axial":
        plane_volume = (views["coronal"] + views["sagittal"]) / 2
    elif plane == "coronal":
        plane_volume = (views["axial"] + views["sagittal"]) / 2
    else:
        plane_volume = (views["axial"] + views["coronal"]) / 2

    save_path = os.path.join(save_dir, "isotropic_volumes", dataset_type, plane, plane, str(case_index))
    save_volume_as_torch(plane_volume, spacing, origin, os.path.join(save_path, "CRIS_volume.pt"))

    if opt.domain.lower() == "mri":
        validate_plane_matches_spacing(spacing, plane)

    print(f"Finished case {case_index} ({dataset_type})")


def export_isotropic_volumes(opt, plane, model_path, saving_dir):
    os.makedirs(saving_dir, exist_ok=True)
    df = pd.read_csv(opt.csv_path)
    df = df.loc[df[plane].notnull()]

    # intensity_min / intensity_max columns are optional — only needed for 'clip' mode.
    has_intensity_min = "intensity_min" in df.columns
    has_intensity_max = "intensity_max" in df.columns

    required_cols = ["index", plane, "dataset"]
    available_cols = required_cols + [c for c in ["intensity_min", "intensity_max"] if c in df.columns]
    df = df[available_cols]

    model = load_model(model_path, opt)
    norm_mode = getattr(opt, 'intensity_norm_mode', 'minmax')

    for _, row in df.iterrows():
        case_index = row["index"]
        dataset_type = row["dataset"]
        path = row[plane]

        # Set per-case intensity bounds when the mode requires them.
        if norm_mode == 'clip':
            opt.intensity_min = float(row["intensity_min"]) if has_intensity_min else opt.intensity_min
            opt.intensity_max = float(row["intensity_max"]) if has_intensity_max else opt.intensity_max
        # 'fixed_range': opt.intensity_min / opt.intensity_max stay constant (set once via CLI).
        # 'stretch' and 'minmax': bounds are auto-computed per volume; no opt fields needed.

        print(
            f"Processing case {case_index} ({dataset_type}) from path: {path} "
            f"[norm_mode={norm_mode}]"
        )
        if dataset_type not in ["test"]:
            continue
        gap_imputation(opt, model, path, saving_dir, case_index, plane, dataset_type)

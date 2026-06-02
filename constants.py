"""Shared constants for the CRIS core package."""

INPUT_NC = 2  # intensity + gap mask
MASK_VALUE = 0.0

PLANE_TO_ARRAY_AXIS = {
    "axial": 0,
    "coronal": 1,
    "sagittal": 2,
}

PLANE_TO_SITK_AXIS = {
    "axial": 2,
    "coronal": 1,
    "sagittal": 0,
}

ARRAY_AXIS_TO_PLANE = {v: k for k, v in PLANE_TO_ARRAY_AXIS.items()}

"""
CRIS — generic launcher example (no hardcoded data paths).

Usage:
    conda activate cris
    python running_script_example.py --csv_path MANIFEST.csv --main_dir OUTPUT/ [options]

Presets set recommended hyperparameters for common setups; override any flag explicitly.
"""

import argparse
import subprocess
import sys


PRESETS = {
    "brain_mri": {
        "default_plane": "coronal",
        "planes": "coronal,axial,sagittal",
        "patch_size": 256,
        "gap": 5,
        "learning_rate": 0.0001,
        "batch_size": 24,
        "base_filters": 118,
        "window_size": 8,
        "domain": "MRI",
        "n_epochs": 120,
        "patience": 20,
        "build_slice_cache": False,
        "export_isotropic_volumes": True,
        "save_epoch_freq": 50,
    },
    "abdomen_mri": {
        "default_plane": "coronal",
        "planes": "axial,coronal",
        "patch_size": 512,
        "gap": 6,
        "learning_rate": 0.00005,
        "batch_size": 16,
        "base_filters": 118,
        "window_size": 8,
        "domain": "MRI",
        "n_epochs": 120,
        "patience": 20,
        "build_slice_cache": True,
        "export_isotropic_volumes": True,
        "save_epoch_freq": 50,
    },
    "microscopy": {
        "default_plane": "axial",
        "planes": "coronal,axial,sagittal",
        "patch_size": 128,
        "gap": 8,
        "learning_rate": 0.00005,
        "batch_size": 64,
        "base_filters": 118,
        "window_size": 4,
        "domain": "microscopy",
        "n_epochs": 120,
        "patience": 15,
        "build_slice_cache": True,
        "export_isotropic_volumes": True,
        "save_epoch_freq": 80,
    },
    "epfl_microscopy": {
        "default_plane": "axial",
        "planes": "coronal,axial,sagittal",
        "patch_size": 128,
        "gap": 4,
        "learning_rate": 0.00005,
        "batch_size": 24,
        "base_filters": 118,
        "window_size": 4,
        "domain": "microscopy",
        "n_epochs": 150,
        "patience": 15,
        "build_slice_cache": True,
        "export_isotropic_volumes": True,
        "save_epoch_freq": 50,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CRIS training or evaluation via train.py (paths supplied on CLI)."
    )
    parser.add_argument("--csv_path", required=True, help="Path to dataset CSV manifest")
    parser.add_argument("--main_dir", required=True, help="Output directory for models and exports")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="brain_mri",
        help="Hyperparameter preset (default: brain_mri)",
    )
    parser.add_argument("--phase", choices=("train", "evaluation"), default="train")
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU indices")
    parser.add_argument("--pre_trained", default=None, help="Optional checkpoint directory to resume from")
    parser.add_argument("--dataset_path", default=None, help="Optional existing slice-cache directory")
    parser.add_argument("--use_only_train", action="store_true", help="Train on train split only")
    return parser


def build_command(args: argparse.Namespace) -> str:
    cfg = dict(PRESETS[args.preset])

    command = (
        "python -u train.py "
        f"--phase {args.phase} "
        f"--planes {cfg['planes']} "
        f"--main_dir {args.main_dir} "
        f"--csv_path {args.csv_path} "
        f"--patch_size {cfg['patch_size']} "
        f"--gpu_ids {args.gpu_ids} "
        f"--batch_size {cfg['batch_size']} "
        f"--n_epochs {cfg['n_epochs']} "
        f"--default_plane {cfg['default_plane']} "
        f"--learning_rate {cfg['learning_rate']} "
        f"--gap {cfg['gap']} "
        f"--base_filters {cfg['base_filters']} "
        f"--window_size {cfg['window_size']} "
        f"--patience {cfg['patience']} "
        f"--save_epoch_freq {cfg['save_epoch_freq']} "
        f"--domain {cfg['domain']} "
    )

    if cfg.get("build_slice_cache"):
        command += "--build_slice_cache True "
    if cfg.get("export_isotropic_volumes"):
        command += "--export_isotropic_volumes True "

    if args.use_only_train:
        command += "--use_only_train True "
    if args.pre_trained:
        command += f"--pre_trained {args.pre_trained} "
    if args.dataset_path:
        command += f"--dataset_path {args.dataset_path} "

    return command


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = build_command(args)
    print("Running CRIS")
    print(command)
    return subprocess.run(command, shell=True).returncode


if __name__ == "__main__":
    sys.exit(main())

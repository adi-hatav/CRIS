# CLI reference

All flags are defined in [`options.py`](../options.py). Run `python train.py --help` to print defaults.

---

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `train` | `train` or `evaluation` (isotropic volume export on the test split) |
| `--csv_path` | *(required)* | Path to the dataset manifest CSV |
| `--main_dir` | *(required)* | Root directory for all outputs |
| `--planes` | *(required)* | Comma-separated planes to train: `coronal`, `axial`, `sagittal`, or any combination |
| `--default_plane` | `coronal` | Plane used as the source for isotropic volume export |
| `--patch_size` | *(required)* | H and W of 2D training patches (pixels). Use as large a patch as your GPU allows. As a guide, if volumes are H × W × D, set `patch_size` to at most 1.25 × min(H, W, D) to limit padding |
| `--gap` | `5` | Simulated acquisition gap — every `gap`-th slice is kept; `(gap−1)` slices are masked |
| `--domain` | `MRI` | `MRI` or `microscopy` — controls loss schedule and dataloader behaviour |
| `--n_epochs` | `150` | Maximum training epochs |
| `--patience` | `15` | Early-stopping patience (epochs without validation improvement) |
| `--batch_size` | `16` | Training batch size |
| `--learning_rate` | `0.0005` | Initial learning rate (Adam) |
| `--base_filters` | `96` | Base channel width — scales all network layers |
| `--window_size` | `8` | Swin-Transformer window size in the bottleneck. **Must match `patch_size`** — see [compatibility](#patch_size--window_size-compatibility) |
| `--output_nc` | `1` | Output channels |
| `--gpu_ids` | `0` | GPU index/indices (`0`, `0,1`, `-1` for CPU) |
| `--intensity_norm_mode` | `minmax` | `minmax`, `stretch`, `clip`, or `fixed_range` — see [README intensity section](../README.md#intensity-normalisation) |
| `--intensity_min` | `0.0` | Lower clip bound for `fixed_range`; fallback for `clip` |
| `--intensity_max` | `2048.0` | Upper clip bound for `fixed_range`; fallback for `clip` |
| `--save_epoch_freq` | `50` | Checkpoint save interval (epochs) |
| `--build_slice_cache` | `False` | Pre-extract all 2D slices to `.pt` files under `<main_dir>/CRIS_Dataset/` |
| `--dataset_path` | `<main_dir>/CRIS_Dataset` | Root of pre-built slice cache (override to reuse a cache across runs) |
| `--export_isotropic_volumes` | `False` | After training, run isotropic export on the test split |
| `--pre_trained` | `None` | Path to a previous `main_dir` to resume from its best checkpoint |
| `--use_only_train` | `False` | Use all CSV rows for training (no val/test split) |
| `--no_degradation` | `False` (off) | Skip the in-plane 1-D blur that simulates thick-slice physics. See [No-degradation mode](#no-degradation-mode) |

---

## `patch_size` / `window_size` compatibility

`patch_size` and `window_size` are **not independent**. An incompatible pair raises an assertion at model initialisation and the run will not start.

**Why?** The U-Net applies five successive ×2 max-pool operations before the Swin-Transformer bottleneck, so the bottleneck spatial size is `patch_size / 32`. `window_size` must evenly divide that size.

| `patch_size` | Bottleneck size (/ 32) | Valid `window_size` values | **Recommended** |
|---|---|---|---|
| `128` | 4 | 1, 2, 4 | **4** |
| `192` | 6 | 1, 2, 3, 6 | **6** |
| `256` | 8 | 1, 2, 4, 8 | **8** |
| `384` | 12 | 1, 2, 3, 4, 6, 12 | **12** |
| `512` | 16 | 1, 2, 4, 8, 16 | **16** |

Any `patch_size` **divisible by 32** also works (e.g. `160`, `224`, `320`, `448`). Set `window_size` to `patch_size / 32` and prefer the **largest** valid value for the best bottleneck attention. Example: `patch_size=128` with `window_size=8` will **not** run.

> **Rule of thumb:** recommended `window_size` = `patch_size / 32` (with `patch_size` divisible by 32).

**Typical pairings**

| Use case | `patch_size` | `window_size` |
|----------|--------------|---------------|
| Microscopy | `128` | `4` |
| MRI (default) | `256` | `8` |
| MRI (large GPU) | `512` | `16` |

---

## No-degradation mode

By default, CRIS blurs the known (in-plane) slices before masking them, simulating the physics of thick-slice acquisition:

- **MRI** — 1-D Gaussian with `sigma = gap / 3.0`
- **Microscopy** — 1-D average filter with `kernel = gap`

This simulates the point-spread function of thick-slice acquisition so that the training and inference distributions match.

Adding `--no_degradation` disables the blur entirely (`sigma = 0`): the network receives the original high-resolution in-plane slices as-is, with no filtering applied.

This is useful when you are running inference on real thick-slice data **without** a ground-truth isotropic reference and you care about visual quality. In that setting, the sharp input slices give the network more information to work with and can produce the best-looking results.

> **Warning — do not use for quantitative GT evaluation.** Do not use `--no_degradation` when computing PSNR / SSIM against a GT volume. The model was trained with the degradation applied, so removing it at inference time creates a distribution mismatch that will lower quantitative scores.

### When to use it

| Scenario | Recommendation |
|----------|----------------|
| Real thick-slice data, no GT available, qualitative inspection | **Use `--no_degradation`** — sharp input slices give the network more signal to work with and produce visually cleaner results |
| Synthetic downsampling experiment with a GT isotropic reference | **Do not use** — the training distribution assumed the blur; removing it creates a mismatch that reduces PSNR/SSIM |
| Benchmark comparison against other super-resolution methods | **Do not use** — quantitative scores will be unreliable |

### Usage

```bash
python train.py \
  --phase         evaluation \
  --csv_path      /path/to/dataset.csv \
  --main_dir      /path/to/run/ \
  --patch_size    256 \
  --gpu_ids       0 \
  --no_degradation
```

`--no_degradation` is a boolean flag (no value required). It defaults to `False` (blur enabled).


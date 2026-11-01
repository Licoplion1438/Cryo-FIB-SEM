# Preprocessing — cryo-FIB-SEM input preparation

This module prepares real acquired volumes for the two trained stages. The scripts are **separate processing operations**, not an additional learned model. Their order in the manuscript workflow is uint8 conversion, optional/appropriate VSNR for curtaining suppression, and global translation-only XY alignment **when required**, followed by Stage-1 inference.

## This module's layout

```text
preprocessing/
├── README.md
├── step1_convert_to_stacked_uint8.py
├── step2_vsnr.py
└── step3_alignment_selection_crop.py
```

All examples below run from the **repository root**. The input data are not included in GitHub. These utilities are not responsible for training either learned stage.

## 1. Convert a volume to a uint8 TIFF stack

`step1_convert_to_stacked_uint8.py` supports compatible microscopy stacks and file/directory formats and writes a ZYX uint8 TIFF. It estimates **one global 0.5th–99.5th percentile intensity window** from the volume; it does not rescale each slice independently.

```bash
python code/preprocessing/step1_convert_to_stacked_uint8.py \
  --input /path/to/acquired_volume \
  --output experiments/preprocessing/input_uint8_stacked.tif
```

It does not align, crop, denoise, or apply Stage-2 domain transfer. Input format-specific `--hdf-key` / `--npz-key` options are available when applicable.

## 2. Suppress curtaining with VSNR

`step2_vsnr.py` wraps the external `pyvsnr` library (the script imports `pyvsnr`; obtain the package from its upstream project) and writes a uint8 TIFF stack with the original ZYX shape. The manuscript results used **VSNR noise level 80**, Gabor scales 1/30 pixels, up to 30 iterations, with optional small orientation sweeps; intensity is robustly matched back to the input slice's median and MAD. VSNR runs on CPU or CUDA (not MPS in this script).

```bash
python code/preprocessing/step2_vsnr.py \
  --input experiments/preprocessing/input_uint8_stacked.tif \
  --output experiments/preprocessing/input_uint8_vsnr.tif \
  --device cpu \
  --noise-level 80
```

Use `--theta-sweep 5` if you want the `-5°` to `+5°` integer-angle sweep on compatible data; the script's default sweep depends on backend. VSNR requires the separately installed `pyvsnr` package and its dependencies (CuPy for CUDA). This module has no dedicated Conda YAML in the current repository.

## 3. Select slices and perform global XY alignment when needed

The final preprocessing script performs translation-only global alignment and a common-overlap crop. It operates **before Stage-1** and is distinct from the patch-local center-referenced alignment inside Stage-2. The 3 in-house manuscript datasets underwent global alignment, whereas **EMPIAR-11275 was already aligned** and did not need it.

```bash
python code/preprocessing/step3_alignment_selection_crop.py \
  --input experiments/preprocessing/input_uint8_vsnr.tif \
  --output-dir experiments/preprocessing/aligned
```

Its main final stack is `experiments/preprocessing/aligned/aligned_stack.tif` (other QC outputs are also written). Pass this aligned stack to Stage-1. For datasets already aligned, pass the prepared uint8 VSNR output directly to Stage-1 instead.

If specifying a Z range, `--start-index` and `--end-index` are **zero-based**, inclusive; `--remove-slices` accepts **original one-based slice numbers**. This script uses bilinear interpolation for global XY translations and crops to the common valid field. It is not a nonrigid registration tool.

## Publication and reproducibility

- The reported example datasets were processed with **the same Stage-1 and Stage-2 checkpoints** after their respective input-preparation steps.
- Global alignment and VSNR are configurable preprocessing choices; they must not be confused with Stage-1's learned artifact suppression or Stage-2's internal intensity preprocessing.
- These scripts preserve acquired Z ordering (except if the user explicitly excludes unusable slices); no additional acquired Z planes are created.
- Run `python code/preprocessing/<script_name>.py --help` for the complete supported interface. See the [project README](../../README.md) for the overall workflow.

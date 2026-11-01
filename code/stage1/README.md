# Stage 1 — Cryo-FIB-SEM Artifact Removal

**Maintainer:** Ningyan Zhang  
**Documentation updated:** November 1, 2026  
**Location in the repository:** `code/stage1/`

For the complete pipeline, begin with the [project README](../../README.md). All relative CLI paths below assume the repository root as the working directory.

## Overview

Stage 1 removes image-domain artifacts from cryo-FIB-SEM data using a supervised, **single-slice (2D)** restoration network. Its primary objective is artifact suppression while retaining structural contrast; the downstream membrane-refinement task belongs to Stage 2.

The module provides the complete Stage-1 data-construction and model pipeline:

1. Analyze real charging examples and construct a reusable charging-template library (four preparatory scripts for source-data regeneration).
2. Map clean JRC Liver images to four cryo-like intensity ranges.
3. Synthesize paired clean/degraded images using charging and other artifact families.
4. Train the 2D artifact-removal network on paired, full-resolution images.
5. Run a pretrained checkpoint on a uint8 TIFF stack.

**Checkpoint inference does not require training data.** To reproduce training from scratch, obtain the separately distributed clean JRC blocks and annotated cryo-FIB-SEM charging-source slices, then regenerate the charging library, four-level clean images, and synthetic training pairs using the provided scripts. Generated synthetic data are not bundled with this repository.

Stage 1 processes Z slices independently. It **does not** align slices, interpolate along Z, perform domain transfer, or apply a separate membrane-sharpening stage during inference.

## This module's layout

This directory contains only Stage-1 code and its installation files; source images, masks, generated data, and weights are stored separately. See the [project README](../../README.md) for the repository-level layout.

```text
stage1/
├── README.md
├── environment/
├── step1_build_charging_library/
├── step2_prepare_clean_data.py
├── step3_synthesize_artifacts.py
├── step4_train.py
└── step5_inference.py
```

The charging-library subfolder contains Steps 1a–1d. All examples below are run **from the repository root**. The clean blocks are expected at `data/JRC/clean_blocks_1024_z16/` after obtaining the separately distributed training-source archive; this local `data/` directory is excluded from Git.

## Requirements

Stage 1 has its **own** installation environment (see
[`environment/ENVIRONMENT.md`](environment/ENVIRONMENT.md)). The environment files are derived from
real macOS package snapshots and preserve the versions used for the recorded
MPS tests.

| Host platform | One-command environment creation |
| --- | --- |
| Apple Silicon macOS | `conda env create -f code/stage1/environment/environment_macos.yml` |
| Linux x86_64 with NVIDIA GPU | `conda env create -f code/stage1/environment/environment_cuda.yml` |

After installation, use `conda activate cryo-fib-sem-stage1`.
The source environment used Python **3.8.20**. PyTorch builds differ
between MPS and CUDA; the Linux CUDA dependency set is not yet runtime-tested.
`environment/requirements.txt` documents the direct non-PyTorch Python dependencies; the
YAML files already provide complete installation instructions, including Torch.

All subsequent code examples assume the working directory is the repository
root, for example:

```bash
cd /path/to/Cryo-FIB-SEM
```

On other machines, substitute the actual repository path for this initial `cd`.
Relative paths resolve from the current working directory.

## Quick start A — Inference with the supplied checkpoint

**Inputs:** one grayscale `uint8` TIFF stack, and `ckpt_model/stage1/best.pt`. The input may be a standard multipage TIFF or a compatible 3D TIFF with explicit ZYX axes. A 2D TIFF is also accepted as a one-slice stack.

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage1/step5_inference.py \
  --input /path/to/input_stack.tif \
  --checkpoint ckpt_model/stage1/best.pt \
  --output-dir experiments/pipeline_stage1 \
  --device mps
```

Replace `--device mps` with `--device cuda`, `--device cpu`, or `--device auto` where appropriate. Optional `--cuda-amp` is **CUDA only**. Choose a new `--output-dir` for each run; the program prevents accidental replacement unless `--overwrite` is supplied.

**Output:**

```text
experiments/pipeline_stage1/
├── output_stack.tif        # uint8, same ZYX dimensions/order as the input
└── visualization/         # five evenly spaced input/output/correction previews
    └── slice_XXXX.png
```

Small native slices are processed directly; large slices are processed with overlapping 2D tiles. No physical or synthetic intermediate Z slices are added. Feed `output_stack.tif` directly to `code/stage2/step2_inference_adaptive.py` for biological refinement.

## Quick start B — Training after rebuilding the synthetic dataset

Run the charging-library preparation and Steps 2–3 below first (using the separately distributed source data). The generated `experiments/rebuilt_synthesis/metadata.csv` links each noisy input to its constructed target. The synthesis contains **1,024 training pairs, 192 validation pairs, and 192 held-out test pairs**. The test pairs are not used for training or model selection.

One full, real-data training epoch on Apple Silicon:

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage1/step4_train.py \
  --data-root experiments/rebuilt_synthesis \
  --output-root experiments/stage1_training_1epoch \
  --device mps \
  --workers 0 \
  --epochs 1
```

For the normal training budget, omit `--epochs 1` (default: **200 epochs**) or set `--epochs 200` explicitly. The run may terminate earlier under its original early-stopping rules. Use a dedicated `--output-root`; training does not overwrite `ckpt_model/stage1/best.pt`.

Important switches:

| Option | Behavior |
| --- | --- |
| `--data-root PATH` | Root containing `metadata.csv` and the paired TIFF files. |
| `--metadata-path PATH` | Use a specific metadata CSV; referenced TIFF paths still resolve under `--data-root`. |
| `--output-root PATH` | Checkpoints, transition masks, plots, logs, and previews. |
| `--epochs N` | Maximum number of epochs; default 200. |
| `--workers N` | DataLoader workers; `0` is a conservative macOS troubleshooting setting. |
| `--learning-rate FLOAT`, `--weight-decay FLOAT` | Override AdamW hyperparameters (defaults: `1e-4` each). |
| `--resume` | Continue from that output directory's `checkpoints/last.pt`, restoring optimizer/scheduler state. |
| `--overwrite` | Explicitly replace an existing output directory for a fresh run; never combine with `--resume`. |

Training artifacts include `checkpoints/best.pt`, `checkpoints/last.pt`, `train_log.csv`, `training_summary.json`, `training_report.txt`, `config.json`, `loss_curve.png`, `lr_curve.png`, transition-mask caches, and validation previews. The best checkpoint is selected from validation performance; it is not necessarily the last epoch.

## Script-by-script reference

| Script | Purpose | Primary input | Principal output |
| --- | --- | --- | --- |
| `step1_build_charging_library/step1a_analyze_cryo_statistics.py` | Characterize charging-core, dark-tail, and halo intensities in annotated real cryo-FIB-SEM slices. | Real 2D TIFFs + manually annotated core-mask TIFFs | Component statistics CSV, summary JSON, diagnostic previews |
| `step1_build_charging_library/step1b_get_bbox.py` | Label charging cores, determine grouping bounding boxes, and merge overlapping regions. | Same real TIFFs and masks | Per-slice/group crops, labeled-core TIFFs, group metadata |
| `step1_build_charging_library/step1c_residual_extraction.py` | Extract exact real core pixels separately from estimated, core-excluded dark-tail residuals. | Group directories from Step 1b | Core, mask, float32 tail-residual TIFFs and metadata |
| `step1_build_charging_library/step1d_build_charging_library.py` | Validate/package extracted core/tail groups into a library usable by synthesis. | Extracted groups from Step 1c | `library_index.csv`, indexed template entries, library summary |
| `step2_prepare_clean_data.py` | Apply deterministic cryo-style intensity mapping at four background levels to clean JRC Liver images. | Clean master blocks + real cryo reference images/core masks | Four-level styled clean TIFFs, metadata, summary |
| `step3_synthesize_artifacts.py` | Generate paired noisy/clean supervision with real charging templates and procedural artifacts. | Styled clean images + template library + real charging statistics | `experiments/rebuilt_synthesis/metadata.csv` (example output path), paired TIFFs, previews, summary |
| `step4_train.py` | Train and validate the whole-slice 2D restoration model. | Paired synthetic TIFFs via metadata CSV | Model checkpoints, metrics, transition cache, plots |
| `step5_inference.py` | Apply a trained 2D model independently to each Z slice in a TIFF volume. | uint8 input stack + pretrained checkpoint | One uint8 `output_stack.tif` and up to five preview PNGs |

### Steps 1a–1d. Charging-library preparation (required for reconstructing synthetic data)

These utilities require **real, preprocessed cryo-FIB-SEM images** and manual charging-core annotations. They do **not** perform VSNR, alignment, or core annotation themselves. Both source images and mask images must be supplied as **individual 2D grayscale TIFF files**, with matching slice indices and dimensions. Masks use `0 = charging core`, `255 = background` (threshold 128 by default).

The first step measures charging intensity statistics used later in synthesis. The second groups nearby charging cores by horizontal bounding boxes; its boxes are for grouping, not an inferred tail extent. The third obtains separate core appearances and dark-tail residuals. The fourth checks and packages usable examples into indexed entries.

Set these two directories to your prepared real data:

- `/path/to/cryo_reference_images/`: `slice_0000.tif`, `slice_0001.tif`, etc.
- `/path/to/cryo_core_masks/`: masks with corresponding slice indices.

Example end-to-end sequence:

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage1/step1_build_charging_library/step1a_analyze_cryo_statistics.py \
  --image-dir /path/to/cryo_reference_images \
  --mask-dir /path/to/cryo_core_masks \
  --output-dir experiments/charging_preparation/statistics \
  --slice-start 0 --slice-end 59

python code/stage1/step1_build_charging_library/step1b_get_bbox.py \
  --image-dir /path/to/cryo_reference_images \
  --mask-dir /path/to/cryo_core_masks \
  --output-dir experiments/charging_preparation/groups \
  --slice-start 0 --slice-end 59

python code/stage1/step1_build_charging_library/step1c_residual_extraction.py \
  --input-dir experiments/charging_preparation/groups \
  --output-dir experiments/charging_preparation/core_tail

python code/stage1/step1_build_charging_library/step1d_build_charging_library.py \
  --input-dir experiments/charging_preparation/core_tail \
  --output-dir experiments/charging_preparation/library
```

The directory `experiments/charging_preparation/library/` will contain `library_index.csv` and `entries/entry_XXXXXX/` with `core_uint8.tif`, `core_mask_uint8.tif`, `tail_residual_float32.tif`, and other template metadata.

**Overwrite warning:** the charging-preparation utilities retain the original overwrite-enabled defaults, unlike the protected training/inference output directories. In particular, library construction can replace its entire output library. Use new output directories, inspect `--help`, and supply `--no-overwrite` when preservation is required.

### Step 2. Prepare cryo-style clean data

This script starts from the immutable JRC Liver master blocks, grouped as 16 train, 3 validation, and 3 test blocks, each with 16 grayscale `1024 × 1024` slices. It uses real non-core cryo image statistics to map images to four target intensity medians: **115, 145, 175, 200**. Each 16-slice block contributes four slices to each level. Biological split membership remains unchanged.

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage1/step2_prepare_clean_data.py \
  --input-root data/JRC/clean_blocks_1024_z16 \
  --cryo-dir /path/to/cryo_reference_images \
  --cryo-mask-dir /path/to/cryo_core_masks \
  --output-root experiments/rebuilt_style_transfer_4levels \
  --check-only
```

Remove `--check-only` to generate the styled images, `metadata.csv`, `summary.json`, and previews. The script pairs reference TIFFs with masks by **sorted filename position**, so inspect the sorted ordering before a first run. Regeneration requires the original reference images and masks; the styled TIFFs alone do not replace that reference input.

### Step 3. Synthesize paired artifact data

Uses the Step-2 styled images, Step-1d charging library, and Step-1a charging-statistics summary. It creates clean targets and degraded inputs at four background levels across unchanged train/val/test partitions. Degradations include real-template and procedural charging, thin bundles, shading, halos, and white stripes. The established synthesis procedure assigns **seven regular charging events, two thin bundles, two joint-shading events, and three white stripes** to a degraded image; approximately **15% identity supervision** is retained per split/background group.

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage1/step3_synthesize_artifacts.py \
  --style-root experiments/rebuilt_style_transfer_4levels \
  --artifact-library-root experiments/charging_preparation/library \
  --charging-statistics-summary experiments/charging_preparation/statistics/summary.json \
  --output-root experiments/rebuilt_synthesis \
  --check-only
```

Remove `--check-only` to generate the paired images. The script expects the styled dataset's `metadata.csv` (by default under `--style-root`); it must be present to perform this regeneration. The train/validation/test charging-template slice ranges are distinct: **0–49 / 50–54 / 55–59** by default.

The synthetic TIFF pairs are **not part of this repository or the planned source-data archive**; generate them with this step, or point `step4_train.py` at an existing locally generated synthesis directory.

### Step 4. Train the 2D restoration network

The network uses a single-channel residual-restoration design with `PixelUnshuffle(2) -> NAFNet (width 16) -> PixelShuffle(2)`, trained on complete `1024 × 1024` slices. The objective includes weighted pixel L1, transition-local signed correction-gradient L1 (weight 2.0), and shallow regional correction-field L1. Deterministic transition masks are built and cached automatically. Optimization uses AdamW, validation-driven checkpoint selection, and the script's original LR scheduler and early-stopping behavior.

The **synthetic `test` split is held out**, not included in training, validation, checkpoint selection, or transition-cache generation. This script is separate from Stage 2's training code and never consumes Stage 2's preprocessing cache.

### Step 5. Run artifact-removal inference

This is the production Stage-1 inference entry point. It accepts a grayscale `uint8` TIFF stack and preserves **Z order, height, width, and uint8 output dtype**. For slices at most 2048 pixels in each spatial dimension, inference is performed at native slice size. Larger slices use overlapping **2048 × 2048** tiles (512-pixel overlap), blended before output conversion.

A user-supplied volume may already be aligned or preprocessed externally, but this script neither requires nor performs inter-slice alignment. It does not create extra slices or modify voxel spacing metadata for downstream software.

## Data contracts and reproducibility

- **Shared clean source:** the same JRC Liver `data/JRC/clean_blocks_1024_z16` split is used by both stages, but only Stage 1 consumes the **locally regenerated synthetic paired dataset** (for example `experiments/rebuilt_synthesis/`).
- **Stage-1 input type:** individual `1024 × 1024`, grayscale `uint8` TIFFs during training; grayscale uint8 TIFF stack during inference.
- **Stage-1 training/test separation:** 16/3/3 source-block split; synthetic pairs are organized as 1024/192/192 train/val/test. Do not use the held-out test pairs for model selection.
- **Model delivery:** the pretrained checkpoint path is `ckpt_model/stage1/best.pt`. Newly trained weights are written only under the chosen experiment output path.
- **Reproducibility:** data generation has deterministic seeds, but identical numerical results are not guaranteed across CPU, CUDA, and MPS backends or different software environments.
- **Input preparation:** VSNR or other microscopy-specific processing is external to these Stage-1 scripts. Charging-template extraction assumes real images and annotations have already been prepared.

## Checking a run

After inference, inspect `output_stack.tif` and the preview images. A minimal shape/dtype check is:

```bash
python - <<'PY'
import tifffile
p = 'experiments/pipeline_stage1/output_stack.tif'
with tifffile.TiffFile(p) as tif:
    data = tif.asarray()
print('shape:', data.shape, 'dtype:', data.dtype)
PY
```

Successful execution verifies runtime compatibility, not biological accuracy. Compare outputs against reference images and downstream segmentation separately; do not infer restored biological structure solely from a visually sharper image.

## Common issues

- **Missing charging templates or real-image masks:** download the authorized charging-source images and corresponding annotations from the separately distributed source-data archive before regenerating Stage-1 synthesis. Checkpoint-only inference does not require these files.
- **No styled-data metadata:** running Step 3 against an image-only `style_transfer_4levels` folder is insufficient; use the original Step-2 output including `metadata.csv`, or regenerate it.
- **Wrong working directory:** start from the repository root before using the relative commands above.
- **Existing output directory:** training/inference protect the output; use a new path or deliberately select `--overwrite`. Charging-library scripts have different overwrite defaults.
- **MPS unavailable:** check the active Python environment and use `--device cpu` or a supported CUDA machine; an explicitly requested unavailable backend is an error.

See the [project README](../../README.md) for the full workflow, [Stage 2](../stage2/README.md) for biological refinement, and [Segmentation](../segmentation/README.md) for CebraNET membrane masks.

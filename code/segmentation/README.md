# CebraNET membrane segmentation

**Cryo-FIB-SEM | Downstream 3-D membrane segmentation | Inference only**

Author: Ningyan Zhang  
Last modified: November 1, 2026

This directory provides the downstream membrane segmentation step after **Stage-1 artifact removal** and **Stage-2 2.5D biological refinement**. It uses a separately supplied, **pretrained CebraNET TorchScript model**; there is **no CebraNET training script** in this module.

## Processing pipeline

```text
Stage-1 inference -> output_stack.tif
                         |
                         v
Stage-2 inference -> output_stack.tif (uint8, ZYX, native geometry)
                         |
                         v
Motion-aware Z resampling (model-space Z scale: 1.5, XY scale: 1.0)
                         |
                         v
CebraNET 3-D patch inference (CUDA, 64^3, stride 16)
                         |
                         v
Weighted overlap of probabilities (FP32)
                         |
                         v
Probability recovery on the ORIGINAL acquisition Z grid
                         |
                         v
membrane_prob_native_float32.tif + mask_th_0.30_uint8.tif
                    (both ORIGINAL ZYX shape)
```

**Important: 1.5× Z is an *internal inference-grid* resampling factor, not the output depth.** The final probability and binary membrane mask are restored to the input's original slice locations and therefore have exactly the same ZYX shape as the Stage-2 input TIFF. This script does **not** produce a final 1.5×-deep segmentation volume. In contrast, the optional model-input TIFF, if saved, is Z-resampled.

The XY pixels are not resampled. The script retains a legacy source-spacing default of `20 5 5` nm (ZYX) for recorded physical metadata. **For all three in-house datasets in the manuscript, explicitly use `--source-spacing 20 10 10`; for EMPIAR-11275 use `--source-spacing 20 5 5`.** This option describes source voxel geometry; it does not resample XY pixels.

For overall installation and the Stage-1 → Stage-2 → Segmentation sequence, see the [project README](../../README.md).

## This module's layout

```text
segmentation/
├── README.md
├── cebranet_inference.py
└── environment/
```

This module requires the **third-party pretrained CebraNET TorchScript checkpoint**, which is not one of the two Stage-1/Stage-2 checkpoints supplied by this project. Obtain it from the original archive ([Zenodo DOI: 10.5281/zenodo.10047209](https://doi.org/10.5281/zenodo.10047209)) and place it at `ckpt_model/segmentation/weights.torchscript`, or specify its location with `--weights`. The repository-level layout is documented in the [project README](../../README.md).

## Installation

From the repository root on **Linux x86_64 with an NVIDIA GPU**:

```bash
conda env create -f code/segmentation/environment/environment_cuda.yml
conda activate cryo-fib-sem-segmentation
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must return `True`. The CUDA-specific runtime has **not** been retested against this portable wrapper; it is derived from an AutoDL CUDA script. See [environment/ENVIRONMENT.md](environment/ENVIRONMENT.md) before installing. **The segmentation script has no Mac MPS inference backend**, unlike Stage-1 and Stage-2. `--selftest-only` and `--check-only` may be used without a CUDA GPU if its Python imports are installed.

## Single-volume inference (recommended)

The input path below is produced by the Stage-2 command in the [project README](../../README.md). The examples assume an **in-house, XY-binned dataset** (10 × 10 × 20 nm). If you ran Stage 2 into another output directory, replace it with that directory's `output_stack.tif`.

All examples below run **from the repository root**:

```bash
python code/segmentation/cebranet_inference.py \
  --input experiments/pipeline_stage2/output_stack.tif \
  --weights ckpt_model/segmentation/weights.torchscript \
  --output-dir experiments/pipeline_segmentation \
  --z-scale 1.5 \
  --source-spacing 20 10 10 \
  --workers 4
```

Replace the Stage-2 input path with the actual `output_stack.tif`. The directory `experiments/pipeline_segmentation` must not already exist unless you **explicitly** pass `--overwrite`.

### Safe preflight (no inference)

```bash
python code/segmentation/cebranet_inference.py \
  --input experiments/pipeline_stage2/output_stack.tif \
  --weights ckpt_model/segmentation/weights.torchscript \
  --output-dir experiments/pipeline_segmentation \
  --z-scale 1.5 \
  --source-spacing 20 10 10 \
  --check-only
```

This checks TIFF shape and dtype, source/weight/output safety, and CPU synthetic invariants. It does **not** load or execute the pretrained model on CUDA, and it does not create or replace **project** output files. The CPU self-tests write and remove temporary files in the system's temporary directory.

### Multiple datasets

Given `experiments/stage2_outputs/<dataset>/output_stack.tif`:

```bash
python code/segmentation/cebranet_inference.py \
  --input-root experiments/stage2_outputs \
  --output-root experiments/pipeline_segmentations \
  --weights ckpt_model/segmentation/weights.torchscript \
  --datasets tile_00 tile_01 \
  --z-scale 1.5 \
  --source-spacing 20 10 10
```

Omit `--datasets` to automatically discover all immediate subfolders containing an `output_stack.tif`. **If datasets have different physical voxel sizes (such as in-house volumes and EMPIAR-11275), process them in separate invocations with the correct `--source-spacing`.** Each dataset writes to `--output-root/<dataset>/`; the parent receives `run_manifest.json`.

### Existing AutoDL layout

If retaining the historical AutoDL data layout, the same portable script can be called with:

```bash
python code/segmentation/cebranet_inference.py \
  --input-root /root/autodl-tmp/AutoDL_UPLOAD \
  --output-root /root/autodl-tmp/cebranet_results \
  --weights /root/autodl-tmp/cebranet/weights/weights.torchscript \
  --datasets tile_00 \
  --z-scale 1.5 \
  --source-spacing 20 10 10
```

These paths are examples, not hardcoded requirements.

## Outputs

Each output directory contains:

| File | Type / meaning |
| --- | --- |
| `membrane_prob_native_float32.tif` | `float32`, original ZYX shape, probability in [0, 1] on original slice coordinates |
| `mask_th_0.30_uint8.tif` | `uint8`, original ZYX shape, membrane=255 and background=0, threshold 0.30 |
| `summary.json` | Input/model shapes, source and realized voxel spacing, matching diagnostics, CUDA backend and timing, mask statistics |
| `run_manifest.json` | Per-run status and dataset summary; in single-stack mode written to the same output directory |
| `cebranet_input_xy1p0_z1p5_uint8.tif` | **Optional** Z-resampled uint8 model input; written when `--save-model-input` is passed (and automatically for dataset named `tile_00`, retaining the original behavior) |

The code can use full-volume FP32 CUDA accumulators if they fit its VRAM safety limit; otherwise it uses *exact absolute-Z slabs* and temporary local scratch files. Both paths implement the same weighted-overlap definition; slabs may re-evaluate patches crossing boundaries. The temporary scratch folder under the requested dataset output is removed after successful processing.

## Frozen scientific settings

| Setting | Value |
| --- | --- |
| Source spacing metadata | CLI default `(Z,Y,X)=(20,5,5)` nm; **in-house manuscript volumes require `--source-spacing 20 10 10`**; EMPIAR-11275 uses `20 5 5` |
| Default model-space XY scale | `1.0` |
| Default model-space Z scale | `1.5` (endpoint-aligned uniform grid) |
| Motion-aware local NCC patch and stride | `160 × 160`, stride `80` |
| Maximum local translation | `15` pixels |
| NCC acceptance cutoff | `0.15` |
| Minimum NCC improvement | `0.01` |
| Forward/backward agreement bound | `4` pixels |
| Warp/blend | `(1-t)*warp(A,t) + t*warp(B,1-t)` |
| CebraNET input normalization | uint8 / 255 → float32 |
| Patch and stride | `64³`, `16` |
| Batch size | `8` |
| FP16 autocast | model forward only (CUDA) |
| Accumulation | FP32, weighted 3-D overlap; floor weight `0.20` |
| Mask threshold | `0.30`, encoded as uint8 `{0,255}` |
| Model-space probability | recovered at original Z before saving |

For `Z=1.5`, not every original interior slice is necessarily an exact model-grid plane. This matters when comparing model-space resampling and the final native-grid segmentation. Z interpolation can introduce biologically implausible structures; its quality must be assessed separately from successful execution.

## Command-line arguments

| Argument | Purpose |
| --- | --- |
| `--input PATH` | Single uint8 ZYX TIFF; requires `--output-dir` |
| `--input-root PATH` | Multi-dataset parent; requires `--output-root` |
| `--datasets NAME ...` | Optional multi-dataset subset; default auto-discovery |
| `--output-dir PATH` | Direct output folder for single stack |
| `--output-root PATH` | Parent output folder for datasets |
| `--weights PATH` | TorchScript checkpoint; default `ckpt_model/segmentation/weights.torchscript` |
| `--z-scale {1.0,1.5,2.0,3.0}` | Z sampling factor for model input; default 1.5 |
| `--workers N` | CPU interpolation workers; default up to 4 |
| `--source-spacing Z_NM Y_NM X_NM` | Source physical spacing recorded in summary; code default `20 5 5`, override to `20 10 10` for in-house manuscript data |
| `--save-model-input` | Save the resampled uint8 model-space stack (may be large) |
| `--check-only` | CPU self-tests + path/TIFF checks, without project output creation |
| `--selftest-only` | CPU geometry/preprocessing/mask tests, no weight/data required |
| `--overwrite` | Explicitly allow replacement of requested output dataset directory |

Both modes require a model-space size of at least `64` voxels in all dimensions. With Z scale 1.5 this typically requires at least **43 native slices**; fewer slices cause a clear preflight error. Motion-aware Z interpolation additionally needs native XY dimensions of at least `160 × 160` pixels.

## Validation and reproducibility

- The provided original code was a CUDA-first inference pipeline and used absolute AutoDL paths. The portable version relocates paths and adds a CLI/output protection layer; the scientific resampling, TorchScript forward, weighted accumulation and native-grid recovery functions are unchanged.
- CPU self-tests check motion-aware preprocessing, Z geometry, global/slab overlap equivalence on synthetic predictions and mask encoding.
- A preflight with `--check-only` does **not** demonstrate that TorchScript CUDA inference works on a target NVIDIA GPU. Run a representative real-volume inference to validate the installed environment and GPU.
- `--overwrite` can replace an existing selected segmentation output folder. Keep completed experimental results in separate output paths.
- Inference outputs are binary membrane *predictions*, not ground-truth biological annotations.

See the [project README](../../README.md), [Stage-1 README](../stage1/README.md), and [Stage-2 README](../stage2/README.md) for upstream preprocessing, artifact removal, and biological refinement.

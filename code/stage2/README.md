# Stage 2 — Cryo-FIB-SEM 2.5D Biological Refinement

**Maintainer:** Ningyan Zhang  
**Documentation updated:** November 1, 2026  
**Location in the repository:** `code/stage2/`

For the complete pipeline, begin with the [project README](../../README.md). All relative CLI paths below assume the repository root as the working directory.

## Overview

Stage 2 performs **2.5D biological refinement** on Stage-1-corrected cryo-FIB-SEM volumes. Its five-slice model uses neighboring Z slices as context to improve membrane contrast and suppress residual artifacts while predicting **only the center slice**. The inference implementation includes local, translation-only neighbor alignment and optionally applies clean-reference domain transfer.

The two executable scripts have distinct responsibilities:

1. `step1_train.py`: construct Stage-2-specific preprocessing caches, dynamically synthesize training corruptions, train and validate the 2.5D network, and save checkpoints.
2. `step2_inference_adaptive.py`: load a pretrained checkpoint and process a grayscale uint8 Stage-1 output stack with **Domain Transfer ON or OFF**.

Stage 2 uses the same immutable JRC Liver train/validation/test **block partition** as Stage 1 but builds **its own** targets, corruptions, and cache from the original clean blocks. It does **not** train on Stage-1's synthetic paired dataset or require Stage-1 charging templates.

Inference preserves the input **(Z, Y, X) dimensions and original number of Z slices**. It does not interpolate through-slice spacing or create synthetic Z slices. Local alignment shifts neighboring model context using integer translations without spatial interpolation; it does not write a separately aligned volume.

## This module's layout

```text
stage2/
├── README.md
├── environment/
├── step1_train.py
└── step2_inference_adaptive.py
```

The directory `code/stage2/experiments/` (when present locally) contains generated caches, not source files needed for the release. Clean JRC blocks are distributed separately and can be placed at `data/JRC/clean_blocks_1024_z16/`, outside Git tracking. Stage-2 training uses the same biological train/validation/test partition as Stage-1 but does **not** consume Stage-1 synthetic pairs or charging templates. All commands are run from the repository root.

## Requirements

Stage 2 has its **own** installation environment (see
[`environment/ENVIRONMENT.md`](environment/ENVIRONMENT.md)). The environment files are derived from
real macOS package snapshots and preserve the versions used for the recorded
MPS tests.

| Host platform | One-command environment creation |
| --- | --- |
| Apple Silicon macOS | `conda env create -f code/stage2/environment/environment_macos.yml` |
| Linux x86_64 with NVIDIA GPU | `conda env create -f code/stage2/environment/environment_cuda.yml` |

After installation, use `conda activate cryo-fib-sem-stage2`.
The source environment used Python **3.11.16**. PyTorch builds differ
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

## Quick start A — Inference with the pretrained checkpoint

**Input:** a Stage-1 `output_stack.tif` (single-channel, uint8, ZYX, and at least `256 × 256` spatial pixels). **Weights:** `ckpt_model/stage2/best.pt`.

### Manuscript mode: with domain transfer (default)

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage2/step2_inference_adaptive.py \
  --input experiments/pipeline_stage1/output_stack.tif \
  --checkpoint ckpt_model/stage2/best.pt \
  --reference-train data/JRC/clean_blocks_1024_z16/train \
  --cache-dir experiments/stage2_domain_cache \
  --output-dir experiments/stage2_inference_with_transfer \
  --device mps \
  --domain-transfer
```

**Domain transfer is enabled by default and was enabled for the real-data examples in the manuscript.** Specify `--no-domain-transfer` explicitly to bypass it for optional comparison or troubleshooting. When enabled, transfer samples 128 clean 256-pixel patches from the 256 training-reference slices, then performs reference-based FFT log-amplitude and spatial intensity CDF matching. Input-stack statistics are computed across **all provided slices**; therefore changing the input stack can change its global mapping. Reference statistics are cached outside the inference output directory and are reused when their content signature matches.


### Optional mode: without domain transfer (no clean reference needed)

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage2/step2_inference_adaptive.py \
  --input experiments/pipeline_stage1/output_stack.tif \
  --checkpoint ckpt_model/stage2/best.pt \
  --output-dir experiments/pipeline_stage2 \
  --device mps \
  --no-domain-transfer
```

Run the Stage-1 inference command in the [project README](../../README.md) to create `experiments/pipeline_stage1/output_stack.tif`, or replace `--input` with **your own Stage-1 output stack**. The `--no-domain-transfer` mode does not need `data/JRC/clean_blocks_1024_z16/train` or access to a domain cache. Full-slice preprocessing, local translation alignment, and model inference **remain active**.

A fast, no-write configuration check for either mode:

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage2/step2_inference_adaptive.py \
  --input experiments/pipeline_stage1/output_stack.tif \
  --checkpoint ckpt_model/stage2/best.pt \
  --output-dir experiments/stage2_preflight_only \
  --device mps \
  --no-domain-transfer \
  --check-only
```

`--check-only` verifies paths, TIFF format, checkpoint compatibility, geometry, and device availability, but does not run the actual forward inference or create outputs.

**Output for both inference modes:**

```text
experiments/stage2_inference_.../
├── output_stack.tif       # uint8; original ZYX shape preserved
└── visualization/
    └── slice_XXXX.png     # up to 5 evenly spaced four-panel comparisons
```

The four preview panels show the Stage-1 output provided to Stage-2, its intensity-preprocessed model input, the Stage-2 output, and the signed change in the processed domain. There are **no extra TIFF stacks or per-slice TIFFs**. The optional domain-reference cache is stored separately under `--cache-dir`.

## Quick start B — Test Stage-2 training in 10 steps

This is the recommended end-to-end runtime check before a long training job. It exercises complete JRC Liver data discovery, deterministic cache construction/reuse, fixed validation-sample generation, stochastic artifact synthesis, GPU/MPS model updates, checkpointing, logs, and preview generation.

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage2/step1_train.py \
  --data-root data/JRC/clean_blocks_1024_z16 \
  --cache-root experiments/test_stage2_cache \
  --output-root experiments/stage2_training_10steps \
  --device mps \
  --cache-workers 2 \
  --steps 10
```

**`--steps` accepts any positive integer; the default is 2,000.** Changing the number of optimizer steps does not change the batch size, loss, artifact probabilities, architecture, or optimizer settings. A short run is started **from scratch**; it is not an exact continuation of a previously trained model.

First-time preprocessing can take longer than the training steps. The preprocessing cache can be reused across runs with unchanged clean TIFF contents and preprocessing settings; it is generated locally and is not a source-data dependency. A 10-step run performs validation and saves previews on the final step even though the normal validation and preview intervals are longer.

To train with the default **2,000 optimizer steps**, omit `--steps 10` and specify a new output directory:

```bash
cd /path/to/Cryo-FIB-SEM

python code/stage2/step1_train.py \
  --data-root data/JRC/clean_blocks_1024_z16 \
  --cache-root experiments/test_stage2_cache \
  --output-root experiments/stage2_training_full \
  --device mps \
  --cache-workers 2
```

The training script does **not** provide exact optimizer/RNG-state resume; do not treat a new run as an uninterrupted continuation. It protects an existing output directory unless `--overwrite` is explicitly supplied. Never use `--overwrite` on a run you want to retain.

## Script reference

| Script | What it does | Inputs | Outputs |
| --- | --- | --- | --- |
| `step1_train.py` | Trains the center-slice 2.5D U-Net. Validates the fixed clean-data split, builds or reuses deterministic caches, generates online corruptions, optimizes the model, scores fixed validation conditions, and saves checkpoints/diagnostics. | `data/JRC/clean_blocks_1024_z16/{train,val,test}/block_*/`, output/cache paths, device, number of steps | `best.pt`, `final.pt`, loss arrays/plot, summary, fixed validation previews, separate deterministic cache |
| `step2_inference_adaptive.py` | Runs five-slice contextual refinement over a Stage-1 grayscale uint8 stack. Supports optional global clean-reference FFT/spatial domain transfer and local, integer translation-only neighbor alignment. | Stage-1 `output_stack.tif`, trained model checkpoint; train split only when transfer is ON | Final uint8 `output_stack.tif`, up to five PNG comparisons, optional separate reference-statistics cache |

### Step 1. Training details

**Input and target construction.** Every complete clean uint8 slice is standardized by its own mean/std, high-pass filtered with a Gaussian low-pass of sigma 10, centered by a 50-bin histogram mode, and clipped to `[-3, 3]`. The center-slice training target applies a deterministic membrane-selective high-pass enhancement in the original clean slice's preprocessing basis. Four deterministic arrays—raw, processed, target, and membrane-placement score—are kept as mmap-backed NPYs. Random corruptions are **not cached**.

**Patch/context sampling.** Each optimizer step samples a batch of **32** patches of size **256 × 256**, with five adjacent Z slices per input. Exactly one major corruption type is selected per patch:

| Major condition | Probability |
| --- | ---: |
| Clean | 40% |
| Missing region | 30% |
| Local drift | 15% |
| Loss of focus | 15% |

Independent overlays additionally include a stripe family (70% total: 35% white-core and 35% black-core, mutually exclusive) and residual thin-bundle artifacts (15%). Gaussian noise with sigma 1.0 is added in processed space; missing-region fill is applied **after** that noise. Local drift shifts neighboring crop locations with the center fixed, while focus loss is synthesized on a complete raw slice before preprocessing.

**Network and optimization.** A five-channel, early-fusion U-Net with **2,204,289 trainable parameters** predicts a single center-slice map; its final full-resolution skip uses only the center input. Training uses LeCun-normal scratch initialization, dropout 0.1, Adam (`lr=2e-5`, betas `0.9/0.999`, zero weight decay), FP32, and the pair loss consisting of squared reconstruction error plus squared negative-structure error. There is no scheduler or early stopping. A single sequential CPU prefetch producer preserves the intended NumPy sampling order.

**Validation and checkpoints.** Thirty-two fixed samples are used to evaluate clean, missing-region, drift, and focus losses. The checkpoint score is:

```text
validation_score = 0.5 * clean_loss
                 + (missing_loss + drift_loss + focus_loss) / 6
```

The stripe and thin-bundle cases are diagnostic visualizations; they are **not** terms of checkpoint selection. Evaluation runs every 100 steps and at the requested final step. Seven-condition previews are saved every 250 steps and at the final step. `best.pt` is selected by the lowest weighted validation score, while `final.pt` stores the final trained model state.

**Training CLI reference:**

| Argument | Meaning |
| --- | --- |
| `--data-root PATH` | Immutable clean block root, with train/val/test subfolders. |
| `--cache-root PATH` | Deterministic train/val preprocessing cache; independent of run outputs. |
| `--output-root PATH` | Dedicated training run directory. |
| `--device {auto,cuda,mps,cpu}` | Explicit backend; no silent fallback if unavailable. |
| `--steps N` | Positive number of optimizer steps, default 2000. |
| `--cache-workers N` | Block-cache build threads, default 2. |
| `--check-only` | Validate the split, paths, device, and shape constraints without writing. |
| `--overwrite` | Explicitly delete an existing training output directory and restart. |

**Training output layout:**

```text
<output-root>/
├── checkpoints/
│   ├── best.pt
│   └── final.pt
├── validation_previews/
│   └── step_XXXX/
│       └── validation_sample_YY.png
├── train_loss.npy
├── train_missing_mask_loss.npy
├── val_steps.npy
├── loss.png
├── train_summary.json
├── experiment.txt
└── step1_train.py            # snapshot of the executed training script
```

The missing-mask loss array is **diagnostic only**, not an added optimization objective.

### Step 2. Inference details

1. **Full-slice preprocessing:** apply per-slice mean/std normalization, Gaussian low-pass subtraction (`sigma=10`), 50-bin modal centering, and clipping to `[-3, 3]`.
2. **Optional domain transfer:** estimate the input stack's global log-amplitude FFT and post-FFT intensity distributions; match against the clean TRAIN reference without changing Fourier phase. Applies across the supplied stack, not independently per slice.
3. **Local alignment:** construct a five-slice context around each output Z index, replicating end slices when necessary; center-referenced integer XY offsets are estimated within a 320 × 320 context, using a central 160 × 160 normalized-correlation region and up to 15-pixel shift. The center stays fixed. **No interpolation/resampling** is used for these local shifts.
4. **2.5D refinement:** run 256 × 256 model tiles with 128-pixel overlap and weighted blending, predicting one output slice at each Z index.
5. **Output encoding:** save the processed-space prediction as `uint8 = clip(round(128 + 32 * prediction), 0, 255)`. The algorithm deliberately does not reconstruct the removed per-slice low-frequency background by adding back original means and low-pass images.

**Inference CLI reference:**

| Argument | Meaning |
| --- | --- |
| `--input PATH` | Stage-1 output grayscale uint8 TIFF stack. |
| `--checkpoint PATH` | `ckpt_model/stage2/best.pt` or a compatible newly trained checkpoint. |
| `--output-dir PATH` | Independent inference output folder. |
| `--domain-transfer` | Enable global FFT + spatial CDF transfer; **default**. |
| `--no-domain-transfer` | Skip all reference/cache logic; preprocessing and local alignment stay active. |
| `--reference-train PATH` | Clean training blocks required only when transfer is enabled. |
| `--cache-dir PATH` | Optional reusable domain reference statistics cache. |
| `--device {auto,cuda,mps,cpu}` | Select runtime backend. |
| `--cuda-amp` | Optional CUDA-only FP16 inference. |
| `--check-only` | Validate configuration without writing or model execution. |
| `--overwrite` | Explicitly replace a previous inference output directory. |

## End-to-end Stage-1 → Stage-2 example

Given a prepared uint8 input volume and the pretrained checkpoints:

```bash
cd /path/to/Cryo-FIB-SEM

# Stage 1: artifact removal
python code/stage1/step5_inference.py \
  --input /path/to/input_stack.tif \
  --checkpoint ckpt_model/stage1/best.pt \
  --output-dir experiments/pipeline_stage1 \
  --device mps

# Stage 2: biological refinement without domain transfer
python code/stage2/step2_inference_adaptive.py \
  --input experiments/pipeline_stage1/output_stack.tif \
  --checkpoint ckpt_model/stage2/best.pt \
  --output-dir experiments/pipeline_stage2 \
  --device mps \
  --no-domain-transfer
```

To compare the domain-transfer variant, run Stage-2 inference again using a **different** `--output-dir` plus `--domain-transfer --reference-train data/JRC/clean_blocks_1024_z16/train`.

## Verifying results

An output stack should have the same shape as its input, with `uint8` dtype:

```bash
cd /path/to/Cryo-FIB-SEM
python - <<'PY'
import tifffile
inp = tifffile.imread('experiments/pipeline_stage1/output_stack.tif')
out = tifffile.imread('experiments/pipeline_stage2/output_stack.tif')
assert inp.shape == out.shape, (inp.shape, out.shape)
assert str(out.dtype) == 'uint8', out.dtype
print('PASS:', inp.shape, out.dtype)
PY
```

For a 10-step training run, check that `len(train_loss.npy) == 10`, the final entry of `val_steps.npy` is `10`, that both checkpoint files load with `torch.load(..., weights_only=True)`, and that four PNG previews exist in `validation_previews/step_0010/`.

**Local verification history (Mac MPS):** the supplied code has completed an actual 10-step train/validation/checkpoint smoke test; inference with the supplied Stage-2 checkpoint has run successfully in both domain-transfer modes, and a checkpoint newly produced by the short training passed inference preflight. This is **runtime compatibility testing**, not evidence that 10 training steps yield an accurate scientific model. The full 2,000-step schedule and CUDA execution were not covered by those local tests.

## Reproducibility and important limitations

- Stage-2 training always starts from a fresh initialization; its checkpoints do not include the complete optimizer/RNG/prefetch state needed for exact mid-run resumption.
- The Stage-2 domain-reference set is **only the clean training split**, not its validation/test blocks and not the input volume itself. The input volume is analyzed to calculate an unsupervised domain mapping, not used as a supervised target.
- Results may differ across devices or when CUDA AMP is enabled. Leave `--cuda-amp` off when prioritizing FP32 reproducibility.
- Domain transfer changes numerical input statistics; its benefit must be evaluated on actual data. Success of the ON and OFF pipelines is not a claim that either is universally better.
- Local translation alignment is not a substitute for full-stack specimen motion correction. Strong nonrigid drift and the physical spacing between acquired Z slices remain outside its scope.
- The model may enhance apparent edges without recovering all missing biological detail. Validate membrane continuity and downstream segmentation independently.
- Keep input TIFFs, checkpoints, training caches, and output roots in separate directories. Training/inference outputs are protected against accidental overwriting unless explicitly authorized.

See the [project README](../../README.md) for full-pipeline execution, [Stage 1](../stage1/README.md) for artifact removal, and [Segmentation](../segmentation/README.md) for downstream CebraNET inference.

> Installation files live in `code/stage2/environment/`. Run all relative commands from the repository root; see [the module README](../README.md) and [the project README](../../../README.md).

# Stage 2 — Environment installation

Stage 2 uses a **separate** Conda environment, not the Stage-1 environment.
The tested **FIB-SEM-214** macOS environment provides Python **3.11.16**,
PyTorch **2.14.0**, and NumPy **2.4.6**.

## Install from repository root

**Apple Silicon macOS (MPS):**

```bash
conda env create -f code/stage2/environment/environment_macos.yml
conda activate cryo-fib-sem-stage2
```

**Linux x86_64 with NVIDIA CUDA:**

```bash
nvidia-smi
conda env create -f code/stage2/environment/environment_cuda.yml
conda activate cryo-fib-sem-stage2
```

`environment_cuda.yml` uses the official **PyTorch 2.14.0 + CUDA 12.6**
wheel, for which a Python 3.11 Linux x86_64 distribution is published.
Verify NVIDIA driver compatibility before installation and check the host
glibc requirement (**manylinux_2_28 / glibc 2.28 or newer**).
This CUDA configuration has **not** yet been run through the Stage-2
training/inference pipeline; the reported local tests used Apple MPS.

**Do not install both YAML files into the same environment.**
`code/stage2/environment/requirements.txt` lists direct non-PyTorch packages only. The appropriate
PyTorch build is included in each YAML file, so no additional pip install
is needed after successful environment creation.

## Check installation

```bash
python -m pip check
python -c "import torch, cv2, numpy, scipy, tifffile, imagecodecs, matplotlib, tqdm; print('Python dependencies: PASS'); print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available())"
```

**Read-only dataset preflight** (from the repository root):

```bash
python code/stage2/step1_train.py \
  --data-root data/clean_blocks_1024_z16 \
  --cache-root experiments/stage2_cache \
  --output-root experiments/stage2_training \
  --device auto \
  --check-only
```

**Short training smoke test** (current `step1_train.py` includes `--steps`):

```bash
python code/stage2/step1_train.py \
  --data-root data/clean_blocks_1024_z16 \
  --cache-root experiments/stage2_cache \
  --output-root experiments/smoke_stage2_10steps \
  --device auto \
  --steps 10
```

Notes:
- The `--steps` value is adjustable; the default is still 2,000.
- The full Stage-2 train/val preprocessing cache is prepared even in a
  10-step smoke test. Reuse the cache directory for subsequent runs.
- Stage-2 OFF inference requires only a valid Stage-1 output TIFF and checkpoint.
  ON inference additionally requires the Stage-2 clean training reference split.
- The YAML files install dependencies only, not JRC Liver data or weights.

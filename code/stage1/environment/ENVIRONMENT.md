> Installation files live in `code/stage1/environment/`. Run all relative commands from the repository root; see [the module README](../README.md) and [the project README](../../../README.md).

# Stage 1 — Environment installation

Stage-1 restoration uses a **dedicated Conda environment**, separate from Stage 2.
The supported configurations retain Python **3.8.20**, PyTorch **2.4.1**,
and NumPy **1.24.4**. Select the YAML for the target platform; do **not**
install the macOS and CUDA YAML files into the same environment.

## Install from repository root

**Apple Silicon macOS (MPS):**

```bash
conda env create -f code/stage1/environment/environment_macos.yml
conda activate cryo-fib-sem-stage1
```

The macOS/MPS configuration has been validated with a complete Stage-1
training run. It remains unchanged by the CUDA packaging fix.

**Linux x86_64 with NVIDIA CUDA:**

```bash
nvidia-smi
conda env create -f code/stage1/environment/environment_cuda.yml
conda activate cryo-fib-sem-stage1
```

`environment_cuda.yml` installs **PyTorch 2.4.1+cu121** (CUDA 12.1) and
**opencv-python-headless==4.8.0.76**. The OpenCV wheel provides the `cv2`
module with version 4.8.0; it is not binary-identical to the Conda OpenCV
build used on macOS. The NVIDIA driver must support the CUDA 12.1 runtime.

**CUDA validation status (October 13, 2026):** The YAML was successfully
installed in a single `conda env create` run on Linux with an NVIDIA
**RTX 4090 D**. Dependency checks, a CUDA forward/backward test, and **two
complete Stage-1 training epochs** passed. **Full CUDA training through
convergence has not been validated**, and CUDA/FP16 results need not be
bitwise identical to macOS/MPS/FP32 results.

**Conda channel note:** The successful CUDA installation used **conda-forge
only**, with **strict channel priority**, to avoid conflicts from the AutoDL
host's preconfigured Conda mirrors/channels. If environment solving fails,
check the host's `.condarc` and retry in an isolated conda-forge-only,
strict-priority configuration. Do not solve the conflict by silently
changing the pinned Python or scientific-package versions.

`requirements.txt` lists direct dependencies for reference and excludes
PyTorch because its build depends on the platform. **Use the appropriate
YAML for installation; do not run `pip install -r requirements.txt` on top
of it.** The CUDA YAML's headless OpenCV package is intentional.

## Check installation

After activating the environment:

```bash
python -m pip check
python -c "import torch, cv2, numpy, scipy, tifffile, imagecodecs, matplotlib, pandas, tqdm; print('Python dependencies: PASS'); print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available())"
```

On an appropriately configured GPU machine, the corresponding device
should be available. CPU-only execution is also supported. On CUDA,
optionally verify that a backward pass runs:

```bash
python -c "import torch; x=torch.ones(4, device='cuda', requires_grad=True); (x.square().sum()).backward(); print('CUDA forward/backward: PASS')"
```

From the repository root, a safe inference preflight is:

```bash
python code/stage1/step5_inference.py --help
```

## Notes

- Python 3.8 is retained for compatibility with the validated Stage-1 setup,
  although the Python 3.8 release line is end-of-life.
- Stage-1 inference requires a supplied checkpoint and a uint8 TIFF stack.
  Stage-1 training requires `synthesis_data/` and `metadata.csv`; for the
  uploaded Deliver layout, pass `--data-root synthesis_data` explicitly.
- Environment YAML files install Python dependencies only; they do not
  download training datasets or model checkpoints.
- Conda solver results can change over time. The CUDA installation and
  short-training validation above do not establish full-training numerical
  equivalence with the macOS/MPS implementation.

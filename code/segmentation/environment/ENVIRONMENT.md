> Installation files live in `code/segmentation/environment/`. Run all relative commands from the repository root; see [the module README](../README.md) and [the project README](../../../README.md).

# Segmentation environment — Linux NVIDIA CUDA

**This module uses its own Conda environment.** Do not reuse Stage-1's `FIB-SEM` (Python 3.8 / PyTorch 2.4) or Stage-2's `FIB-SEM-214` (Python 3.11 / PyTorch 2.14) without checking compatibility.

The original AutoDL setup used Python 3.12.3 and PyTorch 2.5.1 with CUDA 12.4. This distribution targets Linux x86_64 / NVIDIA CUDA with those major versions. Other package versions are selected for compatibility with the imported code, not claimed as a byte-for-byte copy of the source machine's full environment.

## Installation

From the repository root:

```bash
nvidia-smi
conda env create -f code/segmentation/environment/environment_cuda.yml
conda activate cryo-fib-sem-segmentation
python -m pip check
```

The environment includes Python 3.12 and the official CUDA 12.4 build of PyTorch 2.5.1, plus NumPy, SciPy, OpenCV (headless), and tifffile. CUDA compatibility requires a sufficiently recent NVIDIA driver; the CUDA toolkit/compiler `nvcc` is **not required** just to run prebuilt PyTorch inference wheels.

PyTorch 2.5.1 CUDA 12.4 wheels and installation syntax are documented in the [official PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) page.

`code/segmentation/environment/requirements.txt` lists **only** the directly imported non-PyTorch dependencies and is not a substitute for a CUDA PyTorch installation. The YAML is the recommended single-command installation route. It does **not** download the CebraNET TorchScript model weights.

## Verify

```bash
python -c "import torch, numpy, scipy, cv2, tifffile; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
python code/segmentation/cebranet_inference.py --selftest-only
```

The self-tests run on CPU and do not need model weights. Before full CUDA inference, place the checkpoint in `ckpt_model/segmentation/weights.torchscript` or supply `--weights PATH`.

Read-only project preflight (the CPU tests use temporary system files):

```bash
python code/segmentation/cebranet_inference.py \
  --input experiments/stage2_result/output_stack.tif \
  --weights ckpt_model/segmentation/weights.torchscript \
  --output-dir experiments/segmentation_result \
  --check-only
```

## Supported execution

- **CUDA:** The only inference backend in the supplied script. Mixed precision is used for the TorchScript forward; reconstruction accumulators remain FP32.
- **Mac MPS / CPU:** Not supported for real segmentation in this module. The two CPU self-test modes are available if the Python dependencies can be imported.
- **Test status:** The original user's AutoDL work used a CUDA GPU. This new cross-machine package has passed CPU self-tests, interface checks and scientific-function comparisons, but full CUDA execution with the user's actual TorchScript checkpoint has not been reproduced here. Validate against real test volumes in the destination environment.

## Troubleshooting

- `CUDA is unavailable`: check driver availability (`nvidia-smi`), PyTorch wheel variant, and `torch.cuda.is_available()`.
- Missing weight file: copy the original `weights.torchscript` into `ckpt_model/segmentation/` or use `--weights`.
- Model input too short in Z: 64 model-space slices are required. With `--z-scale 1.5`, at least about 43 native slices are needed.
- CUDA out-of-memory: the code automatically uses its exact absolute-Z-slab backend if the **accumulator** estimate is too large. Very large inputs may still exceed CPU RAM, GPU memory required for the model or available scratch disk.
- Existing output directory: choose a new output path, or explicitly use `--overwrite` after verifying the destination.

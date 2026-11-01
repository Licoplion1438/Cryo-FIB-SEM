#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cryo-FIB-SEM Preprocessing Step 2: VSNR Curtaining Removal.

Author: Ningyan Zhang
Last modified: October 10, 2026

Description:
    Remove predominantly vertical curtaining from each slice of an existing
    uint8 ZYX stacked TIFF using the third-party pyvsnr library. Select the
    VSNR execution backend explicitly: NumPy on CPU or CuPy on CUDA.

    For each slice, run VSNR over the specified integer-angle sweep centered
    at zero and select the candidate with the lowest smoothed column-profile
    variance. Match the selected candidate's median and median absolute
    deviation (MAD) to the input slice, as in the original implementation.

    CUDA defaults preserve the original 11-angle sweep and VSNR parameters.
    CPU defaults use only theta=0 to avoid the cost of repeated CPU filtering.
    Backend and parameter choices are explicit and do not alter other steps.

Inputs:
    --input: A uint8 grayscale stacked TIFF with ZYX axes (single-slice OK).
    --output: Optional destination stacked TIFF path.
    --device: Required backend selection: cpu (NumPy) or cuda (CuPy).
    --theta-sweep: Optional nonnegative integer N, angles -N through +N.
                   Defaults to 0 on CPU, 5 on CUDA.
    --noise-level: Optional positive VSNR Gabor noise level; default 80.
    --overwrite: Explicit permission to replace an existing output.
    --check-only: Validate the input TIFF without running VSNR.

Outputs:
    One uint8 ZYX stacked TIFF, preserving slice order and image dimensions.
    No intermediate per-slice files are created.

Requirements:
    Python 3.9+, numpy, scipy, tifffile, tqdm, pyvsnr.
    CuPy with compatible CUDA libraries is required only for --device cuda.
    No Apple MPS backend is used; macOS runs VSNR on CPU.

Usage (from the repository root):
    python code/preprocessing/step2_vsnr.py \
        --input /path/to/input_uint8_stacked.tif --device cpu

    python code/preprocessing/step2_vsnr.py \
        --input /path/to/input_uint8_stacked.tif \
        --device cuda --theta-sweep 5 --noise-level 80

    python code/preprocessing/step2_vsnr.py \
        --input /path/to/input_uint8_stacked.tif \
        --device cpu --theta-sweep 0 --noise-level 60

Notes:
    - This is optional preprocessing, independent of Stage-1 and Stage-2.
    - No resize, crop, alignment, histogram matching, or domain transfer.
    - CUDA with theta-sweep=5 and noise-level=80 retains the original recipe.
    - CPU theta-sweep=0 is a faster alternative, not equivalent to sweep=5.
    - Median/MAD matching is preserved from the original VSNR script.
    - VSNR itself is imported from pyvsnr; its source is not bundled here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter1d
import tifffile
from tqdm import tqdm


# ============================================================
# 1. Fixed VSNR parameters and execution defaults
# ============================================================

MODE = "vertical"
NOISE_LEVEL = 30
SIGMA_LONG = 30
SIGMA_SHORT = 1
THETA_BASE = 0
THETA_SWEEP = 5
MAX_ITERS = 30
SWEEP_SMOOTH_SIGMA = 50
ENABLE_ROBUST_MATCH = True

# Optional backend modules are imported only when real VSNR starts.
# CPU runs never import CuPy, including on macOS.
cp = None
vsnr2d = None


# ============================================================
# 2. Input validation and stacked-TIFF access
# ============================================================

def nonnegative_int(value):
    """Parse a nonnegative integer sweep radius for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("theta-sweep must be a nonnegative integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("theta-sweep must be a nonnegative integer.")
    return parsed


def positive_float(value):
    """Parse a strictly positive, finite VSNR noise level for argparse."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("noise-level must be a positive number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("noise-level must be a positive finite number.")
    return parsed


def parse_args():
    """Parse file paths, backend choice, and the two user-tunable VSNR parameters."""
    parser = argparse.ArgumentParser(
        description="Run CPU or CUDA VSNR on a uint8 ZYX stacked TIFF."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input stacked TIFF.")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), required=True,
        help="Execution backend (required): cpu=NumPy, cuda=CuPy.",
    )
    parser.add_argument(
        "--theta-sweep", type=nonnegative_int, default=None, metavar="N",
        help="Use integer angles -N..+N (default: CPU=0, CUDA=5).",
    )
    parser.add_argument(
        "--noise-level", type=positive_float, default=float(NOISE_LEVEL),
        metavar="VALUE", help="VSNR Gabor noise level (default: 80).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output TIFF (default: <input_stem>_vsnr.tif next to input).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacement of an existing output."
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Validate input only, without running VSNR."
    )
    return parser.parse_args()


@contextmanager
def open_uint8_stack(path):
    """Yield (Z, Y, X), a per-slice reader, and series axes without loading the volume.

    Uncompressed TIFFs are memory mapped when possible. Compressed,
    page-organized TIFFs are decoded one slice at a time to avoid allocating
    a second full-volume array. A 2D single-page TIFF is interpreted as Z=1.
    Color/sample axes and ambiguous multidimensional TIFFs are rejected.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {path}")

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(dim) for dim in series.shape)
        axes = series.axes
        if len(shape) == 2:
            height, width = shape
            shape_zyx = (1, height, width)
        elif len(shape) == 3:
            shape_zyx = shape
        else:
            raise ValueError(f"Expected 2D/3D grayscale TIFF, got {shape} (axes={axes!r}).")

        if series.dtype != np.dtype("uint8"):
            raise ValueError(f"Expected uint8 TIFF, got {series.dtype}.")
        if any(dim < 1 for dim in shape_zyx):
            raise ValueError(f"Invalid TIFF shape: {shape_zyx}.")
        if len(axes) != len(shape) or axes[-2:] != "YX" or any(
            axis in "SC" for axis in axes[:-2]
        ):
            raise ValueError(
                f"Input must be grayscale with trailing YX axes, got axes={axes!r}."
            )

        # First try direct file mapping, which works for our uncompressed
        # stacked-TIFF interface and avoids loading the full volume into RAM.
        mapped = None
        try:
            mapped = tifffile.memmap(path, series=0)
            if mapped.shape != shape:
                mapped = None
        except (ValueError, TypeError, OSError):
            pass

        if mapped is not None:
            def read_slice(z):
                return mapped if len(shape) == 2 else mapped[z]
        elif len(series.pages) == shape_zyx[0]:
            def read_slice(z):
                return series.pages[z].asarray()
        else:
            raise ValueError(
                "This TIFF cannot be read slice-by-slice. Convert it to a "
                "conventional uint8 ZYX stacked TIFF using Preprocessing Step 1."
            )

        try:
            yield shape_zyx, read_slice, axes
        finally:
            del mapped


# ============================================================
# 3. Original CuPy / pyvsnr backend
# ============================================================

def initialize_backend(device):
    """Load pyvsnr and only the dependencies required by the chosen backend.

    Selecting CPU explicitly forces pyvsnr's NumPy path even on machines
    where CuPy is installed; no automatic backend selection occurs.
    """
    global cp, vsnr2d
    try:
        from pyvsnr import vsnr2d as vsnr_function
    except ImportError as exc:
        raise RuntimeError(
            "VSNR requires pyvsnr in the selected Python environment."
        ) from exc

    if device == "cuda":
        try:
            import cupy as cupy_module
        except ImportError as exc:
            raise RuntimeError(
                "CUDA VSNR requires CuPy matching the installed CUDA runtime."
            ) from exc
        cp = cupy_module
    else:
        cp = None
    vsnr2d = vsnr_function


def select_gpu_with_most_free_memory():
    """Select the available CUDA device with the greatest free VRAM."""
    num_gpus = int(cp.cuda.runtime.getDeviceCount())
    if num_gpus < 1:
        raise RuntimeError("CuPy cannot see any CUDA GPU.")

    candidates = []
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        candidates.append((int(free_bytes), int(total_bytes), int(gpu_id)))

    free_bytes, total_bytes, gpu_id = max(candidates, key=lambda item: item[0])
    cp.cuda.Device(gpu_id).use()
    props = cp.cuda.runtime.getDeviceProperties(gpu_id)
    gpu_name = props["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")

    print(f"CUDA GPU: {gpu_name} (device {gpu_id})")
    print(f"Free / total VRAM: {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GiB")
    return gpu_id


def release_cupy_memory():
    """Release cached CuPy buffers on CUDA; do nothing in CPU mode."""
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def to_numpy_float32(array):
    """Convert a NumPy or CuPy pyvsnr result into host-side float32."""
    if cp is not None and isinstance(array, cp.ndarray):
        array = cp.asnumpy(array)
    return np.asarray(array, dtype=np.float32)


# ============================================================
# 4. Original per-slice intensity restoration (unchanged)
# ============================================================

def robust_match(dst, ref):
    """Match destination median/MAD to the input (original VSNR postprocess)."""
    eps = 1e-6
    dst = dst.astype(np.float32, copy=False)
    ref = ref.astype(np.float32, copy=False)

    med_dst = float(np.median(dst))
    med_ref = float(np.median(ref))
    mad_dst = float(np.median(np.abs(dst - med_dst))) + eps
    mad_ref = float(np.median(np.abs(ref - med_ref))) + eps
    return ((dst - med_dst) * (mad_ref / mad_dst) + med_ref).astype(np.float32)


# ============================================================
# 5. Original VSNR candidate and stripe scoring (unchanged)
# ============================================================

def run_vsnr(image, theta, device, noise_level):
    """Run the original VSNR Gabor filter using the explicitly chosen backend.

    Filter sigma, iteration count, input precision, and postprocessing are
    identical for CPU and CUDA; only backend and requested noise level vary.
    """
    filters = [
        {
            "name": "Gabor",
            "noise_level": float(noise_level),
            "sigma": (float(SIGMA_SHORT), float(SIGMA_LONG)),
            "theta": float(theta),
        }
    ]
    array = image.astype(np.float32, copy=False)
    if device == "cuda":
        try:
            output = vsnr2d(array, filters, maxit=MAX_ITERS, algo="cupy")
            cp.cuda.Stream.null.synchronize()
        except cp.cuda.memory.OutOfMemoryError as exc:
            release_cupy_memory()
            raise RuntimeError(
                "CuPy GPU OOM during VSNR. No automatic CPU fallback was used."
            ) from exc
    else:
        output = vsnr2d(array, filters, maxit=MAX_ITERS, algo="numpy")
    return to_numpy_float32(output)


def stripe_score(image):
    """Score residual low-frequency vertical banding (smaller is better)."""
    profile = image.mean(axis=0).astype(np.float32)
    profile = gaussian_filter1d(profile, sigma=SWEEP_SMOOTH_SIGMA)
    return float(np.var(profile))


def process_vsnr(image_uint8, device, theta_sweep, noise_level):
    """Select the best VSNR angle and apply the unchanged intensity restoration.

    The original score-based choice is retained even with a single theta,
    ensuring that postprocessing follows the same computation path.
    """
    src = image_uint8.astype(np.float32)
    best = None
    best_score = math.inf
    best_theta = None

    for delta in range(-theta_sweep, theta_sweep + 1):
        theta = THETA_BASE + delta
        candidate = run_vsnr(src, theta, device, noise_level)
        score = stripe_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score
            best_theta = theta
        release_cupy_memory()

    if best is None:
        raise RuntimeError("VSNR produced no valid candidate.")
    if ENABLE_ROBUST_MATCH:
        best = robust_match(best, src)

    best = np.clip(np.round(best), 0, 255).astype(np.uint8)
    return best, best_theta, best_score


# ============================================================
# 6. Single-stack output and application entry point
# ============================================================

def run_pipeline(
    input_path, output_path, *, device, theta_sweep=None,
    noise_level=NOISE_LEVEL, overwrite=False, check_only=False,
):
    """Validate a volume and optionally execute VSNR into one TIFF stack.

    A temporary TIFF is written beside the destination and published only
    after all slices finish successfully, preventing incomplete final output.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")
    if theta_sweep is None:
        theta_sweep = 0 if device == "cpu" else THETA_SWEEP
    if not isinstance(theta_sweep, int) or theta_sweep < 0:
        raise ValueError("theta_sweep must be a nonnegative integer.")
    noise_level = float(noise_level)
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("noise_level must be a positive finite number.")

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output TIFF must be different paths.")
    if not check_only and output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite.")

    with open_uint8_stack(input_path) as (shape, read_slice, axes):
        z_count, height, width = shape
        print(f"Input: {input_path}")
        print(f"Input axes: {axes} | shape ZYX: {shape} | dtype: uint8")
        print(
            f"VSNR: vertical Gabor, {2 * theta_sweep + 1} theta "
            f"candidate(s) (-{theta_sweep}..+{theta_sweep}), maxit={MAX_ITERS}"
        )
        print(f"VSNR noise level: {noise_level:g}")
        print(f"Median/MAD matching: ON | Backend: {device}")
        if check_only:
            print("Input validation: PASS (VSNR not executed)")
            return

        initialize_backend(device)
        selected_gpu = select_gpu_with_most_free_memory() if device == "cuda" else None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Allocate only the final TIFF on disk and one source/output 2D slice
        # in memory. Avoid constructing a full ZYX output in RAM.
        temp_name = None
        mapped_out = None
        theta_counts = {}
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.stem}.",
                suffix=".tif",
                delete=False,
            ) as temporary:
                temp_name = Path(temporary.name)

            mapped_out = tifffile.memmap(
                temp_name,
                shape=(z_count, height, width),
                dtype=np.uint8,
                photometric="minisblack",
                metadata={"axes": "ZYX"},
                bigtiff=True,
            )
            for z in tqdm(range(z_count), desc="VSNR", dynamic_ncols=True):
                plane = np.asarray(read_slice(z))
                if plane.shape != (height, width) or plane.dtype != np.uint8:
                    raise RuntimeError(
                        f"Slice {z}: expected {(height, width)} uint8, "
                        f"got {plane.shape} {plane.dtype}."
                    )
                restored, theta, _score = process_vsnr(
                    plane, device=device, theta_sweep=theta_sweep,
                    noise_level=noise_level,
                )
                if restored.shape != plane.shape or restored.dtype != np.uint8:
                    raise RuntimeError(f"Slice {z}: VSNR changed shape or dtype unexpectedly.")
                mapped_out[z] = restored
                theta_counts[theta] = theta_counts.get(theta, 0) + 1
                release_cupy_memory()

            mapped_out.flush()
            del mapped_out
            mapped_out = None

            # Verify the actual file before atomically replacing the target.
            with tifffile.TiffFile(temp_name) as tif:
                written_shape = tuple(int(v) for v in tif.series[0].shape)
                written_dtype = tif.series[0].dtype
            expected_shape = (z_count, height, width)
            # Some TIFF readers squeeze singleton Z from single-slice stacks.
            if z_count == 1 and written_shape == (height, width):
                written_shape = expected_shape
            if written_shape != expected_shape or written_dtype != np.dtype("uint8"):
                raise RuntimeError(
                    f"Output verification failed: {written_shape}, {written_dtype}."
                )
            os.replace(temp_name, output_path)
            temp_name = None
        finally:
            if mapped_out is not None:
                mapped_out.flush()
                del mapped_out
            if temp_name is not None:
                temp_name.unlink(missing_ok=True)

        if selected_gpu is not None:
            print(f"Selected CUDA GPU: {selected_gpu}")
        print(f"Processed slices: {z_count}")
        for theta in sorted(theta_counts):
            print(f"  theta {theta:+d}: {theta_counts[theta]}")
        print(f"Output: {output_path}")


def main():
    """Parse command line options and run the stacked-TIFF VSNR pipeline."""
    args = parse_args()
    output_path = args.output or args.input.with_name(f"{args.input.stem}_vsnr.tif")
    run_pipeline(
        args.input, output_path, device=args.device,
        theta_sweep=args.theta_sweep, noise_level=args.noise_level,
        overwrite=args.overwrite, check_only=args.check_only,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM membrane segmentation using a pretrained CebraNET TorchScript model.

Author: Ningyan Zhang
Last modified: October 9, 2026

Pipeline
--------
1. Read the Stage-2 uint8 ZYX output stack.
2. Preserve native XY resolution (scale 1.0) and generate an endpoint-aligned
   Z grid (default scale 1.5) by bidirectional motion-aware interpolation.
3. Run 64^3, stride-16 CebraNET inference on NVIDIA CUDA using FP16 forward
   and FP32 weighted 3-D overlap accumulation; switch to exact Z slabs if
   full-volume GPU accumulators exceed the available budget.
4. Interpolate probabilities back to ORIGINAL acquisition Z positions and
   save the probability volume (float32) and 0.30-threshold mask (uint8 0/255)
   in the ORIGINAL input ZYX shape. The saved mask is NOT Z-upsampled.

This inference-only script requires the separately supplied pretrained
ckpt_model/segmentation/weights.torchscript; it does not train CebraNET.

From the repository root, single stack:
    python code/segmentation/cebranet_inference.py \
      --input experiments/stage2_result/output_stack.tif \
      --weights ckpt_model/segmentation/weights.torchscript \
      --output-dir experiments/segmentation_result --z-scale 1.5

Multiple datasets (each subfolder contains output_stack.tif):
    python code/segmentation/cebranet_inference.py \
      --input-root experiments/stage2_outputs \
      --output-root experiments/segmentation_results \
      --weights ckpt_model/segmentation/weights.torchscript \
      --datasets tile_00 tile_01 --z-scale 1.5

Use --check-only to avoid project output changes and validate headers/paths
(no GPU inference; synthetic CPU tests use temporary files) or
--selftest-only for the built-in synthetic CPU tests. Existing output folders
are never replaced without --overwrite.

Scientific constants, synthesis, network I/O, blending, and original-grid
recovery are unchanged from the supplied AutoDL XY1.0 / Z1.5 weighted v3 code.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Avoid malformed inherited OpenMP settings in some AutoDL images.
_omp = os.environ.get("OMP_NUM_THREADS", "")
if (not _omp.isdigit()) or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np
from scipy import ndimage
import tifffile
import torch


# =============================================================================
# 1. Paths and datasets
# =============================================================================

# Resolve repository paths from this file: code/segmentation/cebranet_inference.py.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY_ROOT / "experiments" / "segmentation"
INPUT_ROOT = REPOSITORY_ROOT / "experiments" / "stage2_outputs"
WEIGHT_PATH = REPOSITORY_ROOT / "ckpt_model" / "segmentation" / "weights.torchscript"

DATASET_ORDER: Sequence[str] = (
    "tile_00",
    "tile_01",
    "tile_02",
    "wt_2026_bin2",
    "symbiotic_algae_stack_bin2_region_1",
    "images2_raw_bin2_region_1",
    "EMPIAR-11275_region_1",
)

INPUT_FILENAME = "output_stack.tif"
PROB_FILENAME = "membrane_prob_native_float32.tif"
MASK_FILENAME = "mask_th_0.30_uint8.tif"
SUMMARY_FILENAME = "summary.json"


# =============================================================================
# 2. Physical geometry
# =============================================================================

SOURCE_SPACING_NM_ZYX = (20.0, 5.0, 5.0)
XY_SCALE_Y = 1.0
XY_SCALE_X = 1.0
XY_UP_INTERPOLATION = cv2.INTER_LINEAR
XY_DOWN_INTERPOLATION = cv2.INTER_LINEAR

SUPPORTED_Z_SCALES = (1.0, 1.5, 2.0, 3.0)


# =============================================================================
# 3. Motion-aware correspondence: frozen accepted parameters
# =============================================================================

MATCH_PATCH = 160
MATCH_STRIDE = 80
MAX_SHIFT_PX = 15

NCC_MIN = 0.15
NCC_IMPROVEMENT_MIN = 0.01
IMPROVEMENT_CONF_SCALE = 0.08

FORWARD_BACKWARD_MAX_ERR_PX = 4.0

FIELD_SMOOTH_SIGMA = 12.0
CONF_SMOOTH_SIGMA = 8.0
MATCH_GAUSSIAN_SIGMA = 1.25
EPS = 1e-6


# =============================================================================
# 4. CebraNET frozen settings
# =============================================================================

PATCH_SIZE = 64
STRIDE = 16
THRESHOLD = 0.30
BLEND_FLOOR = 0.20

BATCH_SIZE = 8
USE_FP16_AUTOCAST = True

ACCUMULATOR_MAX_GIB = 16.0
ACCUMULATOR_MAX_FRACTION_OF_VRAM = 0.68
PROGRESS_EVERY_EVALUATIONS = 2000

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# =============================================================================
# 5. General helpers
# =============================================================================

def gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024 ** 3)


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.1f} min"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def read_uint8_zyx(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected ZYX 3-D TIFF, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"{path}: expected uint8, got {arr.dtype}")
    return np.ascontiguousarray(arr)


def tiff_shape_dtype(path: Path) -> Tuple[Tuple[int, int, int], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(str(path)) as tf:
        shape = tuple(int(v) for v in tf.series[0].shape)
        dtype = str(tf.series[0].dtype)
    if len(shape) != 3:
        raise ValueError(f"{path}: expected 3-D TIFF, got {shape}")
    return shape, dtype


def create_tiff_memmap(path: Path, shape: Sequence[int], dtype):
    ensure_dir(path.parent)
    return tifffile.memmap(
        str(path),
        shape=tuple(int(v) for v in shape),
        dtype=dtype,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
        bigtiff=True,
    )


def output_root_for_scale(z_scale: float) -> Path:
    tag = str(z_scale).replace(".", "p")
    if abs(z_scale - 1.0) < 1e-12:
        name = f"cebranet_xy1p0_z{tag}_weighted_v3"
    else:
        name = f"cebranet_motionaware_xy1p0_z{tag}_weighted_v3"
    return ROOT / name


def model_input_filename(z_scale: float) -> str:
    tag = str(z_scale).replace(".", "p")
    return f"cebranet_input_xy1p0_z{tag}_uint8.tif"


def round_half_up_positive(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def model_z_positions(native_z: int, z_scale: float) -> np.ndarray:
    """
    Uniform endpoint-aligned model-space Z coordinates measured in native-slice units.
    """
    if native_z < 1:
        raise ValueError("native_z must be >= 1")
    if native_z == 1:
        return np.array([0.0], dtype=np.float64)

    intervals = round_half_up_positive((native_z - 1) * float(z_scale))
    intervals = max(1, intervals)
    return np.linspace(
        0.0,
        float(native_z - 1),
        intervals + 1,
        dtype=np.float64,
    )


def model_shape_for_native(
    native_shape: Sequence[int],
    z_scale: float,
) -> Tuple[int, int, int]:
    z, h, w = (int(v) for v in native_shape)
    positions = model_z_positions(z, z_scale)
    return (
        int(len(positions)),
        int(round(h * XY_SCALE_Y)),
        int(round(w * XY_SCALE_X)),
    )


def realized_spacing_nm_zyx(
    native_shape: Sequence[int],
    model_shape: Sequence[int],
) -> Tuple[float, float, float]:
    nz, ny, nx = (int(v) for v in native_shape)
    mz, my, mx = (int(v) for v in model_shape)

    if nz > 1 and mz > 1:
        rz = SOURCE_SPACING_NM_ZYX[0] * (nz - 1) / (mz - 1)
    else:
        rz = SOURCE_SPACING_NM_ZYX[0]

    ry = SOURCE_SPACING_NM_ZYX[1] * ny / my
    rx = SOURCE_SPACING_NM_ZYX[2] * nx / mx
    return float(rz), float(ry), float(rx)


def axis_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        raise ValueError(
            f"Axis length {length} is smaller than patch size {patch_size}."
        )
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


# =============================================================================
# 6. Motion-aware matching
# =============================================================================

def matching_view(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    mean = float(x.mean())
    std = float(x.std())

    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    x = (x - mean) / std

    if MATCH_GAUSSIAN_SIGMA > 0:
        x = ndimage.gaussian_filter(
            x,
            sigma=MATCH_GAUSSIAN_SIGMA,
        ).astype(np.float32)

    return x.astype(np.float32, copy=False)


def patch_grid_positions(length: int) -> List[int]:
    if length < MATCH_PATCH:
        raise ValueError(
            f"Image length {length} is smaller than MATCH_PATCH={MATCH_PATCH}."
        )
    positions = list(range(0, length - MATCH_PATCH + 1, MATCH_STRIDE))
    last = length - MATCH_PATCH
    if positions[-1] != last:
        positions.append(last)
    return positions


def estimate_patch_shift(
    reference_patch: np.ndarray,
    search_image: np.ndarray,
) -> Dict[str, float]:
    score_map = cv2.matchTemplate(
        search_image,
        reference_patch,
        cv2.TM_CCOEFF_NORMED,
    ).astype(np.float32)

    expected = 2 * MAX_SHIFT_PX + 1
    if score_map.shape != (expected, expected):
        raise RuntimeError(
            f"Unexpected NCC score-map shape {score_map.shape}; "
            f"expected {(expected, expected)}"
        )

    flat = int(np.argmax(score_map))
    row, col = np.unravel_index(flat, score_map.shape)

    dy = int(row - MAX_SHIFT_PX)
    dx = int(col - MAX_SHIFT_PX)

    best = float(score_map[row, col])
    zero = float(score_map[MAX_SHIFT_PX, MAX_SHIFT_PX])
    improvement = float(best - zero)

    accepted = (
        np.isfinite(best)
        and best >= NCC_MIN
        and np.isfinite(improvement)
        and improvement >= NCC_IMPROVEMENT_MIN
    )

    return {
        "best_dx": dx,
        "best_dy": dy,
        "best_score": best,
        "zero_score": zero,
        "improvement": improvement,
        "accepted": bool(accepted),
    }


def estimate_translation_grid(
    slice_a: np.ndarray,
    slice_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    if slice_a.shape != slice_b.shape:
        raise ValueError(f"Pair shape mismatch: {slice_a.shape} vs {slice_b.shape}")

    h, w = slice_a.shape
    ys = patch_grid_positions(h)
    xs = patch_grid_positions(w)

    pad = MAX_SHIFT_PX
    b_pad = np.pad(
        slice_b,
        ((pad, pad), (pad, pad)),
        mode="reflect",
    )

    shape = (len(ys), len(xs))
    dx_grid = np.full(shape, np.nan, dtype=np.float32)
    dy_grid = np.full(shape, np.nan, dtype=np.float32)
    conf_grid = np.zeros(shape, dtype=np.float32)
    accepted_grid = np.zeros(shape, dtype=np.uint8)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            ref = slice_a[
                y:y + MATCH_PATCH,
                x:x + MATCH_PATCH,
            ].astype(np.float32, copy=False)

            search = b_pad[
                y:y + MATCH_PATCH + 2 * MAX_SHIFT_PX,
                x:x + MATCH_PATCH + 2 * MAX_SHIFT_PX,
            ].astype(np.float32, copy=False)

            result = estimate_patch_shift(ref, search)

            if not result["accepted"]:
                continue

            accepted_grid[iy, ix] = 1
            dx_grid[iy, ix] = float(result["best_dx"])
            dy_grid[iy, ix] = float(result["best_dy"])

            ncc_conf = np.clip(
                (result["best_score"] - NCC_MIN)
                / max(1.0 - NCC_MIN, EPS),
                0.0,
                1.0,
            )

            improvement_conf = np.clip(
                result["improvement"] / IMPROVEMENT_CONF_SCALE,
                0.0,
                1.0,
            )

            conf_grid[iy, ix] = float(ncc_conf * improvement_conf)

    return {
        "dx_grid": dx_grid,
        "dy_grid": dy_grid,
        "conf_grid": conf_grid,
        "accepted_grid": accepted_grid,
    }


def fill_rejected_nearest(
    values: np.ndarray,
    accepted_grid: np.ndarray,
) -> np.ndarray:
    accepted = accepted_grid.astype(bool)

    if not np.any(accepted):
        return np.zeros_like(values, dtype=np.float32)

    invalid = ~accepted
    nearest_indices = ndimage.distance_transform_edt(
        invalid,
        return_distances=False,
        return_indices=True,
    )
    filled = values[tuple(nearest_indices)]
    return np.nan_to_num(filled, nan=0.0).astype(np.float32)


def resize_field(
    grid: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    h, w = shape
    return cv2.resize(
        grid.astype(np.float32, copy=False),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def sample_field_at_endpoint(
    field: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    h, w = field.shape

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    return cv2.remap(
        field.astype(np.float32, copy=False),
        (xx + dx).astype(np.float32),
        (yy + dy).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)


def build_pair_fields(
    match_a: np.ndarray,
    match_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    ab = estimate_translation_grid(match_a, match_b)
    ba = estimate_translation_grid(match_b, match_a)

    h, w = match_a.shape

    dx_ab_grid = fill_rejected_nearest(
        ab["dx_grid"],
        ab["accepted_grid"],
    )
    dy_ab_grid = fill_rejected_nearest(
        ab["dy_grid"],
        ab["accepted_grid"],
    )
    dx_ba_grid = fill_rejected_nearest(
        ba["dx_grid"],
        ba["accepted_grid"],
    )
    dy_ba_grid = fill_rejected_nearest(
        ba["dy_grid"],
        ba["accepted_grid"],
    )

    dx_ab = ndimage.gaussian_filter(
        resize_field(dx_ab_grid, (h, w)),
        sigma=FIELD_SMOOTH_SIGMA,
    ).astype(np.float32)
    dy_ab = ndimage.gaussian_filter(
        resize_field(dy_ab_grid, (h, w)),
        sigma=FIELD_SMOOTH_SIGMA,
    ).astype(np.float32)
    dx_ba = ndimage.gaussian_filter(
        resize_field(dx_ba_grid, (h, w)),
        sigma=FIELD_SMOOTH_SIGMA,
    ).astype(np.float32)
    dy_ba = ndimage.gaussian_filter(
        resize_field(dy_ba_grid, (h, w)),
        sigma=FIELD_SMOOTH_SIGMA,
    ).astype(np.float32)

    conf_ab = ndimage.gaussian_filter(
        resize_field(ab["conf_grid"], (h, w)),
        sigma=CONF_SMOOTH_SIGMA,
    ).astype(np.float32)
    conf_ba = ndimage.gaussian_filter(
        resize_field(ba["conf_grid"], (h, w)),
        sigma=CONF_SMOOTH_SIGMA,
    ).astype(np.float32)

    conf_ab = np.clip(conf_ab, 0.0, 1.0)
    conf_ba = np.clip(conf_ba, 0.0, 1.0)

    # Forward/backward consistency affects confidence diagnostics only.
    # It does NOT alter displacement and does NOT switch output source.
    dx_ba_ep = sample_field_at_endpoint(dx_ba, dx_ab, dy_ab)
    dy_ba_ep = sample_field_at_endpoint(dy_ba, dx_ab, dy_ab)
    conf_ba_ep = sample_field_at_endpoint(conf_ba, dx_ab, dy_ab)

    err_ab = np.sqrt(
        (dx_ab + dx_ba_ep) ** 2
        + (dy_ab + dy_ba_ep) ** 2
    )

    gate_ab = np.clip(
        1.0 - err_ab / FORWARD_BACKWARD_MAX_ERR_PX,
        0.0,
        1.0,
    )

    dx_ab_ep = sample_field_at_endpoint(dx_ab, dx_ba, dy_ba)
    dy_ab_ep = sample_field_at_endpoint(dy_ab, dx_ba, dy_ba)
    conf_ab_ep = sample_field_at_endpoint(conf_ab, dx_ba, dy_ba)

    err_ba = np.sqrt(
        (dx_ba + dx_ab_ep) ** 2
        + (dy_ba + dy_ab_ep) ** 2
    )

    gate_ba = np.clip(
        1.0 - err_ba / FORWARD_BACKWARD_MAX_ERR_PX,
        0.0,
        1.0,
    )

    conf_ab = np.clip(
        conf_ab
        * gate_ab
        * np.sqrt(np.clip(conf_ba_ep, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)

    conf_ba = np.clip(
        conf_ba
        * gate_ba
        * np.sqrt(np.clip(conf_ab_ep, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)

    diagnostics = {
        "forward_accept_rate": float(ab["accepted_grid"].mean()),
        "reverse_accept_rate": float(ba["accepted_grid"].mean()),
        "forward_confidence_mean": float(conf_ab.mean()),
        "reverse_confidence_mean": float(conf_ba.mean()),
        "forward_consistency_error_mean_px": float(err_ab.mean()),
        "forward_consistency_error_p95_px": float(np.percentile(err_ab, 95)),
        "reverse_consistency_error_mean_px": float(err_ba.mean()),
        "reverse_consistency_error_p95_px": float(np.percentile(err_ba, 95)),
    }

    return {
        "dx_ab": dx_ab,
        "dy_ab": dy_ab,
        "conf_ab": conf_ab,
        "dx_ba": dx_ba,
        "dy_ba": dy_ba,
        "conf_ba": conf_ba,
        "diagnostics": diagnostics,
    }


def warp_image(
    image: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    fraction: float,
) -> np.ndarray:
    h, w = image.shape

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    map_x = (
        xx
        - float(fraction) * dx
    ).astype(np.float32)

    map_y = (
        yy
        - float(fraction) * dy
    ).astype(np.float32)

    return cv2.remap(
        image.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)


def synthesize_fractional_fullfield(
    image_a: np.ndarray,
    image_b: np.ndarray,
    fields: Dict[str, np.ndarray],
    t: float,
) -> np.ndarray:
    """
    Global full-field bidirectional interpolation.

    This is NOT confidence weighting.
    At t=0.5 this is exactly the accepted 50/50 midpoint.
    """
    t = float(t)

    if not (0.0 < t < 1.0):
        raise ValueError(f"Expected 0<t<1, got {t}")

    a_warp = warp_image(
        image_a,
        fields["dx_ab"],
        fields["dy_ab"],
        t,
    )

    b_warp = warp_image(
        image_b,
        fields["dx_ba"],
        fields["dy_ba"],
        1.0 - t,
    )

    output = (
        (1.0 - t) * a_warp
        + t * b_warp
    )

    return np.clip(
        np.round(output),
        0.0,
        255.0,
    ).astype(np.uint8)


def resize_xy_to_model(
    image: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    # XY scale is 1.0 in this experiment: preserve native pixels exactly.
    if image.shape == (int(out_h), int(out_w)):
        return np.ascontiguousarray(image, dtype=np.uint8)

    out = cv2.resize(
        image,
        (int(out_w), int(out_h)),
        interpolation=XY_UP_INTERPOLATION,
    )
    return np.clip(
        np.round(out),
        0.0,
        255.0,
    ).astype(np.uint8)


def build_synthesis_plan(
    native_z: int,
    z_scale: float,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float]]], Dict[int, int]]:
    """
    Returns
    -------
    positions:
        model-plane positions in native-slice coordinates.
    pair_requests:
        left native pair index -> [(model_index, t), ...]
    direct_requests:
        model_index -> exact native index
    """
    positions = model_z_positions(native_z, z_scale)

    pair_requests: Dict[int, List[Tuple[int, float]]] = {}
    direct_requests: Dict[int, int] = {}

    tol = 1e-9

    for model_index, source_pos in enumerate(positions):
        nearest = int(round(float(source_pos)))

        if abs(float(source_pos) - nearest) <= tol:
            direct_requests[int(model_index)] = int(nearest)
            continue

        left = int(math.floor(float(source_pos)))
        right = left + 1

        if left < 0 or right >= native_z:
            raise RuntimeError(
                f"Invalid source position {source_pos} for native_z={native_z}"
            )

        t = float(source_pos - left)
        pair_requests.setdefault(left, []).append(
            (int(model_index), t)
        )

    return positions, pair_requests, direct_requests


def prepare_model_u8(
    native_u8: np.ndarray,
    z_scale: float,
    workers: int,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Build the full CebraNET uint8 model-space volume in RAM.

    Threaded execution changes only execution order, not the scientific
    interpolation definition.
    """
    native_z, native_h, native_w = (
        int(v) for v in native_u8.shape
    )

    model_shape = model_shape_for_native(
        native_u8.shape,
        z_scale,
    )

    model_z, out_h, out_w = model_shape

    positions, pair_requests, direct_requests = (
        build_synthesis_plan(
            native_z,
            z_scale,
        )
    )

    model_u8 = np.empty(
        model_shape,
        dtype=np.uint8,
    )

    cv2.setNumThreads(1)

    # Exact planes that happen to lie on the model Z grid.
    for model_index, native_index in direct_requests.items():
        model_u8[model_index] = resize_xy_to_model(
            native_u8[native_index],
            out_h,
            out_w,
        )

    if not pair_requests:
        return model_u8, [], positions

    max_workers = max(
        1,
        min(
            int(workers),
            len(pair_requests),
        ),
    )

    # Bounded shared cache. At 1920x1920 each cached matching view is ~14 MiB.
    cache_size = max(
        4,
        2 * max_workers + 4,
    )

    @functools.lru_cache(maxsize=cache_size)
    def cached_matching_view(native_index: int) -> np.ndarray:
        return matching_view(
            native_u8[int(native_index)]
        )

    pair_keys = sorted(pair_requests.keys())

    def process_one_pair(left: int):
        image_a = native_u8[left]
        image_b = native_u8[left + 1]

        fields = build_pair_fields(
            cached_matching_view(left),
            cached_matching_view(left + 1),
        )

        generated = []

        for model_index, t in pair_requests[left]:
            synthesized = synthesize_fractional_fullfield(
                image_a,
                image_b,
                fields,
                t,
            )

            generated.append(
                (
                    int(model_index),
                    resize_xy_to_model(
                        synthesized,
                        out_h,
                        out_w,
                    ),
                )
            )

        diag = {
            "left_real_index": int(left),
            "right_real_index": int(left + 1),
            "generated_model_indices": [
                int(idx)
                for idx, _ in pair_requests[left]
            ],
            "generated_t_values": [
                float(t)
                for _, t in pair_requests[left]
            ],
            **fields["diagnostics"],
        }

        return int(left), generated, diag

    pair_diagnostics: List[Dict] = []

    print(
        f"  Motion-aware pairs: {len(pair_keys)}"
        f" | workers={max_workers}"
        f" | matching-cache={cache_size}"
    )

    completed = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="motion",
    ) as pool:
        for left, generated, diag in pool.map(
            process_one_pair,
            pair_keys,
        ):
            for model_index, plane in generated:
                model_u8[model_index] = plane

            pair_diagnostics.append(diag)
            completed += 1

            if (
                completed == 1
                or completed % 8 == 0
                or completed == len(pair_keys)
            ):
                elapsed = time.perf_counter() - t0
                rate = completed / max(elapsed, 1e-12)
                eta = (
                    len(pair_keys) - completed
                ) / max(rate, 1e-12)

                print(
                    f"\r  Motion-aware {completed}/{len(pair_keys)}"
                    f" | {rate:.2f} pair/s"
                    f" | ETA {eta:.1f} s",
                    end="",
                    flush=True,
                )

    print()

    # Hard Z=2 contract: every native plane must occur exactly at even index.
    if abs(z_scale - 2.0) < 1e-12:
        expected_z = 2 * (native_z - 1) + 1

        if model_z != expected_z:
            raise RuntimeError(
                f"Z=2 model length mismatch: {model_z} != {expected_z}"
            )

        for native_index in range(native_z):
            model_index = 2 * native_index
            expected = resize_xy_to_model(
                native_u8[native_index],
                out_h,
                out_w,
            )

            if not np.array_equal(
                model_u8[model_index],
                expected,
            ):
                raise RuntimeError(
                    f"Z=2 preserve-original contract failed at "
                    f"native={native_index}, model={model_index}"
                )

    return model_u8, pair_diagnostics, positions


# =============================================================================
# 7. CebraNET output selection and safe batching
# =============================================================================

def make_blending_weight(size: int) -> np.ndarray:
    x = np.linspace(
        -1.0,
        1.0,
        size,
        dtype=np.float32,
    )

    w = (
        BLEND_FLOOR
        + (1.0 - BLEND_FLOOR)
        * (1.0 - np.abs(x))
    )

    weight = (
        w[:, None, None]
        * w[None, :, None]
        * w[None, None, :]
    )

    return weight.astype(
        np.float32,
        copy=False,
    )


def get_output_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item

    raise TypeError(
        f"Could not find Tensor in model output: {type(output)}"
    )


def select_prediction_channel(
    output,
    expected_batch: int,
) -> torch.Tensor:
    pred = get_output_tensor(output)

    if pred.ndim != 5:
        raise RuntimeError(
            "Expected B,C,Z,Y,X model output, "
            f"got {tuple(pred.shape)}"
        )

    if int(pred.shape[0]) != int(expected_batch):
        raise RuntimeError(
            f"Batch mismatch: expected={expected_batch}, "
            f"got={pred.shape[0]}"
        )

    if int(pred.shape[1]) == 1:
        pred = pred[:, 0]
    else:
        pred = pred[:, -1]

    if tuple(int(v) for v in pred.shape[-3:]) != (
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise RuntimeError(
            "Unexpected CebraNET output spatial shape: "
            f"{tuple(pred.shape)}"
        )

    return pred.float()


def allocate_host_batch():
    host_batch = torch.empty(
        (
            BATCH_SIZE,
            1,
            PATCH_SIZE,
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        dtype=torch.float32,
        pin_memory=True,
    )

    return host_batch, host_batch.numpy()


def fill_host_batch(
    host_np: np.ndarray,
    model_u8: np.ndarray,
    z0: int,
    coords: Sequence[Tuple[int, int]],
) -> None:
    for i, (y0, x0) in enumerate(coords):
        crop = model_u8[
            z0:z0 + PATCH_SIZE,
            y0:y0 + PATCH_SIZE,
            x0:x0 + PATCH_SIZE,
        ]

        if crop.shape != (
            PATCH_SIZE,
            PATCH_SIZE,
            PATCH_SIZE,
        ):
            raise RuntimeError(
                "Patch extraction geometry failure:\n"
                f"z0,y0,x0={(z0, y0, x0)}\n"
                f"crop shape={crop.shape}"
            )

        np.multiply(
            crop,
            1.0 / 255.0,
            out=host_np[i, 0],
            casting="unsafe",
        )


def forward_host_batch(
    model,
    host_batch: torch.Tensor,
    batch_count: int,
    device: torch.device,
) -> torch.Tensor:
    x_gpu = host_batch[:batch_count].to(
        device,
        dtype=torch.float32,
        non_blocking=False,
    )

    with torch.inference_mode():
        if USE_FP16_AUTOCAST:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                raw_output = model(x_gpu)
        else:
            raw_output = model(x_gpu)

    pred = select_prediction_channel(
        raw_output,
        expected_batch=batch_count,
    )

    del x_gpu, raw_output
    return pred


# =============================================================================
# 8. CebraNET accumulator planning
# =============================================================================

def accumulator_budget_bytes(
    device: torch.device,
) -> int:
    total_vram = int(
        torch.cuda.get_device_properties(
            device
        ).total_memory
    )

    absolute_cap = int(
        ACCUMULATOR_MAX_GIB
        * (1024 ** 3)
    )

    fraction_cap = int(
        total_vram
        * ACCUMULATOR_MAX_FRACTION_OF_VRAM
    )

    return min(
        absolute_cap,
        fraction_cap,
    )


def full_accumulator_bytes(
    shape: Sequence[int],
) -> int:
    voxels = int(
        np.prod(
            np.asarray(
                shape,
                dtype=np.int64,
            )
        )
    )
    return voxels * 4 * 2


def choose_slab_depth(
    shape: Sequence[int],
    budget_bytes: int,
) -> int:
    z, y, x = (
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
    )

    bytes_per_z = y * x * 4 * 2
    max_depth = max(
        1,
        budget_bytes // bytes_per_z,
    )

    if max_depth >= STRIDE:
        max_depth = (
            max_depth // STRIDE
        ) * STRIDE

    return min(
        z,
        max(1, int(max_depth)),
    )


def effective_evaluations_for_slab(
    z_starts: Sequence[int],
    yx_count: int,
    z_length: int,
    slab_depth: int,
) -> int:
    total = 0

    for slab_z0 in range(
        0,
        z_length,
        slab_depth,
    ):
        slab_z1 = min(
            z_length,
            slab_z0 + slab_depth,
        )

        overlapping = [
            z0
            for z0 in z_starts
            if z0 < slab_z1
            and z0 + PATCH_SIZE > slab_z0
        ]

        total += len(overlapping) * int(yx_count)

    return int(total)


# =============================================================================
# 9. Weighted CebraNET backends
# =============================================================================

def run_cuda_global_inference(
    model,
    model_u8: np.ndarray,
    blend_gpu: torch.Tensor,
    z_starts: Sequence[int],
    y_starts: Sequence[int],
    x_starts: Sequence[int],
    device: torch.device,
):
    shape = tuple(int(v) for v in model_u8.shape)

    yx_coords = [
        (y0, x0)
        for y0 in y_starts
        for x0 in x_starts
    ]

    total_evaluations = (
        len(z_starts)
        * len(yx_coords)
    )

    print("  Backend: cuda_global")
    print(
        "  Global accumulators:",
        f"{gib(full_accumulator_bytes(shape)):.2f} GiB",
    )
    print(
        "  Tile evaluations:",
        f"{total_evaluations:,}",
    )

    prediction_sum = torch.zeros(
        shape,
        device=device,
        dtype=torch.float32,
    )

    weight_sum = torch.zeros(
        shape,
        device=device,
        dtype=torch.float32,
    )

    host_batch, host_np = allocate_host_batch()

    processed = 0
    t0 = time.perf_counter()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for z0 in z_starts:
        for batch_start in range(
            0,
            len(yx_coords),
            BATCH_SIZE,
        ):
            coords = yx_coords[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            batch_count = len(coords)

            fill_host_batch(
                host_np=host_np,
                model_u8=model_u8,
                z0=z0,
                coords=coords,
            )

            pred = forward_host_batch(
                model=model,
                host_batch=host_batch,
                batch_count=batch_count,
                device=device,
            )

            for i, (y0, x0) in enumerate(coords):
                sl = (
                    slice(z0, z0 + PATCH_SIZE),
                    slice(y0, y0 + PATCH_SIZE),
                    slice(x0, x0 + PATCH_SIZE),
                )

                prediction_sum[sl].add_(
                    pred[i] * blend_gpu
                )

                weight_sum[sl].add_(
                    blend_gpu
                )

            processed += batch_count
            del pred

            if (
                processed <= BATCH_SIZE
                or processed % PROGRESS_EVERY_EVALUATIONS < batch_count
                or processed == total_evaluations
            ):
                torch.cuda.synchronize()

                elapsed = time.perf_counter() - t0
                rate = processed / max(elapsed, 1e-12)
                eta = (
                    total_evaluations - processed
                ) / max(rate, 1e-12)

                print(
                    f"\r    Tiles {processed:,}/{total_evaluations:,}"
                    f" | {rate:.1f} tile/s"
                    f" | ETA {eta / 60.0:.1f} min",
                    end="",
                    flush=True,
                )

    torch.cuda.synchronize()
    print()

    if bool(torch.any(weight_sum <= 0).item()):
        raise RuntimeError(
            "Global weight_sum contains zero or negative voxels."
        )

    prediction_sum.div_(
        weight_sum.clamp_min_(1e-8)
    )

    probability = (
        prediction_sum
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    elapsed = time.perf_counter() - t0
    peak_cuda_mib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    del (
        prediction_sum,
        weight_sum,
        host_batch,
        host_np,
    )

    torch.cuda.empty_cache()
    gc.collect()

    return probability, {
        "backend": "cuda_global",
        "unique_tiles": int(total_evaluations),
        "actual_tile_evaluations": int(total_evaluations),
        "boundary_re_evaluation_factor": 1.0,
        "inference_seconds": float(elapsed),
        "average_tile_evaluations_per_second": float(
            total_evaluations / max(elapsed, 1e-12)
        ),
        "peak_cuda_mib": float(peak_cuda_mib),
    }


def run_cuda_absolute_zslab_inference(
    model,
    model_u8: np.ndarray,
    blend_gpu: torch.Tensor,
    z_starts: Sequence[int],
    y_starts: Sequence[int],
    x_starts: Sequence[int],
    budget_bytes: int,
    scratch_dir: Path,
    device: torch.device,
):
    """
    Exact absolute-Z slabs. Boundary-crossing patches are re-evaluated;
    no ring buffer, wrap, modulo remapping, or carried overlap state.
    """
    shape = tuple(int(v) for v in model_u8.shape)
    z_length, y_length, x_length = shape

    yx_coords = [
        (y0, x0)
        for y0 in y_starts
        for x0 in x_starts
    ]

    unique_tiles = (
        len(z_starts)
        * len(yx_coords)
    )

    slab_depth = choose_slab_depth(
        shape,
        budget_bytes,
    )

    actual_evaluations = effective_evaluations_for_slab(
        z_starts=z_starts,
        yx_count=len(yx_coords),
        z_length=z_length,
        slab_depth=slab_depth,
    )

    re_eval_factor = (
        actual_evaluations
        / max(unique_tiles, 1)
    )

    print("  Backend: cuda_absolute_zslab")
    print(
        "  Full global accumulators would require:",
        f"{gib(full_accumulator_bytes(shape)):.2f} GiB",
    )
    print(
        "  Accumulator budget:",
        f"{gib(budget_bytes):.2f} GiB",
    )
    print(
        "  Absolute slab depth:",
        slab_depth,
    )
    print(
        "  Unique tiles:",
        f"{unique_tiles:,}",
    )
    print(
        "  Actual tile evaluations:",
        f"{actual_evaluations:,}",
        f"(x{re_eval_factor:.3f})",
    )

    ensure_dir(scratch_dir)

    probability_path = (
        scratch_dir
        / "model_probability_float32.dat"
    )

    probability = np.memmap(
        probability_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    host_batch, host_np = allocate_host_batch()

    processed = 0
    t0 = time.perf_counter()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for slab_z0 in range(
        0,
        z_length,
        slab_depth,
    ):
        slab_z1 = min(
            z_length,
            slab_z0 + slab_depth,
        )

        current_depth = slab_z1 - slab_z0

        prediction_sum = torch.zeros(
            (
                current_depth,
                y_length,
                x_length,
            ),
            device=device,
            dtype=torch.float32,
        )

        weight_sum = torch.zeros(
            (
                current_depth,
                y_length,
                x_length,
            ),
            device=device,
            dtype=torch.float32,
        )

        overlapping_z_starts = [
            z0
            for z0 in z_starts
            if z0 < slab_z1
            and z0 + PATCH_SIZE > slab_z0
        ]

        if not overlapping_z_starts:
            raise RuntimeError(
                f"Slab [{slab_z0}:{slab_z1}) has no overlapping patches."
            )

        for patch_z0 in overlapping_z_starts:
            global_overlap_z0 = max(
                patch_z0,
                slab_z0,
            )
            global_overlap_z1 = min(
                patch_z0 + PATCH_SIZE,
                slab_z1,
            )

            pred_z0 = global_overlap_z0 - patch_z0
            pred_z1 = global_overlap_z1 - patch_z0

            slab_local_z0 = global_overlap_z0 - slab_z0
            slab_local_z1 = global_overlap_z1 - slab_z0

            blend_z = blend_gpu[
                pred_z0:pred_z1
            ]

            for batch_start in range(
                0,
                len(yx_coords),
                BATCH_SIZE,
            ):
                coords = yx_coords[
                    batch_start:
                    batch_start + BATCH_SIZE
                ]

                batch_count = len(coords)

                fill_host_batch(
                    host_np=host_np,
                    model_u8=model_u8,
                    z0=patch_z0,
                    coords=coords,
                )

                pred = forward_host_batch(
                    model=model,
                    host_batch=host_batch,
                    batch_count=batch_count,
                    device=device,
                )

                pred_overlap = pred[
                    :,
                    pred_z0:pred_z1,
                ]

                for i, (y0, x0) in enumerate(coords):
                    sl = (
                        slice(
                            slab_local_z0,
                            slab_local_z1,
                        ),
                        slice(
                            y0,
                            y0 + PATCH_SIZE,
                        ),
                        slice(
                            x0,
                            x0 + PATCH_SIZE,
                        ),
                    )

                    prediction_sum[sl].add_(
                        pred_overlap[i] * blend_z
                    )

                    weight_sum[sl].add_(
                        blend_z
                    )

                processed += batch_count

                del pred, pred_overlap

                if (
                    processed <= BATCH_SIZE
                    or processed % PROGRESS_EVERY_EVALUATIONS < batch_count
                    or processed == actual_evaluations
                ):
                    torch.cuda.synchronize()

                    elapsed = time.perf_counter() - t0
                    rate = processed / max(elapsed, 1e-12)
                    eta = (
                        actual_evaluations - processed
                    ) / max(rate, 1e-12)

                    print(
                        f"\r    Tiles {processed:,}/{actual_evaluations:,}"
                        f" | {rate:.1f} tile/s"
                        f" | ETA {eta / 60.0:.1f} min",
                        end="",
                        flush=True,
                    )

        torch.cuda.synchronize()

        if bool(torch.any(weight_sum <= 0).item()):
            raise RuntimeError(
                "Slab weight_sum contains zero or negative voxels.\n"
                f"slab=[{slab_z0}:{slab_z1})"
            )

        prediction_sum.div_(
            weight_sum.clamp_min_(1e-8)
        )

        probability[
            slab_z0:slab_z1
        ] = (
            prediction_sum
            .cpu()
            .numpy()
        )

        probability.flush()

        del prediction_sum, weight_sum
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.synchronize()
    print()

    if processed != actual_evaluations:
        raise RuntimeError(
            "Tile evaluation accounting mismatch:\n"
            f"processed={processed}\n"
            f"expected={actual_evaluations}"
        )

    elapsed = time.perf_counter() - t0
    peak_cuda_mib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    del host_batch, host_np
    torch.cuda.empty_cache()
    gc.collect()

    return probability, {
        "backend": "cuda_absolute_zslab",
        "slab_depth_model_z": int(slab_depth),
        "unique_tiles": int(unique_tiles),
        "actual_tile_evaluations": int(actual_evaluations),
        "boundary_re_evaluation_factor": float(re_eval_factor),
        "inference_seconds": float(elapsed),
        "average_tile_evaluations_per_second": float(
            actual_evaluations / max(elapsed, 1e-12)
        ),
        "peak_cuda_mib": float(peak_cuda_mib),
        "model_probability_scratch_path": str(
            probability_path
        ),
    }


# =============================================================================
# 10. Restore probability to native physical Z positions + native XY
# =============================================================================

def restore_and_save_native_outputs(
    model_probability: np.ndarray,
    model_positions: np.ndarray,
    native_shape: Sequence[int],
    probability_path: Path,
    mask_path: Path,
) -> Dict[str, float]:
    native_z, native_h, native_w = (
        int(v) for v in native_shape
    )

    model_z = int(model_probability.shape[0])

    if len(model_positions) != model_z:
        raise RuntimeError(
            "Model position/probability length mismatch."
        )

    prob_mm = create_tiff_memmap(
        probability_path,
        native_shape,
        np.float32,
    )

    mask_mm = create_tiff_memmap(
        mask_path,
        native_shape,
        np.uint8,
    )

    value_min = math.inf
    value_max = -math.inf
    value_sum = 0.0
    value_count = 0
    positive_count = 0

    try:
        for native_index in range(native_z):
            target_pos = float(native_index)

            right = int(
                np.searchsorted(
                    model_positions,
                    target_pos,
                    side="left",
                )
            )

            if right <= 0:
                j0 = j1 = 0
                alpha = 0.0
            elif right >= model_z:
                j0 = j1 = model_z - 1
                alpha = 0.0
            elif abs(
                float(model_positions[right])
                - target_pos
            ) <= 1e-10:
                j0 = j1 = right
                alpha = 0.0
            else:
                j0 = right - 1
                j1 = right
                p0 = float(model_positions[j0])
                p1 = float(model_positions[j1])
                alpha = (
                    target_pos - p0
                ) / max(p1 - p0, 1e-12)

            if j0 == j1:
                model_plane = np.asarray(
                    model_probability[j0],
                    dtype=np.float32,
                )
            else:
                model_plane = (
                    (1.0 - alpha)
                    * np.asarray(
                        model_probability[j0],
                        dtype=np.float32,
                    )
                    + alpha
                    * np.asarray(
                        model_probability[j1],
                        dtype=np.float32,
                    )
                ).astype(
                    np.float32,
                    copy=False,
                )

            if model_plane.shape == (native_h, native_w):
                native_plane = np.asarray(
                    model_plane,
                    dtype=np.float32,
                )
            else:
                native_plane = cv2.resize(
                    model_plane,
                    (native_w, native_h),
                    interpolation=XY_DOWN_INTERPOLATION,
                ).astype(np.float32)

            native_plane = np.clip(
                native_plane,
                0.0,
                1.0,
            )

            mask_plane = (
                native_plane >= THRESHOLD
            ).astype(np.uint8) * 255

            prob_mm[native_index] = native_plane
            mask_mm[native_index] = mask_plane

            value_min = min(
                value_min,
                float(native_plane.min()),
            )
            value_max = max(
                value_max,
                float(native_plane.max()),
            )
            value_sum += float(
                native_plane.sum(
                    dtype=np.float64
                )
            )
            value_count += int(native_plane.size)
            positive_count += int(
                (mask_plane > 0).sum()
            )

            if (
                native_index == 0
                or (native_index + 1) % 16 == 0
                or native_index == native_z - 1
            ):
                prob_mm.flush()
                mask_mm.flush()

        prob_mm.flush()
        mask_mm.flush()
    finally:
        del prob_mm, mask_mm

    # Exact disk read-back contract for mask coding.
    mask_check = tifffile.memmap(
        str(mask_path),
        mode="r",
    )

    unique_mask = np.unique(mask_check)

    del mask_check

    if not set(
        int(v) for v in unique_mask.tolist()
    ).issubset({0, 255}):
        raise RuntimeError(
            f"Mask coding failure: unique={unique_mask}"
        )

    return {
        "probability_min": float(value_min),
        "probability_max": float(value_max),
        "probability_mean": float(
            value_sum / max(value_count, 1)
        ),
        "positive_fraction": float(
            positive_count / max(value_count, 1)
        ),
        "mask_unique_values": [
            int(v)
            for v in unique_mask.tolist()
        ],
    }


# =============================================================================
# 11. Environment, model, and dataset preflight
# =============================================================================

def validate_environment() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This formal script is for AutoDL RTX4090."
        )

    device = torch.device("cuda:0")

    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GiB:",
        round(
            torch.cuda.get_device_properties(
                device
            ).total_memory
            / (1024 ** 3),
            2,
        ),
    )

    return device


def load_and_precheck_model(
    device: torch.device,
):
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(WEIGHT_PATH)

    model = torch.jit.load(
        str(WEIGHT_PATH),
        map_location=device,
    )
    model.eval()

    x = torch.zeros(
        (
            1,
            1,
            PATCH_SIZE,
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        if USE_FP16_AUTOCAST:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                y = model(x)
        else:
            y = model(x)

    pred = select_prediction_channel(
        y,
        expected_batch=1,
    )

    if tuple(pred.shape) != (
        1,
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected model precheck shape: {tuple(pred.shape)}"
        )

    del x, y, pred
    torch.cuda.empty_cache()

    print("TorchScript CUDA forward: PASS")
    return model


def discover_dataset_records(
    dataset_names: Sequence[str],
    z_scale: float,
) -> List[Dict]:
    """Inspect input stack headers without loading image pixels."""
    records = []
    for name in dataset_names:
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError(f"Invalid dataset folder name: {name!r}")
        path = INPUT_ROOT / name / INPUT_FILENAME
        shape, dtype = tiff_shape_dtype(path)
        records.append(make_dataset_record(name, path, shape, dtype, z_scale))
    return records


def make_dataset_record(
    name: str,
    path: Path,
    shape: Sequence[int],
    dtype: str,
    z_scale: float,
) -> Dict:
    if dtype != "uint8":
        raise TypeError(f"{name}: expected uint8 input, got {dtype}")
    model_shape = model_shape_for_native(shape, z_scale)
    if abs(float(z_scale) - 1.0) > 1e-12 and (shape[1] < MATCH_PATCH or shape[2] < MATCH_PATCH):
        raise ValueError(
            f"{name}: motion-aware interpolation requires native XY >= "
            f"{MATCH_PATCH} pixels; got {tuple(shape)}"
        )
    if any(dim < PATCH_SIZE for dim in model_shape):
        raise ValueError(
            f"{name}: model shape {model_shape} is smaller than CebraNET "
            f"patch {PATCH_SIZE}; the input may not have enough Z slices."
        )
    return {
        "dataset": str(name),
        "input_path": Path(path),
        "native_shape": tuple(int(v) for v in shape),
        "model_shape": model_shape,
        "realized_spacing_nm_zyx": realized_spacing_nm_zyx(shape, model_shape),
    }


def _paths_intersect(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def validate_output_location(output: Path, input_path: Path, weight_path: Path) -> None:
    """Reject source/weight overlap and unsafe output destinations."""
    raw_output = output.expanduser()
    if raw_output.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {raw_output}")
    output = raw_output.resolve()
    input_path = input_path.expanduser().resolve()
    weight_path = weight_path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    forbidden = (home, cwd, REPOSITORY_ROOT.resolve())
    if (output == Path(output.anchor)
            or any(output == p or output in p.parents for p in forbidden)):
        raise ValueError(f"Unsafe output directory: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output}")
    if _paths_intersect(output, input_path) or _paths_intersect(output, weight_path):
        raise ValueError("Output directory must not contain input TIFF or weights.")


def discover_requested_inputs(args, z_scale: float):
    """Return (records, parent output root, optional direct output directory)."""
    global INPUT_ROOT, WEIGHT_PATH
    WEIGHT_PATH = args.weights.expanduser().resolve()
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"CebraNET weights not found: {WEIGHT_PATH}")
    if args.input is not None:
        if args.input_root or args.output_root or args.datasets:
            raise ValueError("--input mode cannot be combined with multi-dataset options.")
        if args.output_dir is None:
            raise ValueError("--input mode requires --output-dir.")
        in_path = args.input.expanduser().resolve()
        shape, dtype = tiff_shape_dtype(in_path)
        name = in_path.parent.name if in_path.name == INPUT_FILENAME else in_path.stem
        rec = make_dataset_record(name, in_path, shape, dtype, z_scale)
        output_dir = args.output_dir.expanduser()
        validate_output_location(output_dir, in_path, WEIGHT_PATH)
        output_dir = output_dir.resolve()
        if output_dir.exists() and not args.overwrite and not args.check_only:
            raise FileExistsError(f"Output directory exists: {output_dir}; use a new path or --overwrite.")
        return [rec], output_dir.parent, output_dir

    if args.output_dir is not None:
        raise ValueError("--output-dir is valid only with --input.")
    if args.input_root is None or args.output_root is None:
        raise ValueError("Multi-dataset mode requires --input-root and --output-root.")
    INPUT_ROOT = args.input_root.expanduser().resolve()
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {INPUT_ROOT}")
    output_root = args.output_root.expanduser()
    if output_root.is_symlink():
        raise ValueError(f"Output root must not be a symlink: {output_root}")
    output_root = output_root.resolve()
    if args.datasets:
        names = list(args.datasets)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate dataset names in --datasets.")
    else:
        names = sorted(p.name for p in INPUT_ROOT.iterdir()
                       if p.is_dir() and (p / INPUT_FILENAME).is_file())
    if not names:
        raise ValueError(f"No {INPUT_FILENAME} datasets found in {INPUT_ROOT}")
    records = discover_dataset_records(names, z_scale)
    for record in records:
        out_dir = output_root / str(record["dataset"])
        validate_output_location(out_dir, record["input_path"], WEIGHT_PATH)
        if out_dir.exists() and not args.overwrite and not args.check_only:
            raise FileExistsError(f"Output dataset exists: {out_dir}; use --overwrite or another output root.")
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError(f"Refusing symlinked manifest: {manifest_path}")
    if manifest_path.exists() and not args.overwrite and not args.check_only:
        raise FileExistsError(f"Output manifest exists: {manifest_path}; use --overwrite or a new root.")
    return records, output_root, None


# =============================================================================
# 12. Per-dataset processing
# =============================================================================

def process_dataset(
    model,
    record: Dict,
    z_scale: float,
    workers: int,
    output_root: Path,
    blend_gpu: torch.Tensor,
    budget_bytes: int,
    device: torch.device,
    output_dir_override: Path | None = None,
    overwrite: bool = False,
    save_model_input: bool = False,
) -> Dict:
    dataset = str(record["dataset"])
    input_path = Path(record["input_path"])
    native_shape = tuple(
        int(v)
        for v in record["native_shape"]
    )
    model_shape = tuple(
        int(v)
        for v in record["model_shape"]
    )

    out_dir = (output_dir_override if output_dir_override is not None
               else output_root / dataset)
    validate_output_location(out_dir, input_path, WEIGHT_PATH)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        reset_dir(out_dir)
    else:
        ensure_dir(out_dir)
    scratch_dir = out_dir / ".scratch"

    print()
    print("=" * 100)
    print("DATASET:", dataset)
    print("=" * 100)
    print("Input:", input_path)
    print("Native shape ZYX:", native_shape)
    print("Requested Z scale:", z_scale)
    print("XY scale YX:", (XY_SCALE_Y, XY_SCALE_X))
    print("Model shape ZYX:", model_shape)
    print(
        "Realized spacing nm ZYX:",
        tuple(
            round(float(v), 6)
            for v in record["realized_spacing_nm_zyx"]
        ),
    )

    total_t0 = time.perf_counter()

    native_u8 = read_uint8_zyx(
        input_path
    )

    if tuple(native_u8.shape) != native_shape:
        raise RuntimeError(
            f"Loaded shape changed: {native_u8.shape} != {native_shape}"
        )

    prep_t0 = time.perf_counter()

    model_u8, pair_diagnostics, model_positions = (
        prepare_model_u8(
            native_u8=native_u8,
            z_scale=z_scale,
            workers=workers,
        )
    )

    prep_seconds = (
        time.perf_counter()
        - prep_t0
    )

    if tuple(model_u8.shape) != model_shape:
        raise RuntimeError(
            f"Model input shape mismatch: "
            f"{model_u8.shape} != {model_shape}"
        )

    del native_u8
    gc.collect()

    print(
        "Model-space preparation:",
        format_seconds(prep_seconds),
    )

    model_input_path = None

    if save_model_input or dataset == "tile_00":
        model_input_path = (
            out_dir
            / model_input_filename(z_scale)
        )

        tifffile.imwrite(
            str(model_input_path),
            model_u8,
            dtype=np.uint8,
            photometric="minisblack",
            bigtiff=True,
            metadata={"axes": "ZYX"},
        )

        print(
            "Saved model-space input:",
            model_input_path,
        )

    z_starts = axis_starts(
        model_shape[0],
        PATCH_SIZE,
        STRIDE,
    )
    y_starts = axis_starts(
        model_shape[1],
        PATCH_SIZE,
        STRIDE,
    )
    x_starts = axis_starts(
        model_shape[2],
        PATCH_SIZE,
        STRIDE,
    )

    unique_tiles = (
        len(z_starts)
        * len(y_starts)
        * len(x_starts)
    )

    needed_bytes = full_accumulator_bytes(
        model_shape
    )

    print(
        "Tile grid:",
        f"{len(z_starts)} x "
        f"{len(y_starts)} x "
        f"{len(x_starts)}",
    )
    print(
        "Unique tiles:",
        f"{unique_tiles:,}",
    )
    print(
        "Full accumulator requirement:",
        f"{gib(needed_bytes):.2f} GiB",
    )
    print(
        "Safe accumulator budget:",
        f"{gib(budget_bytes):.2f} GiB",
    )

    if needed_bytes <= budget_bytes:
        model_probability, infer_stats = (
            run_cuda_global_inference(
                model=model,
                model_u8=model_u8,
                blend_gpu=blend_gpu,
                z_starts=z_starts,
                y_starts=y_starts,
                x_starts=x_starts,
                device=device,
            )
        )
    else:
        model_probability, infer_stats = (
            run_cuda_absolute_zslab_inference(
                model=model,
                model_u8=model_u8,
                blend_gpu=blend_gpu,
                z_starts=z_starts,
                y_starts=y_starts,
                x_starts=x_starts,
                budget_bytes=budget_bytes,
                scratch_dir=scratch_dir,
                device=device,
            )
        )

    del model_u8
    gc.collect()

    probability_path = (
        out_dir
        / PROB_FILENAME
    )

    mask_path = (
        out_dir
        / MASK_FILENAME
    )

    restore_t0 = time.perf_counter()

    output_stats = (
        restore_and_save_native_outputs(
            model_probability=model_probability,
            model_positions=model_positions,
            native_shape=native_shape,
            probability_path=probability_path,
            mask_path=mask_path,
        )
    )

    restore_seconds = (
        time.perf_counter()
        - restore_t0
    )

    del model_probability
    gc.collect()

    if scratch_dir.exists():
        shutil.rmtree(
            scratch_dir
        )

    total_seconds = (
        time.perf_counter()
        - total_t0
    )

    summary = {
        "status": "complete",
        "dataset": dataset,
        "input_path": str(input_path),
        "model_path": str(WEIGHT_PATH),
        "native_shape_zyx": list(native_shape),
        "model_shape_zyx": list(model_shape),
        "source_spacing_nm_zyx": list(
            SOURCE_SPACING_NM_ZYX
        ),
        "requested_scale_zyx": [
            float(z_scale),
            float(XY_SCALE_Y),
            float(XY_SCALE_X),
        ],
        "realized_spacing_nm_zyx": [
            float(v)
            for v in record[
                "realized_spacing_nm_zyx"
            ]
        ],
        "model_z_positions_in_native_slice_units": [
            float(v)
            for v in model_positions.tolist()
        ],
        "motion_aware": {
            "enabled": bool(
                abs(z_scale - 1.0) > 1e-12
            ),
            "workers": int(workers),
            "displacement_multiplied_by_confidence": False,
            "confidence_source_switching": False,
            "fractional_synthesis": (
                "(1-t)*warp(A,t) + t*warp(B,1-t)"
            ),
            "z2_midpoint_is_exact_50_50": True,
            "xy_resize": "identity at XY scale 1.0 (cv2.INTER_LINEAR fallback)",
            "match_patch": MATCH_PATCH,
            "match_stride": MATCH_STRIDE,
            "max_shift_px": MAX_SHIFT_PX,
            "ncc_min": NCC_MIN,
            "ncc_improvement_min": NCC_IMPROVEMENT_MIN,
            "improvement_conf_scale": IMPROVEMENT_CONF_SCALE,
            "forward_backward_max_err_px": (
                FORWARD_BACKWARD_MAX_ERR_PX
            ),
            "field_smooth_sigma": FIELD_SMOOTH_SIGMA,
            "confidence_smooth_sigma": CONF_SMOOTH_SIGMA,
            "matching_gaussian_sigma": MATCH_GAUSSIAN_SIGMA,
            "pair_diagnostics": pair_diagnostics,
        },
        "cebranet": {
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "threshold": THRESHOLD,
            "blend_floor": BLEND_FLOOR,
            "weighted_overlap": True,
            "batch_size": BATCH_SIZE,
            "fp16_autocast_forward": USE_FP16_AUTOCAST,
            "fp32_accumulation": True,
            "blocking_h2d": True,
            "tf32": False,
            "backend_stats": infer_stats,
        },
        "native_recovery": {
            "z_mapping": (
                "physical-coordinate linear sampling from uniform model Z grid"
            ),
            "xy_resize": "identity at native XY size (cv2.INTER_LINEAR fallback)",
            "mask_encoding": "uint8 0/255",
            **output_stats,
        },
        "timing": {
            "model_space_preparation_seconds": float(
                prep_seconds
            ),
            "native_restore_seconds": float(
                restore_seconds
            ),
            "total_seconds": float(
                total_seconds
            ),
        },
        "saved_model_input": (
            None
            if model_input_path is None
            else str(model_input_path)
        ),
        "probability_output": str(
            probability_path
        ),
        "mask_output": str(
            mask_path
        ),
    }

    (
        out_dir
        / SUMMARY_FILENAME
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Native recovery:",
        format_seconds(restore_seconds),
    )
    print(
        "Positive fraction:",
        f"{output_stats['positive_fraction']:.6f}",
    )
    print(
        "Mask unique:",
        output_stats["mask_unique_values"],
    )
    print(
        "DONE:",
        dataset,
        "| total",
        format_seconds(total_seconds),
    )

    return summary


# =============================================================================
# 13. CPU geometry / preprocessing self-tests
# =============================================================================

def reconstruction_geometry_self_test() -> None:
    """
    CPU test: exact absolute-Z slab weighted reconstruction must agree with
    one global weighted reconstruction for deterministic synthetic predictions.
    """
    shape = (23, 20, 19)
    patch = 8
    stride = 4
    floor = 0.20

    def starts(length):
        ss = list(
            range(
                0,
                length - patch + 1,
                stride,
            )
        )
        last = length - patch
        if ss[-1] != last:
            ss.append(last)
        return ss

    zz = starts(shape[0])
    yy = starts(shape[1])
    xx = starts(shape[2])

    q = np.linspace(
        -1.0,
        1.0,
        patch,
        dtype=np.float32,
    )
    q = (
        floor
        + (1.0 - floor)
        * (1.0 - np.abs(q))
    )

    weight = (
        q[:, None, None]
        * q[None, :, None]
        * q[None, None, :]
    ).astype(np.float32)

    local_z = np.arange(
        patch,
        dtype=np.float32,
    )[:, None, None]
    local_y = np.arange(
        patch,
        dtype=np.float32,
    )[None, :, None]
    local_x = np.arange(
        patch,
        dtype=np.float32,
    )[None, None, :]

    def pred(z0, y0, x0):
        return (
            np.float32(z0 * 0.001)
            + np.float32(y0 * 0.002)
            + np.float32(x0 * 0.003)
            + local_z * np.float32(0.0001)
            + local_y * np.float32(0.0002)
            + local_x * np.float32(0.0003)
        ).astype(np.float32)

    global_sum = np.zeros(
        shape,
        dtype=np.float32,
    )
    global_weight = np.zeros(
        shape,
        dtype=np.float32,
    )

    for z0 in zz:
        for y0 in yy:
            for x0 in xx:
                sl = (
                    slice(z0, z0 + patch),
                    slice(y0, y0 + patch),
                    slice(x0, x0 + patch),
                )
                global_sum[sl] += (
                    pred(z0, y0, x0)
                    * weight
                )
                global_weight[sl] += weight

    if np.any(global_weight <= 0):
        raise RuntimeError(
            "Global self-test weight coverage failed."
        )

    reference = global_sum / global_weight

    slab_result = np.empty(
        shape,
        dtype=np.float32,
    )
    slab_depth = 9

    for slab_z0 in range(
        0,
        shape[0],
        slab_depth,
    ):
        slab_z1 = min(
            shape[0],
            slab_z0 + slab_depth,
        )

        depth = slab_z1 - slab_z0

        slab_sum = np.zeros(
            (
                depth,
                shape[1],
                shape[2],
            ),
            dtype=np.float32,
        )
        slab_weight = np.zeros_like(
            slab_sum
        )

        for patch_z0 in zz:
            if not (
                patch_z0 < slab_z1
                and patch_z0 + patch > slab_z0
            ):
                continue

            g0 = max(
                patch_z0,
                slab_z0,
            )
            g1 = min(
                patch_z0 + patch,
                slab_z1,
            )
            p0 = g0 - patch_z0
            p1 = g1 - patch_z0
            s0 = g0 - slab_z0
            s1 = g1 - slab_z0

            for y0 in yy:
                for x0 in xx:
                    sl = (
                        slice(s0, s1),
                        slice(y0, y0 + patch),
                        slice(x0, x0 + patch),
                    )

                    slab_sum[sl] += (
                        pred(
                            patch_z0,
                            y0,
                            x0,
                        )[p0:p1]
                        * weight[p0:p1]
                    )
                    slab_weight[sl] += (
                        weight[p0:p1]
                    )

        if np.any(slab_weight <= 0):
            raise RuntimeError(
                "Slab self-test weight coverage failed."
            )

        slab_result[
            slab_z0:slab_z1
        ] = slab_sum / slab_weight

    if not np.array_equal(
        reference,
        slab_result,
    ):
        max_abs = float(
            np.max(
                np.abs(
                    reference
                    - slab_result
                )
            )
        )
        raise RuntimeError(
            "Absolute-Z slab geometry self-test failed: "
            f"max_abs_diff={max_abs}"
        )


def preprocessing_self_test() -> None:
    """
    Synthetic moving-line test for Z=1 / 1.5 / 2 and Z=2 preserve contract.
    Temporarily shrinks matcher patch only inside this self-test.
    """
    global MATCH_PATCH, MATCH_STRIDE

    old_patch = MATCH_PATCH
    old_stride = MATCH_STRIDE

    MATCH_PATCH = 96
    MATCH_STRIDE = 48

    try:
        h = w = 192

        s0 = np.zeros(
            (h, w),
            dtype=np.uint8,
        )
        s1 = np.zeros_like(s0)
        s2 = np.zeros_like(s0)

        s0[:, 60] = 255
        s1[:, 64] = 255
        s2[:, 68] = 255

        native = np.stack(
            [s0, s1, s2],
            axis=0,
        )

        z1, diag1, pos1 = prepare_model_u8(
            native,
            z_scale=1.0,
            workers=2,
        )
        if z1.shape != (3, 192, 192):
            raise RuntimeError(
                f"Z1 self-test shape failed: {z1.shape}"
            )
        if len(diag1) != 0:
            raise RuntimeError(
                "Z1 should not run motion-aware matching."
            )

        z15, _, pos15 = prepare_model_u8(
            native,
            z_scale=1.5,
            workers=2,
        )
        if z15.shape[0] != 4:
            raise RuntimeError(
                f"Z1.5 self-test Z failed: {z15.shape}"
            )

        z2, _, pos2 = prepare_model_u8(
            native,
            z_scale=2.0,
            workers=2,
        )
        if z2.shape != (5, 192, 192):
            raise RuntimeError(
                f"Z2 self-test shape failed: {z2.shape}"
            )

        for i in range(3):
            expected = resize_xy_to_model(
                native[i],
                192,
                192,
            )
            if not np.array_equal(
                z2[2 * i],
                expected,
            ):
                raise RuntimeError(
                    f"Z2 preserve test failed at native {i}"
                )

        if not np.allclose(
            pos2,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0]
            ),
        ):
            raise RuntimeError(
                f"Unexpected Z2 positions: {pos2}"
            )

    finally:
        MATCH_PATCH = old_patch
        MATCH_STRIDE = old_stride


def mask_coding_self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        model_probability = np.zeros(
            (3, 8, 8),
            dtype=np.float32,
        )
        model_probability[:, 2:6, 2:6] = 0.8

        prob_path = td / "prob.tif"
        mask_path = td / "mask.tif"

        stats = restore_and_save_native_outputs(
            model_probability=model_probability,
            model_positions=np.array(
                [0.0, 1.0, 2.0],
                dtype=np.float64,
            ),
            native_shape=(3, 8, 8),
            probability_path=prob_path,
            mask_path=mask_path,
        )

        unique = np.unique(
            tifffile.imread(
                str(mask_path)
            )
        )

        if not np.array_equal(
            unique,
            np.array(
                [0, 255],
                dtype=np.uint8,
            ),
        ):
            raise RuntimeError(
                f"Mask coding self-test failed: {unique}"
            )


def run_cpu_selftests() -> None:
    reconstruction_geometry_self_test()
    preprocessing_self_test()
    mask_coding_self_test()
    print("CPU self-tests: PASS")


# =============================================================================
# 14. CLI / main
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inference-only, native-XY / motion-aware-Z CebraNET segmentation on CUDA."
    )
    mode = parser.add_argument_group("Input mode (select exactly one)")
    mode.add_argument("--input", type=Path, help="Single uint8 ZYX stacked TIFF.")
    mode.add_argument("--input-root", type=Path,
                      help="Multi-dataset directory containing <dataset>/output_stack.tif.")
    mode.add_argument("--datasets", nargs="+", default=None,
                      help="Optional multi-dataset folder names; default discovers all valid folders.")
    parser.add_argument("--weights", type=Path, default=WEIGHT_PATH,
                        help="Pretrained CebraNET TorchScript model.")
    parser.add_argument("--output-dir", type=Path,
                        help="Single-stack output directory (required with --input).")
    parser.add_argument("--output-root", type=Path,
                        help="Multi-dataset parent output directory (required with --input-root).")
    parser.add_argument("--z-scale", type=float, default=1.5,
                        choices=SUPPORTED_Z_SCALES,
                        help="Model-space Z scale (default: 1.5; native-grid mask output).")
    parser.add_argument("--workers", type=int,
                        default=min(4, max(1, os.cpu_count() or 1)),
                        help="CPU workers for Z interpolation (default: up to 4).")
    parser.add_argument("--source-spacing", nargs=3, type=float,
                        default=list(SOURCE_SPACING_NM_ZYX),
                        metavar=("Z_NM", "Y_NM", "X_NM"),
                        help="Input voxel spacing in nm Z Y X, used in metadata only (default 20 5 5).")
    parser.add_argument("--save-model-input", action="store_true",
                        help="Save the Z-resampled model input TIFF; tile_00 is saved by default as before.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace existing requested output directories.")
    parser.add_argument("--check-only", action="store_true",
                        help="Run CPU self-tests and input/weight/path checks without project output writes (self-tests use temporary files).")
    parser.add_argument("--selftest-only", action="store_true",
                        help="Run CPU synthetic self-tests without CUDA, inputs or weights.")
    return parser.parse_args(argv)


def main(argv=None):
    global SOURCE_SPACING_NM_ZYX
    args = parse_args(argv)
    run_cpu_selftests()
    if args.selftest_only:
        return
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if any(not math.isfinite(v) or v <= 0 for v in args.source_spacing):
        raise ValueError("--source-spacing values must be positive finite numbers")
    SOURCE_SPACING_NM_ZYX = tuple(float(v) for v in args.source_spacing)

    z_scale = float(args.z_scale)
    workers = int(args.workers)
    records, output_root, direct_output_dir = discover_requested_inputs(args, z_scale)

    print("=" * 88)
    print("CebraNET | native XY | motion-aware model Z | weighted 3-D overlap")
    print("Model-space Z scale:", z_scale)
    print("Source voxel spacing (nm, ZYX):", SOURCE_SPACING_NM_ZYX)
    print("Weights:", WEIGHT_PATH)
    print("Datasets:", [r["dataset"] for r in records])
    for rec in records:
        print(f"  {rec['dataset']}: native {rec['native_shape']} "
              f"-> model {rec['model_shape']} | restored mask = native shape")
    print("=" * 88)

    if args.check_only:
        print("Preflight OK (no project outputs): CPU tests, TIFF geometry/dtype, "
              "weights path, and output path checks passed. "
              "CUDA/model forward NOT tested.")
        return

    # Run CUDA validation before any output directory is created or removed.
    device = validate_environment()
    model = load_and_precheck_model(device)
    blend_np = make_blending_weight(PATCH_SIZE)
    blend_gpu = torch.from_numpy(blend_np).to(device=device, dtype=torch.float32)
    if bool(torch.any(blend_gpu <= 0).item()):
        raise RuntimeError("Blend weight must be strictly positive.")
    budget_bytes = accumulator_budget_bytes(device)

    results = []
    completed = []
    for record in records:
        try:
            summary = process_dataset(
                model=model, record=record, z_scale=z_scale, workers=workers,
                output_root=output_root, blend_gpu=blend_gpu,
                budget_bytes=budget_bytes, device=device,
                output_dir_override=direct_output_dir,
                overwrite=args.overwrite, save_model_input=args.save_model_input,
            )
            results.append(summary)
            completed.append(record["dataset"])
        except Exception:
            print("Segmentation failed. Completed datasets:", completed)
            raise

    manifest = {
        "status": "complete",
        "z_scale": z_scale,
        "xy_scale_yx": [XY_SCALE_Y, XY_SCALE_X],
        "output_root": str(output_root),
        "weights": str(WEIGHT_PATH),
        "datasets": [item["dataset"] for item in results],
        "results": [{
            "dataset": item["dataset"],
            "native_shape_zyx": item["native_shape_zyx"],
            "model_shape_zyx": item["model_shape_zyx"],
            "positive_fraction": item["native_recovery"]["positive_fraction"],
            "backend": item["cebranet"]["backend_stats"]["backend"],
        } for item in results],
    }
    manifest_path = ((direct_output_dir if direct_output_dir is not None
                      else output_root) / "run_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ALL REQUESTED DATASETS COMPLETE")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

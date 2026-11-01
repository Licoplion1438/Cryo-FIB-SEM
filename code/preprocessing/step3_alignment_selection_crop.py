#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cryo-FIB-SEM Preprocessing: Slice Selection, Alignment, and Cropping.

Author: Ningyan Zhang
Last modified: October 10, 2026

Description:
    Select a range of slices from a uint8 ZYX TIFF volume, manually exclude
    known bad slices, estimate global XY translations from multi-neighbor
    phase correlation, and crop every retained slice to the common overlap.

    The numerical registration, graph weighting, IRLS fitting, interpolation,
    and crop geometry are preserved from the original translation-only R4
    implementation. This revision changes only the stacked-TIFF input/output
    interface, configurable paths/selection, memory usage, and documentation.

    The code performs NO intensity matching, VSNR, nonrigid registration,
    rotation, image resizing, or automatic bad-slice removal.

Inputs:
    --input: uint8 TIFF stack with axis order (Z, Y, X).
    --start-index / --end-index: ORIGINAL 0-based inclusive Z indices.
    --remove-slices: comma-separated ORIGINAL 1-based Z numbers, excluded
        BEFORE constructing registration images or alignment graph edges.

Outputs (in --output-dir):
    aligned_stack.tif: uint8 ZYX aligned volume, cropped to common overlap.
    before_stack.tif: uint8 ZYX selected input slices before alignment.
    alignment_canvas_stack.tif: uint8 ZYX uncropped alignment QC canvas.
    qc_pairs.csv: measured pairwise translations and confidence values.
    transforms.csv: final per-slice global XY translations.
    removed_slices.txt: selection and removal audit trail.
    summary.json: geometry, parameters, and registration summary.

Dependencies:
    Python 3.8+, numpy, opencv-python, tifffile, tqdm.

Usage (from repository root):
    python code/preprocessing/step3_alignment_selection_crop.py \
        --input /path/to/input_uint8_stacked.tif

    python code/preprocessing/step3_alignment_selection_crop.py \
        --input /path/to/input_uint8_stacked.tif \
        --output-dir /path/to/alignment_results \
        --start-index 0 --end-index 75 \
        --remove-slices 35

Notes:
    - A slice's original Z index never changes when earlier slices are removed.
    - The reference frame is the middle RETAINED slice.
    - Linear interpolation may change pixel intensities on subpixel shifts;
      output is still uint8, without additional contrast normalization.
    - This script is independent of Stage-1, Stage-2, and optional VSNR.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


# ============================================================
# 1. Original R4 algorithm parameters (unchanged)
# ============================================================

GRAPH_GAPS = (1, 2, 4)
REG_SCALES = (0.25, 0.5, 1.0)
REG_PERCENTILE_LOW = 1.0
REG_PERCENTILE_HIGH = 99.0
REG_FINE_BLUR_SIGMA = 1.0
REG_BACKGROUND_SIGMA = 10.0
REG_HIGHPASS_WEIGHT = 0.65
REG_GRADIENT_WEIGHT = 0.35
REG_BORDER_FRACTION = 0.03
REG_MIN_ROI_SIZE = 128
MIN_PAIR_RESPONSE = 0.03
GOOD_PAIR_RESPONSE = 0.10
IRLS_ITERS = 8
HUBER_K = 2.5
MIN_ROBUST_SCALE_PX = 0.5
GOOD_EDGE_RESIDUAL_PX = 2.5
MAX_REASONABLE_SHIFT_FRACTION = 0.12
AUTO_REMOVE_ISOLATED_BAD = False
MAX_AUTO_REMOVED_SLICES = 0
IMAGE_INTERPOLATION = cv2.INTER_LINEAR
ALIGNMENT_CANVAS_MARGIN_PX = 32

BEFORE_STACK_NAME = "before_stack.tif"
ALIGNED_STACK_NAME = "aligned_stack.tif"
ALIGNMENT_CANVAS_STACK_NAME = "alignment_canvas_stack.tif"
ALGORITHM_NAME = "R4 translation-only multi-neighbor graph alignment"


# ============================================================
# 2. CLI configuration, index parsing, and output validation
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Expose file paths and manual Z selection without editing source code."""
    parser = argparse.ArgumentParser(
        description="Select, translation-align, and crop a uint8 ZYX TIFF stack."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input ZYX uint8 TIFF stack.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: INPUT_STEM + '_aligned' beside the input).",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="First original Z index; 0-based and inclusive (default: 0).",
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Last original Z index; 0-based and inclusive (default: final slice).",
    )
    parser.add_argument(
        "--remove-slices", type=str, default="",
        help="Comma-separated ORIGINAL 1-based slice numbers to exclude, e.g. 5,77,78.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace this script's existing output files; preserve unrelated files.",
    )
    return parser


def parse_removed_slices(value: str) -> list[int]:
    """Parse manual exclusions as distinct positive, original 1-based indices."""
    if not value.strip():
        return []
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--remove-slices must contain comma-separated integers.") from exc
    if any(number < 1 for number in parsed):
        raise ValueError("--remove-slices values must be 1-based positive integers.")
    return sorted(set(parsed))


def validate_selection(
    z_count: int, start: int, end: int | None, removed_1based: list[int]
) -> tuple[int, list[int], list[int]]:
    """Validate selection while preserving original Z indices in all records."""
    resolved_end = z_count - 1 if end is None else end
    if not (0 <= start <= resolved_end < z_count):
        raise ValueError(
            f"Invalid slice interval [{start}, {resolved_end}] for Z={z_count}; "
            "start/end are original 0-based inclusive indices."
        )
    if any(item > z_count for item in removed_1based):
        raise ValueError(f"Manual exclusion outside original stack of Z={z_count}.")
    selected = list(range(start, resolved_end + 1))
    removed_in_range = {number - 1 for number in removed_1based if start <= number - 1 <= resolved_end}
    retained = [idx for idx in selected if idx not in removed_in_range]
    if len(retained) < 3:
        raise ValueError("At least three slices must remain after selection/removal.")
    return resolved_end, selected, retained


# ============================================================
# 3. Stacked-TIFF input and memory-aware output utilities
# ============================================================


@contextmanager
def open_uint8_zyx_stack(path: Path):
    """Read an 8-bit ZYX TIFF stack without needlessly loading it into RAM.

    Uncompressed TIFFs are mapped directly. Compressed TIFFs are decoded into
    a temporary file-backed NumPy array, avoiding full-volume resident memory.
    The temporary copy is removed when processing finishes.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input TIFF does not exist: {path}")
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected a TIFF input, got: {path}")

    with tifffile.TiffFile(str(path)) as tf:
        if not tf.series:
            raise ValueError(f"TIFF contains no image series: {path}")
        series = tf.series[0]
        shape = tuple(int(d) for d in series.shape)
        axes = series.axes
        dtype = np.dtype(series.dtype)
        if len(shape) != 3 or shape[0] < 3 or min(shape[1:]) < 1:
            raise ValueError(f"Expected a ZYX volume with at least 3 slices, got shape={shape}.")
        if axes[-2:] != "YX" or axes[0] in "SC":
            raise ValueError(f"Expected grayscale ZYX TIFF, got axes={axes!r}.")
        if dtype != np.dtype("uint8"):
            raise ValueError(
                f"Input must already be uint8 (use preprocessing Step-1), got {dtype}. "
                "This script does not change the intensity mapping."
            )

    # The first path is zero-copy; only switch to disk-backed decompression
    # when tifffile reports that direct memory mapping is not supported.
    try:
        volume = tifffile.memmap(str(path), mode="r")
    except (ValueError, OSError):
        with tempfile.TemporaryDirectory(prefix="cryo_fib_sem_tiff_") as tmp_dir:
            buffer_file = str(Path(tmp_dir) / "decoded_volume.dat")
            volume = tifffile.imread(str(path), out=buffer_file)
            if volume.shape != shape or volume.dtype != np.uint8:
                raise ValueError("Decoded TIFF does not match the declared uint8 ZYX series.")
            try:
                yield volume
            finally:
                del volume
    else:
        if volume.shape != shape:
            raise ValueError(f"Memmap shape mismatch: {volume.shape} != {shape}")
        try:
            yield volume
        finally:
            del volume


def new_stack(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    """Create an uncompressed uint8 ZYX TIFF and return its writable mapping."""
    total_bytes = int(np.prod(shape, dtype=np.int64))
    return tifffile.memmap(
        str(path), shape=shape, dtype=np.uint8,
        photometric="minisblack", metadata={"axes": "ZYX"},
        bigtiff=(total_bytes >= 4_000_000_000),
    )


def prepare_output_dir(folder: Path, names: list[str], overwrite: bool) -> None:
    """Refuse accidental overwrites and never recursively erase user folders."""
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"Output path is not a directory: {folder}")
    existing = [folder / name for name in names if (folder / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing output files detected; choose another --output-dir "
            "or explicitly pass --overwrite: " + ", ".join(str(p) for p in existing)
        )
    folder.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if not path.is_file():
            raise ValueError(f"Cannot overwrite a non-file output path: {path}")
        path.unlink()


def write_json(path: Path, value: dict) -> None:
    """Save deterministic, human-readable run metadata for reproducibility."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


# ============================================================
# 4. Registration features and coarse-to-fine phase correlation
# ============================================================

def robust_normalize_float(image):
    """Build the original percentile-clipped registration intensity channel."""
    x = image.astype(np.float32)

    lo, hi = np.percentile(
        x,
        [REG_PERCENTILE_LOW, REG_PERCENTILE_HIGH],
    )

    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)

    return x.astype(np.float32, copy=False)


def standardize_channel(x):
    """Standardize a registration channel using median and robust dispersion."""
    x = x.astype(np.float32, copy=False)

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))

    scale = max(
        1.4826 * mad,
        float(np.std(x)),
        1e-4,
    )

    return ((x - median) / scale).astype(
        np.float32,
        copy=False,
    )


def make_registration_image(image_uint8):
    """
    Build a structure-focused image.

    High-pass intensity keeps broad membrane / organelle structure.
    Gradient magnitude emphasizes stable boundaries and suppresses absolute
    brightness differences left by 2D artefact removal.
    """
    x = robust_normalize_float(image_uint8)

    fine = cv2.GaussianBlur(
        x,
        (0, 0),
        REG_FINE_BLUR_SIGMA,
    )

    background = cv2.GaussianBlur(
        fine,
        (0, 0),
        REG_BACKGROUND_SIGMA,
    )

    highpass = standardize_channel(
        fine - background
    )

    gx = cv2.Sobel(
        fine,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        fine,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = standardize_channel(
        cv2.magnitude(gx, gy)
    )

    reg = (
        REG_HIGHPASS_WEIGHT * highpass
        + REG_GRADIENT_WEIGHT * gradient
    )

    # Remove a small border from the signal itself. The later phase-correlation
    # ROI also crops, but this reduces strong frame-edge structure.
    h, w = reg.shape
    by = int(round(h * REG_BORDER_FRACTION))
    bx = int(round(w * REG_BORDER_FRACTION))

    if by > 0:
        reg[:by, :] = 0
        reg[-by:, :] = 0

    if bx > 0:
        reg[:, :bx] = 0
        reg[:, -bx:] = 0

    reg -= float(np.mean(reg))

    return reg.astype(np.float32, copy=False)


def resize_registration_image(image, scale):
    """Construct the original phase-correlation pyramid image."""
    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32, copy=False)


def warp_translation_float(image, dx, dy):
    """Apply the original single-image XY translation for registration."""
    h, w = image.shape

    matrix = np.float32(
        [
            [1.0, 0.0, float(dx)],
            [0.0, 1.0, float(dy)],
        ]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def valid_phase_roi(shape, dx, dy):
    """
    After moving has already been warped by (dx,dy), exclude the invalid
    border plus a small safety margin.
    """
    h, w = shape

    safety = max(
        8,
        int(round(min(h, w) * 0.02)),
    )

    left = safety + max(0, int(math.ceil(dx)))
    right = safety + max(0, int(math.ceil(-dx)))
    top = safety + max(0, int(math.ceil(dy)))
    bottom = safety + max(0, int(math.ceil(-dy)))

    x0 = left
    x1 = w - right
    y0 = top
    y1 = h - bottom

    if (
        x1 - x0 < REG_MIN_ROI_SIZE
        or y1 - y0 < REG_MIN_ROI_SIZE
    ):
        return 0, w, 0, h

    return x0, x1, y0, y1


def phase_correlate_pair(reference, moving, scale, current_dx, current_dy):
    """Estimate one pairwise translation correction at a pyramid scale."""
    ref_s = resize_registration_image(
        reference,
        scale,
    )
    mov_s = resize_registration_image(
        moving,
        scale,
    )

    dx_s = current_dx * scale
    dy_s = current_dy * scale

    moved_s = warp_translation_float(
        mov_s,
        dx_s,
        dy_s,
    )

    x0, x1, y0, y1 = valid_phase_roi(
        ref_s.shape,
        dx_s,
        dy_s,
    )

    ref_roi = np.ascontiguousarray(
        ref_s[y0:y1, x0:x1],
        dtype=np.float32,
    )
    mov_roi = np.ascontiguousarray(
        moved_s[y0:y1, x0:x1],
        dtype=np.float32,
    )

    h, w = ref_roi.shape

    if min(h, w) < 32:
        return 0.0, 0.0, 0.0

    window = cv2.createHanningWindow(
        (w, h),
        cv2.CV_32F,
    )

    # phaseCorrelate(reference, moving) returns the displacement of moving
    # relative to reference. Therefore the correction TO APPLY to moving is
    # the negative of this residual.
    (shift_x, shift_y), response = cv2.phaseCorrelate(
        ref_roi,
        mov_roi,
        window,
    )

    correction_dx = -float(shift_x) / scale
    correction_dy = -float(shift_y) / scale

    return (
        correction_dx,
        correction_dy,
        float(response),
    )


def estimate_pair_translation(reference_reg, moving_reg):
    """Accumulate original multi-resolution pairwise displacement estimates."""
    dx = 0.0
    dy = 0.0

    responses = []

    for scale in REG_SCALES:
        residual_dx, residual_dy, response = (
            phase_correlate_pair(
                reference_reg,
                moving_reg,
                scale,
                dx,
                dy,
            )
        )

        dx += residual_dx
        dy += residual_dy
        responses.append(response)

    response = float(
        np.median(
            np.asarray(
                responses,
                dtype=np.float64,
            )
        )
    )

    magnitude = float(
        math.hypot(dx, dy)
    )

    max_reasonable = (
        MAX_REASONABLE_SHIFT_FRACTION
        * min(reference_reg.shape)
    )

    unreasonable = (
        not np.isfinite(dx)
        or not np.isfinite(dy)
        or magnitude > max_reasonable
    )

    if unreasonable:
        response *= 0.05

    return {
        "dx": float(dx),
        "dy": float(dy),
        "magnitude": magnitude,
        "response": response,
        "scale_responses": responses,
        "unreasonable_shift": bool(unreasonable),
    }


# ============================================================
# 5. Multi-neighbor graph and robust translation solution
# ============================================================

def measure_graph_edges(items, reg_images):
    """Build original gap-1/2/4 graph edges and confidence weights."""
    edges = []

    n = len(items)

    total = sum(
        max(0, n - gap)
        for gap in GRAPH_GAPS
    )

    with tqdm(
        total=total,
        desc="Pairwise phase correlation",
        dynamic_ncols=True,
    ) as pbar:
        for gap in GRAPH_GAPS:
            for i in range(n - gap):
                j = i + gap

                info = estimate_pair_translation(
                    reg_images[i],
                    reg_images[j],
                )

                # Equation for global correction:
                # G[j] - G[i] = measured correction to apply to j relative to i.
                response = max(
                    float(info["response"]),
                    0.0,
                )

                base_weight = (
                    max(response, 0.01) ** 2
                    / float(gap)
                )

                if response < MIN_PAIR_RESPONSE:
                    base_weight *= 0.15

                if info["unreasonable_shift"]:
                    base_weight *= 0.05

                edges.append(
                    {
                        "i": i,
                        "j": j,
                        "original_i": items[i]["original_index0"],
                        "original_j": items[j]["original_index0"],
                        "gap": gap,
                        "dx": info["dx"],
                        "dy": info["dy"],
                        "magnitude": info["magnitude"],
                        "response": response,
                        "base_weight": float(base_weight),
                        "unreasonable_shift":
                            info["unreasonable_shift"],
                    }
                )

                pbar.update(1)

    return edges


def solve_weighted_coordinate(
    num_slices,
    edges,
    values,
    weights,
    reference_index,
):
    """Solve one weighted graph coordinate anchored at the middle slice."""
    rows = []
    rhs = []
    row_weights = []

    for edge, value, weight in zip(
        edges,
        values,
        weights,
    ):
        row = np.zeros(
            num_slices,
            dtype=np.float64,
        )

        row[edge["i"]] = -1.0
        row[edge["j"]] = 1.0

        rows.append(row)
        rhs.append(float(value))
        row_weights.append(
            max(float(weight), 1e-8)
        )

    # Anchor the middle retained slice.
    anchor = np.zeros(
        num_slices,
        dtype=np.float64,
    )
    anchor[reference_index] = 1.0

    rows.append(anchor)
    rhs.append(0.0)
    row_weights.append(100.0)

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    w = np.sqrt(
        np.asarray(
            row_weights,
            dtype=np.float64,
        )
    )

    Aw = A * w[:, None]
    bw = b * w

    solution, *_ = np.linalg.lstsq(
        Aw,
        bw,
        rcond=None,
    )

    return solution


def edge_residuals(edges, global_x, global_y):
    """Return Euclidean translation residuals for each graph edge."""
    residuals = np.empty(
        len(edges),
        dtype=np.float64,
    )

    for k, edge in enumerate(edges):
        pred_dx = (
            global_x[edge["j"]]
            - global_x[edge["i"]]
        )
        pred_dy = (
            global_y[edge["j"]]
            - global_y[edge["i"]]
        )

        residuals[k] = math.hypot(
            pred_dx - edge["dx"],
            pred_dy - edge["dy"],
        )

    return residuals


def robust_graph_solution(num_slices, edges):
    """Run original IRLS graph fitting and compute per-edge QC flags."""
    if num_slices < 2:
        raise ValueError(
            "Need at least two retained slices."
        )

    if not edges:
        raise RuntimeError(
            "No pairwise registration edges were measured."
        )

    reference_index = num_slices // 2

    base_weights = np.asarray(
        [
            edge["base_weight"]
            for edge in edges
        ],
        dtype=np.float64,
    )

    weights = base_weights.copy()

    dx_values = np.asarray(
        [
            edge["dx"]
            for edge in edges
        ],
        dtype=np.float64,
    )

    dy_values = np.asarray(
        [
            edge["dy"]
            for edge in edges
        ],
        dtype=np.float64,
    )

    for _ in range(IRLS_ITERS):
        global_x = solve_weighted_coordinate(
            num_slices,
            edges,
            dx_values,
            weights,
            reference_index,
        )

        global_y = solve_weighted_coordinate(
            num_slices,
            edges,
            dy_values,
            weights,
            reference_index,
        )

        residual = edge_residuals(
            edges,
            global_x,
            global_y,
        )

        median = float(
            np.median(residual)
        )

        mad = float(
            np.median(
                np.abs(
                    residual - median
                )
            )
        )

        robust_scale = max(
            1.4826 * mad,
            MIN_ROBUST_SCALE_PX,
        )

        cutoff = HUBER_K * robust_scale

        huber = np.ones_like(residual)

        large = residual > cutoff

        huber[large] = (
            cutoff
            / np.maximum(
                residual[large],
                1e-8,
            )
        )

        weights = (
            base_weights
            * huber
        )

    residual = edge_residuals(
        edges,
        global_x,
        global_y,
    )

    median = float(
        np.median(residual)
    )

    mad = float(
        np.median(
            np.abs(
                residual - median
            )
        )
    )

    robust_scale = max(
        1.4826 * mad,
        MIN_ROBUST_SCALE_PX,
    )

    good_residual_threshold = max(
        GOOD_EDGE_RESIDUAL_PX,
        median + 4.0 * robust_scale,
    )

    for k, edge in enumerate(edges):
        edge["final_weight"] = float(
            weights[k]
        )
        edge["graph_residual_px"] = float(
            residual[k]
        )

        edge["good"] = bool(
            edge["response"] >= GOOD_PAIR_RESPONSE
            and residual[k] <= good_residual_threshold
            and not edge["unreasonable_shift"]
        )

    return {
        "reference_index":
            int(reference_index),
        "global_x":
            global_x,
        "global_y":
            global_y,
        "median_edge_residual_px":
            median,
        "robust_edge_scale_px":
            robust_scale,
        "good_edge_residual_threshold_px":
            float(good_residual_threshold),
    }


# ============================================================
# 6. Conservative bad-slice detection (disabled in the original R4 method)
# ============================================================

def detect_isolated_bad_slices(
    items,
    edges,
):
    """
    Remove only an interior slice k when:
    - edge (k-1,k) is bad,
    - edge (k,k+1) is bad,
    - bridge edge (k-1,k+1) is good.

    This is intentionally much more conservative than treating every weak pair
    as evidence that a slice itself is corrupted.
    """
    if not AUTO_REMOVE_ISOLATED_BAD:
        return {}

    lookup = {
        (edge["i"], edge["j"]):
            edge
        for edge in edges
    }

    removed = {}

    for k in range(1, len(items) - 1):
        left = lookup.get(
            (k - 1, k)
        )
        right = lookup.get(
            (k, k + 1)
        )
        bridge = lookup.get(
            (k - 1, k + 1)
        )

        if (
            left is None
            or right is None
            or bridge is None
        ):
            continue

        if (
            not left["good"]
            and not right["good"]
            and bridge["good"]
        ):
            removed[k] = (
                "isolated registration failure: "
                "both adjacent edges are inconsistent, "
                "but the surrounding slices register reliably"
            )

            if len(removed) >= MAX_AUTO_REMOVED_SLICES:
                break

    return removed


# ============================================================
# 7. Translation-only output geometry
# ============================================================

def translated_axis_bounds(length, shift):
    """Bounds of one image axis after translation only."""
    a = float(shift)
    b = float(length) + float(shift)
    return min(a, b), max(a, b)


def compute_common_overlap(
    image_shape,
    global_x,
    global_y,
):
    """Intersection that is valid in every translation-aligned slice."""
    height, width = image_shape

    x_bounds = [
        translated_axis_bounds(width, global_x[k])
        for k in range(len(global_x))
    ]
    y_bounds = [
        translated_axis_bounds(height, global_y[k])
        for k in range(len(global_y))
    ]

    left = max(bound[0] for bound in x_bounds)
    right = min(bound[1] for bound in x_bounds)
    top = max(bound[0] for bound in y_bounds)
    bottom = min(bound[1] for bound in y_bounds)

    crop_left = int(math.ceil(left))
    crop_top = int(math.ceil(top))
    crop_right = int(math.floor(right))
    crop_bottom = int(math.floor(bottom))

    out_w = crop_right - crop_left
    out_h = crop_bottom - crop_top

    if out_w <= 0 or out_h <= 0:
        raise RuntimeError(
            "No common valid overlap after translation-only alignment."
        )

    return {
        "output_shape_yx": [out_h, out_w],
        "origin_x_global": float(crop_left),
        "origin_y_global": float(crop_top),
    }


def compute_union_canvas(
    image_shape,
    global_x,
    global_y,
):
    """Union canvas for QC after translation only."""
    height, width = image_shape
    margin = int(ALIGNMENT_CANVAS_MARGIN_PX)

    x_bounds = [
        translated_axis_bounds(width, global_x[k])
        for k in range(len(global_x))
    ]
    y_bounds = [
        translated_axis_bounds(height, global_y[k])
        for k in range(len(global_y))
    ]

    origin_x = int(math.floor(min(bound[0] for bound in x_bounds))) - margin
    origin_y = int(math.floor(min(bound[0] for bound in y_bounds))) - margin
    right = int(math.ceil(max(bound[1] for bound in x_bounds))) + margin
    bottom = int(math.ceil(max(bound[1] for bound in y_bounds))) + margin

    return {
        "output_shape_yx": [bottom - origin_y, right - origin_x],
        "origin_x_global": float(origin_x),
        "origin_y_global": float(origin_y),
    }


def warp_to_geometry(
    image,
    global_dx,
    global_dy,
    geometry,
    border_value,
):
    """Apply translation only and write directly into requested geometry."""
    out_h, out_w = geometry["output_shape_yx"]

    tx = float(global_dx) - geometry["origin_x_global"]
    ty = float(global_dy) - geometry["origin_y_global"]

    matrix = np.float32(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
        ]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (int(out_w), int(out_h)),
        flags=IMAGE_INTERPOLATION,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


# ============================================================
# 8. Pairwise and per-slice QC CSV files
# ============================================================

def save_pair_csv(path, items, edges):
    """Write original graph-edge measurements and robust-fitting QC results."""
    fields = [
        "i",
        "j",
        "original_index_i0",
        "original_index_j0",
        "filename_i",
        "filename_j",
        "gap",
        "dx",
        "dy",
        "magnitude",
        "response",
        "base_weight",
        "final_weight",
        "graph_residual_px",
        "good",
        "unreasonable_shift",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for edge in edges:
            writer.writerow(
                {
                    "i": edge["i"],
                    "j": edge["j"],
                    "original_index_i0": edge["original_i"],
                    "original_index_j0": edge["original_j"],
                    "filename_i": items[edge["i"]]["path"].name,
                    "filename_j": items[edge["j"]]["path"].name,
                    "gap": edge["gap"],
                    "dx": edge["dx"],
                    "dy": edge["dy"],
                    "magnitude": edge["magnitude"],
                    "response": edge["response"],
                    "base_weight": edge["base_weight"],
                    "final_weight": edge.get(
                        "final_weight",
                        "",
                    ),
                    "graph_residual_px": edge.get(
                        "graph_residual_px",
                        "",
                    ),
                    "good": edge.get(
                        "good",
                        "",
                    ),
                    "unreasonable_shift": edge["unreasonable_shift"],
                }
            )


def save_transform_csv(
    path,
    items,
    solution,
):
    """Write the global XY translation applied to each retained slice."""
    fields = [
        "retained_index0",
        "original_index0",
        "filename",
        "is_reference",
        "dx_global",
        "dy_global",
    ]

    reference_index = solution["reference_index"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for k, item in enumerate(items):
            writer.writerow(
                {
                    "retained_index0": k,
                    "original_index0": item["original_index0"],
                    "filename": item["path"].name,
                    "is_reference": (k == reference_index),
                    "dx_global": float(solution["global_x"][k]),
                    "dy_global": float(solution["global_y"][k]),
                }
            )


# ============================================================
# 9. Selection audit and stacked-volume processing
# ============================================================


def save_removed_slices(
    path: Path, input_path: Path, z_count: int, start: int,
    resolved_end: int, selected: list[int], retained: list[int],
    removed_1based: list[int],
) -> None:
    """Record manual removal against ORIGINAL slice numbers and Z indices."""
    retained_set = set(retained)
    removed_in_range = [index for index in selected if index not in retained_set]
    outside = [number for number in removed_1based if not (start <= number - 1 <= resolved_end)]
    lines = [
        f"Input: {input_path}",
        "Original Z indices: 0-based; manual exclusion list: 1-based",
        f"Source slice count: {z_count}",
        f"Selected range: {start}..{resolved_end} (inclusive)",
        f"Selected slice count before manual removal: {len(selected)}",
        f"Manually removed before alignment: {len(removed_in_range)}",
        f"Alignment input slice count: {len(retained)}",
        "",
        "Manually removed slices BEFORE alignment:",
    ]
    if removed_in_range:
        lines.extend(
            f"- original slice {index + 1} (1-based) -> original Z index {index} (0-based)"
            for index in removed_in_range
        )
    else:
        lines.append("NONE")
    lines.extend([
        "",
        "Requested manual removals outside selected range (1-based):",
        ", ".join(str(number) for number in outside) if outside else "NONE",
        "",
        "Policy: select range -> manually exclude bad slices -> XY translation-only alignment -> common-overlap crop.",
        "No automatic bad-slice removal, image scaling, rotation, or nonrigid correction.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_registration_images_from_stack(
    volume: np.ndarray, items: list[dict]
) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Generate the ORIGINAL registration representation only for kept slices."""
    shape = tuple(int(value) for value in volume.shape[1:])
    registrations = []
    for item in tqdm(items, desc="Preparing registration images", dynamic_ncols=True):
        image = np.asarray(volume[item["original_index0"]])
        if image.shape != shape:
            raise ValueError(f"Unexpected slice shape: {image.shape} != {shape}")
        registrations.append(make_registration_image(image))
    return shape, registrations


def validate_output_stack(path: Path, expected_shape: tuple[int, int, int]) -> None:
    """Verify that a finished output is an uint8 stacked TIFF of expected ZYX size."""
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        if tuple(series.shape) != tuple(expected_shape) or np.dtype(series.dtype) != np.uint8:
            raise RuntimeError(
                f"Stack verification failed for {path}: "
                f"shape={series.shape}, dtype={series.dtype}, expected={expected_shape}, uint8"
            )


def run_alignment(args: argparse.Namespace) -> dict:
    """Execute original R4 selection/alignment/crop with stacked-TIFF I/O."""
    input_path = args.input.expanduser().resolve()
    removed_1based = parse_removed_slices(args.remove_slices)
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None
        else input_path.with_name(input_path.stem + "_aligned")
    )
    # Report names remain compatible with the original R4 output folder.
    artifact_names = [
        BEFORE_STACK_NAME, ALIGNED_STACK_NAME, ALIGNMENT_CANVAS_STACK_NAME,
        "qc_pairs.csv", "transforms.csv", "removed_slices.txt", "summary.json",
    ]
    with open_uint8_zyx_stack(input_path) as volume:
        z_count, height, width = (int(value) for value in volume.shape)
        resolved_end, selected, retained = validate_selection(
            z_count, args.start_index, args.end_index, removed_1based
        )
        items = [
            {"original_index0": index, "path": Path(f"slice_{index:04d}.tif")}
            for index in retained
        ]
        print("=" * 76)
        print(ALGORITHM_NAME)
        print("=" * 76)
        print("Input:", input_path)
        print("Input ZYX shape:", tuple(volume.shape))
        print("Selected range (original 0-based):", args.start_index, resolved_end)
        print("Requested exclusions (original 1-based):", removed_1based)
        print("Retained slices:", len(retained))
        print("Output directory:", output_dir)

        # Selection and manual bad-slice exclusion happen before ANY graph edge.
        input_shape, reg_images = load_registration_images_from_stack(volume, items)
        edges = measure_graph_edges(items, reg_images)
        solution = robust_graph_solution(len(items), edges)
        del reg_images

        crop_geometry = compute_common_overlap(
            input_shape, solution["global_x"], solution["global_y"]
        )
        canvas_geometry = compute_union_canvas(
            input_shape, solution["global_x"], solution["global_y"]
        )
        cropped_y, cropped_x = crop_geometry["output_shape_yx"]
        canvas_y, canvas_x = canvas_geometry["output_shape_yx"]
        out_shape = (len(retained), cropped_y, cropped_x)
        canvas_shape = (len(retained), canvas_y, canvas_x)
        before_shape = (len(retained), height, width)

        prepare_output_dir(output_dir, artifact_names, args.overwrite)
        before_path = output_dir / BEFORE_STACK_NAME
        aligned_path = output_dir / ALIGNED_STACK_NAME
        canvas_path = output_dir / ALIGNMENT_CANVAS_STACK_NAME
        before = new_stack(before_path, before_shape)
        aligned = new_stack(aligned_path, out_shape)
        canvas = new_stack(canvas_path, canvas_shape)
        try:
            # Read each source plane once and warp it once per requested output.
            for k, item in enumerate(
                tqdm(items, desc="Final align + common-overlap crop", dynamic_ncols=True)
            ):
                original = np.asarray(volume[item["original_index0"]])
                before[k] = original
                dx = solution["global_x"][k]
                dy = solution["global_y"][k]
                aligned[k] = warp_to_geometry(
                    original, dx, dy, crop_geometry, border_value=0
                ).astype(np.uint8, copy=False)
                canvas[k] = warp_to_geometry(
                    original, dx, dy, canvas_geometry, border_value=255
                ).astype(np.uint8, copy=False)
                if k == len(items) - 1 or (k + 1) % 16 == 0:
                    before.flush()
                    aligned.flush()
                    canvas.flush()
            before.flush()
            aligned.flush()
            canvas.flush()
        finally:
            del before, aligned, canvas

        save_pair_csv(output_dir / "qc_pairs.csv", items, edges)
        save_transform_csv(output_dir / "transforms.csv", items, solution)
        save_removed_slices(
            output_dir / "removed_slices.txt", input_path, z_count,
            args.start_index, resolved_end, selected, retained, removed_1based,
        )
        good_edges = sum(int(edge["good"]) for edge in edges)
        summary = {
            "algorithm": ALGORITHM_NAME,
            "input_file": str(input_path),
            "output_dir": str(output_dir),
            "input_shape_zyx": [z_count, height, width],
            "output_shape_zyx": list(out_shape),
            "pre_alignment_shape_zyx": list(before_shape),
            "alignment_canvas_shape_zyx": list(canvas_shape),
            "input_dtype": "uint8",
            "output_dtype": "uint8",
            "start_slice_index0": args.start_index,
            "end_slice_index0_inclusive": resolved_end,
            "manual_remove_requested_1based": removed_1based,
            "manual_removed_before_alignment_index0": sorted(set(selected) - set(retained)),
            "retained_original_indices0": retained,
            "retained_count": len(retained),
            "reference_retained_index0": int(solution["reference_index"]),
            "graph_gaps": list(GRAPH_GAPS),
            "registration_scales": list(REG_SCALES),
            "median_graph_residual_px": float(solution["median_edge_residual_px"]),
            "good_edges": good_edges,
            "total_edges": len(edges),
            "transform_policy": "translation_only",
            "interpolation": "cv2.INTER_LINEAR",
            "intensity_matching": False,
            "automatic_bad_slice_removal": False,
            "aligned_stack": str(aligned_path),
            "before_stack": str(before_path),
            "alignment_canvas_stack": str(canvas_path),
        }
        write_json(output_dir / "summary.json", summary)
        for name, shape in (
            (BEFORE_STACK_NAME, before_shape),
            (ALIGNED_STACK_NAME, out_shape),
            (ALIGNMENT_CANVAS_STACK_NAME, canvas_shape),
        ):
            validate_output_stack(output_dir / name, shape)
        print("DONE; aligned stack:", aligned_path)
        print("Output ZYX shape:", out_shape)
        print(f"Good edges: {good_edges}/{len(edges)}")
        return summary


def main() -> None:
    """Run alignment using validated command-line arguments."""
    args = build_parser().parse_args()
    run_alignment(args)


if __name__ == "__main__":
    main()

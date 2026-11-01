#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM Charging Artifact Intensity Statistics

Author: Ningyan Zhang
Last modified: October 11, 2026

Description:
    Quantify the intensity and geometry of manually annotated charging
    artifacts in real, preprocessed cryo-FIB-SEM images. For each charging
    core, estimate a dual-kernel reference image, identify connected dark
    tails and nearby bright halos, and measure component-level statistics.
    This module analyzes charging artifacts; it does not build the template
    library. Run the subsequent charging-library steps for that purpose.

Inputs:
    --image-stack: grayscale uint8 ZYX stacked TIFF of real cryo-FIB-SEM.
    --mask-stack: matching grayscale uint8 ZYX TIFF containing the original
        manually annotated cores (0 = core, 255 = background). Values below
        --mask-threshold (default 128) are counted as annotated core pixels.
    Both stacks must share the same ZYX shape and 0-based Z indexing.
    Processing uses individual unmodified 2D slices loaded from the stacks.
    No VSNR, spatial alignment, cropping, intensity matching, or resampling
    is performed by the input adapter.

Outputs:
    per_component_stats.csv: component-level charging statistics (CSV).
    summary.json: aggregate statistics, run configuration, and input paths.
    visualization/slice_XXXX_core_YYY.png: optional three-panel previews
        (raw | dual-kernel estimate | colored core/tail/halo overlay).

Default layout (relative to the current working directory):
    data/images_stack.tif            Input uint8 ZYX volume
    data/core_masks_stack.tif        Matching uint8 ZYX core annotations
    data/real_charging_statistics/   Unchanged statistics and previews

Usage:
    python step1a_analyze_cryo_statistics.py \
        --image-stack /path/to/images_stack.tif \
        --mask-stack /path/to/core_masks_stack.tif \
        --output-dir /path/to/real_charging_statistics \
        --slice-start 0 --slice-end 59

Reproducibility:
    All scientific constants default to the values used in the original
    implementation. The --overwrite behavior is enabled by default and
    DELETES the existing output directory before writing new results.
    This program operates on the CPU (NumPy/SciPy/OpenCV); no MPS or CUDA
    configuration is required.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from tqdm import tqdm


# ============================================================
# 1. Paths
# ============================================================
# These are portable defaults, not paths to the original private dataset.
# Users may override every path through the command-line interface.
DATA_ROOT = Path("data")
IMAGE_STACK = DATA_ROOT / "images_stack.tif"
MASK_STACK = DATA_ROOT / "core_masks_stack.tif"
OUTPUT_DIR = DATA_ROOT / "real_charging_statistics"
VIS_DIR = OUTPUT_DIR / "visualization"


# ============================================================
# 2. Slice range
# ============================================================

# Use only the current training/reference region.
SLICE_START = 0
SLICE_END = 59

OVERWRITE = True


# ============================================================
# 3. Core mask
# ============================================================

MASK_THRESHOLD = 128
MIN_CORE_AREA = 5


# ============================================================
# 4. Local crop around each core
# ============================================================

# Large horizontal context because charging tails are long.
BBOX_Y_SCALE = 4.0
BBOX_X_SCALE = 10.0

MIN_CROP_H = 64
MIN_CROP_W = 128


# ============================================================
# 5. Dual-kernel clean estimate
# ============================================================

KERNEL_BIG = 151
KERNEL_MID = 31

ALPHA = 1.0
BETA = 0.75

CLIP_LOW = 0
CLIP_HIGH = 100


# ============================================================
# 6. Tail / halo extraction
# ============================================================

# Intensity-difference thresholds in uint8 gray levels.
DARK_DELTA_THRESHOLD = 2.0
BRIGHT_DELTA_THRESHOLD = 2.0

# Tail must originate from / connect to the core neighbourhood.
TAIL_CORE_SEED_DILATION = 6

# Bridge very small breaks in the thresholded tail.
TAIL_CONNECT_DILATION = 1

# Halo is searched around the core + tail support.
HALO_SEARCH_DILATION = 16


# ============================================================
# 7. Local background
# ============================================================

ARTEFACT_EXCLUDE_DILATION = 8
MIN_BACKGROUND_PIXELS = 50


# ============================================================
# 8. Visualization
# ============================================================

SAVE_PREVIEWS = True
MAX_PREVIEWS = 20


# ============================================================
# 9. Command-line interface and execution configuration
# ============================================================
def parse_arguments():
    """Parse user paths and run settings, preserving original defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure real charging core, dark tail, and bright halo "
            "statistics from matched uint8 ZYX stacked TIFF volumes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("Input and output paths")
    paths.add_argument(
        "--data-root", type=Path, default=DATA_ROOT,
        help="Parent directory for the default image and mask stacks.",
    )
    paths.add_argument(
        "--image-stack", type=Path, default=None,
        help="Path to uint8 ZYX image stacked TIFF.",
    )
    paths.add_argument(
        "--mask-stack", type=Path, default=None,
        help="Path to matching uint8 ZYX core-mask stacked TIFF.",
    )
    paths.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the statistics output folder.",
    )

    run = parser.add_argument_group("Slice selection and output")
    run.add_argument(
        "--slice-start", type=int, default=SLICE_START,
        help="First slice index (inclusive).",
    )
    run.add_argument(
        "--slice-end", type=int, default=SLICE_END,
        help="Last slice index (inclusive).",
    )
    run.add_argument(
        "--max-previews", type=int, default=MAX_PREVIEWS,
        help="Maximum number of per-core diagnostic previews overall.",
    )
    run.add_argument(
        "--save-previews", action="store_true", dest="save_previews",
        help="Enable the original three-panel diagnostic previews.",
    )
    run.add_argument(
        "--no-previews", action="store_false", dest="save_previews",
        help="Disable diagnostic preview generation.",
    )
    run.add_argument(
        "--overwrite", action="store_true", dest="overwrite",
        help="Delete any existing output directory before processing.",
    )
    run.add_argument(
        "--no-overwrite", action="store_false", dest="overwrite",
        help="Preserve existing output directory contents.",
    )
    parser.set_defaults(save_previews=SAVE_PREVIEWS, overwrite=OVERWRITE)

    method = parser.add_argument_group("Scientific parameters (original defaults)")
    method.add_argument("--mask-threshold", type=int, default=MASK_THRESHOLD)
    method.add_argument("--min-core-area", type=int, default=MIN_CORE_AREA)
    method.add_argument("--bbox-y-scale", type=float, default=BBOX_Y_SCALE)
    method.add_argument("--bbox-x-scale", type=float, default=BBOX_X_SCALE)
    method.add_argument("--min-crop-h", type=int, default=MIN_CROP_H)
    method.add_argument("--min-crop-w", type=int, default=MIN_CROP_W)
    method.add_argument("--kernel-big", type=int, default=KERNEL_BIG)
    method.add_argument("--kernel-mid", type=int, default=KERNEL_MID)
    method.add_argument("--alpha", type=float, default=ALPHA)
    method.add_argument("--beta", type=float, default=BETA)
    method.add_argument("--clip-low", type=float, default=CLIP_LOW)
    method.add_argument("--clip-high", type=float, default=CLIP_HIGH)
    method.add_argument(
        "--dark-delta-threshold", type=float, default=DARK_DELTA_THRESHOLD,
    )
    method.add_argument(
        "--bright-delta-threshold", type=float, default=BRIGHT_DELTA_THRESHOLD,
    )
    method.add_argument(
        "--tail-core-seed-dilation", type=int,
        default=TAIL_CORE_SEED_DILATION,
    )
    method.add_argument(
        "--tail-connect-dilation", type=int, default=TAIL_CONNECT_DILATION,
    )
    method.add_argument(
        "--halo-search-dilation", type=int, default=HALO_SEARCH_DILATION,
    )
    method.add_argument(
        "--artefact-exclude-dilation", type=int,
        default=ARTEFACT_EXCLUDE_DILATION,
    )
    method.add_argument(
        "--min-background-pixels", type=int,
        default=MIN_BACKGROUND_PIXELS,
    )
    return parser.parse_args()


def configure_from_arguments(args):
    """Bind CLI values to the original algorithm's configuration variables.

    The original core/tail/halo implementation reads module-level values.
    Setting those values once before processing preserves its calculations.
    """
    global DATA_ROOT, IMAGE_STACK, MASK_STACK, OUTPUT_DIR, VIS_DIR
    global SLICE_START, SLICE_END, OVERWRITE, SAVE_PREVIEWS, MAX_PREVIEWS
    global MASK_THRESHOLD, MIN_CORE_AREA, BBOX_Y_SCALE, BBOX_X_SCALE
    global MIN_CROP_H, MIN_CROP_W, KERNEL_BIG, KERNEL_MID, ALPHA, BETA
    global CLIP_LOW, CLIP_HIGH, DARK_DELTA_THRESHOLD, BRIGHT_DELTA_THRESHOLD
    global TAIL_CORE_SEED_DILATION, TAIL_CONNECT_DILATION
    global HALO_SEARCH_DILATION, ARTEFACT_EXCLUDE_DILATION
    global MIN_BACKGROUND_PIXELS

    DATA_ROOT = args.data_root.expanduser().resolve()
    IMAGE_STACK = (
        args.image_stack if args.image_stack is not None
        else DATA_ROOT / "images_stack.tif"
    ).expanduser().resolve()
    MASK_STACK = (
        args.mask_stack if args.mask_stack is not None
        else DATA_ROOT / "core_masks_stack.tif"
    ).expanduser().resolve()
    OUTPUT_DIR = (
        args.output_dir if args.output_dir is not None
        else DATA_ROOT / "real_charging_statistics"
    ).expanduser().resolve()
    VIS_DIR = OUTPUT_DIR / "visualization"

    # Validate source locations before the original overwrite logic can run.
    for input_stack in (IMAGE_STACK, MASK_STACK):
        if not input_stack.is_file():
            raise FileNotFoundError(f"Input stacked TIFF not found: {input_stack}")
        if OUTPUT_DIR == input_stack or OUTPUT_DIR in input_stack.parents:
            raise ValueError(f"Output directory contains an input stack: {input_stack}")
    if args.slice_start < 0 or args.slice_start > args.slice_end:
        raise ValueError("Require 0 <= --slice-start <= --slice-end.")
    if args.max_previews < 0:
        raise ValueError("--max-previews must be non-negative.")
    SLICE_START = args.slice_start
    SLICE_END = args.slice_end
    OVERWRITE = args.overwrite
    SAVE_PREVIEWS = args.save_previews
    MAX_PREVIEWS = args.max_previews
    MASK_THRESHOLD = args.mask_threshold
    MIN_CORE_AREA = args.min_core_area
    BBOX_Y_SCALE = args.bbox_y_scale
    BBOX_X_SCALE = args.bbox_x_scale
    MIN_CROP_H = args.min_crop_h
    MIN_CROP_W = args.min_crop_w
    KERNEL_BIG = args.kernel_big
    KERNEL_MID = args.kernel_mid
    ALPHA = args.alpha
    BETA = args.beta
    CLIP_LOW = args.clip_low
    CLIP_HIGH = args.clip_high
    DARK_DELTA_THRESHOLD = args.dark_delta_threshold
    BRIGHT_DELTA_THRESHOLD = args.bright_delta_threshold
    TAIL_CORE_SEED_DILATION = args.tail_core_seed_dilation
    TAIL_CONNECT_DILATION = args.tail_connect_dilation
    HALO_SEARCH_DILATION = args.halo_search_dilation
    ARTEFACT_EXCLUDE_DILATION = args.artefact_exclude_dilation
    MIN_BACKGROUND_PIXELS = args.min_background_pixels


# ============================================================
# 10. Helpers
# ============================================================

def ensure_dir(path):
    """Create an output folder and any missing parent folders."""
    path.mkdir(
        parents=True,
        exist_ok=True
    )


def reset_dir(path):
    """Remove an output folder if present, then recreate it."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Stacked-TIFF input adapter (I/O only)
# ============================================================

class StackedTiffReader:
    """Expose logical uint8 ZYX TIFF slices without rescaling or reordering.

    Uncompressed stacks use a read-only memory map when possible. Otherwise,
    multipage TIFFs are read one physical page at a time. This adapter never
    modifies pixel intensities, labels, coordinates, or slice indices.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.tiff = None
        self.mapped = None
        self.shape = None
        self.mode = None

    def __enter__(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"Input TIFF not found: {self.path}")
        if self.path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Input must be a TIFF stack: {self.path}")
        try:
            self.tiff = tifffile.TiffFile(str(self.path))
            series = self.tiff.series[0]
            shape = tuple(int(n) for n in series.shape)
            axes = series.axes
            if len(shape) == 2 and axes == "YX":
                self.shape = (1, *shape)
            elif len(shape) == 3 and axes in ("ZYX", "QYX", "IYX", "TYX"):
                self.shape = shape
            else:
                raise ValueError(
                    f"Expected grayscale ZYX TIFF, got shape={shape}, axes={axes}: {self.path}"
                )
            if np.dtype(series.dtype) != np.dtype('uint8'):
                raise TypeError(f"Expected uint8 TIFF; got {series.dtype}: {self.path}")
            if any(n <= 0 for n in self.shape):
                raise ValueError(f"Invalid stack shape: {self.shape}")

            # Prefer disk-backed access for a contiguous stacked TIFF.
            try:
                mapped = tifffile.memmap(str(self.path), series=0, mode="r")
                if tuple(mapped.shape) == self.shape:
                    self.mapped = mapped
                elif self.shape[0] == 1 and tuple(mapped.shape) == self.shape[1:]:
                    self.mapped = mapped[np.newaxis, ...]
            except (OSError, TypeError, ValueError):
                pass

            if self.mapped is not None:
                self.mode = "memmap"
            else:
                # Compressed conventional stacks: one grayscale page per Z slice.
                if len(self.tiff.pages) != self.shape[0]:
                    raise ValueError(
                        "Non-mappable TIFF requires one 2D page per Z slice: "
                        f"{self.path}"
                    )
                for index, page in enumerate(self.tiff.pages):
                    if (tuple(page.shape) != self.shape[1:]
                            or np.dtype(page.dtype) != np.dtype('uint8')
                            or int(page.samplesperpixel) != 1):
                        raise ValueError(
                            f"Invalid grayscale TIFF page z={index}: {self.path}"
                        )
                self.mode = "pages"
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def read(self, index):
        """Return an unchanged 2D uint8 image from the specified 0-based Z."""
        index = int(index)
        if self.shape is None or not 0 <= index < self.shape[0]:
            raise IndexError(f"Invalid Z index {index}: {self.path}")
        if self.mode == "memmap":
            image = np.asarray(self.mapped[index])
        else:
            image = self.tiff.pages[index].asarray()
        if image.shape != self.shape[1:] or image.dtype != np.uint8:
            raise ValueError(f"Invalid TIFF slice z={index}: {self.path}")
        return image

    def __exit__(self, exc_type, exc_value, traceback):
        self.mapped = None
        if self.tiff is not None:
            self.tiff.close()
            self.tiff = None


class StackSlice:
    """Identify one physical Z slice while keeping the parent TIFF open."""

    def __init__(self, reader, index):
        self.reader = reader
        self.index = int(index)

    def __str__(self):
        return f"{self.reader.path}::z={self.index}"


def open_matching_stacks(stack):
    """Open the paired inputs and check identical (Z, Y, X) geometry."""
    images = stack.enter_context(StackedTiffReader(IMAGE_STACK))
    masks = stack.enter_context(StackedTiffReader(MASK_STACK))
    if images.shape != masks.shape:
        raise ValueError(
            f"Image/mask stack ZYX mismatch: {images.shape} vs {masks.shape}. "
            "Provide the same original slice indexing and XY geometry."
        )
    image_map = {z: StackSlice(images, z) for z in range(images.shape[0])}
    mask_map = {z: StackSlice(masks, z) for z in range(masks.shape[0])}
    return image_map, mask_map


def read_uint8(path):
    """Read one unmodified 2D uint8 image from the input TIFF stack."""
    if not isinstance(path, StackSlice):
        raise TypeError("Expected an indexed stacked-TIFF slice reference.")
    return path.reader.read(path.index)


def ensure_odd(k):
    """Return a positive odd kernel size, using the original rule."""
    k = int(k)

    if k < 1:
        return 1

    return (
        k
        if k % 2 == 1
        else k + 1
    )


# ============================================================
# 11. Dual-kernel clean estimate
# ============================================================

def estimate_clean_dualkernel(image):
    """
    Estimate local clean background.

    Input/output remain in 0-255 intensity scale.
    """
    image = image.astype(
        np.float32
    )

    lo, hi = np.percentile(
        image,
        (
            CLIP_LOW,
            CLIP_HIGH
        )
    )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):
        return image.copy()

    normalized = (
        image - lo
    ) / (
        hi - lo + 1e-8
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    big = cv2.GaussianBlur(
        normalized,
        (
            ensure_odd(KERNEL_BIG),
            1
        ),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT
    )

    mid = cv2.GaussianBlur(
        normalized,
        (
            ensure_odd(KERNEL_MID),
            1
        ),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT
    )

    low_component = (
        big - big.mean()
    )

    mid_component = (
        mid - big
    )

    mid_component -= (
        mid_component.mean()
    )

    clean = (
        normalized
        - ALPHA * low_component
        - BETA * mid_component
    )

    clean = np.clip(
        clean,
        0.0,
        1.0
    )

    clean = (
        clean
        * (hi - lo)
        + lo
    )

    return clean.astype(
        np.float32
    )


# ============================================================
# 12. Core crop
# ============================================================

def get_bbox(binary):
    """Find the inclusive bounding rectangle of a binary component."""
    ys, xs = np.where(binary)

    if ys.size == 0:
        return None

    return {
        "y0": int(ys.min()),
        "y1": int(ys.max()),
        "x0": int(xs.min()),
        "x1": int(xs.max()),
        "h": int(
            ys.max() - ys.min() + 1
        ),
        "w": int(
            xs.max() - xs.min() + 1
        ),
    }


def expand_bbox(
    bbox,
    H,
    W
):
    """Expand the core bounding box with original vertical/horizontal rules."""
    cy = 0.5 * (
        bbox["y0"]
        + bbox["y1"]
    )

    cx = 0.5 * (
        bbox["x0"]
        + bbox["x1"]
    )

    # Keep the old logic:
    # crop scale is based on core height.
    base = max(
        float(bbox["h"]),
        1.0
    )

    crop_h = max(
        int(round(
            BBOX_Y_SCALE * base
        )),
        MIN_CROP_H,
        bbox["h"]
    )

    crop_w = max(
        int(round(
            BBOX_X_SCALE * base
        )),
        MIN_CROP_W,
        bbox["w"]
    )

    y0 = int(round(
        cy - crop_h / 2
    ))

    x0 = int(round(
        cx - crop_w / 2
    ))

    y1 = y0 + crop_h
    x1 = x0 + crop_w

    if y0 < 0:
        y1 -= y0
        y0 = 0

    if x0 < 0:
        x1 -= x0
        x0 = 0

    if y1 > H:
        shift = y1 - H
        y0 = max(
            0,
            y0 - shift
        )
        y1 = H

    if x1 > W:
        shift = x1 - W
        x0 = max(
            0,
            x0 - shift
        )
        x1 = W

    return (
        y0,
        y1,
        x0,
        x1
    )


# ============================================================
# 13. Tail connected to core
# ============================================================

def extract_core_connected_tail(
    tail_candidate,
    core_mask
):
    """
    Keep dark residual components that originate near the core.
    """
    seed = ndimage.binary_dilation(
        core_mask,
        iterations=TAIL_CORE_SEED_DILATION
    )

    connectivity_map = (
        tail_candidate.copy()
    )

    if TAIL_CONNECT_DILATION > 0:
        connectivity_map = (
            ndimage.binary_dilation(
                connectivity_map,
                iterations=TAIL_CONNECT_DILATION
            )
        )

    structure = np.ones(
        (
            3,
            3
        ),
        dtype=np.uint8
    )

    labels, num = ndimage.label(
        connectivity_map,
        structure=structure
    )

    keep_region = np.zeros_like(
        tail_candidate,
        dtype=bool
    )

    for label_id in range(
        1,
        num + 1
    ):
        component = (
            labels == label_id
        )

        if np.any(
            component & seed
        ):
            keep_region |= component

    # Keep only actual thresholded tail pixels.
    return (
        tail_candidate
        & keep_region
    )


# ============================================================
# 14. Statistics
# ============================================================

def percentile_or_nan(
    values,
    percentile
):
    """Compute the percentile of finite values or NaN when empty."""
    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            percentile
        )
    )


def median_or_nan(values):
    """Compute a median via the original 50th-percentile helper."""
    return percentile_or_nan(
        values,
        50
    )


def aggregate_column(
    df,
    column
):
    """Aggregate component values as n, mean, median, and quantiles."""
    values = (
        df[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .to_numpy(
            dtype=np.float32
        )
    )

    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }

    return {
        "n": int(
            values.size
        ),

        "mean": float(
            np.mean(values)
        ),

        "median": float(
            np.percentile(
                values,
                50
            )
        ),

        "p10": float(
            np.percentile(
                values,
                10
            )
        ),

        "p25": float(
            np.percentile(
                values,
                25
            )
        ),

        "p75": float(
            np.percentile(
                values,
                75
            )
        ),

        "p90": float(
            np.percentile(
                values,
                90
            )
        ),
    }


# ============================================================
# 15. Visualization
# ============================================================

def save_preview(
    path,
    raw,
    clean_est,
    core_mask,
    tail_mask,
    halo_mask
):
    """
    Three panels:
        raw | clean estimate | component overlay

    Overlay:
        red   = core
        blue  = tail
        green = halo
    """
    raw_u8 = np.clip(
        np.round(raw),
        0,
        255
    ).astype(np.uint8)

    clean_u8 = np.clip(
        np.round(clean_est),
        0,
        255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(
        raw_u8,
        cv2.COLOR_GRAY2BGR
    )

    overlay[
        core_mask
    ] = (
        0,
        0,
        255
    )

    overlay[
        tail_mask
    ] = (
        255,
        0,
        0
    )

    overlay[
        halo_mask
    ] = (
        0,
        255,
        0
    )

    raw_bgr = cv2.cvtColor(
        raw_u8,
        cv2.COLOR_GRAY2BGR
    )

    clean_bgr = cv2.cvtColor(
        clean_u8,
        cv2.COLOR_GRAY2BGR
    )

    separator = np.full(
        (
            raw.shape[0],
            8,
            3
        ),
        255,
        dtype=np.uint8
    )

    panel = np.concatenate(
        [
            raw_bgr,
            separator,
            clean_bgr,
            separator,
            overlay
        ],
        axis=1
    )

    cv2.imwrite(
        str(path),
        panel
    )


# ============================================================
# 16. Analyze one core component
# ============================================================

def analyze_component(
    slice_idx,
    component_id,
    image,
    component,
    preview_index
):
    """Measure the raw core, background, dark tail, and bright halo of one core."""
    bbox = get_bbox(
        component
    )

    if bbox is None:
        return None

    H, W = image.shape

    y0, y1, x0, x1 = (
        expand_bbox(
            bbox,
            H,
            W
        )
    )

    raw_crop = image[
        y0:y1,
        x0:x1
    ].astype(
        np.float32
    )

    core_crop = component[
        y0:y1,
        x0:x1
    ]

    # --------------------------------------------------------
    # Estimate clean image
    # --------------------------------------------------------

    clean_est = (
        estimate_clean_dualkernel(
            raw_crop
        )
    )

    dark_delta = np.clip(
        clean_est - raw_crop,
        0.0,
        None
    )

    bright_delta = np.clip(
        raw_crop - clean_est,
        0.0,
        None
    )

    # --------------------------------------------------------
    # Tail
    #
    # Core is explicitly excluded.
    # --------------------------------------------------------

    tail_candidate = (
        dark_delta
        > DARK_DELTA_THRESHOLD
    ) & (
        ~core_crop
    )

    tail_mask = (
        extract_core_connected_tail(
            tail_candidate,
            core_crop
        )
    )

    # --------------------------------------------------------
    # Halo
    #
    # Search only near core + tail.
    # --------------------------------------------------------

    dark_support = (
        core_crop
        | tail_mask
    )

    halo_search_region = (
        ndimage.binary_dilation(
            dark_support,
            iterations=HALO_SEARCH_DILATION
        )
    )

    halo_mask = (
        (
            bright_delta
            > BRIGHT_DELTA_THRESHOLD
        )
        & (~core_crop)
        & halo_search_region
    )

    # --------------------------------------------------------
    # Local background
    # --------------------------------------------------------

    artefact_support = (
        core_crop
        | tail_mask
        | halo_mask
    )

    excluded = (
        ndimage.binary_dilation(
            artefact_support,
            iterations=ARTEFACT_EXCLUDE_DILATION
        )
    )

    background_mask = (
        ~excluded
    )

    if (
        int(background_mask.sum())
        < MIN_BACKGROUND_PIXELS
    ):
        core_excluded = (
            ndimage.binary_dilation(
                core_crop,
                iterations=ARTEFACT_EXCLUDE_DILATION
            )
        )

        background_mask = (
            ~core_excluded
        )

    # --------------------------------------------------------
    # Raw / residual values
    # --------------------------------------------------------

    core_values = (
        raw_crop[
            core_crop
        ]
    )

    background_values = (
        raw_crop[
            background_mask
        ]
    )

    tail_values = (
        dark_delta[
            tail_mask
        ]
    )

    halo_values = (
        bright_delta[
            halo_mask
        ]
    )

    # Diagnostic only:
    # dual-kernel residual inside manual core.
    core_residual_values = (
        dark_delta[
            core_crop
        ]
    )

    background_median = (
        median_or_nan(
            background_values
        )
    )

    core_median = (
        median_or_nan(
            core_values
        )
    )

    core_background_delta = (
        background_median
        - core_median
    )

    tail_p90 = (
        percentile_or_nan(
            tail_values,
            90
        )
    )

    halo_p90 = (
        percentile_or_nan(
            halo_values,
            90
        )
    )

    core_residual_p90 = (
        percentile_or_nan(
            core_residual_values,
            90
        )
    )

    # --------------------------------------------------------
    # The key synthesis relationship
    # --------------------------------------------------------

    if (
        np.isfinite(core_background_delta)
        and core_background_delta > 1.0
        and np.isfinite(tail_p90)
    ):
        tail_to_core_ratio = (
            tail_p90
            / core_background_delta
        )
    else:
        tail_to_core_ratio = np.nan

    if (
        np.isfinite(core_background_delta)
        and core_background_delta > 1.0
        and np.isfinite(halo_p90)
    ):
        halo_to_core_ratio = (
            halo_p90
            / core_background_delta
        )
    else:
        halo_to_core_ratio = np.nan

    if (
        np.isfinite(background_median)
        and background_median > 1e-8
        and np.isfinite(core_median)
    ):
        core_to_background_ratio = (
            core_median
            / background_median
        )
    else:
        core_to_background_ratio = (
            np.nan
        )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    if (
        SAVE_PREVIEWS
        and preview_index
        < MAX_PREVIEWS
    ):
        save_preview(
            VIS_DIR
            / (
                f"slice_{slice_idx:04d}"
                f"_core_{component_id:03d}.png"
            ),
            raw_crop,
            clean_est,
            core_crop,
            tail_mask,
            halo_mask
        )

    # --------------------------------------------------------
    # One row = one real core component
    # --------------------------------------------------------

    return {
        "slice_idx": int(
            slice_idx
        ),

        "component_id": int(
            component_id
        ),

        "core_area": int(
            core_crop.sum()
        ),

        "crop_y0": int(y0),
        "crop_y1": int(y1 - 1),
        "crop_x0": int(x0),
        "crop_x1": int(x1 - 1),

        # Background
        "background_pixels": int(
            background_mask.sum()
        ),

        "background_median": (
            background_median
        ),

        # Core
        "core_raw_median": (
            core_median
        ),

        "core_background_delta": (
            core_background_delta
        ),

        "core_to_background_ratio": (
            core_to_background_ratio
        ),

        # Diagnostic only
        "core_residual_p90": (
            core_residual_p90
        ),

        # Tail
        "tail_pixels": int(
            tail_mask.sum()
        ),

        "tail_dark_delta_median": (
            median_or_nan(
                tail_values
            )
        ),

        "tail_dark_delta_p90": (
            tail_p90
        ),

        "tail_dark_delta_p95": (
            percentile_or_nan(
                tail_values,
                95
            )
        ),

        "tail_to_core_ratio": (
            tail_to_core_ratio
        ),

        # Halo
        "halo_pixels": int(
            halo_mask.sum()
        ),

        "halo_bright_delta_median": (
            median_or_nan(
                halo_values
            )
        ),

        "halo_bright_delta_p90": (
            halo_p90
        ),

        "halo_bright_delta_p95": (
            percentile_or_nan(
                halo_values,
                95
            )
        ),

        "halo_to_core_ratio": (
            halo_to_core_ratio
        ),
    }


# ============================================================
# 17. Main
# ============================================================

def main():
    """Validate input stacks, then execute the unchanged per-slice algorithm."""
    configure_from_arguments(parse_arguments())
    with ExitStack() as readers:
        image_map, mask_map = open_matching_stacks(readers)
        run_selected_slices(image_map, mask_map)


def run_selected_slices(image_map, mask_map):
    """Run the original scientific routine using indexed stack slices."""
    if OVERWRITE:
        reset_dir(
            OUTPUT_DIR
        )
    else:
        ensure_dir(
            OUTPUT_DIR
        )

    if SAVE_PREVIEWS:
        ensure_dir(
            VIS_DIR
        )

    indices = sorted(
        set(image_map)
        & set(mask_map)
    )

    indices = [
        idx
        for idx in indices
        if SLICE_START
        <= idx
        <= SLICE_END
    ]

    if not indices:
        raise RuntimeError(
            "No aligned image/mask slices found."
        )

    print(
        f"Matched slices: {len(indices)}"
    )

    print(
        f"Slice range: "
        f"{indices[0]}-{indices[-1]}"
    )

    rows = []

    preview_index = 0

    structure = np.ones(
        (
            3,
            3
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Analyze every manual core component
    # --------------------------------------------------------

    for slice_idx in tqdm(
        indices,
        desc="Analyzing real charging"
    ):

        image = read_uint8(
            image_map[slice_idx]
        )

        mask = read_uint8(
            mask_map[slice_idx]
        )

        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch slice {slice_idx}: "
                f"{image.shape} vs {mask.shape}"
            )

        all_core = (
            mask < MASK_THRESHOLD
        )

        labels, num_components = (
            ndimage.label(
                all_core,
                structure=structure
            )
        )

        for component_id in range(
            1,
            num_components + 1
        ):

            component = (
                labels == component_id
            )

            if (
                int(component.sum())
                < MIN_CORE_AREA
            ):
                continue

            row = analyze_component(
                slice_idx=slice_idx,
                component_id=component_id,
                image=image,
                component=component,
                preview_index=preview_index
            )

            if row is not None:
                rows.append(row)

                if (
                    SAVE_PREVIEWS
                    and preview_index
                    < MAX_PREVIEWS
                ):
                    preview_index += 1

    if not rows:
        raise RuntimeError(
            "No valid charging cores found."
        )

    # --------------------------------------------------------
    # Per-component table
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    csv_path = (
        OUTPUT_DIR
        / "per_component_stats.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Dataset-level statistics
    #
    # Aggregation is over artefact components,
    # not over all pixels.
    # --------------------------------------------------------

    keys = [
        "background_median",
        "core_raw_median",
        "core_background_delta",
        "core_to_background_ratio",
        "core_residual_p90",

        "tail_dark_delta_median",
        "tail_dark_delta_p90",
        "tail_dark_delta_p95",
        "tail_to_core_ratio",

        "halo_bright_delta_median",
        "halo_bright_delta_p90",
        "halo_bright_delta_p95",
        "halo_to_core_ratio",
    ]

    aggregated = {
        key: aggregate_column(
            df,
            key
        )
        for key
        in keys
    }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "image_stack": str(
            IMAGE_STACK
        ),

        "mask_stack": str(
            MASK_STACK
        ),

        "slice_start": int(
            SLICE_START
        ),

        "slice_end": int(
            SLICE_END
        ),

        "num_slices": int(
            len(indices)
        ),

        "num_components": int(
            len(df)
        ),

        "mask_semantic": (
            "0=core, 255=background"
        ),

        "charging_semantics": {
            "core": (
                "manual exact core; "
                "intensity measured directly from real image"
            ),

            "core_background_delta": (
                "local_background_median - core_raw_median"
            ),

            "tail": (
                "dark dual-kernel residual outside core, "
                "restricted to components connected to core neighbourhood"
            ),

            "tail_to_core_ratio": (
                "tail_dark_delta_p90 / core_background_delta"
            ),

            "halo": (
                "bright dual-kernel residual outside core "
                "and near core+tail support"
            ),

            "important": (
                "core, tail and halo are separate synthesis components"
            ),
        },

        "config": {
            "kernel_big": (
                KERNEL_BIG
            ),

            "kernel_mid": (
                KERNEL_MID
            ),

            "alpha": (
                ALPHA
            ),

            "beta": (
                BETA
            ),

            "dark_delta_threshold": (
                DARK_DELTA_THRESHOLD
            ),

            "bright_delta_threshold": (
                BRIGHT_DELTA_THRESHOLD
            ),

            "tail_core_seed_dilation": (
                TAIL_CORE_SEED_DILATION
            ),

            "tail_connect_dilation": (
                TAIL_CONNECT_DILATION
            ),

            "halo_search_dilation": (
                HALO_SEARCH_DILATION
            ),

            "artefact_exclude_dilation": (
                ARTEFACT_EXCLUDE_DILATION
            ),
        },

        "aggregated_component_statistics": (
            aggregated
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Simple report
    # --------------------------------------------------------

    print()
    print("Done.")
    print(
        f"Components: {len(df)}"
    )

    print()
    print("Median real relationships:")

    for key in (
        "background_median",
        "core_raw_median",
        "core_background_delta",
        "tail_dark_delta_p90",
        "tail_to_core_ratio",
        "halo_bright_delta_p90",
    ):
        value = (
            aggregated[key]["median"]
        )

        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "CSV:",
        csv_path
    )

    print(
        "Summary:",
        summary_path
    )

    if SAVE_PREVIEWS:
        print(
            "Previews:",
            VIS_DIR
        )


if __name__ == "__main__":
    main()
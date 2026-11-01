#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM Charging Core and Tail Residual Extraction

Author: Ningyan Zhang
Last modified: October 8, 2026

Description:
    Extract charging core images and dark tail residuals from the charging
    groups produced by step2_group_charging_cores_ZT.py. Exact annotated
    core pixels are copied directly from the source image; tail residuals
    are estimated independently and explicitly zeroed inside the core.
    Core/tail intensities, connectivity, crop bounds, and all scientific
    operations retain the original implementation and default parameters.

Inputs:
    Input directory: charging_bbox_extraction_train/ from charging-library
        Step 2, with one subdirectory named slice_XXXX per Z slice.
    Each processed slice_XXXX directory must contain:
        slice_XXXX_input_uint8.tif      2D grayscale TIFF, uint8
        slice_XXXX_labeled_int32.tif    2D labeled-core TIFF, int32
        groups/group_YYY/group_YYY_metadata.json
                                      Charging-group metadata, JSON
    The grouping metadata specifies the exact member core identifiers and
    inclusive spatial bounds. The source core mask convention is
    0 = core and 255 = background; this program uses labeled core IDs.
    Direct stacked TIFF input: NOT supported. Run the preceding grouping
    step on per-slice TIFF folders first.

Outputs:
    charging_core_tail_extraction_train/ (default):
        slice_XXXX/groups/group_YYY/
            group_YYY_raw_crop_uint8.tif          2D TIFF, uint8
            group_YYY_core_uint8.tif              2D TIFF, uint8;
                                                  255 outside the core
            group_YYY_core_mask_uint8.tif         2D TIFF, uint8;
                                                  0 = core, 255 = background
            group_YYY_tail_residual_float32.tif   2D TIFF, float32;
                                                  0 inside the core
            group_YYY_metadata.json               JSON metadata, including
                                                  source intensity statistics
        visualization/slice_XXXX/groups/group_YYY/ (optional):
            group_YYY_core_tail.png               Four-panel diagnostic PNG

Default directory layout, relative to the current working directory:
    data/charging_bbox_extraction_train/         Input from Step 2
    data/charging_core_tail_extraction_train/     Output from this step

Example:
    python step1c_residual_extraction.py \\
        --input-dir /path/to/charging_bbox_extraction_train \\
        --output-dir /path/to/charging_core_tail_extraction_train

Reproducibility:
    The original computations, order of processing, numeric data types,
    synthesis-relevant residual values, and method defaults are unchanged.
    By default, each selected output slice directory is overwritten, as
    in the original implementation. This tool uses CPU libraries only;
    no CUDA or Apple MPS configuration is required.
"""

import argparse
from pathlib import Path
import json
import re
import shutil

import cv2
import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm


# ============================================================
# 1. Portable input and output paths
# ============================================================
DATA_ROOT = Path("data")
INPUT_DIR = DATA_ROOT / "charging_bbox_extraction_train"
OUTPUT_DIR = DATA_ROOT / "charging_core_tail_extraction_train"
VIS_DIR = OUTPUT_DIR / "visualization"


# ============================================================
# 2. Slice selection
# ============================================================

# This stage reads the slice folders selected by the grouping stage.
#
# None -> all available input slices
# int  -> one slice
# list / tuple / set -> selected slices
SLICE_SELECTOR = None

OVERWRITE = True


# ============================================================
# 3. Group crop padding
# ============================================================

# Original crop padding is preserved by default.
GROUP_PAD_Y = 12
GROUP_PAD_X = 24


# ============================================================
# 4. Dual-kernel clean estimation
# ============================================================

KERNEL_BIG = 151
KERNEL_MID = 31

ALPHA = 1.0
BETA = 0.75

CLIP_LOW = 0
CLIP_HIGH = 100


# ============================================================
# 5. Tail extraction
# ============================================================

# Final tail values smaller than this are removed.
TAIL_VALUE_THRESHOLD = 1.0

# Connectivity is evaluated with this lower threshold
# before final cleanup.
CONNECTIVITY_THRESHOLD = 0.5

# A tail component must touch the neighbourhood
# around an exact manual core.
CORE_SEED_DILATION = 6

KEEP_ONLY_CONNECTED_TO_CORE = True


# ============================================================
# 6. Local source-background statistics
# ============================================================

# Used only to describe the real relationship:
#
# background -> core -> tail
#
# This does NOT alter core or tail.
BACKGROUND_EXCLUDE_DILATION = 8
MIN_BACKGROUND_PIXELS = 50

TAIL_REPORT_PERCENTILE = 90.0


# ============================================================
# 7. Mask convention
# ============================================================

# Saved core mask:
#
# 0   = exact manual core
# 255 = background
CORE_MASK_THRESHOLD = 128


# ============================================================
# 8. Visualization
# ============================================================

SAVE_PREVIEWS = True

# residual intensity difference display:
# zero = white
# strong tail = black
TAIL_DISPLAY_MAX = 120.0


# ============================================================
# Command-line arguments and portable configuration
# ============================================================
def parse_arguments():
    """Parse paths, slice selection, output switches, and method settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract separate charging cores and dark tail residuals from "
            "the output of the preceding charging-core grouping step."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("Input and output paths")
    paths.add_argument(
        "--data-root", type=Path, default=DATA_ROOT,
        help="Parent directory for the default input and output folders.",
    )
    paths.add_argument(
        "--input-dir", type=Path, default=None,
        help="Charging-group output directory from the preceding stage.",
    )
    paths.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for extracted charging cores, tails, and metadata.",
    )

    selection = parser.add_argument_group("Slice selection and output")
    selection.add_argument(
        "--slice-selector", type=int, nargs="+", default=None,
        metavar="INDEX",
        help="Optional exact slice indices (default: all input slices).",
    )
    selection.add_argument(
        "--overwrite", action="store_true", dest="overwrite",
        help="Remove selected per-slice output folders before processing.",
    )
    selection.add_argument(
        "--no-overwrite", action="store_false", dest="overwrite",
        help="Keep existing folders (matching output files are rewritten).",
    )
    selection.add_argument(
        "--save-previews", action="store_true", dest="save_previews",
        help="Write the original four-panel charging diagnostics.",
    )
    selection.add_argument(
        "--no-previews", action="store_false", dest="save_previews",
        help="Do not write diagnostic PNGs.",
    )
    parser.set_defaults(overwrite=OVERWRITE, save_previews=SAVE_PREVIEWS)

    method = parser.add_argument_group("Scientific parameters (original defaults)")
    method.add_argument(
        "--group-pad-y", type=int, default=GROUP_PAD_Y,
        help="Vertical padding around grouped charging bounding boxes.",
    )
    method.add_argument(
        "--group-pad-x", type=int, default=GROUP_PAD_X,
        help="Horizontal padding around grouped charging bounding boxes.",
    )
    method.add_argument(
        "--kernel-big", type=int, default=KERNEL_BIG,
        help="Large horizontal Gaussian-kernel size.",
    )
    method.add_argument(
        "--kernel-mid", type=int, default=KERNEL_MID,
        help="Intermediate horizontal Gaussian-kernel size.",
    )
    method.add_argument(
        "--alpha", type=float, default=ALPHA,
        help="Coefficient of the low-frequency correction.",
    )
    method.add_argument(
        "--beta", type=float, default=BETA,
        help="Coefficient of the intermediate-frequency correction.",
    )
    method.add_argument(
        "--clip-low", type=float, default=CLIP_LOW,
        help="Lower percentile for residual normalization.",
    )
    method.add_argument(
        "--clip-high", type=float, default=CLIP_HIGH,
        help="Upper percentile for residual normalization.",
    )
    method.add_argument(
        "--tail-value-threshold", type=float, default=TAIL_VALUE_THRESHOLD,
        help="Minimum retained dark-tail residual value.",
    )
    method.add_argument(
        "--connectivity-threshold", type=float, default=CONNECTIVITY_THRESHOLD,
        help="Residual threshold for core-connected tail regions.",
    )
    method.add_argument(
        "--core-seed-dilation", type=int, default=CORE_SEED_DILATION,
        help="Core-neighborhood dilation used for tail connectivity.",
    )
    method.add_argument(
        "--keep-only-connected-to-core", action="store_true",
        dest="keep_only_connected_to_core",
        help="Keep only core-connected tail residual components.",
    )
    method.add_argument(
        "--allow-disconnected-tail", action="store_false",
        dest="keep_only_connected_to_core",
        help="Keep tail residuals without core-connectivity filtering.",
    )
    method.add_argument(
        "--background-exclude-dilation", type=int,
        default=BACKGROUND_EXCLUDE_DILATION,
        help="Dilation width for excluding artifacts from background statistics.",
    )
    method.add_argument(
        "--min-background-pixels", type=int, default=MIN_BACKGROUND_PIXELS,
        help="Minimum retained background sample for primary exclusion.",
    )
    method.add_argument(
        "--tail-report-percentile", type=float, default=TAIL_REPORT_PERCENTILE,
        help="Percentile used in tail/core relationship metadata.",
    )
    method.add_argument(
        "--tail-display-max", type=float, default=TAIL_DISPLAY_MAX,
        help="Maximum residual for diagnostic display normalization.",
    )
    parser.set_defaults(keep_only_connected_to_core=KEEP_ONLY_CONNECTED_TO_CORE)
    return parser.parse_args()


def configure_from_arguments(args):
    """Bind parsed arguments to the original module configuration variables."""
    global DATA_ROOT, INPUT_DIR, OUTPUT_DIR, VIS_DIR
    global SLICE_SELECTOR, OVERWRITE, GROUP_PAD_Y, GROUP_PAD_X
    global KERNEL_BIG, KERNEL_MID, ALPHA, BETA, CLIP_LOW, CLIP_HIGH
    global TAIL_VALUE_THRESHOLD, CONNECTIVITY_THRESHOLD
    global CORE_SEED_DILATION, KEEP_ONLY_CONNECTED_TO_CORE
    global BACKGROUND_EXCLUDE_DILATION, MIN_BACKGROUND_PIXELS
    global TAIL_REPORT_PERCENTILE
    global SAVE_PREVIEWS, TAIL_DISPLAY_MAX

    DATA_ROOT = args.data_root.expanduser().resolve()
    INPUT_DIR = (
        args.input_dir if args.input_dir is not None
        else DATA_ROOT / "charging_bbox_extraction_train"
    ).expanduser().resolve()
    OUTPUT_DIR = (
        args.output_dir if args.output_dir is not None
        else DATA_ROOT / "charging_core_tail_extraction_train"
    ).expanduser().resolve()
    VIS_DIR = OUTPUT_DIR / "visualization"

    # The original implementation deletes output slice directories by default.
    # Reject unsafe output locations without changing the extraction algorithm.
    if OUTPUT_DIR == INPUT_DIR or OUTPUT_DIR in INPUT_DIR.parents:
        raise ValueError(
            "The output directory must not equal or contain the input "
            f"directory: output={OUTPUT_DIR}; input={INPUT_DIR}"
        )
    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(
            f"Charging-group input directory does not exist: {INPUT_DIR}"
        )

    SLICE_SELECTOR = args.slice_selector
    OVERWRITE = args.overwrite
    GROUP_PAD_Y = args.group_pad_y
    GROUP_PAD_X = args.group_pad_x
    KERNEL_BIG = args.kernel_big
    KERNEL_MID = args.kernel_mid
    ALPHA = args.alpha
    BETA = args.beta
    CLIP_LOW = args.clip_low
    CLIP_HIGH = args.clip_high
    TAIL_VALUE_THRESHOLD = args.tail_value_threshold
    CONNECTIVITY_THRESHOLD = args.connectivity_threshold
    CORE_SEED_DILATION = args.core_seed_dilation
    KEEP_ONLY_CONNECTED_TO_CORE = args.keep_only_connected_to_core
    BACKGROUND_EXCLUDE_DILATION = args.background_exclude_dilation
    MIN_BACKGROUND_PIXELS = args.min_background_pixels
    TAIL_REPORT_PERCENTILE = args.tail_report_percentile
    SAVE_PREVIEWS = args.save_previews
    TAIL_DISPLAY_MAX = args.tail_display_max


# ============================================================
# 9. Basic helpers
# ============================================================

def ensure_dir(path):
    """Create an output directory and any missing parents."""
    path.mkdir(
        parents=True,
        exist_ok=True
    )


def reset_dir(path):
    """Remove an output directory and recreate it empty."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def read_tif(path):
    """Read an input TIFF without performing intensity conversion."""
    return tifffile.imread(
        str(path)
    )


def save_tif(path, array):
    """Save a NumPy array to TIFF without changing its values."""
    tifffile.imwrite(
        str(path),
        array
    )


def parse_slice_index(name):

    """Extract the integer index from a slice_XXXX directory name."""
    match = re.fullmatch(
        r"slice_(\d+)",
        name
    )

    if match is None:
        raise ValueError(
            f"Invalid slice folder: {name}"
        )

    return int(
        match.group(1)
    )


def find_slice_dirs():

    """Discover all available slice directories in the grouping output."""
    result = {}

    for path in sorted(
        INPUT_DIR.glob("slice_*")
    ):

        if not path.is_dir():
            continue

        try:
            idx = parse_slice_index(
                path.name
            )

        except ValueError:
            continue

        result[idx] = path

    return result


def resolve_indices(
    available
):

    """Select requested slice indices from the discovered folders."""
    available = sorted(
        available
    )

    if SLICE_SELECTOR is None:
        return available

    if isinstance(
        SLICE_SELECTOR,
        int
    ):

        requested = [
            int(SLICE_SELECTOR)
        ]

    elif isinstance(
        SLICE_SELECTOR,
        (
            list,
            tuple,
            set
        )
    ):

        requested = sorted(
            set(
                int(x)
                for x
                in SLICE_SELECTOR
            )
        )

    else:

        raise TypeError(
            "SLICE_SELECTOR must be "
            "None, int, list, tuple, or set."
        )

    missing = [
        idx
        for idx in requested
        if idx not in available
    ]

    if missing:

        raise ValueError(
            f"Missing charging-group slices: {missing}"
        )

    return requested


def ensure_odd(k):

    """Ensure a horizontal Gaussian-kernel size is an odd positive integer."""
    k = int(k)

    if k < 1:
        return 1

    if k % 2 == 0:
        k += 1

    return k


def positive_percentile(
    values,
    percentile
):

    """Calculate a percentile over strictly positive finite values."""
    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if values.size == 0:
        return None

    return float(
        np.percentile(
            values,
            percentile
        )
    )


# ============================================================
# 10. Dual-kernel clean estimation
# ============================================================

def estimate_clean_dualkernel(
    image
):

    """Estimate the local clean intensity using the original dual-kernel correction."""
    image = image.astype(
        np.float32
    )

    values = image[
        np.isfinite(image)
    ]

    if values.size == 0:
        return image.copy()

    lo, hi = np.percentile(
        values,
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
            ensure_odd(
                KERNEL_BIG
            ),
            1
        ),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT
    )

    mid = cv2.GaussianBlur(
        normalized,
        (
            ensure_odd(
                KERNEL_MID
            ),
            1
        ),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT
    )

    low_component = (
        big
        - float(
            big.mean()
        )
    )

    mid_component = (
        mid - big
    )

    mid_component = (
        mid_component
        - float(
            mid_component.mean()
        )
    )

    clean = (
        normalized
        - ALPHA
        * low_component
        - BETA
        * mid_component
    )

    clean = np.clip(
        clean,
        0.0,
        1.0
    )

    clean = (
        clean
        * (
            hi - lo
        )
        + lo
    )

    return clean.astype(
        np.float32
    )


# ============================================================
# 11. Crop bounds
# ============================================================

def compute_padded_bounds(
    bbox,
    image_h,
    image_w
):

    """Expand an inclusive group box by the original X/Y padding."""
    y0 = max(
        0,
        int(bbox["y0"])
        - GROUP_PAD_Y
    )

    y1 = min(
        image_h - 1,
        int(bbox["y1"])
        + GROUP_PAD_Y
    )

    x0 = max(
        0,
        int(bbox["x0"])
        - GROUP_PAD_X
    )

    x1 = min(
        image_w - 1,
        int(bbox["x1"])
        + GROUP_PAD_X
    )

    return (
        y0,
        y1,
        x0,
        x1
    )


# ============================================================
# 12. Keep tail connected to core
# ============================================================

def keep_tail_connected_to_core(
    tail_residual,
    core_mask
):
    """
    tail_residual already has zero values inside core.

    Here we keep only dark residual components connected
    to the exact-core neighbourhood.
    """

    seed = ndimage.binary_dilation(
        core_mask,
        iterations=CORE_SEED_DILATION
    )

    # Include core temporarily only for connectivity.
    #
    # This does NOT put the core intensity back into
    # the final tail residual.
    binary = (
        tail_residual
        >= CONNECTIVITY_THRESHOLD
    ) | core_mask

    structure = np.ones(
        (
            3,
            3
        ),
        dtype=np.uint8
    )

    labeled, num = ndimage.label(
        binary,
        structure=structure
    )

    keep = np.zeros_like(
        core_mask,
        dtype=bool
    )

    for label_id in range(
        1,
        num + 1
    ):

        component = (
            labeled
            == label_id
        )

        if np.any(
            component
            & seed
        ):

            keep |= component

    output = np.zeros_like(
        tail_residual,
        dtype=np.float32
    )

    output[
        keep
    ] = tail_residual[
        keep
    ]

    # Absolute semantic guarantee:
    #
    # tail must NEVER contain core pixels.
    output[
        core_mask
    ] = 0.0

    return output


# ============================================================
# 13. Source relationship statistics
# ============================================================

def compute_source_relationship(
    raw_crop,
    core_mask,
    tail_residual
):
    """
    Measure this real template's core / tail relationship.

    This is metadata only.
    No image is changed here.
    """

    tail_mask = (
        tail_residual
        > TAIL_VALUE_THRESHOLD
    )

    artefact_support = (
        core_mask
        | tail_mask
    )

    excluded = ndimage.binary_dilation(
        artefact_support,
        iterations=BACKGROUND_EXCLUDE_DILATION
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
                core_mask,
                iterations=BACKGROUND_EXCLUDE_DILATION
            )
        )

        background_mask = (
            ~core_excluded
        )

    background_values = (
        raw_crop[
            background_mask
        ]
        .astype(
            np.float32
        )
    )

    core_values = (
        raw_crop[
            core_mask
        ]
        .astype(
            np.float32
        )
    )

    if background_values.size > 0:

        background_median = float(
            np.median(
                background_values
            )
        )

    else:

        background_median = None

    if core_values.size > 0:

        core_median = float(
            np.median(
                core_values
            )
        )

    else:

        core_median = None

    if (
        background_median is not None
        and core_median is not None
    ):

        core_delta = float(
            background_median
            - core_median
        )

    else:

        core_delta = None

    tail_p90 = (
        positive_percentile(
            tail_residual[
                tail_mask
            ],
            TAIL_REPORT_PERCENTILE
        )
    )

    if (
        core_delta is not None
        and core_delta > 1e-6
        and tail_p90 is not None
    ):

        tail_to_core_ratio = float(
            tail_p90
            / core_delta
        )

    else:

        tail_to_core_ratio = None

    return {
        "source_background_median": (
            background_median
        ),

        "source_core_median": (
            core_median
        ),

        "source_core_background_delta": (
            core_delta
        ),

        "source_tail_delta_p90": (
            tail_p90
        ),

        "source_tail_to_core_ratio": (
            tail_to_core_ratio
        ),

        "background_pixels": int(
            background_mask.sum()
        ),

        "core_pixels": int(
            core_mask.sum()
        ),

        "tail_pixels": int(
            tail_mask.sum()
        ),
    }


# ============================================================
# 14. Visualization
# ============================================================

def tail_to_preview(
    tail
):

    """Convert positive tail intensities into an inverted uint8 preview."""
    x = np.clip(
        tail.astype(
            np.float32
        ),
        0.0,
        TAIL_DISPLAY_MAX
    )

    x = (
        x
        / TAIL_DISPLAY_MAX
    )

    return np.round(
        (
            1.0 - x
        )
        * 255.0
    ).astype(
        np.uint8
    )


def save_preview(
    path,
    raw,
    core_mask,
    tail_residual
):

    """Create the original four-panel raw/core/tail/overlay diagnostic PNG."""
    raw_bgr = cv2.cvtColor(
        raw,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Core-only visualization:
    # real core pixels, white elsewhere
    # --------------------------------------------------------

    core_only = np.full(
        raw.shape,
        255,
        dtype=np.uint8
    )

    core_only[
        core_mask
    ] = raw[
        core_mask
    ]

    core_bgr = cv2.cvtColor(
        core_only,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Tail residual visualization
    # --------------------------------------------------------

    tail_preview = (
        tail_to_preview(
            tail_residual
        )
    )

    tail_bgr = cv2.cvtColor(
        tail_preview,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Overlay:
    # red = exact core
    # blue = extracted tail
    # --------------------------------------------------------

    overlay = raw_bgr.copy()

    tail_mask = (
        tail_residual
        > TAIL_VALUE_THRESHOLD
    )

    overlay[
        tail_mask
    ] = (
        255,
        0,
        0
    )

    overlay[
        core_mask
    ] = (
        0,
        0,
        255
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
            core_bgr,
            separator,
            tail_bgr,
            separator,
            overlay,
        ],
        axis=1
    )

    cv2.imwrite(
        str(path),
        panel
    )


# ============================================================
# 15. Process one group
# ============================================================

def process_one_group(
    group_meta_path,
    slice_image,
    slice_labeled,
    output_group_dir,
    vis_group_dir
):

    """Extract and save core pixels, tail residual, and JSON for a group."""
    with open(
        group_meta_path,
        "r"
    ) as f:

        group_meta = json.load(
            f
        )

    group_id = int(
        group_meta[
            "group_id"
        ]
    )

    super_bbox = (
        group_meta[
            "super_bbox"
        ]
    )

    H, W = (
        slice_image.shape
    )

    (
        y0,
        y1,
        x0,
        x1,
    ) = compute_padded_bounds(
        super_bbox,
        H,
        W
    )

    # --------------------------------------------------------
    # Raw crop
    # --------------------------------------------------------

    raw_crop = slice_image[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    labeled_crop = slice_labeled[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    # --------------------------------------------------------
    # Exact manual group-core mask
    #
    # Important:
    # padding can include other nearby cores belonging
    # to another group. Only member IDs count.
    # --------------------------------------------------------

    member_core_ids = [
        int(x)
        for x
        in group_meta[
            "member_core_ids"
        ]
    ]

    core_mask = np.isin(
        labeled_crop,
        member_core_ids
    )

    if not core_mask.any():

        raise RuntimeError(
            f"Group {group_id:03d} "
            f"contains no exact core pixels."
        )

    # --------------------------------------------------------
    # CORE
    #
    # This comes DIRECTLY from the real raw image.
    #
    # No dual-kernel.
    # No residual estimation.
    # No smoothing.
    # --------------------------------------------------------

    core_uint8 = np.full(
        raw_crop.shape,
        255,
        dtype=np.uint8
    )

    core_uint8[
        core_mask
    ] = raw_crop[
        core_mask
    ]

    core_mask_uint8 = np.where(
        core_mask,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # TAIL
    #
    # First estimate complete dark residual...
    # --------------------------------------------------------

    clean_est = (
        estimate_clean_dualkernel(
            raw_crop
        )
    )

    dark_residual = np.clip(
        clean_est
        - raw_crop.astype(
            np.float32
        ),
        0.0,
        None
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # ...then explicitly REMOVE CORE.
    #
    # From this point onward this variable means TAIL ONLY.
    # --------------------------------------------------------

    tail_residual = (
        dark_residual.copy()
    )

    tail_residual[
        core_mask
    ] = 0.0

    # --------------------------------------------------------
    # Keep only tail connected to the core neighbourhood.
    # --------------------------------------------------------

    if KEEP_ONLY_CONNECTED_TO_CORE:

        tail_residual = (
            keep_tail_connected_to_core(
                tail_residual,
                core_mask
            )
        )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    tail_residual[
        tail_residual
        < TAIL_VALUE_THRESHOLD
    ] = 0.0

    # Semantic assertion.
    if np.any(
        tail_residual[
            core_mask
        ] != 0
    ):

        raise RuntimeError(
            "Tail residual contains core pixels."
        )

    # --------------------------------------------------------
    # Measure real source relationship
    # --------------------------------------------------------

    relationship = (
        compute_source_relationship(
            raw_crop,
            core_mask,
            tail_residual
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    ensure_dir(
        output_group_dir
    )

    prefix = (
        f"group_{group_id:03d}"
    )

    save_tif(
        output_group_dir
        / (
            f"{prefix}"
            "_raw_crop_uint8.tif"
        ),
        raw_crop.astype(
            np.uint8
        )
    )

    save_tif(
        output_group_dir
        / (
            f"{prefix}"
            "_core_uint8.tif"
        ),
        core_uint8
    )

    save_tif(
        output_group_dir
        / (
            f"{prefix}"
            "_core_mask_uint8.tif"
        ),
        core_mask_uint8
    )

    save_tif(
        output_group_dir
        / (
            f"{prefix}"
            "_tail_residual_float32.tif"
        ),
        tail_residual.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "slice_idx": int(
            group_meta[
                "slice_idx"
            ]
        ),

        "group_id": int(
            group_id
        ),

        "member_core_ids": (
            member_core_ids
        ),

        "crop_bounds": {
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),

            "h": int(
                y1 - y0 + 1
            ),

            "w": int(
                x1 - x0 + 1
            ),
        },

        "semantics": {
            "core_uint8": (
                "real raw core pixels inside exact "
                "manual core mask; 255 outside"
            ),

            "core_mask_uint8": (
                "0=exact manual core, "
                "255=background"
            ),

            "tail_residual_float32": (
                "positive dark dual-kernel residual "
                "outside core only"
            ),

            "important": (
                "core and tail are separate components; "
                "tail residual is exactly zero inside core"
            ),
        },

        "relationship": (
            relationship
        ),

        "config": {
            "group_pad_y": (
                GROUP_PAD_Y
            ),

            "group_pad_x": (
                GROUP_PAD_X
            ),

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

            "connectivity_threshold": (
                CONNECTIVITY_THRESHOLD
            ),

            "core_seed_dilation": (
                CORE_SEED_DILATION
            ),

            "tail_value_threshold": (
                TAIL_VALUE_THRESHOLD
            ),
        },
    }

    with open(
        output_group_dir
        / (
            f"{prefix}"
            "_metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if SAVE_PREVIEWS:

        ensure_dir(
            vis_group_dir
        )

        save_preview(
            vis_group_dir
            / (
                f"{prefix}"
                "_core_tail.png"
            ),
            raw_crop,
            core_mask,
            tail_residual
        )


# ============================================================
# 16. Process one slice
# ============================================================

def process_one_slice(
    slice_idx,
    slice_dir
):

    """Read one grouped slice and process all listed charging groups."""
    input_path = (
        slice_dir
        / (
            f"slice_{slice_idx:04d}"
            "_input_uint8.tif"
        )
    )

    labeled_path = (
        slice_dir
        / (
            f"slice_{slice_idx:04d}"
            "_labeled_int32.tif"
        )
    )

    groups_dir = (
        slice_dir
        / "groups"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    if not labeled_path.exists():
        raise FileNotFoundError(
            labeled_path
        )

    slice_image = read_tif(
        input_path
    )

    slice_labeled = read_tif(
        labeled_path
    ).astype(
        np.int32
    )

    if (
        slice_image.ndim != 2
        or slice_image.dtype
        != np.uint8
    ):

        raise ValueError(
            f"Invalid grouped image: {input_path}"
        )

    if (
        slice_labeled.shape
        != slice_image.shape
    ):

        raise ValueError(
            f"Image/labeled shape mismatch "
            f"at slice {slice_idx}."
        )

    output_slice_dir = (
        OUTPUT_DIR
        / f"slice_{slice_idx:04d}"
    )

    vis_slice_dir = (
        VIS_DIR
        / f"slice_{slice_idx:04d}"
    )

    if (
        output_slice_dir.exists()
        and OVERWRITE
    ):

        shutil.rmtree(
            output_slice_dir
        )

    if (
        vis_slice_dir.exists()
        and OVERWRITE
    ):

        shutil.rmtree(
            vis_slice_dir
        )

    ensure_dir(
        output_slice_dir
    )

    output_groups_dir = (
        output_slice_dir
        / "groups"
    )

    ensure_dir(
        output_groups_dir
    )

    metadata_paths = sorted(
        groups_dir.glob(
            "group_*/group_*_metadata.json"
        )
    )

    if not metadata_paths:

        return

    for metadata_path in tqdm(
        metadata_paths,
        desc=f"Slice {slice_idx:04d} groups"
    ):

        group_name = (
            metadata_path
            .parent
            .name
        )

        process_one_group(
            group_meta_path=metadata_path,

            slice_image=(
                slice_image
            ),

            slice_labeled=(
                slice_labeled
            ),

            output_group_dir=(
                output_groups_dir
                / group_name
            ),

            vis_group_dir=(
                vis_slice_dir
                / "groups"
                / group_name
            ),
        )


# ============================================================
# 17. Main
# ============================================================

def main():
    """Extract the original core/tail outputs for selected grouped slices."""
    configure_from_arguments(parse_arguments())

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            INPUT_DIR
        )

    ensure_dir(
        OUTPUT_DIR
    )

    if SAVE_PREVIEWS:

        ensure_dir(
            VIS_DIR
        )

    slice_map = (
        find_slice_dirs()
    )

    if not slice_map:

        raise RuntimeError(
            f"No grouped slice folders found in:\n"
            f"{INPUT_DIR}"
        )

    indices = resolve_indices(
        slice_map.keys()
    )

    print(
        f"Grouped slices found: "
        f"{len(slice_map)}"
    )

    print(
        f"Processing: "
        f"{len(indices)} slices"
    )

    for idx in tqdm(
        indices,
        desc="Step3: core/tail extraction"
    ):

        process_one_slice(
            idx,
            slice_map[idx]
        )

    print()
    print("Done.")
    print(
        "Output:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
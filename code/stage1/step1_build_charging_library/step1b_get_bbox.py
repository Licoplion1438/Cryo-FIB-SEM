#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM Charging Core Detection and Grouping

Author: Ningyan Zhang
Last modified: October 11, 2026

Description:
    Identify manually annotated charging cores in each cryo-FIB-SEM slice,
    compute horizontal grouping bounding boxes, and group overlapping boxes
    into charging-artifact regions. Export group crops and metadata for the
    subsequent core/tail extraction stage. Grouping boxes are used only for
    grouping; they do not define or estimate the charging tail itself.

Inputs:
    --image-stack: grayscale uint8 ZYX stacked TIFF of real cryo-FIB-SEM.
    --mask-stack: matching grayscale uint8 ZYX stacked TIFF of manually
        annotated cores (0 = core, 255 = background). Pixels below
        --mask-threshold (default 128) identify annotated charging cores.
    Both stacks must have the same ZYX shape and original 0-based Z indices.
    Input slices are read without normalization, resizing, or resampling.

Outputs:
    charging_bbox_extraction_train/slice_XXXX/ (by default):
        slice_XXXX_input_uint8.tif       Original image, 2D uint8
        slice_XXXX_mask_uint8.tif        Original mask, 2D uint8
        slice_XXXX_labeled_int32.tif     Valid core labels, 2D int32
        slice_XXXX_summary.json         Slice, core, and group metadata
        groups/group_YYY/
            group_YYY_raw_crop_uint8.tif   Group crop, 2D uint8
            group_YYY_metadata.json        Group membership and bounds
    charging_bbox_extraction_train/visualization/slice_XXXX/ (optional):
        slice_XXXX_core_bbox.png        Core and grouping-box overlay
        slice_XXXX_groups.png           Merged group-box overlay

Default layout (relative to the working directory):
    data/images_stack.tif            Input uint8 ZYX volume
    data/core_masks_stack.tif        Matching uint8 ZYX manual annotations
    data/charging_bbox_extraction_train/  Unchanged 2D crop outputs

Example:
    python step1b_get_bbox.py \
        --image-stack /path/to/images_stack.tif \
        --mask-stack /path/to/core_masks_stack.tif \
        --output-dir /path/to/charging_bbox_extraction_train \
        --slice-start 0 --slice-end 59

Reproducibility:
    Scientific calculations, component identifiers, group-merging behavior,
    image data types, and all original default parameters are unchanged.
    With --overwrite (the default), existing output directories for selected
    slices are removed before re-creating those slices. Unselected slice
    directories are left untouched, as in the original implementation.
    This is a CPU-only program; no CUDA or MPS configuration is required.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path
from collections import defaultdict
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm


# ============================================================
# 1. Portable input and output paths
# ============================================================
# Defaults are relative to the user's working directory and can be
# overridden independently by command-line arguments.
DATA_ROOT = Path("data")
IMAGE_STACK = DATA_ROOT / "images_stack.tif"
MASK_STACK = DATA_ROOT / "core_masks_stack.tif"
OUTPUT_DIR = DATA_ROOT / "charging_bbox_extraction_train"
VIS_DIR = OUTPUT_DIR / "visualization"


# ============================================================
# 2. Slice selection
# ============================================================

SLICE_START = 0
SLICE_END = 59

# None -> use all slices in [SLICE_START, SLICE_END]
# int  -> one slice
# list / tuple / set -> selected slices
SLICE_SELECTOR = None

OVERWRITE = True


# ============================================================
# 3. Core mask
# ============================================================

# Manual display mask:
#   0   = charging core
#   255 = background
MASK_THRESHOLD = 128

MIN_COMPONENT_AREA = 5


# ============================================================
# 4. Artefact bbox used ONLY for grouping
# ============================================================

# Use core bbox height as the reference size.
BASE_SIZE_MODE = "bbox_height"

# The grouping box is intentionally narrow vertically
# and long horizontally because charging tails mainly
# extend along x.
BBOX_Y_SCALE = 1.0
BBOX_X_SCALE = 6.0

# Never allow the grouping bbox to become smaller
# than the exact core bbox itself.
ENFORCE_BBOX_COVER_CORE = True


# ============================================================
# 5. Visualization
# ============================================================

SAVE_DEBUG_FIGURES = True

FIGURE_DPI = 150

IMAGE_DISPLAY_VMIN = 80.0
IMAGE_DISPLAY_VMAX = 240.0


# ============================================================
# 6. Command-line arguments and configuration
# ============================================================
def parse_arguments():
    """Parse portable paths, slice selection, and original method settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect and group annotated charging cores from matching "
            "uint8 ZYX stacked TIFF image and mask volumes."
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
        help="Directory for generated per-slice groups and metadata.",
    )

    run = parser.add_argument_group("Slice selection and output")
    run.add_argument(
        "--slice-start", type=int, default=SLICE_START,
        help="First slice index to process (inclusive).",
    )
    run.add_argument(
        "--slice-end", type=int, default=SLICE_END,
        help="Last slice index to process (inclusive).",
    )
    run.add_argument(
        "--slice-selector", type=int, nargs="+", default=None,
        metavar="INDEX",
        help="Optional explicit slice indices, within the selected range.",
    )
    run.add_argument(
        "--overwrite", action="store_true", dest="overwrite",
        help="Remove selected per-slice output folders before processing.",
    )
    run.add_argument(
        "--no-overwrite", action="store_false", dest="overwrite",
        help="Keep existing folders (files with matching names are rewritten).",
    )
    run.add_argument(
        "--save-debug-figures", action="store_true", dest="save_debug_figures",
        help="Write the original bbox and group PNG overlays.",
    )
    run.add_argument(
        "--no-debug-figures", action="store_false", dest="save_debug_figures",
        help="Do not write debug PNG overlays.",
    )
    parser.set_defaults(
        overwrite=OVERWRITE,
        save_debug_figures=SAVE_DEBUG_FIGURES,
    )

    method = parser.add_argument_group("Scientific parameters (original defaults)")
    method.add_argument(
        "--mask-threshold", type=int, default=MASK_THRESHOLD,
        help="A pixel below this value is considered part of the core.",
    )
    method.add_argument(
        "--min-component-area", type=int, default=MIN_COMPONENT_AREA,
        help="Minimum connected-core component size in pixels.",
    )
    method.add_argument(
        "--base-size-mode", choices=("bbox_height", "eq_diameter"),
        default=BASE_SIZE_MODE,
        help="Reference size for each core's grouping bounding box.",
    )
    method.add_argument(
        "--bbox-y-scale", type=float, default=BBOX_Y_SCALE,
        help="Grouping-box height multiplier.",
    )
    method.add_argument(
        "--bbox-x-scale", type=float, default=BBOX_X_SCALE,
        help="Grouping-box width multiplier.",
    )
    method.add_argument(
        "--enforce-bbox-cover-core", action="store_true",
        dest="enforce_bbox_cover_core",
        help="Require grouping boxes to cover exact core bounding boxes.",
    )
    method.add_argument(
        "--no-enforce-bbox-cover-core", action="store_false",
        dest="enforce_bbox_cover_core",
        help="Do not enforce full core coverage in grouping boxes.",
    )
    method.add_argument(
        "--figure-dpi", type=int, default=FIGURE_DPI,
        help="Resolution of visualization PNG files.",
    )
    method.add_argument(
        "--image-display-vmin", type=float, default=IMAGE_DISPLAY_VMIN,
        help="Lower grayscale bound for debug PNGs.",
    )
    method.add_argument(
        "--image-display-vmax", type=float, default=IMAGE_DISPLAY_VMAX,
        help="Upper grayscale bound for debug PNGs.",
    )
    parser.set_defaults(enforce_bbox_cover_core=ENFORCE_BBOX_COVER_CORE)
    return parser.parse_args()


def configure_from_arguments(args):
    """Apply CLI values before executing the unchanged grouping algorithm.

    The original functions read module-level configuration variables.
    This method binds command-line options to those variables once; it does
    not alter the component labeling, grouping, or file-writing algorithms.
    """
    global DATA_ROOT, IMAGE_STACK, MASK_STACK, OUTPUT_DIR, VIS_DIR
    global SLICE_START, SLICE_END, SLICE_SELECTOR, OVERWRITE
    global MASK_THRESHOLD, MIN_COMPONENT_AREA
    global BASE_SIZE_MODE, BBOX_Y_SCALE, BBOX_X_SCALE
    global ENFORCE_BBOX_COVER_CORE, SAVE_DEBUG_FIGURES
    global FIGURE_DPI, IMAGE_DISPLAY_VMIN, IMAGE_DISPLAY_VMAX

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
        else DATA_ROOT / "charging_bbox_extraction_train"
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
    if args.slice_selector is not None:
        requested = set(args.slice_selector)
        outside = sorted(i for i in requested if not args.slice_start <= i <= args.slice_end)
        if outside:
            raise ValueError(f"Requested slice indices outside selected range: {outside}")
    SLICE_START = args.slice_start
    SLICE_END = args.slice_end
    SLICE_SELECTOR = args.slice_selector
    OVERWRITE = args.overwrite
    MASK_THRESHOLD = args.mask_threshold
    MIN_COMPONENT_AREA = args.min_component_area
    BASE_SIZE_MODE = args.base_size_mode
    BBOX_Y_SCALE = args.bbox_y_scale
    BBOX_X_SCALE = args.bbox_x_scale
    ENFORCE_BBOX_COVER_CORE = args.enforce_bbox_cover_core
    SAVE_DEBUG_FIGURES = args.save_debug_figures
    FIGURE_DPI = args.figure_dpi
    IMAGE_DISPLAY_VMIN = args.image_display_vmin
    IMAGE_DISPLAY_VMAX = args.image_display_vmax


# ============================================================
# 7. Basic helpers
# ============================================================

def ensure_dir(path):
    """Create a directory and any missing parent directories."""
    path.mkdir(
        parents=True,
        exist_ok=True
    )


def reset_dir(path):
    """Remove a directory and recreate an empty directory."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def save_tif(path, array):
    """Write a NumPy array to a TIFF file without modifying its values."""
    tifffile.imwrite(
        str(path),
        array
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
    """Read one unmodified 2D uint8 image/mask from a TIFF stack."""
    if not isinstance(path, StackSlice):
        raise TypeError("Expected an indexed stacked-TIFF slice reference.")
    return path.reader.read(path.index)


# ============================================================
# 8. Slice selection
# ============================================================

def resolve_selected_indices(
    common_indices
):
    """Select available slice indices within the configured range."""
    available = [
        idx
        for idx in sorted(common_indices)
        if SLICE_START
        <= idx
        <= SLICE_END
    ]

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
            f"Requested slices not available "
            f"in {SLICE_START}-{SLICE_END}: "
            f"{missing}"
        )

    return requested


# ============================================================
# 9. Exact manual-core connected components
# ============================================================

def equivalent_diameter(area):
    """Compute the diameter of a circle having the given pixel area."""
    return float(
        2.0
        * np.sqrt(
            area / np.pi
        )
    )


def find_core_components(
    core_binary
):
    """
    8-connected components from the exact manual core mask.

    Returns:
        relabeled:
            0 = background
            1,2,3,... = valid core components

        components:
            metadata for each retained core.
    """
    structure = np.ones(
        (
            3,
            3
        ),
        dtype=np.uint8
    )

    labeled, num = ndimage.label(
        core_binary,
        structure=structure
    )

    relabeled = np.zeros(
        labeled.shape,
        dtype=np.int32
    )

    components = []

    new_id = 1

    for old_id in range(
        1,
        num + 1
    ):
        mask = (
            labeled == old_id
        )

        area = int(
            mask.sum()
        )

        if area < MIN_COMPONENT_AREA:
            continue

        ys, xs = np.where(
            mask
        )

        y0 = int(
            ys.min()
        )

        y1 = int(
            ys.max()
        )

        x0 = int(
            xs.min()
        )

        x1 = int(
            xs.max()
        )

        bbox_h = (
            y1 - y0 + 1
        )

        bbox_w = (
            x1 - x0 + 1
        )

        relabeled[
            mask
        ] = new_id

        components.append({
            "comp_id": int(
                new_id
            ),

            "area": int(
                area
            ),

            "centroid_y": float(
                ys.mean()
            ),

            "centroid_x": float(
                xs.mean()
            ),

            "bbox_y0": y0,
            "bbox_y1": y1,
            "bbox_x0": x0,
            "bbox_x1": x1,

            "bbox_h": int(
                bbox_h
            ),

            "bbox_w": int(
                bbox_w
            ),

            "eq_diameter": (
                equivalent_diameter(
                    area
                )
            ),
        })

        new_id += 1

    return (
        relabeled,
        components
    )


# ============================================================
# 10. Grouping bbox around each core
# ============================================================

def get_base_size(component):

    """Choose the unchanged grouping-box reference size for one core."""
    if BASE_SIZE_MODE == "bbox_height":

        return float(
            component[
                "bbox_h"
            ]
        )

    if BASE_SIZE_MODE == "eq_diameter":

        return float(
            component[
                "eq_diameter"
            ]
        )

    raise ValueError(
        "BASE_SIZE_MODE must be "
        "'bbox_height' or 'eq_diameter'."
    )


def compute_grouping_bbox(
    component,
    image_h,
    image_w
):
    """
    Generate the horizontal artefact bbox used ONLY
    to decide whether nearby cores belong to one group.

    It does not define the final tail.
    """
    cy = float(
        component["centroid_y"]
    )

    cx = float(
        component["centroid_x"]
    )

    base_size = (
        get_base_size(
            component
        )
    )

    roi_h = max(
        1,
        int(np.ceil(
            BBOX_Y_SCALE
            * base_size
        ))
    )

    roi_w = max(
        1,
        int(np.ceil(
            BBOX_X_SCALE
            * base_size
        ))
    )

    if ENFORCE_BBOX_COVER_CORE:

        roi_h = max(
            roi_h,
            component["bbox_h"]
        )

        roi_w = max(
            roi_w,
            component["bbox_w"]
        )

    cy = int(round(cy))
    cx = int(round(cx))

    y0 = (
        cy
        - roi_h // 2
    )

    x0 = (
        cx
        - roi_w // 2
    )

    y1 = (
        y0
        + roi_h
        - 1
    )

    x1 = (
        x0
        + roi_w
        - 1
    )

    # --------------------------------------------------------
    # Shift rather than simply truncate whenever possible.
    # --------------------------------------------------------

    if y0 < 0:
        y1 -= y0
        y0 = 0

    if x0 < 0:
        x1 -= x0
        x0 = 0

    if y1 >= image_h:

        shift = (
            y1
            - image_h
            + 1
        )

        y0 -= shift
        y1 = image_h - 1

    if x1 >= image_w:

        shift = (
            x1
            - image_w
            + 1
        )

        x0 -= shift
        x1 = image_w - 1

    y0 = max(
        0,
        y0
    )

    x0 = max(
        0,
        x0
    )

    return {
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
    }


# ============================================================
# 11. Group overlapping artefact bboxes
# ============================================================

def boxes_overlap(
    box_a,
    box_b
):

    """Return whether two inclusive grouping bounding boxes overlap."""
    if (
        box_a["x1"]
        < box_b["x0"]
        or box_b["x1"]
        < box_a["x0"]
    ):
        return False

    if (
        box_a["y1"]
        < box_b["y0"]
        or box_b["y1"]
        < box_a["y0"]
    ):
        return False

    return True


class UnionFind:

    def __init__(self, n):

        """Initialize one disjoint-set entry for each candidate charging core."""
        self.parent = list(
            range(n)
        )

        self.rank = [
            0
        ] * n

    def find(self, x):

        """Find the representative element with path compression."""
        if (
            self.parent[x]
            != x
        ):

            self.parent[x] = (
                self.find(
                    self.parent[x]
                )
            )

        return self.parent[x]

    def union(
        self,
        a,
        b
    ):

        """Merge the sets of two overlapping charging-core candidates."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):

            self.parent[root_a] = (
                root_b
            )

        elif (
            self.rank[root_a]
            > self.rank[root_b]
        ):

            self.parent[root_b] = (
                root_a
            )

        else:

            self.parent[root_b] = (
                root_a
            )

            self.rank[
                root_a
            ] += 1


def build_groups(
    components,
    grouping_bboxes
):

    """Merge overlapping per-core grouping boxes using union-find."""
    n = len(
        components
    )

    if n == 0:
        return []

    union_find = UnionFind(
        n
    )

    # --------------------------------------------------------
    # Cores whose horizontal artefact bboxes overlap
    # are treated as one charging group.
    # --------------------------------------------------------

    for i in range(n):

        for j in range(
            i + 1,
            n
        ):

            if boxes_overlap(
                grouping_bboxes[i],
                grouping_bboxes[j]
            ):

                union_find.union(
                    i,
                    j
                )

    grouped_indices = defaultdict(
        list
    )

    for i in range(n):

        root = union_find.find(
            i
        )

        grouped_indices[
            root
        ].append(i)

    groups = []

    for group_id, member_indices in enumerate(
        grouped_indices.values(),
        start=1
    ):

        member_indices = sorted(
            member_indices
        )

        y0 = min(
            grouping_bboxes[i]["y0"]
            for i
            in member_indices
        )

        y1 = max(
            grouping_bboxes[i]["y1"]
            for i
            in member_indices
        )

        x0 = min(
            grouping_bboxes[i]["x0"]
            for i
            in member_indices
        )

        x1 = max(
            grouping_bboxes[i]["x1"]
            for i
            in member_indices
        )

        groups.append({
            "group_id": int(
                group_id
            ),

            "member_indices": (
                member_indices
            ),

            "super_bbox": {
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
        })

    return groups


# ============================================================
# 12. Visualization
# ============================================================

def save_core_bbox_overlay(
    image,
    components,
    grouping_bboxes,
    output_path
):

    """Plot core and grouping boxes over the grayscale input slice."""
    fig, ax = plt.subplots(
        figsize=(
            8,
            8
        )
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=IMAGE_DISPLAY_VMIN,
        vmax=IMAGE_DISPLAY_VMAX
    )

    for component, box in zip(
        components,
        grouping_bboxes
    ):

        core_rect = plt.Rectangle(
            (
                component["bbox_x0"],
                component["bbox_y0"]
            ),
            component["bbox_w"],
            component["bbox_h"],
            fill=False,
            edgecolor="lime",
            linewidth=1.2
        )

        ax.add_patch(
            core_rect
        )

        grouping_rect = (
            plt.Rectangle(
                (
                    box["x0"],
                    box["y0"]
                ),
                box["w"],
                box["h"],
                fill=False,
                edgecolor="yellow",
                linewidth=1.2,
                linestyle="--"
            )
        )

        ax.add_patch(
            grouping_rect
        )

        ax.text(
            component["centroid_x"],
            component["centroid_y"],
            str(
                component["comp_id"]
            ),
            color="red",
            fontsize=8,
            ha="center",
            va="center"
        )

    ax.set_title(
        "Exact core bbox (green) | grouping bbox (yellow)"
    )

    ax.axis(
        "off"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def save_group_overlay(
    image,
    groups,
    components,
    output_path
):

    """Plot merged charging-group boxes and their member core labels."""
    fig, ax = plt.subplots(
        figsize=(
            8,
            8
        )
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=IMAGE_DISPLAY_VMIN,
        vmax=IMAGE_DISPLAY_VMAX
    )

    for group in groups:

        box = (
            group[
                "super_bbox"
            ]
        )

        rect = plt.Rectangle(
            (
                box["x0"],
                box["y0"]
            ),
            box["w"],
            box["h"],
            fill=False,
            edgecolor="cyan",
            linewidth=1.8
        )

        ax.add_patch(
            rect
        )

        member_ids = [
            components[i][
                "comp_id"
            ]
            for i
            in group[
                "member_indices"
            ]
        ]

        ax.text(
            box["x0"],
            max(
                0,
                box["y0"] - 2
            ),
            f"G{group['group_id']}: "
            f"{member_ids}",
            color="cyan",
            fontsize=8,
            ha="left",
            va="bottom"
        )

    ax.set_title(
        "Charging groups"
    )

    ax.axis(
        "off"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 13. Process one slice
# ============================================================

def process_one_slice(
    slice_idx,
    image_path,
    mask_path
):

    """Label cores, form charging groups, and export one slice and its groups."""
    image = read_uint8(
        image_path
    )

    mask = read_uint8(
        mask_path
    )

    if image.shape != mask.shape:

        raise ValueError(
            f"Shape mismatch slice {slice_idx}: "
            f"image={image.shape}, "
            f"mask={mask.shape}"
        )

    H, W = image.shape

    core_binary = (
        mask
        < MASK_THRESHOLD
    )

    (
        labeled,
        components
    ) = find_core_components(
        core_binary
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    slice_dir = (
        OUTPUT_DIR
        / f"slice_{slice_idx:04d}"
    )

    slice_vis_dir = (
        VIS_DIR
        / f"slice_{slice_idx:04d}"
    )

    if (
        slice_dir.exists()
        and OVERWRITE
    ):
        shutil.rmtree(
            slice_dir
        )

    if (
        slice_vis_dir.exists()
        and OVERWRITE
    ):
        shutil.rmtree(
            slice_vis_dir
        )

    ensure_dir(
        slice_dir
    )

    groups_dir = (
        slice_dir
        / "groups"
    )

    ensure_dir(
        groups_dir
    )

    if SAVE_DEBUG_FIGURES:

        ensure_dir(
            slice_vis_dir
        )

    # --------------------------------------------------------
    # Files the next charging-library stage will use
    # --------------------------------------------------------

    save_tif(
        slice_dir
        / (
            f"slice_{slice_idx:04d}"
            "_input_uint8.tif"
        ),
        image
    )

    save_tif(
        slice_dir
        / (
            f"slice_{slice_idx:04d}"
            "_mask_uint8.tif"
        ),
        mask
    )

    save_tif(
        slice_dir
        / (
            f"slice_{slice_idx:04d}"
            "_labeled_int32.tif"
        ),
        labeled.astype(
            np.int32
        )
    )

    # --------------------------------------------------------
    # No core in this slice
    # --------------------------------------------------------

    if not components:

        summary = {
            "slice_idx": int(
                slice_idx
            ),

            "num_cores": 0,
            "num_groups": 0,
        }

        with open(
            slice_dir
            / (
                f"slice_{slice_idx:04d}"
                "_summary.json"
            ),
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

        return

    # --------------------------------------------------------
    # Build grouping bboxes and groups
    # --------------------------------------------------------

    grouping_bboxes = [
        compute_grouping_bbox(
            component,
            H,
            W
        )
        for component
        in components
    ]

    groups = build_groups(
        components,
        grouping_bboxes
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if SAVE_DEBUG_FIGURES:

        save_core_bbox_overlay(
            image=image,
            components=components,
            grouping_bboxes=grouping_bboxes,
            output_path=(
                slice_vis_dir
                / (
                    f"slice_{slice_idx:04d}"
                    "_core_bbox.png"
                )
            )
        )

        save_group_overlay(
            image=image,
            groups=groups,
            components=components,
            output_path=(
                slice_vis_dir
                / (
                    f"slice_{slice_idx:04d}"
                    "_groups.png"
                )
            )
        )

    # --------------------------------------------------------
    # Slice summary
    # --------------------------------------------------------

    summary = {
        "slice_idx": int(
            slice_idx
        ),

        "image_path": str(
            image_path
        ),

        "mask_path": str(
            mask_path
        ),

        "num_cores": int(
            len(components)
        ),

        "num_groups": int(
            len(groups)
        ),

        "config": {
            "mask_threshold": (
                MASK_THRESHOLD
            ),

            "min_component_area": (
                MIN_COMPONENT_AREA
            ),

            "base_size_mode": (
                BASE_SIZE_MODE
            ),

            "bbox_y_scale": (
                BBOX_Y_SCALE
            ),

            "bbox_x_scale": (
                BBOX_X_SCALE
            ),
        },

        "cores": [],

        "groups": [],
    }

    for component, box in zip(
        components,
        grouping_bboxes
    ):

        summary["cores"].append({
            **component,

            "grouping_bbox": (
                box
            ),
        })

    # --------------------------------------------------------
    # Save group crops + metadata
    # --------------------------------------------------------

    for group in groups:

        group_id = (
            group[
                "group_id"
            ]
        )

        member_indices = (
            group[
                "member_indices"
            ]
        )

        member_ids = [
            components[i][
                "comp_id"
            ]
            for i
            in member_indices
        ]

        box = (
            group[
                "super_bbox"
            ]
        )

        y0 = box["y0"]
        y1 = box["y1"]

        x0 = box["x0"]
        x1 = box["x1"]

        raw_crop = image[
            y0:y1 + 1,
            x0:x1 + 1
        ]

        group_name = (
            f"group_{group_id:03d}"
        )

        group_dir = (
            groups_dir
            / group_name
        )

        ensure_dir(
            group_dir
        )

        save_tif(
            group_dir
            / (
                f"{group_name}"
                "_raw_crop_uint8.tif"
            ),
            raw_crop
        )

        metadata = {
            "slice_idx": int(
                slice_idx
            ),

            "group_id": int(
                group_id
            ),

            "member_core_ids": [
                int(x)
                for x
                in member_ids
            ],

            "num_member_cores": int(
                len(member_ids)
            ),

            "super_bbox": (
                box
            ),

            "members": [],
        }

        for i in member_indices:

            component = (
                components[i]
            )

            metadata[
                "members"
            ].append({
                "comp_id": int(
                    component[
                        "comp_id"
                    ]
                ),

                "core_bbox": {
                    "y0": int(
                        component[
                            "bbox_y0"
                        ]
                    ),

                    "y1": int(
                        component[
                            "bbox_y1"
                        ]
                    ),

                    "x0": int(
                        component[
                            "bbox_x0"
                        ]
                    ),

                    "x1": int(
                        component[
                            "bbox_x1"
                        ]
                    ),

                    "h": int(
                        component[
                            "bbox_h"
                        ]
                    ),

                    "w": int(
                        component[
                            "bbox_w"
                        ]
                    ),
                },

                "grouping_bbox": (
                    grouping_bboxes[i]
                ),
            })

        with open(
            group_dir
            / (
                f"{group_name}"
                "_metadata.json"
            ),
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2
            )

        summary[
            "groups"
        ].append({
            "group_id": int(
                group_id
            ),

            "member_core_ids": [
                int(x)
                for x
                in member_ids
            ],

            "super_bbox": (
                box
            ),
        })

    with open(
        slice_dir
        / (
            f"slice_{slice_idx:04d}"
            "_summary.json"
        ),
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


# ============================================================
# 14. Main
# ============================================================

def main():
    """Validate input stacks, then execute the unchanged per-slice algorithm."""
    configure_from_arguments(parse_arguments())
    with ExitStack() as readers:
        image_map, mask_map = open_matching_stacks(readers)
        run_selected_slices(image_map, mask_map)


def run_selected_slices(image_map, mask_map):
    """Run the original scientific routine using indexed stack slices."""
    ensure_dir(
        OUTPUT_DIR
    )

    if SAVE_DEBUG_FIGURES:
        ensure_dir(
            VIS_DIR
        )

    common_indices = sorted(
        set(image_map)
        & set(mask_map)
    )

    selected_indices = (
        resolve_selected_indices(
            common_indices
        )
    )

    if not selected_indices:
        raise RuntimeError(
            "No matched image/mask slices "
            "in the requested range."
        )

    print(
        f"Matched slices selected: "
        f"{len(selected_indices)}"
    )

    print(
        f"Range: "
        f"{selected_indices[0]}"
        f"-"
        f"{selected_indices[-1]}"
    )

    for idx in tqdm(
        selected_indices,
        desc="Step2: core grouping"
    ):

        process_one_slice(
            slice_idx=idx,
            image_path=image_map[idx],
            mask_path=mask_map[idx]
        )

    print()
    print("Done.")
    print(
        "Output:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
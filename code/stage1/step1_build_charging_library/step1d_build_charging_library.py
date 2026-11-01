#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM Charging Artifact Library Construction

Author: Ningyan Zhang
Last modified: October 8, 2026

Description:
    Package validated charging core and tail templates into an artifact
    library for procedural synthesis. Each accepted charging group yields
    exactly one indexed library entry. Core appearances are copied from
    real image pixels, and extracted tail residuals remain separate.
    The original selection criteria, statistics, ordering, and data values
    are preserved.

Inputs:
    charging_core_tail_extraction_train/ from charging-library Step 3.
    Input is a directory hierarchy, NOT a stacked TIFF:
        slice_XXXX/groups/group_YYY/
            group_YYY_raw_crop_uint8.tif          Grayscale 2D TIFF, uint8
            group_YYY_core_uint8.tif              Grayscale 2D TIFF, uint8
            group_YYY_core_mask_uint8.tif         Grayscale 2D TIFF, uint8
                                                (0 = core; 255 = background)
            group_YYY_tail_residual_float32.tif   Grayscale 2D TIFF, float32
                                                (zero inside the core)
            group_YYY_metadata.json               Per-group metadata, JSON
    Each group's TIFF arrays must have identical spatial dimensions.

Outputs:
    charging_artefact_library_train/:
        entries/entry_000000/
            raw_crop_uint8.tif          2D TIFF, uint8
            core_uint8.tif              2D TIFF, uint8
            core_mask_uint8.tif         2D TIFF, uint8
            tail_residual_float32.tif   2D TIFF, float32
            meta.json                   JSON metadata and QC annotations
        library_index.csv               Index and per-entry measurements
        library_summary.json            Library counts and configuration
    The original filename and metadata schema are maintained for synthesis
    compatibility. Source paths recorded in metadata follow user inputs.

Default directory layout (relative to the working directory):
    data/charging_core_tail_extraction_train/    Step 3 input
    data/charging_artefact_library_train/         Output library

Example:
    python step1d_build_charging_library.py \
        --input-dir /path/to/charging_core_tail_extraction_train \
        --output-dir /path/to/charging_artefact_library_train

Reproducibility:
    The original scientific algorithm, group selection, numeric thresholds,
    array contents, and metadata fields remain unchanged. The original
    default behavior OVERWRITES the entire output library directory.
    This code uses CPU-only NumPy/TIFF operations (no CUDA/MPS required).
"""

import argparse
from pathlib import Path
import csv
import json
import shutil

import numpy as np
import tifffile
from tqdm import tqdm


# ============================================================
# 1. Portable input and output directories
# ============================================================
DATA_ROOT = Path("data")
INPUT_ROOT = DATA_ROOT / "charging_core_tail_extraction_train"
LIBRARY_ROOT = DATA_ROOT / "charging_artefact_library_train"
ENTRIES_ROOT = LIBRARY_ROOT / "entries"


# ============================================================
# 2. Settings
# ============================================================

OVERWRITE = True

# Skip templates that effectively contain no tail.
MIN_TAIL_SUM = 1.0
MIN_TAIL_AREA = 5

TAIL_NONZERO_THRESHOLD = 1e-6

CORE_MASK_THRESHOLD = 128

# Numerical tolerance for the semantic check:
# tail must be zero inside core.
CORE_TAIL_ZERO_TOLERANCE = 1e-6


# ============================================================
# Command-line arguments and portable path configuration
# ============================================================
def parse_arguments():
    """Parse portable paths, overwrite behavior, and original QC thresholds."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a charging artifact library from grouped core and tail "
            "templates produced by charging-library Step 3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("Input and output paths")
    paths.add_argument(
        "--data-root", type=Path, default=DATA_ROOT,
        help="Parent directory of default input and output folders.",
    )
    paths.add_argument(
        "--input-dir", type=Path, default=None,
        help="Step 3 charging core/tail extraction directory.",
    )
    paths.add_argument(
        "--output-dir", type=Path, default=None,
        help="Destination charging artifact library directory.",
    )

    output = parser.add_argument_group("Output behavior")
    output.add_argument(
        "--overwrite", action="store_true", dest="overwrite",
        help="Delete the existing output library before writing entries.",
    )
    output.add_argument(
        "--no-overwrite", action="store_false", dest="overwrite",
        help="Preserve the output directory (existing matching files are rewritten).",
    )
    parser.set_defaults(overwrite=OVERWRITE)

    qc = parser.add_argument_group("Scientific and QC parameters (original defaults)")
    qc.add_argument(
        "--min-tail-sum", type=float, default=MIN_TAIL_SUM,
        help="Minimum summed tail residual required to retain a group.",
    )
    qc.add_argument(
        "--min-tail-area", type=int, default=MIN_TAIL_AREA,
        help="Minimum nonzero tail area required to retain a group.",
    )
    qc.add_argument(
        "--tail-nonzero-threshold", type=float,
        default=TAIL_NONZERO_THRESHOLD,
        help="Threshold used to determine the nonzero tail mask.",
    )
    qc.add_argument(
        "--core-mask-threshold", type=int, default=CORE_MASK_THRESHOLD,
        help="Core pixels have mask intensity below this threshold.",
    )
    qc.add_argument(
        "--core-tail-zero-tolerance", type=float,
        default=CORE_TAIL_ZERO_TOLERANCE,
        help="Tolerance when verifying that the tail is zero inside cores.",
    )
    return parser.parse_args()


def configure_from_arguments(args):
    """Bind CLI inputs to the original module configuration variables."""
    global DATA_ROOT, INPUT_ROOT, LIBRARY_ROOT, ENTRIES_ROOT
    global OVERWRITE, MIN_TAIL_SUM, MIN_TAIL_AREA
    global TAIL_NONZERO_THRESHOLD, CORE_MASK_THRESHOLD
    global CORE_TAIL_ZERO_TOLERANCE

    DATA_ROOT = args.data_root.expanduser().resolve()
    INPUT_ROOT = (
        args.input_dir if args.input_dir is not None
        else DATA_ROOT / "charging_core_tail_extraction_train"
    ).expanduser().resolve()
    LIBRARY_ROOT = (
        args.output_dir if args.output_dir is not None
        else DATA_ROOT / "charging_artefact_library_train"
    ).expanduser().resolve()
    ENTRIES_ROOT = LIBRARY_ROOT / "entries"

    # The original main() removes LIBRARY_ROOT when OVERWRITE is enabled.
    # Reject destinations that could delete the input extraction directory.
    if LIBRARY_ROOT == INPUT_ROOT or LIBRARY_ROOT in INPUT_ROOT.parents:
        raise ValueError(
            "The output directory must not equal or contain the input "
            f"directory: output={LIBRARY_ROOT}; input={INPUT_ROOT}"
        )
    if not INPUT_ROOT.is_dir():
        raise NotADirectoryError(
            f"Charging core/tail input directory does not exist: {INPUT_ROOT}"
        )

    OVERWRITE = args.overwrite
    MIN_TAIL_SUM = args.min_tail_sum
    MIN_TAIL_AREA = args.min_tail_area
    TAIL_NONZERO_THRESHOLD = args.tail_nonzero_threshold
    CORE_MASK_THRESHOLD = args.core_mask_threshold
    CORE_TAIL_ZERO_TOLERANCE = args.core_tail_zero_tolerance


# ============================================================
# 3. Helpers
# ============================================================

def ensure_dir(path):
    """Create the requested directory and missing parent directories."""
    path.mkdir(
        parents=True,
        exist_ok=True
    )


def reset_dir(path):
    """Remove and recreate the target directory (destructive)."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def read_tif(path):
    """Read a TIFF array using the original TIFF loading procedure."""
    return tifffile.imread(
        str(path)
    )


def save_tif(path, array):
    """Write an array to a TIFF file without changing its values."""
    tifffile.imwrite(
        str(path),
        array
    )


def read_json(path):
    """Read per-group JSON metadata from disk."""
    with open(
        path,
        "r"
    ) as f:
        return json.load(f)


def find_group_dirs():
    """Collect Step 3 charging-group directories in their original sorted order."""
    group_dirs = []

    for slice_dir in sorted(
        INPUT_ROOT.glob("slice_*")
    ):
        if not slice_dir.is_dir():
            continue

        groups_dir = (
            slice_dir
            / "groups"
        )

        if not groups_dir.exists():
            continue

        for group_dir in sorted(
            groups_dir.glob("group_*")
        ):
            if group_dir.is_dir():
                group_dirs.append(
                    group_dir
                )

    return group_dirs


def parse_slice_idx(group_dir):
    """Read the source slice index from a group directory path."""
    slice_name = (
        group_dir
        .parent
        .parent
        .name
    )

    if not slice_name.startswith(
        "slice_"
    ):
        raise ValueError(
            f"Invalid slice folder: {slice_name}"
        )

    return int(
        slice_name.split("_")[1]
    )


def parse_group_id(group_dir):
    """Read the group identifier from its directory name."""
    name = group_dir.name

    if not name.startswith(
        "group_"
    ):
        raise ValueError(
            f"Invalid group folder: {name}"
        )

    return int(
        name.split("_")[1]
    )


def get_bbox(binary):
    """Return the inclusive spatial bounding box of a nonempty binary mask."""
    ys, xs = np.where(
        binary
    )

    if ys.size == 0:
        return None

    return {
        "y0": int(
            ys.min()
        ),

        "y1": int(
            ys.max()
        ),

        "x0": int(
            xs.min()
        ),

        "x1": int(
            xs.max()
        ),

        "h": int(
            ys.max()
            - ys.min()
            + 1
        ),

        "w": int(
            xs.max()
            - xs.min()
            + 1
        ),
    }


def positive_percentile(
    values,
    percentile
):
    """Return the percentile of finite positive values or None if empty."""
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
# 4. Load and validate one Step2 group
# ============================================================

def load_group(
    group_dir
):

    """Load and validate one extracted core/tail template.
    
        Input:
            group_dir: Directory containing the five Step 3 group files.
    
        Returns:
            Template arrays and metadata, or None if the tail fails the original
            minimum-sum or minimum-area criteria. Invalid arrays raise errors."""
    slice_idx = (
        parse_slice_idx(
            group_dir
        )
    )

    group_id = (
        parse_group_id(
            group_dir
        )
    )

    prefix = (
        f"group_{group_id:03d}"
    )

    raw_path = (
        group_dir
        / (
            f"{prefix}"
            "_raw_crop_uint8.tif"
        )
    )

    core_path = (
        group_dir
        / (
            f"{prefix}"
            "_core_uint8.tif"
        )
    )

    core_mask_path = (
        group_dir
        / (
            f"{prefix}"
            "_core_mask_uint8.tif"
        )
    )

    tail_path = (
        group_dir
        / (
            f"{prefix}"
            "_tail_residual_float32.tif"
        )
    )

    metadata_path = (
        group_dir
        / (
            f"{prefix}"
            "_metadata.json"
        )
    )

    required = [
        raw_path,
        core_path,
        core_mask_path,
        tail_path,
        metadata_path,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Incomplete Step 3 group:\n"
            + "\n".join(
                str(path)
                for path
                in missing
            )
        )

    raw = read_tif(
        raw_path
    )

    core = read_tif(
        core_path
    )

    core_mask_uint8 = read_tif(
        core_mask_path
    )

    tail = read_tif(
        tail_path
    ).astype(
        np.float32
    )

    source_meta = read_json(
        metadata_path
    )

    # --------------------------------------------------------
    # Shape / dtype checks
    # --------------------------------------------------------

    if (
        raw.ndim != 2
        or core.ndim != 2
        or core_mask_uint8.ndim != 2
        or tail.ndim != 2
    ):
        raise ValueError(
            f"Expected 2D arrays: {group_dir}"
        )

    if not (
        raw.shape
        == core.shape
        == core_mask_uint8.shape
        == tail.shape
    ):
        raise ValueError(
            f"Shape mismatch: {group_dir}\n"
            f"raw={raw.shape}, "
            f"core={core.shape}, "
            f"mask={core_mask_uint8.shape}, "
            f"tail={tail.shape}"
        )

    if raw.dtype != np.uint8:
        raise TypeError(
            f"raw crop is not uint8: "
            f"{group_dir}"
        )

    if core.dtype != np.uint8:
        raise TypeError(
            f"core is not uint8: "
            f"{group_dir}"
        )

    if core_mask_uint8.dtype != np.uint8:
        raise TypeError(
            f"core mask is not uint8: "
            f"{group_dir}"
        )

    # --------------------------------------------------------
    # Exact core
    # --------------------------------------------------------

    core_mask = (
        core_mask_uint8
        < CORE_MASK_THRESHOLD
    )

    core_area = int(
        core_mask.sum()
    )

    if core_area == 0:
        raise ValueError(
            f"Empty core mask: {group_dir}"
        )

    # Step2 definition:
    # outside exact core, core image must be white.
    if np.any(
        core[
            ~core_mask
        ] != 255
    ):
        raise ValueError(
            f"core_uint8 contains non-white "
            f"pixels outside core: {group_dir}"
        )

    # Exact real pixels inside the core must agree
    # with the raw crop.
    if not np.array_equal(
        core[
            core_mask
        ],
        raw[
            core_mask
        ]
    ):
        raise ValueError(
            f"core pixels do not match raw crop: "
            f"{group_dir}"
        )

    # --------------------------------------------------------
    # Tail
    # --------------------------------------------------------

    if not np.isfinite(
        tail
    ).all():
        raise ValueError(
            f"Tail contains NaN/Inf: "
            f"{group_dir}"
        )

    if float(
        tail.min()
    ) < -CORE_TAIL_ZERO_TOLERANCE:
        raise ValueError(
            f"Tail contains negative values: "
            f"{group_dir}"
        )

    # Most important semantic assertion:
    #
    # TAIL MUST NOT CONTAIN CORE.
    core_tail_max = float(
        np.max(
            np.abs(
                tail[
                    core_mask
                ]
            )
        )
    )

    if (
        core_tail_max
        > CORE_TAIL_ZERO_TOLERANCE
    ):
        raise ValueError(
            f"Tail is non-zero inside core: "
            f"{group_dir}, "
            f"max={core_tail_max}"
        )

    tail_mask = (
        tail
        > TAIL_NONZERO_THRESHOLD
    )

    tail_sum = float(
        tail.sum()
    )

    tail_area = int(
        tail_mask.sum()
    )

    tail_max = float(
        tail.max()
    )

    if (
        tail_sum < MIN_TAIL_SUM
        or tail_area < MIN_TAIL_AREA
    ):
        return None

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    core_bbox = get_bbox(
        core_mask
    )

    tail_bbox = get_bbox(
        tail_mask
    )

    # --------------------------------------------------------
    # Relationship metadata from Step2
    # --------------------------------------------------------

    relationship = (
        source_meta.get(
            "relationship",
            {}
        )
    )

    return {
        "slice_idx": (
            slice_idx
        ),

        "group_id": (
            group_id
        ),

        "group_dir": (
            group_dir
        ),

        "raw": (
            raw
        ),

        "core": (
            core
        ),

        "core_mask_uint8": (
            core_mask_uint8
        ),

        "core_mask": (
            core_mask
        ),

        "tail": (
            tail
        ),

        "core_area": (
            core_area
        ),

        "tail_area": (
            tail_area
        ),

        "tail_sum": (
            tail_sum
        ),

        "tail_max": (
            tail_max
        ),

        "tail_p90": (
            positive_percentile(
                tail[
                    tail_mask
                ],
                90
            )
        ),

        "core_bbox": (
            core_bbox
        ),

        "tail_bbox": (
            tail_bbox
        ),

        "relationship": (
            relationship
        ),

        "source_meta": (
            source_meta
        ),
    }


# ============================================================
# 5. Write CSV
# ============================================================

def write_csv(
    path,
    rows
):

    """Write a CSV index with keys and row order from the original data."""
    if not rows:
        return

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 6. Main
# ============================================================

def main():
    """Build library entries, index, and summary in the original source order.
    
        Every accepted Step 3 group becomes one numbered library entry. No
        image normalization, resizing, or core/tail recomputation occurs."""
    configure_from_arguments(parse_arguments())

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Step 3 output not found:\n"
            f"{INPUT_ROOT}"
        )

    if OVERWRITE:
        reset_dir(
            LIBRARY_ROOT
        )
    else:
        ensure_dir(
            LIBRARY_ROOT
        )

    ensure_dir(
        ENTRIES_ROOT
    )

    group_dirs = (
        find_group_dirs()
    )

    if not group_dirs:
        raise RuntimeError(
            f"No Step 3 groups found in:\n"
            f"{INPUT_ROOT}"
        )

    print(
        f"Step 3 groups found: "
        f"{len(group_dirs)}"
    )

    index_rows = []

    kept = 0
    skipped_empty_tail = 0

    # --------------------------------------------------------
    # One Step2 group -> one library entry
    # --------------------------------------------------------

    for group_dir in tqdm(
        group_dirs,
        desc="Step 4: building charging artifact library"
    ):

        data = load_group(
            group_dir
        )

        if data is None:
            skipped_empty_tail += 1
            continue

        entry_id = (
            kept
        )

        entry_name = (
            f"entry_{entry_id:06d}"
        )

        entry_dir = (
            ENTRIES_ROOT
            / entry_name
        )

        ensure_dir(
            entry_dir
        )

        # ----------------------------------------------------
        # Copy computational data EXACTLY.
        #
        # No intensity normalization.
        # No thresholding.
        # No resizing.
        # No core/tail modification.
        # ----------------------------------------------------

        save_tif(
            entry_dir
            / "raw_crop_uint8.tif",
            data["raw"]
        )

        save_tif(
            entry_dir
            / "core_uint8.tif",
            data["core"]
        )

        save_tif(
            entry_dir
            / "core_mask_uint8.tif",
            data[
                "core_mask_uint8"
            ]
        )

        save_tif(
            entry_dir
            / "tail_residual_float32.tif",
            data["tail"].astype(
                np.float32
            )
        )

        relationship = (
            data[
                "relationship"
            ]
        )

        # ----------------------------------------------------
        # Entry metadata
        # ----------------------------------------------------

        meta = {
            "entry_id": int(
                entry_id
            ),

            "entry_name": (
                entry_name
            ),

            "source": {
                "slice_idx": int(
                    data[
                        "slice_idx"
                    ]
                ),

                "group_id": int(
                    data[
                        "group_id"
                    ]
                ),

                "source_group_dir": str(
                    data[
                        "group_dir"
                    ]
                ),
            },

            "shape": {
                "h": int(
                    data[
                        "tail"
                    ].shape[0]
                ),

                "w": int(
                    data[
                        "tail"
                    ].shape[1]
                ),
            },

            "semantics": {
                "core_uint8": (
                    "real core pixels from raw image; "
                    "255 outside exact core"
                ),

                "core_mask_uint8": (
                    "0=exact manual core, "
                    "255=background"
                ),

                "tail_residual_float32": (
                    "dark tail residual only; "
                    "exactly zero inside core"
                ),

                "important": (
                    "core and tail must be rendered "
                    "as separate components in Step5"
                ),
            },

            "core": {
                "area": int(
                    data[
                        "core_area"
                    ]
                ),

                "bbox": (
                    data[
                        "core_bbox"
                    ]
                ),
            },

            "tail": {
                "area": int(
                    data[
                        "tail_area"
                    ]
                ),

                "sum": float(
                    data[
                        "tail_sum"
                    ]
                ),

                "max": float(
                    data[
                        "tail_max"
                    ]
                ),

                "p90": (
                    data[
                        "tail_p90"
                    ]
                ),

                "bbox": (
                    data[
                        "tail_bbox"
                    ]
                ),
            },

            "source_relationship": (
                relationship
            ),

            "qc": {
                "core_matches_raw": (
                    True
                ),

                "tail_zero_inside_core": (
                    True
                ),

                "tail_nonnegative": (
                    True
                ),

                "all_shapes_identical": (
                    True
                ),
            },
        }

        with open(
            entry_dir
            / "meta.json",
            "w"
        ) as f:

            json.dump(
                meta,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Global index
        # ----------------------------------------------------

        index_rows.append({
            "entry_id": int(
                entry_id
            ),

            "entry_name": (
                entry_name
            ),

            "slice_idx": int(
                data[
                    "slice_idx"
                ]
            ),

            "group_id": int(
                data[
                    "group_id"
                ]
            ),

            "h": int(
                data[
                    "tail"
                ].shape[0]
            ),

            "w": int(
                data[
                    "tail"
                ].shape[1]
            ),

            "core_area": int(
                data[
                    "core_area"
                ]
            ),

            "tail_area": int(
                data[
                    "tail_area"
                ]
            ),

            "tail_sum": float(
                data[
                    "tail_sum"
                ]
            ),

            "tail_max": float(
                data[
                    "tail_max"
                ]
            ),

            "tail_p90": (
                data[
                    "tail_p90"
                ]
            ),

            "source_background_median": (
                relationship.get(
                    "source_background_median"
                )
            ),

            "source_core_median": (
                relationship.get(
                    "source_core_median"
                )
            ),

            "source_core_background_delta": (
                relationship.get(
                    "source_core_background_delta"
                )
            ),

            "source_tail_delta_p90": (
                relationship.get(
                    "source_tail_delta_p90"
                )
            ),

            "source_tail_to_core_ratio": (
                relationship.get(
                    "source_tail_to_core_ratio"
                )
            ),

            "entry_dir": str(
                entry_dir
            ),

            "source_group_dir": str(
                data[
                    "group_dir"
                ]
            ),
        })

        kept += 1

    if not index_rows:
        raise RuntimeError(
            "No valid artefact entries were kept."
        )

    # --------------------------------------------------------
    # Keep source order.
    #
    # Do NOT reorder the library by tail intensity.
    # --------------------------------------------------------

    index_rows = sorted(
        index_rows,
        key=lambda row: (
            row["slice_idx"],
            row["group_id"]
        )
    )

    write_csv(
        LIBRARY_ROOT
        / "library_index.csv",
        index_rows
    )

    # --------------------------------------------------------
    # Library summary
    # --------------------------------------------------------

    summary = {
        "input_root": str(
            INPUT_ROOT
        ),

        "library_root": str(
            LIBRARY_ROOT
        ),

        "num_groups_found": int(
            len(group_dirs)
        ),

        "num_entries_kept": int(
            kept
        ),

        "num_entries_skipped_empty_tail": int(
            skipped_empty_tail
        ),

        "entry_files": [
            "raw_crop_uint8.tif",
            "core_uint8.tif",
            "core_mask_uint8.tif",
            "tail_residual_float32.tif",
            "meta.json",
        ],

        "semantics": {
            "core": (
                "real raw core appearance"
            ),

            "tail": (
                "dual-kernel dark residual "
                "outside core only"
            ),

            "core_tail_separated": (
                True
            ),

            "step5_rule": (
                "render core and tail separately"
            ),
        },

        "qc": {
            "min_tail_sum": (
                MIN_TAIL_SUM
            ),

            "min_tail_area": (
                MIN_TAIL_AREA
            ),

            "tail_nonzero_threshold": (
                TAIL_NONZERO_THRESHOLD
            ),

            "core_tail_zero_tolerance": (
                CORE_TAIL_ZERO_TOLERANCE
            ),
        },
    }

    with open(
        LIBRARY_ROOT
        / "library_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("Done.")
    print(
        "Library:",
        LIBRARY_ROOT
    )

    print(
        "Entries kept:",
        kept
    )

    print(
        "Empty-tail entries skipped:",
        skipped_empty_tail
    )


if __name__ == "__main__":
    main()
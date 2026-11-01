#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-1 Step 1: Prepare Four-Level Cryo-Style Clean Images

Author: Ningyan Zhang
Last modified: October 8, 2026

Description:
    Apply deterministic whole-slice weak affine intensity mappings to clean
    FIB-SEM images. Real cryo-FIB-SEM non-core pixels provide reference
    contrast statistics; the target median is one of 115, 145, 175, or 200.
    The original scientific algorithm, sampling, and split definitions are
    preserved. This stage does not synthesize artifacts.

Inputs:
    --input-root: Directory containing the immutable clean master split:
        train/block_*/<16 individual 1024x1024 uint8 TIFF files>
        val/block_*/<16 individual 1024x1024 uint8 TIFF files>
        test/block_*/<16 individual 1024x1024 uint8 TIFF files>
        The default split is 16 training / 3 validation / 3 test blocks.
    --cryo-dir: Directory of separate 2D uint8 TIFF reference images.
    --cryo-mask-dir: Directory of separate 2D uint8 TIFF masks of the
        corresponding reference images (0 = core; 255 = non-core).
        Reference images and masks are independently sorted by filename
        and paired POSITIONALLY. They must have matching sorted order.
        By default, sorted positions 0 through 59 are used (inclusive).
    All inputs are TIFF directories, NOT stacked TIFF volumes.

Outputs:
    --output-root:
        train/level_115/, level_145/, level_175/, level_200/
        val/level_115/, level_145/, level_175/, level_200/
        test/level_115/, level_145/, level_175/, level_200/
        visualization/{train,val,test}/: selected comparison TIFF images
        metadata.csv: per-slice assignments, reference statistics, mappings
        summary.json: dataset-level processing report
    Generated images are individual 1024x1024 2D uint8 TIFF files.

Scientific protocol:
    Each 16-slice block receives exactly four assignments per level.
    Brightness and reference assignments use stable SHA256-derived seeds.
    Train/validation/test membership is not changed. No image alignment,
    resampling, HPF, local normalization, or artifact synthesis is performed.
    Source images are never modified. Only the output directory is written.

Environment:
    CPU and disk I/O; CUDA and MPS are not required.

Example:
    python step2_prepare_clean_data.py \
        --input-root /path/to/clean_blocks_1024_z16 \
        --cryo-dir /path/to/cryo_reference_images \
        --cryo-mask-dir /path/to/cryo_core_masks \
        --output-root /path/to/stage1_style_transfer_4levels

    Run with --check-only to validate inputs and preview assignments without
    generating output. Use --overwrite only when regeneration is intended.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil

import numpy as np
import tifffile
from tqdm import tqdm


# ============================================================================
# 1. Paths
# ============================================================================

# Portable paths; argparse configuration replaces these before processing.
# Paths are resolved against the current working directory by default.
DEFAULT_DATA_ROOT = Path("data")
INPUT_ROOT = DEFAULT_DATA_ROOT / "clean_blocks_1024_z16"
CRYO_DIR = DEFAULT_DATA_ROOT / "cryo_reference_images"
CRYO_MASK_DIR = DEFAULT_DATA_ROOT / "cryo_core_masks"
OUTPUT_ROOT = DEFAULT_DATA_ROOT / "stage1_style_transfer_4levels"

METADATA_PATH = OUTPUT_ROOT / "metadata.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
VIS_ROOT = OUTPUT_ROOT / "visualization"


# ============================================================================
# 2. Dataset / split definition
# ============================================================================

SPLITS = ("train", "val", "test")

EXPECTED_BLOCK_COUNTS = {
    "train": 16,
    "val": 3,
    "test": 3,
}

EXPECTED_SLICES_PER_BLOCK = 16
EXPECTED_SIZE = 1024

TARGET_MEDIAN_LEVELS = (115.0, 145.0, 175.0, 200.0)
SLICES_PER_LEVEL_PER_BLOCK = 4

if (
    len(TARGET_MEDIAN_LEVELS) * SLICES_PER_LEVEL_PER_BLOCK
    != EXPECTED_SLICES_PER_BLOCK
):
    raise RuntimeError(
        "Brightness balancing does not sum to 16 slices per block."
    )


# ============================================================================
# 3. Cryo reference / style-transfer definition
# ============================================================================

# Inclusive reference positions within the lexicographically sorted TIFF lists.
CRYO_SLICE_START = 0
CRYO_SLICE_END = 59

CORE_MASK_THRESHOLD = 128

LOW_PERCENTILE = 5.0
MID_PERCENTILE = 50.0
HIGH_PERCENTILE = 95.0
MIN_RANGE = 20.0

# Original weak affine style-transfer parameters.
WEAK_AFFINE_STRENGTH = 0.25
SCALE_MIN = 0.75
SCALE_MAX = 1.35

REFERENCE_PIXEL_STRIDE = 4

# Global reproducibility anchor.
MASTER_SEED = 123

# Default CPU/I/O parallelism, independent of a specific workstation.
DEFAULT_NUM_WORKERS = min(6, max(1, os.cpu_count() or 1))

# Small number of before/after pairs for visual inspection.
SAVE_VISUALIZATION = True
NUM_VISUALIZATIONS_PER_SPLIT = 4


# ============================================================================
# 4. Basic helpers
# ============================================================================

def list_tifs(root: Path):
    """List lowercase .tif/.tiff files in sorted filename order."""
    return sorted(
        list(root.glob("*.tif"))
        + list(root.glob("*.tiff"))
    )


def read_uint8_2d(path: Path):
    """Read one 2D TIFF image, requiring uint8 pixel data."""
    img = tifffile.imread(str(path))

    if img.ndim != 2:
        raise ValueError(
            f"Expected 2D TIFF, got shape={img.shape}: {path}"
        )

    if img.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 TIFF, got dtype={img.dtype}: {path}"
        )

    return img


def to_uint8(x):
    """Round, clip, and convert an intensity array to uint8."""
    return np.clip(
        np.round(x),
        0,
        255,
    ).astype(np.uint8)


def stable_seed(*parts):
    """
    Deterministic 32-bit seed independent of Python's randomized hash().
    """
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def robust_stats(values):
    """Compute p5/p50/p95 and a minimum-clamped robust intensity range."""
    values = np.asarray(
        values,
        dtype=np.float32,
    ).ravel()

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError(
            "No valid pixels available for robust style statistics."
        )

    low, mid, high = np.percentile(
        values,
        [
            LOW_PERCENTILE,
            MID_PERCENTILE,
            HIGH_PERCENTILE,
        ],
    )

    low = float(low)
    mid = float(mid)
    high = float(high)

    value_range = high - low

    if value_range < MIN_RANGE:
        low = mid - MIN_RANGE / 2.0
        high = mid + MIN_RANGE / 2.0
        value_range = MIN_RANGE

    return {
        "low": low,
        "mid": mid,
        "high": high,
        "range": float(value_range),
        "num_values": int(values.size),
    }


# ============================================================================
# 5. Input discovery and strict split validation
# ============================================================================

def discover_master_dataset():
    """
    Returns:
        records_by_split = {
            "train": [
                {
                    "split": "train",
                    "block_name": "...",
                    "block_dir": Path(...),
                    "slice_paths": [16 sorted TIFF paths],
                },
                ...
            ],
            ...
        }
    """
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Master INPUT_ROOT does not exist:\n{INPUT_ROOT}"
        )

    records_by_split = {}
    seen_block_names = set()

    for split in SPLITS:
        split_dir = INPUT_ROOT / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing master split directory:\n{split_dir}"
            )

        block_dirs = sorted(
            p
            for p in split_dir.iterdir()
            if p.is_dir()
            and p.name.startswith("block_")
        )

        expected_blocks = EXPECTED_BLOCK_COUNTS[split]

        if len(block_dirs) != expected_blocks:
            raise RuntimeError(
                f"{split}: expected {expected_blocks} block folders, "
                f"found {len(block_dirs)} under:\n{split_dir}"
            )

        split_records = []

        for block_dir in block_dirs:
            if block_dir.name in seen_block_names:
                raise RuntimeError(
                    "A block folder name appears in more than one split: "
                    f"{block_dir.name}"
                )

            seen_block_names.add(block_dir.name)

            slice_paths = list_tifs(block_dir)

            if len(slice_paths) != EXPECTED_SLICES_PER_BLOCK:
                raise RuntimeError(
                    f"{split}/{block_dir.name}: expected "
                    f"{EXPECTED_SLICES_PER_BLOCK} TIFFs, "
                    f"found {len(slice_paths)}."
                )

            # Validate all 16 source slices before any output is written.
            for path in slice_paths:
                img = read_uint8_2d(path)

                if img.shape != (
                    EXPECTED_SIZE,
                    EXPECTED_SIZE,
                ):
                    raise RuntimeError(
                        f"{path}: expected "
                        f"{EXPECTED_SIZE}x{EXPECTED_SIZE}, "
                        f"got {img.shape}."
                    )

            split_records.append(
                {
                    "split": split,
                    "block_name": block_dir.name,
                    "block_dir": block_dir,
                    "slice_paths": slice_paths,
                }
            )

        records_by_split[split] = split_records

    return records_by_split


# ============================================================================
# 6. Real cryo reference statistics
# ============================================================================

def compute_one_reference_stat(
    slice_index,
    image_path,
    mask_path,
):
    """Collect non-core intensity statistics for one reference image/mask pair."""
    img = read_uint8_2d(image_path)
    mask = read_uint8_2d(mask_path)

    if img.shape != mask.shape:
        raise ValueError(
            f"Cryo image/mask shape mismatch at reference index "
            f"{slice_index}: {img.shape} vs {mask.shape}"
        )

    # Existing manual convention:
    #   0   = charging core
    #   255 = non-core/background/biology
    non_core = mask >= CORE_MASK_THRESHOLD
    values = img[non_core]

    if values.size == 0:
        raise RuntimeError(
            f"No non-core pixels in reference mask: {mask_path}"
        )

    sampled_values = values[
        ::REFERENCE_PIXEL_STRIDE
    ].astype(np.float32)

    stats = robust_stats(sampled_values)
    stats.update(
        {
            "slice_index": int(slice_index),
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "sampled_values": sampled_values,
        }
    )

    return stats


def compute_cryo_reference_stats(num_workers):
    """Analyze sorted reference-image/mask pairs in parallel and summarize them."""
    image_paths = list_tifs(CRYO_DIR)
    mask_paths = list_tifs(CRYO_MASK_DIR)

    if not image_paths:
        raise RuntimeError(
            f"No cryo TIFFs found in:\n{CRYO_DIR}"
        )

    if not mask_paths:
        raise RuntimeError(
            f"No cryo core-mask TIFFs found in:\n{CRYO_MASK_DIR}"
        )

    if len(image_paths) != len(mask_paths):
        raise RuntimeError(
            "Cryo image/mask count mismatch: "
            f"images={len(image_paths)}, masks={len(mask_paths)}"
        )

    if (
        CRYO_SLICE_START < 0
        or CRYO_SLICE_END < CRYO_SLICE_START
        or CRYO_SLICE_END >= len(image_paths)
    ):
        raise ValueError(
            f"Invalid cryo reference range "
            f"{CRYO_SLICE_START}-{CRYO_SLICE_END}; "
            f"found {len(image_paths)} slices."
        )

    indices = list(
        range(
            CRYO_SLICE_START,
            CRYO_SLICE_END + 1,
        )
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        future_map = {
            executor.submit(
                compute_one_reference_stat,
                idx,
                image_paths[idx],
                mask_paths[idx],
            ): idx
            for idx in indices
        }

        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Cryo reference stats",
            unit="slice",
        ):
            results.append(
                future.result()
            )

    # Restore deterministic index order after parallel execution.
    results.sort(
        key=lambda x: x["slice_index"]
    )

    pooled_values = np.concatenate(
        [
            item["sampled_values"]
            for item in results
        ]
    )

    pooled_stats = robust_stats(
        pooled_values
    )

    # sampled_values are not JSON-serializable and are no longer needed.
    for item in results:
        item.pop("sampled_values")

    return pooled_stats, results


# ============================================================================
# 7. Whole-slice weak affine mapping
# ============================================================================

def build_mapping(
    source_stats,
    target_stats,
):
    """Construct the original bounded weak affine contrast/median mapping."""
    source_range = max(
        float(source_stats["range"]),
        MIN_RANGE,
    )

    target_range = max(
        float(target_stats["range"]),
        MIN_RANGE,
    )

    raw_scale = (
        target_range
        / source_range
    )

    clipped_scale = float(
        np.clip(
            raw_scale,
            SCALE_MIN,
            SCALE_MAX,
        )
    )

    source_mid = float(
        source_stats["mid"]
    )

    target_mid = float(
        target_stats["mid"]
    )

    # Original weak-affine mapping definition:
    #
    # median-only:
    #   y = x + (target_mid - source_mid)
    #
    # full affine:
    #   y = scale*x + (target_mid - scale*source_mid)
    #
    # weakly blend the two.
    s = float(
        np.clip(
            WEAK_AFFINE_STRENGTH,
            0.0,
            1.0,
        )
    )

    A = (
        (1.0 - s)
        + s * clipped_scale
    )

    B = (
        (1.0 - s)
        * (target_mid - source_mid)
        + s
        * (
            target_mid
            - clipped_scale * source_mid
        )
    )

    return {
        "A": float(A),
        "B": float(B),
        "raw_scale": float(raw_scale),
        "clipped_scale": float(clipped_scale),
    }


def apply_mapping(
    img,
    mapping,
):
    """Apply the affine mapping to a whole clean image and return uint8."""
    x = img.astype(
        np.float32,
        copy=False,
    )

    y = (
        float(mapping["A"]) * x
        + float(mapping["B"])
    )

    return to_uint8(y)


# ============================================================================
# 8. Deterministic assignment plan
# ============================================================================

def build_assignment_plan(
    records_by_split,
    reference_stats,
):
    """
    Generate every brightness and reference choice BEFORE processing.

    Important:
    - brightness labels are shuffled only within each 16-slice block;
    - each block has 4 slices per target level;
    - split membership is never changed;
    - reference choice is independently deterministic for each slice.
    """
    plan = []

    base_levels = []

    for level in TARGET_MEDIAN_LEVELS:
        base_levels.extend(
            [float(level)]
            * SLICES_PER_LEVEL_PER_BLOCK
        )

    if len(base_levels) != EXPECTED_SLICES_PER_BLOCK:
        raise RuntimeError(
            "Internal brightness schedule length mismatch."
        )

    for split in SPLITS:
        for block_record in records_by_split[split]:
            block_name = block_record["block_name"]

            block_seed = stable_seed(
                MASTER_SEED,
                "brightness",
                split,
                block_name,
            )

            block_rng = np.random.default_rng(
                block_seed
            )

            levels = np.asarray(
                base_levels,
                dtype=np.float32,
            )

            levels = levels[
                block_rng.permutation(
                    len(levels)
                )
            ]

            # Verify exactly 4 of each level in this block.
            for level in TARGET_MEDIAN_LEVELS:
                count = int(
                    np.sum(
                        levels == float(level)
                    )
                )

                if count != SLICES_PER_LEVEL_PER_BLOCK:
                    raise RuntimeError(
                        f"{split}/{block_name}: "
                        f"target level {level} has count {count}, "
                        f"expected {SLICES_PER_LEVEL_PER_BLOCK}."
                    )

            for local_index, (
                slice_path,
                target_median,
            ) in enumerate(
                zip(
                    block_record["slice_paths"],
                    levels.tolist(),
                )
            ):
                ref_seed = stable_seed(
                    MASTER_SEED,
                    "cryo_reference",
                    split,
                    block_name,
                    slice_path.name,
                )

                ref_rng = np.random.default_rng(
                    ref_seed
                )

                ref_position = int(
                    ref_rng.integers(
                        0,
                        len(reference_stats),
                    )
                )

                reference = reference_stats[
                    ref_position
                ]

                plan.append(
                    {
                        "split": split,
                        "block_name": block_name,
                        "source_path": slice_path,
                        "source_filename": slice_path.name,
                        "local_index": int(local_index),
                        "target_median": float(target_median),
                        "brightness_seed": int(block_seed),
                        "reference_seed": int(ref_seed),
                        "reference_position": int(ref_position),
                        "reference_stats": reference,
                    }
                )

    return plan


# ============================================================================
# 9. Output layout
# ============================================================================

def prepare_output_root(
    overwrite,
):
    """Create per-split level folders, respecting the original overwrite policy."""
    if OUTPUT_ROOT.exists():
        if not overwrite:
            raise FileExistsError(
                f"OUTPUT_ROOT already exists:\n{OUTPUT_ROOT}\n\n"
                "Nothing was overwritten. "
                "Use --overwrite only if you intentionally want "
                "to regenerate this Stage-1 style dataset."
            )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in SPLITS:
        for level in TARGET_MEDIAN_LEVELS:
            level_int = int(
                round(level)
            )

            (
                OUTPUT_ROOT
                / split
                / f"level_{level_int}"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        (
            VIS_ROOT
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================================
# 10. Per-slice processing
# ============================================================================

def process_one_slice(
    item,
):
    """Style-match one clean image and record its reproducibility metadata."""
    split = item["split"]
    block_name = item["block_name"]
    source_path = item["source_path"]
    source_filename = item["source_filename"]
    local_index = item["local_index"]

    target_median = float(
        item["target_median"]
    )

    reference_stats = item[
        "reference_stats"
    ]

    img = read_uint8_2d(
        source_path
    )

    if img.shape != (
        EXPECTED_SIZE,
        EXPECTED_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected source shape for {source_path}: "
            f"{img.shape}"
        )

    source_stats = robust_stats(
        img
    )

    # Preserve the established reference-contrast mapping:
    # use the selected real cryo slice's p5/p50/p95 shape, but shift
    # its whole reference intensity interval so its median equals one
    # of the explicit target levels.
    target_stats = {
        "low": float(
            reference_stats["low"]
        ),
        "mid": float(
            reference_stats["mid"]
        ),
        "high": float(
            reference_stats["high"]
        ),
        "range": float(
            reference_stats["range"]
        ),
    }

    median_shift = (
        target_median
        - float(
            reference_stats["mid"]
        )
    )

    target_stats["mid"] = (
        target_median
    )

    target_stats["low"] = (
        float(reference_stats["low"])
        + median_shift
    )

    target_stats["high"] = (
        float(reference_stats["high"])
        + median_shift
    )

    mapping = build_mapping(
        source_stats,
        target_stats,
    )

    matched = apply_mapping(
        img,
        mapping,
    )

    level_int = int(
        round(
            target_median
        )
    )

    output_name = (
        f"{block_name}__"
        f"{source_filename}"
    )

    output_path = (
        OUTPUT_ROOT
        / split
        / f"level_{level_int}"
        / output_name
    )

    tifffile.imwrite(
        str(output_path),
        matched,
        photometric="minisblack",
    )

    # Immediate read-back verification.
    verify = tifffile.imread(
        str(output_path)
    )

    if verify.shape != (
        EXPECTED_SIZE,
        EXPECTED_SIZE,
    ):
        raise RuntimeError(
            f"Read-back shape mismatch: "
            f"{output_path} -> {verify.shape}"
        )

    if verify.dtype != np.uint8:
        raise RuntimeError(
            f"Read-back dtype mismatch: "
            f"{output_path} -> {verify.dtype}"
        )

    metadata = {
        "split": split,
        "block_name": block_name,
        "local_index": int(local_index),
        "source_filename": source_filename,
        "source_path": str(source_path),
        "target_level": int(level_int),
        "target_median": float(target_median),
        "brightness_seed": int(
            item["brightness_seed"]
        ),
        "reference_seed": int(
            item["reference_seed"]
        ),
        "reference_position": int(
            item["reference_position"]
        ),
        "reference_slice_index": int(
            reference_stats["slice_index"]
        ),
        "reference_image_name": (
            reference_stats["image_name"]
        ),
        "reference_mask_name": (
            reference_stats["mask_name"]
        ),
        "source_p5": float(
            source_stats["low"]
        ),
        "source_p50": float(
            source_stats["mid"]
        ),
        "source_p95": float(
            source_stats["high"]
        ),
        "reference_p5": float(
            reference_stats["low"]
        ),
        "reference_p50": float(
            reference_stats["mid"]
        ),
        "reference_p95": float(
            reference_stats["high"]
        ),
        "target_p5": float(
            target_stats["low"]
        ),
        "target_p50": float(
            target_stats["mid"]
        ),
        "target_p95": float(
            target_stats["high"]
        ),
        "mapping_A": float(
            mapping["A"]
        ),
        "mapping_B": float(
            mapping["B"]
        ),
        "raw_scale": float(
            mapping["raw_scale"]
        ),
        "clipped_scale": float(
            mapping["clipped_scale"]
        ),
        "output_mean": float(
            matched.mean()
        ),
        "output_p50": float(
            np.median(matched)
        ),
        "output_path": str(
            output_path
        ),
    }

    return metadata, img, matched


# ============================================================================
# 11. Metadata / summary
# ============================================================================

def write_metadata(
    rows,
):
    """Sort and write the per-image CSV metadata in split/block/slice order."""
    if not rows:
        raise RuntimeError(
            "No metadata rows were produced."
        )

    rows = sorted(
        rows,
        key=lambda r: (
            SPLITS.index(r["split"]),
            r["block_name"],
            r["local_index"],
        ),
    )

    with open(
        METADATA_PATH,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return rows


def verify_final_counts(
    rows,
):
    """Check exact per-split and per-level output image counts."""
    counts = {}

    for split in SPLITS:
        counts[split] = {}

        split_rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        expected_total = (
            EXPECTED_BLOCK_COUNTS[split]
            * EXPECTED_SLICES_PER_BLOCK
        )

        if len(split_rows) != expected_total:
            raise RuntimeError(
                f"{split}: expected {expected_total} output slices, "
                f"got {len(split_rows)}."
            )

        for level in TARGET_MEDIAN_LEVELS:
            level_int = int(
                round(level)
            )

            count = sum(
                1
                for row in split_rows
                if row["target_level"] == level_int
            )

            expected_level_count = (
                EXPECTED_BLOCK_COUNTS[split]
                * SLICES_PER_LEVEL_PER_BLOCK
            )

            if count != expected_level_count:
                raise RuntimeError(
                    f"{split}/level_{level_int}: "
                    f"expected {expected_level_count}, got {count}."
                )

            counts[split][
                f"level_{level_int}"
            ] = int(count)

    return counts


def save_pair_tif(
    original,
    matched,
    path,
):
    """Write a side-by-side original/style-matched comparison TIFF."""
    separator = np.zeros(
        (
            original.shape[0],
            8,
        ),
        dtype=np.uint8,
    )

    pair = np.concatenate(
        [
            original,
            separator,
            matched,
        ],
        axis=1,
    )

    tifffile.imwrite(
        str(path),
        pair,
        photometric="minisblack",
    )


# ============================================================================
# 12. Command-line path and run configuration
# ============================================================================


def configure_runtime(args):
    """Resolve user-supplied directories and retain the original defaults."""
    global INPUT_ROOT, CRYO_DIR, CRYO_MASK_DIR, OUTPUT_ROOT
    global METADATA_PATH, SUMMARY_PATH, VIS_ROOT
    global CRYO_SLICE_START, CRYO_SLICE_END, MASTER_SEED

    data_root = args.data_root.expanduser().resolve()
    INPUT_ROOT = (
        args.input_root if args.input_root is not None
        else data_root / "clean_blocks_1024_z16"
    ).expanduser().resolve()
    CRYO_DIR = (
        args.cryo_dir if args.cryo_dir is not None
        else data_root / "cryo_reference_images"
    ).expanduser().resolve()
    CRYO_MASK_DIR = (
        args.cryo_mask_dir if args.cryo_mask_dir is not None
        else data_root / "cryo_core_masks"
    ).expanduser().resolve()
    OUTPUT_ROOT = (
        args.output_root if args.output_root is not None
        else data_root / "stage1_style_transfer_4levels"
    ).expanduser().resolve()

    METADATA_PATH = OUTPUT_ROOT / "metadata.csv"
    SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
    VIS_ROOT = OUTPUT_ROOT / "visualization"

    CRYO_SLICE_START = args.cryo_slice_start
    CRYO_SLICE_END = args.cryo_slice_end
    MASTER_SEED = args.master_seed


def validate_runtime_paths():
    """Fail early for missing input folders and unsafe output destinations."""
    for name, path in (
        ("clean block input", INPUT_ROOT),
        ("cryo reference images", CRYO_DIR),
        ("cryo reference core masks", CRYO_MASK_DIR),
    ):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing {name} directory: {path}"
            )

    if CRYO_SLICE_START < 0 or CRYO_SLICE_END < CRYO_SLICE_START:
        raise ValueError(
            "Reference slice positions must satisfy "
            "0 <= --cryo-slice-start <= --cryo-slice-end."
        )

    # Never allow --overwrite to remove input images, masks, or their parents.
    for name, path in (
        ("clean block input", INPUT_ROOT),
        ("cryo reference images", CRYO_DIR),
        ("cryo reference core masks", CRYO_MASK_DIR),
    ):
        if OUTPUT_ROOT == path or OUTPUT_ROOT in path.parents or path in OUTPUT_ROOT.parents:
            raise ValueError(
                f"Unsafe output directory: {OUTPUT_ROOT}. "
                f"It overlaps the {name} directory: {path}"
            )


# ============================================================================
# 13. Main
# ============================================================================

def parse_args():
    """Parse dataset paths and original or optional runtime parameters."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic Stage-1 "
            "1024x1024 four-level style-matched JRC data."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Base directory for the default input/reference/output folders "
            "(default: ./data). Individual path arguments override this."
        ),
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Clean block split root (train/val/test/block_* with 16 uint8 "
            "TIFF slices per block). Default: DATA_ROOT/clean_blocks_1024_z16."
        ),
    )

    parser.add_argument(
        "--cryo-dir",
        type=Path,
        default=None,
        help=(
            "Real cryo reference images as individual uint8 TIFFs. "
            "Default: DATA_ROOT/cryo_reference_images."
        ),
    )

    parser.add_argument(
        "--cryo-mask-dir",
        type=Path,
        default=None,
        help=(
            "Corresponding reference core masks as individual uint8 TIFFs "
            "(0=core, 255=non-core). Default: DATA_ROOT/cryo_core_masks."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output dataset directory. "
            "Default: DATA_ROOT/stage1_style_transfer_4levels."
        ),
    )

    parser.add_argument(
        "--cryo-slice-start",
        type=int,
        default=CRYO_SLICE_START,
        help="First sorted reference slice position (inclusive; default: 0).",
    )

    parser.add_argument(
        "--cryo-slice-end",
        type=int,
        default=CRYO_SLICE_END,
        help="Last sorted reference slice position (inclusive; default: 59).",
    )

    parser.add_argument(
        "--master-seed",
        type=int,
        default=MASTER_SEED,
        help="Deterministic assignment seed (original default: 123).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and regenerate OUTPUT_ROOT if it already exists."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate all source paths/splits and print the assignment "
            "plan without writing outputs."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            f"CPU/I/O worker threads; default={DEFAULT_NUM_WORKERS}."
        ),
    )

    return parser.parse_args()


def main():
    """Configure paths, validate inputs, and run the original preparation pipeline."""
    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    configure_runtime(args)
    validate_runtime_paths()

    print("=" * 80)
    print("Stage-1 Step 1 | 1024 whole-slice style transfer")
    print(f"Input master : {INPUT_ROOT}")
    print(f"Output       : {OUTPUT_ROOT}")
    print(f"Workers      : {args.workers}")
    print(f"Master seed  : {MASTER_SEED}")
    print("=" * 80)

    # 1. Validate immutable master split and every 1024x1024 source TIFF.
    records_by_split = discover_master_dataset()

    print("\nMaster split validated:")
    for split in SPLITS:
        print(
            f"  {split:5s}: "
            f"{len(records_by_split[split])} blocks x "
            f"{EXPECTED_SLICES_PER_BLOCK} slices"
        )

    # 2. Compute real-cryo non-core reference statistics.
    pooled_reference_stats, reference_stats = (
        compute_cryo_reference_stats(
            num_workers=args.workers
        )
    )

    print("\nCryo reference statistics:")
    print(
        f"  p{LOW_PERCENTILE:g}  = "
        f"{pooled_reference_stats['low']:.2f}"
    )
    print(
        f"  p{MID_PERCENTILE:g} = "
        f"{pooled_reference_stats['mid']:.2f}"
    )
    print(
        f"  p{HIGH_PERCENTILE:g} = "
        f"{pooled_reference_stats['high']:.2f}"
    )
    print(
        f"  references = {len(reference_stats)} slices"
    )

    # 3. Build deterministic, split-preserving assignments.
    plan = build_assignment_plan(
        records_by_split,
        reference_stats,
    )

    print("\nPlanned style assignments:")
    for split in SPLITS:
        split_plan = [
            item
            for item in plan
            if item["split"] == split
        ]

        counts = {
            int(level): sum(
                1
                for item in split_plan
                if int(item["target_median"])
                == int(level)
            )
            for level in TARGET_MEDIAN_LEVELS
        }

        print(
            f"  {split:5s}: total={len(split_plan)} | "
            + " | ".join(
                f"{level}={counts[int(level)]}"
                for level in TARGET_MEDIAN_LEVELS
            )
        )

    if args.check_only:
        print(
            "\nCHECK ONLY complete. "
            "No Stage-1 style-transfer output was written."
        )
        return

    # 4. Create output tree only after every preflight check succeeds.
    prepare_output_root(
        overwrite=args.overwrite
    )

    # 5. Process slices in parallel. All randomness was fixed above.
    rows = []
    vis_saved = {
        split: 0
        for split in SPLITS
    }

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_map = {
            executor.submit(
                process_one_slice,
                item,
            ): item
            for item in plan
        }

        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Style transfer",
            unit="slice",
        ):
            item = future_map[future]

            try:
                metadata, original, matched = future.result()
            except Exception as exc:
                raise RuntimeError(
                    "Stage-1 style transfer failed for:\n"
                    f"  split={item['split']}\n"
                    f"  block={item['block_name']}\n"
                    f"  file={item['source_filename']}"
                ) from exc

            rows.append(
                metadata
            )

            split = metadata["split"]

            if (
                SAVE_VISUALIZATION
                and vis_saved[split]
                < NUM_VISUALIZATIONS_PER_SPLIT
            ):
                level_int = metadata[
                    "target_level"
                ]

                vis_name = (
                    f"{metadata['block_name']}__"
                    f"{Path(metadata['source_filename']).stem}__"
                    f"level_{level_int}__pair.tif"
                )

                save_pair_tif(
                    original,
                    matched,
                    VIS_ROOT
                    / split
                    / vis_name,
                )

                vis_saved[split] += 1

    # 6. Metadata first, then exact final count validation.
    rows = write_metadata(
        rows
    )

    final_counts = verify_final_counts(
        rows
    )

    reference_mids = np.asarray(
        [
            item["mid"]
            for item in reference_stats
        ],
        dtype=np.float32,
    )

    summary = {
        "step": "stage1_step1_whole_slice_style_transfer_1024_4levels",
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "master_split_counts": EXPECTED_BLOCK_COUNTS,
        "slices_per_block": EXPECTED_SLICES_PER_BLOCK,
        "expected_xy": [
            EXPECTED_SIZE,
            EXPECTED_SIZE,
        ],
        "target_median_levels": [
            int(v)
            for v in TARGET_MEDIAN_LEVELS
        ],
        "brightness_assignment": (
            "exactly 4 slices per target level within every 16-slice block; "
            "shuffled only within that block using a stable SHA256-derived seed"
        ),
        "split_policy": (
            "master train/val/test block membership is preserved exactly; "
            "no cross-split shuffle or reassignment"
        ),
        "master_seed": MASTER_SEED,
        "stable_seed_method": (
            "SHA256-derived 32-bit seeds from MASTER_SEED + identifiers"
        ),
        "cryo_reference_dir": str(CRYO_DIR),
        "cryo_mask_dir": str(CRYO_MASK_DIR),
        "cryo_slice_start": CRYO_SLICE_START,
        "cryo_slice_end": CRYO_SLICE_END,
        "num_reference_slices": len(reference_stats),
        "pooled_reference_stats": pooled_reference_stats,
        "reference_p50_min": float(
            reference_mids.min()
        ),
        "reference_p50_max": float(
            reference_mids.max()
        ),
        "reference_p50_median": float(
            np.median(reference_mids)
        ),
        "low_percentile": LOW_PERCENTILE,
        "mid_percentile": MID_PERCENTILE,
        "high_percentile": HIGH_PERCENTILE,
        "weak_affine_strength": WEAK_AFFINE_STRENGTH,
        "scale_min": SCALE_MIN,
        "scale_max": SCALE_MAX,
        "reference_pixel_stride": REFERENCE_PIXEL_STRIDE,
        "workers_used": args.workers,
        "final_counts": final_counts,
        "processing": (
            "one selected real cryo non-core contrast reference plus one "
            "explicit target median and one whole-slice weak affine mapping "
            "per 1024x1024 JRC TIFF; no alignment, interpolation, resizing, "
            "HPF, local normalization, gamma/local augmentation, or "
            "artifact synthesis"
        ),
        "metadata_csv": str(METADATA_PATH),
    }

    with open(
        SUMMARY_PATH,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nDone.")
    print(f"Output   : {OUTPUT_ROOT}")
    print(f"Metadata : {METADATA_PATH}")
    print(f"Summary  : {SUMMARY_PATH}")

    for split in SPLITS:
        print(
            f"  {split:5s}: "
            + ", ".join(
                f"{level}={final_counts[split][f'level_{level}']}"
                for level in (
                    115,
                    145,
                    175,
                    200,
                )
            )
        )


if __name__ == "__main__":
    main()

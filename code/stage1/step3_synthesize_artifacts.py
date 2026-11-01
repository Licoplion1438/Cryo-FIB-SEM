#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM Stage-1 Step 2: Synthetic Artifact Dataset Generation

Author: Ningyan Zhang
Last modified: October 8, 2026

Description:
    Generate paired 2D clean and degraded cryo-FIB-SEM-style training images
    using real charging templates and procedural dark charging, thin bundles,
    joint shading, halos, and horizontal white stripes. Training, validation,
    and test membership is inherited from Stage-1 Step 1 without resplitting.

Inputs:
    --style-root:
        Stage-1 Step 1 output directory, containing individual 1024x1024
        uint8 TIFF files in train/level_{115,145,175,200}/, val/, and test/.
    --style-metadata:
        Stage-1 Step 1 metadata.csv (default: <style-root>/metadata.csv).
    --artifact-library-root:
        Charging Library Step 4 output, including library_index.csv and
        entries/entry_XXXXXX/ with 2D uint8 core/core-mask and float32 tail
        TIFF files, plus optional supporting TIFF/JSON metadata.
    --charging-statistics-summary:
        Charging Library Step 1 output summary.json, containing
        aggregated_component_statistics. Original lookup and fallbacks are
        deliberately preserved to reproduce the accepted training definition.
    All image inputs are individual 2D TIFF files, not stacked TIFF volumes.

Outputs:
    --output-root:
        {train,val,test}/bg_{115,145,175,200}/clean/*.tif
        {train,val,test}/bg_{115,145,175,200}/noisy_charging/*.tif
        {train,val,test}/bg_{115,145,175,200}/noisy/*.tif
        {train,val,test}/bg_{115,145,175,200}/preview/*.png
        {train,val,test}/bg_{115,145,175,200}/
            preview_core_membrane_diagnostic/*.png
        metadata.csv, summary.json, experiment.txt, running-source snapshot.
    Every output image is a separate 1024x1024 uint8 TIFF, retaining the
    original scientific synthesis, intensities, split, and RNG sequences.

Scientific defaults:
    Clean images: 256 train, 48 val, 48 test; four variants per clean image.
    Exactly 15% identity supervision per split x brightness group (rounded).
    Each degraded image: 7 regular charging events, 2 thin bundles,
    2 joint shading events, and 3 horizontal white stripes.
    Artifact template slices: train 0-49, validation 50-54, test 55-59.
    The original seeded procedure, rendering mathematics, and sampling are
    not changed by the release-only engineering/documentation updates.

Environment:
    CPU, NumPy, OpenCV, tifffile and tqdm; no CUDA or MPS required.

Example:
    python step3_synthesize_artifacts.py \
        --style-root /path/to/stage1_style_transfer_4levels \
        --artifact-library-root /path/to/charging_artefact_library_train \
        --charging-statistics-summary /path/to/real_charging_statistics/summary.json \
        --output-root /path/to/stage1_synthesis

    Use --check-only to validate split membership and planned assignments
    without generating images. Existing output directories are not overwritten
    unless --overwrite is supplied; --overwrite deletes the output directory.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


# ============================================================
# 1. Paths
# ============================================================
# Portable defaults. All four independent inputs/outputs can be set via CLI.
# Stage-1 Step 1 already defines the biological train/val/test split.
DEFAULT_DATA_ROOT = Path("data")
STYLE_ROOT = DEFAULT_DATA_ROOT / "stage1_style_transfer_4levels"
STEP1_METADATA_PATH = STYLE_ROOT / "metadata.csv"
ARTEFACT_LIBRARY_ROOT = DEFAULT_DATA_ROOT / "charging_artefact_library_train"
STEP0_SUMMARY_PATH = DEFAULT_DATA_ROOT / "real_charging_statistics" / "summary.json"
OUTPUT_ROOT = DEFAULT_DATA_ROOT / "stage1_synthesis"
EXPERIMENT_DESCRIPTION = OUTPUT_ROOT / "experiment.txt"

# Descriptive release metadata, not scientific configuration.
SYNTHESIS_EXPERIMENT = "stage1_artifact_synthesis"
CODE_VERSION = "2026-10-08"

SPLITS = ("train", "val", "test")
BACKGROUND_LEVELS = (115, 145, 175, 200)
BACKGROUND_TAGS = {
    level: f"bg_{level}"
    for level in BACKGROUND_LEVELS
}

EXPECTED_SOURCE_COUNTS = {
    "train": 256,
    "val": 48,
    "test": 48,
}


# ============================================================
# 2. Dataset
# ============================================================
IMAGE_SIZE = 1024
MASTER_SEED = 123

VARIANTS_PER_CLEAN = 4

# Real artifact-library split. Biological source split and artifact-template
# split are both isolated across train / val / test.
ARTEFACT_TRAIN_SLICE_START = 0
ARTEFACT_TRAIN_SLICE_END = 49
ARTEFACT_VAL_SLICE_START = 50
ARTEFACT_VAL_SLICE_END = 54
ARTEFACT_TEST_SLICE_START = 55
ARTEFACT_TEST_SLICE_END = 59

# Sample-level CPU parallelism. OpenCV internal threading is set to 1
# to avoid oversubscription when several synthesis samples run concurrently.
DEFAULT_NUM_WORKERS = min(4, max(1, os.cpu_count() or 1))


# ============================================================
# 3. Charging-event schedule
# ============================================================
NUM_EVENTS_PER_SLICE = 11

# Native 1024-pixel synthesis schedule:
#   7 regular separate
#   2 thin-bundle
#   2 joint shading
# = 11 charging-related events per degraded 1024 slice
# White-stripe objects are handled separately below (exactly 3 per degraded slice).
NUM_REGULAR_SEPARATE_EVENTS = 7
NUM_THIN_BUNDLE_EVENTS = 2
NUM_SEPARATE_EVENTS = NUM_REGULAR_SEPARATE_EVENTS + NUM_THIN_BUNDLE_EVENTS
NUM_JOINT_EVENTS = 2

if NUM_SEPARATE_EVENTS + NUM_JOINT_EVENTS != NUM_EVENTS_PER_SLICE:
    raise ValueError("Charging-event counts must sum to NUM_EVENTS_PER_SLICE.")

ADDITIVE_FRACTION = 0.0

RANDOM_POSITION_MARGIN = 64
MIN_ANCHOR_DISTANCE = 115
ANCHOR_CANDIDATE_COUNT = 96


# ============================================================
# 4. Structure reference for clean-core synthesis ONLY
#    NOT applied to the final clean target.
# ============================================================
HPF_SIGMA = 2.0
HPF_GAIN = 3.0

# Extra gain is applied only to high-confidence structure through gate^2.
# Weak responses therefore stay close to the current noise-safe baseline, while
# strong membrane responses can reach an effective gain of 3.0 + 1.5 = 4.5.
HPF_STRONG_EXTRA_GAIN = 1.5

# Extract HPF detail from an edge-preserving bilateral structure layer. Its
# range smoothing rejects weak pixel-scale fluctuations while retaining the
# stronger membrane transitions that a Gaussian preblur can soften.
HPF_BILATERAL_DIAMETER = 7
HPF_BILATERAL_SIGMA_COLOR = 12.0
HPF_BILATERAL_SIGMA_SPACE = 2.0

# A soft gate then keeps coherent/strong structure detail while flat-background
# detail remains close to zero.
HPF_GATE_LOW = 1.5
HPF_GATE_HIGH = 4.0
HPF_GATE_SMOOTH_SIGMA = 0.8


# ============================================================
# 6. Spatial scaling
# ============================================================
CLEAN_PIXEL_SIZE_NM = 8.0
REFERENCE_CLEAN_PIXEL_SIZE_NM = 5.0
REFERENCE_ARTEFACT_SCALE = 2.2
# Base physical scale is 2.2 * 5 / 8 = 1.375. Starting the random multiplier
# at 0.7273 makes the final spatial scale start at ~1.0 instead of shrinking
# already-small artefacts. Final spatial scale is therefore about 1.0-1.65x.
RANDOM_ARTEFACT_SCALE_RANGE = (0.7273, 1.2)

# Joint events are enlarged again after the current resize + tail stretch.
JOINT_EXTRA_SCALE_RANGE = (2.0, 3.0)


# ============================================================
# 7. Tail length
# ============================================================
# Maximum local X-stretch at the tail root. The actual stretch decays
# continuously with distance from the core and reaches 1x at the distal end.
# This prevents a very short distal tail from being expanded to the width of the
# whole core+tail template.
TAIL_LENGTH_SCALE_RANGE = (4.0, 8.0)
TAIL_STRETCH_DECAY_POWER = 1.0
TAIL_ROOT_WIDTH_TO_CORE_WIDTH = 0.25
TAIL_ROOT_WIDTH_MIN = 2
TAIL_ROOT_WIDTH_MAX = 24
MAX_ARTEFACT_WIDTH = 700


# ============================================================
# 8. Core appearance
# ============================================================
CORE_MASK_THRESHOLD = 128

CORE_TARGET_MEDIAN_LIMITS = (35.0, 70.0)
CORE_TARGET_MEDIAN_JITTER = 4.0
CORE_CONTRAST_JITTER = 0.05

# Clean core no longer reuses the degraded core interior. Its interior texture is
# sampled from low-structure regions of the real clean HPF-difference image, so it
# follows the same background texture style without copying hidden biological content.
CORE_CLEAN_RING_WIDTH = 8

# Difference-derived background texture.
# Strong membrane responses are excluded when choosing a source patch. Among many
# random candidates, the patch with the smallest strong-structure fraction is used.
CORE_BG_DIFF_STRUCTURE_THRESHOLD = 2.0
CORE_BG_TEXTURE_CANDIDATES = 64
CORE_BG_TEXTURE_GAIN = 1.0
CORE_BG_TEXTURE_CLIP_PERCENTILE = 99.0

# Membrane darkness is still calibrated from real negative HPF responses.
CORE_MEMBRANE_HPF_NEGATIVE_THRESHOLD = 2.0
CORE_MEMBRANE_LOCAL_BG_SIGMA = 3.0
CORE_MEMBRANE_CONTRAST_PERCENTILE = 75.0
CORE_MEMBRANE_DARKEN_SCALE = 1.25
CORE_MEMBRANE_DARKEN_LIMITS = (10.0, 36.0)

# Uniform clean-core membrane.
# Every membrane pixel within one core receives exactly the same gray value before
# the final clean HPF. No distance-profile fading and no Gaussian edge smoothing.
CORE_MEMBRANE_WIDTH_PX = 4.0
CORE_MEMBRANE_DARKEN_FACTOR = 0.68


# ============================================================
# 9. Tail appearance
# ============================================================
TAIL_RATIO_JITTER = 0.08
TAIL_RATIO_LIMITS = (0.30, 0.70)

TAIL_DARKEN_FACTOR = 1.40
TAIL_TARGET_DELTA_LIMITS = (60.0, 100.0)
TAIL_MAX_CORE_DELTA_FRACTION = 0.90
TAIL_STRENGTH_PERCENTILE = 90.0
TAIL_VALUE_THRESHOLD = 1.0

# Joint core+tail represents shallow shading, not a dark charging core.
# Separate-core target intensity is 35-70; newly rendered joint pixels are
# therefore kept at 85 or brighter so the two appearance ranges cannot cross.
JOINT_TARGET_DELTA_RANGE = (50.0, 58.0)
JOINT_STRENGTH_PERCENTILE = 90.0
JOINT_PIXEL_DELTA_MAX = 62.0
JOINT_MIN_REMAINING_FRACTION = 0.35


# ============================================================
# 10. Black thin-bundle charging morphology
# ============================================================
# Stronger morphology diversity than the earlier attempt:
# tiny / small / medium / large bundles can all appear, including very wide fields.
THIN_BUNDLE_SIZE_NAMES = ("tiny", "small", "medium")
THIN_BUNDLE_SIZE_PROBS = (0.15, 0.65, 0.20)
THIN_BUNDLE_LENGTH_RANGES = (
    (40, 120),
    (100, 260),
    (240, 420),
)
THIN_BUNDLE_LINE_COUNT_RANGES = (
    (8, 22),
    (20, 40),
    (32, 64),
)

# Lines are densely stacked in Y and remain very thin.
THIN_BUNDLE_LINE_SPACING_RANGE = (1, 4)
THIN_BUNDLE_LINE_THICKNESS_RANGE = (1, 2)

# Thin-bundle lines are strictly horizontal.

# Neighboring lines have correlated start/end positions and lengths, but the
# bundle no longer assumes one common center. Left and right extents drift
# separately, so the whole bundle can look clearly off-centered.
THIN_BUNDLE_LENGTH_VARIATION_FRACTION = 0.38
THIN_BUNDLE_CENTER_DRIFT_FRACTION = 0.35
THIN_BUNDLE_SIDE_ASYMMETRY_FRACTION = 0.45
THIN_BUNDLE_EDGE_SMOOTH_SIGMA_LINES = 2.2

# Horizontal line endpoints fade only along X.
THIN_BUNDLE_END_TAPER_RANGE = (4, 22)

# One smooth darker core-like zone lives inside each bundle. The surrounding line
# field remains visible but weaker.
THIN_BUNDLE_OUTER_STRENGTH_RANGE = (0.10, 0.26)
THIN_BUNDLE_CORE_SIGMA_X_FRACTION_RANGE = (0.08, 0.26)
THIN_BUNDLE_CORE_SIGMA_Y_FRACTION_RANGE = (0.14, 0.40)
THIN_BUNDLE_CORE_CENTER_X_FRACTION_RANGE = (0.10, 0.90)
THIN_BUNDLE_CORE_CENTER_Y_FRACTION_RANGE = (0.18, 0.82)

# Peak darkness is intentionally the same range as the current separate tail.
THIN_BUNDLE_TARGET_DELTA_RANGE = TAIL_TARGET_DELTA_LIMITS
THIN_BUNDLE_FIELD_SCALE = 100.0
THIN_BUNDLE_PAD = 20

# Thin-bundle placement only:
# Keep the existing random-maximin anchor philosophy, but reject a candidate
# position when more than 20% of the ACTUAL rendered bundle support would
# overlap dark artifact support that is already present on this slice.
#
# This does NOT forbid overlap. It only prevents the pathological case where
# most of a thin bundle lands on top of existing charging artifacts.
THIN_BUNDLE_MAX_OVERLAP_FRACTION = 0.20
THIN_BUNDLE_PLACEMENT_TRIES = 30


# ============================================================
# 11. Two-pass renderer
# ============================================================
TWO_PASS_MAP_STRENGTH_RANGE = (2.4, 3.0)
TWO_PASS_SOFT_SIGMA_RANGE = (0.4, 0.8)
TWO_PASS_GAMMA_RANGE = (0.7, 0.9)
TWO_PASS_STRUCTURE_BLEND_RANGE = (0.08, 0.15)
TWO_PASS_MIN_REMAINING_FRACTION = 0.25
TWO_PASS_FOURIER_CUTOFF_RANGE = (0.03, 0.05)
TWO_PASS_SPATIAL_BLUR_RANGE = (1.0, 2.0)
TWO_PASS_FFT_BLEND_RANGE = (0.65, 0.80)

# Fuse overlapping dark residuals with a smooth maximum. At equal darkness,
# temperature=4 adds only ~2.8 gray levels, avoiding a dark overlap stripe while
# removing the ownership seam between separate tail and joint shading.
DARK_UNION_SMOOTHMAX_TEMPERATURE = 4.0


# ============================================================
# 12. White halo
# ============================================================
# Keep halo brightness restrained; enlarge only its spatial footprint.
HALO_DELTA_LIMITS = (45.0, 62.0)
HALO_DELTA_JITTER = 4.0
HALO_OUTPUT_CEILING = 236.0

# Wider vertical spread above / below the core-tail support.
# Old: ratio=(0.50, 0.80), max=16 px.
HALO_SIGMA_TO_SUPPORT_HEIGHT_RANGE = (0.85, 1.25)
HALO_SIGMA_MIN = 5.0
HALO_SIGMA_MAX = 28.0

# Only mild horizontal smoothing: halo should still follow the tail rather
# than becoming a broad isotropic glow.
HALO_X_SMOOTH_SIGMA = 2.0

# Slightly lower gamma makes the weak outer halo remain visible farther away
# without increasing the peak brightness.
HALO_GAMMA_RANGE = (0.55, 0.72)


# ============================================================
# 13. Clean identity supervision
# ============================================================
# Preserve the previous best dataset composition: 15% no-artefact samples.
# Clean-identity targets are exact identity supervision.
# The fraction is still enforced independently inside train and validation.
CLEAN_IDENTITY_FRACTION = 0.15


# ============================================================
# 14. Misc
# ============================================================
MAX_EVENT_TRIES = 30
MAX_ANCHOR_PLACEMENT_TRIES = 30

# Placement retry budget for regular separate events:
# With 7 regular-separate events on a 1024 canvas, the unchanged strict rule
# ("the new separate core must not overlap any existing dark artifact support")
# can occasionally fail to find a legal anchor within only 30 random-maximin
# attempts even when legal empty space still exists.
#
# Keep the SAME anchor sampler and SAME acceptance/rejection rule; only give
# regular-separate events more chances before declaring synthesis failure.
REGULAR_SEPARATE_ANCHOR_PLACEMENT_TRIES = 120

SAVE_PREVIEWS = True
NUM_PREVIEWS_PER_BACKGROUND = 8
PREVIEW_SIZE = 1024

# All four background groups use three artifact contrast factors.
# Dark annotated core is EXCLUDED from amplification.
# White halo / all positive residual is also EXCLUDED from amplification.
# Only NON-CORE DARK artefacts are amplified.
# Brighter backgrounds receive wider/stronger factor ranges.
# There is no global safe limiter.
# We apply the requested factor directly to non-core dark residuals,
# then clip to [0, 255].
ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND = {
    115: (1.0, 1.15, 1.30),
    145: (1.0, 1.15, 1.30),
    175: (1.0, 1.30, 1.50),
    200: (1.0, 1.35, 1.70),
}


# ============================================================
# 14b. Horizontal white-stripe family
# ============================================================
# Horizontal stripe morphologies retained from the accepted training definition.
#
# IMPORTANT:
# - Bright and dark stripe bands use the accepted paired rendering.
# - No custom overlap mask / overlap rejection is added here.
# - Placement uses the existing random-maximin sample_anchor() logic and the
#   already-used artefact anchor list from the current synthetic slice.
# - Applied only to degraded samples after existing artefact amplification.
# - Clean target is never modified.

NEW_WHITE_STRIPE_SLICE_PROB = 1.00

# Fixed to exactly 3 white stripes per degraded 1024 slice after visual tuning.
NEW_WHITE_STRIPE_COUNT_RANGE = (3, 3)

NEW_WHITE_STRIPE_SUBTYPE_PROBS = {
    "broad_double_region_seam": 0.55,
    "alternating_narrow_band": 0.45,
}

# Broad type: straight horizontal seam + large shallow dark regions on both sides.
# Preserve the old 2048-FOV physical/pixel length distribution:
# old half-length fraction 0.12-0.34 of 2048 -> about 246-696 px.
# Values longer than the 1024 field are naturally clipped by the image boundary.
BROAD_HALF_LENGTH_RANGE_PX = (246, 696)
BROAD_SIDE_WIDTH_RANGE_PX = (60, 120)
BROAD_SEAM_HALF_WIDTH_RANGE_PX = (3, 6)
BROAD_SIDE_DARK_DELTA_RANGE = (6.0, 10.0)
BROAD_SEAM_BRIGHT_DELTA_RANGE = (8.0, 18.0)
BROAD_SIDE_PLATEAU_FRAC = 0.80
BROAD_X_PLATEAU_FRAC = 0.82
BROAD_SEAM_SOFT_SIGMA_RANGE = (1.5, 3.0)

# Alternating type: shorter groups of narrow horizontal dark/bright bands.
# Preserve old 2048-FOV half-length: 0.05-0.18 of 2048 -> about 102-369 px.
ALT_HALF_LENGTH_RANGE_PX = (102, 369)
ALT_BAND_COUNT_RANGE = (4, 7)
ALT_STRIPE_HALF_WIDTH_RANGE_PX = (2, 5)
ALT_BAND_GAP_RANGE_PX = (2, 8)
ALT_BASE_DARK_DELTA_RANGE = (4.0, 9.0)
ALT_BASE_BRIGHT_DELTA_RANGE = (4.5, 10.5)
ALT_LENGTH_JITTER_PX = 40
ALT_X_JITTER_PX = 10
ALT_Y_JITTER_PX = 6
ALT_SOFT_SIGMA_RANGE = (1.5, 3.5)


class PlacementBlockedError(RuntimeError):
    pass


# ============================================================
# 15. General and data-preparation helpers
# ============================================================
def ensure_dir(path):
    """Create an output directory and its missing parent directories."""
    path.mkdir(parents=True, exist_ok=True)


def find_tifs(folder):
    """Return sorted unique TIFF paths within one directory."""
    out = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        out.extend(folder.glob(pattern))
    return sorted(set(out))


def read_tif(path):
    """Read a TIFF image without changing its dtype or pixel values."""
    return tifffile.imread(str(path))


def save_tif(path, image):
    """Write a TIFF image using grayscale photometric metadata."""
    tifffile.imwrite(str(path), image, photometric="minisblack")


def to_uint8(image):
    """Round, clip to [0, 255], and convert an array to uint8."""
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def positive_percentile(values, percentile):
    """Compute a percentile over finite, strictly positive values."""
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def normalize_positive(image):
    """Clip to positive values and normalize by the maximum."""
    image = np.clip(image.astype(np.float32), 0.0, None)
    maximum = float(image.max())
    if maximum <= 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return image / maximum


def get_bbox(mask):
    """Find the inclusive-exclusive bounding box of nonzero mask pixels."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def stable_seed(*parts):
    """Stable 32-bit seed independent of Python's randomized hash()."""
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_style_source_records():
    """
    Read the Stage-1 Step-1 metadata and preserve its train/val/test assignment
    exactly. No source is re-split here.
    """
    if not STEP1_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Step-1 metadata not found: {STEP1_METADATA_PATH}"
        )

    required_columns = {
        "split",
        "block_name",
        "source_filename",
        "target_level",
        "target_median",
    }

    records = []
    seen_paths = set()

    with open(STEP1_METADATA_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        missing = required_columns - fieldnames
        if missing:
            raise RuntimeError(
                "Step-1 metadata is missing required columns: "
                f"{sorted(missing)}\n"
                f"Metadata: {STEP1_METADATA_PATH}"
            )

        for row in reader:
            split = row["split"].strip()
            if split not in SPLITS:
                raise ValueError(
                    f"Unexpected Step-1 split '{split}' in metadata."
                )

            block_name = row["block_name"].strip()
            source_filename = row["source_filename"].strip()
            bg_level = int(round(float(row["target_level"])))

            if bg_level not in BACKGROUND_LEVELS:
                raise ValueError(
                    f"Unsupported target level {bg_level}: {row}"
                )

            # Step-1 output naming is deterministic:
            # <block_name>__<original source TIFF filename>
            styled_name = f"{block_name}__{source_filename}"
            styled_path = (
                STYLE_ROOT
                / split
                / f"level_{bg_level}"
                / styled_name
            )

            if not styled_path.exists():
                raise FileNotFoundError(
                    "Styled source TIFF listed by metadata is missing:\n"
                    f"{styled_path}"
                )

            if styled_path in seen_paths:
                raise RuntimeError(
                    f"Duplicate styled source in metadata: {styled_path}"
                )
            seen_paths.add(styled_path)

            img = read_tif(styled_path)
            if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}: "
                    f"{styled_path}, got {img.shape}"
                )
            if img.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8: {styled_path}, got {img.dtype}"
                )

            records.append({
                "path": styled_path,
                "split": split,
                "block_name": block_name,
                "source_filename": source_filename,
                "styled_filename": styled_name,
                "target_p50": float(row["target_median"]),
                "background_level": bg_level,
                "background_tag": BACKGROUND_TAGS[bg_level],
                "step1_brightness_seed": row.get("brightness_seed", ""),
                "step1_reference_seed": row.get("reference_seed", ""),
                "step1_reference_slice_index": row.get("reference_slice_index", ""),
            })

    if not records:
        raise RuntimeError(
            f"No rows found in Step-1 metadata: {STEP1_METADATA_PATH}"
        )

    # Stable ordering independent of filesystem traversal.
    records.sort(
        key=lambda r: (
            SPLITS.index(r["split"]),
            r["block_name"],
            r["source_filename"],
        )
    )

    # Strict expected source counts from the frozen 16/3/3 block split.
    for split in SPLITS:
        split_records = [r for r in records if r["split"] == split]
        expected = EXPECTED_SOURCE_COUNTS[split]

        if len(split_records) != expected:
            raise RuntimeError(
                f"{split}: expected {expected} Step-1 source slices, "
                f"found {len(split_records)}."
            )

        # Four brightness groups must remain exactly balanced.
        expected_per_level = expected // len(BACKGROUND_LEVELS)
        for level in BACKGROUND_LEVELS:
            count = sum(
                int(r["background_level"]) == int(level)
                for r in split_records
            )
            if count != expected_per_level:
                raise RuntimeError(
                    f"{split}/level_{level}: expected "
                    f"{expected_per_level} slices, found {count}."
                )

    return records


def apply_noncore_signed_delta_amplification(
    base_u8,
    image_u8,
    factor,
    core_mask,
):
    """Amplify only negative non-core pixel differences against the base image."""
    base_f = base_u8.astype(np.float32)
    image_f = image_u8.astype(np.float32)
    delta = image_f - base_f
    core_mask = core_mask.astype(bool)

    amplified_delta = delta.copy()

    # Only NON-CORE DARK artifacts are amplified.
    noncore_dark_mask = (delta < 0.0) & (~core_mask)
    amplified_delta[noncore_dark_mask] = factor * delta[noncore_dark_mask]

    # White halo / all positive residual stays unchanged.
    amplified = base_f + amplified_delta
    return to_uint8(amplified)


def choose_target_factor_for_background(background_level, rng):
    """Select one contrast multiplier from the active background group."""
    background_level = int(background_level)

    if background_level not in ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND:
        raise KeyError(
            f"Unsupported background level for artefact contrast policy: "
            f"{background_level}"
        )

    candidates = ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND[background_level]
    target_factor = float(
        candidates[
            int(rng.integers(0, len(candidates)))
        ]
    )

    return target_factor, tuple(float(x) for x in candidates)


def build_clean_identity_keys(source_records):
    """
    Select clean-identity variants independently within each
    split x brightness group, using a stable group-specific RNG.
    """
    chosen_keys = set()
    stats = {}

    for split_name in SPLITS:
        for bg_level in BACKGROUND_LEVELS:
            keys = [
                (source["path"], variant)
                for source in source_records
                if source["split"] == split_name
                and int(source["background_level"]) == int(bg_level)
                for variant in range(VARIANTS_PER_CLEAN)
            ]

            keys = sorted(
                keys,
                key=lambda x: (str(x[0]), int(x[1])),
            )

            group_seed = stable_seed(
                MASTER_SEED,
                "clean_identity",
                split_name,
                bg_level,
            )
            rng = np.random.default_rng(group_seed)

            count = int(round(len(keys) * CLEAN_IDENTITY_FRACTION))
            count = min(max(count, 0), len(keys))

            if count > 0:
                selected_idx = rng.choice(
                    len(keys),
                    size=count,
                    replace=False,
                )
                for idx in selected_idx:
                    chosen_keys.add(keys[int(idx)])

            stats[f"{split_name}_bg_{bg_level}"] = {
                "num_candidates": int(len(keys)),
                "num_clean_identity": int(count),
                "fraction": (
                    float(count) / float(len(keys))
                    if keys else 0.0
                ),
                "seed": int(group_seed),
            }

    return chosen_keys, stats


def build_balanced_factor_assignments(
    source_records,
    clean_identity_keys,
):
    """
    Build strict / near-strict factor balance independently inside every
    split x background group. Test is isolated exactly like train and val.
    """
    assignments = {}
    stats = {}

    for split_name in SPLITS:
        for bg_level in BACKGROUND_LEVELS:
            keys = [
                (source["path"], variant)
                for source in source_records
                if source["split"] == split_name
                and int(source["background_level"]) == int(bg_level)
                for variant in range(VARIANTS_PER_CLEAN)
                if (source["path"], variant) not in clean_identity_keys
            ]

            keys = sorted(
                keys,
                key=lambda x: (str(x[0]), int(x[1])),
            )

            candidates = tuple(
                float(x)
                for x in ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND[int(bg_level)]
            )

            group_seed = stable_seed(
                MASTER_SEED,
                "contrast_factor",
                split_name,
                bg_level,
            )
            rng = np.random.default_rng(group_seed)

            # Randomize which source gets which factor, but preserve near-equal counts.
            permutation = rng.permutation(len(keys)) if keys else np.asarray([], dtype=np.int64)
            shuffled_keys = [keys[int(i)] for i in permutation]

            n = len(shuffled_keys)
            q, r = divmod(n, len(candidates))

            factor_list = []
            for factor in candidates:
                factor_list.extend([factor] * q)

            if r > 0:
                remainder_order = list(candidates)
                rng.shuffle(remainder_order)
                factor_list.extend(remainder_order[:r])

            rng.shuffle(factor_list)

            if len(factor_list) != len(shuffled_keys):
                raise RuntimeError(
                    f"Balanced factor assignment size mismatch: "
                    f"split={split_name}, bg={bg_level}, "
                    f"keys={len(shuffled_keys)}, factors={len(factor_list)}"
                )

            for key, factor in zip(shuffled_keys, factor_list):
                assignments[key] = float(factor)

            counts = {
                str(factor): int(sum(abs(x - factor) < 1e-8 for x in factor_list))
                for factor in candidates
            }

            if counts:
                count_values = list(counts.values())
                if max(count_values) - min(count_values) > 1:
                    raise RuntimeError(
                        f"Factor balancing failed: split={split_name}, "
                        f"bg={bg_level}, counts={counts}"
                    )

            stats[f"{split_name}_bg_{bg_level}"] = {
                "num_degraded_samples": int(n),
                "factor_counts": counts,
                "seed": int(group_seed),
            }

    return assignments, stats


def amplify_slice_artefacts(
    base_u8,
    charging_u8,
    noisy_u8,
    rng,
    background_level,
    core_mask,
    forced_target_factor=None,
):
    """Apply a chosen contrast factor and assemble diagnostics."""
    if forced_target_factor is None:
        target_factor, candidate_factors = choose_target_factor_for_background(
            background_level,
            rng,
        )
    else:
        target_factor = float(forced_target_factor)
        candidate_factors = tuple(
            float(x)
            for x in ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND[int(background_level)]
        )

    target_factor = max(1.0, float(target_factor))
    effective_factor = target_factor
    core_mask = core_mask.astype(bool)

    charging_amp = apply_noncore_signed_delta_amplification(
        base_u8,
        charging_u8,
        effective_factor,
        core_mask,
    )
    noisy_amp = apply_noncore_signed_delta_amplification(
        base_u8,
        noisy_u8,
        effective_factor,
        core_mask,
    )

    noisy_delta_before = noisy_u8.astype(np.float32) - base_u8.astype(np.float32)
    noisy_delta_after = noisy_amp.astype(np.float32) - base_u8.astype(np.float32)

    negative_before = noisy_delta_before < 0.0
    positive_before = noisy_delta_before > 0.0
    core_negative_before = negative_before & core_mask
    noncore_negative_before = negative_before & (~core_mask)

    negative_after = noisy_delta_after < 0.0
    positive_after = noisy_delta_after > 0.0
    core_negative_after = negative_after & core_mask
    noncore_negative_after = negative_after & (~core_mask)

    meta = {
        "background_level": int(background_level),
        "candidate_factors": [float(x) for x in candidate_factors],
        "amplification_enabled": int(max(candidate_factors) > 1.0),
        "core_amplification_enabled": 0,
        "noncore_dark_amplification_enabled": int(max(candidate_factors) > 1.0),
        "bright_halo_amplification_enabled": 0,
        "target_factor": float(target_factor),
        "effective_factor": float(effective_factor),
        "limited_by_headroom": 0,
        "max_factor_charging": None,
        "max_factor_noisy": None,
        "safe_limit_after_margin": None,
        "dark_floor": None,
        "bright_ceiling": None,
        "core_negative_pixels_before": int(np.sum(core_negative_before)),
        "noncore_negative_pixels_before": int(np.sum(noncore_negative_before)),
        "positive_delta_pixels_before": int(np.sum(positive_before)),
        "core_negative_pixels_after": int(np.sum(core_negative_after)),
        "noncore_negative_pixels_after": int(np.sum(noncore_negative_after)),
        "positive_delta_pixels_after": int(np.sum(positive_after)),
        "max_core_negative_delta_before": (
            float(np.min(noisy_delta_before[core_negative_before]))
            if np.any(core_negative_before) else 0.0
        ),
        "max_noncore_negative_delta_before": (
            float(np.min(noisy_delta_before[noncore_negative_before]))
            if np.any(noncore_negative_before) else 0.0
        ),
        "max_positive_delta_before": (
            float(np.max(noisy_delta_before[positive_before]))
            if np.any(positive_before) else 0.0
        ),
        "max_core_negative_delta_after": (
            float(np.min(noisy_delta_after[core_negative_after]))
            if np.any(core_negative_after) else 0.0
        ),
        "max_noncore_negative_delta_after": (
            float(np.min(noisy_delta_after[noncore_negative_after]))
            if np.any(noncore_negative_after) else 0.0
        ),
        "max_positive_delta_after": (
            float(np.max(noisy_delta_after[positive_after]))
            if np.any(positive_after) else 0.0
        ),
    }

    return charging_amp, noisy_amp, meta


def _normalize_probability_dict(prob_dict):
    """Return ordered keys with normalized sampling probabilities."""
    keys = list(prob_dict.keys())
    probs = np.asarray([float(prob_dict[k]) for k in keys], dtype=np.float64)
    probs /= probs.sum()
    return keys, probs


def _stripe_band_alpha(shape, x0, x1, center_y, half_width, sigma):
    """Horizontal soft band used only by the alternating stripe subtype."""
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    x0 = int(round(np.clip(x0, 0, w - 1)))
    x1 = int(round(np.clip(x1, 0, w - 1)))
    yy = int(round(np.clip(center_y, 0, h - 1)))

    if x1 < x0:
        x0, x1 = x1, x0

    cv2.line(
        canvas,
        (x0, yy),
        (x1, yy),
        255,
        thickness=int(max(1, round(2 * half_width))),
        lineType=cv2.LINE_AA,
    )

    alpha = cv2.GaussianBlur(
        canvas.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    maximum = float(alpha.max())
    if maximum > 1e-8:
        alpha /= maximum

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _make_straight_broad_masks(shape, params):
    """
    Horizontal broad-band morphology.

    Strictly horizontal / separable geometry:
    - X: long support, mostly flat with taper only near the two ends;
    - Y dark side regions: immediately outside the seam, strong for inner 80%,
      fading only over the outer 20%;
    - Y bright seam: thicker central Gaussian-like seam.
    """
    h, w = shape

    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)

    center_x = float(params["center_x"])
    center_y = float(params["center_y"])
    half_length = float(params["half_length"])
    side_width = float(params["side_width"])
    seam_half_width = float(params["seam_half_width"])

    x_norm = np.abs(x - center_x) / max(half_length, 1.0)
    x_env = np.zeros(w, dtype=np.float32)

    x_inside = x_norm <= BROAD_X_PLATEAU_FRAC
    x_taper = (x_norm > BROAD_X_PLATEAU_FRAC) & (x_norm < 1.0)
    x_env[x_inside] = 1.0

    if np.any(x_taper):
        u = (
            x_norm[x_taper] - BROAD_X_PLATEAU_FRAC
        ) / max(1.0 - BROAD_X_PLATEAU_FRAC, 1e-6)
        x_env[x_taper] = 0.5 * (1.0 + np.cos(np.pi * u))

    abs_dy = np.abs(y - center_y)
    distance_from_seam = np.maximum(abs_dy - seam_half_width, 0.0)
    side_t = distance_from_seam / max(side_width, 1.0)

    dark_y = np.zeros(h, dtype=np.float32)
    side_valid = abs_dy >= seam_half_width
    plateau = side_valid & (side_t <= BROAD_SIDE_PLATEAU_FRAC)
    fade = side_valid & (side_t > BROAD_SIDE_PLATEAU_FRAC) & (side_t < 1.0)

    dark_y[plateau] = 1.0

    if np.any(fade):
        u = (
            side_t[fade] - BROAD_SIDE_PLATEAU_FRAC
        ) / max(1.0 - BROAD_SIDE_PLATEAU_FRAC, 1e-6)
        dark_y[fade] = 0.5 * (1.0 + np.cos(np.pi * u))

    seam_sigma = max(
        1.2,
        seam_half_width + float(params["seam_soft_sigma"]),
    )
    seam_y = np.exp(
        -0.5 * ((y - center_y) / seam_sigma) ** 2
    ).astype(np.float32)
    seam_y[abs_dy > seam_sigma * 2.4] = 0.0

    dark_alpha = (dark_y[:, None] * x_env[None, :]).astype(np.float32)
    seam_alpha = (seam_y[:, None] * x_env[None, :]).astype(np.float32)

    return (
        np.clip(dark_alpha, 0.0, 1.0).astype(np.float32),
        np.clip(seam_alpha, 0.0, 1.0).astype(np.float32),
    )


def _sample_new_white_stripe_object(shape, anchor, rng):
    """Sample a horizontal stripe object around an existing-style anchor."""
    h, w = shape
    anchor_y, anchor_x = anchor

    keys, probs = _normalize_probability_dict(NEW_WHITE_STRIPE_SUBTYPE_PROBS)
    subtype = str(rng.choice(keys, p=probs))

    if subtype == "broad_double_region_seam":
        return {
            "subtype": subtype,
            "center_x": float(anchor_x),
            "center_y": float(anchor_y),
            "half_length": int(
                rng.integers(
                    BROAD_HALF_LENGTH_RANGE_PX[0],
                    BROAD_HALF_LENGTH_RANGE_PX[1] + 1,
                )
            ),
            "side_width": int(
                rng.integers(BROAD_SIDE_WIDTH_RANGE_PX[0], BROAD_SIDE_WIDTH_RANGE_PX[1] + 1)
            ),
            "seam_half_width": int(
                rng.integers(BROAD_SEAM_HALF_WIDTH_RANGE_PX[0], BROAD_SEAM_HALF_WIDTH_RANGE_PX[1] + 1)
            ),
            "side_dark_delta": float(rng.uniform(*BROAD_SIDE_DARK_DELTA_RANGE)),
            "seam_bright_delta": float(rng.uniform(*BROAD_SEAM_BRIGHT_DELTA_RANGE)),
            "seam_soft_sigma": float(rng.uniform(*BROAD_SEAM_SOFT_SIGMA_RANGE)),
        }

    if subtype == "alternating_narrow_band":
        return {
            "subtype": subtype,
            "center_x": float(anchor_x),
            "center_y": float(anchor_y),
            "half_length": int(
                rng.integers(
                    ALT_HALF_LENGTH_RANGE_PX[0],
                    ALT_HALF_LENGTH_RANGE_PX[1] + 1,
                )
            ),
            "band_count": int(
                rng.integers(ALT_BAND_COUNT_RANGE[0], ALT_BAND_COUNT_RANGE[1] + 1)
            ),
            "base_dark_delta": float(rng.uniform(*ALT_BASE_DARK_DELTA_RANGE)),
            "base_bright_delta": float(rng.uniform(*ALT_BASE_BRIGHT_DELTA_RANGE)),
            "soft_sigma": float(rng.uniform(*ALT_SOFT_SIGMA_RANGE)),
            # Rendering can be repeated deterministically for metadata / pair application.
            "shape_seed": int(rng.integers(0, 2**31 - 1)),
        }

    raise ValueError(f"Unknown new white-stripe subtype: {subtype}")


def _render_new_white_stripe_object(shape, params):
    """Render one sampled signed stripe field and its support."""
    subtype = params["subtype"]

    if subtype == "broad_double_region_seam":
        dark_alpha, seam_alpha = _make_straight_broad_masks(shape, params)
        delta = (
            - float(params["side_dark_delta"]) * dark_alpha
            + float(params["seam_bright_delta"]) * seam_alpha
        ).astype(np.float32)

        support = np.maximum(dark_alpha, seam_alpha) > 0.06
        info = dict(params)
        info.update({
            "dark_mean_alpha": float(dark_alpha.mean()),
            "seam_mean_alpha": float(seam_alpha.mean()),
            "support_pixels": int(np.sum(support)),
            "side_plateau_fraction": float(BROAD_SIDE_PLATEAU_FRAC),
            "x_plateau_fraction": float(BROAD_X_PLATEAU_FRAC),
        })
        return delta, support, info

    if subtype == "alternating_narrow_band":
        rng = np.random.default_rng(int(params["shape_seed"]))
        h, w = shape
        delta = np.zeros((h, w), dtype=np.float32)
        support_acc = np.zeros((h, w), dtype=np.float32)

        band_count = int(params["band_count"])
        center_y = float(params["center_y"])
        center_x = float(params["center_x"])
        base_half_length = int(params["half_length"])

        half_widths = [
            int(rng.integers(ALT_STRIPE_HALF_WIDTH_RANGE_PX[0], ALT_STRIPE_HALF_WIDTH_RANGE_PX[1] + 1))
            for _ in range(band_count)
        ]
        gaps = [
            int(rng.integers(ALT_BAND_GAP_RANGE_PX[0], ALT_BAND_GAP_RANGE_PX[1] + 1))
            for _ in range(max(0, band_count - 1))
        ]

        offsets = [0.0]
        cursor = 0.0
        for i in range(1, band_count):
            cursor += half_widths[i - 1] + half_widths[i] + gaps[i - 1]
            offsets.append(cursor)

        mean_offset = 0.5 * (offsets[0] + offsets[-1]) if len(offsets) > 1 else 0.0
        offsets = [off - mean_offset for off in offsets]
        band_infos = []

        for i, (offset, half_width) in enumerate(zip(offsets, half_widths)):
            sign = 1.0 if (i % 2 == 1) else -1.0
            if sign < 0:
                amplitude = float(params["base_dark_delta"]) * float(rng.uniform(0.90, 1.15))
            else:
                amplitude = float(params["base_bright_delta"]) * float(rng.uniform(0.90, 1.15))

            local_half_length = int(max(
                25,
                base_half_length + int(rng.integers(-ALT_LENGTH_JITTER_PX, ALT_LENGTH_JITTER_PX + 1)),
            ))
            local_center_x = center_x + float(rng.integers(-ALT_X_JITTER_PX, ALT_X_JITTER_PX + 1))
            local_center_y = center_y + float(offset) + float(rng.integers(-ALT_Y_JITTER_PX, ALT_Y_JITTER_PX + 1))

            alpha = _stripe_band_alpha(
                shape,
                local_center_x - local_half_length,
                local_center_x + local_half_length,
                local_center_y,
                half_width,
                params["soft_sigma"],
            )

            delta += sign * amplitude * alpha
            support_acc = np.maximum(support_acc, alpha)
            band_infos.append({
                "band_index": int(i),
                "sign": "bright" if sign > 0 else "dark",
                "center_y_offset": float(offset),
                "half_width": int(half_width),
                "half_length": int(local_half_length),
                "amplitude": float(amplitude),
            })

        support = support_acc > 0.06
        info = dict(params)
        info.update({
            "support_pixels": int(np.sum(support)),
            "bands": band_infos,
            "support_mean_alpha": float(support_acc.mean()),
        })
        return delta.astype(np.float32), support, info

    raise ValueError(f"Unknown new white-stripe subtype: {subtype}")


def sample_new_white_stripe_configuration(shape, rng, used_anchors):
    """
    Sample the configured stripe count using the same placement mechanism used
    by the charging artefacts: sample_anchor(rng, used_anchors).

    No new overlap-specific mask / rejection rule is added.
    """
    object_count = int(
        rng.integers(
            NEW_WHITE_STRIPE_COUNT_RANGE[0],
            NEW_WHITE_STRIPE_COUNT_RANGE[1] + 1,
        )
    )

    params_list = []
    for _ in range(object_count):
        anchor = sample_anchor(rng, used_anchors)
        used_anchors.append(anchor)
        params = _sample_new_white_stripe_object(shape, anchor, rng)
        params["anchor_y"] = int(anchor[0])
        params["anchor_x"] = int(anchor[1])
        params_list.append(params)

    return params_list


def build_new_white_stripe_delta(shape, params_list):
    """Accumulate signed deltas and support masks for stripe objects."""
    total_delta = np.zeros(shape, dtype=np.float32)
    support_total = np.zeros(shape, dtype=bool)
    object_infos = []

    for params in params_list:
        delta, support, info = _render_new_white_stripe_object(shape, params)
        total_delta += delta
        support_total |= support
        object_infos.append(info)

    return total_delta, support_total, object_infos


def apply_same_new_white_stripes_to_pair(
    charging_u8,
    noisy_u8,
    rng,
    used_anchors,
):
    """
    Apply the SAME signed white-stripe field to charging and noisy.

    The clean target is intentionally not an argument and cannot be modified.
    """
    if float(rng.random()) >= NEW_WHITE_STRIPE_SLICE_PROB:
        meta = {
            "enabled": 0,
            "count": 0,
            "objects": [],
            "charging_changed_pixels": 0,
            "noisy_changed_pixels": 0,
            "positive_delta_pixels": 0,
            "negative_delta_pixels": 0,
        }
        return charging_u8.copy(), noisy_u8.copy(), meta

    params_list = sample_new_white_stripe_configuration(
        charging_u8.shape,
        rng,
        used_anchors,
    )
    signed_delta, _, object_infos = build_new_white_stripe_delta(
        charging_u8.shape,
        params_list,
    )

    charging_out = to_uint8(charging_u8.astype(np.float32) + signed_delta)
    noisy_out = to_uint8(noisy_u8.astype(np.float32) + signed_delta)

    charging_change = charging_out.astype(np.float32) - charging_u8.astype(np.float32)
    noisy_change = noisy_out.astype(np.float32) - noisy_u8.astype(np.float32)

    meta = {
        "enabled": 1,
        "count": int(len(params_list)),
        "objects": object_infos,
        "charging_changed_pixels": int(np.sum(charging_change != 0.0)),
        "noisy_changed_pixels": int(np.sum(noisy_change != 0.0)),
        "positive_delta_pixels": int(np.sum(signed_delta > 0.0)),
        "negative_delta_pixels": int(np.sum(signed_delta < 0.0)),
        "max_positive_delta": float(np.max(signed_delta)) if signed_delta.size else 0.0,
        "min_negative_delta": float(np.min(signed_delta)) if signed_delta.size else 0.0,
    }

    return charging_out, noisy_out, meta


def build_output_paths_for_background(split, level, name):
    """Construct output paths for one split, brightness level, and sample."""
    split = str(split)
    level = int(level)

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if level not in BACKGROUND_LEVELS:
        raise ValueError(f"Unknown background level: {level}")

    root = OUTPUT_ROOT / split / BACKGROUND_TAGS[level]

    return {
        "clean": root / "clean" / name,
        "noisy_charging": root / "noisy_charging" / name,
        "noisy": root / "noisy" / name,
        "preview": root / "preview" / f"{Path(name).stem}__preview.png",
        "core_diagnostic": (
            root
            / "preview_core_membrane_diagnostic"
            / f"{Path(name).stem}__core_diagnostic.png"
        ),
    }


def relative_to_output_root(path):
    """Convert an output file path to a path relative to the dataset root."""
    return str(path.relative_to(OUTPUT_ROOT))


def build_structure_reference(image):
    """Derive gated, edge-preserving structure used only for clean-core synthesis."""
    image_f = image.astype(np.float32)

    # Same reference transform as the previous best. It is used only for
    # clean-core membrane calibration / texture selection, never as target.
    structure_source = cv2.bilateralFilter(
        image_f,
        d=HPF_BILATERAL_DIAMETER,
        sigmaColor=HPF_BILATERAL_SIGMA_COLOR,
        sigmaSpace=HPF_BILATERAL_SIGMA_SPACE,
        borderType=cv2.BORDER_REFLECT,
    )
    structure_blur = cv2.GaussianBlur(
        structure_source,
        (0, 0),
        sigmaX=HPF_SIGMA,
        sigmaY=HPF_SIGMA,
        borderType=cv2.BORDER_REFLECT,
    )
    structure_detail = structure_source - structure_blur

    strength = np.abs(structure_detail)
    gate = np.clip(
        (strength - HPF_GATE_LOW) / (HPF_GATE_HIGH - HPF_GATE_LOW),
        0.0,
        1.0,
    )
    # Smoothstep avoids a hard threshold; spatial smoothing keeps the enhanced
    # membrane band continuous instead of creating a new gate boundary.
    gate = gate * gate * (3.0 - 2.0 * gate)
    gate = cv2.GaussianBlur(
        gate.astype(np.float32),
        (0, 0),
        sigmaX=HPF_GATE_SMOOTH_SIGMA,
        sigmaY=HPF_GATE_SMOOTH_SIGMA,
        borderType=cv2.BORDER_REFLECT,
    )

    # Keep the existing 3x response for all gated structure, then add extra
    # enhancement only where the gate is already strong. Because gate^2 falls
    # rapidly for weak responses, residual pixel-scale noise is barely changed.
    enhancement_gain = (
        HPF_GAIN * gate
        + HPF_STRONG_EXTRA_GAIN * gate * gate
    )
    return to_uint8(image_f + structure_detail * enhancement_gain)


def estimate_real_membrane_darkening(clean_input, structure_reference):
    """Estimate clean-core membrane darkness from the unchanged structure reference."""
    clean_input_f = clean_input.astype(np.float32)
    reference_f = structure_reference.astype(np.float32)

    hpf_difference = reference_f - clean_input_f
    local_background = cv2.GaussianBlur(
        reference_f,
        (0, 0),
        sigmaX=CORE_MEMBRANE_LOCAL_BG_SIGMA,
        sigmaY=CORE_MEMBRANE_LOCAL_BG_SIGMA,
        borderType=cv2.BORDER_REFLECT,
    )
    local_dark_contrast = np.clip(local_background - reference_f, 0.0, None)

    membrane_mask = (
        (hpf_difference <= -CORE_MEMBRANE_HPF_NEGATIVE_THRESHOLD)
        & (local_dark_contrast > 0.0)
    )
    values = local_dark_contrast[membrane_mask]

    if values.size < 128:
        values = local_dark_contrast[local_dark_contrast > 0.0]

    if values.size == 0:
        raw_contrast = CORE_MEMBRANE_DARKEN_LIMITS[0]
    else:
        raw_contrast = float(np.percentile(values, CORE_MEMBRANE_CONTRAST_PERCENTILE))

    membrane_darkening = float(np.clip(
        raw_contrast * CORE_MEMBRANE_DARKEN_SCALE,
        *CORE_MEMBRANE_DARKEN_LIMITS,
    ))

    return membrane_darkening, {
        "candidate_pixels": int(values.size),
        "raw_contrast_p75": raw_contrast,
        "applied_darkening": membrane_darkening,
    }



def prepare_difference_texture_source(clean_input, structure_reference):
    """Return reference detail and a mask used only for clean-core synthesis."""
    difference = (
        structure_reference.astype(np.float32)
        - clean_input.astype(np.float32)
    )
    strong_structure = np.abs(difference) >= CORE_BG_DIFF_STRUCTURE_THRESHOLD
    return difference, strong_structure


def sample_clean_background_texture(
    clean_input,
    strong_structure,
    shape,
    rng,
):
    """Sample one background-style texture patch from real clean input.

    We use HPF difference only to locate low-structure regions. The actual texture
    comes from the corresponding clean-input patch, so the synthetic core interior
    can inherit the same slow background variation / texture style as real clean
    background, rather than the sparse membrane-only difference image.
    """
    patch_h, patch_w = shape
    image_h, image_w = clean_input.shape

    patch_h = min(patch_h, image_h)
    patch_w = min(patch_w, image_w)

    best = None
    best_score = float("inf")

    for _ in range(CORE_BG_TEXTURE_CANDIDATES):
        top = int(rng.integers(0, image_h - patch_h + 1))
        left = int(rng.integers(0, image_w - patch_w + 1))

        structure_patch = strong_structure[
            top:top + patch_h,
            left:left + patch_w,
        ]
        score = float(np.mean(structure_patch))

        if score < best_score:
            best_score = score
            best = (top, left)

    top, left = best
    texture = clean_input[
        top:top + patch_h,
        left:left + patch_w,
    ].astype(np.float32).copy()
    structure_patch = strong_structure[
        top:top + patch_h,
        left:left + patch_w,
    ]

    # Remove any accidentally retained semantic structure from the source patch.
    # Keep only general background-style variation.
    background_values = texture[~structure_patch]
    neutral = float(np.median(background_values)) if background_values.size else float(np.median(texture))
    texture[structure_patch] = neutral
    texture -= float(np.median(texture))

    abs_values = np.abs(texture[np.isfinite(texture)])
    if abs_values.size:
        limit = float(np.percentile(abs_values, CORE_BG_TEXTURE_CLIP_PERCENTILE))
        if limit > 1e-6:
            texture = np.clip(texture, -limit, limit)

    if bool(rng.integers(0, 2)):
        texture = texture[:, ::-1]
    if bool(rng.integers(0, 2)):
        texture = texture[::-1, :]

    return (texture * CORE_BG_TEXTURE_GAIN).astype(np.float32), best_score


def sample_anchor(rng, used_anchors):
    # Random maximin sampling: draw many legal candidates and choose the one
    # farthest from existing anchors. This keeps randomness but avoids large
    # clusters / empty regions much better than accepting the first random point.
    """Draw a maximin anchor relative to previously used artifact anchors."""
    candidates = np.column_stack([
        rng.integers(
            RANDOM_POSITION_MARGIN,
            IMAGE_SIZE - RANDOM_POSITION_MARGIN,
            size=ANCHOR_CANDIDATE_COUNT,
        ),
        rng.integers(
            RANDOM_POSITION_MARGIN,
            IMAGE_SIZE - RANDOM_POSITION_MARGIN,
            size=ANCHOR_CANDIDATE_COUNT,
        ),
    ]).astype(np.int32)

    if not used_anchors:
        y, x = candidates[int(rng.integers(0, len(candidates)))]
        return int(y), int(x)

    used = np.asarray(used_anchors, dtype=np.float32)
    diff = candidates[:, None, :].astype(np.float32) - used[None, :, :]
    min_dist2 = np.min(np.sum(diff * diff, axis=2), axis=1)

    legal = min_dist2 >= MIN_ANCHOR_DISTANCE ** 2
    if np.any(legal):
        legal_indices = np.where(legal)[0]
        best = legal_indices[int(np.argmax(min_dist2[legal_indices]))]
    else:
        best = int(np.argmax(min_dist2))

    y, x = candidates[best]
    return int(y), int(x)


def sample_thin_bundle_anchor(rng, used_anchors, bundle_shape):
    """Sample a maximin anchor for one already-generated thin bundle.

    Unlike the generic sampler, the legal center range depends on the actual
    bundle footprint. This prevents a large bundle near an image boundary from
    being silently shifted inward by top/left clipping while its recorded anchor
    remains near the edge.

    The maximin policy itself is unchanged:
      1) draw ANCHOR_CANDIDATE_COUNT random legal centers;
      2) prefer centers at least MIN_ANCHOR_DISTANCE from earlier anchors;
      3) choose the candidate with the largest nearest-anchor distance.

    Actual footprint overlap is checked separately in add_thin_bundle_event().
    """
    h, w = (int(bundle_shape[0]), int(bundle_shape[1]))

    if h <= 0 or w <= 0 or h >= IMAGE_SIZE or w >= IMAGE_SIZE:
        raise PlacementBlockedError(
            f"Invalid thin-bundle footprint for {IMAGE_SIZE} canvas: {(h, w)}"
        )

    half_h = h // 2
    half_w = w // 2

    # If top = anchor_y - h//2, requiring 0 <= top <= IMAGE_SIZE-h gives
    # exactly this legal center interval. Apply the existing 64-px margin
    # whenever the bundle footprint still leaves room for it.
    y_low = max(RANDOM_POSITION_MARGIN, half_h)
    y_high = min(
        IMAGE_SIZE - RANDOM_POSITION_MARGIN,
        IMAGE_SIZE - h + half_h,
    )
    x_low = max(RANDOM_POSITION_MARGIN, half_w)
    x_high = min(
        IMAGE_SIZE - RANDOM_POSITION_MARGIN,
        IMAGE_SIZE - w + half_w,
    )

    # A very large footprint can make the old fixed margin impossible. In that
    # case relax ONLY the margin, never the requirement that the entire bundle
    # fits on the 1024 canvas.
    if y_low > y_high:
        y_low = half_h
        y_high = IMAGE_SIZE - h + half_h
    if x_low > x_high:
        x_low = half_w
        x_high = IMAGE_SIZE - w + half_w

    if y_low > y_high or x_low > x_high:
        raise PlacementBlockedError(
            f"No legal thin-bundle center for footprint {(h, w)}"
        )

    candidates = np.column_stack([
        rng.integers(
            int(y_low),
            int(y_high) + 1,
            size=ANCHOR_CANDIDATE_COUNT,
        ),
        rng.integers(
            int(x_low),
            int(x_high) + 1,
            size=ANCHOR_CANDIDATE_COUNT,
        ),
    ]).astype(np.int32)

    if not used_anchors:
        y, x = candidates[int(rng.integers(0, len(candidates)))]
        return int(y), int(x)

    used = np.asarray(used_anchors, dtype=np.float32)
    diff = candidates[:, None, :].astype(np.float32) - used[None, :, :]
    min_dist2 = np.min(np.sum(diff * diff, axis=2), axis=1)

    legal = min_dist2 >= MIN_ANCHOR_DISTANCE ** 2
    if np.any(legal):
        legal_indices = np.where(legal)[0]
        best = legal_indices[int(np.argmax(min_dist2[legal_indices]))]
    else:
        best = int(np.argmax(min_dist2))

    y, x = candidates[best]
    return int(y), int(x)


# ============================================================
# 16. Charging Library Step 1 statistics
# ============================================================
def load_step0_stats():
    """Load charging statistics with the original field names and fallbacks."""
    with open(STEP0_SUMMARY_PATH, "r") as f:
        summary = json.load(f)["aggregated_component_statistics"]

    def stat(name, key="median", fallback=None):
        item = summary.get(name, {})
        value = item.get(key)
        if value is None:
            return fallback
        value = float(value)
        if not np.isfinite(value):
            return fallback
        return value

    core_raw_median = stat("core_raw_median", "median", 50.0)
    core_delta_median = stat("bg_minus_core_median", "median", 130.0)
    tail_dark_p90 = stat("tail_dark_delta_p90", "median", 60.0)
    halo_delta_median = stat("halo_bright_delta_p90", "median", 45.0)

    tail_ratio_median = tail_dark_p90 / max(core_delta_median, 1e-6)
    tail_ratio_p25 = max(TAIL_RATIO_LIMITS[0], 0.85 * tail_ratio_median)
    tail_ratio_p75 = min(TAIL_RATIO_LIMITS[1], 1.15 * tail_ratio_median)

    return {
        "core_raw_median": core_raw_median,
        "core_delta_median": core_delta_median,
        "tail_ratio_median": float(np.clip(tail_ratio_median, *TAIL_RATIO_LIMITS)),
        "tail_ratio_p25": tail_ratio_p25,
        "tail_ratio_p75": tail_ratio_p75,
        "halo_delta_median": halo_delta_median,
    }


# ============================================================
# 17. Load artefact library
# ============================================================
def resolve_entry_dir(row):
    """Resolve an artifact library CSV record to an existing entry folder."""
    p = Path(row["entry_dir"])
    candidates = [p]

    if not p.is_absolute():
        candidates.append(ARTEFACT_LIBRARY_ROOT / p)

    if row.get("entry_name"):
        candidates.append(ARTEFACT_LIBRARY_ROOT / "entries" / row["entry_name"])

    candidates.append(ARTEFACT_LIBRARY_ROOT / "entries" / p.name)

    for q in candidates:
        if q.exists():
            return q

    raise FileNotFoundError(f"Cannot resolve entry_dir: {row['entry_dir']}")


def first_existing(folder, names):
    """Return the first existing file among supported historical filenames."""
    for name in names:
        p = folder / name
        if p.exists():
            return p
    return None


def load_library_index():
    """Load artifact entries and assign them to non-overlapping slice splits."""
    index_path = ARTEFACT_LIBRARY_ROOT / "library_index.csv"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Artifact library index not found: {index_path}"
        )

    out = {
        "train": [],
        "val": [],
        "test": [],
    }

    with open(index_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["slice_idx"])
            record = {
                "entry_id": int(row["entry_id"]),
                "slice_idx": z,
                "group_id": int(row.get("group_id", -1)),
                "entry_dir": resolve_entry_dir(row),
            }

            if ARTEFACT_TRAIN_SLICE_START <= z <= ARTEFACT_TRAIN_SLICE_END:
                out["train"].append(record)
            elif ARTEFACT_VAL_SLICE_START <= z <= ARTEFACT_VAL_SLICE_END:
                out["val"].append(record)
            elif ARTEFACT_TEST_SLICE_START <= z <= ARTEFACT_TEST_SLICE_END:
                out["test"].append(record)

    for split_name in SPLITS:
        if not out[split_name]:
            raise RuntimeError(
                "Bad artifact-library split: "
                + ", ".join(
                    f"{s}={len(out[s])}"
                    for s in SPLITS
                )
            )

    # Strictly prevent one resolved artifact entry from appearing in >1 split.
    owners = {}
    for split_name in SPLITS:
        for record in out[split_name]:
            key = str(record["entry_dir"].resolve())
            previous = owners.get(key)
            if previous is not None and previous != split_name:
                raise RuntimeError(
                    f"Artifact entry leaks across splits: {key} "
                    f"({previous}, {split_name})"
                )
            owners[key] = split_name

    return out


def load_entry(record, cache):
    """Load, validate semantically, and cache one charging core/tail template."""
    key = str(record["entry_dir"])
    if key in cache:
        return cache[key]

    folder = record["entry_dir"]

    core_path = first_existing(folder, ("core_uint8.tif", "pure_core_uint8.tif"))
    mask_path = first_existing(folder, ("core_mask_uint8.tif", "core_mask.tif"))
    tail_path = first_existing(folder, ("tail_residual_float32.tif", "residual_float32.tif"))
    raw_path = first_existing(folder, ("raw_crop_uint8.tif", "raw_crop_padded_uint8.tif"))

    if core_path is None or mask_path is None or tail_path is None:
        raise FileNotFoundError(f"Incomplete artefact entry: {folder}")

    core = read_tif(core_path).astype(np.uint8)
    core_mask = read_tif(mask_path).astype(np.uint8) < CORE_MASK_THRESHOLD
    tail = np.clip(read_tif(tail_path).astype(np.float32), 0.0, None)
    tail[core_mask] = 0.0

    raw_crop = read_tif(raw_path).astype(np.uint8) if raw_path is not None else None

    meta = {}
    if (folder / "meta.json").exists():
        with open(folder / "meta.json", "r") as f:
            meta = json.load(f)

    relationship = meta.get("source_relationship", {})

    source_bg = relationship.get(
        "source_background_median",
        relationship.get("source_background"),
    )

    if source_bg is None or not np.isfinite(float(source_bg)):
        if raw_crop is not None:
            support = core_mask | (tail > TAIL_VALUE_THRESHOLD)
            outside = raw_crop[~support]
            source_bg = float(np.median(outside if outside.size >= 50 else raw_crop))
        else:
            source_bg = 190.0

    core_delta = np.zeros_like(tail, dtype=np.float32)
    core_delta[core_mask] = np.clip(
        float(source_bg) - core[core_mask].astype(np.float32),
        0.0,
        None,
    )

    core_template = np.zeros_like(core, dtype=np.uint8)
    core_template[core_mask] = core[core_mask]

    entry = {
        "entry_id": record["entry_id"],
        "slice_idx": record["slice_idx"],
        "group_id": record["group_id"],
        "core_template": core_template,
        "core_delta": core_delta,
        "core_mask": core_mask,
        "tail": tail,
        "relationship": relationship,
    }
    cache[key] = entry
    return entry


# ============================================================
# 18. Resize / stretch template
# ============================================================
def crop_template(core_template, core_delta, core_mask, tail):
    """Crop core and tail arrays to their combined nonempty support."""
    box = get_bbox(core_mask | (tail > TAIL_VALUE_THRESHOLD))
    if box is None:
        raise RuntimeError("Empty artefact template.")
    y0, y1, x0, x1 = box
    return (
        core_template[y0:y1, x0:x1].copy(),
        core_delta[y0:y1, x0:x1].copy(),
        core_mask[y0:y1, x0:x1].copy(),
        tail[y0:y1, x0:x1].copy(),
    )


def resize_template(core_template, core_delta, core_mask, tail, scale):
    """Scale template pixels and maintain core-mask and tail separation."""
    h, w = tail.shape
    new_h = max(2, int(round(h * scale)))
    new_w = max(2, int(round(w * scale)))

    core_template = cv2.resize(core_template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    core_delta = cv2.resize(core_delta, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    core_mask = cv2.resize(
        core_mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    tail = cv2.resize(tail, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    core_template = np.round(core_template).astype(np.uint8)
    core_template[~core_mask] = 0
    core_delta[~core_mask] = 0.0
    tail[core_mask] = 0.0
    return core_template, core_delta, core_mask, tail


def stretch_tail(core_template, core_delta, core_mask, tail, rng):
    """Stretch distal tail non-uniformly: strongest near core, weakest distally.

    The old implementation chose a target width from the full core+tail template
    and uniformly resized each distal side to fill that width. A very short tail
    could therefore be expanded into a very long thin line simply because the
    core itself made the template wide.

    Here the amount of new X extent is determined only by the *existing distal
    tail width*. Within each side, local stretch decreases smoothly from a sampled
    4-8x at the root to 1x at the distal edge. The Y geometry is untouched.
    """
    box = get_bbox(core_mask)
    if box is None:
        raise RuntimeError("No core in template.")

    _, _, cx0, cx1 = box
    h, w = tail.shape

    root_width = int(np.clip(
        round((cx1 - cx0) * TAIL_ROOT_WIDTH_TO_CORE_WIDTH),
        TAIL_ROOT_WIDTH_MIN,
        TAIL_ROOT_WIDTH_MAX,
    ))

    left_root_start = max(0, cx0 - root_width)
    right_root_end = min(w, cx1 + root_width)

    left_part = tail[:, :left_root_start]
    middle_part = tail[:, left_root_start:right_root_end]
    right_part = tail[:, right_root_end:]

    if left_part.shape[1] + right_part.shape[1] <= 0:
        return core_template, core_delta, core_mask, tail, 1.0

    proximal_stretch = float(rng.uniform(*TAIL_LENGTH_SCALE_RANGE))

    def stretched_width(width, root_scale):
        if width <= 1:
            return width
        r = np.linspace(0.0, 1.0, width - 1, dtype=np.float32)
        local_scale = 1.0 + (root_scale - 1.0) * (1.0 - r) ** TAIL_STRETCH_DECAY_POWER
        return int(round(1.0 + float(local_scale.sum())))

    desired_left = stretched_width(left_part.shape[1], proximal_stretch)
    desired_right = stretched_width(right_part.shape[1], proximal_stretch)
    middle_width = middle_part.shape[1]

    original_distal = left_part.shape[1] + right_part.shape[1]
    desired_distal = desired_left + desired_right
    max_distal = max(MAX_ARTEFACT_WIDTH - middle_width, original_distal)

    if desired_distal > max_distal and desired_distal > original_distal:
        keep_extra = (max_distal - original_distal) / float(desired_distal - original_distal)
        keep_extra = float(np.clip(keep_extra, 0.0, 1.0))
        effective_root_stretch = 1.0 + (proximal_stretch - 1.0) * keep_extra
    else:
        effective_root_stretch = proximal_stretch

    def stretch_distal(part, proximal_on_left):
        width = part.shape[1]
        if width <= 1:
            return part.astype(np.float32, copy=True)

        work = part if proximal_on_left else part[:, ::-1]

        r = np.linspace(0.0, 1.0, width - 1, dtype=np.float32)
        local_scale = 1.0 + (effective_root_stretch - 1.0) * (1.0 - r) ** TAIL_STRETCH_DECAY_POWER

        target_pos = np.zeros(width, dtype=np.float32)
        target_pos[1:] = np.cumsum(local_scale, dtype=np.float32)
        new_width = max(width, int(round(float(target_pos[-1]))) + 1)

        target_x = np.arange(new_width, dtype=np.float32)
        source_x = np.interp(
            target_x,
            target_pos,
            np.arange(width, dtype=np.float32),
        ).astype(np.float32)

        map_x = np.broadcast_to(source_x[None, :], (h, new_width)).copy()
        map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, new_width)).copy()

        stretched = cv2.remap(
            work.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return stretched if proximal_on_left else stretched[:, ::-1]

    left_new = stretch_distal(left_part, proximal_on_left=False)
    right_new = stretch_distal(right_part, proximal_on_left=True)

    tail_new = np.concatenate([left_new, middle_part, right_new], axis=1).astype(np.float32)

    middle_mask = core_mask[:, left_root_start:right_root_end]
    middle_delta = core_delta[:, left_root_start:right_root_end]
    middle_template = core_template[:, left_root_start:right_root_end]

    mask_new = np.zeros(tail_new.shape, dtype=bool)
    delta_new = np.zeros(tail_new.shape, dtype=np.float32)
    template_new = np.zeros(tail_new.shape, dtype=np.uint8)

    start = left_new.shape[1]
    end = start + middle_mask.shape[1]
    mask_new[:, start:end] = middle_mask
    delta_new[:, start:end] = middle_delta
    template_new[:, start:end] = middle_template

    tail_new[mask_new] = 0.0
    template_new[~mask_new] = 0
    return template_new, delta_new, mask_new, tail_new, tail_new.shape[1] / float(w)


def prepare_template(record, cache, rng):
    """Apply physical-scale normalization and tail-length stretching."""
    entry = load_entry(record, cache)

    spatial_scale = (
        REFERENCE_ARTEFACT_SCALE
        * REFERENCE_CLEAN_PIXEL_SIZE_NM
        / CLEAN_PIXEL_SIZE_NM
        * float(rng.uniform(*RANDOM_ARTEFACT_SCALE_RANGE))
    )

    core_template, core_delta, core_mask, tail = resize_template(
        entry["core_template"],
        entry["core_delta"],
        entry["core_mask"],
        entry["tail"],
        spatial_scale,
    )

    core_template, core_delta, core_mask, tail = crop_template(
        core_template, core_delta, core_mask, tail
    )
    core_template, core_delta, core_mask, tail, tail_length_scale = stretch_tail(
        core_template, core_delta, core_mask, tail, rng
    )
    core_template, core_delta, core_mask, tail = crop_template(
        core_template, core_delta, core_mask, tail
    )

    return {
        "entry_id": entry["entry_id"],
        "slice_idx": entry["slice_idx"],
        "group_id": entry["group_id"],
        "relationship": entry["relationship"],
        "core_template": core_template,
        "core_delta": core_delta,
        "core_mask": core_mask,
        "tail": tail,
        "spatial_scale": float(spatial_scale),
        "tail_length_scale": float(tail_length_scale),
    }


# ============================================================
# 19. Place one template on the synthesis canvas
# ============================================================
def paste_with_clipping(canvas, src, top, left, mode="replace"):
    """Paste a 2D source into a target canvas, clipping to image bounds."""
    h, w = src.shape
    y0 = max(0, top)
    y1 = min(canvas.shape[0], top + h)
    x0 = max(0, left)
    x1 = min(canvas.shape[1], left + w)

    if y1 <= y0 or x1 <= x0:
        return

    sy0 = y0 - top
    sy1 = sy0 + (y1 - y0)
    sx0 = x0 - left
    sx1 = sx0 + (x1 - x0)

    src_crop = src[sy0:sy1, sx0:sx1]

    if mode == "replace":
        canvas[y0:y1, x0:x1] = src_crop
    elif mode == "max":
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], src_crop)
    else:
        raise ValueError(mode)


def make_local_canvases(core_template, core_delta, core_mask, tail, top, left, pad_y=84, pad_x=80):
    """Build padded core/tail masks and residuals around one placement."""
    h, w = tail.shape

    y0 = max(0, top - pad_y)
    y1 = min(IMAGE_SIZE, top + h + pad_y)
    x0 = max(0, left - pad_x)
    x1 = min(IMAGE_SIZE, left + w + pad_x)

    shape = (y1 - y0, x1 - x0)
    local_top = top - y0
    local_left = left - x0

    core_template_canvas = np.zeros(shape, dtype=np.uint8)
    core_delta_canvas = np.zeros(shape, dtype=np.float32)
    core_mask_canvas = np.zeros(shape, dtype=np.uint8)
    tail_canvas = np.zeros(shape, dtype=np.float32)

    paste_with_clipping(core_template_canvas, core_template, local_top, local_left, mode="replace")
    paste_with_clipping(core_delta_canvas, core_delta, local_top, local_left, mode="replace")
    paste_with_clipping(core_mask_canvas, core_mask.astype(np.uint8), local_top, local_left, mode="max")
    paste_with_clipping(tail_canvas, tail, local_top, local_left, mode="max")

    core_mask_canvas = core_mask_canvas > 0
    core_template_canvas[~core_mask_canvas] = 0
    tail_canvas[core_mask_canvas] = 0.0

    return (y0, y1, x0, x1), core_template_canvas, core_delta_canvas, core_mask_canvas, tail_canvas


def place_full_or_tail_template(core_template, core_delta, core_mask, tail, anchor):
    """Center a real core template at an anchor on the synthesis canvas."""
    box = get_bbox(core_mask)
    if box is None:
        raise RuntimeError("No core in template.")
    cy0, cy1, cx0, cx1 = box

    core_center_y = int(round(0.5 * (cy0 + cy1 - 1)))
    core_center_x = int(round(0.5 * (cx0 + cx1 - 1)))

    top = int(anchor[0] - core_center_y)
    left = int(anchor[1] - core_center_x)

    top = int(np.clip(top, -cy0, IMAGE_SIZE - cy1))
    left = int(np.clip(left, -cx0, IMAGE_SIZE - cx1))

    return make_local_canvases(core_template, core_delta, core_mask, tail, top, left)


# ============================================================
# 20. Core / tail rendering
# ============================================================
def render_core(core_template, core_mask, background, step0_stats, rng):
    """Reproduce real core appearance at the configured target intensity."""
    if not core_mask.any():
        return background.copy(), None, None

    core_values = core_template[core_mask].astype(np.float32)
    if core_values.size == 0:
        return background.copy(), None, None

    template_median = float(np.median(core_values))
    contrast_scale = float(
        rng.uniform(1.0 - CORE_CONTRAST_JITTER, 1.0 + CORE_CONTRAST_JITTER)
    )

    centered = (core_values - template_median) * contrast_scale

    center = float(np.clip(
        step0_stats["core_raw_median"],
        *CORE_TARGET_MEDIAN_LIMITS,
    ))

    desired_median = float(np.clip(
        center + rng.uniform(-CORE_TARGET_MEDIAN_JITTER, CORE_TARGET_MEDIAN_JITTER),
        *CORE_TARGET_MEDIAN_LIMITS,
    ))

    rendered_values = np.clip(centered + desired_median, 0.0, 255.0)

    output = background.copy()
    output[core_mask] = np.round(rendered_values).astype(np.uint8)

    realized_core_median = float(np.median(output[core_mask]))
    realized_core_delta = float(np.median(
        background[core_mask].astype(np.float32)
        - output[core_mask].astype(np.float32)
    ))

    return output, realized_core_median, realized_core_delta


def local_ring_median(background, core_mask):
    """Compute a local background median outside a charging core."""
    if not core_mask.any():
        return None

    mask_u8 = core_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * CORE_CLEAN_RING_WIDTH + 1, 2 * CORE_CLEAN_RING_WIDTH + 1),
    )
    dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
    ring = dilated & (~core_mask)

    values = background[ring].astype(np.float32)
    if values.size < 16:
        values = background[core_mask].astype(np.float32)
    if values.size == 0:
        return None
    return float(np.median(values))


def build_clean_core_target(
    core_image,
    background,
    core_mask,
    membrane_darkening,
    clean_texture_source,
    difference_strong_structure,
    rng,
):
    """Build a textured clean core plus an exact uniform closed membrane ring.

    The interior texture still comes from a low-structure region of the real clean
    input, located using the HPF-difference structure mask.

    The membrane is intentionally simple for diagnosis:
      - binary constant-width inner ring from the final core mask;
      - one constant gray value for the entire ring of this core;
      - no transition profile;
      - no Gaussian smoothing;
      - no interior texture is allowed to overwrite membrane pixels.

    Therefore, if `membrane_mask` is closed but the final clean image looks broken,
    the break must be introduced after this target construction (most notably HPF).
    """
    if not core_mask.any():
        empty = np.zeros_like(core_mask, dtype=bool)
        return core_image.copy(), None, None, None, None, empty

    ring_median = local_ring_median(background, core_mask)
    if ring_median is None:
        ring_median = float(np.median(background[core_mask]))

    box = get_bbox(core_mask)
    if box is None:
        empty = np.zeros_like(core_mask, dtype=bool)
        return core_image.copy(), None, ring_median, None, None, empty
    y0, y1, x0, x1 = box

    bbox_mask = core_mask[y0:y1, x0:x1]
    texture_patch, texture_structure_fraction = sample_clean_background_texture(
        clean_texture_source,
        difference_strong_structure,
        bbox_mask.shape,
        rng,
    )

    output_f = core_image.astype(np.float32).copy()

    # Background-like interior: local level + real clean-input texture style.
    clean_bbox = np.full(bbox_mask.shape, ring_median, dtype=np.float32)
    clean_bbox += texture_patch

    # Exact binary inner ring. IMPORTANT: compute boundary geometry on a zero-padded
    # mask rather than a tight crop. Otherwise the core can touch the local array
    # border, and distance/morphology near those edges may behave inconsistently and
    # create artificial membrane gaps.
    membrane_pad = max(int(np.ceil(float(CORE_MEMBRANE_WIDTH_PX))) + 2, 4)
    padded_bbox_mask = np.pad(
        bbox_mask.astype(np.uint8),
        ((membrane_pad, membrane_pad), (membrane_pad, membrane_pad)),
        mode="constant",
        constant_values=0,
    )
    distance = cv2.distanceTransform(
        padded_bbox_mask,
        cv2.DIST_L2,
        5,
    ).astype(np.float32)
    padded_membrane_band = (
        (padded_bbox_mask > 0)
        & (distance > 0.0)
        & (distance <= float(CORE_MEMBRANE_WIDTH_PX))
    )
    membrane_band = padded_membrane_band[
        membrane_pad:-membrane_pad,
        membrane_pad:-membrane_pad,
    ]
    membrane_band = membrane_band & bbox_mask

    # One single gray value for this entire core membrane. This assignment is done
    # AFTER texture construction, so texture cannot make any segment lighter/darker.
    membrane_value = float(np.clip(
        ring_median - membrane_darkening * CORE_MEMBRANE_DARKEN_FACTOR,
        0.0,
        255.0,
    ))
    clean_bbox[membrane_band] = membrane_value

    target_roi = output_f[y0:y1, x0:x1]
    target_roi[bbox_mask] = clean_bbox[bbox_mask]

    membrane_mask_full = np.zeros_like(core_mask, dtype=bool)
    membrane_mask_full[y0:y1, x0:x1] = membrane_band

    output = to_uint8(output_f)
    target_median = float(np.median(output[core_mask]))

    return (
        output,
        None,
        ring_median,
        target_median,
        texture_structure_fraction,
        membrane_mask_full,
    )


def choose_tail_reference_core_delta(relationship, step0_stats, realized_core_delta):
    """Choose realized or source reference contrast for tail calibration."""
    if realized_core_delta is not None:
        return float(realized_core_delta)

    source_core_delta = relationship.get("source_core_background_delta")
    if source_core_delta is None or not np.isfinite(float(source_core_delta)):
        source_core_delta = step0_stats["core_delta_median"]

    return float(source_core_delta)


def calibrate_tail(tail, relationship, reference_core_delta, step0_stats, rng):
    """Adjust tail strength using real template metadata and global statistics."""
    ratio = relationship.get("source_tail_to_core_ratio")
    if ratio is None or not np.isfinite(float(ratio)):
        ratio = step0_stats["tail_ratio_median"]

    ratio = float(ratio)

    data_low = max(TAIL_RATIO_LIMITS[0], float(step0_stats["tail_ratio_p25"]))
    data_high = min(TAIL_RATIO_LIMITS[1], float(step0_stats["tail_ratio_p75"]))
    if data_high <= data_low:
        data_low, data_high = TAIL_RATIO_LIMITS

    ratio = float(np.clip(ratio, data_low, data_high))
    ratio *= float(rng.uniform(1.0 - TAIL_RATIO_JITTER, 1.0 + TAIL_RATIO_JITTER))
    ratio = float(np.clip(ratio, *TAIL_RATIO_LIMITS))

    raw_target_delta = reference_core_delta * ratio * TAIL_DARKEN_FACTOR

    # The broad Step-1 brightness levels can make the realized core contrast
    # much smaller on dark-gray slices. Never let a fixed tail floor make the
    # tail darker than its corresponding core.
    max_tail_delta = max(
        reference_core_delta * TAIL_MAX_CORE_DELTA_FRACTION,
        1.0,
    )
    lower = min(TAIL_TARGET_DELTA_LIMITS[0], max_tail_delta)
    upper = min(TAIL_TARGET_DELTA_LIMITS[1], max_tail_delta)
    target_delta = float(np.clip(raw_target_delta, lower, upper))

    support = tail > TAIL_VALUE_THRESHOLD
    current_delta = positive_percentile(tail[support], TAIL_STRENGTH_PERCENTILE)
    if current_delta is None:
        raise RuntimeError("Tail has no positive residual.")

    scale = target_delta / (current_delta + 1e-8)
    calibrated = (tail * scale).astype(np.float32)

    return calibrated, ratio, target_delta


def calibrate_joint_core_tail(core_delta, tail, rng):
    # Core and tail are one residual field here. Their original internal
    # relationship is preserved, but the whole event is kept shallower than a
    # visible separate core.
    """Calibrate merged core/tail residuals for shallow joint shading."""
    joint = np.maximum(
        core_delta.astype(np.float32),
        tail.astype(np.float32),
    )
    support = joint > TAIL_VALUE_THRESHOLD

    current_delta = positive_percentile(
        joint[support],
        JOINT_STRENGTH_PERCENTILE,
    )
    if current_delta is None:
        raise RuntimeError("Joint core-tail field has no positive residual.")

    target_delta = float(rng.uniform(*JOINT_TARGET_DELTA_RANGE))
    joint *= target_delta / (current_delta + 1e-8)
    return joint.astype(np.float32), target_delta


def render_additive_tail(background, tail):
    """Subtract a calibrated residual from the original background."""
    return to_uint8(background.astype(np.float32) - tail)


def fourier_lowpass(image, cutoff):
    """Compute a Gaussian low-pass image in Fourier space."""
    image = image.astype(np.float32)
    h, w = image.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")

    filt = np.exp(
        -(yy * yy + xx * xx) / (2.0 * cutoff * cutoff + 1e-12)
    )

    return np.real(np.fft.ifft2(np.fft.fft2(image) * filt)).astype(np.float32)


def render_two_pass_tail(background, tail, target_delta, rng):
    """Render dark shading using structure-preserving low-pass blending."""
    raw_map = normalize_positive(tail)

    strength = float(rng.uniform(*TWO_PASS_MAP_STRENGTH_RANGE))
    sigma = float(rng.uniform(*TWO_PASS_SOFT_SIGMA_RANGE))
    gamma = float(rng.uniform(*TWO_PASS_GAMMA_RANGE))
    structure_blend = float(rng.uniform(*TWO_PASS_STRUCTURE_BLEND_RANGE))
    cutoff = float(rng.uniform(*TWO_PASS_FOURIER_CUTOFF_RANGE))
    blur_sigma = float(rng.uniform(*TWO_PASS_SPATIAL_BLUR_RANGE))
    fft_blend = float(rng.uniform(*TWO_PASS_FFT_BLEND_RANGE))

    render_map = np.clip(raw_map * strength, 0.0, 1.0)
    render_map = cv2.GaussianBlur(
        render_map,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    render_map = np.clip(render_map, 0.0, 1.0) ** gamma
    render_map[render_map < 0.01] = 0.0

    background_f = background.astype(np.float32)

    fft_low = fourier_lowpass(background_f, cutoff)
    blur_low = cv2.GaussianBlur(
        background_f,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    degraded = fft_blend * fft_low + (1.0 - fft_blend) * blur_low

    # Keep most of the original membrane structure. Darkness is added in a
    # separate step, so a darker artefact no longer requires stronger blur.
    mix = structure_blend * render_map
    lightly_filtered = background_f * (1.0 - mix) + degraded * mix

    support = tail > TAIL_VALUE_THRESHOLD
    minimum_output = background_f * TWO_PASS_MIN_REMAINING_FRACTION

    # Solve only the independent darkening strength. The relative lower bound
    # prevents dark membrane pixels from collapsing to zero and disappearing.
    low, high = 0.0, 255.0
    for _ in range(18):
        darkening = 0.5 * (low + high)
        candidate = np.maximum(
            lightly_filtered - darkening * render_map,
            minimum_output,
        )
        realized_delta = positive_percentile(
            np.clip(background_f - candidate, 0.0, None)[support],
            TAIL_STRENGTH_PERCENTILE,
        )
        if realized_delta is None:
            raise RuntimeError("Light two-pass produced no tail.")
        if realized_delta < target_delta:
            low = darkening
        else:
            high = darkening

    darkening = 0.5 * (low + high)
    output = np.maximum(
        lightly_filtered - darkening * render_map,
        minimum_output,
    )
    return to_uint8(output), darkening


# ============================================================
# 21. White halo
# ============================================================
def build_halo_map(tail, core_mask, rng):
    """Generate a spatial bright-halo envelope around dark artifact support."""
    tail_map = normalize_positive(tail)

    geometry = tail_map.copy()
    geometry[core_mask] = 1.0

    support = geometry > 0.01
    if not support.any():
        return np.zeros_like(tail, dtype=np.float32)

    box = get_bbox(support)
    if box is None:
        return np.zeros_like(tail, dtype=np.float32)

    y0, y1, _, _ = box
    support_h = max(y1 - y0, 2)

    sigma_y = support_h * float(rng.uniform(*HALO_SIGMA_TO_SUPPORT_HEIGHT_RANGE))
    sigma_y = float(np.clip(sigma_y, HALO_SIGMA_MIN, HALO_SIGMA_MAX))

    vertical = cv2.GaussianBlur(
        geometry,
        (0, 0),
        sigmaX=0.01,
        sigmaY=sigma_y,
        borderType=cv2.BORDER_REFLECT,
    )

    halo = cv2.GaussianBlur(
        vertical,
        (0, 0),
        sigmaX=HALO_X_SMOOTH_SIGMA,
        sigmaY=0.01,
        borderType=cv2.BORDER_REFLECT,
    )

    halo[support] = 0.0

    p99 = positive_percentile(halo, 99.0)
    if p99 is None:
        return np.zeros_like(halo, dtype=np.float32)

    halo = np.clip(halo / (p99 + 1e-8), 0.0, 1.0)

    gamma = float(rng.uniform(*HALO_GAMMA_RANGE))
    halo = halo ** gamma

    halo[core_mask] = 0.0
    return halo.astype(np.float32)


def sample_halo_delta(step0_stats, rng):
    """Sample calibrated halo brightness from the global reference statistics."""
    center = float(np.clip(
        step0_stats["halo_delta_median"],
        *HALO_DELTA_LIMITS,
    ))
    return float(np.clip(
        center + rng.uniform(-HALO_DELTA_JITTER, HALO_DELTA_JITTER),
        *HALO_DELTA_LIMITS,
    ))


def apply_halo_layer(charging, halo_layer, protected_mask):
    """Apply bright halo without changing protected dark pixels."""
    charging_f = charging.astype(np.float32)
    addition = halo_layer.astype(np.float32).copy()
    addition[protected_mask] = 0.0

    available = np.maximum(HALO_OUTPUT_CEILING - charging_f, 0.0)
    addition = np.minimum(addition, available)
    output = to_uint8(charging_f + addition)
    output[protected_mask] = charging[protected_mask]
    return output


def constrain_joint_shading(background, rendered):
    """Keep joint shading dark but retain a fraction of every local pixel."""
    background_f = background.astype(np.float32)
    requested_delta = np.clip(
        background_f - rendered.astype(np.float32),
        0.0,
        None,
    )

    # A relative bound works across the different Step-1 brightness levels and
    # avoids the membrane-flattening behavior of one absolute output floor.
    allowed_by_fraction = background_f * (1.0 - JOINT_MIN_REMAINING_FRACTION)
    applied_delta = np.minimum(requested_delta, JOINT_PIXEL_DELTA_MAX)
    applied_delta = np.minimum(applied_delta, allowed_by_fraction)
    output = to_uint8(background_f - applied_delta)

    changed = applied_delta > 0.0
    min_new_value = float(np.min(output[changed])) if changed.any() else None
    max_added_delta = float(np.max(applied_delta)) if changed.any() else 0.0
    return output, min_new_value, max_added_delta


def compose_dark_union(current, candidate, background, candidate_support, occupied):
    """Fuse overlapping dark artefacts into one continuous residual field."""
    background_f = background.astype(np.float32)
    current_delta = np.clip(
        background_f - current.astype(np.float32),
        0.0,
        None,
    )
    candidate_delta = np.clip(
        background_f - candidate.astype(np.float32),
        0.0,
        None,
    )

    overlap = candidate_support & occupied

    # Smooth maximum in residual space: if one event is clearly darker it wins;
    # if they have similar darkness they merge without an ownership boundary.
    temperature = DARK_UNION_SMOOTHMAX_TEMPERATURE
    maximum = np.maximum(current_delta, candidate_delta)
    combined_delta = maximum.copy()

    # Apply the smooth maximum only where both dark fields truly overlap.
    # Outside overlap, use the original residual exactly so weak tail edges are
    # not artificially strengthened merely because the other field is zero.
    overlap_field = (current_delta > 0.0) & (candidate_delta > 0.0)
    if overlap_field.any():
        difference = np.abs(current_delta - candidate_delta)
        combined_delta[overlap_field] = (
            maximum[overlap_field]
            + temperature
            * np.log1p(
                np.exp(
                    -difference[overlap_field]
                    / max(temperature, 1e-6)
                )
            )
        )

    output = to_uint8(background_f - combined_delta)
    return output, int(overlap.sum())



# ============================================================
# 22. Black thin-bundle charging
# ============================================================
def smooth_bundle_sequence(values, sigma):
    """Smooth correlated across-line variability of a thin bundle."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if values.shape[0] <= 2 or sigma <= 0:
        return values[:, 0]

    smoothed = cv2.GaussianBlur(
        values,
        (0, 0),
        sigmaX=0.01,
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    return smoothed[:, 0]


def normalized_smooth_bundle_noise(rng, count):
    """Sample a normalized, smooth random sequence for bundle geometry."""
    values = rng.normal(0.0, 1.0, size=count).astype(np.float32)
    values = smooth_bundle_sequence(values, THIN_BUNDLE_EDGE_SMOOTH_SIGMA_LINES)
    values -= float(np.mean(values))
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs > 1e-6:
        values /= max_abs
    return values


def build_thin_bundle_field(rng):
    """Create one dense horizontal black-line bundle as a positive residual field.

    Important properties:
    - lines are strictly horizontal;
    - neighboring lines have correlated lengths and endpoints;
    - the whole bundle is NOT forced to be center-aligned;
    - one local region is darker (core-like), and surrounding lines fade gradually.
    """
    size_index = int(rng.choice(len(THIN_BUNDLE_SIZE_PROBS), p=np.asarray(THIN_BUNDLE_SIZE_PROBS, dtype=np.float64)))
    size_name = THIN_BUNDLE_SIZE_NAMES[size_index]

    length_low, length_high = THIN_BUNDLE_LENGTH_RANGES[size_index]
    line_low, line_high = THIN_BUNDLE_LINE_COUNT_RANGES[size_index]

    base_length = int(rng.integers(length_low, length_high + 1))
    line_count = int(rng.integers(line_low, line_high + 1))

    # Dense but imperfect vertical packing.
    spacings = rng.integers(
        THIN_BUNDLE_LINE_SPACING_RANGE[0],
        THIN_BUNDLE_LINE_SPACING_RANGE[1] + 1,
        size=max(line_count - 1, 1),
    )
    y_centers = np.zeros(line_count, dtype=np.int32)
    if line_count > 1:
        y_centers[1:] = np.cumsum(spacings[:line_count - 1], dtype=np.int32)

    # Smooth correlated variation: line lengths should not jump wildly.
    left_noise = normalized_smooth_bundle_noise(rng, line_count)
    right_noise = normalized_smooth_bundle_noise(rng, line_count)
    center_noise = normalized_smooth_bundle_noise(rng, line_count)

    # A bundle-level left/right asymmetry prevents everything from being centered.
    asymmetry = float(rng.uniform(-THIN_BUNDLE_SIDE_ASYMMETRY_FRACTION, THIN_BUNDLE_SIDE_ASYMMETRY_FRACTION))
    base_left = 0.5 * base_length * (1.0 + asymmetry)
    base_right = 0.5 * base_length * (1.0 - asymmetry)

    left_extents = np.maximum(
        8.0,
        base_left * np.clip(
            1.0 + THIN_BUNDLE_LENGTH_VARIATION_FRACTION * left_noise,
            0.45,
            1.55,
        ),
    )
    right_extents = np.maximum(
        8.0,
        base_right * np.clip(
            1.0 + THIN_BUNDLE_LENGTH_VARIATION_FRACTION * right_noise,
            0.45,
            1.55,
        ),
    )

    # Smooth drift across lines: the whole packet can slant in position without
    # changing orientation; this creates off-centered bundles and unequal margins.
    center_offsets = (
        THIN_BUNDLE_CENTER_DRIFT_FRACTION * base_length * center_noise
    )
    drift_end = float(rng.uniform(-0.22, 0.22) * base_length)
    center_offsets += np.linspace(
        -0.5 * drift_end,
        0.5 * drift_end,
        line_count,
        dtype=np.float32,
    )

    starts = center_offsets - left_extents
    ends = center_offsets + right_extents

    min_x = float(np.min(starts))
    max_x = float(np.max(ends))
    width = int(np.ceil(max_x - min_x)) + 2 * THIN_BUNDLE_PAD + 4

    max_thickness = THIN_BUNDLE_LINE_THICKNESS_RANGE[1]
    height = int(y_centers[-1]) + 2 * THIN_BUNDLE_PAD + 2 * max_thickness + 4

    geometry = np.zeros((height, width), dtype=np.float32)

    for i in range(line_count):
        x0 = int(round(starts[i] - min_x + THIN_BUNDLE_PAD + 2))
        x1 = int(round(ends[i] - min_x + THIN_BUNDLE_PAD + 2))
        yc = int(round(y_centers[i] + THIN_BUNDLE_PAD + max_thickness + 2))

        x0 = int(np.clip(x0, 0, width - 1))
        x1 = int(np.clip(x1, x0 + 1, width))

        thickness = int(rng.integers(
            THIN_BUNDLE_LINE_THICKNESS_RANGE[0],
            THIN_BUNDLE_LINE_THICKNESS_RANGE[1] + 1,
        ))

        line_length = x1 - x0
        taper = int(rng.integers(
            THIN_BUNDLE_END_TAPER_RANGE[0],
            THIN_BUNDLE_END_TAPER_RANGE[1] + 1,
        ))
        taper = max(1, min(taper, max(line_length // 3, 1)))

        profile = np.ones(line_length, dtype=np.float32)
        if taper > 1:
            ramp = np.linspace(0.0, 1.0, taper, dtype=np.float32)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            profile[:taper] *= ramp
            profile[-taper:] *= ramp[::-1]

        y_start = max(0, yc - (thickness - 1) // 2)
        y_end = min(height, y_start + thickness)
        for yy_line in range(y_start, y_end):
            geometry[yy_line, x0:x1] = np.maximum(
                geometry[yy_line, x0:x1],
                profile,
            )

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    core_x = float(rng.uniform(*THIN_BUNDLE_CORE_CENTER_X_FRACTION_RANGE) * (width - 1))
    core_y = float(rng.uniform(*THIN_BUNDLE_CORE_CENTER_Y_FRACTION_RANGE) * (height - 1))
    sigma_x = max(
        4.0,
        float(rng.uniform(*THIN_BUNDLE_CORE_SIGMA_X_FRACTION_RANGE) * width),
    )
    sigma_y = max(
        3.0,
        float(rng.uniform(*THIN_BUNDLE_CORE_SIGMA_Y_FRACTION_RANGE) * height),
    )

    core_envelope = np.exp(
        -0.5 * (
            ((xx - core_x) / sigma_x) ** 2
            + ((yy - core_y) / sigma_y) ** 2
        )
    ).astype(np.float32)

    outer_strength = float(rng.uniform(*THIN_BUNDLE_OUTER_STRENGTH_RANGE))
    darkness_envelope = outer_strength + (1.0 - outer_strength) * core_envelope

    field = (
        geometry
        * darkness_envelope
        * THIN_BUNDLE_FIELD_SCALE
    ).astype(np.float32)

    info = {
        "thin_bundle_size": size_name,
        "thin_bundle_base_length": base_length,
        "thin_bundle_line_count": line_count,
        "thin_bundle_height": height,
        "thin_bundle_width": width,
        "thin_bundle_angle_deg": 0.0,
        "thin_bundle_outer_strength": outer_strength,
        "thin_bundle_core_x_fraction": core_x / max(width - 1, 1),
        "thin_bundle_core_y_fraction": core_y / max(height - 1, 1),
        "thin_bundle_core_sigma_x": sigma_x,
        "thin_bundle_core_sigma_y": sigma_y,
        "thin_bundle_asymmetry": asymmetry,
    }
    return field, info


def render_direct_thin_bundle(background, field, target_delta):
    """Directly place the predesigned horizontal-line bundle onto the image.

    No two-pass processing, no extra blur, no structure filtering.
    The darker core-like region is already encoded inside `field`.
    """
    render_map = normalize_positive(field)

    if not np.any(render_map > 0.0):
        raise RuntimeError("Empty thin-bundle render map.")

    background_f = background.astype(np.float32)
    support = render_map > 0.01
    minimum_output = background_f * TWO_PASS_MIN_REMAINING_FRACTION

    low, high = 0.0, 255.0
    for _ in range(18):
        darkening = 0.5 * (low + high)
        candidate = np.maximum(
            background_f - darkening * render_map,
            minimum_output,
        )

        realized_delta = positive_percentile(
            np.clip(background_f - candidate, 0.0, None)[support],
            TAIL_STRENGTH_PERCENTILE,
        )
        if realized_delta is None:
            raise RuntimeError("Direct thin-bundle rendering produced no dark lines.")

        if realized_delta < target_delta:
            low = darkening
        else:
            high = darkening

    darkening = 0.5 * (low + high)
    output = np.maximum(
        background_f - darkening * render_map,
        minimum_output,
    )

    return to_uint8(output), darkening


def add_thin_bundle_event(
    base_image,
    clean,
    charging,
    noisy,
    occupied_dark_mask,
    used_anchors,
    rng,
):
    """Add one black thin-bundle charging event by direct placement.

    Placement uses the accepted footprint-aware rule:
    - generate the bundle exactly as before;
    - keep the existing maximin anchor philosophy;
    - evaluate overlap using the bundle's ACTUAL support;
    - accept positions with <= THIN_BUNDLE_MAX_OVERLAP_FRACTION overlap;
    - allow small/natural overlap rather than requiring zero overlap.

    All morphology, intensity, direct rendering, dark-union fusion, and
    Occupancy semantics are preserved from the original implementation.
    """
    render_field, bundle_info = build_thin_bundle_field(rng)
    h, w = render_field.shape

    # Preserve the original safety clipping. With the tiny/small/
    # medium ranges this should rarely be needed, but retaining it avoids
    # changing any morphology logic outside placement.
    original_h, original_w = h, w
    max_h = IMAGE_SIZE - 2
    max_w = IMAGE_SIZE - 2

    if h > max_h:
        y0_crop = max(0, (h - max_h) // 2)
        render_field = render_field[y0_crop:y0_crop + max_h, :]
    if render_field.shape[1] > max_w:
        w_now = render_field.shape[1]
        x0_crop = max(0, (w_now - max_w) // 2)
        render_field = render_field[:, x0_crop:x0_crop + max_w]

    h, w = render_field.shape

    # Placement support is based on the actual synthetic thin-bundle residual,
    # not on a bounding box or just the anchor center.
    placement_support = render_field > TAIL_VALUE_THRESHOLD
    placement_support_pixels = int(np.sum(placement_support))
    if placement_support_pixels <= 0:
        raise PlacementBlockedError("Thin-bundle placement support is empty.")

    accepted = None
    best_seen_overlap_fraction = None

    for placement_try in range(1, THIN_BUNDLE_PLACEMENT_TRIES + 1):
        anchor = sample_thin_bundle_anchor(
            rng=rng,
            used_anchors=used_anchors,
            bundle_shape=(h, w),
        )

        # sample_thin_bundle_anchor guarantees the full footprint fits, so no
        # np.clip() is used here. This avoids silently shifting a bundle inward
        # relative to the anchor used by the maximin placement rule.
        top = int(anchor[0] - h // 2)
        left = int(anchor[1] - w // 2)
        y0, y1 = top, top + h
        x0, x1 = left, left + w

        if y0 < 0 or x0 < 0 or y1 > IMAGE_SIZE or x1 > IMAGE_SIZE:
            raise RuntimeError(
                "Internal thin-bundle placement error: "
                f"anchor={anchor}, footprint={(h, w)}, "
                f"roi={(y0, y1, x0, x1)}"
            )

        occupied_roi = occupied_dark_mask[y0:y1, x0:x1]
        overlap_pixels_before = int(
            np.sum(placement_support & occupied_roi)
        )
        overlap_fraction_before = (
            overlap_pixels_before / float(placement_support_pixels)
        )

        if (
            best_seen_overlap_fraction is None
            or overlap_fraction_before < best_seen_overlap_fraction
        ):
            best_seen_overlap_fraction = float(overlap_fraction_before)

        if (
            overlap_fraction_before
            <= THIN_BUNDLE_MAX_OVERLAP_FRACTION + 1e-12
        ):
            accepted = (
                anchor,
                y0, y1, x0, x1,
                overlap_pixels_before,
                float(overlap_fraction_before),
                int(placement_try),
            )
            break

    if accepted is None:
        raise PlacementBlockedError(
            "Thin-bundle footprint overlap remained above "
            f"{THIN_BUNDLE_MAX_OVERLAP_FRACTION:.0%} after "
            f"{THIN_BUNDLE_PLACEMENT_TRIES} placement attempts; "
            f"best={best_seen_overlap_fraction:.3f}"
        )

    (
        anchor,
        y0, y1, x0, x1,
        placement_overlap_pixels,
        placement_overlap_fraction,
        placement_tries_used,
    ) = accepted

    render_background = base_image[y0:y1, x0:x1].copy()
    charging_roi = charging[y0:y1, x0:x1].copy()
    noisy_roi = noisy[y0:y1, x0:x1].copy()
    occupied_roi = occupied_dark_mask[y0:y1, x0:x1]

    target_delta = float(rng.uniform(*THIN_BUNDLE_TARGET_DELTA_RANGE))
    charging_candidate, two_pass_darkening = render_direct_thin_bundle(
        render_background,
        render_field,
        target_delta,
    )

    candidate_support = (
        placement_support
        | (charging_candidate != render_background)
    )

    # Diagnostic only: rendering can add a few faint changed pixels outside the
    # thresholded placement support. We retain the pre-render placement rule
    # above so that morphology/intensity logic itself is untouched.
    candidate_support_pixels = int(np.sum(candidate_support))
    candidate_overlap_pixels = int(np.sum(candidate_support & occupied_roi))
    candidate_overlap_fraction = (
        candidate_overlap_pixels / float(candidate_support_pixels)
        if candidate_support_pixels > 0 else 0.0
    )

    charging_event, overlap_fused_pixels = compose_dark_union(
        current=charging_roi,
        candidate=charging_candidate,
        background=render_background,
        candidate_support=candidate_support,
        occupied=occupied_roi,
    )

    newly_written_mask = charging_event != charging_roi
    if not newly_written_mask.any():
        raise PlacementBlockedError(
            "Thin-bundle event produced no visible change after dark-union fusion."
        )

    # Thin bundles are scheduled before the halo phase. They have no white halo of
    # their own; marking them occupied also protects their black lines from later
    # separate-event halo additions.
    noisy_event = charging_event.copy()

    charging[y0:y1, x0:x1] = charging_event
    noisy[y0:y1, x0:x1] = noisy_event
    occupied_roi[candidate_support] = True

    meta = {
        "event_type": "thin_bundle",
        "renderer": "direct_place",
        "anchor_y": int(anchor[0]),
        "anchor_x": int(anchor[1]),
        "entry_id": None,
        "artefact_slice_idx": None,
        "artefact_group_id": None,
        "spatial_scale": None,
        "tail_length_scale": None,
        "joint_extra_scale": None,
        "realized_core_median": None,
        "realized_core_delta": None,
        "tail_ratio": None,
        "target_tail_delta_p90": target_delta,
        "joint_target_delta_p90": None,
        "two_pass_darkening": two_pass_darkening,
        "halo_delta": None,
        "joint_min_new_value": None,
        "joint_max_added_delta": None,
        "overlap_pixels_fused": overlap_fused_pixels,
        "newly_occupied_pixels": int(newly_written_mask.sum()),
        "clean_core_copied": False,
        "clean_core_source_median": None,
        "clean_core_background_median": None,
        "clean_core_target_median": None,
        "clean_core_membrane_darkening": None,
        "visible_core_pixels": 0,
        "render_field_pixels": int(np.sum(render_field > TAIL_VALUE_THRESHOLD)),
        "thin_bundle_original_height": int(original_h),
        "thin_bundle_original_width": int(original_w),
        "thin_bundle_clipped_to_fov": int(
            original_h != h or original_w != w
        ),
        "thin_bundle_placement_overlap_limit": float(
            THIN_BUNDLE_MAX_OVERLAP_FRACTION
        ),
        "thin_bundle_placement_overlap_pixels": int(
            placement_overlap_pixels
        ),
        "thin_bundle_placement_overlap_fraction": float(
            placement_overlap_fraction
        ),
        "thin_bundle_candidate_overlap_pixels_after_render": int(
            candidate_overlap_pixels
        ),
        "thin_bundle_candidate_overlap_fraction_after_render": float(
            candidate_overlap_fraction
        ),
        "thin_bundle_placement_tries": int(placement_tries_used),
        **bundle_info,
    }

    return clean, charging, noisy, meta


# ============================================================
# 23. Add one regular-library charging event
# ============================================================
def add_one_event(
    base_image,
    clean,
    charging,
    noisy,
    halo_layer,
    preserved_core_mask,
    clean_core_target,
    clean_membrane_mask,
    occupied_dark_mask,
    event_type,
    renderer,
    entries,
    cache,
    step0_stats,
    membrane_darkening,
    clean_texture_source,
    difference_texture,
    difference_strong_structure,
    anchor,
    rng,
):
    """Render one real-template charging or joint-shading artifact event."""
    last_error = None

    for _ in range(MAX_EVENT_TRIES):
        try:
            record = entries[int(rng.integers(0, len(entries)))]
            template = prepare_template(record, cache, rng)

            joint_extra_scale = 1.0
            if event_type == "joint":
                joint_extra_scale = float(rng.uniform(*JOINT_EXTRA_SCALE_RANGE))
                core_template, core_delta, core_mask, tail = resize_template(
                    template["core_template"],
                    template["core_delta"],
                    template["core_mask"],
                    template["tail"],
                    joint_extra_scale,
                )
                core_template, core_delta, core_mask, tail = crop_template(
                    core_template,
                    core_delta,
                    core_mask,
                    tail,
                )
            else:
                # Separate events use only the base spatial scale. Explicit
                # copies prevent any joint resize from affecting them.
                core_template = template["core_template"].copy()
                core_delta = template["core_delta"].copy()
                core_mask = template["core_mask"].copy()
                tail = template["tail"].copy()

            (y0, y1, x0, x1), core_template_canvas, core_delta_canvas, core_mask_canvas, tail_canvas = (
                place_full_or_tail_template(
                    core_template,
                    core_delta,
                    core_mask,
                    tail,
                    anchor,
                )
            )

            if not np.any(tail_canvas > TAIL_VALUE_THRESHOLD):
                raise RuntimeError("No visible tail after placement.")

            base_roi = base_image[y0:y1, x0:x1]
            clean_roi = clean[y0:y1, x0:x1].copy()
            charging_roi = charging[y0:y1, x0:x1].copy()
            noisy_roi = noisy[y0:y1, x0:x1].copy()
            halo_layer_roi = halo_layer[y0:y1, x0:x1]
            clean_core_target_roi = clean_core_target[y0:y1, x0:x1]
            clean_membrane_mask_roi = clean_membrane_mask[y0:y1, x0:x1]
            occupied_roi = occupied_dark_mask[y0:y1, x0:x1]

            # Every separate event must contribute one complete core to clean.
            # If its core touches an earlier dark artefact, reject this placement
            # and let synthesize_one_slice sample another anchor.
            if event_type == "separate" and np.any(core_mask_canvas & occupied_roi):
                raise PlacementBlockedError(
                    "Separate core overlaps an earlier artefact."
                )

            # Render every new event on one continuous original-background ROI.
            # Candidate and existing dark artefacts are fused afterwards in
            # darkness-residual space, rather than assigning pixel ownership.
            render_background = base_roi.copy()

            joint_min_new_value = None
            joint_max_added_delta = None

            if event_type == "separate":
                visible_core_mask = core_mask_canvas.copy()

                core_only, realized_core_median, realized_core_delta = render_core(
                    core_template_canvas,
                    visible_core_mask,
                    render_background,
                    step0_stats,
                    rng,
                )

                reference_core_delta = choose_tail_reference_core_delta(
                    template["relationship"],
                    step0_stats,
                    realized_core_delta,
                )

                render_field, tail_ratio, target_tail_delta = calibrate_tail(
                    tail_canvas,
                    template["relationship"],
                    reference_core_delta,
                    step0_stats,
                    rng,
                )
                joint_target_delta = None

            elif event_type == "joint":
                visible_core_mask = np.zeros_like(core_mask_canvas, dtype=bool)
                core_only = None
                realized_core_median = None
                realized_core_delta = None
                tail_ratio = None
                target_tail_delta = None

                render_field, joint_target_delta = calibrate_joint_core_tail(
                    core_delta_canvas,
                    tail_canvas,
                    rng,
                )
            else:
                raise ValueError(event_type)

            if renderer == "additive":
                charging_event = render_additive_tail(render_background, render_field)
                two_pass_darkening = None

            elif renderer == "two_pass":
                charging_event, two_pass_darkening = render_two_pass_tail(
                    render_background,
                    render_field,
                    target_tail_delta if event_type == "separate" else joint_target_delta,
                    rng,
                )
            else:
                raise ValueError(renderer)

            if event_type == "separate":
                charging_event[visible_core_mask] = core_only[visible_core_mask]

            elif event_type == "joint":
                charging_event, joint_min_new_value, joint_max_added_delta = (
                    constrain_joint_shading(
                        render_background,
                        charging_event,
                    )
                )

            # Continuous dark union: overlap is allowed and fused in residual
            # space. There is no first-write ownership boundary between events.
            candidate_change_mask = charging_event != render_background
            candidate_support_mask = candidate_change_mask | visible_core_mask
            charging_event, overlap_fused_pixels = compose_dark_union(
                current=charging_roi,
                candidate=charging_event,
                background=render_background,
                candidate_support=candidate_support_mask,
                occupied=occupied_roi,
            )

            newly_written_mask = (charging_event != charging_roi) | visible_core_mask
            if not newly_written_mask.any():
                raise PlacementBlockedError(
                    "Event produced no visible change after dark-union fusion."
                )

            clean_core_copied = bool(visible_core_mask.any())
            core_clean_source_median = None
            core_clean_background_median = None
            core_clean_target_median = None
            core_clean_texture_structure_fraction = None

            if clean_core_copied:
                # Keep the old placement semantics first: copy the exact dark pasted noisy
                # core into clean. This temporary copy is also used at the end to
                # restore the noisy core after any later halo/joint overlap.
                clean_roi[visible_core_mask] = charging_event[visible_core_mask]
                clean[y0:y1, x0:x1] = clean_roi

                # Clean target: preserve the core's exact shape and local
                # texture, but remove its abnormal dark level with ONE constant
                # offset. The target median is the underlying pre-artefact background
                # at the same core footprint. No blur, inpainting, or contrast scale.
                (
                    matched_core,
                    core_clean_source_median,
                    core_clean_background_median,
                    core_clean_target_median,
                    core_clean_texture_structure_fraction,
                    core_membrane_mask_event,
                ) = build_clean_core_target(
                    charging_event,
                    render_background,
                    visible_core_mask,
                    membrane_darkening,
                    clean_texture_source,
                    difference_strong_structure,
                    rng,
                )
                clean_core_target_roi[visible_core_mask] = matched_core[visible_core_mask]
                clean_membrane_mask_roi[core_membrane_mask_event] = True

                mask_roi = preserved_core_mask[y0:y1, x0:x1]
                mask_roi[visible_core_mask] = True

            if event_type == "separate":
                # Build the halo now but store it in a separate layer. The layer
                # is applied only after all scheduled separate events are complete.
                halo_map = build_halo_map(render_field, visible_core_mask, rng)
                halo_delta = sample_halo_delta(step0_stats, rng)
                current_halo_addition = halo_map * halo_delta
                np.maximum(
                    halo_layer_roi,
                    current_halo_addition,
                    out=halo_layer_roi,
                )
                noisy_event = charging_event.copy()

            elif event_type == "joint":
                # Joint shading is added after the separate-halo stage. Apply
                # exactly the realized joint darkening to the current noisy
                # image so that it also darkens any halo beneath it.
                halo_delta = None
                joint_delta = np.clip(
                    charging_roi.astype(np.float32)
                    - charging_event.astype(np.float32),
                    0.0,
                    None,
                )
                noisy_event = to_uint8(noisy_roi.astype(np.float32) - joint_delta)

            else:
                raise ValueError(event_type)

            noisy_event[visible_core_mask] = charging_event[visible_core_mask]

            charging[y0:y1, x0:x1] = charging_event
            noisy[y0:y1, x0:x1] = noisy_event
            occupied_roi[candidate_support_mask] = True

            meta = {
                "event_type": event_type,
                "renderer": renderer,
                "anchor_y": int(anchor[0]),
                "anchor_x": int(anchor[1]),
                "entry_id": int(template["entry_id"]),
                "artefact_slice_idx": int(template["slice_idx"]),
                "artefact_group_id": int(template["group_id"]),
                "spatial_scale": float(template["spatial_scale"]),
                "tail_length_scale": float(template["tail_length_scale"]),
                "joint_extra_scale": joint_extra_scale,
                "realized_core_median": realized_core_median,
                "realized_core_delta": realized_core_delta,
                "tail_ratio": tail_ratio,
                "target_tail_delta_p90": target_tail_delta,
                "joint_target_delta_p90": joint_target_delta,
                "two_pass_darkening": two_pass_darkening,
                "halo_delta": halo_delta,
                "joint_min_new_value": joint_min_new_value,
                "joint_max_added_delta": joint_max_added_delta,
                "overlap_pixels_fused": overlap_fused_pixels,
                "newly_occupied_pixels": int(newly_written_mask.sum()),
                "clean_core_copied": clean_core_copied,
                "clean_core_source_median": core_clean_source_median,
                "clean_core_background_median": core_clean_background_median,
                "clean_core_target_median": core_clean_target_median,
                "clean_core_membrane_darkening": (
                    float(membrane_darkening) if clean_core_copied else None
                ),
                "visible_core_pixels": int(visible_core_mask.sum()),
                "render_field_pixels": int(np.sum(render_field > TAIL_VALUE_THRESHOLD)),
            }

            return clean, charging, noisy, meta

        except PlacementBlockedError:
            raise
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(f"Event synthesis failed after {MAX_EVENT_TRIES} tries: {last_error}")


# ============================================================
# 24. Schedule helper
# ============================================================
def build_event_schedule(rng):
    """Sample the fixed event schedule without altering event counts."""
    regular_separate_schedule = (
        [("separate", "additive")] * int(round(NUM_REGULAR_SEPARATE_EVENTS * ADDITIVE_FRACTION))
        + [("separate", "two_pass")] * (
            NUM_REGULAR_SEPARATE_EVENTS
            - int(round(NUM_REGULAR_SEPARATE_EVENTS * ADDITIVE_FRACTION))
        )
    )
    regular_separate_schedule = list(regular_separate_schedule)
    rng.shuffle(regular_separate_schedule)

    # Preserve regular separate-core placement priority. Thin bundles are then
    # added before the shared halo phase, followed by the existing joint shading.
    thin_bundle_schedule = [("thin_bundle", "direct_place")] * NUM_THIN_BUNDLE_EVENTS
    joint_schedule = [("joint", "two_pass")] * NUM_JOINT_EVENTS
    return regular_separate_schedule + thin_bundle_schedule + joint_schedule


# ============================================================
# 25. One synthetic 1024 slice
# ============================================================
def synthesize_one_slice(
    styled_clean,
    split,
    library,
    cache,
    step0_stats,
    membrane_darkening,
    clean_texture_source,
    difference_texture,
    difference_strong_structure,
    rng,
    make_clean_identity,
):
    # Keep the previous 15% no-artefact sample composition, but remove the
    # membrane-enhancement target. These samples are now strict identity pairs.
    """Create all artifacts and clean-core targets for one styled slice."""
    if make_clean_identity:
        raw_input = styled_clean.copy()
        clean = raw_input.copy()
        empty_mask = np.zeros_like(raw_input, dtype=bool)
        return (
            clean, raw_input.copy(), raw_input, [], "clean_identity", clean.copy(),
            empty_mask, empty_mask.copy(), [],
        )

    # Degraded samples: keep all previous artefact and clean-core logic unchanged.
    clean = styled_clean.copy()
    charging = styled_clean.copy()
    noisy = charging.copy()
    halo_layer = np.zeros_like(clean, dtype=np.float32)

    preserved_core_mask = np.zeros_like(clean, dtype=bool)
    clean_core_target = styled_clean.copy()
    clean_membrane_mask = np.zeros_like(clean, dtype=bool)
    occupied_dark_mask = np.zeros_like(clean, dtype=bool)

    entries = library[split]
    schedule = build_event_schedule(rng)
    used_anchors = []
    event_metas = []

    for event_index, (event_type, renderer) in enumerate(schedule):
        # Phase boundary: apply all separate halos together before the first
        # joint shading event. Dark artefact pixels remain protected.
        if event_index == NUM_SEPARATE_EVENTS:
            noisy = apply_halo_layer(
                charging,
                halo_layer,
                occupied_dark_mask,
            )

        last_placement_error = None

        placement_try_limit = (
            REGULAR_SEPARATE_ANCHOR_PLACEMENT_TRIES
            if event_type == "separate"
            else MAX_ANCHOR_PLACEMENT_TRIES
        )

        for _ in range(placement_try_limit):
            try:
                if event_type == "thin_bundle":
                    clean, charging, noisy, meta = add_thin_bundle_event(
                        base_image=styled_clean,
                        clean=clean,
                        charging=charging,
                        noisy=noisy,
                        occupied_dark_mask=occupied_dark_mask,
                        used_anchors=used_anchors,
                        rng=rng,
                    )
                    # The thin-bundle routine chooses its accepted anchor using
                    # the actual bundle footprint. Use that anchor everywhere
                    # downstream (used_anchors, metadata, later white stripes).
                    anchor = (
                        int(meta["anchor_y"]),
                        int(meta["anchor_x"]),
                    )
                else:
                    # All non-thin-bundle event placement remains exactly on the
                    # original generic random-maximin sampler.
                    anchor = sample_anchor(rng, used_anchors)
                    clean, charging, noisy, meta = add_one_event(
                        base_image=styled_clean,
                        clean=clean,
                        charging=charging,
                        noisy=noisy,
                        halo_layer=halo_layer,
                        preserved_core_mask=preserved_core_mask,
                        clean_core_target=clean_core_target,
                        clean_membrane_mask=clean_membrane_mask,
                        occupied_dark_mask=occupied_dark_mask,
                        event_type=event_type,
                        renderer=renderer,
                        entries=entries,
                        cache=cache,
                        step0_stats=step0_stats,
                        membrane_darkening=membrane_darkening,
                        clean_texture_source=clean_texture_source,
                        difference_texture=difference_texture,
                        difference_strong_structure=difference_strong_structure,
                        anchor=anchor,
                        rng=rng,
                    )
            except RuntimeError as exc:
                last_placement_error = exc
                continue

            used_anchors.append(anchor)
            event_metas.append(meta)
            break
        else:
            raise RuntimeError(
                f"Could not place {event_type}/{renderer} after "
                f"{placement_try_limit} anchors: {last_placement_error}"
            )

    if any(
        meta["event_type"] == "separate" and not meta["clean_core_copied"]
        for meta in event_metas
    ):
        raise RuntimeError("A separate event failed to copy its core to clean.")

    if any(
        meta["event_type"] == "joint" and meta["clean_core_copied"]
        for meta in event_metas
    ):
        raise RuntimeError("A joint shading event incorrectly copied a core to clean.")

    if any(
        meta["event_type"] == "thin_bundle" and meta["clean_core_copied"]
        for meta in event_metas
    ):
        raise RuntimeError("A thin-bundle event incorrectly copied a core to clean.")

    # Preserve the exact dark noisy core against later halo/joint overlap, as before.
    # At this moment `clean` still contains the exact copied dark core.
    charging[preserved_core_mask] = clean[preserved_core_mask]
    noisy[preserved_core_mask] = charging[preserved_core_mask]

    if not np.array_equal(
        charging[preserved_core_mask],
        clean[preserved_core_mask],
    ):
        raise RuntimeError("Dark noisy core protection failed before clean target update.")

    # Replace ONLY the clean core pixels with the same noisy core
    # texture shifted to local-background level, then STOP. No global HPF.
    clean[preserved_core_mask] = clean_core_target[preserved_core_mask]
    clean_target = clean.copy()

    return (
        clean_target, charging, noisy, event_metas, "degraded", clean_target.copy(),
        preserved_core_mask.copy(), clean_membrane_mask.copy(), list(used_anchors),
    )


# ============================================================
# 26. Preview
# ============================================================
def make_preview_row(items):
    """Concatenate labeled uint8 preview panels without altering image data."""
    panels = []

    for title, image in items:
        if image.dtype != np.uint8:
            raise ValueError(f"Preview expects uint8, got {image.dtype}: {title}")
        small = cv2.resize(
            image,
            (PREVIEW_SIZE, PREVIEW_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        header = np.full((28, PREVIEW_SIZE), 255, dtype=np.uint8)
        cv2.putText(
            header,
            title,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            0,
            1,
            cv2.LINE_AA,
        )

        panels.append(np.concatenate([header, small], axis=0))

    separator = np.zeros((panels[0].shape[0], 8), dtype=np.uint8)
    row = [panels[0]]
    for panel in panels[1:]:
        row.extend([separator, panel])
    return np.concatenate(row, axis=1)


def make_preview(clean, charging, noisy):
    """Build a three-panel clean/charging/noisy diagnostic preview."""
    return make_preview_row((
        ("clean", clean),
        ("noisy_charging", charging),
        ("noisy", noisy),
    ))


def make_core_diagnostic_preview(
    core_mask,
    membrane_mask,
    clean_target,
):
    """Visualize synthesized charging-core and membrane masks."""
    core_mask_u8 = (core_mask.astype(np.uint8) * 255)
    membrane_mask_u8 = (membrane_mask.astype(np.uint8) * 255)

    return make_preview_row((
        ("core mask", core_mask_u8),
        ("membrane mask", membrane_mask_u8),
        ("clean target (no global HPF)", clean_target),
    ))



# ============================================================
# 27. Dataset generation and CLI
# ============================================================
def prepare_output_root(overwrite):
    """Safely initialize the original synthesis output directory structure."""
    if OUTPUT_ROOT.exists():
        if not overwrite:
            raise FileExistsError(
                f"OUTPUT_ROOT already exists:\n{OUTPUT_ROOT}\n\n"
                "Nothing was overwritten. Use --overwrite only when you "
                "intentionally want to regenerate the synthetic dataset."
            )
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for level in BACKGROUND_LEVELS:
            root = OUTPUT_ROOT / split / BACKGROUND_TAGS[level]
            for subdir in (
                "clean",
                "noisy_charging",
                "noisy",
                "preview",
                "preview_core_membrane_diagnostic",
            ):
                (root / subdir).mkdir(parents=True, exist_ok=True)


def preload_artifact_cache(library):
    """
    Load every real artifact entry once before sample-level threading starts.
    The resulting cache is read-only during parallel synthesis.
    """
    cache = {}
    unique_records = {}

    for split in SPLITS:
        for record in library[split]:
            key = str(record["entry_dir"])
            unique_records.setdefault(key, record)

    for record in tqdm(
        unique_records.values(),
        total=len(unique_records),
        desc="Preload artifact library",
        unit="entry",
    ):
        load_entry(record, cache)

    return cache


def build_jobs(
    source_records,
    clean_identity_keys,
    factor_assignments,
):
    """
    Build one deterministic job per (source slice, variant).

    sample_id is assigned only from the stable sorted source order and variant
    number, so it is invariant to worker count and completion order.
    """
    jobs = []
    sample_id = 0

    preview_quota = {
        (split, level): NUM_PREVIEWS_PER_BACKGROUND
        for split in SPLITS
        for level in BACKGROUND_LEVELS
    }

    for source in source_records:
        split = source["split"]
        bg_level = int(source["background_level"])

        for variant in range(VARIANTS_PER_CLEAN):
            key = (source["path"], variant)
            make_clean_identity = key in clean_identity_keys

            if make_clean_identity:
                forced_factor = 1.0
            else:
                if key not in factor_assignments:
                    raise KeyError(
                        "Missing balanced contrast-factor assignment for "
                        f"{source['path'].name}, variant={variant}"
                    )
                forced_factor = float(factor_assignments[key])

            sample_seed = stable_seed(
                MASTER_SEED,
                "synthesis_sample",
                split,
                source["block_name"],
                source["source_filename"],
                variant,
            )

            # Preview only degraded examples; deterministic because jobs are
            # built in stable source/variant order.
            save_preview = False
            preview_key = (split, bg_level)
            if (
                SAVE_PREVIEWS
                and not make_clean_identity
                and preview_quota[preview_key] > 0
            ):
                save_preview = True
                preview_quota[preview_key] -= 1

            jobs.append({
                "sample_id": int(sample_id),
                "source": source,
                "variant": int(variant),
                "make_clean_identity": bool(make_clean_identity),
                "forced_factor": float(forced_factor),
                "sample_seed": int(sample_seed),
                "save_preview": bool(save_preview),
            })
            sample_id += 1

    return jobs


def process_one_job(
    job,
    library,
    cache,
    step0_stats,
):
    """Synthesize and write one deterministic variant, returning its metadata."""
    source = job["source"]
    split = source["split"]
    bg_level = int(source["background_level"])
    variant = int(job["variant"])
    sample_seed = int(job["sample_seed"])
    rng = np.random.default_rng(sample_seed)

    styled_clean = read_tif(source["path"])

    if (
        styled_clean.shape != (IMAGE_SIZE, IMAGE_SIZE)
        or styled_clean.dtype != np.uint8
    ):
        raise ValueError(
            f"Expected uint8 {IMAGE_SIZE}x{IMAGE_SIZE}: "
            f"{source['path']}; got {styled_clean.shape}, {styled_clean.dtype}"
        )

    # Reference-only signal for unchanged clean-core synthesis.
    # Never written globally into the clean target.
    structure_reference = build_structure_reference(styled_clean)

    membrane_darkening, membrane_stats = estimate_real_membrane_darkening(
        styled_clean,
        structure_reference,
    )

    difference_texture, difference_strong_structure = (
        prepare_difference_texture_source(
            styled_clean,
            structure_reference,
        )
    )

    (
        clean,
        charging,
        noisy,
        event_metas,
        sample_type,
        clean_target_reference,
        core_mask_diagnostic,
        membrane_mask_diagnostic,
        used_anchors,
    ) = synthesize_one_slice(
        styled_clean=styled_clean,
        split=split,
        library=library,
        cache=cache,
        step0_stats=step0_stats,
        membrane_darkening=membrane_darkening,
        clean_texture_source=styled_clean,
        difference_texture=difference_texture,
        difference_strong_structure=difference_strong_structure,
        rng=rng,
        make_clean_identity=job["make_clean_identity"],
    )

    if sample_type == "degraded":
        charging, noisy, amplification_meta = amplify_slice_artefacts(
            styled_clean,
            charging,
            noisy,
            rng,
            background_level=bg_level,
            core_mask=core_mask_diagnostic,
            forced_target_factor=job["forced_factor"],
        )

        charging, noisy, new_white_stripe_meta = (
            apply_same_new_white_stripes_to_pair(
                charging,
                noisy,
                rng,
                used_anchors,
            )
        )
    else:
        amplification_meta = {
            "background_level": int(bg_level),
            "candidate_factors": [
                float(x)
                for x in ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND[bg_level]
            ],
            "amplification_enabled": 0,
            "core_amplification_enabled": 0,
            "noncore_dark_amplification_enabled": 0,
            "bright_halo_amplification_enabled": 0,
            "target_factor": 1.0,
            "effective_factor": 1.0,
            "limited_by_headroom": 0,
            "max_factor_charging": None,
            "max_factor_noisy": None,
            "safe_limit_after_margin": None,
            "dark_floor": None,
            "bright_ceiling": None,
            "core_negative_pixels_before": 0,
            "noncore_negative_pixels_before": 0,
            "positive_delta_pixels_before": 0,
            "core_negative_pixels_after": 0,
            "noncore_negative_pixels_after": 0,
            "positive_delta_pixels_after": 0,
            "max_core_negative_delta_before": 0.0,
            "max_noncore_negative_delta_before": 0.0,
            "max_positive_delta_before": 0.0,
            "max_core_negative_delta_after": 0.0,
            "max_noncore_negative_delta_after": 0.0,
            "max_positive_delta_after": 0.0,
        }

        new_white_stripe_meta = {
            "enabled": 0,
            "count": 0,
            "objects": [],
            "charging_changed_pixels": 0,
            "noisy_changed_pixels": 0,
            "positive_delta_pixels": 0,
            "negative_delta_pixels": 0,
        }

    source_stem = Path(source["source_filename"]).stem
    name = (
        f"{job['sample_id']:06d}__"
        f"{source['block_name']}__"
        f"{source_stem}__"
        f"v{variant:02d}.tif"
    )

    out_paths = build_output_paths_for_background(
        split,
        bg_level,
        name,
    )

    save_tif(out_paths["clean"], clean)
    save_tif(out_paths["noisy_charging"], charging)
    save_tif(out_paths["noisy"], noisy)

    if job["save_preview"]:
        cv2.imwrite(
            str(out_paths["preview"]),
            make_preview(clean, charging, noisy),
        )

        cv2.imwrite(
            str(out_paths["core_diagnostic"]),
            make_core_diagnostic_preview(
                core_mask_diagnostic,
                membrane_mask_diagnostic,
                clean_target_reference,
            ),
        )

    local_event_counter = {
        "separate": 0,
        "thin_bundle": 0,
        "joint": 0,
        "additive": 0,
        "two_pass": 0,
        "direct_place": 0,
        "thin_bundle_without_clean_core": 0,
        "joint_without_clean_core": 0,
    }

    for meta in event_metas:
        event_type = meta["event_type"]
        renderer = meta["renderer"]

        local_event_counter[event_type] = (
            local_event_counter.get(event_type, 0) + 1
        )
        local_event_counter[renderer] = (
            local_event_counter.get(renderer, 0) + 1
        )

        if event_type == "thin_bundle":
            if meta["clean_core_copied"]:
                raise RuntimeError(
                    "Thin bundle unexpectedly copied a core."
                )
            local_event_counter["thin_bundle_without_clean_core"] += 1

        if event_type == "joint":
            if meta["clean_core_copied"]:
                raise RuntimeError(
                    "Joint shading unexpectedly copied a core."
                )
            local_event_counter["joint_without_clean_core"] += 1

    row = {
        "sample_id": int(job["sample_id"]),
        "split": split,
        "block_name": source["block_name"],
        "source_file": source["source_filename"],
        "styled_source_file": source["styled_filename"],
        "styled_source_path": str(source["path"]),
        "variant": int(variant),
        "synthesis_seed": int(sample_seed),
        "step1_target_p50": float(source["target_p50"]),
        "background_level": int(bg_level),
        "background_tag": source["background_tag"],
        "step1_brightness_seed": source["step1_brightness_seed"],
        "step1_reference_seed": source["step1_reference_seed"],
        "step1_reference_slice_index": source["step1_reference_slice_index"],
        "artifact_library_split": split,
        "output_name": name,
        "clean_relpath": relative_to_output_root(out_paths["clean"]),
        "noisy_charging_relpath": relative_to_output_root(
            out_paths["noisy_charging"]
        ),
        "noisy_relpath": relative_to_output_root(out_paths["noisy"]),
        "sample_type": sample_type,
        "is_clean_identity": int(sample_type == "clean_identity"),
        "real_membrane_candidate_pixels": membrane_stats["candidate_pixels"],
        "real_membrane_raw_contrast_p75": membrane_stats["raw_contrast_p75"],
        "real_membrane_applied_darkening": membrane_stats["applied_darkening"],
        "num_charging_events": len(event_metas),
        "num_separate_events": sum(
            m["event_type"] == "separate"
            for m in event_metas
        ),
        "num_thin_bundle_events": sum(
            m["event_type"] == "thin_bundle"
            for m in event_metas
        ),
        "num_joint_events": sum(
            m["event_type"] == "joint"
            for m in event_metas
        ),
        "num_joint_cores_copied": sum(
            m["event_type"] == "joint"
            and m["clean_core_copied"]
            for m in event_metas
        ),
        "num_overlap_pixels_fused": sum(
            m["overlap_pixels_fused"]
            for m in event_metas
        ),
        "artefact_contrast_amplification_enabled": (
            amplification_meta["amplification_enabled"]
        ),
        "artefact_contrast_target_factor": (
            amplification_meta["target_factor"]
        ),
        "artefact_contrast_effective_factor": (
            amplification_meta["effective_factor"]
        ),
        "artefact_contrast_limited_by_headroom": (
            amplification_meta["limited_by_headroom"]
        ),
        "charging_events_json": json.dumps(event_metas),
        "artefact_contrast_json": json.dumps(amplification_meta),
        "new_white_stripe_enabled": new_white_stripe_meta["enabled"],
        "new_white_stripe_count": new_white_stripe_meta["count"],
        "new_white_stripe_json": json.dumps(new_white_stripe_meta),
    }

    return row, local_event_counter


def aggregate_event_counters(counter_list):
    """Aggregate diagnostic event counts across synthesis jobs."""
    total = {}

    for counter in counter_list:
        for key, value in counter.items():
            total[key] = total.get(key, 0) + int(value)

    return total


def validate_rows_and_outputs(rows):
    """Validate expected split and brightness group output counts."""
    expected_total = sum(
        EXPECTED_SOURCE_COUNTS[s] * VARIANTS_PER_CLEAN
        for s in SPLITS
    )

    if len(rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} metadata rows, got {len(rows)}."
        )

    sample_ids = [int(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Duplicate sample_id detected.")

    output_names = [row["output_name"] for row in rows]
    if len(set(output_names)) != len(output_names):
        raise RuntimeError("Duplicate output_name detected.")

    count_summary = {}

    for split in SPLITS:
        split_rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        expected_split = (
            EXPECTED_SOURCE_COUNTS[split]
            * VARIANTS_PER_CLEAN
        )

        if len(split_rows) != expected_split:
            raise RuntimeError(
                f"{split}: expected {expected_split} synthetic samples, "
                f"got {len(split_rows)}."
            )

        count_summary[split] = {}

        expected_per_level = (
            EXPECTED_SOURCE_COUNTS[split]
            // len(BACKGROUND_LEVELS)
            * VARIANTS_PER_CLEAN
        )

        for level in BACKGROUND_LEVELS:
            level_rows = [
                row
                for row in split_rows
                if int(row["background_level"]) == int(level)
            ]

            if len(level_rows) != expected_per_level:
                raise RuntimeError(
                    f"{split}/bg_{level}: expected {expected_per_level}, "
                    f"got {len(level_rows)}."
                )

            for kind in ("clean", "noisy_charging", "noisy"):
                folder = (
                    OUTPUT_ROOT
                    / split
                    / BACKGROUND_TAGS[level]
                    / kind
                )

                tif_count = len(find_tifs(folder))
                if tif_count != expected_per_level:
                    raise RuntimeError(
                        f"{folder}: expected {expected_per_level} TIFFs, "
                        f"found {tif_count}."
                    )

            count_summary[split][f"bg_{level}"] = {
                "all": int(len(level_rows)),
                "clean_identity": int(sum(
                    row["sample_type"] == "clean_identity"
                    for row in level_rows
                )),
                "degraded": int(sum(
                    row["sample_type"] == "degraded"
                    for row in level_rows
                )),
            }

    return count_summary


def write_metadata(rows):
    """Sort synthesis metadata by sample ID and write a CSV file."""
    rows = sorted(
        rows,
        key=lambda r: int(r["sample_id"]),
    )

    path = OUTPUT_ROOT / "metadata.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def parse_args():
    """Parse portable data paths and original-default synthesis controls."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic 1024x1024 Stage-1 artifact synthesis."
    )

    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Base directory for default paths (default: data).",
    )
    parser.add_argument(
        "--style-root", type=Path, default=None,
        help="Stage-1 Step 1 styled 1024x1024 TIFF dataset root.",
    )
    parser.add_argument(
        "--style-metadata", type=Path, default=None,
        help="Stage-1 Step 1 metadata CSV (default: <style-root>/metadata.csv).",
    )
    parser.add_argument(
        "--artifact-library-root", type=Path, default=None,
        help="Charging Library Step 4 root containing library_index.csv and entries/.",
    )
    parser.add_argument(
        "--charging-statistics-summary", type=Path, default=None,
        help="Charging Library Step 1 summary.json for global calibration.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Destination for synthetic TIFF pairs, metadata, and previews.",
    )
    parser.add_argument(
        "--artifact-train-start", type=int, default=ARTEFACT_TRAIN_SLICE_START,
        help=f"First training template slice (default: {ARTEFACT_TRAIN_SLICE_START}).",
    )
    parser.add_argument(
        "--artifact-train-end", type=int, default=ARTEFACT_TRAIN_SLICE_END,
        help=f"Last training template slice (default: {ARTEFACT_TRAIN_SLICE_END}).",
    )
    parser.add_argument(
        "--artifact-val-start", type=int, default=ARTEFACT_VAL_SLICE_START,
        help=f"First validation template slice (default: {ARTEFACT_VAL_SLICE_START}).",
    )
    parser.add_argument(
        "--artifact-val-end", type=int, default=ARTEFACT_VAL_SLICE_END,
        help=f"Last validation template slice (default: {ARTEFACT_VAL_SLICE_END}).",
    )
    parser.add_argument(
        "--artifact-test-start", type=int, default=ARTEFACT_TEST_SLICE_START,
        help=f"First test template slice (default: {ARTEFACT_TEST_SLICE_START}).",
    )
    parser.add_argument(
        "--artifact-test-end", type=int, default=ARTEFACT_TEST_SLICE_END,
        help=f"Last test template slice (default: {ARTEFACT_TEST_SLICE_END}).",
    )
    parser.add_argument(
        "--master-seed", type=int, default=MASTER_SEED,
        help=f"Deterministic synthesis seed (default: {MASTER_SEED}).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_NUM_WORKERS,
        help=f"CPU sample-level worker threads (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--no-previews", action="store_true",
        help="Disable preview PNG generation (does not change synthetic TIFF outputs).",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help=(
            "Validate source split, metadata, artifact-library index, and "
            "assignments without synthesis; does not validate all artifact TIFF payloads."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete and recreate the output directory if it already exists.",
    )
    return parser.parse_args()


def configure_from_args(args):
    """Set module paths and user-selected controls before any image computation."""
    global STYLE_ROOT, STEP1_METADATA_PATH, ARTEFACT_LIBRARY_ROOT
    global STEP0_SUMMARY_PATH, OUTPUT_ROOT, EXPERIMENT_DESCRIPTION
    global MASTER_SEED, SAVE_PREVIEWS
    global ARTEFACT_TRAIN_SLICE_START, ARTEFACT_TRAIN_SLICE_END
    global ARTEFACT_VAL_SLICE_START, ARTEFACT_VAL_SLICE_END
    global ARTEFACT_TEST_SLICE_START, ARTEFACT_TEST_SLICE_END

    data_root = args.data_root
    STYLE_ROOT = args.style_root if args.style_root is not None else (
        data_root / "stage1_style_transfer_4levels"
    )
    STEP1_METADATA_PATH = args.style_metadata if args.style_metadata is not None else (
        STYLE_ROOT / "metadata.csv"
    )
    ARTEFACT_LIBRARY_ROOT = (
        args.artifact_library_root if args.artifact_library_root is not None
        else data_root / "charging_artefact_library_train"
    )
    STEP0_SUMMARY_PATH = (
        args.charging_statistics_summary if args.charging_statistics_summary is not None
        else data_root / "real_charging_statistics" / "summary.json"
    )
    OUTPUT_ROOT = args.output_root if args.output_root is not None else (
        data_root / "stage1_synthesis"
    )
    EXPERIMENT_DESCRIPTION = OUTPUT_ROOT / "experiment.txt"

    ARTEFACT_TRAIN_SLICE_START = args.artifact_train_start
    ARTEFACT_TRAIN_SLICE_END = args.artifact_train_end
    ARTEFACT_VAL_SLICE_START = args.artifact_val_start
    ARTEFACT_VAL_SLICE_END = args.artifact_val_end
    ARTEFACT_TEST_SLICE_START = args.artifact_test_start
    ARTEFACT_TEST_SLICE_END = args.artifact_test_end
    MASTER_SEED = args.master_seed
    SAVE_PREVIEWS = not args.no_previews

    for split, start, end in (
        ("train", ARTEFACT_TRAIN_SLICE_START, ARTEFACT_TRAIN_SLICE_END),
        ("val", ARTEFACT_VAL_SLICE_START, ARTEFACT_VAL_SLICE_END),
        ("test", ARTEFACT_TEST_SLICE_START, ARTEFACT_TEST_SLICE_END),
    ):
        if start < 0 or end < start:
            raise ValueError(
                f"Invalid {split} artifact template range: {start}-{end}."
            )
    if not (
        ARTEFACT_TRAIN_SLICE_END < ARTEFACT_VAL_SLICE_START
        and ARTEFACT_VAL_SLICE_END < ARTEFACT_TEST_SLICE_START
    ):
        raise ValueError("Artifact train/val/test slice ranges must be disjoint and ordered.")

    input_roots = (STYLE_ROOT, ARTEFACT_LIBRARY_ROOT)
    resolved_output = OUTPUT_ROOT.resolve()
    for input_root in input_roots:
        resolved_input = input_root.resolve()
        if resolved_output == resolved_input or resolved_output in resolved_input.parents:
            raise ValueError(
                f"Output directory must not equal or contain an input directory: {input_root}"
            )
        if resolved_input in resolved_output.parents:
            raise ValueError(
                f"Output directory must not be inside an input directory: {input_root}"
            )
    if resolved_output == STEP0_SUMMARY_PATH.resolve() or resolved_output in STEP0_SUMMARY_PATH.resolve().parents:
        raise ValueError("Output directory must not overwrite charging statistics summary.")

    for purpose, path in (
        ("styled data directory", STYLE_ROOT),
        ("style metadata CSV", STEP1_METADATA_PATH),
        ("artifact library directory", ARTEFACT_LIBRARY_ROOT),
        ("artifact library index CSV", ARTEFACT_LIBRARY_ROOT / "library_index.csv"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {purpose}: {path}")
    # Preserve the original --check-only behavior: summary values are only read
    # when synthesis starts; report an absent summary immediately for normal runs.
    if not args.check_only and not STEP0_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing charging statistics summary JSON: {STEP0_SUMMARY_PATH}"
        )


def main():
    """Validate inputs, deterministically synthesize samples, and write metadata."""
    args = parse_args()
    configure_from_args(args)

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    # Avoid nested OpenCV threading when ThreadPoolExecutor handles sample-level
    # parallelism; the numerical rendering itself remains unchanged.
    cv2.setNumThreads(1)

    print("=" * 88)
    print("Stage-1 Step 2 | deterministic artifact synthesis")
    print(f"Style input    : {STYLE_ROOT}")
    print(f"Step-1 metadata: {STEP1_METADATA_PATH}")
    print(f"Artifact library: {ARTEFACT_LIBRARY_ROOT}")
    print(f"Charging stats  : {STEP0_SUMMARY_PATH}")
    print(f"Output         : {OUTPUT_ROOT}")
    print(f"Workers        : {args.workers}")
    print(f"Master seed    : {MASTER_SEED}")
    print("=" * 88)

    # ------------------------------------------------------------------
    # 1. Strict source validation: fixed split from Stage-1 Step 1.
    # ------------------------------------------------------------------
    source_records = load_style_source_records()

    print("\nStep-1 source split validated:")
    for split in SPLITS:
        split_records = [
            r for r in source_records
            if r["split"] == split
        ]

        counts = {
            level: sum(
                int(r["background_level"]) == int(level)
                for r in split_records
            )
            for level in BACKGROUND_LEVELS
        }

        print(
            f"  {split:5s}: total={len(split_records)} | "
            + " | ".join(
                f"{level}={counts[level]}"
                for level in BACKGROUND_LEVELS
            )
        )

    # ------------------------------------------------------------------
    # 2. Strict artifact-library split validation.
    # ------------------------------------------------------------------
    library = load_library_index()

    print("\nArtifact-library split:")
    library_summary = {}

    for split in SPLITS:
        records = library[split]
        z_values = sorted({
            int(record["slice_idx"])
            for record in records
        })
        group_ids = sorted({
            int(record["group_id"])
            for record in records
        })

        library_summary[split] = {
            "num_entries": int(len(records)),
            "slice_indices": z_values,
            "group_ids": group_ids,
        }

        print(
            f"  {split:5s}: entries={len(records)}, "
            f"z={z_values[0]}..{z_values[-1]}, "
            f"groups={len(group_ids)}"
        )

    # ------------------------------------------------------------------
    # 3. Deterministic clean-identity and contrast-factor assignments.
    # ------------------------------------------------------------------
    clean_identity_keys, clean_identity_stats = (
        build_clean_identity_keys(source_records)
    )

    factor_assignments, factor_balance_stats = (
        build_balanced_factor_assignments(
            source_records,
            clean_identity_keys,
        )
    )

    jobs = build_jobs(
        source_records,
        clean_identity_keys,
        factor_assignments,
    )

    print("\nPlanned synthesis:")
    for split in SPLITS:
        split_jobs = [
            job
            for job in jobs
            if job["source"]["split"] == split
        ]

        clean_count = sum(
            job["make_clean_identity"]
            for job in split_jobs
        )

        print(
            f"  {split:5s}: "
            f"sources={EXPECTED_SOURCE_COUNTS[split]}, "
            f"variants/source={VARIANTS_PER_CLEAN}, "
            f"samples={len(split_jobs)}, "
            f"clean_identity={clean_count}, "
            f"degraded={len(split_jobs) - clean_count}"
        )

    print(
        "\nDegraded 1024 event schedule: "
        f"{NUM_REGULAR_SEPARATE_EVENTS} regular + "
        f"{NUM_THIN_BUNDLE_EVENTS} thin-bundle + "
        f"{NUM_JOINT_EVENTS} joint + "
        f"{NEW_WHITE_STRIPE_COUNT_RANGE[0]}-"
        f"{NEW_WHITE_STRIPE_COUNT_RANGE[1]} white-stripe"
    )

    print(
        "White-stripe half-length px: "
        f"broad={BROAD_HALF_LENGTH_RANGE_PX}, "
        f"alternating={ALT_HALF_LENGTH_RANGE_PX}"
    )

    print("Contrast-factor balance:")
    for key in sorted(factor_balance_stats):
        print(
            f"  {key}: "
            f"{factor_balance_stats[key]['factor_counts']}"
        )

    if args.check_only:
        print(
            "\nCHECK ONLY complete. "
            "No synthesis output was written."
        )
        return

    # ------------------------------------------------------------------
    # 4. Output creation only after all preflight checks succeeded.
    # ------------------------------------------------------------------
    prepare_output_root(
        overwrite=args.overwrite
    )

    this_file = Path(__file__).resolve()
    if not this_file.exists():
        raise FileNotFoundError(
            f"Cannot snapshot running source file: {this_file}"
        )

    code_snapshot_path = OUTPUT_ROOT / this_file.name
    shutil.copy2(
        this_file,
        code_snapshot_path,
    )

    EXPERIMENT_DESCRIPTION.write_text(
        "\n".join([
            "Stage-1 synthetic artifact dataset",
            "Scientific implementation: original accepted rendering and sampling.",
            "Master biological split: inherited unchanged from Stage-1 Step 1.",
            "Canvas: native uint8 1024x1024.",
            f"Variants per clean: {VARIANTS_PER_CLEAN}.",
            f"Degraded event schedule: {NUM_REGULAR_SEPARATE_EVENTS} regular charging + "
            f"{NUM_THIN_BUNDLE_EVENTS} thin bundles + {NUM_JOINT_EVENTS} joint shading + "
            f"{NEW_WHITE_STRIPE_COUNT_RANGE[0]}-{NEW_WHITE_STRIPE_COUNT_RANGE[1]} "
            "horizontal white-stripe objects.",
            "Artifact physical/pixel scale retained at JRC 8 nm/pixel.",
            "White-stripe length uses the original absolute-pixel distribution.",
            f"Clean identity: {CLEAN_IDENTITY_FRACTION:.0%}, independently balanced "
            "per split x brightness.",
            "Contrast factors: balanced independently per split x brightness.",
            f"Artifact-library split: train {ARTEFACT_TRAIN_SLICE_START}-{ARTEFACT_TRAIN_SLICE_END}; "
            f"val {ARTEFACT_VAL_SLICE_START}-{ARTEFACT_VAL_SLICE_END}; "
            f"test {ARTEFACT_TEST_SLICE_START}-{ARTEFACT_TEST_SLICE_END}.",
            "Per-sample RNG: stable SHA256-derived seed.",
            "Global clean-target HPF: disabled.",
            "Final white noise: disabled.",
        ]) + "\n",
        encoding="utf-8",
    )

    # Load non-random global calibration once.
    step0_stats = load_step0_stats()

    # Preload artifact entries once. During parallel synthesis the cache is read-only.
    cache = preload_artifact_cache(
        library
    )

    # ------------------------------------------------------------------
    # 5. Parallel deterministic synthesis.
    # ------------------------------------------------------------------
    rows = []
    counters = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_map = {
            executor.submit(
                process_one_job,
                job,
                library,
                cache,
                step0_stats,
            ): job
            for job in jobs
        }

        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Artifact synthesis",
            unit="sample",
        ):
            job = future_map[future]

            try:
                row, local_counter = future.result()
            except Exception as exc:
                source = job["source"]
                raise RuntimeError(
                    "Synthesis failed for:\n"
                    f"  sample_id={job['sample_id']}\n"
                    f"  split={source['split']}\n"
                    f"  block={source['block_name']}\n"
                    f"  source={source['source_filename']}\n"
                    f"  variant={job['variant']}\n"
                    f"  seed={job['sample_seed']}"
                ) from exc

            rows.append(row)
            counters.append(local_counter)

    rows.sort(
        key=lambda r: int(r["sample_id"])
    )

    if not rows:
        raise RuntimeError(
            "No synthesis samples were produced."
        )

    metadata_path = write_metadata(
        rows
    )

    count_summary = validate_rows_and_outputs(
        rows
    )

    event_counter = aggregate_event_counters(
        counters
    )

    # ------------------------------------------------------------------
    # 6. Full reproducibility summary.
    # ------------------------------------------------------------------
    realized_clean_identity_samples = int(sum(
        row["sample_type"] == "clean_identity"
        for row in rows
    ))

    summary = {
        "step": "stage1_step2_artifact_synthesis",
        "synthesis_experiment": SYNTHESIS_EXPERIMENT,
        "code_version": CODE_VERSION,
        "master_seed": MASTER_SEED,
        "stable_seed_method": (
            "SHA256-derived 32-bit seed from MASTER_SEED + "
            "split + block + source slice + variant"
        ),
        "workers_used": int(args.workers),
        "input_style_root": str(STYLE_ROOT),
        "step1_metadata": str(STEP1_METADATA_PATH),
        "output_root": str(OUTPUT_ROOT),
        "image_size": int(IMAGE_SIZE),
        "variants_per_clean": int(VARIANTS_PER_CLEAN),
        "source_counts": EXPECTED_SOURCE_COUNTS,
        "output_sample_counts": {
            split: int(
                EXPECTED_SOURCE_COUNTS[split]
                * VARIANTS_PER_CLEAN
            )
            for split in SPLITS
        },
        "split_policy": (
            "Stage-1 Step-1 train/val/test membership is inherited exactly; "
            "no region resplitting and no cross-split shuffle"
        ),
        "artifact_library_split": library_summary,
        "clean_identity": {
            "target_fraction": CLEAN_IDENTITY_FRACTION,
            "selection_scope": "independent within every split x brightness group",
            "selection_stats": clean_identity_stats,
            "realized_total": realized_clean_identity_samples,
        },
        "charging_event_schedule_1024": {
            "regular_separate": NUM_REGULAR_SEPARATE_EVENTS,
            "thin_bundle": NUM_THIN_BUNDLE_EVENTS,
            "joint": NUM_JOINT_EVENTS,
            "charging_total": NUM_EVENTS_PER_SLICE,
            "new_white_stripe_count_range": list(
                NEW_WHITE_STRIPE_COUNT_RANGE
            ),
            "approx_total_artifact_objects_range": [
                NUM_EVENTS_PER_SLICE
                + NEW_WHITE_STRIPE_COUNT_RANGE[0],
                NUM_EVENTS_PER_SLICE
                + NEW_WHITE_STRIPE_COUNT_RANGE[1],
            ],
        },
        "density_adaptation": (
            "the previous 2048-pixel setup used 25 charging events and 4-6 stripe "
            "objects; this tuned 1024 version uses 11 charging events + exactly 3 white-stripe "
            "objects to preserve approximate artifact-object density"
        ),
        "spatial_scale": {
            "clean_pixel_size_nm": CLEAN_PIXEL_SIZE_NM,
            "reference_clean_pixel_size_nm": REFERENCE_CLEAN_PIXEL_SIZE_NM,
            "reference_artifact_scale": REFERENCE_ARTEFACT_SCALE,
            "random_multiplier_range": list(
                RANDOM_ARTEFACT_SCALE_RANGE
            ),
            "base_physical_scale": (
                REFERENCE_ARTEFACT_SCALE
                * REFERENCE_CLEAN_PIXEL_SIZE_NM
                / CLEAN_PIXEL_SIZE_NM
            ),
        },
        "anchor_sampling": {
            "random_position_margin_px": RANDOM_POSITION_MARGIN,
            "minimum_anchor_distance_px": MIN_ANCHOR_DISTANCE,
            "candidate_count": ANCHOR_CANDIDATE_COUNT,
            "default_placement_tries": MAX_ANCHOR_PLACEMENT_TRIES,
            "regular_separate_placement_tries": (
                REGULAR_SEPARATE_ANCHOR_PLACEMENT_TRIES
            ),
            "thin_bundle": {
                "mode": "footprint-aware maximin with overlap cap",
                "max_overlap_fraction_with_existing_dark_support": (
                    THIN_BUNDLE_MAX_OVERLAP_FRACTION
                ),
                "placement_tries": THIN_BUNDLE_PLACEMENT_TRIES,
                "full_footprint_must_fit_canvas": True,
                "small_overlap_allowed": True,
            },
        },
        "artifact_contrast_amplification": {
            "policy": (
                "only non-core dark artifacts are amplified; "
                "background, dark core, and positive halo remain unchanged"
            ),
            "factors_by_background": {
                str(level): [
                    float(x)
                    for x in ARTEFACT_CONTRAST_FACTORS_BY_BACKGROUND[level]
                ]
                for level in BACKGROUND_LEVELS
            },
            "assignment_scope": (
                "near-equal counts independently within every "
                "split x brightness group"
            ),
            "balance_stats": factor_balance_stats,
        },
        "white_stripe": {
            "enabled": True,
            "count_range": list(
                NEW_WHITE_STRIPE_COUNT_RANGE
            ),
            "subtype_probabilities": NEW_WHITE_STRIPE_SUBTYPE_PROBS,
            "broad_half_length_range_px": list(
                BROAD_HALF_LENGTH_RANGE_PX
            ),
            "alternating_half_length_range_px": list(
                ALT_HALF_LENGTH_RANGE_PX
            ),
            "length_policy": (
                "absolute-pixel ranges derived from the original 2048-pixel setup; "
                "not halved with the 1024 canvas"
            ),
            "broad_side_width_range_px": list(
                BROAD_SIDE_WIDTH_RANGE_PX
            ),
            "broad_seam_half_width_range_px": list(
                BROAD_SEAM_HALF_WIDTH_RANGE_PX
            ),
            "broad_side_dark_delta_range": list(
                BROAD_SIDE_DARK_DELTA_RANGE
            ),
            "broad_seam_bright_delta_range": list(
                BROAD_SEAM_BRIGHT_DELTA_RANGE
            ),
            "alternating_band_count_range": list(
                ALT_BAND_COUNT_RANGE
            ),
            "alternating_stripe_half_width_range_px": list(
                ALT_STRIPE_HALF_WIDTH_RANGE_PX
            ),
            "alternating_band_gap_range_px": list(
                ALT_BAND_GAP_RANGE_PX
            ),
        },
        "clean_target": {
            "global_hpf_enhancement": False,
            "clean_identity_mapping": "styled clean -> same styled clean",
            "degraded_target_rule": (
                "original clean-core target construction; "
                "no global HPF"
            ),
        },
        "final_white_noise": {
            "enabled": False,
        },
        "count_summary": count_summary,
        "realized_event_counter": event_counter,
        "metadata_csv": str(metadata_path),
        "code_snapshot_path": str(code_snapshot_path),
    }

    with open(
        OUTPUT_ROOT / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nDone.")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Metadata    : {metadata_path}")
    print(f"Samples     : {len(rows)}")
    print(
        f"Clean identity: "
        f"{realized_clean_identity_samples}/{len(rows)}"
    )

    for split in SPLITS:
        print(f"  {split}:")
        for level in BACKGROUND_LEVELS:
            item = count_summary[split][f"bg_{level}"]
            print(
                f"    bg_{level}: "
                f"all={item['all']}, "
                f"degraded={item['degraded']}, "
                f"identity={item['clean_identity']}"
            )

    print(
        "Realized charging-event counts: "
        f"{event_counter}"
    )


if __name__ == "__main__":
    main()

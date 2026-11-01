#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cryo-FIB-SEM Stage-2 Step 1: 2.5D Biological Refinement Training

Author: Ningyan Zhang
Last modified: October 9, 2026

Description:
    Train the fixed five-slice refinement network to restore membrane
    structure and suppress residual artifacts in cryo-FIB-SEM images.
    Stage-1 and Stage-2 share the same JRC Liver clean master blocks and
    identical train/validation/test partition, but they do NOT share their
    processed training inputs. Stage-2 generates its own 2.5D training
    samples and membrane-enhanced targets from the original clean images.

Required data:
    --data-root PATH       Shared JRC Liver clean master directory containing:
                               train/block_*/*.tif  (16 blocks)
                               val/block_*/*.tif    ( 3 blocks)
                               test/block_*/*.tif   ( 3 blocks)
                           Each block contains exactly 16 grayscale uint8
                           TIFF slices of size 1024x1024 (ZYX source blocks).
                           The test split is checked for consistency but is
                           NEVER used to train, validate, select checkpoints,
                           or compute preprocessing statistics.

Optional arguments:
    --cache-root PATH      Separate deterministic preprocessing cache.
                           Default: experiments/stage2_cache
                           Automatically built/reused from TRAIN and VAL;
                           this is NOT an additional required input dataset.
    --output-root PATH     Dedicated training run directory.
                           Default: experiments/stage2_training
    --device {auto,cuda,mps,cpu}
                           Default: auto (CUDA > Apple MPS > CPU).
                           Explicit device choices never silently fall back.
    --steps N              Optimizer steps (default: 2000). For short end-to-end
                           testing, specify e.g. --steps 10 or --steps 20.
                           This controls the complete training loop and the
                           checkpoint, loss-history, and summary step counts.
    --cache-workers N      Parallel block-cache build workers (default: 2).
                           Does not parallelize stochastic batch sampling.
    --check-only           Validate dataset geometry, device availability,
                           and directory safety without creating files.
    --overwrite            Explicitly allow deletion of an existing output
                           directory before training; disabled by default.

Output structure:
    --output-root/
        checkpoints/
            best.pt                  Lowest weighted validation score.
            final.pt                 Model after the last optimizer step.
        validation_previews/
            step_XXXX/               Fixed seven-condition comparisons.
        train_loss.npy               Per-step training pair loss.
        train_missing_mask_loss.npy  Missing-core diagnostic loss only.
        val_steps.npy                Steps at which validation ran.
        loss.png                     Training and validation loss curves.
        train_summary.json           Hyperparameters, metrics, and timing.
        experiment.txt               Human-readable experiment provenance.
        step1_train.py               Snapshot of the executed source code.
    Cache files reside outside the output directory and are not deliverables.

Usage example A: Train on an NVIDIA GPU
    python step1_train.py \
        --data-root /path/to/clean_blocks_1024_z16 \
        --cache-root /path/to/stage2_cache \
        --output-root /path/to/stage2_training \
        --device cuda

Usage example B: Train on Apple Silicon (MPS)
    python step1_train.py \
        --data-root /path/to/clean_blocks_1024_z16 \
        --output-root /path/to/stage2_training_mps \
        --device mps \
        --steps 20

Usage example C: Validate without writing files
    python step1_train.py \
        --data-root /path/to/clean_blocks_1024_z16 \
        --output-root /path/to/stage2_training \
        --device cuda \
        --check-only

Scientific processing (preserve when maintaining this script):
    1. For each complete clean slice, standardize using its own mean and
       standard deviation; subtract a Gaussian low-pass (sigma=10), center
       by the mode of a 50-bin high-pass histogram, and clip to [-3, 3].
       This processed representation is the model's float32 input space.
    2. Construct the center-slice target by applying the frozen deterministic
       membrane-selective HPF to the original uint8 image, then expressing
       the enhanced image in the ORIGINAL center-slice preprocessing basis.
    3. Cache original uint8 slices, full-slice processed float32 arrays,
       HPF target arrays, and membrane placement scores as mmap-backed NPYs.
       Cache validity includes source-content SHA-256 and preprocessing
       parameters. No random training artifacts are cached.
    4. Sample a five-slice context [z-2, z-1, z, z+1, z+2], then independent
       256x256 patches. Exactly one major condition is selected per patch:
           clean 40% | missing region 30% | local drift 15% | focus loss 15%.
       Local drift moves neighboring crop locations while keeping the center
       fixed. Focus loss blurs one COMPLETE raw slice before preprocessing.
       Missing-region replacement is blended AFTER Gaussian training noise.
    5. Apply independent processed-space overlays: stripe family 70%, with
       white-core/black-side and black-core/white-side variants at 35% each
       (mutually exclusive per patch); residual thin bundle 15%.
       Add Gaussian noise (sigma=1.0) in the original training order.
    6. Train the fixed 2,204,289-parameter early-fusion 2.5D U-Net using
       five input channels and one center-slice output. The full-resolution
       decoder skip uses ONLY the center input. Preserve scratch LeCun-normal
       initialization, dropout=0.1, and the original pair loss (squared
       reconstruction error plus squared negative-structure error).
    7. Train with Adam (lr=2e-5, betas=0.9/0.999, no weight decay), batch=32,
       2,000 optimization steps by default (--steps changes run length only),
       seed=0, and no scheduler or early stopping.
       A single sequential CPU producer preserves NumPy sampling order.
    8. Validate on 32 fixed samples every 100 steps AND at the last requested
       step. Select best.pt using 0.5*clean + (missing + drift + focus)/6.
       Stripe and thin-bundle cases are visual diagnostics, not checkpoint-
       selection terms. Save fixed visual comparisons every 250 steps and at
       the last requested step; the standard 2000-step schedule is unchanged.

Reproducibility and safety:
    - Training runs in FP32 on CUDA, MPS, or CPU; no AMP is used.
    - Different hardware backends need not generate identical Torch noise,
      dropout masks, optimizer trajectories, or selected best steps.
    - --steps sets the run's total length, not a continuation point. A shorter
      smoke test starts from a freshly initialized model and does not modify
      the default 2000-step training protocol.
    - Checkpoints contain model weights and metadata, but not the complete
      optimizer, stochastic generator, and prefetch state. Exact mid-run
      resume is NOT supported.
    - Master data, cache, and output paths must be separate non-overlapping
      directory trees. Existing outputs are protected unless --overwrite
      is provided explicitly.

Requirements: PyTorch, NumPy, SciPy, OpenCV, tifffile, matplotlib, tqdm.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Paths
# ============================================================

# The clean master is shared with Stage-1 (same 16/3/3 block partition).
# User-provided paths override these portable defaults at invocation time.
DATA_ROOT = Path("data") / "clean_blocks_1024_z16"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

# Cache must be separate from both immutable master TIFFs and training output.
CACHE_ROOT = Path("experiments") / "stage2_cache"

EXPERIMENT_NAME = "stage2_biological_refinement"
TRAINING_CODE_VERSION = "2026-10-08"
EXPERIMENT_ROOT = Path("experiments") / "stage2_training"

CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"

# Only two checkpoints are kept:
# - best.pt  : lowest weighted validation score seen during training
# - final.pt : model state after the last training step
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "final.pt"

VALIDATION_PREVIEW_DIR = EXPERIMENT_ROOT / "validation_previews"

OVERWRITE_EXPERIMENT = False

# Strict frozen master-split geometry.
EXPECTED_BLOCK_COUNTS = {
    "train": 16,
    "val": 3,
    "test": 3,
}
EXPECTED_SLICES_PER_BLOCK = 16
EXPECTED_HEIGHT = 1024
EXPECTED_WIDTH = 1024

# Engineering-only acceleration knobs.
CACHE_BUILD_WORKERS = 2
ENABLE_BATCH_PREFETCH = True
PREFETCH_DEPTH = 1  # scientifically intentional: one sequential producer only
CACHE_FORMAT_VERSION = 2  # Adds per-source content SHA-256 to the manifest


# ============================================================
# 2. Data / training settings
# ============================================================

NUM_CONTEXT_SLICES = 5
CONTEXT_RADIUS = 2

PATCH_SIZE = 256
BATCH_SIZE = 32

NOISE_SIGMA = 1.0
DROPOUT = 0.1

LR = 2e-5
TRAIN_STEPS = 2000
SEED = 0

# ------------------------------------------------------------
# Major patch-type mixture
# ------------------------------------------------------------
# The three major structural corruptions are mutually exclusive.
#
# Preserve the 30% missing-region rate and a substantial
# clean fraction, while introducing drift and lose-focus conservatively.
TRAIN_CLEAN_PROB = 0.40
TRAIN_MISSING_PROB = 0.30
TRAIN_LOCAL_DRIFT_PROB = 0.15
TRAIN_LOSE_FOCUS_PROB = 0.15

if not np.isclose(
    TRAIN_CLEAN_PROB
    + TRAIN_MISSING_PROB
    + TRAIN_LOCAL_DRIFT_PROB
    + TRAIN_LOSE_FOCUS_PROB,
    1.0,
):
    raise ValueError("Training sample probabilities must sum to 1.0.")


# ------------------------------------------------------------
# Stripe family overlay - Stage-2 original + polarity inversion
# ------------------------------------------------------------
# The major clean/missing/local-drift/lose-focus mixture above is untouched.
#
# For every 256x256 patch, stripe-family selection is ONE categorical draw:
#   30% -> no stripe
#   35% -> original white-core + black-side stripe
#   35% -> inverted black-core + white-side stripe
#
# Therefore original and inverted stripes can NEVER occur together in the
# same 256x256 patch.
TRAIN_STRIPE_FAMILY_PROB = 0.70
STRIPE_ORIGINAL_FRACTION = 0.50
STRIPE_INVERTED_FRACTION = 0.50

if not np.isclose(
    STRIPE_ORIGINAL_FRACTION
    + STRIPE_INVERTED_FRACTION,
    1.0,
):
    raise ValueError(
        "Stripe polarity fractions must sum to 1.0."
    )

TRAIN_ORIGINAL_STRIPE_MARGINAL_PROB = (
    TRAIN_STRIPE_FAMILY_PROB
    * STRIPE_ORIGINAL_FRACTION
)

TRAIN_INVERTED_STRIPE_MARGINAL_PROB = (
    TRAIN_STRIPE_FAMILY_PROB
    * STRIPE_INVERTED_FRACTION
)

STRIPE_TRAIN_RNG_SEED_OFFSET = 3000
STRIPE_VAL_RNG_SEED_OFFSET = 2000

STRIPE_COUNT_RANGE = (4, 7)
STRIPE_HALF_WIDTH_RANGE_PX = (2, 4)
STRIPE_HALF_LENGTH_FRAC_RANGE = (0.30, 0.48)
STRIPE_DARK_BAND_HALF_WIDTH_RANGE_PX = (2, 4)
STRIPE_SOFT_EDGE_SIGMA = 2.0
STRIPE_DARK_SOFT_EDGE_SIGMA = 1.0
STRIPE_BRIGHT_DELTA_RANGE = (1.00, 1.28)
STRIPE_DARK_DELTA_RANGE = (0.72, 0.92)
STRIPE_CENTER_Y_JITTER_PX = 2.5
STRIPE_CENTER_X_JITTER_PX = 5.0
STRIPE_HALF_WIDTH_JITTER_PX = 0.6
STRIPE_DARK_BAND_HALF_WIDTH_JITTER_PX = 0.4
STRIPE_HALF_LENGTH_JITTER_PX = 8
STRIPE_STRENGTH_SCALE_RANGE = (0.86, 1.14)
STRIPE_ANGLE_DEG = 0.0
MIN_STRIPE_VERTICAL_GAP = 14


# ------------------------------------------------------------
# Independent residual thin-bundle overlay configuration
# ------------------------------------------------------------
TRAIN_RESIDUAL_THIN_BUNDLE_PROB = 0.15
THIN_BUNDLE_TRAIN_RNG_SEED_OFFSET = 4000
THIN_BUNDLE_VAL_RNG_SEED_OFFSET = 5000
THIN_BUNDLE_LENGTH_RANGE_PX = (80, 190)
THIN_BUNDLE_LINE_COUNT_RANGE = (8, 20)
THIN_BUNDLE_LINE_SPACING_RANGE_PX = (2, 5)
THIN_BUNDLE_LINE_THICKNESS_RANGE_PX = (1, 2)
THIN_BUNDLE_ENDPOINT_JITTER_FRAC = 0.16
THIN_BUNDLE_CENTER_DRIFT_FRAC = 0.10
THIN_BUNDLE_LINE_KEEP_PROB_RANGE = (0.68, 0.90)
THIN_BUNDLE_GAP_COUNT_RANGE = (0, 2)
THIN_BUNDLE_GAP_LENGTH_RANGE_PX = (6, 24)
THIN_BUNDLE_LINE_STRENGTH_RANGE = (0.72, 1.00)
THIN_BUNDLE_DELTA_RANGE = (0.32, 0.62)
THIN_BUNDLE_Z_X_JITTER_PX = 4
THIN_BUNDLE_Z_Y_JITTER_PX = 2
THIN_BUNDLE_Z_STRENGTH_RANGE = (0.88, 1.12)
THIN_BUNDLE_SIGNATURE_SIGMA = 10.0


# ------------------------------------------------------------
# Missing-region synthesis parameters
# ------------------------------------------------------------

MISSING_CENTER_ONLY_PROB = 0.50
MISSING_CENTER_DOMINANT_PROB = 0.25
MISSING_NEIGHBOR_ONLY_PROB = 0.15
MISSING_IRREGULAR_MULTI_PROB = 0.10

MISSING_BASE_AREA_RANGE = (2500.0, 8000.0)
MISSING_POSITION_JITTER_PX = 14.0

MISSING_SOFT_EDGE_SIGMA = 2.5
MISSING_RING_INNER_DILATE = 6
MISSING_RING_OUTER_DILATE = 18

MISSING_GRAY_OFFSET_POSITIVE_PROB = 0.80
MISSING_GRAY_OFFSET_POSITIVE = (0.10, 0.24)
MISSING_GRAY_OFFSET_NEGATIVE = (-0.14, -0.06)

MISSING_TEXTURE_SIGMA = (5.0, 12.0)
MISSING_TEXTURE_AMPLITUDE = (0.015, 0.055)
MISSING_GRADIENT_AMPLITUDE = (0.00, 0.05)

# Membrane-aware crop / placement retained from the baseline logic.
MISSING_CROP_CANDIDATES = 24
MISSING_CROP_TOP_K = 6
MISSING_PLACEMENT_VALID_MARGIN = 28
MISSING_PLACEMENT_TOP_PERCENTILE = 78.0

MISSING_SCORE_RAW_DARK_WEIGHT = 0.40
MISSING_SCORE_PROCESSED_NEG_WEIGHT = 0.40
MISSING_SCORE_GRADIENT_WEIGHT = 0.20
MISSING_SCORE_SMOOTH_SIGMA = 1.4


# ------------------------------------------------------------
# Local inter-slice drift parameters
# ------------------------------------------------------------

# Center slice is always fixed. Only neighbors can be shifted.
LOCAL_DRIFT_AFFECTED_NEIGHBOR_COUNT = (1, 4)

# Magnitude is Euclidean displacement length in pixels.
LOCAL_DRIFT_MAG_RANGE_PX = (4.0, 15.0)


# ------------------------------------------------------------
# Loss-of-focus blur parameters
# ------------------------------------------------------------

# Exactly one slice per lose-focus positive.
# Center receives the highest probability.
LOSE_FOCUS_INDEX_WEIGHTS = np.asarray(
    [0.10, 0.15, 0.50, 0.15, 0.10],
    dtype=np.float64,
)
LOSE_FOCUS_INDEX_WEIGHTS /= LOSE_FOCUS_INDEX_WEIGHTS.sum()

LOSE_FOCUS_SIGMA_RANGE = (4.0, 5.8)
LOSE_FOCUS_ALPHA_RANGE = (0.92, 0.96)


# ------------------------------------------------------------
# Validation / preview
# ------------------------------------------------------------

VAL_SAMPLES = 32
VAL_BATCH_SIZE = 8
VAL_EVERY = 100

# Clean behavior remains dominant in checkpoint selection.
VAL_SCORE_CLEAN_WEIGHT = 0.50
VAL_SCORE_MISSING_WEIGHT = 1.0 / 6.0
VAL_SCORE_DRIFT_WEIGHT = 1.0 / 6.0
VAL_SCORE_FOCUS_WEIGHT = 1.0 / 6.0

# Fixed validation visualization.
VALIDATION_PREVIEW_EVERY = 250
VALIDATION_PREVIEW_COUNT = 4

# ============================================================
# 3. full-slice preprocessing
# ============================================================

PREPROCESS_GAUSSIAN_SIGMA = 10.0
PREPROCESS_HIST_BINS = 50

CLIP_MIN = -3.0
CLIP_MAX = 3.0


# ============================================================
# 4. Membrane-selective HPF
#    Exact parameters from the previous accepted HPF version.
# ============================================================

HPF_SIGMA = 2.0
HPF_GAIN = 3.0
HPF_STRONG_EXTRA_GAIN = 1.5

HPF_BILATERAL_DIAMETER = 7
HPF_BILATERAL_SIGMA_COLOR = 12.0
HPF_BILATERAL_SIGMA_SPACE = 2.0

HPF_GATE_LOW = 1.5
HPF_GATE_HIGH = 4.0
HPF_GATE_SMOOTH_SIGMA = 0.8


# ============================================================
# 5. Basic utilities / frozen master split discovery
# ============================================================

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def save_current_code_snapshot():
    source = Path(__file__).resolve()
    target = EXPERIMENT_ROOT / source.name

    if source != target.resolve():
        shutil.copy2(source, target)

    print("Code snapshot:", target)
    return target


def get_device(requested="auto"):
    requested = str(requested).lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but CUDA is unavailable.")
        device = torch.device("cuda")
    elif requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested Apple MPS, but MPS is unavailable.")
        device = torch.device("mps")
    elif requested == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} (FP32; no AMP)")
    return device

def to_uint8(image):
    return np.clip(
        np.round(image),
        0,
        255,
    ).astype(np.uint8)


def natural_key(path):
    parts = re.split(r"(\d+)", path.name)
    return [
        int(p) if p.isdigit() else p.lower()
        for p in parts
    ]


def list_tifs(folder):
    paths = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        paths.extend(folder.glob(pattern))
    return sorted(paths, key=natural_key)


def read_uint8_tif(path):
    image = tifffile.imread(
        str(path)
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D TIFF, got {image.shape}: {path}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 TIFF, got {image.dtype}: {path}"
        )

    return image


def discover_split_blocks(folder, split):
    if not folder.exists():
        raise FileNotFoundError(folder)

    block_dirs = sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_dir()
            and p.name.startswith("block_")
        ],
        key=natural_key,
    )

    expected = EXPECTED_BLOCK_COUNTS[split]
    if len(block_dirs) != expected:
        raise RuntimeError(
            f"{split}: expected {expected} block folders, "
            f"found {len(block_dirs)} under {folder}"
        )

    records = []
    for block_dir in block_dirs:
        slice_paths = list_tifs(block_dir)

        if len(slice_paths) != EXPECTED_SLICES_PER_BLOCK:
            raise RuntimeError(
                f"{split}/{block_dir.name}: expected "
                f"{EXPECTED_SLICES_PER_BLOCK} TIFFs, "
                f"found {len(slice_paths)}"
            )

        for path in slice_paths:
            image = read_uint8_tif(path)
            if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
                raise RuntimeError(
                    f"Expected {(EXPECTED_HEIGHT, EXPECTED_WIDTH)}, "
                    f"got {image.shape}: {path}"
                )

        records.append({
            "split": split,
            "name": block_dir.name,
            "dir": block_dir,
            "slice_paths": slice_paths,
        })

    return records


def discover_master_split():
    records = {
        "train": discover_split_blocks(TRAIN_DIR, "train"),
        "val": discover_split_blocks(VAL_DIR, "val"),
        "test": discover_split_blocks(TEST_DIR, "test"),
    }

    seen = set()
    for split in ("train", "val", "test"):
        for record in records[split]:
            key = record["name"]
            if key in seen:
                raise RuntimeError(
                    f"Block folder name occurs in multiple splits: {key}"
                )
            seen.add(key)

    return records


# ============================================================
# 6. Deterministic membrane HPF
# ============================================================

def apply_hpf(image):
    image_f = image.astype(np.float32)

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

    structure_detail = (
        structure_source
        - structure_blur
    )

    strength = np.abs(
        structure_detail
    )

    gate = np.clip(
        (
            strength
            - HPF_GATE_LOW
        )
        / (
            HPF_GATE_HIGH
            - HPF_GATE_LOW
        ),
        0.0,
        1.0,
    )

    # Smoothstep.
    gate = (
        gate
        * gate
        * (3.0 - 2.0 * gate)
    )

    gate = cv2.GaussianBlur(
        gate.astype(np.float32),
        (0, 0),
        sigmaX=HPF_GATE_SMOOTH_SIGMA,
        sigmaY=HPF_GATE_SMOOTH_SIGMA,
        borderType=cv2.BORDER_REFLECT,
    )

    enhancement_gain = (
        HPF_GAIN * gate
        + HPF_STRONG_EXTRA_GAIN
        * gate
        * gate
    )

    return to_uint8(
        image_f
        + structure_detail
        * enhancement_gain
    )


def block_cache_dir(record):
    candidate = CACHE_ROOT / record["split"] / record["name"]
    resolved = candidate.resolve()
    if resolved == CACHE_ROOT or CACHE_ROOT not in resolved.parents:
        raise ValueError(
            f"Unsafe cache path outside dedicated cache root: {candidate}"
        )
    return candidate


def source_fingerprint(record):
    if "_source_fingerprint" in record:
        return record["_source_fingerprint"]

    files = []
    for path in record["slice_paths"]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        files.append({
            "name": path.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
        })

    record["_source_fingerprint"] = files
    return files

def cache_manifest_expected(record):
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "split": record["split"],
        "block_name": record["name"],
        "shape": [
            EXPECTED_SLICES_PER_BLOCK,
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH,
        ],
        "source_files": source_fingerprint(record),
        "preprocess_gaussian_sigma": PREPROCESS_GAUSSIAN_SIGMA,
        "preprocess_hist_bins": PREPROCESS_HIST_BINS,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "hpf_sigma": HPF_SIGMA,
        "hpf_gain": HPF_GAIN,
        "hpf_strong_extra_gain": HPF_STRONG_EXTRA_GAIN,
        "hpf_bilateral_diameter": HPF_BILATERAL_DIAMETER,
        "hpf_bilateral_sigma_color": HPF_BILATERAL_SIGMA_COLOR,
        "hpf_bilateral_sigma_space": HPF_BILATERAL_SIGMA_SPACE,
        "hpf_gate_low": HPF_GATE_LOW,
        "hpf_gate_high": HPF_GATE_HIGH,
        "hpf_gate_smooth_sigma": HPF_GATE_SMOOTH_SIGMA,
        "placement_score_raw_dark_weight": MISSING_SCORE_RAW_DARK_WEIGHT,
        "placement_score_processed_neg_weight": MISSING_SCORE_PROCESSED_NEG_WEIGHT,
        "placement_score_gradient_weight": MISSING_SCORE_GRADIENT_WEIGHT,
        "placement_score_smooth_sigma": MISSING_SCORE_SMOOTH_SIGMA,
    }


def cache_paths(record):
    root = block_cache_dir(record)
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "raw": root / "raw.npy",
        "processed": root / "processed.npy",
        "target": root / "target.npy",
        "score": root / "membrane_score.npy",
    }


def cache_array_valid(path, shape, dtype):
    if not path.exists():
        return False

    try:
        array = np.load(path, mmap_mode="r")
        return (
            tuple(array.shape) == tuple(shape)
            and array.dtype == np.dtype(dtype)
        )
    except Exception:
        return False


def block_cache_valid(record):
    paths = cache_paths(record)
    if not paths["manifest"].exists():
        return False

    try:
        manifest = json.loads(
            paths["manifest"].read_text(encoding="utf-8")
        )
    except Exception:
        return False

    if manifest != cache_manifest_expected(record):
        return False

    shape = (
        EXPECTED_SLICES_PER_BLOCK,
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
    )

    return (
        cache_array_valid(paths["raw"], shape, np.uint8)
        and cache_array_valid(paths["processed"], shape, np.float32)
        and cache_array_valid(paths["target"], shape, np.float32)
        and cache_array_valid(paths["score"], shape, np.float32)
    )


def build_block_cache(record):
    if block_cache_valid(record):
        return {
            "name": record["name"],
            "split": record["split"],
            "status": "hit",
        }

    paths = cache_paths(record)
    ensure_dir(paths["root"])

    # Write into temporary files and atomically replace only after the entire
    # block succeeds, so interrupted first runs cannot create a false valid cache.
    tmp_paths = {
        key: value.with_name(value.name + ".tmp")
        for key, value in paths.items()
        if key in {"raw", "processed", "target", "score"}
    }

    for path in tmp_paths.values():
        if path.exists():
            path.unlink()

    shape = (
        EXPECTED_SLICES_PER_BLOCK,
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
    )

    raw_mm = np.lib.format.open_memmap(
        tmp_paths["raw"],
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )
    processed_mm = np.lib.format.open_memmap(
        tmp_paths["processed"],
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    target_mm = np.lib.format.open_memmap(
        tmp_paths["target"],
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    score_mm = np.lib.format.open_memmap(
        tmp_paths["score"],
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    try:
        for zi, source_path in enumerate(record["slice_paths"]):
            raw = read_uint8_tif(source_path)
            processed, basis = preprocess_slice(raw)
            hpf_target = apply_hpf(raw)
            target = target_to_processed_space(
                hpf_target,
                basis,
            )
            score = build_membrane_score(
                raw,
                processed,
            )

            raw_mm[zi] = raw
            processed_mm[zi] = processed
            target_mm[zi] = target
            score_mm[zi] = score

        raw_mm.flush()
        processed_mm.flush()
        target_mm.flush()
        score_mm.flush()
    finally:
        del raw_mm
        del processed_mm
        del target_mm
        del score_mm

    tmp_paths["raw"].replace(paths["raw"])
    tmp_paths["processed"].replace(paths["processed"])
    tmp_paths["target"].replace(paths["target"])
    tmp_paths["score"].replace(paths["score"])

    paths["manifest"].write_text(
        json.dumps(
            cache_manifest_expected(record),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "name": record["name"],
        "split": record["split"],
        "status": "built",
    }


def prepare_stage2_cache(train_records, val_records):
    records = list(train_records) + list(val_records)

    print(
        "\nPreparing deterministic Stage-2 cache "
        f"({len(records)} train+val blocks)..."
    )
    print("Cache root:", CACHE_ROOT)
    print("Cache workers:", CACHE_BUILD_WORKERS)

    hits = 0
    builds = 0

    # Block-level parallelism only. Cached values themselves are deterministic
    # and use no RNG, so this cannot alter the Stage-2 training distribution.
    with ThreadPoolExecutor(
        max_workers=max(1, CACHE_BUILD_WORKERS)
    ) as executor:
        futures = [
            executor.submit(
                build_block_cache,
                record,
            )
            for record in records
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Stage-2 cache",
            dynamic_ncols=True,
        ):
            result = future.result()
            if result["status"] == "hit":
                hits += 1
            else:
                builds += 1

    print(
        f"Stage-2 cache ready: hits={hits}, built={builds}"
    )


# ============================================================
# 7. full-slice preprocessing
# ============================================================

def preprocess_slice(raw):
    """
    Exact full-slice preprocessing on one COMPLETE slice at its actual native YX size.

    Returns:
        processed : clipped high-pass representation
        basis     : values needed to express/reconstruct the center target
    """
    raw_f = raw.astype(
        np.float32
    )

    mean = float(
        raw_f.mean()
    )

    std = float(
        raw_f.std()
    )

    if std < 1e-8:
        raise ValueError(
            "Slice standard deviation is zero."
        )

    normalized = (
        raw_f - mean
    ) / std

    lowpass = ndimage.gaussian_filter(
        normalized,
        sigma=PREPROCESS_GAUSSIAN_SIGMA,
    ).astype(np.float32)

    highpass = (
        normalized
        - lowpass
    )

    hist, edges = np.histogram(
        highpass.ravel(),
        bins=PREPROCESS_HIST_BINS,
    )

    i = int(
        np.argmax(hist)
    )

    mode = float(
        (
            edges[i]
            + edges[i + 1]
        )
        / 2.0
    )

    processed = np.clip(
        highpass - mode,
        CLIP_MIN,
        CLIP_MAX,
    ).astype(np.float32)

    basis = {
        "mean": mean,
        "std": std,
        "lowpass": lowpass,
        "mode": mode,
    }

    return processed, basis


def target_to_processed_space(
    hpf_target,
    center_basis,
):
    """
    Express the HPF-enhanced raw center target in the ORIGINAL center slice's
    full-slice reconstruction basis.

    Therefore:
        inverse(prediction, center_basis)
    is trained to reconstruct the HPF-enhanced center image.
    """
    target_f = hpf_target.astype(
        np.float32
    )

    target_normalized = (
        target_f
        - center_basis["mean"]
    ) / center_basis["std"]

    target = (
        target_normalized
        - center_basis["lowpass"]
        - center_basis["mode"]
    )

    return np.clip(
        target,
        CLIP_MIN,
        CLIP_MAX,
    ).astype(np.float32)




# ============================================================
# 8. Block access
# ============================================================

def open_block_pairs(
    records,
):
    pairs = []

    for record in records:
        if not block_cache_valid(record):
            raise RuntimeError(
                f"Stage-2 cache missing/invalid for {record['split']}/{record['name']}"
            )

        paths = cache_paths(record)

        raw = np.load(
            paths["raw"],
            mmap_mode="r",
        )
        processed = np.load(
            paths["processed"],
            mmap_mode="r",
        )
        target = np.load(
            paths["target"],
            mmap_mode="r",
        )
        score = np.load(
            paths["score"],
            mmap_mode="r",
        )

        shape = (
            EXPECTED_SLICES_PER_BLOCK,
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH,
        )

        for name, array, dtype in (
            ("raw", raw, np.uint8),
            ("processed", processed, np.float32),
            ("target", target, np.float32),
            ("score", score, np.float32),
        ):
            if tuple(array.shape) != shape:
                raise RuntimeError(
                    f"Cached {name} shape mismatch for {record['name']}: "
                    f"{array.shape} vs {shape}"
                )
            if array.dtype != np.dtype(dtype):
                raise RuntimeError(
                    f"Cached {name} dtype mismatch for {record['name']}: "
                    f"{array.dtype} vs {np.dtype(dtype)}"
                )

        pairs.append({
            "name": record["name"],
            "split": record["split"],
            "raw": raw,
            "processed": processed,
            "target": target,
            "score": score,
        })

    return pairs


# ============================================================
# 9. New Stage-2 synthesis / dynamic patch sampling
# ============================================================

def prepare_clean_context(
    pair,
    center_z,
):
    """
    Prepare one exact clean 5-slice context from deterministic mmap caches.

    Scientific equivalence to accepted Stage-2 baseline:
    - raw_stack is the original uint8 master data;
    - processed_full is exact full-slice full-slice preprocessing;
    - target_full is the exact deterministic HPF target represented in the
      original center slice's full-slice basis;
    - placement_full is built from the exact cached per-slice membrane score.

    No stochastic corruption is cached here.
    """
    z0 = center_z - CONTEXT_RADIUS
    z1 = center_z + CONTEXT_RADIUS + 1

    clean_raw_stack = np.asarray(
        pair["raw"][z0:z1],
        dtype=np.uint8,
    ).copy()

    clean_processed_full = np.asarray(
        pair["processed"][z0:z1],
        dtype=np.float32,
    ).copy()

    target_full = np.asarray(
        pair["target"][center_z],
        dtype=np.float32,
    ).copy()

    score_stack = np.asarray(
        pair["score"][z0:z1],
        dtype=np.float32,
    ).copy()

    center_score = score_stack[
        CONTEXT_RADIUS
    ]

    neighbor_score = np.mean(
        np.delete(
            score_stack,
            CONTEXT_RADIUS,
            axis=0,
        ),
        axis=0,
    )

    placement_full = (
        0.72
        * center_score
        + 0.28
        * neighbor_score
    ).astype(np.float32)

    return {
        "raw_stack": clean_raw_stack,
        "processed_full": clean_processed_full,
        "target_full": target_full,
        "score_stack": score_stack,
        "placement_full": placement_full,
    }

def crop_stack(
    stack,
    y,
    x,
):
    return stack[
        :,
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE,
    ].copy()


def random_crop_start(
    h,
    w,
    rng,
    margin=0,
):
    if h < PATCH_SIZE or w < PATCH_SIZE:
        raise ValueError(
            f"Slice {(h, w)} smaller than patch {PATCH_SIZE}."
        )

    margin = int(max(0, margin))

    y_min = margin
    x_min = margin
    y_max = h - PATCH_SIZE - margin
    x_max = w - PATCH_SIZE - margin

    # Fall back to the full valid crop range if the requested margin is
    # impossible for a smaller image.
    if y_max < y_min:
        y_min = 0
        y_max = h - PATCH_SIZE

    if x_max < x_min:
        x_min = 0
        x_max = w - PATCH_SIZE

    y = int(
        rng.integers(
            y_min,
            y_max + 1,
        )
    )

    x = int(
        rng.integers(
            x_min,
            x_max + 1,
        )
    )

    return y, x


# ------------------------------------------------------------
# Missing-region placement / appearance
# ------------------------------------------------------------

def robust01(image):
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    lo, hi = np.percentile(
        image,
        [5.0, 95.0],
    )

    if hi - lo < 1e-8:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    return np.clip(
        (image - lo) / (hi - lo),
        0.0,
        1.0,
    ).astype(np.float32)


def build_membrane_score(
    raw_uint8,
    processed,
):
    raw_dark = (
        1.0
        - robust01(
            raw_uint8.astype(
                np.float32
            )
        )
    )

    negative = np.maximum(
        -processed.astype(
            np.float32
        ),
        0.0,
    )

    negative = robust01(
        negative
    )

    gx = cv2.Sobel(
        processed.astype(
            np.float32
        ),
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        processed.astype(
            np.float32
        ),
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = robust01(
        np.sqrt(
            gx * gx
            + gy * gy
        )
    )

    score = (
        MISSING_SCORE_RAW_DARK_WEIGHT
        * raw_dark
        * (
            0.30
            + 0.70
            * gradient
        )
        + MISSING_SCORE_PROCESSED_NEG_WEIGHT
        * negative
        * (
            0.40
            + 0.60
            * gradient
        )
        + MISSING_SCORE_GRADIENT_WEIGHT
        * gradient
    )

    score = ndimage.gaussian_filter(
        score.astype(
            np.float32
        ),
        sigma=MISSING_SCORE_SMOOTH_SIGMA,
    )

    return robust01(
        score
    )




def integral_crop_mean(
    integral,
    y,
    x,
):
    y2 = y + PATCH_SIZE
    x2 = x + PATCH_SIZE

    total = (
        integral[y2, x2]
        - integral[y, x2]
        - integral[y2, x]
        + integral[y, x]
    )

    return float(
        total
        / (
            PATCH_SIZE
            * PATCH_SIZE
        )
    )


def choose_membrane_rich_crop(
    placement_score,
    rng,
):
    h, w = placement_score.shape

    integral = cv2.integral(
        placement_score.astype(
            np.float32
        ),
        sdepth=cv2.CV_64F,
    )

    candidates = []

    for _ in range(
        MISSING_CROP_CANDIDATES
    ):
        y, x = random_crop_start(
            h,
            w,
            rng,
            margin=0,
        )

        candidates.append(
            (
                integral_crop_mean(
                    integral,
                    y,
                    x,
                ),
                y,
                x,
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top = candidates[
        :max(
            1,
            min(
                MISSING_CROP_TOP_K,
                len(candidates),
            ),
        )
    ]

    weights = np.asarray(
        [
            max(
                item[0],
                1e-8,
            )
            for item in top
        ],
        dtype=np.float64,
    )

    weights /= weights.sum()

    index = int(
        rng.choice(
            len(top),
            p=weights,
        )
    )

    return (
        int(top[index][1]),
        int(top[index][2]),
    )


def choose_structure_center(
    score_patch,
    rng,
):
    h, w = score_patch.shape

    valid = np.zeros(
        (h, w),
        dtype=bool,
    )

    margin = int(
        MISSING_PLACEMENT_VALID_MARGIN
    )

    valid[
        margin:h - margin,
        margin:w - margin,
    ] = True

    values = score_patch[
        valid
    ]

    if values.size == 0:
        return (
            float(w / 2.0),
            float(h / 2.0),
        )

    threshold = float(
        np.percentile(
            values,
            MISSING_PLACEMENT_TOP_PERCENTILE,
        )
    )

    candidates = (
        valid
        & (
            score_patch
            >= threshold
        )
    )

    ys, xs = np.where(
        candidates
    )

    if len(xs) == 0:
        return (
            float(w / 2.0),
            float(h / 2.0),
        )

    weights = np.maximum(
        score_patch[
            ys,
            xs,
        ].astype(
            np.float64
        ),
        1e-8,
    ) ** 2.0

    weights /= weights.sum()

    index = int(
        rng.choice(
            len(xs),
            p=weights,
        )
    )

    return (
        float(xs[index]),
        float(ys[index]),
    )


def choose_missing_pattern(
    rng,
):
    r = float(
        rng.random()
    )

    a = MISSING_CENTER_ONLY_PROB
    b = a + MISSING_CENTER_DOMINANT_PROB
    c = b + MISSING_NEIGHBOR_ONLY_PROB

    if r < a:
        return "center_only"

    if r < b:
        return "center_dominant"

    if r < c:
        return "neighbor_only"

    return "irregular_multi"


def build_missing_severity_plan(
    pattern,
    rng,
):
    severity = np.zeros(
        NUM_CONTEXT_SLICES,
        dtype=np.float32,
    )

    if pattern == "center_only":
        severity[2] = float(
            rng.uniform(
                1.00,
                1.35,
            )
        )

    elif pattern == "center_dominant":
        severity[2] = float(
            rng.uniform(
                1.00,
                1.35,
            )
        )

        side = (
            -1
            if rng.random() < 0.5
            else 1
        )

        severity[
            2 + side
        ] = float(
            rng.uniform(
                0.30,
                0.70,
            )
        )

        if rng.random() < 0.30:
            far = 2 + 2 * side

            if 0 <= far < 5:
                severity[
                    far
                ] = float(
                    rng.uniform(
                        0.15,
                        0.45,
                    )
                )

    elif pattern == "neighbor_only":
        choices = np.asarray(
            [0, 1, 3, 4],
            dtype=np.int64,
        )

        count = int(
            rng.integers(
                1,
                3,
            )
        )

        picked = rng.choice(
            choices,
            size=count,
            replace=False,
        )

        for zi in picked:
            severity[
                int(zi)
            ] = float(
                rng.uniform(
                    0.55,
                    1.10,
                )
            )

    elif pattern == "irregular_multi":
        count = int(
            rng.integers(
                2,
                4,
            )
        )

        picked = rng.choice(
            np.arange(
                NUM_CONTEXT_SLICES
            ),
            size=count,
            replace=False,
        )

        for zi in picked:
            zi = int(zi)

            if zi == CONTEXT_RADIUS:
                severity[zi] = float(
                    rng.uniform(
                        0.80,
                        1.30,
                    )
                )
            else:
                severity[zi] = float(
                    rng.uniform(
                        0.25,
                        1.00,
                    )
                )

    else:
        raise ValueError(
            f"Unknown missing pattern: {pattern}"
        )

    return severity


def make_irregular_soft_mask(
    h,
    w,
    rng,
    center,
    area,
):
    cx, cy = center

    base_radius = math.sqrt(
        max(
            area,
            100.0,
        )
        / math.pi
    )

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        256,
        endpoint=False,
        dtype=np.float32,
    )

    radial = np.ones_like(
        angles,
        dtype=np.float32,
    )

    wave_count = int(
        rng.integers(
            2,
            5,
        )
    )

    for frequency in range(
        1,
        wave_count + 1,
    ):
        amplitude = float(
            rng.uniform(
                0.04,
                0.14,
            )
        ) / float(
            frequency
        )

        phase = float(
            rng.uniform(
                0.0,
                2.0 * math.pi,
            )
        )

        radial += (
            amplitude
            * np.sin(
                frequency
                * angles
                + phase
            )
        )

    radial *= base_radius

    aspect_x = float(
        rng.uniform(
            0.75,
            1.45,
        )
    )

    aspect_y = float(
        rng.uniform(
            0.75,
            1.45,
        )
    )

    rotation = float(
        rng.uniform(
            0.0,
            2.0 * math.pi,
        )
    )

    local_x = (
        radial
        * np.cos(
            angles
        )
        * aspect_x
    )

    local_y = (
        radial
        * np.sin(
            angles
        )
        * aspect_y
    )

    cos_r = math.cos(
        rotation
    )

    sin_r = math.sin(
        rotation
    )

    x = (
        cx
        + local_x
        * cos_r
        - local_y
        * sin_r
    )

    y = (
        cy
        + local_x
        * sin_r
        + local_y
        * cos_r
    )

    polygon = np.stack(
        [x, y],
        axis=1,
    )

    binary = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        binary,
        [
            np.round(
                polygon
            ).astype(
                np.int32
            )
        ],
        255,
    )

    binary = ndimage.binary_fill_holes(
        binary > 0
    ).astype(np.uint8)

    soft = ndimage.gaussian_filter(
        binary.astype(
            np.float32
        ),
        sigma=MISSING_SOFT_EDGE_SIGMA,
    )

    if soft.max() > 1e-8:
        soft /= float(
            soft.max()
        )

    return np.clip(
        soft,
        0.0,
        1.0,
    ).astype(np.float32)


def missing_ring(
    soft_mask,
):
    binary = (
        soft_mask > 0.5
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    inner = cv2.dilate(
        binary,
        kernel,
        iterations=MISSING_RING_INNER_DILATE,
    )

    outer = cv2.dilate(
        binary,
        kernel,
        iterations=MISSING_RING_OUTER_DILATE,
    )

    return (
        (outer > 0)
        & (inner == 0)
    )


def make_missing_gray_fill(
    processed_patch,
    soft_mask,
    rng,
):
    ring = missing_ring(
        soft_mask
    )

    if ring.sum() >= 30:
        local_base = float(
            np.median(
                processed_patch[
                    ring
                ]
            )
        )
    else:
        local_base = float(
            np.median(
                processed_patch
            )
        )

    if (
        rng.random()
        < MISSING_GRAY_OFFSET_POSITIVE_PROB
    ):
        offset = float(
            rng.uniform(
                *MISSING_GRAY_OFFSET_POSITIVE
            )
        )
    else:
        offset = float(
            rng.uniform(
                *MISSING_GRAY_OFFSET_NEGATIVE
            )
        )

    h, w = processed_patch.shape

    fill = np.full(
        (h, w),
        local_base + offset,
        dtype=np.float32,
    )

    texture_sigma = float(
        rng.uniform(
            *MISSING_TEXTURE_SIGMA
        )
    )

    texture_amp = float(
        rng.uniform(
            *MISSING_TEXTURE_AMPLITUDE
        )
    )

    texture = rng.normal(
        0.0,
        1.0,
        size=(h, w),
    ).astype(np.float32)

    texture = ndimage.gaussian_filter(
        texture,
        sigma=texture_sigma,
    )

    texture -= float(
        texture.mean()
    )

    texture_std = float(
        texture.std()
    )

    if texture_std > 1e-8:
        texture /= texture_std

    fill += (
        texture_amp
        * texture
    )

    gradient_amp = float(
        rng.uniform(
            *MISSING_GRADIENT_AMPLITUDE
        )
    )

    if gradient_amp > 0.0:
        yy, xx = np.mgrid[
            0:h,
            0:w,
        ].astype(np.float32)

        xx = (
            xx
            - (w - 1) / 2.0
        ) / max(
            float(w),
            1.0,
        )

        yy = (
            yy
            - (h - 1) / 2.0
        ) / max(
            float(h),
            1.0,
        )

        direction = float(
            rng.uniform(
                0.0,
                2.0 * math.pi,
            )
        )

        fill += (
            gradient_amp
            * (
                math.cos(
                    direction
                )
                * xx
                + math.sin(
                    direction
                )
                * yy
            )
        )

    return np.clip(
        fill,
        CLIP_MIN,
        CLIP_MAX,
    ).astype(np.float32)


def build_missing_corruption(
    clean_patch,
    preferred_center,
    rng,
):
    """
    Build the missing-region replacement and soft alpha.

    The actual blending into the training input happens AFTER Gaussian noise,
    preserving the original baseline ordering for this corruption family.
    """
    h, w = clean_patch.shape[-2:]

    pattern = choose_missing_pattern(
        rng
    )

    severity = build_missing_severity_plan(
        pattern,
        rng,
    )

    base_center = (
        float(
            preferred_center[0]
        ),
        float(
            preferred_center[1]
        ),
    )

    replacement = np.zeros_like(
        clean_patch,
        dtype=np.float32,
    )

    alpha = np.zeros_like(
        clean_patch,
        dtype=np.float32,
    )

    binary_masks = np.zeros_like(
        clean_patch,
        dtype=np.float32,
    )

    metadata = {
        "pattern": pattern,
        "severity": [
            float(v)
            for v in severity
        ],
        "affected_indices": [],
    }

    for zi in range(
        NUM_CONTEXT_SLICES
    ):
        scale = float(
            severity[zi]
        )

        if scale <= 0.0:
            continue

        jitter_x = float(
            rng.uniform(
                -MISSING_POSITION_JITTER_PX,
                MISSING_POSITION_JITTER_PX,
            )
        )

        jitter_y = float(
            rng.uniform(
                -MISSING_POSITION_JITTER_PX,
                MISSING_POSITION_JITTER_PX,
            )
        )

        center = (
            np.clip(
                base_center[0] + jitter_x,
                MISSING_PLACEMENT_VALID_MARGIN,
                w - MISSING_PLACEMENT_VALID_MARGIN - 1,
            ),
            np.clip(
                base_center[1] + jitter_y,
                MISSING_PLACEMENT_VALID_MARGIN,
                h - MISSING_PLACEMENT_VALID_MARGIN - 1,
            ),
        )

        base_area = float(
            rng.uniform(
                *MISSING_BASE_AREA_RANGE
            )
        )

        area = (
            base_area
            * scale
        )

        soft_mask = make_irregular_soft_mask(
            h=h,
            w=w,
            rng=rng,
            center=center,
            area=area,
        )

        fill = make_missing_gray_fill(
            clean_patch[zi],
            soft_mask,
            rng,
        )

        alpha[zi] = soft_mask
        replacement[zi] = fill
        binary_masks[zi] = (
            soft_mask > 0.5
        ).astype(np.float32)

        metadata[
            "affected_indices"
        ].append(
            int(zi)
        )

    return (
        replacement,
        alpha,
        binary_masks,
        metadata,
    )


# ------------------------------------------------------------
# Whole-patch crop-offset local drift
# ------------------------------------------------------------

def compute_shifted_crop_start(
    base_y,
    base_x,
    dy,
    dx,
    full_h,
    full_w,
):
    y = int(
        round(
            base_y
            + dy
        )
    )

    x = int(
        round(
            base_x
            + dx
        )
    )

    y = int(
        np.clip(
            y,
            0,
            full_h - PATCH_SIZE,
        )
    )

    x = int(
        np.clip(
            x,
            0,
            full_w - PATCH_SIZE,
        )
    )

    return y, x


def apply_local_drift_crop(
    clean_processed_full,
    crop_y,
    crop_x,
    rng,
):
    """
    Center patch stays at (crop_y, crop_x).

    Each affected neighbor is cropped from its own shifted location:
        (crop_y + dy, crop_x + dx)

    Therefore the entire affected 256x256 patch shifts together.
    """
    full_h, full_w = clean_processed_full.shape[-2:]

    output = crop_stack(
        clean_processed_full,
        crop_y,
        crop_x,
    )

    neighbors = np.asarray(
        [0, 1, 3, 4],
        dtype=np.int64,
    )

    affected_count = int(
        rng.integers(
            LOCAL_DRIFT_AFFECTED_NEIGHBOR_COUNT[0],
            LOCAL_DRIFT_AFFECTED_NEIGHBOR_COUNT[1] + 1,
        )
    )

    affected = [
        int(v)
        for v in rng.choice(
            neighbors,
            size=affected_count,
            replace=False,
        )
    ]

    vectors = []

    for zi in affected:
        magnitude = float(
            rng.uniform(
                *LOCAL_DRIFT_MAG_RANGE_PX
            )
        )

        angle = float(
            rng.uniform(
                0.0,
                2.0 * math.pi,
            )
        )

        dx = (
            magnitude
            * math.cos(
                angle
            )
        )

        dy = (
            magnitude
            * math.sin(
                angle
            )
        )

        drift_y, drift_x = compute_shifted_crop_start(
            base_y=crop_y,
            base_x=crop_x,
            dy=dy,
            dx=dx,
            full_h=full_h,
            full_w=full_w,
        )

        output[zi] = clean_processed_full[
            zi,
            drift_y:drift_y + PATCH_SIZE,
            drift_x:drift_x + PATCH_SIZE,
        ]

        vectors.append(
            {
                "slice_index": int(zi),
                "dx": float(dx),
                "dy": float(dy),
                "magnitude": float(magnitude),
                "angle_deg": float(
                    math.degrees(
                        angle
                    )
                    % 360.0
                ),
                "drifted_crop_yx": [
                    int(drift_y),
                    int(drift_x),
                ],
            }
        )

    return (
        output.astype(np.float32),
        {
            "affected_indices": affected,
            "vectors": vectors,
        },
    )


# ------------------------------------------------------------
# Whole-slice lose-focus
# ------------------------------------------------------------

def blur_raw_whole_slice(
    raw,
    sigma,
    alpha,
):
    raw_f = raw.astype(
        np.float32
    )

    blurred = cv2.GaussianBlur(
        raw_f,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    degraded = (
        (
            1.0
            - alpha
        )
        * raw_f
        + alpha
        * blurred
    )

    return to_uint8(
        degraded
    )


def apply_lose_focus_crop(
    clean_raw_stack,
    clean_processed_full,
    crop_y,
    crop_x,
    rng,
):
    """
    Exactly one COMPLETE raw slice is blurred, then that full degraded slice
    receives full-slice preprocessing before the 256 crop is extracted.
    """
    output = crop_stack(
        clean_processed_full,
        crop_y,
        crop_x,
    )

    slice_index = int(
        rng.choice(
            np.arange(
                NUM_CONTEXT_SLICES
            ),
            p=LOSE_FOCUS_INDEX_WEIGHTS,
        )
    )

    sigma = float(
        rng.uniform(
            *LOSE_FOCUS_SIGMA_RANGE
        )
    )

    alpha = float(
        rng.uniform(
            *LOSE_FOCUS_ALPHA_RANGE
        )
    )

    blurred_raw = blur_raw_whole_slice(
        clean_raw_stack[
            slice_index
        ],
        sigma=sigma,
        alpha=alpha,
    )

    blurred_processed, _ = preprocess_slice(
        blurred_raw
    )

    output[
        slice_index
    ] = blurred_processed[
        crop_y:crop_y + PATCH_SIZE,
        crop_x:crop_x + PATCH_SIZE,
    ]

    metadata = {
        "blurred_index": int(
            slice_index
        ),
        "sigma": float(
            sigma
        ),
        "alpha": float(
            alpha
        ),
    }

    return (
        output.astype(np.float32),
        metadata,
    )


# ------------------------------------------------------------
# Composite stripe synthesis - Stage-2
# ------------------------------------------------------------

def stripe_smooth_binary(
    binary,
    sigma,
):
    soft = ndimage.gaussian_filter(
        binary.astype(np.float32),
        sigma=sigma,
    )

    maxv = float(
        soft.max()
    )

    if maxv > 1e-8:
        soft /= maxv

    return np.clip(
        soft,
        0.0,
        1.0,
    ).astype(np.float32)


def stripe_rectangle_mask(
    h,
    w,
    cx,
    cy,
    half_length,
    half_width,
    angle_deg=0.0,
):
    yy, xx = np.mgrid[
        0:h,
        0:w,
    ].astype(np.float32)

    x = xx - float(cx)
    y = yy - float(cy)

    angle = math.radians(
        angle_deg
    )

    cos_a = math.cos(
        angle
    )

    sin_a = math.sin(
        angle
    )

    xr = (
        x * cos_a
        + y * sin_a
    )

    yr = (
        -x * sin_a
        + y * cos_a
    )

    mask = (
        (np.abs(xr) <= float(half_length))
        & (np.abs(yr) <= float(half_width))
    )

    return mask.astype(
        np.float32
    )


def choose_stripe_centers(
    rng,
    count,
):
    centers = []
    attempts = 0

    y_min = 24
    y_max = PATCH_SIZE - 25

    while (
        len(centers) < count
        and attempts < 500
    ):
        attempts += 1

        y = float(
            rng.uniform(
                y_min,
                y_max,
            )
        )

        if all(
            abs(y - old_y)
            >= MIN_STRIPE_VERTICAL_GAP
            for old_y in centers
        ):
            centers.append(
                y
            )

    if len(centers) < count:
        centers.extend(
            list(
                rng.uniform(
                    y_min,
                    y_max,
                    size=(
                        count
                        - len(centers)
                    ),
                )
            )
        )

    return sorted(
        centers
    )


def sample_stripe_objects(
    rng,
):
    stripe_count = int(
        rng.integers(
            STRIPE_COUNT_RANGE[0],
            STRIPE_COUNT_RANGE[1] + 1,
        )
    )

    center_ys = choose_stripe_centers(
        rng,
        stripe_count,
    )

    objects = []

    for base_y in center_ys:
        half_width = float(
            rng.uniform(
                *STRIPE_HALF_WIDTH_RANGE_PX
            )
        )

        dark_band_width = float(
            rng.uniform(
                *STRIPE_DARK_BAND_HALF_WIDTH_RANGE_PX
            )
        )

        half_length = (
            float(
                rng.uniform(
                    *STRIPE_HALF_LENGTH_FRAC_RANGE
                )
            )
            * PATCH_SIZE
        )

        bright_delta = float(
            rng.uniform(
                *STRIPE_BRIGHT_DELTA_RANGE
            )
        )

        dark_delta = float(
            rng.uniform(
                *STRIPE_DARK_DELTA_RANGE
            )
        )

        center_x = float(
            PATCH_SIZE / 2.0
            + rng.uniform(
                -12.0,
                12.0,
            )
        )

        # Keep the exact approved visual-test behavior:
        # base widths are stored as integer pixel widths before per-slice jitter.
        objects.append(
            {
                "base_center_y":
                    float(base_y),
                "base_center_x":
                    float(center_x),
                "base_half_width":
                    int(half_width),
                "base_dark_band_width":
                    int(dark_band_width),
                "base_half_length":
                    float(half_length),
                "base_bright_delta":
                    float(bright_delta),
                "base_dark_delta":
                    float(dark_delta),
            }
        )

    return objects


def apply_stripe_artifact(
    clean_patch,
    rng,
    inverted=False,
):
    """
    Apply one polarity of the approved Stage-2 stripe family.

    inverted=False:
        original white core + black side bands

    inverted=True:
        new black core + white side bands

    Geometry, widths, lengths, strengths, soft edges and across-Z jitter use
    the SAME accepted Stage-2 baseline distributions. Only the signs are reversed.
    """
    output = clean_patch.copy()

    stripe_objects = sample_stripe_objects(
        rng
    )

    metadata = {
        "inverted":
            bool(
                inverted
            ),
        "stripe_count":
            int(
                len(
                    stripe_objects
                )
            ),
        "stripes":
            [],
    }

    for stripe_id, stripe in enumerate(
        stripe_objects
    ):
        stripe_meta = {
            "stripe_id":
                int(
                    stripe_id
                ),
            "per_slice":
                [],
        }

        for zi in range(
            NUM_CONTEXT_SLICES
        ):
            center_y = float(
                stripe[
                    "base_center_y"
                ]
                + rng.uniform(
                    -STRIPE_CENTER_Y_JITTER_PX,
                    STRIPE_CENTER_Y_JITTER_PX,
                )
            )

            center_x = float(
                stripe[
                    "base_center_x"
                ]
                + rng.uniform(
                    -STRIPE_CENTER_X_JITTER_PX,
                    STRIPE_CENTER_X_JITTER_PX,
                )
            )

            half_width = float(
                np.clip(
                    stripe[
                        "base_half_width"
                    ]
                    + rng.uniform(
                        -STRIPE_HALF_WIDTH_JITTER_PX,
                        STRIPE_HALF_WIDTH_JITTER_PX,
                    ),
                    1.0,
                    16.0,
                )
            )

            dark_band_width = float(
                np.clip(
                    stripe[
                        "base_dark_band_width"
                    ]
                    + rng.uniform(
                        -STRIPE_DARK_BAND_HALF_WIDTH_JITTER_PX,
                        STRIPE_DARK_BAND_HALF_WIDTH_JITTER_PX,
                    ),
                    1.0,
                    10.0,
                )
            )

            half_length = float(
                np.clip(
                    stripe[
                        "base_half_length"
                    ]
                    + rng.integers(
                        -STRIPE_HALF_LENGTH_JITTER_PX,
                        STRIPE_HALF_LENGTH_JITTER_PX + 1,
                    ),
                    20.0,
                    PATCH_SIZE / 2.0 - 4.0,
                )
            )

            strength_scale = float(
                rng.uniform(
                    *STRIPE_STRENGTH_SCALE_RANGE
                )
            )

            bright_delta = (
                stripe[
                    "base_bright_delta"
                ]
                * strength_scale
            )

            dark_delta = (
                stripe[
                    "base_dark_delta"
                ]
                * strength_scale
            )

            core_binary = stripe_rectangle_mask(
                PATCH_SIZE,
                PATCH_SIZE,
                center_x,
                center_y,
                half_length=half_length,
                half_width=half_width,
                angle_deg=STRIPE_ANGLE_DEG,
            )

            outer_binary = stripe_rectangle_mask(
                PATCH_SIZE,
                PATCH_SIZE,
                center_x,
                center_y,
                half_length=half_length,
                half_width=(
                    half_width
                    + dark_band_width
                ),
                angle_deg=STRIPE_ANGLE_DEG,
            )

            dark_binary = np.clip(
                outer_binary
                - core_binary,
                0.0,
                1.0,
            ).astype(
                np.float32
            )

            core_soft = stripe_smooth_binary(
                core_binary,
                sigma=STRIPE_SOFT_EDGE_SIGMA,
            )

            dark_soft = stripe_smooth_binary(
                dark_binary,
                sigma=STRIPE_DARK_SOFT_EDGE_SIGMA,
            )

            if inverted:
                output[
                    zi
                ] = (
                    output[
                        zi
                    ]
                    - bright_delta
                    * core_soft
                    + dark_delta
                    * dark_soft
                ).astype(
                    np.float32
                )
            else:
                # Preserve the exact accepted Stage-2 baseline original-polarity operation
                # order so the white-core branch is numerically unchanged.
                output[
                    zi
                ] = (
                    output[
                        zi
                    ]
                    + bright_delta
                    * core_soft
                    - dark_delta
                    * dark_soft
                ).astype(
                    np.float32
                )

            stripe_meta[
                "per_slice"
            ].append(
                {
                    "slice_index":
                        int(zi),
                    "center_y":
                        float(center_y),
                    "center_x":
                        float(center_x),
                    "half_width":
                        float(half_width),
                    "dark_band_width":
                        float(
                            dark_band_width
                        ),
                    "half_length":
                        float(
                            half_length
                        ),
                    "bright_delta":
                        float(
                            bright_delta
                        ),
                    "dark_delta":
                        float(
                            dark_delta
                        ),
                }
            )

        metadata[
            "stripes"
        ].append(
            stripe_meta
        )

    return (
        np.clip(
            output,
            CLIP_MIN,
            CLIP_MAX,
        ).astype(np.float32),
        metadata,
    )




def smooth_random_sequence(
    rng,
    length,
    sigma,
):
    values = rng.normal(
        0.0,
        1.0,
        size=length,
    ).astype(np.float32)

    if length > 1:
        values = ndimage.gaussian_filter1d(
            values,
            sigma=max(
                float(sigma),
                0.25,
            ),
            mode="reflect",
        )

    scale = float(
        np.max(
            np.abs(values)
        )
    )

    if scale > 1e-8:
        values /= scale

    return values.astype(np.float32)


def build_residual_thin_bundle_field(
    rng,
):
    """
    Build one partial horizontal-line bundle directly on a 256x256 canvas.

    It deliberately keeps:
        - 1--2 px thin lines
        - horizontal geometry
        - correlated neighboring line extents

    It deliberately weakens:
        - line count
        - completeness
        - contrast

    relative to a full Stage-1 thin-bundle corruption.
    """
    field = np.zeros(
        (
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        dtype=np.float32,
    )

    line_count = int(
        rng.integers(
            THIN_BUNDLE_LINE_COUNT_RANGE[0],
            THIN_BUNDLE_LINE_COUNT_RANGE[1] + 1,
        )
    )

    base_length = int(
        rng.integers(
            THIN_BUNDLE_LENGTH_RANGE_PX[0],
            THIN_BUNDLE_LENGTH_RANGE_PX[1] + 1,
        )
    )

    spacing = int(
        rng.integers(
            THIN_BUNDLE_LINE_SPACING_RANGE_PX[0],
            THIN_BUNDLE_LINE_SPACING_RANGE_PX[1] + 1,
        )
    )

    total_height = max(
        8,
        (
            line_count - 1
        )
        * spacing
        + THIN_BUNDLE_LINE_THICKNESS_RANGE_PX[1]
        + 4,
    )

    margin_x = (
        base_length // 2
        + 12
    )

    margin_y = (
        total_height // 2
        + 8
    )

    if margin_x >= PATCH_SIZE // 2:
        margin_x = PATCH_SIZE // 2 - 4

    if margin_y >= PATCH_SIZE // 2:
        margin_y = PATCH_SIZE // 2 - 4

    center_x = int(
        rng.integers(
            margin_x,
            PATCH_SIZE - margin_x,
        )
    )

    center_y = int(
        rng.integers(
            margin_y,
            PATCH_SIZE - margin_y,
        )
    )

    start_drift = smooth_random_sequence(
        rng,
        line_count,
        sigma=max(
            1.0,
            line_count / 6.0,
        ),
    )

    end_drift = smooth_random_sequence(
        rng,
        line_count,
        sigma=max(
            1.0,
            line_count / 6.0,
        ),
    )

    center_drift = smooth_random_sequence(
        rng,
        line_count,
        sigma=max(
            1.0,
            line_count / 5.0,
        ),
    )

    keep_prob = float(
        rng.uniform(
            *THIN_BUNDLE_LINE_KEEP_PROB_RANGE
        )
    )

    line_strengths = rng.uniform(
        THIN_BUNDLE_LINE_STRENGTH_RANGE[0],
        THIN_BUNDLE_LINE_STRENGTH_RANGE[1],
        size=line_count,
    ).astype(np.float32)

    kept_lines = 0

    for i in range(
        line_count
    ):
        if float(
            rng.random()
        ) > keep_prob:
            continue

        kept_lines += 1

        relative_y = (
            i
            - 0.5
            * (
                line_count - 1
            )
        )

        y = int(
            round(
                center_y
                + relative_y
                * spacing
                + center_drift[i]
                * THIN_BUNDLE_CENTER_DRIFT_FRAC
                * total_height
            )
        )

        half_length = (
            0.5
            * base_length
        )

        x0 = int(
            round(
                center_x
                - half_length
                + start_drift[i]
                * THIN_BUNDLE_ENDPOINT_JITTER_FRAC
                * base_length
            )
        )

        x1 = int(
            round(
                center_x
                + half_length
                + end_drift[i]
                * THIN_BUNDLE_ENDPOINT_JITTER_FRAC
                * base_length
            )
        )

        x0 = int(
            np.clip(
                x0,
                2,
                PATCH_SIZE - 3,
            )
        )

        x1 = int(
            np.clip(
                x1,
                2,
                PATCH_SIZE - 3,
            )
        )

        if x1 <= x0 + 8:
            continue

        thickness = int(
            rng.integers(
                THIN_BUNDLE_LINE_THICKNESS_RANGE_PX[0],
                THIN_BUNDLE_LINE_THICKNESS_RANGE_PX[1] + 1,
            )
        )

        line_mask = np.zeros_like(
            field,
            dtype=np.uint8,
        )

        cv2.line(
            line_mask,
            (
                x0,
                y,
            ),
            (
                x1,
                y,
            ),
            255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        # Make the residual incomplete by removing a few local pieces.
        gap_count = int(
            rng.integers(
                THIN_BUNDLE_GAP_COUNT_RANGE[0],
                THIN_BUNDLE_GAP_COUNT_RANGE[1] + 1,
            )
        )

        for _ in range(
            gap_count
        ):
            gap_length = int(
                rng.integers(
                    THIN_BUNDLE_GAP_LENGTH_RANGE_PX[0],
                    THIN_BUNDLE_GAP_LENGTH_RANGE_PX[1] + 1,
                )
            )

            gap_center = int(
                rng.integers(
                    x0,
                    x1 + 1,
                )
            )

            gx0 = max(
                x0,
                gap_center
                - gap_length // 2,
            )

            gx1 = min(
                x1 + 1,
                gap_center
                + gap_length // 2
                + 1,
            )

            gy0 = max(
                0,
                y - thickness - 1,
            )

            gy1 = min(
                PATCH_SIZE,
                y + thickness + 2,
            )

            line_mask[
                gy0:gy1,
                gx0:gx1,
            ] = 0

        field = np.maximum(
            field,
            (
                line_mask.astype(
                    np.float32
                )
                / 255.0
            )
            * float(
                line_strengths[i]
            ),
        )

    if kept_lines == 0 or not np.any(
        field > 0
    ):
        raise RuntimeError(
            "Residual thin-bundle generator produced an empty field."
        )

    return (
        np.clip(
            field,
            0.0,
            1.0,
        ).astype(np.float32),
        {
            "line_count":
                int(line_count),
            "kept_lines":
                int(kept_lines),
            "base_length":
                int(base_length),
            "spacing":
                int(spacing),
            "keep_probability":
                float(keep_prob),
            "center_xy":
                [
                    int(center_x),
                    int(center_y),
                ],
        },
    )


def shift_field_integer(
    field,
    dx,
    dy,
):
    shifted = np.zeros_like(
        field,
        dtype=np.float32,
    )

    h, w = field.shape

    src_x0 = max(
        0,
        -dx,
    )
    src_x1 = min(
        w,
        w - dx,
    )

    src_y0 = max(
        0,
        -dy,
    )
    src_y1 = min(
        h,
        h - dy,
    )

    dst_x0 = max(
        0,
        dx,
    )
    dst_x1 = (
        dst_x0
        + (
            src_x1
            - src_x0
        )
    )

    dst_y0 = max(
        0,
        dy,
    )
    dst_y1 = (
        dst_y0
        + (
            src_y1
            - src_y0
        )
    )

    if (
        src_x1 <= src_x0
        or src_y1 <= src_y0
    ):
        return shifted

    shifted[
        dst_y0:dst_y1,
        dst_x0:dst_x1,
    ] = field[
        src_y0:src_y1,
        src_x0:src_x1,
    ]

    return shifted


def processed_signature_from_dark_field(
    field,
):
    """
    Mimic the local signature produced when a weak DARK thin-line residual
    passes through the sigma=10 full-slice high-pass.

    raw-like artifact:
        -field

    high-pass artifact:
        (-field) - Gaussian(-field)
        = -field + Gaussian(field)

    Result:
        dark thin core + faint broad positive surround
    """
    field = np.clip(
        field,
        0.0,
        1.0,
    ).astype(np.float32)

    broad = ndimage.gaussian_filter(
        field,
        sigma=THIN_BUNDLE_SIGNATURE_SIGMA,
    ).astype(np.float32)

    signature = (
        -field
        + broad
    ).astype(np.float32)

    negative = np.maximum(
        -signature,
        0.0,
    )

    support = (
        negative
        > 1e-5
    )

    if not np.any(
        support
    ):
        raise RuntimeError(
            "Thin-bundle processed signature has no negative support."
        )

    scale = float(
        np.percentile(
            negative[
                support
            ],
            98.0,
        )
    )

    if scale < 1e-8:
        raise RuntimeError(
            "Thin-bundle processed signature normalization failed."
        )

    return (
        signature
        / scale
    ).astype(np.float32)


def apply_residual_thin_bundle(
    clean_patch,
    rng,
    strength_multiplier=1.0,
):
    if clean_patch.shape != (
        NUM_CONTEXT_SLICES,
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise ValueError(
            f"Expected [5,256,256], got {clean_patch.shape}"
        )

    base_field, geometry_meta = (
        build_residual_thin_bundle_field(
            rng
        )
    )

    base_delta = float(
        rng.uniform(
            *THIN_BUNDLE_DELTA_RANGE
        )
    )

    output = clean_patch.copy()

    per_slice = []

    for zi in range(
        NUM_CONTEXT_SLICES
    ):
        dx = int(
            rng.integers(
                -THIN_BUNDLE_Z_X_JITTER_PX,
                THIN_BUNDLE_Z_X_JITTER_PX + 1,
            )
        )

        dy = int(
            rng.integers(
                -THIN_BUNDLE_Z_Y_JITTER_PX,
                THIN_BUNDLE_Z_Y_JITTER_PX + 1,
            )
        )

        z_strength = float(
            rng.uniform(
                *THIN_BUNDLE_Z_STRENGTH_RANGE
            )
        )

        shifted_field = shift_field_integer(
            base_field,
            dx=dx,
            dy=dy,
        )

        signature = (
            processed_signature_from_dark_field(
                shifted_field
            )
        )

        realized_delta = (
            base_delta
            * z_strength
            * float(
                strength_multiplier
            )
        )

        output[zi] = (
            output[zi]
            + realized_delta
            * signature
        ).astype(np.float32)

        per_slice.append(
            {
                "slice_index":
                    int(zi),
                "dx":
                    int(dx),
                "dy":
                    int(dy),
                "strength_scale":
                    float(z_strength),
                "target_dark_delta":
                    float(realized_delta),
            }
        )

    return (
        np.clip(
            output,
            CLIP_MIN,
            CLIP_MAX,
        ).astype(np.float32),
        {
            "base_delta":
                float(base_delta),
            "strength_multiplier":
                float(
                    strength_multiplier
                ),
            "geometry":
                geometry_meta,
            "per_slice":
                per_slice,
        },
    )

# ------------------------------------------------------------
# Stripe-family categorical sampling
# ------------------------------------------------------------

def sample_stripe_variant(
    rng,
):
    """
    Returns exactly one of:
        None / "original" / "inverted"

    Because this is one categorical return value, both stripe polarities
    can never appear together in the same 256x256 patch.
    """
    r = float(
        rng.random()
    )

    original_cut = (
        TRAIN_STRIPE_FAMILY_PROB
        * STRIPE_ORIGINAL_FRACTION
    )

    if r < original_cut:
        return "original"

    if r < TRAIN_STRIPE_FAMILY_PROB:
        return "inverted"

    return None



# ------------------------------------------------------------
# Patch type sampling
# ------------------------------------------------------------

def sample_patch_type(
    rng,
):
    r = float(
        rng.random()
    )

    a = TRAIN_CLEAN_PROB
    b = a + TRAIN_MISSING_PROB
    c = b + TRAIN_LOCAL_DRIFT_PROB

    if r < a:
        return "clean"

    if r < b:
        return "missing"

    if r < c:
        return "local_drift"

    return "lose_focus"




def sample_training_batch(
    train_pairs,
    rng,
    stripe_rng,
    thin_bundle_rng,
):
    """
    Build one Stage-2 batch.

    As in the accepted baseline, one full 5-slice context is prepared and multiple independent
    256x256 training patches are sampled from it. Each 256 patch independently
    chooses exactly ONE major structural condition:
        clean / missing / local_drift / lose_focus

    After the mutually exclusive major condition, artifact overlays are added:
        stripe family: one categorical choice, never both polarities in one patch
        residual thin bundle: independent Bernoulli overlay

    Gaussian noise is NOT added here. It is added in the optimization loop.
    Missing-region replacement is also blended after Gaussian noise, matching
    the baseline ordering for processed-space restoration corruption.
    """
    pair = train_pairs[
        int(
            rng.integers(
                0,
                len(train_pairs),
            )
        )
    ]

    z_count, h, w = pair[
        "raw"
    ].shape

    center_z = int(
        rng.integers(
            CONTEXT_RADIUS,
            z_count - CONTEXT_RADIUS,
        )
    )

    context = prepare_clean_context(
        pair,
        center_z,
    )

    clean_raw_stack = context[
        "raw_stack"
    ]

    clean_processed_full = context[
        "processed_full"
    ]

    target_full = context[
        "target_full"
    ]

    placement_full = context[
        "placement_full"
    ]

    input_patches = []
    target_patches = []
    missing_alphas = []
    missing_replacements = []
    missing_center_masks = []

    type_counts = {
        "clean": 0,
        "missing": 0,
        "local_drift": 0,
        "lose_focus": 0,
        "stripe": 0,
        "stripe_original": 0,
        "stripe_inverted": 0,
        "thin_bundle": 0,
    }

    drift_margin = int(
        math.ceil(
            LOCAL_DRIFT_MAG_RANGE_PX[1]
        )
    )

    for _ in range(
        BATCH_SIZE
    ):
        patch_type = sample_patch_type(
            rng
        )

        type_counts[
            patch_type
        ] += 1

        if patch_type == "missing":
            y, x = choose_membrane_rich_crop(
                placement_full,
                rng,
            )
        elif patch_type == "local_drift":
            y, x = random_crop_start(
                h,
                w,
                rng,
                margin=drift_margin,
            )
        else:
            y, x = random_crop_start(
                h,
                w,
                rng,
                margin=0,
            )

        clean_patch = crop_stack(
            clean_processed_full,
            y,
            x,
        )

        target_patch = target_full[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ].copy()

        structural_input = (
            clean_patch.copy()
        )

        missing_alpha = np.zeros_like(
            clean_patch,
            dtype=np.float32,
        )

        missing_replacement = np.zeros_like(
            clean_patch,
            dtype=np.float32,
        )

        missing_binary = np.zeros_like(
            clean_patch,
            dtype=np.float32,
        )

        metadata = {
            "type": patch_type,
        }

        if patch_type == "missing":
            placement_patch = placement_full[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ]

            preferred_center = choose_structure_center(
                placement_patch,
                rng,
            )

            (
                missing_replacement,
                missing_alpha,
                missing_binary,
                missing_meta,
            ) = build_missing_corruption(
                clean_patch,
                preferred_center,
                rng,
            )

            metadata.update(
                missing_meta
            )

            preview_corrupted = (
                clean_patch
                * (
                    1.0
                    - missing_alpha
                )
                + missing_replacement
                * missing_alpha
            )

        elif patch_type == "local_drift":
            (
                structural_input,
                drift_meta,
            ) = apply_local_drift_crop(
                clean_processed_full,
                y,
                x,
                rng,
            )

            metadata.update(
                drift_meta
            )

            preview_corrupted = (
                structural_input
            )

        elif patch_type == "lose_focus":
            (
                structural_input,
                focus_meta,
            ) = apply_lose_focus_crop(
                clean_raw_stack,
                clean_processed_full,
                y,
                x,
                rng,
            )

            metadata.update(
                focus_meta
            )

            preview_corrupted = (
                structural_input
            )

        else:
            preview_corrupted = (
                structural_input
            )

        # Stripe family: ONE categorical draw.
        # Original and inverted stripes can NEVER coexist in one 256 patch.
        stripe_variant = sample_stripe_variant(
            stripe_rng
        )

        if stripe_variant is not None:
            is_inverted = (
                stripe_variant
                == "inverted"
            )

            (
                structural_input,
                stripe_meta,
            ) = apply_stripe_artifact(
                structural_input,
                stripe_rng,
                inverted=is_inverted,
            )

            metadata[
                "stripe"
            ] = {
                "variant":
                    stripe_variant,
                "artifact":
                    stripe_meta,
            }

            type_counts[
                "stripe"
            ] += 1

            type_counts[
                (
                    "stripe_inverted"
                    if is_inverted
                    else "stripe_original"
                )
            ] += 1

        # Independent residual thin-bundle overlay.
        if (
            float(
                thin_bundle_rng.random()
            )
            < TRAIN_RESIDUAL_THIN_BUNDLE_PROB
        ):
            (
                structural_input,
                thin_meta,
            ) = apply_residual_thin_bundle(
                structural_input,
                thin_bundle_rng,
                strength_multiplier=1.0,
            )

            metadata[
                "residual_thin_bundle"
            ] = thin_meta

            type_counts[
                "thin_bundle"
            ] += 1


        input_patches.append(
            structural_input
        )

        target_patches.append(
            target_patch[
                None,
                :,
                :,
            ]
        )

        missing_alphas.append(
            missing_alpha
        )

        missing_replacements.append(
            missing_replacement
        )

        missing_center_masks.append(
            missing_binary[
                CONTEXT_RADIUS:
                CONTEXT_RADIUS + 1
            ]
        )

    return (
        torch.from_numpy(
            np.stack(
                input_patches,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                target_patches,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                missing_alphas,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                missing_replacements,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                missing_center_masks,
                axis=0,
            )
        ),
        type_counts,
    )


# ============================================================
# 10. Fixed validation
# ============================================================

def build_fixed_validation(
    val_pairs,
):
    rng = np.random.default_rng(
        SEED + 1000
    )

    stripe_rng = np.random.default_rng(
        SEED
        + STRIPE_VAL_RNG_SEED_OFFSET
    )

    inverted_stripe_rng = np.random.default_rng(
        SEED
        + STRIPE_VAL_RNG_SEED_OFFSET
    )

    thin_bundle_rng = np.random.default_rng(
        SEED
        + THIN_BUNDLE_VAL_RNG_SEED_OFFSET
    )

    clean_inputs = []
    missing_inputs = []
    drift_inputs = []
    focus_inputs = []
    stripe_inputs = []
    inverted_stripe_inputs = []
    thin_bundle_inputs = []
    targets = []
    missing_masks = []

    print(
        "\nBuilding fixed clean + missing + drift + lose-focus validation..."
    )

    drift_margin = int(
        math.ceil(
            LOCAL_DRIFT_MAG_RANGE_PX[1]
        )
    )

    for _ in tqdm(
        range(
            VAL_SAMPLES
        ),
        desc="Validation samples",
        dynamic_ncols=True,
    ):
        pair = val_pairs[
            int(
                rng.integers(
                    0,
                    len(val_pairs),
                )
            )
        ]

        z_count, h, w = pair[
            "raw"
        ].shape

        center_z = int(
            rng.integers(
                CONTEXT_RADIUS,
                z_count - CONTEXT_RADIUS,
            )
        )

        context = prepare_clean_context(
            pair,
            center_z,
        )

        raw_stack = context[
            "raw_stack"
        ]

        processed_full = context[
            "processed_full"
        ]

        target_full = context[
            "target_full"
        ]

        placement_full = context[
            "placement_full"
        ]

        y, x = random_crop_start(
            h,
            w,
            rng,
            margin=drift_margin,
        )

        clean_patch = crop_stack(
            processed_full,
            y,
            x,
        )

        target_patch = target_full[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ].copy()

        placement_patch = placement_full[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ]

        preferred_center = choose_structure_center(
            placement_patch,
            rng,
        )

        (
            replacement,
            alpha,
            binary_masks,
            _missing_meta,
        ) = build_missing_corruption(
            clean_patch,
            preferred_center,
            rng,
        )

        drift_patch, _ = apply_local_drift_crop(
            processed_full,
            y,
            x,
            rng,
        )

        focus_patch, _ = apply_lose_focus_crop(
            raw_stack,
            processed_full,
            y,
            x,
            rng,
        )

        stripe_patch, _ = apply_stripe_artifact(
            clean_patch,
            stripe_rng,
            inverted=False,
        )

        inverted_stripe_patch, _ = apply_stripe_artifact(
            clean_patch,
            inverted_stripe_rng,
            inverted=True,
        )

        thin_bundle_patch, _ = apply_residual_thin_bundle(
            clean_patch,
            thin_bundle_rng,
            strength_multiplier=1.0,
        )

        # Same fixed Gaussian noise realization is used across all conditions
        # so validation differences isolate the structural corruption.
        noise = rng.normal(
            0.0,
            NOISE_SIGMA,
            size=clean_patch.shape,
        ).astype(np.float32)

        noisy_clean = (
            clean_patch
            + noise
        )

        noisy_missing = (
            noisy_clean
            * (
                1.0
                - alpha
            )
            + replacement
            * alpha
        )

        noisy_drift = (
            drift_patch
            + noise
        )

        noisy_focus = (
            focus_patch
            + noise
        )

        noisy_stripe = (
            stripe_patch
            + noise
        )

        noisy_inverted_stripe = (
            inverted_stripe_patch
            + noise
        )

        noisy_thin_bundle = (
            thin_bundle_patch
            + noise
        )

        clean_inputs.append(
            noisy_clean.astype(
                np.float32
            )
        )

        missing_inputs.append(
            noisy_missing.astype(
                np.float32
            )
        )

        drift_inputs.append(
            noisy_drift.astype(
                np.float32
            )
        )

        focus_inputs.append(
            noisy_focus.astype(
                np.float32
            )
        )

        stripe_inputs.append(
            noisy_stripe.astype(
                np.float32
            )
        )

        inverted_stripe_inputs.append(
            noisy_inverted_stripe.astype(
                np.float32
            )
        )

        thin_bundle_inputs.append(
            noisy_thin_bundle.astype(
                np.float32
            )
        )

        targets.append(
            target_patch[
                None,
                :,
                :,
            ]
        )

        missing_masks.append(
            binary_masks[
                CONTEXT_RADIUS:
                CONTEXT_RADIUS + 1
            ]
        )

    return (
        torch.from_numpy(
            np.stack(
                clean_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                missing_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                drift_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                focus_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                stripe_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                inverted_stripe_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                thin_bundle_inputs,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                targets,
                axis=0,
            )
        ),
        torch.from_numpy(
            np.stack(
                missing_masks,
                axis=0,
            )
        ),
    )


def lecun_normal_(layer):
    if not isinstance(
        layer,
        (
            nn.Conv2d,
            nn.ConvTranspose2d,
        ),
    ):
        return

    fan_in = (
        layer.in_channels
        * layer.kernel_size[0]
        * layer.kernel_size[1]
    )

    target_std = math.sqrt(
        1.0 / fan_in
    )

    trunc_std = (
        target_std
        / 0.87962566103423978
    )

    with torch.no_grad():
        nn.init.trunc_normal_(
            layer.weight,
            mean=0.0,
            std=trunc_std,
            a=-2.0 * trunc_std,
            b=2.0 * trunc_std,
        )

        if layer.bias is not None:
            nn.init.zeros_(
                layer.bias
            )


# ============================================================
# 12. full-slice 2.5D U-Net
# ============================================================

class UNet2p5D(nn.Module):
    def __init__(self):
        super().__init__()

        # Only encoder input changes:
        # 1 channel -> 5 adjacent z slices.
        self.conv_x0 = nn.Conv2d(
            NUM_CONTEXT_SLICES,
            32,
            kernel_size=5,
            padding=2,
        )

        self.conv_x1 = nn.Conv2d(
            32,
            64,
            kernel_size=5,
            padding=2,
        )

        self.conv_x2 = nn.Conv2d(
            64,
            128,
            kernel_size=5,
            padding=2,
        )

        self.conv_x3 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1,
        )

        self.conv_y4 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )

        self.deconv_y3 = nn.ConvTranspose2d(
            512,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.deconv_y2 = nn.ConvTranspose2d(
            256,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.deconv_y1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        self.deconv_y0 = nn.ConvTranspose2d(
            96,
            32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        # Keep the original 33-channel final skip:
        # decoder 32 + CENTER input 1.
        self.conv_out1 = nn.Conv2d(
            33,
            32,
            kernel_size=5,
            padding=2,
        )

        self.conv_out2 = nn.Conv2d(
            32,
            16,
            kernel_size=5,
            padding=2,
        )

        self.conv_out3 = nn.Conv2d(
            16,
            1,
            kernel_size=5,
            padding=2,
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.apply(
            lecun_normal_
        )

    def forward(self, x):
        # Direct final skip remains anchored to the center slice only.
        center = x[
            :,
            CONTEXT_RADIUS:
            CONTEXT_RADIUS + 1,
            :,
            :,
        ]

        x0 = torch.sin(
            self.conv_x0(x)
        )
        x0 = F.max_pool2d(
            x0,
            2,
            2,
        )
        x0 = self.dropout(x0)

        x1 = torch.sin(
            self.conv_x1(x0)
        )
        x1 = F.max_pool2d(
            x1,
            2,
            2,
        )
        x1 = self.dropout(x1)

        x2 = torch.sin(
            self.conv_x2(x1)
        )
        x2 = F.max_pool2d(
            x2,
            2,
            2,
        )
        x2 = self.dropout(x2)

        x3 = torch.sin(
            self.conv_x3(x2)
        )
        x3 = F.max_pool2d(
            x3,
            2,
            2,
        )
        x3 = self.dropout(x3)

        y4 = torch.sin(
            self.conv_y4(x3)
        )
        y4 = torch.cat(
            [y4, x3],
            dim=1,
        )
        y4 = self.dropout(y4)

        y3 = torch.sin(
            self.deconv_y3(y4)
        )
        y3 = torch.cat(
            [y3, x2],
            dim=1,
        )
        y3 = self.dropout(y3)

        y2 = torch.sin(
            self.deconv_y2(y3)
        )
        y2 = torch.cat(
            [y2, x1],
            dim=1,
        )
        y2 = self.dropout(y2)

        y1 = torch.sin(
            self.deconv_y1(y2)
        )
        y1 = torch.cat(
            [y1, x0],
            dim=1,
        )
        y1 = self.dropout(y1)

        y0 = torch.sin(
            self.deconv_y0(y1)
        )

        # Important:
        # 2.5D context is used through the encoder;
        # direct high-resolution skip uses only z center.
        y0 = torch.cat(
            [y0, center],
            dim=1,
        )
        y0 = self.dropout(y0)

        y0 = torch.sin(
            self.conv_out1(y0)
        )
        y0 = self.dropout(y0)

        # No activation here, matching the validated full-slice topology.
        y0 = self.conv_out2(y0)
        y0 = self.dropout(y0)

        return self.conv_out3(y0)


# ============================================================
# 13. Pair loss
# ============================================================

def pair_loss(pred, clean):
    loss_dt = torch.mean(
        (pred - clean) ** 2
    )

    loss_neg = (
        F.relu(-clean)
        - F.relu(-pred)
    )

    return (
        loss_dt
        + torch.mean(
            loss_neg ** 2
        )
    )


def masked_pair_loss(
    pred,
    clean,
    mask,
):
    """
    Same pair-loss formulation, but evaluated ONLY inside the fully
    saturated restoration core.

    mask shape: [B,1,H,W]
    """
    mask = mask.to(
        dtype=pred.dtype
    )

    denom = torch.clamp(
        mask.sum(),
        min=1.0,
    )

    loss_dt = (
        (
            (pred - clean) ** 2
        )
        * mask
    ).sum() / denom

    neg_diff = (
        F.relu(-clean)
        - F.relu(-pred)
    )

    loss_neg = (
        (
            neg_diff ** 2
        )
        * mask
    ).sum() / denom

    return (
        loss_dt
        + loss_neg
    )





# ============================================================
# 13. Fixed validation visualization
# ============================================================

def display_processed(image):
    """
    Convert processed-space values to the same stable grayscale visualization:
        0 -> 128
        +/- 3 -> approximately 32 / 224
    """
    return np.clip(
        np.round(
            128.0
            + 32.0
            * image
        ),
        0,
        255,
    ).astype(np.uint8)


@torch.no_grad()
def save_validation_previews(
    model,
    val_clean_input,
    val_missing_input,
    val_drift_input,
    val_focus_input,
    val_stripe_input,
    val_inverted_stripe_input,
    val_thin_bundle_input,
    val_target,
    device,
    step,
):
    """
    Save the SAME fixed validation examples every 250 steps.

    For each selected validation sample, create one figure:
        rows = clean / missing / local drift / lose focus /
               original stripe / inverted stripe / residual thin bundle
        columns = center input / prediction / target / absolute error

    No checkpoint is saved here. This is purely visual training diagnostics.
    """
    ensure_dir(
        VALIDATION_PREVIEW_DIR
    )

    step_dir = (
        VALIDATION_PREVIEW_DIR
        / f"step_{step:04d}"
    )

    ensure_dir(
        step_dir
    )

    was_training = model.training
    model.eval()

    count = min(
        VALIDATION_PREVIEW_COUNT,
        len(val_target),
    )

    condition_names = [
        "clean",
        "missing",
        "local drift",
        "lose focus",
        "original stripe",
        "inverted stripe",
        "residual thin bundle",
    ]

    condition_tensors = [
        val_clean_input,
        val_missing_input,
        val_drift_input,
        val_focus_input,
        val_stripe_input,
        val_inverted_stripe_input,
        val_thin_bundle_input,
    ]

    for sample_index in range(
        count
    ):
        target_np = (
            val_target[
                sample_index,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        target_display = display_processed(
            target_np
        )

        fig, axes = plt.subplots(
            7,
            4,
            figsize=(
                13,
                21,
            ),
        )

        for row, (
            condition_name,
            condition_tensor,
        ) in enumerate(
            zip(
                condition_names,
                condition_tensors,
            )
        ):
            input_tensor = (
                condition_tensor[
                    sample_index:
                    sample_index + 1
                ]
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            prediction = (
                model(
                    input_tensor
                )[
                    0,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            center_input = (
                condition_tensor[
                    sample_index,
                    CONTEXT_RADIUS,
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            abs_error = np.abs(
                prediction
                - target_np
            )

            axes[
                row,
                0,
            ].imshow(
                display_processed(
                    center_input
                ),
                cmap="gray",
                vmin=0,
                vmax=255,
            )

            axes[
                row,
                1,
            ].imshow(
                display_processed(
                    prediction
                ),
                cmap="gray",
                vmin=0,
                vmax=255,
            )

            axes[
                row,
                2,
            ].imshow(
                target_display,
                cmap="gray",
                vmin=0,
                vmax=255,
            )

            axes[
                row,
                3,
            ].imshow(
                abs_error,
                cmap="gray",
                vmin=0.0,
                vmax=1.5,
            )

            axes[
                row,
                0,
            ].set_ylabel(
                condition_name,
                fontsize=11,
            )

            for col in range(
                4
            ):
                axes[
                    row,
                    col,
                ].axis(
                    "off"
                )

        column_titles = [
            "center input",
            "prediction",
            "target",
            "|prediction-target|",
        ]

        for col, title in enumerate(
            column_titles
        ):
            axes[
                0,
                col,
            ].set_title(
                title,
                fontsize=12,
            )

        fig.suptitle(
            (
                f"{EXPERIMENT_NAME} {TRAINING_CODE_VERSION} "
                f"fixed validation | step {step} | "
                f"sample {sample_index:02d}"
            ),
            fontsize=14,
        )

        fig.tight_layout(
            rect=[
                0.0,
                0.0,
                1.0,
                0.97,
            ]
        )

        fig.savefig(
            step_dir
            / f"validation_sample_{sample_index:02d}.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    if was_training:
        model.train()


# ============================================================
# 14. Validation / checkpoint / bookkeeping
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    val_clean_input,
    val_missing_input,
    val_drift_input,
    val_focus_input,
    val_target,
    val_missing_mask,
    device,
):
    model.eval()

    clean_losses = []
    missing_losses = []
    drift_losses = []
    focus_losses = []
    missing_mask_losses = []

    for start in range(
        0,
        len(val_target),
        VAL_BATCH_SIZE,
    ):
        end = start + VAL_BATCH_SIZE

        y = val_target[
            start:end
        ].to(
            device=device,
            dtype=torch.float32,
        )

        clean_x = val_clean_input[
            start:end
        ].to(
            device=device,
            dtype=torch.float32,
        )

        missing_x = val_missing_input[
            start:end
        ].to(
            device=device,
            dtype=torch.float32,
        )

        drift_x = val_drift_input[
            start:end
        ].to(
            device=device,
            dtype=torch.float32,
        )

        focus_x = val_focus_input[
            start:end
        ].to(
            device=device,
            dtype=torch.float32,
        )

        missing_mask = val_missing_mask[
            start:end
        ].to(
            device=device,
            dtype=torch.float32,
        )

        clean_pred = model(
            clean_x
        )

        missing_pred = model(
            missing_x
        )

        drift_pred = model(
            drift_x
        )

        focus_pred = model(
            focus_x
        )

        clean_losses.append(
            float(
                pair_loss(
                    clean_pred,
                    y,
                )
                .detach()
                .cpu()
            )
        )

        missing_losses.append(
            float(
                pair_loss(
                    missing_pred,
                    y,
                )
                .detach()
                .cpu()
            )
        )

        drift_losses.append(
            float(
                pair_loss(
                    drift_pred,
                    y,
                )
                .detach()
                .cpu()
            )
        )

        focus_losses.append(
            float(
                pair_loss(
                    focus_pred,
                    y,
                )
                .detach()
                .cpu()
            )
        )

        missing_mask_losses.append(
            float(
                masked_pair_loss(
                    missing_pred,
                    y,
                    missing_mask,
                )
                .detach()
                .cpu()
            )
        )

    model.train()

    clean_mean = float(
        np.mean(
            clean_losses
        )
    )

    missing_mean = float(
        np.mean(
            missing_losses
        )
    )

    drift_mean = float(
        np.mean(
            drift_losses
        )
    )

    focus_mean = float(
        np.mean(
            focus_losses
        )
    )

    missing_mask_mean = float(
        np.mean(
            missing_mask_losses
        )
    )

    score = (
        VAL_SCORE_CLEAN_WEIGHT
        * clean_mean
        + VAL_SCORE_MISSING_WEIGHT
        * missing_mean
        + VAL_SCORE_DRIFT_WEIGHT
        * drift_mean
        + VAL_SCORE_FOCUS_WEIGHT
        * focus_mean
    )

    return {
        "score": float(
            score
        ),
        "clean_loss": clean_mean,
        "missing_loss": missing_mean,
        "drift_loss": drift_mean,
        "focus_loss": focus_mean,
        "missing_mask_loss": missing_mask_mean,
    }


def save_checkpoint(
    path,
    model,
    step,
    metrics,
    parameter_count,
):
    checkpoint = {
        "model_state_dict":
            model.state_dict(),
        "experiment_name":
            EXPERIMENT_NAME,
        "training_code_version":
            TRAINING_CODE_VERSION,
        "step":
            int(
                step
            ),
        "val_loss":
            (
                None
                if metrics is None
                else float(
                    metrics[
                        "score"
                    ]
                )
            ),
        "val_metrics":
            metrics,
        "parameter_count":
            int(
                parameter_count
            ),
        "settings": {
            "architecture":
                "five_slice_early_fusion_unet_center_skip",
            "initialization":
                "scratch_lecun_normal",
            "num_context_slices":
                NUM_CONTEXT_SLICES,
            "patch_size":
                PATCH_SIZE,
            "batch_size":
                BATCH_SIZE,
            "noise_sigma":
                NOISE_SIGMA,
            "dropout":
                DROPOUT,
            "lr":
                LR,
            "train_steps":
                TRAIN_STEPS,
            "train_sample_probabilities": {
                "clean":
                    TRAIN_CLEAN_PROB,
                "missing":
                    TRAIN_MISSING_PROB,
                "local_drift":
                    TRAIN_LOCAL_DRIFT_PROB,
                "lose_focus":
                    TRAIN_LOSE_FOCUS_PROB,
            },
            "stripe_enabled":
                True,
            "stripe_family_probability":
                TRAIN_STRIPE_FAMILY_PROB,
            "original_stripe_marginal_probability":
                TRAIN_ORIGINAL_STRIPE_MARGINAL_PROB,
            "inverted_stripe_marginal_probability":
                TRAIN_INVERTED_STRIPE_MARGINAL_PROB,
            "stripe_polarities_mutually_exclusive_per_patch":
                True,
            "residual_thin_bundle_probability":
                TRAIN_RESIDUAL_THIN_BUNDLE_PROB,
            "residual_thin_bundle_processed_space":
                True,
            "missing": {
                "pattern_probabilities": {
                    "center_only":
                        MISSING_CENTER_ONLY_PROB,
                    "center_dominant":
                        MISSING_CENTER_DOMINANT_PROB,
                    "neighbor_only":
                        MISSING_NEIGHBOR_ONLY_PROB,
                    "irregular_multi":
                        MISSING_IRREGULAR_MULTI_PROB,
                },
                "base_area_range":
                    list(
                        MISSING_BASE_AREA_RANGE
                    ),
                "position_jitter_px":
                    MISSING_POSITION_JITTER_PX,
                "soft_edge_sigma":
                    MISSING_SOFT_EDGE_SIGMA,
                "processed_space":
                    True,
                "membrane_aware_placement":
                    True,
                "shared_core_required":
                    False,
                "forced_z_continuity":
                    False,
            },
            "local_drift": {
                "whole_patch_crop_offset":
                    True,
                "center_fixed":
                    True,
                "magnitude_range_px":
                    list(
                        LOCAL_DRIFT_MAG_RANGE_PX
                    ),
                "direction":
                    "uniform_continuous_0_to_2pi",
                "affected_neighbor_count":
                    list(
                        LOCAL_DRIFT_AFFECTED_NEIGHBOR_COUNT
                    ),
            },
            "lose_focus": {
                "exactly_one_slice":
                    True,
                "index_weights":
                    LOSE_FOCUS_INDEX_WEIGHTS.tolist(),
                "sigma_range":
                    list(
                        LOSE_FOCUS_SIGMA_RANGE
                    ),
                "alpha_range":
                    list(
                        LOSE_FOCUS_ALPHA_RANGE
                    ),
                "application_stage":
                    "raw_full_slice_before_processed_space_normalization",
            },
            "preprocess_gaussian_sigma":
                PREPROCESS_GAUSSIAN_SIGMA,
            "preprocess_hist_bins":
                PREPROCESS_HIST_BINS,
            "hpf_sigma":
                HPF_SIGMA,
            "hpf_gain":
                HPF_GAIN,
            "hpf_strong_extra_gain":
                HPF_STRONG_EXTRA_GAIN,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def write_experiment_description(
    train_records,
    val_records,
    test_records,
    parameter_count,
):
    text = f"""Biological refinement experiment: {EXPERIMENT_NAME}
Code version: {TRAINING_CODE_VERSION}

Scientific base:
- accepted Stage-2 architecture, target, loss, optimization, data split,
  major-condition mixture, missing, local-drift and lose-focus logic unchanged
- only artifact overlays are extended:
  original stripe + inverted stripe + residual thin bundle

Architecture:
- EXACT reference 5-slice early-fusion U-Net
- parameters: {parameter_count:,}
- center-only full-resolution processed skip
- scratch LeCun-normal initialization

Optimization:
- Adam
- LR: {LR}
- batch: {BATCH_SIZE}
- steps: {TRAIN_STEPS}
- total stochastic training patches: {TRAIN_STEPS * BATCH_SIZE}
- Gaussian noise sigma: {NOISE_SIGMA}
- pair loss only
- no scheduler / weight decay / early stopping

Training sample mixture:
- clean: {TRAIN_CLEAN_PROB}
- missing: {TRAIN_MISSING_PROB}
- local drift: {TRAIN_LOCAL_DRIFT_PROB}
- lose focus: {TRAIN_LOSE_FOCUS_PROB}
- stripe family overall probability: {TRAIN_STRIPE_FAMILY_PROB}
- original white-core stripe marginal probability: {TRAIN_ORIGINAL_STRIPE_MARGINAL_PROB}
- inverted black-core stripe marginal probability: {TRAIN_INVERTED_STRIPE_MARGINAL_PROB}
- original/inverted stripe are mutually exclusive within each 256x256 patch
- residual thin-bundle independent probability: {TRAIN_RESIDUAL_THIN_BUNDLE_PROB}
- missing / drift / focus are mutually exclusive

Missing region:
- after complete-slice full-slice preprocessing
- membrane-aware placement
- z-pattern probabilities:
    center_only={MISSING_CENTER_ONLY_PROB}
    center_dominant={MISSING_CENTER_DOMINANT_PROB}
    neighbor_only={MISSING_NEIGHBOR_ONLY_PROB}
    irregular_multi={MISSING_IRREGULAR_MULTI_PROB}
- base area range: {MISSING_BASE_AREA_RANGE}
- no shared-core constraint
- no forced smooth z-continuity
- gray/low-frequency replacement only

Local drift:
- center slice fixed
- affected neighbors: {LOCAL_DRIFT_AFFECTED_NEIGHBOR_COUNT}
- whole 256 patch is cropped from shifted XY location
- magnitude range: {LOCAL_DRIFT_MAG_RANGE_PX} px
- arbitrary continuous direction

Lose focus:
- exactly one slice
- index weights: {LOSE_FOCUS_INDEX_WEIGHTS.tolist()}
- sigma: {LOSE_FOCUS_SIGMA_RANGE}
- alpha: {LOSE_FOCUS_ALPHA_RANGE}
- applied to full RAW slice before full-slice preprocessing

Validation/checkpoint:
- fixed validation samples: {VAL_SAMPLES}
- validation every {VAL_EVERY} steps
- preview every {VALIDATION_PREVIEW_EVERY} steps
- checkpoint score remains clean/missing/drift/focus only
- original stripe / inverted stripe / residual thin bundle remain
  visualization-only for checkpoint selection; checkpoint score stays
  clean/missing/drift/focus only

Frozen master data:
- root: {DATA_ROOT}
- train: {len(train_records)} blocks x {EXPECTED_SLICES_PER_BLOCK} slices = {len(train_records) * EXPECTED_SLICES_PER_BLOCK} source slices
- val: {len(val_records)} blocks x {EXPECTED_SLICES_PER_BLOCK} slices = {len(val_records) * EXPECTED_SLICES_PER_BLOCK} source slices
- test: {len(test_records)} blocks x {EXPECTED_SLICES_PER_BLOCK} slices = {len(test_records) * EXPECTED_SLICES_PER_BLOCK} held-out source slices
- train blocks: {', '.join(r['name'] for r in train_records)}
- val blocks: {', '.join(r['name'] for r in val_records)}
- test blocks: {', '.join(r['name'] for r in test_records)}

Engineering acceleration:
- cache root: {CACHE_ROOT}
- cached raw uint8 + full-slice processed float32 + HPF target float32 + membrane score float32
- mmap-backed cache; no scientific-value quantization
- one sequential CPU batch prefetch producer: {ENABLE_BATCH_PREFETCH}
- prefetch depth: {PREFETCH_DEPTH}
- cache build workers: {CACHE_BUILD_WORKERS}
"""

    path = EXPERIMENT_ROOT / "experiment.txt"
    path.write_text(
        text,
        encoding="utf-8",
    )
    return path

def save_loss_plot(
    train_losses,
    val_steps,
    val_metrics_history,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            10,
            10,
        ),
    )

    axes[0].plot(
        np.arange(
            1,
            len(train_losses) + 1,
        ),
        train_losses,
        linewidth=1.0,
    )

    axes[0].set_title(
        "Training pair loss"
    )

    axes[0].set_xlabel(
        "step"
    )

    axes[0].set_ylabel(
        "loss"
    )

    if val_steps:
        axes[1].plot(
            val_steps,
            [
                item["score"]
                for item in val_metrics_history
            ],
            label="score",
        )

        axes[1].plot(
            val_steps,
            [
                item["clean_loss"]
                for item in val_metrics_history
            ],
            label="clean",
        )

        axes[1].plot(
            val_steps,
            [
                item["missing_loss"]
                for item in val_metrics_history
            ],
            label="missing",
        )

        axes[1].plot(
            val_steps,
            [
                item["drift_loss"]
                for item in val_metrics_history
            ],
            label="drift",
        )

        axes[1].plot(
            val_steps,
            [
                item["focus_loss"]
                for item in val_metrics_history
            ],
            label="focus",
        )

        axes[1].legend()

    axes[1].set_title(
        "Validation"
    )

    axes[1].set_xlabel(
        "step"
    )

    axes[1].set_ylabel(
        "pair loss"
    )

    fig.tight_layout()

    fig.savefig(
        EXPERIMENT_ROOT
        / "loss.png",
        dpi=160,
    )

    plt.close(
        fig
    )


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)

    if hours >= 1.0:
        return f"{int(hours)}h {int(minutes):02d}m {sec:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m {sec:04.1f}s"
    return f"{sec:.1f}s"


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def make_batch_prefetch_executor():
    if not ENABLE_BATCH_PREFETCH:
        return None

    if PREFETCH_DEPTH != 1:
        raise ValueError(
            "Stage-2 intentionally supports PREFETCH_DEPTH=1 only so the "
            "single producer preserves sequential NumPy RNG semantics."
        )

    return ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="stage2_batch_prefetch",
    )


# ============================================================
# 15. Command-line interface and main
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-2 2.5D biological refinement: frozen clean blocks -> training."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="Immutable JRC Liver clean master block directory (train/val/test).")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT,
                        help="Separate deterministic train/val .npy cache directory.")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT,
                        help="Dedicated training run directory (checkpoints, logs, previews).")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto", help="Explicit device does not fall back.")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                        help="Number of optimizer steps (default: 2000; e.g. 10 for a smoke test).")
    parser.add_argument("--cache-workers", type=int, default=CACHE_BUILD_WORKERS,
                        help="Thread workers for deterministic block-cache build (default: 2).")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate dataset and directory safety; create no files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing output-root and retrain; never deletes input/cache.")
    return parser.parse_args(argv)


def _paths_intersect(a, b):
    return a == b or a in b.parents or b in a.parents


def configure_run(args):
    """Validate root paths before touching the source, cache or output."""
    global DATA_ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR, CACHE_ROOT
    global EXPERIMENT_ROOT, CHECKPOINT_DIR, BEST_CHECKPOINT, FINAL_CHECKPOINT
    global VALIDATION_PREVIEW_DIR, CACHE_BUILD_WORKERS, OVERWRITE_EXPERIMENT, TRAIN_STEPS

    data = args.data_root.expanduser().resolve()
    cache = args.cache_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()

    if not data.is_dir():
        raise FileNotFoundError(f"Master dataset directory does not exist: {data}")
    if args.output_root.expanduser().is_symlink():
        raise ValueError("Output-root must not itself be a symbolic link.")
    if args.cache_root.expanduser().is_symlink():
        raise ValueError("Cache-root must not itself be a symbolic link.")
    if not isinstance(args.steps, int) or args.steps < 1:
        raise ValueError("--steps must be a positive integer.")
    if not isinstance(args.cache_workers, int) or args.cache_workers < 1:
        raise ValueError("--cache-workers must be >= 1.")
    if _paths_intersect(data, cache) or _paths_intersect(data, output) or _paths_intersect(cache, output):
        raise ValueError("Data, cache and output roots must be separate non-overlapping trees.")

    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    if output == Path(output.anchor) or output in (cwd, home) or output in cwd.parents or output in home.parents:
        raise ValueError("Unsafe output-root: choose a dedicated subdirectory.")
    if output.exists() and not args.check_only and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output}. "
            "Choose a new directory or explicitly pass --overwrite."
        )
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output-root exists but is not a directory: {output}")
    if cache.exists() and not cache.is_dir():
        raise ValueError(f"Cache-root exists but is not a directory: {cache}")

    DATA_ROOT = data
    TRAIN_DIR, VAL_DIR, TEST_DIR = (data / "train", data / "val", data / "test")
    CACHE_ROOT = cache
    EXPERIMENT_ROOT = output
    CHECKPOINT_DIR = output / "checkpoints"
    BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
    FINAL_CHECKPOINT = CHECKPOINT_DIR / "final.pt"
    VALIDATION_PREVIEW_DIR = output / "validation_previews"
    CACHE_BUILD_WORKERS = args.cache_workers
    TRAIN_STEPS = args.steps
    OVERWRITE_EXPERIMENT = args.overwrite


def main(argv=None):
    args = parse_args(argv)
    configure_run(args)
    cv2.setNumThreads(1)

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    print("=" * 88)
    print("Stage-2 biological refinement training | fixed 1024x16 split + deterministic cache/prefetch")
    print("Conda env:", conda_env or "(not reported)")
    print("Master data:", DATA_ROOT)
    print("Experiment:", EXPERIMENT_ROOT)
    print("=" * 88)

    # ------------------------------------------------------------------
    # 1. Strictly validate the complete frozen train/val/test split.
    # ------------------------------------------------------------------
    master = discover_master_split()
    train_records = master["train"]
    val_records = master["val"]
    test_records = master["test"]

    print("\nFrozen master split validated:")
    for split, records in (
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ):
        print(
            f"  {split:5s}: {len(records)} blocks x "
            f"{EXPECTED_SLICES_PER_BLOCK} slices = "
            f"{len(records) * EXPECTED_SLICES_PER_BLOCK} slices"
        )

    # Test is deliberately not cached/opened by the training loop.
    print(
        "Held-out test split: validated only; not used for cache, training, "
        "validation, checkpointing, or preprocessing statistics."
    )

    if args.check_only:
        # Availability check performs no training or file writes.
        get_device(args.device)
        print("Check-only: master split, device and directory safety validated; no files written.")
        print("Cache root:", CACHE_ROOT)
        print("Output root:", EXPERIMENT_ROOT)
        print("Device request:", args.device)
        return

    # Verify the requested hardware before costly cache building.
    device = get_device(args.device)

    # ------------------------------------------------------------------
    # 2. Build/reuse deterministic train+val caches BEFORE any stochastic RNG.
    # ------------------------------------------------------------------
    prepare_stage2_cache(
        train_records,
        val_records,
    )

    # ------------------------------------------------------------------
    # 3. Experiment output. Cache lives outside EXPERIMENT_ROOT and survives
    #    experiment overwrite/reruns.
    # ------------------------------------------------------------------
    if OVERWRITE_EXPERIMENT:
        reset_dir(EXPERIMENT_ROOT)
    else:
        ensure_dir(EXPERIMENT_ROOT)

    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(VALIDATION_PREVIEW_DIR)

    code_snapshot = save_current_code_snapshot()

    # Preserve accepted Stage-2 baseline RNG anchors exactly.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    stripe_rng = np.random.default_rng(
        SEED + STRIPE_TRAIN_RNG_SEED_OFFSET
    )

    thin_bundle_rng = np.random.default_rng(
        SEED + THIN_BUNDLE_TRAIN_RNG_SEED_OFFSET
    )

    train_pairs = open_block_pairs(train_records)
    val_pairs = open_block_pairs(val_records)

    # ------------------------------------------------------------------
    # 4. Fixed validation: same scientific construction as Stage-2, now reading
    #    deterministic full-slice values from cache.
    # ------------------------------------------------------------------
    validation_build_start = time.perf_counter()

    (
        val_clean_input,
        val_missing_input,
        val_drift_input,
        val_focus_input,
        val_stripe_input,
        val_inverted_stripe_input,
        val_thin_bundle_input,
        val_target,
        val_missing_mask,
    ) = build_fixed_validation(val_pairs)

    validation_build_seconds = (
        time.perf_counter()
        - validation_build_start
    )

    print(
        "Validation:",
        {
            "clean": tuple(val_clean_input.shape),
            "missing": tuple(val_missing_input.shape),
            "drift": tuple(val_drift_input.shape),
            "focus": tuple(val_focus_input.shape),
            "stripe_original": tuple(val_stripe_input.shape),
            "stripe_inverted": tuple(val_inverted_stripe_input.shape),
            "thin_bundle": tuple(val_thin_bundle_input.shape),
            "target": tuple(val_target.shape),
        },
    )

    model = UNet2p5D().to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    expected_parameters = 2204289
    if parameter_count != expected_parameters:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count:,}; "
            f"expected {expected_parameters:,}."
        )

    # Preflight forward pass on the real model before optimization.
    # Use eval() so Dropout consumes NO torch RNG and the accepted Stage-2 baseline training
    # RNG sequence remains unchanged.
    model.eval()
    with torch.no_grad():
        preflight_x = torch.zeros(
            (1, NUM_CONTEXT_SLICES, PATCH_SIZE, PATCH_SIZE),
            dtype=torch.float32,
            device=device,
        )
        preflight_y = model(preflight_x)
        if tuple(preflight_y.shape) != (1, 1, PATCH_SIZE, PATCH_SIZE):
            raise RuntimeError(
                f"Preflight output shape mismatch: {tuple(preflight_y.shape)}"
            )
        if not torch.isfinite(preflight_y).all():
            raise RuntimeError("NaN/Inf in model preflight forward pass.")
        del preflight_x, preflight_y
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    experiment_path = write_experiment_description(
        train_records,
        val_records,
        test_records,
        parameter_count,
    )

    initial_metrics = evaluate(
        model,
        val_clean_input,
        val_missing_input,
        val_drift_input,
        val_focus_input,
        val_target,
        val_missing_mask,
        device,
    )

    print("Initial validation:", initial_metrics)
    print("\nExperiment:", EXPERIMENT_ROOT)
    print("Parameters:", f"{parameter_count:,}")
    print(
        "Source train data:",
        f"{len(train_records)} blocks / "
        f"{len(train_records) * EXPECTED_SLICES_PER_BLOCK} slices",
    )
    print(
        "Training exposure:",
        f"{TRAIN_STEPS} steps x {BATCH_SIZE} patches = "
        f"{TRAIN_STEPS * BATCH_SIZE} stochastic patches",
    )
    print(
        "Major sample probabilities:",
        {
            "clean": TRAIN_CLEAN_PROB,
            "missing": TRAIN_MISSING_PROB,
            "local_drift": TRAIN_LOCAL_DRIFT_PROB,
            "lose_focus": TRAIN_LOSE_FOCUS_PROB,
        },
    )
    print(
        "Stripe family probability:",
        TRAIN_STRIPE_FAMILY_PROB,
    )
    print(
        "  original white-core marginal:",
        TRAIN_ORIGINAL_STRIPE_MARGINAL_PROB,
    )
    print(
        "  inverted black-core marginal:",
        TRAIN_INVERTED_STRIPE_MARGINAL_PROB,
    )
    print(
        "  both stripe polarities in one patch:",
        "IMPOSSIBLE by categorical sampling",
    )
    print(
        "Residual thin-bundle probability:",
        TRAIN_RESIDUAL_THIN_BUNDLE_PROB,
    )
    print("CPU batch prefetch:", ENABLE_BATCH_PREFETCH)
    print(
        "Fixed-validation build:",
        format_seconds(validation_build_seconds),
    )

    train_losses = []
    train_missing_mask_losses = []
    val_steps = []
    val_history = []

    best_val = float("inf")
    best_step = None
    best_metrics = None

    model.train()

    batch_wait_times = []
    optimize_times = []
    validation_times = []
    step_wall_times = []

    training_start = time.perf_counter()

    prefetch_executor = make_batch_prefetch_executor()
    next_batch_future = None

    try:
        if prefetch_executor is not None:
            next_batch_future = prefetch_executor.submit(
                sample_training_batch,
                train_pairs,
                rng,
                stripe_rng,
                thin_bundle_rng,
            )

        pbar = tqdm(
            range(1, TRAIN_STEPS + 1),
            desc="Stage-2 biological refinement",
            dynamic_ncols=True,
        )

        for step in pbar:
            step_start = time.perf_counter()

            # ----------------------------------------------------------
            # CPU batch generation. With prefetch, the one sequential
            # producer has normally prepared this while the device trained the
            # previous batch. No extra RNG stream is introduced.
            # ----------------------------------------------------------
            wait_start = time.perf_counter()

            if prefetch_executor is None:
                batch = sample_training_batch(
                    train_pairs,
                    rng,
                    stripe_rng,
                    thin_bundle_rng,
                )
            else:
                batch = next_batch_future.result()

                if step < TRAIN_STEPS:
                    next_batch_future = prefetch_executor.submit(
                        sample_training_batch,
                        train_pairs,
                        rng,
                        stripe_rng,
                        thin_bundle_rng,
                    )
                else:
                    next_batch_future = None

            batch_wait_seconds = (
                time.perf_counter()
                - wait_start
            )
            batch_wait_times.append(batch_wait_seconds)

            (
                structural_input,
                target,
                missing_alpha,
                missing_replacement,
                missing_center_mask,
                type_counts,
            ) = batch

            optimize_start = time.perf_counter()

            structural_input = structural_input.to(
                device=device,
                dtype=torch.float32,
            )
            target = target.to(
                device=device,
                dtype=torch.float32,
            )
            missing_alpha = missing_alpha.to(
                device=device,
                dtype=torch.float32,
            )
            missing_replacement = missing_replacement.to(
                device=device,
                dtype=torch.float32,
            )
            missing_center_mask = missing_center_mask.to(
                device=device,
                dtype=torch.float32,
            )

            # Accepted Stage-2 ordering: Gaussian noise first, then missing fill.
            noisy_input = (
                structural_input
                + torch.randn_like(
                    structural_input
                )
                * NOISE_SIGMA
            )

            noisy_input = (
                noisy_input
                * (1.0 - missing_alpha)
                + missing_replacement
                * missing_alpha
            )

            optimizer.zero_grad(set_to_none=True)

            pred = model(noisy_input)

            if not torch.isfinite(pred).all():
                raise RuntimeError(
                    f"NaN/Inf in prediction at training step {step}."
                )

            loss = pair_loss(
                pred,
                target,
            )

            missing_mask_loss = masked_pair_loss(
                pred,
                target,
                missing_center_mask,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NaN/Inf in training loss at step {step}."
                )

            loss.backward()
            optimizer.step()

            # Transferring a scalar to CPU synchronizes outstanding accelerator work.
            loss_value = float(
                loss.detach().cpu()
            )
            missing_mask_value = float(
                missing_mask_loss.detach().cpu()
            )

            optimize_seconds = (
                time.perf_counter()
                - optimize_start
            )
            optimize_times.append(optimize_seconds)

            train_losses.append(loss_value)
            train_missing_mask_losses.append(
                missing_mask_value
            )

            postfix = {
                "train": f"{loss_value:.5f}",
                "miss_mask": f"{missing_mask_value:.5f}",
                "C": type_counts["clean"],
                "M": type_counts["missing"],
                "D": type_counts["local_drift"],
                "F": type_counts["lose_focus"],
                "SW": type_counts["stripe_original"],
                "SI": type_counts["stripe_inverted"],
                "TB": type_counts["thin_bundle"],
                "wait": f"{batch_wait_seconds:.2f}s",
                "opt": f"{optimize_seconds:.2f}s",
            }

            if (
                step % VAL_EVERY == 0
                or step == TRAIN_STEPS
            ):
                val_start = time.perf_counter()

                metrics = evaluate(
                    model,
                    val_clean_input,
                    val_missing_input,
                    val_drift_input,
                    val_focus_input,
                    val_target,
                    val_missing_mask,
                    device,
                )

                val_seconds = (
                    time.perf_counter()
                    - val_start
                )
                validation_times.append(val_seconds)

                val_steps.append(int(step))
                val_history.append(dict(metrics))

                postfix.update({
                    "val": f"{metrics['score']:.5f}",
                    "vC": f"{metrics['clean_loss']:.5f}",
                    "vM": f"{metrics['missing_loss']:.5f}",
                    "vD": f"{metrics['drift_loss']:.5f}",
                    "vF": f"{metrics['focus_loss']:.5f}",
                    "vsec": f"{val_seconds:.1f}",
                })

                if metrics["score"] < best_val:
                    best_val = float(metrics["score"])
                    best_step = int(step)
                    best_metrics = dict(metrics)

                    save_checkpoint(
                        BEST_CHECKPOINT,
                        model,
                        step,
                        metrics,
                        parameter_count,
                    )

            if (
                step % VALIDATION_PREVIEW_EVERY == 0
                or step == TRAIN_STEPS
            ):
                save_validation_previews(
                    model,
                    val_clean_input,
                    val_missing_input,
                    val_drift_input,
                    val_focus_input,
                    val_stripe_input,
                    val_inverted_stripe_input,
                    val_thin_bundle_input,
                    val_target,
                    device,
                    step,
                )

            step_seconds = (
                time.perf_counter()
                - step_start
            )
            step_wall_times.append(step_seconds)

            postfix["step"] = f"{step_seconds:.2f}s"
            postfix["avg"] = (
                f"{mean_or_zero(step_wall_times):.2f}s"
            )

            pbar.set_postfix(postfix)

    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(
                wait=True,
                cancel_futures=False,
            )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    final_metrics = evaluate(
        model,
        val_clean_input,
        val_missing_input,
        val_drift_input,
        val_focus_input,
        val_target,
        val_missing_mask,
        device,
    )

    save_checkpoint(
        FINAL_CHECKPOINT,
        model,
        TRAIN_STEPS,
        final_metrics,
        parameter_count,
    )

    np.save(
        EXPERIMENT_ROOT / "train_loss.npy",
        np.asarray(
            train_losses,
            dtype=np.float32,
        ),
    )

    np.save(
        EXPERIMENT_ROOT / "train_missing_mask_loss.npy",
        np.asarray(
            train_missing_mask_losses,
            dtype=np.float32,
        ),
    )

    np.save(
        EXPERIMENT_ROOT / "val_steps.npy",
        np.asarray(
            val_steps,
            dtype=np.int32,
        ),
    )

    save_loss_plot(
        train_losses,
        val_steps,
        val_history,
    )

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "training_code_version": TRAINING_CODE_VERSION,
        "parameter_count": int(parameter_count),
        "architecture": "five_slice_early_fusion_unet_center_skip",
        "scientific_base": "fixed_refinement_baseline_with_artifact_overlays",
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "total_training_patches": TRAIN_STEPS * BATCH_SIZE,
        "lr": LR,
        "noise_sigma": NOISE_SIGMA,
        "stripe_enabled": True,
        "stripe_family_probability": TRAIN_STRIPE_FAMILY_PROB,
        "original_stripe_marginal_probability":
            TRAIN_ORIGINAL_STRIPE_MARGINAL_PROB,
        "inverted_stripe_marginal_probability":
            TRAIN_INVERTED_STRIPE_MARGINAL_PROB,
        "stripe_polarities_mutually_exclusive_per_patch": True,
        "residual_thin_bundle_probability":
            TRAIN_RESIDUAL_THIN_BUNDLE_PROB,
        "train_sample_probabilities": {
            "clean": TRAIN_CLEAN_PROB,
            "missing": TRAIN_MISSING_PROB,
            "local_drift": TRAIN_LOCAL_DRIFT_PROB,
            "lose_focus": TRAIN_LOSE_FOCUS_PROB,
        },
        "data": {
            "root": str(DATA_ROOT),
            "cache_root": str(CACHE_ROOT),
            "train_blocks": len(train_records),
            "val_blocks": len(val_records),
            "test_blocks": len(test_records),
            "slices_per_block": EXPECTED_SLICES_PER_BLOCK,
            "train_source_slices": (
                len(train_records)
                * EXPECTED_SLICES_PER_BLOCK
            ),
            "val_source_slices": (
                len(val_records)
                * EXPECTED_SLICES_PER_BLOCK
            ),
            "test_source_slices": (
                len(test_records)
                * EXPECTED_SLICES_PER_BLOCK
            ),
            "test_used_in_training": False,
        },
        "acceleration": {
            "environment": conda_env,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_build_workers": CACHE_BUILD_WORKERS,
            "batch_prefetch": ENABLE_BATCH_PREFETCH,
            "prefetch_depth": PREFETCH_DEPTH,
            "cache_dtype_raw": "uint8",
            "cache_dtype_processed": "float32",
            "cache_dtype_target": "float32",
            "cache_dtype_membrane_score": "float32",
            "rng_semantics": (
                "single sequential NumPy producer; no worker-local RNG streams"
            ),
        },
        "missing": {
            "pattern_probabilities": {
                "center_only": MISSING_CENTER_ONLY_PROB,
                "center_dominant": MISSING_CENTER_DOMINANT_PROB,
                "neighbor_only": MISSING_NEIGHBOR_ONLY_PROB,
                "irregular_multi": MISSING_IRREGULAR_MULTI_PROB,
            },
            "base_area_range": list(
                MISSING_BASE_AREA_RANGE
            ),
            "shared_core": False,
            "forced_z_continuity": False,
        },
        "local_drift": {
            "whole_patch_crop_offset": True,
            "magnitude_range_px": list(
                LOCAL_DRIFT_MAG_RANGE_PX
            ),
            "affected_neighbor_count": list(
                LOCAL_DRIFT_AFFECTED_NEIGHBOR_COUNT
            ),
        },
        "lose_focus": {
            "index_weights": (
                LOSE_FOCUS_INDEX_WEIGHTS.tolist()
            ),
            "sigma_range": list(
                LOSE_FOCUS_SIGMA_RANGE
            ),
            "alpha_range": list(
                LOSE_FOCUS_ALPHA_RANGE
            ),
        },
        "initial_validation": initial_metrics,
        "best_step": best_step,
        "best_validation": best_metrics,
        "final_validation": final_metrics,
        "timing": {
            "validation_build_seconds": float(
                validation_build_seconds
            ),
            "training_loop_seconds": float(
                training_seconds
            ),
            "average_step_wall_seconds": mean_or_zero(
                step_wall_times
            ),
            "average_batch_wait_seconds": mean_or_zero(
                batch_wait_times
            ),
            "average_optimize_seconds": mean_or_zero(
                optimize_times
            ),
            "average_validation_seconds": mean_or_zero(
                validation_times
            ),
        },
        "code_snapshot": str(code_snapshot),
        "experiment_description": str(experiment_path),
    }

    summary_path = (
        EXPERIMENT_ROOT
        / "train_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print("Best checkpoint:", BEST_CHECKPOINT)
    print(
        "Best step / val:",
        best_step,
        best_val,
        best_metrics,
    )
    print("Final checkpoint:", FINAL_CHECKPOINT)
    print("Final validation:", final_metrics)
    print("Training loop:", format_seconds(training_seconds))
    print(
        "Average step / wait / optimize:",
        f"{mean_or_zero(step_wall_times):.3f}s / "
        f"{mean_or_zero(batch_wait_times):.3f}s / "
        f"{mean_or_zero(optimize_times):.3f}s",
    )
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()

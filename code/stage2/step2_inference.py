#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cryo-FIB-SEM Stage-2 Step 2: 2.5D Biological Refinement Inference

Author: Ningyan Zhang
Last modified: October 13, 2026

Description:
    Refine membrane contrast and suppress residual imaging artifacts in a
    Stage-1 corrected cryo-FIB-SEM volume using a pretrained five-slice,
    early-fusion U-Net. Read the Stage-1 grayscale uint8 stacked TIFF directly,
    preserve the original ZYX geometry, and write one restored uint8 stack.

    Optionally perform clean-reference domain transfer BEFORE local alignment
    and model inference. Both modes share the same network, checkpoint format,
    five-slice context, center-referenced integer translation, overlapping
    tiles, and canonical output decoding. Neither mode adds training-time
    corruptions or further membrane-enhancement postprocessing.

Required arguments:
    --input PATH            Stage-1 output_stack.tif: grayscale uint8 TIFF
                            volume in ZYX order. Conventional multipage TIFFs
                            and compatible explicitly labeled ZYX stacks are
                            supported, including Z=1. Height and width must
                            each be at least 256 pixels. Input is read-only.
    --checkpoint PATH       Trained best.pt or final.pt containing
                            model_state_dict, or a standalone model state_dict.
                            Weights are loaded onto CPU, checked against the
                            fixed 2,204,289-parameter network, then moved
                            onto the requested inference device.
    --output-dir PATH       New output directory. Existing directories are
                            protected unless --overwrite is specified.

Domain-transfer options (mutually exclusive):
    --domain-transfer       Enable FFT log-amplitude and spatial-intensity
                            CDF matching (DEFAULT). Requires access to the
                            clean JRC Liver training-reference split.
    --no-domain-transfer    Disable domain transfer entirely. Do NOT read,
                            validate, create, or use any clean reference
                            directory or domain-statistics cache. Full-slice
                            preprocessing and local alignment remain enabled.
    --reference-train PATH  Reference directory used only when matching is on.
                            Default: data/clean_blocks_1024_z16/train
                            Expected: 16 block folders, 16 grayscale uint8
                            1024x1024 TIFF slices per block (256 in total).
                            Only the clean TRAIN split is used; validation
                            and test clean blocks are never reference inputs.
    --cache-dir PATH        Reusable, content-verified global domain-statistics
                            cache (only when matching is on).
                            Default: experiments/stage2_domain_cache
                            Cached statistics are NOT a second input stack.

Other options:
    --device {auto,cuda,mps,cpu}
                            Default: auto (CUDA > Apple MPS > CPU). Explicitly
                            requested unavailable devices raise an error.
    --execution {auto,legacy}  Default auto: device-aware execution optimization.
                                Legacy reproduces original scheduling, alignment
                                extraction, batch=4, and inference structure.
    --batch-size INT          Override execution batch size (auto if omitted).
    --max-batch INT           Upper bound for CUDA auto-tuning (default 32).
    --prepare-workers INT     CPU patch-preparation workers (auto if omitted).
    --timing-json PATH        Optional timing report, kept OUTSIDE result folder.
    --cuda-amp              Opt in to float16 autocast on CUDA. Default FP32;
                            not supported for CPU or MPS in this program.
    --check-only            Validate inputs, selected mode, checkpoint,
                            geometry and runtime availability; write nothing.
    --overwrite             Replace an existing output directory explicitly.
                            Never replaces source data, model weights or the
                            separate clean-training reference directory.

Output structure (identical for BOTH transfer modes):
    --output-dir/
        output_stack.tif            uint8 ZYX restored volume; identical Z,
                                    height, width and slice order to input.
        visualization/
            slice_XXXX.png          Five evenly spaced four-panel comparisons
                                    (one per slice if Z < 5). Each panel shows:
                                      1. Artifact-corrected Stage-1 input
                                      2. Actual processed model input
                                      3. Refined uint8 output
                                      4. Signed model correction (1x display)

    No per-slice TIFF outputs, additional TIFF volumes, diagnostics, summaries,
    metadata JSON files, or source-code snapshots are written into the result
    directory. The optional reference-statistics cache is kept separately.

Usage example A: Inference WITH domain transfer (default)
    python step2_inference.py \
        --input /path/to/stage1_output/output_stack.tif \
        --checkpoint /path/to/stage2_training/checkpoints/best.pt \
        --reference-train /path/to/clean_blocks_1024_z16/train \
        --output-dir /path/to/refined_with_transfer \
        --device cuda --domain-transfer

Usage example B: Inference WITHOUT domain transfer
    python step2_inference.py \
        --input /path/to/stage1_output/output_stack.tif \
        --checkpoint /path/to/stage2_training/checkpoints/best.pt \
        --output-dir /path/to/refined_without_transfer \
        --device mps --no-domain-transfer

Usage example C: Validate configuration without creating files
    python step2_inference.py \
        --input /path/to/stage1_output/output_stack.tif \
        --checkpoint /path/to/stage2_training/checkpoints/best.pt \
        --output-dir /path/to/stage2_output \
        --device cuda --no-domain-transfer --check-only

Usage example D: Optional CUDA half-precision inference
    python step2_inference.py \
        --input /path/to/stage1_output/output_stack.tif \
        --checkpoint /path/to/stage2_training/checkpoints/best.pt \
        --reference-train /path/to/clean_blocks_1024_z16/train \
        --output-dir /path/to/stage2_output \
        --device cuda --domain-transfer --cuda-amp

Scientific inference procedure (preserve these definitions):
    1. Process EACH complete native-resolution uint8 Z slice: standardize by
       its own mean and standard deviation, subtract a Gaussian low-pass
       (sigma=10), subtract the modal high-pass histogram-bin center (50
       bins), and clip to [-3, 3]. Keep processed data as float32.
    2. If domain transfer is enabled, deterministically sample 128 clean
       256x256 patches from the 256 available training-reference slices,
       using the identical full-slice preprocessing. Estimate source FFT
       log-amplitude CDFs from ALL overlapping 256x256 tiles of ALL input
       slices. Match source to reference amplitude while retaining Fourier
       phase and DC, using 128-pixel tile overlap and Hann blending. Then
       apply the fixed spatial CDF mapping estimated from ALL pixels after
       the FFT step. The mapping belongs to the whole supplied stack, not
       separately to each Z slice. No test images enter the clean reference.
    3. Build a five-slice input [z-2, z-1, z, z+1, z+2], replicating the
       nearest valid slice at Z boundaries. Align neighboring slices to the
       CENTER slice separately for each model tile: estimate integer XY
       offsets from a 320x320 context using the central 160x160 normalized
       cross-correlation region, searching within a circular radius of 15
       pixels with confidence rejection. Keep the center fixed. Apply shifts
       by direct pixel cropping, WITHOUT spatial interpolation or resampling.
    4. Run the center-refinement U-Net on 256x256 five-channel tiles, with
       128-pixel overlap and inference batch size 4. Blend predicted center
       slices with the original clipped Hann weights. Model topology and
       checkpoint tensor shapes are unchanged from training.
    5. Decode the float32 prediction using one dataset-wide mapping:
           uint8 = clip(round(128 + 32 * prediction), 0, 255)
       Do NOT re-add per-slice mean, standard deviation, low-pass or mode;
       low-frequency background is deliberately not reconstructed.
    6. Save exactly one final uint8 stack and up to five PNG comparisons.
       Previews are display-only and never feed back into inference.

Reproducibility and safety:
    - Adaptive execution changes scheduling and may choose different batch sizes;
      floating-point results can vary by device, batch, and library backend.
      Use --execution legacy for baseline output comparisons.
    - Default inference uses FP32. CUDA AMP is an explicit optional change in
      precision; different hardware or AMP may yield slightly different pixels.
    - Reference statistics are content-addressed by the full input TIFF,
      clean-reference TIFFs and numerical mapping settings. A renamed cache
      signature in this documented release may rebuild an old cache once;
      it does not alter the FFT/CDF algorithms or statistical sampling seed.
    - --check-only performs no writes. Domain transfer OFF does not require
      access to JRC Liver reference images or to a domain cache.
    - The final stack is produced incrementally in a temporary TIFF and
      renamed after all slices complete. Source and checkpoint files remain
      unchanged. --overwrite protects unrelated source/reference directories.

Requirements: Python 3.9+, PyTorch (2.4+ recommended), NumPy, SciPy,
              OpenCV, tifffile, tqdm.
"""

import argparse
from collections import OrderedDict
from pathlib import Path
import hashlib
import math
import re
import shutil
import tempfile
import os
import time
import gc
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import cv2
import numpy as np
import tifffile
from scipy import fft as sp_fft
from scipy import ndimage
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================
# 1. Frozen inference, preprocessing and visualization parameters
# ========================================================================
# These constants preserve the validated model and integer-alignment numerical settings.
# They must not be changed as part of code cleanup.

NUM_CONTEXT_SLICES = 5
CONTEXT_RADIUS = 2
TILE_SIZE = 256
TILE_OVERLAP = 128
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP
TILE_BATCH = 4
BLEND_EPS = 1e-3
LOCAL_ALIGNMENT_ENABLED = True
ALIGN_CONTEXT_SIZE = 320
ALIGN_CONTEXT_MARGIN = (ALIGN_CONTEXT_SIZE - TILE_SIZE) // 2
MAX_LOCAL_SHIFT_PX = 15
ALIGN_ESTIMATION_GAUSSIAN_SIGMA = 1.25
ALIGN_MATCH_SIZE = 160
ALIGN_MIN_BEST_NCC = 0.15
ALIGN_MIN_NCC_IMPROVEMENT = 0.020
ALIGN_MIN_PEAK_MARGIN = 0.0
ALIGN_PEAK_EXCLUSION_RADIUS = 2
PREPROCESS_CACHE_SIZE = 7
PREPROCESS_GAUSSIAN_SIGMA = 10.0
PREPROCESS_HIST_BINS = 50
CLIP_MIN = -3.0
CLIP_MAX = 3.0
DROPOUT = 0.1
CANONICAL_BACKGROUND = 128.0
CANONICAL_GAIN = 32.0
CORRECTION_PREVIEW_GAIN = 1.0
VIS_SEPARATOR_WIDTH = 6
VIS_HEADER_HEIGHT = 22
VIS_FONT_SCALE = 0.36
VIS_FONT_THICKNESS = 1
VIS_TEXT_X = 5
VIS_TEXT_Y = 15

# ========================================================================
# 2. Optional domain-transfer distribution parameters
# ========================================================================
# The reference uses the frozen train split; real-source statistics use
# every slice of the input stack. No statistics are built when disabled.

DOMAIN_REFERENCE_EXPECTED_BLOCKS = 16
DOMAIN_REFERENCE_EXPECTED_SLICES_PER_BLOCK = 16
DOMAIN_REFERENCE_EXPECTED_SIZE = 1024
DOMAIN_PATCH_SIZE = 256
DOMAIN_TRAIN_PATCH_COUNT = 128
DOMAIN_REFERENCE_RANDOM_SEED = 20260924
DOMAIN_TRANSFER_TILE_OVERLAP = 128
DOMAIN_TRANSFER_TILE_STRIDE = DOMAIN_PATCH_SIZE - DOMAIN_TRANSFER_TILE_OVERLAP
DOMAIN_BLEND_EPS = 1e-3
DOMAIN_FFT_HIST_BINS = 4096
DOMAIN_SPATIAL_HIST_BINS = 4096
DOMAIN_FFT_WORKERS = -1
DOMAIN_FFT_EPS = 1e-12
DOMAIN_FFT_BATCH_TILES = 32
DOMAIN_REFERENCE_DIR = None
DOMAIN_REFERENCE_CACHE_DIR = None


# ========================================================================
# 3. General filesystem and uint8 image utilities
# ========================================================================
# Utilities for frozen TRAIN reference TIFFs and image conversion.

def ensure_dir(path):
    """Create a directory and missing parent directories for temporary/cache I/O.
    
        Args: path: Target directory path."""
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def natural_key(path):
    """Build the numeric-aware filename sort key for frozen TIFF blocks.
    
        Args: path: Filesystem path.
        Returns: Comparable sequence alternating lowercase strings and integers."""
    parts = re.split(
        r"(\d+)",
        path.name,
    )

    return [
        int(p) if p.isdigit()
        else p.lower()
        for p in parts
    ]


def list_tifs(folder):
    """Find and naturally sort 2D TIFF files in a reference block directory.
    
        Args: folder: Directory containing individual TIFF slices.
        Returns: Sorted list of TIFF paths; raises if the directory has no TIFFs."""
    paths = []

    for pattern in (
        "*.tif",
        "*.tiff",
        "*.TIF",
        "*.TIFF",
    ):
        paths.extend(
            folder.glob(pattern)
        )

    paths = sorted(
        paths,
        key=natural_key,
    )

    if not paths:
        raise RuntimeError(
            f"No TIFF files found in: {folder}"
        )

    return paths


def read_uint8(path):
    """Read either one frozen reference TIFF or a slice in the input stack."""
    if isinstance(path, InputStackSlice):
        image = path.reader.get(path.index)
    else:
        image = tifffile.imread(str(path))
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"Expected grayscale uint8 2D image, got {image.shape}/{image.dtype}: {path}")
    return image


def to_uint8(image):
    """Round and clip scalar/array values to the original uint8 representation.
    
        Args: image: Numeric image, in grayscale intensity units.
        Returns: uint8 array with values clipped to [0, 255]."""
    return np.clip(
        np.round(image),
        0,
        255,
    ).astype(np.uint8)



# ========================================================================
# 4. Frozen JRC Liver TRAIN reference discovery
# ========================================================================
# The reference is used solely in the enabled domain-transfer mode.

def _validate_reference_tiff_header(path):
    """Validate one clean uint8 training-reference TIFF without loading its pixels."""
    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            shape = tuple(series.shape)
            dtype = np.dtype(series.dtype)
    except Exception as error:
        raise RuntimeError(
            f"Cannot inspect domain-reference TIFF: {path}"
        ) from error

    expected_shape = (
        DOMAIN_REFERENCE_EXPECTED_SIZE,
        DOMAIN_REFERENCE_EXPECTED_SIZE,
    )
    if shape != expected_shape:
        raise ValueError(
            f"Expected clean TRAIN reference shape {expected_shape}, "
            f"got {shape}: {path}"
        )
    if dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 clean TRAIN reference, got {dtype}: {path}"
        )


def discover_domain_reference_slices():
    """
    Discover the complete shared clean training-reference split:
        16 block folders x 16 TIFF slices = 256 clean source slices.
    """
    if not DOMAIN_REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Clean training-reference directory not found: "
            f"{DOMAIN_REFERENCE_DIR}"
        )

    block_dirs = sorted(
        [
            p
            for p in DOMAIN_REFERENCE_DIR.iterdir()
            if p.is_dir() and p.name.startswith("block_")
        ],
        key=natural_key,
    )

    if len(block_dirs) != DOMAIN_REFERENCE_EXPECTED_BLOCKS:
        raise RuntimeError(
            f"Expected {DOMAIN_REFERENCE_EXPECTED_BLOCKS} clean training-reference blocks, "
            f"found {len(block_dirs)} under {DOMAIN_REFERENCE_DIR}"
        )

    slice_paths = []
    for block_dir in block_dirs:
        paths = list_tifs(block_dir)
        if len(paths) != DOMAIN_REFERENCE_EXPECTED_SLICES_PER_BLOCK:
            raise RuntimeError(
                f"Expected {DOMAIN_REFERENCE_EXPECTED_SLICES_PER_BLOCK} TIFFs in "
                f"{block_dir}, found {len(paths)}"
            )

        for path in paths:
            _validate_reference_tiff_header(path)
            slice_paths.append(path)

    expected_total = (
        DOMAIN_REFERENCE_EXPECTED_BLOCKS
        * DOMAIN_REFERENCE_EXPECTED_SLICES_PER_BLOCK
    )
    if len(slice_paths) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} clean training-reference slices, "
            f"found {len(slice_paths)}"
        )

    return slice_paths


def _random_crop_start(h, w, rng):
    """Sample a valid 256x256 training-reference crop origin using the given RNG.
    
        Args: h, w: Full processed slice dimensions; rng: seeded NumPy generator.
        Returns: Integer (y, x) crop origin."""
    if h < DOMAIN_PATCH_SIZE or w < DOMAIN_PATCH_SIZE:
        raise ValueError(
            f"Image {(h, w)} is smaller than DOMAIN_PATCH_SIZE={DOMAIN_PATCH_SIZE}."
        )
    y = int(rng.integers(0, h - DOMAIN_PATCH_SIZE + 1))
    x = int(rng.integers(0, w - DOMAIN_PATCH_SIZE + 1))
    return y, x


def _sample_train_processed_patches(train_paths, rng):
    """
    Sample exactly 128 processed patches from clean training-reference slices:
        read one frozen 1024x1024 uint8 TRAIN slice
        -> full-slice normalization and high-pass preprocessing
        -> random 256x256 crop.
    """
    if not train_paths:
        raise RuntimeError("No clean training-reference slices available.")

    replace = len(train_paths) < DOMAIN_TRAIN_PATCH_COUNT
    indices = rng.choice(
        len(train_paths),
        size=DOMAIN_TRAIN_PATCH_COUNT,
        replace=replace,
    )

    patches = []
    descriptors = []

    for idx in tqdm(
        indices,
        desc="Sample processed clean training-reference patches",
        dynamic_ncols=True,
    ):
        path = train_paths[int(idx)]
        raw = read_uint8(path)

        expected_shape = (
            DOMAIN_REFERENCE_EXPECTED_SIZE,
            DOMAIN_REFERENCE_EXPECTED_SIZE,
        )
        if raw.shape != expected_shape:
            raise ValueError(
                f"Expected reference shape {expected_shape}, got {raw.shape}: {path}"
            )

        processed = preprocess_slice(raw)["processed"]
        y, x = _random_crop_start(*processed.shape, rng)
        patch = processed[
            y:y + DOMAIN_PATCH_SIZE,
            x:x + DOMAIN_PATCH_SIZE,
        ].copy()

        patches.append(patch.astype(np.float32))
        descriptors.append({
            "block": path.parent.name,
            "file": path.name,
            "y": int(y),
            "x": int(x),
        })

    return np.stack(patches, axis=0), descriptors



# ========================================================================
# 5. Domain-transfer tile geometry and CDF construction
# ========================================================================
# These functions preserve the accepted FFT log-amplitude, spatial
# histogram and overlap-blending mathematics.

def _domain_positions(length):
    """Enumerate edge-complete overlapping 256px domain-transfer tile origins.
    
        Args: length: Height or width of the full slice.
        Returns: Ordered valid tile starts including the final image edge."""
    if length < DOMAIN_PATCH_SIZE:
        raise ValueError(
            f"Domain transfer requires dimension >= {DOMAIN_PATCH_SIZE}, got {length}."
        )
    if length == DOMAIN_PATCH_SIZE:
        return [0]

    positions = list(range(
        0,
        length - DOMAIN_PATCH_SIZE + 1,
        DOMAIN_TRANSFER_TILE_STRIDE,
    ))
    last = length - DOMAIN_PATCH_SIZE
    if positions[-1] != last:
        positions.append(last)
    return positions


def _domain_blend_weight():
    """Build the original nonzero Hann weight used for FFT tile blending.
    
        Returns: float32 [256,256] weights clipped by DOMAIN_BLEND_EPS."""
    one_d = np.hanning(DOMAIN_PATCH_SIZE).astype(np.float32)
    weight = np.outer(one_d, one_d).astype(np.float32)
    return np.maximum(weight, DOMAIN_BLEND_EPS)


def _spatial_hist_edges():
    # Exact full-slice processed inputs are clipped to [CLIP_MIN, CLIP_MAX].
    """Create fixed [-3,3] histogram boundaries for processed-space CDFs.
    
        Returns: float32 spatial intensity histogram bin edges."""
    return np.linspace(
        CLIP_MIN,
        CLIP_MAX,
        DOMAIN_SPATIAL_HIST_BINS + 1,
        dtype=np.float32,
    )


def _fft_hist_edges():
    # For a 256x256 patch clipped to +/-3, every DFT coefficient magnitude is
    # bounded by N * max_abs. This gives a deterministic complete histogram
    # range and lets us stream ALL test tiles without retaining coefficients.
    """Create deterministic bounds for per-tile FFT log-amplitude histograms.
    
        Returns: float32 log-amplitude histogram bin edges."""
    n = DOMAIN_PATCH_SIZE * DOMAIN_PATCH_SIZE
    max_abs = max(abs(float(CLIP_MIN)), abs(float(CLIP_MAX)))
    max_log_amp = math.log1p(float(n) * max_abs)
    return np.linspace(
        0.0,
        max_log_amp,
        DOMAIN_FFT_HIST_BINS + 1,
        dtype=np.float32,
    )


def _cdf_from_hist(hist):
    """Normalize histogram counts into a cumulative probability distribution.
    
        Args: hist: Nonnegative counts for one feature distribution.
        Returns: Monotonically nondecreasing float64 cumulative distribution."""
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if total <= 0:
        raise RuntimeError("Cannot build CDF from an empty histogram.")
    return np.cumsum(hist, dtype=np.float64) / float(total)


def _accumulate_spatial_hist(image, edges, hist):
    """Accumulate full processed-space pixel counts without discarding slices.
    
        Args: image: float32 pixel values; edges: fixed histogram boundaries;
            hist: mutable count array updated in place."""
    values = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite values found while accumulating spatial histogram.")
    counts, _ = np.histogram(values, bins=edges)
    hist += counts.astype(np.int64, copy=False)


def _tile_coordinates(h, w):
    """Construct the full Y and X tile origin lists for domain statistics.
    
        Args: h, w: Native image dimensions.
        Returns: Pair (ys, xs) of edge-complete tile origin sequences."""
    ys = _domain_positions(h)
    xs = _domain_positions(w)
    return ys, xs


def _accumulate_fft_hist_from_image(image, edges, hist):
    """
    Accumulate FFT log-amplitude histogram from EVERY 256x256 tile on the
    domain-transfer grid. DC is excluded independently for every tile.
    """
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    ys, xs = _tile_coordinates(h, w)
    coords = [(y, x) for y in ys for x in xs]

    batch_size = max(1, int(DOMAIN_FFT_BATCH_TILES))
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        patches = np.stack(
            [
                image[y:y + DOMAIN_PATCH_SIZE, x:x + DOMAIN_PATCH_SIZE]
                for y, x in chunk
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        spectra = sp_fft.rfft2(
            patches,
            axes=(-2, -1),
            workers=DOMAIN_FFT_WORKERS,
        )
        log_amp = np.log1p(np.abs(spectra)).astype(np.float32)
        values = log_amp.reshape(log_amp.shape[0], -1)[:, 1:].reshape(-1)
        counts, _ = np.histogram(values, bins=edges)
        hist += counts.astype(np.int64, copy=False)

    return len(coords)


def _unique_inverse_cdf(cdf, values):
    """Build an interpolation-safe inverse of an empirical CDF.
    
        Args: cdf: Cumulative probabilities; values: associated histogram centers.
        Returns: Unique increasing percentiles and matching intensity values."""
    cdf = np.asarray(cdf, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    unique_cdf, first = np.unique(cdf, return_index=True)
    unique_values = values[first]
    if unique_cdf[0] > 0.0:
        unique_cdf = np.concatenate(([0.0], unique_cdf))
        unique_values = np.concatenate(([values[0]], unique_values))
    if unique_cdf[-1] < 1.0:
        unique_cdf = np.concatenate((unique_cdf, [1.0]))
        unique_values = np.concatenate((unique_values, [values[-1]]))
    return unique_cdf, unique_values


def _cdf_map(values, source_edges, source_cdf, target_edges, target_cdf):
    """Quantile-map source values to target values via empirical histogram CDFs.
    
        Args: values: Source values; source_edges/source_cdf: source histogram;
            target_edges/target_cdf: target histogram.
        Returns: Float64 values mapped through the reference distribution."""
    source_centers = (
        source_edges[:-1].astype(np.float64)
        + source_edges[1:].astype(np.float64)
    ) * 0.5
    target_centers = (
        target_edges[:-1].astype(np.float64)
        + target_edges[1:].astype(np.float64)
    ) * 0.5

    percentiles = np.interp(
        np.asarray(values, dtype=np.float64),
        source_centers,
        source_cdf,
        left=0.0,
        right=1.0,
    )
    target_cdf_u, target_values_u = _unique_inverse_cdf(
        target_cdf,
        target_centers,
    )
    return np.interp(
        percentiles,
        target_cdf_u,
        target_values_u,
    )


def _fft_transfer_patch(patch, transfer):
    """Map the log-amplitude CDF of one 256px tile without changing FFT phase.
    
        The DC coefficient is explicitly preserved, and inverse rFFT reconstructs
        the float32 tile. This operation is unchanged from the accepted method.
    
        Args: patch: float32 [256,256] processed tile; transfer: global CDF maps.
        Returns: Transferred float32 [256,256] tile."""
    patch = np.asarray(patch, dtype=np.float32)
    if patch.shape != (DOMAIN_PATCH_SIZE, DOMAIN_PATCH_SIZE):
        raise ValueError(
            f"FFT transfer expects {DOMAIN_PATCH_SIZE}x{DOMAIN_PATCH_SIZE} patch, "
            f"got {patch.shape}"
        )

    spectrum = sp_fft.rfft2(patch, workers=DOMAIN_FFT_WORKERS)
    amplitude = np.abs(spectrum).astype(np.float32)
    log_amp = np.log1p(amplitude).astype(np.float32)

    flat = log_amp.reshape(-1)
    matched = flat.copy()
    matched[1:] = _cdf_map(
        flat[1:],
        transfer["test_fft_edges"],
        transfer["test_fft_cdf"],
        transfer["train_fft_edges"],
        transfer["train_fft_cdf"],
    ).astype(np.float32)
    # Preserve the tile mean exactly: DC is never changed.
    matched[0] = flat[0]
    matched = matched.reshape(log_amp.shape)
    matched_amp = np.expm1(matched).astype(np.float32)

    unit_phase = np.ones_like(spectrum)
    np.divide(
        spectrum,
        amplitude,
        out=unit_phase,
        where=amplitude > DOMAIN_FFT_EPS,
    )
    shifted = sp_fft.irfft2(
        unit_phase * matched_amp,
        s=patch.shape,
        workers=DOMAIN_FFT_WORKERS,
    ).astype(np.float32)
    if not np.all(np.isfinite(shifted)):
        raise RuntimeError("Non-finite values produced by patch FFT transfer.")
    return shifted


def fft_domain_shift(processed, transfer):
    """Apply the learned 256x256 FFT map tile-wise with overlap blending."""
    image = np.asarray(processed, dtype=np.float32)
    h, w = image.shape
    ys, xs = _tile_coordinates(h, w)
    weight = _domain_blend_weight()
    out_sum = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)

    for y in ys:
        for x in xs:
            patch = image[y:y + DOMAIN_PATCH_SIZE, x:x + DOMAIN_PATCH_SIZE]
            shifted = _fft_transfer_patch(patch, transfer)
            out_sum[
                y:y + DOMAIN_PATCH_SIZE,
                x:x + DOMAIN_PATCH_SIZE,
            ] += shifted * weight
            weight_sum[
                y:y + DOMAIN_PATCH_SIZE,
                x:x + DOMAIN_PATCH_SIZE,
            ] += weight

    result = out_sum / np.maximum(weight_sum, DOMAIN_BLEND_EPS)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Non-finite values produced by full-slice FFT transfer.")
    return result.astype(np.float32)


def spatial_domain_shift(image_float, transfer):
    """Apply one FIXED all-test-after-FFT -> clean-train processed CDF map."""
    image = np.asarray(image_float, dtype=np.float32)
    mapped = _cdf_map(
        image.reshape(-1),
        transfer["test_after_fft_spatial_edges"],
        transfer["test_after_fft_spatial_cdf"],
        transfer["train_spatial_edges"],
        transfer["train_spatial_cdf"],
    ).reshape(image.shape).astype(np.float32)
    mapped = np.clip(mapped, CLIP_MIN, CLIP_MAX).astype(np.float32)
    if not np.all(np.isfinite(mapped)):
        raise RuntimeError("Non-finite values produced by spatial domain transfer.")
    return mapped



# ========================================================================
# 6. Global clean-target and whole-stack source statistics
# ========================================================================
# Float32 disk-backed intermediates prevent memory exhaustion.
# Domain maps must be learned from ALL input slices before inference.

def build_domain_transfer_statistics(train_paths, test_paths):
    """Measure clean-TRAIN and ALL-real-stack distributions for global transfer.
    
        The same frozen 128 clean TRAIN patches define the target FFT/spatial CDFs.
        Every test slice contributes processed-space pixels and FFT tile values.
        A disk-backed temporary float32 stack supports the post-FFT spatial CDF.
        This function never modifies the Stage-1 input or final inference output.
    
        Args: train_paths: Frozen clean TRAIN TIFF paths; test_paths: input-stack
            slice adapters exposing the same read_uint8 interface as 2D TIFFs.
        Returns: Mapping containing FFT/spatial histogram edges, CDFs and counts."""
    if not test_paths:
        raise RuntimeError("No Stage-1 test TIFFs available for domain source.")

    rng = np.random.default_rng(DOMAIN_REFERENCE_RANDOM_SEED)

    # ------------------------------
    # TRAIN target/reference domain.
    # ------------------------------
    train_patches, train_desc = _sample_train_processed_patches(
        train_paths,
        rng,
    )

    spatial_edges = _spatial_hist_edges()
    train_spatial_hist = np.zeros(
        DOMAIN_SPATIAL_HIST_BINS,
        dtype=np.int64,
    )
    _accumulate_spatial_hist(
        train_patches,
        spatial_edges,
        train_spatial_hist,
    )
    train_spatial_cdf = _cdf_from_hist(train_spatial_hist)

    fft_edges = _fft_hist_edges()
    train_fft_hist = np.zeros(
        DOMAIN_FFT_HIST_BINS,
        dtype=np.int64,
    )
    for patch in train_patches:
        _accumulate_fft_hist_from_image(
            patch,
            fft_edges,
            train_fft_hist,
        )
    train_fft_cdf = _cdf_from_hist(train_fft_hist)

    # ------------------------------
    # TEST source domain: ALL data.
    # ------------------------------
    first_raw = read_uint8(test_paths[0])
    h, w = first_raw.shape
    if h < DOMAIN_PATCH_SIZE or w < DOMAIN_PATCH_SIZE:
        raise ValueError(
            f"Test slice {(h, w)} is smaller than domain patch {DOMAIN_PATCH_SIZE}."
        )

    ensure_dir(DOMAIN_REFERENCE_CACHE_DIR)
    processed_tmp_path = (
        DOMAIN_TEMP_DIR
        / "processed_tmp.npy"
    )
    after_fft_tmp_path = (
        DOMAIN_TEMP_DIR
        / "afterfft_tmp.npy"
    )

    for temp_path in (processed_tmp_path, after_fft_tmp_path):
        if temp_path.exists():
            temp_path.unlink()

    processed_stack = None
    after_fft_stack = None
    try:
        processed_stack = np.lib.format.open_memmap(
            processed_tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(test_paths), h, w),
        )

        test_spatial_hist = np.zeros(
            DOMAIN_SPATIAL_HIST_BINS,
            dtype=np.int64,
        )
        test_fft_hist = np.zeros(
            DOMAIN_FFT_HIST_BINS,
            dtype=np.int64,
        )
        num_test_fft_patches = 0

        for zi, path in enumerate(tqdm(
            test_paths,
            desc="Measure ALL real-test post-full-slice data",
            dynamic_ncols=True,
        )):
            raw = first_raw if zi == 0 else read_uint8(path)
            if raw.shape != (h, w):
                raise ValueError(
                    f"All test slices must share one shape. First={(h, w)}, "
                    f"got={raw.shape}: {path}"
                )

            processed = preprocess_slice(raw)["processed"]
            processed_stack[zi] = processed
            _accumulate_spatial_hist(
                processed,
                spatial_edges,
                test_spatial_hist,
            )
            num_test_fft_patches += _accumulate_fft_hist_from_image(
                processed,
                fft_edges,
                test_fft_hist,
            )

        processed_stack.flush()
        test_spatial_cdf = _cdf_from_hist(test_spatial_hist)
        test_fft_cdf = _cdf_from_hist(test_fft_hist)

        transfer = {
            "train_spatial_edges": spatial_edges,
            "train_spatial_cdf": train_spatial_cdf,
            "test_spatial_edges": spatial_edges.copy(),
            "test_spatial_cdf": test_spatial_cdf,
            "train_fft_edges": fft_edges,
            "train_fft_cdf": train_fft_cdf,
            "test_fft_edges": fft_edges.copy(),
            "test_fft_cdf": test_fft_cdf,
        }

        # Apply the learned FFT map to EVERY processed test slice. The spatial
        # source CDF is then measured from EVERY post-FFT pixel, not from a
        # sampled subset of patches.
        after_fft_stack = np.lib.format.open_memmap(
            after_fft_tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(test_paths), h, w),
        )
        after_min = np.inf
        after_max = -np.inf

        for zi in tqdm(
            range(len(test_paths)),
            desc="Apply FFT transfer to ALL test slices",
            dynamic_ncols=True,
        ):
            after_fft = fft_domain_shift(
                np.asarray(processed_stack[zi], dtype=np.float32),
                transfer,
            )
            after_fft_stack[zi] = after_fft
            after_min = min(after_min, float(after_fft.min()))
            after_max = max(after_max, float(after_fft.max()))

        after_fft_stack.flush()
        if not np.isfinite(after_min) or not np.isfinite(after_max):
            raise RuntimeError("Invalid global range after FFT domain transfer.")
        if not after_max > after_min:
            after_max = after_min + 1e-6

        after_fft_edges = np.linspace(
            after_min,
            after_max,
            DOMAIN_SPATIAL_HIST_BINS + 1,
            dtype=np.float32,
        )
        after_fft_hist = np.zeros(
            DOMAIN_SPATIAL_HIST_BINS,
            dtype=np.int64,
        )
        for zi in tqdm(
            range(len(test_paths)),
            desc="Measure ALL post-FFT test pixels",
            dynamic_ncols=True,
        ):
            _accumulate_spatial_hist(
                after_fft_stack[zi],
                after_fft_edges,
                after_fft_hist,
            )

        transfer["test_after_fft_spatial_edges"] = after_fft_edges
        transfer["test_after_fft_spatial_cdf"] = _cdf_from_hist(after_fft_hist)

        transfer.update({
            "num_train_patches": int(len(train_patches)),
            "num_test_slices": int(len(test_paths)),
            "num_test_fft_patches": int(num_test_fft_patches),
            "num_test_spatial_pixels": int(len(test_paths) * h * w),
            "test_height": int(h),
            "test_width": int(w),
        })
        return transfer

    finally:
        if processed_stack is not None:
            try:
                processed_stack.flush()
            except Exception:
                pass
            del processed_stack
        if after_fft_stack is not None:
            try:
                after_fft_stack.flush()
            except Exception:
                pass
            del after_fft_stack
        for temp_path in (processed_tmp_path, after_fft_tmp_path):
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass



# ========================================================================
# 7. Content-addressed domain-statistics cache
# ========================================================================
# The cache is outside the final output directory and is unused in
# --no-domain-transfer mode.

def sha256_file(path):
    """Content fingerprint so cache reuse cannot depend on mtime alone."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def domain_signature(train_paths, input_path):
    """Fingerprint reference TIFF content, complete input stack and mapping settings.
    
        Args: train_paths: Frozen reference TIFF paths; input_path: Stage-1 stack.
        Returns: Stable SHA-256 hex identifier for cache invalidation."""
    digest = hashlib.sha256()
    params = ("five_slice_processed_fft_spatial_cdf_v1", DOMAIN_PATCH_SIZE,
              DOMAIN_TRAIN_PATCH_COUNT, DOMAIN_REFERENCE_RANDOM_SEED,
              DOMAIN_TRANSFER_TILE_OVERLAP, DOMAIN_FFT_HIST_BINS,
              DOMAIN_SPATIAL_HIST_BINS, DOMAIN_FFT_BATCH_TILES,
              PREPROCESS_GAUSSIAN_SIGMA, PREPROCESS_HIST_BINS,
              CLIP_MIN, CLIP_MAX)
    digest.update(repr(params).encode("utf8"))
    digest.update(str(input_path.resolve()).encode("utf8"))
    digest.update(sha256_file(input_path).encode("ascii"))
    for file in train_paths:
        digest.update(str(file.resolve()).encode("utf8"))
        digest.update(sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def load_or_build_domain_transfer_statistics(test_paths, input_path):
    """Reuse content-addressed FFT/CDF statistics or recompute them once.
    
        Disk-backed build intermediates live only in a temporary cache directory.
        This function is never called when --no-domain-transfer is selected.
    
        Args: test_paths: Stage-1 input-slice adapters; input_path: stacked TIFF.
        Returns: Transfer dictionary, including a loaded_from_cache flag."""
    train_paths = discover_domain_reference_slices()
    signature = domain_signature(train_paths, input_path)
    cache_path = DOMAIN_REFERENCE_CACHE_DIR / f"reference_{signature}.npz"
    array_keys = (
        "train_spatial_edges", "train_spatial_cdf",
        "test_spatial_edges", "test_spatial_cdf",
        "train_fft_edges", "train_fft_cdf",
        "test_fft_edges", "test_fft_cdf",
        "test_after_fft_spatial_edges", "test_after_fft_spatial_cdf",
    )
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["signature"].item()) == signature:
                    result = {key: cached[key] for key in array_keys}
                    result["loaded_from_cache"] = True
                    print(f"Domain reference cache hit: {cache_path}")
                    return result
        except (OSError, ValueError, KeyError):
            print(f"Rebuilding invalid domain reference cache: {cache_path}")

    global DOMAIN_TEMP_DIR
    with tempfile.TemporaryDirectory(prefix=".build_", dir=DOMAIN_REFERENCE_CACHE_DIR) as temp:
        DOMAIN_TEMP_DIR = Path(temp)
        result = build_domain_transfer_statistics(train_paths, test_paths)
    result["loaded_from_cache"] = False
    with tempfile.NamedTemporaryFile(prefix=".reference_", suffix=".npz",
                                     dir=DOMAIN_REFERENCE_CACHE_DIR, delete=False) as out_file:
        tmp = Path(out_file.name)
    try:
        np.savez_compressed(tmp, signature=np.asarray(signature),
                            **{key: result[key] for key in array_keys})
        tmp.replace(cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"Domain reference cache created: {cache_path}")
    return result


def apply_stage2_domain_shift(processed, transfer):
    """Apply both domain maps or bypass them when the reference is absent.
    
        Args: processed: Original float32 full-slice processed image;
            transfer: global FFT/CDF maps, or None when transfer is disabled.
        Returns: Named original, post-FFT, and final float32 processed arrays."""
    processed = np.asarray(processed, dtype=np.float32)
    if transfer is None:
        copy = processed.copy()
        return {
            "original_processed": copy,
            "fft_shift_processed": copy,
            "domain_shifted_processed": copy,
        }

    after_fft = fft_domain_shift(processed, transfer)
    final = spatial_domain_shift(after_fft, transfer)
    return {
        "original_processed": processed,
        "fft_shift_processed": after_fft,
        "domain_shifted_processed": final,
    }



# ========================================================================
# 8. Five-slice context, preprocessing and output decoding
# ========================================================================
# Keep full-slice processing and center-only canonical decoding.
# Do NOT inverse-transform the output back into the old lowpass background.

def context_indices(
    center_index,
    num_slices,
):
    """
    Five-slice context selection with replicated Z boundaries.

    Use [z-2, z-1, z, z+1, z+2]. At stack boundaries, replicate the
    nearest available slice by clipping indices to [0, num_slices - 1].
    """
    return [
        int(
            np.clip(
                center_index + dz,
                0,
                num_slices - 1,
            )
        )
        for dz in range(
            -CONTEXT_RADIUS,
            CONTEXT_RADIUS + 1,
        )
    ]


def preprocess_slice(raw):
    """
    Full-slice preprocessing matched to the trained five-slice refinement model.
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

    return {
        "processed": processed,
        "mean": mean,
        "std": std,
        "lowpass": lowpass,
        "mode": mode,
    }


def decode_canonical_membrane(processed):
    """
    Fixed dataset-wide decoding.

    IMPORTANT:
    This intentionally does NOT use the current center slice's:
        mean / std / lowpass / mode

    Therefore low-frequency Stage-1 shading is not explicitly added back.
    """
    return to_uint8(
        CANONICAL_BACKGROUND
        + CANONICAL_GAIN
        * processed
    )



# ========================================================================
# 9. Small LRU cache for five-slice model context
# ========================================================================
# Only deterministic full-slice preprocessing is cached here.

class SlicePreprocessCache:
    """LRU cache of preprocessed full slices and optional domain-shifted arrays.
    
        Holds at most max_items slices to limit RAM while respecting five-slice
        context reuse. It does not alter the scientific image transformation."""
    def __init__(
        self,
        paths,
        max_items,
        domain_reference,
    ):
        """Configure source slices, LRU capacity and the optional global CDF maps."""
        self.paths = paths
        self.max_items = max_items
        self.domain_reference = domain_reference
        self.drop_unused_lowpass = False
        self.cache = OrderedDict()

    def get(self, index):
        """Return one cached or freshly processed slice, updating LRU ordering.
        
            Args: index: Original Z index in the stacked Stage-1 input.
            Returns: Dictionary containing the uint8 input, float32 processed image
                and original/FFT-shifted processed views for the selected mode."""
        if index in self.cache:
            value = self.cache.pop(
                index
            )
            self.cache[index] = value
            return value

        raw = read_uint8(
            self.paths[index]
        )

        # Full-slice normalization and high-pass preprocessing are applied first.
        basis = preprocess_slice(
            raw
        )

        shifted = apply_stage2_domain_shift(
            basis["processed"],
            self.domain_reference,
        )

        # Keep the original acquisition image and each processed-domain stage.
        # infer_processed_stack() consumes basis["processed"], so local integer
        # alignment and the refinement network see the final processed-domain image.
        basis["raw"] = raw
        basis["processed_original"] = shifted[
            "original_processed"
        ]
        basis["fft_shift_processed"] = shifted[
            "fft_shift_processed"
        ]
        basis["processed"] = shifted[
            "domain_shifted_processed"
        ]
        if self.drop_unused_lowpass:
            # Only preprocess_slice uses this large intermediary; no subsequent
            # inference or preview reads it. Save ~128 MB per 7589x4186 slice.
            basis.pop("lowpass", None)

        self.cache[index] = basis

        while (
            len(self.cache)
            > self.max_items
        ):
            self.cache.popitem(
                last=False
            )

        return basis

    def clear(self):
        """Release references to all cached preprocessed slice arrays."""
        self.cache.clear()



# ========================================================================
# 10. Fixed five-slice U-Net topology and checkpoint compatibility
# ========================================================================
# No architecture, layer order, tensor shape or model forward change.

def lecun_normal_(layer):
    """Apply the frozen LeCun initialization to convolutional model layers.
    
        The checkpoint subsequently overwrites all initialized weights; keeping
        the original initialization preserves model-construction behavior."""
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


class UNet2p5D(nn.Module):
    """Five-slice early-fusion U-Net for center-slice refinement.
    
        Uses four encoder/decoder scales, sine activations and only the center
        input for the final full-resolution skip. Checkpoint tensor shapes are
        preserved exactly (2,204,289 parameters)."""
    def __init__(self):
        """Construct the trained network topology and initialize convolution weights."""
        super().__init__()

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

        # decoder 32 + CENTER input 1
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
        """Predict one processed-space center slice from five adjacent slices.
        
            Args: x: float32 tensor [B,5,256,256].
            Returns: float32 tensor [B,1,256,256]; no canonical uint8 decoding here."""
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

        y0 = torch.cat(
            [y0, center],
            dim=1,
        )
        y0 = self.dropout(y0)

        y0 = torch.sin(
            self.conv_out1(y0)
        )
        y0 = self.dropout(y0)

        y0 = self.conv_out2(y0)
        y0 = self.dropout(y0)

        return self.conv_out3(y0)



# ========================================================================
# 11. Center-referenced local integer translation alignment
# ========================================================================
# 320px estimation context, 160px NCC ROI, integer shift <= 15px.
# Neighbor model crops are direct pixel crops, never interpolated.

def make_alignment_shift_mask():
    """
    Valid integer translations in a circular search radius.

    score_map has shape:
        [2*MAX_LOCAL_SHIFT_PX + 1,
         2*MAX_LOCAL_SHIFT_PX + 1]

    row -> dy, column -> dx.
    """
    radius = int(
        MAX_LOCAL_SHIFT_PX
    )

    values = np.arange(
        -radius,
        radius + 1,
        dtype=np.int32,
    )

    dy, dx = np.meshgrid(
        values,
        values,
        indexing="ij",
    )

    return (
        dx * dx
        + dy * dy
        <= radius * radius
    )


ALIGN_SHIFT_MASK = make_alignment_shift_mask()


class ReflectWindowSampler:
    """Exact numpy-reflect border semantics without padding whole images per tile.

    The previous extractor performed np.pad(full_slice, (47,47)) once for the
    center AND once for every neighbor, for EVERY tile. On large slices that
    copies hundreds of megabytes for a 320x320 matching window. Here interior
    windows are views; border windows use reflected index mapping with exactly
    the same indices as np.pad(..., mode="reflect").
    """
    def __init__(self, height, width):
        self.height = int(height)
        self.width = int(width)
        self.pad = ALIGN_CONTEXT_MARGIN + MAX_LOCAL_SHIFT_PX
        self.ys = np.pad(np.arange(height, dtype=np.int32),
                         (self.pad, self.pad), mode="reflect")
        self.xs = np.pad(np.arange(width, dtype=np.int32),
                         (self.pad, self.pad), mode="reflect")

    def window(self, image, y, x, size):
        if (0 <= y and 0 <= x and
                y + size <= self.height and x + size <= self.width):
            return image[y:y + size, x:x + size]
        iy = self.ys[y + self.pad:y + self.pad + size]
        ix = self.xs[x + self.pad:x + self.pad + size]
        if len(iy) != size or len(ix) != size:
            raise RuntimeError("Reflected alignment window exceeds validated pad.")
        return image[np.ix_(iy, ix)]


def _extract_alignment_reference_and_search(
    center_slice,
    neighbor_slice,
    tile_y,
    tile_x,
    sampler=None,
):
    """
    Extract:
      reference: ALIGN_CONTEXT_SIZE x ALIGN_CONTEXT_SIZE
      search:    reference size + 2*MAX_LOCAL_SHIFT_PX

    Reflect padding is used ONLY for estimating alignment near full-image
    borders. Actual model-input crops never use this padded array.
    """
    if (
        ALIGN_CONTEXT_SIZE
        < TILE_SIZE
    ):
        raise ValueError(
            "ALIGN_CONTEXT_SIZE must be >= TILE_SIZE."
        )

    if (
        ALIGN_CONTEXT_SIZE
        - TILE_SIZE
    ) % 2 != 0:
        raise ValueError(
            "ALIGN_CONTEXT_SIZE - TILE_SIZE must be even."
        )

    pad = (
        ALIGN_CONTEXT_MARGIN
        + MAX_LOCAL_SHIFT_PX
    )

    if sampler is not None:
        # Same 320/350 windows and reflect-border samples as legacy, with no
        # full-image allocation. The NCC / Gaussian algorithm is unchanged.
        top = tile_y - ALIGN_CONTEXT_MARGIN
        left = tile_x - ALIGN_CONTEXT_MARGIN
        reference = sampler.window(center_slice, top, left, ALIGN_CONTEXT_SIZE)
        search = sampler.window(neighbor_slice, top - MAX_LOCAL_SHIFT_PX,
                                left - MAX_LOCAL_SHIFT_PX,
                                ALIGN_CONTEXT_SIZE + 2 * MAX_LOCAL_SHIFT_PX)
        return (reference.astype(np.float32, copy=False),
                search.astype(np.float32, copy=False))

    center_padded = np.pad(
        center_slice,
        (
            (pad, pad),
            (pad, pad),
        ),
        mode="reflect",
    )

    neighbor_padded = np.pad(
        neighbor_slice,
        (
            (pad, pad),
            (pad, pad),
        ),
        mode="reflect",
    )

    ref_y = (
        tile_y
        + pad
        - ALIGN_CONTEXT_MARGIN
    )

    ref_x = (
        tile_x
        + pad
        - ALIGN_CONTEXT_MARGIN
    )

    reference = center_padded[
        ref_y:
        ref_y + ALIGN_CONTEXT_SIZE,
        ref_x:
        ref_x + ALIGN_CONTEXT_SIZE,
    ]

    search_y = (
        ref_y
        - MAX_LOCAL_SHIFT_PX
    )

    search_x = (
        ref_x
        - MAX_LOCAL_SHIFT_PX
    )

    search_size = (
        ALIGN_CONTEXT_SIZE
        + 2 * MAX_LOCAL_SHIFT_PX
    )

    search = neighbor_padded[
        search_y:
        search_y + search_size,
        search_x:
        search_x + search_size,
    ]

    return (
        reference.astype(
            np.float32,
            copy=False,
        ),
        search.astype(
            np.float32,
            copy=False,
        ),
    )


def _valid_shift_mask_for_output_crop(
    tile_y,
    tile_x,
    image_h,
    image_w,
):
    """
    Do not accept a translation if the final 256x256 neighbor crop would
    leave the real processed image.

    This prevents padding or interpolation from entering the actual model inputs.
    """
    radius = int(
        MAX_LOCAL_SHIFT_PX
    )

    values = np.arange(
        -radius,
        radius + 1,
        dtype=np.int32,
    )

    dy, dx = np.meshgrid(
        values,
        values,
        indexing="ij",
    )

    shifted_y = (
        tile_y
        + dy
    )

    shifted_x = (
        tile_x
        + dx
    )

    valid = (
        ALIGN_SHIFT_MASK
        & (shifted_y >= 0)
        & (shifted_x >= 0)
        & (
            shifted_y
            + TILE_SIZE
            <= image_h
        )
        & (
            shifted_x
            + TILE_SIZE
            <= image_w
        )
    )

    return valid


def estimate_local_translation(
    center_slice,
    neighbor_slice,
    tile_y,
    tile_x,
    sampler=None,
):
    """
    Estimate one integer translation (dx,dy) for one neighbor relative to
    the center tile.

    Positive dx/dy means:
        crop the neighbor from (x+dx, y+dy).

    The center-region matching configuration:
    - the larger 320x320 context is still extracted;
    - NCC is computed only on a CENTRAL ALIGN_MATCH_SIZE x ALIGN_MATCH_SIZE ROI;
    - the final model input is still a direct 256x256 crop from the original
      processed full slice.

    The CENTER slice itself is never shifted.
    """
    h, w = center_slice.shape

    (
        reference_context,
        search_context,
    ) = (
        _extract_alignment_reference_and_search(
            center_slice,
            neighbor_slice,
            tile_y,
            tile_x,
            sampler=sampler,
        )
    )

    if (
        ALIGN_MATCH_SIZE
        > ALIGN_CONTEXT_SIZE
    ):
        raise ValueError(
            "ALIGN_MATCH_SIZE must be <= ALIGN_CONTEXT_SIZE."
        )

    if (
        ALIGN_CONTEXT_SIZE
        - ALIGN_MATCH_SIZE
    ) % 2 != 0:
        raise ValueError(
            "ALIGN_CONTEXT_SIZE - ALIGN_MATCH_SIZE must be even."
        )

    # Central ROI inside the 320x320 reference context.
    ref_margin = (
        ALIGN_CONTEXT_SIZE
        - ALIGN_MATCH_SIZE
    ) // 2

    reference = reference_context[
        ref_margin:
        ref_margin + ALIGN_MATCH_SIZE,
        ref_margin:
        ref_margin + ALIGN_MATCH_SIZE,
    ]

    # search_context shape:
    #   ALIGN_CONTEXT_SIZE + 2*MAX_LOCAL_SHIFT_PX
    #
    # To search +/- MAX_LOCAL_SHIFT_PX for the CENTRAL ROI, crop a search
    # window centered on the same central location but with the full search
    # margin retained around it.
    search_start = ref_margin

    search_size = (
        ALIGN_MATCH_SIZE
        + 2 * MAX_LOCAL_SHIFT_PX
    )

    search = search_context[
        search_start:
        search_start + search_size,
        search_start:
        search_start + search_size,
    ]

    if (
        ALIGN_ESTIMATION_GAUSSIAN_SIGMA
        > 0
    ):
        reference_for_match = (
            ndimage.gaussian_filter(
                reference,
                sigma=(
                    ALIGN_ESTIMATION_GAUSSIAN_SIGMA
                ),
            ).astype(np.float32)
        )

        search_for_match = (
            ndimage.gaussian_filter(
                search,
                sigma=(
                    ALIGN_ESTIMATION_GAUSSIAN_SIGMA
                ),
            ).astype(np.float32)
        )
    else:
        reference_for_match = (
            reference
        )

        search_for_match = (
            search
        )

    score_map = cv2.matchTemplate(
        search_for_match,
        reference_for_match,
        cv2.TM_CCOEFF_NORMED,
    ).astype(np.float32)

    expected_size = (
        2 * MAX_LOCAL_SHIFT_PX
        + 1
    )

    if score_map.shape != (
        expected_size,
        expected_size,
    ):
        raise RuntimeError(
            "Unexpected NCC score-map shape: "
            f"{score_map.shape}; "
            f"expected "
            f"{(expected_size, expected_size)}."
        )

    valid = (
        _valid_shift_mask_for_output_crop(
            tile_y,
            tile_x,
            h,
            w,
        )
    )

    if not np.any(
        valid
    ):
        return {
            "dx": 0,
            "dy": 0,
            "accepted": False,
            "reason":
                "no_valid_candidate",
            "best_score":
                None,
            "zero_score":
                None,
            "improvement":
                None,
            "peak_margin":
                None,
        }

    masked_scores = np.where(
        valid,
        score_map,
        -np.inf,
    )

    best_flat = int(
        np.argmax(
            masked_scores
        )
    )

    best_row, best_col = (
        np.unravel_index(
            best_flat,
            masked_scores.shape,
        )
    )

    best_dy = int(
        best_row
        - MAX_LOCAL_SHIFT_PX
    )

    best_dx = int(
        best_col
        - MAX_LOCAL_SHIFT_PX
    )

    best_score = float(
        masked_scores[
            best_row,
            best_col,
        ]
    )

    zero_index = int(
        MAX_LOCAL_SHIFT_PX
    )

    zero_score = float(
        score_map[
            zero_index,
            zero_index,
        ]
    )

    improvement = float(
        best_score
        - zero_score
    )

    # Diagnostic second-peak score outside a small neighborhood around
    # the best shift.
    second_mask = (
        valid.copy()
    )

    y0 = max(
        0,
        best_row
        - ALIGN_PEAK_EXCLUSION_RADIUS,
    )

    y1 = min(
        second_mask.shape[0],
        best_row
        + ALIGN_PEAK_EXCLUSION_RADIUS
        + 1,
    )

    x0 = max(
        0,
        best_col
        - ALIGN_PEAK_EXCLUSION_RADIUS,
    )

    x1 = min(
        second_mask.shape[1],
        best_col
        + ALIGN_PEAK_EXCLUSION_RADIUS
        + 1,
    )

    second_mask[
        y0:y1,
        x0:x1,
    ] = False

    if np.any(
        second_mask
    ):
        second_score = float(
            np.max(
                np.where(
                    second_mask,
                    score_map,
                    -np.inf,
                )
            )
        )

        peak_margin = float(
            best_score
            - second_score
        )
    else:
        second_score = None
        peak_margin = None

    accepted = True
    reason = "accepted"

    if (
        best_dx == 0
        and best_dy == 0
    ):
        accepted = False
        reason = "zero_shift_is_best"

    elif (
        not np.isfinite(
            best_score
        )
        or best_score
        < ALIGN_MIN_BEST_NCC
    ):
        accepted = False
        reason = "best_ncc_too_low"

    elif (
        not np.isfinite(
            improvement
        )
        or improvement
        < ALIGN_MIN_NCC_IMPROVEMENT
    ):
        accepted = False
        reason = "ncc_improvement_too_small"

    elif (
        ALIGN_MIN_PEAK_MARGIN
        > 0
        and (
            peak_margin is None
            or peak_margin
            < ALIGN_MIN_PEAK_MARGIN
        )
    ):
        accepted = False
        reason = "peak_margin_too_small"

    if not accepted:
        final_dx = 0
        final_dy = 0
    else:
        final_dx = best_dx
        final_dy = best_dy

    return {
        "dx":
            int(
                final_dx
            ),
        "dy":
            int(
                final_dy
            ),
        "accepted":
            bool(
                accepted
            ),
        "reason":
            reason,
        "best_dx":
            int(
                best_dx
            ),
        "best_dy":
            int(
                best_dy
            ),
        "best_score":
            float(
                best_score
            ),
        "zero_score":
            float(
                zero_score
            ),
        "improvement":
            float(
                improvement
            ),
        "second_score":
            (
                None
                if second_score is None
                else float(
                    second_score
                )
            ),
        "peak_margin":
            (
                None
                if peak_margin is None
                else float(
                    peak_margin
                )
            ),
    }


def build_locally_aligned_tile(
    processed_stack,
    tile_y,
    tile_x,
    sampler=None,
):
    """
    Build one [5,256,256] five-slice refinement-model input.

    The center crop is always exactly:
        [tile_y:tile_y+256, tile_x:tile_x+256]

    Each neighbor is either:
      - cropped from a confidence-accepted translated location, or
      - left at the exact center coordinates when alignment is rejected.
    """
    _, h, w = (
        processed_stack.shape
    )

    center_slice = (
        processed_stack[
            CONTEXT_RADIUS
        ]
    )

    aligned_patches = []
    neighbor_records = []

    for zi in range(
        NUM_CONTEXT_SLICES
    ):
        if (
            zi
            == CONTEXT_RADIUS
        ):
            dx = 0
            dy = 0

            record = {
                "slice_offset":
                    0,
                "dx":
                    0,
                "dy":
                    0,
                "accepted":
                    True,
                "reason":
                    "center_reference",
            }

        elif not LOCAL_ALIGNMENT_ENABLED:
            dx = 0
            dy = 0

            record = {
                "slice_offset":
                    int(
                        zi
                        - CONTEXT_RADIUS
                    ),
                "dx":
                    0,
                "dy":
                    0,
                "accepted":
                    False,
                "reason":
                    "local_alignment_disabled",
            }

        else:
            result = estimate_local_translation(
                center_slice,
                processed_stack[
                    zi
                ],
                tile_y,
                tile_x,
                sampler=sampler,
            )

            dx = int(
                result["dx"]
            )

            dy = int(
                result["dy"]
            )

            record = {
                "slice_offset":
                    int(
                        zi
                        - CONTEXT_RADIUS
                    ),
                **result,
            }

        crop_y = (
            tile_y
            + dy
        )

        crop_x = (
            tile_x
            + dx
        )

        patch = processed_stack[
            zi,
            crop_y:
            crop_y + TILE_SIZE,
            crop_x:
            crop_x + TILE_SIZE,
        ]

        if patch.shape != (
            TILE_SIZE,
            TILE_SIZE,
        ):
            raise RuntimeError(
                "Local-alignment crop has wrong shape: "
                f"slice={zi}, "
                f"tile={(tile_y, tile_x)}, "
                f"shift={(dy, dx)}, "
                f"shape={patch.shape}."
            )

        aligned_patches.append(
            patch
        )

        neighbor_records.append(
            record
        )

    tile_record = {
        "tile_y":
            int(
                tile_y
            ),
        "tile_x":
            int(
                tile_x
            ),
        "neighbors":
            neighbor_records,
    }

    return (
        np.stack(
            aligned_patches,
            axis=0,
        ).astype(np.float32),
        tile_record,
    )


def summarize_alignment_records(
    tile_records,
):
    """Aggregate accepted and rejected local neighbor translations over tiles.
    
        Args: tile_records: Per-tile NCC alignment diagnostics.
        Returns: Counts, acceptance fraction and accepted shift magnitude summary."""
    neighbor_records = []

    for tile in tile_records:
        for neighbor in tile[
            "neighbors"
        ]:
            if (
                neighbor[
                    "slice_offset"
                ]
                != 0
            ):
                neighbor_records.append(
                    neighbor
                )

    if not neighbor_records:
        return {
            "num_neighbor_attempts":
                0,
            "num_accepted":
                0,
            "acceptance_rate":
                0.0,
            "mean_accepted_shift_magnitude":
                0.0,
            "max_accepted_shift_magnitude":
                0.0,
        }

    accepted = [
        item
        for item in neighbor_records
        if item[
            "accepted"
        ]
    ]

    magnitudes = [
        math.sqrt(
            float(
                item["dx"]
            ) ** 2
            + float(
                item["dy"]
            ) ** 2
        )
        for item in accepted
    ]

    return {
        "num_neighbor_attempts":
            int(
                len(
                    neighbor_records
                )
            ),
        "num_accepted":
            int(
                len(
                    accepted
                )
            ),
        "acceptance_rate":
            float(
                len(
                    accepted
                )
                / max(
                    len(
                        neighbor_records
                    ),
                    1,
                )
            ),
        "mean_accepted_shift_magnitude":
            (
                0.0
                if not magnitudes
                else float(
                    np.mean(
                        magnitudes
                    )
                )
            ),
        "max_accepted_shift_magnitude":
            (
                0.0
                if not magnitudes
                else float(
                    np.max(
                        magnitudes
                    )
                )
            ),
    }



# ========================================================================
# 12. Overlapping 2.5D center-slice refinement inference
# ========================================================================
# 256px tiles, 128px overlap and original clipped Hann blend.

def get_positions(
    length,
):
    """List inference tile starts, including the rightmost/bottommost border.
    
        Args: length: Height or width of one output slice.
        Returns: Ordered 256px tile origins using the original 128px stride."""
    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - TILE_SIZE + 1,
            TILE_STRIDE,
        )
    )

    last = (
        length
        - TILE_SIZE
    )

    if positions[-1] != last:
        positions.append(
            last
        )

    return positions


def make_blend_weight():
    """Construct the validated clipped Hann blend weights for tile predictions.
    
        Returns: float32 [256,256] weight map with nonzero border weights."""
    one_d = np.hanning(
        TILE_SIZE
    ).astype(np.float32)

    weight = np.outer(
        one_d,
        one_d,
    ).astype(np.float32)

    # Never allow zero denominator at outer image borders.
    return np.maximum(
        weight,
        BLEND_EPS,
    )


class ExecutionPlan:
    """Hardware-aware execution settings. Never alters scientific constants."""
    def __init__(self, device, args):
        self.device = device
        self.enabled = args.execution == "auto"
        self.batch_size = TILE_BATCH
        self.max_batch = args.max_batch
        self.override_batch = args.batch_size
        self.tuned = False
        self.benchmarks = []
        if args.prepare_workers is not None:
            self.prepare_workers = args.prepare_workers
        elif device.type == "cuda":
            # Bounded overlap between CPU alignment and CUDA model forward.
            self.prepare_workers = min(2, max(1, len(os.sched_getaffinity(0)) - 1)) \
                if hasattr(os, "sched_getaffinity") else 2
        elif device.type == "mps":
            self.prepare_workers = 1
        else:
            self.prepare_workers = 0
        if not self.enabled:
            self.prepare_workers = 0
        if self.enabled and self.override_batch is None and device.type == "mps":
            # Apple Silicon has unified memory: leave headroom for whole-slice
            # SciPy preprocessing and alignment; safe fallback remains active.
            try:
                recommended = int(torch.mps.recommended_max_memory())
                allocated = int(torch.mps.driver_allocated_memory())
                free_budget = max(0, recommended - allocated)
                if free_budget < 3 * (1024 ** 3):
                    self.batch_size = 1
                elif free_budget < 6 * (1024 ** 3):
                    self.batch_size = 2
                else:
                    self.batch_size = TILE_BATCH
            except (AttributeError, RuntimeError):
                self.batch_size = TILE_BATCH
        elif self.enabled and device.type == "cpu" and self.override_batch is None:
            self.batch_size = 1
        elif self.override_batch is not None:
            self.batch_size = self.override_batch

        if self.enabled:
            self.batch_size = min(self.batch_size, self.max_batch)

    def describe(self):
        return {"execution": "auto" if self.enabled else "legacy",
                "device": str(self.device), "batch_size": self.batch_size,
                "prepare_workers": self.prepare_workers,
                "cuda_batch_benchmarks": self.benchmarks}


def _device_oom(exc, device):
    """Only retry genuine device memory errors; never swallow unrelated failures."""
    if device.type not in ("cuda", "mps"):
        return False
    if device.type == "cuda" and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in message or "mps backend out of memory" in message)


def _release_device_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _forward_batch(model, patches, device):
    tensor = torch.from_numpy(patches).to(
        device=device, dtype=torch.float32,
        non_blocking=(device.type == "cuda"))
    return model(tensor)[:, 0].detach().cpu().numpy()


@torch.inference_mode()
def _autotune_cuda_batch(model, processed_stack, plan):
    """Tiny real-input inference-only calibration. No weight/state updates.

    Select based on actual throughput, not GPU name or blindly assuming that
    all available VRAM should be filled. OOM during calibration is recoverable;
    no output tiles have been written at this point.
    """
    if plan.tuned or plan.device.type != "cuda":
        return
    plan.tuned = True
    if plan.override_batch is not None:
        return
    tile = np.ascontiguousarray(processed_stack[:, :TILE_SIZE, :TILE_SIZE])
    candidates = [n for n in (1, 2, 4, 8, 16, 32) if n <= plan.max_batch]
    results = []
    for candidate in candidates:
        sample = np.repeat(tile[None, ...], candidate, axis=0)
        try:
            # Keep the precision state owned by the caller (AMP / FP32).
            for _ in range(1):
                _forward_batch(model, sample, plan.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(2):
                _forward_batch(model, sample, plan.device)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - start) / 2.0
            throughput = candidate / max(dt, 1e-6)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            result = {"batch": candidate, "seconds": round(dt, 4),
                      "tiles_per_second": round(throughput, 2),
                      "free_vram_gb": round(free_bytes / (1024 ** 3), 2)}
            results.append(result)
            # Avoid aggressively consuming already tight available VRAM.
            if free_bytes < max(512 * 1024**2, total_bytes * 0.10):
                break
        except (RuntimeError, MemoryError) as exc:
            if not _device_oom(exc, plan.device):
                raise
            print(f"[Adaptive] Calibration batch {candidate} OOM, stopping growth.")
            _release_device_cache(plan.device)
            break
        finally:
            del sample
    if results:
        best = max(item["tiles_per_second"] for item in results)
        # The smallest batch within 95% of best CUDA throughput avoids excess
        # memory use on shared GPUs without compromising much throughput.
        plan.batch_size = next(item["batch"] for item in results
                               if item["tiles_per_second"] >= 0.95 * best)
    plan.benchmarks = results
    print(f"[Adaptive] CUDA batch={plan.batch_size}; calibration={results}", flush=True)


def _prepare_aligned_batch(processed_stack, batch_coords, sampler):
    start = time.perf_counter()
    tiles = [build_locally_aligned_tile(processed_stack, y, x, sampler=sampler)[0]
             for y, x in batch_coords]
    patches = np.stack(tiles, axis=0).astype(np.float32, copy=False)
    return patches, time.perf_counter() - start


def _prepared_batches(processed_stack, coords, batch_size, sampler, workers):
    """Ordered, bounded CPU prefetch; no skip/reorder of output tile positions."""
    grouped = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]
    if workers == 0:
        for positions in grouped:
            patches, prep_seconds = _prepare_aligned_batch(processed_stack, positions, sampler)
            yield positions, patches, prep_seconds
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        cursor = 0
        limit = min(workers + 1, 4)  # memory-bounded in-flight batches
        while cursor < len(grouped) and len(pending) < limit:
            pending.append((grouped[cursor], pool.submit(
                _prepare_aligned_batch, processed_stack, grouped[cursor], sampler)))
            cursor += 1
        while pending:
            positions, future = pending.popleft()
            if cursor < len(grouped):
                pending.append((grouped[cursor], pool.submit(
                    _prepare_aligned_batch, processed_stack, grouped[cursor], sampler)))
                cursor += 1
            patches, prep_seconds = future.result()
            yield positions, patches, prep_seconds


def infer_processed_stack_adaptive(model, processed_stack, device, plan, timing=None):
    """Same tile math, reflect pixels, and sequential Hann accumulation as legacy.

    Only CPU sampling, ordered prefetch, and inference batch scheduling change.
    """
    if processed_stack.ndim != 3 or processed_stack.shape[0] != NUM_CONTEXT_SLICES:
        raise ValueError(f"Expected [5,H,W], got {processed_stack.shape}")
    _, h, w = processed_stack.shape
    if min(h, w) < TILE_SIZE:
        raise ValueError("Input slice smaller than 256x256 tile")
    ys, xs = get_positions(h), get_positions(w)
    coords = [(y, x) for y in ys for x in xs]
    blend = make_blend_weight()
    output_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    sampler = ReflectWindowSampler(h, w)
    model.eval()
    if device.type == "cuda":
        _autotune_cuda_batch(model, processed_stack, plan)

    # Fallback only retries uncommitted model batches: CPU blending is done
    # strictly after a successful forward, with each tile accumulated once.
    batch_size = int(plan.batch_size)
    if batch_size < 1:
        raise ValueError("Execution batch must be positive")
    batch_preparation = 0.0
    batch_wait = 0.0
    device_seconds = 0.0
    blend_seconds = 0.0
    oom_retries = 0
    with torch.inference_mode():
        for positions, patches, prep_time in _prepared_batches(
                processed_stack, coords, batch_size, sampler, plan.prepare_workers):
            batch_preparation += prep_time
            offset = 0
            current_batch = min(batch_size, plan.batch_size)
            while offset < len(positions):
                chunk_coords = positions[offset:offset + current_batch]
                chunk_patches = patches[offset:offset + current_batch]
                try:
                    start = time.perf_counter()
                    pred = _forward_batch(model, chunk_patches, device)
                    device_seconds += time.perf_counter() - start
                except (RuntimeError, MemoryError) as exc:
                    if not _device_oom(exc, device):
                        raise
                    if current_batch == 1:
                        raise RuntimeError("Even one 256px tile cannot fit device memory; "
                                           "reduce competing memory use.") from exc
                    oom_retries += 1
                    current_batch = max(1, current_batch // 2)
                    plan.batch_size = min(plan.batch_size, current_batch)
                    _release_device_cache(device)
                    print(f"[Adaptive] {device.type.upper()} OOM: retry pending tiles "
                          f"at batch={current_batch}", flush=True)
                    continue
                start = time.perf_counter()
                for (y, x), patch_pred in zip(chunk_coords, pred):
                    output_sum[y:y + TILE_SIZE, x:x + TILE_SIZE] += patch_pred * blend
                    weight_sum[y:y + TILE_SIZE, x:x + TILE_SIZE] += blend
                blend_seconds += time.perf_counter() - start
                offset += len(chunk_coords)
            del patches
    predicted = (output_sum / np.maximum(weight_sum, BLEND_EPS)).astype(np.float32)
    if timing is not None:
        timing.update({"aligned_tiles": len(coords), "patch_preparation_cpu_seconds": batch_preparation,
                       "model_transfer_wall_seconds": device_seconds,
                       "blend_wall_seconds": blend_seconds, "oom_retries": oom_retries,
                       "effective_batch_size": plan.batch_size})
    return predicted, {"enabled": LOCAL_ALIGNMENT_ENABLED,
                       "tile_count": len(coords), "tile_size": TILE_SIZE,
                       "summary": "Diagnostics not retained in adaptive execution to reduce memory"}


def infer_processed_stack(
    model,
    processed_stack,
    device,
):
    """
    processed_stack:
        [5,H,W]

    returns:
        predicted center representation [H,W]
        alignment diagnostic dict

    Important:
    - Local alignment changes only where neighbor crops are sampled.
    - The center crop and output placement are never shifted.
    - Model architecture/checkpoint are unchanged.
    """
    if (
        processed_stack.ndim != 3
        or processed_stack.shape[0]
        != NUM_CONTEXT_SLICES
    ):
        raise ValueError(
            f"Expected [5,H,W], "
            f"got {processed_stack.shape}"
        )

    _, h, w = (
        processed_stack.shape
    )

    if (
        h < TILE_SIZE
        or w < TILE_SIZE
    ):
        raise ValueError(
            f"Input image {h}x{w} "
            f"is smaller than TILE_SIZE={TILE_SIZE}."
        )

    ys = get_positions(
        h
    )

    xs = get_positions(
        w
    )

    coords = [
        (y, x)
        for y in ys
        for x in xs
    ]

    blend_weight = (
        make_blend_weight()
    )

    output_sum = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    alignment_records = []

    model.eval()

    for start in range(
        0,
        len(coords),
        TILE_BATCH,
    ):
        batch_coords = coords[
            start:
            start + TILE_BATCH
        ]

        batch_patches = []
        batch_records = []

        for y, x in batch_coords:
            (
                aligned_patch,
                tile_record,
            ) = build_locally_aligned_tile(
                processed_stack,
                y,
                x,
            )

            batch_patches.append(
                aligned_patch
            )

            batch_records.append(
                tile_record
            )

        patches = np.stack(
            batch_patches,
            axis=0,
        ).astype(np.float32)

        tensor = torch.from_numpy(
            patches
        ).to(
            device=device,
            dtype=torch.float32,
        )

        pred = model(
            tensor
        )[
            :,
            0,
        ].detach().cpu().numpy()

        alignment_records.extend(
            batch_records
        )

        for (
            (y, x),
            patch_pred,
        ) in zip(
            batch_coords,
            pred,
        ):
            output_sum[
                y:
                y + TILE_SIZE,
                x:
                x + TILE_SIZE,
            ] += (
                patch_pred
                * blend_weight
            )

            weight_sum[
                y:
                y + TILE_SIZE,
                x:
                x + TILE_SIZE,
            ] += blend_weight

    predicted = (
        output_sum
        / np.maximum(
            weight_sum,
            BLEND_EPS,
        )
    ).astype(np.float32)

    diagnostic = {
        "enabled":
            bool(
                LOCAL_ALIGNMENT_ENABLED
            ),
        "tile_size":
            int(
                TILE_SIZE
            ),
        "tile_overlap":
            int(
                TILE_OVERLAP
            ),
        "tile_stride":
            int(
                TILE_STRIDE
            ),
        "alignment_context_size":
            int(
                ALIGN_CONTEXT_SIZE
            ),
        "alignment_match_size":
            int(
                ALIGN_MATCH_SIZE
            ),
        "max_local_shift_px":
            int(
                MAX_LOCAL_SHIFT_PX
            ),
        "estimation_gaussian_sigma":
            float(
                ALIGN_ESTIMATION_GAUSSIAN_SIGMA
            ),
        "min_best_ncc":
            float(
                ALIGN_MIN_BEST_NCC
            ),
        "min_ncc_improvement":
            float(
                ALIGN_MIN_NCC_IMPROVEMENT
            ),
        "min_peak_margin":
            float(
                ALIGN_MIN_PEAK_MARGIN
            ),
        "summary":
            summarize_alignment_records(
                alignment_records
            ),
        "tiles":
            alignment_records,
    }

    return (
        predicted,
        diagnostic,
    )



# ========================================================================
# 13. Five evenly spaced four-panel visualizations
# ========================================================================
# These panels are display-only and cannot feed back to model input.

def add_title(image, title):
    """Add a compact header while preserving the image at original H x W."""
    if image.dtype != np.uint8:
        raise ValueError(f"Visualization expects uint8, got {image.dtype}")

    header = np.full(
        (VIS_HEADER_HEIGHT, image.shape[1]),
        255,
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        title,
        (VIS_TEXT_X, VIS_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        VIS_FONT_SCALE,
        0,
        VIS_FONT_THICKNESS,
        cv2.LINE_AA,
    )

    return np.concatenate([header, image], axis=0)


def _join_panels(panels):
    """Join titled grayscale panels with the preserved separators/borders.
    
        Args: panels: Equally sized uint8 grayscale images.
        Returns: Horizontally concatenated comparison image."""
    bordered = [
        _add_black_border_uint8(panel, 1)
        for panel in panels
    ]
    separator = np.zeros(
        (
            bordered[0].shape[0],
            VIS_SEPARATOR_WIDTH,
        ),
        dtype=np.uint8,
    )

    pieces = []
    for i, panel in enumerate(bordered):
        if i > 0:
            pieces.append(separator)
        pieces.append(panel)

    return np.concatenate(
        pieces,
        axis=1,
    )


def make_visualization(stage1_input, processed_input, output_image):
    """Four diagnostic panels. Visualization only; never used by the model."""
    if stage1_input.shape != processed_input.shape or stage1_input.shape != output_image.shape:
        raise ValueError("Visualization requires matching YX shapes.")
    correction_display = to_uint8(
        128.0 + CORRECTION_PREVIEW_GAIN * (
            output_image.astype(np.float32) - processed_input.astype(np.float32)
        )
    )
    return _join_panels([
        add_title(stage1_input, "Stage-1 output"),
        add_title(processed_input, "Processed model input"),
        add_title(output_image, "Refined output"),
        add_title(correction_display, "Model correction x1"),
    ])


def _add_black_border_uint8(image, border):
    """Add an unscaled solid black border to a grayscale uint8 preview."""
    if border <= 0:
        return image
    return cv2.copyMakeBorder(
        image,
        border,
        border,
        border,
        border,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )



# ========================================================================
# 14. Strict grayscale uint8 stacked-TIFF input handling
# ========================================================================
# Preserve the original ZYX order and read slices lazily.

def inspect_input_stack(path):
    """Validate a uint8 stack without loading all slice data into RAM.

    Most ordinary TIFF stacks store one grayscale 2D frame per physical page.
    For a single physical page containing a 3D ZYX array, allow an uncompressed
    memory-mappable volume with explicit ZYX axis metadata.
    """
    with tifffile.TiffFile(str(path)) as tif:
        if not tif.pages:
            raise ValueError(f"Empty TIFF stack: {path}")

        # Standard multipage TIFF: one 2D grayscale image per page.
        page = tif.pages[0]
        if len(page.shape) == 2:
            shape = tuple(int(s) for s in page.shape)
            for z, current in enumerate(tif.pages):
                if tuple(current.shape) != shape:
                    raise ValueError(
                        f"Inconsistent TIFF page shape at z={z}: "
                        f"{current.shape}, expected {shape}"
                    )
                if np.dtype(current.dtype) != np.dtype("uint8"):
                    raise ValueError(
                        f"Expected uint8 grayscale slices; z={z} has dtype {current.dtype}"
                    )
                if int(current.samplesperpixel) != 1:
                    raise ValueError(f"Expected single-channel TIFF; z={z} has multiple samples")
            return len(tif.pages), shape, "pages"

        # Rare TIFFs encode a ZYX volume as one physical page.
        if len(tif.pages) == 1 and len(page.shape) == 3:
            series = tif.series[0]
            if tuple(series.shape) != tuple(page.shape) or series.axes != "ZYX":
                raise ValueError(
                    "Single-page 3D TIFF requires explicit ZYX axes metadata. "
                    f"Found shape={series.shape}, axes={series.axes}."
                )
            if np.dtype(page.dtype) != np.dtype("uint8") or int(page.samplesperpixel) != 1:
                raise ValueError("Expected grayscale uint8 TIFF stack.")
            try:
                mapped = tifffile.memmap(str(path), series=0, mode="r")
                if mapped.ndim != 3 or mapped.shape != tuple(series.shape):
                    raise ValueError("Invalid memory-mapped ZYX TIFF volume.")
            except Exception as exc:
                raise ValueError(
                    "This single-page 3D TIFF cannot be read slice-by-slice. "
                    "Provide a standard multipage TIFF or an uncompressed ZYX TIFF."
                ) from exc
            return int(series.shape[0]), tuple(int(s) for s in series.shape[1:]), "memmap"

        raise ValueError(
            "Expected a grayscale uint8 stacked TIFF with one 2D image per "
            "page or an explicitly labeled ZYX volume. "
            f"First page shape: {page.shape}"
        )


def select_visualization_indices(z_count):
    """Choose up to five unique evenly spaced Z indices including both ends."""
    if z_count <= 0:
        raise ValueError("Stack must have at least one slice.")
    if z_count <= 5:
        return list(range(z_count))
    return [int(round(i * (z_count - 1) / 4)) for i in range(5)]


class InputStackSlice:
    """View of an input stack page consumed by original per-slice algorithms."""
    def __init__(self, reader, index):
        """Bind one logical Z slice of an open stacked TIFF reader."""
        self.reader = reader
        self.index = int(index)
        self.name = f"slice_{index:04d}.tif"


class InputStackReader:
    """Read the Stage-1 stacked TIFF lazily without loading the volume into RAM.
    
        Multipage TIFF pages are read individually; memory mapping is preferred
        for uncompressed stacks when supported by tifffile."""
    def __init__(self, path, stack_info):
        """Store the stack shape and selected page-based/memmap reader mode."""
        self.path = Path(path)
        self.z_count, self.yx_shape, self.mode = stack_info
        self.tif = None
        self.array = None

    def __enter__(self):
        """Open the TIFF and optional memory-mapped array for sequential access."""
        self.tif = tifffile.TiffFile(str(self.path))
        if self.mode == "memmap":
            self.array = tifffile.memmap(str(self.path), series=0, mode="r")
        elif self.mode == "pages":
            # Prefer disk-backed access to the uncompressed Stage-1 BigTIFF.
            try:
                mapped = tifffile.memmap(str(self.path), series=0, mode="r")
                if mapped.shape == (self.z_count, *self.yx_shape):
                    self.array = mapped
            except (ValueError, OSError, TypeError):
                pass
        return self

    def get(self, index):
        """Load or view one uint8 YX slice from the selected Z index.
        
            Returns: 2D uint8 array in the original Stage-1 coordinate system."""
        if not 0 <= index < self.z_count:
            raise IndexError(index)
        image = np.asarray(self.array[index]) if self.array is not None else self.tif.pages[index].asarray()
        if image.shape != self.yx_shape or image.dtype != np.uint8:
            raise ValueError(f"Invalid uint8 slice z={index}: {image.shape}/{image.dtype}")
        return image

    def __exit__(self, exc_type, exc, tb):
        """Close the TIFF handle and release mapped-array references."""
        self.array = None
        if self.tif is not None:
            self.tif.close()
        self.tif = None



# ========================================================================
# 15. Cross-device execution and checkpoint loading
# ========================================================================
# Load checkpoint tensors onto CPU, then move the model to the selected device.

def resolve_device(selected):
    """Choose CUDA, MPS, or CPU; reject explicitly unavailable devices.
    
        Args: selected: One of auto/cuda/mps/cpu.
        Returns: Available torch.device; no silent fallback when explicitly set."""
    if selected == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; no CPU fallback.")
    if selected == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was selected but is unavailable; no CPU fallback.")
    return torch.device(selected)


def load_checkpoint(model, path):
    """Load cross-backend weights on CPU; validate scientific configuration."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"Cannot load PyTorch weights safely from {path}") from error
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dictionary or standalone state_dict.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Model weights are missing or invalid.")
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    actual_count = sum(p.numel() for p in model.parameters())
    if actual_count != 2204289:
        raise RuntimeError(f"Five-slice model parameter-count mismatch: {actual_count}")
    declared_count = checkpoint.get("parameter_count")
    if declared_count is not None and int(declared_count) != actual_count:
        raise RuntimeError("Checkpoint parameter_count does not match the five-slice model.")
    settings = checkpoint.get("settings", {})
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise TypeError("Checkpoint settings metadata must be a dict.")
    for key, expected in (("num_context_slices", NUM_CONTEXT_SLICES),
                          ("patch_size", TILE_SIZE), ("dropout", DROPOUT)):
        if key in settings and settings[key] != expected:
            raise ValueError(f"Checkpoint training setting {key} mismatch: {settings[key]} != {expected}")
    # Check both archived and new metadata spelling without retaining any
    # personal-name-specific keys in the released inference implementation.
    for key, value in settings.items():
        if key.endswith("gaussian_sigma") and key != "hpf_sigma":
            if float(value) != PREPROCESS_GAUSSIAN_SIGMA:
                raise ValueError(f"Checkpoint Gaussian preprocessing mismatch: {key}={value}")
        if key.endswith("hist_bins") and int(value) != PREPROCESS_HIST_BINS:
            raise ValueError(f"Checkpoint histogram preprocessing mismatch: {key}={value}")
    return checkpoint



# ========================================================================
# 16. Filesystem safety and explicit overwrite
# ========================================================================
# No output or domain cache may overwrite frozen master data or input.

def protect_output_location(input_path, checkpoint_path, output_dir,
                            reference_train=None, cache_dir=None):
    """Reject destructive output overlaps without requiring unused references."""
    out = output_dir.resolve()
    for name, protected in (("Stage-1 stack", input_path.resolve()),
                            ("checkpoint", checkpoint_path.resolve())):
        if out == protected or out in protected.parents:
            raise ValueError(f"Output directory contains {name}: {protected}")
    if reference_train is not None:
        protected_dir = reference_train.resolve()
        if (out == protected_dir or out in protected_dir.parents
                or protected_dir in out.parents):
            raise ValueError(f"Output directory overlaps clean train references: {protected_dir}")
    if cache_dir is not None:
        protected_dir = cache_dir.resolve()
        if (out == protected_dir or out in protected_dir.parents
                or protected_dir in out.parents):
            raise ValueError(f"Output directory overlaps domain reference cache: {protected_dir}")
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError("Refusing to write to a symlink/non-directory output path.")

def protect_cache_location(input_path, checkpoint_path, reference_train, cache_dir):
    """Reference-statistics cache must never modify the original source tree."""
    cache = cache_dir.resolve()
    source_files = (("input stack", input_path.resolve()),
                    ("model checkpoint", checkpoint_path.resolve()))
    for label, file in source_files:
        if cache == file or cache in file.parents:
            raise ValueError(f"Domain cache would contain {label}: {file}")
    train = reference_train.resolve()
    if cache == train or cache in train.parents or train in cache.parents:
        raise ValueError(f"Domain cache overlaps frozen reference training images: {train}")
    if cache_dir.is_symlink() or (cache_dir.exists() and not cache_dir.is_dir()):
        raise ValueError("Refusing to use a symlink or non-directory cache path.")


def prepare_output(output_dir, overwrite):
    """Create only the final result directory and visualization subfolder.
    
        Existing output directories require --overwrite and are protected against
        symlink replacement and accidental deletion of protected inputs."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; use --overwrite to replace it.")
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("Refusing to overwrite symlink or non-directory output.")
        shutil.rmtree(output_dir)
    (output_dir / "visualization").mkdir(parents=True, exist_ok=True)


def create_output_stack(path, z_count, shape_yx):
    """Preallocate an uint8 ZYX BigTIFF stack on disk for incremental writes.
    
        Args: path: Temporary output TIFF; z_count: number of slices;
            shape_yx: image height and width.
        Returns: Writable tifffile memory-map array."""
    return tifffile.memmap(str(path), shape=(int(z_count), *shape_yx),
                          dtype=np.uint8, bigtiff=True,
                          photometric="minisblack", metadata={"axes": "ZYX"})



# ========================================================================
# 17. Command-line interface and run settings
# ========================================================================
# Documented --domain-transfer / --no-domain-transfer flags are exclusive.

def parse_args():
    """Define the documented CLI with mutually exclusive domain-transfer modes.
    
        Returns: argparse.Namespace containing input, checkpoint, paths, device,
            precision, overwrite/check-only flags and transfer selection."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Stage-1 output_stack.tif (uint8 ZYX).")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Five-slice model best.pt or final.pt.")
    domain = parser.add_mutually_exclusive_group()
    domain.add_argument("--domain-transfer", dest="domain_transfer", action="store_true",
                        help="Enable FFT and spatial CDF domain matching (default).")
    domain.add_argument("--no-domain-transfer", dest="domain_transfer", action="store_false",
                        help="Use direct full-slice processed model inputs; skip domain references and cache.")
    parser.set_defaults(domain_transfer=True)
    parser.add_argument("--reference-train", type=Path,
                        default=Path("data/clean_blocks_1024_z16/train"),
                        help="Clean TRAIN reference required only with --domain-transfer.")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("experiments/stage2_domain_cache"),
                        help="Domain statistics cache (only used with --domain-transfer).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--execution", choices=("auto", "legacy"), default="auto",
                        help="Auto optimizes memory/throughput; legacy uses original scheduling.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override inference batch (default: automatic selection).")
    parser.add_argument("--max-batch", type=int, default=32,
                        help="Maximum CUDA autotuning batch (default 32).")
    parser.add_argument("--prepare-workers", type=int, default=None,
                        help="CPU alignment prefetch workers; auto if not specified.")
    parser.add_argument("--timing-json", type=Path, default=None,
                        help="Optional timing JSON path OUTSIDE the result folder.")
    parser.add_argument("--cuda-amp", action="store_true",
                        help="Opt-in float16 autocast on CUDA.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing output folder.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate selected mode, inputs and checkpoint; write nothing.")
    return parser.parse_args()


# ========================================================================
# 18. Complete stack inference and output writing
# ========================================================================
# The final result directory contains ONLY output_stack.tif and
# visualization/ with up to five PNGs.

def run_stack(reader, model, device, output_dir, domain_reference, cuda_amp,
              plan=None, timings=None):
    """Run five-slice refinement and write one final TIFF and up to five previews.
    
        Reads full-slice processed data from an LRU cache, assembles the original
        locally aligned 256px overlapping tiles and applies the fixed uint8 decoding.
        Preview images do not participate in model inference. The temporary output
        TIFF is renamed into place only after successful processing.
    
        Args: reader: Lazy uint8 Stage-1 TIFF reader; model: loaded refinement network;
            device: selected PyTorch backend; output_dir: final output directory;
            domain_reference: transfer maps or None; cuda_amp: optional FP16 mode.
        Returns: (output TIFF path, number of visualization PNG files)."""
    z_count, shape_yx = reader.z_count, reader.yx_shape
    sources = [InputStackSlice(reader, z) for z in range(z_count)]
    visualization_indices = set(select_visualization_indices(z_count))
    pending = output_dir / "output_stack.partial.tif"
    dest = output_dir / "output_stack.tif"
    stack = None
    try:
        stack = create_output_stack(pending, z_count, shape_yx)
        cache_size = NUM_CONTEXT_SLICES if plan is not None and plan.enabled else PREPROCESS_CACHE_SIZE
        cache = SlicePreprocessCache(sources, cache_size, domain_reference)
        cache.drop_unused_lowpass = bool(plan is not None and plan.enabled)
        try:
            for z in tqdm(range(z_count), desc=("2.5D refinement with domain transfer"
                                               if domain_reference is not None else "2.5D refinement without domain transfer"),
                          unit="slice"):
                t_slice = time.perf_counter()
                t_preprocess = time.perf_counter()
                indices = context_indices(z, z_count)
                entries = [cache.get(i) for i in indices]
                processed_stack = np.stack([e["processed"] for e in entries], axis=0).astype(np.float32)
                preprocess_seconds = time.perf_counter() - t_preprocess
                tile_timing = {}
                amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                           if cuda_amp else nullcontext())
                with amp_ctx:
                    if plan is not None and plan.enabled:
                        predicted, _ = infer_processed_stack_adaptive(
                            model, processed_stack, device, plan, timing=tile_timing)
                    else:
                        predicted, _ = infer_processed_stack(model, processed_stack, device)
                result = decode_canonical_membrane(predicted)
                stack[z] = result
                del processed_stack, predicted
                if z in visualization_indices:
                    center = entries[CONTEXT_RADIUS]
                    actual_input = decode_canonical_membrane(center["processed"])
                    panel = make_visualization(center["raw"], actual_input, result)
                    png = output_dir / "visualization" / f"slice_{z:04d}.png"
                    if not cv2.imwrite(str(png), panel):
                        raise IOError(f"Cannot write visualization: {png}")
                elapsed = time.perf_counter() - t_slice
                if timings is not None:
                    timings["inference_slices"].append({
                        "slice": z, "wall_seconds": round(elapsed, 4),
                        "full_slice_preprocess_seconds": round(preprocess_seconds, 4),
                        **tile_timing})
                summary = (f"[Timing] slice={z + 1}/{z_count} total={elapsed:.2f}s "
                           f"preprocess={preprocess_seconds:.2f}s")
                if tile_timing:
                    summary += (f" model_and_transfers={tile_timing['model_transfer_wall_seconds']:.2f}s "
                                f"alignment_CPU_sum={tile_timing['patch_preparation_cpu_seconds']:.2f}s "
                                f"tiles={tile_timing['aligned_tiles']}")
                else:
                    summary += " execution=legacy"
                print(summary, flush=True)
        finally:
            cache.clear()
        stack.flush()
        del stack
        stack = None
        pending.replace(dest)
    finally:
        if stack is not None:
            stack.flush()
            del stack
        if pending.exists():
            pending.unlink()
    return dest, len(visualization_indices)



# ========================================================================
# 19. Entrypoint
# ========================================================================
# Validate first; only enabled domain transfer reads reference/cache.

def main():
    """Validate CLI settings, select optional domain transfer, and run inference.
    
        Enforces input/checkpoint/reference/cache separation, validates stack
        geometry, preflights the five-slice model, and prints the final output location."""
    args = parse_args()
    if args.max_batch < 1 or args.max_batch > 128:
        raise ValueError("--max-batch must be within 1..128")
    if args.batch_size is not None and not (1 <= args.batch_size <= args.max_batch):
        raise ValueError("--batch-size must be between 1 and --max-batch")
    if args.prepare_workers is not None and not (0 <= args.prepare_workers <= 16):
        raise ValueError("--prepare-workers must be within 0..16")
    if args.execution == "legacy" and (args.batch_size is not None or
                                       args.prepare_workers is not None):
        raise ValueError("--execution legacy cannot use adaptive batch/workers overrides")
    t_whole_start = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().absolute()
    if not input_path.is_file() or input_path.suffix.lower() not in (".tif", ".tiff"):
        raise FileNotFoundError(f"Stage-1 stacked TIFF missing: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Refinement-model checkpoint missing: {checkpoint_path}")

    # Only domain-transfer mode requires or even resolves reference/cache paths.
    reference_train = None
    cache_dir = None
    if args.domain_transfer:
        reference_train = args.reference_train.expanduser().resolve()
        cache_dir = args.cache_dir.expanduser().absolute()
        if not reference_train.is_dir():
            raise FileNotFoundError(f"Clean training-reference directory missing: {reference_train}")

    protect_output_location(input_path, checkpoint_path, output_dir,
                            reference_train=reference_train, cache_dir=cache_dir)
    if args.domain_transfer:
        protect_cache_location(input_path, checkpoint_path, reference_train, cache_dir)
    if args.timing_json is not None:
        tp = args.timing_json.expanduser().absolute().resolve()
        protected = [input_path, checkpoint_path]
        if args.domain_transfer:
            protected.extend([reference_train, cache_dir])
        if (tp == output_dir or tp in output_dir.parents or output_dir in tp.parents
                or any(tp == path or (path.is_dir() and path in tp.parents)
                       for path in protected)):
            raise ValueError("Timing JSON must be outside inputs, reference/cache and result folder")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}; use --overwrite.")
    stack_info = inspect_input_stack(input_path)
    if min(stack_info[1]) < TILE_SIZE:
        raise ValueError(f"The five-slice refinement model requires H,W >= {TILE_SIZE}; got {stack_info[1]}")
    device = resolve_device(args.device)
    if args.cuda_amp and device.type != "cuda":
        raise ValueError("--cuda-amp requires CUDA.")

    if args.domain_transfer:
        global DOMAIN_REFERENCE_DIR, DOMAIN_REFERENCE_CACHE_DIR
        DOMAIN_REFERENCE_DIR = reference_train
        DOMAIN_REFERENCE_CACHE_DIR = cache_dir
        train_slices = discover_domain_reference_slices()
    else:
        train_slices = None

    model = UNet2p5D()
    load_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    plan = ExecutionPlan(device, args)
    if args.check_only:
        mode_info = (f"domain transfer ON, reference slices={len(train_slices)}"
                     if args.domain_transfer else "domain transfer OFF, no reference/cache required")
        print(f"Preflight OK: input ZYX={(stack_info[0], *stack_info[1])}, "
              f"{mode_info}, device={device}.")
        return

    print(f"Input: {input_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Domain transfer: {'ON' if args.domain_transfer else 'OFF'}")
    if args.domain_transfer:
        print(f"Reference train split: {reference_train}")
        print(f"Domain cache: {cache_dir}")
    print(f"Device: {device}; CUDA FP16 AMP: {args.cuda_amp}")
    print(f"Output: {output_dir}")
    print(f"Visualization indices: {select_visualization_indices(stack_info[0])}")
    print(f"Execution: {args.execution}; preparation workers: {plan.prepare_workers}")
    timings = {"device": str(device), "execution": args.execution,
               "domain_transfer_enabled": args.domain_transfer, "inference_slices": []}

    with InputStackReader(input_path, stack_info) as reader:
        reference = None
        if args.domain_transfer:
            cache_dir.mkdir(parents=True, exist_ok=True)
            sources = [InputStackSlice(reader, z) for z in range(reader.z_count)]
            t_domain = time.perf_counter()
            reference = load_or_build_domain_transfer_statistics(sources, input_path)
            timings["domain_transfer_reference_seconds"] = time.perf_counter() - t_domain
            print(f"[Timing] domain transfer statistics: {timings['domain_transfer_reference_seconds']:.2f}s",
                  flush=True)
        prepare_output(output_dir, args.overwrite)
        t_inference = time.perf_counter()
        dest, count = run_stack(reader, model, device, output_dir, reference, args.cuda_amp,
                                plan=plan, timings=timings)
        timings["stack_inference_seconds"] = time.perf_counter() - t_inference
    timings["overall_seconds"] = time.perf_counter() - t_whole_start
    timings["execution_plan"] = plan.describe()
    print(f"[Timing] inference={timings['stack_inference_seconds']:.2f}s "
          f"overall={timings['overall_seconds']:.2f}s", flush=True)
    if args.timing_json is not None:
        timing_path = args.timing_json.expanduser().absolute()
        if (timing_path == input_path or timing_path == checkpoint_path or
                timing_path == output_dir or output_dir in timing_path.parents or
                timing_path in output_dir.parents):
            raise ValueError("Timing JSON must not overlap output, input or checkpoint")
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        print(f"[Timing] report: {timing_path}")
    print(f"Completed: {dest}")
    print(f"Visualization PNGs: {count}")

if __name__ == "__main__":
    main()



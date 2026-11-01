#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryo-FIB-SEM Preprocessing Step 1: Universal Input Conversion
=============================================================

Author: Ningyan Zhang
Last modified: October 10, 2026

Description:
    Convert microscopy image data from supported source formats into one
    grayscale uint8 stacked TIFF volume with logical ZYX axis order.

    The intensity mapping is IDENTICAL to the established
    preprocessing procedure. Each input plane is read as float32, sampled
    using the original deterministic stride rule, and pooled into one global
    P0.5-P99.5 percentile window. The same window is applied to every plane.

    This script performs no per-slice normalization, histogram matching,
    resizing, cropping, registration, VSNR, or other artifact correction.

Inputs:
    - HDF5/HDF (.hdf, .h5, .hdf5, .he5): single 2D/3D dataset.
    - MRC-family volumes (.mrc, .mrcs, .map, .rec).
    - Single grayscale 2D/3D TIFF, NPY or NPZ arrays.
    - Grayscale PNG/BMP/JPEG images, or naturally sorted slice directories.
    - A directory of independent HDF files or other supported file families.

Outputs:
    - <input_name>_uint8_stacked.tif: an uncompressed uint8 BigTIFF volume.
    - <input_name>_uint8_stacked_summary.json: conversion metadata (optional).

Requirements:
    - Python 3.8 or newer.
    - numpy, h5py, tifffile, tqdm.
    - mrcfile for MRC input; Pillow for PNG/BMP/JPEG input.

Usage:
    python code/preprocessing/step1_convert_to_stacked_uint8.py \
        --input /path/to/volume.hdf

    python code/preprocessing/step1_convert_to_stacked_uint8.py \
        --input /path/to/volume.mrc \
        --output /path/to/converted_uint8_stacked.tif

    python code/preprocessing/step1_convert_to_stacked_uint8.py \
        --input /path/to/slice_directory

    python code/preprocessing/step1_convert_to_stacked_uint8.py \
        --input /path/to/ambiguous.h5 --hdf-key /data/volume

Scientific invariants:
    - Percentiles: P0.5 and P99.5, estimated ONCE for the full volume.
    - Sampling per slice: flat[::max(1, len(flat) // 200000)][:200000].
    - Conversion: float32 -> linear percentile window -> clip(0, 1) ->
      np.round(value * 255) -> uint8, with original arithmetic order.
    - Original image dimensions, slice order and pixel spatial scale remain
      unchanged. Differing slice shapes are rejected, not resampled.

Notes:
    - The numeric Z dimension is retained in output metadata. TIFF readers
      may display a single-plane (Z=1) stack as a two-dimensional YX array.
    - HDF4, RGB/multichannel data, ambiguous arrays and non-finite pixel
      values are not silently converted.
    - Running this conversion again on an already normalized uint8 stack
      applies ANOTHER global percentile mapping and is not idempotent.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import tifffile
from tqdm import tqdm


# ============================================================
# 1. Scientific constants and supported input formats
# ============================================================

PERCENTILE_LOW = 0.5
PERCENTILE_HIGH = 99.5
MAX_SAMPLES_PER_SLICE = 200_000

HDF_EXT = {".hdf", ".h5", ".hdf5", ".he5"}
MRC_EXT = {".mrc", ".mrcs", ".map", ".rec"}
TIFF_EXT = {".tif", ".tiff"}
IMAGE_EXT = {".png", ".bmp", ".jpg", ".jpeg"}
ARRAY_EXT = {".npy", ".npz"}
SUPPORTED_EXT = HDF_EXT | MRC_EXT | TIFF_EXT | IMAGE_EXT | ARRAY_EXT


# ============================================================
# 2. Common validation and source abstractions
# ============================================================


def natural_sort_key(path: Path):
    """Sort filenames by numeric index (slice_2 precedes slice_10)."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", path.name)]


def ensure_numeric_array(dtype: Any, label: str) -> None:
    """Reject non-real and structured arrays instead of guessing channels."""
    if np.dtype(dtype).kind not in "iuf":
        raise ValueError(
            f"{label}: expected real numeric grayscale data; got {dtype}. "
            "Complex, RGB and structured data are not supported."
        )


def volume_shape(shape: Any, label: str) -> tuple[int, int, int]:
    """Interpret a 2D YX image as one-plane ZYX, or validate native ZYX.

    The input is never transposed: sources must already be in ZYX order.
    """
    shape = tuple(int(x) for x in shape)
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    if len(shape) != 3 or any(x < 1 for x in shape):
        raise ValueError(
            f"{label}: expected 2D YX or 3D ZYX grayscale array, got {shape}. "
            "Four-dimensional/color volumes must be converted explicitly."
        )
    return shape


class SliceSource:
    """Lazy, read-only, uniform-volume 2D slice source."""

    def __init__(self, shape, dtype, read, description, stack: ExitStack):
        """Store shape, reader callback and cleanup stack without loading the volume.

        Args:
            shape: Logical (Z, Y, X) dimensions.
            dtype: Original source pixel dtype.
            read: Callback accepting a zero-based Z index and returning a 2D plane.
            description: User-readable description of the detected source layout.
            stack: Context manager for all open input file handles.
        """
        self.shape = tuple(shape)
        self.dtype = str(np.dtype(dtype))
        self.read = read
        self.description = description
        self.stack = stack

    def close(self):
        """Close all source files acquired during input discovery."""
        self.stack.close()

    def __enter__(self):
        """Enter a managed source-reading context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Release all source handles, including when conversion fails."""
        self.close()


# ============================================================
# 3. HDF5 dataset discovery
# ============================================================


def find_hdf_layout(h5: h5py.File, hdf_key: str | None):
    """Return a list of (dataset path, local-z-index or None) records.

    Supports single 2D/3D datasets and indexed MDF/images/<z>/image groups.
    An explicitly specified --hdf-key may also identify a numbered group.
    Ambiguous inputs fail rather than silently selecting unrelated datasets.

    Args:
        h5: Open HDF5 file.
        hdf_key: Optional explicit internal dataset/group path.

    Returns:
        Pair (records, description), where each record is
        (dataset_path, local_z_index_or_None).
    """
    # Prefer an explicitly specified internal path over heuristic discovery.
    if hdf_key:
        key = hdf_key.strip("/")
        if key not in h5:
            raise KeyError(f"HDF path not found: {key}")
        node = h5[key]
        if isinstance(node, h5py.Dataset):
            shape = volume_shape(node.shape, key)
            ensure_numeric_array(node.dtype, key)
            return [(key, None if len(node.shape) == 2 else i)
                    for i in range(shape[0])], f"HDF dataset '{key}'"
        if not isinstance(node, h5py.Group):
            raise ValueError(f"Unsupported HDF object at {key}")
        group = node
        prefix = key
    elif "MDF/images" in h5 and isinstance(h5["MDF/images"], h5py.Group):
        group = h5["MDF/images"]
        prefix = "MDF/images"
    else:
        group = None
        prefix = ""

    # Many instrument HDF5 files are containers of per-slice 2D datasets,
    # not one conventional 3D array. Index order, not HDF iteration order,
    # determines the source Z order.
    if group is not None:
        numbered = []
        for name in group:
            child = group[name]
            if name.isdigit() and isinstance(child, h5py.Group) and "image" in child:
                if isinstance(child["image"], h5py.Dataset):
                    numbered.append((int(name), f"{prefix}/{name}/image"))
        numbered.sort()
        if numbered:
            indices = [i for i, _ in numbered]
            if len(set(indices)) != len(indices):
                raise ValueError("Duplicate numeric HDF slice indices")
            # Do not silently reassign missing index positions.
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(
                    "Missing HDF slice indices in numbered group. "
                    f"Found {indices[:20]}...; inspect the input file."
                )
            if len(numbered) == 1:
                path = numbered[0][1]
                ds = h5[path]
                shape = volume_shape(ds.shape, path)
                ensure_numeric_array(ds.dtype, path)
                return [(path, None if ds.ndim == 2 else i)
                        for i in range(shape[0])], f"numbered HDF: {path}"
            records = []
            for _, path in numbered:
                ds = h5[path]
                if ds.ndim != 2:
                    raise ValueError(f"{path}: expected 2D among numbered HDF slices, got {ds.shape}")
                ensure_numeric_array(ds.dtype, path)
                records.append((path, None))
            return records, f"numbered HDF datasets under {prefix}"

        # Explicitly chosen groups may themselves contain numbered 2D datasets.
        if hdf_key:
            numbered_flat = sorted(
                [(int(k), f"{prefix}/{k}") for k in group
                 if k.isdigit() and isinstance(group[k], h5py.Dataset)]
            )
            if numbered_flat:
                indices = [i for i, _ in numbered_flat]
                if indices != list(range(indices[0], indices[-1] + 1)):
                    raise ValueError("Missing indexed HDF datasets in selected group")
                records = []
                for _, path in numbered_flat:
                    ds = h5[path]
                    if ds.ndim != 2:
                        raise ValueError(f"{path}: expected 2D indexed dataset")
                    ensure_numeric_array(ds.dtype, path)
                    records.append((path, None))
                return records, f"indexed datasets under {prefix}"

    # Generic HDF: auto-pick only an unambiguous numeric 2D/3D dataset.
    # If the user selected a group explicitly, never search outside that group.
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3) and np.dtype(obj.dtype).kind in "iuf":
            found.append(f"{prefix}/{name}" if hdf_key else name)

    if hdf_key:
        group.visititems(visit)
    else:
        h5.visititems(visit)
    if len(found) != 1:
        preview = ", ".join(found[:12]) or "none"
        raise ValueError(
            f"Ambiguous HDF: {len(found)} candidate grayscale 2D/3D datasets "
            f"({preview}). Use --hdf-key to choose a dataset or numbered group."
        )
    path = found[0]
    ds = h5[path]
    shape = volume_shape(ds.shape, path)
    return [(path, None if ds.ndim == 2 else i)
            for i in range(shape[0])], f"HDF dataset '{path}'"


# ============================================================
# 4. Format-specific lazy image readers
# ============================================================


def open_single_file(path: Path, stack: ExitStack,
                     hdf_key: str | None, npz_key: str | None):
    """Open one file without eagerly loading the entire microscopy volume.

    Args:
        path: Supported image/volume file.
        stack: Collects cleanup callbacks for open files.
        hdf_key: Selected HDF5 dataset/group, if needed.
        npz_key: Selected NumPy archive member, if needed.

    Returns:
        (shape_zyx, source_dtype, read_plane, layout_description).

    Raises:
        ValueError: Ambiguous axes, invalid dimensions or unsupported data.
        ImportError: Optional format-specific library is not installed.
    """
    ext = path.suffix.lower()
    # Read only one plane at a time whenever the file format supports it.
    if ext in HDF_EXT:
        h5 = h5py.File(path, "r")
        stack.callback(h5.close)
        records, description = find_hdf_layout(h5, hdf_key)
        ds0 = h5[records[0][0]]
        yx = tuple(int(x) for x in (ds0.shape if ds0.ndim == 2 else ds0.shape[1:]))
        for key, _ in records:
            ds = h5[key]
            ds_yx = tuple(int(x) for x in (ds.shape if ds.ndim == 2 else ds.shape[1:]))
            if ds_yx != yx:
                raise ValueError(
                    f"HDF slices have inconsistent YX dimensions: {ds_yx} != {yx} ({key}). "
                    "No automatic cropping/resizing is allowed."
                )
        def read(i):
            key, local_z = records[i]
            ds = h5[key]
            return ds[()] if local_z is None else ds[local_z]
        return (len(records), *yx), ds0.dtype, read, description

    if ext in MRC_EXT:
        try:
            import mrcfile
        except ImportError as exc:
            raise ImportError("MRC input requires: pip install mrcfile") from exc
        handle = mrcfile.mmap(str(path), mode="r", permissive=False)
        stack.callback(handle.close)
        arr = handle.data
        shape = volume_shape(arr.shape, str(path))
        ensure_numeric_array(arr.dtype, str(path))
        return shape, arr.dtype, (lambda i: arr if arr.ndim == 2 else arr[i]), "MRC volume"

    if ext in TIFF_EXT:
        tf = tifffile.TiffFile(str(path))
        stack.callback(tf.close)
        if len(tf.series) != 1:
            raise ValueError(
                f"TIFF has {len(tf.series)} series; only one grayscale series is supported."
            )
        series = tf.series[0]
        shape = volume_shape(series.shape, str(path))
        axes = series.axes
        if len(shape) == 3 and axes not in ("ZYX", "IYX", "QYX", "TYX", "YX"):
            raise ValueError(
                f"TIFF axes '{axes}' are ambiguous (possibly color/multichannel). "
                "Expected ZYX/IYX/QYX/TYX or plain YX."
            )
        ensure_numeric_array(series.dtype, str(path))
        try:
            arr = tifffile.memmap(str(path), series=0, mode="r")
            read = lambda i: arr if arr.ndim == 2 else arr[i]
        except (OSError, ValueError, TypeError):
            # Ordinary compressed multipage TIFF: decode only requested page.
            if series.shape == shape[1:] and len(tf.pages) == 1:
                read = lambda i: tf.pages[0].asarray()
            elif len(series.pages) == shape[0] and all(len(p.shape) == 2 for p in series.pages):
                read = lambda i: series.pages[i].asarray()
            else:
                raise ValueError(
                    f"This TIFF cannot be read plane-by-plane (shape={series.shape}, "
                    f"pages={len(series.pages)}). Export as uncompressed ZYX TIFF first."
                )
        return shape, series.dtype, read, f"TIFF series axes={axes}"

    if ext in ARRAY_EXT:
        if ext == ".npy":
            arr = np.load(str(path), mmap_mode="r", allow_pickle=False)
            description = "NumPy NPY"
        else:
            npz = np.load(str(path), allow_pickle=False)
            stack.callback(npz.close)
            candidates = list(npz.files)
            if npz_key:
                if npz_key not in candidates:
                    raise KeyError(f"NPZ key '{npz_key}' not found; available: {candidates}")
                key = npz_key
            elif len(candidates) == 1:
                key = candidates[0]
            else:
                raise ValueError(f"NPZ contains multiple arrays {candidates}; use --npz-key")
            arr = npz[key]
            description = f"NumPy NPZ array '{key}'"
        shape = volume_shape(arr.shape, str(path))
        ensure_numeric_array(arr.dtype, str(path))
        return shape, arr.dtype, (lambda i: arr if arr.ndim == 2 else arr[i]), description

    if ext in IMAGE_EXT:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("PNG/BMP/JPEG input requires: pip install Pillow") from exc
        with Image.open(path) as im:
            if im.mode not in ("L", "I;16", "I;16L", "I;16B", "I", "F", "1"):
                raise ValueError(
                    f"{path.name}: input is mode '{im.mode}', expected grayscale. "
                    "RGB/multichannel images are not converted silently."
                )
            yx = (im.height, im.width)
            dtype = np.asarray(im).dtype
        def read(i):
            with Image.open(path) as im:
                return np.asarray(im)
        ensure_numeric_array(dtype, str(path))
        return (1, *yx), dtype, read, f"2D image {ext}"

    raise ValueError(f"Unsupported file format: {path.suffix} ({path})")


# ============================================================
# 5. Unified file/directory volume input
# ============================================================


def open_source(path: Path, hdf_key: str | None = None,
                npz_key: str | None = None) -> SliceSource:
    """Create one logical ZYX source from a volume file or a file directory.

    Directory files are naturally ordered and concatenated in Z. This is
    intended for independent HDF slice files or image-sequence folders.
    Files must belong to one compatible format family and have matching YX.

    Returns:
        A context-managed, lazy slice source.
    """
    stack = ExitStack()
    try:
        if path.is_file():
            shape, dtype, read, desc = open_single_file(path, stack, hdf_key, npz_key)
            return SliceSource(shape, dtype, read, desc, stack)
        if not path.is_dir():
            raise FileNotFoundError(f"Input does not exist: {path}")

        files = sorted((p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT),
                       key=natural_sort_key)
        if not files:
            raise ValueError(f"No supported image files in directory: {path}")
        # A directory is interpreted as a Z-concatenation of slice files or
        # independent volumes. No spatial resampling or channel selection.
        # A directory is a collection of slices or independent volumes.
        # Require one logical file family to avoid silently mixing modalities.
        def family(p):
            e = p.suffix.lower()
            if e in HDF_EXT: return "hdf"
            if e in MRC_EXT: return "mrc"
            if e in (TIFF_EXT | IMAGE_EXT): return "images"
            return "arrays"
        families = {family(p) for p in files}
        if len(families) != 1:
            raise ValueError(
                "Input directory mixes unrelated data formats; split it first: "
                + ", ".join(sorted(families))
            )
        records = []
        ref_yx = None
        ref_dtype = None
        for f in files:
            shape, dtype, reader, description = open_single_file(f, stack, hdf_key, npz_key)
            if ref_yx is None:
                ref_yx = shape[1:]
                ref_dtype = dtype
            elif shape[1:] != ref_yx:
                raise ValueError(
                    f"Unequal slice shape in directory: {f.name}: {shape[1:]} != {ref_yx}. "
                    "This tool never resizes/crops images."
                )
            for local_idx in range(shape[0]):
                records.append((reader, local_idx))
        def read(i):
            reader, local_idx = records[i]
            return reader(local_idx)
        return SliceSource((len(records), *ref_yx), ref_dtype, read,
                           f"directory ({families.pop()}, {len(files)} files)", stack)
    except Exception:
        stack.close()
        raise


# ============================================================
# 6. Original global intensity-window calculation and mapping
# ============================================================


def to_uint8(plane: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Apply the unchanged original global-window uint8 mapping to one plane.

    Args:
        plane: Source image plane. Converted to float32 exactly as before.
        p_low: Full-volume sampled 0.5th percentile.
        p_high: Full-volume sampled 99.5th percentile.

    Returns:
        A uint8 YX array using the original arithmetic order and np.round.
    """
    x = plane.astype(np.float32, copy=False)
    x = (x - p_low) / (p_high - p_low)
    x = np.clip(x, 0.0, 1.0)
    return np.round(x * 255.0).astype(np.uint8)


def estimate_global_window(src: SliceSource):
    """Estimate the original single intensity window across ALL Z slices.

    The sample stride and 200,000-pixel cap apply independently to each
    plane, but the final percentile calculation pools ALL plane samples.
    Keeping the exact sample order ensures reproducibility across formats.
    An implementation-only sample copy prevents keeping entire source
    images resident when the source returns a temporary decoded array.

    Args:
        src: Lazy source whose reader returns 2D pixel arrays.

    Returns:
        (raw_min, raw_max, p_low, p_high) from the complete input volume.
    """
    samples = []
    raw_min = np.inf
    raw_max = -np.inf
    for i in tqdm(range(src.shape[0]), desc="Sampling", dynamic_ncols=True):
        plane = np.asarray(src.read(i), dtype=np.float32)
        if plane.shape != src.shape[1:]:
            raise ValueError(f"Slice {i} has shape {plane.shape}, expected {src.shape[1:]}")
        if not np.all(np.isfinite(plane)):
            raise ValueError(f"Nonfinite pixel values in slice {i}")
        raw_min = min(raw_min, float(plane.min()))
        raw_max = max(raw_max, float(plane.max()))
        # Preserve the original deterministic flat-array sampling formula.
        # Using every Nth pixel (not a random sample) is part of the method.
        flat = plane.ravel()
        step = max(1, len(flat) // MAX_SAMPLES_PER_SLICE)
        sample = flat[::step][:MAX_SAMPLES_PER_SLICE].copy()
        samples.append(sample)
    all_samples = np.concatenate(samples)
    # This percentile window is computed only once for the full Z volume.
    # Per-slice normalization would alter inter-slice intensity relationships.
    low, high = np.percentile(all_samples, [PERCENTILE_LOW, PERCENTILE_HIGH])
    del all_samples, samples
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(
            f"Invalid global percentile window: low={low}, high={high}. "
            "Cannot normalize constant-intensity volumes."
        )
    # Preserve the NumPy float64 percentile scalars from the original Step-1.
    # Casting to Python float changes NumPy 2.x promotion for float32 arrays
    # and can change uint8 rounding at a small number of pixels.
    return float(raw_min), float(raw_max), low, high


# ============================================================
# 7. Output selection, stacked TIFF writing and verification
# ============================================================


def resolve_output(input_path: Path, output_arg: str | None) -> Path:
    """Choose a predictable stacked-TIFF filename beside the source."""
    if output_arg is None:
        return input_path.parent / (input_path.stem + "_uint8_stacked.tif")
    output = Path(output_arg).expanduser().resolve()
    if output.suffix.lower() not in TIFF_EXT:
        raise ValueError("--output must name a .tif or .tiff file")
    return output


def convert(input_path: Path, output_path: Path, overwrite: bool,
            hdf_key: str | None, npz_key: str | None, write_summary: bool):
    """Convert one logical microscopy volume and verify the final TIFF.

    Source planes are read twice: the first pass estimates the original
    global percentile window, and the second pass writes uint8 output.
    Memory usage is limited by one decoded plane, the sampled intensity
    pool, and the OS-backed output memmap rather than the full volume.

    The output is written via a temporary TIFF. The completed TIFF is
    published only after its dimensions, dtype and axes are verified.
    Existing outputs are never replaced without --overwrite.
    """
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path == input_path or (input_path.is_dir() and input_path in output_path.parents):
        raise ValueError("Output must be outside the input directory and cannot overwrite its source")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite: {output_path}")
    if write_summary and summary_path.exists() and not overwrite:
        raise FileExistsError(f"Summary already exists; use --overwrite: {summary_path}")

    with open_source(input_path, hdf_key=hdf_key, npz_key=npz_key) as src:
        shape = src.shape
        print("=" * 78)
        print("Unified input -> uint8 ZYX stacked TIFF")
        print("Input       :", input_path)
        print("Detected    :", src.description)
        print("Input dtype :", src.dtype)
        print("Shape (ZYX) :", shape)
        print("Output      :", output_path)
        print("Mapping     : global P0.5-P99.5 (original Step-1; unchanged)")
        print("No resize, crop, alignment, VSNR, or histogram matching")
        print("=" * 78)

        raw_min, raw_max, p_low, p_high = estimate_global_window(src)
        print(f"Global window: P0.5={p_low:.8g}, P99.5={p_high:.8g}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete, validated outputs. BigTIFF and a disk-backed
        # memmap avoid constructing the entire output volume in RAM.
        # Write an incomplete output to a temporary location, then replace.
        temp_path = output_path.with_name(output_path.name + ".partial.tif")
        if temp_path.exists():
            if not overwrite:
                raise FileExistsError(f"Incomplete previous run: {temp_path}; use --overwrite")
            temp_path.unlink()
        try:
            out = tifffile.memmap(
                str(temp_path), shape=shape, dtype=np.uint8,
                photometric="minisblack", metadata={"axes": "ZYX"},
                bigtiff=True,
            )
            try:
                for i in tqdm(range(shape[0]), desc="Writing", dynamic_ncols=True):
                    plane = np.asarray(src.read(i), dtype=np.float32)
                    if plane.shape != shape[1:]:
                        raise ValueError(f"Slice shape changed during conversion at z={i}")
                    out[i] = to_uint8(plane, p_low, p_high)
                    if (i + 1) % 16 == 0 or i == shape[0] - 1:
                        out.flush()
                out.flush()
            finally:
                del out

            with tifffile.TiffFile(str(temp_path)) as tf:
                actual = tuple(int(v) for v in tf.series[0].shape)
                dtype = str(tf.series[0].dtype)
                axes = tf.series[0].axes
            # tifffile may squeeze a singleton Z axis while the TIFF's shaped
            # metadata still records the original logical (1, Y, X) volume.
            single_plane = shape[0] == 1 and actual == shape[1:] and axes == "YX"
            if not ((actual == shape and axes == "ZYX") or single_plane) or dtype != "uint8":
                raise RuntimeError(
                    f"Output verification failed: shape={actual}, dtype={dtype}, axes={axes}"
                )
            temp_path.replace(output_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        if write_summary:
            summary = {
                "input": str(input_path), "source_layout": src.description,
                "input_dtype": src.dtype, "output_tif": str(output_path),
                "shape_zyx": [int(x) for x in shape], "output_dtype": "uint8",
                "output_axes": "ZYX" if shape[0] > 1 else "ZYX metadata; some readers expose YX for Z=1",
                "raw_global_min": raw_min,
                "raw_global_max": raw_max, "percentile_low": PERCENTILE_LOW,
                "percentile_high": PERCENTILE_HIGH, "p_low_value": p_low,
                "p_high_value": p_high,
                "max_samples_per_slice": MAX_SAMPLES_PER_SLICE,
                "normalization": "single global sampled percentile window; original Step-1 mapping",
                "crop": False, "resize": False, "alignment": False,
                "vsnr": False, "histogram_matching": False,
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print("DONE: ", output_path)
        if write_summary:
            print("Summary:", summary_path)
        print("Verified shape/dtype/axes:", actual, "uint8", axes)
        if shape[0] == 1:
            print("Note: TIFF readers represent a single Z plane as 2D YX; logical Z=1.")


# ============================================================
# 8. Command-line interface
# ============================================================


def main(argv=None):
    """Parse CLI arguments and report user-facing conversion errors."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="HDF/MRC/TIFF/NPY/NPZ/PNG/BMP/JPEG file or an image directory")
    parser.add_argument("--output", default=None, help="Output .tif path (default: <input>_uint8_stacked.tif beside source)")
    parser.add_argument("--hdf-key", default=None, help="Explicit HDF dataset path, e.g. /MDF/images/0/image or /volume")
    parser.add_argument("--npz-key", default=None, help="Explicit NPZ array name if archive contains multiple arrays")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output and summary")
    parser.add_argument("--no-summary", action="store_true", help="Write stacked TIFF only, no JSON summary")
    args = parser.parse_args(argv)
    path = Path(args.input).expanduser().resolve()
    try:
        convert(path, resolve_output(path, args.output), args.overwrite,
                args.hdf_key, args.npz_key, not args.no_summary)
    except (ValueError, KeyError, FileNotFoundError, OSError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cryo-FIB-SEM Stage-1 Step 5: Whole-Slice Artifact Removal Inference

Author: Ningyan Zhang
Last modified: October 8, 2026

Description:
    Apply the trained Stage-1 artifact-removal network independently to each
    Z slice of a grayscale TIFF stack. The model removes image artifacts while
    preserving the original slice dimensions and Z ordering. Inference does
    not use neighboring slices, and it does not introduce interpolation,
    registration, domain matching, or additional image enhancement.

    This release preserves the verified model architecture, whole-image and
    overlapping-tile inference algorithms, residual-correction blending,
    and final uint8 conversion. It accepts checkpoints produced on CUDA,
    Apple MPS, or CPU and can perform inference on any available backend.

Required inputs:
    --input PATH           Input grayscale uint8 TIFF stack, supplied as:
                           (a) a standard multipage TIFF with one 2D slice
                               per physical page; or
                           (b) a single-page, memory-mappable 3D TIFF with
                               explicit ZYX axes metadata.
                           A single 2D TIFF is accepted as a Z=1 stack.
                           All slices must have the same height, width,
                           single-channel format, and uint8 dtype.

    --checkpoint PATH      PyTorch checkpoint containing model_state_dict,
                           such as checkpoints/best.pt, or a standalone
                           model state_dict with matching tensor names and
                           shapes. Weights are first loaded onto the CPU
                           and then moved to the inference device.

    --output-dir PATH      New directory for the restored stack and previews.
                           The input TIFF and checkpoint remain read-only.

Optional arguments:
    --device {auto,cuda,mps,cpu}
                           Default: auto (CUDA > Apple MPS > CPU).
                           Explicit unavailable devices cause an error;
                           there is no silent fallback.
    --cuda-amp             Enable CUDA float16 autocast. Disabled by default;
                           this option is invalid for MPS or CPU.
    --overwrite            Explicitly delete an existing output directory
                           and replace it. Disabled by default.

Output structure:
    --output-dir/
        output_stack.tif           Single uint8 TIFF stack with the same
                                   (Z, Y, X) shape and slice order as input.
        visualization/
            slice_0000.png         Example Z-indexed preview filename.
                                   Five evenly spaced Z positions, including
                                   the first and last slice (or all Z if
                                   fewer than five slices are present).

    Each preview contains three labeled grayscale panels:
        original input | artifact-removed output | signed correction.
    The correction panel uses a neutral gray value of 128 for no correction;
    brighter/darker pixels indicate positive/negative model correction.
    Preview images may be downsampled for display, but the TIFF output is
    always written at the original pixel resolution.

    No per-slice TIFFs, metadata JSON, training caches, or source snapshots
    are written. A temporary partial TIFF is used while inference runs and
    is renamed to output_stack.tif only after successful completion.

Usage example A: Inference on an NVIDIA GPU
    python step5_inference.py \
        --input /path/to/input_stack.tif \
        --checkpoint /path/to/checkpoints/best.pt \
        --output-dir /path/to/stage1_output \
        --device cuda

Usage example B: Inference on Apple Silicon (MPS)
    python step5_inference.py \
        --input /path/to/input_stack.tif \
        --checkpoint /path/to/checkpoints/best.pt \
        --output-dir /path/to/stage1_output_mps \
        --device mps

Usage example C: Optional CUDA mixed-precision inference
    python step5_inference.py \
        --input /path/to/input_stack.tif \
        --checkpoint /path/to/checkpoints/best.pt \
        --output-dir /path/to/stage1_output_amp \
        --device cuda --cuda-amp

Usage example D: Explicitly replace an existing result
    python step5_inference.py \
        --input /path/to/input_stack.tif \
        --checkpoint /path/to/checkpoints/best.pt \
        --output-dir /path/to/stage1_output \
        --device cpu --overwrite

Scientific inference procedure (preserve when maintaining this script):
    1. Validate the input TIFF layout, axes, uint8 dtype, and consistent
       slice dimensions. Read one 2D Z slice at a time; do not load the
       complete stack into host RAM. Convert uint8 pixels to float32 [0, 1]
       solely as the numerical input representation expected by the model.
    2. Load the single-channel residual restoration network using:
           PixelUnshuffle(2) -> NAFNet(width=16) -> PixelShuffle(2).
       Encoder blocks: (1, 1, 1, 2); middle blocks: 4;
       decoder blocks: (1, 1, 1, 1); dropout: 0.1.
       Use strict checkpoint weight loading and model.eval() throughout.
    3. For slices with both H and W <= 2048, infer at the native dimensions.
       Apply only the minimal reflect padding required for pixel-unshuffle
       and the network's internal encoder stride, then crop back to H x W.
    4. For larger slices, use 2048x2048 tiles with 512-pixel overlap and
       stride 1536. Include terminal tiles at the image boundaries; reflect
       pad any incomplete boundary tiles only as needed for inference.
    5. Compute each tile's predicted residual correction relative to its
       input and blend corrections using the original floored Hann weights.
       Add the blended correction to the original full-resolution input,
       clip restored values to [0, 1], multiply by 255, and convert to uint8
       using NumPy round-then-clip semantics.
    6. Write restored slices directly into one on-disk uint8 ZYX BigTIFF.
       Select up to five unique, evenly spaced Z indices for visualization;
       no slice selection or cropping affects the actual output volume.

Reproducibility and safety:
    - Default inference uses float32 on CUDA, MPS, and CPU. CUDA float16
      autocast is strictly opt-in and may produce different numerical results.
    - The model is applied independently to each original Z slice. No
      preprocessing beyond uint8-to-float conversion, no synthetic noise,
      and no histogram/style normalization are introduced at inference.
    - The restored volume keeps the input ZYX geometry; visual previews are
      diagnostic only and never feed back into inference computations.
    - Existing outputs are protected unless --overwrite is specified.
      The output directory must not contain the input TIFF or checkpoint.

Requirements:
    Python 3.9+, PyTorch (2.4+ recommended), NumPy, OpenCV, tifffile, tqdm.
    A GPU is optional; large-image inference can be slower on CPU.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import shutil

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# Frozen scientific inference constants.
TILE_SIZE = 2048
TILE_OVERLAP = 512
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP
BLEND_EPS = 1e-3
VIS_MAX_SIDE = 1920
VIS_SEPARATOR_WIDTH = 8
VIS_HEADER_HEIGHT = 28
VIS_CORRECTION_GAIN = 1.0
OUTPUT_STACK_NAME = "output_stack.tif"

# Exact training model configuration; names and tensor shapes are unchanged.
PIXEL_UNSHUFFLE_FACTOR = 2
WIDTH = 16
MIDDLE_BLK_NUM = 4
ENC_BLK_NUMS = (1, 1, 1, 2)
DEC_BLK_NUMS = (1, 1, 1, 1)
DROP_OUT_RATE = 0.1

# Configured once before inference. Default FP32 matches the original MPS path.
USE_AMP = False

# Model definition: identical executable statements to the frozen training model.
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-06):
        """Create per-channel normalization parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        """Apply channelwise two-dimensional layer normalization."""
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):

    def forward(self, x):
        """Multiply paired channel halves as in the original model."""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):

    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        """Construct the unchanged NAFBlock convolution and gate layers."""
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channels // 2, dw_channels // 2, 1))
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp):
        """Apply the original residual NAFBlock transformation."""
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums, drop_out_rate=0.0):
        """Construct the encoder-middle-decoder network and skip pathways."""
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = width
        for num_blocks in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels, drop_out_rate=drop_out_rate) for _ in range(num_blocks)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(channels, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)])
        for num_blocks in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels, drop_out_rate=drop_out_rate) for _ in range(num_blocks)]))
        self.padder_size = 2 ** len(enc_blk_nums)

    def check_image_size(self, x):
        """Reflect-pad spatial sizes to the encoder stride multiple."""
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return (x, h, w)

    def forward(self, inp):
        """Predict the original residual-corrected image."""
        inp, original_h, original_w = self.check_image_size(inp)
        x = self.intro(inp)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)
        correction = self.ending(x)
        output = inp + correction
        return output[:, :, :original_h, :original_w]


class WholeSliceNAFNet(nn.Module):
    """Whole-slice architecture: PixelUnshuffle(2) -> NAFNet -> PixelShuffle(2)."""

    def __init__(self):
        """Construct the packed-channel network unchanged."""
        super().__init__()
        packed_channels = PIXEL_UNSHUFFLE_FACTOR ** 2
        self.net = NAFNet(img_channel=packed_channels, width=WIDTH, middle_blk_num=MIDDLE_BLK_NUM, enc_blk_nums=ENC_BLK_NUMS, dec_blk_nums=DEC_BLK_NUMS, drop_out_rate=DROP_OUT_RATE)

    def forward(self, x):
        """Unshuffle, restore with NAFNet, then pixel-shuffle."""
        x = F.pixel_unshuffle(x, PIXEL_UNSHUFFLE_FACTOR)
        x = self.net(x)
        return F.pixel_shuffle(x, PIXEL_UNSHUFFLE_FACTOR)


# Original inference algorithms: computation, tiling and visualization unchanged.
def to_uint8(x):
    """Convert a restored intensity array to uint8 using the original round-and-clip rule."""
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def get_positions(length):
    """Preserve the original tile schedule, including the terminal boundary tile."""
    if length <= TILE_SIZE:
        return [0]
    positions = list(range(0, length - TILE_SIZE + 1, TILE_STRIDE))
    last = length - TILE_SIZE
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_blend_weight():
    """Construct the original floored Hann window for correction-field blending."""
    one_d = np.hanning(TILE_SIZE).astype(np.float32)
    one_d = np.maximum(one_d, BLEND_EPS)
    weight = np.outer(one_d, one_d).astype(np.float32)
    return np.maximum(weight, BLEND_EPS)


def extract_tile(image, top, left):
    """Extract a 2048 tile, reflect-padding only outside the original image."""
    h, w = image.shape
    bottom = min(top + TILE_SIZE, h)
    right = min(left + TILE_SIZE, w)
    tile = image[top:bottom, left:right]
    valid_h, valid_w = tile.shape
    pad_h = TILE_SIZE - valid_h
    pad_w = TILE_SIZE - valid_w
    if pad_h or pad_w:
        tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
    return (tile, valid_h, valid_w)


@torch.no_grad()
def infer_native_image(model, source_uint8, device):
    """
    Direct whole-image inference at the native input size.

    This path is used when both H and W are <= TILE_SIZE.

    Important:
    - The image is NOT padded to 2048 x 2048.
    - Only a minimal bottom/right reflect pad is added if needed for
      pixel_unshuffle (factor=2).
    - NAFNet internally handles its own encoder divisibility padding.
    - 960 x 960 and 1920 x 1920 therefore run directly with no external pad.
    """
    h, w = source_uint8.shape
    factor = PIXEL_UNSHUFFLE_FACTOR
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        padded_u8 = np.pad(source_uint8, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded_u8 = source_uint8
    x = torch.from_numpy(padded_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with amp_context(device):
        pred = model(x)
    pred = pred.squeeze().float().cpu().numpy().astype(np.float32)
    pred = pred[:h, :w]
    source_float01 = source_uint8.astype(np.float32) / 255.0
    correction = pred - source_float01
    restored = np.clip(source_float01 + correction, 0.0, 1.0).astype(np.float32)
    meta = {'height': int(h), 'width': int(w), 'inference_mode': 'native_whole_image', 'tile_size_threshold': TILE_SIZE, 'external_pad_h': int(pad_h), 'external_pad_w': int(pad_w), 'num_tiles': 1, 'mean_abs_final_correction': float(np.mean(np.abs(correction)))}
    return (restored, correction, meta)


@torch.no_grad()
def infer_large_image(model, source_uint8, device):
    """Use native inference when possible; otherwise blend overlapping tile corrections."""
    h, w = source_uint8.shape
    if h <= TILE_SIZE and w <= TILE_SIZE:
        return infer_native_image(model, source_uint8, device)
    ys = get_positions(h)
    xs = get_positions(w)
    blend = make_blend_weight()
    correction_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    correction_abs_means = []
    model.eval()
    for top in ys:
        for left in xs:
            tile_u8, valid_h, valid_w = extract_tile(source_uint8, top, left)
            x = torch.from_numpy(tile_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            with amp_context(device):
                pred = model(x)
            pred = pred.squeeze().float().cpu().numpy().astype(np.float32)
            inp = tile_u8.astype(np.float32) / 255.0
            correction = pred - inp
            correction = correction[:valid_h, :valid_w]
            weight = blend[:valid_h, :valid_w]
            correction_sum[top:top + valid_h, left:left + valid_w] += correction * weight
            weight_sum[top:top + valid_h, left:left + valid_w] += weight
            correction_abs_means.append(float(np.mean(np.abs(correction))))
    correction = correction_sum / np.maximum(weight_sum, 1e-08)
    source_float01 = source_uint8.astype(np.float32) / 255.0
    restored = np.clip(source_float01 + correction, 0.0, 1.0).astype(np.float32)
    meta = {'height': int(h), 'width': int(w), 'inference_mode': 'tiled_residual_blend', 'tile_size': TILE_SIZE, 'tile_overlap': TILE_OVERLAP, 'tile_stride': TILE_STRIDE, 'tiles_y': len(ys), 'tiles_x': len(xs), 'num_tiles': len(ys) * len(xs), 'mean_abs_tile_correction': float(np.mean(correction_abs_means)), 'mean_abs_final_correction': float(np.mean(np.abs(correction)))}
    return (restored, correction, meta)


def resize_for_visualization(image):
    """Downsample preview panels only, never the restored TIFF output."""
    h, w = image.shape
    scale = min(1.0, VIS_MAX_SIDE / max(h, w))
    if scale == 1.0:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_title(image, title):
    """Append a labeled grayscale preview header."""
    header = np.full((VIS_HEADER_HEIGHT, image.shape[1]), 255, dtype=np.uint8)
    cv2.putText(header, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, 0, 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def make_visualization(source, restored, correction):
    """Create the original input/restoration/signed-correction comparison panel."""
    correction_vis = to_uint8(128.0 + correction * 255.0 * VIS_CORRECTION_GAIN)
    panels = [add_title(resize_for_visualization(source), 'stage1 input'), add_title(resize_for_visualization(restored), 'artefact-removed'), add_title(resize_for_visualization(correction_vis), f'artefact correction x{VIS_CORRECTION_GAIN:g}')]
    sep = np.zeros((panels[0].shape[0], VIS_SEPARATOR_WIDTH), dtype=np.uint8)
    return np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)


# Portable input/output and device interfaces.



def parse_args():
    """Parse the portable stack-in/stack-out command-line interface."""
    parser = argparse.ArgumentParser(
        description="Stage-1 2D artifact removal from one uint8 stacked TIFF."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input grayscale uint8 multipage/ZYX TIFF stack.",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Stage-1 training best.pt or model state_dict checkpoint.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New directory for output_stack.tif and visualization/.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
        help="Inference device (default: auto; explicit selections never fall back).",
    )
    parser.add_argument(
        "--cuda-amp", action="store_true",
        help="Opt in to CUDA float16 autocast. Default is full-precision inference.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete an existing output directory before inference.",
    )
    return parser.parse_args()


def resolve_device(requested="auto"):
    """Resolve an explicit or automatically selected device without silent fallback."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested Apple MPS, but MPS is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError(f"Unknown device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device):
    """Keep CUDA FP16 optional; MPS and CPU always run in full precision."""
    if USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_checkpoint(model, path):
    """Safely load either a training checkpoint or model weights onto CPU.

    Model weights have no training-device dependence. The full training
    checkpoint can contain optimizer/scheduler states, but these are not used
    in inference. Strict loading detects incompatible model architectures.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint safely: {path}. "
            "Expected a trusted PyTorch weights checkpoint containing tensors "
            "and basic Python data types."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary or model state_dict.")

    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise TypeError("Checkpoint model_state_dict must be a nonempty dictionary.")
    state = {
        (key.replace("module.", "", 1) if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model


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


def protect_output_location(input_path, checkpoint_path, output_dir):
    """Prevent an overwrite from deleting either of the two inputs."""
    input_path = input_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    for label, protected in (("input", input_path), ("checkpoint", checkpoint_path)):
        if output_dir == protected or output_dir in protected.parents:
            raise ValueError(
                f"Output directory {output_dir} contains the {label} file "
                f"{protected}; choose a separate output directory."
            )


def prepare_output(output_dir, overwrite):
    """Create a clean output tree only after validating the input and model."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. "
                "Use --overwrite only to intentionally replace this directory."
            )
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(f"Refusing to overwrite a non-directory or symlink: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "visualization").mkdir(parents=True, exist_ok=True)


def create_output_stack_memmap(path, num_slices, image_shape):
    """Allocate an on-disk uint8 ZYX BigTIFF; do not buffer the volume in RAM."""
    height, width = image_shape
    return tifffile.memmap(
        str(path),
        shape=(int(num_slices), int(height), int(width)),
        dtype=np.uint8,
        bigtiff=True,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )


def process_stack(input_path, output_dir, model, device, stack_info):
    """Restore each TIFF page, write one stacked TIFF, and save five previews."""
    z_count, yx_shape, reader_type = stack_info
    vis_indices = set(select_visualization_indices(z_count))
    pending_path = output_dir / "output_stack.partial.tif"
    final_path = output_dir / OUTPUT_STACK_NAME
    output_stack = None
    input_map = None
    try:
        output_stack = create_output_stack_memmap(pending_path, z_count, yx_shape)
        if reader_type == "memmap":
            input_map = tifffile.memmap(str(input_path), series=0, mode="r")
        with tifffile.TiffFile(str(input_path)) as tif:
            for z in tqdm(range(z_count), desc="Stage-1 inference", unit="slice"):
                source_u8 = (
                    np.asarray(input_map[z]) if reader_type == "memmap"
                    else tif.pages[z].asarray()
                )
                if source_u8.shape != yx_shape or source_u8.dtype != np.uint8:
                    raise ValueError(
                        f"Unexpected slice at z={z}: shape={source_u8.shape}, "
                        f"dtype={source_u8.dtype}; expected uint8 {yx_shape}"
                    )

                restored_float01, correction, _ = infer_large_image(model, source_u8, device)
                restored_u8 = to_uint8(restored_float01 * 255.0)
                output_stack[z] = restored_u8

                if z in vis_indices:
                    panel = make_visualization(source_u8, restored_u8, correction)
                    png_path = output_dir / "visualization" / f"slice_{z:04d}.png"
                    if not cv2.imwrite(str(png_path), panel):
                        raise IOError(f"Failed to write visualization: {png_path}")

        output_stack.flush()
        del output_stack
        output_stack = None
        pending_path.replace(final_path)
    finally:
        if output_stack is not None:
            output_stack.flush()
            del output_stack
        if input_map is not None:
            del input_map
        if pending_path.exists():
            pending_path.unlink()

    return final_path, len(vis_indices)


def main():
    """Validate inputs, load portable weights, then process a TIFF stack."""
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    # Do not dereference the final output path: --overwrite must reject
    # symlinked directories rather than deleting their resolved targets.
    output_dir = args.output_dir.expanduser().absolute()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")
    if input_path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected a .tif or .tiff input: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    protect_output_location(input_path, checkpoint_path, output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use --overwrite to replace it."
        )

    stack_info = inspect_input_stack(input_path)
    device = resolve_device(args.device)
    if args.cuda_amp and device.type != "cuda":
        raise ValueError("--cuda-amp requires a CUDA inference device.")

    global USE_AMP
    USE_AMP = bool(args.cuda_amp)

    model = WholeSliceNAFNet()
    load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"Input: {input_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | CUDA FP16 AMP: {USE_AMP}")
    print(f"Input ZYX: {(stack_info[0], *stack_info[1])}")
    print(f"Output: {output_dir}")
    print(f"Visualization Z indices: {select_visualization_indices(stack_info[0])}")

    prepare_output(output_dir, args.overwrite)
    output_path, preview_count = process_stack(
        input_path, output_dir, model, device, stack_info
    )

    print(f"Completed: {output_path}")
    print(f"Visualization: {output_dir / 'visualization'} ({preview_count} PNGs)")


if __name__ == "__main__":
    main()

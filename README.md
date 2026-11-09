# Cryo-FIB-SEM artifact suppression and ultrastructural visualization enhancement

This repository provides the **post-acquisition two-stage processing framework** evaluated in our cryo-FIB-SEM manuscript, together with data-preparation scripts, training code, inference entry points, and an optional downstream CebraNET membrane-segmentation workflow.

The workflow uses real charging-artifact information to synthesize Stage-1 training pairs from conventional (RT) FIB-SEM images. A separately trained Stage-2 model uses five consecutive slices for local structural contrast enhancement, denoising, and residual artifact suppression. The two pretrained stages can be applied to real cryo-FIB-SEM data **without dataset-specific model retraining**. The framework is intended to improve visual interpretation and support downstream image analysis; predictions in severely obscured regions should not be treated as independently verified ultrastructure.

## Repository structure

```text
Cryo-FIB-SEM/                 # The directory containing this README.md
├── README.md
├── .gitignore
├── code/
│   ├── preprocessing/       # uint8 conversion, VSNR, global XY alignment
│   ├── stage1/              # 2D artifact synthesis, training, inference
│   ├── stage2/              # 2.5D training and inference
│   └── segmentation/        # optional third-party CebraNET workflow
└── ckpt_model/
    ├── stage1/best.pt
    └── stage2/best.pt
```

Each module has its own README covering only its scripts, requirements, and commands:

- [Preprocessing](code/preprocessing/README.md)
- [Stage-1 artifact suppression](code/stage1/README.md)
- [Stage-2 five-slice enhancement](code/stage2/README.md)
- [Downstream membrane segmentation](code/segmentation/README.md)

The repository deliberately does **not** contain full training data, Stage-1 synthetic TIFF pairs, charging-library intermediates, output stacks, or domain-transfer caches. They are not required to be stored in Git.

## Data and checkpoints

**Source data will be distributed separately**, subject to the applicable reuse and sharing permissions. The planned training-source package contains:

- The exact `JRC/clean_blocks_1024_z16/` train/validation/test blocks (or a documented way to obtain and reconstruct them from the official JRC source).
- The cryo-FIB-SEM `tile_00` source **slices used for charging-template extraction**, plus their corresponding manual charging-core masks. The masks are encoded with `0` for core and `255` for non-core and must match slice identifiers and dimensions.

**Download link:** _to be added after the source-data archive and permissions are finalized_. The package is not included in GitHub. After obtaining it, place its contents in a local `data/` directory (ignored by Git), for example:

```text
data/                         # local only; not committed
├── JRC/clean_blocks_1024_z16/
└── charging_source/          # source TIFF slices and corresponding masks
```

The charging-source directory names are flexible: supply the actual image/mask directories via the Stage-1 command-line options. The JRC blocks are used in **both** training stages. Neither `style_transfer_4levels/` nor the complete `synthesis_data/` nor intermediate charging-library outputs need to be distributed; the Stage-1 scripts regenerate those artifacts.

Pretrained model checkpoints are stored under `ckpt_model/stage1/best.pt` and `ckpt_model/stage2/best.pt`. The optional **CebraNET** TorchScript weights are *not* part of these two models; they are available from the [original Zenodo archive](https://doi.org/10.5281/zenodo.10047209).

Real evaluation volumes used for manuscript figures are distinct from the training-source package; their availability or access conditions will be documented separately. No unpublished internal data are implied to be public merely by the presence of the scripts.

## Workflow

```text
Real cryo-FIB-SEM volume
        │
        ▼
Input preparation: uint8 conversion → optional VSNR → global XY alignment if needed
        │
        ▼
Stage-1: 2D large-scale artifact suppression
        │
        ▼
Stage-2: 2.5D local structural/membrane contrast enhancement
        │
        ▼
Processed volume → optional CebraNET membrane predictions / 3D visualization
```

For the manuscript's displayed real datasets, VSNR was used before learned processing. The three in-house volumes were globally aligned; EMPIAR-11275 was already aligned and did not undergo additional global alignment. Stage-2 includes mandatory full-slice intensity preprocessing and an **optional** intensity/Fourier-statistics domain-transfer switch; **domain transfer was enabled for the real-data results in the manuscript**. None of these steps creates additional acquired Z slices.

## Quick inference example

All commands assume you run them **from the repository root** (the folder containing this README), in the separately installed environments described in the module READMEs. First prepare a uint8 ZYX TIFF stack as needed using [`code/preprocessing/`](code/preprocessing/README.md). The example below assumes this preparation is already complete.

```bash
# Stage-1: suppress large-scale artifacts
python code/stage1/step5_inference.py \
  --input /path/to/prepared_uint8_stack.tif \
  --checkpoint ckpt_model/stage1/best.pt \
  --output-dir experiments/pipeline_stage1 \
  --device mps

# Stage-2: the manuscript's default (domain transfer ON)
python code/stage2/step2_inference_adaptive.py \
  --input experiments/pipeline_stage1/output_stack.tif \
  --checkpoint ckpt_model/stage2/best.pt \
  --reference-train data/JRC/clean_blocks_1024_z16/train \
  --cache-dir experiments/stage2_domain_cache \
  --output-dir experiments/pipeline_stage2 \
  --device mps \
  --domain-transfer
```

Replace `--device mps` with `--device cuda` on a compatible NVIDIA GPU. To run Stage-2 without domain transfer, supply `--no-domain-transfer` and omit `--reference-train`; this optional mode is **not** the configuration used for the reported real-data figures.

**Important:** Stage-2 domain transfer in this implementation needs reference statistics constructed from the **clean JRC training split**. If the separate source-data package is not yet available, use the optional `--no-domain-transfer` mode for an initial inference check, not as an exact reproduction of the manuscript setting.

## Reproducing training and evaluation

Stage-1 data generation follows this reproducible chain:

```text
JRC clean blocks + annotated charging-source images
       ├── charging-source analysis → template library
       └── four-level clean-data preparation
                           ▼
                synthetic paired Stage-1 data
                           ▼
                     Stage-1 training
```

Stage-2 training independently uses the original JRC clean blocks to construct five-slice synthetic examples. See the module READMEs for commands, fixed split definitions, training defaults, and generated outputs. The source blocks are partitioned into **16 train / 3 validation / 3 held-out test** blocks for both stages. Stage-1 test pairs and Stage-2 test patches are evaluated against **constructed synthetic references**, not paired artifact-free cryo-FIB-SEM acquisitions.

The downstream [CebraNET workflow](code/segmentation/README.md) is a separate use of an existing pretrained model and does not provide independent ground-truth segmentation labels. For the three in-house analyzed volumes the voxel spacing is `(X,Y,Z)=(10,10,20)` nm after XY binning; the public EMPIAR-11275 data use `(5,5,20)` nm. Segmentation geometry arguments must match the dataset (see its README).

## Reproducibility notes

- The Stage-1 and Stage-2 models are independently trained. Stage-1 does not perform Stage-2 domain transfer.
- The supplied code uses explicit command-line arguments for data, reference, checkpoint, and output paths; do not assume the ignored local `data/` directory exists after cloning.
- Generated caches and results should remain outside the tracked source-code modules.
- The published figures' qualitative structural checks should not be interpreted as independently established accuracy in fully obscured regions.
- The public source-data URL, full real-data availability statement, and release license will be finalized before publication. Please verify permission before redistributing any sample images.

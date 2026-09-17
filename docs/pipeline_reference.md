# Glue Factory — Pipeline Reference

Single source of truth for every pipeline stage, from raw images through evaluation.

---

## Table of Contents

- [Conventions](#conventions)
- [Scripts Directory](#scripts-directory)
- [Pipeline A — SuperPoint Training (Homography Adaptation)](#pipeline-a--superpoint-training-homography-adaptation)
- [Pipeline B — SuperPoint Training (SLAM Pose-Based)](#pipeline-b--superpoint-training-slam-pose-based)
- [SuperGlue Training](#superglue-training)
- [LightGlue Training](#lightglue-training)
- [Model Export](#model-export)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Monitoring](#monitoring)
- [Dataset Format Spec](#dataset-format-spec)

---

## Conventions

| Concept | Canonical form |
| :--- | :--- |
| Training runs | `{model}_{task}_run` e.g. `superpoint_slam_run`, `superglue_slam_run`, `lightglue_slam_run` |
| Quick debug runs | `{model}_sample_run` |
| Exported model files | `superpoint_slam.pt`, `superglue_slam.pt`, `lightglue_slam.pt` |
| Configs | `{extractor}+{matcher}_{task}.yaml` |

**Data paths (docs/code only):**
- SLAM dataset → `data/output/slam` (canonical). If your data is elsewhere (e.g. `data/output/map2`), pass `data.data_dir=output/map2` as a CLI override or create a symlink.
- Custom LiDAR dataset → `data/output/dataset`
- Sample dataset → `data/output/sample_data` / `data/output/sample_slam`

---

## Scripts Directory

All helper tools live in `gluefactory/scripts/` and are run via
`python3 -m gluefactory.scripts.<name>`.

| Script | Purpose |
| :--- | :--- |
| `visualize_dataset` | Unified visualization CLI — generates labels, pairs, training, custom, and attention overlays |
| `run_inference` | SuperPoint / SuperGlue / LightGlue inference and visualization (checkpoint or `.pt`) |
| `export_features` | Batch-export keypoints + descriptors to HDF5 for HPatches/eth3d benchmarks |
| `export_slam_features` | Export features at consensus keypoints (SLAM SuperGlue/LightGlue training) |
| `export_model` | Convert a trained checkpoint to a standalone `.pt` for deployment |
| `prepare_slam_labels` | Generate SLAM-format pseudo-labels with pose-driven 3D warps |
| `generate_slam_pairs` | Build SLAM pair list + ground-truth from pose file and image dir |
| `boost_h5_scores` | Scale matching scores in an exported HDF5 |
| `filter_h5_file` | Filter keypoints by confidence threshold |

---

## Pipeline A — SuperPoint Training (Homography Adaptation)

Trains SuperPoint on reflectivity LiDAR images using homographic + 3D-projective
adaptation with pre-computed pseudo-label targets.

**Config:** `gluefactory/configs/superpoint_custom_homography.yaml`

### A1 — Prepare Pseudo-Labels

```bash
# Full custom dataset
python3 -m gluefactory.scripts.prepare_slam_labels \
    --data_dir data/output/dataset \
    --warp_mode 3d \
    --use_gpu

# Sample dataset (fast iteration)
python3 -m gluefactory.scripts.prepare_slam_labels \
    --data_dir data/output/sample_data \
    --warp_mode 3d \
    --use_gpu
```

Output: `<data_dir>/exports/pseudo_labels_slam.h5`

Key CLI flags: `--num_warps 14`, `--warp_mode 3d`, `--weights <trained_sp_ckpt>`,
`--use_gpu`. Camera calibration YAML must exist under `<data_dir>/images/calib/`.

### A2 — Train SuperPoint

```bash
# Fresh training
python3 -m gluefactory.train superpoint_custom_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml

# Resume interrupted run
python3 -m gluefactory.train superpoint_custom_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml \
    --restore

# Fine-tune from official Magic Leap weights
python3 -m gluefactory.train superpoint_custom_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml \
    train.load_experiment=superpoint_open
```

Output: `outputs/training/superpoint_custom_run/checkpoint_best.tar`

### A3 — Quick Debug Run (Small Sample)

```bash
python3 -m gluefactory.train superpoint_sample_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml \
    data.data_dir=output/sample_data \
    data.image_list=custom_image_list.txt \
    data.train_size=15 data.val_size=4 \
    data.load_features.path=output/sample_data/exports/pseudo_labels.h5 \
    train.epochs=50
```

### Training CLI Flags

| Flag | Effect |
| :--- | :--- |
| `--restore` | Resume from last checkpoint |
| `--distributed` | Multi-GPU DDP training |
| `--overfit` | Train and eval on a single batch (debug) |
| `--mixed_precision float16` | FP16 mixed precision |
| `--compile default` | `torch.compile` for speed |
| `--print_arch` | Print full model architecture |
| `--no_eval_0` | Skip validation at iteration 0 |
| `dotlist` | Override config keys inline, e.g. `train.lr=5e-5` |

---

## Pipeline B — SuperPoint Training (SLAM Pose-Based)

Trains SuperPoint using SLAM pose data (TUM format) for pose-driven 3D warps.

**Config:** `gluefactory/configs/superpoint_slam.yaml`

### B1 — Prepare SLAM Pair List

```bash
python3 -m gluefactory.scripts.generate_slam_pairs \
    --data_dir data/output/slam \
    --poses_file poses_odom_RGBD_slam.txt \
    --output_dir data/output/slam/exports
```

Output: `exports/slam_pairs.csv`

### B2 — Train SuperPoint

```bash
python3 -m gluefactory.train superpoint_slam_run \
    --conf gluefactory/configs/superpoint_slam.yaml
```

Output: `outputs/training/superpoint_slam_run/checkpoint_best.tar`

---

## SuperGlue Training

SuperGlue requires a frozen SuperPoint feature cache (HDF5) — it does not re-run
the extractor at training time.

### Homography Dataset

**Config:** `gluefactory/configs/superpoint_custom+superglue_homography.yaml`

```bash
python3 -m gluefactory.train superglue_custom_run \
    --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml

# Resume
python3 -m gluefactory.train superglue_custom_run \
    --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml \
    --restore

# Fine-tune from official SuperGlue
python3 -m gluefactory.train superglue_custom_run \
    --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml \
    train.load_experiment=superglue_outdoor
```

Output: `outputs/training/superglue_custom_run/checkpoint_best.tar`

### SLAM Dataset

**Config:** `gluefactory/configs/superpoint+superglue_slam.yaml`

First generate the consensus feature cache:

```bash
# Generate pseudo-labels (SLAM format)
python3 -m gluefactory.scripts.prepare_slam_labels \
    --data_dir data/output/slam \
    --warp_mode 3d --use_gpu

# Export features at consensus keypoints (HDF5)
python3 -m gluefactory.scripts.export_slam_features \
    --dataset output/slam \
    --pseudo_labels_h5 data/output/slam/exports/pseudo_labels_slam.h5 \
    --weights outputs/training/superpoint_slam_run/checkpoint_best.tar \
    --output_h5 data/output/slam/exports/slam_consensus_SP.h5
```

Then train:

```bash
python3 -m gluefactory.train superglue_slam_run \
    --conf gluefactory/configs/superpoint+superglue_slam.yaml
```

Output: `outputs/training/superglue_slam_run/checkpoint_best.tar`

---

## LightGlue Training

LightGlue uses the same data format as SuperGlue (no matcher re-run at train time).

### Homography Dataset

**Config:** `gluefactory/configs/superpoint_custom+lightglue_homography.yaml`

```bash
python3 -m gluefactory.train lightglue_custom_run \
    --conf gluefactory/configs/superpoint_custom+lightglue_homography.yaml
```

### SLAM Dataset

**Config:** `gluefactory/configs/superpoint+lightglue_slam.yaml`

```bash
python3 -m gluefactory.train lightglue_slam_run \
    --conf gluefactory/configs/superpoint+lightglue_slam.yaml
```

Output: `outputs/training/lightglue_slam_run/checkpoint_best.tar`

---

## Model Export

Converts trained checkpoints to standalone `.pt` files for deployment.

```bash
# SuperPoint
python3 -m gluefactory.scripts.export_model \
    --config gluefactory/configs/superpoint_slam.yaml \
    --checkpoint outputs/training/superpoint_slam_run/checkpoint_best.tar \
    --output superpoint_slam.pt \
    --match_input_size --simplify --dynamic

# SuperGlue (produces optimized .pt)
python3 -m gluefactory.scripts.export_model \
    --config gluefactory/configs/superpoint+superglue_slam.yaml \
    --checkpoint outputs/training/superglue_slam_run/checkpoint_best.tar \
    --output superglue_slam.pt

# LightGlue
python3 -m gluefactory.scripts.export_model \
    --config gluefactory/configs/superpoint+lightglue_slam.yaml \
    --checkpoint outputs/training/lightglue_slam_run/checkpoint_best.tar \
    --output lightglue_slam.pt
```

See `docs/evaluation.md` for deployment examples.

---

## Evaluation

### HPatches

```bash
# Custom SuperPoint only
python3 -m gluefactory.eval.hpatches \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint-open+NN

# Custom SuperPoint + NN (homography metric)
python3 -m gluefactory.eval.homographies \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint-open+NN
```

### SLAM

```bash
python3 -m gluefactory.scripts.run_inference \
    --config gluefactory/configs/superpoint+superglue_slam.yaml \
    --checkpoint outputs/training/superglue_slam_run/checkpoint_best.tar \
    --backend checkpoint \
    --split test
```

### ZEB-REMO / LiDAR

See `docs/evaluation.md` for ZEB-REMO benchmark commands.

---

## Visualization

All dataset-creation visualizations go through the unified CLI:

```bash
python3 -m gluefactory.scripts.visualize_dataset <kind> [options]
#   kinds: labels | pairs | training | custom | attention

# Labels — keypoints + scores + match pairs
python3 -m gluefactory.scripts.visualize_dataset labels --data_dir data/output/slam --num_vis 50

# Pairs — sparse correspondences
python3 -m gluefactory.scripts.visualize_dataset pairs --data_dir data/output/slam

# Training — SuperPoint/LightGlue match overlays
python3 -m gluefactory.scripts.visualize_dataset training --data_dir data/output/slam

# Custom — custom backend visualization
python3 -m gluefactory.scripts.visualize_dataset custom --data_dir data/output/slam

# Attention — SuperPoint attention patterns (checkpoints only)
python3 -m gluefactory.scripts.visualize_dataset attention \
    --data_dir data/output/slam \
    --checkpoint outputs/training/superglue_slam_run/checkpoint_best.tar
```

See `docs/attention_visualization.md` for attention analysis.

### Inference Visualisation (run_inference)

```bash
# Keypoint detection only
python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher none \
    --extractor_ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \
    --input data/output/slam/images/rgb \
    --output all --output_dir data/output/slam/visualizations/keypoints

# Full matching (SuperPoint + LightGlue)
python3 -m gluefactory.scripts.run_inference \
    --config gluefactory/configs/superpoint+lightglue_slam.yaml \
    --backend checkpoint \
    --input data/output/slam/images/rgb \
    --output all
```

---

## Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/training/
```

### Inspect HDF5

```bash
python3 -c "
import h5py, sys
f = h5py.File(sys.argv[1], 'r')
keys = list(f.keys())
print(f'Entries: {len(keys)}, first: {keys[0]}')
for n, ds in f[keys[0]].items():
    print(f'  {n}: {ds.shape} {ds.dtype}')
" data/output/slam/exports/slam_consensus_SP.h5
```

---

## Dataset Format Spec

### SLAM Dataset (`slam_posed_images`)

```
data/output/slam/
├── images/
│   └── rgb/
│       ├── frame_000000.png
│       └── ...
└── exports/
    ├── slam_pairs.csv          ← generated by generate_slam_pairs
    └── pseudo_labels_slam.h5   ← generated by prepare_slam_labels
```

The H5 contains per-pair entries with: `image_path0`, `image_path1` (str), `keypoints0`,
`scores0`, `descriptors0`, `keypoints1`, `scores1`, `descriptors1` (float tensors),
`T_0to1` (4×4 pose), `image_size0`, `image_size1` (2D int).

### Homography Dataset

```
data/output/dataset/
├── images/
│   ├── nearir/
│   ├── range/
│   ├── reflectivity/
│   └── signal/
├── calib/
│   └── <frame>_calib.yaml
├── custom_image_list.txt
└── exports/
    ├── pseudo_labels.h5
    └── custom_dataset_consensus_SP.h5
```

The image list contains one relative path per line (from `images/`):

```
nearir/img_00001.png
range/img_00001.png
reflectivity/img_00001.png
signal/img_00001.png
...
```

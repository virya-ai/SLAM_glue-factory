# Naming Conventions

## Training Runs

```
{model}_{task}_run
```

| Pipeline | Runs |
| :--- | :--- |
| SLAM (pose-based) | `superpoint_slam_run`, `superglue_slam_run`, `lightglue_slam_run` |
| Homography (custom LiDAR) | `superpoint_custom_run`, `superglue_custom_run`, `lightglue_custom_run` |
| Quick debug/overfit | `{model}_sample_run` |

Obsolete forms to never use: `_force_true`, `_model_tr_`, `_finetune`, `_slam_ran_`, config-as-run-name.

## Exported Models

```
{model}_{task}.pt
```

Examples: `superpoint_slam.pt`, `superglue_slam.pt`, `lightglue_slam.pt`

## Config Files

```
{extractor}+{matcher}_{dataset_task}.yaml
```

- **SLAM dataset** → task `_slam`
- **Homography dataset** (LiDAR custom) → task `_homography`
- `+` joins extractor and matcher (e.g. `superpoint+superglue_slam.yaml`)
- `-` for upstream naming convention only (e.g. `lightglue_official`)
- `_` separates dataset/task qualifier

| Format | Example |
| :--- | :--- |
| SP detector (homography) | `superpoint_custom_homography.yaml` |
| SP + LightGlue (homography) | `superpoint_custom+lightglue_homography.yaml` |
| SP + SuperGlue (SLAM) | `superpoint+superglue_slam.yaml` |
| SP open + NN baseline | `superpoint-open+NN.yaml` |

## Data Paths (in code/docs, relative to `DATA_PATH = data/`)

| Pipeline | Canonical `data_dir` |
| :--- | :--- |
| SLAM | `output/slam` |
| Custom LiDAR homography | `output/custom_dataset` |
| Sample (fast iteration) | `output/sample_data` / `output/sample_slam` |

Physical on-disk directories may differ from the canonical name above. Use CLI
overrides (`data.data_dir=output/map2`) or symlinks if the real data lives elsewhere.

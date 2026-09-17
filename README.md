# Glue Factory
Glue Factory is CVG's library for training and evaluating deep neural networks that extract and match local visual features. It enables you to:
- Train and evaluate feature extractors (SuperPoint) and matchers (SuperGlue, LightGlue)
- Run custom training pipelines (homography-based and SLAM pose-based)
- Evaluate on standard benchmarks like HPatches, MegaDepth-1500, Scannet-1500, and ETH3D

<p align="center">
  <a href="https://github.com/cvg/LightGlue"><img src="docs/lightglue_matches.svg" width="60%"/></a>
  <br /><em>Point matching with LightGlue.</em>
</p>

## Installation
```bash
git clone https://github.com/cvg/glue-factory
cd glue-factory
python3 -m pip install -e .  # editable mode
```
Some advanced features might require:
```bash
python3 -m pip install -e .[extra]
```

## License
The code and trained models in Glue Factory are released with an Apache-2.0 license. This includes LightGlue and an [open version of SuperPoint](https://github.com/rpautrat/SuperPoint). Third-party models that are not compatible with this license, such as SuperPoint (original) and SuperGlue, are provided in `gluefactory/models/nonfree`, where each model might follow its own, restrictive license.

## Custom Pipelines

This fork extends Glue Factory with SLAM pose-based and custom LiDAR training pipelines. See the full reference:

- **[Pipeline Reference](docs/pipeline_reference.md)** — all commands from raw images through evaluation
- **[Evaluation Guide](docs/evaluation.md)** — benchmark instructions and results
- **[Config Parameters](docs/config_parameters_guide.md)** — all config keys explained
- **[System Architecture](docs/architecture.md)** — codebase layout and module relationships

### Naming Conventions

| Concept | Form |
| :--- | :--- |
| Training runs | `{model}_{task}_run` e.g. `superpoint_slam_run`, `superglue_slam_run`, `lightglue_slam_run` |
| Quick debug runs | `{model}_sample_run` |
| Exported models | `superpoint_slam.pt`, `superglue_slam.pt`, `lightglue_slam.pt` |
| Configs | `{extractor}+{matcher}_{task}.yaml` |

## Evaluation

Running the evaluation commands automatically downloads the dataset, by default to `data/`.

### HPatches
```bash
# LightGlue (SuperPoint open + LightGlue)
python -m gluefactory.eval.hpatches --conf superpoint+lightglue_megadepth --overwrite

# SuperPoint (open) + NN baseline
python -m gluefactory.eval.hpatches --conf superpoint-open+NN --overwrite
```

### MegaDepth-1500
```bash
python -m gluefactory.eval.megadepth1500 --conf superpoint+lightglue_megadepth
# or with PoseLib estimator
python -m gluefactory.eval.megadepth1500 --conf superpoint+lightglue_megadepth \
    eval.estimator=poselib eval.ransac_th=2.0
```

### Scannet-1500
```bash
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue_megadepth
# adaptive variant
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue_megadepth \
    model.matcher.{depth_confidence=0.95,width_confidence=0.95}
```

### Visual inspection
```bash
python -m gluefactory.eval.inspect hpatches superpoint+lightglue_megadepth
# compare multiple methods
python -m gluefactory.eval.inspect hpatches superpoint+lightglue_megadepth superpoint-open+NN
```

Detailed evaluation instructions: [docs/evaluation.md](docs/evaluation.md).

## Training

### LightGlue

Pre-train on the homography dataset (requires ~450 GB, 2x 3090 GPUs for default batch size):
```bash
python -m gluefactory.train sp+lg_homography \
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml
# reduce batch size for smaller GPUs
python -m gluefactory.train sp+lg_homography \
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml \
    data.batch_size=32
```

Fine-tune on MegaDepth:
```bash
python -m gluefactory.train sp+lg_megadepth \
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml \
    train.load_experiment=sp+lg_homography
```

With cached features (saves GPU time, requires ~150 GB):
```bash
python -m gluefactory.scripts.export_features megadepth --method sp --num_workers 8
python -m gluefactory.train sp+lg_megadepth \
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml \
    train.load_experiment=sp+lg_homography \
    data.load_features.do=True
```

### SuperPoint (Custom Training)

See [Pipeline Reference](docs/pipeline_reference.md) for SLAM and homography-based training.

## Available Models

| Model | Training | Evaluation |
| --- | --- | --- |
| [LightGlue](https://github.com/cvg/LightGlue) | ✅ | ✅ |
| [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) | ✅ | ✅ |

Feature extractors:

| Model | Config prefix |
| --- | --- |
| [SuperPoint (open)](https://github.com/rpautrat/SuperPoint) | `superpoint-open+*` |
| [SuperPoint (original)](https://github.com/magicleap/SuperPointPretrainedNetwork) | `superpoint+*` |
| SIFT (via [pycolmap](https://github.com/colmap/pycolmap)) | `sift+*` |
| [ALIKED](https://github.com/Shiaoming/ALIKED) | `aliked+*` |
| [DISK](https://github.com/cvlab-epfl/disk) | `disk+*` |

## BibTeX Citation
Please consider citing the following papers if you found this library useful:
```bibtex
@InProceedings{lindenberger_2023_lightglue,
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```
```bibtex
@InProceedings{sarlin_2020_superglue,
  title     = {{SuperGlue: Learning Feature Matching with Graph Neural Networks}},
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```

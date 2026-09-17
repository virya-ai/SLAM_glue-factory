"""
Export local features (keypoints, descriptors, scores) from a dataset to H5.

Iterates over a dataset split, runs the configured extractor (SuperPoint,
DISK, ALIKED, SIFT, KeyNet, …), and writes per-image features to an HDF5 file
that can be loaded by the training pipeline via the load_features cache
mechanism. Supports both custom image-folder datasets and MegaDepth.

Usage:
    # Custom dataset (image_folder):
    python -m gluefactory.scripts.export_features my_dataset --method sp \
        --export_prefix sp_r1600_

    # Custom-trained SuperPoint weights:
    python -m gluefactory.scripts.export_features my_dataset --method sp_custom \
        --weights outputs/training/superpoint_custom_run/checkpoint_best.tar

    # MegaDepth (per-scene exports, optionally with keypoint depth):
    python -m gluefactory.scripts.export_features megadepth --method sp_open \
        --scenes train_scenes.txt --export_sparse_depth
"""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..geometry.depth import sample_depth
from ..models import get_model
from ..settings import DATA_PATH
from ..utils.export_predictions import export_predictions

resize = 1600
n_kpts = 2048
configs = {
    "sp": {
        "name": f"r{resize}_SP-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "gluefactory_nonfree.superpoint",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "sp_open": {
        "name": f"r{resize}_SP-open-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "extractors.superpoint_open",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "sp_custom": {
        "name": f"r{resize}_SP-custom-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "extractors.superpoint_open",
            "weights": None,  # filled from --weights
        },
    },
    "sift": {
        "name": f"r{resize}_SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
        },
    },
    "sift_pycolmap": {
        "name": f"r{resize}_pycolmap-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap",
        },
    },
    "sift_pycolmap_gpu": {
        "name": f"r{resize}_pycolmap-SIFTGPU-nms3-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap_cuda",
            "nms_radius": 3,
        },
    },
    "keynet": {
        "name": f"r{resize}_KeyNetAffNetHardNet-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.keynet_affnet_hardnet",
            "max_num_keypoints": n_kpts,
        },
    },
    "disk": {
        "name": f"r{resize}_DISK-k{n_kpts}-nms5",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.disk_kornia",
            "max_num_keypoints": n_kpts,
        },
    },
    "aliked": {
        "name": f"r{resize}_ALIKED-k{n_kpts}-n16",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.aliked",
            "max_num_keypoints": n_kpts,
        },
    },
}


def get_kp_depth(pred, data):
    d, valid = sample_depth(pred["keypoints"], data["depth"])
    return {"depth_keypoints": d, "valid_depth_keypoints": valid}


def export_megadepth(args):
    data_root = Path(DATA_PATH, "megadepth/Undistorted_SfM")
    export_root = Path(DATA_PATH, "exports", "megadepth-undist-depth-" + configs[args.method]["name"])
    export_root.mkdir(parents=True, exist_ok=True)

    if args.scenes is None:
        scenes = [p.name for p in data_root.iterdir() if p.is_dir()]
    else:
        with open(DATA_PATH / "megadepth" / args.scenes, "r") as f:
            scenes = f.read().split()

    for i, scene in enumerate(scenes):
        print(f"{i} / {len(scenes)}", scene)
        feature_file = export_root / (scene + ".h5")
        if feature_file.exists():
            continue
        if not (data_root / scene / "images").exists():
            logging.info("Skip " + scene)
            continue

        conf = OmegaConf.create(
            {
                "data": {
                    "name": "megadepth",
                    "views": 1,
                    "grayscale": configs[args.method]["gray"],
                    "preprocessing": {"resize": resize, "side": "long"},
                    "batch_size": 1,
                    "num_workers": args.num_workers,
                    "read_depth": True,
                    "train_split": [scene],
                    "train_num_per_scene": None,
                },
                "split": "train",
                "model": configs[args.method]["conf"],
            }
        )
        keys = configs[args.method]["keys"]
        dataset = get_dataset(conf.data.name)(conf.data)
        loader = dataset.get_data_loader(conf.split)
        model = get_model(conf.model.name)(conf.model).to(device).eval()

        if args.export_sparse_depth:
            callback_fn = get_kp_depth
            keys = keys + ["depth_keypoints", "valid_depth_keypoints"]
        else:
            callback_fn = None
        logging.info(f"Export local features for scene {scene}")
        export_predictions(
            loader, model, feature_file, as_half=True, keys=keys, callback_fn=callback_fn
        )


def export_image_folder(args):
    data_root = Path(DATA_PATH, args.dataset)
    feature_file = Path(
        DATA_PATH, "exports", args.export_prefix + configs[args.method]["name"] + ".h5"
    )
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Export local features for dataset {args.dataset} to {feature_file}")

    image_list_path = data_root / "custom_image_list.txt"
    images = image_list_path if image_list_path.exists() else data_root / "images"
    root_folder = data_root / "images"

    conf = OmegaConf.create(
        {
            "data": {
                "name": "image_folder",
                "grayscale": configs[args.method]["gray"],
                "preprocessing": {"resize": None if args.method == "sp_custom" else resize},
                "images": str(images),
                "root_folder": str(root_folder),
                "batch_size": 1,
                "num_workers": args.num_workers,
            },
            "split": "train",
            "model": configs[args.method]["conf"],
        }
    )
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(conf.split)
    model = get_model(conf.model.name)(conf.model).to(device).eval()
    export_predictions(
        loader, model, feature_file, as_half=True, keys=configs[args.method]["keys"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str,
        help="Dataset name (use 'megadepth' for the MegaDepth export path).",
    )
    parser.add_argument("--method", type=str, default="sp", choices=list(configs))
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint path for method 'sp_custom'.")
    parser.add_argument("--export_prefix", type=str, default="")
    parser.add_argument("--scenes", type=str, default=None,
                        help="File listing MegaDepth scenes.")
    parser.add_argument("--export_sparse_depth", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if args.weights:
        configs[args.method]["conf"]["weights"] = args.weights
    elif args.method == "sp_custom":
        parser.error("--weights is required for method 'sp_custom'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "megadepth":
        export_megadepth(args)
    else:
        export_image_folder(args)
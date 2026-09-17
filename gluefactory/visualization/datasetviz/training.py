"""kind=training — visualize training-pair samples: keypoints + GT matches.

Loads a two-view training dataset (e.g. homographies or superglue_slam) from
its YAML config, runs the configured extractor on random samples, computes
ground-truth matches with the config's ground_truth matcher, and renders a
side-by-side visualization per sample. Also writes a CSV of per-sample
statistics (keypoint counts and GT match counts) alongside the PNGs.

Consolidates the former visualize_training_dataset.py / visualize_training_pair.py
entry points — a single sample (``--num_vis 1``) reproduces the old quick
single-pair check, larger counts produce the batch + stats run.

Usage:
    python -m gluefactory.scripts.visualize_dataset training \\
        --conf gluefactory/configs/superpoint+superglue_slam.yaml \\
        --split train --num_vis 20
"""

import csv
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.visualization.datasetviz.base import make_out_dir
from gluefactory.visualization import viz2d

logger = logging.getLogger(__name__)


def _tensor_to_uint8(img_tensor):
    img = img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def _get_gt_matches(gt_matcher, sample, kp0, kp1, device):
    data = {
        **sample,
        "keypoints0": kp0.unsqueeze(0),
        "keypoints1": kp1.unsqueeze(0),
    }
    if data.get("H_0to1") is not None and not isinstance(data["H_0to1"], torch.Tensor):
        data["H_0to1"] = torch.from_numpy(np.asarray(data["H_0to1"])).float()
    if data.get("T_0to1") is not None and not isinstance(data["T_0to1"], torch.Tensor):
        data["T_0to1"] = torch.from_numpy(np.asarray(data["T_0to1"])).float()

    gt = gt_matcher(data)
    matches0 = gt["matches0"][0].cpu().numpy()
    valid = np.where(matches0 >= 0)[0]
    return [(int(i), int(matches0[i])) for i in valid]


def _save_visualization(out_file, img0, img1, kp0, kp1, matches, title,
                        max_matches=300):
    vis = viz2d.draw_matches(img0, img1, kp0, kp1, matches[: max_matches])
    plt.figure(figsize=(18, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_training(conf_path, split="train", num_vis=50, output=None,
                       max_matches=300):
    """Render ``num_vis`` training-pair samples with GT matches + a CSV.

    Args:
        conf_path: YAML config with ``data``, ``model.extractor`` and
                   ``model.ground_truth`` sections.
        split:     dataset split to sample from ("train"/"val"/"test").
        num_vis:   number of pairs to visualize.
        output:    output directory (default
                   ``outputs/visualizations/training_dataset``).
        max_matches: cap on GT match lines drawn per pair.

    Returns:
        directory containing the PNGs and training_statistics.csv.
    """
    out_dir = make_out_dir(output or "outputs/visualizations/training_dataset")

    conf = OmegaConf.load(conf_path)
    dataset = get_dataset(conf.data.name)(conf.data)
    split_ds = dataset.get_dataset(split)

    logger.info(
        f"Dataset: {conf.data.name} ({len(split_ds)} samples, split={split})"
    )

    extractor_name = conf.model.extractor.get("name", None)
    sp = None
    if extractor_name:
        logger.info(f"Extractor: {extractor_name}")
        sp = get_model(extractor_name)(conf.model.extractor).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sp = sp.to(device)
    else:
        logger.info(
            "Extractor: none configured (using pre-extracted keypoints from the "
            "sample, e.g. the load_features path)."
        )

    gt_matcher = get_model(conf.model.ground_truth.name)(conf.model.ground_truth)

    stats = []
    n = max(1, int(num_vis))
    for step in range(n):
        idx = step % len(split_ds)
        sample = split_ds[idx]

        if sp is None:
            kp0 = kp1 = None
            for key in ("keypoints0",):
                if key in sample and isinstance(sample[key], dict) and "keypoints" in sample[key]:
                    kp0 = sample[key]["keypoints"][0]
                    break
            if "keypoints1" in sample and isinstance(sample["keypoints1"], dict):
                kp1 = sample["keypoints1"]["keypoints"][0]
            if kp0 is None and "cache" in sample:
                cache = sample["cache"]
                for lk in ("keypoints0", "keypoints1"):
                    if lk in cache and isinstance(cache[lk], dict):
                        vals = cache[lk].get("keypoints")
                        if vals is not None:
                            v = vals[0]
                            if lk == "keypoints0":
                                kp0 = v
                            else:
                                kp1 = v
            if kp0 is None or kp1 is None:
                raise RuntimeError(
                    "No extractor configured and the dataset sample does not "
                    "contain pre-extracted keypoints0/keypoints1 (enable "
                    "'data.load_features' or set 'model.extractor')."
                )
        else:
            img0 = sample["view0"]["image"].unsqueeze(0).to(device)
            img1 = sample["view1"]["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred0 = sp({"image": img0})
                pred1 = sp({"image": img1})
            kp0, kp1 = pred0["keypoints"][0], pred1["keypoints"][0]

        matches = _get_gt_matches(gt_matcher, sample, kp0, kp1, device)
        num_matches = len(matches)
        stats.append([step, idx, sample["name"], kp0.shape[0], kp1.shape[0], num_matches])
        logger.info(
            f"[{step + 1}/{n}] {sample['name']} KP0={kp0.shape[0]} "
            f"KP1={kp1.shape[0]} GT={num_matches}"
        )

        out_file = out_dir / f"iter_{step:05d}.png"
        _save_visualization(
            out_file,
            _tensor_to_uint8(sample["view0"]["image"]),
            _tensor_to_uint8(sample["view1"]["image"]),
            kp0.cpu().numpy(),
            kp1.cpu().numpy(),
            matches,
            f"Step={step} GT Matches={num_matches}",
            max_matches,
        )

    csv_file = out_dir / "training_statistics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "dataset_idx", "image_name", "kp0", "kp1", "gt_matches"])
        writer.writerows(stats)

    counts = np.array([s[5] for s in stats])
    logger.info(
        f"Min GT Matches: {counts.min()} | Max: {counts.max()} | "
        f"Mean: {counts.mean():.1f} | Median: {np.median(counts):.1f}"
    )
    logger.info(f"Visualizations + stats saved to: {out_dir}")
    return str(out_dir)


if __name__ == "__main__":
    # Keep this module runnable for quick debugging.
    visualize_training("gluefactory/configs/superpoint+superglue_slam.yaml")
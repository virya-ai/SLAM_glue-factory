"""kind=labels — visualize cached pseudo-label keypoints overlaid on images.

Reads an H5 cache of per-image keypoints (e.g. the pseudo-labels produced by
``gluefactory.scripts.prepare_slam_labels``), draws the bounding frames with a
score-coloured keypoint scatter, and generates a browsable HTML grid. This is
the visualization that dataset creation auto-runs unless ``--num_vis 0``.

Usage:
    python -m gluefactory.scripts.visualize_dataset labels \\
        --data_dir data/output/slam --num_vis 50
"""

import logging
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gluefactory.visualization.datasetviz.base import (
    finalize_html_grid,
    load_h5_kpts,
    make_out_dir,
)
from gluefactory.visualization import viz2d

logger = logging.getLogger(__name__)


def visualize_labels(data_dir, h5_path=None, modality="rgb", num_vis=50,
                     score_thresh=0.01, output_dir=None):
    """Render per-image keypoint overlays from an H5 pseudo-label cache.

    Args:
        data_dir: dataset root (contains ``images/<modality>`` and, by
                  default, ``exports/pseudo_labels_slam.h5``).
        h5_path:  override the pseudo-label H5 path.
        modality: image subdirectory holding the frames (default "rgb").
        num_vis:  maximum number of images to visualize (0 = all).
        score_thresh: keypoint scores below this are not drawn.
        output_dir: where PNGs + index.html are written (default
                    ``<data_dir>/visualizations/kpt_labels``).

    Returns:
        list of saved PNG paths.
    """
    data_dir = Path(data_dir)
    h5_path = Path(h5_path) if h5_path else data_dir / "exports/pseudo_labels_slam.h5"
    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        return []

    rgb_dir = data_dir / "images" / modality
    with h5py.File(h5_path, "r") as f:
        image_names = list(f.keys())
    if not image_names:
        logger.warning(f"No entries found in {h5_path}")
        return []

    if num_vis and num_vis > 0 and len(image_names) > num_vis:
        np.random.seed(42)
        np.random.shuffle(image_names)
        image_names = image_names[: num_vis]

    out_dir = make_out_dir(output_dir or data_dir / "visualizations/kpt_labels")

    saved = []
    with h5py.File(h5_path, "r") as f:
        for name in tqdm(image_names, desc="Visualizing labels"):
            rgb_path = rgb_dir / name
            if not rgb_path.exists():
                logger.warning(f"Image {name} not found in {rgb_dir}, skipping.")
                continue
            img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read image {rgb_path}, skipping.")
                continue
            kpts, scores = load_h5_kpts(f, name, score_thresh)
            out_path = out_dir / name
            viz2d.plot_keypoints_overlay(
                img, kpts, scores, out_path,
                title=f"{name} ({len(kpts)} keypoints)", colorbar=True,
            )
            saved.append(out_path)

    finalize_html_grid(saved, out_dir, "SLAM Pseudo-Labels Visualization")
    logger.info(f"Saved {len(saved)} label visualization(s) to {out_dir}")
    return [str(p) for p in saved]


if __name__ == "__main__":
    # Keep this module runnable for quick debugging.
    visualize_labels("data/output/sample_slam")
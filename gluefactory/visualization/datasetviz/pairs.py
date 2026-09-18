"""kind=pairs — visualize image pairs with SuperPoint keypoints overlaid.

Reads a pairs file (one pair per line, e.g. ``rgb/a.png rgb/b.png``) and an
optional H5 keypoint cache, then draws both frames side by side with the
cached keypoints and generates a browsable HTML grid. This is the visualization
that ``gluefactory.scripts.prepare_slam_pairs --extract_features`` auto-runs
unless ``--num_vis 0``.

Usage:
    python -m gluefactory.scripts.visualize_dataset pairs \\
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


def _load_pairs(pairs_path, num_vis):
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    if num_vis and num_vis > 0 and len(pairs) > num_vis:
        np.random.seed(42)
        np.random.shuffle(pairs)
        pairs = pairs[: num_vis]
    return pairs


def visualize_pairs(data_dir, h5_path=None, pairs_file="pairs_train.txt",
                    num_vis=50, score_thresh=0.01, output_dir=None):
    """Render side-by-side pair images with cached keypoints.

    Args:
        data_dir: dataset root (contains ``images/<modality>`` and the pairs
                  file; H5 features default to ``exports/sp_features_slam.h5``).
        h5_path:  override the H5 features path (``None`` draws images only).
        pairs_file: pairs list filename under ``data_dir``.
        num_vis:  maximum number of pairs to visualize (0 = all).
        score_thresh: keypoint scores below this are not drawn.
        output_dir: where PNGs + index.html are written (default
                    ``<data_dir>/visualizations/pair_matches``).

    Returns:
        list of saved PNG paths.
    """
    data_dir = Path(data_dir)
    out_dir = make_out_dir(output_dir or data_dir / "visualizations/pair_matches")

    pairs_path = data_dir / pairs_file
    if not pairs_path.exists():
        logger.error(f"Pairs file not found: {pairs_path}")
        return []

    pairs = _load_pairs(pairs_path, num_vis)
    if not pairs:
        logger.warning("No pairs found.")
        return []

    h5_path = Path(h5_path) if h5_path else data_dir / "exports/sp_features_slam.h5"
    f_h5 = h5py.File(h5_path, "r") if h5_path.exists() else None
    if f_h5 is None:
        logger.warning(f"Could not load H5 features at {h5_path}. Only drawing images.")

    saved = []
    try:
        for p1, p2 in tqdm(pairs, desc="Visualizing pairs"):
            n1, n2 = Path(p1).name, Path(p2).name
            img1 = cv2.imread(str(data_dir / "images" / p1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(data_dir / "images" / p2), cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                logger.warning(f"Could not read one of {p1}, {p2}; skipping.")
                continue

            kpts1, scores1 = load_h5_kpts(f_h5, n1, score_thresh)
            kpts2, scores2 = load_h5_kpts(f_h5, n2, score_thresh)
            colors1 = plt.get_cmap("plasma")(scores1).tolist() if len(kpts1) else []
            colors2 = plt.get_cmap("plasma")(scores2).tolist() if len(kpts2) else []

            viz2d.plot_images([img1, img2], titles=[n1, n2], cmaps="gray")
            viz2d.plot_keypoints([kpts1, kpts2], colors=[colors1, colors2], ps=15, a=0.8)

            out_path = out_dir / f"{Path(n1).stem}_{Path(n2).stem}.png"
            viz2d.save_plot(out_path, facecolor="#121212")
            plt.close()
            saved.append(out_path)
    finally:
        if f_h5 is not None:
            f_h5.close()

    finalize_html_grid(saved, out_dir, "SLAM Pairs Visualization", grid_min_width=800)
    logger.info(f"Saved {len(saved)} pair visualization(s) to {out_dir}")
    return [str(p) for p in saved]


if __name__ == "__main__":
    # Keep this module runnable for quick debugging.
    visualize_pairs("data/output/sample_slam")
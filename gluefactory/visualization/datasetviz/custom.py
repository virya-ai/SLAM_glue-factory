"""kind=custom — visualize custom-model keypoint detections on an image set.

Runs a trained/exported SuperPoint model over every image in ``--input``,
saves a score-coloured keypoint overlay per image, a 2x3 summary collage of
the first few frames, and a browsable HTML grid. Consolidates the old
top-level visualize_custom.py (already superseded by run_inference) behind the
unified CLI.

Usage:
    python -m gluefactory.scripts.visualize_dataset custom \\
        --weights outputs/training/superpoint_custom_run/checkpoint_best.tar \\
        --input data/inputs/cases3_indoor_rgb --num_vis 50
"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.slam.io import list_images, load_image
from gluefactory.slam.matcher import SLAMMatcher
from gluefactory.scripts.run_inference import ExportedMatcher
from gluefactory.visualization import viz2d
from gluefactory.visualization.datasetviz.base import (
    finalize_html_grid,
    make_out_dir,
)

logger = logging.getLogger(__name__)


def _make_backend(weights, pt, conf):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pt:
        return ExportedMatcher(pt, None, device=device, conf=conf, matcher="none")
    return SLAMMatcher(None, weights, device=device, conf=conf, matcher="none")


def _make_collage(image_paths, kpts_by_name, out_dir, max_images=6, top_k=150):
    """Render a 2x3 collage of green keypoint dots on the first frames."""
    titles, overlays = [], []
    for p in image_paths:
        if len(overlays) >= max_images:
            break
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        kpts = kpts_by_name.get(p.name, np.zeros((0, 2)))
        for kp in kpts[:top_k]:
            x, y = int(round(kp[0])), int(round(kp[1]))
            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        overlays.append(overlay)
        titles.append(f"{p.name} ({len(kpts)} kpts)")
    if not overlays:
        return None

    rows = (len(overlays) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows), dpi=150)
    fig.patch.set_facecolor("#0b0c10")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0b0c10")
    for ax, img, title in zip(axes, overlays, titles):
        ax.imshow(img)
        ax.set_title(title, color="#66fcf1", fontsize=12, pad=8)
    fig.suptitle(
        "Custom SuperPoint Detections Collage (Top 150 Keypoints shown in Green)",
        color="#fc4445", fontsize=18, weight="bold", y=0.96,
    )
    out_path = out_dir / "collage.png"
    plt.savefig(out_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return out_path


def visualize_custom(weights=None, pt=None, input=None, num_vis=50,
                     output_dir=None, **model_conf):
    """Render keypoint detections for a set of images with a custom model.

    Args:
        weights: SuperPoint training checkpoint (.tar) path.
        pt:      exported SuperPoint TorchScript (.pt) path (alternative to weights).
        input:   image directory / file list.
        num_vis: maximum number of images (0 = all).
        output_dir: where PNGs + index.html are written.
        **model_conf: inference overrides (nms_radius, max_num_keypoints, ...).

    Returns:
        list of saved PNG paths.
    """
    if not weights and not pt:
        raise ValueError("Either --weights or --pt is required for kind=custom")
    image_paths = list_images(input)
    if not image_paths:
        logger.error(f"No images found in: {input}")
        return []
    if num_vis and num_vis > 0:
        image_paths = image_paths[: num_vis]

    out_dir = make_out_dir(output_dir or "outputs/visualizations/custom_detections")
    conf = {
        "nms_radius": model_conf.pop("nms_radius", 3),
        "max_num_keypoints": model_conf.pop("max_num_keypoints", 512),
        "detection_threshold": model_conf.pop("detection_threshold", 0.005),
        "filter_threshold": model_conf.pop("filter_threshold", 0.01),
    }
    backend = _make_backend(weights, pt, conf)

    saved = []
    kpts_by_name = {}
    for idx, p in enumerate(tqdm(image_paths, desc="Extracting")):
        img_bgr, img_gray = load_image(p)
        out = backend.extract(img_gray)
        kpts, scores = out["keypoints"], out["scores"]
        kpts_by_name[p.name] = kpts
        logger.info(f"[{idx}] {p.name}: {len(kpts)} keypoints")

        out_path = out_dir / f"{p.stem}_detections.png"
        viz2d.plot_keypoints_overlay(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), kpts, scores, out_path,
            title=f"{p.name} ({len(kpts)} keypoints)", colorbar=True,
        )
        saved.append(out_path)

    collage = _make_collage(image_paths, kpts_by_name, out_dir)
    if collage is not None:
        saved.append(collage)

    finalize_html_grid(saved, [p for p in saved if p.name != "collage.png"],
                       "Custom SuperPoint Detections")
    logger.info(f"Saved {len(saved)} visualization(s) to {out_dir}")
    return [str(p) for p in saved]


if __name__ == "__main__":
    # Keep this module runnable for quick debugging.
    visualize_custom(
        weights="outputs/training/superpoint_custom_run/checkpoint_best.tar",
        input="data/inputs/cases3_indoor_rgb",
    )
"""kind=attention — visualize SuperGlue attention maps + SuperPoint feature maps.

Runs the two models on an image pair entirely on CPU (safe while GPU training
is in progress) and produces the full set of analysis figures:
SuperPoint encoder feature maps, detector heatmap, descriptor PCA, per-layer
and summary attention grids, match-confidence heatmap, and the research-paper
style attention overlays / line figures.

Usage:
    python -m gluefactory.scripts.visualize_dataset attention \\
        --superglue outputs/training/superglue_slam_run/checkpoint_best.tar \\
        --superpoint outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --img0 data/output/sample_slam/images/rgb/1775717079.407016.png \\
        --img1 data/output/sample_slam/images/rgb/1775717079.507152.png
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from gluefactory.utils.experiments import load_experiment
from gluefactory.visualization.attention_viz import (
    collect_attention_maps,
    patch_superglue_attention,
    plot_attention_layer,
    plot_attention_summary,
    plot_superglue_match_heatmap,
    plot_superglue_attention_on_image,
    plot_superglue_attention_lines,
    plot_superpoint_descriptor_pca,
    plot_superpoint_detector_heatmap,
    plot_superpoint_feature_maps,
    register_superpoint_hooks,
)
from gluefactory.visualization.datasetviz.base import make_out_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (kept local to this kind)
# ---------------------------------------------------------------------------

def _load_image(path, size):
    """Load image as a (1,1,H,W) float32 tensor and an (H,W) numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if size > 0:
        h, w = img.shape
        scale = size / max(h, w)
        new_w = int(round(w * scale / 8) * 8)
        new_h = int(round(h * scale / 8) * 8)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor, img


def _run_superpoint(model, image_tensor):
    sp = model.extractor if hasattr(model, "extractor") and model.extractor is not None else model
    with torch.inference_mode():
        pred = sp({"image": image_tensor})
    return pred


def _compute_dense_scores(pred):
    """Reconstruct the (H, W) dense score map from a prediction dict."""
    if "logits" in pred:
        import torch.nn.functional as F

        logits = pred["logits"].detach().cpu()  # (1, stride²+1, H', W')
        probs = F.softmax(logits, dim=1)[:, :-1]  # drop dustbin
        b, c, h, w = probs.shape
        stride = int(round(c ** 0.5))
        probs = probs.permute(0, 2, 3, 1).reshape(b, h, w, stride, stride)
        probs = probs.permute(0, 1, 3, 2, 4).reshape(b, h * stride, w * stride)
        return probs[0].numpy()
    if "keypoint_scores" in pred:
        sc = pred["keypoint_scores"]
        if sc.dim() == 3:
            return sc[0].detach().cpu().numpy()
    return None


def _build_superglue_data(view0_tensor, view1_tensor, pred0, pred1):
    def _b(t):
        return t.unsqueeze(0) if t.dim() == 2 else t

    return {
        "view0": {"image": view0_tensor},
        "view1": {"image": view1_tensor},
        "keypoints0": _b(pred0["keypoints"]),
        "keypoints1": _b(pred1["keypoints"]),
        "descriptors0": _b(pred0["descriptors"]),
        "descriptors1": _b(pred1["descriptors"]),
        "keypoint_scores0": _b(pred0["keypoint_scores"]),
        "keypoint_scores1": _b(pred1["keypoint_scores"]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def visualize_attention(superglue, superpoint, img0, img1,
                        output_dir="outputs/visualizations/attention_maps",
                        resize=512, layers="all", num_vis=None):
    """Run the full SuperPoint + SuperGlue attention/feature-map visualization.

    Args:
        superglue: SuperGlue checkpoint (.tar) path.
        superpoint: SuperPoint checkpoint (.tar) path.
        img0, img1: path to the left/right image to analyze.
        output_dir: where the PNGs are written.
        resize: max image dimension to resize to (0 = no resize).
        layers: which GNN layers to plot individually ("all", "none", or
                comma-separated indices).
        num_vis: accepted for the shared kind interface; unused (this analysis
                 always renders the single given image pair).

    Returns:
        sorted list of saved PNG paths under ``output_dir``.
    """
    out_dir = make_out_dir(output_dir)

    logger.info("=== SuperGlue + SuperPoint attention/feature-map visualizer ===")
    logger.info(f"Output directory: {out_dir}")

    # 1. Load images
    img0_tensor, img0_np = _load_image(img0, resize)
    img1_tensor, img1_np = _load_image(img1, resize)
    logger.info(f"  img0: {img0_np.shape}  img1: {img1_np.shape}")

    # 2. Load + run SuperPoint
    logger.info(f"Loading SuperPoint from {superpoint} …")
    sp_model = load_experiment(superpoint).eval()
    sp_hooks, sp_activations = register_superpoint_hooks(sp_model)

    logger.info("Running SuperPoint on img0 …")
    pred0 = _run_superpoint(sp_model, img0_tensor)
    sp_activations_img0 = {k: v.clone() for k, v in sp_activations.items()}
    dense_scores_img0 = _compute_dense_scores(pred0)
    dense_desc_img0 = pred0.get("dense_descriptors")

    logger.info("Running SuperPoint on img1 …")
    pred1 = _run_superpoint(sp_model, img1_tensor)

    for h in sp_hooks:
        h.remove()

    n0 = pred0["keypoints"].shape[-2]
    n1 = pred1["keypoints"].shape[-2]
    logger.info(f"  Keypoints: img0={n0}  img1={n1}")

    # 3. SuperPoint visualizations
    logger.info("Generating SuperPoint visualizations …")
    plot_superpoint_feature_maps(sp_activations_img0, out_dir)

    if dense_scores_img0 is not None:
        plot_superpoint_detector_heatmap(
            dense_scores_img0, img0_np, out_dir / "superpoint_detector_heatmap.png"
        )
        logger.info("  Saved superpoint_detector_heatmap.png")

    if dense_desc_img0 is not None:
        plot_superpoint_descriptor_pca(
            dense_desc_img0, img0_np, out_dir / "superpoint_descriptor_pca.png"
        )
        logger.info("  Saved superpoint_descriptor_pca.png")
    else:
        logger.warning("  dense_descriptors not found; skipping PCA plot")

    # 4. Load + run SuperGlue
    logger.info(f"Loading SuperGlue from {superglue} …")
    sg_model = load_experiment(superglue).eval()
    patch_superglue_attention(sg_model)

    data = _build_superglue_data(img0_tensor, img1_tensor, pred0, pred1)
    logger.info("Running SuperGlue …")
    with torch.inference_mode():
        sg_pred = sg_model(data)

    attention_maps = collect_attention_maps(sg_model)
    n_captured = sum(1 for m in attention_maps if m["prob_img0"] is not None)
    logger.info(f"  Captured attention maps: {n_captured}/{len(attention_maps)} layers")

    # 5. Per-layer attention plots
    if layers == "none":
        layer_indices = []
    elif layers == "all":
        layer_indices = list(range(len(attention_maps)))
    else:
        layer_indices = [int(x) for x in str(layers).split(",")]

    logger.info(f"Generating per-layer attention plots for {len(layer_indices)} layers …")
    for i in layer_indices:
        m = attention_maps[i]
        if m["prob_img0"] is None:
            continue
        fname = out_dir / f"superglue_attn_layer_{i:02d}_{m['type']}_img0.png"
        plot_attention_layer(m["prob_img0"], i, m["type"], save_path=fname)
        if m["prob_img1"] is not None:
            fname1 = out_dir / f"superglue_attn_layer_{i:02d}_{m['type']}_img1.png"
            plot_attention_layer(m["prob_img1"], i, m["type"], save_path=fname1)

    # 6-9. Summary + heatmap + paper-style figures
    plot_attention_summary(attention_maps, save_path=out_dir / "superglue_attn_summary.png")
    plot_superglue_match_heatmap(
        sg_pred,
        data["keypoints0"], data["keypoints1"],
        img0_np, img1_np,
        out_dir / "superglue_match_heatmap.png",
    )
    plot_superglue_attention_on_image(
        attention_maps,
        data["keypoints0"], data["keypoints1"],
        img0_np, img1_np,
        out_dir / "superglue_attention_overlay.png",
    )
    plot_superglue_attention_lines(
        attention_maps,
        data["keypoints0"], data["keypoints1"],
        img0_np, img1_np,
        out_dir / "superglue_attention_lines.png",
        n_layers=5, top_k=15,
    )

    saved = sorted(out_dir.glob("*.png"))
    logger.info(f"\nDone. {len(saved)} PNG files in {out_dir}:")
    for f in saved:
        logger.info(f"  {f.name}")
    return [str(p) for p in saved]


if __name__ == "__main__":
    # Keep this module runnable for quick debugging.
    visualize_attention(
        superglue="outputs/training/superglue_slam_run/checkpoint_best.tar",
        superpoint="outputs/training/superpoint_slam_run/checkpoint_best.tar",
        img0="data/output/sample_slam/images/rgb/1775717079.407016.png",
        img1="data/output/sample_slam/images/rgb/1775717079.507152.png",
    )
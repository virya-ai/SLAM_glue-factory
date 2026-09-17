#!/usr/bin/env python3
"""
Unified dataset-visualization CLI.

Single entry point for every visualization in the dataset pipeline, each backed
by a dedicated module in ``gluefactory.visualization.datasetviz``:

    labels      cached pseudo-labels overlaid on frames     -> HTML grid
    pairs       side-by-side pairs with cached keypoints    -> HTML grid
    training    training-pair samples (SP + GT matches)     -> PNGs + CSV
    custom      custom-model keypoint detections + collage  -> PNGs + HTML grid
    attention   SuperPoint/SuperGlue attention & feature maps -> PNGs

All kinds share the ``--num_vis`` count (default 50; ``--num_vis 0`` disables).
The dataset-creation scripts (prepare_slam_labels.py, generate_slam_pairs.py)
call this machinery automatically after building their artifacts; this CLI is
for manual re-runs or solo inspection.

Usage:
    python -m gluefactory.scripts.visualize_dataset labels \\
        --data_dir data/output/slam --num_vis 50

    python -m gluefactory.scripts.visualize_dataset pairs \\
        --data_dir data/output/slam --num_vis 50

    python -m gluefactory.scripts.visualize_dataset training \\
        --conf gluefactory/configs/superpoint+superglue_slam.yaml --num_vis 20

    python -m gluefactory.scripts.visualize_dataset custom \\
        --weights outputs/training/superpoint_custom_run/checkpoint_best.tar \\
        --input data/inputs/cases3_indoor_rgb

    python -m gluefactory.scripts.visualize_dataset attention \\
        --superglue outputs/training/superglue_slam_run/checkpoint_best.tar \\
        --superpoint outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --img0 data/output/slam/images/rgb/1775717079.407016.png \\
        --img1 data/output/slam/images/rgb/1775717079.507152.png
"""

import argparse
import logging

from gluefactory.visualization.datasetviz import DEFAULT_NUM_VIS, visualize

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("visualize_dataset")


def _add_num_vis(p, default=DEFAULT_NUM_VIS):
    p.add_argument("--num_vis", type=int, default=default,
                   help=f"Maximum number of samples/pairs to visualize "
                        f"(default {default}, 0 = disabled).")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="kind", required=True)

    # labels ----------------------------------------------------------------
    p = sub.add_parser("labels", help="Cached pseudo-label keypoints overlaid on frames.")
    p.add_argument("--data_dir", type=str, default="data/output/sample_slam")
    p.add_argument("--h5_path", type=str, default=None)
    p.add_argument("--modality", type=str, default="rgb")
    p.add_argument("--score_thresh", type=float, default=0.01)
    p.add_argument("--output_dir", type=str, default=None)
    _add_num_vis(p)

    # pairs -----------------------------------------------------------------
    p = sub.add_parser("pairs", help="Side-by-side pairs with cached keypoints.")
    p.add_argument("--data_dir", type=str, default="data/output/sample_slam")
    p.add_argument("--h5_path", type=str, default=None)
    p.add_argument("--pairs_file", type=str, default="pairs_train.txt")
    p.add_argument("--score_thresh", type=float, default=0.01)
    p.add_argument("--output_dir", type=str, default=None)
    _add_num_vis(p)

    # training --------------------------------------------------------------
    p = sub.add_parser("training", help="Training-pair samples (extractor + GT matches).")
    p.add_argument("--conf", type=str,
                   default="gluefactory/configs/superpoint+superglue_slam.yaml")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max_matches", type=int, default=300)
    _add_num_vis(p)

    # custom ----------------------------------------------------------------
    p = sub.add_parser("custom", help="Custom-model keypoint detections + collage.")
    p.add_argument("--weights", type=str, default=None,
                   help="SuperPoint training checkpoint (.tar).")
    p.add_argument("--pt", type=str, default=None,
                   help="Exported SuperPoint model (.pt); alternative to --weights.")
    p.add_argument("--input", type=str, required=True,
                   help="Image directory to run the extractor over.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--nms_radius", type=int, default=3)
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--detection_threshold", type=float, default=0.005)
    _add_num_vis(p)

    # attention -------------------------------------------------------------
    p = sub.add_parser("attention", help="SuperPoint/SuperGlue attention & feature maps.")
    p.add_argument("--superglue", type=str,
                   default="outputs/training/superglue_slam_run/checkpoint_best.tar")
    p.add_argument("--superpoint", type=str,
                   default="outputs/training/superpoint_slam_run/checkpoint_best.tar")
    p.add_argument("--img0", type=str,
                   default="data/output/sample_slam/images/rgb/1775717079.407016.png")
    p.add_argument("--img1", type=str,
                   default="data/output/sample_slam/images/rgb/1775717079.507152.png")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--layers", type=str, default="all",
                   help='Which GNN layers to plot individually: "all", "none", '
                        'or comma-separated indices e.g. "0,1,8,9"')
    _add_num_vis(p)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    kind = args.kind
    params = vars(args)
    params.pop("kind", None)

    if kind == "training" and params.get("output") is None:
        params["output"] = "outputs/visualizations/training_dataset"
    if kind in ("labels", "pairs") and params.get("output_dir") is None:
        params["output_dir"] = None  # kind computes its default under data_dir
    if kind == "custom" and params.get("output_dir") is None:
        params["output_dir"] = "outputs/visualizations/custom_detections"
    if kind == "attention" and params.get("output_dir") is None:
        params["output_dir"] = "outputs/visualizations/attention_maps"

    result = visualize(kind, **params)
    if result in (None, []):
        logger.warning(
            f"Visualization kind={kind} produced nothing (was it disabled "
            f"via --num_vis 0, or did inputs go missing?)."
        )
        return
    logger.info(f"Visualization kind={kind} complete.")


if __name__ == "__main__":
    main()
"""Unified dataset-visualization kinds.

One registry, one CLI (``gluefactory.scripts.visualize_dataset``), many kinds —
each kind is a self-contained module that knows how to render its own data:

    labels      cached pseudo-label keypoints overlaid on images  (H5 → PNG grid)
    pairs       side-by-side image pairs with cached keypoints     (pairs.txt + H5)
    training    training-pair samples: extractor + GT matches      (YAML config)
    custom      custom-model keypoint detections + collage         (images dir)
    attention   SuperPoint/SuperGlue attention & feature maps      (two images)

Every kind follows the same contract: ``num_vis`` controls how many samples are
rendered (default 50; 0 disables), ``output_dir``/``data_dir``-style arguments
keep the default on-disk locations, and the result always ends with a browsable
HTML grid. Dataset-creation scripts call :func:`visualize` after the artifacts
are written so a visualization pass is automatic unless ``--num_vis 0``.
"""

import logging

from gluefactory.visualization.datasetviz import (
    attention as _attention,
    custom as _custom,
    labels as _labels,
    pairs as _pairs,
    training as _training,
)

logger = logging.getLogger(__name__)

KINDS = {
    "labels": _labels.visualize_labels,
    "pairs": _pairs.visualize_pairs,
    "training": _training.visualize_training,
    "custom": _custom.visualize_custom,
    "attention": _attention.visualize_attention,
}

DEFAULT_NUM_VIS = 50


def visualize(kind, **kwargs):
    """Dispatch to the unified visualization for ``kind``.

    Args:
        kind: one of ``KINDS``.
        **kwargs: passed through to the kind's ``visualize_*`` function with
                  ``num_vis=0`` removed (0 = disabled => return without doing
                  anything).

    Returns:
        the kind's return value (saved paths / output dir), or ``[]`` when the
        kind is unknown or disabled.
    """
    if kind not in KINDS:
        logger.error(
            f"Unknown visualization kind {kind!r}. Choose from {sorted(KINDS)}"
        )
        return []
    num_vis = kwargs.pop("num_vis", DEFAULT_NUM_VIS)
    if not num_vis or num_vis <= 0:
        logger.info(f"Visualization kind={kind} disabled (num_vis={num_vis}).")
        return []
    return KINDS[kind](num_vis=num_vis, **kwargs)
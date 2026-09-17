"""Shared helpers for the unified dataset-visualization kinds.

Every kind lives in its own module (labels, pairs, training, custom,
attention) and exposes a ``visualize_*`` entry function with a common shape:
    1. create its output directory,
    2. render one PNG per sample/pair,
    3. finish with an HTML grid via :func:`finalize_html_grid`.

The number of samples is always controlled by ``num_vis`` (0 disables; the
default is 50) so dataset-creation scripts can auto-run a visualization pass
with a single consistent argument.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def make_out_dir(out_dir):
    """Create ``out_dir`` (and parents) and return it as a Path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_h5_kpts(f_h5, name, score_thresh):
    """Read keypoints + scores for ``name`` from an open H5 file.

    Returns ``(kpts, scores)`` filtered by ``score_thresh``. Missing entries
    yield empty arrays so callers can skip the None checks.
    """
    if f_h5 is None or name not in f_h5:
        return np.zeros((0, 2)), np.zeros(0)
    kpts = f_h5[name]["keypoints"][:]
    scores = f_h5[name]["keypoint_scores"][:]
    mask = scores > score_thresh
    return kpts[mask], scores[mask]


def finalize_html_grid(saved, out_dir, title, grid_min_width=600):
    """Render a browsable HTML grid over the saved PNGs in ``out_dir``.

    Args:
        saved: iterable of saved image paths (must sit directly under out_dir
               for the dashboard to resolve their basenames).
        out_dir: directory the images live in.
        title: page title.
        grid_min_width: CSS min column width for the grid template.

    Returns:
        the generated index.html path, or None if nothing was saved.
    """
    if not saved:
        logger.warning("No visualizations produced; skipping HTML grid.")
        return None
    from gluefactory.visualization import dashboard

    html_path = dashboard.render_image_grid(
        Path(out_dir),
        [Path(p) for p in saved],
        title=title,
        grid_min_width=grid_min_width,
    )
    logger.info(f"Generated HTML grid at {html_path}")
    return html_path
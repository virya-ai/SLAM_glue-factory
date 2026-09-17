"""Visualization helpers for the SLAM & feature-matching pipelines.

Import submodules explicitly to keep dependencies (matplotlib, seaborn, cv2)
lazy: e.g. ``from gluefactory.visualization.viz2d import draw_matches``.
"""

__all__ = [
    "viz2d",
    "visualize_batch",
    "attention_viz",
    "two_view_frame",
    "global_frame",
    "dashboard",
    "tools",
    "datasetviz",
]
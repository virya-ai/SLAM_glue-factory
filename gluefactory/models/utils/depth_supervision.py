"""Depth-aware supervision helpers shared by the detector and matcher losses.

These functions never enter the model inputs: depth is used exclusively during
training to re-weight the detector and matching losses, so inference remains
RGB-only.

Conventions
-----------
* ``max_depth`` is the hard distance cutoff (in meters). Depth beyond the cutoff
  is *far*, depth that is non-finite, non-positive, or below ``min_depth`` is
  *missing*. The two are kept separate so that clipping artifacts are never
  interpreted as validity.
* ``distance_weight`` combines a hard validity gate with one of the smooth
  in-range weighting methods (spec section 5)::

      hard          w = valid_weight            if d <= max_depth else invalid_weight
      linear        w = valid_weight * (1 - d / max_depth)
      exponential   w = valid_weight * exp(-alpha * d)
      inverse_depth w = valid_weight * eps / (eps + d)   (normalized 1/(d + eps))

  The hard cutoff (``w = invalid_weight`` for ``d > max_depth``) is always
  applied on top of the chosen smooth method.
"""

import torch
import torch.nn.functional as F

# Hard default shared by SuperPoint (detector) and LightGlue (matcher) losses.
DEFAULT_DEPTH_SUPERVISION_CONF = {
    "do": False,
    "min_depth": 2.0,
    "max_depth": 70.0,
    "method": "inverse_depth",
    "alpha": 0.03,
    "eps": 1.0,
    "valid_weight": 1.0,
    "invalid_weight": 0.0,
}


def _get(conf, key, default=None):
    """Read a field from a dict-like, OmegaConf or SimpleNamespace-like config."""
    if conf is None:
        return default
    if hasattr(conf, "get") and callable(conf.get):
        val = conf.get(key)
    else:
        val = getattr(conf, key, None)
    if val is None:
        return default
    return val


def valid_depth(depth, min_depth=2.0, max_depth=70.0):
    """Boolean mask for usable depth values: ``min_depth < d <= max_depth``."""
    return torch.isfinite(depth) & (depth > min_depth) & (depth <= max_depth)


def far_mask(depth, max_depth=70.0):
    """Boolean mask for finite depth beyond the maximum distance."""
    return torch.isfinite(depth) & (depth > max_depth)


def missing_mask(depth, min_depth=2.0):
    """Boolean mask for missing depth: non-finite or non-positive values."""
    return ~torch.isfinite(depth) | (depth <= min_depth)


def normalize_depth(depth, max_depth=70.0):
    """Map depth to the ``[0, 1]`` range using the maximum distance."""
    return torch.clamp(depth / max_depth, 0.0, 1.0)


def inverse_depth(depth, eps=1.0):
    """Stable reciprocal depth used for the smooth in-range weighting."""
    return 1.0 / (depth + eps)


def distance_weight(
    depth,
    method="inverse_depth",
    min_depth=2.0,
    max_depth=70.0,
    alpha=0.03,
    eps=1.0,
    valid_weight=1.0,
    invalid_weight=0.0,
):
    """Per-element weight with a hard validity gate plus a smooth in-range method.

    Args:
        depth: tensor of depth values (any shape).
        method: one of ``"hard"``, ``"linear"``, ``"exponential"``, ``"inverse_depth"``.
        min_depth, max_depth: hard validity bounds in meters.
        alpha: decay rate for ``"exponential"``.
        eps: regularization for ``"inverse_depth"``.
        valid_weight: weight assigned to valid in-range depths (upper bound).
        invalid_weight: weight assigned to far/missing depths (usually 0).
    """
    if depth.numel() == 0:
        return depth
    valid = valid_depth(depth, min_depth, max_depth)
    d = torch.clamp(depth, min=0.0)
    if method == "hard":
        w = torch.full_like(d, valid_weight)
    elif method == "linear":
        w = valid_weight * (1.0 - torch.clamp(d / max_depth, 0.0, 1.0))
    elif method == "exponential":
        w = valid_weight * torch.exp(-alpha * d)
    elif method == "inverse_depth":
        w = valid_weight * eps / (d + eps)
    else:
        raise ValueError(f"Unknown depth weighting method: {method}")
    return torch.where(valid, w, torch.full_like(w, invalid_weight))


def sample_depth_at_centers(depth, centers):
    """Bilinear-sample a depth map at pixel centers.

    Args:
        depth: tensor of shape ``[B, H, W]`` (or ``[H, W]``), pixel meters.
        centers: tensor of shape ``[N, 2]`` with ``(x, y)`` pixel coordinates.

    Returns:
        Sampled depth of shape ``[B, N]`` (or ``[N]``); NaN where invalid.
    """
    batched = depth.dim() == 3
    if not batched:
        depth = depth[None]
    h, w = depth.shape[-2:]
    grid = centers / centers.new_tensor([w, h]).float() * 2 - 1
    grid = grid[None, :, None].expand(depth.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        depth[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    out = sampled[:, 0, :, 0]
    if not batched:
        out = out[0]
    return out


def cell_depth_weights(logits, depth, conf, cell_size=8):
    """Per-cell depth weights and validity for a detector heatmap.

    Args:
        logits: detector logits of shape ``[B, 65, hc, wc]``.
        depth: depth map of shape ``[B, H, W]`` aligned with ``logits``.
        conf: depth supervision config (dict-like).
        cell_size: detector cell size (stride).

    Returns:
        ``(weights, valid, d)``: weights and validity of shape ``[B, hc, wc]``,
        plus the raw sampled cell depths ``[B, hc, wc]`` (NaN where invalid).
    """
    b, _, hc, wc = logits.shape
    device = logits.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(hc, device=device), torch.arange(wc, device=device), indexing="ij"
    )
    centers = torch.stack([grid_x, grid_y], dim=-1).float() * cell_size + cell_size / 2
    centers = centers.reshape(-1, 2)
    d = sample_depth_at_centers(depth, centers).reshape(b, hc, wc)
    min_depth = _get(conf, "min_depth", 0.0)
    max_depth = _get(conf, "max_depth", 70.0)
    valid = valid_depth(d, min_depth, max_depth)
    weights = depth_weight_from_conf(d, conf)
    return weights, valid, d


def keypoint_depths(data, i):
    """Per-keypoint depth values ``[B, M]`` for view ``i`` (or ``None``).

    Prefers the ground-truth depth already sampled at the keypoints
    (``gt_depth_keypoints{i}``, produced by the depth matcher in the loss
    phase); otherwise samples the view's depth map at the predicted keypoints.
    """
    gt_key = f"gt_depth_keypoints{i}"
    if isinstance(data, dict):
        gt_depth = data.get(gt_key)
    else:
        gt_depth = getattr(data, gt_key, None)
    if gt_depth is not None:
        return gt_depth
    view = data.get(f"view{i}") if isinstance(data, dict) else None
    kpts = data.get(f"keypoints{i}") if isinstance(data, dict) else None
    if view is not None and view.get("depth") is not None and kpts is not None:
        return sample_depth_at_keypoints(kpts, view["depth"])
    return None


def sample_depth_at_keypoints(keypoints, depth):
    """Sample a batched depth map at keypoints.

    Args:
        keypoints: tensor of shape ``[B, M, 2]`` with ``(x, y)`` pixel coords.
        depth: tensor of shape ``[B, H, W]``.

    Returns:
        Depth of shape ``[B, M]``; NaN where out of bounds.
    """
    b, m, _ = keypoints.shape
    h, w = depth.shape[-2:]
    grid = keypoints / keypoints.new_tensor([w, h]).float() * 2 - 1
    grid = grid[:, :, None]
    sampled = F.grid_sample(
        depth[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled[:, 0, :, 0].reshape(b, m)


def depth_weight_from_conf(depth, conf):
    """Compute per-element depth weights from a depth-supervision config block."""
    return distance_weight(
        depth,
        method=_get(conf, "method", "inverse_depth"),
        min_depth=_get(conf, "min_depth", 2.0),
        max_depth=_get(conf, "max_depth", 70.0),
        alpha=_get(conf, "alpha", 0.03),
        eps=_get(conf, "eps", 1.0),
        valid_weight=_get(conf, "valid_weight", 1.0),
        invalid_weight=_get(conf, "invalid_weight", 0.0),
    )


def weighted_cross_entropy(logits, targets, cell_weight):
    """Per-cell cross-entropy with depth weighting.

    ``cell_weight`` has shape ``[B, hc, wc]``; far/missing cells receive the
    configured ``invalid_weight`` (0 suppresses them without deleting them).
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    return (ce * cell_weight).mean()


def depth_aware_penalty(logits, invalid_cell):
    """Conceptual ``L_depth_aware``: penalize detector confidence in bad cells.

    ``P_kp`` is the probability that any keypoint class is selected; invalid
    cells (far or missing depth) get their contribution counted against the
    model. Returns ``mean(P_kp * invalid_cell)``.
    """
    prob = F.softmax(logits, dim=1)[:, :-1].sum(dim=1)
    return (prob * invalid_cell.float()).mean()

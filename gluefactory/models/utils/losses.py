import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .depth_supervision import _get, depth_weight_from_conf, keypoint_depths


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
        "depth_aware": {
            "do": False,
            "min_depth": 2.0,
            "max_depth": 70.0,
            "method": "inverse_depth",
            "alpha": 0.03,
            "eps": 1.0,
            "valid_weight": 1.0,
            "invalid_weight": 0.0,
            "pair_combine": "min",
        },
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
                **getattr(self, "depth_metrics", {}),
            },
        )

    def nll_loss(self, log_assignment, data):
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return self.apply_depth_weights(weights, data, m, n)

    def apply_depth_weights(self, weights, data, m, n):
        """Re-weight the matching loss matrix with the keypoint depths.

        Positive (matched) pairs are scaled by the combined depth weight of
        both keypoints (``pair_combine: "min"`` drags the whole pair weight
        to ``invalid_weight`` if either point is far/invalid) -- this trusts
        a claimed match less as it gets farther away, matching the base
        detector's pseudo-labels being less reliable at range.

        The dustbin rows/columns (weights[:, :m, -1] / weights[:, -1, :n]:
        "this keypoint has no match") are NOT distance-weighted -- they
        always get the constant ``valid_weight``. gt_matches0/1 == -1 (used
        to build these) already excludes IGNORE_FEATURE points upstream, so
        these are confirmed genuine negatives; discounting that "should be
        unmatched" signal for far keypoints would under-supervise negative
        matching the same way it under-supervised SuperPoint's background
        cells and caused a false-positive detection grid there.
        """
        da = self.conf.depth_aware
        self.depth_metrics = {}
        if not _get(da, "do", False):
            return weights
        d0 = keypoint_depths(data, 0)
        d1 = keypoint_depths(data, 1)
        if d0 is None or d1 is None:
            return weights
        w0 = depth_weight_from_conf(d0, da)
        w1 = depth_weight_from_conf(d1, da)
        invalid_weight = float(_get(da, "invalid_weight", 0.0))
        valid_weight = float(_get(da, "valid_weight", 1.0))
        if _get(da, "pair_combine", "min") == "prod":
            pair = w0[:, :, None] * w1[:, None, :]
        else:
            pair = torch.min(w0[:, :, None], w1[:, None, :])
        weights = weights.clone()
        weights[:, :m, :n] = weights[:, :m, :n] * pair
        weights[:, :m, -1] = weights[:, :m, -1] * valid_weight
        weights[:, -1, :n] = weights[:, -1, :n] * valid_weight
        self.depth_metrics = {
            "depth/far_pair_frac": (pair == invalid_weight).float().mean().detach(),
            "depth/valid_keypoints0": (w0 > invalid_weight).float().mean().detach(),
            "depth/valid_keypoints1": (w1 > invalid_weight).float().mean().detach(),
        }
        return weights

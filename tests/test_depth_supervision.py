import unittest

import torch

from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.utils.depth_supervision import (
    cell_depth_weights,
    depth_aware_penalty,
    distance_weight,
    far_mask,
    inverse_depth,
    keypoint_depths,
    missing_mask,
    normalize_depth,
    valid_depth,
    weighted_cross_entropy,
)
from gluefactory.models.utils.losses import NLLLoss


class TestDepthBasis(unittest.TestCase):
    def test_valid_depth_separates_buckets(self):
        d = torch.tensor(
            [0.0, -5.0, 1.0, 2.0, float("nan"), float("inf"), 30.0, 70.0, 200.0]
        )
        valid = valid_depth(d, min_depth=2.0, max_depth=70.0)
        expected = torch.tensor(
            [False, False, False, False, False, False, True, True, False]
        )
        self.assertTrue(torch.equal(valid, expected))

    def test_far_and_missing_masks(self):
        d = torch.tensor([0.0, 1.0, 2.0, float("nan"), float("inf"), 70.0, 200.0])
        # default min_depth=2.0: everything at/below 2 m is missing
        self.assertTrue(
            torch.equal(
                missing_mask(d),
                torch.tensor([True, True, True, True, True, False, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                far_mask(d, max_depth=70.0),
                torch.tensor([False, False, False, False, False, False, True]),
            )
        )

    def test_hard_weight(self):
        d = torch.tensor([0.0, 10.0, 70.0, 200.0])
        w = distance_weight(
            d, method="hard", max_depth=70.0, valid_weight=1.0, invalid_weight=0.0
        )
        self.assertTrue(torch.equal(w, torch.tensor([0.0, 1.0, 1.0, 0.0])))

    def test_linear_weight_decreasing_and_cutoff(self):
        d = torch.arange(0.0, 200.0, 10.0)
        w = distance_weight(d, method="linear", max_depth=70.0)
        self.assertEqual(w[0], 0.0)  # d=0 is missing, not valid at weight 1
        self.assertTrue(torch.equal(w[1:8], 1.0 - (d[1:8] / 70.0)))  # 10..70 m
        self.assertTrue(w[7] == 0.0)  # d=70
        clipped = (w[8:] == 0.0).all()  # beyond cutoff suppressed, not clipped
        self.assertTrue(clipped)
        mono = torch.diff(w[1:8]).max() <= 0  # monotonically non-increasing in-range
        self.assertTrue(mono)

    def test_exponential_weight(self):
        d = torch.tensor([1.0, 3.0, 10.0, 50.0, 200.0])
        w = distance_weight(d, method="exponential", max_depth=70.0, alpha=0.03)
        self.assertEqual(w[0], 0.0)  # below 2 m suppressed
        self.assertTrue(torch.allclose(w[1:4], torch.exp(-0.03 * d[1:4])))
        self.assertEqual(w[4], 0.0)  # beyond cutoff

    def test_inverse_depth_weight_normalized(self):
        d = torch.tensor([1.0, 3.0, 10.0])
        eps = 1.0
        w = distance_weight(d, method="inverse_depth", max_depth=70.0, eps=eps)
        self.assertEqual(w[0], 0.0)  # below 2 m is not valid
        self.assertTrue(torch.allclose(w[1:], eps / (d[1:] + eps)))

    def test_inverse_depth_hard_cutoff(self):
        d = torch.tensor([1.0, 2.0, 3.0, 70.0, 200.0])
        eps = 1.0
        w = distance_weight(d, method="inverse_depth", max_depth=70.0, eps=eps)
        self.assertEqual(w[0], 0.0)  # below min excluded
        self.assertEqual(w[1], 0.0)  # exactly 2 m excluded (strict lower bound)
        self.assertAlmostEqual(w[2].item(), eps / (3.0 + eps), places=5)
        self.assertAlmostEqual(w[3].item(), eps / (70.0 + eps), places=5)
        self.assertEqual(w[4], 0.0)  # beyond cutoff suppressed
        nan_w = distance_weight(torch.tensor([float("nan")]), method="inverse_depth")
        self.assertEqual(nan_w, 0.0)

    def test_min_depth_2m_boundary(self):
        d = torch.tensor([0.0, 1.0, 2.0, 2.01, 3.0, 10.0])
        w_lin = distance_weight(
            d, method="linear", min_depth=2.0, max_depth=70.0, invalid_weight=0.0
        )
        self.assertTrue((w_lin[:3] == 0.0).all())  # d <= 2 m suppressed
        self.assertGreater(w_lin[3], 0.0)  # 2.01 m enters the valid range
        self.assertGreater(w_lin[3], w_lin[4])  # decreasing with distance
        w_inv = distance_weight(
            d, method="inverse_depth", min_depth=2.0, max_depth=70.0, eps=1.0
        )
        self.assertTrue((w_inv[:3] == 0.0).all())
        self.assertGreater(w_inv[3], w_inv[4])

    def test_invalid_weight_is_honoured(self):
        d = torch.tensor([10.0, 200.0, float("nan")])
        w = distance_weight(d, method="linear", max_depth=70.0, invalid_weight=0.5)
        self.assertTrue(torch.equal(w[:1], torch.tensor([1.0 - 10.0 / 70.0])))
        self.assertTrue((w[1:] == 0.5).all())

    def test_normalize_and_inverse_depth(self):
        d = torch.tensor([35.0, 70.0, 140.0])
        normalized = normalize_depth(d, max_depth=70.0)
        self.assertTrue(torch.allclose(normalized, torch.tensor([0.5, 1.0, 1.0])))
        inv = inverse_depth(d, eps=1.0)
        self.assertTrue((inv > 0).all() and (inv < 2.0).all())


class TestCellWeighting(unittest.TestCase):
    def test_cell_depth_weights_shapes_and_far_suppression(self):
        logits = torch.zeros(1, 65, 4, 8)
        depth = torch.ones(1, 32, 64) * 100.0  # all cells far
        conf = {"max_depth": 70.0, "method": "linear", "invalid_weight": 0.0}
        w, valid, d = cell_depth_weights(logits, depth, conf)
        self.assertEqual(w.shape, (1, 4, 8))
        self.assertEqual(valid.shape, (1, 4, 8))
        self.assertTrue((w == 0.0).all())
        self.assertFalse(valid.any())

    def test_cell_depth_weights_near(self):
        logits = torch.zeros(1, 65, 4, 8)
        depth = torch.ones(1, 32, 64) * 35.0
        conf = {"do": True, "max_depth": 70.0, "method": "linear"}
        w, valid, _ = cell_depth_weights(logits, depth, conf)
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(w, torch.full((1, 4, 8), 0.5)))

    def test_weighted_cross_entropy_suppresses_far(self):
        # Logits strongly predict class 0 in every cell.
        b, hc, wc = 1, 2, 2
        logits = torch.zeros(b, 65, hc, wc)
        logits[:, 0] = 10.0
        targets = torch.zeros(b, hc, wc, dtype=torch.long)
        near_w = torch.ones(b, hc, wc)
        far_w = torch.zeros(b, hc, wc)
        far_w.requires_grad_(True)
        # Weighted CE is (near) zero on cells the model is confident in, and a
        # far cell with weight 0 contributes nothing regardless of its prediction.
        self.assertLess(weighted_cross_entropy(logits, targets, near_w), 0.01)
        self.assertEqual(weighted_cross_entropy(logits, targets, far_w), 0.0)
        # But if the model is wrong in a *near* cell the loss is larger.
        logits[:, 0] = 2.0
        self.assertGreater(weighted_cross_entropy(logits, targets, near_w), 0.01)

    def test_depth_aware_penalty(self):
        b, hc, wc = 1, 2, 2
        invalid = torch.zeros(b, hc, wc, dtype=torch.bool)
        invalid[0, 0, 0] = True
        high_logits = torch.zeros(b, 65, hc, wc)
        low_logits = torch.zeros(b, 65, hc, wc)
        high_logits[:, 0] = 10.0
        low_logits[:, 0] = -10.0
        hi = depth_aware_penalty(high_logits, invalid)
        lo = depth_aware_penalty(low_logits, invalid)
        # Confidence in an invalid cell is penalized: P_kp in that cell ~ 1.
        self.assertGreater(hi, 0.0)
        self.assertGreater(hi, lo)


class TestNLLDepthWeights(unittest.TestCase):
    @staticmethod
    def make_data(m=3, n=4, b=1):
        data = {
            "gt_matches0": torch.tensor([[0, -1, -2]]).repeat(b, 1),
            "gt_matches1": torch.tensor([[0, -1, -1, -1]]).repeat(b, 1),
            "gt_assignment": torch.zeros(b, m, n, dtype=torch.bool),
        }
        data["gt_assignment"][:, 0, 0] = True
        data["gt_depth_keypoints0"] = torch.tensor([[5.0, 3.0, 1.0]]).repeat(b, 1)
        data["gt_depth_keypoints1"] = torch.tensor(
            [[5.0, 50.0, 100.0, float("nan")]]
        ).repeat(b, 1)
        return data

    @staticmethod
    def linear_weight(d):
        d = torch.as_tensor(d, dtype=torch.float32)
        return torch.clamp(1.0 - d / 70.0, min=0.0)

    def test_no_depth_is_backward_compatible(self):
        data = self.make_data()
        loss = NLLLoss({"nll_balancing": 0.5, "depth_aware": {"do": False}})
        log_assignment = torch.zeros(1, 4, 5)
        _, weights, _ = loss({"log_assignment": log_assignment}, data)
        expected = torch.zeros(1, 4, 5)
        expected[0, :3, :4] = data["gt_assignment"].float()
        expected[0, :3, -1] = (data["gt_matches0"] == -1).float()
        expected[0, -1, :4] = (data["gt_matches1"] == -1).float()
        self.assertTrue(torch.equal(weights, expected))

    def test_far_pair_zeroed(self):
        data = self.make_data()
        conf = {"depth_aware": {"do": True, "method": "linear", "max_depth": 70.0}}
        loss = NLLLoss(conf)
        _, weights, metrics = loss({"log_assignment": torch.zeros(1, 4, 5)}, data)
        # (0, 2) is a far pair (100 m) and (0, 3) pairs with missing depth (NaN).
        self.assertEqual(weights[0, 0, 2], 0.0)
        self.assertEqual(weights[0, 0, 3], 0.0)
        # near/near pairs keep a positive reduced weight
        expected = self.linear_weight(5.0)
        self.assertAlmostEqual(weights[0, 0, 0].item(), expected.item(), places=5)
        # weighted sums normalize like the plain masks
        self.assertGreater(metrics["num_matchable"][0], 0)
        self.assertGreater(float(metrics["depth/far_pair_frac"]), 0)

    def test_negative_dustbin_scaled_by_view_weights(self):
        data = self.make_data()
        # min_depth pinned to 0: this test exercises dustbin scaling, including
        # a deliberately sub-2 m keypoint.
        loss = NLLLoss(
            {"depth_aware": {"do": True, "method": "linear", "min_depth": 0.0}}
        )
        _, weights, _ = loss({"log_assignment": torch.zeros(1, 4, 5)}, data)
        w0 = self.linear_weight(torch.tensor([5.0, 3.0, 1.0]))
        w1 = self.linear_weight(torch.tensor([5.0, 50.0, 100.0, float("nan")]))
        # unmatched keypoints go to the dustbin and are scaled by their depth weight
        self.assertAlmostEqual(weights[0, 1, -1].item(), w0[1].item(), places=5)
        self.assertAlmostEqual(weights[0, -1, 1].item(), w1[1].item(), places=5)
        self.assertEqual(weights[0, -1, 2], 0.0)  # far unmatched keypoint is suppressed
        dustbin0 = weights[0, 0, -1] + weights[0, 1, -1] + weights[0, 2, -1]
        self.assertEqual(dustbin0, w0[1])

    def test_pair_combine_min_and_prod(self):
        data = self.make_data()
        data["gt_depth_keypoints0"] = torch.full((1, 3), 35.0)
        data["gt_depth_keypoints1"] = torch.full((1, 4), 35.0)
        loss_min = NLLLoss(
            {"depth_aware": {"do": True, "method": "linear", "pair_combine": "min"}}
        )
        loss_prod = NLLLoss(
            {"depth_aware": {"do": True, "method": "linear", "pair_combine": "prod"}}
        )
        _, w_min, _ = loss_min({"log_assignment": torch.zeros(1, 4, 5)}, data)
        _, w_prod, _ = loss_prod({"log_assignment": torch.zeros(1, 4, 5)}, data)
        self.assertAlmostEqual(w_min[0, 0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(w_prod[0, 0, 0].item(), 0.25, places=5)
        self.assertNotEqual(loss_prod.conf.depth_aware.pair_combine, "min")

    def test_keypoint_depths_prefers_gt_and_falls_back_to_view(self):
        data = self.make_data()
        gt = keypoint_depths(data, 0)
        self.assertTrue(torch.equal(gt, data["gt_depth_keypoints0"]))
        data.pop("gt_depth_keypoints0")
        depth = torch.full((1, 32, 32), 42.0)
        kpts = torch.tensor([[[16.0, 16.0]]])
        data["view0"] = {"depth": depth}
        data["keypoints0"] = kpts
        sampled = keypoint_depths(data, 0)
        self.assertEqual(sampled.shape, (1, 1))
        self.assertGreater(sampled[0, 0], 0.0)
        self.assertLess(sampled[0, 0], 42.5)


class TestSuperPointDepthMetrics(unittest.TestCase):
    """Regression test: SP depth metrics must report per-view fractions,
    not the pooled mean halved again (see the /2 bug on [B, H, W] masks)."""

    @classmethod
    def setUpClass(cls):
        cls.sp = SuperPoint(
            {
                "name": "extractors.superpoint_open",
                "max_num_keypoints": 128,
                "force_num_keypoints": False,
                "dense_outputs": True,
                "depth_supervision": {
                    "do": True,
                    "min_depth": 2.0,
                    "max_depth": 70.0,
                    "method": "linear",
                    "valid_weight": 1.0,
                    "invalid_weight": 0.0,
                    "lambda_depth": 0.1,
                },
            }
        )
        cls.sp.eval()

    @staticmethod
    def make_data(depth_value=30.0):
        h = w = 96
        kpts = torch.tensor([[[24.0, 24.0], [48.0, 48.0]]]).repeat(2, 1, 1)

        def view():
            return {
                "image": torch.zeros(2, 1, h, w),
                "depth": torch.full((2, h, w), depth_value),
                "cache": {"keypoints": kpts},
            }

        pred = {
            "logits0": torch.zeros(2, 65, h // 8, w // 8),
            "logits1": torch.zeros(2, 65, h // 8, w // 8),
            "dense_descriptors0": torch.randn(2, 256, h // 8, w // 8),
            "dense_descriptors1": torch.randn(2, 256, h // 8, w // 8),
        }
        data = {"view0": view(), "view1": view()}
        return pred, data

    def test_metrics_report_true_fractions(self):
        pred, data = self.make_data(depth_value=30.0)
        _, metrics = self.sp.loss(pred, data)
        # All cells have valid in-range depth -> fraction must be ~1, not ~0.5.
        self.assertAlmostEqual(metrics["depth/valid_frac"].item(), 1.0, places=3)
        self.assertEqual(metrics["depth/missing_frac"].item(), 0.0)
        self.assertEqual(metrics["depth/far_frac"].item(), 0.0)
        # all cells valid -> no depth-aware penalty contribution
        self.assertEqual(metrics["loss/depth_aware"].item(), 0.0)

    def test_metrics_with_invalid_depth(self):
        pred, data = self.make_data(depth_value=float("nan"))
        _, metrics = self.sp.loss(pred, data)
        self.assertEqual(metrics["depth/valid_frac"].item(), 0.0)
        self.assertAlmostEqual(metrics["depth/missing_frac"].item(), 1.0, places=3)
        self.assertEqual(metrics["depth/far_frac"].item(), 0.0)
        # all cells invalid -> depth-aware penalty is active
        self.assertGreater(metrics["loss/depth_aware"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()

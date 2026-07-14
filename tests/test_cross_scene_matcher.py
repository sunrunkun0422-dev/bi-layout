import unittest

import torch

from models.cross_scene_matcher import (
    DualPanoramaCrossAttentionModel,
    OpeningGuidedCrossAttentionMatcher,
    OpeningSignalHead,
    candidate_intervals_to_mask,
    cyclic_yaw_loss,
    opening_matching_loss,
    relative_pose_loss,
)


class DummyBiLayout(torch.nn.Module):
    def __init__(self, tokens=16, channels=32):
        super().__init__()
        self.patch_dim = channels
        self.tokens = tokens
        self.channels = channels

    def forward(self, image, return_features=False):
        batch = image.shape[0]
        features = image.new_zeros(batch, self.tokens, self.channels)
        depth = image.new_ones(batch, self.tokens)
        new_depth = depth.clone()
        new_depth[:, 3:7] = 2.0
        output = {"depth": depth, "new_depth": new_depth, "ratio": depth[:, :1]}
        if return_features:
            output.update({
                "layout_feature": features,
                "feature_pos": features.clone(),
                "enc_feature": features.clone(),
                "ext_feature": features.clone(),
            })
        return output


class CrossSceneMatcherTest(unittest.TestCase):
    def test_circular_candidate_intervals(self):
        masks = candidate_intervals_to_mask([(2, 4), (6, 1)], length=8)

        self.assertEqual(masks.shape, (2, 8))
        self.assertEqual(torch.where(masks[0])[0].tolist(), [2, 3, 4])
        self.assertEqual(torch.where(masks[1])[0].tolist(), [0, 1, 6, 7])

    def test_opening_head_uses_extended_minus_enclosed_prior(self):
        head = OpeningSignalHead(feature_dim=16, hidden_dim=8)
        features = torch.zeros(1, 12, 16)
        enclosed = torch.ones(1, 12)
        extended = enclosed.clone()
        extended[:, 4:8] = 2.0

        output = head(features, enclosed, extended)

        self.assertGreater(output["opening_probability"][:, 4:8].mean().item(), 0.8)
        self.assertLess(output["opening_probability"][:, :4].mean().item(), 0.2)
        self.assertGreater(output["expansion_depth"][:, 4:8].mean().item(), 0.8)
        self.assertEqual(output["expansion_depth"][:, :4].abs().sum().item(), 0.0)

    def test_bidirectional_token_and_candidate_matching(self):
        torch.manual_seed(7)
        batch, tokens, channels = 2, 16, 32
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=channels,
            heads=4,
            hidden_dim=16,
        )
        features_a = torch.randn(batch, tokens, channels, requires_grad=True)
        features_b = torch.randn(batch, tokens, channels, requires_grad=True)
        enclosed_a = torch.ones(batch, tokens)
        enclosed_b = torch.ones(batch, tokens)
        extended_a = enclosed_a.clone()
        extended_b = enclosed_b.clone()
        extended_a[:, 3:7] = 2.0
        extended_b[:, 9:13] = 2.0
        masks_a = candidate_intervals_to_mask([(3, 6), (14, 1)], tokens)
        masks_b = candidate_intervals_to_mask([(9, 12), (0, 2)], tokens)

        output = matcher(
            features_a,
            features_b,
            enclosed_a,
            extended_a,
            enclosed_b,
            extended_b,
            candidate_masks_a=masks_a,
            candidate_masks_b=masks_b,
        )

        self.assertEqual(output["Aff_AB"].shape, (batch, tokens, tokens))
        self.assertEqual(output["candidate_affinity"].shape, (batch, 2, 2))
        self.assertEqual(output["best_candidate_pair"].shape, (batch, 2))
        self.assertEqual(output["cyclic_shift_score"].shape, (batch, tokens))
        self.assertEqual(output["relative_yaw_radians"].shape, (batch,))
        torch.testing.assert_close(
            output["Aff_AB"].sum(dim=-1),
            torch.ones(batch, tokens),
        )
        target_mask_b = masks_b.any(dim=0)
        self.assertEqual(output["Aff_AB"][:, :, ~target_mask_b].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(output["candidate_pair_score"]).all())
        torch.testing.assert_close(
            output["cyclic_shift_score"].sum(dim=-1),
            torch.ones(batch),
        )

        affinity_target = torch.zeros(batch, tokens, tokens)
        affinity_target[:, 3:7, 9:13] = 1.0
        opening_target_a = masks_a[0].float().expand(batch, -1)
        opening_target_b = masks_b[0].float().expand(batch, -1)
        losses = opening_matching_loss(
            output,
            opening_target_a,
            opening_target_b,
            affinity_target_ab=affinity_target,
        )
        losses["loss_total"].backward()
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertIsNotNone(features_a.grad)
        self.assertIsNotNone(features_b.grad)

    def test_end_to_end_wrapper_connects_bi_layout_outputs(self):
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=32,
            heads=4,
            hidden_dim=16,
        )
        model = DualPanoramaCrossAttentionModel(DummyBiLayout(), matcher=matcher)
        image_a = torch.zeros(1, 3, 8, 16)
        image_b = torch.zeros(1, 3, 8, 16)

        output = model(image_a, image_b)

        self.assertIn("depth", output["layout_A"])
        self.assertEqual(output["matches"]["Aff_AB"].shape, (1, 16, 16))
        self.assertEqual(output["matches"]["S_A"].shape, (1, 16))

    def test_cyclic_shift_recovers_wrapped_horizontal_rotation(self):
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=32,
            heads=4,
            hidden_dim=16,
        )
        tokens = 16
        affinity = torch.zeros(1, tokens, tokens)
        source = torch.arange(tokens)
        affinity[0, source, (source + 14) % tokens] = 1.0
        opening = torch.ones(1, tokens)

        output = matcher._cyclic_shift_scores(affinity, opening, opening)

        self.assertEqual(output["best_cyclic_shift"].item(), 14)
        self.assertAlmostEqual(output["relative_yaw_radians"].item(), -torch.pi / 4, places=5)

        yaw_loss = cyclic_yaw_loss(output["cyclic_shift_score"], torch.tensor([-torch.pi / 4]))
        self.assertLess(yaw_loss.item(), 1e-4)

    def test_relative_pose_loss_is_zero_for_equal_transforms(self):
        transform = torch.eye(3).unsqueeze(0).requires_grad_(True)

        losses = relative_pose_loss(transform, transform.detach().clone())
        losses["loss_pose"].backward()

        self.assertAlmostEqual(losses["loss_pose"].item(), 0.0, places=6)
        self.assertIsNotNone(transform.grad)


if __name__ == "__main__":
    unittest.main()

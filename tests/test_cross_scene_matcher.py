import unittest

import torch

from models.cross_scene_matcher import (
    DualPanoramaCrossAttentionModel,
    OpeningGuidedCrossAttentionMatcher,
    OpeningSignalHead,
    bidirectional_candidate_consistency_loss,
    candidate_assignment_loss,
    candidate_intervals_to_mask,
    cyclic_token_shift_loss,
    cyclic_yaw_loss,
    opening_matching_loss,
    opening_detection_loss,
    opening_probabilities_to_intervals,
    relative_pose_loss,
    resolve_enclosed_extended_depth,
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
        new_depth = image.new_ones(batch, self.tokens)  # raw/enclosed branch
        depth = new_depth.clone()  # visible/extended branch
        depth[:, 3:7] = 2.0
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

    def test_opening_components_preserve_width_and_wrap_at_seam(self):
        probability = torch.tensor(
            [0.9, 0.8, 0.1, 0.1, 0.7, 0.8, 0.95], dtype=torch.float32
        )

        intervals = opening_probabilities_to_intervals(
            probability,
            threshold=0.5,
            min_width_tokens=2,
            max_intervals=4,
        )

        self.assertEqual(intervals, [(4, 1)])
        mask = candidate_intervals_to_mask(intervals, len(probability))
        self.assertEqual(torch.where(mask[0])[0].tolist(), [0, 1, 4, 5, 6])

    def test_opening_components_drop_narrow_noise(self):
        probability = torch.tensor([0.1, 0.9, 0.1, 0.8, 0.8, 0.1])

        intervals = opening_probabilities_to_intervals(
            probability, threshold=0.5, min_width_tokens=2
        )

        self.assertEqual(intervals, [(3, 4)])

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

    def test_recall_oriented_opening_loss_is_finite_and_backpropagates(self):
        logits = torch.zeros(2, 8, requires_grad=True)
        target = torch.zeros(2, 8)
        target[:, 3:6] = 1.0

        losses = opening_detection_loss(
            logits,
            target,
            pos_weight=2.5,
            tversky_weight=0.5,
            tversky_alpha=0.3,
            tversky_beta=0.7,
        )
        losses["loss_total"].backward()

        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertGreater(losses["loss_bce"].item(), 0.0)
        self.assertGreater(losses["loss_tversky"].item(), 0.0)
        self.assertIsNotNone(logits.grad)

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
        self.assertEqual(output["candidate_assignment_logits"].shape, (batch, 3, 3))
        self.assertEqual(output["candidate_assignment_AB"].shape, (batch, 3, 3))
        self.assertEqual(output["candidate_assignment_BA"].shape, (batch, 3, 3))
        self.assertEqual(
            output["candidate_no_match_probability_A"].shape, (batch, 2)
        )
        self.assertEqual(
            output["candidate_no_match_probability_B"].shape, (batch, 2)
        )
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

    def test_gt_guidance_is_matcher_only_and_opening_head_stays_trainable(self):
        torch.manual_seed(13)
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=32,
            heads=4,
            hidden_dim=16,
        )
        features_a = torch.randn(1, 16, 32)
        features_b = torch.randn(1, 16, 32)
        enclosed = torch.ones(1, 16)
        extended = enclosed.clone()
        extended[:, 3:7] = 2.0
        guidance = torch.zeros(1, 16)
        guidance[:, 10:13] = 1.0

        output = matcher(
            features_a,
            features_b,
            enclosed,
            extended,
            enclosed,
            extended,
            opening_guidance_a=guidance,
            opening_guidance_b=guidance,
            opening_guidance_mode="gt",
        )

        torch.testing.assert_close(output["opening_guidance_A"], guidance)
        self.assertGreater(output["P_A_open"][:, 3:7].mean().item(), 0.8)
        self.assertLess(output["P_A_open"][:, 10:13].mean().item(), 0.2)
        self.assertFalse(torch.equal(output["P_A_open"], output["opening_guidance_A"]))

        opening_target = torch.zeros(1, 16)
        opening_target[:, 3:7] = 1.0
        losses = opening_matching_loss(output, opening_target, opening_target)
        losses["loss_total"].backward()
        self.assertIsNotNone(matcher.opening_head.output.weight.grad)
        self.assertGreater(matcher.opening_head.output.weight.grad.abs().sum().item(), 0.0)

    def test_negative_pair_supervises_dustbin_and_padding_is_ignored(self):
        torch.manual_seed(17)
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=32,
            heads=4,
            hidden_dim=16,
        )
        features_a = torch.randn(1, 16, 32)
        features_b = torch.randn(1, 16, 32)
        enclosed = torch.ones(1, 16)
        extended = enclosed.clone()
        extended[:, 2:12] = 2.0
        masks = candidate_intervals_to_mask([(2, 5), (8, 11)], 16)
        # Candidate 1 is deliberately non-empty but marked as padding.
        valid = torch.tensor([[True, False]])

        output = matcher(
            features_a,
            features_b,
            enclosed,
            extended,
            enclosed,
            extended,
            candidate_masks_a=masks,
            candidate_masks_b=masks,
            candidate_valid_a=valid,
            candidate_valid_b=valid,
        )
        self.assertEqual(output["candidate_valid_A"].tolist(), [[True, False]])
        self.assertEqual(
            output["candidate_assignment_AB"][:, 1, :].abs().sum().item(),
            0.0,
        )
        torch.testing.assert_close(
            output["candidate_assignment_AB"][:, 0, :].sum(dim=-1),
            torch.ones(1),
        )

        losses = candidate_assignment_loss(
            output,
            is_match=torch.tensor([False]),
            consistency_weight=0.0,
        )
        losses["loss_candidate_total"].backward()
        self.assertTrue(torch.isfinite(losses["loss_candidate_total"]))
        self.assertIsNotNone(matcher.dustbin_score.grad)
        self.assertLess(matcher.dustbin_score.grad.item(), 0.0)

    def test_positive_and_negative_candidate_targets_choose_real_or_dustbin(self):
        valid = torch.tensor([[True]])
        real_preferred = {
            "candidate_assignment_AB": torch.tensor([[[0.9, 0.1], [0.1, 0.9]]]),
            "candidate_assignment_BA": torch.tensor([[[0.9, 0.1], [0.1, 0.9]]]),
            "candidate_valid_A": valid,
            "candidate_valid_B": valid,
        }
        dustbin_preferred = {
            "candidate_assignment_AB": torch.tensor([[[0.1, 0.9], [0.9, 0.1]]]),
            "candidate_assignment_BA": torch.tensor([[[0.1, 0.9], [0.9, 0.1]]]),
            "candidate_valid_A": valid,
            "candidate_valid_B": valid,
        }
        target = torch.ones(1, 1, 1)

        positive_real = candidate_assignment_loss(
            real_preferred,
            candidate_target_ab=target,
            is_match=torch.tensor([True]),
            consistency_weight=0.0,
        )["loss_candidate_total"]
        positive_dustbin = candidate_assignment_loss(
            dustbin_preferred,
            candidate_target_ab=target,
            is_match=torch.tensor([True]),
            consistency_weight=0.0,
        )["loss_candidate_total"]
        negative_real = candidate_assignment_loss(
            real_preferred,
            is_match=torch.tensor([False]),
            consistency_weight=0.0,
        )["loss_candidate_total"]
        negative_dustbin = candidate_assignment_loss(
            dustbin_preferred,
            is_match=torch.tensor([False]),
            consistency_weight=0.0,
        )["loss_candidate_total"]

        self.assertLess(positive_real.item(), positive_dustbin.item())
        self.assertLess(negative_dustbin.item(), negative_real.item())

    def test_bidirectional_candidate_consistency_detects_disagreement(self):
        valid = torch.tensor([[True]])
        aligned = {
            "candidate_assignment_AB": torch.tensor([[[0.8, 0.2], [0.2, 0.8]]]),
            "candidate_assignment_BA": torch.tensor([[[0.8, 0.2], [0.2, 0.8]]]),
            "candidate_valid_A": valid,
            "candidate_valid_B": valid,
        }
        disagreed = dict(aligned)
        disagreed["candidate_assignment_BA"] = torch.tensor(
            [[[0.2, 0.8], [0.8, 0.2]]]
        )

        aligned_loss = bidirectional_candidate_consistency_loss(aligned)
        disagreed_loss = bidirectional_candidate_consistency_loss(disagreed)

        self.assertAlmostEqual(aligned_loss.item(), 0.0, places=6)
        self.assertGreater(disagreed_loss.item(), aligned_loss.item())

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
        self.assertGreater(
            output["matches"]["P_A_open"][:, 3:7].mean().item(), 0.8
        )

    def test_zind_branch_order_maps_raw_to_enclosed(self):
        output = {
            "depth": torch.full((1, 8), 2.0),
            "new_depth": torch.ones(1, 8),
        }

        enclosed, extended = resolve_enclosed_extended_depth(output)

        torch.testing.assert_close(enclosed, output["new_depth"])
        torch.testing.assert_close(extended, output["depth"])

    def test_cyclic_shift_recovers_wrapped_portal_token_offset(self):
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
        self.assertAlmostEqual(
            output["relative_token_shift_radians"].item(), -torch.pi / 4, places=5
        )
        torch.testing.assert_close(
            output["relative_yaw_radians"],
            output["relative_token_shift_radians"],
        )

        shift_loss = cyclic_token_shift_loss(
            output["cyclic_shift_score"], torch.tensor([-torch.pi / 4])
        )
        compatibility_loss = cyclic_yaw_loss(
            output["cyclic_shift_score"], torch.tensor([-torch.pi / 4])
        )
        self.assertLess(shift_loss.item(), 1e-4)
        torch.testing.assert_close(compatibility_loss, shift_loss)

    def test_relative_pose_loss_is_zero_for_equal_transforms(self):
        transform = torch.eye(3).unsqueeze(0).requires_grad_(True)

        losses = relative_pose_loss(transform, transform.detach().clone())
        losses["loss_pose"].backward()

        self.assertAlmostEqual(losses["loss_pose"].item(), 0.0, places=6)
        self.assertIsNotNone(transform.grad)


if __name__ == "__main__":
    unittest.main()

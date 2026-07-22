import unittest
from types import SimpleNamespace

import torch

from models.cross_scene_matcher import OpeningGuidedCrossAttentionMatcher
from tools.train_zind_cross_scene_matcher import (
    _losses,
    candidate_targets_from_affinity,
    guidance_mode_for_epoch,
    predicted_candidate_masks,
)


class ZInDMatcherTrainingTest(unittest.TestCase):
    def test_guidance_schedule_transitions_at_zero_based_boundaries(self):
        expected = {
            0: "gt",
            3: "gt",
            4: "mix",
            7: "mix",
            8: "predicted",
            20: "predicted",
        }

        for epoch, mode in expected.items():
            with self.subTest(epoch=epoch):
                self.assertEqual(
                    guidance_mode_for_epoch(epoch, gt_epochs=4, mix_epochs=4),
                    mode,
                )

        self.assertEqual(
            guidance_mode_for_epoch(0, gt_epochs=0, mix_epochs=0),
            "predicted",
        )

    def test_candidate_targets_project_dense_affinity_to_candidate_pairs(self):
        masks_a = torch.zeros(2, 2, 6, dtype=torch.bool)
        masks_b = torch.zeros(2, 2, 6, dtype=torch.bool)
        masks_a[:, 0, 0:2] = True
        masks_a[:, 1, 3:5] = True
        masks_b[:, 0, 1:3] = True
        masks_b[:, 1, 4:6] = True

        affinity = torch.zeros(2, 6, 6)
        affinity[0, 0, 4] = 0.25
        affinity[0, 3, 2] = 2.0

        target = candidate_targets_from_affinity(masks_a, affinity, masks_b)

        self.assertEqual(tuple(target.shape), (2, 2, 2))
        self.assertEqual(target.dtype, affinity.dtype)
        torch.testing.assert_close(
            target[0],
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        )
        torch.testing.assert_close(target[1], torch.zeros(2, 2))

    def test_candidate_target_rejects_accidental_one_token_overlap(self):
        masks_a = torch.ones(1, 1, 8, dtype=torch.bool)
        masks_b = torch.ones(1, 1, 8, dtype=torch.bool)
        affinity = torch.zeros(1, 8, 8)
        affinity[0, 2, 5] = 1.0

        target = candidate_targets_from_affinity(
            masks_a, affinity, masks_b, min_iou=0.30
        )

        self.assertEqual(target.item(), 0.0)

    def test_predicted_candidate_masks_filter_noise_and_pad_batch(self):
        probability = torch.tensor(
            [
                [0.1, 0.9, 0.8, 0.1, 0.7, 0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8, 0.9, 0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )

        masks, valid = predicted_candidate_masks(
            probability,
            threshold=0.5,
            min_width_tokens=2,
            max_openings=4,
        )

        self.assertEqual(tuple(masks.shape), (3, 2, 8))
        self.assertEqual(tuple(valid.shape), (3, 2))
        self.assertEqual(masks.dtype, torch.bool)
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(valid.tolist(), [[True, True], [True, False], [False, False]])
        self.assertEqual(torch.where(masks[0, 0])[0].tolist(), [1, 2])
        self.assertEqual(torch.where(masks[0, 1])[0].tolist(), [4, 5])
        self.assertEqual(torch.where(masks[1, 0])[0].tolist(), [2, 3])
        self.assertFalse(bool(masks[1, 1].any()))
        self.assertFalse(bool(masks[2].any()))

    def test_small_matcher_forward_and_training_losses_backpropagate(self):
        torch.manual_seed(23)
        batch_size, token_count, feature_dim = 2, 8, 16
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=feature_dim,
            heads=4,
            hidden_dim=8,
            dropout=0.0,
        )
        matcher.opening_head.requires_grad_(False)

        features_a = torch.randn(batch_size, token_count, feature_dim)
        features_b = torch.randn(batch_size, token_count, feature_dim)
        enclosed_a = torch.ones(batch_size, token_count)
        enclosed_b = torch.ones(batch_size, token_count)
        extended_a = enclosed_a.clone()
        extended_b = enclosed_b.clone()
        extended_a[:, 1:6] = 2.0
        extended_b[:, 2:7] = 2.0

        masks_a = torch.zeros(batch_size, 2, token_count, dtype=torch.bool)
        masks_b = torch.zeros(batch_size, 2, token_count, dtype=torch.bool)
        masks_a[:, 0, 1:3] = True
        masks_a[:, 1, 4:6] = True
        masks_b[:, 0, 2:4] = True
        masks_b[:, 1, 5:7] = True
        valid_a = torch.ones(batch_size, 2, dtype=torch.bool)
        valid_b = torch.ones(batch_size, 2, dtype=torch.bool)
        opening_all_a = masks_a.any(dim=1).float()
        opening_all_b = masks_b.any(dim=1).float()

        affinity_ab = torch.zeros(batch_size, token_count, token_count)
        affinity_ab[0, 1:3, 5:7] = 1.0
        candidate_target = candidate_targets_from_affinity(
            masks_a, affinity_ab, masks_b
        )
        pair = SimpleNamespace(
            is_match=torch.tensor([True, False]),
            affinity_ab=affinity_ab,
            shared_portal_a=torch.stack(
                (masks_a[0, 0], torch.zeros(token_count, dtype=torch.bool))
            ),
            shared_portal_b=torch.stack(
                (masks_b[0, 1], torch.zeros(token_count, dtype=torch.bool))
            ),
            relative_yaw=torch.zeros(batch_size),
        )
        args = SimpleNamespace(
            consistency_weight=0.1,
            candidate_loss_weight=1.0,
            token_affinity_weight=1.0,
            shared_response_weight=0.25,
            portal_shift_loss_weight=0.2,
        )

        outputs = matcher(
            features_a,
            features_b,
            enclosed_a,
            extended_a,
            enclosed_b,
            extended_b,
            candidate_masks_a=masks_a,
            candidate_masks_b=masks_b,
            candidate_valid_a=valid_a,
            candidate_valid_b=valid_b,
            opening_guidance_a=opening_all_a,
            opening_guidance_b=opening_all_b,
            opening_guidance_mode="gt",
        )
        losses = _losses(
            outputs,
            pair,
            candidate_target,
            valid_a,
            valid_b,
            args,
        )
        losses["loss_total"].backward()

        self.assertEqual(tuple(outputs["candidate_assignment_AB"].shape), (2, 3, 3))
        self.assertEqual(candidate_target[0].tolist(), [[0.0, 1.0], [0.0, 0.0]])
        self.assertEqual(candidate_target[1].tolist(), [[0.0, 0.0], [0.0, 0.0]])
        for name, loss in losses.items():
            with self.subTest(loss=name):
                self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(matcher.query_projection.weight.grad)
        self.assertGreater(
            float(matcher.query_projection.weight.grad.abs().sum().item()), 0.0
        )
        self.assertIsNotNone(matcher.dustbin_score.grad)
        self.assertTrue(bool(torch.isfinite(matcher.dustbin_score.grad)))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from dataset.zind_bipair_adapter import (
    adapt_zind_bipair_batch,
    canonicalize_zind_bipair_batch,
    opening_mask_to_candidate_masks,
)


def _batch():
    opening_a = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 0],
        ],
        dtype=torch.uint8,
    )
    opening_b = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
    )
    portal_a = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ],
        dtype=torch.uint8,
    )
    portal_b = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
    )
    affinity = torch.einsum("bi,bj->bij", portal_a.float(), portal_b.float())
    return {
        "pair_id": ["positive", "negative"],
        "image_A": torch.randn(2, 3, 4, 8),
        "image_B": torch.randn(2, 3, 4, 8),
        "depth_enclosed_A": torch.ones(2, 8),
        "depth_enclosed_B": torch.ones(2, 8),
        "depth_extended_A": torch.ones(2, 8) * 2,
        "depth_extended_B": torch.ones(2, 8) * 2,
        "opening_mask_all_A": opening_a,
        "opening_mask_all_B": opening_b,
        "portal_mask_A": portal_a,
        "portal_mask_B": portal_b,
        "affinity_gt": affinity,
        "is_positive": torch.tensor([True, False]),
        "T_B_to_A": torch.eye(3).repeat(2, 1, 1),
        "relative_yaw_gt": torch.tensor([[0.1], [-0.2]]),
        "scale_meters_per_coordinate": torch.tensor([[0.5], [0.5]]),
    }


class ZInDBiPairAdapterTest(unittest.TestCase):
    def test_circular_opening_components_keep_seam_interval(self):
        opening = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 1]], dtype=torch.bool)

        candidates = opening_mask_to_candidate_masks(opening)

        self.assertEqual(candidates.valid.tolist(), [[True, True]])
        self.assertEqual(candidates.intervals.tolist(), [[[4, 4], [7, 1]]])
        torch.testing.assert_close(candidates.union_mask(), opening)

    def test_empty_opening_has_only_invalid_padding_candidate(self):
        candidates = opening_mask_to_candidate_masks(torch.zeros(2, 8))

        self.assertEqual(tuple(candidates.masks.shape), (2, 1, 8))
        self.assertFalse(bool(candidates.valid.any()))
        self.assertEqual(candidates.candidate_ids.tolist(), [[-1], [-1]])

    def test_adapter_separates_all_openings_from_shared_portal(self):
        legacy = _batch()

        pair = adapt_zind_bipair_batch(legacy)

        torch.testing.assert_close(pair.candidates_a.union_mask(), pair.opening_all_a)
        torch.testing.assert_close(pair.candidates_b.union_mask(), pair.opening_all_b)
        self.assertEqual(pair.candidates_a.valid[0].sum().item(), 2)
        self.assertEqual(pair.candidates_b.valid[0].sum().item(), 2)
        self.assertEqual(pair.shared_portal_a[0].sum().item(), 3)
        self.assertFalse(torch.equal(pair.opening_all_a, pair.shared_portal_a))
        self.assertEqual(pair.is_match.tolist(), [True, False])
        self.assertEqual(pair.frame.transform_direction, "b_to_a")

    def test_mapping_adapter_preserves_legacy_fields(self):
        legacy = _batch()

        output = canonicalize_zind_bipair_batch(legacy)

        self.assertIs(output["opening_mask_all_A"], legacy["opening_mask_all_A"])
        self.assertIn("opening_all_a", output)
        self.assertIn("candidate_masks_all_a", output)
        self.assertIn("shared_portal_a", output)
        self.assertEqual(tuple(output["candidate_masks_all_a"].shape), (2, 2, 8))


if __name__ == "__main__":
    unittest.main()

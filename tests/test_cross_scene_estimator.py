import unittest

import numpy as np

from utils.cross_scene_estimator import (
    OpeningCandidate,
    estimate_wall_pair_candidates,
    extract_opening_candidates,
    opening_candidates_from_intervals,
    wall_token_assignment,
)


def make_layout(points, depth=None, new_depth=None):
    layout = {
        "cameraHeight": 1.6,
        "layoutHeight": 3.2,
        "layoutPoints": {
            "num": len(points),
            "points": [
                {"id": index, "xyz": [point[0], 1.6, point[1]]}
                for index, point in enumerate(points)
            ],
        },
        "layoutWalls": {
            "num": len(points),
            "walls": [
                {"pointsIdx": [index, (index + 1) % len(points)]}
                for index in range(len(points))
            ],
        },
    }
    if depth is not None:
        layout["biLayoutOutputs"] = {"depth": np.asarray(depth, dtype=float).tolist()}
        if new_depth is not None:
            layout["biLayoutOutputs"]["new_depth"] = np.asarray(new_depth, dtype=float).tolist()
    return layout


class CrossSceneEstimatorTest(unittest.TestCase):
    def test_learned_intervals_keep_matcher_candidate_order(self):
        layout = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        wall_ids, _ = wall_token_assignment(layout, sample_count=256)
        token_wall_1 = int(np.flatnonzero(wall_ids == 1)[len(np.flatnonzero(wall_ids == 1)) // 2])
        token_wall_3 = int(np.flatnonzero(wall_ids == 3)[len(np.flatnonzero(wall_ids == 3)) // 2])
        intervals = [(token_wall_3, token_wall_3), (token_wall_1, token_wall_1)]
        probabilities = np.full(256, 0.05, dtype=float)
        probabilities[token_wall_3] = 0.9
        probabilities[token_wall_1] = 0.7

        candidates = opening_candidates_from_intervals(
            layout,
            intervals,
            probabilities,
        )

        self.assertEqual([item.candidate_index for item in candidates], [0, 1])
        self.assertEqual([item.wall_index for item in candidates], [3, 1])
        self.assertEqual(
            [(item.token_start, item.token_end) for item in candidates],
            intervals,
        )
        self.assertAlmostEqual(candidates[0].confidence, 0.9)
        self.assertEqual(candidates[0].to_json()["candidateIndex"], 0)
        self.assertEqual(
            candidates[0].to_json()["confidenceType"],
            "learned_opening_probability_mean",
        )

    def test_extracts_openings_from_extended_minus_enclosed_depth(self):
        depth = np.ones(256, dtype=float)
        new_depth = depth.copy()
        new_depth[40:52] = 1.8
        layout = make_layout(
            [[-2, -2], [2, -2], [2, 2], [-2, 2]],
            depth=depth,
            new_depth=new_depth,
        )

        candidates, summary = extract_opening_candidates(
            layout,
            threshold=0.2,
            min_width_tokens=3,
            max_candidates=4,
        )

        self.assertEqual(summary["source"], "extended_minus_enclosed")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "extended_minus_enclosed")
        self.assertGreater(candidates[0].confidence, 0.5)
        candidate_json = candidates[0].to_json()
        self.assertEqual(
            candidate_json["confidenceType"],
            "heuristic_normalized_depth_contrast",
        )
        self.assertFalse(candidate_json["isCalibratedProbability"])

    def test_passability_confidence_changes_wall_pair_ranking(self):
        layout_a = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        layout_b = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        openings_a = [
            OpeningCandidate(0, 0.25, 0.75, 0.1),
            OpeningCandidate(1, 0.25, 0.75, 0.95),
        ]
        openings_b = [
            OpeningCandidate(0, 0.25, 0.75, 0.1),
            OpeningCandidate(3, 0.25, 0.75, 0.95),
        ]

        candidates, _ = estimate_wall_pair_candidates(
            layout_a,
            layout_b,
            openings_a=openings_a,
            openings_b=openings_b,
            passability_weight=10.0,
            top_k=1,
        )

        self.assertEqual(candidates[0].wall_a, 1)
        self.assertEqual(candidates[0].wall_b, 3)
        self.assertGreater(candidates[0].metrics["passabilityReward"], 9.0)


if __name__ == "__main__":
    unittest.main()

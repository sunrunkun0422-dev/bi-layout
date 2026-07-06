import unittest

import numpy as np

from utils.cross_scene_estimator import (
    OpeningCandidate,
    estimate_wall_pair_candidates,
    extract_opening_candidates,
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

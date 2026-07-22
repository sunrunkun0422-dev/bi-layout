import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dataset.zind_pair_mining import (
    interface_token_interval,
    mine_zind_adjacent_pairs,
    mine_zind_matching_records,
    relative_zind_transform,
    segment_endpoint_error,
)


def make_pano(image_path, transform, door):
    return {
        "is_primary": True,
        "label": "test room",
        "image_path": image_path,
        "floor_plan_transformation": transform,
        "layout_complete": {
            "vertices": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            "doors": [door[0], door[1], [0, 0]],
            "windows": [],
            "openings": [],
            "internal": [],
        },
    }


class ZindPairMiningTest(unittest.TestCase):
    def test_relative_transform_maps_b_into_a_coordinates(self):
        transform_a = {"translation": [0, 0], "rotation": 0, "scale": 2}
        transform_b = {"translation": [4, 0], "rotation": 90, "scale": 1}

        relative = relative_zind_transform(transform_a, transform_b)

        self.assertAlmostEqual(math.atan2(relative[1, 0], relative[0, 0]), math.pi / 2)
        self.assertAlmostEqual(math.hypot(relative[0, 0], relative[1, 0]), 0.5)
        np.testing.assert_allclose(relative[:2, 2], [2, 0], atol=1e-7)

    def test_segment_error_accepts_reversed_endpoint_order(self):
        first = np.asarray([[1, 0], [1, 1]], dtype=float)
        second = np.asarray([[1, 1], [1, 0]], dtype=float)

        self.assertAlmostEqual(segment_endpoint_error(first, second), 0.0)

    def test_interface_interval_uses_short_arc_across_panorama_seam(self):
        interval = interface_token_interval([[0.1, -1.0], [-0.1, -1.0]], 8)

        self.assertEqual(interval, (7, 0))

    def test_mines_labeled_shared_door_between_complete_rooms(self):
        identity = {"translation": [0, 0], "rotation": 0, "scale": 1}
        reversed_room = {"translation": [2, 0], "rotation": 180, "scale": 1}
        payload = {
            "merger": {
                "floor_01": {
                    "complete_room_01": {
                        "partial_room_01": {
                            "pano_1": make_pano(
                                "panos/a.jpg", identity, [[1, 0], [1, 1]]
                            )
                        }
                    },
                    "complete_room_02": {
                        "partial_room_02": {
                            "pano_2": make_pano(
                                "panos/b.jpg", reversed_room, [[1, 0], [1, -1]]
                            )
                        }
                    },
                }
            }
        }
        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "0000" / "zind_data.json"
            json_path.parent.mkdir()
            json_path.write_text(json.dumps(payload), encoding="utf-8")

            pairs = mine_zind_adjacent_pairs(str(json_path), endpoint_tolerance=1e-4)

        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertAlmostEqual(pair.interface_endpoint_error, 0.0, places=7)
        self.assertAlmostEqual(pair.camera_distance, 2.0, places=7)
        np.testing.assert_allclose(pair.interface_global, [[1, 0], [1, 1]], atol=1e-7)
        np.testing.assert_allclose(
            pair.ground_truth_transform,
            [[-1, 0, 2], [0, -1, 0], [0, 0, 1]],
            atol=1e-7,
        )

    def test_builds_compact_positive_and_safe_negative_matching_records(self):
        identity = {"translation": [0, 0], "rotation": 0, "scale": 1}
        reversed_room = {"translation": [2, 0], "rotation": 180, "scale": 1}
        distant_room = {"translation": [10, 0], "rotation": 0, "scale": 1}
        payload = {
            "merger": {
                "floor_01": {
                    "complete_room_01": {
                        "partial_room_01": {
                            "pano_1": make_pano(
                                "panos/a.jpg", identity, [[1, 0], [1, 1]]
                            )
                        }
                    },
                    "complete_room_02": {
                        "partial_room_02": {
                            "pano_2": make_pano(
                                "panos/b.jpg", reversed_room, [[1, 0], [1, -1]]
                            )
                        }
                    },
                    "complete_room_03": {
                        "partial_room_03": {
                            "pano_3": make_pano(
                                "panos/c.jpg", distant_room, [[1, 0], [1, 1]]
                            )
                        }
                    },
                }
            }
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "0000" / "zind_data.json"
            json_path.parent.mkdir()
            json_path.write_text(json.dumps(payload), encoding="utf-8")

            records = mine_zind_matching_records(
                str(json_path),
                endpoint_tolerance=1e-4,
                negative_ratio=1.0,
                negative_min_endpoint_error=0.1,
                token_count=16,
                data_root=str(root),
            )

        positives = [record for record in records if record["supervision"]["is_match"]]
        negatives = [record for record in records if not record["supervision"]["is_match"]]
        self.assertEqual(len(positives), 1)
        self.assertEqual(len(negatives), 1)
        self.assertEqual(positives[0]["token_count"], 16)
        self.assertEqual(positives[0]["supervision"]["target_candidate_a"], 0)
        self.assertEqual(positives[0]["supervision"]["target_candidate_b"], 0)
        self.assertTrue(positives[0]["supervision"]["pose_valid"])
        self.assertFalse(negatives[0]["supervision"]["pose_valid"])
        self.assertEqual(negatives[0]["supervision"]["target_candidate_a"], -1)
        self.assertFalse(Path(positives[0]["image_a"]).is_absolute())


if __name__ == "__main__":
    unittest.main()

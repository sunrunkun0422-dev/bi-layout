import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from dataset.zind_bipair_builder import (
    PairThresholds,
    build_house_pair_examples,
    build_pair_record_and_arrays,
    build_view_label_arrays,
)
from dataset.zind_bipair_dataset import ZInDBiPairDataset, collate_zind_bipair


def make_layout(vertices, opening):
    return {
        "vertices": vertices,
        "openings": [opening[0], opening[1], [-1, 0.6]],
        "doors": [],
        "windows": [],
        "internal": [],
    }


def make_pano(image_path, transform, opening, vertices=None):
    vertices = vertices or [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    raw = make_layout(vertices, opening)
    visible = make_layout(
        [[point[0] * 1.2, point[1] * 1.2] for point in vertices], opening
    )
    complete = make_layout(
        [[point[0] * 1.5, point[1] * 1.5] for point in vertices], opening
    )
    return {
        "is_primary": True,
        "is_inside": True,
        "is_ceiling_flat": True,
        "label": "test room",
        "image_path": image_path,
        "camera_height": 1.6,
        "ceiling_height": 3.2,
        "floor_plan_transformation": transform,
        "layout_raw": raw,
        "layout_visible": visible,
        "layout_complete": complete,
    }


class ZInDBiPairBuilderTest(unittest.TestCase):
    def test_mines_partial_opening_pair_and_dense_training_labels(self):
        identity = {"translation": [0, 0], "rotation": 0, "scale": 1}
        reversed_room = {"translation": [2, 0], "rotation": 180, "scale": 1}
        distant_room = {"translation": [8, 0], "rotation": 0, "scale": 1}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            house = root / "0000"
            (house / "panos").mkdir(parents=True)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.fromarray(np.full((8, 16, 3), 128, dtype=np.uint8)).save(
                    house / "panos" / name
                )
            payload = {
                "scale_meters_per_coordinate": {"floor_01": 1.0},
                "merger": {
                    "floor_01": {
                        "complete_room_01": {
                            "partial_room_01": {
                                "pano_1": make_pano(
                                    "panos/a.jpg", identity, [[1, 0], [1, 0.5]]
                                )
                            },
                            "partial_room_02": {
                                "pano_2": make_pano(
                                    "panos/b.jpg",
                                    reversed_room,
                                    [[1, 0], [1, -0.5]],
                                )
                            },
                            "partial_room_03": {
                                "pano_3": make_pano(
                                    "panos/c.jpg",
                                    distant_room,
                                    [[1, 0], [1, 0.5]],
                                )
                            },
                        }
                    }
                },
            }

            positives, negatives, invalid, eligible = build_house_pair_examples(
                payload,
                "0000",
                root,
                256,
                PairThresholds(),
            )

            self.assertEqual(eligible, 3)
            self.assertFalse(invalid)
            self.assertEqual(len(positives), 1)
            self.assertEqual(len(negatives), 2)
            match = positives[0].portal_match
            self.assertAlmostEqual(match.endpoint_error_m, 0.0, places=7)
            cache = {
                view.view_id: build_view_label_arrays(view, 256)
                for view in (positives[0].view_a, positives[0].view_b)
            }
            record, arrays = build_pair_record_and_arrays(
                positives[0], "train", "labels/train/example.npz", 256, cache
            )

        self.assertEqual(record["pair_type"], "partial_opening")
        self.assertTrue(record["is_positive"])
        self.assertEqual(record["shared_portal"]["type"], "opening")
        self.assertEqual(arrays["depth_enclosed_A"].shape, (256,))
        self.assertEqual(arrays["depth_extended_B"].shape, (256,))
        self.assertGreater(arrays["portal_mask_A"].sum(), 0)
        self.assertGreater(arrays["affinity_gt"].sum(), 0)
        self.assertEqual(arrays["T_B_to_A"].shape, (3, 3))
        self.assertTrue(np.isfinite(arrays["joint_layout_overlap_ratio"]).all())

    def test_loader_reads_generated_jsonl_and_npz_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "ZInD-BiPair-v1"
            data_root = root / "data"
            (dataset_root / "manifests").mkdir(parents=True)
            (dataset_root / "labels/train").mkdir(parents=True)
            (data_root / "0000/panos").mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                Image.fromarray(np.full((8, 16, 3), 64, dtype=np.uint8)).save(
                    data_root / "0000/panos" / name
                )
            info = {"source": {"dataRoot": str(data_root)}}
            (dataset_root / "dataset_info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
            arrays = {
                "depth_enclosed_A": np.ones(256, dtype=np.float32),
                "depth_enclosed_B": np.ones(256, dtype=np.float32),
                "affinity_gt": np.eye(256, dtype=np.uint8),
                "corners_enclosed_A": np.zeros((4, 2), dtype=np.float32),
                "corners_enclosed_B": np.zeros((6, 2), dtype=np.float32),
                "is_positive": np.ones(1, dtype=np.uint8),
            }
            np.savez_compressed(dataset_root / "labels/train/pair.npz", **arrays)
            record = {
                "pair_id": "pair",
                "pair_type": "partial_opening",
                "house_id": "0000",
                "floor_id": "floor_01",
                "complete_room_id": "complete_room_01",
                "is_positive": True,
                "label_cache": "labels/train/pair.npz",
                "view_A": {
                    "image_path": "0000/panos/a.jpg",
                    "partial_room_id": "partial_room_01",
                    "pano_id": "pano_1",
                },
                "view_B": {
                    "image_path": "0000/panos/b.jpg",
                    "partial_room_id": "partial_room_02",
                    "pano_id": "pano_2",
                },
            }
            (dataset_root / "manifests/train_pairs.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            dataset = ZInDBiPairDataset(
                str(dataset_root / "manifests/train_pairs.jsonl"),
                image_shape=(8, 16),
            )
            sample = dataset[0]
            batch = collate_zind_bipair([sample])

        self.assertEqual(sample["image_A"].shape, (3, 8, 16))
        self.assertEqual(batch["depth_enclosed_A"].shape, (1, 256))
        self.assertEqual(batch["affinity_gt"].shape, (1, 256, 256))
        self.assertEqual(batch["is_positive"].shape, (1,))
        self.assertEqual(batch["is_positive"].dtype, torch.bool)
        self.assertEqual(len(batch["corners_enclosed_B"]), 1)


if __name__ == "__main__":
    unittest.main()

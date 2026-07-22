import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dataset.zind_bipair_adapter import adapt_zind_bipair_batch
from dataset.zind_bipair_dataset import collate_zind_bipair
from dataset.zind_bipair_feature_dataset import ZInDBiPairFeatureDataset


class ZInDBiPairFeatureDatasetTest(unittest.TestCase):
    def test_joins_frozen_features_and_preserves_ground_truth_depth(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            dataset_root = root / "pairs"
            manifest_dir = dataset_root / "manifests"
            labels_dir = dataset_root / "labels/train"
            cache_dir = root / "cache"
            for path in (data_root, manifest_dir, labels_dir, cache_dir):
                path.mkdir(parents=True, exist_ok=True)

            dataset_info = {
                "tokenCount": 8,
                "source": {"dataRoot": str(data_root)},
            }
            (dataset_root / "dataset_info.json").write_text(
                json.dumps(dataset_info), encoding="utf-8"
            )
            record = {
                "pair_id": "pair_0",
                "pair_type": "partial_opening",
                "house_id": "0000",
                "floor_id": "floor_01",
                "complete_room_id": "complete_room_01",
                "label_cache": "labels/train/pair_0.npz",
                "is_positive": True,
                "view_A": {
                    "image_path": "a.jpg",
                    "partial_room_id": "partial_a",
                    "pano_id": "pano_a",
                },
                "view_B": {
                    "image_path": "b.jpg",
                    "partial_room_id": "partial_b",
                    "pano_id": "pano_b",
                },
            }
            manifest = manifest_dir / "train_pairs.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            opening_a = np.asarray([1, 1, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
            opening_b = np.asarray([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
            np.savez(
                labels_dir / "pair_0.npz",
                depth_enclosed_A=np.full(8, 10, dtype=np.float32),
                depth_enclosed_B=np.full(8, 20, dtype=np.float32),
                depth_extended_A=np.full(8, 30, dtype=np.float32),
                depth_extended_B=np.full(8, 40, dtype=np.float32),
                opening_mask_all_A=opening_a,
                opening_mask_all_B=opening_b,
                portal_mask_A=opening_a,
                portal_mask_B=opening_b,
                affinity_gt=np.outer(opening_a, opening_b).astype(np.uint8),
                T_B_to_A=np.eye(3, dtype=np.float32),
                relative_yaw_gt=np.asarray([0.0], dtype=np.float32),
                scale_meters_per_coordinate=np.asarray([2.0], dtype=np.float32),
            )

            views = [
                {"image_path": str(data_root / "a.jpg"), "relative_image_path": "a.jpg"},
                {"image_path": str(data_root / "b.jpg"), "relative_image_path": "b.jpg"},
            ]
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "sample_count": 2,
                        "token_count": 8,
                        "feature_dim": 4,
                        "branch_order": "extended_first",
                        "views": views,
                    }
                ),
                encoding="utf-8",
            )
            np.save(cache_dir / "features.npy", np.arange(64, dtype=np.float16).reshape(2, 8, 4))
            np.save(
                cache_dir / "enclosed_depth.npy",
                np.asarray([[1] * 8, [2] * 8], dtype=np.float32),
            )
            np.save(
                cache_dir / "extended_depth.npy",
                np.asarray([[3] * 8, [4] * 8], dtype=np.float32),
            )

            dataset = ZInDBiPairFeatureDataset(
                str(manifest), str(cache_dir), validate_paths=False
            )
            sample = dataset[0]
            batch = collate_zind_bipair([sample])
            canonical = adapt_zind_bipair_batch(batch)

        self.assertEqual(tuple(sample["feature_A"].shape), (8, 4))
        self.assertTrue(torch.all(sample["depth_enclosed_A"] == 1))
        self.assertTrue(torch.all(sample["depth_enclosed_gt_A"] == 10))
        self.assertTrue(torch.all(canonical.depth_extended_b == 4))
        self.assertEqual(canonical.candidates_a.intervals.tolist(), [[[7, 1]]])

    def test_merges_multiple_true_portals_without_cross_product_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            dataset_root = root / "pairs"
            manifest_dir = dataset_root / "manifests"
            labels_dir = dataset_root / "labels/train"
            cache_dir = root / "cache"
            for path in (data_root, manifest_dir, labels_dir, cache_dir):
                path.mkdir(parents=True, exist_ok=True)
            (dataset_root / "dataset_info.json").write_text(
                json.dumps({"tokenCount": 8, "source": {"dataRoot": str(data_root)}}),
                encoding="utf-8",
            )
            opening = np.asarray([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.uint8)
            records = []
            for index, (token_a, token_b) in enumerate(((1, 3), (5, 7))):
                portal_a = np.zeros(8, dtype=np.uint8)
                portal_b = np.zeros(8, dtype=np.uint8)
                portal_a[token_a] = 1
                portal_b[token_b] = 1
                label_name = f"pair_{index}.npz"
                np.savez(
                    labels_dir / label_name,
                    depth_enclosed_A=np.ones(8, dtype=np.float32),
                    depth_enclosed_B=np.ones(8, dtype=np.float32),
                    depth_extended_A=np.ones(8, dtype=np.float32) * 2,
                    depth_extended_B=np.ones(8, dtype=np.float32) * 2,
                    opening_mask_all_A=opening,
                    opening_mask_all_B=np.roll(opening, 2),
                    portal_mask_A=portal_a,
                    portal_mask_B=portal_b,
                    affinity_gt=np.outer(portal_a, portal_b).astype(np.uint8),
                    T_B_to_A=np.eye(3, dtype=np.float32),
                    relative_yaw_gt=np.asarray([0.0], dtype=np.float32),
                    scale_meters_per_coordinate=np.asarray([1.0], dtype=np.float32),
                )
                records.append(
                    {
                        "pair_id": f"pair_{index}",
                        "pair_type": "partial_opening",
                        "house_id": "0000",
                        "floor_id": "floor_01",
                        "complete_room_id": "complete_room_01",
                        "label_cache": f"labels/train/{label_name}",
                        "is_positive": True,
                        "view_A": {
                            "image_path": "a.jpg",
                            "partial_room_id": "partial_a",
                            "pano_id": "pano_a",
                        },
                        "view_B": {
                            "image_path": "b.jpg",
                            "partial_room_id": "partial_b",
                            "pano_id": "pano_b",
                        },
                    }
                )
            (manifest_dir / "train_pairs.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            views = [
                {"image_path": str(data_root / name), "relative_image_path": name}
                for name in ("a.jpg", "b.jpg")
            ]
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "sample_count": 2,
                        "token_count": 8,
                        "feature_dim": 4,
                        "branch_order": "extended_first",
                        "views": views,
                    }
                ),
                encoding="utf-8",
            )
            np.save(cache_dir / "features.npy", np.zeros((2, 8, 4), dtype=np.float16))
            np.save(cache_dir / "enclosed_depth.npy", np.ones((2, 8), dtype=np.float32))
            np.save(cache_dir / "extended_depth.npy", np.ones((2, 8), dtype=np.float32) * 2)

            dataset = ZInDBiPairFeatureDataset(
                str(manifest_dir / "train_pairs.jsonl"),
                str(cache_dir),
                validate_paths=False,
            )
            sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["merged_pair_count"].item(), 2)
        self.assertEqual(sample["portal_mask_A"].sum().item(), 2)
        self.assertEqual(sample["portal_mask_B"].sum().item(), 2)
        self.assertEqual(sample["affinity_gt"].sum().item(), 2)
        self.assertTrue(sample["affinity_gt"][1, 3])
        self.assertTrue(sample["affinity_gt"][5, 7])
        self.assertFalse(sample["affinity_gt"][1, 7])


if __name__ == "__main__":
    unittest.main()

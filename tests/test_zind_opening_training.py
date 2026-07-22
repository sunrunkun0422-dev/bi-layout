import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from dataset.zind_opening_dataset import (
    OpeningFeatureCacheDataset,
    ZInDOpeningViewDataset,
    synchronize_opening_augmentation,
)
from tools.train_zind_opening_head import (
    select_validation_threshold,
    train_from_cache,
)


def _write_cache(path: Path, sample_count: int, seed: int) -> None:
    path.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    token_count = 16
    feature_dim = 4
    features = rng.normal(size=(sample_count, token_count, feature_dim)).astype(
        np.float16
    )
    enclosed = np.ones((sample_count, token_count), dtype=np.float32)
    extended = enclosed.copy()
    targets = np.zeros((sample_count, token_count), dtype=np.uint8)
    for index in range(sample_count):
        start = 2 + index % 5
        targets[index, start : start + 4] = 1
        extended[index, start : start + 4] = 2.0
    np.save(str(path / "features.npy"), features)
    np.save(str(path / "enclosed_depth.npy"), enclosed)
    np.save(str(path / "extended_depth.npy"), extended)
    np.save(str(path / "targets.npy"), targets)
    metadata = {
        "complete": True,
        "sample_count": sample_count,
        "token_count": token_count,
        "feature_dim": feature_dim,
        "views": [{"index": index} for index in range(sample_count)],
        "manifest": {"path": "synthetic.jsonl", "sha256": "synthetic"},
        "bi_layout_config": {"path": "synthetic.yaml", "sha256": "synthetic"},
        "bi_layout_checkpoint": {"path": "synthetic.pt", "size": 1},
        "image_shape": [32, 64],
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _training_args(epochs: int, resume=None, overwrite: bool = False):
    return SimpleNamespace(
        roll_probability=1.0,
        flip_probability=0.5,
        batch_size=2,
        workers=0,
        seed=7,
        hidden_dim=8,
        kernel_size=3,
        prior_strength=4.0,
        prior_relative_scale=0.1,
        lr=1e-3,
        weight_decay=0.0,
        resume=resume,
        overwrite=overwrite,
        epochs=epochs,
        pos_weight=2.5,
        tversky_weight=0.5,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        grad_clip=5.0,
        scan_min=0.0,
        scan_max=1.0,
        scan_steps=21,
        precision_target=0.8,
    )


class ZInDOpeningDatasetTest(unittest.TestCase):
    def test_unique_view_dataset_uses_side_specific_opening_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "ZInD-BiPair-v1"
            manifest_dir = dataset_root / "manifests"
            label_dir = dataset_root / "labels/train"
            data_root = root / "data"
            manifest_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (data_root / "house/panos").mkdir(parents=True)
            for name, value in (("a.jpg", 64), ("b.jpg", 192)):
                Image.new("RGB", (16, 8), (value, value, value)).save(
                    data_root / "house/panos" / name
                )
            (dataset_root / "dataset_info.json").write_text(
                json.dumps(
                    {
                        "tokenCount": 4,
                        "source": {"dataRoot": str(data_root)},
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                str(label_dir / "pair.npz"),
                opening_mask_all_A=np.asarray([1, 0, 0, 0], dtype=np.uint8),
                opening_mask_all_B=np.asarray([0, 0, 1, 1], dtype=np.uint8),
            )
            record = {
                "pair_id": "pair",
                "label_cache": "labels/train/pair.npz",
                "view_A": {"image_path": "house/panos/a.jpg"},
                "view_B": {"image_path": "house/panos/b.jpg"},
            }
            manifest = manifest_dir / "train_pairs.jsonl"
            manifest.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            dataset = ZInDOpeningViewDataset(
                str(manifest), image_shape=(8, 16)
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["index"], 0)
            self.assertEqual(tuple(dataset[0]["image"].shape), (3, 8, 16))
            torch.testing.assert_close(
                dataset[0]["target"], torch.tensor([1.0, 0.0, 0.0, 0.0])
            )
            torch.testing.assert_close(
                dataset[1]["target"], torch.tensor([0.0, 0.0, 1.0, 1.0])
            )

    def test_cached_augmentation_keeps_every_token_stream_aligned(self):
        feature = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        enclosed = torch.arange(4, dtype=torch.float32)
        extended = enclosed + 10
        target = enclosed + 20

        result = synchronize_opening_augmentation(
            feature, enclosed, extended, target, shift=1, flip=True
        )

        expected_order = torch.tensor([2.0, 1.0, 0.0, 3.0])
        torch.testing.assert_close(result[0][:, 0] / 3, expected_order)
        torch.testing.assert_close(result[1], expected_order)
        torch.testing.assert_close(result[2] - 10, expected_order)
        torch.testing.assert_close(result[3] - 20, expected_order)

    def test_cache_training_writes_best_and_resumes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_cache = root / "cache/train"
            val_cache = root / "cache/val"
            output = root / "output"
            _write_cache(train_cache, sample_count=6, seed=1)
            _write_cache(val_cache, sample_count=4, seed=2)

            dataset = OpeningFeatureCacheDataset(str(train_cache))
            self.assertEqual(tuple(dataset[0]["feature"].shape), (16, 4))
            first = train_from_cache(
                train_cache,
                val_cache,
                output,
                torch.device("cpu"),
                _training_args(epochs=1),
            )
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            self.assertEqual(first["epoch"], 0)

            second = train_from_cache(
                train_cache,
                val_cache,
                output,
                torch.device("cpu"),
                _training_args(epochs=2, resume="auto"),
            )
            checkpoint = torch.load(str(output / "last.pt"), map_location="cpu")
            self.assertEqual(second["epoch"], 1)
            self.assertEqual(checkpoint["next_epoch"], 2)
            self.assertIn("operating_threshold", checkpoint)
            self.assertEqual(
                len((output / "metrics.jsonl").read_text().splitlines()), 2
            )

    def test_validation_threshold_maximizes_recall_at_precision_target(self):
        args = SimpleNamespace(
            scan_min=0.0,
            scan_max=1.0,
            scan_steps=11,
            precision_target=0.8,
        )
        result = select_validation_threshold(
            [np.asarray([0, 0, 1, 1], dtype=np.uint8)],
            [np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)],
            args,
        )

        self.assertEqual(result["policy"], "max_recall_at_precision>=0.80")
        selected = result["evaluation"]["metricsAtSelectedThreshold"]
        self.assertEqual(selected["precision"], 1.0)
        self.assertEqual(selected["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

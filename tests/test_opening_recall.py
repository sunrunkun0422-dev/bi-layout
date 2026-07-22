import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluation.opening_recall import (
    UniqueViewReference,
    binary_average_precision,
    circular_components,
    circular_interval_iou,
    circular_interval_mask,
    deduplicate_manifest_views,
    evaluate_opening_scores,
    opening_geometry_probability,
    resolve_depth_branches,
    scan_opening_thresholds,
)
from models.cross_scene_matcher import OpeningSignalHead
from tools.evaluate_zind_opening_recall import (
    _learned_opening_probability,
    _load_opening_head,
    _predicted_depth_scores,
    _resolve_evaluation_threshold,
    _validate_args,
    parse_args,
)


class OpeningRecallTest(unittest.TestCase):
    def test_circular_components_merge_across_panorama_seam(self):
        mask = np.asarray([1, 1, 0, 0, 1], dtype=np.uint8)

        components = circular_components(mask)

        self.assertEqual(components, [(4, 1)])
        np.testing.assert_array_equal(
            circular_interval_mask(components[0], len(mask)), mask.astype(bool)
        )
        self.assertAlmostEqual(
            circular_interval_iou((4, 1), (4, 0), len(mask)), 2.0 / 3.0
        )
        self.assertEqual(circular_components(np.zeros(5)), [])
        self.assertEqual(circular_components(np.ones(5)), [(0, 4)])

    def test_manifest_views_are_deduplicated_by_image_path_in_stable_order(self):
        records = [
            {
                "pair_id": "pair_1",
                "house_id": "0000",
                "floor_id": "floor_01",
                "complete_room_id": "complete_room_01",
                "label_cache": "labels/val/pair_1.npz",
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
            },
            {
                "pair_id": "pair_2",
                "house_id": "0000",
                "floor_id": "floor_01",
                "complete_room_id": "complete_room_01",
                "label_cache": "labels/val/pair_2.npz",
                "view_A": {
                    "image_path": "0000/panos/c.jpg",
                    "partial_room_id": "partial_room_03",
                    "pano_id": "pano_3",
                },
                "view_B": {
                    "image_path": "0000/panos/a.jpg",
                    "partial_room_id": "partial_room_01",
                    "pano_id": "pano_1",
                },
            },
        ]

        views = deduplicate_manifest_views(records)

        self.assertEqual(
            [view.image_path for view in views],
            ["0000/panos/a.jpg", "0000/panos/b.jpg", "0000/panos/c.jpg"],
        )
        self.assertEqual(views[0].side, "A")
        self.assertEqual(views[0].label_cache, "labels/val/pair_1.npz")
        self.assertEqual(views[2].pair_id, "pair_2")

    def test_depth_branch_order_reproduces_correct_and_reversed_signal(self):
        extended_first = np.ones(8, dtype=np.float32)
        extended_first[3:6] = 2.0
        enclosed_second = np.ones(8, dtype=np.float32)

        enclosed, extended = resolve_depth_branches(
            extended_first, enclosed_second, "extended_first"
        )
        correct = opening_geometry_probability(enclosed, extended)
        reversed_enclosed, reversed_extended = resolve_depth_branches(
            extended_first, enclosed_second, "enclosed_first"
        )
        reversed_score = opening_geometry_probability(
            reversed_enclosed, reversed_extended
        )

        self.assertGreater(float(correct[3:6].mean()), float(correct[:3].mean()))
        np.testing.assert_allclose(
            reversed_score,
            np.full(8, 1.0 / (1.0 + np.exp(2.0)), dtype=np.float32),
            rtol=1e-6,
        )

    def test_average_precision_is_tie_aware(self):
        targets = np.asarray([1, 0, 1, 0], dtype=np.uint8)
        scores = np.full(4, 0.5, dtype=np.float32)

        self.assertAlmostEqual(binary_average_precision(targets, scores), 0.5)

    def test_threshold_scan_reports_both_selection_policies(self):
        targets = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        scores = np.asarray([0.9, 0.6, 0.7, 0.1], dtype=np.float32)

        result = scan_opening_thresholds(
            targets,
            scores,
            thresholds=[0.5, 0.65, 0.85],
            precision_target=0.85,
        )

        self.assertEqual(result["bestF1"]["threshold"], 0.5)
        self.assertEqual(
            result["precisionTargetMaxRecall"]["threshold"], 0.85
        )
        self.assertAlmostEqual(
            result["precisionTargetMaxRecall"]["precision"], 1.0
        )
        self.assertAlmostEqual(
            result["precisionTargetMaxRecall"]["recall"], 0.5
        )

    def test_evaluation_reports_token_and_wrap_component_recall(self):
        targets = [np.asarray([1, 1, 0, 0, 1], dtype=np.uint8)]
        scores = [np.asarray([0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float32)]

        result = evaluate_opening_scores(targets, scores, threshold=0.5)

        metrics = result["metricsAtSelectedThreshold"]
        self.assertEqual(metrics["truePositive"], 3)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        components = result["componentMetricsAtSelectedThreshold"]
        self.assertEqual(components["groundTruthComponentCount"], 1)
        self.assertEqual(components["predictedComponentCount"], 1)
        self.assertAlmostEqual(components["overlap"]["iouAtLeast0.50"]["recall"], 1.0)

    def test_learned_checkpoint_restores_saved_model_configuration_and_weights(self):
        original = OpeningSignalHead(
            feature_dim=4,
            hidden_dim=6,
            kernel_size=3,
            prior_strength=2.5,
            prior_relative_scale=0.2,
        )
        with torch.no_grad():
            original.output.bias[0] = 1.25
        payload = {
            "format_version": 1,
            "task": "ZInD-BiPair-v1 single-view Opening Head",
            "completed_epoch": 3,
            "operating_threshold": 0.37,
            "threshold_policy": "max_recall_at_precision>=0.80",
            "threshold_fallback": False,
            "model": {
                "feature_dim": 4,
                "hidden_dim": 6,
                "kernel_size": 3,
                "prior_strength": 2.5,
                "prior_relative_scale": 0.2,
                "branch_order": "extended_first",
            },
            "opening_head_state_dict": original.state_dict(),
        }

        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            torch.save(payload, str(checkpoint_path))
            restored, report = _load_opening_head(
                checkpoint_path,
                torch.device("cpu"),
                expected_feature_dim=4,
                branch_order="extended_first",
            )

        self.assertFalse(restored.training)
        self.assertEqual(restored.input_projection.in_channels, 7)
        self.assertEqual(restored.input_projection.out_channels, 6)
        torch.testing.assert_close(restored.output.bias, original.output.bias)
        self.assertEqual(report["operatingThreshold"], 0.37)
        self.assertEqual(report["completedEpoch"], 3)
        self.assertEqual(report["model"]["kernel_size"], 3)

    def test_learned_checkpoint_rejects_feature_or_branch_contract_mismatch(self):
        head = OpeningSignalHead(feature_dim=4, hidden_dim=6, kernel_size=3)
        payload = {
            "model": {
                "feature_dim": 4,
                "hidden_dim": 6,
                "kernel_size": 3,
                "prior_strength": 4.0,
                "prior_relative_scale": 0.1,
                "branch_order": "extended_first",
            },
            "opening_head_state_dict": head.state_dict(),
            "operating_threshold": 0.4,
        }
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            torch.save(payload, str(checkpoint_path))
            with self.assertRaisesRegex(ValueError, "feature_dim"):
                _load_opening_head(
                    checkpoint_path,
                    torch.device("cpu"),
                    expected_feature_dim=8,
                    branch_order="extended_first",
                )
            with self.assertRaisesRegex(ValueError, "branch_order"):
                _load_opening_head(
                    checkpoint_path,
                    torch.device("cpu"),
                    expected_feature_dim=4,
                    branch_order="enclosed_first",
                )

    def test_learned_scoring_consumes_layout_feature_and_resolved_depths(self):
        class CapturingHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = None

            def forward(self, feature, enclosed, extended):
                self.inputs = (feature, enclosed, extended)
                return {"opening_probability": torch.sigmoid(feature[..., 0])}

        feature = torch.tensor(
            [[[0.0, 1.0], [2.0, 3.0], [-2.0, 4.0]]], dtype=torch.float32
        )
        enclosed = torch.ones(1, 3)
        extended = torch.full((1, 3), 2.0)
        output = {
            "layout_feature": feature,
            "depth": extended,
            "new_depth": enclosed,
        }
        head = CapturingHead()

        probability = _learned_opening_probability(
            output, head, "extended_first"
        )

        torch.testing.assert_close(head.inputs[0], feature)
        torch.testing.assert_close(head.inputs[1], enclosed)
        torch.testing.assert_close(head.inputs[2], extended)
        torch.testing.assert_close(probability, torch.sigmoid(feature[..., 0]))

    def test_predicted_depth_path_runs_loaded_head_on_returned_features(self):
        class DummyBiLayout(torch.nn.Module):
            patch_num = 3
            patch_dim = 2

            def __init__(self):
                super().__init__()
                self.return_features = None

            def forward(self, images, return_features=False):
                self.return_features = return_features
                enclosed = torch.ones(images.shape[0], self.patch_num)
                extended = enclosed.clone()
                extended[:, 1] = 2.0
                output = {
                    "depth": extended,
                    "new_depth": enclosed,
                }
                if return_features:
                    output["layout_feature"] = torch.zeros(
                        images.shape[0], self.patch_num, self.patch_dim
                    )
                return output

        learned = OpeningSignalHead(
            feature_dim=2,
            hidden_dim=4,
            kernel_size=3,
            prior_strength=4.0,
            prior_relative_scale=0.1,
        )
        with torch.no_grad():
            learned.output.bias[0] = 2.0
        payload = {
            "format_version": 1,
            "task": "ZInD-BiPair-v1 single-view Opening Head",
            "completed_epoch": 2,
            "operating_threshold": 0.37,
            "threshold_policy": "max_recall_at_precision>=0.80",
            "threshold_fallback": False,
            "model": {
                "feature_dim": 2,
                "hidden_dim": 4,
                "kernel_size": 3,
                "prior_strength": 4.0,
                "prior_relative_scale": 0.1,
                "branch_order": "extended_first",
            },
            "opening_head_state_dict": learned.state_dict(),
        }
        view = UniqueViewReference(
            image_path="house/pano.jpg",
            label_cache="labels/test/pair.npz",
            side="A",
            pair_id="pair",
            house_id="house",
            floor_id="floor",
            complete_room_id="complete",
            partial_room_id="partial",
            pano_id="pano",
        )
        model = DummyBiLayout()
        target = np.asarray([0, 1, 0], dtype=np.uint8)
        cached_extended = np.asarray([1.0, 2.0, 1.0], dtype=np.float32)
        cached_enclosed = np.ones(3, dtype=np.float32)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            backbone_path = root / "backbone.pt"
            opening_path = root / "best.pt"
            config_path.touch()
            backbone_path.touch()
            torch.save(payload, str(opening_path))
            args = SimpleNamespace(
                torch_threads=1,
                device="cpu",
                config=str(config_path),
                checkpoint=str(backbone_path),
                opening_checkpoint=str(opening_path),
                branch_order="extended_first",
                batch_size=1,
                image_height=8,
                image_width=16,
                progress_every=1,
                prior_strength=4.0,
                prior_relative_scale=0.1,
            )
            with patch(
                "tools.debug_cross_scene_flow.load_bi_layout",
                return_value=(model, {"loaded": True}),
            ), patch(
                "tools.evaluate_zind_opening_recall._load_image",
                return_value=torch.zeros(3, 8, 16),
            ), patch(
                "tools.evaluate_zind_opening_recall._load_view_cache",
                return_value=(target, cached_extended, cached_enclosed),
            ):
                targets, scores, report = _predicted_depth_scores(
                    root, root, [view], 3, args
                )

        self.assertTrue(model.return_features)
        np.testing.assert_array_equal(targets[0], target)
        self.assertGreater(float(scores[0][0]), 0.49)
        self.assertGreater(float(scores[0][1]), float(scores[0][0]))
        self.assertTrue(report["openingScore"]["isLearnedOpeningCheckpoint"])
        self.assertEqual(
            report["openingHeadCheckpoint"]["operatingThreshold"], 0.37
        )

    def test_test_threshold_comes_from_checkpoint_and_explicit_value_wins(self):
        args = SimpleNamespace(split="test", threshold=None)
        source = {
            "openingHeadCheckpoint": {"operatingThreshold": 0.37}
        }

        threshold, provenance = _resolve_evaluation_threshold(args, source)

        self.assertEqual(threshold, 0.37)
        self.assertEqual(provenance, "opening_checkpoint")
        args.threshold = 0.55
        threshold, provenance = _resolve_evaluation_threshold(args, source)
        self.assertEqual(threshold, 0.55)
        self.assertEqual(provenance, "explicit_cli")

    def test_test_never_scans_when_checkpoint_threshold_is_missing(self):
        args = SimpleNamespace(split="test", threshold=None)

        with self.assertRaisesRegex(ValueError, "scanning is forbidden"):
            _resolve_evaluation_threshold(
                args,
                {"openingHeadCheckpoint": {"operatingThreshold": None}},
            )

        val_args = SimpleNamespace(split="val", threshold=None)
        threshold, provenance = _resolve_evaluation_threshold(val_args, {})
        self.assertIsNone(threshold)
        self.assertEqual(provenance, "validation_scan")

    def test_cli_preserves_legacy_fixed_test_threshold_contract(self):
        learned = parse_args(
            [
                "--source",
                "predicted-depth",
                "--split",
                "test",
                "--opening_checkpoint",
                "best.pt",
            ]
        )
        _validate_args(learned)

        legacy_without_threshold = parse_args(
            ["--source", "predicted-depth", "--split", "test"]
        )
        with self.assertRaisesRegex(ValueError, "fixed --threshold"):
            _validate_args(legacy_without_threshold)

        legacy_with_threshold = parse_args(
            [
                "--source",
                "predicted-depth",
                "--split",
                "test",
                "--threshold",
                "0.12",
            ]
        )
        _validate_args(legacy_with_threshold)

        invalid_gt = parse_args(
            ["--source", "gt-depth", "--opening_checkpoint", "best.pt"]
        )
        with self.assertRaisesRegex(ValueError, "predicted-depth"):
            _validate_args(invalid_gt)


if __name__ == "__main__":
    unittest.main()

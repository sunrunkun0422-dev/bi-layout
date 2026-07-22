import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    OpeningSignalHead,
)
from utils.opening_checkpoint import (
    load_opening_head_checkpoint,
    resolve_opening_probability_threshold,
)


def _identity(path: Path, with_hash: bool):
    stat = path.stat()
    result = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if with_hash:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class OpeningCheckpointTest(unittest.TestCase):
    def _checkpoint(self, directory: Path, head: OpeningSignalHead):
        config_path = directory / "zind.yaml"
        backbone_path = directory / "bilayout.pt"
        config_path.write_bytes(b"MODEL: synthetic\n")
        backbone_path.write_bytes(b"frozen-bilayout")
        checkpoint_path = directory / "opening.pt"
        payload = {
            "format_version": 1,
            "task": "synthetic opening head",
            "completed_epoch": 7,
            "operating_threshold": 0.375,
            "threshold_policy": "validation",
            "threshold_fallback": False,
            "model": {
                "feature_dim": 8,
                "hidden_dim": 6,
                "kernel_size": 5,
                "prior_strength": 4.0,
                "prior_relative_scale": 0.1,
                "branch_order": "extended_first",
            },
            "train_cache_contract": {
                "feature_dim": 8,
                "token_count": 16,
                "bi_layout_config": _identity(config_path, with_hash=True),
                "bi_layout_checkpoint": _identity(
                    backbone_path, with_hash=False
                ),
            },
            "opening_head_state_dict": head.state_dict(),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path, config_path, backbone_path, payload

    def test_loads_weights_threshold_and_validates_backbone_contract(self):
        source = OpeningSignalHead(feature_dim=8, hidden_dim=6, kernel_size=5)
        with torch.no_grad():
            source.output.bias.fill_(1.25)
        target = OpeningSignalHead(feature_dim=8, hidden_dim=6, kernel_size=5)

        with TemporaryDirectory() as directory:
            checkpoint, config, backbone, _ = self._checkpoint(
                Path(directory), source
            )
            report = load_opening_head_checkpoint(
                target,
                str(checkpoint),
                expected_feature_dim=8,
                expected_branch_order="extended_first",
                expected_token_count=16,
                bi_layout_config_path=str(config),
                bi_layout_checkpoint_path=str(backbone),
            )

        torch.testing.assert_close(target.output.bias, source.output.bias)
        self.assertFalse(target.training)
        self.assertEqual(report["operatingThreshold"], 0.375)
        self.assertEqual(report["branchOrder"], "extended_first")
        self.assertTrue(report["backboneContract"]["validated"])

    def test_explicit_opening_checkpoint_overrides_only_matcher_opening_head(self):
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=8, hidden_dim=6, heads=2
        )
        source = OpeningSignalHead(feature_dim=8, hidden_dim=6, kernel_size=5)
        with torch.no_grad():
            matcher.query_projection.weight.fill_(0.75)
            matcher.opening_head.output.bias.fill_(-2.0)
            source.output.bias.fill_(2.0)
        query_before = matcher.query_projection.weight.detach().clone()

        with TemporaryDirectory() as directory:
            checkpoint, config, backbone, _ = self._checkpoint(
                Path(directory), source
            )
            load_opening_head_checkpoint(
                matcher.opening_head,
                str(checkpoint),
                expected_feature_dim=8,
                expected_branch_order="extended_first",
                expected_token_count=16,
                bi_layout_config_path=str(config),
                bi_layout_checkpoint_path=str(backbone),
            )

        torch.testing.assert_close(matcher.opening_head.output.bias, source.output.bias)
        torch.testing.assert_close(matcher.query_projection.weight, query_before)

    def test_rejects_branch_token_and_backbone_mismatch(self):
        source = OpeningSignalHead(feature_dim=8, hidden_dim=6, kernel_size=5)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, backbone, _ = self._checkpoint(root, source)
            target = OpeningSignalHead(feature_dim=8, hidden_dim=6, kernel_size=5)
            common = {
                "expected_feature_dim": 8,
                "expected_token_count": 16,
                "bi_layout_config_path": str(config),
                "bi_layout_checkpoint_path": str(backbone),
            }
            with self.assertRaisesRegex(ValueError, "branch_order"):
                load_opening_head_checkpoint(
                    target,
                    str(checkpoint),
                    expected_branch_order="enclosed_first",
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "token_count"):
                load_opening_head_checkpoint(
                    target,
                    str(checkpoint),
                    expected_branch_order="extended_first",
                    **{**common, "expected_token_count": 32},
                )
            config.write_bytes(b"MODEL: changed\n")
            with self.assertRaisesRegex(ValueError, "bi_layout_config"):
                load_opening_head_checkpoint(
                    target,
                    str(checkpoint),
                    expected_branch_order="extended_first",
                    **common,
                )

    def test_threshold_precedence_and_cli_defaults(self):
        report = {"loaded": True, "operatingThreshold": 0.375}
        self.assertEqual(
            resolve_opening_probability_threshold(0.6, report),
            {"value": 0.6, "source": "cli"},
        )
        self.assertEqual(
            resolve_opening_probability_threshold(None, report),
            {"value": 0.375, "source": "opening_checkpoint"},
        )
        self.assertEqual(
            resolve_opening_probability_threshold(None, None),
            {"value": 0.12, "source": "legacy_default"},
        )
        matcher_report = {
            "loaded": True,
            "openingProbabilityThreshold": 0.42,
        }
        self.assertEqual(
            resolve_opening_probability_threshold(None, None, matcher_report),
            {"value": 0.42, "source": "matcher_checkpoint"},
        )
        self.assertEqual(
            resolve_opening_probability_threshold(None, report, matcher_report),
            {"value": 0.375, "source": "opening_checkpoint"},
        )

        from tools import debug_cross_scene_flow
        from tools import evaluate_zind_cross_scene_pairs

        with patch.object(sys, "argv", ["debug_cross_scene_flow.py"]):
            debug_args = debug_cross_scene_flow.parse_args()
        with patch.object(sys, "argv", ["evaluate_zind_cross_scene_pairs.py"]):
            eval_args = evaluate_zind_cross_scene_pairs.parse_args()
        self.assertIsNone(debug_args.opening_probability_threshold)
        self.assertIsNone(eval_args.opening_probability_threshold)
        self.assertEqual(debug_args.branch_order, "extended_first")
        self.assertEqual(eval_args.branch_order, "extended_first")


if __name__ == "__main__":
    unittest.main()

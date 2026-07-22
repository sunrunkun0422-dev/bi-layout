import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from models.cross_scene_matcher import OpeningGuidedCrossAttentionMatcher
from utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint
from utils.opening_checkpoint import resolve_opening_probability_threshold


def _identity(path: Path, with_hash: bool):
    stat = path.stat()
    value = {"path": str(path), "size": stat.st_size, "mtimeNs": stat.st_mtime_ns}
    if with_hash:
        value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


class MatcherCheckpointTest(unittest.TestCase):
    def _write_checkpoint(self, root: Path):
        config = root / "config.yaml"
        backbone = root / "bilayout.pt"
        config.write_bytes(b"MODEL: synthetic\n")
        backbone.write_bytes(b"synthetic-backbone")
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=8,
            heads=2,
            hidden_dim=6,
            dropout=0.2,
            opening_bias_strength=1.5,
            candidate_temperature=0.3,
            shift_temperature=0.07,
        )
        with torch.no_grad():
            matcher.dustbin_score.fill_(0.75)
        payload = {
            "format_version": 1,
            "task": "synthetic cross-scene matcher",
            "completed_epoch": 3,
            "matcher_state_dict": matcher.state_dict(),
            "opening_probability_threshold": 0.375,
            "model": {
                "feature_dim": 8,
                "heads": 2,
                "hidden_dim": 6,
                "dropout": 0.2,
                "opening_bias_strength": 1.5,
                "candidate_temperature": 0.3,
                "shift_temperature": 0.07,
                "has_dustbin": True,
            },
            "opening_head_dependency": {
                "branchOrder": "extended_first",
                "model": {
                    "feature_dim": 8,
                    "hidden_dim": 6,
                    "kernel_size": 5,
                    "prior_strength": 4.0,
                    "prior_relative_scale": 0.1,
                },
                "backboneContract": {
                    "tokenCount": 16,
                    "featureDim": 8,
                    "biLayoutConfig": _identity(config, True),
                    "biLayoutCheckpoint": _identity(backbone, False),
                },
            },
        }
        checkpoint = root / "matcher.pt"
        torch.save(payload, checkpoint)
        return checkpoint, config, backbone

    def test_rebuilds_saved_architecture_and_restores_threshold(self):
        with TemporaryDirectory() as directory:
            checkpoint, config, backbone = self._write_checkpoint(Path(directory))
            matcher, report = load_cross_scene_matcher_checkpoint(
                str(checkpoint),
                expected_feature_dim=8,
                expected_token_count=16,
                expected_branch_order="extended_first",
                bi_layout_config_path=str(config),
                bi_layout_checkpoint_path=str(backbone),
                device=torch.device("cpu"),
            )

        self.assertEqual(matcher.heads, 2)
        self.assertAlmostEqual(matcher.dropout.p, 0.2)
        self.assertAlmostEqual(matcher.candidate_temperature, 0.3)
        self.assertAlmostEqual(matcher.dustbin_score.item(), 0.75)
        self.assertEqual(report["openingProbabilityThreshold"], 0.375)
        self.assertEqual(
            resolve_opening_probability_threshold(None, None, report),
            {"value": 0.375, "source": "matcher_checkpoint"},
        )

    def test_rejects_changed_backbone_dependency(self):
        with TemporaryDirectory() as directory:
            checkpoint, config, backbone = self._write_checkpoint(Path(directory))
            config.write_bytes(b"MODEL: changed\n")
            with self.assertRaisesRegex(ValueError, "Bi-Layout config"):
                load_cross_scene_matcher_checkpoint(
                    str(checkpoint),
                    expected_feature_dim=8,
                    expected_token_count=16,
                    expected_branch_order="extended_first",
                    bi_layout_config_path=str(config),
                    bi_layout_checkpoint_path=str(backbone),
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()

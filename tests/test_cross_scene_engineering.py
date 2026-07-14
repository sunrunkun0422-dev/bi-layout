import json
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from dataset.panorama_pair_dataset import PanoramaPairDataset, build_pair_dataloader
from models.geometry_consistency_selector import (
    GEOMETRY_METRIC_NAMES,
    GeometryConsistencySelector,
    geometry_selector_loss,
)
from utils.cross_scene_estimator import (
    OpeningCandidate,
    estimate_wall_pair_candidates,
    polygon_overlap_ratio,
    polygon_validity,
)
from utils.cross_scene_logger import CrossSceneExperimentLogger
from utils.cross_scene_pipeline import (
    CrossScenePipeline,
    CrossScenePipelineConfig,
    atomic_write_json,
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
            layout["biLayoutOutputs"]["new_depth"] = np.asarray(
                new_depth, dtype=float
            ).tolist()
    return layout


class FixedSelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, metrics):
        count = metrics.shape[-2]
        logits = torch.arange(count, device=metrics.device, dtype=metrics.dtype)
        return {
            "selector_logits": logits,
            "selector_probability": torch.softmax(logits, dim=-1),
            "best_index": torch.tensor(count - 1, device=metrics.device),
        }


class CrossSceneEngineeringTest(unittest.TestCase):
    def test_polygon_validity_detects_self_intersection(self):
        valid = polygon_validity(np.asarray([[-2, -2], [2, -2], [2, 2], [-2, 2]]))
        invalid = polygon_validity(np.asarray([[0, 0], [2, 2], [0, 2], [2, 0]]))

        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])
        self.assertGreater(invalid["selfIntersectionCount"], 0)

    def test_polygon_overlap_has_dependency_free_fallback(self):
        first = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        second = np.asarray([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)

        overlap = polygon_overlap_ratio(first, second)

        self.assertAlmostEqual(overlap, 0.25, places=2)

    def test_cross_attention_evidence_changes_geometry_ranking(self):
        layout_a = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        layout_b = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        openings_a = [
            OpeningCandidate(0, 0.25, 0.75, 0.5),
            OpeningCandidate(1, 0.25, 0.75, 0.5),
        ]
        openings_b = [
            OpeningCandidate(0, 0.25, 0.75, 0.5),
            OpeningCandidate(3, 0.25, 0.75, 0.5),
        ]
        match_evidence = {
            "candidate_pair_score": np.asarray([[0.0, 0.0], [0.0, 1.0]])
        }

        candidates, _ = estimate_wall_pair_candidates(
            layout_a,
            layout_b,
            openings_a=openings_a,
            openings_b=openings_b,
            match_evidence=match_evidence,
            feature_weight=10.0,
            top_k=4,
        )

        self.assertEqual((candidates[0].wall_a, candidates[0].wall_b), (1, 3))
        self.assertEqual(candidates[0].metrics["featureScore"], 1.0)
        self.assertAlmostEqual(sum(candidate.confidence for candidate in candidates), 1.0)

    def test_pipeline_produces_validated_versioned_output(self):
        depth = np.ones(256, dtype=float)
        new_depth = depth.copy()
        new_depth[30:42] = 1.8
        layout = make_layout(
            [[-2, -2], [2, -2], [2, 2], [-2, 2]], depth, new_depth
        )
        pipeline = CrossScenePipeline(CrossScenePipelineConfig(top_k=4))

        result = pipeline.run(layout, layout)
        payload = result.candidates_json({"pairId": "demo"})

        self.assertEqual(payload["formatVersion"], 2)
        self.assertGreater(payload["candidateCount"], 0)
        self.assertTrue(payload["layoutPreparation"]["A"]["validity"]["valid"])
        self.assertIn("selection", result.best_joint_layout)

    def test_pipeline_can_apply_learned_selector(self):
        layout = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        pipeline = CrossScenePipeline(
            CrossScenePipelineConfig(top_k=4, use_passability=False),
            selector=FixedSelector(),
        )

        result = pipeline.run(layout, layout)

        self.assertIn("learned_selector", result.method)
        self.assertEqual(
            result.best_joint_layout["selection"]["method"],
            "learned_geometry_consistency_selector",
        )
        self.assertIn("selectorProbability", result.candidates[0].metrics)

    def test_pair_dataset_batches_manifest_records(self):
        with TemporaryDirectory() as directory:
            image_a = f"{directory}/a.png"
            image_b = f"{directory}/b.png"
            manifest = f"{directory}/pairs.json"
            Image.fromarray(np.full((4, 8, 3), 64, dtype=np.uint8)).save(image_a)
            Image.fromarray(np.full((4, 8, 3), 192, dtype=np.uint8)).save(image_b)
            with open(manifest, "w", encoding="utf-8") as file:
                json.dump({"pairs": [{"id": "a_b", "image_a": "a.png", "image_b": "b.png"}]}, file)

            dataset = PanoramaPairDataset(manifest, image_shape=(8, 16))
            batch = next(iter(build_pair_dataloader(
                manifest, batch_size=1, image_shape=(8, 16)
            )))

        self.assertEqual(len(dataset), 1)
        self.assertEqual(batch["image_A"].shape, (1, 3, 8, 16))
        self.assertEqual(batch["pair_id"][0], "a_b")

    def test_learned_selector_masks_candidates_and_backpropagates(self):
        selector = GeometryConsistencySelector(hidden_dim=16, dropout=0.0)
        metrics = torch.randn(2, 3, len(GEOMETRY_METRIC_NAMES), requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])

        output = selector(metrics, candidate_mask=mask)
        loss = geometry_selector_loss(
            output["selector_logits"], torch.tensor([1, 0]), candidate_mask=mask
        )
        loss.backward()

        self.assertEqual(output["selector_probability"][:, 2].abs().sum().item(), 0.0)
        self.assertIsNotNone(metrics.grad)

    def test_atomic_output_and_experiment_log_summary(self):
        layout = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        candidates, best = estimate_wall_pair_candidates(layout, layout, top_k=2)
        with TemporaryDirectory() as directory:
            output_path = f"{directory}/result.json"
            log_path = f"{directory}/runs.jsonl"
            atomic_write_json(output_path, {"ok": True})
            logger = CrossSceneExperimentLogger(log_path)
            logger.log_result(
                "pair_1",
                candidates,
                best,
                ground_truth={"wallA": candidates[0].wall_a, "wallB": candidates[0].wall_b},
            )
            summary = logger.summarize()
            with open(output_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

        self.assertTrue(payload["ok"])
        self.assertEqual(summary["recordCount"], 1)
        self.assertEqual(summary["meanMetrics"]["openingTop1Correct"], 1.0)


if __name__ == "__main__":
    unittest.main()

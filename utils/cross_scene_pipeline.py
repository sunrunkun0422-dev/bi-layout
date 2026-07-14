"""Reusable engineering pipeline for cross-scene opening fusion."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from utils.cross_scene_estimator import (
    WallPairCandidate,
    estimate_wall_pair_candidates,
    extract_opening_candidates,
    polygon_validity,
    simplify_layout_for_estimation,
)
from utils.joint_layout import build_joint_layout


@dataclass(frozen=True)
class CrossScenePipelineConfig:
    anchor_ratio: float = 0.3
    top_k: int = 8
    calibrate_scale: bool = True
    simplify_tolerance: float = 0.05
    max_walls: int = 64
    use_passability: bool = True
    opening_threshold: float = 0.25
    min_opening_width_tokens: int = 3
    max_openings_per_layout: int = 12
    passability_weight: float = 1.0
    feature_weight: float = 1.0
    nms_overlap_threshold: float = 0.8
    confidence_temperature: float = 1.0
    strict_polygon_validation: bool = True

    def __post_init__(self):
        if not 0.0 < self.anchor_ratio <= 1.0:
            raise ValueError("anchor_ratio must be in (0, 1]")
        if self.top_k <= 0 or self.max_walls < 3:
            raise ValueError("top_k must be positive and max_walls must be at least 3")
        if not 0.0 <= self.opening_threshold <= 1.0:
            raise ValueError("opening_threshold must be in [0, 1]")
        if self.min_opening_width_tokens <= 0 or self.max_openings_per_layout <= 0:
            raise ValueError("opening candidate counts must be positive")
        if not 0.0 <= self.nms_overlap_threshold <= 1.0:
            raise ValueError("nms_overlap_threshold must be in [0, 1]")
        if self.confidence_temperature <= 0:
            raise ValueError("confidence_temperature must be positive")


@dataclass
class CrossScenePipelineResult:
    candidates: List[WallPairCandidate]
    best_joint_layout: Dict[str, Any]
    opening_summary: Dict[str, Any]
    layout_preparation: Dict[str, Any]
    config: CrossScenePipelineConfig
    method: str

    def candidates_json(self, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        output = {
            "formatVersion": 2,
            "method": self.method,
            "config": asdict(self.config),
            "layoutPreparation": self.layout_preparation,
            "passability": self.opening_summary,
            "candidateCount": len(self.candidates),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }
        if metadata:
            output["metadata"] = dict(metadata)
        return output


class CrossScenePipeline:
    def __init__(self, config: Optional[CrossScenePipelineConfig] = None, selector=None):
        self.config = config or CrossScenePipelineConfig()
        self.selector = selector

    def _apply_learned_selector(self, candidates, layout_a, layout_b, best_joint_layout):
        if self.selector is None or not candidates:
            return candidates, best_joint_layout

        import torch
        from models.geometry_consistency_selector import candidate_metrics_to_tensor

        try:
            device = next(self.selector.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        metrics = candidate_metrics_to_tensor(
            [candidate.metrics for candidate in candidates], device=device
        )
        self.selector.eval()
        with torch.no_grad():
            selector_output = self.selector(metrics)
        probabilities = selector_output["selector_probability"].detach().cpu().numpy()
        order = np.argsort(-probabilities)

        reranked = []
        for rank, index in enumerate(order.tolist(), start=1):
            candidate = candidates[index]
            updated_metrics = dict(candidate.metrics)
            updated_metrics["selectorProbability"] = float(probabilities[index])
            reranked.append(replace(
                candidate,
                rank=rank,
                confidence=float(probabilities[index]),
                metrics=updated_metrics,
            ))

        selected = reranked[0]
        selected_joint = build_joint_layout(
            layout_a,
            layout_b,
            selected.door_a,
            selected.door_b,
            calibrate_scale=self.config.calibrate_scale,
        )
        selected_joint["confidence"] = selected.confidence
        selected_joint["selection"] = {
            "rank": 1,
            "score": selected.score,
            "confidence": selected.confidence,
            "wallA": selected.wall_a,
            "wallB": selected.wall_b,
            "metrics": selected.metrics,
            "method": "learned_geometry_consistency_selector",
        }
        if best_joint_layout.get("diagnostics"):
            selected_joint["diagnostics"] = deepcopy(best_joint_layout["diagnostics"])
        return reranked, selected_joint

    def _prepare_layout(self, layout: Dict, name: str):
        prepared, simplification = simplify_layout_for_estimation(
            layout,
            tolerance=self.config.simplify_tolerance,
            max_walls=self.config.max_walls,
        )
        validity = polygon_validity(prepared)
        if self.config.strict_polygon_validation and not validity["valid"]:
            raise ValueError(f"{name} polygon failed validation: {validity}")
        return prepared, {"simplification": simplification, "validity": validity}

    def run(
        self,
        layout_a: Dict,
        layout_b: Dict,
        extended_layout_a: Optional[Dict] = None,
        extended_layout_b: Optional[Dict] = None,
        match_evidence: Optional[Mapping[str, Any]] = None,
    ) -> CrossScenePipelineResult:
        prepared_a, preparation_a = self._prepare_layout(layout_a, "layout A")
        prepared_b, preparation_b = self._prepare_layout(layout_b, "layout B")

        prepared_ext_a = None
        prepared_ext_b = None
        preparation_ext_a = None
        preparation_ext_b = None
        if extended_layout_a is not None:
            prepared_ext_a, preparation_ext_a = self._prepare_layout(
                extended_layout_a, "extended layout A"
            )
        if extended_layout_b is not None:
            prepared_ext_b, preparation_ext_b = self._prepare_layout(
                extended_layout_b, "extended layout B"
            )

        openings_a = None
        openings_b = None
        if self.config.use_passability:
            openings_a, summary_a = extract_opening_candidates(
                prepared_a,
                extended_layout=prepared_ext_a,
                threshold=self.config.opening_threshold,
                min_width_tokens=self.config.min_opening_width_tokens,
                max_candidates=self.config.max_openings_per_layout,
                fallback_anchor_ratio=self.config.anchor_ratio,
            )
            openings_b, summary_b = extract_opening_candidates(
                prepared_b,
                extended_layout=prepared_ext_b,
                threshold=self.config.opening_threshold,
                min_width_tokens=self.config.min_opening_width_tokens,
                max_candidates=self.config.max_openings_per_layout,
                fallback_anchor_ratio=self.config.anchor_ratio,
            )
            opening_summary = {
                "enabled": True,
                "A": summary_a,
                "B": summary_b,
                "candidatesA": [candidate.to_json() for candidate in openings_a],
                "candidatesB": [candidate.to_json() for candidate in openings_b],
            }
        else:
            opening_summary = {
                "enabled": False,
                "A": {"source": "disabled_wall_center_fallback"},
                "B": {"source": "disabled_wall_center_fallback"},
                "candidatesA": [],
                "candidatesB": [],
            }

        candidates, best_joint_layout = estimate_wall_pair_candidates(
            prepared_a,
            prepared_b,
            anchor_ratio=self.config.anchor_ratio,
            top_k=self.config.top_k,
            calibrate_scale=self.config.calibrate_scale,
            openings_a=openings_a,
            openings_b=openings_b,
            passability_weight=(
                self.config.passability_weight if self.config.use_passability else 0.0
            ),
            match_evidence=match_evidence,
            feature_weight=self.config.feature_weight,
            nms_overlap_threshold=self.config.nms_overlap_threshold,
            confidence_temperature=self.config.confidence_temperature,
            validate_polygons=self.config.strict_polygon_validation,
        )
        if not candidates:
            raise RuntimeError("no valid cross-scene opening alignment candidates were found")
        candidates, best_joint_layout = self._apply_learned_selector(
            candidates, prepared_a, prepared_b, best_joint_layout
        )

        method_parts = ["validated_geometry", "opening_nms", "calibrated_selection"]
        if self.config.use_passability:
            method_parts.insert(0, "passability")
        if match_evidence:
            method_parts.insert(1, "cross_attention")
        if self.selector is not None:
            method_parts.append("learned_selector")
        return CrossScenePipelineResult(
            candidates=candidates,
            best_joint_layout=best_joint_layout,
            opening_summary=opening_summary,
            layout_preparation={
                "A": preparation_a,
                "B": preparation_b,
                "AExtended": preparation_ext_a,
                "BExtended": preparation_ext_b,
            },
            config=self.config,
            method="_".join(method_parts),
        )


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

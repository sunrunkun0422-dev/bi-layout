"""Structured JSONL experiment logging for cross-scene layout estimation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


def _wrapped_angle_error(predicted: float, target: float) -> float:
    return float(abs((predicted - target + math.pi) % (2.0 * math.pi) - math.pi))


def _alignment_pose(best_joint_layout: Mapping[str, Any]):
    alignment = best_joint_layout.get("alignment", {})
    matrix = np.asarray(alignment.get("roomBToWorld", []), dtype=np.float64)
    if matrix.shape != (3, 3):
        return None
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    yaw = float(math.atan2(matrix[1, 0], matrix[0, 0]))
    translation = [float(matrix[0, 2]), float(matrix[1, 2])]
    return {"yawRadians": yaw, "translation": translation, "scale": scale}


class CrossSceneExperimentLogger:
    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload.setdefault("schemaVersion", 1)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as file:
            if fcntl is not None:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            file.write(line + "\n")
            file.flush()
            if fcntl is not None:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        return payload

    def log_result(
        self,
        pair_id: str,
        candidates: Iterable,
        best_joint_layout: Mapping[str, Any],
        ground_truth: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        candidate_list = [
            candidate.to_json() if hasattr(candidate, "to_json") else dict(candidate)
            for candidate in candidates
        ]
        pose = _alignment_pose(best_joint_layout)
        metrics: Dict[str, Any] = {
            "candidateCount": len(candidate_list),
            "bestConfidence": (
                float(candidate_list[0].get("confidence", 0.0)) if candidate_list else 0.0
            ),
            "invalidFusion": not bool(best_joint_layout),
        }
        if ground_truth and candidate_list:
            wall_a = ground_truth.get("wallA")
            wall_b = ground_truth.get("wallB")
            if wall_a is not None and wall_b is not None:
                matches = [
                    item.get("wallA") == wall_a and item.get("wallB") == wall_b
                    for item in candidate_list
                ]
                metrics["openingTop1Correct"] = bool(matches[0])
                metrics["openingTopKRecall"] = bool(any(matches))
            if pose and ground_truth.get("yawRadians") is not None:
                metrics["yawErrorRadians"] = _wrapped_angle_error(
                    pose["yawRadians"], float(ground_truth["yawRadians"])
                )
            if pose and ground_truth.get("translation") is not None:
                target = np.asarray(ground_truth["translation"], dtype=np.float64)
                metrics["translationError"] = float(np.linalg.norm(
                    np.asarray(pose["translation"]) - target
                ))
            if pose and ground_truth.get("scale") is not None:
                metrics["scaleError"] = abs(pose["scale"] - float(ground_truth["scale"]))

        record = {
            "pairId": str(pair_id),
            "status": "success" if best_joint_layout else "failed",
            "metrics": metrics,
            "prediction": pose,
        }
        if extra:
            record["extra"] = dict(extra)
        return self.append(record)

    def records(self):
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid experiment log line {line_number}") from exc
        return records

    def summarize(self) -> Dict[str, Any]:
        records = self.records()
        metric_values: Dict[str, list] = {}
        for record in records:
            for name, value in record.get("metrics", {}).items():
                if isinstance(value, (bool, int, float)):
                    metric_values.setdefault(name, []).append(float(value))
        return {
            "recordCount": len(records),
            "successCount": sum(record.get("status") == "success" for record in records),
            "meanMetrics": {
                name: float(np.mean(values)) for name, values in metric_values.items() if values
            },
        }

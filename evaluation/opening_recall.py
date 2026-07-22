"""Reusable metrics for one-dimensional circular opening detection.

The ZInD-BiPair labels and Bi-Layout predictions are horizontal panorama
sequences.  Token zero and the last token are neighbours, so ordinary linear
connected-component code gives incorrect results for openings that cross the
panorama seam.  This module keeps that circular contract explicit and has no
PyTorch dependency, which makes it usable from evaluation scripts and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class UniqueViewReference:
    """A stable pointer from one unique panorama to its side-specific cache."""

    image_path: str
    label_cache: str
    side: str
    pair_id: str
    house_id: str
    floor_id: str
    complete_room_id: str
    partial_room_id: str
    pano_id: str


def deduplicate_manifest_views(
    records: Sequence[Mapping[str, Any]],
) -> List[UniqueViewReference]:
    """Return the first occurrence of every ``view.image_path`` in a manifest.

    A panorama can occur in several pairs.  Counting every occurrence would
    overweight highly connected partial rooms, so opening detection is
    evaluated once per image path.  Manifest order is preserved.
    """

    unique: List[UniqueViewReference] = []
    seen = set()
    for record in records:
        label_cache = str(record["label_cache"])
        for side in ("A", "B"):
            view = record[f"view_{side}"]
            image_path = str(view["image_path"])
            if image_path in seen:
                continue
            seen.add(image_path)
            unique.append(
                UniqueViewReference(
                    image_path=image_path,
                    label_cache=label_cache,
                    side=side,
                    pair_id=str(record.get("pair_id", "")),
                    house_id=str(record.get("house_id", "")),
                    floor_id=str(record.get("floor_id", "")),
                    complete_room_id=str(record.get("complete_room_id", "")),
                    partial_room_id=str(view.get("partial_room_id", "")),
                    pano_id=str(view.get("pano_id", "")),
                )
            )
    return unique


def opening_geometry_probability(
    enclosed_depth: np.ndarray,
    extended_depth: np.ndarray,
    *,
    prior_strength: float = 4.0,
    prior_relative_scale: float = 0.1,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply the zero-weight ``OpeningSignalHead`` geometry prior in NumPy.

    This intentionally mirrors :class:`models.cross_scene_matcher.OpeningSignalHead`.
    The order is significant: ``extended_depth - enclosed_depth`` is the
    passability signal.  For the existing ``zind_all`` checkpoint the adapter
    in the CLI maps ``new_depth`` to enclosed and ``depth`` to extended.
    """

    if prior_relative_scale <= 0 or eps <= 0:
        raise ValueError("prior_relative_scale and eps must be positive")
    enclosed = np.asarray(enclosed_depth, dtype=np.float64)
    extended = np.asarray(extended_depth, dtype=np.float64)
    if enclosed.shape != extended.shape:
        raise ValueError("enclosed_depth and extended_depth must have the same shape")
    if enclosed.ndim == 0 or enclosed.shape[-1] <= 0:
        raise ValueError("depth arrays must have a non-empty token dimension")
    if not np.isfinite(enclosed).all() or not np.isfinite(extended).all():
        raise ValueError("depth arrays must contain only finite values")

    enclosed = np.maximum(np.abs(enclosed), eps)
    extended = np.maximum(np.abs(extended), eps)
    relative_delta = np.maximum(extended - enclosed, 0.0) / enclosed
    relative_delta = np.minimum(relative_delta, 10.0)
    relative_peak = relative_delta / np.maximum(
        np.max(relative_delta, axis=-1, keepdims=True), eps
    )
    geometry_prior = relative_peak * (
        relative_delta / (relative_delta + float(prior_relative_scale))
    )
    logits = float(prior_strength) * (geometry_prior - 0.5)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def resolve_depth_branches(
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    branch_order: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(enclosed, extended)`` from two source-ordered depth streams.

    ``extended_first`` is the native order of the existing ZInD checkpoint:
    ``depth`` is visible/extended and ``new_depth`` is raw/enclosed.
    ``enclosed_first`` is retained as an explicit diagnostic mode so an
    evaluator can reproduce the previously reversed interpretation.
    """

    first = np.asarray(first_depth)
    second = np.asarray(second_depth)
    if first.shape != second.shape:
        raise ValueError("first_depth and second_depth must have the same shape")
    if branch_order == "extended_first":
        return second, first
    if branch_order == "enclosed_first":
        return first, second
    raise ValueError("branch_order must be extended_first or enclosed_first")


def _as_bool_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return value.astype(bool, copy=False)


def circular_components(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive ``(start, end)`` components on a circular sequence.

    A component crossing the panorama seam is represented with ``start > end``.
    For example, ``[1, 1, 0, 0, 1]`` becomes ``[(4, 1)]``.
    """

    value = _as_bool_mask(mask)
    length = int(value.size)
    if not value.any():
        return []
    if value.all():
        return [(0, length - 1)]

    starts = np.flatnonzero(value & ~np.roll(value, 1))
    components: List[Tuple[int, int]] = []
    for raw_start in starts:
        start = int(raw_start)
        end = start
        while value[(end + 1) % length]:
            end = (end + 1) % length
        components.append((start, end))
    return components


def circular_interval_mask(
    interval: Tuple[int, int], token_count: int
) -> np.ndarray:
    """Convert an inclusive, possibly wrapping interval to a boolean mask."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    start, end = (int(value) % token_count for value in interval)
    mask = np.zeros(token_count, dtype=bool)
    if start <= end:
        mask[start : end + 1] = True
    else:
        mask[start:] = True
        mask[: end + 1] = True
    return mask


def circular_interval_iou(
    first: Tuple[int, int], second: Tuple[int, int], token_count: int
) -> float:
    """Compute IoU between two inclusive circular token intervals."""

    mask_first = circular_interval_mask(first, token_count)
    mask_second = circular_interval_mask(second, token_count)
    union = np.logical_or(mask_first, mask_second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(mask_first, mask_second).sum() / union)


def _validate_targets_scores(
    targets: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    target = np.asarray(targets).astype(bool, copy=False).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != score.shape:
        raise ValueError("targets and scores must have the same non-empty shape")
    if not np.isfinite(score).all():
        raise ValueError("scores must contain only finite values")
    return target, score


def binary_average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware non-interpolated average precision for binary token scores."""

    target, score = _validate_targets_scores(targets, scores)
    positive_count = int(target.sum())
    if positive_count == 0:
        return 0.0

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order].astype(np.int64)
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(score) - 1]
    true_positive = np.cumsum(sorted_target)[distinct_ends]
    predicted_positive = distinct_ends + 1
    precision = true_positive / predicted_positive
    recall = true_positive / positive_count
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def binary_token_metrics(
    targets: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Compute micro token metrics at one probability threshold."""

    target, score = _validate_targets_scores(targets, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = score >= float(threshold)
    true_positive = int(np.logical_and(predicted, target).sum())
    false_positive = int(np.logical_and(predicted, ~target).sum())
    false_negative = int(np.logical_and(~predicted, target).sum())
    true_negative = int(np.logical_and(~predicted, ~target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "threshold": float(threshold),
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "predictedPositive": int(predicted.sum()),
        "actualPositive": int(target.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def scan_opening_thresholds(
    targets: np.ndarray,
    scores: np.ndarray,
    thresholds: Iterable[float],
    *,
    precision_target: float = 0.85,
) -> Dict[str, Any]:
    """Select best-F1 and maximum-recall operating points from a val scan."""

    if not 0.0 <= precision_target <= 1.0:
        raise ValueError("precision_target must be in [0, 1]")
    threshold_values = sorted({float(value) for value in thresholds})
    if not threshold_values or not np.isfinite(threshold_values).all():
        raise ValueError("thresholds must contain at least one finite value")
    curve = [binary_token_metrics(targets, scores, value) for value in threshold_values]
    best_f1 = max(
        curve,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["precision"],
            -item["threshold"],
        ),
    )
    qualifying = [
        item
        for item in curve
        if item["precision"] >= precision_target and item["predictedPositive"] > 0
    ]
    precision_constrained = (
        max(
            qualifying,
            key=lambda item: (
                item["recall"],
                item["f1"],
                item["precision"],
                -item["threshold"],
            ),
        )
        if qualifying
        else None
    )
    return {
        "precisionTarget": float(precision_target),
        "bestF1": dict(best_f1),
        "precisionTargetMaxRecall": (
            None if precision_constrained is None else dict(precision_constrained)
        ),
        "curve": curve,
    }


def component_detection_metrics(
    targets_by_view: Sequence[np.ndarray],
    scores_by_view: Sequence[np.ndarray],
    threshold: float,
    *,
    iou_thresholds: Sequence[float] = (0.3, 0.5),
) -> Dict[str, Any]:
    """Measure circular connected-component recall and precision across views."""

    if len(targets_by_view) != len(scores_by_view) or not targets_by_view:
        raise ValueError("targets_by_view and scores_by_view must have equal non-zero length")
    overlap_thresholds = [float(value) for value in iou_thresholds]
    if any(value < 0 or value > 1 for value in overlap_thresholds):
        raise ValueError("component IoU thresholds must be in [0, 1]")

    gt_components: List[List[Tuple[int, int]]] = []
    predicted_components: List[List[Tuple[int, int]]] = []
    token_counts: List[int] = []
    for target_raw, score_raw in zip(targets_by_view, scores_by_view):
        target = _as_bool_mask(target_raw, "target")
        score = np.asarray(score_raw, dtype=np.float64)
        if score.ndim != 1 or score.shape != target.shape or not np.isfinite(score).all():
            raise ValueError("each score must be finite and match its one-dimensional target")
        gt_components.append(circular_components(target))
        predicted_components.append(circular_components(score >= float(threshold)))
        token_counts.append(int(target.size))

    gt_count = sum(len(items) for items in gt_components)
    predicted_count = sum(len(items) for items in predicted_components)
    overlap_metrics: Dict[str, Any] = {}
    for overlap_threshold in overlap_thresholds:
        matched_gt = 0
        matched_predicted = 0
        for gt_items, predicted_items, token_count in zip(
            gt_components, predicted_components, token_counts
        ):
            matrix = np.asarray(
                [
                    [circular_interval_iou(gt, pred, token_count) for pred in predicted_items]
                    for gt in gt_items
                ],
                dtype=np.float64,
            )
            if gt_items:
                if predicted_items:
                    matched_gt += int((matrix.max(axis=1) >= overlap_threshold).sum())
            if predicted_items:
                if gt_items:
                    matched_predicted += int(
                        (matrix.max(axis=0) >= overlap_threshold).sum()
                    )
        key = f"iouAtLeast{overlap_threshold:.2f}"
        overlap_metrics[key] = {
            "matchedGroundTruth": matched_gt,
            "matchedPredicted": matched_predicted,
            "recall": float(matched_gt / max(gt_count, 1)),
            "precision": float(matched_predicted / max(predicted_count, 1)),
        }

    return {
        "threshold": float(threshold),
        "viewCount": len(targets_by_view),
        "groundTruthComponentCount": int(gt_count),
        "predictedComponentCount": int(predicted_count),
        "meanPredictedComponentsPerView": float(
            predicted_count / len(targets_by_view)
        ),
        "overlap": overlap_metrics,
    }


def evaluate_opening_scores(
    targets_by_view: Sequence[np.ndarray],
    scores_by_view: Sequence[np.ndarray],
    *,
    threshold: Optional[float] = None,
    thresholds: Optional[Iterable[float]] = None,
    precision_target: float = 0.85,
    component_iou_thresholds: Sequence[float] = (0.3, 0.5),
) -> Dict[str, Any]:
    """Evaluate per-view opening scores and select an operating threshold."""

    if len(targets_by_view) != len(scores_by_view) or not targets_by_view:
        raise ValueError("targets_by_view and scores_by_view must have equal non-zero length")
    normalized_targets = []
    normalized_scores = []
    token_count = None
    for target_raw, score_raw in zip(targets_by_view, scores_by_view):
        target = _as_bool_mask(target_raw, "target")
        score = np.asarray(score_raw, dtype=np.float64)
        if score.ndim != 1 or score.shape != target.shape or not np.isfinite(score).all():
            raise ValueError("each score must be finite and match its one-dimensional target")
        if token_count is None:
            token_count = int(target.size)
        elif token_count != int(target.size):
            raise ValueError("all views must use the same token count")
        normalized_targets.append(target)
        normalized_scores.append(score)

    targets_flat = np.concatenate(normalized_targets)
    scores_flat = np.concatenate(normalized_scores)
    average_precision = binary_average_precision(targets_flat, scores_flat)
    scan = None
    if threshold is None:
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 201)
        scan = scan_opening_thresholds(
            targets_flat,
            scores_flat,
            thresholds,
            precision_target=precision_target,
        )
        selected_threshold = float(scan["bestF1"]["threshold"])
        selected_reason = "validation_best_f1"
    else:
        selected_threshold = float(threshold)
        selected_reason = "fixed_threshold"

    selected_metrics = binary_token_metrics(
        targets_flat, scores_flat, selected_threshold
    )
    selected_metrics["averagePrecision"] = average_precision
    return {
        "viewCount": len(normalized_targets),
        "tokenCountPerView": token_count,
        "totalTokenCount": int(targets_flat.size),
        "positiveTokenRate": float(targets_flat.mean()),
        "averagePrecision": average_precision,
        "selectedThreshold": selected_threshold,
        "selectedThresholdReason": selected_reason,
        "metricsAtSelectedThreshold": selected_metrics,
        "thresholdSelection": scan,
        "componentMetricsAtSelectedThreshold": component_detection_metrics(
            normalized_targets,
            normalized_scores,
            selected_threshold,
            iou_thresholds=component_iou_thresholds,
        ),
    }


__all__ = [
    "UniqueViewReference",
    "binary_average_precision",
    "binary_token_metrics",
    "circular_components",
    "circular_interval_iou",
    "circular_interval_mask",
    "component_detection_metrics",
    "deduplicate_manifest_views",
    "evaluate_opening_scores",
    "opening_geometry_probability",
    "resolve_depth_branches",
    "scan_opening_thresholds",
]

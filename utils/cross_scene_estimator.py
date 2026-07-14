"""Geometry-first cross-scene layout estimation utilities.

This module estimates candidate relative poses for two independently predicted
room layouts. It does not require door detections; instead it treats the center
part of each wall as a possible shared opening/interface and ranks all wall-pair
alignments.
"""

from __future__ import annotations

import math
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from utils.joint_layout import DoorSpec, build_joint_layout, layout_xz

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover - optional runtime dependency
    Polygon = None


@dataclass(frozen=True)
class OpeningCandidate:
    wall_index: int
    start_ratio: float
    end_ratio: float
    confidence: float
    token_start: int = -1
    token_end: int = -1
    source: str = "wall_center_fallback"
    metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.wall_index < 0:
            raise ValueError("wall_index must be non-negative")
        if not 0 <= self.start_ratio < self.end_ratio <= 1:
            raise ValueError("opening ratios must satisfy 0 <= start < end <= 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("opening confidence must be in [0, 1]")

    @property
    def spec(self) -> DoorSpec:
        return DoorSpec(self.wall_index, self.start_ratio, self.end_ratio)

    def to_json(self) -> Dict:
        return {
            "wallIndex": self.wall_index,
            "startRatio": self.start_ratio,
            "endRatio": self.end_ratio,
            "confidence": self.confidence,
            "tokenStart": self.token_start,
            "tokenEnd": self.token_end,
            "source": self.source,
            "metrics": self.metrics or {},
        }


@dataclass(frozen=True)
class WallPairCandidate:
    rank: int
    score: float
    confidence: float
    wall_a: int
    wall_b: int
    door_a: DoorSpec
    door_b: DoorSpec
    opening_a: OpeningCandidate
    opening_b: OpeningCandidate
    metrics: Dict[str, float]
    alignment: Dict

    def to_json(self) -> Dict:
        opening_a = asdict(self.door_a)
        opening_b = asdict(self.door_b)
        return {
            "rank": self.rank,
            "score": self.score,
            "confidence": self.confidence,
            "wallA": self.wall_a,
            "wallB": self.wall_b,
            "openingA": opening_a,
            "openingB": opening_b,
            "doorA": opening_a,
            "doorB": opening_b,
            "openingEvidenceA": self.opening_a.to_json(),
            "openingEvidenceB": self.opening_b.to_json(),
            "metrics": self.metrics,
            "alignment": self.alignment,
        }


def wall_count(layout: Dict) -> int:
    walls = layout.get("layoutWalls", {}).get("walls", [])
    if walls:
        return len(walls)
    return len(layout_xz(layout))


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _lon_samples(sample_count: int) -> np.ndarray:
    u = (np.arange(sample_count, dtype=np.float64) + 0.5) / float(sample_count)
    return (u - 0.5) * 2.0 * math.pi


def _depth_to_xz(depth: np.ndarray) -> np.ndarray:
    lon = _lon_samples(len(depth))
    return np.stack([depth * np.sin(lon), depth * np.cos(lon)], axis=-1)


def _as_float_array(value, expected_length: Optional[int] = None) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if expected_length is not None and len(array) != expected_length:
        return None
    if len(array) == 0 or not np.isfinite(array).all():
        return None
    return np.abs(array)


def _find_first_array_by_keys(obj, keys: Iterable[str], expected_length: Optional[int] = None) -> Optional[np.ndarray]:
    keys = set(keys)
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                array = _as_float_array(obj[key], expected_length=expected_length)
                if array is not None:
                    return array
        for value in obj.values():
            array = _find_first_array_by_keys(value, keys, expected_length=expected_length)
            if array is not None:
                return array
    elif isinstance(obj, list):
        for value in obj:
            array = _find_first_array_by_keys(value, keys, expected_length=expected_length)
            if array is not None:
                return array
    return None


def polygon_depth_profile(layout: Dict, sample_count: int = 256) -> np.ndarray:
    points = layout_xz(layout)
    if len(points) == sample_count:
        return np.linalg.norm(points, axis=-1)

    depths = np.full(sample_count, np.nan, dtype=np.float64)
    lon = _lon_samples(sample_count)
    directions = np.stack([np.sin(lon), np.cos(lon)], axis=-1)

    for token, direction in enumerate(directions):
        hits = []
        for wall_index in range(len(points)):
            start = points[wall_index]
            end = points[(wall_index + 1) % len(points)]
            edge = end - start
            denominator = _cross_2d(direction, edge)
            if abs(denominator) < 1e-10:
                continue
            distance = _cross_2d(start, edge) / denominator
            ratio = _cross_2d(start, direction) / denominator
            if distance > 1e-8 and -1e-8 <= ratio <= 1.0 + 1e-8:
                hits.append(distance)
        if hits:
            depths[token] = min(hits)

    if np.isnan(depths).any():
        finite = depths[np.isfinite(depths)]
        if len(finite) == 0:
            raise ValueError("could not sample a depth profile from the layout polygon")
        depths[~np.isfinite(depths)] = float(np.median(finite))
    return depths


def layout_depth_profile(layout: Dict, sample_count: int = 256,
                         keys: Optional[Iterable[str]] = None) -> np.ndarray:
    if keys is None:
        keys = (
            "depth",
            "layoutDepth",
            "layout_depth",
            "enclosedDepth",
            "enclosed_depth",
            "originDepth",
            "origin_depth",
        )
    array = _find_first_array_by_keys(layout, keys, expected_length=sample_count)
    if array is not None:
        return array
    return polygon_depth_profile(layout, sample_count=sample_count)


def extended_depth_profile(layout: Dict, extended_layout: Optional[Dict] = None,
                           sample_count: int = 256) -> Optional[np.ndarray]:
    if extended_layout is not None:
        return layout_depth_profile(
            extended_layout,
            sample_count=sample_count,
            keys=(
                "new_depth",
                "newDepth",
                "extendedDepth",
                "extended_depth",
                "depth",
                "layoutDepth",
                "layout_depth",
            ),
        )
    return _find_first_array_by_keys(
        layout,
        (
            "new_depth",
            "newDepth",
            "extendedDepth",
            "extended_depth",
            "newLayoutDepth",
            "new_layout_depth",
        ),
        expected_length=sample_count,
    )


def _smooth_signal(signal: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return signal
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / float(2 * radius + 1)
    padded = np.concatenate([signal[-radius:], signal, signal[:radius]])
    return np.convolve(padded, kernel, mode="valid")


def _segments_from_mask(mask: np.ndarray, min_width: int) -> List[Tuple[int, int]]:
    if len(mask) == 0 or not mask.any():
        return []
    runs = []
    start = None
    for index, enabled in enumerate(mask.tolist()):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            if index - start >= min_width:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= min_width:
        runs.append((start, len(mask) - 1))

    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == len(mask) - 1:
        first = runs.pop(0)
        last = runs.pop(-1)
        merged = (last[0], first[1])
        runs.insert(0, merged)
    return runs


def _tokens_in_segment(start: int, end: int, length: int) -> np.ndarray:
    if start <= end:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.concatenate([
        np.arange(start, length, dtype=np.int64),
        np.arange(0, end + 1, dtype=np.int64),
    ])


def wall_token_assignment(layout: Dict, sample_count: int = 256,
                          depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if depth is None:
        depth = layout_depth_profile(layout, sample_count=sample_count)
    token_points = _depth_to_xz(depth)
    wall_ids = np.zeros(sample_count, dtype=np.int64)
    ratios = np.zeros(sample_count, dtype=np.float64)

    for token, point in enumerate(token_points):
        best_wall = 0
        best_ratio = 0.5
        best_distance = float("inf")
        for wall_index in range(wall_count(layout)):
            start, end = wall_endpoints(layout, wall_index)
            vector = end - start
            denom = float(np.dot(vector, vector))
            if denom <= 1e-12:
                continue
            ratio = float(np.dot(point - start, vector) / denom)
            ratio_clamped = min(max(ratio, 0.0), 1.0)
            projection = start + ratio_clamped * vector
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_wall = wall_index
                best_ratio = ratio_clamped
        wall_ids[token] = best_wall
        ratios[token] = best_ratio
    return wall_ids, ratios


def centered_opening_candidate(wall_index: int, ratio: float, confidence: float = 0.0,
                               source: str = "wall_center_fallback") -> OpeningCandidate:
    spec = centered_wall_spec(wall_index, ratio)
    return OpeningCandidate(
        wall_index=spec.wall_index,
        start_ratio=spec.start_ratio,
        end_ratio=spec.end_ratio,
        confidence=confidence,
        source=source,
        metrics={"anchorRatio": float(ratio)},
    )


def fallback_opening_candidates(layout: Dict, anchor_ratio: float = 0.3) -> List[OpeningCandidate]:
    return [
        centered_opening_candidate(wall_index, anchor_ratio)
        for wall_index in range(wall_count(layout))
    ]


def simplify_layout_for_estimation(layout: Dict, tolerance: float = 0.05, max_walls: int = 64) -> Tuple[Dict, Dict]:
    count = wall_count(layout)
    if count <= max_walls or tolerance <= 0:
        return layout, {
            "simplified": False,
            "originalWallCount": count,
            "wallCount": count,
            "tolerance": tolerance,
        }

    points = layout_xz(layout)
    simplified = None
    if Polygon is not None:
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            simple_poly = poly.simplify(tolerance, preserve_topology=True)
            coords = np.asarray(simple_poly.exterior.coords[:-1], dtype=np.float64)
            if len(coords) >= 3:
                simplified = coords
        except Exception:
            simplified = None

    if simplified is None or len(simplified) >= count:
        return layout, {
            "simplified": False,
            "originalWallCount": count,
            "wallCount": count,
            "tolerance": tolerance,
        }

    y = float(layout.get("cameraHeight", 1.6))
    new_layout = deepcopy(layout)
    new_layout["layoutPoints"] = {
        "num": int(len(simplified)),
        "points": [
            {
                "xyz": [float(point[0]), y, float(point[1])],
                "id": index,
            }
            for index, point in enumerate(simplified)
        ],
    }
    new_layout["layoutWalls"] = {
        "num": 0,
        "walls": [],
    }
    return new_layout, {
        "simplified": True,
        "originalWallCount": count,
        "wallCount": int(len(simplified)),
        "tolerance": tolerance,
    }


def wall_endpoints(layout: Dict, wall_index: int) -> Tuple[np.ndarray, np.ndarray]:
    points = layout_xz(layout)
    walls = layout.get("layoutWalls", {}).get("walls", [])
    if walls:
        if wall_index >= len(walls):
            raise ValueError(f"wall_index {wall_index} is outside the {len(walls)} layout walls")
        start_index, end_index = walls[wall_index]["pointsIdx"]
    else:
        if wall_index >= len(points):
            raise ValueError(f"wall_index {wall_index} is outside the {len(points)} layout walls")
        start_index, end_index = wall_index, (wall_index + 1) % len(points)
    return points[start_index], points[end_index]


def wall_length(layout: Dict, wall_index: int) -> float:
    start, end = wall_endpoints(layout, wall_index)
    return float(np.linalg.norm(end - start))


def centered_wall_spec(wall_index: int, ratio: float) -> DoorSpec:
    ratio = min(max(float(ratio), 1e-3), 1.0)
    margin = (1.0 - ratio) / 2.0
    return DoorSpec(wall_index, margin, 1.0 - margin)


def extract_opening_candidates(
    layout: Dict,
    extended_layout: Optional[Dict] = None,
    sample_count: int = 256,
    threshold: float = 0.25,
    min_width_tokens: int = 3,
    max_candidates: int = 12,
    fallback_anchor_ratio: float = 0.3,
) -> Tuple[List[OpeningCandidate], Dict]:
    enclosed_depth = layout_depth_profile(layout, sample_count=sample_count)
    extended_depth = extended_depth_profile(layout, extended_layout, sample_count=sample_count)
    if extended_depth is None:
        candidates = fallback_opening_candidates(layout, anchor_ratio=fallback_anchor_ratio)
        return candidates, {
            "source": "wall_center_fallback",
            "hasExtendedDepth": False,
            "candidateCount": len(candidates),
        }

    delta = np.maximum(extended_depth - enclosed_depth, 0.0)
    relative_delta = delta / np.maximum(enclosed_depth, 1e-6)
    heat = _smooth_signal(relative_delta, radius=2)
    positive = heat[heat > 1e-8]
    if len(positive) > 0:
        scale = max(float(np.percentile(positive, 95)), 1e-6)
        heat_norm = np.clip(heat / scale, 0.0, 1.0)
    else:
        heat_norm = np.zeros_like(heat)

    mask = heat_norm >= threshold
    segments = _segments_from_mask(mask, max(1, int(min_width_tokens)))
    wall_ids, wall_ratios = wall_token_assignment(layout, sample_count=sample_count, depth=enclosed_depth)

    candidates: List[OpeningCandidate] = []
    for start, end in segments:
        tokens = _tokens_in_segment(start, end, sample_count)
        grouped = Counter(wall_ids[tokens].tolist())
        for wall_index, count in grouped.most_common():
            wall_tokens = tokens[wall_ids[tokens] == wall_index]
            if len(wall_tokens) < max(1, min_width_tokens):
                continue
            ratios = wall_ratios[wall_tokens]
            pad = max(0.01, 0.5 / max(1, len(wall_tokens)))
            start_ratio = max(0.0, float(np.min(ratios)) - pad)
            end_ratio = min(1.0, float(np.max(ratios)) + pad)
            if end_ratio - start_ratio < 1e-3:
                center = (start_ratio + end_ratio) / 2.0
                start_ratio = max(0.0, center - 0.05)
                end_ratio = min(1.0, center + 0.05)
            segment_heat = heat_norm[wall_tokens]
            segment_delta = relative_delta[wall_tokens]
            candidates.append(OpeningCandidate(
                wall_index=int(wall_index),
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                confidence=float(np.clip(np.mean(segment_heat), 0.0, 1.0)),
                token_start=int(start),
                token_end=int(end),
                source="extended_minus_enclosed",
                metrics={
                    "tokenCount": int(len(wall_tokens)),
                    "segmentTokenStart": int(start),
                    "segmentTokenEnd": int(end),
                    "meanRelativeDelta": float(np.mean(segment_delta)),
                    "maxRelativeDelta": float(np.max(segment_delta)),
                    "meanHeat": float(np.mean(segment_heat)),
                    "threshold": float(threshold),
                },
            ))

    if not candidates:
        candidates = fallback_opening_candidates(layout, anchor_ratio=fallback_anchor_ratio)
        return candidates, {
            "source": "wall_center_fallback_after_empty_passability",
            "hasExtendedDepth": True,
            "threshold": float(threshold),
            "maxHeat": float(np.max(heat_norm)) if len(heat_norm) else 0.0,
            "candidateCount": len(candidates),
        }

    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    candidates = candidates[:max_candidates]
    return candidates, {
        "source": "extended_minus_enclosed",
        "hasExtendedDepth": True,
        "threshold": float(threshold),
        "minWidthTokens": int(min_width_tokens),
        "candidateCount": len(candidates),
        "maxHeat": float(np.max(heat_norm)),
        "meanPositiveRelativeDelta": float(np.mean(positive)) if len(positive) else 0.0,
    }


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-9) -> int:
    value = _cross_2d(b - a, c - a)
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray,
                      eps: float = 1e-9) -> bool:
    if _orientation(start, end, point, eps=eps) != 0:
        return False
    return bool(
        min(start[0], end[0]) - eps <= point[0] <= max(start[0], end[0]) + eps
        and min(start[1], end[1]) - eps <= point[1] <= max(start[1], end[1]) + eps
    )


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray,
                        eps: float = 1e-9) -> bool:
    o1 = _orientation(a0, a1, b0, eps=eps)
    o2 = _orientation(a0, a1, b1, eps=eps)
    o3 = _orientation(b0, b1, a0, eps=eps)
    o4 = _orientation(b0, b1, a1, eps=eps)
    if o1 != o2 and o3 != o4:
        return True
    return bool(
        (o1 == 0 and _point_on_segment(b0, a0, a1, eps=eps))
        or (o2 == 0 and _point_on_segment(b1, a0, a1, eps=eps))
        or (o3 == 0 and _point_on_segment(a0, b0, b1, eps=eps))
        or (o4 == 0 and _point_on_segment(a1, b0, b1, eps=eps))
    )


def polygon_validity(layout_or_points: Any, min_area: float = 1e-6,
                     min_edge_length: float = 1e-6) -> Dict[str, Any]:
    """Return dependency-free polygon validity diagnostics."""
    try:
        if isinstance(layout_or_points, dict):
            points = layout_xz(layout_or_points)
        else:
            points = np.asarray(layout_or_points, dtype=np.float64)
    except Exception as exc:
        return {
            "valid": False,
            "isSimple": False,
            "error": str(exc),
            "vertexCount": 0,
            "selfIntersectionCount": 0,
        }

    finite = bool(points.ndim == 2 and points.shape[1:] == (2,) and np.isfinite(points).all())
    vertex_count = int(len(points)) if points.ndim >= 1 else 0
    if not finite or vertex_count < 3:
        return {
            "valid": False,
            "isSimple": False,
            "finite": finite,
            "vertexCount": vertex_count,
            "area": 0.0,
            "minimumEdgeLength": 0.0,
            "duplicateEdgeCount": 0,
            "selfIntersectionCount": 0,
        }

    edges = np.roll(points, -1, axis=0) - points
    edge_lengths = np.linalg.norm(edges, axis=-1)
    duplicate_edges = int(np.sum(edge_lengths <= min_edge_length))
    intersections = []
    for first in range(vertex_count):
        first_next = (first + 1) % vertex_count
        for second in range(first + 1, vertex_count):
            second_next = (second + 1) % vertex_count
            if first == second or first_next == second or second_next == first:
                continue
            if first == 0 and second_next == 0:
                continue
            if _segments_intersect(
                points[first], points[first_next], points[second], points[second_next]
            ):
                intersections.append([first, second])

    signed_area = 0.5 * float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    ))
    area = abs(signed_area)
    is_simple = duplicate_edges == 0 and len(intersections) == 0
    valid = finite and vertex_count >= 3 and area > min_area and is_simple
    return {
        "valid": bool(valid),
        "isSimple": bool(is_simple),
        "finite": finite,
        "vertexCount": vertex_count,
        "area": float(area),
        "clockwise": bool(signed_area < 0),
        "minimumEdgeLength": float(edge_lengths.min()),
        "maximumEdgeLength": float(edge_lengths.max()),
        "duplicateEdgeCount": duplicate_edges,
        "selfIntersectionCount": len(intersections),
        "selfIntersections": intersections,
    }


def polygon_overlap_ratio(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if Polygon is not None:
        try:
            poly_a = Polygon(points_a)
            poly_b = Polygon(points_b)
            if not poly_a.is_valid:
                poly_a = poly_a.buffer(0)
            if not poly_b.is_valid:
                poly_b = poly_b.buffer(0)
            denom = min(poly_a.area, poly_b.area)
            if denom <= 1e-8:
                return 0.0
            return float(poly_a.intersection(poly_b).area / denom)
        except Exception:
            pass

    # Shapely is optional in the project environment. Use a deterministic
    # raster estimate instead of silently treating every overlap as zero.
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    minimum = np.maximum(points_a.min(axis=0), points_b.min(axis=0))
    maximum = np.minimum(points_a.max(axis=0), points_b.max(axis=0))
    span = maximum - minimum
    if np.any(span <= 0):
        return 0.0

    resolution = 128
    x = minimum[0] + (np.arange(resolution) + 0.5) * span[0] / resolution
    y = minimum[1] + (np.arange(resolution) + 0.5) * span[1] / resolution
    grid_x, grid_y = np.meshgrid(x, y)

    def points_inside_polygon(polygon: np.ndarray) -> np.ndarray:
        inside = np.zeros(grid_x.shape, dtype=bool)
        previous = polygon[-1]
        for current in polygon:
            crosses_y = (current[1] > grid_y) != (previous[1] > grid_y)
            intersection_x = (
                (previous[0] - current[0])
                * (grid_y - current[1])
                / (previous[1] - current[1] + 1e-12)
                + current[0]
            )
            inside ^= crosses_y & (grid_x < intersection_x)
            previous = current
        return inside

    intersection_fraction = np.mean(
        points_inside_polygon(points_a) & points_inside_polygon(points_b)
    )
    intersection_area = float(intersection_fraction * span[0] * span[1])
    denominator = min(polygon_area(points_a), polygon_area(points_b))
    return float(np.clip(intersection_area / max(denominator, 1e-8), 0.0, 1.0))


def _as_match_array(value: Any, expected_ndim: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if array.ndim == expected_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != expected_ndim or not np.isfinite(array).all():
        return None
    return array


def _opening_tokens(opening: OpeningCandidate, layout: Dict, sample_count: int,
                    wall_ids: np.ndarray, wall_ratios: np.ndarray) -> np.ndarray:
    if opening.token_start >= 0 and opening.token_end >= 0:
        return _tokens_in_segment(opening.token_start, opening.token_end, sample_count)
    mask = (
        (wall_ids == opening.wall_index)
        & (wall_ratios >= opening.start_ratio)
        & (wall_ratios <= opening.end_ratio)
    )
    return np.flatnonzero(mask)


def opening_pair_feature_metrics(
    opening_a: OpeningCandidate,
    opening_b: OpeningCandidate,
    tokens_a: np.ndarray,
    tokens_b: np.ndarray,
    match_evidence: Optional[Mapping[str, Any]],
    candidate_index_a: int,
    candidate_index_b: int,
) -> Dict[str, float]:
    """Aggregate token and candidate neural evidence for one opening pair."""
    if not match_evidence:
        return {
            "featureScore": 0.0,
            "featureEvidenceAvailable": 0.0,
        }

    components = []
    metrics: Dict[str, float] = {}
    affinity_ab = _as_match_array(match_evidence.get("Aff_AB"), expected_ndim=2)
    affinity_ba = _as_match_array(match_evidence.get("Aff_BA"), expected_ndim=2)
    shared_a = _as_match_array(match_evidence.get("S_A"), expected_ndim=1)
    shared_b = _as_match_array(match_evidence.get("S_B"), expected_ndim=1)
    candidate_scores = _as_match_array(
        match_evidence.get("candidate_pair_score"), expected_ndim=2
    )

    if len(tokens_a) and len(tokens_b) and affinity_ab is not None:
        if affinity_ab.shape[0] > int(tokens_a.max()) and affinity_ab.shape[1] > int(tokens_b.max()):
            block = affinity_ab[np.ix_(tokens_a, tokens_b)]
            score = float(np.clip(block.sum(axis=1).mean(), 0.0, 1.0))
            metrics["affinityScoreAB"] = score
            components.append(score)
    if len(tokens_a) and len(tokens_b) and affinity_ba is not None:
        if affinity_ba.shape[0] > int(tokens_b.max()) and affinity_ba.shape[1] > int(tokens_a.max()):
            block = affinity_ba[np.ix_(tokens_b, tokens_a)]
            score = float(np.clip(block.sum(axis=1).mean(), 0.0, 1.0))
            metrics["affinityScoreBA"] = score
            components.append(score)
    if len(tokens_a) and shared_a is not None and shared_a.shape[0] > int(tokens_a.max()):
        score = float(np.clip(shared_a[tokens_a].mean(), 0.0, 1.0))
        metrics["sharedHeatScoreA"] = score
        components.append(score)
    if len(tokens_b) and shared_b is not None and shared_b.shape[0] > int(tokens_b.max()):
        score = float(np.clip(shared_b[tokens_b].mean(), 0.0, 1.0))
        metrics["sharedHeatScoreB"] = score
        components.append(score)
    if candidate_scores is not None:
        if (candidate_index_a < candidate_scores.shape[0]
                and candidate_index_b < candidate_scores.shape[1]):
            score = float(np.clip(
                candidate_scores[candidate_index_a, candidate_index_b], 0.0, 1.0
            ))
            metrics["candidateMatchScore"] = score
            components.append(score)

    metrics["featureScore"] = float(np.mean(components)) if components else 0.0
    metrics["featureEvidenceAvailable"] = 1.0 if components else 0.0
    return metrics


def _score_candidate(layout_a: Dict, layout_b: Dict, joint_layout: Dict,
                     opening_a: OpeningCandidate, opening_b: OpeningCandidate,
                     passability_weight: float = 1.0,
                     feature_metrics: Optional[Dict[str, float]] = None,
                     feature_weight: float = 1.0) -> Dict[str, float]:
    points_a = np.asarray(joint_layout["rooms"][0]["boundary"], dtype=np.float64)
    points_b = np.asarray(joint_layout["rooms"][1]["boundary"], dtype=np.float64)
    area_a = polygon_area(points_a)
    area_b = polygon_area(points_b)
    len_a = wall_length(layout_a, opening_a.wall_index)
    len_b = wall_length(layout_b, opening_b.wall_index)
    opening_width_a = len_a * (opening_a.end_ratio - opening_a.start_ratio)
    opening_width_b = len_b * (opening_b.end_ratio - opening_b.start_ratio)
    scale = float(joint_layout["alignment"]["roomBScale"])
    overlap = polygon_overlap_ratio(points_a, points_b)
    length_ratio_penalty = abs(math.log((len_b + 1e-8) / (len_a + 1e-8)))
    opening_width_penalty = abs(math.log(
        (opening_width_b + 1e-8) / (opening_width_a + 1e-8)
    ))
    scale_penalty = abs(math.log(scale + 1e-8))
    area_ratio_penalty = abs(math.log((area_b + 1e-8) / (area_a + 1e-8)))
    side_product = float(joint_layout["alignment"]["centroidSideProduct"])
    same_side_penalty = 1.0 if side_product > 0 else 0.0
    opening_confidence = (opening_a.confidence + opening_b.confidence) / 2.0
    passability_reward = passability_weight * opening_confidence
    feature_metrics = feature_metrics or {
        "featureScore": 0.0,
        "featureEvidenceAvailable": 0.0,
    }
    feature_score = float(feature_metrics.get("featureScore", 0.0))
    feature_reward = float(feature_weight) * feature_score

    geometry_score = (
        0.25 * length_ratio_penalty
        + 0.75 * opening_width_penalty
        + 0.5 * scale_penalty
        + 0.25 * area_ratio_penalty
        + 3.0 * overlap
        + same_side_penalty
    )
    score = geometry_score - passability_reward - feature_reward
    metrics = {
        "wallALength": len_a,
        "wallBLength": len_b,
        "openingAWidth": opening_width_a,
        "openingBWidth": opening_width_b,
        "areaA": area_a,
        "areaBWorld": area_b,
        "lengthRatioPenalty": length_ratio_penalty,
        "openingWidthPenalty": opening_width_penalty,
        "scalePenalty": scale_penalty,
        "areaRatioPenalty": area_ratio_penalty,
        "overlapRatio": overlap,
        "centroidSideProduct": side_product,
        "sameSidePenalty": same_side_penalty,
        "openingConfidenceA": opening_a.confidence,
        "openingConfidenceB": opening_b.confidence,
        "openingConfidenceMean": opening_confidence,
        "passabilityWeight": float(passability_weight),
        "passabilityReward": float(passability_reward),
        "featureWeight": float(feature_weight),
        "featureReward": float(feature_reward),
        "geometryScore": float(geometry_score),
        "score": float(score),
    }
    metrics.update({key: float(value) for key, value in feature_metrics.items()})
    return metrics


def _interval_iou(first: OpeningCandidate, second: OpeningCandidate) -> float:
    if first.wall_index != second.wall_index:
        return 0.0
    intersection = max(
        0.0,
        min(first.end_ratio, second.end_ratio) - max(first.start_ratio, second.start_ratio),
    )
    union = max(first.end_ratio, second.end_ratio) - min(first.start_ratio, second.start_ratio)
    return float(intersection / max(union, 1e-8))


def opening_pair_nms(raw_candidates: List[Dict], overlap_threshold: float = 0.8) -> List[Dict]:
    """Suppress duplicate opening pairs while preserving different wall hypotheses."""
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")
    kept = []
    for candidate in raw_candidates:
        duplicate = any(
            _interval_iou(candidate["opening_a"], previous["opening_a"]) >= overlap_threshold
            and _interval_iou(candidate["opening_b"], previous["opening_b"]) >= overlap_threshold
            for previous in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def estimate_wall_pair_candidates(
    layout_a: Dict,
    layout_b: Dict,
    anchor_ratio: float = 0.3,
    top_k: int = 8,
    calibrate_scale: bool = True,
    openings_a: Optional[List[OpeningCandidate]] = None,
    openings_b: Optional[List[OpeningCandidate]] = None,
    passability_weight: float = 1.0,
    match_evidence: Optional[Mapping[str, Any]] = None,
    feature_weight: float = 1.0,
    nms_overlap_threshold: float = 0.8,
    confidence_temperature: float = 1.0,
    validate_polygons: bool = True,
) -> Tuple[List[WallPairCandidate], Dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if confidence_temperature <= 0:
        raise ValueError("confidence_temperature must be positive")
    validity_a = polygon_validity(layout_a)
    validity_b = polygon_validity(layout_b)
    if validate_polygons and not validity_a["valid"]:
        raise ValueError(f"layout A polygon is invalid: {validity_a}")
    if validate_polygons and not validity_b["valid"]:
        raise ValueError(f"layout B polygon is invalid: {validity_b}")

    if openings_a is None:
        openings_a = fallback_opening_candidates(layout_a, anchor_ratio=anchor_ratio)
    if openings_b is None:
        openings_b = fallback_opening_candidates(layout_b, anchor_ratio=anchor_ratio)

    sample_count = 256
    wall_ids_a, wall_ratios_a = wall_token_assignment(layout_a, sample_count=sample_count)
    wall_ids_b, wall_ratios_b = wall_token_assignment(layout_b, sample_count=sample_count)
    opening_tokens_a = [
        _opening_tokens(item, layout_a, sample_count, wall_ids_a, wall_ratios_a)
        for item in openings_a
    ]
    opening_tokens_b = [
        _opening_tokens(item, layout_b, sample_count, wall_ids_b, wall_ratios_b)
        for item in openings_b
    ]

    raw_candidates = []
    rejected_count = 0
    for index_a, opening_a in enumerate(openings_a):
        door_a = opening_a.spec
        for index_b, opening_b in enumerate(openings_b):
            door_b = opening_b.spec
            try:
                joint_layout = build_joint_layout(
                    layout_a,
                    layout_b,
                    door_a,
                    door_b,
                    calibrate_scale=calibrate_scale,
                )
            except Exception as exc:
                rejected_count += 1
                raw_candidates.append({
                    "wall_a": opening_a.wall_index,
                    "wall_b": opening_b.wall_index,
                    "error": str(exc),
                })
                continue
            feature_metrics = opening_pair_feature_metrics(
                opening_a,
                opening_b,
                opening_tokens_a[index_a],
                opening_tokens_b[index_b],
                match_evidence,
                index_a,
                index_b,
            )
            metrics = _score_candidate(
                layout_a,
                layout_b,
                joint_layout,
                opening_a,
                opening_b,
                passability_weight=passability_weight,
                feature_metrics=feature_metrics,
                feature_weight=feature_weight,
            )
            raw_candidates.append({
                "wall_a": opening_a.wall_index,
                "wall_b": opening_b.wall_index,
                "door_a": door_a,
                "door_b": door_b,
                "opening_a": opening_a,
                "opening_b": opening_b,
                "joint_layout": joint_layout,
                "metrics": metrics,
            })

    valid = [item for item in raw_candidates if "joint_layout" in item]
    valid.sort(key=lambda item: item["metrics"]["score"])
    valid_before_nms = len(valid)
    valid = opening_pair_nms(valid, overlap_threshold=nms_overlap_threshold)
    selected = valid[:top_k]
    if selected:
        scores = np.asarray([item["metrics"]["score"] for item in selected], dtype=np.float64)
        logits = -(scores - scores.min()) / confidence_temperature
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
    else:
        probabilities = np.asarray([], dtype=np.float64)

    candidates: List[WallPairCandidate] = []
    for rank, (item, confidence) in enumerate(zip(selected, probabilities), start=1):
        score = float(item["metrics"]["score"])
        candidates.append(WallPairCandidate(
            rank=rank,
            score=score,
            confidence=float(confidence),
            wall_a=item["wall_a"],
            wall_b=item["wall_b"],
            door_a=item["door_a"],
            door_b=item["door_b"],
            opening_a=item["opening_a"],
            opening_b=item["opening_b"],
            metrics=item["metrics"],
            alignment=item["joint_layout"]["alignment"],
        ))

    best_joint_layout = deepcopy(selected[0]["joint_layout"]) if selected else {}
    if candidates:
        best_joint_layout["confidence"] = candidates[0].confidence
        best_joint_layout["selection"] = {
            "rank": 1,
            "score": candidates[0].score,
            "confidence": candidates[0].confidence,
            "wallA": candidates[0].wall_a,
            "wallB": candidates[0].wall_b,
            "metrics": candidates[0].metrics,
        }
        best_joint_layout["diagnostics"] = {
            "layoutAValidity": validity_a,
            "layoutBValidity": validity_b,
            "rawCandidateCount": len(openings_a) * len(openings_b),
            "rejectedCandidateCount": rejected_count,
            "validCandidateCount": valid_before_nms,
            "postNmsCandidateCount": len(valid),
            "nmsOverlapThreshold": float(nms_overlap_threshold),
        }
    return candidates, best_joint_layout

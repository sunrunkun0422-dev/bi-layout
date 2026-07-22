"""Build the ZInD-BiPair-v1 partial-opening supervision contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dataset.communal.base_dataset import BaseDataset
from dataset.zind_pair_mining import (
    interface_token_interval,
    relative_zind_transform,
    transform_points,
    zind_transform_matrix,
)
from utils.boundary import visibility_corners
from utils.conversion import xyz2uv
from utils.cross_scene_estimator import polygon_overlap_ratio, polygon_validity


DATASET_NAME = "ZInD-BiPair-v1"
PAIR_TYPE = "partial_opening"


@dataclass(frozen=True)
class PortalSegment:
    index: int
    local_endpoints: np.ndarray
    global_endpoints: np.ndarray
    token_interval: Tuple[int, int]


@dataclass(frozen=True)
class EligibleView:
    house_id: str
    floor_id: str
    complete_room_id: str
    partial_room_id: str
    pano_id: str
    image_path: str
    room_label: str
    camera_height: float
    ceiling_height: float
    transform: Mapping[str, Any]
    layout_raw: Mapping[str, Any]
    layout_visible: Mapping[str, Any]
    layout_complete: Mapping[str, Any]
    portals: Tuple[PortalSegment, ...]

    @property
    def view_id(self) -> str:
        return "{}_{}_{}_{}_{}".format(
            self.house_id,
            self.floor_id,
            self.complete_room_id,
            self.partial_room_id,
            self.pano_id,
        )


@dataclass(frozen=True)
class PortalMatch:
    portal_a: PortalSegment
    portal_b: PortalSegment
    shared_global_endpoints: np.ndarray
    midpoint_distance_m: float
    direction_difference_degrees: float
    relative_length_error: float
    endpoint_error_m: float
    length_a_m: float
    length_b_m: float


@dataclass(frozen=True)
class PairExample:
    view_a: EligibleView
    view_b: EligibleView
    floor_scale_meters: float
    portal_match: Optional[PortalMatch]
    camera_distance_m: float

    @property
    def is_positive(self) -> bool:
        return self.portal_match is not None

    @property
    def pair_id(self) -> str:
        base = "{}_{}_{}_{}_{}__{}_{}".format(
            self.view_a.house_id,
            self.view_a.floor_id,
            self.view_a.complete_room_id,
            self.view_a.partial_room_id,
            self.view_a.pano_id,
            self.view_b.partial_room_id,
            self.view_b.pano_id,
        )
        if self.portal_match is None:
            return f"{base}_negative"
        return "{}_opening_{}_{}".format(
            base,
            self.portal_match.portal_a.index,
            self.portal_match.portal_b.index,
        )


@dataclass(frozen=True)
class PairThresholds:
    midpoint_distance_m: float = 0.20
    direction_difference_degrees: float = 10.0
    relative_length_error: float = 0.20
    endpoint_error_m: float = 0.25

    def validate(self) -> None:
        values = (
            self.midpoint_distance_m,
            self.direction_difference_degrees,
            self.relative_length_error,
            self.endpoint_error_m,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all portal matching thresholds must be positive")


def _opening_segments(
    pano: Mapping[str, Any], token_count: int
) -> Tuple[PortalSegment, ...]:
    layout = pano["layout_raw"]
    vertices = layout.get("openings", [])
    local_to_global = zind_transform_matrix(pano["floor_plan_transformation"])
    segments = []
    for offset in range(0, len(vertices), 3):
        if offset + 1 >= len(vertices):
            continue
        local = np.asarray(vertices[offset : offset + 2], dtype=np.float64)
        if local.shape != (2, 2) or not np.isfinite(local).all():
            continue
        segments.append(
            PortalSegment(
                index=offset // 3,
                local_endpoints=local,
                global_endpoints=transform_points(local, local_to_global),
                token_interval=interface_token_interval(local, token_count),
            )
        )
    return tuple(segments)


def eligible_view_from_pano(
    *,
    house_id: str,
    floor_id: str,
    complete_room_id: str,
    partial_room_id: str,
    pano_id: str,
    pano: Mapping[str, Any],
    data_root: Path,
    token_count: int,
) -> Tuple[Optional[EligibleView], Optional[str]]:
    """Apply the strict v1 view filter from the dataset specification."""
    if not pano.get("is_primary", False):
        return None, "not_primary"
    if not pano.get("is_inside", False):
        return None, "not_inside"
    if not pano.get("is_ceiling_flat", False):
        return None, "ceiling_not_flat"
    required = (
        "layout_raw",
        "layout_visible",
        "layout_complete",
        "floor_plan_transformation",
        "camera_height",
        "ceiling_height",
        "image_path",
    )
    missing = [key for key in required if key not in pano or pano[key] is None]
    if missing:
        return None, "missing_" + "_".join(missing)
    if any(len(pano[key].get("vertices", [])) < 3 for key in (
        "layout_raw", "layout_visible", "layout_complete"
    )):
        return None, "invalid_layout_vertices"
    image_path = data_root / house_id / str(pano["image_path"])
    if not image_path.is_file():
        return None, "missing_image"
    portals = _opening_segments(pano, token_count)
    if not portals:
        return None, "no_layout_raw_opening"
    return (
        EligibleView(
            house_id=house_id,
            floor_id=floor_id,
            complete_room_id=complete_room_id,
            partial_room_id=partial_room_id,
            pano_id=pano_id,
            image_path=f"{house_id}/{pano['image_path']}",
            room_label=str(pano.get("label", "")),
            camera_height=float(pano["camera_height"]),
            ceiling_height=float(pano["ceiling_height"]),
            transform=pano["floor_plan_transformation"],
            layout_raw=pano["layout_raw"],
            layout_visible=pano["layout_visible"],
            layout_complete=pano["layout_complete"],
            portals=portals,
        ),
        None,
    )


def _segment_metrics(
    portal_a: PortalSegment,
    portal_b: PortalSegment,
    floor_scale_meters: float,
) -> Dict[str, Any]:
    a = portal_a.global_endpoints
    b = portal_b.global_endpoints
    midpoint_distance_m = float(
        np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) * floor_scale_meters
    )
    vector_a = a[1] - a[0]
    vector_b = b[1] - b[0]
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    if min(length_a, length_b) <= 1e-9:
        return {"valid": False}
    cosine = float(
        np.clip(
            abs(np.dot(vector_a, vector_b) / (length_a * length_b)), 0.0, 1.0
        )
    )
    direction_difference = math.degrees(math.acos(cosine))
    relative_length_error = abs(length_a - length_b) / max(length_a, length_b)
    direct = (
        np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[1] - b[1])
    ) / 2.0
    reversed_error = (
        np.linalg.norm(a[0] - b[1]) + np.linalg.norm(a[1] - b[0])
    ) / 2.0
    if direct <= reversed_error:
        aligned_b = b
        endpoint_error = direct
    else:
        aligned_b = b[::-1]
        endpoint_error = reversed_error
    return {
        "valid": True,
        "midpoint_distance_m": midpoint_distance_m,
        "direction_difference_degrees": float(direction_difference),
        "relative_length_error": float(relative_length_error),
        "endpoint_error_m": float(endpoint_error * floor_scale_meters),
        "length_a_m": float(length_a * floor_scale_meters),
        "length_b_m": float(length_b * floor_scale_meters),
        "shared_global_endpoints": (a + aligned_b) / 2.0,
    }


def match_opening_portals(
    view_a: EligibleView,
    view_b: EligibleView,
    floor_scale_meters: float,
    thresholds: PairThresholds,
) -> List[PortalMatch]:
    """Greedily form one-to-one portal matches using metric ZInD geometry."""
    thresholds.validate()
    candidates = []
    for portal_a in view_a.portals:
        for portal_b in view_b.portals:
            metrics = _segment_metrics(portal_a, portal_b, floor_scale_meters)
            if not metrics["valid"]:
                continue
            if metrics["midpoint_distance_m"] > thresholds.midpoint_distance_m:
                continue
            if (
                metrics["direction_difference_degrees"]
                > thresholds.direction_difference_degrees
            ):
                continue
            if metrics["relative_length_error"] > thresholds.relative_length_error:
                continue
            if metrics["endpoint_error_m"] > thresholds.endpoint_error_m:
                continue
            score = (
                metrics["endpoint_error_m"] / thresholds.endpoint_error_m
                + metrics["midpoint_distance_m"] / thresholds.midpoint_distance_m
                + metrics["direction_difference_degrees"]
                / thresholds.direction_difference_degrees
                + metrics["relative_length_error"]
                / thresholds.relative_length_error
            )
            candidates.append((score, portal_a.index, portal_b.index, portal_a, portal_b, metrics))
    candidates.sort(key=lambda item: item[:3])

    matches = []
    used_a = set()
    used_b = set()
    for _, _, _, portal_a, portal_b, metrics in candidates:
        if portal_a.index in used_a or portal_b.index in used_b:
            continue
        used_a.add(portal_a.index)
        used_b.add(portal_b.index)
        matches.append(
            PortalMatch(
                portal_a=portal_a,
                portal_b=portal_b,
                shared_global_endpoints=metrics["shared_global_endpoints"],
                midpoint_distance_m=metrics["midpoint_distance_m"],
                direction_difference_degrees=metrics[
                    "direction_difference_degrees"
                ],
                relative_length_error=metrics["relative_length_error"],
                endpoint_error_m=metrics["endpoint_error_m"],
                length_a_m=metrics["length_a_m"],
                length_b_m=metrics["length_b_m"],
            )
        )
    return matches


def camera_distance_m(
    view_a: EligibleView, view_b: EligibleView, floor_scale_meters: float
) -> float:
    point_a = np.asarray(view_a.transform["translation"], dtype=np.float64)
    point_b = np.asarray(view_b.transform["translation"], dtype=np.float64)
    return float(np.linalg.norm(point_a - point_b) * floor_scale_meters)


def build_house_pair_examples(
    payload: Mapping[str, Any],
    house_id: str,
    data_root: Path,
    token_count: int,
    thresholds: PairThresholds,
) -> Tuple[List[PairExample], List[PairExample], List[Dict[str, Any]], int]:
    """Return positive pairs, safe same-room negatives, invalid views, and view count."""
    positives: List[PairExample] = []
    negatives: List[PairExample] = []
    invalid = []
    eligible_count = 0
    floor_scales = payload.get("scale_meters_per_coordinate", {})
    for floor_id, floor_data in payload.get("merger", {}).items():
        floor_scale = floor_scales.get(floor_id)
        if floor_scale is None or float(floor_scale) <= 0:
            invalid.append(
                {
                    "house_id": house_id,
                    "floor_id": floor_id,
                    "reason": "missing_scale_meters_per_coordinate",
                }
            )
            continue
        floor_scale = float(floor_scale)
        for complete_room_id, complete_room in floor_data.items():
            views = []
            for partial_room_id, partial_room in complete_room.items():
                primary_seen = False
                for pano_id, pano in partial_room.items():
                    if pano.get("is_primary", False):
                        primary_seen = True
                    view, reason = eligible_view_from_pano(
                        house_id=house_id,
                        floor_id=floor_id,
                        complete_room_id=complete_room_id,
                        partial_room_id=partial_room_id,
                        pano_id=pano_id,
                        pano=pano,
                        data_root=data_root,
                        token_count=token_count,
                    )
                    if view is not None:
                        views.append(view)
                        eligible_count += 1
                    elif pano.get("is_primary", False):
                        invalid.append(
                            {
                                "house_id": house_id,
                                "floor_id": floor_id,
                                "complete_room_id": complete_room_id,
                                "partial_room_id": partial_room_id,
                                "pano_id": pano_id,
                                "reason": reason,
                            }
                        )
                if not primary_seen:
                    invalid.append(
                        {
                            "house_id": house_id,
                            "floor_id": floor_id,
                            "complete_room_id": complete_room_id,
                            "partial_room_id": partial_room_id,
                            "reason": "no_primary_panorama",
                        }
                    )
            views.sort(key=lambda view: (view.partial_room_id, view.pano_id))
            for index, view_a in enumerate(views):
                for view_b in views[index + 1 :]:
                    if view_a.partial_room_id == view_b.partial_room_id:
                        continue
                    matches = match_opening_portals(
                        view_a, view_b, floor_scale, thresholds
                    )
                    distance = camera_distance_m(view_a, view_b, floor_scale)
                    if matches:
                        positives.extend(
                            PairExample(view_a, view_b, floor_scale, match, distance)
                            for match in matches
                        )
                    else:
                        negatives.append(
                            PairExample(view_a, view_b, floor_scale, None, distance)
                        )
    positives.sort(key=lambda pair: pair.pair_id)
    negatives.sort(key=lambda pair: (pair.camera_distance_m, pair.pair_id))
    return positives, negatives, invalid, eligible_count


def _interval_mask(interval: Tuple[int, int], token_count: int) -> np.ndarray:
    mask = np.zeros(token_count, dtype=np.uint8)
    start, end = (int(value) % token_count for value in interval)
    if start <= end:
        mask[start : end + 1] = 1
    else:
        mask[start:] = 1
        mask[: end + 1] = 1
    return mask


def _union_portal_mask(
    portals: Sequence[PortalSegment], token_count: int
) -> np.ndarray:
    mask = np.zeros(token_count, dtype=np.uint8)
    for portal in portals:
        mask |= _interval_mask(portal.token_interval, token_count)
    return mask


def _layout_uv_and_depth(
    layout: Mapping[str, Any], camera_height: float, token_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    corner_xz = np.asarray(layout["vertices"], dtype=np.float64).copy()
    if corner_xz.ndim != 2 or corner_xz.shape[1] != 2 or len(corner_xz) < 3:
        raise ValueError("layout vertices must have shape [N, 2] with N >= 3")
    corner_xz[:, 0] *= -1.0
    corner_xyz = np.insert(corner_xz, 1, float(camera_height), axis=1)
    corners = xyz2uv(corner_xyz).astype(np.float32)
    visible = visibility_corners(corners.copy())
    depth = BaseDataset.get_depth(
        visible, plan_y=1, length=token_count, visible=False
    )
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (token_count,) or not np.isfinite(depth).all():
        raise ValueError("layout projection did not produce a finite token depth profile")
    return corners, depth


def build_view_label_arrays(
    view: EligibleView, token_count: int
) -> Dict[str, np.ndarray]:
    corners_raw, depth_raw = _layout_uv_and_depth(
        view.layout_raw, view.camera_height, token_count
    )
    corners_visible, depth_visible = _layout_uv_and_depth(
        view.layout_visible, view.camera_height, token_count
    )
    ratio = (view.ceiling_height - view.camera_height) / view.camera_height
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("invalid camera/ceiling height ratio")
    complete_global = transform_points(
        view.layout_complete["vertices"], zind_transform_matrix(view.transform)
    ).astype(np.float32)
    validity = polygon_validity(complete_global)
    if not validity["valid"]:
        raise ValueError("layout_complete is not a valid global polygon")
    return {
        "depth_enclosed": depth_raw,
        "depth_extended": depth_visible,
        "extension_depth": np.maximum(0.0, depth_visible - depth_raw).astype(
            np.float32
        ),
        "ratio": np.asarray([ratio], dtype=np.float32),
        "corners_enclosed": corners_raw,
        "corners_extended": corners_visible,
        "opening_mask_all": _union_portal_mask(view.portals, token_count),
        "joint_layout_global": complete_global,
    }


def _portal_center_width(mask: np.ndarray) -> Tuple[int, int]:
    tokens = np.flatnonzero(mask)
    if len(tokens) == 0:
        return -1, 0
    # The source interval is always the shortest circular arc. Circular mean
    # remains stable for portals crossing token 255 -> 0.
    angle = tokens.astype(np.float64) * (2.0 * math.pi / len(mask))
    center_angle = math.atan2(np.sin(angle).mean(), np.cos(angle).mean())
    center = int(round((center_angle % (2.0 * math.pi)) * len(mask) / (2.0 * math.pi)))
    return center % len(mask), int(len(tokens))


def build_pair_record_and_arrays(
    example: PairExample,
    split: str,
    label_cache_path: str,
    token_count: int,
    view_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    view_a = example.view_a
    view_b = example.view_b
    labels_a = view_cache[view_a.view_id]
    labels_b = view_cache[view_b.view_id]
    transform = relative_zind_transform(view_a.transform, view_b.transform)
    relative_scale = float(math.hypot(transform[0, 0], transform[1, 0]))
    relative_yaw = float(math.atan2(transform[1, 0], transform[0, 0]))
    translation = transform[:2, 2].astype(np.float64)
    translation_meters = (
        translation
        * float(view_a.transform["scale"])
        * example.floor_scale_meters
    )

    portal_mask_a = np.zeros(token_count, dtype=np.uint8)
    portal_mask_b = np.zeros(token_count, dtype=np.uint8)
    shared_portal = None
    if example.portal_match is not None:
        match = example.portal_match
        portal_mask_a = _interval_mask(match.portal_a.token_interval, token_count)
        portal_mask_b = _interval_mask(match.portal_b.token_interval, token_count)
        shared_portal = {
            "type": "opening",
            "portal_id_A": match.portal_a.index,
            "portal_id_B": match.portal_b.index,
            "segment_floor": match.shared_global_endpoints.tolist(),
            "segment_floor_meters": (
                match.shared_global_endpoints * example.floor_scale_meters
            ).tolist(),
            "token_interval_A": list(match.portal_a.token_interval),
            "token_interval_B": list(match.portal_b.token_interval),
            "matching_metrics": {
                "midpoint_distance_m": match.midpoint_distance_m,
                "direction_difference_degrees": match.direction_difference_degrees,
                "relative_length_error": match.relative_length_error,
                "endpoint_error_m": match.endpoint_error_m,
                "length_A_m": match.length_a_m,
                "length_B_m": match.length_b_m,
            },
        }
    center_a, width_a = _portal_center_width(portal_mask_a)
    center_b, width_b = _portal_center_width(portal_mask_b)
    affinity = np.outer(portal_mask_a, portal_mask_b).astype(np.uint8)
    shared_global = (
        np.empty((0, 2), dtype=np.float32)
        if example.portal_match is None
        else example.portal_match.shared_global_endpoints.astype(np.float32)
    )

    polygon_a = labels_a["joint_layout_global"]
    polygon_b = labels_b["joint_layout_global"]
    overlap = float(polygon_overlap_ratio(polygon_a, polygon_b))
    record = {
        "schema_version": "1.0",
        "dataset_name": DATASET_NAME,
        "pair_id": example.pair_id,
        "split": split,
        "pair_type": PAIR_TYPE,
        "is_positive": example.is_positive,
        "house_id": view_a.house_id,
        "floor_id": view_a.floor_id,
        "complete_room_id": view_a.complete_room_id,
        "label_cache": label_cache_path,
        "view_A": {
            "image_path": view_a.image_path,
            "complete_room_id": view_a.complete_room_id,
            "partial_room_id": view_a.partial_room_id,
            "pano_id": view_a.pano_id,
            "room_label": view_a.room_label,
            "transformation": dict(view_a.transform),
            "opening_count": len(view_a.portals),
        },
        "view_B": {
            "image_path": view_b.image_path,
            "complete_room_id": view_b.complete_room_id,
            "partial_room_id": view_b.partial_room_id,
            "pano_id": view_b.pano_id,
            "room_label": view_b.room_label,
            "transformation": dict(view_b.transform),
            "opening_count": len(view_b.portals),
        },
        "shared_portal": shared_portal,
        "relative_pose_gt": {
            "T_B_to_A": transform.tolist(),
            "relative_yaw": relative_yaw,
            "translation": translation.tolist(),
            "translation_meters": translation_meters.tolist(),
            "relative_scale": relative_scale,
            "valid": True,
        },
        "joint_layout_gt": {
            "source": "layout_complete",
            "cache": label_cache_path,
            "overlap_ratio_A_B": overlap,
            "valid_A": bool(polygon_validity(polygon_a)["valid"]),
            "valid_B": bool(polygon_validity(polygon_b)["valid"]),
        },
        "camera_distance_m": example.camera_distance_m,
        "scale_meters_per_coordinate": example.floor_scale_meters,
    }
    arrays = {
        "depth_enclosed_A": labels_a["depth_enclosed"],
        "depth_extended_A": labels_a["depth_extended"],
        "depth_enclosed_B": labels_b["depth_enclosed"],
        "depth_extended_B": labels_b["depth_extended"],
        "extension_depth_A": labels_a["extension_depth"],
        "extension_depth_B": labels_b["extension_depth"],
        "ratio_A": labels_a["ratio"],
        "ratio_B": labels_b["ratio"],
        "corners_enclosed_A": labels_a["corners_enclosed"],
        "corners_extended_A": labels_a["corners_extended"],
        "corners_enclosed_B": labels_b["corners_enclosed"],
        "corners_extended_B": labels_b["corners_extended"],
        "opening_mask_all_A": labels_a["opening_mask_all"],
        "opening_mask_all_B": labels_b["opening_mask_all"],
        "portal_mask_A": portal_mask_a,
        "portal_mask_B": portal_mask_b,
        "portal_center_A": np.asarray([center_a], dtype=np.int64),
        "portal_center_B": np.asarray([center_b], dtype=np.int64),
        "portal_width_A": np.asarray([width_a], dtype=np.int64),
        "portal_width_B": np.asarray([width_b], dtype=np.int64),
        "affinity_gt": affinity,
        "T_B_to_A": transform.astype(np.float32),
        "relative_yaw_gt": np.asarray([relative_yaw], dtype=np.float32),
        "translation_gt": translation.astype(np.float32),
        "translation_meters_gt": translation_meters.astype(np.float32),
        "relative_scale_gt": np.asarray([relative_scale], dtype=np.float32),
        "joint_layout_global": polygon_a.astype(np.float32),
        "joint_layout_global_A": polygon_a.astype(np.float32),
        "joint_layout_global_B": polygon_b.astype(np.float32),
        "joint_layout_overlap_ratio": np.asarray([overlap], dtype=np.float32),
        "shared_portal_global": shared_global,
        "scale_meters_per_coordinate": np.asarray(
            [example.floor_scale_meters], dtype=np.float32
        ),
        "is_positive": np.asarray([example.is_positive], dtype=np.uint8),
    }
    return record, arrays


def interleave_pairs(
    positives: Sequence[PairExample], negatives: Sequence[PairExample]
) -> List[PairExample]:
    output = []
    count = max(len(positives), len(negatives))
    for index in range(count):
        if index < len(positives):
            output.append(positives[index])
        if index < len(negatives):
            output.append(negatives[index])
    return output

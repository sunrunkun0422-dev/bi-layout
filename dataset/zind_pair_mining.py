"""Mine panorama pairs that share a labeled ZInD door or opening."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def zind_transform_matrix(transform: Mapping[str, Any]) -> np.ndarray:
    """Return the local-to-floor 2D similarity transform used by ZInD."""
    angle = math.radians(float(transform["rotation"]))
    scale = float(transform["scale"])
    translation = np.asarray(transform["translation"], dtype=np.float64)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [
            [scale * cosine, -scale * sine, translation[0]],
            [scale * sine, scale * cosine, translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def relative_zind_transform(
    transform_a: Mapping[str, Any], transform_b: Mapping[str, Any]
) -> np.ndarray:
    """Return the ground-truth transform from panorama B into panorama A."""
    return np.linalg.inv(zind_transform_matrix(transform_a)) @ zind_transform_matrix(
        transform_b
    )


def transform_points(points: Sequence[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        (points_array, np.ones((len(points_array), 1), dtype=np.float64)), axis=1
    )
    return (matrix @ homogeneous.T).T[:, :2]


def segment_endpoint_error(first: np.ndarray, second: np.ndarray) -> float:
    """Average endpoint distance with direction-independent segment matching."""
    direct = np.linalg.norm(first[0] - second[0]) + np.linalg.norm(
        first[1] - second[1]
    )
    reversed_order = np.linalg.norm(first[0] - second[1]) + np.linalg.norm(
        first[1] - second[0]
    )
    return float(min(direct, reversed_order) / 2.0)


def interface_token_interval(
    local_endpoints: Sequence[Sequence[float]], token_count: int = 256
) -> Tuple[int, int]:
    """Project a ZInD floor-plan segment to the shortest circular pano interval."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    endpoints = np.asarray(local_endpoints, dtype=np.float64)
    if endpoints.shape != (2, 2):
        raise ValueError("local_endpoints must have shape [2, 2]")

    # ZInD local [x, y] becomes Bi-Layout horizontal [-x, z]. Camera height
    # does not affect the longitude, so only the two horizontal axes are used.
    longitude = np.arctan2(-endpoints[:, 0], endpoints[:, 1])
    normalized_u = np.mod(longitude / (2.0 * math.pi) + 0.5, 1.0)
    tokens = np.floor(normalized_u * token_count).astype(np.int64) % token_count
    first, second = int(tokens[0]), int(tokens[1])
    forward = (second - first) % token_count
    backward = (first - second) % token_count
    return (first, second) if forward <= backward else (second, first)


def _portable_path(path: str, data_root: Optional[Path]) -> str:
    resolved = Path(path).expanduser().resolve()
    if data_root is not None:
        try:
            return resolved.relative_to(data_root).as_posix()
        except ValueError:
            pass
    return str(resolved)


def _serialized_interface(interface: Mapping[str, Any], token_count: int) -> Dict[str, Any]:
    local = np.asarray(interface["local"], dtype=np.float64)
    global_endpoints = np.asarray(interface["global"], dtype=np.float64)
    return {
        "type": str(interface["type"]),
        "index": int(interface["index"]),
        "localEndpoints": local.tolist(),
        "globalEndpoints": global_endpoints.tolist(),
        "tokenInterval": list(interface_token_interval(local, token_count)),
    }


@dataclass(frozen=True)
class ZindAdjacentPair:
    pair_id: str
    house_id: str
    floor_id: str
    room_a: str
    room_b: str
    partial_room_a: str
    partial_room_b: str
    pano_a: str
    pano_b: str
    label_a: str
    label_b: str
    image_a: str
    image_b: str
    transform_a: Mapping[str, Any]
    transform_b: Mapping[str, Any]
    interface_type_a: str
    interface_type_b: str
    interface_index_a: int
    interface_index_b: int
    interface_local_a: Sequence[Sequence[float]]
    interface_local_b: Sequence[Sequence[float]]
    interface_global: Sequence[Sequence[float]]
    interface_endpoint_error: float
    camera_distance: float
    camera_to_interface_distance: float
    candidate_interfaces_a: Sequence[Mapping[str, Any]]
    candidate_interfaces_b: Sequence[Mapping[str, Any]]

    @property
    def ground_truth_transform(self) -> np.ndarray:
        return relative_zind_transform(self.transform_a, self.transform_b)

    def to_manifest_record(self) -> Dict[str, Any]:
        return {
            "id": self.pair_id,
            "scene_id": self.house_id,
            "floor_id": self.floor_id,
            "image_a": self.image_a,
            "image_b": self.image_b,
            "ground_truth": {
                "roomA": self.room_a,
                "roomB": self.room_b,
                "partialRoomA": self.partial_room_a,
                "partialRoomB": self.partial_room_b,
                "panoA": self.pano_a,
                "panoB": self.pano_b,
                "labelA": self.label_a,
                "labelB": self.label_b,
                "interfaceTypeA": self.interface_type_a,
                "interfaceTypeB": self.interface_type_b,
                "interfaceIndexA": self.interface_index_a,
                "interfaceIndexB": self.interface_index_b,
                "interfaceLocalA": self.interface_local_a,
                "interfaceLocalB": self.interface_local_b,
                "interfaceGlobal": self.interface_global,
                "interfaceEndpointError": self.interface_endpoint_error,
                "cameraDistance": self.camera_distance,
                "cameraToInterfaceDistance": self.camera_to_interface_distance,
                "transformBToA": self.ground_truth_transform.tolist(),
            },
        }

    def to_matching_record(
        self, token_count: int = 256, data_root: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return the compact manifest contract consumed by ZInDPairDataset."""
        root = None if data_root is None else Path(data_root).expanduser().resolve()
        candidates_a = [
            _serialized_interface(candidate, token_count)
            for candidate in self.candidate_interfaces_a
        ]
        candidates_b = [
            _serialized_interface(candidate, token_count)
            for candidate in self.candidate_interfaces_b
        ]
        target_a = next(
            index
            for index, candidate in enumerate(self.candidate_interfaces_a)
            if candidate["type"] == self.interface_type_a
            and int(candidate["index"]) == self.interface_index_a
        )
        target_b = next(
            index
            for index, candidate in enumerate(self.candidate_interfaces_b)
            if candidate["type"] == self.interface_type_b
            and int(candidate["index"]) == self.interface_index_b
        )
        matrix = self.ground_truth_transform
        return {
            "id": self.pair_id,
            "scene_id": self.house_id,
            "floor_id": self.floor_id,
            "image_a": _portable_path(self.image_a, root),
            "image_b": _portable_path(self.image_b, root),
            "room_a": self.room_a,
            "room_b": self.room_b,
            "partial_room_a": self.partial_room_a,
            "partial_room_b": self.partial_room_b,
            "pano_a": self.pano_a,
            "pano_b": self.pano_b,
            "room_label_a": self.label_a,
            "room_label_b": self.label_b,
            "token_count": int(token_count),
            "candidates_a": candidates_a,
            "candidates_b": candidates_b,
            "supervision": {
                "is_match": True,
                "target_candidate_a": target_a,
                "target_candidate_b": target_b,
                "relative_transform_b_to_a": matrix.tolist(),
                "relative_yaw_radians": float(math.atan2(matrix[1, 0], matrix[0, 0])),
                "pose_valid": True,
                "interface_endpoint_error": self.interface_endpoint_error,
            },
        }


def _interfaces(pano: Mapping[str, Any]) -> List[Dict[str, Any]]:
    layout = pano.get("layout_complete") or pano.get("layout_raw")
    if not layout:
        return []
    local_to_global = zind_transform_matrix(pano["floor_plan_transformation"])
    output = []
    for interface_type in ("doors", "openings"):
        vertices = layout.get(interface_type, [])
        for offset in range(0, len(vertices), 3):
            if offset + 1 >= len(vertices):
                continue
            local = np.asarray(vertices[offset : offset + 2], dtype=np.float64)
            output.append(
                {
                    "type": interface_type[:-1],
                    "index": offset // 3,
                    "candidate_index": len(output),
                    "local": local,
                    "global": transform_points(local, local_to_global),
                }
            )
    return output


def _primary_panoramas(
    floor_data: Mapping[str, Any], house_dir: Path
) -> Dict[str, List[Dict[str, Any]]]:
    rooms: Dict[str, List[Dict[str, Any]]] = {}
    for room_id, room_data in floor_data.items():
        for partial_room_id, panorama_data in room_data.items():
            selected = next(
                (
                    (pano_id, pano)
                    for pano_id, pano in panorama_data.items()
                    if pano.get("is_primary")
                ),
                next(iter(panorama_data.items())),
            )
            pano_id, pano = selected
            interfaces = _interfaces(pano)
            if not interfaces:
                continue
            rooms.setdefault(room_id, []).append(
                {
                    "room_id": room_id,
                    "partial_room_id": partial_room_id,
                    "pano_id": pano_id,
                    "label": str(pano.get("label", "")),
                    "image": str((house_dir / pano["image_path"]).resolve()),
                    "transform": pano["floor_plan_transformation"],
                    "interfaces": interfaces,
                }
            )
    return rooms


def _pair_candidate(
    pano_a: Mapping[str, Any], pano_b: Mapping[str, Any]
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    best = None
    camera_a = np.asarray(pano_a["transform"]["translation"], dtype=np.float64)
    camera_b = np.asarray(pano_b["transform"]["translation"], dtype=np.float64)
    for interface_a in pano_a["interfaces"]:
        for interface_b in pano_b["interfaces"]:
            direct_error = float(
                (
                    np.linalg.norm(interface_a["global"][0] - interface_b["global"][0])
                    + np.linalg.norm(
                        interface_a["global"][1] - interface_b["global"][1]
                    )
                )
                / 2.0
            )
            reversed_error = float(
                (
                    np.linalg.norm(interface_a["global"][0] - interface_b["global"][1])
                    + np.linalg.norm(
                        interface_a["global"][1] - interface_b["global"][0]
                    )
                )
                / 2.0
            )
            if direct_error <= reversed_error:
                error = direct_error
                aligned_global_b = interface_b["global"]
            else:
                error = reversed_error
                aligned_global_b = interface_b["global"][::-1]
            midpoint = (
                interface_a["global"].mean(axis=0)
                + interface_b["global"].mean(axis=0)
            ) / 2.0
            camera_to_interface = float(
                np.linalg.norm(camera_a - midpoint) + np.linalg.norm(camera_b - midpoint)
            )
            key = (error, camera_to_interface)
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "pano_a": pano_a,
                        "pano_b": pano_b,
                        "interface_a": interface_a,
                        "interface_b": interface_b,
                        "interface_global": (
                            interface_a["global"] + aligned_global_b
                        )
                        / 2.0,
                        "camera_distance": float(np.linalg.norm(camera_a - camera_b)),
                        "camera_to_interface_distance": camera_to_interface,
                    },
                )
    if best is None:
        raise ValueError("both panoramas must contain at least one door or opening")
    return best


def mine_zind_adjacent_pairs(
    zind_json_path: str, endpoint_tolerance: float = 0.06
) -> List[ZindAdjacentPair]:
    """Mine one best primary-panorama pair per pair of adjacent complete rooms."""
    if endpoint_tolerance <= 0:
        raise ValueError("endpoint_tolerance must be positive")
    json_path = Path(zind_json_path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    house_id = json_path.parent.name
    pairs: List[ZindAdjacentPair] = []
    for floor_id, floor_data in payload.get("merger", {}).items():
        rooms = _primary_panoramas(floor_data, json_path.parent)
        room_ids = sorted(rooms)
        for room_index, room_a in enumerate(room_ids):
            for room_b in room_ids[room_index + 1 :]:
                best = None
                for pano_a in rooms[room_a]:
                    for pano_b in rooms[room_b]:
                        candidate = _pair_candidate(pano_a, pano_b)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
                if best is None or best[0][0] > endpoint_tolerance:
                    continue
                key, match = best
                pano_a = match["pano_a"]
                pano_b = match["pano_b"]
                interface_a = match["interface_a"]
                interface_b = match["interface_b"]
                pair_id = "{}_{}_{}_{}_{}".format(
                    house_id,
                    floor_id,
                    pano_a["pano_id"],
                    pano_b["pano_id"],
                    len(pairs),
                )
                pairs.append(
                    ZindAdjacentPair(
                        pair_id=pair_id,
                        house_id=house_id,
                        floor_id=floor_id,
                        room_a=room_a,
                        room_b=room_b,
                        partial_room_a=pano_a["partial_room_id"],
                        partial_room_b=pano_b["partial_room_id"],
                        pano_a=pano_a["pano_id"],
                        pano_b=pano_b["pano_id"],
                        label_a=pano_a["label"],
                        label_b=pano_b["label"],
                        image_a=pano_a["image"],
                        image_b=pano_b["image"],
                        transform_a=pano_a["transform"],
                        transform_b=pano_b["transform"],
                        interface_type_a=interface_a["type"],
                        interface_type_b=interface_b["type"],
                        interface_index_a=interface_a["index"],
                        interface_index_b=interface_b["index"],
                        interface_local_a=interface_a["local"].tolist(),
                        interface_local_b=interface_b["local"].tolist(),
                        interface_global=match["interface_global"].tolist(),
                        interface_endpoint_error=key[0],
                        camera_distance=match["camera_distance"],
                        camera_to_interface_distance=match[
                            "camera_to_interface_distance"
                        ],
                        candidate_interfaces_a=tuple(pano_a["interfaces"]),
                        candidate_interfaces_b=tuple(pano_b["interfaces"]),
                    )
                )
    return sorted(
        pairs,
        key=lambda pair: (
            pair.interface_endpoint_error,
            pair.camera_to_interface_distance,
            pair.pair_id,
        ),
    )


def _negative_matching_record(
    house_id: str,
    floor_id: str,
    pano_a: Mapping[str, Any],
    pano_b: Mapping[str, Any],
    endpoint_error: float,
    token_count: int,
    data_root: Optional[str],
) -> Dict[str, Any]:
    root = None if data_root is None else Path(data_root).expanduser().resolve()
    matrix = relative_zind_transform(pano_a["transform"], pano_b["transform"])
    pair_id = "{}_{}_{}_{}_negative".format(
        house_id, floor_id, pano_a["pano_id"], pano_b["pano_id"]
    )
    return {
        "id": pair_id,
        "scene_id": house_id,
        "floor_id": floor_id,
        "image_a": _portable_path(pano_a["image"], root),
        "image_b": _portable_path(pano_b["image"], root),
        "room_a": pano_a["room_id"],
        "room_b": pano_b["room_id"],
        "partial_room_a": pano_a["partial_room_id"],
        "partial_room_b": pano_b["partial_room_id"],
        "pano_a": pano_a["pano_id"],
        "pano_b": pano_b["pano_id"],
        "room_label_a": pano_a["label"],
        "room_label_b": pano_b["label"],
        "token_count": int(token_count),
        "candidates_a": [
            _serialized_interface(candidate, token_count)
            for candidate in pano_a["interfaces"]
        ],
        "candidates_b": [
            _serialized_interface(candidate, token_count)
            for candidate in pano_b["interfaces"]
        ],
        "supervision": {
            "is_match": False,
            "target_candidate_a": -1,
            "target_candidate_b": -1,
            # Kept for diagnostics, but pose_valid prevents pose supervision on
            # pairs that do not have a shared interface.
            "relative_transform_b_to_a": matrix.tolist(),
            "relative_yaw_radians": float(math.atan2(matrix[1, 0], matrix[0, 0])),
            "pose_valid": False,
            "minimum_interface_endpoint_error": float(endpoint_error),
        },
    }


def mine_zind_matching_records(
    zind_json_path: str,
    endpoint_tolerance: float = 0.06,
    token_count: int = 256,
    negative_ratio: float = 1.0,
    negative_min_endpoint_error: Optional[float] = None,
    data_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build positive shared-interface pairs and deterministic same-floor negatives."""
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if negative_min_endpoint_error is None:
        negative_min_endpoint_error = max(0.15, endpoint_tolerance * 2.0)
    if negative_min_endpoint_error <= endpoint_tolerance:
        raise ValueError("negative_min_endpoint_error must exceed endpoint_tolerance")

    json_path = Path(zind_json_path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    positives = mine_zind_adjacent_pairs(
        str(json_path), endpoint_tolerance=endpoint_tolerance
    )
    records = [
        pair.to_matching_record(token_count=token_count, data_root=data_root)
        for pair in positives
    ]
    positives_by_floor: Dict[str, List[ZindAdjacentPair]] = {}
    for pair in positives:
        positives_by_floor.setdefault(pair.floor_id, []).append(pair)

    house_id = json_path.parent.name
    for floor_id, floor_data in payload.get("merger", {}).items():
        floor_positives = positives_by_floor.get(floor_id, [])
        requested_negatives = int(math.ceil(len(floor_positives) * negative_ratio))
        if requested_negatives == 0:
            continue
        positive_room_pairs = {
            tuple(sorted((pair.room_a, pair.room_b))) for pair in floor_positives
        }
        rooms = _primary_panoramas(floor_data, json_path.parent)
        room_ids = sorted(rooms)
        negative_candidates = []
        for room_index, room_a in enumerate(room_ids):
            for room_b in room_ids[room_index + 1 :]:
                if (room_a, room_b) in positive_room_pairs:
                    continue
                best = None
                for pano_a in rooms[room_a]:
                    for pano_b in rooms[room_b]:
                        candidate = _pair_candidate(pano_a, pano_b)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
                if best is None or best[0][0] < negative_min_endpoint_error:
                    continue
                key, match = best
                negative_candidates.append(
                    (
                        match["camera_distance"],
                        key[0],
                        match["pano_a"]["pano_id"],
                        match["pano_b"]["pano_id"],
                        match,
                    )
                )
        negative_candidates.sort(key=lambda value: value[:4])
        for _, endpoint_error, _, _, match in negative_candidates[:requested_negatives]:
            records.append(
                _negative_matching_record(
                    house_id,
                    floor_id,
                    match["pano_a"],
                    match["pano_b"],
                    endpoint_error,
                    token_count,
                    data_root,
                )
            )

    return sorted(
        records,
        key=lambda record: (
            record["floor_id"],
            not record["supervision"]["is_match"],
            record["id"],
        ),
    )

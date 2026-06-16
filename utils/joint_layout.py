"""
Shared-door alignment for two independently predicted room layouts.

The layout network currently predicts one polygon per panorama. This module
keeps that model unchanged and aligns two polygons through a doorway observed
from both rooms.
"""
from dataclasses import asdict, dataclass
from html import escape
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class DoorSpec:
    wall_index: int
    start_ratio: float
    end_ratio: float

    def __post_init__(self):
        if self.wall_index < 0:
            raise ValueError("wall_index must be non-negative")
        if not 0 <= self.start_ratio < self.end_ratio <= 1:
            raise ValueError("door ratios must satisfy 0 <= start < end <= 1")

    @classmethod
    def parse(cls, value: str) -> "DoorSpec":
        try:
            wall_index, start_ratio, end_ratio = value.split(":")
            return cls(int(wall_index), float(start_ratio), float(end_ratio))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "door spec must use wall_index:start_ratio:end_ratio, for example 1:0.25:0.55"
            ) from exc


def _as_xz(point: Dict) -> np.ndarray:
    xyz = np.asarray(point["xyz"], dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("layout point xyz must contain three finite values")
    return xyz[[0, 2]]


def layout_xz(layout: Dict) -> np.ndarray:
    points = layout.get("layoutPoints", {}).get("points", [])
    if len(points) < 3:
        raise ValueError("layout must contain at least three layout points")
    return np.asarray([_as_xz(point) for point in points], dtype=np.float64)


def _wall_indices(layout: Dict, wall_index: int) -> Tuple[int, int]:
    points = layout["layoutPoints"]["points"]
    walls = layout.get("layoutWalls", {}).get("walls", [])
    if walls:
        if wall_index >= len(walls):
            raise ValueError(f"wall_index {wall_index} is outside the {len(walls)} layout walls")
        indices = walls[wall_index]["pointsIdx"]
        if len(indices) != 2:
            raise ValueError("layout wall pointsIdx must contain exactly two point indices")
        start_index, end_index = indices
    else:
        if wall_index >= len(points):
            raise ValueError(f"wall_index {wall_index} is outside the {len(points)} layout walls")
        start_index, end_index = wall_index, (wall_index + 1) % len(points)

    if not 0 <= start_index < len(points) or not 0 <= end_index < len(points):
        raise ValueError("layout wall references an invalid layout point")
    return start_index, end_index


def door_endpoints(layout: Dict, spec: DoorSpec) -> np.ndarray:
    start_index, end_index = _wall_indices(layout, spec.wall_index)
    points = layout_xz(layout)
    wall_start = points[start_index]
    wall_vector = points[end_index] - wall_start
    if np.linalg.norm(wall_vector) < 1e-8:
        raise ValueError("door wall must have non-zero length")
    return np.asarray([
        wall_start + spec.start_ratio * wall_vector,
        wall_start + spec.end_ratio * wall_vector,
    ])


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :2]


def _fit_similarity(source: np.ndarray, target: np.ndarray, calibrate_scale: bool) -> Tuple[np.ndarray, float]:
    source_vector = source[1] - source[0]
    target_vector = target[1] - target[0]
    source_width = np.linalg.norm(source_vector)
    target_width = np.linalg.norm(target_vector)
    if source_width < 1e-8 or target_width < 1e-8:
        raise ValueError("shared door must have non-zero width in both rooms")

    scale = target_width / source_width if calibrate_scale else 1.0
    source_angle = np.arctan2(source_vector[1], source_vector[0])
    target_angle = np.arctan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation = np.asarray([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle],
    ])

    source_midpoint = source.mean(axis=0)
    target_midpoint = target.mean(axis=0)
    translation = target_midpoint - scale * rotation @ source_midpoint
    matrix = np.asarray([
        [scale * cos_angle, -scale * sin_angle, translation[0]],
        [scale * sin_angle, scale * cos_angle, translation[1]],
        [0.0, 0.0, 1.0],
    ])
    return matrix, float(scale)


def _boundary_lines(points: np.ndarray) -> List[Dict]:
    return [
        {
            "wallIndex": index,
            "start": points[index].tolist(),
            "end": points[(index + 1) % len(points)].tolist(),
        }
        for index in range(len(points))
    ]


def _room_output(room_id: str, layout: Dict, points: np.ndarray, camera_center: np.ndarray) -> Dict:
    return {
        "id": room_id,
        "layoutHeight": layout.get("layoutHeight"),
        "cameraHeight": layout.get("cameraHeight"),
        "cameraCenter": camera_center.tolist(),
        "boundary": points.tolist(),
        "boundaryLines": _boundary_lines(points),
    }


def build_joint_layout(layout_a: Dict, layout_b: Dict, door_a: DoorSpec, door_b: DoorSpec,
                       calibrate_scale: bool = True) -> Dict:
    points_a = layout_xz(layout_a)
    points_b = layout_xz(layout_b)
    endpoints_a = door_endpoints(layout_a, door_a)
    endpoints_b = door_endpoints(layout_b, door_b)

    centroid_a = points_a.mean(axis=0)
    door_vector_a = endpoints_a[1] - endpoints_a[0]
    side_a = _cross_2d(door_vector_a, centroid_a - endpoints_a.mean(axis=0))

    candidates = []
    for mapping, target in (("same", endpoints_a), ("reversed", endpoints_a[::-1])):
        matrix, scale = _fit_similarity(endpoints_b, target, calibrate_scale)
        transformed_points_b = _transform_points(points_b, matrix)
        transformed_endpoints_b = _transform_points(endpoints_b, matrix)
        centroid_b = transformed_points_b.mean(axis=0)
        side_b = _cross_2d(door_vector_a, centroid_b - endpoints_a.mean(axis=0))
        candidates.append({
            "mapping": mapping,
            "matrix": matrix,
            "scale": scale,
            "points_b": transformed_points_b,
            "endpoints_b": transformed_endpoints_b,
            "side_product": side_a * side_b,
        })

    # The shared wall is an interior interface, so the room centroids should be
    # on opposite sides of the doorway after alignment.
    best = min(candidates, key=lambda candidate: candidate["side_product"])
    camera_center_a = np.zeros(2, dtype=np.float64)
    camera_center_b = _transform_points(np.zeros((1, 2), dtype=np.float64), best["matrix"])[0]
    shared_door = endpoints_a

    return {
        "formatVersion": 1,
        "coordinateSystem": "floorplan-xz-meters",
        "rooms": [
            _room_output("A", layout_a, points_a, camera_center_a),
            _room_output("B", layout_b, best["points_b"], camera_center_b),
        ],
        "sharedDoor": {
            "roomA": {
                "spec": asdict(door_a),
                "localEndpoints": endpoints_a.tolist(),
                "worldEndpoints": endpoints_a.tolist(),
            },
            "roomB": {
                "spec": asdict(door_b),
                "localEndpoints": endpoints_b.tolist(),
                "worldEndpoints": best["endpoints_b"].tolist(),
            },
            "worldEndpoints": shared_door.tolist(),
            "width": float(np.linalg.norm(shared_door[1] - shared_door[0])),
        },
        "alignment": {
            "roomBToWorld": best["matrix"].tolist(),
            "roomBScale": best["scale"],
            "endpointMapping": best["mapping"],
            "centroidSideProduct": float(best["side_product"]),
            "scaleCalibratedFromSharedDoor": calibrate_scale,
        },
    }


def _joint_boundary_projector(joint_layout: Dict, side_length: int, padding: int):
    rooms = joint_layout["rooms"]
    room_points = [np.asarray(room["boundary"], dtype=np.float64) for room in rooms]
    door = np.asarray(joint_layout["sharedDoor"]["worldEndpoints"], dtype=np.float64)
    all_points = np.concatenate(room_points + [door], axis=0)

    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = (side_length - 2 * padding) / span.max()
    center = (minimum + maximum) / 2

    def to_pixel(points: np.ndarray) -> np.ndarray:
        pixels = (points - center) * scale
        pixels[:, 1] *= -1
        pixels += side_length / 2
        return np.rint(pixels).astype(np.int32)

    return rooms, room_points, door, to_pixel


def render_joint_boundary_svg(joint_layout: Dict, save_path: str, side_length: int = 900, padding: int = 70) -> None:
    rooms, room_points, door, to_pixel = _joint_boundary_projector(joint_layout, side_length, padding)
    colors = ["#1e6ecd", "#289128"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side_length}" height="{side_length}" '
        f'viewBox="0 0 {side_length} {side_length}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for room, points, color in zip(rooms, room_points, colors):
        pixels = to_pixel(points)
        polygon = " ".join(f"{x},{y}" for x, y in pixels)
        svg.append(f'<polygon points="{polygon}" fill="none" stroke="{color}" stroke-width="4"/>')
        room_center = to_pixel(np.asarray([points.mean(axis=0)]))[0]
        svg.append(
            f'<text x="{room_center[0]}" y="{room_center[1]}" fill="{color}" '
            f'font-size="24">Room {escape(str(room["id"]))}</text>'
        )
        for wall_index, wall_start in enumerate(pixels):
            wall_end = pixels[(wall_index + 1) % len(pixels)]
            wall_midpoint = (wall_start + wall_end) // 2
            svg.append(
                f'<text x="{wall_midpoint[0]}" y="{wall_midpoint[1]}" fill="{color}" '
                f'font-size="18">{escape(str(room["id"]))}:{wall_index}</text>'
            )
        camera = to_pixel(np.asarray([room["cameraCenter"]], dtype=np.float64))[0]
        svg.append(f'<circle cx="{camera[0]}" cy="{camera[1]}" r="7" fill="{color}"/>')

    door_pixels = to_pixel(door)
    svg.append(
        f'<line x1="{door_pixels[0, 0]}" y1="{door_pixels[0, 1]}" '
        f'x2="{door_pixels[1, 0]}" y2="{door_pixels[1, 1]}" stroke="#dc1e1e" stroke-width="8"/>'
    )
    door_midpoint = door_pixels.mean(axis=0).astype(np.int32)
    svg.append(
        f'<text x="{door_midpoint[0]}" y="{door_midpoint[1]}" fill="#dc1e1e" '
        f'font-size="20">shared door</text>'
    )
    svg.append("</svg>")
    with open(save_path, "w") as file:
        file.write("\n".join(svg) + "\n")


def render_joint_boundary(joint_layout: Dict, save_path: str, side_length: int = 900, padding: int = 70) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render the joint boundary PNG image") from exc

    rooms, room_points, door, to_pixel = _joint_boundary_projector(joint_layout, side_length, padding)
    canvas = np.full((side_length, side_length, 3), 255, dtype=np.uint8)
    colors = [(205, 110, 30), (40, 145, 40)]
    for room, points, color in zip(rooms, room_points, colors):
        pixels = to_pixel(points)
        cv2.polylines(canvas, [pixels], isClosed=True, color=color, thickness=4, lineType=cv2.LINE_AA)
        label_at = tuple(to_pixel(np.asarray([points.mean(axis=0)]))[0])
        cv2.putText(canvas, f"Room {room['id']}", label_at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        for wall_index, wall_start in enumerate(pixels):
            wall_end = pixels[(wall_index + 1) % len(pixels)]
            wall_midpoint = tuple(((wall_start + wall_end) // 2).tolist())
            cv2.putText(canvas, f"{room['id']}:{wall_index}", wall_midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        camera = to_pixel(np.asarray([room["cameraCenter"]], dtype=np.float64))[0]
        cv2.drawMarker(canvas, tuple(camera), color, cv2.MARKER_CROSS, 18, 3)

    door_pixels = to_pixel(door)
    cv2.line(canvas, tuple(door_pixels[0]), tuple(door_pixels[1]), (30, 30, 220), 8, cv2.LINE_AA)
    cv2.putText(canvas, "shared door", tuple(door_pixels.mean(axis=0).astype(np.int32)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 220), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, canvas)

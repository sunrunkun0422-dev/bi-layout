#!/usr/bin/env python3
"""Join two predicted room layouts through a shared opening/interface."""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cross_scene_estimator import polygon_validity
from utils.cross_scene_pipeline import atomic_write_json
from utils.joint_layout import DoorSpec, build_joint_layout, render_joint_boundary, render_joint_boundary_svg


def parse_option():
    parser = argparse.ArgumentParser(
        description="Align two room-layout predictions through a shared opening/interface and export boundary lines."
    )
    parser.add_argument("--layout_a", required=True, help="room A prediction JSON from inference.py")
    parser.add_argument("--layout_b", required=True, help="room B prediction JSON from inference.py")
    parser.add_argument(
        "--opening_a",
        "--door_a",
        required=True,
        type=DoorSpec.parse,
        dest="door_a",
        metavar="WALL:START:END",
        help="shared opening segment on room A, for example 1:0.25:0.55",
    )
    parser.add_argument(
        "--opening_b",
        "--door_b",
        required=True,
        type=DoorSpec.parse,
        dest="door_b",
        metavar="WALL:START:END",
        help="shared opening segment on room B, for example 3:0.30:0.60",
    )
    parser.add_argument("--output_dir", default="src/output/joint_layout", help="output directory")
    parser.add_argument("--name", default="a_b_joint", help="output file prefix")
    parser.add_argument(
        "--preserve_scale",
        action="store_true",
        help="keep each room prediction scale instead of calibrating room B with the shared opening width",
    )
    parser.add_argument(
        "--allow_invalid_polygon",
        action="store_true",
        help="continue with invalid polygon diagnostics instead of rejecting the input.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = parse_option()
    layout_a = load_json(args.layout_a)
    layout_b = load_json(args.layout_b)
    validity_a = polygon_validity(layout_a)
    validity_b = polygon_validity(layout_b)
    if not args.allow_invalid_polygon and not validity_a["valid"]:
        raise ValueError(f"layout A polygon is invalid: {validity_a}")
    if not args.allow_invalid_polygon and not validity_b["valid"]:
        raise ValueError(f"layout B polygon is invalid: {validity_b}")
    joint_layout = build_joint_layout(
        layout_a,
        layout_b,
        args.door_a,
        args.door_b,
        calibrate_scale=not args.preserve_scale,
    )
    joint_layout["diagnostics"] = {
        "layoutAValidity": validity_a,
        "layoutBValidity": validity_b,
        "selectionMethod": "manual_opening_specification",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.name}.json"
    svg_path = output_dir / f"{args.name}_boundaries.svg"
    image_path = output_dir / f"{args.name}_boundaries.png"
    atomic_write_json(str(json_path), joint_layout)
    print(f"joint layout: {json_path}")
    render_joint_boundary_svg(joint_layout, str(svg_path))
    print(f"boundary visualization: {svg_path}")
    try:
        render_joint_boundary(joint_layout, str(image_path))
        print(f"boundary visualization: {image_path}")
    except RuntimeError as exc:
        print(f"boundary visualization skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Join two predicted room layouts through a shared doorway."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.joint_layout import DoorSpec, build_joint_layout, render_joint_boundary, render_joint_boundary_svg


def parse_option():
    parser = argparse.ArgumentParser(
        description="Align two room-layout predictions through a shared door and export their boundary lines."
    )
    parser.add_argument("--layout_a", required=True, help="room A prediction JSON from inference.py")
    parser.add_argument("--layout_b", required=True, help="room B prediction JSON from inference.py")
    parser.add_argument(
        "--door_a",
        required=True,
        type=DoorSpec.parse,
        metavar="WALL:START:END",
        help="shared door segment on room A, for example 1:0.25:0.55",
    )
    parser.add_argument(
        "--door_b",
        required=True,
        type=DoorSpec.parse,
        metavar="WALL:START:END",
        help="shared door segment on room B, for example 3:0.30:0.60",
    )
    parser.add_argument("--output_dir", default="src/output/joint_layout", help="output directory")
    parser.add_argument("--name", default="a_b_joint", help="output file prefix")
    parser.add_argument(
        "--preserve_scale",
        action="store_true",
        help="keep each room prediction scale instead of calibrating room B with the shared door width",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def main():
    args = parse_option()
    layout_a = load_json(args.layout_a)
    layout_b = load_json(args.layout_b)
    joint_layout = build_joint_layout(
        layout_a,
        layout_b,
        args.door_a,
        args.door_b,
        calibrate_scale=not args.preserve_scale,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.name}.json")
    svg_path = os.path.join(args.output_dir, f"{args.name}_boundaries.svg")
    image_path = os.path.join(args.output_dir, f"{args.name}_boundaries.png")
    with open(json_path, "w") as file:
        json.dump(joint_layout, file, indent=4)
        file.write("\n")
    print(f"joint layout: {json_path}")
    render_joint_boundary_svg(joint_layout, svg_path)
    print(f"boundary visualization: {svg_path}")
    try:
        render_joint_boundary(joint_layout, image_path)
        print(f"boundary visualization: {image_path}")
    except RuntimeError as exc:
        print(f"boundary visualization skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

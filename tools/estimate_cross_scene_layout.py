#!/usr/bin/env python3
"""Estimate cross-scene alignment candidates from two room-layout JSON files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cross_scene_estimator import (
    estimate_wall_pair_candidates,
    extract_opening_candidates,
    simplify_layout_for_estimation,
)
from utils.joint_layout import render_joint_boundary, render_joint_boundary_svg


def parse_option() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank possible cross-scene shared-opening alignments for two predicted room layouts."
    )
    parser.add_argument("--layout_a", required=True, help="room A prediction JSON from inference.py")
    parser.add_argument("--layout_b", required=True, help="room B prediction JSON from inference.py")
    parser.add_argument(
        "--layout_a_extended",
        help="optional extended/new-branch layout JSON for room A; used to infer passable openings.",
    )
    parser.add_argument(
        "--layout_b_extended",
        help="optional extended/new-branch layout JSON for room B; used to infer passable openings.",
    )
    parser.add_argument("--output_dir", default="src/output/cross_scene_estimation")
    parser.add_argument("--name", default="a_b_cross_scene")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument(
        "--anchor_ratio",
        type=float,
        default=0.3,
        help="centered ratio of each wall used as the possible shared opening/interface.",
    )
    parser.add_argument(
        "--preserve_scale",
        action="store_true",
        help="keep each room prediction scale instead of calibrating room B from the shared opening/interface.",
    )
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=0.05,
        help="polygon simplification tolerance used when layout JSON contains dense boundary samples.",
    )
    parser.add_argument(
        "--max_walls",
        type=int,
        default=64,
        help="simplify layouts with more than this number of wall segments before estimation.",
    )
    parser.add_argument(
        "--disable_passability",
        action="store_true",
        help="disable extended-minus-enclosed opening proposals and use wall-center fallback only.",
    )
    parser.add_argument(
        "--opening_threshold",
        type=float,
        default=0.25,
        help="normalized threshold for extended-minus-enclosed passability heat.",
    )
    parser.add_argument(
        "--min_opening_width_tokens",
        type=int,
        default=3,
        help="minimum contiguous token count for a passable opening candidate.",
    )
    parser.add_argument(
        "--max_openings_per_layout",
        type=int,
        default=12,
        help="maximum passable opening candidates kept per layout.",
    )
    parser.add_argument(
        "--passability_weight",
        type=float,
        default=1.0,
        help="reward weight for passable-opening confidence in candidate ranking.",
    )
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r") as file:
        return json.load(file)


def main() -> None:
    args = parse_option()
    raw_layout_a = load_json(args.layout_a)
    raw_layout_b = load_json(args.layout_b)
    raw_layout_a_extended = load_json(args.layout_a_extended) if args.layout_a_extended else None
    raw_layout_b_extended = load_json(args.layout_b_extended) if args.layout_b_extended else None
    layout_a, simplify_a = simplify_layout_for_estimation(
        raw_layout_a,
        tolerance=args.simplify_tolerance,
        max_walls=args.max_walls,
    )
    layout_b, simplify_b = simplify_layout_for_estimation(
        raw_layout_b,
        tolerance=args.simplify_tolerance,
        max_walls=args.max_walls,
    )
    layout_a_extended, simplify_a_extended = (None, None)
    layout_b_extended, simplify_b_extended = (None, None)
    if raw_layout_a_extended is not None:
        layout_a_extended, simplify_a_extended = simplify_layout_for_estimation(
            raw_layout_a_extended,
            tolerance=args.simplify_tolerance,
            max_walls=args.max_walls,
        )
    if raw_layout_b_extended is not None:
        layout_b_extended, simplify_b_extended = simplify_layout_for_estimation(
            raw_layout_b_extended,
            tolerance=args.simplify_tolerance,
            max_walls=args.max_walls,
        )

    opening_summary = None
    openings_a = None
    openings_b = None
    if not args.disable_passability:
        openings_a, opening_summary_a = extract_opening_candidates(
            layout_a,
            extended_layout=layout_a_extended,
            threshold=args.opening_threshold,
            min_width_tokens=args.min_opening_width_tokens,
            max_candidates=args.max_openings_per_layout,
            fallback_anchor_ratio=args.anchor_ratio,
        )
        openings_b, opening_summary_b = extract_opening_candidates(
            layout_b,
            extended_layout=layout_b_extended,
            threshold=args.opening_threshold,
            min_width_tokens=args.min_opening_width_tokens,
            max_candidates=args.max_openings_per_layout,
            fallback_anchor_ratio=args.anchor_ratio,
        )
        opening_summary = {
            "A": opening_summary_a,
            "B": opening_summary_b,
            "candidatesA": [candidate.to_json() for candidate in openings_a],
            "candidatesB": [candidate.to_json() for candidate in openings_b],
        }

    candidates, best_joint_layout = estimate_wall_pair_candidates(
        layout_a,
        layout_b,
        anchor_ratio=args.anchor_ratio,
        top_k=args.top_k,
        calibrate_scale=not args.preserve_scale,
        openings_a=openings_a,
        openings_b=openings_b,
        passability_weight=0.0 if args.disable_passability else args.passability_weight,
    )
    if not candidates:
        raise RuntimeError("No valid wall-pair alignment candidates were found.")

    os.makedirs(args.output_dir, exist_ok=True)
    candidates_path = os.path.join(args.output_dir, f"{args.name}_candidates.json")
    joint_path = os.path.join(args.output_dir, f"{args.name}_best_joint.json")
    svg_path = os.path.join(args.output_dir, f"{args.name}_best_joint.svg")
    image_path = os.path.join(args.output_dir, f"{args.name}_best_joint.png")

    output = {
        "formatVersion": 1,
        "method": "passability_geometry_opening_search" if not args.disable_passability else "geometry_wall_pair_search",
        "layoutA": args.layout_a,
        "layoutB": args.layout_b,
        "layoutAExtended": args.layout_a_extended,
        "layoutBExtended": args.layout_b_extended,
        "layoutSimplification": {
            "A": simplify_a,
            "B": simplify_b,
            "AExtended": simplify_a_extended,
            "BExtended": simplify_b_extended,
        },
        "anchorRatio": args.anchor_ratio,
        "scaleCalibratedFromSharedOpening": not args.preserve_scale,
        "scaleCalibratedFromSharedInterface": not args.preserve_scale,
        "passability": {
            "enabled": not args.disable_passability,
            "openingThreshold": args.opening_threshold,
            "minOpeningWidthTokens": args.min_opening_width_tokens,
            "maxOpeningsPerLayout": args.max_openings_per_layout,
            "passabilityWeight": args.passability_weight,
            "summary": opening_summary,
        },
        "topK": args.top_k,
        "candidates": [candidate.to_json() for candidate in candidates],
    }
    with open(candidates_path, "w") as file:
        json.dump(output, file, indent=4)
        file.write("\n")
    with open(joint_path, "w") as file:
        json.dump(best_joint_layout, file, indent=4)
        file.write("\n")

    render_joint_boundary_svg(best_joint_layout, svg_path)
    try:
        render_joint_boundary(best_joint_layout, image_path)
        rendered = image_path
    except RuntimeError:
        rendered = svg_path

    best = candidates[0]
    print(f"candidates: {candidates_path}")
    print(f"best joint layout: {joint_path}")
    print(f"best visualization: {rendered}")
    print(
        "best candidate: "
        f"A wall {best.wall_a} <-> B wall {best.wall_b}, "
        f"score={best.score:.4f}, confidence={best.confidence:.4f}"
    )


if __name__ == "__main__":
    main()

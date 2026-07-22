#!/usr/bin/env python3
"""Estimate cross-scene alignment candidates from two room-layout JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cross_scene_logger import CrossSceneExperimentLogger
from utils.cross_scene_pipeline import (
    CrossScenePipeline,
    CrossScenePipelineConfig,
    atomic_write_json,
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
    parser.add_argument(
        "--match_evidence",
        help="optional matcher output in JSON, NPZ, PT, or PTH format containing Aff_AB/S_A/S_B.",
    )
    parser.add_argument(
        "--feature_weight",
        type=float,
        default=1.0,
        help="reward weight for cross-attention feature evidence.",
    )
    parser.add_argument(
        "--nms_overlap_threshold",
        type=float,
        default=0.8,
        help="opening-pair interval IoU threshold used by candidate NMS.",
    )
    parser.add_argument(
        "--confidence_temperature",
        type=float,
        default=1.0,
        help=(
            "temperature used to normalize relative candidate scores; the output "
            "is not a calibrated probability."
        ),
    )
    parser.add_argument(
        "--allow_invalid_polygon",
        action="store_true",
        help="continue with invalid polygon diagnostics instead of rejecting the pair.",
    )
    parser.add_argument(
        "--selector_checkpoint",
        help="optional GeometryConsistencySelector checkpoint used to rerank top-k candidates.",
    )
    parser.add_argument("--selector_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--experiment_log",
        help="JSONL run log path; defaults to <output_dir>/cross_scene_runs.jsonl.",
    )
    parser.add_argument("--ground_truth", help="optional pair ground-truth JSON for evaluation logging.")
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_match_evidence(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    if suffix in (".pt", ".pth"):
        import torch

        payload = torch.load(path, map_location="cpu")
    else:
        payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("matches"), dict):
        return payload["matches"]
    if not isinstance(payload, dict):
        raise ValueError("match evidence must contain a dictionary")
    return payload


def load_selector(path: str, hidden_dim: int):
    import torch

    from models.geometry_consistency_selector import GeometryConsistencySelector

    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    selector = GeometryConsistencySelector(hidden_dim=hidden_dim)
    selector.load_state_dict(state_dict)
    return selector


def main() -> None:
    args = parse_option()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_log = args.experiment_log or str(output_dir / "cross_scene_runs.jsonl")
    experiment_logger = CrossSceneExperimentLogger(experiment_log)

    raw_layout_a = load_json(args.layout_a)
    raw_layout_b = load_json(args.layout_b)
    raw_layout_a_extended = load_json(args.layout_a_extended) if args.layout_a_extended else None
    raw_layout_b_extended = load_json(args.layout_b_extended) if args.layout_b_extended else None
    match_evidence = load_match_evidence(args.match_evidence) if args.match_evidence else None
    selector = (
        load_selector(args.selector_checkpoint, args.selector_hidden_dim)
        if args.selector_checkpoint else None
    )
    config = CrossScenePipelineConfig(
        anchor_ratio=args.anchor_ratio,
        top_k=args.top_k,
        calibrate_scale=not args.preserve_scale,
        simplify_tolerance=args.simplify_tolerance,
        max_walls=args.max_walls,
        use_passability=not args.disable_passability,
        opening_threshold=args.opening_threshold,
        min_opening_width_tokens=args.min_opening_width_tokens,
        max_openings_per_layout=args.max_openings_per_layout,
        passability_weight=args.passability_weight,
        feature_weight=args.feature_weight,
        nms_overlap_threshold=args.nms_overlap_threshold,
        confidence_temperature=args.confidence_temperature,
        strict_polygon_validation=not args.allow_invalid_polygon,
    )
    pipeline = CrossScenePipeline(config, selector=selector)
    try:
        result = pipeline.run(
            raw_layout_a,
            raw_layout_b,
            extended_layout_a=raw_layout_a_extended,
            extended_layout_b=raw_layout_b_extended,
            match_evidence=match_evidence,
        )
    except Exception as exc:
        experiment_logger.append({
            "pairId": args.name,
            "status": "failed",
            "error": str(exc),
            "layoutA": args.layout_a,
            "layoutB": args.layout_b,
        })
        raise

    candidates_path = output_dir / f"{args.name}_candidates.json"
    joint_path = output_dir / f"{args.name}_best_joint.json"
    svg_path = output_dir / f"{args.name}_best_joint.svg"
    image_path = output_dir / f"{args.name}_best_joint.png"
    metadata = {
        "pairId": args.name,
        "layoutA": args.layout_a,
        "layoutB": args.layout_b,
        "layoutAExtended": args.layout_a_extended,
        "layoutBExtended": args.layout_b_extended,
        "matchEvidence": args.match_evidence,
        "selectorCheckpoint": args.selector_checkpoint,
    }
    atomic_write_json(str(candidates_path), result.candidates_json(metadata))
    atomic_write_json(str(joint_path), result.best_joint_layout)

    render_joint_boundary_svg(result.best_joint_layout, str(svg_path))
    try:
        render_joint_boundary(result.best_joint_layout, str(image_path))
        rendered = str(image_path)
    except RuntimeError:
        rendered = str(svg_path)

    ground_truth = load_json(args.ground_truth) if args.ground_truth else None
    experiment_logger.log_result(
        args.name,
        result.candidates,
        result.best_joint_layout,
        ground_truth=ground_truth,
        extra={
            "method": result.method,
            "candidatesPath": str(candidates_path),
            "jointPath": str(joint_path),
        },
    )

    best = result.candidates[0]
    print(f"candidates: {candidates_path}")
    print(f"best joint layout: {joint_path}")
    print(f"best visualization: {rendered}")
    print(f"experiment log: {experiment_log}")
    print(
        "best candidate: "
        f"A wall {best.wall_a} <-> B wall {best.wall_b}, "
        f"score={best.score:.4f}, confidence={best.confidence:.4f}"
    )


if __name__ == "__main__":
    main()

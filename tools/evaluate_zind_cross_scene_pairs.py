#!/usr/bin/env python3
"""Evaluate the dual-panorama pipeline on ZInD labeled adjacent-room pairs."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.panorama_pair_dataset import build_pair_dataloader
from dataset.zind_pair_mining import ZindAdjacentPair, mine_zind_adjacent_pairs
from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    candidate_intervals_to_mask,
    resolve_enclosed_extended_depth,
)
from tools.debug_cross_scene_flow import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    _top_intervals,
    load_bi_layout,
    load_module_checkpoint,
    prediction_to_layout,
)
from utils.cross_scene_estimator import (
    opening_candidates_from_intervals,
    wall_token_assignment,
)
from utils.cross_scene_pipeline import (
    CrossScenePipeline,
    CrossScenePipelineConfig,
    atomic_write_json,
)
from utils.joint_layout import render_joint_boundary, render_joint_boundary_svg
from utils.opening_checkpoint import (
    DEFAULT_OPENING_PROBABILITY_THRESHOLD,
    load_opening_head_checkpoint,
    resolve_opening_probability_threshold,
)
from utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint


DEFAULT_ZIND_ROOT = REPO_ROOT.parent / "zind/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine labeled adjacent ZInD rooms and evaluate shared-wall selection, "
            "relative pose, and forward/reverse consistency."
        )
    )
    parser.add_argument("--zind_root", default=str(DEFAULT_ZIND_ROOT))
    parser.add_argument("--house_ids", help="comma-separated house ids; default scans in order")
    parser.add_argument("--house_count", type=int, default=3)
    parser.add_argument("--pairs_per_house", type=int, default=3)
    parser.add_argument("--max_pairs", type=int, default=8)
    parser.add_argument("--endpoint_tolerance", type=float, default=0.06)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--matcher_checkpoint",
        help=(
            "optional trained OpeningGuidedCrossAttentionMatcher checkpoint; "
            "without it, only geometry variants are scored"
        ),
    )
    parser.add_argument(
        "--opening_checkpoint",
        help=(
            "optional trained OpeningSignalHead checkpoint; loaded after "
            "--matcher_checkpoint and therefore overrides its opening head"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument(
        "--opening_probability_threshold",
        type=float,
        default=None,
        help=(
            "explicit opening candidate threshold; otherwise use the value saved "
            "by --opening_checkpoint, then --matcher_checkpoint, or the legacy "
            "0.12 default"
        ),
    )
    parser.add_argument(
        "--branch_order",
        choices=("extended_first", "enclosed_first"),
        default="extended_first",
        help="semantic order of the two Bi-Layout depth branches",
    )
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument(
        "--output_dir", default="src/output/zind_cross_scene_ground_truth_eval"
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.house_count,
        args.pairs_per_house,
        args.max_pairs,
        args.top_k,
        args.torch_threads,
        args.min_opening_width_tokens,
        args.max_openings_per_view,
    )
    if min(positive) <= 0:
        raise ValueError("house and pair counts, top_k, and torch_threads must be positive")
    if args.endpoint_tolerance <= 0:
        raise ValueError("endpoint_tolerance must be positive")
    if (
        args.opening_probability_threshold is not None
        and not 0.0 <= args.opening_probability_threshold <= 1.0
    ):
        raise ValueError("opening_probability_threshold must be in [0, 1]")


def select_pairs(args: argparse.Namespace) -> List[ZindAdjacentPair]:
    root = Path(args.zind_root).expanduser().resolve()
    if args.house_ids:
        house_ids = [item.strip() for item in args.house_ids.split(",") if item.strip()]
        json_paths = [root / house_id / "zind_data.json" for house_id in house_ids]
    else:
        json_paths = sorted(root.glob("*/zind_data.json"))

    selected: List[ZindAdjacentPair] = []
    selected_house_count = 0
    for json_path in json_paths:
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
        house_pairs = mine_zind_adjacent_pairs(
            str(json_path), endpoint_tolerance=args.endpoint_tolerance
        )
        if not house_pairs:
            continue
        selected.extend(house_pairs[: args.pairs_per_house])
        selected_house_count += 1
        if selected_house_count >= args.house_count or len(selected) >= args.max_pairs:
            break
    return selected[: args.max_pairs]


def _wrapped_angle_error(first: float, second: float) -> float:
    return float(abs((first - second + math.pi) % (2.0 * math.pi) - math.pi))


def _pose(matrix: Sequence[Sequence[float]]) -> Dict[str, Any]:
    array = np.asarray(matrix, dtype=np.float64)
    scale = float(math.hypot(array[0, 0], array[1, 0]))
    return {
        "matrix": array.tolist(),
        "yawRadians": float(math.atan2(array[1, 0], array[0, 0])),
        "translation": array[:2, 2].tolist(),
        "scale": scale,
    }


def _opening_token(local_segment: Sequence[Sequence[float]], token_count: int) -> int:
    midpoint = np.asarray(local_segment, dtype=np.float64).mean(axis=0)
    longitude = math.atan2(-float(midpoint[0]), float(midpoint[1]))
    continuous = (longitude / (2.0 * math.pi) + 0.5) * token_count - 0.5
    return int(round(continuous)) % token_count


def _target_wall(layout: Mapping[str, Any], local_segment, token_count: int) -> Dict[str, int]:
    token = _opening_token(local_segment, token_count)
    wall_ids, _ = wall_token_assignment(layout, sample_count=token_count)
    return {"token": token, "wall": int(wall_ids[token])}


def _candidate_evaluation(result, target_a: Mapping[str, int], target_b: Mapping[str, int]):
    candidates = result.candidates
    passability_enabled = bool(result.opening_summary["enabled"])
    matches = [
        candidate.wall_a == target_a["wall"] and candidate.wall_b == target_b["wall"]
        for candidate in candidates
    ]
    selected = candidates[0]
    predicted_pose = _pose(result.best_joint_layout["alignment"]["roomBToWorld"])
    return {
        "candidateCount": len(candidates),
        "targetWallA": target_a["wall"],
        "targetWallB": target_b["wall"],
        "selectedWallA": selected.wall_a,
        "selectedWallB": selected.wall_b,
        "openingTop1Correct": bool(matches and matches[0]),
        "openingTopKRecall": bool(any(matches)),
        "bestConfidence": float(selected.confidence),
        "predictedPose": predicted_pose,
        "passabilitySourceA": result.opening_summary["A"]["source"],
        "passabilitySourceB": result.opening_summary["B"]["source"],
        "openingWallRecallA": not passability_enabled
        or any(
            item["wallIndex"] == target_a["wall"]
            for item in result.opening_summary["candidatesA"]
        ),
        "openingWallRecallB": not passability_enabled
        or any(
            item["wallIndex"] == target_b["wall"]
            for item in result.opening_summary["candidatesB"]
        ),
    }


def _add_pose_errors(evaluation: Dict[str, Any], target_pose: Mapping[str, Any]) -> None:
    prediction = evaluation["predictedPose"]
    evaluation["yawErrorDegrees"] = math.degrees(
        _wrapped_angle_error(prediction["yawRadians"], target_pose["yawRadians"])
    )
    evaluation["translationError"] = float(
        np.linalg.norm(
            np.asarray(prediction["translation"], dtype=np.float64)
            - np.asarray(target_pose["translation"], dtype=np.float64)
        )
    )
    evaluation["absoluteLogScaleError"] = float(
        abs(math.log(max(prediction["scale"], 1e-8) / max(target_pose["scale"], 1e-8)))
    )


def _mean(records: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def _variant_summary(records: List[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    values = [record["variants"][name] for record in records if name in record.get("variants", {})]
    yaw_errors = [float(value["yawErrorDegrees"]) for value in values]
    return {
        "evaluatedPairCount": len(values),
        "openingTop1Accuracy": _mean(values, "openingTop1Correct"),
        "openingTopKRecall": _mean(values, "openingTopKRecall"),
        "meanYawErrorDegrees": _mean(values, "yawErrorDegrees"),
        "medianYawErrorDegrees": float(median(yaw_errors)) if yaw_errors else 0.0,
        "meanTranslationError": _mean(values, "translationError"),
        "meanAbsoluteLogScaleError": _mean(values, "absoluteLogScaleError"),
        "meanCandidateCount": _mean(values, "candidateCount"),
        "openingWallRecallA": _mean(values, "openingWallRecallA"),
        "openingWallRecallB": _mean(values, "openingWallRecallB"),
        "bothSidesWithoutFallback": float(
            np.mean(
                [
                    "fallback" not in value["passabilitySourceA"]
                    and "fallback" not in value["passabilitySourceB"]
                    for value in values
                ]
            )
        )
        if values
        else 0.0,
    }


def _cycle_metrics(forward: Mapping[str, Any], reverse: Mapping[str, Any]) -> Dict[str, Any]:
    matrix_forward = np.asarray(forward["predictedPose"]["matrix"], dtype=np.float64)
    matrix_reverse = np.asarray(reverse["predictedPose"]["matrix"], dtype=np.float64)
    cycle = matrix_forward @ matrix_reverse
    cycle_pose = _pose(cycle)
    return {
        "wallSwapConsistent": bool(
            forward["selectedWallA"] == reverse["selectedWallB"]
            and forward["selectedWallB"] == reverse["selectedWallA"]
        ),
        "poseCycleYawDegrees": math.degrees(abs(cycle_pose["yawRadians"])),
        "poseCycleTranslation": float(np.linalg.norm(cycle_pose["translation"])),
        "poseCycleAbsoluteLogScale": abs(math.log(max(cycle_pose["scale"], 1e-8))),
    }


def _render_pair(output_dir: Path, pair_id: str, joint_layout: Mapping[str, Any]) -> None:
    svg_path = output_dir / "visualizations" / f"{pair_id}.svg"
    png_path = output_dir / "visualizations" / f"{pair_id}.png"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    render_joint_boundary_svg(
        dict(joint_layout), str(svg_path), show_wall_indices=False
    )
    render_joint_boundary(dict(joint_layout), str(png_path), show_wall_indices=False)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_pairs(args)
    if not pairs:
        raise RuntimeError("no labeled adjacent panorama pairs were found")
    pair_lookup = {pair.pair_id: pair for pair in pairs}
    manifest = {
        "formatVersion": 1,
        "source": "ZInD shared door/opening endpoint matching",
        "endpointTolerance": args.endpoint_tolerance,
        "pairs": [pair.to_manifest_record() for pair in pairs],
    }
    manifest_path = output_dir / "pair_manifest.json"
    atomic_write_json(str(manifest_path), manifest)

    model, checkpoint_report = load_bi_layout(
        args.config, args.checkpoint, device, load_checkpoint=True
    )
    torch.manual_seed(args.seed)
    matcher_report = {"path": None, "loaded": False}
    if args.matcher_checkpoint:
        matcher, matcher_report = load_cross_scene_matcher_checkpoint(
            args.matcher_checkpoint,
            expected_feature_dim=int(model.patch_dim),
            expected_token_count=int(model.patch_num),
            expected_branch_order=args.branch_order,
            bi_layout_config_path=args.config,
            bi_layout_checkpoint_path=args.checkpoint,
            device=device,
        )
    else:
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=model.patch_dim
        ).to(device)
    opening_report = {"path": None, "loaded": False}
    if args.opening_checkpoint:
        opening_report = load_opening_head_checkpoint(
            matcher.opening_head,
            args.opening_checkpoint,
            expected_feature_dim=int(model.patch_dim),
            expected_branch_order=args.branch_order,
            expected_token_count=int(model.patch_num),
            bi_layout_config_path=args.config,
            bi_layout_checkpoint_path=args.checkpoint,
        )
        opening_report["loadOrder"] = (
            "matcher_checkpoint_then_opening_override"
            if matcher_report["loaded"]
            else "opening_checkpoint_only"
        )
    opening_threshold = resolve_opening_probability_threshold(
        args.opening_probability_threshold,
        opening_report,
        matcher_report,
        fallback=DEFAULT_OPENING_PROBABILITY_THRESHOLD,
    )
    learned_opening_weights_loaded = bool(
        opening_report["loaded"] or matcher_report["loaded"]
    )
    matcher.eval()
    geometry_pipeline = CrossScenePipeline(
        CrossScenePipelineConfig(top_k=args.top_k, feature_weight=0.0)
    )
    cross_pipeline = (
        CrossScenePipeline(
            CrossScenePipelineConfig(top_k=args.top_k, feature_weight=1.0)
        )
        if matcher_report["loaded"]
        else None
    )
    all_walls_geometry_pipeline = CrossScenePipeline(
        CrossScenePipelineConfig(
            top_k=args.top_k, use_passability=False, feature_weight=0.0
        )
    )
    all_walls_cross_pipeline = (
        CrossScenePipeline(
            CrossScenePipelineConfig(
                top_k=args.top_k, use_passability=False, feature_weight=1.0
            )
        )
        if matcher_report["loaded"]
        else None
    )
    loader = build_pair_dataloader(str(manifest_path), batch_size=1, workers=0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records = []
    start_time = time.perf_counter()
    for pair_index, batch in enumerate(loader, start=1):
        pair_id = batch["pair_id"][0]
        pair = pair_lookup[pair_id]
        pair_start = time.perf_counter()
        print(f"[{pair_index}/{len(pairs)}] {pair_id}: {pair.label_a} <-> {pair.label_b}")
        record: Dict[str, Any] = {
            "pairId": pair_id,
            "houseId": pair.house_id,
            "floorId": pair.floor_id,
            "labels": [pair.label_a, pair.label_b],
            "interfaceEndpointError": pair.interface_endpoint_error,
            "cameraDistance": pair.camera_distance,
        }
        try:
            image_a = batch["image_A"].to(device)
            image_b = batch["image_B"].to(device)
            with torch.no_grad():
                output_a = model(image_a, return_features=True)
                output_b = model(image_b, return_features=True)
                enclosed_a, extended_a = resolve_enclosed_extended_depth(
                    output_a, args.branch_order
                )
                enclosed_b, extended_b = resolve_enclosed_extended_depth(
                    output_b, args.branch_order
                )
                unmasked = matcher(
                    output_a["layout_feature"],
                    output_b["layout_feature"],
                    enclosed_a,
                    extended_a,
                    enclosed_b,
                    extended_b,
                )
                unmasked_reverse = matcher(
                    output_b["layout_feature"],
                    output_a["layout_feature"],
                    enclosed_b,
                    extended_b,
                    enclosed_a,
                    extended_a,
                )

            intervals_a = _top_intervals(
                unmasked["P_A_open"][0],
                count=args.max_openings_per_view,
                threshold=opening_threshold["value"],
                min_width_tokens=args.min_opening_width_tokens,
                ensure_non_empty=False,
            )
            intervals_b = _top_intervals(
                unmasked["P_B_open"][0],
                count=args.max_openings_per_view,
                threshold=opening_threshold["value"],
                min_width_tokens=args.min_opening_width_tokens,
                ensure_non_empty=False,
            )
            probability_peak_fallback_a = not intervals_a
            probability_peak_fallback_b = not intervals_b
            if probability_peak_fallback_a:
                intervals_a = _top_intervals(
                    unmasked["P_A_open"][0],
                    count=args.max_openings_per_view,
                    threshold=opening_threshold["value"],
                    min_width_tokens=args.min_opening_width_tokens,
                )
            if probability_peak_fallback_b:
                intervals_b = _top_intervals(
                    unmasked["P_B_open"][0],
                    count=args.max_openings_per_view,
                    threshold=opening_threshold["value"],
                    min_width_tokens=args.min_opening_width_tokens,
                )
            masks_a = candidate_intervals_to_mask(
                intervals_a, model.patch_num, device=device
            )
            masks_b = candidate_intervals_to_mask(
                intervals_b, model.patch_num, device=device
            )
            with torch.no_grad():
                matches = matcher(
                    output_a["layout_feature"],
                    output_b["layout_feature"],
                    enclosed_a,
                    extended_a,
                    enclosed_b,
                    extended_b,
                    candidate_masks_a=masks_a,
                    candidate_masks_b=masks_b,
                )
                reverse_matches = matcher(
                    output_b["layout_feature"],
                    output_a["layout_feature"],
                    enclosed_b,
                    extended_b,
                    enclosed_a,
                    extended_a,
                    candidate_masks_a=masks_b,
                    candidate_masks_b=masks_a,
                )

            layout_a, report_a = prediction_to_layout(enclosed_a, output_a["ratio"])
            layout_b, report_b = prediction_to_layout(enclosed_b, output_b["ratio"])
            extended_layout_a, _ = prediction_to_layout(extended_a, output_a["ratio"])
            extended_layout_b, _ = prediction_to_layout(extended_b, output_b["ratio"])
            token_count = int(enclosed_a.shape[-1])
            target_a = _target_wall(layout_a, pair.interface_local_a, token_count)
            target_b = _target_wall(layout_b, pair.interface_local_b, token_count)
            target_pose = _pose(pair.ground_truth_transform)
            learned_openings_a = None
            learned_openings_b = None
            if learned_opening_weights_loaded:
                learned_openings_a = opening_candidates_from_intervals(
                    layout_a,
                    intervals_a,
                    unmasked["P_A_open"][0],
                )
                learned_openings_b = opening_candidates_from_intervals(
                    layout_b,
                    intervals_b,
                    unmasked["P_B_open"][0],
                )

            geometry_result = geometry_pipeline.run(
                layout_a,
                layout_b,
                extended_layout_a=extended_layout_a,
                extended_layout_b=extended_layout_b,
                openings_a=learned_openings_a,
                openings_b=learned_openings_b,
            )
            all_walls_geometry_result = all_walls_geometry_pipeline.run(
                layout_a,
                layout_b,
                extended_layout_a=extended_layout_a,
                extended_layout_b=extended_layout_b,
            )
            geometry_eval = _candidate_evaluation(geometry_result, target_a, target_b)
            all_walls_geometry_eval = _candidate_evaluation(
                all_walls_geometry_result, target_a, target_b
            )
            _add_pose_errors(geometry_eval, target_pose)
            _add_pose_errors(all_walls_geometry_eval, target_pose)

            variants = {
                "geometryOpening": geometry_eval,
                "allWallsGeometry": all_walls_geometry_eval,
            }
            selected_result = geometry_result
            forward_reverse = None
            if matcher_report["loaded"]:
                cross_result = cross_pipeline.run(
                    layout_a,
                    layout_b,
                    extended_layout_a=extended_layout_a,
                    extended_layout_b=extended_layout_b,
                    match_evidence=matches,
                    openings_a=learned_openings_a,
                    openings_b=learned_openings_b,
                )
                reverse_result = cross_pipeline.run(
                    layout_b,
                    layout_a,
                    extended_layout_a=extended_layout_b,
                    extended_layout_b=extended_layout_a,
                    match_evidence=reverse_matches,
                    openings_a=learned_openings_b,
                    openings_b=learned_openings_a,
                )
                all_walls_cross_result = all_walls_cross_pipeline.run(
                    layout_a,
                    layout_b,
                    extended_layout_a=extended_layout_a,
                    extended_layout_b=extended_layout_b,
                    match_evidence=matches,
                )
                cross_eval = _candidate_evaluation(cross_result, target_a, target_b)
                reverse_eval = _candidate_evaluation(reverse_result, target_b, target_a)
                all_walls_cross_eval = _candidate_evaluation(
                    all_walls_cross_result, target_a, target_b
                )
                _add_pose_errors(cross_eval, target_pose)
                _add_pose_errors(all_walls_cross_eval, target_pose)
                variants.update(
                    crossAttention=cross_eval,
                    allWallsCrossAttention=all_walls_cross_eval,
                )
                forward_reverse = _cycle_metrics(cross_eval, reverse_eval)
                selected_result = cross_result

            forward_token_shift = float(
                unmasked["relative_token_shift_radians"].item()
            )
            reverse_token_shift = float(
                unmasked_reverse["relative_token_shift_radians"].item()
            )
            token_count = int(unmasked["P_A_open"].shape[-1])
            target_token_shift = (
                int(target_b["token"]) - int(target_a["token"])
            ) * (2.0 * math.pi / token_count)
            target_token_shift = math.atan2(
                math.sin(target_token_shift), math.cos(target_token_shift)
            )
            opening_a = float(unmasked["P_A_open"][0, target_a["token"]].item())
            opening_b = float(unmasked["P_B_open"][0, target_b["token"]].item())
            record.update(
                status="success",
                target={
                    "wallA": target_a["wall"],
                    "wallB": target_b["wall"],
                    "tokenA": target_a["token"],
                    "tokenB": target_b["token"],
                    "pose": target_pose,
                },
                layout={"A": report_a, "B": report_b},
                openingResponseAtTarget={"A": opening_a, "B": opening_b},
                openingCandidateSource=(
                    "learned_opening_probability"
                    if learned_opening_weights_loaded
                    else "legacy_depth_contrast"
                ),
                openingCandidateFallback={
                    "A": (
                        learned_opening_weights_loaded
                        and probability_peak_fallback_a
                    ),
                    "B": (
                        learned_opening_weights_loaded
                        and probability_peak_fallback_b
                    ),
                },
                crossAttention={
                    "eligibleForAccuracy": matcher_report["loaded"],
                    "predictedTokenShiftRadians": forward_token_shift,
                    "targetTokenShiftRadians": target_token_shift,
                    "tokenShiftErrorDegrees": math.degrees(
                        _wrapped_angle_error(
                            forward_token_shift, target_token_shift
                        )
                    ),
                    "reversePredictedTokenShiftRadians": reverse_token_shift,
                    "reciprocalTokenShiftErrorDegrees": math.degrees(
                        _wrapped_angle_error(
                            forward_token_shift, -reverse_token_shift
                        )
                    ),
                },
                variants=variants,
            )
            if forward_reverse is not None:
                record["forwardReverse"] = forward_reverse
            pair_dir = output_dir / "pairs" / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                str(pair_dir / "selected_candidates.json"),
                selected_result.candidates_json(
                    {
                        "pairId": pair_id,
                        "matcherCheckpointLoaded": matcher_report["loaded"],
                        "openingCheckpointLoaded": opening_report["loaded"],
                        "openingCandidateThreshold": opening_threshold,
                        "openingCandidateFallback": {
                            "A": (
                                learned_opening_weights_loaded
                                and probability_peak_fallback_a
                            ),
                            "B": (
                                learned_opening_weights_loaded
                                and probability_peak_fallback_b
                            ),
                        },
                    }
                ),
            )
            atomic_write_json(
                str(pair_dir / "selected_best_joint.json"),
                selected_result.best_joint_layout,
            )
            _render_pair(output_dir, pair_id, selected_result.best_joint_layout)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"  FAILED: {record['error']}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["runtimeSeconds"] = time.perf_counter() - pair_start
        records.append(record)

    successful = [record for record in records if record.get("status") == "success"]
    variant_summaries = {
        "geometryOpening": _variant_summary(successful, "geometryOpening"),
        "allWallsGeometry": _variant_summary(successful, "allWallsGeometry"),
    }
    cross_attention_diagnostics = {
        "eligibleForAccuracy": matcher_report["loaded"],
        "checkpoint": matcher_report,
    }
    if matcher_report["loaded"]:
        variant_summaries.update(
            crossAttention=_variant_summary(successful, "crossAttention"),
            allWallsCrossAttention=_variant_summary(
                successful, "allWallsCrossAttention"
            ),
        )
        cross_attention_diagnostics.update(
            meanTokenShiftErrorDegrees=_mean(
                [record["crossAttention"] for record in successful],
                "tokenShiftErrorDegrees",
            ),
            meanReciprocalTokenShiftErrorDegrees=_mean(
                [record["crossAttention"] for record in successful],
                "reciprocalTokenShiftErrorDegrees",
            ),
            meanOpeningResponseAtTargetA=_mean(
                [record["openingResponseAtTarget"] for record in successful], "A"
            ),
            meanOpeningResponseAtTargetB=_mean(
                [record["openingResponseAtTarget"] for record in successful], "B"
            ),
            wallSwapConsistency=_mean(
                [record["forwardReverse"] for record in successful],
                "wallSwapConsistent",
            ),
            meanPoseCycleYawDegrees=_mean(
                [record["forwardReverse"] for record in successful],
                "poseCycleYawDegrees",
            ),
            meanPoseCycleTranslation=_mean(
                [record["forwardReverse"] for record in successful],
                "poseCycleTranslation",
            ),
        )
    summary = {
        "formatVersion": 1,
        "task": "ZInD labeled adjacent-room dual-panorama evaluation",
        "pairCount": len(records),
        "successCount": len(successful),
        "failureCount": len(records) - len(successful),
        "houseCount": len({record["houseId"] for record in records}),
        "runtimeSeconds": time.perf_counter() - start_time,
        "meanPairRuntimeSeconds": _mean(records, "runtimeSeconds"),
        "checkpoint": checkpoint_report,
        "neuralWeights": {
            "biLayout": "trained checkpoint",
            "openingCrossAttention": matcher_report,
            "crossAttentionMatcher": matcher_report,
            "openingHead": opening_report,
            "geometrySelector": "excluded from evaluation",
        },
        "openingCandidateThreshold": opening_threshold,
        "openingCandidateSource": (
            "learned_opening_probability"
            if learned_opening_weights_loaded
            else "legacy_depth_contrast"
        ),
        "probabilityPeakFallbackPairCount": sum(
            bool(record.get("openingCandidateFallback", {}).get("A"))
            or bool(record.get("openingCandidateFallback", {}).get("B"))
            for record in successful
        ),
        "depthBranchOrder": args.branch_order,
        "variants": variant_summaries,
        "crossAttentionDiagnostics": cross_attention_diagnostics,
        "peakCudaMemoryMiB": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_write_json(str(output_dir / "records.json"), {"records": records})
    atomic_write_json(str(output_dir / "summary.json"), summary)
    print(f"summary: {output_dir / 'summary.json'}")
    completion = "completed: {}/{} pairs, geometry top-1={:.3f}".format(
        len(successful),
        len(records),
        summary["variants"]["geometryOpening"]["openingTop1Accuracy"],
    )
    if matcher_report["loaded"]:
        completion += ", cross top-1={:.3f}".format(
            summary["variants"]["crossAttention"]["openingTop1Accuracy"]
        )
    else:
        completion += ", cross accuracy skipped (no trained matcher checkpoint)"
    print(completion)
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

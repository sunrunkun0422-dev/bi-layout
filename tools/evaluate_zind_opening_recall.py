#!/usr/bin/env python3
"""Evaluate ZInD-BiPair-v1 opening recall once per unique panorama."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.opening_recall import (
    UniqueViewReference,
    deduplicate_manifest_views,
    evaluate_opening_scores,
    opening_geometry_probability,
    resolve_depth_branches,
)


DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "zind/ZInD-BiPair-v1"
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--manifest",
        help="override <dataset_root>/manifests/<split>_pairs.jsonl",
    )
    parser.add_argument(
        "--data_root",
        help="override source.dataRoot from dataset_info.json",
    )
    parser.add_argument(
        "--source",
        choices=("gt-depth", "predicted-depth"),
        default="gt-depth",
        help="score openings from cached GT depth or Bi-Layout checkpoint predictions",
    )
    parser.add_argument(
        "--branch_order",
        choices=("extended_first", "enclosed_first"),
        default="extended_first",
        help=(
            "interpret the native first/second depth streams; zind_all checkpoint "
            "uses extended_first, while enclosed_first reproduces the reversed diagnostic"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "fixed threshold; when omitted, val scans thresholds, while test "
            "requires --opening_checkpoint and uses its operating_threshold"
        ),
    )
    parser.add_argument("--scan_min", type=float, default=0.0)
    parser.add_argument("--scan_max", type=float, default=1.0)
    parser.add_argument("--scan_steps", type=int, default=201)
    parser.add_argument("--precision_target", type=float, default=0.85)
    parser.add_argument("--prior_strength", type=float, default=4.0)
    parser.add_argument("--prior_relative_scale", type=float, default=0.1)
    parser.add_argument(
        "--max_views",
        type=int,
        help="evaluate only the first N deduplicated views; useful for CPU smoke tests",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--opening_checkpoint",
        help=(
            "trained Opening Head best.pt; only valid with --source predicted-depth"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--output",
        help="output JSON path; default is src/output/zind_opening_recall/<name>.json",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("batch_size and torch_threads must be positive")
    if min(args.image_height, args.image_width) <= 0:
        raise ValueError("image dimensions must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if args.max_views is not None and args.max_views <= 0:
        raise ValueError("max_views must be positive when provided")
    if args.scan_steps < 2 or args.scan_min >= args.scan_max:
        raise ValueError("threshold scan requires scan_steps >= 2 and scan_min < scan_max")
    if not 0.0 <= args.precision_target <= 1.0:
        raise ValueError("precision_target must be in [0, 1]")
    if args.prior_relative_scale <= 0:
        raise ValueError("prior_relative_scale must be positive")
    if args.threshold is not None and not math.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    if args.opening_checkpoint and args.source != "predicted-depth":
        raise ValueError(
            "--opening_checkpoint requires --source predicted-depth because the "
            "learned head consumes Bi-Layout layout_feature"
        )
    if args.threshold is None and args.split != "val":
        if not (args.split == "test" and args.opening_checkpoint):
            raise ValueError(
                "a fixed --threshold is required for train/test unless test can "
                "read operating_threshold from --opening_checkpoint"
            )


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            records.append(value)
    if not records:
        raise ValueError(f"manifest contains no records: {path}")
    return records


def _load_view_cache(
    dataset_root: Path,
    view: UniqueViewReference,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = dataset_root / view.label_cache
    with np.load(path, allow_pickle=False) as cache:
        target_key = f"opening_mask_all_{view.side}"
        enclosed_key = f"depth_enclosed_{view.side}"
        extended_key = f"depth_extended_{view.side}"
        missing = [
            key
            for key in (target_key, enclosed_key, extended_key)
            if key not in cache.files
        ]
        if missing:
            raise ValueError(f"{path} is missing labels: {', '.join(missing)}")
        target = np.asarray(cache[target_key], dtype=np.uint8).copy()
        # The GT source is exposed in the same native order as the checkpoint:
        # first=extended/visible and second=enclosed/raw.
        first_depth = np.asarray(cache[extended_key], dtype=np.float32).copy()
        second_depth = np.asarray(cache[enclosed_key], dtype=np.float32).copy()
    return target, first_depth, second_depth


def _validate_view_arrays(
    view: UniqueViewReference,
    target: np.ndarray,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    token_count: int,
) -> None:
    expected = (token_count,)
    for name, value in (
        ("opening target", target),
        ("first depth", first_depth),
        ("second depth", second_depth),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{view.image_path} {name} has shape {value.shape}, expected {expected}"
            )


def _score_depth_streams(
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    enclosed, extended = resolve_depth_branches(
        first_depth, second_depth, args.branch_order
    )
    return opening_geometry_probability(
        enclosed,
        extended,
        prior_strength=args.prior_strength,
        prior_relative_scale=args.prior_relative_scale,
    )


def _gt_depth_scores(
    dataset_root: Path,
    views: Sequence[UniqueViewReference],
    token_count: int,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    targets: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for index, view in enumerate(views, start=1):
        target, first_depth, second_depth = _load_view_cache(dataset_root, view)
        _validate_view_arrays(view, target, first_depth, second_depth, token_count)
        targets.append(target)
        scores.append(_score_depth_streams(first_depth, second_depth, args))
        if index % args.progress_every == 0 or index == len(views):
            print(f"[{index}/{len(views)}] GT-depth views scored")
    return targets, scores, {
        "predictionSource": "gt-depth",
        "nativeFirst": "depth_extended (layout_visible)",
        "nativeSecond": "depth_enclosed (layout_raw)",
    }


def _load_image(path: Path, height: int, width: int):
    import torch
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _checkpoint_positive_int(model_config: Mapping[str, Any], key: str) -> int:
    value = model_config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(
            "opening checkpoint model.{0} must be a positive integer".format(key)
        )
    return value


def _checkpoint_finite_float(
    model_config: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> float:
    value = model_config.get(key)
    if isinstance(value, bool):
        raise ValueError(
            "opening checkpoint model.{0} must be finite".format(key)
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "opening checkpoint model.{0} must be finite".format(key)
        ) from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(
            "opening checkpoint model.{0} must be {1}".format(key, qualifier)
        )
    return number


def _load_opening_head(
    checkpoint_path: Path,
    device,
    *,
    expected_feature_dim: int,
    branch_order: str,
):
    """Load the exact OpeningSignalHead architecture saved by the trainer."""

    import torch

    from models.cross_scene_matcher import OpeningSignalHead

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("opening checkpoint root must be a mapping")
    model_config = payload.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("opening checkpoint is missing model configuration")
    state_dict = payload.get("opening_head_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("opening checkpoint is missing opening_head_state_dict")

    feature_dim = _checkpoint_positive_int(model_config, "feature_dim")
    hidden_dim = _checkpoint_positive_int(model_config, "hidden_dim")
    kernel_size = _checkpoint_positive_int(model_config, "kernel_size")
    if feature_dim != int(expected_feature_dim):
        raise ValueError(
            "opening checkpoint feature_dim={} does not match Bi-Layout patch_dim={}".format(
                feature_dim, expected_feature_dim
            )
        )
    saved_branch_order = model_config.get("branch_order")
    if saved_branch_order not in ("extended_first", "enclosed_first"):
        raise ValueError(
            "opening checkpoint model.branch_order must be extended_first or enclosed_first"
        )
    if saved_branch_order != branch_order:
        raise ValueError(
            "opening checkpoint branch_order={!r} does not match --branch_order={!r}".format(
                saved_branch_order, branch_order
            )
        )

    prior_strength = _checkpoint_finite_float(model_config, "prior_strength")
    prior_relative_scale = _checkpoint_finite_float(
        model_config, "prior_relative_scale", positive=True
    )
    head = OpeningSignalHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        prior_strength=prior_strength,
        prior_relative_scale=prior_relative_scale,
    )
    head.load_state_dict(state_dict, strict=True)
    head.to(device).eval()

    raw_threshold = payload.get("operating_threshold")
    operating_threshold = None
    if raw_threshold is not None:
        if isinstance(raw_threshold, bool):
            raise ValueError(
                "opening checkpoint operating_threshold must be in [0, 1]"
            )
        try:
            operating_threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "opening checkpoint operating_threshold must be in [0, 1]"
            ) from exc
        if not math.isfinite(operating_threshold) or not 0.0 <= operating_threshold <= 1.0:
            raise ValueError(
                "opening checkpoint operating_threshold must be in [0, 1]"
            )

    report = {
        "path": str(path),
        "loaded": True,
        "formatVersion": payload.get("format_version"),
        "task": payload.get("task"),
        "completedEpoch": payload.get("completed_epoch"),
        "operatingThreshold": operating_threshold,
        "thresholdPolicy": payload.get("threshold_policy"),
        "thresholdFallback": payload.get("threshold_fallback"),
        "model": dict(model_config),
    }
    return head, report


def _learned_opening_probability(
    output: Mapping[str, Any],
    opening_head,
    branch_order: str,
):
    """Score a Bi-Layout batch with its learned single-view Opening Head."""

    from models.cross_scene_matcher import resolve_enclosed_extended_depth

    if "layout_feature" not in output:
        raise ValueError(
            "Bi-Layout output must contain layout_feature for learned Opening Head"
        )
    enclosed, extended = resolve_enclosed_extended_depth(output, branch_order)
    head_output = opening_head(output["layout_feature"], enclosed, extended)
    probability = head_output.get("opening_probability")
    if probability is None:
        raise ValueError("Opening Head output must contain opening_probability")
    return probability


def _predicted_depth_scores(
    dataset_root: Path,
    data_root: Path,
    views: Sequence[UniqueViewReference],
    token_count: int,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    import torch

    from tools.debug_cross_scene_flow import load_bi_layout

    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model, checkpoint_report = load_bi_layout(
        str(config_path), str(checkpoint_path), device, load_checkpoint=True
    )
    if int(model.patch_num) != token_count:
        raise ValueError(
            f"model patch_num={model.patch_num} does not match dataset tokenCount={token_count}"
        )

    opening_head = None
    opening_checkpoint_report = None
    if args.opening_checkpoint:
        opening_head, opening_checkpoint_report = _load_opening_head(
            Path(args.opening_checkpoint),
            device,
            expected_feature_dim=int(model.patch_dim),
            branch_order=args.branch_order,
        )

    targets: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for offset in range(0, len(views), args.batch_size):
        batch_views = views[offset : offset + args.batch_size]
        images = torch.stack(
            [
                _load_image(
                    data_root / view.image_path,
                    args.image_height,
                    args.image_width,
                )
                for view in batch_views
            ]
        ).to(device)
        with torch.no_grad():
            output = (
                model(images)
                if opening_head is None
                else model(images, return_features=True)
            )
        if "depth" not in output or "new_depth" not in output:
            raise ValueError("checkpoint model must output both depth and new_depth")
        # Existing zind_all contract: first depth=visible/extended, second
        # new_depth=raw/enclosed.  branch_order controls only interpretation.
        first_batch = output["depth"].detach().cpu().numpy()
        second_batch = output["new_depth"].detach().cpu().numpy()
        if opening_head is None:
            score_batch = _score_depth_streams(first_batch, second_batch, args)
        else:
            with torch.no_grad():
                score_batch = (
                    _learned_opening_probability(
                        output, opening_head, args.branch_order
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        for batch_index, view in enumerate(batch_views):
            target, _, _ = _load_view_cache(dataset_root, view)
            first_depth = np.asarray(first_batch[batch_index])
            second_depth = np.asarray(second_batch[batch_index])
            _validate_view_arrays(
                view, target, first_depth, second_depth, token_count
            )
            targets.append(target)
            scores.append(np.asarray(score_batch[batch_index], dtype=np.float32))
        completed = min(offset + len(batch_views), len(views))
        if completed % args.progress_every == 0 or completed == len(views):
            print(f"[{completed}/{len(views)}] checkpoint-predicted views scored")

    report = {
        "predictionSource": "predicted-depth",
        "nativeFirst": "checkpoint output['depth'] (layout_visible/extended)",
        "nativeSecond": "checkpoint output['new_depth'] (layout_raw/enclosed)",
        "checkpoint": checkpoint_report,
    }
    if opening_checkpoint_report is not None:
        report["openingHeadCheckpoint"] = opening_checkpoint_report
        report["openingScore"] = {
            "type": "learned OpeningSignalHead",
            "isLearnedOpeningCheckpoint": True,
            "model": opening_checkpoint_report["model"],
        }
    return targets, scores, report


def _depth_contract(source_report: Mapping[str, Any], branch_order: str) -> Dict[str, Any]:
    native_first = source_report["nativeFirst"]
    native_second = source_report["nativeSecond"]
    if branch_order == "extended_first":
        enclosed = native_second
        extended = native_first
    else:
        enclosed = native_first
        extended = native_second
    return {
        "branchOrder": branch_order,
        "nativeFirst": native_first,
        "nativeSecond": native_second,
        "interpretedEnclosed": enclosed,
        "interpretedExtended": extended,
        "geometryOperation": "relu(interpretedExtended - interpretedEnclosed)",
        "isRecommendedZindOrder": branch_order == "extended_first",
    }


def _resolve_evaluation_threshold(
    args: argparse.Namespace,
    source_report: Mapping[str, Any],
) -> Tuple[Optional[float], str]:
    """Resolve threshold provenance without ever scanning train/test data."""

    if args.threshold is not None:
        return float(args.threshold), "explicit_cli"
    if args.split == "val":
        return None, "validation_scan"
    if args.split != "test":
        raise ValueError("train evaluation requires an explicit --threshold")

    checkpoint_report = source_report.get("openingHeadCheckpoint")
    if not isinstance(checkpoint_report, Mapping):
        raise ValueError(
            "test evaluation without --threshold requires --opening_checkpoint"
        )
    threshold = checkpoint_report.get("operatingThreshold")
    if threshold is None:
        raise ValueError(
            "opening checkpoint has no operating_threshold; test threshold scanning is forbidden"
        )
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "opening checkpoint operating_threshold must be in [0, 1]"
        )
    return threshold, "opening_checkpoint"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    start_time = time.perf_counter()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info_path = dataset_root / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    dataset_info = _read_json(info_path)
    token_count = int(dataset_info["tokenCount"])
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else dataset_root / "manifests" / f"{args.split}_pairs.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records = _read_jsonl(manifest_path)
    all_unique_views = deduplicate_manifest_views(records)
    if not all_unique_views:
        raise RuntimeError("manifest contains no panorama views")
    views = (
        all_unique_views[: args.max_views]
        if args.max_views is not None
        else all_unique_views
    )
    data_root_value = args.data_root or dataset_info.get("source", {}).get("dataRoot")
    if not data_root_value:
        raise ValueError("data_root is not provided and dataset_info has no source.dataRoot")
    data_root = Path(data_root_value).expanduser().resolve()

    if args.source == "gt-depth":
        targets, scores, source_report = _gt_depth_scores(
            dataset_root, views, token_count, args
        )
    else:
        targets, scores, source_report = _predicted_depth_scores(
            dataset_root, data_root, views, token_count, args
        )

    evaluation_threshold, threshold_source = _resolve_evaluation_threshold(
        args, source_report
    )
    thresholds = (
        np.linspace(args.scan_min, args.scan_max, args.scan_steps)
        if evaluation_threshold is None
        else None
    )
    evaluation = evaluate_opening_scores(
        targets,
        scores,
        threshold=evaluation_threshold,
        thresholds=thresholds,
        precision_target=args.precision_target,
    )
    evaluation["thresholdInputSource"] = threshold_source
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else REPO_ROOT
        / "src/output/zind_opening_recall"
        / f"{args.split}_{args.source}_{args.branch_order}.json"
    )
    report: Dict[str, Any] = {
        "formatVersion": 1,
        "task": "ZInD-BiPair-v1 unique-panorama opening recall",
        "dataset": {
            "name": dataset_info.get("datasetName", "ZInD-BiPair-v1"),
            "datasetRoot": str(dataset_root),
            "dataRoot": str(data_root),
            "manifest": str(manifest_path),
            "split": args.split,
            "manifestPairCount": len(records),
            "manifestViewReferenceCount": len(records) * 2,
            "uniqueViewCountBeforeLimit": len(all_unique_views),
            "evaluatedUniqueViewCount": len(views),
            "duplicateViewReferenceCount": len(records) * 2 - len(all_unique_views),
            "maxViews": args.max_views,
            "deduplicationKey": "view.image_path",
            "target": "opening_mask_all",
        },
        "source": {
            **source_report,
            "branchOrder": args.branch_order,
            "depthContract": _depth_contract(source_report, args.branch_order),
            "openingScore": source_report.get(
                "openingScore",
                {
                    "type": "OpeningSignalHead zero-weight geometry prior",
                    "priorStrength": args.prior_strength,
                    "priorRelativeScale": args.prior_relative_scale,
                    "isLearnedOpeningCheckpoint": False,
                },
            ),
        },
        "evaluation": evaluation,
        "runtimeSeconds": time.perf_counter() - start_time,
    }
    _write_json(output_path, report)
    selected = evaluation["metricsAtSelectedThreshold"]
    print(f"report: {output_path}")
    print(
        "views={}, AP={:.4f}, threshold={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}, IoU={:.4f}".format(
            len(views),
            evaluation["averagePrecision"],
            evaluation["selectedThreshold"],
            selected["precision"],
            selected["recall"],
            selected["f1"],
            selected["iou"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

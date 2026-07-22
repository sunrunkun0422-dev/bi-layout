#!/usr/bin/env python3
"""Generate the ZInD-BiPair-v1 dataset beside the native ZInD checkout."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_bipair_builder import (
    DATASET_NAME,
    PairExample,
    PairThresholds,
    build_house_pair_examples,
    build_pair_record_and_arrays,
    build_view_label_arrays,
    interleave_pairs,
)


DEFAULT_ZIND_ROOT = REPO_ROOT.parent / "zind/data"
DEFAULT_PARTITION = REPO_ROOT.parent / "zind/zind_partition.json"
DEFAULT_OUTPUT = REPO_ROOT.parent / f"zind/{DATASET_NAME}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ZInD partial rooms into pair-centered opening matching, "
            "relative pose, and layout-completion supervision."
        )
    )
    parser.add_argument("--zind_root", default=str(DEFAULT_ZIND_ROOT))
    parser.add_argument("--partition", default=str(DEFAULT_PARTITION))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--token_count", type=int, default=256)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--midpoint_distance_m", type=float, default=0.20)
    parser.add_argument("--direction_difference_degrees", type=float, default=10.0)
    parser.add_argument("--relative_length_error", type=float, default=0.20)
    parser.add_argument("--endpoint_error_m", type=float, default=0.25)
    parser.add_argument(
        "--max_houses_per_split",
        type=int,
        default=0,
        help="limit each split for a smoke test; 0 uses the complete split",
    )
    parser.add_argument(
        "--max_pairs_per_split",
        type=int,
        default=0,
        help="limit emitted pairs after mining; 0 emits all selected pairs",
    )
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace generated manifests/caches in an existing output directory",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> PairThresholds:
    if args.token_count <= 0:
        raise ValueError("--token_count must be positive")
    if args.negative_ratio < 0:
        raise ValueError("--negative_ratio must be non-negative")
    if min(args.max_houses_per_split, args.max_pairs_per_split) < 0:
        raise ValueError("dataset limits must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be positive")
    thresholds = PairThresholds(
        midpoint_distance_m=args.midpoint_distance_m,
        direction_difference_degrees=args.direction_difference_degrees,
        relative_length_error=args.relative_length_error,
        endpoint_error_m=args.endpoint_error_m,
    )
    thresholds.validate()
    return thresholds


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def _load_partition(path: Path) -> Mapping[str, Sequence[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"partition does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        partition = json.load(file)
    for split in ("train", "val", "test"):
        if split not in partition or not isinstance(partition[split], list):
            raise ValueError(f"partition is missing a valid {split} house list")
    return partition


def _load_house(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _select_negatives(
    positives: Sequence[PairExample],
    negatives: Sequence[PairExample],
    negative_ratio: float,
) -> List[PairExample]:
    requested = int(math.ceil(len(positives) * negative_ratio))
    return list(negatives[:requested])


def _metric_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _dataset_readme() -> str:
    return """# ZInD-BiPair-v1

ZInD-BiPair-v1 is a pair-centered derivative of ZInD for Bi-Layout
cross-panorama opening matching and layout merging. Version 1 contains strict
`partial_opening` pairs only: both views come from different partial rooms in
the same complete room, and positive pairs share a matched `layout_raw.openings`
segment.

The dataset does not copy panorama images. `view_A.image_path` and
`view_B.image_path` are relative to the native ZInD data root recorded in
`dataset_info.json`.

## Files

- `manifests/{train,val,test}_pairs.jsonl`: pair metadata and geometry.
- `labels/<split>/<pair_id>.npz`: 256-token Bi-Layout, opening, affinity,
  relative-pose, and complete-layout targets.
- `statistics/*_stats.json`: split counts and portal matching diagnostics.
- `statistics/invalid_pairs.jsonl`: filtered views/pairs and reasons.

## Scope

- Primary panoramas only.
- `is_inside=true` and `is_ceiling_flat=true`.
- `layout_raw`, `layout_visible`, `layout_complete`, metric floor scale, and
  `floor_plan_transformation` are required.
- House-level train/val/test partitions are inherited from ZInD.
"""


def convert_split(
    split: str,
    house_ids: Sequence[str],
    zind_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    thresholds: PairThresholds,
) -> tuple:
    selected_houses = list(house_ids)
    if args.max_houses_per_split:
        selected_houses = selected_houses[: args.max_houses_per_split]
    positives: List[PairExample] = []
    negative_candidates: List[PairExample] = []
    invalid_records: List[Dict[str, Any]] = []
    eligible_view_count = 0
    started = time.perf_counter()

    for house_index, house_id in enumerate(selected_houses, start=1):
        annotation = zind_root / house_id / "zind_data.json"
        if not annotation.is_file():
            invalid_records.append(
                {"split": split, "house_id": house_id, "reason": "missing_zind_data_json"}
            )
            continue
        payload = _load_house(annotation)
        house_positive, house_negative, house_invalid, house_views = (
            build_house_pair_examples(
                payload,
                house_id,
                zind_root,
                args.token_count,
                thresholds,
            )
        )
        positives.extend(house_positive)
        negative_candidates.extend(house_negative)
        invalid_records.extend({"split": split, **record} for record in house_invalid)
        eligible_view_count += house_views
        if house_index % args.progress_every == 0 or house_index == len(selected_houses):
            print(
                f"[{split}] mined houses {house_index}/{len(selected_houses)}: "
                f"positive={len(positives)}, negative_candidates={len(negative_candidates)}",
                flush=True,
            )

    positives.sort(key=lambda example: example.pair_id)
    negative_candidates.sort(key=lambda example: (example.camera_distance_m, example.pair_id))

    # Project each unique view only once. Invalid projections remove every pair
    # that references the view and are recorded explicitly.
    view_cache: Dict[str, Mapping[str, np.ndarray]] = {}
    invalid_view_ids = set()
    for example in [*positives, *negative_candidates]:
        for view in (example.view_a, example.view_b):
            if view.view_id in view_cache or view.view_id in invalid_view_ids:
                continue
            try:
                view_cache[view.view_id] = build_view_label_arrays(
                    view, args.token_count
                )
            except Exception as exc:
                invalid_view_ids.add(view.view_id)
                invalid_records.append(
                    {
                        "split": split,
                        "house_id": view.house_id,
                        "floor_id": view.floor_id,
                        "complete_room_id": view.complete_room_id,
                        "partial_room_id": view.partial_room_id,
                        "pano_id": view.pano_id,
                        "reason": "label_projection_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    positives = [
        example
        for example in positives
        if example.view_a.view_id not in invalid_view_ids
        and example.view_b.view_id not in invalid_view_ids
    ]
    negative_candidates = [
        example
        for example in negative_candidates
        if example.view_a.view_id not in invalid_view_ids
        and example.view_b.view_id not in invalid_view_ids
    ]
    negatives = _select_negatives(positives, negative_candidates, args.negative_ratio)
    pairs = interleave_pairs(positives, negatives)
    if args.max_pairs_per_split:
        pairs = pairs[: args.max_pairs_per_split]

    records = []
    emitted_positive = 0
    endpoint_errors = []
    midpoint_distances = []
    direction_errors = []
    length_errors = []
    overlaps = []
    for pair_index, example in enumerate(pairs, start=1):
        label_relative = f"labels/{split}/{example.pair_id}.npz"
        try:
            record, arrays = build_pair_record_and_arrays(
                example,
                split,
                label_relative,
                args.token_count,
                view_cache,
            )
            _atomic_npz(output_dir / label_relative, arrays)
            records.append(record)
            emitted_positive += int(example.is_positive)
            overlaps.append(float(record["joint_layout_gt"]["overlap_ratio_A_B"]))
            if example.portal_match is not None:
                endpoint_errors.append(example.portal_match.endpoint_error_m)
                midpoint_distances.append(example.portal_match.midpoint_distance_m)
                direction_errors.append(
                    example.portal_match.direction_difference_degrees
                )
                length_errors.append(example.portal_match.relative_length_error)
        except Exception as exc:
            invalid_records.append(
                {
                    "split": split,
                    "pair_id": example.pair_id,
                    "reason": "pair_cache_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if pair_index % 1000 == 0 or pair_index == len(pairs):
            print(
                f"[{split}] wrote pairs {pair_index}/{len(pairs)}",
                flush=True,
            )

    manifest_path = output_dir / "manifests" / f"{split}_pairs.jsonl"
    _atomic_jsonl(manifest_path, records)
    invalid_reasons = Counter(record["reason"] for record in invalid_records)
    stats = {
        "datasetName": DATASET_NAME,
        "split": split,
        "housesRequested": len(selected_houses),
        "eligibleViewCount": eligible_view_count,
        "cachedViewCount": len(view_cache),
        "pairCount": len(records),
        "positivePairCount": emitted_positive,
        "negativePairCount": len(records) - emitted_positive,
        "negativeCandidateCount": len(negative_candidates),
        "invalidRecordCount": len(invalid_records),
        "invalidReasonCounts": dict(sorted(invalid_reasons.items())),
        "portalEndpointErrorMeters": _metric_summary(endpoint_errors),
        "portalMidpointDistanceMeters": _metric_summary(midpoint_distances),
        "portalDirectionDifferenceDegrees": _metric_summary(direction_errors),
        "portalRelativeLengthError": _metric_summary(length_errors),
        "jointLayoutOverlapRatio": _metric_summary(overlaps),
        "runtimeSeconds": time.perf_counter() - started,
        "manifest": str(manifest_path),
    }
    _atomic_json(output_dir / "statistics" / f"{split}_stats.json", stats)
    return stats, invalid_records


def main() -> int:
    args = parse_args()
    thresholds = _validate_args(args)
    zind_root = Path(args.zind_root).expanduser().resolve()
    partition_path = Path(args.partition).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not zind_root.is_dir():
        raise FileNotFoundError(f"ZInD data root does not exist: {zind_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --overwrite to replace generated files"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    partition = _load_partition(partition_path)

    split_stats = []
    invalid_records = []
    for split in args.splits:
        stats, split_invalid = convert_split(
            split,
            partition[split],
            zind_root,
            output_dir,
            args,
            thresholds,
        )
        split_stats.append(stats)
        invalid_records.extend(split_invalid)

    _atomic_jsonl(output_dir / "statistics" / "invalid_pairs.jsonl", invalid_records)
    info = {
        "datasetName": DATASET_NAME,
        "schemaVersion": "1.0",
        "pairType": "partial_opening",
        "description": (
            "Strict primary-panorama pairs from different partial rooms in the "
            "same ZInD complete room, with opening, pose, and complete-layout labels."
        ),
        "source": {
            "dataRoot": str(zind_root),
            "partition": str(partition_path),
            "imagesCopied": False,
        },
        "tokenCount": args.token_count,
        "negativeRatio": args.negative_ratio,
        "thresholds": {
            "midpointDistanceMeters": thresholds.midpoint_distance_m,
            "directionDifferenceDegrees": thresholds.direction_difference_degrees,
            "relativeLengthError": thresholds.relative_length_error,
            "endpointErrorMeters": thresholds.endpoint_error_m,
        },
        "filters": {
            "isPrimary": True,
            "isInside": True,
            "isCeilingFlat": True,
            "requiresLayouts": ["layout_raw", "layout_visible", "layout_complete"],
            "requiresMetricFloorScale": True,
        },
        "splits": split_stats,
        "totals": {
            "pairs": sum(stats["pairCount"] for stats in split_stats),
            "positivePairs": sum(stats["positivePairCount"] for stats in split_stats),
            "negativePairs": sum(stats["negativePairCount"] for stats in split_stats),
            "invalidRecords": len(invalid_records),
        },
    }
    _atomic_json(output_dir / "dataset_info.json", info)
    (output_dir / "README.md").write_text(_dataset_readme(), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate manifests, paths, NPZ targets, and split isolation for ZInD-BiPair-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--data_root", help="override source.dataRoot in dataset_info.json")
    return parser.parse_args()


def _jsonl(path: Path) -> List[Mapping[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def _require_shape(cache: Mapping[str, np.ndarray], key: str, shape) -> None:
    if key not in cache:
        raise ValueError(f"label cache is missing {key}")
    if tuple(cache[key].shape) != tuple(shape):
        raise ValueError(f"{key} has shape {cache[key].shape}, expected {shape}")


def _validate_cache(path: Path, record: Mapping[str, Any], token_count: int) -> None:
    with np.load(path, allow_pickle=False) as cache:
        for side in ("A", "B"):
            for key in (
                f"depth_enclosed_{side}",
                f"depth_extended_{side}",
                f"extension_depth_{side}",
                f"opening_mask_all_{side}",
                f"portal_mask_{side}",
            ):
                _require_shape(cache, key, (token_count,))
            _require_shape(cache, f"ratio_{side}", (1,))
            if not np.isfinite(cache[f"depth_enclosed_{side}"]).all():
                raise ValueError(f"non-finite enclosed depth in {path}")
            if not np.isfinite(cache[f"depth_extended_{side}"]).all():
                raise ValueError(f"non-finite extended depth in {path}")
            if (cache[f"extension_depth_{side}"] < 0).any():
                raise ValueError(f"negative extension depth in {path}")
            portal = cache[f"portal_mask_{side}"].astype(bool)
            openings = cache[f"opening_mask_all_{side}"].astype(bool)
            if np.any(portal & ~openings):
                raise ValueError(f"shared portal is not part of all-opening mask in {path}")
        _require_shape(cache, "affinity_gt", (token_count, token_count))
        _require_shape(cache, "T_B_to_A", (3, 3))
        _require_shape(cache, "relative_yaw_gt", (1,))
        _require_shape(cache, "translation_gt", (2,))
        _require_shape(cache, "translation_meters_gt", (2,))
        _require_shape(cache, "relative_scale_gt", (1,))
        if not np.isfinite(cache["T_B_to_A"]).all():
            raise ValueError(f"non-finite relative transform in {path}")
        mask_a_mass = int(cache["portal_mask_A"].sum())
        mask_b_mass = int(cache["portal_mask_B"].sum())
        affinity_mass = int(cache["affinity_gt"].sum())
        positive = bool(record["is_positive"])
        if positive:
            if min(mask_a_mass, mask_b_mass) <= 0:
                raise ValueError(f"positive pair has an empty portal mask: {path}")
            if affinity_mass != mask_a_mass * mask_b_mass:
                raise ValueError(f"positive affinity mass is inconsistent: {path}")
        elif mask_a_mass or mask_b_mass or affinity_mass:
            raise ValueError(f"negative pair contains shared-portal positives: {path}")
        overlap = float(cache["joint_layout_overlap_ratio"][0])
        if not np.isfinite(overlap) or overlap < 0.99:
            raise ValueError(f"unexpected complete-layout overlap {overlap}: {path}")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    info_path = dataset_root / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    token_count = int(info["tokenCount"])
    data_root = Path(args.data_root or info["source"]["dataRoot"]).expanduser().resolve()
    all_pair_ids = set()
    house_sets = {}
    summaries: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        manifest = dataset_root / "manifests" / f"{split}_pairs.jsonl"
        records = _jsonl(manifest)
        positive_count = 0
        houses = set()
        for record in records:
            pair_id = record["pair_id"]
            if pair_id in all_pair_ids:
                raise ValueError(f"duplicate pair_id: {pair_id}")
            all_pair_ids.add(pair_id)
            houses.add(record["house_id"])
            if record["split"] != split:
                raise ValueError(f"pair {pair_id} has the wrong split")
            if record["pair_type"] != "partial_opening":
                raise ValueError(f"pair {pair_id} has an unsupported pair_type")
            view_a = record["view_A"]
            view_b = record["view_B"]
            if view_a["complete_room_id"] != view_b["complete_room_id"]:
                raise ValueError(f"pair {pair_id} crosses complete rooms")
            if view_a["partial_room_id"] == view_b["partial_room_id"]:
                raise ValueError(f"pair {pair_id} does not cross partial rooms")
            positive = bool(record["is_positive"])
            positive_count += int(positive)
            if positive != (record.get("shared_portal") is not None):
                raise ValueError(f"pair {pair_id} has inconsistent portal metadata")
            for view in (view_a, view_b):
                image_path = data_root / view["image_path"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"missing panorama: {image_path}")
            cache_path = dataset_root / record["label_cache"]
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing label cache: {cache_path}")
            _validate_cache(cache_path, record, token_count)
        house_sets[split] = houses
        summaries[split] = {
            "pairs": len(records),
            "positivePairs": positive_count,
            "negativePairs": len(records) - positive_count,
            "housesWithPairs": len(houses),
        }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = house_sets[first] & house_sets[second]
        if overlap:
            raise ValueError(f"house split leakage between {first}/{second}: {sorted(overlap)[:3]}")
    output = {
        "datasetName": info["datasetName"],
        "valid": True,
        "pairCount": len(all_pair_ids),
        "splitLeakageHouseCount": 0,
        "splits": summaries,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

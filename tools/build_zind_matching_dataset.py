#!/usr/bin/env python3
"""Convert native ZInD annotations into supervised panorama matching manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_pair_mining import mine_zind_matching_records


DEFAULT_ZIND_ROOT = REPO_ROOT.parent / "zind/data"
DEFAULT_PARTITION = REPO_ROOT.parent / "zind/zind_partition.json"
DEFAULT_OUTPUT = REPO_ROOT / "src/dataset/ZInD_matching"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val/test manifests containing ZInD panorama pairs, "
            "door/opening token intervals, correspondence targets, and relative pose."
        )
    )
    parser.add_argument("--zind_root", default=str(DEFAULT_ZIND_ROOT))
    parser.add_argument("--partition", default=str(DEFAULT_PARTITION))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        choices=("train", "val", "test"),
    )
    parser.add_argument("--token_count", type=int, default=256)
    parser.add_argument("--endpoint_tolerance", type=float, default=0.06)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument(
        "--negative_min_endpoint_error",
        type=float,
        help="minimum endpoint mismatch for a safe negative; default=max(0.15, 2*tolerance)",
    )
    parser.add_argument(
        "--max_houses_per_split",
        type=int,
        default=0,
        help="limit each split for smoke tests; 0 converts the complete split",
    )
    parser.add_argument(
        "--progress_every", type=int, default=50, help="print progress every N houses"
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="record and skip houses without zind_data.json instead of failing",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.token_count <= 0:
        raise ValueError("--token_count must be positive")
    if args.endpoint_tolerance <= 0:
        raise ValueError("--endpoint_tolerance must be positive")
    if args.negative_ratio < 0:
        raise ValueError("--negative_ratio must be non-negative")
    if args.max_houses_per_split < 0:
        raise ValueError("--max_houses_per_split must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be positive")


def _load_partition(path: Path) -> Mapping[str, Sequence[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"ZInD partition does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    for split in ("train", "val", "test"):
        if split not in payload or not isinstance(payload[split], list):
            raise ValueError(f"partition is missing a valid '{split}' house list")
    return payload


def _record_images_exist(record: Mapping[str, Any], root: Path) -> bool:
    return all((root / record[key]).is_file() for key in ("image_a", "image_b"))


def convert_split(
    split: str,
    house_ids: Sequence[str],
    zind_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    missing_houses = []
    missing_image_records = []
    started = time.perf_counter()
    selected_houses = list(house_ids)
    if args.max_houses_per_split:
        selected_houses = selected_houses[: args.max_houses_per_split]

    for house_index, house_id in enumerate(selected_houses, start=1):
        annotation = zind_root / house_id / "zind_data.json"
        if not annotation.is_file():
            if not args.skip_missing:
                raise FileNotFoundError(f"missing ZInD annotation: {annotation}")
            missing_houses.append(house_id)
            continue
        house_records = mine_zind_matching_records(
            str(annotation),
            endpoint_tolerance=args.endpoint_tolerance,
            token_count=args.token_count,
            negative_ratio=args.negative_ratio,
            negative_min_endpoint_error=args.negative_min_endpoint_error,
            data_root=str(zind_root),
        )
        for record in house_records:
            if _record_images_exist(record, zind_root):
                records.append(record)
            elif args.skip_missing:
                missing_image_records.append(record["id"])
            else:
                raise FileNotFoundError(
                    f"pair {record['id']} references a missing panorama image"
                )
        if house_index % args.progress_every == 0 or house_index == len(selected_houses):
            print(
                f"[{split}] houses {house_index}/{len(selected_houses)}, "
                f"matching records {len(records)}",
                flush=True,
            )

    positive_count = sum(record["supervision"]["is_match"] for record in records)
    negative_count = len(records) - positive_count
    manifest = {
        "formatVersion": 2,
        "task": "ZInD cross-scene shared-interface matching",
        "split": split,
        "dataRoot": str(zind_root),
        "tokenCount": args.token_count,
        "candidateEncoding": "inclusive circular [startToken, endToken]",
        "coordinateConvention": {
            "relativeTransform": "B local floor coordinates to A local floor coordinates",
            "relativeYaw": "atan2(transformBToA[1,0], transformBToA[0,0])",
        },
        "counts": {
            "housesRequested": len(selected_houses),
            "housesConverted": len(selected_houses) - len(missing_houses),
            "pairs": len(records),
            "positivePairs": int(positive_count),
            "negativePairs": int(negative_count),
            "missingHouses": len(missing_houses),
            "missingImagePairs": len(missing_image_records),
        },
        "sourceHouseIds": selected_houses,
        "missingHouseIds": missing_houses,
        "missingImagePairIds": missing_image_records,
        "pairs": records,
    }
    manifest_path = output_dir / f"{split}.json"
    _atomic_json(manifest_path, manifest)
    return {
        "split": split,
        "manifest": str(manifest_path),
        **manifest["counts"],
        "runtimeSeconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    _validate_args(args)
    zind_root = Path(args.zind_root).expanduser().resolve()
    partition_path = Path(args.partition).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not zind_root.is_dir():
        raise FileNotFoundError(f"ZInD root does not exist: {zind_root}")
    partition = _load_partition(partition_path)

    summaries = [
        convert_split(
            split,
            partition[split],
            zind_root,
            output_dir,
            args,
        )
        for split in args.splits
    ]
    summary = {
        "formatVersion": 1,
        "task": "ZInD matching dataset conversion summary",
        "zindRoot": str(zind_root),
        "partition": str(partition_path),
        "tokenCount": args.token_count,
        "endpointTolerance": args.endpoint_tolerance,
        "negativeRatio": args.negative_ratio,
        "splits": summaries,
        "totalPairs": sum(item["pairs"] for item in summaries),
        "totalPositivePairs": sum(item["positivePairs"] for item in summaries),
        "totalNegativePairs": sum(item["negativePairs"] for item in summaries),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

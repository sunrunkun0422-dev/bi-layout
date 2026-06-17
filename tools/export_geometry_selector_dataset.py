#!/usr/bin/env python3
"""Export a formal geometry-selector dataset from Bi-Layout predictions.

Unlike the visualization-proxy experiment, this script runs the Bi-Layout model
and extracts features directly from predicted depth/ratio sequences. Oracle
labels are computed from per-sample full_2d/full_3d metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config.defaults import get_config
from dataset.build import build_dataset
from evaluation.accuracy import calc_accuracy
from models.build import build_model
from utils.conversion import depth2xyz
from utils.misc import tensor2np_d


class SimpleLogger:
    def info(self, msg: object) -> None:
        print(msg)

    def warning(self, msg: object) -> None:
        print(f"WARNING: {msg}")

    def error(self, msg: object) -> None:
        print(f"ERROR: {msg}")


class ConfigArgs(SimpleNamespace):
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


META_COLUMNS = [
    "sample_id",
    "split",
    "original_full_2d",
    "original_full_3d",
    "new_as_origin_full_2d",
    "new_as_origin_full_3d",
    "label_full_2d",
    "label_full_3d",
    "oracle_full_2d",
    "oracle_full_3d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="src/config/mp3d_test_o0.yaml")
    parser.add_argument("--mode", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt-option", default="best", choices=["last", "best", "oracle", "average"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset size for smoke tests.")
    parser.add_argument("--output", default="geometry_selector_formal/test_selector_dataset.csv")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    cfg_args = ConfigArgs(
        cfg=args.cfg,
        mode=args.mode,
        debug=False,
        hidden_bar=True,
        bs=args.batch_size,
        for_test_index=args.limit,
        device=args.device,
        ckpt_option=args.ckpt_option,
    )
    config = get_config(cfg_args)
    config.defrost()
    config.DATA.NUM_WORKERS = args.num_workers
    config.DATA.BATCH_SIZE = args.batch_size
    config.SHOW_BAR = False
    config.freeze()
    return config


def disable_pretrained_encoder_download() -> None:
    """Avoid torchvision network downloads; the Bi-Layout checkpoint is loaded next."""
    from models.modules import horizon_net_feature_extractor as hnf

    def resnet_init(self, backbone: str = "resnet50", pretrained: bool = True) -> None:
        hnf.nn.Module.__init__(self)
        assert backbone in hnf.ENCODER_RESNET
        try:
            self.encoder = getattr(hnf.models, backbone)(weights=None)
        except TypeError:
            self.encoder = getattr(hnf.models, backbone)(pretrained=False)
        del self.encoder.fc, self.encoder.avgpool

    hnf.Resnet.__init__ = resnet_init


def as_scalar(value: np.ndarray | float) -> float:
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def robust_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def cyclic_diff(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.diff(np.concatenate([values, values[:1]]))


def safe_polygon(xz: np.ndarray) -> Polygon | None:
    try:
        poly = Polygon(np.asarray(xz, dtype=np.float64))
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def polygon_iou(poly_a: Polygon | None, poly_b: Polygon | None) -> float:
    if poly_a is None or poly_b is None:
        return 0.0
    try:
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
        if not poly_b.is_valid:
            poly_b = poly_b.buffer(0)
        union = poly_a.union(poly_b).area
        if union <= 0:
            return 0.0
        return float(poly_a.intersection(poly_b).area / union)
    except Exception:
        return 0.0


def polygon_features(xz: np.ndarray, prefix: str) -> Tuple[Dict[str, float], Polygon | None]:
    feats: Dict[str, float] = {}
    poly = safe_polygon(xz)
    if poly is None:
        feats.update({
            f"{prefix}_poly_valid": 0.0,
            f"{prefix}_poly_area": 0.0,
            f"{prefix}_poly_perimeter": 0.0,
            f"{prefix}_poly_hull_area_ratio": 0.0,
            f"{prefix}_poly_minx": 0.0,
            f"{prefix}_poly_miny": 0.0,
            f"{prefix}_poly_width": 0.0,
            f"{prefix}_poly_height": 0.0,
            f"{prefix}_poly_centroid_radius": 0.0,
        })
        return feats, None

    area = float(abs(poly.area))
    perimeter = float(poly.length)
    hull_area = float(poly.convex_hull.area) if not poly.convex_hull.is_empty else 0.0
    minx, miny, maxx, maxy = poly.bounds
    centroid = poly.centroid
    feats.update({
        f"{prefix}_poly_valid": float(poly.is_valid),
        f"{prefix}_poly_area": area,
        f"{prefix}_poly_perimeter": perimeter,
        f"{prefix}_poly_hull_area_ratio": float(area / (hull_area + 1e-8)),
        f"{prefix}_poly_minx": float(minx),
        f"{prefix}_poly_miny": float(miny),
        f"{prefix}_poly_width": float(maxx - minx),
        f"{prefix}_poly_height": float(maxy - miny),
        f"{prefix}_poly_centroid_radius": float(math.sqrt(centroid.x ** 2 + centroid.y ** 2)),
    })
    return feats, poly


def layout_features(depth: np.ndarray, ratio: float, prefix: str) -> Tuple[Dict[str, float], Polygon | None]:
    raw_depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    abs_depth = np.abs(raw_depth)
    xyz = depth2xyz(abs_depth)
    xz = xyz[..., ::2]
    radius = np.linalg.norm(xz, axis=-1)
    d1 = cyclic_diff(abs_depth)
    d2 = cyclic_diff(d1)
    abs_d1 = np.abs(d1)
    abs_d2 = np.abs(d2)
    peak_threshold = abs_d1.mean() + abs_d1.std()

    feats: Dict[str, float] = {
        f"{prefix}_ratio": float(ratio),
        f"{prefix}_height": float(1.0 + ratio),
        f"{prefix}_depth_negative_frac": float((raw_depth < 0).mean()),
        f"{prefix}_depth_nonfinite_frac": float((~np.isfinite(raw_depth)).mean()),
        f"{prefix}_boundary_jump_count": float((abs_d1 > peak_threshold).sum()),
        f"{prefix}_boundary_jump_frac": float((abs_d1 > peak_threshold).mean()),
        f"{prefix}_boundary_jump_threshold": float(peak_threshold),
    }
    feats.update(robust_stats(abs_depth, f"{prefix}_depth"))
    feats.update(robust_stats(abs_d1, f"{prefix}_depth_grad1_abs"))
    feats.update(robust_stats(abs_d2, f"{prefix}_depth_grad2_abs"))
    feats.update(robust_stats(radius, f"{prefix}_radius"))

    poly_feats, poly = polygon_features(xz, prefix)
    feats.update(poly_feats)
    return feats, poly


def pair_features(
    original_depth: np.ndarray,
    new_depth: np.ndarray,
    original_feats: Dict[str, float],
    new_feats: Dict[str, float],
    original_poly: Polygon | None,
    new_poly: Polygon | None,
) -> Dict[str, float]:
    original_abs = np.abs(np.asarray(original_depth, dtype=np.float64).reshape(-1))
    new_abs = np.abs(np.asarray(new_depth, dtype=np.float64).reshape(-1))
    diff = new_abs - original_abs
    feats: Dict[str, float] = {}
    feats.update(robust_stats(np.abs(diff), "branch_depth_absdiff"))
    feats.update(robust_stats(diff, "branch_depth_signeddiff"))
    feats["branch_depth_l1"] = float(np.mean(np.abs(diff)))
    feats["branch_depth_l2"] = float(np.sqrt(np.mean(diff ** 2)))
    if original_abs.std() > 1e-8 and new_abs.std() > 1e-8:
        feats["branch_depth_corr"] = float(np.corrcoef(original_abs, new_abs)[0, 1])
    else:
        feats["branch_depth_corr"] = 0.0

    for raw_name in [
        "poly_area",
        "poly_perimeter",
        "poly_hull_area_ratio",
        "poly_width",
        "poly_height",
        "poly_centroid_radius",
        "depth_grad1_abs_mean",
        "depth_grad1_abs_max",
        "depth_grad2_abs_mean",
        "boundary_jump_count",
        "boundary_jump_frac",
        "radius_mean",
        "radius_std",
        "radius_max",
    ]:
        old_key = f"original_{raw_name}"
        new_key = f"new_{raw_name}"
        if old_key in original_feats and new_key in new_feats:
            old = original_feats[old_key]
            new = new_feats[new_key]
            feats[f"branch_diff_{raw_name}"] = float(new - old)
            feats[f"branch_absdiff_{raw_name}"] = float(abs(new - old))
            feats[f"branch_ratio_{raw_name}"] = float(new / (old + 1e-8))

    feats["branch_polygon_iou"] = polygon_iou(original_poly, new_poly)
    return feats


def build_row(sample_id: str, split: str, dt_np: Dict[str, np.ndarray], gt_np: Dict[str, np.ndarray], idx: int,
              original_metrics: Tuple[List[float], List[float]],
              new_metrics: Tuple[List[float], List[float]]) -> Dict[str, float | str | int]:
    original_full_2d, original_full_3d = original_metrics[0][idx], original_metrics[1][idx]
    new_full_2d, new_full_3d = new_metrics[0][idx], new_metrics[1][idx]

    original_depth = dt_np["depth"][idx]
    new_depth = dt_np["new_depth"][idx]
    ratio = as_scalar(dt_np["ratio"][idx])
    original_feats, original_poly = layout_features(original_depth, ratio, "original")
    new_feats, new_poly = layout_features(new_depth, ratio, "new")
    branch_feats = pair_features(original_depth, new_depth, original_feats, new_feats, original_poly, new_poly)

    row: Dict[str, float | str | int] = {
        "sample_id": sample_id,
        "split": split,
        "original_full_2d": float(original_full_2d),
        "original_full_3d": float(original_full_3d),
        "new_as_origin_full_2d": float(new_full_2d),
        "new_as_origin_full_3d": float(new_full_3d),
        "label_full_2d": int(new_full_2d >= original_full_2d),
        "label_full_3d": int(new_full_3d >= original_full_3d),
        "oracle_full_2d": float(max(original_full_2d, new_full_2d)),
        "oracle_full_3d": float(max(original_full_3d, new_full_3d)),
    }
    row.update(original_feats)
    row.update(new_feats)
    row.update(branch_feats)
    return row


def summarize(rows: List[Dict[str, float | str | int]]) -> Dict[str, float | int]:
    original_2d = np.array([r["original_full_2d"] for r in rows], dtype=np.float64)
    original_3d = np.array([r["original_full_3d"] for r in rows], dtype=np.float64)
    new_2d = np.array([r["new_as_origin_full_2d"] for r in rows], dtype=np.float64)
    new_3d = np.array([r["new_as_origin_full_3d"] for r in rows], dtype=np.float64)
    oracle_2d = np.maximum(original_2d, new_2d)
    oracle_3d = np.maximum(original_3d, new_3d)
    labels_3d = np.array([r["label_full_3d"] for r in rows], dtype=np.int64)
    return {
        "n": int(len(rows)),
        "original_full_2d": float(original_2d.mean()),
        "original_full_3d": float(original_3d.mean()),
        "new_as_origin_full_2d": float(new_2d.mean()),
        "new_as_origin_full_3d": float(new_3d.mean()),
        "oracle_full_2d": float(oracle_2d.mean()),
        "oracle_full_3d": float(oracle_3d.mean()),
        "label_full_3d_new_count": int(labels_3d.sum()),
        "label_full_3d_original_count": int((labels_3d == 0).sum()),
        "original_low_iou_2d_rate": float((original_2d < 0.5).mean()),
        "new_low_iou_2d_rate": float((new_2d < 0.5).mean()),
        "oracle_low_iou_2d_rate": float((oracle_2d < 0.5).mean()),
    }


def write_csv(path: Path, rows: List[Dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(META_COLUMNS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = SimpleLogger()
    config = make_config(args)

    dataset = build_dataset(args.mode, config, logger)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    disable_pretrained_encoder_download()
    model, _, _, _ = build_model(config, logger)
    model.eval()

    rows: List[Dict[str, float | str | int]] = []
    device = torch.device(args.device)
    with torch.no_grad():
        for gt in tqdm(loader, desc=f"export-{args.mode}", ncols=100):
            imgs = gt["image"].to(device, non_blocking=False)
            dt = model(imgs)
            gt_np = tensor2np_d(gt)
            dt_np = tensor2np_d(dt)

            _, full_iou, _, _, original_full_2ds, original_full_3ds, _ = calc_accuracy(
                dt_np, gt_np, visualization=False
            )
            _, new_full_iou, _, _, new_full_2ds, new_full_3ds, _ = calc_accuracy(
                dt_np, gt_np, visualization=False, second_type=True, gt_label="origin"
            )

            sample_ids = gt_np["id"]
            for idx, sample_id in enumerate(sample_ids):
                row = build_row(
                    sample_id=sample_id,
                    split=args.mode,
                    dt_np=dt_np,
                    gt_np=gt_np,
                    idx=idx,
                    original_metrics=(original_full_2ds, original_full_3ds),
                    new_metrics=(new_full_2ds, new_full_3ds),
                )
                rows.append(row)

    output_path = Path(args.output)
    write_csv(output_path, rows)
    summary = summarize(rows)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

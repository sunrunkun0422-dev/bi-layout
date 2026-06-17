#!/usr/bin/env python3
"""Run a lightweight geometry-selector experiment from saved eval visualizations.

This script is a fast proxy experiment. It uses the saved Bi-Layout test
visualizations under checkpoints/.../results/test_best and extracts only
prediction-colored geometry cues from the images. The IoU embedded in filenames
is used as the oracle selector label and for evaluation, not as an input
feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_RE = re.compile(r"^(?P<sample_id>.+)_(?P<iou>\d+\.\d+)$")


@dataclass
class PredictionImage:
    sample_id: str
    iou_2d: float
    path: Path
    mtime: float


def parse_prediction_dir(path: Path) -> Dict[str, PredictionImage]:
    by_id: Dict[str, List[PredictionImage]] = {}
    for img_path in path.glob("*.png"):
        match = ID_RE.match(img_path.stem)
        if not match:
            continue
        pred = PredictionImage(
            sample_id=match.group("sample_id"),
            iou_2d=float(match.group("iou")),
            path=img_path,
            mtime=img_path.stat().st_mtime,
        )
        by_id.setdefault(pred.sample_id, []).append(pred)

    newest = {}
    for sample_id, items in by_id.items():
        newest[sample_id] = sorted(items, key=lambda item: item.mtime)[-1]
    return newest


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def prediction_green_mask(rgb: np.ndarray) -> np.ndarray:
    """Extract the prediction overlay.

    In Bi-Layout visualizations, GT is drawn blue and prediction is drawn green.
    The threshold is intentionally conservative to avoid using blue GT pixels.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (g > 145) & (r < 110) & (b < 130)


def safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def mask_features(mask: np.ndarray, prefix: str) -> Dict[str, float]:
    h, w = mask.shape
    mask_u8 = mask.astype(np.uint8)
    area = int(mask_u8.sum())
    feats: Dict[str, float] = {
        f"{prefix}_area_frac": area / float(h * w),
    }

    ys, xs = np.where(mask)
    if area == 0:
        feats.update({
            f"{prefix}_bbox_w": 0.0,
            f"{prefix}_bbox_h": 0.0,
            f"{prefix}_bbox_area": 0.0,
            f"{prefix}_x_coverage": 0.0,
            f"{prefix}_y_coverage": 0.0,
            f"{prefix}_component_count": 0.0,
            f"{prefix}_largest_component_frac": 0.0,
            f"{prefix}_contour_count": 0.0,
            f"{prefix}_contour_area_frac": 0.0,
            f"{prefix}_contour_perimeter_norm": 0.0,
            f"{prefix}_boundary_jump_mean": 0.0,
            f"{prefix}_boundary_jump_max": 0.0,
        })
        feats.update(safe_stats(np.array([]), f"{prefix}_col_count"))
        feats.update(safe_stats(np.array([]), f"{prefix}_row_count"))
        return feats

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_w = (x_max - x_min + 1) / float(w)
    bbox_h = (y_max - y_min + 1) / float(h)
    feats.update({
        f"{prefix}_bbox_w": float(bbox_w),
        f"{prefix}_bbox_h": float(bbox_h),
        f"{prefix}_bbox_area": float(bbox_w * bbox_h),
        f"{prefix}_x_coverage": float(np.unique(xs).size / w),
        f"{prefix}_y_coverage": float(np.unique(ys).size / h),
    })

    labels, stats = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)[1:3]
    component_areas = stats[1:, cv2.CC_STAT_AREA] if stats.shape[0] > 1 else np.array([])
    feats[f"{prefix}_component_count"] = float(component_areas.size)
    feats[f"{prefix}_largest_component_frac"] = float(component_areas.max() / area) if component_areas.size else 0.0

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    contour_perimeters = np.array([cv2.arcLength(c, True) for c in contours], dtype=np.float32)
    feats[f"{prefix}_contour_count"] = float(len(contours))
    feats[f"{prefix}_contour_area_frac"] = float(contour_areas.sum() / (h * w)) if contour_areas.size else 0.0
    feats[f"{prefix}_contour_perimeter_norm"] = float(contour_perimeters.sum() / (h + w)) if contour_perimeters.size else 0.0

    col_counts = mask_u8.sum(axis=0).astype(np.float32) / h
    row_counts = mask_u8.sum(axis=1).astype(np.float32) / w
    feats.update(safe_stats(col_counts[col_counts > 0], f"{prefix}_col_count"))
    feats.update(safe_stats(row_counts[row_counts > 0], f"{prefix}_row_count"))

    # For each occupied x column, estimate the centerline y. Large jumps indicate
    # jagged or discontinuous layout boundaries.
    occupied_x = np.unique(xs)
    if occupied_x.size > 1:
        y_by_x = np.array([ys[xs == x].mean() for x in occupied_x], dtype=np.float32) / h
        jumps = np.abs(np.diff(y_by_x))
        feats[f"{prefix}_boundary_jump_mean"] = float(jumps.mean()) if jumps.size else 0.0
        feats[f"{prefix}_boundary_jump_max"] = float(jumps.max()) if jumps.size else 0.0
    else:
        feats[f"{prefix}_boundary_jump_mean"] = 0.0
        feats[f"{prefix}_boundary_jump_max"] = 0.0

    return feats


def image_features(path: Path, prefix: str) -> Dict[str, float]:
    rgb = load_rgb(path)
    # Saved visualization is [panorama 1024px | floorplan 512px].
    pano = rgb[:, :1024]
    floor = rgb[:, 1024:]
    full_mask = prediction_green_mask(rgb)
    pano_mask = prediction_green_mask(pano)
    floor_mask = prediction_green_mask(floor)

    feats: Dict[str, float] = {}
    feats.update(mask_features(full_mask, f"{prefix}_full"))
    feats.update(mask_features(pano_mask, f"{prefix}_pano"))
    feats.update(mask_features(floor_mask, f"{prefix}_floor"))
    return feats


def ratio(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / (b + eps))


def pair_features(ext_feats: Dict[str, float], enc_feats: Dict[str, float]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for key, ext_val in ext_feats.items():
        raw = key[len("extended_"):] if key.startswith("extended_") else key
        enc_key = "enclosed_" + raw
        if enc_key not in enc_feats:
            continue
        enc_val = enc_feats[enc_key]
        feats[f"diff_{raw}"] = float(enc_val - ext_val)
        feats[f"absdiff_{raw}"] = float(abs(enc_val - ext_val))
        feats[f"ratio_{raw}"] = ratio(enc_val, ext_val)
    return feats


def build_dataset(root: Path) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    extended = parse_prediction_dir(root / "extended_results")
    enclosed = parse_prediction_dir(root / "enclosed_results")
    sample_ids = sorted(set(extended).intersection(enclosed))

    rows: List[Dict[str, float]] = []
    meta: List[Dict[str, object]] = []
    for sample_id in sample_ids:
        ext = extended[sample_id]
        enc = enclosed[sample_id]
        ext_feats = image_features(ext.path, "extended")
        enc_feats = image_features(enc.path, "enclosed")
        feats: Dict[str, float] = {}
        feats.update(ext_feats)
        feats.update(enc_feats)
        feats.update(pair_features(ext_feats, enc_feats))
        rows.append(feats)
        meta.append({
            "sample_id": sample_id,
            "extended_iou_2d": ext.iou_2d,
            "enclosed_iou_2d": enc.iou_2d,
            "oracle_label": int(enc.iou_2d >= ext.iou_2d),
            "extended_path": str(ext.path),
            "enclosed_path": str(enc.path),
        })
    return rows, meta


def matrix_from_rows(rows: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    feature_names = sorted(rows[0].keys())
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, feature_names


def evaluate_predictions(meta: List[Dict[str, object]], pred: np.ndarray) -> Dict[str, float]:
    ext_iou = np.array([row["extended_iou_2d"] for row in meta], dtype=np.float32)
    enc_iou = np.array([row["enclosed_iou_2d"] for row in meta], dtype=np.float32)
    labels = np.array([row["oracle_label"] for row in meta], dtype=np.int64)
    selected = np.where(pred == 1, enc_iou, ext_iou)
    best_fixed_pred = np.ones_like(labels) if enc_iou.mean() >= ext_iou.mean() else np.zeros_like(labels)
    best_fixed = enc_iou if enc_iou.mean() >= ext_iou.mean() else ext_iou
    oracle = np.maximum(ext_iou, enc_iou)
    return {
        "n": int(labels.size),
        "extended_full_2d_mean": float(ext_iou.mean()),
        "enclosed_full_2d_mean": float(enc_iou.mean()),
        "best_fixed_full_2d_mean": float(best_fixed.mean()),
        "selector_full_2d_mean": float(selected.mean()),
        "oracle_full_2d_mean": float(oracle.mean()),
        "selector_accuracy": float(accuracy_score(labels, pred)),
        "best_fixed_accuracy": float(accuracy_score(labels, best_fixed_pred)),
        "oracle_gap_captured": float((selected.mean() - best_fixed.mean()) / max(oracle.mean() - best_fixed.mean(), 1e-8)),
        "extended_low_iou_rate": float((ext_iou < 0.5).mean()),
        "enclosed_low_iou_rate": float((enc_iou < 0.5).mean()),
        "selector_low_iou_rate": float((selected < 0.5).mean()),
        "oracle_low_iou_rate": float((oracle < 0.5).mean()),
        "enclosed_better_count": int(labels.sum()),
        "extended_better_count": int((labels == 0).sum()),
    }


def save_csv(path: Path, rows: List[Dict[str, float]], meta: List[Dict[str, object]], feature_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "extended_iou_2d",
        "enclosed_iou_2d",
        "oracle_label",
        *feature_names,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat_row, meta_row in zip(rows, meta):
            row = {name: meta_row[name] for name in fieldnames[:4]}
            row.update({name: feat_row[name] for name in feature_names})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="checkpoints/Bi_Layout_Net/mp3d/results/test_best")
    parser.add_argument("--output-dir", default="geometry_selector_experiment")
    parser.add_argument("--model", choices=["random_forest", "logreg"], default="random_forest")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    rows, meta = build_dataset(result_root)
    if not rows:
        raise RuntimeError(f"No paired samples found in {result_root}")
    x, feature_names = matrix_from_rows(rows)
    y = np.array([row["oracle_label"] for row in meta], dtype=np.int64)

    if args.model == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
        )

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    pred = cross_val_predict(estimator, x, y, cv=cv, method="predict")
    metrics = evaluate_predictions(meta, pred)
    metrics["model"] = args.model
    metrics["folds"] = args.folds
    metrics["feature_count"] = len(feature_names)
    metrics["confusion_matrix"] = confusion_matrix(y, pred).tolist()
    metrics["note"] = (
        "Fast visualization-proxy experiment. Features are extracted from prediction-colored "
        "green overlays in saved eval images. Filename IoU is used only for labels/evaluation."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "selector_dataset.csv", rows, meta, feature_names)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Train once on all data only for feature importance diagnostics.
    estimator.fit(x, y)
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "named_steps") and "logisticregression" in estimator.named_steps:
        importances = np.abs(estimator.named_steps["logisticregression"].coef_[0])
    if importances is not None:
        top = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:30]
        with (output_dir / "feature_importance.json").open("w") as f:
            json.dump([{"feature": name, "importance": float(value)} for name, value in top], f, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

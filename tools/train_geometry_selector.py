#!/usr/bin/env python3
"""Train/evaluate a geometry-aware branch selector from exported CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV for CV mode, or test CSV when --train-csv is set.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--label", default="label_full_3d", choices=["label_full_2d", "label_full_3d"])
    parser.add_argument("--metric", default="full_3d", choices=["full_2d", "full_3d"])
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logreg"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for switching away from the best fixed branch.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="geometry_selector_formal")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def feature_names(rows: Sequence[Dict[str, str]]) -> List[str]:
    names = []
    for key in rows[0].keys():
        if key in NON_FEATURE_COLUMNS:
            continue
        names.append(key)
    return names


def to_matrix(rows: Sequence[Dict[str, str]], names: Sequence[str]) -> np.ndarray:
    values = []
    for row in rows:
        values.append([float(row.get(name, 0.0) or 0.0) for name in names])
    x = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def labels(rows: Sequence[Dict[str, str]], label_col: str) -> np.ndarray:
    return np.asarray([int(float(row[label_col])) for row in rows], dtype=np.int64)


def metric_arrays(rows: Sequence[Dict[str, str]], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    if metric == "full_2d":
        original_col = "original_full_2d"
        new_col = "new_as_origin_full_2d"
    else:
        original_col = "original_full_3d"
        new_col = "new_as_origin_full_3d"
    original = np.asarray([float(row[original_col]) for row in rows], dtype=np.float64)
    new = np.asarray([float(row[new_col]) for row in rows], dtype=np.float64)
    return original, new


def confidence_predictions(rows: Sequence[Dict[str, str]], metric: str, prob_new: np.ndarray,
                           threshold: float) -> np.ndarray:
    original, new = metric_arrays(rows, metric)
    default_new = bool(new.mean() >= original.mean())
    if default_new:
        return (prob_new >= (1.0 - threshold)).astype(np.int64)
    return (prob_new >= threshold).astype(np.int64)


def make_estimator(name: str, seed: int):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def evaluate(rows: Sequence[Dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> Dict[str, object]:
    original, new = metric_arrays(rows, metric)
    selected = np.where(y_pred == 1, new, original)
    oracle = np.maximum(original, new)
    best_fixed_new = bool(new.mean() >= original.mean())
    best_fixed = new if best_fixed_new else original
    best_fixed_pred = np.ones_like(y_true) if best_fixed_new else np.zeros_like(y_true)
    oracle_gap = max(float(oracle.mean() - best_fixed.mean()), 1e-8)
    return {
        "n": int(len(rows)),
        "metric": metric,
        "original_mean": float(original.mean()),
        "new_mean": float(new.mean()),
        "best_fixed_branch": "new" if best_fixed_new else "original",
        "best_fixed_mean": float(best_fixed.mean()),
        "selector_mean": float(selected.mean()),
        "oracle_mean": float(oracle.mean()),
        "selector_gain_vs_best_fixed": float(selected.mean() - best_fixed.mean()),
        "oracle_gap_captured": float((selected.mean() - best_fixed.mean()) / oracle_gap),
        "selector_accuracy": float(accuracy_score(y_true, y_pred)),
        "best_fixed_accuracy": float(accuracy_score(y_true, best_fixed_pred)),
        "original_low_iou_rate": float((original < 0.5).mean()),
        "new_low_iou_rate": float((new < 0.5).mean()),
        "best_fixed_low_iou_rate": float((best_fixed < 0.5).mean()),
        "selector_low_iou_rate": float((selected < 0.5).mean()),
        "oracle_low_iou_rate": float((oracle < 0.5).mean()),
        "new_better_count": int(y_true.sum()),
        "original_better_count": int((y_true == 0).sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def write_predictions(path: Path, rows: Sequence[Dict[str, str]], pred: np.ndarray,
                      prob_new: np.ndarray, metric: str) -> None:
    original, new = metric_arrays(rows, metric)
    selected = np.where(pred == 1, new, original)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "sample_id",
            "prediction",
            "prob_new",
            f"selected_{metric}",
            f"original_{metric}",
            f"new_{metric}",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, p, prob, s, o, n in zip(rows, pred, prob_new, selected, original, new):
            writer.writerow({
                "sample_id": row["sample_id"],
                "prediction": "new" if p == 1 else "original",
                "prob_new": prob,
                f"selected_{metric}": s,
                f"original_{metric}": o,
                f"new_{metric}": n,
            })


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_rows = read_rows(Path(args.csv))
    names = feature_names(test_rows)
    estimator = make_estimator(args.model, args.seed)

    if args.train_csv:
        train_rows = read_rows(Path(args.train_csv))
        train_names = feature_names(train_rows)
        names = [name for name in names if name in set(train_names)]
        x_train = to_matrix(train_rows, names)
        y_train = labels(train_rows, args.label)
        estimator.fit(x_train, y_train)
        x_test = to_matrix(test_rows, names)
        y_test = labels(test_rows, args.label)
        prob_new = estimator.predict_proba(x_test)[:, 1]
        pred = confidence_predictions(test_rows, args.metric, prob_new, args.decision_threshold)
        mode = "train_test"
    else:
        x_test = to_matrix(test_rows, names)
        y_test = labels(test_rows, args.label)
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        prob_new = cross_val_predict(estimator, x_test, y_test, cv=cv, method="predict_proba")[:, 1]
        pred = confidence_predictions(test_rows, args.metric, prob_new, args.decision_threshold)
        estimator.fit(x_test, y_test)
        mode = "cross_val"

    metrics = evaluate(test_rows, y_test, pred, args.metric)
    metrics.update({
        "mode": mode,
        "model": args.model,
        "label": args.label,
        "folds": args.folds if not args.train_csv else None,
        "feature_count": len(names),
        "decision_threshold": args.decision_threshold,
        "mean_prob_new": float(prob_new.mean()),
    })

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_predictions(output_dir / "predictions.csv", test_rows, pred, prob_new, args.metric)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "named_steps") and "logisticregression" in estimator.named_steps:
        importances = np.abs(estimator.named_steps["logisticregression"].coef_[0])
    else:
        importances = None
    if importances is not None:
        top = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:40]
        with (output_dir / "feature_importance.json").open("w") as f:
            json.dump([{"feature": name, "importance": float(value)} for name, value in top], f, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

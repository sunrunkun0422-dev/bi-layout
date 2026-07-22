#!/usr/bin/env python3
"""Cache frozen Bi-Layout outputs and train the ZInD single-view Opening Head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_opening_dataset import (
    OpeningFeatureCacheDataset,
    ZInDOpeningViewDataset,
)
from evaluation.opening_recall import evaluate_opening_scores
from models.cross_scene_matcher import (
    OpeningSignalHead,
    opening_detection_loss,
    resolve_enclosed_extended_depth,
)
from tools.debug_cross_scene_flow import load_bi_layout


DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "zind/ZInD-BiPair-v1"
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"
DEFAULT_BACKBONE_CHECKPOINT = (
    REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints/Opening_Head/zind_bipair_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--train_manifest")
    parser.add_argument("--val_manifest")
    parser.add_argument("--data_root")
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--bi_layout_checkpoint", default=str(DEFAULT_BACKBONE_CHECKPOINT)
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache_dir")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="resume from a checkpoint path, or from output_dir/last.pt when omitted",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=2.5)
    parser.add_argument("--tversky_weight", type=float, default=0.5)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--prior_strength", type=float, default=4.0)
    parser.add_argument("--prior_relative_scale", type=float, default=0.1)
    parser.add_argument("--roll_probability", type=float, default=1.0)
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--precision_target", type=float, default=0.80)
    parser.add_argument("--scan_min", type=float, default=0.0)
    parser.add_argument("--scan_max", type=float, default=1.0)
    parser.add_argument("--scan_steps", type=int, default=201)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--max_train_views", type=int)
    parser.add_argument("--max_val_views", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.cache_batch_size,
        args.lr,
        args.pos_weight,
        args.tversky_alpha,
        args.tversky_beta,
        args.hidden_dim,
        args.kernel_size,
        args.prior_relative_scale,
        args.scan_steps,
        args.torch_threads,
        args.image_height,
        args.image_width,
        args.progress_every,
    )
    if min(positive) <= 0:
        raise ValueError("positive training, model, and data arguments must be > 0")
    if args.workers < 0 or args.weight_decay < 0 or args.tversky_weight < 0:
        raise ValueError("workers, weight_decay, and tversky_weight must be non-negative")
    if args.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if not 0.0 <= args.roll_probability <= 1.0:
        raise ValueError("roll_probability must be in [0, 1]")
    if not 0.0 <= args.flip_probability <= 1.0:
        raise ValueError("flip_probability must be in [0, 1]")
    if not 0.0 <= args.precision_target <= 1.0:
        raise ValueError("precision_target must be in [0, 1]")
    if args.scan_min >= args.scan_max:
        raise ValueError("scan_min must be lower than scan_max")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    for value in (args.max_train_views, args.max_val_views):
        if value is not None and value <= 0:
            raise ValueError("max view limits must be positive")


def resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this process; use --device cpu "
            "or run the command with GPU device access"
        )
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(path))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, hash_contents: bool = True) -> Dict[str, Any]:
    stat = path.stat()
    identity = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_contents:
        identity["sha256"] = _sha256(path)
    return identity


def _cache_contract(
    dataset: ZInDOpeningViewDataset,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    feature_dim: int,
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "task": "ZInD-BiPair-v1 frozen Bi-Layout opening feature cache",
        "manifest": _file_identity(manifest_path, hash_contents=True),
        "bi_layout_config": _file_identity(config_path, hash_contents=True),
        # Hashing a 1.2 GiB checkpoint on every resume is unnecessary; size and
        # nanosecond mtime are enough to reject an accidentally changed file.
        "bi_layout_checkpoint": _file_identity(
            checkpoint_path, hash_contents=False
        ),
        "image_shape": list(image_shape),
        "branch_order": "extended_first",
        "sample_count": len(dataset),
        "token_count": int(dataset.token_count),
        "feature_dim": int(feature_dim),
        "feature_dtype": "float16",
        "depth_dtype": "float32",
        "target_dtype": "uint8",
    }


def _same_cache_contract(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    ignored = {"complete", "completed_count", "runtime_seconds", "views"}
    left = {key: value for key, value in first.items() if key not in ignored}
    right = {key: value for key, value in second.items() if key not in ignored}
    return left == right


def _serialize_views(dataset: ZInDOpeningViewDataset) -> Sequence[Dict[str, Any]]:
    records = []
    for view in dataset.views:
        if hasattr(view, "__dict__"):
            records.append(dict(view.__dict__))
        elif isinstance(view, Mapping):
            records.append(dict(view))
        else:
            records.append({"image_path": str(view)})
    return records


def _initialize_cache(
    cache_dir: Path,
    contract: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    rebuild: bool,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    metadata = None
    if metadata_path.is_file() and not rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _same_cache_contract(metadata, contract):
            raise ValueError(
                f"cache contract changed at {cache_dir}; pass --rebuild_cache"
            )

    count = int(contract["sample_count"])
    tokens = int(contract["token_count"])
    channels = int(contract["feature_dim"])
    specs = {
        "features": (np.float16, (count, tokens, channels)),
        "enclosed_depth": (np.float32, (count, tokens)),
        "extended_depth": (np.float32, (count, tokens)),
        "targets": (np.uint8, (count, tokens)),
        "completed": (np.uint8, (count,)),
    }
    arrays: Dict[str, np.ndarray] = {}
    create = metadata is None or rebuild
    for name, (dtype, shape) in specs.items():
        path = cache_dir / f"{name}.npy"
        mode = "w+" if create else "r+"
        arrays[name] = np.lib.format.open_memmap(
            str(path), mode=mode, dtype=dtype, shape=shape
        )
    if create:
        arrays["completed"][:] = 0
        metadata = {
            **dict(contract),
            "complete": False,
            "completed_count": 0,
            "views": list(views),
        }
        _atomic_write_json(metadata_path, metadata)
    return dict(metadata), arrays


def build_feature_cache(
    dataset: ZInDOpeningViewDataset,
    cache_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    contract = _cache_contract(
        dataset,
        manifest_path,
        config_path,
        checkpoint_path,
        feature_dim=int(model.patch_dim),
        image_shape=(args.image_height, args.image_width),
    )
    metadata, arrays = _initialize_cache(
        cache_dir,
        contract,
        _serialize_views(dataset),
        rebuild=args.rebuild_cache,
    )
    if bool(metadata.get("complete")) and int(arrays["completed"].sum()) == len(dataset):
        print(f"cache already complete: {cache_dir} ({len(dataset)} views)")
        return metadata

    pending = np.flatnonzero(np.asarray(arrays["completed"]) == 0).tolist()
    subset = Subset(dataset, pending)
    loader = DataLoader(
        subset,
        batch_size=args.cache_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    model.requires_grad_(False)
    start = time.perf_counter()
    processed = int(arrays["completed"].sum())
    for batch in loader:
        indices = batch["index"].cpu().numpy().astype(np.int64)
        images = batch["image"].to(
            device, dtype=torch.float32, non_blocking=device.type == "cuda"
        )
        with torch.no_grad():
            output = model(images, return_features=True)
            enclosed, extended = resolve_enclosed_extended_depth(output)
        arrays["features"][indices] = (
            output["layout_feature"].detach().cpu().numpy().astype(np.float16)
        )
        arrays["enclosed_depth"][indices] = (
            enclosed.detach().cpu().numpy().astype(np.float32)
        )
        arrays["extended_depth"][indices] = (
            extended.detach().cpu().numpy().astype(np.float32)
        )
        arrays["targets"][indices] = (
            batch["target"].cpu().numpy().astype(np.uint8)
        )
        arrays["completed"][indices] = 1
        processed += len(indices)
        if processed % args.progress_every < len(indices) or processed == len(dataset):
            for value in arrays.values():
                value.flush()
            elapsed = time.perf_counter() - start
            metadata.update(
                complete=False,
                completed_count=int(arrays["completed"].sum()),
                runtime_seconds=float(metadata.get("runtime_seconds", 0.0)) + elapsed,
            )
            _atomic_write_json(cache_dir / "metadata.json", metadata)
            print(f"cache {cache_dir.name}: [{processed}/{len(dataset)}]")
            start = time.perf_counter()

    for value in arrays.values():
        value.flush()
    completed_count = int(arrays["completed"].sum())
    if completed_count != len(dataset):
        raise RuntimeError(
            f"cache incomplete at {cache_dir}: {completed_count}/{len(dataset)}"
        )
    metadata.update(complete=True, completed_count=completed_count)
    _atomic_write_json(cache_dir / "metadata.json", metadata)
    print(f"cache complete: {cache_dir} ({completed_count} views)")
    return metadata


def select_validation_threshold(
    targets: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    thresholds = np.linspace(args.scan_min, args.scan_max, args.scan_steps)
    scan_evaluation = evaluate_opening_scores(
        targets,
        scores,
        thresholds=thresholds,
        precision_target=args.precision_target,
    )
    scan = scan_evaluation["thresholdSelection"]
    selected = scan["precisionTargetMaxRecall"]
    fallback = selected is None
    if fallback:
        selected = scan["bestF1"]
    threshold = float(selected["threshold"])
    fixed = evaluate_opening_scores(targets, scores, threshold=threshold)
    fixed["averagePrecision"] = scan_evaluation["averagePrecision"]
    fixed["metricsAtSelectedThreshold"]["averagePrecision"] = scan_evaluation[
        "averagePrecision"
    ]
    return {
        "policy": (
            f"max_recall_at_precision>={args.precision_target:.2f}"
            if not fallback
            else "best_f1_fallback"
        ),
        "fallback": fallback,
        "selectedThreshold": threshold,
        "precisionTarget": float(args.precision_target),
        "evaluation": fixed,
        "thresholdSelection": scan,
    }


@torch.no_grad()
def validate_opening_head(
    head: OpeningSignalHead,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    head.eval()
    targets = []
    scores = []
    for batch in loader:
        feature = batch["feature"].to(device, dtype=torch.float32, non_blocking=True)
        enclosed = batch["enclosed_depth"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        extended = batch["extended_depth"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        output = head(feature, enclosed, extended)
        probability = output["opening_probability"].detach().cpu().numpy()
        target = batch["target"].detach().cpu().numpy()
        targets.extend(np.asarray(item) for item in target)
        scores.extend(np.asarray(item) for item in probability)
    return select_validation_threshold(targets, scores, args)


def train_one_epoch(
    head: OpeningSignalHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    head.train()
    totals = {"loss_total": 0.0, "loss_bce": 0.0, "loss_tversky": 0.0}
    sample_count = 0
    for batch in loader:
        feature = batch["feature"].to(device, dtype=torch.float32, non_blocking=True)
        enclosed = batch["enclosed_depth"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        extended = batch["extended_depth"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad()
        output = head(feature, enclosed, extended)
        losses = opening_detection_loss(
            output["opening_logits"],
            target,
            pos_weight=args.pos_weight,
            tversky_weight=args.tversky_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
        )
        losses["loss_total"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        batch_size = int(feature.shape[0])
        sample_count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * batch_size
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    head: OpeningSignalHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best: Mapping[str, Any],
    validation: Mapping[str, Any],
    args: argparse.Namespace,
    train_cache_metadata: Mapping[str, Any],
    val_cache_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "task": "ZInD-BiPair-v1 single-view Opening Head",
        "completed_epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "global_step": int(global_step),
        "opening_head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "operating_threshold": float(validation["selectedThreshold"]),
        "threshold_policy": validation["policy"],
        "threshold_fallback": bool(validation["fallback"]),
        "validation": dict(validation),
        "best": dict(best),
        "model": {
            "feature_dim": int(train_cache_metadata["feature_dim"]),
            "hidden_dim": int(args.hidden_dim),
            "kernel_size": int(args.kernel_size),
            "prior_strength": float(args.prior_strength),
            "prior_relative_scale": float(args.prior_relative_scale),
            "branch_order": "extended_first",
        },
        "training_args": _jsonable_args(args),
        "train_cache_contract": {
            key: train_cache_metadata[key]
            for key in (
                "manifest",
                "bi_layout_config",
                "bi_layout_checkpoint",
                "image_shape",
                "sample_count",
                "token_count",
                "feature_dim",
            )
        },
        "val_cache_contract": {
            key: val_cache_metadata[key]
            for key in (
                "manifest",
                "sample_count",
                "token_count",
                "feature_dim",
            )
        },
        "rng_state": _rng_state(),
    }


def train_from_cache(
    train_cache_dir: Path,
    val_cache_dir: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    train_dataset = OpeningFeatureCacheDataset(
        str(train_cache_dir),
        augment=True,
        roll_probability=args.roll_probability,
        flip_probability=args.flip_probability,
    )
    val_dataset = OpeningFeatureCacheDataset(str(val_cache_dir), augment=False)
    if train_dataset.feature_dim != val_dataset.feature_dim:
        raise ValueError("train/val cache feature dimensions differ")
    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    head = OpeningSignalHead(
        feature_dim=train_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        prior_strength=args.prior_strength,
        prior_relative_scale=args.prior_relative_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 0
    global_step = 0
    best: Dict[str, Any] = {
        "epoch": -1,
        "component_recall_iou_0_3": -1.0,
        "average_precision": -1.0,
        "token_f1": -1.0,
        "operating_threshold": None,
    }
    resume_path = None
    if args.resume:
        resume_path = (
            output_dir / "last.pt"
            if args.resume == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        head.load_state_dict(checkpoint["opening_head_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint.get("global_step", 0))
        best = dict(checkpoint.get("best", best))
        _restore_rng_state(checkpoint.get("rng_state"))
        print(f"resumed: {resume_path}, next_epoch={start_epoch}")
    elif (output_dir / "last.pt").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir / 'last.pt'} exists; pass --resume or --overwrite"
        )

    train_meta = train_dataset.metadata
    val_meta = val_dataset.metadata
    last_report: Dict[str, Any] = {}
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(head, train_loader, optimizer, device, args)
        global_step += len(train_loader)
        validation = validate_opening_head(head, val_loader, device, args)
        selected = validation["evaluation"]["metricsAtSelectedThreshold"]
        component = validation["evaluation"]["componentMetricsAtSelectedThreshold"]
        component_recall = float(
            component["overlap"]["iouAtLeast0.30"]["recall"]
        )
        score = (
            component_recall,
            float(validation["evaluation"]["averagePrecision"]),
            float(selected["f1"]),
        )
        best_score = (
            float(best["component_recall_iou_0_3"]),
            float(best["average_precision"]),
            float(best["token_f1"]),
        )
        improved = score > best_score
        if improved:
            best = {
                "epoch": int(epoch),
                "component_recall_iou_0_3": score[0],
                "average_precision": score[1],
                "token_f1": score[2],
                "operating_threshold": float(validation["selectedThreshold"]),
            }
        report = {
            "epoch": int(epoch),
            "runtime_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
            "validation": validation,
            "improved": improved,
            "best": best,
        }
        payload = _checkpoint_payload(
            head,
            optimizer,
            epoch,
            global_step,
            best,
            validation,
            args,
            train_meta,
            val_meta,
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
        _append_jsonl(output_dir / "metrics.jsonl", report)
        _atomic_write_json(output_dir / "latest_metrics.json", report)
        print(
            "epoch {}/{} loss={:.4f} val_P={:.4f} val_R={:.4f} "
            "val_F1={:.4f} interval_R@0.3={:.4f} threshold={:.3f}{}".format(
                epoch + 1,
                args.epochs,
                train_metrics["loss_total"],
                selected["precision"],
                selected["recall"],
                selected["f1"],
                component_recall,
                validation["selectedThreshold"],
                " BEST" if improved else "",
            )
        )
        last_report = report
    if start_epoch >= args.epochs:
        print(f"checkpoint already reached requested epochs={args.epochs}")
        checkpoint = torch.load(str(output_dir / "last.pt"), map_location="cpu")
        last_report = dict(checkpoint.get("validation", {}))
    return last_report


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.max_train_views = args.max_train_views or 8
        args.max_val_views = args.max_val_views or 4
        args.cache_batch_size = min(args.cache_batch_size, 2)
        args.batch_size = min(args.batch_size, 4)
        args.workers = 0
        args.progress_every = min(args.progress_every, 4)
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    print(f"device: {device}")
    if args.device == "auto" and device.type == "cpu":
        print("warning: CUDA unavailable; caching frozen Bi-Layout outputs on CPU")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_manifest = (
        Path(args.train_manifest).expanduser().resolve()
        if args.train_manifest
        else dataset_root / "manifests/train_pairs.jsonl"
    )
    val_manifest = (
        Path(args.val_manifest).expanduser().resolve()
        if args.val_manifest
        else dataset_root / "manifests/val_pairs.jsonl"
    )
    config_path = Path(args.bi_layout_config).expanduser().resolve()
    backbone_checkpoint = Path(args.bi_layout_checkpoint).expanduser().resolve()
    for path in (train_manifest, val_manifest, config_path, backbone_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_root = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output_dir / "cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_views = ZInDOpeningViewDataset(
        str(train_manifest),
        data_root=args.data_root,
        image_shape=(args.image_height, args.image_width),
        max_views=args.max_train_views,
    )
    val_views = ZInDOpeningViewDataset(
        str(val_manifest),
        data_root=args.data_root,
        image_shape=(args.image_height, args.image_width),
        max_views=args.max_val_views,
    )
    print(f"unique views: train={len(train_views)}, val={len(val_views)}")

    model, backbone_report = load_bi_layout(
        str(config_path), str(backbone_checkpoint), device, load_checkpoint=True
    )
    if not backbone_report.get("loaded"):
        raise RuntimeError("Bi-Layout backbone checkpoint was not loaded")
    missing_keys = list(backbone_report.get("missingKeys", ()))
    unexpected_keys = list(backbone_report.get("unexpectedKeys", ()))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Bi-Layout checkpoint is incompatible: missing_keys={}, "
            "unexpected_keys={}".format(missing_keys, unexpected_keys)
        )
    model.requires_grad_(False).eval()
    build_feature_cache(
        train_views,
        cache_root / "train",
        model,
        device,
        train_manifest,
        config_path,
        backbone_checkpoint,
        args,
    )
    build_feature_cache(
        val_views,
        cache_root / "val",
        model,
        device,
        val_manifest,
        config_path,
        backbone_checkpoint,
        args,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.cache_only:
        print("cache-only run completed")
        return 0

    train_from_cache(
        cache_root / "train",
        cache_root / "val",
        output_dir,
        device,
        args,
    )
    print(f"training complete: {output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

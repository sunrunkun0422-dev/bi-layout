#!/usr/bin/env python3
"""Train the 1D cross-scene matcher on frozen Bi-Layout/Openings features.

This is the first complete trainable path for the project:

ZInD pair labels -> frozen Bi-Layout cache -> trained Opening Head ->
opening-guided cross attention -> candidate dustbin assignment -> checkpoint.

The geometry/BEV stage remains an inference post-process and is intentionally
not part of this differentiable training loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_bipair_adapter import adapt_zind_bipair_batch
from dataset.zind_bipair_dataset import collate_zind_bipair
from dataset.zind_bipair_feature_dataset import ZInDBiPairFeatureDataset
from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    candidate_assignment_loss,
    candidate_intervals_to_mask,
    cyclic_token_shift_loss,
    opening_probabilities_to_intervals,
)
from utils.opening_checkpoint import load_opening_head_checkpoint


DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "zind/ZInD-BiPair-v1"
DEFAULT_OPENING_DIR = REPO_ROOT / "checkpoints/Opening_Head/zind_bipair_v1"
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"
DEFAULT_BACKBONE = REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_bipair_v1"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--train_manifest")
    parser.add_argument("--val_manifest")
    parser.add_argument("--data_root")
    parser.add_argument("--train_cache_dir", default=str(DEFAULT_OPENING_DIR / "cache/train"))
    parser.add_argument("--val_cache_dir", default=str(DEFAULT_OPENING_DIR / "cache/val"))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING_DIR / "best.pt"))
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BACKBONE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="resume from a checkpoint path or output_dir/last.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--candidate_temperature", type=float, default=0.2)
    parser.add_argument("--shift_temperature", type=float, default=0.05)
    parser.add_argument("--dustbin_score", type=float, default=0.0)
    parser.add_argument("--gt_guidance_epochs", type=int, default=4)
    parser.add_argument("--mix_guidance_epochs", type=int, default=4)
    parser.add_argument("--mix_guidance_weight", type=float, default=0.5)
    parser.add_argument("--candidate_loss_weight", type=float, default=1.0)
    parser.add_argument("--token_affinity_weight", type=float, default=1.0)
    parser.add_argument(
        "--shared_response_weight",
        type=float,
        default=0.0,
        help=(
            "experimental loss on the current analytic S_A/S_B response; keep 0 "
            "until an explicit learned shared-opening head is introduced"
        ),
    )
    parser.add_argument(
        "--portal_shift_loss_weight",
        "--yaw_loss_weight",
        dest="portal_shift_loss_weight",
        type=float,
        default=0.2,
        help=(
            "weight for circular shared-portal token shift; --yaw_loss_weight is "
            "kept as a compatibility alias, but this is not camera pose yaw"
        ),
    )
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--opening_threshold", type=float)
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument("--metric_top_k", type=int, default=3)
    parser.add_argument("--max_train_pairs", type=int)
    parser.add_argument("--max_val_pairs", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.lr,
        args.heads,
        args.hidden_dim,
        args.candidate_temperature,
        args.shift_temperature,
        args.min_opening_width_tokens,
        args.max_openings_per_view,
        args.metric_top_k,
        args.torch_threads,
        args.progress_every,
    )
    if min(positive) <= 0:
        raise ValueError("positive training/model arguments must be > 0")
    nonnegative = (
        args.workers,
        args.weight_decay,
        args.grad_clip,
        args.gt_guidance_epochs,
        args.mix_guidance_epochs,
        args.candidate_loss_weight,
        args.token_affinity_weight,
        args.shared_response_weight,
        args.portal_shift_loss_weight,
        args.consistency_weight,
    )
    if min(nonnegative) < 0:
        raise ValueError("workers, schedules, and loss weights must be non-negative")
    for name in ("dropout", "mix_guidance_weight"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.opening_threshold is not None and not 0.0 <= args.opening_threshold <= 1.0:
        raise ValueError("opening_threshold must be in [0, 1]")
    for value in (args.max_train_pairs, args.max_val_pairs):
        if value is not None and value <= 0:
            raise ValueError("max pair limits must be positive")


def resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _identity_fingerprint(value: Any, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a file identity mapping")
    size = value.get("size")
    if type(size) is not int or size < 0:
        raise ValueError(f"{name}.size must be a non-negative integer")
    sha256 = value.get("sha256")
    if sha256 is not None:
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"{name}.sha256 must be a SHA-256 string")
        version = ("sha256", sha256)
    else:
        mtime_ns = value.get("mtime_ns")
        if type(mtime_ns) is not int or mtime_ns < 0:
            raise ValueError(f"{name}.mtime_ns must be a non-negative integer")
        version = ("mtime_ns", mtime_ns)
    return (size,) + version


def validate_feature_cache_contract(
    dataset: ZInDBiPairFeatureDataset,
    expected: Mapping[str, Any],
    split: str,
    backbone_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fail fast when frozen features differ from the Opening Head cache."""
    for key, actual in (
        ("sample_count", dataset.sample_count),
        ("token_count", dataset.token_count),
        ("feature_dim", dataset.feature_dim),
    ):
        if int(expected.get(key, -1)) != int(actual):
            raise ValueError(
                f"{split} feature cache {key}={actual} does not match "
                f"Opening checkpoint contract {expected.get(key)!r}"
            )
    if dataset.metadata.get("branch_order") != "extended_first":
        raise ValueError(f"{split} feature cache must use extended_first branch order")
    manifest_actual = _identity_fingerprint(
        dataset.metadata.get("manifest"), f"{split} cache manifest"
    )
    manifest_expected = _identity_fingerprint(
        expected.get("manifest"), f"opening checkpoint {split} manifest"
    )
    if manifest_actual != manifest_expected:
        raise ValueError(
            f"{split} feature cache manifest identity does not match Opening checkpoint"
        )

    reference = expected if backbone_contract is None else backbone_contract
    checked_backbone = {}
    for key in ("bi_layout_config", "bi_layout_checkpoint"):
        saved = reference.get(key)
        if saved is None:
            raise ValueError(f"Opening checkpoint cache contract is missing {key}")
        actual = _identity_fingerprint(
            dataset.metadata.get(key), f"{split} cache {key}"
        )
        required = _identity_fingerprint(saved, f"opening checkpoint {key}")
        if actual != required:
            raise ValueError(
                f"{split} feature cache {key} identity does not match Opening checkpoint"
            )
        checked_backbone[key] = True
    return {
        "validated": True,
        "split": split,
        "sampleCount": int(dataset.sample_count),
        "pairCountAfterMultiPortalMerge": int(len(dataset)),
        "tokenCount": int(dataset.token_count),
        "featureDim": int(dataset.feature_dim),
        "manifestIdentity": list(manifest_actual),
        "backboneIdentity": checked_backbone,
    }


def validate_resume_contract(
    checkpoint: Mapping[str, Any],
    matcher: OpeningGuidedCrossAttentionMatcher,
    opening_report: Mapping[str, Any],
    opening_threshold: float,
    train_dataset: ZInDBiPairFeatureDataset,
    val_dataset: ZInDBiPairFeatureDataset,
    args: argparse.Namespace,
) -> None:
    """Reject a resume that would silently mix model/data dependencies."""
    saved_model = checkpoint.get("model")
    saved_data = checkpoint.get("data_contract")
    saved_opening = checkpoint.get("opening_head_dependency")
    if not all(isinstance(value, Mapping) for value in (saved_model, saved_data, saved_opening)):
        raise ValueError("matcher resume checkpoint is missing model/data/opening contracts")
    expected_model = {
        "feature_dim": int(matcher.feature_dim),
        "heads": int(matcher.heads),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "opening_bias_strength": float(matcher.opening_bias_strength),
        "candidate_temperature": float(args.candidate_temperature),
        "shift_temperature": float(args.shift_temperature),
        "has_dustbin": True,
    }
    for key, expected_value in expected_model.items():
        saved_value = saved_model.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(saved_value, (int, float)) and math.isclose(
                float(saved_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = saved_value == expected_value
        if not matches:
            raise ValueError(
                f"resume model contract mismatch for {key}: {saved_value!r} != {expected_value!r}"
            )
    expected_data = {
        "train_manifest": str(train_dataset.pairs.manifest_path),
        "val_manifest": str(val_dataset.pairs.manifest_path),
        "train_feature_cache": str(train_dataset.cache_dir),
        "val_feature_cache": str(val_dataset.cache_dir),
        "token_count": int(train_dataset.token_count),
        "feature_dim": int(train_dataset.feature_dim),
    }
    for key, expected_value in expected_data.items():
        if saved_data.get(key) != expected_value:
            raise ValueError(
                f"resume data contract mismatch for {key}: "
                f"{saved_data.get(key)!r} != {expected_value!r}"
            )
    saved_backbone = saved_opening.get("backboneContract")
    current_backbone = opening_report.get("backboneContract")
    if saved_backbone != current_backbone:
        raise ValueError("resume Opening Head backbone dependency changed")
    saved_model_config = saved_opening.get("model")
    if saved_model_config != opening_report.get("model"):
        raise ValueError("resume Opening Head model dependency changed")
    saved_threshold = checkpoint.get("opening_probability_threshold")
    if not isinstance(saved_threshold, (int, float)) or not math.isclose(
        float(saved_threshold), float(opening_threshold), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("resume opening probability threshold changed")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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


def guidance_mode_for_epoch(
    epoch: int, gt_epochs: int, mix_epochs: int
) -> str:
    if epoch < gt_epochs:
        return "gt"
    if epoch < gt_epochs + mix_epochs:
        return "mix"
    return "predicted"


def candidate_targets_from_affinity(
    candidate_masks_a: torch.Tensor,
    affinity_ab: torch.Tensor,
    candidate_masks_b: torch.Tensor,
    min_iou: float = 0.30,
) -> torch.Tensor:
    """Project dense portal affinity into candidate pairs with IoU gating.

    Conditioning each side's target tokens on the candidate from the other
    side keeps multiple true portals separate after duplicate records are
    merged.  A one-token accidental overlap therefore no longer counts as a
    recalled/matched opening.
    """
    if not 0.0 <= float(min_iou) <= 1.0:
        raise ValueError("min_iou must be in [0, 1]")
    masks_a = candidate_masks_a.to(dtype=affinity_ab.dtype)
    masks_b = candidate_masks_b.to(dtype=affinity_ab.dtype)
    mass = torch.einsum(
        "bkn,bnm,blm->bkl",
        masks_a,
        affinity_ab,
        masks_b,
    )
    affinity_binary = (affinity_ab > 0).to(dtype=affinity_ab.dtype)
    connected_a = torch.einsum("bnm,blm->bln", affinity_binary, masks_b) > 0
    connected_b = torch.einsum("bkn,bnm->bkm", masks_a, affinity_binary) > 0
    connected_a = connected_a.to(dtype=affinity_ab.dtype)
    connected_b = connected_b.to(dtype=affinity_ab.dtype)

    overlap_a = torch.einsum("bkn,bln->bkl", masks_a, connected_a)
    union_a = (
        masks_a.sum(dim=-1).unsqueeze(-1)
        + connected_a.sum(dim=-1).unsqueeze(1)
        - overlap_a
    )
    overlap_b = torch.einsum("bkm,blm->bkl", connected_b, masks_b)
    union_b = (
        connected_b.sum(dim=-1).unsqueeze(-1)
        + masks_b.sum(dim=-1).unsqueeze(1)
        - overlap_b
    )
    iou_a = overlap_a / union_a.clamp_min(1.0)
    iou_b = overlap_b / union_b.clamp_min(1.0)
    target = (mass > 0) & (iou_a >= float(min_iou)) & (iou_b >= float(min_iou))
    return target.to(dtype=affinity_ab.dtype)


def predicted_candidate_masks(
    probability: torch.Tensor,
    threshold: float,
    min_width_tokens: int,
    max_openings: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of frozen Opening Head scores to padded candidates."""
    if probability.ndim != 2:
        raise ValueError("opening probability must have shape [B, N]")
    per_sample = []
    for sample in probability:
        intervals = opening_probabilities_to_intervals(
            sample,
            threshold=threshold,
            min_width_tokens=min_width_tokens,
            max_intervals=max_openings,
        )
        per_sample.append(
            candidate_intervals_to_mask(
                intervals, int(probability.shape[1]), device=probability.device
            )
        )
    max_count = max(1, max(int(item.shape[0]) for item in per_sample))
    masks = torch.zeros(
        (probability.shape[0], max_count, probability.shape[1]),
        dtype=torch.bool,
        device=probability.device,
    )
    valid = torch.zeros(
        (probability.shape[0], max_count),
        dtype=torch.bool,
        device=probability.device,
    )
    for batch_index, sample_masks in enumerate(per_sample):
        count = int(sample_masks.shape[0])
        if count:
            masks[batch_index, :count] = sample_masks
            valid[batch_index, :count] = sample_masks.any(dim=-1)
    return masks, valid


def dense_token_affinity_loss(
    outputs: Mapping[str, torch.Tensor], target_ab: torch.Tensor
) -> torch.Tensor:
    """Bidirectional row-wise CE for positive shared-opening tokens."""
    target_ab = target_ab.to(outputs["Aff_AB"])

    def directional(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        row_mass = target.sum(dim=-1, keepdim=True)
        valid = row_mass.squeeze(-1) > 0
        normalized = target / row_mass.clamp_min(1e-6)
        row_loss = -(normalized * prediction.clamp_min(1e-8).log()).sum(dim=-1)
        return row_loss[valid].mean() if valid.any() else row_loss.new_zeros(())

    return 0.5 * (
        directional(outputs["Aff_AB"], target_ab)
        + directional(outputs["Aff_BA"], target_ab.transpose(-2, -1))
    )


def portal_shift_target(
    shared_portal_a: torch.Tensor, shared_portal_b: torch.Tensor
) -> torch.Tensor:
    """Return the circular B-token minus A-token shift for shared portals.

    This is the quantity represented by ``cyclic_shift_score``.  It is not the
    camera-to-camera pose yaw when the panorama centers are translated, so the
    matcher must not supervise it directly with ``T_B_to_A`` rotation.
    """
    if shared_portal_a.shape != shared_portal_b.shape or shared_portal_a.ndim != 2:
        raise ValueError("shared portal masks must have the same shape [B, N]")
    token_count = int(shared_portal_a.shape[-1])
    angles = torch.arange(
        token_count, device=shared_portal_a.device, dtype=torch.float32
    ) * (2.0 * math.pi / token_count)

    def center(mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=angles.dtype)
        sine = (weights * torch.sin(angles)).sum(dim=-1)
        cosine = (weights * torch.cos(angles)).sum(dim=-1)
        return torch.atan2(sine, cosine)

    center_a = center(shared_portal_a)
    center_b = center(shared_portal_b)
    return torch.atan2(
        torch.sin(center_b - center_a), torch.cos(center_b - center_a)
    )


def positive_shared_response_loss(
    outputs: Mapping[str, torch.Tensor],
    shared_a: torch.Tensor,
    shared_b: torch.Tensor,
    is_match: torch.Tensor,
) -> torch.Tensor:
    """Supervise the existing shared-opening response on positive pairs only."""
    positive = is_match.to(device=outputs["S_A"].device, dtype=torch.bool).reshape(-1)
    if not positive.any():
        return outputs["S_A"].new_zeros(())
    loss_a = F.binary_cross_entropy(
        outputs["S_A"][positive].clamp(1e-6, 1.0 - 1e-6),
        shared_a.to(outputs["S_A"])[positive],
    )
    loss_b = F.binary_cross_entropy(
        outputs["S_B"][positive].clamp(1e-6, 1.0 - 1e-6),
        shared_b.to(outputs["S_B"])[positive],
    )
    return 0.5 * (loss_a + loss_b)


def _forward_batch(
    matcher: OpeningGuidedCrossAttentionMatcher,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    guidance_mode: str,
    mix_weight: float,
    candidate_source: str,
    opening_threshold: float,
    min_width_tokens: int,
    max_openings: int,
) -> Tuple[Mapping[str, torch.Tensor], Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair = adapt_zind_bipair_batch(raw_batch).to(device)
    feature_a = raw_batch["feature_A"].to(device, dtype=torch.float32, non_blocking=True)
    feature_b = raw_batch["feature_B"].to(device, dtype=torch.float32, non_blocking=True)

    if candidate_source == "gt":
        masks_a, valid_a = pair.candidates_a.masks, pair.candidates_a.valid
        masks_b, valid_b = pair.candidates_b.masks, pair.candidates_b.valid
    elif candidate_source == "predicted":
        with torch.no_grad():
            signal_a = matcher.opening_head(
                feature_a, pair.depth_enclosed_a, pair.depth_extended_a
            )
            signal_b = matcher.opening_head(
                feature_b, pair.depth_enclosed_b, pair.depth_extended_b
            )
        masks_a, valid_a = predicted_candidate_masks(
            signal_a["opening_probability"],
            opening_threshold,
            min_width_tokens,
            max_openings,
        )
        masks_b, valid_b = predicted_candidate_masks(
            signal_b["opening_probability"],
            opening_threshold,
            min_width_tokens,
            max_openings,
        )
    else:
        raise ValueError("candidate_source must be gt or predicted")

    outputs = matcher(
        feature_a,
        feature_b,
        pair.depth_enclosed_a,
        pair.depth_extended_a,
        pair.depth_enclosed_b,
        pair.depth_extended_b,
        candidate_masks_a=masks_a,
        candidate_masks_b=masks_b,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        opening_guidance_a=pair.opening_all_a.float(),
        opening_guidance_b=pair.opening_all_b.float(),
        opening_guidance_mode=guidance_mode,
        opening_guidance_mix_weight=mix_weight,
    )
    candidate_target = candidate_targets_from_affinity(
        masks_a, pair.affinity_ab, masks_b
    )
    return outputs, pair, candidate_target, valid_a, valid_b


def _losses(
    outputs: Mapping[str, torch.Tensor],
    pair: Any,
    candidate_target: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    candidate = candidate_assignment_loss(
        outputs,
        candidate_target_ab=candidate_target,
        is_match=pair.is_match,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        consistency_weight=args.consistency_weight,
    )
    token = dense_token_affinity_loss(outputs, pair.affinity_ab)
    shared = (
        positive_shared_response_loss(
            outputs, pair.shared_portal_a, pair.shared_portal_b, pair.is_match
        )
        if args.shared_response_weight > 0
        else token.new_zeros(())
    )
    positive = pair.is_match.bool()
    metadata = getattr(pair, "metadata", None)
    merged_count = metadata.get("merged_pair_count") if metadata else None
    if torch.is_tensor(merged_count):
        shift_valid = positive & (merged_count.to(positive.device).reshape(-1) <= 1)
    else:
        shift_valid = positive
    shift_target = portal_shift_target(pair.shared_portal_a, pair.shared_portal_b)
    portal_shift = (
        cyclic_token_shift_loss(
            outputs["cyclic_shift_score"][shift_valid], shift_target[shift_valid]
        )
        if shift_valid.any()
        else token.new_zeros(())
    )
    total = (
        args.candidate_loss_weight * candidate["loss_candidate_total"]
        + args.token_affinity_weight * token
        + args.shared_response_weight * shared
        + args.portal_shift_loss_weight * portal_shift
    )
    return {
        "loss_total": total,
        "loss_candidate": candidate["loss_candidate_total"],
        "loss_candidate_assignment": candidate["loss_candidate_assignment"],
        "loss_consistency": candidate["loss_bidirectional_consistency"],
        "loss_token_affinity": token,
        "loss_shared_response": shared,
        "loss_portal_shift": portal_shift,
    }


class MetricAccumulator:
    def __init__(self, top_k: int) -> None:
        self.top_k = int(top_k)
        self.samples = 0
        self.loss_sums: Dict[str, float] = {}
        self.positive = 0
        self.negative = 0
        self.target_available = 0
        self.accepted_positive = 0
        self.correct_top1 = 0
        self.correct_topk = 0
        self.rejected_negative = 0
        self.portal_shift_error_sum = 0.0
        self.portal_shift_count = 0
        self.pair_match_positive_sum = 0.0
        self.pair_match_negative_sum = 0.0
        self.valid_candidates_a = 0
        self.valid_candidates_b = 0

    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        pair: Any,
        candidate_target: torch.Tensor,
        valid_a: torch.Tensor,
        valid_b: torch.Tensor,
        losses: Mapping[str, torch.Tensor],
    ) -> None:
        batch_size = int(pair.batch_size)
        self.samples += batch_size
        for key, value in losses.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(
                value.detach().item()
            ) * batch_size
        positive = pair.is_match.bool()
        negative = ~positive
        self.positive += int(positive.sum().item())
        self.negative += int(negative.sum().item())
        self.valid_candidates_a += int(valid_a.sum().item())
        self.valid_candidates_b += int(valid_b.sum().item())

        target = candidate_target > 0
        has_target = target.flatten(1).any(dim=-1) & positive
        self.target_available += int(has_target.sum().item())
        best = outputs["best_candidate_pair"]
        accepted = (best >= 0).all(dim=-1)
        self.accepted_positive += int((accepted & positive).sum().item())
        for batch_index in torch.where(has_target)[0].tolist():
            index_a, index_b = best[batch_index].tolist()
            if index_a >= 0 and bool(target[batch_index, index_a, index_b]):
                self.correct_top1 += 1
            score = outputs["candidate_pair_score"][batch_index].masked_fill(
                ~(valid_a[batch_index].unsqueeze(-1) & valid_b[batch_index].unsqueeze(0)),
                -1.0,
            )
            flat_target = target[batch_index].flatten()
            count = min(self.top_k, int((score.flatten() >= 0).sum().item()))
            if count > 0:
                indices = torch.topk(score.flatten(), k=count).indices
                if bool(flat_target[indices].any()):
                    self.correct_topk += 1

        rejected = (best < 0).all(dim=-1)
        self.rejected_negative += int((rejected & negative).sum().item())
        pair_match = outputs["pair_match_probability"].detach()
        if positive.any():
            self.pair_match_positive_sum += float(pair_match[positive].sum().item())
            metadata = getattr(pair, "metadata", None)
            merged_count = metadata.get("merged_pair_count") if metadata else None
            if torch.is_tensor(merged_count):
                shift_valid = positive & (
                    merged_count.to(positive.device).reshape(-1) <= 1
                )
            else:
                shift_valid = positive
            shift_target = portal_shift_target(
                pair.shared_portal_a, pair.shared_portal_b
            )
            predicted_shift = outputs.get(
                "relative_token_shift_radians", outputs["relative_yaw_radians"]
            )
            error = torch.atan2(
                torch.sin(
                    predicted_shift[shift_valid] - shift_target[shift_valid]
                ),
                torch.cos(
                    predicted_shift[shift_valid] - shift_target[shift_valid]
                ),
            ).abs()
            self.portal_shift_error_sum += float(torch.rad2deg(error).sum().item())
            self.portal_shift_count += int(shift_valid.sum().item())
        if negative.any():
            self.pair_match_negative_sum += float(pair_match[negative].sum().item())

    def result(self) -> Dict[str, float]:
        positive = max(self.positive, 1)
        negative = max(self.negative, 1)
        available = max(self.target_available, 1)
        candidate_recall = self.target_available / positive
        conditional_top1 = self.correct_top1 / available
        conditional_topk = self.correct_topk / available
        end_to_end_top1 = self.correct_top1 / positive
        negative_rejection = self.rejected_negative / negative
        return {
            **{
                key: value / max(self.samples, 1)
                for key, value in self.loss_sums.items()
            },
            "pair_count": float(self.samples),
            "positive_pair_count": float(self.positive),
            "negative_pair_count": float(self.negative),
            "positive_candidate_recall": candidate_recall,
            "positive_acceptance_rate": self.accepted_positive / positive,
            "conditional_candidate_top1": conditional_top1,
            "conditional_candidate_topk": conditional_topk,
            "end_to_end_candidate_top1": end_to_end_top1,
            "negative_rejection_rate": negative_rejection,
            "balanced_flow_score": 0.5 * (end_to_end_top1 + negative_rejection),
            "portal_shift_pair_count": float(self.portal_shift_count),
            "mean_portal_shift_error_degrees": self.portal_shift_error_sum
            / max(self.portal_shift_count, 1),
            "mean_pair_match_positive": self.pair_match_positive_sum / positive,
            "mean_pair_match_negative": self.pair_match_negative_sum / negative,
            "mean_candidate_count_a": self.valid_candidates_a / max(self.samples, 1),
            "mean_candidate_count_b": self.valid_candidates_b / max(self.samples, 1),
        }


def train_one_epoch(
    matcher: OpeningGuidedCrossAttentionMatcher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Dict[str, float]:
    mode = guidance_mode_for_epoch(
        epoch, args.gt_guidance_epochs, args.mix_guidance_epochs
    )
    matcher.train()
    matcher.opening_head.eval()
    metrics = MetricAccumulator(args.metric_top_k)
    start = time.perf_counter()
    for step, raw_batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs, pair, target, valid_a, valid_b = _forward_batch(
            matcher,
            raw_batch,
            device,
            guidance_mode=mode,
            mix_weight=args.mix_guidance_weight,
            candidate_source="gt",
            opening_threshold=opening_threshold,
            min_width_tokens=args.min_opening_width_tokens,
            max_openings=args.max_openings_per_view,
        )
        losses = _losses(outputs, pair, target, valid_a, valid_b, args)
        losses["loss_total"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in matcher.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
        optimizer.step()
        metrics.update(outputs, pair, target, valid_a, valid_b, losses)
        if step % args.progress_every == 0 or step == len(loader):
            print(
                "  train {}/{} pairs={} loss={:.4f} elapsed={:.1f}s".format(
                    step,
                    len(loader),
                    metrics.samples,
                    metrics.loss_sums.get("loss_total", 0.0) / max(metrics.samples, 1),
                    time.perf_counter() - start,
                ),
                flush=True,
            )
    result = metrics.result()
    result["guidance_mode"] = mode
    result["runtime_seconds"] = time.perf_counter() - start
    return result


@torch.no_grad()
def evaluate(
    matcher: OpeningGuidedCrossAttentionMatcher,
    loader: DataLoader,
    device: torch.device,
    candidate_source: str,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Dict[str, float]:
    matcher.eval()
    metrics = MetricAccumulator(args.metric_top_k)
    start = time.perf_counter()
    guidance_mode = "gt" if candidate_source == "gt" else "predicted"
    for raw_batch in loader:
        outputs, pair, target, valid_a, valid_b = _forward_batch(
            matcher,
            raw_batch,
            device,
            guidance_mode=guidance_mode,
            mix_weight=args.mix_guidance_weight,
            candidate_source=candidate_source,
            opening_threshold=opening_threshold,
            min_width_tokens=args.min_opening_width_tokens,
            max_openings=args.max_openings_per_view,
        )
        losses = _losses(outputs, pair, target, valid_a, valid_b, args)
        metrics.update(outputs, pair, target, valid_a, valid_b, losses)
    result = metrics.result()
    result["guidance_mode"] = guidance_mode
    result["candidate_source"] = candidate_source
    result["runtime_seconds"] = time.perf_counter() - start
    return result


def _make_loader(
    manifest: Path,
    cache_dir: Path,
    max_pairs: Optional[int],
    args: argparse.Namespace,
    shuffle: bool,
    device: torch.device,
) -> Tuple[ZInDBiPairFeatureDataset, DataLoader]:
    dataset = ZInDBiPairFeatureDataset(
        str(manifest),
        str(cache_dir),
        data_root=args.data_root,
        max_pairs=max_pairs,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_zind_bipair,
        generator=generator if shuffle else None,
    )
    return dataset, loader


def _checkpoint_payload(
    matcher: OpeningGuidedCrossAttentionMatcher,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_gt: Mapping[str, Any],
    validation_predicted: Mapping[str, Any],
    opening_report: Mapping[str, Any],
    opening_threshold: float,
    cache_validation: Mapping[str, Any],
    train_dataset: ZInDBiPairFeatureDataset,
    val_dataset: ZInDBiPairFeatureDataset,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "task": "ZInD-BiPair-v1 1D cross-scene opening matcher",
        "completed_epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "matcher_state_dict": matcher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "opening_head_dependency": dict(opening_report),
        "opening_probability_threshold": float(opening_threshold),
        "model": {
            "feature_dim": int(matcher.feature_dim),
            "heads": int(matcher.heads),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "opening_bias_strength": float(matcher.opening_bias_strength),
            "candidate_temperature": float(args.candidate_temperature),
            "shift_temperature": float(args.shift_temperature),
            "has_dustbin": True,
        },
        "data_contract": {
            "train_manifest": str(train_dataset.pairs.manifest_path),
            "val_manifest": str(val_dataset.pairs.manifest_path),
            "train_feature_cache": str(train_dataset.cache_dir),
            "val_feature_cache": str(val_dataset.cache_dir),
            "token_count": int(train_dataset.token_count),
            "feature_dim": int(train_dataset.feature_dim),
            "opening_guidance": "all openings; never pair shared_portal",
            "shared_supervision": "portal_mask/affinity_gt",
            "coordinate_frame": "zind_pano_local_xy",
            "transform_direction": "b_to_a",
            "feature_cache_validation": dict(cache_validation),
        },
        "training_args": vars(args),
        "train": dict(train_metrics),
        "validation": {
            "teacher_forced": dict(validation_gt),
            "predicted_openings": dict(validation_predicted),
        },
        "best": dict(best),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        args.epochs = 1
        args.max_train_pairs = args.max_train_pairs or 8
        args.max_val_pairs = args.max_val_pairs or 4
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.progress_every = min(args.progress_every, 2)
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    print("device: {}".format(device), flush=True)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_manifest = Path(args.train_manifest).expanduser().resolve() if args.train_manifest else dataset_root / "manifests/train_pairs.jsonl"
    val_manifest = Path(args.val_manifest).expanduser().resolve() if args.val_manifest else dataset_root / "manifests/val_pairs.jsonl"
    train_cache = Path(args.train_cache_dir).expanduser().resolve()
    val_cache = Path(args.val_cache_dir).expanduser().resolve()
    opening_checkpoint = Path(args.opening_checkpoint).expanduser().resolve()
    config_path = Path(args.bi_layout_config).expanduser().resolve()
    backbone_path = Path(args.bi_layout_checkpoint).expanduser().resolve()
    for path in (
        train_manifest,
        val_manifest,
        train_cache / "metadata.json",
        val_cache / "metadata.json",
        opening_checkpoint,
        config_path,
        backbone_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_dataset, train_loader = _make_loader(
        train_manifest, train_cache, args.max_train_pairs, args, True, device
    )
    val_dataset, val_loader = _make_loader(
        val_manifest, val_cache, args.max_val_pairs, args, False, device
    )
    if train_dataset.feature_dim != val_dataset.feature_dim:
        raise ValueError("train/val feature cache dimensions differ")
    if train_dataset.token_count != val_dataset.token_count:
        raise ValueError("train/val token counts differ")
    if train_dataset.feature_dim % args.heads != 0:
        raise ValueError("feature_dim must be divisible by --heads")
    print(
        "pairs: train={} val={} feature=[{}, {}]".format(
            len(train_dataset),
            len(val_dataset),
            train_dataset.token_count,
            train_dataset.feature_dim,
        ),
        flush=True,
    )

    matcher = OpeningGuidedCrossAttentionMatcher(
        feature_dim=train_dataset.feature_dim,
        heads=args.heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        candidate_temperature=args.candidate_temperature,
        shift_temperature=args.shift_temperature,
        dustbin_score=args.dustbin_score,
    ).to(device)
    opening_report = load_opening_head_checkpoint(
        matcher.opening_head,
        str(opening_checkpoint),
        expected_feature_dim=train_dataset.feature_dim,
        expected_branch_order="extended_first",
        expected_token_count=train_dataset.token_count,
        bi_layout_config_path=str(config_path),
        bi_layout_checkpoint_path=str(backbone_path),
    )
    opening_threshold = (
        float(args.opening_threshold)
        if args.opening_threshold is not None
        else float(opening_report["operatingThreshold"])
    )
    opening_payload = torch.load(str(opening_checkpoint), map_location="cpu")
    if not isinstance(opening_payload, Mapping):
        raise ValueError("opening checkpoint root must be a mapping")
    train_opening_contract = opening_payload.get("train_cache_contract")
    val_opening_contract = opening_payload.get("val_cache_contract")
    if not isinstance(train_opening_contract, Mapping) or not isinstance(
        val_opening_contract, Mapping
    ):
        raise ValueError("opening checkpoint is missing train/val cache contracts")
    cache_validation = {
        "train": validate_feature_cache_contract(
            train_dataset, train_opening_contract, "train"
        ),
        "val": validate_feature_cache_contract(
            val_dataset,
            val_opening_contract,
            "val",
            backbone_contract=train_opening_contract,
        ),
    }
    matcher.opening_head.requires_grad_(False)
    matcher.opening_head.eval()
    trainable = [parameter for parameter in matcher.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best: Dict[str, Any] = {
        "epoch": -1,
        "balanced_flow_score": -1.0,
        "end_to_end_candidate_top1": -1.0,
        "negative_rejection_rate": -1.0,
    }
    if args.resume:
        resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("matcher resume checkpoint root must be a mapping")
        validate_resume_contract(
            checkpoint,
            matcher,
            opening_report,
            opening_threshold,
            train_dataset,
            val_dataset,
            args,
        )
        matcher.load_state_dict(checkpoint["matcher_state_dict"], strict=True)
        # The explicitly requested Opening Head is authoritative.  Reload it
        # after the full matcher so resume cannot silently replace it with a
        # stale embedded copy.
        opening_report = load_opening_head_checkpoint(
            matcher.opening_head,
            str(opening_checkpoint),
            expected_feature_dim=train_dataset.feature_dim,
            expected_branch_order="extended_first",
            expected_token_count=train_dataset.token_count,
            bi_layout_config_path=str(config_path),
            bi_layout_checkpoint_path=str(backbone_path),
        )
        matcher.opening_head.requires_grad_(False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["next_epoch"])
        best = dict(checkpoint.get("best", best))
        print("resumed {} at epoch {}".format(resume_path, start_epoch), flush=True)
    elif (output_dir / "last.pt").exists() and not args.overwrite:
        raise FileExistsError(
            "{} exists; use --resume or --overwrite".format(output_dir / "last.pt")
        )

    for epoch in range(start_epoch, args.epochs):
        print("epoch {}/{}".format(epoch + 1, args.epochs), flush=True)
        train_metrics = train_one_epoch(
            matcher, train_loader, optimizer, device, epoch, opening_threshold, args
        )
        validation_gt = evaluate(
            matcher, val_loader, device, "gt", opening_threshold, args
        )
        validation_predicted = evaluate(
            matcher, val_loader, device, "predicted", opening_threshold, args
        )
        current_score = (
            float(validation_predicted["balanced_flow_score"]),
            float(validation_predicted["end_to_end_candidate_top1"]),
            float(validation_predicted["negative_rejection_rate"]),
        )
        best_score = (
            float(best["balanced_flow_score"]),
            float(best["end_to_end_candidate_top1"]),
            float(best["negative_rejection_rate"]),
        )
        improved = current_score > best_score
        if improved:
            best = {
                "epoch": int(epoch),
                "balanced_flow_score": current_score[0],
                "end_to_end_candidate_top1": current_score[1],
                "negative_rejection_rate": current_score[2],
            }
        report = {
            "epoch": int(epoch),
            "train": train_metrics,
            "validation": {
                "teacher_forced": validation_gt,
                "predicted_openings": validation_predicted,
            },
            "opening_threshold": opening_threshold,
            "improved": improved,
            "best": best,
        }
        payload = _checkpoint_payload(
            matcher,
            optimizer,
            epoch,
            best,
            train_metrics,
            validation_gt,
            validation_predicted,
            opening_report,
            opening_threshold,
            cache_validation,
            train_dataset,
            val_dataset,
            args,
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
            _atomic_json(output_dir / "best_metrics.json", report)
        _append_jsonl(output_dir / "metrics.jsonl", report)
        _atomic_json(output_dir / "latest_metrics.json", report)
        print(
            "  val predicted: candidate_recall={:.4f} top1={:.4f} "
            "negative_reject={:.4f} balanced={:.4f} shift_mae={:.2f}{}".format(
                validation_predicted["positive_candidate_recall"],
                validation_predicted["end_to_end_candidate_top1"],
                validation_predicted["negative_rejection_rate"],
                validation_predicted["balanced_flow_score"],
                validation_predicted["mean_portal_shift_error_degrees"],
                " BEST" if improved else "",
            ),
            flush=True,
        )

    if start_epoch >= args.epochs:
        print("checkpoint already reached requested epochs={}".format(args.epochs))
    best_metrics_path = output_dir / "best_metrics.json"
    best_checkpoint_path = output_dir / "best.pt"
    if not best_metrics_path.is_file() and best_checkpoint_path.is_file():
        best_checkpoint = torch.load(str(best_checkpoint_path), map_location="cpu")
        if not isinstance(best_checkpoint, Mapping):
            raise ValueError("best matcher checkpoint root must be a mapping")
        _atomic_json(
            best_metrics_path,
            {
                "epoch": int(best_checkpoint["completed_epoch"]),
                "train": dict(best_checkpoint["train"]),
                "validation": dict(best_checkpoint["validation"]),
                "opening_threshold": float(
                    best_checkpoint["opening_probability_threshold"]
                ),
                "improved": True,
                "best": dict(best_checkpoint["best"]),
            },
        )
    print("training complete: {}".format(output_dir / "best.pt"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

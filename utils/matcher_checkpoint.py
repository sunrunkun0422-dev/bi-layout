"""Contract-aware construction of a trained cross-scene matcher."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    OpeningSignalHead,
)


def _finite(mapping: Mapping[str, Any], key: str, positive: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"matcher checkpoint model.{key} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"matcher checkpoint model.{key} must be finite") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"matcher checkpoint model.{key} has an invalid value")
    return number


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"matcher checkpoint model.{key} must be a positive integer")
    return int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_saved_identity(saved: Any, path_value: str, name: str) -> Dict[str, Any]:
    if not isinstance(saved, Mapping):
        raise ValueError(f"matcher opening dependency is missing {name} identity")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if saved.get("size") != int(stat.st_size):
        raise ValueError(f"matcher opening dependency {name} size changed")
    report = {"path": str(path), "size": int(stat.st_size)}
    saved_hash = saved.get("sha256")
    if saved_hash is not None:
        current_hash = _sha256(path)
        if current_hash != saved_hash:
            raise ValueError(f"matcher opening dependency {name} SHA-256 changed")
        report["sha256"] = current_hash
    else:
        saved_mtime = saved.get("mtimeNs", saved.get("mtime_ns"))
        if saved_mtime != int(stat.st_mtime_ns):
            raise ValueError(f"matcher opening dependency {name} mtime changed")
        report["mtimeNs"] = int(stat.st_mtime_ns)
    return report


def load_cross_scene_matcher_checkpoint(
    checkpoint_path: str,
    *,
    expected_feature_dim: int,
    expected_token_count: int,
    expected_branch_order: str,
    bi_layout_config_path: str,
    bi_layout_checkpoint_path: str,
    device: torch.device,
) -> Tuple[OpeningGuidedCrossAttentionMatcher, Dict[str, Any]]:
    """Rebuild and load a matcher from its saved architecture contract."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("matcher checkpoint root must be a mapping")
    state_dict = payload.get("matcher_state_dict")
    model = payload.get("model")
    opening_dependency = payload.get("opening_head_dependency")
    if not isinstance(state_dict, Mapping):
        raise ValueError("matcher checkpoint is missing matcher_state_dict")
    if not isinstance(model, Mapping):
        raise ValueError("matcher checkpoint is missing model contract")
    if not isinstance(opening_dependency, Mapping):
        raise ValueError("matcher checkpoint is missing Opening Head dependency")

    feature_dim = _positive_int(model, "feature_dim")
    heads = _positive_int(model, "heads")
    hidden_dim = _positive_int(model, "hidden_dim")
    if feature_dim != int(expected_feature_dim):
        raise ValueError(
            f"matcher feature_dim={feature_dim} does not match Bi-Layout {expected_feature_dim}"
        )
    if feature_dim % heads != 0:
        raise ValueError("matcher feature_dim must be divisible by saved heads")
    dropout = _finite(model, "dropout")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("matcher checkpoint dropout must be in [0, 1]")
    opening_bias = _finite(model, "opening_bias_strength")
    candidate_temperature = _finite(model, "candidate_temperature", positive=True)
    shift_temperature = _finite(model, "shift_temperature", positive=True)
    if model.get("has_dustbin") is not True:
        raise ValueError("matcher checkpoint does not declare the required dustbin")

    opening_model = opening_dependency.get("model")
    backbone = opening_dependency.get("backboneContract")
    if not isinstance(opening_model, Mapping) or not isinstance(backbone, Mapping):
        raise ValueError("matcher checkpoint has an incomplete Opening Head dependency")
    if int(opening_model.get("feature_dim", -1)) != feature_dim:
        raise ValueError("matcher and Opening Head feature dimensions disagree")
    if int(opening_model.get("hidden_dim", -1)) != hidden_dim:
        raise ValueError("matcher and Opening Head hidden dimensions disagree")
    if opening_dependency.get("branchOrder") != expected_branch_order:
        raise ValueError("matcher Opening Head branch order does not match inference")
    if int(backbone.get("tokenCount", -1)) != int(expected_token_count):
        raise ValueError("matcher Opening Head token count does not match Bi-Layout")
    if int(backbone.get("featureDim", -1)) != feature_dim:
        raise ValueError("matcher Opening Head backbone feature dimension changed")
    config_report = _validate_saved_identity(
        backbone.get("biLayoutConfig"), bi_layout_config_path, "Bi-Layout config"
    )
    backbone_report = _validate_saved_identity(
        backbone.get("biLayoutCheckpoint"),
        bi_layout_checkpoint_path,
        "Bi-Layout checkpoint",
    )

    matcher = OpeningGuidedCrossAttentionMatcher(
        feature_dim=feature_dim,
        heads=heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        opening_bias_strength=opening_bias,
        candidate_temperature=candidate_temperature,
        shift_temperature=shift_temperature,
    )
    matcher.opening_head = OpeningSignalHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        kernel_size=int(opening_model.get("kernel_size", 5)),
        prior_strength=float(opening_model.get("prior_strength", 4.0)),
        prior_relative_scale=float(opening_model.get("prior_relative_scale", 0.1)),
    )
    matcher.load_state_dict(state_dict, strict=True)
    matcher.to(device).eval()

    threshold = payload.get("opening_probability_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("matcher checkpoint is missing opening_probability_threshold")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("matcher opening_probability_threshold must be in [0, 1]")
    return matcher, {
        "path": str(path),
        "loaded": True,
        "formatVersion": payload.get("format_version"),
        "task": payload.get("task"),
        "completedEpoch": payload.get("completed_epoch"),
        "model": dict(model),
        "openingProbabilityThreshold": threshold,
        "openingHeadDependency": {
            "branchOrder": opening_dependency.get("branchOrder"),
            "backboneContractValidated": True,
            "biLayoutConfig": config_report,
            "biLayoutCheckpoint": backbone_report,
        },
    }


__all__ = ["load_cross_scene_matcher_checkpoint"]

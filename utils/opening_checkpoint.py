"""Strict loading helpers for a trained :class:`OpeningSignalHead`.

The opening head is trained against cached features from one exact Bi-Layout
backbone.  Loading only the tensor weights is therefore not sufficient: the
feature width, depth-branch meaning, token count, and backbone files must all
match the cache contract saved by the trainer.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


DEFAULT_OPENING_PROBABILITY_THRESHOLD = 0.12
SUPPORTED_BRANCH_ORDERS = ("extended_first", "enclosed_first")


def _positive_int(mapping: Mapping[str, Any], key: str, scope: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{scope}.{key} must be a positive integer")
    return int(value)


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    scope: str,
    *,
    positive: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{scope}.{key} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{scope}.{key} must be finite") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{scope}.{key} must be {qualifier}")
    return number


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file_identity(
    saved: Any,
    current_path: Path,
    contract_name: str,
) -> Dict[str, Any]:
    if not isinstance(saved, Mapping):
        raise ValueError(
            f"opening checkpoint train_cache_contract.{contract_name} "
            "must be a file identity mapping"
        )
    path = Path(current_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    stat = path.stat()
    saved_size = saved.get("size")
    if type(saved_size) is not int or saved_size < 0:
        raise ValueError(
            f"opening checkpoint {contract_name}.size must be a non-negative integer"
        )
    if int(stat.st_size) != saved_size:
        raise ValueError(
            f"opening checkpoint backbone contract mismatch for {contract_name}: "
            f"size {stat.st_size} != {saved_size}"
        )

    report: Dict[str, Any] = {
        "path": str(path),
        "savedPath": saved.get("path"),
        "size": int(stat.st_size),
    }
    saved_sha256 = saved.get("sha256")
    if saved_sha256 is not None:
        if not isinstance(saved_sha256, str) or len(saved_sha256) != 64:
            raise ValueError(
                f"opening checkpoint {contract_name}.sha256 must be a SHA-256 string"
            )
        current_sha256 = _sha256(path)
        if current_sha256 != saved_sha256:
            raise ValueError(
                f"opening checkpoint backbone contract mismatch for {contract_name}: "
                "SHA-256 differs"
            )
        report["sha256"] = current_sha256
    else:
        saved_mtime_ns = saved.get("mtime_ns")
        if type(saved_mtime_ns) is not int or saved_mtime_ns < 0:
            raise ValueError(
                f"opening checkpoint {contract_name}.mtime_ns must be a "
                "non-negative integer when sha256 is absent"
            )
        if int(stat.st_mtime_ns) != saved_mtime_ns:
            raise ValueError(
                f"opening checkpoint backbone contract mismatch for {contract_name}: "
                f"mtime_ns {stat.st_mtime_ns} != {saved_mtime_ns}"
            )
        report["mtimeNs"] = int(stat.st_mtime_ns)
    return report


def _opening_head_contract(opening_head: torch.nn.Module) -> Dict[str, Any]:
    """Read the architecture fields that affect checkpoint compatibility."""
    try:
        normalized_shape = opening_head.feature_norm.normalized_shape
        feature_dim = int(normalized_shape[0])
        hidden_dim = int(opening_head.input_projection.out_channels)
        kernel_size = int(opening_head.context.kernel_size[0])
        prior_strength = float(opening_head.prior_strength)
        prior_relative_scale = float(opening_head.prior_relative_scale)
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError(
            "opening_head must be an OpeningSignalHead-compatible module"
        ) from exc
    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "kernel_size": kernel_size,
        "prior_strength": prior_strength,
        "prior_relative_scale": prior_relative_scale,
    }


def load_opening_head_checkpoint(
    opening_head: torch.nn.Module,
    checkpoint_path: str,
    *,
    expected_feature_dim: int,
    expected_branch_order: str,
    expected_token_count: int,
    bi_layout_config_path: str,
    bi_layout_checkpoint_path: str,
) -> Dict[str, Any]:
    """Load a trained head after validating its full frozen-backbone contract.

    The caller decides ordering.  Cross-scene entry points intentionally load a
    full matcher first and this head second, so an explicit opening checkpoint
    overrides only ``matcher.opening_head`` while preserving all other matcher
    weights.
    """
    if type(expected_feature_dim) is not int or expected_feature_dim <= 0:
        raise ValueError("expected_feature_dim must be a positive integer")
    if type(expected_token_count) is not int or expected_token_count <= 0:
        raise ValueError("expected_token_count must be a positive integer")
    if expected_branch_order not in SUPPORTED_BRANCH_ORDERS:
        raise ValueError(
            "expected_branch_order must be extended_first or enclosed_first"
        )

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("opening checkpoint root must be a mapping")
    state_dict = payload.get("opening_head_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("opening checkpoint is missing opening_head_state_dict")
    model_config = payload.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("opening checkpoint is missing model configuration")

    saved_model = {
        "feature_dim": _positive_int(
            model_config, "feature_dim", "opening checkpoint model"
        ),
        "hidden_dim": _positive_int(
            model_config, "hidden_dim", "opening checkpoint model"
        ),
        "kernel_size": _positive_int(
            model_config, "kernel_size", "opening checkpoint model"
        ),
        "prior_strength": _finite_float(
            model_config, "prior_strength", "opening checkpoint model"
        ),
        "prior_relative_scale": _finite_float(
            model_config,
            "prior_relative_scale",
            "opening checkpoint model",
            positive=True,
        ),
    }
    saved_branch_order = model_config.get("branch_order")
    if saved_branch_order not in SUPPORTED_BRANCH_ORDERS:
        raise ValueError(
            "opening checkpoint model.branch_order must be extended_first or "
            "enclosed_first"
        )
    if saved_model["feature_dim"] != expected_feature_dim:
        raise ValueError(
            "opening checkpoint feature_dim={} does not match Bi-Layout "
            "patch_dim={}".format(saved_model["feature_dim"], expected_feature_dim)
        )
    if saved_branch_order != expected_branch_order:
        raise ValueError(
            "opening checkpoint branch_order={!r} does not match requested "
            "branch_order={!r}".format(saved_branch_order, expected_branch_order)
        )

    target_model = _opening_head_contract(opening_head)
    for key, saved_value in saved_model.items():
        target_value = target_model[key]
        if isinstance(saved_value, float):
            matches = math.isclose(saved_value, target_value, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = saved_value == target_value
        if not matches:
            raise ValueError(
                "opening checkpoint model.{}={} does not match target "
                "OpeningSignalHead {}={}".format(
                    key, saved_value, key, target_value
                )
            )

    cache_contract = payload.get("train_cache_contract")
    if not isinstance(cache_contract, Mapping):
        raise ValueError(
            "opening checkpoint is missing train_cache_contract; cannot verify "
            "the frozen Bi-Layout backbone"
        )
    cache_feature_dim = _positive_int(
        cache_contract, "feature_dim", "opening checkpoint train_cache_contract"
    )
    cache_token_count = _positive_int(
        cache_contract, "token_count", "opening checkpoint train_cache_contract"
    )
    if cache_feature_dim != expected_feature_dim:
        raise ValueError(
            "opening checkpoint train cache feature_dim={} does not match "
            "Bi-Layout patch_dim={}".format(cache_feature_dim, expected_feature_dim)
        )
    if cache_token_count != expected_token_count:
        raise ValueError(
            "opening checkpoint token_count={} does not match Bi-Layout "
            "patch_num={}".format(cache_token_count, expected_token_count)
        )

    config_identity = _validate_file_identity(
        cache_contract.get("bi_layout_config"),
        Path(bi_layout_config_path),
        "bi_layout_config",
    )
    checkpoint_identity = _validate_file_identity(
        cache_contract.get("bi_layout_checkpoint"),
        Path(bi_layout_checkpoint_path),
        "bi_layout_checkpoint",
    )

    raw_threshold = payload.get("operating_threshold")
    if raw_threshold is None or isinstance(raw_threshold, bool):
        raise ValueError(
            "opening checkpoint operating_threshold must be present and in [0, 1]"
        )
    try:
        operating_threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "opening checkpoint operating_threshold must be present and in [0, 1]"
        ) from exc
    if not math.isfinite(operating_threshold) or not 0.0 <= operating_threshold <= 1.0:
        raise ValueError(
            "opening checkpoint operating_threshold must be present and in [0, 1]"
        )

    opening_head.load_state_dict(state_dict, strict=True)
    opening_head.eval()
    return {
        "path": str(path),
        "loaded": True,
        "formatVersion": payload.get("format_version"),
        "task": payload.get("task"),
        "completedEpoch": payload.get("completed_epoch"),
        "operatingThreshold": operating_threshold,
        "thresholdPolicy": payload.get("threshold_policy"),
        "thresholdFallback": payload.get("threshold_fallback"),
        "branchOrder": saved_branch_order,
        "model": dict(model_config),
        "backboneContract": {
            "validated": True,
            "tokenCount": cache_token_count,
            "featureDim": cache_feature_dim,
            "biLayoutConfig": config_identity,
            "biLayoutCheckpoint": checkpoint_identity,
        },
    }


def resolve_opening_probability_threshold(
    explicit_threshold: Optional[float],
    opening_checkpoint_report: Optional[Mapping[str, Any]],
    matcher_checkpoint_report: Optional[Mapping[str, Any]] = None,
    *,
    fallback: float = DEFAULT_OPENING_PROBABILITY_THRESHOLD,
) -> Dict[str, Any]:
    """Resolve CLI > explicit Opening Head > full Matcher > legacy fallback."""
    if explicit_threshold is not None:
        threshold = float(explicit_threshold)
        source = "cli"
    elif opening_checkpoint_report and opening_checkpoint_report.get("loaded"):
        threshold = float(opening_checkpoint_report["operatingThreshold"])
        source = "opening_checkpoint"
    elif matcher_checkpoint_report and matcher_checkpoint_report.get("loaded"):
        threshold = float(matcher_checkpoint_report["openingProbabilityThreshold"])
        source = "matcher_checkpoint"
    else:
        threshold = float(fallback)
        source = "legacy_default"
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("opening_probability_threshold must be in [0, 1]")
    return {"value": threshold, "source": source}

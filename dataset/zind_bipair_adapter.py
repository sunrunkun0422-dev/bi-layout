"""Adapter from legacy ZInD-BiPair fields to canonical cross-scene contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from models.cross_scene_contracts import (
    CoordinateFrameSpec,
    OpeningCandidates,
    PairBatch,
)


Tensor = torch.Tensor


def _require_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"ZInD-BiPair batch is missing tensor field {key!r}")
    return value


def _batched_tokens(value: Tensor, key: str) -> Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{key} must have shape [B, N] or [N]")
    return value


def _circular_components(mask: Tensor) -> List[Tuple[int, int, Tensor]]:
    """Return inclusive intervals and masks for circular connected components."""
    mask = mask.bool().flatten()
    token_count = int(mask.numel())
    if token_count == 0:
        raise ValueError("opening mask must contain at least one token")
    if not bool(mask.any()):
        return []
    if bool(mask.all()):
        return [(0, token_count - 1, mask.clone())]

    starts = torch.nonzero(mask & ~torch.roll(mask, shifts=1), as_tuple=False)
    components: List[Tuple[int, int, Tensor]] = []
    for start_tensor in starts.flatten():
        start = int(start_tensor.item())
        end = start
        while bool(mask[(end + 1) % token_count]):
            end = (end + 1) % token_count
        component = torch.zeros_like(mask, dtype=torch.bool)
        if start <= end:
            component[start : end + 1] = True
        else:
            component[start:] = True
            component[: end + 1] = True
        components.append((start, end, component))
    components.sort(key=lambda item: item[0])
    return components


def opening_mask_to_candidate_masks(opening_all: Tensor) -> OpeningCandidates:
    """Split every opening union mask into padded circular candidate masks.

    This function intentionally accepts only the all-opening mask.  Pair-level
    ``portal_mask`` labels are not an input because they reveal which opening is
    shared and would leak the matching target into candidate generation.
    """
    opening_all = _batched_tokens(opening_all, "opening_all").bool()
    components = [_circular_components(mask) for mask in opening_all]
    batch_size, token_count = opening_all.shape
    # Keep one invalid padding row for an all-empty batch.  This makes the
    # contract network-friendly without inventing a valid opening candidate.
    max_candidates = max(1, max(len(sample) for sample in components))
    masks = torch.zeros(
        (batch_size, max_candidates, token_count),
        dtype=torch.bool,
        device=opening_all.device,
    )
    valid = torch.zeros(
        (batch_size, max_candidates), dtype=torch.bool, device=opening_all.device
    )
    candidate_ids = torch.full(
        (batch_size, max_candidates),
        -1,
        dtype=torch.long,
        device=opening_all.device,
    )
    intervals = torch.full(
        (batch_size, max_candidates, 2),
        -1,
        dtype=torch.long,
        device=opening_all.device,
    )
    for batch_index, sample_components in enumerate(components):
        for candidate_index, (start, end, component) in enumerate(sample_components):
            masks[batch_index, candidate_index] = component
            valid[batch_index, candidate_index] = True
            candidate_ids[batch_index, candidate_index] = candidate_index
            intervals[batch_index, candidate_index] = torch.tensor(
                [start, end], dtype=torch.long, device=opening_all.device
            )
    return OpeningCandidates(
        masks=masks,
        valid=valid,
        candidate_ids=candidate_ids,
        intervals=intervals,
    )


def _metadata(batch: Mapping[str, Any]) -> Dict[str, Any]:
    tensor_or_label_keys = {
        "image_A",
        "image_B",
        "depth_enclosed_A",
        "depth_enclosed_B",
        "depth_extended_A",
        "depth_extended_B",
        "opening_mask_all_A",
        "opening_mask_all_B",
        "portal_mask_A",
        "portal_mask_B",
        "affinity_gt",
        "is_positive",
        "T_B_to_A",
        "relative_yaw_gt",
        "scale_meters_per_coordinate",
    }
    return {key: value for key, value in batch.items() if key not in tensor_or_label_keys}


def adapt_zind_bipair_batch(batch: Mapping[str, Any]) -> PairBatch:
    """Adapt a collated legacy ZInD-BiPair batch without mutating it."""
    opening_all_a = _batched_tokens(
        _require_tensor(batch, "opening_mask_all_A"), "opening_mask_all_A"
    ).bool()
    opening_all_b = _batched_tokens(
        _require_tensor(batch, "opening_mask_all_B"), "opening_mask_all_B"
    ).bool()
    batch_size = int(opening_all_a.shape[0])
    if opening_all_b.shape != opening_all_a.shape:
        raise ValueError("opening_mask_all_A/B must have the same [B, N] shape")

    transform = _require_tensor(batch, "T_B_to_A")
    if transform.ndim == 2:
        transform = transform.unsqueeze(0)
    relative_yaw = _require_tensor(batch, "relative_yaw_gt").reshape(batch_size)
    is_match = _require_tensor(batch, "is_positive").reshape(batch_size).bool()
    pose_valid_value = batch.get("pose_valid")
    if torch.is_tensor(pose_valid_value):
        pose_valid = pose_valid_value.reshape(batch_size).bool()
    else:
        # ZInD-BiPair stores a valid relative transform for every mined pair,
        # including no-shared-opening negatives.
        pose_valid = torch.ones_like(is_match, dtype=torch.bool)

    scale = batch.get("scale_meters_per_coordinate")
    if torch.is_tensor(scale):
        scale = scale.reshape(batch_size, -1)

    return PairBatch(
        image_a=batch.get("image_A") if torch.is_tensor(batch.get("image_A")) else None,
        image_b=batch.get("image_B") if torch.is_tensor(batch.get("image_B")) else None,
        depth_enclosed_a=_batched_tokens(
            _require_tensor(batch, "depth_enclosed_A"), "depth_enclosed_A"
        ).float(),
        depth_enclosed_b=_batched_tokens(
            _require_tensor(batch, "depth_enclosed_B"), "depth_enclosed_B"
        ).float(),
        depth_extended_a=_batched_tokens(
            _require_tensor(batch, "depth_extended_A"), "depth_extended_A"
        ).float(),
        depth_extended_b=_batched_tokens(
            _require_tensor(batch, "depth_extended_B"), "depth_extended_B"
        ).float(),
        opening_all_a=opening_all_a,
        opening_all_b=opening_all_b,
        candidates_a=opening_mask_to_candidate_masks(opening_all_a),
        candidates_b=opening_mask_to_candidate_masks(opening_all_b),
        shared_portal_a=_batched_tokens(
            _require_tensor(batch, "portal_mask_A"), "portal_mask_A"
        ).bool(),
        shared_portal_b=_batched_tokens(
            _require_tensor(batch, "portal_mask_B"), "portal_mask_B"
        ).bool(),
        affinity_ab=_require_tensor(batch, "affinity_gt").float(),
        is_match=is_match,
        transform_b_to_a=transform.float(),
        relative_yaw=relative_yaw.float(),
        pose_valid=pose_valid,
        frame=CoordinateFrameSpec(
            frame="zind_pano_local_xy",
            units="zind_coordinate_units",
            transform_direction="b_to_a",
        ),
        scale_meters_per_coordinate=scale.float() if torch.is_tensor(scale) else None,
        metadata=_metadata(batch),
    )


def canonicalize_zind_bipair_batch(
    batch: Mapping[str, Any], keep_legacy: bool = True
) -> Dict[str, Any]:
    """Return mapping form for gradual migration of existing training code.

    With the default ``keep_legacy=True`` all current uppercase/raw keys remain
    available and canonical snake_case keys are added alongside them.
    """
    canonical = adapt_zind_bipair_batch(batch).as_dict()
    if not keep_legacy:
        return canonical
    output = dict(batch)
    output.update(canonical)
    return output

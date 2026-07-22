"""Typed tensor contracts shared by cross-scene training stages.

The existing project historically passes loose dictionaries whose field names
depend on the dataset or model entry point.  The dataclasses in this module are
the canonical, snake_case interface.  They deliberately do not perform model
logic, thresholding, or geometry post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class CoordinateFrameSpec:
    """Explicit coordinate convention for a tensor or transform."""

    frame: str
    units: str
    transform_direction: str = ""

    def __post_init__(self) -> None:
        if not self.frame:
            raise ValueError("coordinate frame must be non-empty")
        if not self.units:
            raise ValueError("coordinate units must be non-empty")


@dataclass(frozen=True)
class OpeningCandidates:
    """Padded opening candidates for a batch of circular token sequences.

    ``masks`` has shape ``[B, K, N]``.  Padding rows are represented by an
    all-false mask, ``valid=False``, ``candidate_ids=-1`` and interval
    ``[-1, -1]``.  Intervals are inclusive; ``start > end`` denotes an
    equirectangular seam-crossing interval.
    """

    masks: Tensor
    valid: Tensor
    candidate_ids: Tensor
    intervals: Tensor

    def __post_init__(self) -> None:
        if self.masks.ndim != 3:
            raise ValueError("opening candidate masks must have shape [B, K, N]")
        batch_size, candidate_count, _ = self.masks.shape
        expected = (batch_size, candidate_count)
        if tuple(self.valid.shape) != expected:
            raise ValueError("opening candidate valid mask must have shape [B, K]")
        if tuple(self.candidate_ids.shape) != expected:
            raise ValueError("opening candidate ids must have shape [B, K]")
        if tuple(self.intervals.shape) != expected + (2,):
            raise ValueError("opening candidate intervals must have shape [B, K, 2]")
        if self.masks.dtype != torch.bool or self.valid.dtype != torch.bool:
            raise TypeError("opening candidate masks and valid flags must be bool")
        if self.candidate_ids.dtype != torch.long or self.intervals.dtype != torch.long:
            raise TypeError("opening candidate ids and intervals must be torch.long")

    @property
    def batch_size(self) -> int:
        return int(self.masks.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.masks.shape[1])

    @property
    def token_count(self) -> int:
        return int(self.masks.shape[2])

    def union_mask(self) -> Tensor:
        return (self.masks & self.valid.unsqueeze(-1)).any(dim=1)

    def to(self, *args, **kwargs) -> "OpeningCandidates":
        return replace(
            self,
            masks=self.masks.to(*args, **kwargs),
            valid=self.valid.to(*args, **kwargs),
            candidate_ids=self.candidate_ids.to(*args, **kwargs),
            intervals=self.intervals.to(*args, **kwargs),
        )


@dataclass(frozen=True)
class SingleViewOutput:
    """Canonical output of one panorama's shared Bi-Layout branch."""

    feature_1d: Tensor
    depth_enclosed: Tensor
    depth_extended: Tensor
    opening_logits: Optional[Tensor] = None
    opening_probability: Optional[Tensor] = None
    opening_candidates: Optional[OpeningCandidates] = None
    feature_2d_pyramid: Optional[Tuple[Tensor, ...]] = None
    frame: CoordinateFrameSpec = CoordinateFrameSpec(
        frame="camera_local_xz", units="model_depth_units"
    )

    def __post_init__(self) -> None:
        if self.feature_1d.ndim != 3:
            raise ValueError("feature_1d must have shape [B, N, C]")
        batch_size, token_count, _ = self.feature_1d.shape
        expected = (batch_size, token_count)
        for name, tensor in (
            ("depth_enclosed", self.depth_enclosed),
            ("depth_extended", self.depth_extended),
            ("opening_logits", self.opening_logits),
            ("opening_probability", self.opening_probability),
        ):
            if tensor is not None and tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape [B, N]")
        if self.opening_candidates is not None:
            candidates = self.opening_candidates
            if candidates.batch_size != batch_size or candidates.token_count != token_count:
                raise ValueError("opening candidates must align with feature_1d")


@dataclass(frozen=True)
class PairBatch:
    """Canonical supervision contract for a pair of panorama views.

    ``opening_all_*`` supervises single-view opening detection.
    ``candidates_*`` contains every opening in that view.
    ``shared_portal_*`` identifies only the opening shared by this pair and
    must never be substituted for ``opening_all_*`` or all-view candidates.
    """

    image_a: Optional[Tensor]
    image_b: Optional[Tensor]
    depth_enclosed_a: Tensor
    depth_enclosed_b: Tensor
    depth_extended_a: Tensor
    depth_extended_b: Tensor
    opening_all_a: Tensor
    opening_all_b: Tensor
    candidates_a: OpeningCandidates
    candidates_b: OpeningCandidates
    shared_portal_a: Tensor
    shared_portal_b: Tensor
    affinity_ab: Tensor
    is_match: Tensor
    transform_b_to_a: Tensor
    relative_yaw: Tensor
    pose_valid: Tensor
    frame: CoordinateFrameSpec
    scale_meters_per_coordinate: Optional[Tensor] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.opening_all_a.ndim != 2 or self.opening_all_b.ndim != 2:
            raise ValueError("opening_all tensors must have shape [B, N]")
        if tuple(self.opening_all_a.shape) != tuple(self.opening_all_b.shape):
            raise ValueError("paired opening_all tensors must share shape")
        batch_size, token_count = self.opening_all_a.shape
        token_shape = (batch_size, token_count)
        for name, tensor in (
            ("depth_enclosed_a", self.depth_enclosed_a),
            ("depth_enclosed_b", self.depth_enclosed_b),
            ("depth_extended_a", self.depth_extended_a),
            ("depth_extended_b", self.depth_extended_b),
            ("shared_portal_a", self.shared_portal_a),
            ("shared_portal_b", self.shared_portal_b),
        ):
            if tuple(tensor.shape) != token_shape:
                raise ValueError(f"{name} must have shape [B, N]")
        if self.candidates_a.batch_size != batch_size or self.candidates_a.token_count != token_count:
            raise ValueError("candidates_a must align with opening_all_a")
        if self.candidates_b.batch_size != batch_size or self.candidates_b.token_count != token_count:
            raise ValueError("candidates_b must align with opening_all_b")
        if not torch.equal(self.candidates_a.union_mask(), self.opening_all_a.bool()):
            raise ValueError("candidates_a must be derived from every opening in opening_all_a")
        if not torch.equal(self.candidates_b.union_mask(), self.opening_all_b.bool()):
            raise ValueError("candidates_b must be derived from every opening in opening_all_b")
        if tuple(self.affinity_ab.shape) != (batch_size, token_count, token_count):
            raise ValueError("affinity_ab must have shape [B, N, N]")
        if tuple(self.transform_b_to_a.shape) != (batch_size, 3, 3):
            raise ValueError("transform_b_to_a must have shape [B, 3, 3]")
        for name, tensor in (("is_match", self.is_match), ("pose_valid", self.pose_valid)):
            if tuple(tensor.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
        if self.relative_yaw.numel() != batch_size:
            raise ValueError("relative_yaw must contain one value per pair")
        for name, image in (("image_a", self.image_a), ("image_b", self.image_b)):
            if image is not None and (image.ndim != 4 or image.shape[0] != batch_size):
                raise ValueError(f"{name} must have shape [B, C, H, W]")

    @property
    def batch_size(self) -> int:
        return int(self.opening_all_a.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.opening_all_a.shape[1])

    def as_dict(self) -> Dict[str, Any]:
        """Return canonical snake_case fields for model/trainer entry points."""
        return {
            "image_a": self.image_a,
            "image_b": self.image_b,
            "depth_enclosed_a": self.depth_enclosed_a,
            "depth_enclosed_b": self.depth_enclosed_b,
            "depth_extended_a": self.depth_extended_a,
            "depth_extended_b": self.depth_extended_b,
            "opening_all_a": self.opening_all_a,
            "opening_all_b": self.opening_all_b,
            "candidate_masks_all_a": self.candidates_a.masks,
            "candidate_masks_all_b": self.candidates_b.masks,
            "candidate_valid_all_a": self.candidates_a.valid,
            "candidate_valid_all_b": self.candidates_b.valid,
            "candidate_ids_all_a": self.candidates_a.candidate_ids,
            "candidate_ids_all_b": self.candidates_b.candidate_ids,
            "candidate_intervals_all_a": self.candidates_a.intervals,
            "candidate_intervals_all_b": self.candidates_b.intervals,
            "shared_portal_a": self.shared_portal_a,
            "shared_portal_b": self.shared_portal_b,
            "affinity_ab": self.affinity_ab,
            "is_match": self.is_match,
            "transform_b_to_a": self.transform_b_to_a,
            "relative_yaw": self.relative_yaw,
            "pose_valid": self.pose_valid,
            "scale_meters_per_coordinate": self.scale_meters_per_coordinate,
            "frame": self.frame,
            "metadata": self.metadata,
        }

    def to(self, *args, **kwargs) -> "PairBatch":
        def move(value):
            return None if value is None else value.to(*args, **kwargs)

        return replace(
            self,
            image_a=move(self.image_a),
            image_b=move(self.image_b),
            depth_enclosed_a=move(self.depth_enclosed_a),
            depth_enclosed_b=move(self.depth_enclosed_b),
            depth_extended_a=move(self.depth_extended_a),
            depth_extended_b=move(self.depth_extended_b),
            opening_all_a=move(self.opening_all_a),
            opening_all_b=move(self.opening_all_b),
            candidates_a=self.candidates_a.to(*args, **kwargs),
            candidates_b=self.candidates_b.to(*args, **kwargs),
            shared_portal_a=move(self.shared_portal_a),
            shared_portal_b=move(self.shared_portal_b),
            affinity_ab=move(self.affinity_ab),
            is_match=move(self.is_match),
            transform_b_to_a=move(self.transform_b_to_a),
            relative_yaw=move(self.relative_yaw),
            pose_valid=move(self.pose_valid),
            scale_meters_per_coordinate=move(self.scale_meters_per_coordinate),
        )


@dataclass(frozen=True)
class MatcherOutput:
    """Canonical output contract of a cross-scene matching network."""

    token_affinity_ab: Tensor
    token_affinity_ba: Tensor
    candidate_assignment: Optional[Tensor] = None
    token_no_match_a: Optional[Tensor] = None
    token_no_match_b: Optional[Tensor] = None
    shared_opening_logits_a: Optional[Tensor] = None
    shared_opening_logits_b: Optional[Tensor] = None
    pose_hypotheses: Optional[Tensor] = None
    pose_scores: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.token_affinity_ab.ndim != 3 or self.token_affinity_ba.ndim != 3:
            raise ValueError("token affinities must have shape [B, N, N]")
        if self.token_affinity_ab.shape != self.token_affinity_ba.transpose(1, 2).shape:
            raise ValueError("AB and BA token affinity shapes are inconsistent")

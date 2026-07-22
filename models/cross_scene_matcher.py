"""Opening-guided cross attention for matching two panorama layouts."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interval_endpoints(interval) -> Tuple[int, int]:
    if hasattr(interval, "token_start") and hasattr(interval, "token_end"):
        return int(interval.token_start), int(interval.token_end)
    if isinstance(interval, Mapping):
        start = interval.get("token_start", interval.get("tokenStart", -1))
        end = interval.get("token_end", interval.get("tokenEnd", -1))
        return int(start), int(end)
    if len(interval) != 2:
        raise ValueError("Each opening interval must contain (token_start, token_end)")
    return int(interval[0]), int(interval[1])


def candidate_intervals_to_mask(
    intervals: Iterable,
    length: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert inclusive, possibly circular opening intervals to [K, N] masks."""
    if length <= 0:
        raise ValueError("length must be positive")

    intervals = list(intervals)
    masks = torch.zeros((len(intervals), length), dtype=torch.bool, device=device)
    for index, interval in enumerate(intervals):
        start, end = _interval_endpoints(interval)
        if start < 0 or end < 0:
            continue
        start %= length
        end %= length
        if start <= end:
            masks[index, start:end + 1] = True
        else:
            masks[index, start:] = True
            masks[index, :end + 1] = True
    return masks


def opening_probabilities_to_intervals(
    probability: torch.Tensor,
    threshold: float = 0.12,
    min_width_tokens: int = 2,
    max_intervals: int = 12,
) -> list:
    """Extract confidence-ranked circular components from a 1D opening score.

    Unlike fixed-radius top-k peaks, this preserves the predicted opening
    width and merges a component that crosses the panorama seam.  The default
    threshold is calibrated on the ZInD-BiPair-v1 validation split for the
    untrained geometry prior; callers using a trained opening head should pass
    their own validation threshold.
    """
    values = probability.detach().reshape(-1)
    token_count = int(values.numel())
    if token_count <= 0:
        raise ValueError("probability must contain at least one token")
    if not torch.isfinite(values).all():
        raise ValueError("probability must contain only finite values")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if min_width_tokens <= 0 or max_intervals <= 0:
        raise ValueError("min_width_tokens and max_intervals must be positive")

    mask = values >= float(threshold)
    if not bool(mask.any()):
        return []
    if bool(mask.all()):
        return [(0, token_count - 1)] if token_count >= min_width_tokens else []

    starts = torch.where(mask & ~torch.roll(mask, shifts=1))[0].tolist()
    ranked = []
    for start in starts:
        end = int(start)
        tokens = [int(start)]
        while bool(mask[(end + 1) % token_count]):
            end = (end + 1) % token_count
            tokens.append(end)
        if len(tokens) < int(min_width_tokens):
            continue
        index = torch.as_tensor(tokens, device=values.device, dtype=torch.long)
        component_score = float(values[index].mean().item())
        ranked.append((component_score, len(tokens), int(start), int(end)))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [(start, end) for _, _, start, end in ranked[:max_intervals]]


def resolve_enclosed_extended_depth(
    output: Mapping[str, torch.Tensor],
    branch_order: str = "extended_first",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map Bi-Layout's two depth branches to explicit geometry semantics.

    The ZInD two-head checkpoint is trained with ``depth=layout_visible`` and
    ``new_depth=layout_raw``.  In the opening pipeline those tensors mean
    ``extended`` and ``enclosed`` respectively.  Keeping this mapping explicit
    prevents the positive opening contrast from being accidentally reversed.

    ``enclosed_first`` remains available for checkpoints trained with the
    opposite label order.
    """
    if "depth" not in output or "new_depth" not in output:
        raise ValueError("Bi-Layout output must contain depth and new_depth")
    if branch_order == "extended_first":
        return output["new_depth"], output["depth"]
    if branch_order == "enclosed_first":
        return output["depth"], output["new_depth"]
    raise ValueError(
        "branch_order must be 'extended_first' or 'enclosed_first'"
    )


class OpeningSignalHead(nn.Module):
    """Predict per-token opening response and usable expansion depth."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        prior_strength: float = 4.0,
        prior_relative_scale: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if prior_relative_scale <= 0:
            raise ValueError("prior_relative_scale must be positive")

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Conv1d(feature_dim + 3, hidden_dim, kernel_size=1)
        self.context = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=0)
        self.output = nn.Conv1d(hidden_dim, 2, kernel_size=1)
        self.kernel_padding = kernel_size // 2
        self.prior_strength = float(prior_strength)
        self.prior_relative_scale = float(prior_relative_scale)
        self.eps = float(eps)

        # Before training, D_ext - D_enc still provides a meaningful opening prior.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        enclosed_depth: torch.Tensor,
        extended_depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, N, C]")
        if enclosed_depth.shape != features.shape[:2]:
            raise ValueError("enclosed_depth must have shape [B, N]")
        if extended_depth.shape != features.shape[:2]:
            raise ValueError("extended_depth must have shape [B, N]")

        enclosed = enclosed_depth.abs().clamp_min(self.eps)
        extended = extended_depth.abs().clamp_min(self.eps)
        delta = F.relu(extended - enclosed)
        relative_delta = (delta / enclosed).clamp(max=10.0)
        log_ratio = F.relu(torch.log(extended / enclosed)).clamp(max=4.0)

        relative_scale = relative_delta.amax(dim=1, keepdim=True).clamp_min(self.eps)
        relative_peak = (relative_delta / relative_scale).clamp(0.0, 1.0)
        magnitude_gate = relative_delta / (relative_delta + self.prior_relative_scale)
        geometry_prior = relative_peak * magnitude_gate
        geometry = torch.stack((delta, relative_delta, log_ratio), dim=-1)

        x = torch.cat((self.feature_norm(features), geometry), dim=-1).transpose(1, 2)
        x = F.gelu(self.input_projection(x))
        x = F.pad(x, (self.kernel_padding, self.kernel_padding), mode="circular")
        x = F.gelu(self.context(x))
        learned_opening_logits, depth_scale_logits = self.output(x).chunk(2, dim=1)
        learned_opening_logits = learned_opening_logits.squeeze(1)
        depth_scale_logits = depth_scale_logits.squeeze(1)

        prior_logits = self.prior_strength * (geometry_prior - 0.5)
        opening_logits = learned_opening_logits + prior_logits
        opening_probability = torch.sigmoid(opening_logits)
        depth_scale = torch.exp(torch.tanh(depth_scale_logits))
        expansion_depth = delta * opening_probability * depth_scale

        return {
            "opening_logits": opening_logits,
            "opening_probability": opening_probability,
            "expansion_depth": expansion_depth,
            "delta": delta,
            "relative_delta": relative_delta,
            "log_ratio": log_ratio,
        }


def opening_detection_loss(
    opening_logits: torch.Tensor,
    opening_target: torch.Tensor,
    pos_weight: float = 2.5,
    tversky_weight: float = 0.5,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Recall-oriented supervised loss for the single-view Opening Head."""
    if opening_logits.ndim != 2:
        raise ValueError("opening_logits must have shape [B, N]")
    if opening_target.shape != opening_logits.shape:
        raise ValueError("opening_target must match opening_logits")
    if pos_weight <= 0 or tversky_weight < 0:
        raise ValueError("pos_weight must be positive and tversky_weight non-negative")
    if min(tversky_alpha, tversky_beta, eps) <= 0:
        raise ValueError("Tversky alpha, beta, and eps must be positive")

    target = opening_target.to(
        device=opening_logits.device,
        dtype=opening_logits.dtype,
    )
    positive_weight = opening_logits.new_tensor(float(pos_weight))
    bce = F.binary_cross_entropy_with_logits(
        opening_logits,
        target,
        pos_weight=positive_weight,
    )
    probability = torch.sigmoid(opening_logits)
    reduce_dims = tuple(range(1, opening_logits.ndim))
    true_positive = (probability * target).sum(dim=reduce_dims)
    false_positive = (probability * (1.0 - target)).sum(dim=reduce_dims)
    false_negative = ((1.0 - probability) * target).sum(dim=reduce_dims)
    tversky_index = (true_positive + eps) / (
        true_positive
        + float(tversky_alpha) * false_positive
        + float(tversky_beta) * false_negative
        + eps
    )
    tversky = (1.0 - tversky_index).mean()
    total = bce + float(tversky_weight) * tversky
    return {
        "loss_total": total,
        "loss_bce": bce,
        "loss_tversky": tversky,
    }


class OpeningTokenPooler(nn.Module):
    """Pool token features inside each candidate opening interval."""

    def forward(
        self,
        features: torch.Tensor,
        opening_probability: torch.Tensor,
        candidate_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_masks.ndim == 2:
            candidate_masks = candidate_masks.unsqueeze(0)
        if candidate_masks.ndim != 3:
            raise ValueError("candidate_masks must have shape [B, K, N] or [K, N]")
        if candidate_masks.shape[0] == 1 and features.shape[0] > 1:
            candidate_masks = candidate_masks.expand(features.shape[0], -1, -1)
        if candidate_masks.shape[0] != features.shape[0]:
            raise ValueError("candidate mask batch size does not match features")
        if candidate_masks.shape[2] != features.shape[1]:
            raise ValueError("candidate mask token count does not match features")

        masks = candidate_masks.to(device=features.device, dtype=features.dtype)
        valid = masks.sum(dim=-1) > 0
        weights = masks * opening_probability.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bkn,bnc->bkc", weights, features)
        pooled = pooled * valid.unsqueeze(-1).to(features.dtype)
        return pooled, weights, valid


class OpeningGuidedCrossAttentionMatcher(nn.Module):
    """Bidirectionally match opening tokens and opening candidates."""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        opening_bias_strength: float = 1.0,
        candidate_temperature: float = 0.2,
        shift_temperature: float = 0.05,
        dustbin_score: float = 0.0,
    ):
        super().__init__()
        if feature_dim % heads != 0:
            raise ValueError("feature_dim must be divisible by heads")
        if candidate_temperature <= 0 or shift_temperature <= 0:
            raise ValueError("matching temperatures must be positive")

        self.feature_dim = int(feature_dim)
        self.heads = int(heads)
        self.head_dim = feature_dim // heads
        self.scale = self.head_dim ** -0.5
        self.opening_bias_strength = float(opening_bias_strength)
        self.candidate_temperature = float(candidate_temperature)
        self.shift_temperature = float(shift_temperature)

        self.opening_head = OpeningSignalHead(feature_dim, hidden_dim=hidden_dim)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.query_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        with torch.no_grad():
            self.key_projection.weight.copy_(self.query_projection.weight)
        self.value_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
        )
        self.pooler = OpeningTokenPooler()
        # A learned score for assigning an opening candidate to ``no match``.
        # Keeping it in the matcher (instead of using a fixed threshold) lets
        # negative pairs directly supervise rejection.
        self.dustbin_score = nn.Parameter(torch.tensor(float(dustbin_score)))

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _prepare_candidate_valid(
        candidate_valid: Optional[torch.Tensor],
        candidate_masks: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a batch-shaped validity mask and reject ambiguous padding."""
        mask_valid = candidate_masks.to(device=device, dtype=torch.bool).any(dim=-1)
        if candidate_valid is None:
            return mask_valid
        if candidate_valid.ndim == 1:
            candidate_valid = candidate_valid.unsqueeze(0)
        if candidate_valid.ndim != 2:
            raise ValueError("candidate_valid must have shape [B, K] or [K]")
        if candidate_valid.shape[0] == 1 and batch_size > 1:
            candidate_valid = candidate_valid.expand(batch_size, -1)
        if candidate_valid.shape != mask_valid.shape:
            raise ValueError("candidate_valid shape must match candidate_masks [B, K]")
        return mask_valid & candidate_valid.to(device=device, dtype=torch.bool)

    @classmethod
    def _token_mask(
        cls,
        candidate_masks: Optional[torch.Tensor],
        reference: torch.Tensor,
        candidate_valid: Optional[torch.Tensor] = None,
    ):
        if candidate_masks is None:
            return None
        if candidate_masks.ndim == 2:
            candidate_masks = candidate_masks.unsqueeze(0)
        if candidate_masks.shape[0] == 1 and reference.shape[0] > 1:
            candidate_masks = candidate_masks.expand(reference.shape[0], -1, -1)
        if candidate_masks.ndim != 3:
            raise ValueError("candidate_masks must have shape [B, K, N] or [K, N]")
        if candidate_masks.shape[0] != reference.shape[0]:
            raise ValueError("candidate mask batch size does not match features")
        if candidate_masks.shape[-1] != reference.shape[1]:
            raise ValueError("candidate mask token count does not match features")
        valid = cls._prepare_candidate_valid(
            candidate_valid,
            candidate_masks,
            reference.shape[0],
            reference.device,
        )
        mask = (
            candidate_masks.to(device=reference.device, dtype=torch.bool)
            & valid.unsqueeze(-1)
        ).any(dim=1)
        has_opening = mask.any(dim=-1, keepdim=True)
        return torch.where(has_opening, mask, torch.ones_like(mask))

    @staticmethod
    def _opening_guidance(
        prediction: torch.Tensor,
        external: Optional[torch.Tensor],
        mode: str,
        mix_weight: float,
    ) -> torch.Tensor:
        """Select the signal used by matching without replacing head outputs."""
        if mode not in ("predicted", "gt", "mix"):
            raise ValueError(
                "opening_guidance_mode must be 'predicted', 'gt', or 'mix'"
            )
        if not 0.0 <= float(mix_weight) <= 1.0:
            raise ValueError("opening_guidance_mix_weight must be in [0, 1]")
        if mode == "predicted":
            return prediction
        if external is None:
            raise ValueError(
                "external opening guidance is required for 'gt' and 'mix' modes"
            )
        if external.ndim == 1:
            external = external.unsqueeze(0)
        if external.shape[0] == 1 and prediction.shape[0] > 1:
            external = external.expand(prediction.shape[0], -1)
        if external.shape != prediction.shape:
            raise ValueError("opening guidance must match prediction shape [B, N]")
        external = external.to(device=prediction.device, dtype=prediction.dtype)
        if not torch.isfinite(external).all():
            raise ValueError("opening guidance must contain only finite values")
        external = external.clamp(0.0, 1.0)
        if mode == "gt":
            return external
        weight = float(mix_weight)
        return (1.0 - weight) * prediction + weight * external

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_opening: torch.Tensor,
        key_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        opening_bias = torch.log(key_opening.clamp_min(1e-6))
        logits = logits + self.opening_bias_strength * opening_bias[:, None, None, :]
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(value.shape[0], -1, self.feature_dim)
        return context, attention

    def _cyclic_shift_scores(
        self,
        affinity_ab: torch.Tensor,
        opening_a: torch.Tensor,
        opening_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        token_count = affinity_ab.shape[-1]
        source = torch.arange(token_count, device=affinity_ab.device)[:, None]
        shifts = torch.arange(token_count, device=affinity_ab.device)[None, :]
        target = (source + shifts) % token_count
        diagonal_affinity = affinity_ab[:, source, target]
        opening_pairs = opening_a[:, source] * opening_b[:, target]
        shift_mass = (diagonal_affinity * opening_pairs).sum(dim=1)
        shift_mass = shift_mass / opening_a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        shift_probability = torch.softmax(shift_mass / self.shift_temperature, dim=-1)
        best_shift = shift_mass.argmax(dim=-1)
        signed_shift = torch.where(
            best_shift > token_count // 2,
            best_shift - token_count,
            best_shift,
        )
        yaw = signed_shift.to(affinity_ab.dtype) * (2.0 * math.pi / token_count)
        return {
            "cyclic_shift_mass": shift_mass,
            "cyclic_shift_score": shift_probability,
            "best_cyclic_shift": best_shift,
            # Canonical meaning: horizontal B-token minus A-token shift.  This
            # is not generally the camera relative yaw when camera centers
            # differ.  Keep the historical key below as a compatibility alias.
            "relative_token_shift_radians": yaw,
            "relative_yaw_radians": yaw,
        }

    def _candidate_matches(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        opening_a: torch.Tensor,
        opening_b: torch.Tensor,
        candidate_masks_a: torch.Tensor,
        candidate_masks_b: torch.Tensor,
        candidate_valid_a: Optional[torch.Tensor] = None,
        candidate_valid_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if candidate_masks_a.ndim == 2:
            candidate_masks_a = candidate_masks_a.unsqueeze(0)
        if candidate_masks_b.ndim == 2:
            candidate_masks_b = candidate_masks_b.unsqueeze(0)
        if candidate_masks_a.shape[0] == 1 and features_a.shape[0] > 1:
            candidate_masks_a = candidate_masks_a.expand(features_a.shape[0], -1, -1)
        if candidate_masks_b.shape[0] == 1 and features_b.shape[0] > 1:
            candidate_masks_b = candidate_masks_b.expand(features_b.shape[0], -1, -1)
        candidate_masks_a = candidate_masks_a.to(device=features_a.device, dtype=torch.bool)
        candidate_masks_b = candidate_masks_b.to(device=features_b.device, dtype=torch.bool)

        pooled_a, weights_a, mask_valid_a = self.pooler(
            features_a, opening_a, candidate_masks_a
        )
        pooled_b, weights_b, mask_valid_b = self.pooler(
            features_b, opening_b, candidate_masks_b
        )
        batch, count_a, _ = pooled_a.shape
        count_b = pooled_b.shape[1]
        valid_a = mask_valid_a & self._prepare_candidate_valid(
            candidate_valid_a,
            candidate_masks_a,
            batch,
            features_a.device,
        )
        valid_b = mask_valid_b & self._prepare_candidate_valid(
            candidate_valid_b,
            candidate_masks_b,
            batch,
            features_b.device,
        )

        candidate_a = F.normalize(self.query_projection(pooled_a), dim=-1)
        candidate_b = F.normalize(self.key_projection(pooled_b), dim=-1)
        logits = torch.einsum("bkc,blc->bkl", candidate_a, candidate_b)
        logits = logits / self.candidate_temperature

        pair_valid = valid_a.unsqueeze(-1) & valid_b.unsqueeze(1)
        neg_inf = torch.finfo(logits.dtype).min

        # The last row/column are dustbins.  Two directional row-softmaxes are
        # exposed because one matrix cannot simultaneously express all A->B
        # and B->A rejection probabilities when candidate counts differ.
        assignment_logits = logits.new_full(
            (batch, count_a + 1, count_b + 1),
            neg_inf,
        )
        assignment_logits[:, :count_a, :count_b] = logits.masked_fill(
            ~pair_valid, neg_inf
        )
        assignment_logits[:, :count_a, count_b] = torch.where(
            valid_a,
            self.dustbin_score.to(logits),
            logits.new_full((), neg_inf),
        )
        assignment_logits[:, count_a, :count_b] = torch.where(
            valid_b,
            self.dustbin_score.to(logits),
            logits.new_full((), neg_inf),
        )
        assignment_logits[:, count_a, count_b] = self.dustbin_score.to(logits)

        valid_rows_ab = torch.cat(
            (valid_a, torch.ones((batch, 1), dtype=torch.bool, device=logits.device)),
            dim=-1,
        )
        assignment_ab = torch.softmax(assignment_logits, dim=-1)
        assignment_ab = assignment_ab * valid_rows_ab.unsqueeze(-1).to(logits.dtype)

        assignment_ba = torch.softmax(assignment_logits.transpose(-2, -1), dim=-1)
        valid_rows_ba = torch.cat(
            (valid_b, torch.ones((batch, 1), dtype=torch.bool, device=logits.device)),
            dim=-1,
        )
        assignment_ba = assignment_ba * valid_rows_ba.unsqueeze(-1).to(logits.dtype)

        affinity_ab = assignment_ab[:, :count_a, :count_b]
        affinity_ba = assignment_ba[:, :count_b, :count_a]
        mutual_affinity = torch.sqrt(
            (affinity_ab * affinity_ba.transpose(-2, -1)).clamp_min(0.0)
        )
        no_match_a = assignment_ab[:, :count_a, count_b]
        no_match_b = assignment_ba[:, :count_b, count_a]

        assignment_probability = assignment_logits.new_zeros(
            assignment_logits.shape
        )
        assignment_probability[:, :count_a, :count_b] = mutual_affinity
        assignment_probability[:, :count_a, count_b] = no_match_a
        assignment_probability[:, count_a, :count_b] = no_match_b
        assignment_probability[:, count_a, count_b] = torch.sqrt(
            (
                assignment_ab[:, count_a, count_b]
                * assignment_ba[:, count_b, count_a]
            ).clamp_min(0.0)
        )

        mask_a = candidate_masks_a.to(features_a.dtype)
        mask_b = candidate_masks_b.to(features_b.dtype)
        confidence_a = (mask_a * opening_a.unsqueeze(1)).sum(-1)
        confidence_a = confidence_a / mask_a.sum(-1).clamp_min(1.0)
        confidence_b = (mask_b * opening_b.unsqueeze(1)).sum(-1)
        confidence_b = confidence_b / mask_b.sum(-1).clamp_min(1.0)
        pair_confidence = torch.sqrt(confidence_a.unsqueeze(-1) * confidence_b.unsqueeze(1))
        pair_score = mutual_affinity * pair_confidence * pair_valid.to(logits.dtype)

        best = torch.full((batch, 2), -1, dtype=torch.long, device=pooled_a.device)
        if count_a > 0 and count_b > 0:
            flat_best = pair_score.flatten(1).argmax(dim=-1)
            best_a = torch.div(flat_best, count_b, rounding_mode="floor")
            proposed_best = torch.stack(
                (best_a, flat_best.remainder(count_b)), dim=-1
            )
            chosen_a_to_b = affinity_ab > no_match_a.unsqueeze(-1)
            chosen_b_to_a = affinity_ba.transpose(-2, -1) > no_match_b.unsqueeze(1)
            accepted_pair = pair_valid & chosen_a_to_b & chosen_b_to_a
            has_pair = accepted_pair.flatten(1).any(dim=-1)
            # Select the best mutually accepted pair, not merely the largest
            # real-real cell when every candidate prefers its dustbin.
            accepted_score = pair_score.masked_fill(~accepted_pair, -1.0)
            accepted_best = accepted_score.flatten(1).argmax(dim=-1)
            accepted_a = torch.div(accepted_best, count_b, rounding_mode="floor")
            proposed_best = torch.stack(
                (accepted_a, accepted_best.remainder(count_b)), dim=-1
            )
            best = torch.where(has_pair.unsqueeze(-1), proposed_best, best)

        def valid_mean_or_one(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            count = valid.sum(dim=-1)
            mean = (values * valid.to(values.dtype)).sum(dim=-1)
            mean = mean / count.clamp_min(1).to(values.dtype)
            return torch.where(count > 0, mean, torch.ones_like(mean))

        mean_no_match_a = valid_mean_or_one(no_match_a, valid_a)
        mean_no_match_b = valid_mean_or_one(no_match_b, valid_b)
        pair_no_match = torch.sqrt(
            (mean_no_match_a * mean_no_match_b).clamp_min(0.0)
        )
        if count_a > 0 and count_b > 0:
            pair_match = pair_score.flatten(1).amax(dim=-1)
        else:
            pair_match = logits.new_zeros((batch,))

        return {
            "E_A_open": pooled_a,
            "E_B_open": pooled_b,
            "candidate_weights_A": weights_a,
            "candidate_weights_B": weights_b,
            "candidate_masks_A": candidate_masks_a,
            "candidate_masks_B": candidate_masks_b,
            "candidate_valid_A": valid_a,
            "candidate_valid_B": valid_b,
            "candidate_logits": logits,
            # Backward-compatible real-candidate view. Rows now sum to <= 1;
            # the remaining mass is explicitly assigned to the dustbin.
            "candidate_affinity": affinity_ab,
            "candidate_pair_score": pair_score,
            "best_candidate_pair": best,
            "candidate_assignment_logits": assignment_logits,
            "candidate_assignment_AB": assignment_ab,
            "candidate_assignment_BA": assignment_ba,
            "candidate_assignment_probability": assignment_probability,
            "candidate_assignment": assignment_probability,
            "candidate_no_match_probability_A": no_match_a,
            "candidate_no_match_probability_B": no_match_b,
            "no_match_probability_A": no_match_a,
            "no_match_probability_B": no_match_b,
            "pair_match_probability": pair_match,
            "pair_no_match_probability": pair_no_match,
        }

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        enclosed_depth_a: torch.Tensor,
        extended_depth_a: torch.Tensor,
        enclosed_depth_b: torch.Tensor,
        extended_depth_b: torch.Tensor,
        position_a: Optional[torch.Tensor] = None,
        position_b: Optional[torch.Tensor] = None,
        candidate_masks_a: Optional[torch.Tensor] = None,
        candidate_masks_b: Optional[torch.Tensor] = None,
        candidate_valid_a: Optional[torch.Tensor] = None,
        candidate_valid_b: Optional[torch.Tensor] = None,
        opening_guidance_a: Optional[torch.Tensor] = None,
        opening_guidance_b: Optional[torch.Tensor] = None,
        opening_guidance_mode: str = "predicted",
        opening_guidance_mix_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        if features_a.shape != features_b.shape:
            raise ValueError("features_a and features_b must have the same shape")

        signal_a = self.opening_head(features_a, enclosed_depth_a, extended_depth_a)
        signal_b = self.opening_head(features_b, enclosed_depth_b, extended_depth_b)
        opening_a = signal_a["opening_probability"]
        opening_b = signal_b["opening_probability"]
        guidance_a = self._opening_guidance(
            opening_a,
            opening_guidance_a,
            opening_guidance_mode,
            opening_guidance_mix_weight,
        )
        guidance_b = self._opening_guidance(
            opening_b,
            opening_guidance_b,
            opening_guidance_mode,
            opening_guidance_mix_weight,
        )
        token_mask_a = self._token_mask(
            candidate_masks_a, features_a, candidate_valid_a
        )
        token_mask_b = self._token_mask(
            candidate_masks_b, features_b, candidate_valid_b
        )

        normalized_a = self.input_norm(features_a)
        normalized_b = self.input_norm(features_b)
        query_input_a = normalized_a if position_a is None else normalized_a + position_a
        query_input_b = normalized_b if position_b is None else normalized_b + position_b
        query_a = self._split_heads(self.query_projection(query_input_a))
        query_b = self._split_heads(self.query_projection(query_input_b))
        key_a = self._split_heads(self.key_projection(query_input_a))
        key_b = self._split_heads(self.key_projection(query_input_b))
        value_a = self._split_heads(self.value_projection(normalized_a))
        value_b = self._split_heads(self.value_projection(normalized_b))

        context_ab, attention_ab = self._attend(
            query_a, key_b, value_b, guidance_b, token_mask_b
        )
        context_ba, attention_ba = self._attend(
            query_b, key_a, value_a, guidance_a, token_mask_a
        )
        cross_a = features_a + self.dropout(self.output_projection(context_ab))
        cross_b = features_b + self.dropout(self.output_projection(context_ba))
        cross_a = self.output_norm(cross_a + self.feed_forward(self.output_norm(cross_a)))
        cross_b = self.output_norm(cross_b + self.feed_forward(self.output_norm(cross_b)))

        affinity_ab = attention_ab.mean(dim=1)
        affinity_ba = attention_ba.mean(dim=1)
        effective_opening_a = guidance_a
        effective_opening_b = guidance_b
        if token_mask_a is not None:
            effective_opening_a = effective_opening_a * token_mask_a.to(opening_a.dtype)
        if token_mask_b is not None:
            effective_opening_b = effective_opening_b * token_mask_b.to(opening_b.dtype)
        shared_a = effective_opening_a * torch.einsum(
            "bnm,bm->bn", affinity_ab, effective_opening_b
        )
        shared_b = effective_opening_b * torch.einsum(
            "bnm,bm->bn", affinity_ba, effective_opening_a
        )

        result = {
            "opening_logits_A": signal_a["opening_logits"],
            "opening_logits_B": signal_b["opening_logits"],
            "P_A_open": opening_a,
            "P_B_open": opening_b,
            # Guidance is matcher-only. P_A/P_B and opening_logits always
            # remain the Opening Head predictions and can still be supervised.
            "opening_guidance_A": guidance_a,
            "opening_guidance_B": guidance_b,
            "G_A_open": signal_a["expansion_depth"],
            "G_B_open": signal_b["expansion_depth"],
            "Delta_A": signal_a["delta"],
            "Delta_B": signal_b["delta"],
            "S_A": shared_a,
            "S_B": shared_b,
            "Aff_AB": affinity_ab,
            "Aff_BA": affinity_ba,
            "cross_feature_A": cross_a,
            "cross_feature_B": cross_b,
        }
        result.update(self._cyclic_shift_scores(
            affinity_ab,
            effective_opening_a,
            effective_opening_b,
        ))
        if candidate_masks_a is not None and candidate_masks_b is not None:
            result.update(self._candidate_matches(
                cross_a,
                cross_b,
                guidance_a,
                guidance_b,
                candidate_masks_a,
                candidate_masks_b,
                candidate_valid_a,
                candidate_valid_b,
            ))
        return result


class DualPanoramaCrossAttentionModel(nn.Module):
    """Run a shared Bi-Layout model twice and match both panoramas."""

    def __init__(
        self,
        bi_layout: nn.Module,
        matcher: Optional[OpeningGuidedCrossAttentionMatcher] = None,
        use_position_encoding: bool = False,
        depth_branch_order: str = "extended_first",
    ):
        super().__init__()
        self.bi_layout = bi_layout
        self.matcher = matcher or OpeningGuidedCrossAttentionMatcher(
            feature_dim=bi_layout.patch_dim
        )
        self.use_position_encoding = bool(use_position_encoding)
        if depth_branch_order not in ("extended_first", "enclosed_first"):
            raise ValueError(
                "depth_branch_order must be 'extended_first' or 'enclosed_first'"
            )
        self.depth_branch_order = depth_branch_order

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        candidate_masks_a: Optional[torch.Tensor] = None,
        candidate_masks_b: Optional[torch.Tensor] = None,
        candidate_valid_a: Optional[torch.Tensor] = None,
        candidate_valid_b: Optional[torch.Tensor] = None,
        opening_guidance_a: Optional[torch.Tensor] = None,
        opening_guidance_b: Optional[torch.Tensor] = None,
        opening_guidance_mode: str = "predicted",
        opening_guidance_mix_weight: float = 0.5,
    ) -> Dict[str, object]:
        output_a = self.bi_layout(image_a, return_features=True)
        output_b = self.bi_layout(image_b, return_features=True)
        if "new_depth" not in output_a or "new_depth" not in output_b:
            raise ValueError("Cross-scene matching requires Bi_Layout output_number=2")

        enclosed_a, extended_a = resolve_enclosed_extended_depth(
            output_a, self.depth_branch_order
        )
        enclosed_b, extended_b = resolve_enclosed_extended_depth(
            output_b, self.depth_branch_order
        )

        matches = self.matcher(
            output_a["layout_feature"],
            output_b["layout_feature"],
            enclosed_a,
            extended_a,
            enclosed_b,
            extended_b,
            position_a=output_a["feature_pos"] if self.use_position_encoding else None,
            position_b=output_b["feature_pos"] if self.use_position_encoding else None,
            candidate_masks_a=candidate_masks_a,
            candidate_masks_b=candidate_masks_b,
            candidate_valid_a=candidate_valid_a,
            candidate_valid_b=candidate_valid_b,
            opening_guidance_a=opening_guidance_a,
            opening_guidance_b=opening_guidance_b,
            opening_guidance_mode=opening_guidance_mode,
            opening_guidance_mix_weight=opening_guidance_mix_weight,
        )
        return {"layout_A": output_a, "layout_B": output_b, "matches": matches}


def bidirectional_candidate_consistency_loss(
    outputs: Mapping[str, torch.Tensor],
    candidate_valid_a: Optional[torch.Tensor] = None,
    candidate_valid_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize disagreement between A->B and B->A real-candidate scores."""
    assignment_ab = outputs["candidate_assignment_AB"]
    assignment_ba = outputs["candidate_assignment_BA"]
    if assignment_ab.ndim != 3 or assignment_ba.ndim != 3:
        raise ValueError("candidate assignments must have shape [B, K+1, L+1]")
    batch, count_a_plus_bin, count_b_plus_bin = assignment_ab.shape
    if assignment_ba.shape != (batch, count_b_plus_bin, count_a_plus_bin):
        raise ValueError("candidate_assignment_BA must be the reverse-shaped assignment")
    count_a = count_a_plus_bin - 1
    count_b = count_b_plus_bin - 1

    if candidate_valid_a is None:
        candidate_valid_a = outputs.get("candidate_valid_A")
    if candidate_valid_b is None:
        candidate_valid_b = outputs.get("candidate_valid_B")
    if candidate_valid_a is None:
        candidate_valid_a = torch.ones(
            (batch, count_a), dtype=torch.bool, device=assignment_ab.device
        )
    if candidate_valid_b is None:
        candidate_valid_b = torch.ones(
            (batch, count_b), dtype=torch.bool, device=assignment_ab.device
        )
    candidate_valid_a = candidate_valid_a.to(
        device=assignment_ab.device, dtype=torch.bool
    )
    candidate_valid_b = candidate_valid_b.to(
        device=assignment_ab.device, dtype=torch.bool
    )
    if candidate_valid_a.shape != (batch, count_a):
        raise ValueError("candidate_valid_a must have shape [B, K_A]")
    if candidate_valid_b.shape != (batch, count_b):
        raise ValueError("candidate_valid_b must have shape [B, K_B]")

    real_ab = assignment_ab[:, :count_a, :count_b]
    real_ba = assignment_ba[:, :count_b, :count_a].transpose(-2, -1)
    pair_valid = candidate_valid_a.unsqueeze(-1) & candidate_valid_b.unsqueeze(1)
    squared_error = (real_ab - real_ba).square()
    if pair_valid.any():
        return squared_error[pair_valid].mean()
    return squared_error.new_zeros(())


def candidate_assignment_loss(
    outputs: Mapping[str, torch.Tensor],
    candidate_target_ab: Optional[torch.Tensor] = None,
    target_candidate_pair: Optional[torch.Tensor] = None,
    is_match: Optional[torch.Tensor] = None,
    candidate_valid_a: Optional[torch.Tensor] = None,
    candidate_valid_b: Optional[torch.Tensor] = None,
    consistency_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Supervise candidate correspondence including explicit no-match bins.

    ``candidate_target_ab`` may contain one or several positive cells per
    source candidate. Candidates without a positive cell are supervised to
    their dustbin. For a negative image pair (``is_match=False``), all valid
    candidates on both sides are supervised to no-match.
    """
    if consistency_weight < 0:
        raise ValueError("consistency_weight must be non-negative")
    assignment_ab = outputs["candidate_assignment_AB"]
    assignment_ba = outputs["candidate_assignment_BA"]
    batch, count_a_plus_bin, count_b_plus_bin = assignment_ab.shape
    count_a = count_a_plus_bin - 1
    count_b = count_b_plus_bin - 1
    if assignment_ba.shape != (batch, count_b_plus_bin, count_a_plus_bin):
        raise ValueError("candidate_assignment_BA must be the reverse-shaped assignment")

    if candidate_valid_a is None:
        candidate_valid_a = outputs.get("candidate_valid_A")
    if candidate_valid_b is None:
        candidate_valid_b = outputs.get("candidate_valid_B")
    if candidate_valid_a is None:
        candidate_valid_a = torch.ones(
            (batch, count_a), dtype=torch.bool, device=assignment_ab.device
        )
    if candidate_valid_b is None:
        candidate_valid_b = torch.ones(
            (batch, count_b), dtype=torch.bool, device=assignment_ab.device
        )
    candidate_valid_a = candidate_valid_a.to(
        device=assignment_ab.device, dtype=torch.bool
    )
    candidate_valid_b = candidate_valid_b.to(
        device=assignment_ab.device, dtype=torch.bool
    )
    if candidate_valid_a.shape != (batch, count_a):
        raise ValueError("candidate_valid_a must have shape [B, K_A]")
    if candidate_valid_b.shape != (batch, count_b):
        raise ValueError("candidate_valid_b must have shape [B, K_B]")

    if candidate_target_ab is not None and target_candidate_pair is not None:
        raise ValueError(
            "provide candidate_target_ab or target_candidate_pair, not both"
        )
    if candidate_target_ab is None:
        candidate_target_ab = assignment_ab.new_zeros((batch, count_a, count_b))
    else:
        candidate_target_ab = candidate_target_ab.to(assignment_ab)
        if candidate_target_ab.shape != (batch, count_a, count_b):
            raise ValueError("candidate_target_ab must have shape [B, K_A, K_B]")
        if (candidate_target_ab < 0).any():
            raise ValueError("candidate_target_ab must be non-negative")

    if target_candidate_pair is not None:
        target_candidate_pair = torch.as_tensor(
            target_candidate_pair, device=assignment_ab.device, dtype=torch.long
        )
        if target_candidate_pair.ndim == 1:
            target_candidate_pair = target_candidate_pair.unsqueeze(0)
        if target_candidate_pair.shape[0] == 1 and batch > 1:
            target_candidate_pair = target_candidate_pair.expand(batch, -1)
        if target_candidate_pair.shape != (batch, 2):
            raise ValueError("target_candidate_pair must have shape [B, 2]")
        target_from_pair = assignment_ab.new_zeros((batch, count_a, count_b))
        for batch_index, pair in enumerate(target_candidate_pair.tolist()):
            index_a, index_b = pair
            if 0 <= index_a < count_a and 0 <= index_b < count_b:
                target_from_pair[batch_index, index_a, index_b] = 1.0
        candidate_target_ab = target_from_pair

    if is_match is None:
        is_match = candidate_target_ab.flatten(1).sum(dim=-1) > 0
    else:
        is_match = torch.as_tensor(is_match, device=assignment_ab.device)
        if is_match.ndim == 0:
            is_match = is_match.unsqueeze(0)
        is_match = is_match.reshape(-1).to(dtype=torch.bool)
        if is_match.shape[0] == 1 and batch > 1:
            is_match = is_match.expand(batch)
        if is_match.shape != (batch,):
            raise ValueError("is_match must have shape [B]")

    pair_valid = candidate_valid_a.unsqueeze(-1) & candidate_valid_b.unsqueeze(1)
    candidate_target_ab = candidate_target_ab * pair_valid.to(
        candidate_target_ab.dtype
    )
    candidate_target_ab = candidate_target_ab * is_match[:, None, None].to(
        candidate_target_ab.dtype
    )
    candidate_target_ba = candidate_target_ab.transpose(-2, -1)

    def directional_loss(
        probability: torch.Tensor,
        real_target: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        real_count = real_target.shape[-1]
        row_mass = real_target.sum(dim=-1, keepdim=True)
        normalized_real = real_target / row_mass.clamp_min(1e-6)
        dustbin_target = (row_mass <= 0).to(real_target.dtype)
        target = torch.cat((normalized_real, dustbin_target), dim=-1)
        row_loss = -(target * probability[..., :real_count + 1].clamp_min(1e-8).log()).sum(-1)
        if source_valid.any():
            return row_loss[source_valid].mean()
        return row_loss.new_zeros(())

    loss_ab = directional_loss(
        assignment_ab[:, :count_a, :],
        candidate_target_ab,
        candidate_valid_a,
    )
    loss_ba = directional_loss(
        assignment_ba[:, :count_b, :],
        candidate_target_ba,
        candidate_valid_b,
    )
    assignment_loss = 0.5 * (loss_ab + loss_ba)
    consistency_loss = bidirectional_candidate_consistency_loss(
        outputs,
        candidate_valid_a=candidate_valid_a,
        candidate_valid_b=candidate_valid_b,
    )
    total = assignment_loss + float(consistency_weight) * consistency_loss
    return {
        "loss_candidate_total": total,
        "loss_candidate_assignment": assignment_loss,
        "loss_candidate_assignment_ab": loss_ab,
        "loss_candidate_assignment_ba": loss_ba,
        "loss_bidirectional_consistency": consistency_loss,
    }


def opening_matching_loss(
    outputs: Mapping[str, torch.Tensor],
    opening_target_a: torch.Tensor,
    opening_target_b: torch.Tensor,
    affinity_target_ab: Optional[torch.Tensor] = None,
    expansion_target_a: Optional[torch.Tensor] = None,
    expansion_target_b: Optional[torch.Tensor] = None,
    expansion_weight: float = 1.0,
    affinity_weight: float = 1.0,
    candidate_target_ab: Optional[torch.Tensor] = None,
    target_candidate_pair: Optional[torch.Tensor] = None,
    is_match: Optional[torch.Tensor] = None,
    candidate_valid_a: Optional[torch.Tensor] = None,
    candidate_valid_b: Optional[torch.Tensor] = None,
    candidate_assignment_weight: float = 1.0,
    consistency_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute supervised losses for opening response, depth, and matching."""
    opening_target_a = opening_target_a.to(outputs["P_A_open"])
    opening_target_b = opening_target_b.to(outputs["P_B_open"])
    if "opening_logits_A" in outputs and "opening_logits_B" in outputs:
        opening_loss = F.binary_cross_entropy_with_logits(
            outputs["opening_logits_A"], opening_target_a
        )
        opening_loss = opening_loss + F.binary_cross_entropy_with_logits(
            outputs["opening_logits_B"], opening_target_b
        )
    else:
        opening_loss = F.binary_cross_entropy(outputs["P_A_open"], opening_target_a)
        opening_loss = opening_loss + F.binary_cross_entropy(
            outputs["P_B_open"], opening_target_b
        )
    total = opening_loss
    zero = opening_loss.new_zeros(())
    expansion_loss = zero
    affinity_loss = zero
    candidate_loss = zero
    candidate_assignment_component = zero
    consistency_loss = zero

    if expansion_target_a is not None and expansion_target_b is not None:
        expansion_target_a = expansion_target_a.to(outputs["G_A_open"])
        expansion_target_b = expansion_target_b.to(outputs["G_B_open"])
        expansion_loss = F.smooth_l1_loss(outputs["G_A_open"], expansion_target_a)
        expansion_loss = expansion_loss + F.smooth_l1_loss(
            outputs["G_B_open"], expansion_target_b
        )
        total = total + float(expansion_weight) * expansion_loss

    if affinity_target_ab is not None:
        affinity_target_ab = affinity_target_ab.to(outputs["Aff_AB"])
        affinity_target_ba = affinity_target_ab.transpose(-2, -1)

        def dense_match_loss(prediction, target):
            row_mass = target.sum(dim=-1, keepdim=True)
            valid_rows = row_mass.squeeze(-1) > 0
            normalized_target = target / row_mass.clamp_min(1e-6)
            row_loss = -(normalized_target * prediction.clamp_min(1e-8).log()).sum(-1)
            if valid_rows.any():
                return row_loss[valid_rows].mean()
            return row_loss.new_zeros(())

        affinity_loss_ab = dense_match_loss(outputs["Aff_AB"], affinity_target_ab)
        affinity_loss_ba = dense_match_loss(outputs["Aff_BA"], affinity_target_ba)
        affinity_loss = affinity_loss_ab + affinity_loss_ba
        total = total + float(affinity_weight) * affinity_loss

    has_candidate_assignment = (
        "candidate_assignment_AB" in outputs
        and "candidate_assignment_BA" in outputs
    )
    candidate_supervision = (
        candidate_target_ab is not None
        or target_candidate_pair is not None
        or is_match is not None
        or (affinity_target_ab is not None and "candidate_masks_A" in outputs)
    )
    if has_candidate_assignment and candidate_supervision:
        if (
            candidate_target_ab is None
            and target_candidate_pair is None
            and affinity_target_ab is not None
            and "candidate_masks_A" in outputs
            and "candidate_masks_B" in outputs
        ):
            masks_a = outputs["candidate_masks_A"].to(
                device=affinity_target_ab.device,
                dtype=affinity_target_ab.dtype,
            )
            masks_b = outputs["candidate_masks_B"].to(
                device=affinity_target_ab.device,
                dtype=affinity_target_ab.dtype,
            )
            candidate_target_ab = torch.einsum(
                "bkn,bnm,blm->bkl",
                masks_a,
                affinity_target_ab,
                masks_b,
            )
        candidate_losses = candidate_assignment_loss(
            outputs,
            candidate_target_ab=candidate_target_ab,
            target_candidate_pair=target_candidate_pair,
            is_match=is_match,
            candidate_valid_a=candidate_valid_a,
            candidate_valid_b=candidate_valid_b,
            consistency_weight=consistency_weight,
        )
        candidate_loss = candidate_losses["loss_candidate_total"]
        candidate_assignment_component = candidate_losses[
            "loss_candidate_assignment"
        ]
        consistency_loss = candidate_losses["loss_bidirectional_consistency"]
        total = total + float(candidate_assignment_weight) * candidate_loss

    return {
        "loss_total": total,
        "loss_opening": opening_loss,
        "loss_expansion": expansion_loss,
        "loss_affinity": affinity_loss,
        "loss_candidate": candidate_loss,
        "loss_candidate_assignment": candidate_assignment_component,
        "loss_bidirectional_consistency": consistency_loss,
    }


def cyclic_token_shift_loss(
    cyclic_shift_score: torch.Tensor,
    target_shift_radians: torch.Tensor,
) -> torch.Tensor:
    """Supervise the circular B-token minus A-token shift in radians."""
    if cyclic_shift_score.ndim != 2:
        raise ValueError("cyclic_shift_score must have shape [B, N]")
    token_count = cyclic_shift_score.shape[-1]
    target_shift_radians = target_shift_radians.to(cyclic_shift_score).reshape(-1)
    if target_shift_radians.shape[0] != cyclic_shift_score.shape[0]:
        raise ValueError("target_shift_radians batch size does not match predictions")
    target_shift = torch.round(
        target_shift_radians * token_count / (2.0 * math.pi)
    ).to(torch.long).remainder(token_count)
    return F.nll_loss(cyclic_shift_score.clamp_min(1e-8).log(), target_shift)


def cyclic_yaw_loss(
    cyclic_shift_score: torch.Tensor,
    target_yaw_radians: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for :func:`cyclic_token_shift_loss`.

    The historical name is retained for callers, but the target must be the
    shared portal's B-token minus A-token shift, not camera relative yaw.
    """
    return cyclic_token_shift_loss(cyclic_shift_score, target_yaw_radians)


def relative_pose_loss(
    predicted_transform: torch.Tensor,
    target_transform: torch.Tensor,
    yaw_weight: float = 1.0,
    translation_weight: float = 1.0,
    scale_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Supervise differentiable 2D similarity transforms shaped [B, 3, 3]."""
    if predicted_transform.shape != target_transform.shape:
        raise ValueError("predicted and target transforms must have the same shape")
    if predicted_transform.ndim != 3 or predicted_transform.shape[-2:] != (3, 3):
        raise ValueError("transforms must have shape [B, 3, 3]")
    target_transform = target_transform.to(predicted_transform)

    predicted_yaw = torch.atan2(predicted_transform[:, 1, 0], predicted_transform[:, 0, 0])
    target_yaw = torch.atan2(target_transform[:, 1, 0], target_transform[:, 0, 0])
    yaw_loss = (1.0 - torch.cos(predicted_yaw - target_yaw)).mean()
    translation_loss = F.smooth_l1_loss(
        predicted_transform[:, :2, 2], target_transform[:, :2, 2]
    )
    predicted_scale = torch.sqrt(
        predicted_transform[:, 0, 0].square()
        + predicted_transform[:, 1, 0].square()
    ).clamp_min(1e-6)
    target_scale = torch.sqrt(
        target_transform[:, 0, 0].square()
        + target_transform[:, 1, 0].square()
    ).clamp_min(1e-6)
    scale_loss = F.smooth_l1_loss(predicted_scale.log(), target_scale.log())
    total = (
        float(yaw_weight) * yaw_loss
        + float(translation_weight) * translation_loss
        + float(scale_weight) * scale_loss
    )
    return {
        "loss_pose": total,
        "loss_yaw": yaw_loss,
        "loss_translation": translation_loss,
        "loss_scale": scale_loss,
    }

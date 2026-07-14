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

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _token_mask(candidate_masks: Optional[torch.Tensor], reference: torch.Tensor):
        if candidate_masks is None:
            return None
        if candidate_masks.ndim == 2:
            candidate_masks = candidate_masks.unsqueeze(0)
        if candidate_masks.shape[0] == 1 and reference.shape[0] > 1:
            candidate_masks = candidate_masks.expand(reference.shape[0], -1, -1)
        mask = candidate_masks.to(device=reference.device, dtype=torch.bool).any(dim=1)
        has_opening = mask.any(dim=-1, keepdim=True)
        return torch.where(has_opening, mask, torch.ones_like(mask))

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

        pooled_a, weights_a, valid_a = self.pooler(features_a, opening_a, candidate_masks_a)
        pooled_b, weights_b, valid_b = self.pooler(features_b, opening_b, candidate_masks_b)
        batch, count_a, _ = pooled_a.shape
        count_b = pooled_b.shape[1]

        if count_a == 0 or count_b == 0:
            shape = (batch, count_a, count_b)
            empty = pooled_a.new_zeros(shape)
            best = torch.full((batch, 2), -1, dtype=torch.long, device=pooled_a.device)
            return {
                "E_A_open": pooled_a,
                "E_B_open": pooled_b,
                "candidate_logits": empty,
                "candidate_affinity": empty,
                "candidate_pair_score": empty,
                "best_candidate_pair": best,
            }

        candidate_a = F.normalize(self.query_projection(pooled_a), dim=-1)
        candidate_b = F.normalize(self.key_projection(pooled_b), dim=-1)
        logits = torch.einsum("bkc,blc->bkl", candidate_a, candidate_b)
        logits = logits / self.candidate_temperature

        pair_valid = valid_a.unsqueeze(-1) & valid_b.unsqueeze(1)
        safe_logits = logits.masked_fill(~valid_b.unsqueeze(1), torch.finfo(logits.dtype).min)
        affinity = torch.softmax(safe_logits, dim=-1)
        affinity = affinity * pair_valid.to(affinity.dtype)
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        mask_a = candidate_masks_a.to(features_a.dtype)
        mask_b = candidate_masks_b.to(features_b.dtype)
        confidence_a = (mask_a * opening_a.unsqueeze(1)).sum(-1)
        confidence_a = confidence_a / mask_a.sum(-1).clamp_min(1.0)
        confidence_b = (mask_b * opening_b.unsqueeze(1)).sum(-1)
        confidence_b = confidence_b / mask_b.sum(-1).clamp_min(1.0)
        pair_confidence = torch.sqrt(confidence_a.unsqueeze(-1) * confidence_b.unsqueeze(1))
        pair_score = torch.sigmoid(logits) * pair_confidence * pair_valid.to(logits.dtype)

        flat_best = pair_score.flatten(1).argmax(dim=-1)
        best_a = torch.div(flat_best, count_b, rounding_mode="floor")
        best = torch.stack((best_a, flat_best.remainder(count_b)), dim=-1)
        has_pair = pair_valid.flatten(1).any(dim=-1)
        best = torch.where(has_pair.unsqueeze(-1), best, torch.full_like(best, -1))

        return {
            "E_A_open": pooled_a,
            "E_B_open": pooled_b,
            "candidate_weights_A": weights_a,
            "candidate_weights_B": weights_b,
            "candidate_logits": logits,
            "candidate_affinity": affinity,
            "candidate_pair_score": pair_score,
            "best_candidate_pair": best,
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
    ) -> Dict[str, torch.Tensor]:
        if features_a.shape != features_b.shape:
            raise ValueError("features_a and features_b must have the same shape")

        signal_a = self.opening_head(features_a, enclosed_depth_a, extended_depth_a)
        signal_b = self.opening_head(features_b, enclosed_depth_b, extended_depth_b)
        opening_a = signal_a["opening_probability"]
        opening_b = signal_b["opening_probability"]
        token_mask_a = self._token_mask(candidate_masks_a, features_a)
        token_mask_b = self._token_mask(candidate_masks_b, features_b)

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
            query_a, key_b, value_b, opening_b, token_mask_b
        )
        context_ba, attention_ba = self._attend(
            query_b, key_a, value_a, opening_a, token_mask_a
        )
        cross_a = features_a + self.dropout(self.output_projection(context_ab))
        cross_b = features_b + self.dropout(self.output_projection(context_ba))
        cross_a = self.output_norm(cross_a + self.feed_forward(self.output_norm(cross_a)))
        cross_b = self.output_norm(cross_b + self.feed_forward(self.output_norm(cross_b)))

        affinity_ab = attention_ab.mean(dim=1)
        affinity_ba = attention_ba.mean(dim=1)
        effective_opening_a = opening_a
        effective_opening_b = opening_b
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
            "P_A_open": opening_a,
            "P_B_open": opening_b,
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
                opening_a,
                opening_b,
                candidate_masks_a,
                candidate_masks_b,
            ))
        return result


class DualPanoramaCrossAttentionModel(nn.Module):
    """Run a shared Bi-Layout model twice and match both panoramas."""

    def __init__(
        self,
        bi_layout: nn.Module,
        matcher: Optional[OpeningGuidedCrossAttentionMatcher] = None,
        use_position_encoding: bool = False,
    ):
        super().__init__()
        self.bi_layout = bi_layout
        self.matcher = matcher or OpeningGuidedCrossAttentionMatcher(
            feature_dim=bi_layout.patch_dim
        )
        self.use_position_encoding = bool(use_position_encoding)

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        candidate_masks_a: Optional[torch.Tensor] = None,
        candidate_masks_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        output_a = self.bi_layout(image_a, return_features=True)
        output_b = self.bi_layout(image_b, return_features=True)
        if "new_depth" not in output_a or "new_depth" not in output_b:
            raise ValueError("Cross-scene matching requires Bi_Layout output_number=2")

        matches = self.matcher(
            output_a["layout_feature"],
            output_b["layout_feature"],
            output_a["depth"],
            output_a["new_depth"],
            output_b["depth"],
            output_b["new_depth"],
            position_a=output_a["feature_pos"] if self.use_position_encoding else None,
            position_b=output_b["feature_pos"] if self.use_position_encoding else None,
            candidate_masks_a=candidate_masks_a,
            candidate_masks_b=candidate_masks_b,
        )
        return {"layout_A": output_a, "layout_B": output_b, "matches": matches}


def opening_matching_loss(
    outputs: Mapping[str, torch.Tensor],
    opening_target_a: torch.Tensor,
    opening_target_b: torch.Tensor,
    affinity_target_ab: Optional[torch.Tensor] = None,
    expansion_target_a: Optional[torch.Tensor] = None,
    expansion_target_b: Optional[torch.Tensor] = None,
    expansion_weight: float = 1.0,
    affinity_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute supervised losses for opening response, depth, and matching."""
    opening_target_a = opening_target_a.to(outputs["P_A_open"])
    opening_target_b = opening_target_b.to(outputs["P_B_open"])
    opening_loss = F.binary_cross_entropy(outputs["P_A_open"], opening_target_a)
    opening_loss = opening_loss + F.binary_cross_entropy(outputs["P_B_open"], opening_target_b)
    total = opening_loss
    zero = opening_loss.new_zeros(())
    expansion_loss = zero
    affinity_loss = zero

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

    return {
        "loss_total": total,
        "loss_opening": opening_loss,
        "loss_expansion": expansion_loss,
        "loss_affinity": affinity_loss,
    }


def cyclic_yaw_loss(
    cyclic_shift_score: torch.Tensor,
    target_yaw_radians: torch.Tensor,
) -> torch.Tensor:
    """Supervise circular panorama shift using relative yaw in radians."""
    if cyclic_shift_score.ndim != 2:
        raise ValueError("cyclic_shift_score must have shape [B, N]")
    token_count = cyclic_shift_score.shape[-1]
    target_yaw_radians = target_yaw_radians.to(cyclic_shift_score).reshape(-1)
    if target_yaw_radians.shape[0] != cyclic_shift_score.shape[0]:
        raise ValueError("target_yaw_radians batch size does not match predictions")
    target_shift = torch.round(
        target_yaw_radians * token_count / (2.0 * math.pi)
    ).to(torch.long).remainder(token_count)
    return F.nll_loss(cyclic_shift_score.clamp_min(1e-8).log(), target_shift)


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

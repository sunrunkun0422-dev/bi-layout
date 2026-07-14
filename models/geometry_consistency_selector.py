"""Learned selector for ranked cross-scene geometry candidates."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


GEOMETRY_METRIC_NAMES = (
    "openingWidthPenalty",
    "lengthRatioPenalty",
    "scalePenalty",
    "areaRatioPenalty",
    "overlapRatio",
    "sameSidePenalty",
    "openingConfidenceMean",
    "featureScore",
    "passabilityReward",
    "featureReward",
)


def candidate_metrics_to_tensor(
    candidate_metrics: Iterable[Mapping[str, float]],
    metric_names: Sequence[str] = GEOMETRY_METRIC_NAMES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    rows = [
        [float(metrics.get(name, 0.0)) for name in metric_names]
        for metrics in candidate_metrics
    ]
    if not rows:
        return torch.empty((0, len(metric_names)), dtype=torch.float32, device=device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


class GeometryConsistencySelector(nn.Module):
    """Predict a calibrated distribution over geometry candidates."""

    def __init__(self, input_dim: int = len(GEOMETRY_METRIC_NAMES), hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, metrics: torch.Tensor,
                candidate_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        squeeze_batch = metrics.ndim == 2
        if squeeze_batch:
            metrics = metrics.unsqueeze(0)
        if metrics.ndim != 3 or metrics.shape[-1] != self.input_dim:
            raise ValueError(f"metrics must have shape [B, K, {self.input_dim}] or [K, {self.input_dim}]")

        batch, count, _ = metrics.shape
        if count == 0:
            empty = metrics.new_empty((batch, 0))
            best = torch.full((batch,), -1, dtype=torch.long, device=metrics.device)
            result = {"selector_logits": empty, "selector_probability": empty, "best_index": best}
            return {key: value.squeeze(0) for key, value in result.items()} if squeeze_batch else result

        logits = self.network(metrics).squeeze(-1)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            if candidate_mask.ndim == 1:
                candidate_mask = candidate_mask.unsqueeze(0)
            candidate_mask = candidate_mask.to(device=metrics.device, dtype=torch.bool)
            if candidate_mask.shape != logits.shape:
                raise ValueError("candidate_mask must have shape [B, K] or [K]")
        safe_mask = torch.where(
            candidate_mask.any(dim=-1, keepdim=True),
            candidate_mask,
            torch.ones_like(candidate_mask),
        )
        logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        probability = torch.softmax(logits, dim=-1) * candidate_mask.to(logits.dtype)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        best = probability.argmax(dim=-1)
        best = torch.where(candidate_mask.any(dim=-1), best, torch.full_like(best, -1))
        result = {
            "selector_logits": logits,
            "selector_probability": probability,
            "best_index": best,
        }
        return {key: value.squeeze(0) for key, value in result.items()} if squeeze_batch else result


def geometry_selector_loss(
    selector_logits: torch.Tensor,
    target_index: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if selector_logits.ndim == 1:
        selector_logits = selector_logits.unsqueeze(0)
    target_index = target_index.to(device=selector_logits.device, dtype=torch.long).reshape(-1)
    if candidate_mask is not None:
        if candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0)
        selector_logits = selector_logits.masked_fill(
            ~candidate_mask.to(device=selector_logits.device, dtype=torch.bool),
            torch.finfo(selector_logits.dtype).min,
        )
    return F.cross_entropy(selector_logits, target_index)

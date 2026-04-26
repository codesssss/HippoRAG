"""Autoregressive evidence-path selector for D-PathRAG.

The selector operates over a fixed candidate pool.  DAEC-derived signals enter
as input features and warm-start labels only; the main method deliberately does
not add a hand-weighted ``DAEC_score + neural_score`` residual prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.dpathrag.gumbel import masked_softmax, st_gumbel_top1


@dataclass
class PathSelectorOutput:
    """Outputs from an autoregressive no-replacement selector pass."""

    path_mask: Tensor
    selected_indices: Tensor
    step_logits: Tensor
    step_probs: Tensor
    states: Tensor


class AutoregressivePathSelector(nn.Module):
    """Per-step conditioned listwise selector.

    Inputs are dense per-candidate feature tensors.  A later cache/encoder stage
    can replace these with q-doc encoder representations while preserving this
    selection API.
    """

    def __init__(
        self,
        *,
        candidate_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.list_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.initial_state = nn.Parameter(torch.zeros(hidden_dim))
        self.state_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: Tensor,
        *,
        path_len: int,
        query_features: Tensor | None = None,
        tau: float = 1.0,
        candidate_mask: Tensor | None = None,
        hard: bool = True,
        add_gumbel_noise: bool | None = None,
    ) -> PathSelectorOutput:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [batch, candidates, feature_dim]")
        if path_len <= 0:
            raise ValueError("path_len must be positive")
        batch_size, candidate_count, _ = candidate_features.shape
        add_noise = self.training if add_gumbel_noise is None else bool(add_gumbel_noise)
        encoded = self.feature_projection(candidate_features)
        query_token = self._project_query_token(query_features, batch_size=batch_size) if query_features is not None else None
        valid_mask = torch.ones(batch_size, candidate_count, dtype=torch.bool, device=candidate_features.device)
        if candidate_mask is not None:
            valid_mask = candidate_mask.bool()
        available = valid_mask.clone()
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        if query_token is not None:
            state = state + query_token
        path_mask = torch.zeros(batch_size, candidate_count, dtype=candidate_features.dtype, device=candidate_features.device)
        selected_indices: list[Tensor] = []
        step_logits: list[Tensor] = []
        step_probs: list[Tensor] = []
        states: list[Tensor] = []

        for _ in range(int(path_len)):
            conditioned = self._condition_candidates(encoded, state, query_token=query_token)
            scorer_state = state.unsqueeze(1).expand(-1, candidate_count, -1)
            logits = self.scorer(torch.cat([conditioned, scorer_state], dim=-1)).squeeze(-1)
            probs = masked_softmax(logits, available, dim=-1)
            sample = st_gumbel_top1(logits, tau=tau, mask=available, hard=hard, add_noise=add_noise)
            chosen = sample.argmax(dim=-1)
            selected_rep = torch.bmm(sample.unsqueeze(1), conditioned).squeeze(1)
            state = self.state_gru(selected_rep, state)
            path_mask = torch.clamp(path_mask + sample, max=1.0)
            available = available.scatter(1, chosen.unsqueeze(1), False)
            selected_indices.append(chosen)
            step_logits.append(logits)
            step_probs.append(probs)
            states.append(state)

        return PathSelectorOutput(
            path_mask=path_mask,
            selected_indices=torch.stack(selected_indices, dim=1),
            step_logits=torch.stack(step_logits, dim=1),
            step_probs=torch.stack(step_probs, dim=1),
            states=torch.stack(states, dim=1),
        )

    def _project_query_token(self, query_features: Tensor, *, batch_size: int) -> Tensor:
        if query_features.ndim == 2:
            pass
        elif query_features.ndim == 3 and query_features.shape[1] == 1:
            query_features = query_features.squeeze(1)
        else:
            raise ValueError("query_features must have shape [batch, feature_dim] or [batch, 1, feature_dim]")
        if query_features.shape[0] != batch_size:
            raise ValueError("query_features batch size must match candidate_features")
        return self.feature_projection(query_features)

    def _condition_candidates(self, encoded: Tensor, state: Tensor, *, query_token: Tensor | None = None) -> Tensor:
        state_token = state.unsqueeze(1)
        sequence_parts = [state_token]
        if query_token is not None:
            sequence_parts.append(query_token.unsqueeze(1))
        sequence_parts.append(encoded)
        sequence = torch.cat(sequence_parts, dim=1)
        conditioned = self.list_encoder(sequence)
        candidate_start = 2 if query_token is not None else 1
        return conditioned[:, candidate_start:, :]


def teacher_forced_path_nll(
    step_logits: Tensor,
    target_indices: Tensor,
    *,
    candidate_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Cross-entropy warm-start loss with no-replacement masking.

    This trains the selector to imitate a gold/DAEC evidence path during Stage 1.
    Stage 2 should optimize answer NLL only.
    """

    if step_logits.ndim != 3:
        raise ValueError("step_logits must have shape [batch, steps, candidates]")
    if target_indices.ndim != 2:
        raise ValueError("target_indices must have shape [batch, steps]")
    batch_size, steps, candidate_count = step_logits.shape
    if target_indices.shape != (batch_size, steps):
        raise ValueError("target_indices shape must match [batch, steps]")
    available = torch.ones(batch_size, candidate_count, dtype=torch.bool, device=step_logits.device)
    if candidate_mask is not None:
        available = candidate_mask.bool().clone()
    losses: list[Tensor] = []
    for step in range(steps):
        logits = step_logits[:, step, :].masked_fill(~available, torch.finfo(step_logits.dtype).min)
        target = target_indices[:, step]
        valid = target.ne(int(ignore_index))
        if valid.any():
            safe_target = target.masked_fill(~valid, 0)
            step_loss = F.cross_entropy(logits, safe_target, reduction="none")
            losses.append(step_loss[valid])
            valid_rows = valid.nonzero(as_tuple=False).squeeze(1)
            available[valid_rows] = available[valid_rows].scatter(1, target[valid].unsqueeze(1), False)
    if not losses:
        return step_logits.sum() * 0.0
    return torch.cat(losses, dim=0).mean()

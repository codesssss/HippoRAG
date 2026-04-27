"""Straight-through Gumbel utilities for discrete path selection."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_gumbel(shape: torch.Size | tuple[int, ...], *, device: torch.device | None = None, eps: float = 1e-10) -> Tensor:
    uniform = torch.rand(shape, device=device)
    return -torch.log((-torch.log(uniform.clamp_min(eps))).clamp_min(eps))


def masked_softmax(logits: Tensor, mask: Tensor | None = None, dim: int = -1) -> Tensor:
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=dim)


def st_gumbel_top1(logits: Tensor, *, tau: float = 1.0, mask: Tensor | None = None, hard: bool = True, add_noise: bool = True) -> Tensor:
    """Return a straight-through one-hot sample over the last dimension.

    The forward path is one-hot when ``hard`` is true; gradients flow through the
    relaxed softmax.  ``mask`` marks valid positions and is used for
    no-replacement autoregressive selection.
    """

    if tau <= 0:
        raise ValueError("tau must be positive")
    work_logits = logits
    if mask is not None:
        work_logits = work_logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    if add_noise:
        work_logits = work_logits + sample_gumbel(work_logits.shape, device=work_logits.device)
    relaxed = torch.softmax(work_logits / float(tau), dim=-1)
    if not hard:
        return relaxed
    index = relaxed.argmax(dim=-1, keepdim=True)
    hard_sample = torch.zeros_like(relaxed).scatter_(-1, index, 1.0)
    return hard_sample.detach() - relaxed.detach() + relaxed

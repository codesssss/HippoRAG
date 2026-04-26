from __future__ import annotations

import torch

from src.dpathrag.gumbel import st_gumbel_top1
from src.dpathrag.selector import AutoregressivePathSelector, teacher_forced_path_nll


def test_st_gumbel_top1_respects_mask_without_noise() -> None:
    logits = torch.tensor([[0.1, 10.0, 0.2]])
    mask = torch.tensor([[True, False, True]])
    sample = st_gumbel_top1(logits, tau=1.0, mask=mask, hard=True, add_noise=False)
    assert sample.shape == logits.shape
    assert sample.argmax(dim=-1).item() == 2


def test_autoregressive_selector_shapes_and_no_replacement() -> None:
    torch.manual_seed(7)
    selector = AutoregressivePathSelector(candidate_feature_dim=3, hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    selector.eval()
    features = torch.randn(2, 6, 3)
    output = selector(features, path_len=4, tau=1.0, hard=True, add_gumbel_noise=False)
    assert output.path_mask.shape == (2, 6)
    assert output.selected_indices.shape == (2, 4)
    assert output.step_logits.shape == (2, 4, 6)
    for row in output.selected_indices.tolist():
        assert len(row) == len(set(row))


def test_autoregressive_selector_accepts_query_features() -> None:
    torch.manual_seed(11)
    selector = AutoregressivePathSelector(candidate_feature_dim=4, hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    selector.eval()
    features = torch.randn(2, 5, 4)
    query_features = torch.randn(2, 4, requires_grad=True)
    output = selector(features, query_features=query_features, path_len=3, tau=1.0, hard=True, add_gumbel_noise=False)
    assert output.selected_indices.shape == (2, 3)
    output.step_logits.sum().backward()
    assert query_features.grad is not None


def test_teacher_forced_path_nll_backpropagates() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[0, 2, 4], [1, 3, 0]])
    loss = teacher_forced_path_nll(logits, targets)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_teacher_forced_path_nll_ignores_padded_targets() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[0, -100, -100], [1, 3, -100]])
    loss = teacher_forced_path_nll(logits, targets)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None

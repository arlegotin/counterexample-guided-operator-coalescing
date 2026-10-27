"""Catch reversed SVD factors, rank mistakes, and dead zero-delta adapters."""
import importlib
import importlib.util

import pytest
import torch


def bank_api():
    assert importlib.util.find_spec('poc.bank') is not None, 'canonical bank implementation missing'
    return importlib.import_module('poc.bank')


@pytest.mark.parametrize('rank', [0, 1, 3])
def test_svd_factors_reconstruct_known_diagonal_truncation(rank):
    delta = torch.tensor([[9., 0., 0.], [0., 4., 0.], [0., 0., 1.], [0., 0., 0.]], dtype=torch.float64)
    a, b = bank_api().initialize_residual(delta, rank, 7)
    assert a.shape == (rank, 3) and b.shape == (4, rank)
    expected = torch.zeros_like(delta)
    for i in range(rank):
        expected[i, i] = [9., 4., 1.][i]
    torch.testing.assert_close(b @ a, expected, atol=0, rtol=0)
    torch.testing.assert_close(a @ a.T, b.T @ b, atol=0, rtol=0)


def test_zero_delta_is_exact_seeded_and_can_start_learning():
    api = bank_api()
    delta = torch.zeros((4, 3), dtype=torch.float64)
    rng = torch.random.get_rng_state().clone()
    a, b = api.initialize_residual(delta, 2, 7)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(a, api.initialize_residual(delta, 2, 7)[0])
    assert not torch.equal(a, api.initialize_residual(delta, 2, 8)[0])
    assert torch.count_nonzero(a) == a.numel() and torch.count_nonzero(b) == 0
    a.requires_grad_(); b.requires_grad_()
    torch.nn.functional.linear(torch.nn.functional.linear(torch.tensor([[1., 2., 3.]], dtype=torch.float64), a), b).sum().backward()
    assert torch.count_nonzero(a.grad) == 0 and torch.count_nonzero(b.grad) > 0


@pytest.mark.parametrize('rank', [-1, 4, True, 1.5])
def test_rank_must_fit_both_projection_dimensions(rank):
    with pytest.raises(ValueError, match='rank'):
        bank_api().initialize_residual(torch.zeros((4, 3)), rank, 0)


@pytest.mark.parametrize('delta', [torch.ones(3), torch.ones((2, 2), dtype=torch.int64), torch.full((2, 2), float('nan'))])
def test_initializer_rejects_invalid_matrices(delta):
    with pytest.raises(ValueError):
        bank_api().initialize_residual(delta, 1, 0)

"""Catch wrong whitening orientation, centered covariance and data-role leakage."""
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from poc.bank import initialize_residual
from poc.data.manifest import DataManifest, DocumentRecord


def api():
    assert importlib.util.find_spec('poc.initializers') is not None, 'weighted initializer is not implemented'
    return importlib.import_module('poc.initializers')


def population(role='fit'):
    return DataManifest((DocumentRecord('one', 'owned', 'v1', 'fixture', 0),),
                        {'one': role}, {'one': 'component-one'}, 'owned')


def calibrate(x, **kwargs):
    return api().calibrate_projection_inputs(
        (x,), manifest=population(), document_ids=('one',),
        window_sha256s=('a' * 64,), projection='gate_proj', depth=15,
        input_source='teacher', source_checkpoint_sha256='b'*64, tokenizer_sha256='c'*64, **kwargs)


def test_weighted_rank_one_uses_actual_input_energy_and_uncentered_moment():
    # Raw SVD keeps delta[0,0]=3. Input energy makes delta[1,1]=2 worth more.
    # The second coordinate is constant: centering incorrectly discards it.
    delta = torch.diag(torch.tensor([3., 2.], dtype=torch.float64))
    expected = torch.tensor([[0., 0.], [0., 2.]], dtype=torch.float64)
    x = torch.tensor([[1., 10.], [-1., 10.]], dtype=torch.float64)
    result = api().initialize_weighted_residual(delta, 1, 17, calibrate(x))
    torch.testing.assert_close(result.B @ result.A, expected, atol=1e-10, rtol=1e-10)
    raw_a, raw_b = initialize_residual(delta, 1, 17)
    assert ((x @ (delta-result.B@result.A).T).square().sum()
            < (x @ (delta-raw_b@raw_a).T).square().sum())
    assert result.record['method'] == 'weighted_delta_svd_v1'
    assert result.record['valid_input_rows'] == 2
    assert result.record['projection'] == 'gate_proj'


def test_correlated_covariance_attains_weighted_tail_energy():
    g = torch.Generator().manual_seed(23)
    x = torch.randn(100, 5, generator=g, dtype=torch.float64)
    x = x @ torch.tensor([[3., 2., 0., 0., 0.], [0., 1., 1., 0., 0.],
                          [0., 0., .2, .1, 0.], [0., 0., 0., .5, .1],
                          [0., 0., 0., 0., .01]], dtype=torch.float64)
    delta = torch.randn(4, 5, generator=g, dtype=torch.float64)
    result = api().initialize_weighted_residual(delta, 2, 17, calibrate(x))
    ridge = result.record['ridge']
    C = x.T@x/len(x) + ridge*torch.eye(5, dtype=torch.float64)
    error = delta-result.B@result.A
    actual = torch.trace(error@C@error.T)
    singular = torch.linalg.svdvals(delta@torch.linalg.cholesky(C))
    torch.testing.assert_close(actual, singular[2:].square().sum(), atol=1e-10, rtol=1e-10)
    assert result.A.is_contiguous() and result.B.is_contiguous()
    assert not result.A.requires_grad and not result.B.requires_grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_full_rank_reconstructs_and_rank_zero_is_empty(dtype):
    delta = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], dtype=dtype, requires_grad=True)
    cal = calibrate(torch.tensor([[1., 0., 2.], [3., 1., 0.]], dtype=dtype))
    full = api().initialize_weighted_residual(delta, 2, 17, cal)
    torch.testing.assert_close(full.B@full.A, delta.detach(), atol=1e-6, rtol=1e-6)
    zero = api().initialize_weighted_residual(delta, 0, 17, cal)
    assert zero.A.shape == (0, 3) and zero.B.shape == (2, 0)
    assert full.A.dtype == dtype and full.A.device == delta.device


def test_zero_difference_retains_seeded_live_gradient_without_global_rng_change():
    delta = torch.zeros(4, 3, dtype=torch.float64)
    cal = calibrate(torch.ones(5, 3, dtype=torch.float64))
    before = torch.random.get_rng_state().clone()
    result = api().initialize_weighted_residual(delta, 2, 17, cal)
    a, b = initialize_residual(delta, 2, 17)
    torch.testing.assert_close(result.A, a, atol=0, rtol=0)
    torch.testing.assert_close(result.B, b, atol=0, rtol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    result.A.requires_grad_(); result.B.requires_grad_()
    (torch.ones(1, 3, dtype=torch.float64)@result.A.T@result.B.T).sum().backward()
    assert result.B.grad.abs().max()>0


def test_calibration_is_snapshot_and_detects_mutation():
    x = torch.eye(3, dtype=torch.float64)
    cal = calibrate(x)
    original = cal.digest()
    x.fill_(100)
    assert cal.digest() == original
    cal.second_moment[0, 0] += 1
    with pytest.raises(ValueError, match='mutat|digest'):
        api().initialize_weighted_residual(torch.eye(3, dtype=torch.float64), 1, 17, cal)


@pytest.mark.parametrize('role', ['dev', 'probe_score', 'confirmation', 'probe_fit'])
def test_nonfit_calibration_is_denied_before_consuming_inputs(role):
    class Unreadable:
        def __iter__(self):
            raise AssertionError('inputs consumed before role check')
    with pytest.raises(ValueError, match='fit'):
        api().calibrate_projection_inputs(Unreadable(), manifest=population(role),
            document_ids=('one',), window_sha256s=('a'*64,),
            projection='gate_proj', depth=15, input_source='teacher', source_checkpoint_sha256='b'*64, tokenizer_sha256='c'*64)


@pytest.mark.parametrize('change', [
    {'document_ids': ('missing',)}, {'window_sha256s': ('invalid',)},
    {'projection': 'output_adapter'}, {'depth': -1}, {'input_source': 'unknown'},
])
def test_calibration_requires_complete_declared_provenance(change):
    args = dict(manifest=population(), document_ids=('one',), window_sha256s=('a'*64,),
                projection='gate_proj', depth=15, input_source='teacher', source_checkpoint_sha256='b'*64, tokenizer_sha256='c'*64)
    args.update(change)
    with pytest.raises(ValueError):
        api().calibrate_projection_inputs((torch.eye(3),), **args)


@pytest.mark.parametrize('x', [torch.empty(0, 3), torch.ones(3),
                              torch.full((2, 3), float('nan')), torch.ones(2, 3, dtype=torch.int64)])
def test_invalid_activation_inputs_are_rejected(x):
    with pytest.raises(ValueError): calibrate(x)


def test_singular_calibration_gets_finite_ridge_and_bound_condition():
    result = api().initialize_weighted_residual(torch.eye(3, dtype=torch.float64), 1, 17,
                    calibrate(torch.ones(5, 3, dtype=torch.float64)), max_condition=1e4)
    assert result.record['ridge']>0
    assert result.record['condition']<=1e4*(1+1e-10)
    assert torch.isfinite(result.A).all() and torch.isfinite(result.B).all()


@pytest.mark.parametrize('kwargs', [{'rank': -1}, {'rank': True}, {'seed': -1},
                                  {'ridge_relative': 0.}, {'max_condition': 1.},
                                  {'max_condition': float('inf')}])
def test_invalid_initializer_configuration_is_rejected(kwargs):
    args = dict(rank=1, seed=17, calibration=calibrate(torch.eye(3)))
    args.update(kwargs)
    with pytest.raises(ValueError): api().initialize_weighted_residual(torch.eye(3), **args)


def test_wrong_input_width_and_quarantined_documents_fail_closed():
    with pytest.raises(ValueError, match='width'):
        api().initialize_weighted_residual(torch.eye(2), 1, 17, calibrate(torch.eye(3)))
    manifest = replace(population(), quarantine={'one': 'owned exclusion'})
    with pytest.raises(ValueError, match='quarantin'):
        api().calibrate_projection_inputs((torch.eye(3),), manifest=manifest,
            document_ids=('one',), window_sha256s=('a'*64,), projection='gate_proj',
            depth=15, input_source='teacher', source_checkpoint_sha256='b'*64, tokenizer_sha256='c'*64)


@pytest.mark.mps
@pytest.mark.parametrize('mps_inputs',[False,True])
def test_weighted_initializer_keeps_nonzero_mps_inputs_and_delta(mps_inputs):
    assert torch.backends.mps.is_available(),'explicit weighted-initializer regression requires Metal'
    x=torch.tensor([[1.,10.],[-1.,10.]],dtype=torch.float32)
    delta=torch.diag(torch.tensor([3.,2.],dtype=torch.float32))
    cal=calibrate(x.to('mps') if mps_inputs else x)
    torch.testing.assert_close(cal.second_moment,torch.diag(torch.tensor([1.,100.],dtype=torch.float64)),atol=0,rtol=0)
    actual=api().initialize_weighted_residual(delta.to('mps'),1,17,cal)
    expected=api().initialize_weighted_residual(delta,1,17,calibrate(x))
    product=(actual.B@actual.A).cpu()
    assert product.abs().max()>1
    torch.testing.assert_close(product,expected.B@expected.A,atol=1e-6,rtol=1e-6)

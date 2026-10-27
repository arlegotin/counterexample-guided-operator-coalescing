"""Catch detached candidate suffixes, target autograd, and nonlinear fixture gaps."""
from dataclasses import replace
import pytest
import torch
from test_model_parity import api, batch


@pytest.mark.cpu
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_frozen_suffix_transmits_candidate_weight_gradient_and_target_is_no_grad(depth):
    m = api().make_tiny_decoder(api().TinyConfig(), seed=3, dtype=torch.float64)
    bound = m.prefix_boundary(batch(), depth)
    candidate_weight = torch.nn.Parameter(torch.eye(m.config.hidden_size, dtype=torch.float64) * .02)
    with torch.no_grad():
        target = m.suffix_logits(m.ff_residual(bound), bound)
    candidate = m.suffix_logits(m.ff_residual(bound, lambda x: x @ candidate_weight), bound)
    (candidate - target).square().mean().backward()
    assert candidate_weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in m.parameters())
    assert not target.requires_grad and candidate.requires_grad


@pytest.mark.cpu
def test_input_gradient_matches_central_difference_through_final_norm_and_readout():
    m = api().make_tiny_decoder(api().TinyConfig(), seed=6, dtype=torch.float64)
    bound = m.prefix_boundary(batch(), 1)
    z = m.ff_residual(bound).detach().requires_grad_()
    loss = m.suffix_logits(z, bound).square().mean()
    gradient = torch.autograd.grad(loss, z)[0]
    h = 1e-6
    delta = torch.zeros_like(z)
    delta[0,2,3] = h
    finite_difference = (m.suffix_logits(z.detach()+delta, bound).square().mean() -
                         m.suffix_logits(z.detach()-delta, bound).square().mean()) / (2*h)
    torch.testing.assert_close(gradient[0,2,3], finite_difference, atol=1e-10, rtol=1e-6)


@pytest.mark.cpu
def test_nullspace_missing_coordinate_and_nonlinear_composition_controls():
    from poc.synthetic import make_causal_controls
    f = make_causal_controls()
    assert torch.equal(f.readout(f.base), f.readout(f.base + f.null_delta))
    assert not torch.equal(f.suffix(f.base), f.suffix(f.base + f.null_delta))
    assert torch.equal(f.missing_coordinate(f.sampled), f.sampled)
    assert not torch.equal(f.missing_coordinate(f.counterexample), f.counterexample)
    a, b, c = f.nontransitive_states
    assert abs(f.nontransitive_readout(a)-f.nontransitive_readout(b)) <= f.tolerance
    assert abs(f.nontransitive_readout(b)-f.nontransitive_readout(c)) <= f.tolerance
    assert abs(f.nontransitive_readout(a)-f.nontransitive_readout(c)) > f.tolerance
    assert f.joint_suffix(False, False) == f.joint_suffix(True, False) == f.joint_suffix(False, True)
    assert f.joint_suffix(True, True) != f.joint_suffix(False, False)


@pytest.mark.cpu
def test_candidate_prefix_boundary_runs_through_teacher_suffix_without_rewriting_provenance():
    module = api()
    teacher = module.make_tiny_decoder(module.TinyConfig(), seed=11, dtype=torch.float64)
    candidate = module.make_tiny_decoder(module.TinyConfig(), seed=12, dtype=torch.float64)
    candidate.requires_grad_(True)
    boundary = candidate.prefix_boundary(batch(), 1)
    origin = boundary.model_digest
    assert origin != teacher.manifest().checkpoint_digest
    weight = torch.nn.Parameter(torch.eye(candidate.config.hidden_size, dtype=torch.float64)*.03)
    with torch.no_grad():
        target = teacher.suffix_logits(teacher.ff_residual(boundary), boundary)
    # Student-state fitting detaches the prefix, retaining its original digest.
    detached = replace(boundary, u=boundary.u.detach())
    actual = teacher.suffix_logits(teacher.ff_residual(detached, lambda x: x @ weight), detached)
    (actual-target).square().mean().backward()
    assert weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in candidate.parameters())
    assert boundary.model_digest == detached.model_digest == origin

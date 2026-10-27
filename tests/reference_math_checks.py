"""Owned CPU controls from blueprint 11.4; historical decimal output is not evidence.

Run explicitly with pytest (also part of `poc validate --suite component`).
"""
import torch

from poc.losses import kl_sum


def test_missing_coordinate_has_zero_observed_kl_but_positive_intervened_loss_and_gradient():
    x = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]], dtype=torch.float64)
    gain = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    frozen_readout = torch.tensor([[1., 1.], [-1., -1.]], dtype=torch.float64)
    def output(z, g):
        return torch.stack((z[:, 0], g*z[:, 1]), dim=1) @ frozen_readout.T
    valid = torch.ones(3, dtype=torch.bool)
    observed = kl_sum(output(x, 1.), output(x, gain), valid)
    patched = x.clone(); patched[:, 1] = 1.
    intervened = kl_sum(output(patched, 1.), output(patched, gain), valid)
    intervened.backward()
    assert abs(observed.item()) < 1e-12
    assert intervened.item() > .1 and gain.grad.item() < -.1
    assert not frozen_readout.requires_grad


def test_readout_nullspace_can_have_large_state_error_and_zero_output_kl():
    x = torch.tensor([[.25, 0.]], dtype=torch.float64)
    changed = x + torch.tensor([[0., 100.]], dtype=torch.float64)
    readout = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    assert abs(kl_sum(x @ readout.T, changed @ readout.T, torch.tensor([True])).item()) < 1e-12
    assert (x-changed).square().mean().item() == 5000.


def test_shared_scalar_accumulates_all_use_gradients():
    w = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    (3*w+5*w).backward()
    assert w.grad.item() == 8.


def test_approximate_relation_is_nontransitive_and_joint_substitutions_interact():
    from poc.synthetic import make_causal_controls
    fixture = make_causal_controls()
    a, b, c = map(fixture.nontransitive_readout, fixture.nontransitive_states)
    assert abs(a-b) <= fixture.tolerance
    assert abs(b-c) <= fixture.tolerance
    assert abs(a-c) > fixture.tolerance
    assert fixture.joint_suffix(True, False) == fixture.joint_suffix(False, True) == 0.
    assert fixture.joint_suffix(True, True) > 0.

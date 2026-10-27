"""Catch reversed KL, vocabulary truncation, clamping, and masked reductions."""
import importlib
import importlib.util
import math

import pytest
import torch

pytestmark = pytest.mark.cpu


def api():
    assert importlib.util.find_spec('poc.losses') is not None, 'exact loss reductions are missing'
    return importlib.import_module('poc.losses')


def test_asymmetric_teacher_left_kl_and_identity():
    loss = api()
    a = torch.tensor([[.75, .25]], dtype=torch.float64).log()
    b = torch.tensor([[.5, .5]], dtype=torch.float64).log()
    valid = torch.tensor([True])
    assert abs(loss.kl_sum(a, b, valid).item() - .13081203594113697) < 1e-12
    assert abs(loss.kl_sum(b, a, valid).item() - .14384103622589045) < 1e-12
    assert abs(loss.kl_sum(a, a, valid).item()) < 1e-15


def test_all_vocabulary_coordinates_contribute_and_eos_is_scored():
    loss = api()
    a = torch.tensor([[.6, .3, .1]], dtype=torch.float64).log()
    b = torch.tensor([[.6, .2, .2]], dtype=torch.float64).log()
    mask = torch.tensor([True])
    assert abs(loss.kl_sum(a, b, mask).item() - (.3*math.log(1.5)+.1*math.log(.5))) < 1e-12
    assert abs(loss.nll_sum(b, torch.tensor([2]), mask).item() + math.log(.2)) < 1e-12


def test_mask_excludes_invalid_padding_before_log_softmax_and_gather():
    loss = api()
    x = torch.tensor([[.5, .5], [float('nan'), 0]], dtype=torch.float64).log()
    mask = torch.tensor([True, False])
    assert loss.kl_sum(x, x, mask).item() == 0
    assert abs(loss.nll_sum(x, torch.tensor([0, -999]), mask).item() - math.log(2)) < 1e-12
    assert loss.nll_sum(x, torch.tensor([0, -999]), ~torch.ones_like(mask)).item() == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_valid_logits_fail_explicitly(value):
    loss = api()
    x = torch.tensor([[value, 0.]])
    with pytest.raises(FloatingPointError, match='finite'):
        loss.kl_sum(torch.zeros_like(x), x, torch.tensor([True]))
    with pytest.raises(FloatingPointError, match='finite'):
        loss.nll_sum(x, torch.tensor([0]), torch.tensor([True]))


def test_teacher_target_is_detached_and_accelerator_math_is_at_least_fp32():
    loss = api()
    a = torch.tensor([[1., 2., 3.]], requires_grad=True)
    b = torch.tensor([[3., 2., 1.]], dtype=torch.float16, requires_grad=True)
    result = loss.kl_sum(a, b, torch.tensor([True]))
    assert result.dtype == torch.float32
    result.backward()
    assert a.grad is None
    assert b.grad.abs().sum() > 0


def test_raw_finite_negative_metric_is_preserved():
    api()
    from poc.contracts import MetricSums
    value = MetricSums(kl=-1e-8, nll=2., valid_targets=3, disagreements=1)
    assert value.kl == -1e-8
    merged = value + MetricSums(kl=2e-8, nll=4., valid_targets=2, disagreements=0)
    assert merged.kl == 1e-8 and merged.valid_targets == 5
    assert merged.nll == 6. and merged.disagreements == 1
    with pytest.raises(ValueError, match='finite'):
        MetricSums(kl=float('nan'), nll=0., valid_targets=1, disagreements=0)


def test_raw_roundoff_negative_kl_is_not_clamped():
    loss = api()
    # Fixed near-identical distributions measured with this native FP32 build.
    teacher = torch.tensor([[.6613521575927734, .266924113035202, .06167725846171379]])
    student = torch.tensor([[.6613583564758301, .2669195830821991, .06167559698224068]])
    actual = loss.kl_sum(teacher, student, torch.tensor([True])).item()
    assert -1e-7 < actual < 0

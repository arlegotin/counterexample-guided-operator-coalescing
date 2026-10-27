"""Catch lost tied-head paths, per-chunk normalization, and microbatch resets."""
from copy import deepcopy
import importlib
import importlib.util

import pytest
import torch

from poc.contracts import Batch
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder
from poc.bank import make_bank, canonicalize_partition

pytestmark = pytest.mark.cpu


def api():
    assert importlib.util.find_spec('poc.losses') is not None, 'chunked loss/backward is missing'
    return importlib.import_module('poc.losses')


def batch(length=8, *, device='cpu', empty=False):
    ids = torch.tensor([[1, 4, 2, 3, 8, 2, 2, 2][:length],
                        [1, 3, 5, 2, 9, 2, 3, 2][:length]], device=device)
    attention = torch.ones_like(ids, dtype=torch.bool)
    if length == 8:
        attention[0, -2:] = False
    mask = attention[:, :-1] & attention[:, 1:]
    if mask.numel():
        mask[1, 0] = False
    if empty:
        mask.zero_()
    return Batch(ids, attention, mask, torch.arange(length, device=device).expand(2, -1), ('a','b'), ('a','b'))


def models(*, device='cpu', dtype=torch.float64, tied=True):
    teacher = make_tiny_decoder(TinyConfig(vocab_size=13), seed=72, dtype=torch.float64).to(device=device, dtype=dtype)
    student = make_bank(teacher, canonicalize_partition([[0, 2], [1]], 3), 2, 8)
    if tied:
        student.embed_tokens.weight.requires_grad_(True)
    return teacher, student


def full(teacher, student, batches, alpha):
    """Independent unchunked teacher-left reference; no production loss helper."""
    total = sum(int(b.target_mask.sum()) for b in batches)
    sums = [0., 0., 0]
    objective = None
    for b in batches:
        with torch.no_grad():
            t = teacher.logits(b)[:, :-1][b.target_mask].log_softmax(-1)
        s = student.logits(b)[:, :-1][b.target_mask].log_softmax(-1)
        kl = (t.exp() * (t-s)).sum()
        nll = -s.gather(-1, b.input_ids[:, 1:][b.target_mask, None]).sum()
        sums[0] += kl.item(); sums[1] += nll.item()
        sums[2] += int((t.argmax(-1) != s.argmax(-1)).sum())
        value = (kl + alpha*nll)/total
        objective = value if objective is None else objective+value
    objective.backward()
    return sums


@pytest.mark.parametrize('chunk_size', [1, 4, 32])
@pytest.mark.parametrize('alpha', [0., .1, 1.])
@pytest.mark.parametrize('tied', [False, True])
def test_every_owner_gradient_matches_full_fp64_across_ragged_microbatches(chunk_size, alpha, tied):
    loss = api()
    teacher, reference = models(tied=tied)
    candidate = deepcopy(reference)
    batches = [batch(), batch(5)]
    expected = full(teacher, reference, batches, alpha)
    denominator = sum(int(b.target_mask.sum()) for b in batches)
    results = [loss.backward_distillation(teacher, candidate, b, alpha=alpha,
               denominator=denominator, chunk_size=chunk_size) for b in batches]
    merged = results[0] + results[1]
    assert merged.valid_targets == denominator
    assert abs(merged.kl - expected[0]) < 1e-12
    assert abs(merged.nll - expected[1]) < 1e-12
    assert merged.disagreements == expected[2]
    for (name, p), (other_name, q) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None, name
            torch.testing.assert_close(q.grad, p.grad, atol=1e-12, rtol=1e-11, msg=name)
        else:
            assert p.grad is None and q.grad is None, name
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize('chunk_size', [1, 4, 32])
def test_stream_metrics_matches_full_with_ragged_padding_and_eos(chunk_size):
    loss = api(); teacher, student = models()
    b = batch()
    # ID 2 represents both real EOS and padding; target_mask owns inclusion.
    expected = full(teacher, deepcopy(student), [b], .1)
    actual = loss.stream_metrics(teacher, student, b, chunk_size)
    assert actual.valid_targets == 11
    assert abs(actual.kl-expected[0]) < 1e-12
    assert abs(actual.nll-expected[1]) < 1e-12
    assert actual.disagreements == expected[2]
    assert all(p.grad is None for p in student.parameters())


def test_empty_batch_explicit_skip_preserves_gradients_and_optimizer_count():
    loss = api(); teacher, student = models()
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad])
    b = batch(empty=True)
    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    empty_batch_event = loss.backward_distillation(teacher, student, b, alpha=.1, denominator=0, chunk_size=4)
    assert empty_batch_event.reason == 'no_valid_targets'
    assert empty_batch_event.status == 'skipped'
    # Trainer owns stepping, and must obey the explicit skip event.
    if empty_batch_event.status != 'skipped':
        optimizer.step()
    assert not optimizer.state
    for name, p in student.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, before[name], atol=0, rtol=0)
    assert loss.stream_metrics(teacher, student, b, 4).reason == 'no_valid_targets'


@pytest.mark.parametrize('kwargs', [{'denominator':0}, {'denominator':1}, {'alpha':-1.}, {'chunk_size':0}])
def test_invalid_update_normalization_fails_before_backward(kwargs):
    loss = api(); teacher, student = models()
    options = dict(alpha=.1, denominator=11, chunk_size=4) | kwargs
    with pytest.raises(ValueError):
        loss.backward_distillation(teacher, student, batch(), **options)
    assert all(p.grad is None for p in student.parameters())


@pytest.mark.mps
@pytest.mark.parametrize('chunk_size', [1, 4, 32])
@pytest.mark.parametrize('alpha', [0., .1, 1.])
def test_mps_chunked_loss_and_tied_gradients_match_measured_tiny_envelope(chunk_size, alpha):
    import json
    from pathlib import Path
    assert torch.backends.mps.is_available(), 'explicit MPS loss check requires real Metal'
    loss = api()
    cpu_t, cpu_s = models()
    mps_t, mps_s = models(device='mps', dtype=torch.float32)
    inputs = [batch(), batch(5)]
    ref = full(cpu_t, cpu_s, inputs, alpha)
    count = sum(int(b.target_mask.sum()) for b in inputs)
    results = [loss.backward_distillation(mps_t, mps_s, batch(length, device='mps'), alpha=alpha,
               denominator=count, chunk_size=chunk_size) for length in (8, 5)]
    result = results[0] + results[1]
    path = Path(__file__).resolve().parents[1] / 'provenance/loss-tiny-numerics-2026-10-27.json'
    tolerance = json.loads(path.read_text())['tolerances']
    assert abs(result.kl-ref[0]) < tolerance['kl_sum_atol']
    assert abs(result.nll-ref[1]) < tolerance['nll_sum_atol']
    for (name,p), (_,q) in zip(cpu_s.named_parameters(), mps_s.named_parameters(), strict=True):
        if p.requires_grad:
            torch.testing.assert_close(q.grad.cpu().double(), p.grad, atol=tolerance['gradient_atol'], rtol=0, msg=name)
    assert all(p.grad is None for p in mps_t.parameters())


def test_teacher_branch_frozen_even_when_target_parameters_require_gradients():
    loss = api(); teacher, student = models()
    teacher.requires_grad_(True)
    loss.backward_distillation(teacher, student, batch(), alpha=.1, denominator=11, chunk_size=4)
    assert all(p.grad is None for p in teacher.parameters())
    assert student.embed_tokens.weight.grad.abs().sum() > 0


def test_empty_microbatch_preserves_existing_accumulated_gradients():
    loss = api(); teacher, student = models()
    loss.backward_distillation(teacher, student, batch(), alpha=.1, denominator=11, chunk_size=4)
    before = {name: p.grad.clone() for name,p in student.named_parameters() if p.requires_grad}
    event = loss.backward_distillation(teacher, student, batch(empty=True), alpha=.1, denominator=11, chunk_size=1)
    assert event.reason == 'no_valid_targets'
    for name,p in student.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, before[name], atol=0, rtol=0)


def test_readout_positions_bounded_and_full_logit_adapter_never_required():
    loss = api(); teacher, student = models()
    # Instrument real readout execution; forward values and VJPs remain real.
    class ReadoutOnly:
        def __init__(self, model):
            self.model = model
            self.widths = []
        def final_hidden(self, batch):
            return self.model.final_hidden(batch)
        def readout(self, hidden):
            self.widths.append(hidden.shape[0])
            return self.model.readout(hidden)
    t, s = ReadoutOnly(teacher), ReadoutOnly(student)
    loss.backward_distillation(t, s, batch(), alpha=.1, denominator=11, chunk_size=4)
    assert max(t.widths) <= 4 and max(s.widths) <= 4
    assert len(t.widths) == len(s.widths) == 4

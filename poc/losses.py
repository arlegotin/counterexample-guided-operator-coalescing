"""Exact temperature-one, teacher-left losses with bounded readout position chunks.

Readout and upstream VJPs are separated so trainable tied owners receive both
paths. No optimizer operation or gradient reset belongs to this module. A
nonfinite exception invalidates the caller's whole update, including any earlier
chunk gradients; the caller must discard that update instead of stepping it.
"""
import math

import torch

from poc.contracts import Batch, DecoderAdapter, MetricSums


def _validate_logits(logits: torch.Tensor, mask: torch.Tensor) -> None:
    if logits.ndim < 2 or logits.shape[-1] < 1 or tuple(logits.shape[:-1]) != tuple(mask.shape):
        raise ValueError('logits [...,V] and mask [...] must align')
    if not logits.is_floating_point() or mask.dtype != torch.bool or mask.device != logits.device:
        raise ValueError('floating logits and boolean mask must share a device')


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if logits.device.type == 'cpu' and logits.dtype == torch.float64 else torch.float32
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError('valid logits must be finite')
    result = logits.to(dtype).log_softmax(-1)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError('log-softmax must be finite')
    return result


def _finite(value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).item():
        raise FloatingPointError('loss sum must be finite')
    return value


def kl_sum(teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum KL(T || M) over valid positions and the complete vocabulary.

    Teacher targets are always detached. Mask selection precedes normalization,
    allowing unused padding logits to contain meaningless values.
    """
    _validate_logits(teacher_logits, mask)
    _validate_logits(student_logits, mask)
    if teacher_logits.shape != student_logits.shape:
        raise ValueError('teacher and student must use the same complete vocabulary')
    teacher = _log_probs(teacher_logits.detach()[mask])
    student = _log_probs(student_logits[mask])
    return _finite((teacher.exp() * (teacher - student)).sum())


def nll_sum(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum next-token NLL using target_mask, never inferred pad/EOS identity."""
    _validate_logits(logits, mask)
    if targets.shape != mask.shape or targets.dtype != torch.int64 or targets.device != logits.device:
        raise ValueError('int64 targets must align with mask on the logits device')
    selected = targets[mask]
    if ((selected < 0) | (selected >= logits.shape[-1])).any().item():
        raise ValueError('valid target outside vocabulary')
    return _finite(-_log_probs(logits[mask]).gather(-1, selected.unsqueeze(-1)).sum())


def _count(batch: Batch, chunk_size: int) -> int:
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if batch.target_mask.dtype != torch.bool:
        raise ValueError('target_mask must be boolean')
    return int(batch.target_mask.sum().item())


def _skipped() -> MetricSums:
    return MetricSums(status='skipped', reason='no_valid_targets')


def _chunks(teacher, student, teacher_hidden, student_hidden, batch, chunk_size):
    mask = batch.target_mask.reshape(-1)
    targets = batch.input_ids[:, 1:].reshape(-1)
    for start in range(0, mask.numel(), chunk_size):
        valid = mask[start:start+chunk_size]
        if not valid.any().item():
            continue
        # Every class survives: chunk only positions, never the vocabulary.
        with torch.no_grad():
            t = teacher.readout(teacher_hidden[start:start+chunk_size])
        s = student.readout(student_hidden[start:start+chunk_size])
        kl = kl_sum(t, s, valid)
        nll = nll_sum(s, targets[start:start+chunk_size], valid)
        disagreements = int((t.argmax(-1)[valid] != s.argmax(-1)[valid]).sum().item())
        yield kl, nll, disagreements


def _metrics(parts, count):
    return MetricSums(kl=math.fsum(p[0] for p in parts), nll=math.fsum(p[1] for p in parts),
                      valid_targets=count, disagreements=sum(p[2] for p in parts))


@torch.no_grad()
def stream_metrics(teacher: DecoderAdapter, student: DecoderAdapter, batch: Batch, chunk_size: int) -> MetricSums:
    """Return raw sums; a caller evaluating a population must reject count zero."""
    count = _count(batch, chunk_size)
    if not count:
        return _skipped()
    t = teacher.final_hidden(batch)[:, :-1]
    h = student.final_hidden(batch)[:, :-1]
    t, h = t.reshape(-1, t.shape[-1]), h.reshape(-1, h.shape[-1])
    parts = [(float(k.item()), float(n.item()), d) for k, n, d in _chunks(teacher, student, t, h, batch, chunk_size)]
    return _metrics(parts, count)


def backward_distillation(teacher: DecoderAdapter, student: DecoderAdapter, batch: Batch, *,
                          alpha: float, denominator: int, chunk_size: int) -> MetricSums:
    """Accumulate (KL + alpha*NLL)/whole-update-target-count gradients.

    Caller zeros once before all microbatches and steps once after a nonempty,
    successful update. Empty batches return a skip event without a model forward.
    Teacher final_hidden/readout are frozen target branches even if some target
    parameters happen to require gradients. Candidate suffix graphs can be
    supplied through an adapter implementing final_hidden/readout.
    """
    count = _count(batch, chunk_size)
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    if type(denominator) is not int or denominator < count:
        raise ValueError('denominator must cover this batch and the entire update')
    if not count:
        return _skipped()
    with torch.no_grad():
        t = teacher.final_hidden(batch)[:, :-1]
        t = t.reshape(-1, t.shape[-1])
    h = student.final_hidden(batch)[:, :-1]
    z = h.detach().reshape(-1, h.shape[-1]).requires_grad_(True)
    parts = []
    for kl, nll, disagreements in _chunks(teacher, student, t, z, batch, chunk_size):
        parts.append((float(kl.detach().item()), float(nll.detach().item()), disagreements))
        _finite((kl + alpha*nll)/denominator).backward()
    # Head chunks accumulate directly into the canonical head owner. Add lookup
    # and upstream contributions once, preserving the sum for tied weights.
    if h.requires_grad:
        h.backward(z.grad.reshape_as(h))
    return _metrics(parts, count)

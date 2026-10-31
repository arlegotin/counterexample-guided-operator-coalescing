"""Named nonlinear FFN-output reconstruction loss for a separately costed warm start."""
from dataclasses import dataclass, replace

import torch

from poc.contracts import Boundary, CostRecord
from poc.probes import _match_boundary


@dataclass(frozen=True)
class ReconstructionSums:
    squared_error_sum: float
    valid_targets: int
    output_width: int
    depth: int


@torch.no_grad()
def collect_teacher_boundaries(teacher, batch, depths, *, recorder):
    """One deterministic prefix traversal; retain only requested detached states.

    This dispatch-local result is consumed before an optimizer update, never a
    persistent cache of mutable tensors. Original contexts/masks are preserved.
    A call records one prefix invocation plus its real context processing; the
    number of traversed blocks remains determined by the explicit depths.
    """
    from poc.train import _call
    from poc.runtime.torch_decoder import TorchDecoder
    if not isinstance(teacher, TorchDecoder):
        raise ValueError('single-pass reconstruction requires the qualified Torch decoder')
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('teacher must remain frozen')
    depths = tuple(depths)
    if (not depths or any(type(d) is not int or not 0 <= d < len(teacher.layers) for d in depths)
            or depths != tuple(sorted(set(depths)))):
        raise ValueError('sorted unique executed depths required')
    teacher._validate_batch(batch)
    def collect():
        digest = teacher.manifest().checkpoint_digest
        x = teacher.embed_tokens(batch.input_ids)
        result = []
        for depth in range(depths[-1]+1):
            if depth in depths:
                u = teacher._attention(x, batch, depth)
                boundary = Boundary(u.detach(), batch, depth, digest)
                result.append(boundary)
                if depth < depths[-1]:
                    x = teacher.ff_residual(boundary)
            else:
                x = teacher._block(x, batch, depth)
        if teacher.manifest().checkpoint_digest != digest:
            raise ValueError('teacher changed during boundary collection')
        return tuple(result)
    return _call(recorder, collect, teacher_forward_calls=1,
                 processed_inputs=batch.input_ids.shape[0], processed_tokens=int(batch.attention_mask.sum()))


def backward_ffn_reconstruction(teacher, candidate_operator, boundary, *, denominator,
                                record, loss_mask, recorder):
    """Sum mean-coordinate FFN squared errors / common raw-target denominator.

    Full original context and provenance remain in boundary/record; only the
    explicit loss mask is sliced. Teacher prefix/normalization/targets are frozen.
    CPU FP64 accumulation retains gradients back to the candidate device.
    KL and NLL are not measured by this objective.
    """
    from poc.train import _call
    if (record.use != 'fit' or record.recipient_role != 'fit' or record.source != 'natural'
            or record.prefix_source != 'teacher'):
        raise PermissionError('FFN reconstruction requires fit-only natural teacher boundaries')
    _match_boundary(boundary, record)
    if teacher.manifest().checkpoint_digest != record.teacher_digest:
        raise ValueError('stale reconstruction teacher identity')
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('reconstruction teacher must remain frozen')
    original_mask = torch.tensor([record.target_mask], dtype=torch.bool, device=boundary.u.device)
    if not torch.equal(original_mask, boundary.batch.target_mask):
        raise ValueError('full reconstruction template mask differs from provenance')
    if (not isinstance(loss_mask, torch.Tensor) or loss_mask.dtype != torch.bool
            or loss_mask.shape != original_mask.shape or loss_mask.device != original_mask.device
            or (loss_mask & ~original_mask).any().item()):
        raise ValueError('invalid reconstruction loss-mask slice')
    n = int(loss_mask.sum())
    if type(denominator) is not int or not 0 < n <= denominator:
        raise ValueError('positive common raw-target denominator required')
    recorder.record(CostRecord(valid_target_presentations=n))
    detached = replace(boundary, u=boundary.u.detach())
    layer = teacher.layers[boundary.depth]
    with torch.no_grad():
        x = _call(recorder, lambda: teacher._rms(detached.u, layer.post_attention_layernorm), teacher_forward_calls=1)
        target = _call(recorder, lambda: layer.mlp(x), teacher_forward_calls=1)
    actual = _call(recorder, lambda: candidate_operator(x), student_forward_calls=1)
    if actual.shape != target.shape or actual.shape != boundary.u.shape:
        raise ValueError('reconstruction FFN output shape mismatch')
    error = actual[:, :-1][loss_mask].cpu().double()-target[:, :-1][loss_mask].cpu().double()
    total = error.square().mean(-1).sum()
    if not torch.isfinite(total).item():
        raise FloatingPointError('nonfinite FFN reconstruction loss; no optimizer step')
    _call(recorder, lambda: (total/denominator).backward(), student_backward_calls=1)
    return ReconstructionSums(float(total.detach()), n, actual.shape[-1], boundary.depth)

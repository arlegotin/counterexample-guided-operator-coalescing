"""Explicit recovery-only initialization, with fixed parent and canonical owners."""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch

from poc.bank import BankDecoder, PROJECTIONS, _seed, initialize_residual
from poc.contracts import thaw
from poc.initializers import calibrate_projection_inputs, initialize_weighted_residual
from poc.merge import _call
from poc.records import canonical_json


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class RecoveryInitializerSpec:
    method: str = 'parent_svd'
    calibration_components: int = 16
    ridge_relative: float = 1e-6
    max_condition: float = 1e8
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unknown recovery initializer schema')
        if self.method not in ('parent_svd', 'zero_product', 'weighted_delta_svd'):
            raise ValueError('unknown recovery initializer; nonlinear warm start is a separate training phase')
        if type(self.calibration_components) is not int or self.calibration_components < 1:
            raise ValueError('positive calibration component count required')
        if (type(self.ridge_relative) not in (float, int) or not math.isfinite(self.ridge_relative)
                or self.ridge_relative <= 0 or type(self.max_condition) not in (float, int)
                or not math.isfinite(self.max_condition) or self.max_condition <= 1):
            raise ValueError('positive ridge and finite condition limit >1 required')

    def digest(self):
        return _hash(asdict(self))


def initialize_recovery_corrections(model, teacher, affected_depths, fit, batches, seed, spec, *, recorder=None):
    """Apply one named warm initialization before optimizer construction.

    The existing parent-SVD trial chooses the bank once. This opt-in stage keeps
    that bank, rank, owner identities/trainability and every unaffected tensor.
    The caller supplies fit batches produced by the guarded role-bound reader;
    this routine performs no raw-file reads. All work is staged before mutation.
    """
    if not isinstance(model, BankDecoder) or not isinstance(spec, RecoveryInitializerSpec):
        raise ValueError('bank decoder and versioned initializer spec required')
    _seed(seed)
    depths = tuple(affected_depths)
    if (not depths or len(set(depths)) != len(depths)
            or any(type(d) is not int or not 0 <= d < len(model.layers) for d in depths)):
        raise ValueError('valid affected depths required')
    indices = {model.partition.reuse_map[d] for d in depths}
    if len(indices) != 1 or tuple(sorted(depths)) != model.partition.groups[next(iter(indices))]:
        raise ValueError('initializer must cover all uses of one complete canonical bank')
    if model.base_digest != teacher.manifest().checkpoint_digest:
        raise ValueError('initialization teacher lineage mismatch')
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('teacher must remain frozen')
    if not fit.records or any(r != 'fit' for r in fit.roles.values()):
        raise PermissionError('initialization calibration requires fit-only manifest')
    batches = tuple(batches)
    ids = tuple(b.document_ids[0] for b in batches if len(b.document_ids) == 1)
    if (len(ids) != len(batches) or len(ids) != len(set(ids)) or set(ids) != set(fit.roles)
            or len(set(fit.membership.values())) != spec.calibration_components):
        raise ValueError('exact component-disjoint calibration manifest/batches required')
    for b in batches:
        if (b.cluster_ids != (fit.membership[b.document_ids[0]],)
                or b.target_mask.dtype != torch.bool or not b.target_mask.any().item()
                or b.input_ids.shape[0] != 1):
            raise ValueError('calibration component identity and nonempty target mask required')
    if fit.quarantine:
        raise ValueError('quarantined calibration is forbidden')
    before = model.manifest().checkpoint_digest
    record = {'schema_version': 1, 'spec': asdict(spec), 'spec_sha256': spec.digest(),
              'before_digest': before, 'teacher_digest': teacher.manifest().checkpoint_digest,
              'fit_manifest_object_sha256': fit.digest(), 'rank': model.rank, 'seed': seed,
              'affected_depths': list(depths), 'bank_index': next(iter(indices)),
              'projection_records': [], 'calibration_operator_input_presentations': 0,
              'calibration_raw_target_presentations':0, 'calibration_inventory_valid_positions': sum(int(b.target_mask.sum()) for b in batches)}
    if spec.method == 'parent_svd':
        record['after_digest'] = before
        return record
    staged = {}
    window_hashes = tuple(_hash({'tokens': b.input_ids.tolist(), 'attention': b.attention_mask.tolist(),
                                 'target_mask': b.target_mask.tolist(), 'positions': b.positions.tolist(),
                                 'document_ids': list(b.document_ids), 'components': list(b.cluster_ids)}) for b in batches)
    bank = model.bank[next(iter(indices))]
    for depth in depths:
        xs, zs = [], []
        if spec.method == 'weighted_delta_svd':
            with torch.no_grad():
                for b in batches:
                    boundary = _call(recorder, lambda: teacher.prefix_boundary(b, depth),
                                     teacher_forward_calls=1, processed_inputs=1,
                                     processed_tokens=int(b.attention_mask.sum()))
                    x = _call(recorder, lambda: teacher._rms(boundary.u, teacher.layers[depth].post_attention_layernorm),
                              teacher_forward_calls=1)
                    valid = x[:, :-1][b.target_mask]
                    z = _call(recorder, lambda: torch.nn.functional.silu(teacher.layers[depth].mlp.gate_proj(valid))
                              * teacher.layers[depth].mlp.up_proj(valid), teacher_forward_calls=1)
                    xs.append(valid);zs.append(z)
                    record['calibration_raw_target_presentations']+=int(b.target_mask.sum())
        for qi, projection in enumerate(PROJECTIONS):
            private_seed = (seed+depth*3+qi) % 2**63
            target = getattr(teacher.layers[depth].mlp, projection).weight
            delta = target.detach()-getattr(bank, projection).weight.detach()
            if spec.method == 'zero_product':
                a, b = _call(recorder, lambda: initialize_residual(torch.zeros_like(delta), model.rank, private_seed))
                details = {'projection': projection, 'depth': depth, 'valid_input_rows': 0,
                           'input_width': delta.shape[1], 'method': 'zero_product'}
            else:
                samples = zs if projection == 'down_proj' else xs
                calibration = _call(recorder, lambda: calibrate_projection_inputs(samples, manifest=fit,
                    document_ids=ids, window_sha256s=window_hashes, projection=projection, depth=depth,
                    input_source='teacher', source_checkpoint_sha256=record['teacher_digest'],
                    tokenizer_sha256=teacher.tokenizer_digest))
                result = _call(recorder, lambda: initialize_weighted_residual(delta, model.rank, private_seed, calibration,
                                         ridge_relative=spec.ridge_relative, max_condition=spec.max_condition))
                a, b = result.A, result.B
                details = dict(thaw(result.record), input_width=delta.shape[1], calibration=thaw(calibration.provenance))
                record['calibration_operator_input_presentations'] += calibration.provenance['valid_input_rows']
            staged[(depth, projection)] = a, b
            record['projection_records'].append(details)
    # All validation/calibration/SVD work completed without modifying any owner.
    with torch.no_grad():
        for (depth, projection), (a, b) in staged.items():
            if not model.rank:
                continue
            factors = model.layers[depth].mlp.residuals[projection]
            factors.A.copy_(a);factors.B.copy_(b)
    record['after_digest'] = model.manifest().checkpoint_digest
    return record

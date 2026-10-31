"""Opt-in fit-calibrated delta SVD; the historical initializer is unchanged.

The weighted optimum concerns a fixed linear projection and its fixed inputs.
It does not imply optimal nonlinear FFN reconstruction or downstream model KL.
"""
from dataclasses import dataclass
import hashlib
import math

import torch

from poc.bank import PROJECTIONS, _rank, _seed, initialize_residual
from poc.contracts import freeze, thaw
from poc.data.manifest import DataManifest
from poc.records import canonical_json


def _sha(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _digest_tensor(value):
    work = value.detach().cpu().contiguous()
    h = hashlib.sha256(canonical_json({'dtype': str(work.dtype), 'shape': list(work.shape)}))
    h.update(work.numpy().tobytes())
    return h.hexdigest()


def _hash_string(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


@dataclass(frozen=True)
class ProjectionCalibration:
    """A CPU FP64 snapshot, with a content check before every use.

    Tensor mutability cannot be disabled by a frozen dataclass. Mutation of this
    snapshot is detected, rather than trusting a cached digest of live tensors.
    """
    second_moment: torch.Tensor
    provenance: object
    content_sha256: str

    def digest(self):
        current = _sha({'provenance': thaw(self.provenance),
                        'second_moment_sha256': _digest_tensor(self.second_moment)})
        if current != self.content_sha256:
            raise ValueError('calibration snapshot was mutated; digest mismatch')
        return current


def calibrate_projection_inputs(inputs, *, manifest, document_ids, window_sha256s,
                                projection, depth, input_source,
                                source_checkpoint_sha256, tokenizer_sha256):
    """Accumulate actual projection inputs supplied by a role-guarded caller.

    One nonempty 2D tensor per document/window, containing only the selected
    valid positions. Down inputs must be SiLU(gate(x))*up(x), not norm2(x).
    This function performs no text I/O: callers must use the guarded fit reader
    to produce activations; all declared roles are checked before iteration.
    """
    if not isinstance(manifest, DataManifest):
        raise ValueError('a fit-only data manifest is required')
    ids, windows = tuple(document_ids), tuple(window_sha256s)
    if (not ids or len(ids) != len(windows) or len(set(ids)) != len(ids)
            or any(i not in manifest.roles for i in ids)):
        raise ValueError('calibration requires unique declared fit documents/windows')
    if any(manifest.roles[i] != 'fit' for i in ids):
        raise ValueError('only fit documents may calibrate initialization')
    if any(i in manifest.quarantine for i in ids):
        raise ValueError('quarantined documents cannot calibrate initialization')
    if (projection not in PROJECTIONS or type(depth) is not int or depth < 0
            or input_source not in ('teacher', 'student')
            or any(not _hash_string(h) for h in (*windows, source_checkpoint_sha256, tokenizer_sha256))):
        raise ValueError('complete projection/window/checkpoint/tokenizer provenance is required')
    values = iter(inputs)
    moment, count, documents = None, 0, []
    for document_id, window_sha in zip(ids, windows):
        try:
            x = next(values)
        except StopIteration as exc:
            raise ValueError('one activation tensor is required per declared window') from exc
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1
                or x.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(x).all().item()):
            raise ValueError('activation inputs must be nonempty finite FP32/FP64 matrices')
        work = x.detach().cpu().double().contiguous()
        if moment is not None and moment.shape[0] != work.shape[1]:
            raise ValueError('projection input width must be constant')
        contribution = work.T @ work
        if not torch.isfinite(contribution).all().item():
            raise ValueError('nonfinite activation second moment')
        moment = contribution if moment is None else moment + contribution
        count += len(work)
        documents.append({'document_id': document_id, 'component_id': manifest.membership[document_id],
                          'window_sha256': window_sha, 'input_sha256': _digest_tensor(work),
                          'valid_input_rows': len(work)})
    try:
        next(values)
    except StopIteration:
        pass
    else:
        raise ValueError('undeclared activation input window')
    moment = (moment / count).contiguous()
    if not torch.isfinite(moment).all().item():
        raise ValueError('nonfinite accumulated second moment')
    provenance = freeze({'schema_version': 1, 'kind': 'fit_projection_second_moment_v1',
                         'manifest_object_sha256': manifest.digest(), 'documents': documents,
                         'projection': projection, 'depth': depth, 'input_source': input_source,
                         'source_checkpoint_sha256': source_checkpoint_sha256,
                         'tokenizer_sha256': tokenizer_sha256, 'valid_input_rows': count,
                         'input_width': moment.shape[0], 'centered': False})
    digest = _sha({'provenance': thaw(provenance), 'second_moment_sha256': _digest_tensor(moment)})
    return ProjectionCalibration(moment, provenance, digest)


@dataclass(frozen=True)
class WeightedInitialization:
    A: torch.Tensor
    B: torch.Tensor
    record: object


def initialize_weighted_residual(delta, rank, seed, calibration, *,
                                 ridge_relative=1e-6, max_condition=1e8):
    """Solve min_rank<=r tr((delta-BA) C (delta-BA)^T), returning A,B.

    C is the fit-only uncentered second moment plus a positive ridge. CPU FP64
    Cholesky/SVD and a triangular solve avoid forming an inverse. Returned factors
    retain delta's dtype/device and are contiguous, detached tensors.
    """
    if (not isinstance(delta, torch.Tensor) or delta.ndim != 2 or min(delta.shape) < 1
            or delta.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(delta).all().item()):
        raise ValueError('delta must be a finite floating projection matrix')
    _rank(rank, delta.shape)
    _seed(seed)
    if (type(ridge_relative) not in (float, int) or not math.isfinite(ridge_relative)
            or ridge_relative <= 0 or type(max_condition) not in (float, int)
            or not math.isfinite(max_condition) or max_condition <= 1):
        raise ValueError('positive finite ridge and finite condition limit >1 are required')
    if not isinstance(calibration, ProjectionCalibration):
        raise ValueError('fit projection calibration is required')
    calibration_sha = calibration.digest()
    moment = calibration.second_moment
    if moment.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError('calibration input width differs from projection input width')
    work = delta.detach().cpu().double()
    eigen = torch.linalg.eigvalsh(moment)
    low, high = float(eigen[0]), float(eigen[-1])
    ridge = max(ridge_relative * max(float(moment.diagonal().mean()), 1e-12),
                (high-max_condition*low)/(max_condition-1))
    # Resolve rounding at the condition boundary with a fixed deterministic step.
    ridge = math.nextafter(ridge, math.inf)
    condition = (high+ridge)/(low+ridge)
    if not math.isfinite(ridge) or not math.isfinite(condition) or condition > max_condition*(1+1e-10):
        raise ValueError('calibration cannot be regularized within declared condition bound')
    C = moment + ridge*torch.eye(moment.shape[0], dtype=torch.float64)
    S = torch.linalg.cholesky(C)
    weighted = work @ S
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    if rank == 0 or not torch.count_nonzero(work).item():
        a, b = initialize_residual(delta, rank, seed)
    else:
        root = singular[:rank].sqrt()
        b = u[:, :rank]*root
        a = torch.linalg.solve_triangular(S.T, (root[:, None]*vh[:rank]).T, upper=True).T
        a = a.to(device=delta.device, dtype=delta.dtype).contiguous()
        b = b.to(device=delta.device, dtype=delta.dtype).contiguous()
    error = work-b.detach().cpu().double()@a.detach().cpu().double()
    config = {'schema_version': 1, 'method': 'weighted_delta_svd_v1', 'rank': rank,
              'seed': seed, 'ridge_relative': ridge_relative, 'max_condition': max_condition,
              'calibration_sha256': calibration_sha, 'delta_sha256': _digest_tensor(delta)}
    record = dict(config, initializer_sha256=_sha(config), ridge=ridge, condition=condition,
                  valid_input_rows=calibration.provenance['valid_input_rows'],
                  projection=calibration.provenance['projection'], depth=calibration.provenance['depth'],
                  weighted_tail_energy=float(singular[rank:].square().sum()),
                  returned_factor_objective=float(((error@C)*error).sum()),
                  input_reconstruction_mse=float(((error@moment)*error).sum()/delta.shape[0]),
                  singular_values=singular.tolist(), A_sha256=_digest_tensor(a), B_sha256=_digest_tensor(b))
    return WeightedInitialization(a, b, freeze(record))

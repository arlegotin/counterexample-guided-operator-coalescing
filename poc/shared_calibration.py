"""Fit-only actual-input statistics for every use of a canonical FFN bank.

CPU FP32 execution and FP64 uncentered moments form a separate calibration
profile. No training, quantization, or deployment-quality claim is made here.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import re

import torch
from torch import nn

from poc.bank import BankDecoder, PROJECTIONS, _BankUse
from poc.contracts import Batch
from poc.data.manifest import DataManifest
from poc.records import canonical_json
from poc.runtime.torch_decoder import TorchDecoder

NUMERICAL_PROFILE = 'shared-inputs-cpu-fp32-moments-fp64-v1'


def _sha(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class SharedCalibrationSpec:
    layout_sha256: str
    source_sha256: str
    population_sha256: str
    window_plan_sha256: str
    tokenizer_sha256: str
    source: str = 'current_student'
    schema_version: int = 1
    context_length: int = 512
    max_windows: int = 16
    max_statistics_bytes: int = 1073741824
    numerical_profile: str = NUMERICAL_PROFILE

    def __post_init__(self):
        for name in ('layout_sha256', 'source_sha256', 'population_sha256',
                     'window_plan_sha256', 'tokenizer_sha256'):
            value = getattr(self, name)
            if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError(f'{name} must be a SHA256')
        for name in ('schema_version', 'context_length', 'max_windows', 'max_statistics_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.schema_version != 1 or self.context_length != 512:
            raise ValueError('only version 1 and context 512 are qualified')
        if self.source not in ('current_student', 'teacher'):
            raise ValueError('source must be current_student or teacher')
        if self.numerical_profile != NUMERICAL_PROFILE:
            raise ValueError('unqualified numerical profile')

    def identity(self):
        return asdict(self)

    def digest(self):
        return _sha(self.identity())

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('calibration spec requires exactly the versioned fields')
        return cls(**value)


def tensor_sha256(tensor):
    """Hash actual content, shape and dtype; never cache mutable tensor digests."""
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
        raise ValueError('calibration tensors must be on CPU')
    value = tensor.detach().contiguous()
    h = hashlib.sha256(canonical_json({'shape': list(value.shape), 'dtype': str(value.dtype)}))
    h.update(value.reshape(-1).clone().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _context_identity(batch):
    if not isinstance(batch, Batch):
        raise ValueError('explicit Batch windows required')
    return {'input_ids': tensor_sha256(batch.input_ids),
            'attention_mask': tensor_sha256(batch.attention_mask),
            'positions': tensor_sha256(batch.positions),
            'document_ids': list(batch.document_ids), 'component_ids': list(batch.cluster_ids)}


def window_plan_sha256(windows):
    return _sha({'windows': [dict(_context_identity(b), target_mask=tensor_sha256(b.target_mask)) for b in windows]})


def _model_sha(model):
    model.invalidate_state_snapshot()
    return model.manifest().checkpoint_digest


def _plain_model(model):
    if not isinstance(model, TorchDecoder):
        raise ValueError('a TorchDecoder source is required')
    from torch.nn.modules import module as module_api
    if any(getattr(module_api, key, {}) for key in (
            '_global_forward_pre_hooks', '_global_forward_hooks',
            '_global_backward_pre_hooks', '_global_backward_hooks')):
        raise ValueError('global execution instrumentation is not permitted')
    for module in model.modules():
        if any(getattr(module, key, {}) for key in (
                '_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')):
            raise ValueError('pre-existing instrumentation is not permitted')
    for parameter in model.parameters():
        if parameter.device.type != 'cpu' or parameter.dtype != torch.float32:
            raise ValueError('qualified calibration requires CPU FP32 owners')
        if parameter.grad is not None:
            raise ValueError('calibration requires no existing gradients')
    for layer in model.layers:
        if 'projection' in layer.mlp.__dict__:
            raise ValueError('projection overrides are not permitted')


def calibration_preflight(layout, source_model, population, windows, spec):
    """Validate identities, roles and finite storage before any neural dispatch."""
    if not isinstance(spec, SharedCalibrationSpec) or not isinstance(layout, BankDecoder):
        raise ValueError('a versioned spec and BankDecoder layout are required')
    if not isinstance(population, DataManifest) or not isinstance(windows, (tuple, list)):
        raise ValueError('a role manifest and finite explicit window sequence are required')
    if not windows or len(windows) > spec.max_windows:
        raise ValueError('window count exceeds the finite contract')
    if any(role != 'fit' for role in population.roles.values()) or population.quarantine:
        raise ValueError('only unquarantined fit components may calibrate')
    if spec.source == 'current_student':
        if source_model is not layout:
            raise ValueError('current_student source must be this exact layout')
    elif not isinstance(source_model, TorchDecoder) or isinstance(source_model, BankDecoder):
        raise ValueError('teacher source must be a dense TorchDecoder')
    _plain_model(layout)
    if source_model is not layout:
        _plain_model(source_model)
    if _model_sha(layout) != spec.layout_sha256 or _model_sha(source_model) != spec.source_sha256:
        raise ValueError('model identity changed')
    if spec.source == 'teacher' and (layout.base_digest != spec.source_sha256
                                    or len(layout.layers) != len(source_model.layers)):
        raise ValueError('teacher must be the layout base')
    if any(m.tokenizer_digest != spec.tokenizer_sha256 for m in (layout, source_model)):
        raise ValueError('tokenizer identity mismatch')
    if population.digest() != spec.population_sha256 or window_plan_sha256(windows) != spec.window_plan_sha256:
        raise ValueError('population or window identity changed')
    if len(layout.bank) != len(layout.partition.groups):
        raise ValueError('every canonical bank must have a declared use group')
    owners, parameter_ids = {}, set()
    for index, group in enumerate(layout.partition.groups):
        for projection in PROJECTIONS:
            weight = getattr(layout.bank[index], projection).weight
            if id(weight) in parameter_ids:
                raise ValueError('undeclared alias across canonical bank/projection owners')
            parameter_ids.add(id(weight))
            owners[f'bank.{index}.{projection}.weight'] = {
                'use_sites': list(group), 'shape': list(weight.shape)}
    for depth, layer in enumerate(layout.layers):
        wrapper = layer.mlp
        if (not isinstance(wrapper, _BankUse) or wrapper._owners_ref() is not layout.bank
                or wrapper.index != layout.partition.reuse_map[depth]):
            raise ValueError('noncanonical use-site mapping')
        if spec.source == 'teacher':
            for projection in PROJECTIONS:
                module = getattr(source_model.layers[depth].mlp, projection, None)
                expected = owners[f'bank.{wrapper.index}.{projection}.weight']['shape']
                if not isinstance(module, nn.Linear) or module.bias is not None or list(module.weight.shape) != expected:
                    raise ValueError('teacher projection layout mismatch')
    statistics_bytes = sum(8 * (row['shape'][1] ** 2 + 2 * row['shape'][1]) for row in owners.values())
    if statistics_bytes > spec.max_statistics_bytes:
        raise ValueError('statistics exceed the finite storage bound')
    contexts, unique_positions, raw_targets = {}, set(), 0
    for batch in windows:
        source_model._validate_batch(batch)
        if batch.input_ids.shape != (1, spec.context_length) or batch.positions[0, 0].item() != 0:
            raise ValueError('independent batch-one context 512 with position reset required')
        count = int(batch.target_mask.sum())
        if count == 0:
            raise ValueError('empty loss masks may not calibrate')
        doc, component = batch.document_ids[0], batch.cluster_ids[0]
        if population.roles.get(doc) != 'fit' or population.membership.get(doc) != component:
            raise ValueError('window document/component does not belong to fit population')
        context_hash = _sha(_context_identity(batch))
        if doc in contexts and contexts[doc] != context_hash:
            raise ValueError('version 1 permits one distinct context per document')
        contexts[doc] = context_hash
        unique_positions.update((doc, context_hash, t) for t in batch.target_mask[0].nonzero().flatten().tolist())
        raw_targets += count
    return {'spec_sha256': spec.digest(), 'owners': owners, 'statistics_bytes': statistics_bytes,
            'raw_target_presentations': raw_targets,
            'unique_context_target_positions': len(unique_positions),
            'unique_contexts': len(contexts), 'processed_context_tokens': len(windows) * spec.context_length,
            'operator_input_presentations': raw_targets * len(layout.layers),
            'projection_input_presentations': raw_targets * len(layout.layers) * len(PROJECTIONS)}


@dataclass
class OwnerInputStatistics:
    moment_sum: torch.Tensor
    abs_sum: torch.Tensor
    max_abs: torch.Tensor
    count: int
    use_counts: dict

    def identity(self):
        return {'moment_sum_sha256': tensor_sha256(self.moment_sum),
                'abs_sum_sha256': tensor_sha256(self.abs_sum),
                'max_abs_sha256': tensor_sha256(self.max_abs), 'count': self.count,
                'use_counts': {str(k): v for k, v in sorted(self.use_counts.items())}}


@dataclass(frozen=True)
class SharedCalibrationResult:
    spec_sha256: str
    spec: SharedCalibrationSpec
    owners: dict
    owner_layout: dict
    counters: dict
    hidden_sha256: tuple
    window_records: tuple

    def identity(self):
        return {'schema_version': 1, 'spec_sha256': self.spec_sha256,
                'spec': self.spec.identity(), 'owner_layout': self.owner_layout,
                'owners': {k: v.identity() for k, v in sorted(self.owners.items())},
                'counters': dict(self.counters), 'hidden_sha256': list(self.hidden_sha256),
                'window_records': list(self.window_records)}

    def digest(self):
        return _sha(self.identity())


def collect_shared_inputs(layout, source_model, population, windows, spec, recorder=None):
    """Observe actual projection inputs; keep one statistic owner per shared bank.

    The returned tensor sums are mutable calibration data. identity()/digest()
    always hash their actual content. A failed dispatch publishes no result.
    This function temporarily instruments a model and is not concurrent-safe.
    """
    from types import MethodType
    from poc.recovery_controls import ControlRecorder
    from poc.train import _call

    if recorder is None:
        recorder = ControlRecorder()
    preflight = calibration_preflight(layout, source_model, population, windows, spec)
    owners = {}
    for key, row in preflight['owners'].items():
        width = row['shape'][1]
        owners[key] = OwnerInputStatistics(torch.zeros((width, width), dtype=torch.float64),
                                           torch.zeros(width, dtype=torch.float64),
                                           torch.zeros(width, dtype=torch.float64), 0,
                                           {depth: 0 for depth in row['use_sites']})
    modes = [(module, module.training) for module in source_model.modules()]
    trainability = [(parameter, parameter.requires_grad) for parameter in source_model.parameters()]
    handles, overridden, outputs, window_records = [], [], [], []
    active, seen = None, set()

    def observe(depth, projection, inputs):
        if active is None or projection not in PROJECTIONS:
            raise ValueError('projection call outside declared window/projection')
        use = (depth, projection)
        if use in seen:
            raise ValueError('duplicate projection call for a logical use')
        seen.add(use)
        key = f'bank.{layout.partition.reuse_map[depth]}.{projection}.weight'
        stats = owners[key]
        if (not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float32
                or inputs.device.type != 'cpu'
                or tuple(inputs.shape) != (1, spec.context_length, len(stats.abs_sum))
                or not torch.isfinite(inputs).all().item()):
            raise ValueError('invalid or nonfinite actual projection inputs')
        selected = inputs[:, :-1][active.target_mask].detach().double()
        stats.moment_sum.addmm_(selected.T, selected)
        absolute = selected.abs()
        stats.abs_sum.add_(absolute.sum(0))
        torch.maximum(stats.max_abs, absolute.max(0).values, out=stats.max_abs)
        stats.count += len(selected)
        stats.use_counts[depth] += len(selected)

    try:
        source_model.eval()
        for depth, layer in enumerate(source_model.layers):
            if spec.source == 'current_student':
                wrapper, original = layer.mlp, layer.mlp.projection

                def projection(this, inputs, name, *, _depth=depth, _original=original):
                    observe(_depth, name, inputs)
                    return _original(inputs, name)

                wrapper.projection = MethodType(projection, wrapper)
                overridden.append(wrapper)
            else:
                for name in PROJECTIONS:
                    def before(module, args, *, _depth=depth, _name=name):
                        if len(args) != 1:
                            raise ValueError('unexpected projection arguments')
                        observe(_depth, _name, args[0])
                    handles.append(getattr(layer.mlp, name).register_forward_pre_hook(before))
        expected = {(depth, projection) for depth in range(len(layout.layers)) for projection in PROJECTIONS}
        with torch.no_grad():
            for presentation_index, batch in enumerate(windows):
                active, seen = batch, set()
                before_window = window_plan_sha256((batch,))
                raw_count = int(batch.target_mask.sum())
                source = 'teacher' if spec.source == 'teacher' else 'student'
                value = _call(recorder, lambda: source_model.final_hidden(batch),
                              processed_inputs=1, processed_tokens=spec.context_length,
                              raw_corpus_target_presentations=raw_count,
                              **{source + '_forward_calls': 1})
                if seen != expected:
                    raise ValueError('missing projection use-site coverage')
                if (not isinstance(value, torch.Tensor) or value.shape != (1, spec.context_length, layout.config.hidden_size)
                        or value.dtype != torch.float32 or not torch.isfinite(value).all().item()):
                    raise ValueError('invalid final hidden state')
                if window_plan_sha256((batch,)) != before_window:
                    raise ValueError('window mutated during dispatch')
                outputs.append(tensor_sha256(value))
                window_records.append({
                    'spec_sha256': spec.digest(), 'source_sha256': spec.source_sha256,
                    'window_plan_sha256': spec.window_plan_sha256,
                    'window_sha256': before_window, 'presentation_index': presentation_index,
                    'raw_targets': raw_count, 'projection_inputs': raw_count * len(layout.layers) * 3,
                    'hidden_sha256': outputs[-1]})
                active = None
    finally:
        for handle in handles:
            handle.remove()
        for wrapper in overridden:
            del wrapper.projection
        for module, training in modes:
            module.training = training
    if (any(getattr(model, 'tokenizer_digest', None) != spec.tokenizer_sha256
            for model in (layout, source_model))
            or _model_sha(source_model) != spec.source_sha256 or _model_sha(layout) != spec.layout_sha256
            or window_plan_sha256(windows) != spec.window_plan_sha256
            or population.digest() != spec.population_sha256):
        raise ValueError('model or data identity changed during calibration')
    if any(p.requires_grad != required or p.grad is not None for p, required in trainability):
        raise ValueError('calibration altered gradient state')
    raw = preflight['raw_target_presentations']
    for key, stats in owners.items():
        if stats.count != raw * len(stats.use_counts) or any(n != raw for n in stats.use_counts.values()):
            raise ValueError('incomplete canonical owner/use-site counts')
        if not all(torch.isfinite(t).all().item() for t in (stats.moment_sum, stats.abs_sum, stats.max_abs)):
            raise ValueError('nonfinite accumulated moments')
    counters = {k: v for k, v in preflight.items() if k not in ('spec_sha256', 'owners')}
    return SharedCalibrationResult(spec.digest(), spec, owners, preflight['owners'], counters, tuple(outputs), tuple(window_records))

"""Canonical floating-point FFN owners and private unit-scale low-rank residuals.

Construction chooses the first canonical depth as the initial medoid. It does
not infer neuron alignment or select fit-scored merge medoids (Task 10).
"""
from collections.abc import Sequence
from copy import deepcopy
import hashlib
import json
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from poc.contracts import DecoderAdapter, ModelManifest, OwnerRecord, Partition
from poc.runtime.torch_decoder import TorchDecoder

PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


def canonicalize_partition(groups: Sequence[Sequence[int]], depth: int) -> Partition:
    groups = tuple(tuple(group) for group in groups)
    # Validate coverage and scalar types before sorting potentially invalid inputs.
    if (type(depth) is not int or depth < 1 or not 1 <= len(groups) <= depth
            or any(not group for group in groups)):
        raise ValueError('partition requires nonempty groups and 1 <= K <= actual depth')
    flat = [value for group in groups for value in group]
    if any(type(value) is not int for value in flat) or sorted(flat) != list(range(depth)):
        raise ValueError('partition must cover every depth exactly once')
    return Partition(tuple(sorted((tuple(sorted(group)) for group in groups), key=lambda group: group[0])), depth)


def _rank(rank, shape):
    if type(rank) is not int or not 0 <= rank <= min(shape):
        raise ValueError('rank must fit both projection dimensions')


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be a nonnegative int64')


def initialize_residual(delta: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU SVD reference, returned in delta's original dtype/device, as (A,B)."""
    if (not isinstance(delta, torch.Tensor) or delta.ndim != 2 or min(delta.shape) < 1
            or delta.dtype not in (torch.float32, torch.float64) or not torch.isfinite(delta).all().item()):
        raise ValueError('delta must be a finite floating projection matrix')
    _rank(rank, delta.shape); _seed(seed)
    out_width, in_width = delta.shape
    work = delta.detach().cpu().double()
    if rank == 0:
        a, b = work.new_empty((0, in_width)), work.new_empty((out_width, 0))
    elif not torch.count_nonzero(work).item():
        generator = torch.Generator(device='cpu').manual_seed(seed)
        a = torch.randn((rank, in_width), generator=generator, dtype=torch.float64) * .01
        b = work.new_zeros((out_width, rank))
    else:
        u, singular, vh = torch.linalg.svd(work, full_matrices=False)
        root = singular[:rank].sqrt()
        a, b = root[:, None] * vh[:rank], u[:, :rank] * root[None, :]
    return (a.to(device=delta.device, dtype=delta.dtype).contiguous(),
            b.to(device=delta.device, dtype=delta.dtype).contiguous())


class _Factors(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.A, self.B = nn.Parameter(a), nn.Parameter(b)

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)


class _BankUse(nn.Module):
    def __init__(self, owners, index: int, residuals: dict):
        super().__init__()
        # A weak reference avoids registration of bank copies at every depth.
        self._owners_ref = weakref.ref(owners)
        self.index = index
        self.residuals = nn.ModuleDict(residuals)

    def projection(self, x, name):
        owners = self._owners_ref()
        if owners is None:
            raise RuntimeError('bank owner has expired')
        result = F.linear(x, getattr(owners[self.index], name).weight)
        if name in self.residuals:
            result = result + self.residuals[name](x)
        return result

    def forward(self, x):
        return self.projection(F.silu(self.projection(x, 'gate_proj')) * self.projection(x, 'up_proj'), 'down_proj')


class _DenseFFN(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name, weight in weights.items():
            # Meta construction avoids allocating/randomizing a discarded dense matrix.
            projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device='meta', dtype=weight.dtype)
            projection.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
            setattr(self, name, projection)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _copy_backbone(source, target):
    clone = deepcopy(source)
    for name in ('config', 'embed_tokens', 'layers', 'norm', 'lm_head', 'tokenizer_digest', 'head_dim', 'rope_theta'):
        setattr(target, name, getattr(clone, name))
    target.eval()


def _validate_teacher(teacher, partition, rank, seed):
    if not isinstance(teacher, TorchDecoder) or isinstance(teacher, BankDecoder):
        raise ValueError('bank construction requires a dense TorchDecoder base')
    if not isinstance(partition, Partition) or partition.depth != len(teacher.layers):
        raise ValueError('partition must match actual loaded depth')
    # Recheck canonical coverage even for state reconstructed outside the helper.
    Partition(partition.groups, partition.depth)
    _seed(seed)
    hidden = teacher.embed_tokens.weight.shape[1]
    width = None
    for layer in teacher.layers:
        for name in PROJECTIONS:
            projection = getattr(layer.mlp, name, None)
            if not isinstance(projection, nn.Linear) or projection.bias is not None:
                raise ValueError('only bias-free linear FFN projections are supported')
            weight = projection.weight
            if width is None:
                width = weight.shape[0]
            expected = (hidden, width) if name == 'down_proj' else (width, hidden)
            if tuple(weight.shape) != expected or weight.device != teacher.device or weight.dtype != teacher.dtype:
                raise ValueError('actual FFN projection shapes, dtype and device must be compatible')
            _rank(rank, weight.shape)
            if weight.dtype not in (torch.float32, torch.float64) or not torch.isfinite(weight).all().item():
                raise ValueError('bank requires finite scientific FP32 or CPU FP64 weights')


class BankDecoder(TorchDecoder):
    def __init__(self, teacher, partition, rank, seed, *, initialize=True):
        _validate_teacher(teacher, partition, rank, seed)
        nn.Module.__init__(self)
        _copy_backbone(teacher, self)
        self.requires_grad_(False)
        self.partition, self.rank = partition, rank
        self.base_digest = teacher.manifest().checkpoint_digest
        self.bank = nn.ModuleList()
        for group in partition.groups:
            owner = _DenseFFN({name: getattr(teacher.layers[group[0]].mlp, name).weight for name in PROJECTIONS})
            owner.requires_grad_(True)
            self.bank.append(owner)
        for depth, index in enumerate(partition.reuse_map):
            residuals = {}
            if rank:
                for projection_index, name in enumerate(PROJECTIONS):
                    weight = getattr(teacher.layers[depth].mlp, name).weight
                    if initialize:
                        delta = weight.detach() - getattr(self.bank[index], name).weight.detach()
                        a, b = initialize_residual(delta, rank, (seed + depth * 3 + projection_index) % 2**63)
                    else:
                        a, b = weight.new_zeros((rank, weight.shape[1])), weight.new_zeros((weight.shape[0], rank))
                    residuals[name] = _Factors(a, b)
            self.layers[depth].mlp = _BankUse(self.bank, index, residuals)

    def __deepcopy__(self, memo):
        # Python copies weak references unchanged; rebind execution references to
        # the copied canonical owners for independently mutable candidates.
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            setattr(result, name, deepcopy(value, memo))
        for layer in result.layers:
            layer.mlp._owners_ref = weakref.ref(result.bank)
        return result

    @property
    def residual_owner_count(self):
        return sum(len(list(layer.mlp.parameters())) for layer in self.layers)

    def _digest(self):
        # Non-tensor wrapper references affect execution; verify before consulting
        # the tensor snapshot or issuing cache/probe provenance.
        for depth, layer in enumerate(self.layers):
            wrapper = layer.mlp
            if (not isinstance(wrapper, _BankUse) or wrapper._owners_ref() is not self.bank
                    or wrapper.index != self.partition.reuse_map[depth]):
                raise ValueError('actual bank mapping/owner violates declared partition')
        payload = (super()._digest(), self.partition.groups, self.rank, self.base_digest)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def manifest(self) -> ModelManifest:
        original = super().manifest()
        return ModelManifest(original.config, {record.name: record.use_sites for record in owner_census(self)},
                             original.tokenizer_digest, original.checkpoint_digest, original.dtype, original.backend)


def make_bank(teacher: TorchDecoder, partition: Partition, rank: int, seed: int) -> BankDecoder:
    return BankDecoder(teacher, partition, rank, seed)


def owner_census(decoder: DecoderAdapter) -> list[OwnerRecord]:
    """Count real registered parameter identities, attaching executed bank use sites."""
    if not isinstance(decoder, TorchDecoder):
        raise ValueError('owner census currently supports torch decoders')
    aliases = {}
    for name, parameter in decoder.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    categories, uses = {}, {}
    if not isinstance(decoder, BankDecoder):
        for layer in decoder.layers:
            for parameter in layer.mlp.parameters():
                categories[id(parameter)] = 'dense_ffn'
    if isinstance(decoder, BankDecoder):
        for depth, layer in enumerate(decoder.layers):
            wrapper = layer.mlp
            if not isinstance(wrapper, _BankUse) or wrapper._owners_ref() is not decoder.bank or wrapper.index != decoder.partition.reuse_map[depth]:
                raise ValueError('bank depth reference violates canonical ownership')
            for name in PROJECTIONS:
                parameter = getattr(decoder.bank[wrapper.index], name).weight
                categories[id(parameter)] = 'bank'
                uses.setdefault(id(parameter), []).append(f'layers.{depth}.mlp.{name}.weight')
            for parameter in wrapper.parameters():
                categories[id(parameter)] = 'residual'
    return [OwnerRecord(name, tuple(parameter.shape), parameter.numel(), parameter.requires_grad,
                        categories.get(id(parameter), 'non_ffn'), tuple(uses.get(id(parameter), aliases[id(parameter)])))
            for name, parameter in decoder.named_parameters()]


def expand_for_diagnostic(bank: BankDecoder) -> TorchDecoder:
    """Materialize independent dense per-depth FFNs for diagnostic parity only."""
    if not isinstance(bank, BankDecoder):
        raise ValueError('diagnostic expansion requires a bank')
    owner_census(bank)
    result = TorchDecoder.__new__(TorchDecoder)
    nn.Module.__init__(result)
    _copy_backbone(bank, result)
    with torch.no_grad():
        for depth, index in enumerate(bank.partition.reuse_map):
            weights = {}
            for name in PROJECTIONS:
                weight = getattr(bank.bank[index], name).weight
                residual = bank.layers[depth].mlp.residuals
                weights[name] = weight if name not in residual else weight + residual[name].B @ residual[name].A
            result.layers[depth].mlp = _DenseFFN(weights)
    result.requires_grad_(False)
    return result

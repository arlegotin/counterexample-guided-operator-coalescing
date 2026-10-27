"""Backend-independent identity and expenditure contracts (units are explicit)."""
from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
import re
from types import MappingProxyType
from typing import Any


def _number(name: str, value: Any, *, integer: bool = False) -> None:
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative {'integer' if integer else 'number'}")


def freeze(value: Any) -> Any:
    """Snapshot JSON-compatible inputs, recursively, without mutable references."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("mapping keys must be strings")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"not a finite JSON value: {type(value).__name__}")


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Budget:
    """Finite cumulative ceilings; disk and memory are high-water bytes."""
    updates: int
    valid_target_presentations: int
    teacher_forward_calls: int
    suffix_backward_calls: int
    student_backward_calls: int
    search_proposals: int
    search_seconds: float
    wall_seconds: float
    disk_bytes: int
    memory_bytes: int

    def __post_init__(self) -> None:
        for field in fields(self):
            _number(field.name, getattr(self, field.name), integer=not field.name.endswith("seconds"))

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class CostRecord:
    """Actual execution increments; recomputation is charged again.

    Missing memory is explicitly unavailable, never a measured zero.
    Disk/memory values are current observations aggregated by maximum.
    """
    wall_seconds: float = 0
    processed_inputs: int = 0
    processed_tokens: int = 0
    valid_target_presentations: int = 0
    unique_tokens: int = 0
    teacher_forward_calls: int = 0
    suffix_backward_calls: int = 0
    student_backward_calls: int = 0
    estimated_operations: float = 0
    estimation_method: str | None = None
    memory_bytes: int | None = None
    memory_unavailable_reason: str | None = "not measured"
    disk_bytes: int = 0
    updates: int = 0
    search_proposals: int = 0
    search_seconds: float = 0
    student_forward_calls: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ("estimation_method", "memory_unavailable_reason"):
                if value is not None and (type(value) is not str or not value.strip()):
                    raise ValueError(f"{field.name} must be a nonempty reason or null")
            elif field.name == "memory_bytes" and value is None:
                continue
            else:
                _number(field.name, value, integer=field.name not in ("wall_seconds", "search_seconds", "estimated_operations"))
        if self.estimated_operations and not self.estimation_method:
            raise ValueError("estimated operations require an estimation_method")
        if self.memory_bytes is None and not self.memory_unavailable_reason:
            raise ValueError("missing memory requires memory_unavailable_reason")

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class RunSpec:
    method_id: str
    method_variant: str
    resolved_config: Mapping
    source_commit: str
    dirty_patch_sha256: str | None
    dependency_lock_sha256: str
    input_manifests: Mapping
    hardware_policy: Mapping
    seeds: Mapping
    parent_run_ids: tuple[str, ...]
    budgets: Budget

    def __post_init__(self) -> None:
        if self.method_id not in {"T", "Q", "R", "RC", "CKA", "BS", "SV", "P", "H", "QCE", "M"}:
            raise ValueError("unknown method_id")
        if not isinstance(self.method_variant, str) or not self.method_variant.strip():
            raise ValueError("method_variant must be explicit")
        if not isinstance(self.source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", self.source_commit):
            raise ValueError("source_commit must be a full Git commit")
        hashes = [self.dependency_lock_sha256, *self.input_manifests.values(), *self.parent_run_ids]
        if self.dirty_patch_sha256 is not None:
            hashes.append(self.dirty_patch_sha256)
        if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes):
            raise ValueError("manifest, lock, patch and parent hashes must be full SHA-256 digests")
        if not isinstance(self.budgets, Budget):
            raise ValueError("resolved finite budgets required")
        for name in ("resolved_config", "input_manifests", "hardware_policy", "seeds"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a nonempty mapping")
            object.__setattr__(self, name, freeze(value))
        for name, seed in self.seeds.items():
            _number(f"seeds.{name}", seed, integer=True)
        object.__setattr__(self, "parent_run_ids", tuple(self.parent_run_ids))

    def to_dict(self) -> dict:
        return {field.name: (self.budgets.to_dict() if field.name == "budgets" else thaw(getattr(self, field.name)))
                for field in fields(self)}


# These contracts intentionally do not import a numerical backend. Live backend
# tensors pass through untouched; adapters enforce dtype/device/value semantics.
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

Tensor = Any
FFNOperator = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class Batch:
    input_ids: Tensor
    attention_mask: Tensor
    target_mask: Tensor
    positions: Tensor
    document_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = tuple(self.input_ids.shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError('input_ids must be nonempty [B,S]')
        b, s = shape
        if tuple(self.attention_mask.shape) != shape or tuple(self.positions.shape) != shape:
            raise ValueError('attention_mask and positions must be [B,S]')
        if tuple(self.target_mask.shape) != (b, s - 1):
            raise ValueError('target_mask must be [B,S-1]')
        for name in ('document_ids', 'cluster_ids'):
            values = tuple(getattr(self, name))
            if len(values) != b or any(type(x) is not str or not x for x in values):
                raise ValueError(f'{name} must contain one nonempty identity per row')
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class Boundary:
    u: Tensor
    batch: Batch
    depth: int
    model_digest: str

    def __post_init__(self) -> None:
        if len(self.u.shape) != 3 or tuple(self.u.shape[:2]) != tuple(self.batch.input_ids.shape):
            raise ValueError('boundary u must be [B,S,d]')
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError('boundary depth must be a nonnegative integer')
        if not re.fullmatch(r'[0-9a-f]{64}', self.model_digest):
            raise ValueError('boundary requires exact model digest')


@dataclass
class LayerCache:
    """Mutable inference buffers; never shared between executed depths."""
    key: Tensor | None = None
    value: Tensor | None = None


@dataclass
class CacheState:
    """Inference state, unlike immutable scientific records, mutates once per call."""
    layers: tuple[LayerCache, ...]
    start_position: int = 0
    length: int = 0
    model_digest: str | None = None

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        if not self.layers or len({id(layer) for layer in self.layers}) != len(self.layers):
            raise ValueError('cache requires distinct per-depth buffers')
        if type(self.start_position) is not int or self.start_position < 0 or type(self.length) is not int or self.length < 0:
            raise ValueError('cache positions and length must be nonnegative integers')


@dataclass(frozen=True)
class ModelManifest:
    config: Mapping
    owner_use_sites: Mapping
    tokenizer_digest: str | None
    checkpoint_digest: str
    dtype: str
    backend: str

    def __post_init__(self) -> None:
        for name in ('config', 'owner_use_sites'):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        for digest in (self.tokenizer_digest, self.checkpoint_digest):
            if digest is not None and not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('manifest requires SHA-256 digests or explicit missing tokenizer')


@dataclass(frozen=True)
class ContinuationRequest:
    prompt_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]
    boundary_policy_digest: str
    item_id: str

    def __post_init__(self) -> None:
        for name in ('prompt_ids', 'continuation_ids'):
            ids = tuple(getattr(self, name))
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError(f'{name} must contain nonnegative token IDs')
            object.__setattr__(self, name, ids)
        if not self.boundary_policy_digest or not self.item_id:
            raise ValueError('continuations require boundary policy and item identity')


@dataclass(frozen=True)
class ContinuationScore:
    item_id: str
    log_likelihood: float
    scored_tokens: int


@runtime_checkable
class DecoderAdapter(Protocol):
    def final_hidden(self, batch: Batch, *, cache: CacheState | None = None) -> Tensor: ...
    def readout(self, hidden: Tensor) -> Tensor: ...
    def logits(self, batch: Batch, *, cache: CacheState | None = None) -> Tensor: ...
    def prefix_boundary(self, batch: Batch, depth: int) -> Boundary: ...
    def ff_residual(self, boundary: Boundary, operator: FFNOperator | None = None) -> Tensor: ...
    def suffix_hidden(self, after_ffn: Tensor, boundary: Boundary) -> Tensor: ...
    def suffix_logits(self, after_ffn: Tensor, boundary: Boundary) -> Tensor: ...
    def score_continuations(self, requests: Sequence[ContinuationRequest]) -> list[ContinuationScore]: ...
    def manifest(self) -> ModelManifest: ...


@dataclass(frozen=True)
class Partition:
    """Canonical groups ordered by their first original depth; complete coverage."""
    groups: tuple[tuple[int, ...], ...]
    depth: int

    def __post_init__(self) -> None:
        groups = tuple(tuple(group) for group in self.groups)
        if type(self.depth) is not int or self.depth < 1 or not 1 <= len(groups) <= self.depth:
            raise ValueError('partition requires 1 <= K <= actual depth')
        flat = [depth for group in groups for depth in group]
        if (any(not group for group in groups) or any(type(depth) is not int for depth in flat)
                or sorted(flat) != list(range(self.depth))):
            raise ValueError('partition must cover every depth exactly once')
        canonical = tuple(sorted((tuple(sorted(group)) for group in groups), key=lambda group: group[0]))
        if groups != canonical:
            raise ValueError('partition groups must be canonicalized')
        object.__setattr__(self, 'groups', groups)

    @property
    def reuse_map(self) -> tuple[int, ...]:
        mapping = {depth: owner for owner, group in enumerate(self.groups) for depth in group}
        return tuple(mapping[depth] for depth in range(self.depth))


@dataclass(frozen=True)
class OwnerRecord:
    name: str
    shape: tuple[int, ...]
    numel: int
    trainable: bool
    category: str
    use_sites: tuple[str, ...]


@dataclass(frozen=True)
class BankStateRef:
    """Hashed tiny bank fixture requiring an external base; not a deployment bundle."""
    directory: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.directory, str) or not self.directory:
            raise ValueError('state directory required')
        if not isinstance(self.manifest_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.manifest_sha256):
            raise ValueError('bank state manifest requires SHA-256')


@dataclass(frozen=True)
class MetricSums:
    """Unnormalized token sums, aggregated on CPU in FP64; raw KL is preserved.

    A skipped batch is not an empty-population evaluation result: evaluation
    callers must reject an aggregate with zero valid targets.
    """
    kl: float = 0.0
    nll: float = 0.0
    valid_targets: int = 0
    disagreements: int = 0
    status: str = 'ok'
    reason: str | None = None

    def __post_init__(self) -> None:
        for name in ('kl', 'nll'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('valid_targets', 'disagreements'):
            _number(name, getattr(self, name), integer=True)
        if self.disagreements > self.valid_targets:
            raise ValueError('disagreements cannot exceed valid targets')
        if self.status not in ('ok', 'skipped'):
            raise ValueError('unknown metric status')
        if self.status == 'skipped':
            if self.reason != 'no_valid_targets' or self.valid_targets or self.kl or self.nll or self.disagreements:
                raise ValueError('skipped metrics require no_valid_targets and empty sums')
        elif self.reason is not None:
            raise ValueError('successful metrics cannot have a skip reason')

    def __add__(self, other: 'MetricSums') -> 'MetricSums':
        if not isinstance(other, MetricSums):
            return NotImplemented
        count = self.valid_targets + other.valid_targets
        return MetricSums(math.fsum((self.kl, other.kl)), math.fsum((self.nll, other.nll)), count,
                          self.disagreements + other.disagreements,
                          'ok' if count else 'skipped', None if count else 'no_valid_targets')


def _optional_reason(name, value):
    if value is not None and (type(value) is not str or not value.strip()):
        raise ValueError(f'{name} must be a nonempty reason or null')


@dataclass(frozen=True)
class RuntimeSpec:
    """Export-only local inputs. Paths never enter the artifact's identity."""
    tokenizer_files: Mapping
    license_files: Mapping
    dependency_lock: str
    install_bytes: int | None = None
    install_unknown_reason: str | None = 'installed environment not measured'

    def __post_init__(self):
        from pathlib import PurePosixPath
        for name in ('tokenizer_files', 'license_files'):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping) or not mapping:
                raise ValueError(f'{name} must explicitly supply files')
            for filename, path in mapping.items():
                if (type(filename) is not str or PurePosixPath(filename).name != filename
                        or filename in ('.', '..') or '\\' in filename or not isinstance(path, str) or not path):
                    raise ValueError('export inputs require simple file names and local paths')
            object.__setattr__(self, name, freeze(mapping))
        if ('tokenizer.json' not in self.tokenizer_files or type(self.dependency_lock) is not str
                or not self.dependency_lock.strip()):
            raise ValueError('a self-contained tokenizer.json and dependency lock are required')
        _optional_reason('install_unknown_reason',self.install_unknown_reason)
        if self.install_bytes is not None: _number('install_bytes', self.install_bytes, integer=True)
        elif not self.install_unknown_reason: raise ValueError('unknown installation size requires a reason')


@dataclass(frozen=True)
class BundleRef:
    directory: str
    manifest_sha256: str

    def __post_init__(self):
        if (type(self.directory) is not str or not self.directory or type(self.manifest_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.manifest_sha256)):
            raise ValueError('bundle requires a directory and detached manifest SHA-256')


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int
    file_entries: Mapping
    metadata: Mapping

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported bundle schema')
        object.__setattr__(self, 'file_entries', freeze(self.file_entries))
        object.__setattr__(self, 'metadata', freeze(self.metadata))


@dataclass(frozen=True)
class ByteAccounting:
    payload_bytes: int
    archive_bytes: int | None = None
    archive_unknown_reason: str | None = 'archive not measured'
    install_bytes: int | None = None
    install_unknown_reason: str | None = 'installation not measured'
    resident_bytes: int | None = None
    resident_unknown_reason: str | None = 'no materialized decoder supplied'
    owner_count: int | None = None
    phase_peaks: Mapping | None = None

    def __post_init__(self):
        _number('payload_bytes', self.payload_bytes, integer=True)
        for name in ('archive', 'install', 'resident'):
            _optional_reason(name+'_unknown_reason',getattr(self,name+'_unknown_reason'))
            value = getattr(self, name + '_bytes')
            if value is None:
                if not getattr(self, name + '_unknown_reason'): raise ValueError(f'unknown {name} bytes need reason')
            else: _number(name + '_bytes', value, integer=True)
        if self.owner_count is not None: _number('owner_count', self.owner_count, integer=True)
        peaks = self.phase_peaks or {'load': {'bytes': None, 'reason': 'not measured'},
                                    'inference': {'bytes': None, 'reason': 'not measured'}}
        if not isinstance(peaks,Mapping): raise ValueError('phase peaks must be a mapping')
        for value in peaks.values():
            if not isinstance(value,Mapping): raise ValueError('phase peak must record bytes and reason')
            _optional_reason('phase reason',value.get('reason'))
            if value.get('bytes') is None:
                if not value.get('reason'): raise ValueError('unknown phase peak requires reason')
            else: _number('phase peak', value['bytes'], integer=True)
        object.__setattr__(self, 'phase_peaks', freeze(peaks))


@dataclass(frozen=True)
class ArchiveRecord:
    path: str
    archive_bytes: int
    sha256: str
    compressor: str
    wall_seconds: float
    temporary_disk_peak_bytes: int
    unpack_seconds: float | None = None
    unpack_unknown_reason: str | None = 'unpack not measured'

    def __post_init__(self):
        _number('archive_bytes', self.archive_bytes, integer=True)
        _number('wall_seconds', self.wall_seconds)
        _number('temporary_disk_peak_bytes', self.temporary_disk_peak_bytes, integer=True)
        if type(self.sha256) is not str or not re.fullmatch('[0-9a-f]{64}', self.sha256): raise ValueError('archive requires SHA-256')
        if type(self.path) is not str or not self.path or type(self.compressor) is not str or not self.compressor.strip():
            raise ValueError('archive requires a path and explicit compressor identity')
        _optional_reason('unpack_unknown_reason',self.unpack_unknown_reason)
        if self.unpack_seconds is not None: _number('unpack_seconds', self.unpack_seconds)
        elif not self.unpack_unknown_reason: raise ValueError('unknown unpack time requires reason')


@runtime_checkable
class Recorder(Protocol):
    """Preflight and append actual expenditure, including failed operations."""
    def check(self, next_cost: CostRecord) -> None: ...
    def record(self, actual_cost: CostRecord) -> None: ...

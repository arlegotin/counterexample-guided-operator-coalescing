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

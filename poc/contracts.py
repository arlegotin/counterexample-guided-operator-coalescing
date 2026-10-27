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

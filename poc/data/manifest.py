"""Small metadata contracts; original text and tokens live only in ignored storage."""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

from poc.records import canonical_json, locked_ledger, read_rows, write_row, timestamp


class FrozenDict(dict):
    """JSON/asdict-compatible immutable mapping snapshot."""
    def _deny(self, *args, **kwargs):
        raise TypeError('manifest mappings are immutable')
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny
    def __deepcopy__(self, memo):
        return self


def _snapshot(value):
    if isinstance(value, dict): return FrozenDict({key:_snapshot(item) for key,item in value.items()})
    if isinstance(value, (tuple,list)): return tuple(_snapshot(item) for item in value)
    return value


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source: str
    revision: str
    shard: str
    row: int
    text: str | None = None
    text_path: str | None = None
    text_sha256: str | None = None
    url: str | None = None
    article_id: str | None = None
    source_record_id: str | None = None
    row_group: int | None = None
    local_source_sha256: str | None = None
    token_path: str | None = None
    token_sha256: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'metadata', _snapshot(self.metadata))

    def read_text(self) -> str:
        value = self.text if self.text is not None else Path(self.text_path).read_bytes().decode('utf-8')
        if self.text_sha256 is not None and hashlib.sha256(value.encode()).hexdigest() != self.text_sha256:
            raise ValueError('document text hash mismatch')
        return value


@dataclass(frozen=True)
class ComponentManifest:
    records: tuple[DocumentRecord, ...]
    membership: dict[str, str]
    near_edges: tuple[tuple[str, str], ...]
    policy: dict
    candidate_pairs: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'membership', _snapshot(self.membership))
        object.__setattr__(self, 'policy', _snapshot(self.policy))


@dataclass(frozen=True)
class DataManifest:
    records: tuple[DocumentRecord, ...]
    roles: dict[str, str]
    membership: dict[str, str]
    salt: str
    exposure_path: str | None = None
    lineage_sha256: str | None = None
    quarantine: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ('roles','membership','quarantine'):
            object.__setattr__(self, name, _snapshot(getattr(self,name)))
        ids = [r.document_id for r in self.records]
        if len(ids) != len(set(ids)) or set(ids) != set(self.roles) or set(ids) != set(self.membership):
            raise ValueError('manifest record/role/component IDs must match uniquely')
        component_roles = {}
        for i in ids:
            c = self.membership[i]
            if c in component_roles and component_roles[c] != self.roles[i]:
                raise ValueError('a component cannot cross roles')
            component_roles[c] = self.roles[i]

    def for_roles(self, *roles: str):
        selected = tuple(r for r in self.records if self.roles[r.document_id] in roles)
        ids = {r.document_id for r in selected}
        return replace(self, records=selected, roles={i:self.roles[i] for i in ids},
                       membership={i:self.membership[i] for i in ids},
                       quarantine={i:v for i,v in self.quarantine.items() if i in ids})

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


class ExposureError(ValueError):
    """A rebuild would promote previously exposed content into protection."""


class ExposureLedger:
    """Append-only lineage snapshots, including dedup fingerprints, not content.

    Rebuilds after access fail closed unless all prospective protected documents
    are checked against every prior exposure, regardless of record/component IDs.
    Preserve this file across pool additions; deletion is never a reset protocol.
    """
    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, manifest: DataManifest, ids: Iterable[str], *, purpose: str):
        from poc.data.dedup import fingerprint
        selected = set(ids)
        exposed_components = {manifest.membership[i] for i in selected}
        rows = [r for r in manifest.records if manifest.membership[r.document_id] in exposed_components]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with locked_ledger(self.path) as stream:
            existing = read_rows(stream)
            for r in rows:
                write_row(stream, {'event_id':len(existing)+1, 'timestamp':timestamp(), 'purpose':purpose,
                                   'manifest_sha256':manifest.digest(), 'document_id':r.document_id,
                                   'component_id':manifest.membership[r.document_id], **fingerprint(r)})
                existing.append({})

    def validate_rebuild(self, manifest: DataManifest, policy):
        from poc.data.dedup import fingerprint, fingerprint_overlap
        if not self.path.exists():
            return
        with locked_ledger(self.path) as stream:
            exposures = read_rows(stream)
        # Compare exact shingle intersections, not approximate MinHash screening.
        for r in manifest.records:
            if manifest.roles[r.document_id] in {'confirm_A','confirm_B','protected_benchmark'}:
                fp = fingerprint(r)
                if any(r.document_id == prior['document_id'] or manifest.membership[r.document_id] == prior['component_id'] or
                       fingerprint_overlap(fp, prior, policy.threshold) for prior in exposures):
                    raise ExposureError('previous exploratory exposure overlaps a prospective protected component')


def load_manifest(path: Path, *, expected_sha256: str, roles: set[str]) -> DataManifest:
    """Verify an immutable metadata manifest and return only explicitly named roles.

    Callers receive this restricted object, never a root directory. Reading text
    remains exclusively through open_records and its exposure/permit checks.
    """
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('data manifest hash mismatch')
    raw = json.loads(data)
    manifest = DataManifest(records=tuple(DocumentRecord(**r) for r in raw.pop('records')), **raw)
    return manifest.for_roles(*roles)

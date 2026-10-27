"""Canonical identities, immutable attempt directories and durable JSONL ledgers."""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from poc.contracts import RunSpec, freeze, thaw


class RecordError(ValueError):
    """An audit record is corrupt or violates its append-only contract."""


def canonical_json(value: Mapping) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("canonical_json requires a mapping")
    return (json.dumps(thaw(freeze(value)), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def run_id(spec: RunSpec) -> str:
    return hashlib.sha256(canonical_json(spec.to_dict())).hexdigest()


@dataclass
class AttemptRecord:
    run_id: str
    attempt_id: str
    path: Path
    spec: RunSpec
    status: str = "planned"


@contextmanager
def locked_ledger(path: Path):
    """Cross-process advisory lock; inspect and append under the same lock."""
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_rows(stream, *, first_event_id: int = 1) -> list[dict]:
    data = stream.read()
    if data and not data.endswith(b"\n"):
        raise RecordError("truncated ledger tail; retained for diagnosis")
    try:
        rows = [json.loads(line) for line in data.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise RecordError("invalid ledger JSON; retained for diagnosis") from exc
    if any(not isinstance(row, dict) or type(row.get("event_id")) is not int or row["event_id"] != index
           for index, row in enumerate(rows, first_event_id)):
        raise RecordError("nonmonotonic ledger event IDs")
    return rows


def write_row(stream, row: Mapping) -> bytes:
    encoded = canonical_json(row)
    stream.seek(0, os.SEEK_END)
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
    return encoded


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_attempt(spec: RunSpec, root: Path) -> AttemptRecord:
    identity = run_id(spec)
    run_path = Path(root).resolve() / identity
    run_path.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(spec.to_dict())
    spec_path = run_path / "spec.json"
    # Exclusive creation prevents two retries from replacing the identity manifest.
    with locked_ledger(run_path / ".identity.lock"):
        if spec_path.exists():
            if spec_path.read_bytes() != encoded:
                raise RecordError("run identity manifest mismatch")
        else:
            with spec_path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
    attempt_id = uuid4().hex
    path = run_path / "attempts" / attempt_id
    path.mkdir(parents=True, exist_ok=False)
    attempt = AttemptRecord(identity, attempt_id, path, spec)
    record = {"schema_version": 1, "run_id": identity, "attempt_id": attempt_id,
              "status": "planned", "spec": spec.to_dict(), "created_at": timestamp(),
              "output_manifest_sha256": None, "failure_category": None, "failure_details": None}
    with (path / "run.json").open("xb") as stream:
        stream.write(canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    (path / "events.jsonl").touch(exist_ok=False)
    return attempt


def append_event(attempt: AttemptRecord, event: Mapping) -> None:
    if not isinstance(event, Mapping) or any(key in event for key in ("event_id", "timestamp", "run_id", "attempt_id")):
        raise RecordError("event contains reserved ledger fields")
    if not isinstance(event.get("kind"), str) or not event["kind"].strip():
        raise RecordError("event requires a nonempty kind")
    payload = thaw(freeze(event))
    with locked_ledger(attempt.path / "events.jsonl") as stream:
        rows = read_rows(stream)
        write_row(stream, {**payload, "event_id": len(rows) + 1, "timestamp": timestamp(),
                           "run_id": attempt.run_id, "attempt_id": attempt.attempt_id})

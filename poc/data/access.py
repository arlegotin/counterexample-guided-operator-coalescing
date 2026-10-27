"""Preflight role denial before any text I/O. Final permit issuance is Task22."""
from dataclasses import replace
from poc.data.manifest import DataManifest, ExposureLedger


class AccessDenied(PermissionError):
    pass


class EvaluationPermit:
    """No public constructor/issuer until immutable freeze verification exists."""
    def __new__(cls, *args, **kwargs):
        raise TypeError('evaluation permits require the future freeze-verifying issuer')


_ALLOWED = {'fit':{'fit'}, 'calibration':{'fit'}, 'counterexample':{'fit','probe_fit'},
            'ranking':{'probe_score'}, 'selection':{'dev'}}


def open_records(manifest: DataManifest, *, purpose: str, permit: EvaluationPermit | None = None):
    allowed = _ALLOWED.get(purpose)
    if allowed is None or any(role not in allowed for role in manifest.roles.values()):
        raise AccessDenied('purpose denied by role matrix; protected readers require verified freeze issuer')
    # Never allow exploratory access without preserving immutable exposure lineage.
    if manifest.records and manifest.exposure_path is None:
        raise AccessDenied('exploratory reader requires an append-only exposure ledger')
    materialized = tuple(replace(r,text=r.read_text()) for r in manifest.records)
    if materialized:
        ExposureLedger(manifest.exposure_path).record(replace(manifest,records=materialized),
                                                      [r.document_id for r in materialized],purpose=purpose)
    yield from materialized

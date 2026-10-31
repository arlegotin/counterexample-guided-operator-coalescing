"""Preflight role denial; confirmation requires a verified freeze and durable grant."""
from dataclasses import replace
from poc.data.manifest import DataManifest, ExposureLedger


class AccessDenied(PermissionError):
    pass


class EvaluationPermit:
    """Opaque immutable capability; only the freeze-verifying issuer can construct it."""
    __slots__ = ("_grant_path", "_grant_sha256", "_nonce", "_grant_inode", "_journal_inode")

    def __new__(cls, *args, **kwargs):
        raise TypeError("evaluation permits require the freeze-verifying issuer")

    def __setattr__(self, name, value):
        raise AttributeError("evaluation permits are immutable")


_ALLOWED = {'fit':{'fit'}, 'calibration':{'fit'}, 'counterexample':{'fit','probe_fit'},
            'ranking':{'probe_score'}, 'selection':{'dev'}}


def open_records(manifest: DataManifest, *, purpose: str, permit: EvaluationPermit | None = None,
                 evaluation=None):
    if purpose == "confirmation":
        from poc.data.confirmation import open_confirmation_records
        yield from open_confirmation_records(manifest, permit=permit, evaluation=evaluation)
        return
    allowed = _ALLOWED.get(purpose)
    if allowed is None or any(role not in allowed for role in manifest.roles.values()):
        raise AccessDenied('purpose denied by role matrix; protected readers require verified freeze issuer')
    # Never allow exploratory access without preserving immutable exposure lineage.
    if manifest.records and manifest.exposure_path is None:
        raise AccessDenied('exploratory reader requires an append-only exposure ledger')
    if manifest.records:
        intent_digest, intent_id = ExposureLedger(manifest.exposure_path).intent(manifest, purpose=purpose)
    materialized = tuple(replace(r,text=r.read_text()) for r in manifest.records)
    if materialized:
        ExposureLedger(manifest.exposure_path).record(replace(manifest,records=materialized),
                                                      [r.document_id for r in materialized],purpose=purpose,
                                                      intent_manifest_sha256=intent_digest,intent_id=intent_id)
    yield from materialized

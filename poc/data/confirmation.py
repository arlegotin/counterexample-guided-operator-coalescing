"""Versioned local freeze/consumption gate; this confers no external data permission.

Only the new-study confirmation path is implemented. Operators must preserve the
exclusive grant directory and the population exposure ledger across interruptions.
Deliberate deletion of all audit state is outside this local workflow contract.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from poc.data.access import AccessDenied, EvaluationPermit
from poc.data.manifest import DataManifest, ExposureLedger
from poc.records import canonical_json, read_rows, timestamp, write_row

_HEX = re.compile(r'[0-9a-f]{64}\Z')
_ROLES = {'confirm_A', 'confirm_B'}
_REF = {'path', 'sha256'}
_FREEZE = {'schema_version','study','purpose','population_sha256','population_roles',
           'exposure_ledger','protocol','analysis','method_artifacts'}
_AUTH = {'schema_version','study','purpose','population_sha256','population_roles',
         'exposure_ledger','freeze','permit_directory','population_origin'}


def _deny(message):
    raise AccessDenied(message)


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        _deny('expected exact lowercase SHA256 identity')
    return value


def _path(value, *, directory=False, missing=False):
    if not isinstance(value, str) or not value or '\x00' in value:
        _deny('expected canonical absolute path')
    p = Path(value)
    if not p.is_absolute() or str(p) != value or p.resolve() != p or '..' in p.parts:
        _deny('path aliases and noncanonical paths are denied')
    if missing:
        if p.exists() or p.is_symlink():
            _deny('permit directory already exists; issuance cannot reset consumption')
        if not p.parent.is_dir():
            _deny('permit parent directory must already exist')
        return p
    try:
        s = p.lstat()
    except OSError as exc:
        raise AccessDenied('required audit or frozen file is missing') from exc
    if directory:
        if not stat.S_ISDIR(s.st_mode): _deny('expected real directory')
    elif not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        _deny('frozen and audit files must be regular single-link files')
    return p


def _read_ref(reference):
    if type(reference) is not dict or set(reference) != _REF:
        _deny('invalid closed file reference')
    _sha(reference['sha256'])
    p = _path(reference['path'])
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        _deny('frozen file hash mismatch')
    return data


def _object(reference, keys):
    try:
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj: _deny('duplicate metadata key')
                obj[key] = value
            return obj
        value = json.loads(_read_ref(reference), object_pairs_hook=unique)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise AccessDenied('invalid frozen JSON metadata') from exc
    if type(value) is not dict or set(value) != keys or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        _deny('unknown or incomplete confirmation schema')
    return value


@dataclass(frozen=True)
class EvaluationContext:
    """Caller-declared identities of the actual frozen evaluation invocation."""
    study: str
    protocol_sha256: str
    analysis_sha256: str
    method_artifacts: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.study, str) or not self.study:
            _deny('study identity required')
        _sha(self.protocol_sha256); _sha(self.analysis_sha256)
        pairs = tuple(tuple(pair) for pair in self.method_artifacts)
        if not pairs or any(len(pair) != 2 for pair in pairs): _deny('method/artifact identities required')
        for method, artifact in pairs: _sha(method); _sha(artifact)
        if len(set(pair[0] for pair in pairs)) != len(pairs): _deny('duplicate method identity')
        object.__setattr__(self, 'method_artifacts', tuple(sorted(pairs)))


def _metadata_paths_only(manifest, references):
    """Deny known population payloads before opening any metadata reference."""
    if not isinstance(manifest, DataManifest) or not manifest.records:
        _deny('nonempty exact population required')
    payloads = []
    for record in manifest.records:
        for value in (record.text_path, record.token_path):
            if value is not None:
                payloads.append(Path(value).resolve())
    for reference in references:
        if type(reference) is not dict or set(reference) != _REF:
            _deny('invalid closed metadata reference')
        candidate = _path(reference['path'])
        for payload in payloads:
            if candidate == payload or (payload.exists() and _inode(candidate) == _inode(payload)):
                _deny('frozen metadata overlaps population content')


def _bindings(authorization, freeze, manifest):
    _metadata_paths_only(manifest, (authorization, freeze))
    auth = _object(authorization, _AUTH)
    frozen = _object(freeze, _FREEZE)
    if auth['freeze'] != freeze or auth['population_origin'] != 'new_confirmation':
        _deny('authorization does not bind this new confirmation freeze')
    for key in ('study','purpose','population_sha256','population_roles','exposure_ledger'):
        if auth[key] != frozen[key]: _deny('authorization/freeze binding mismatch')
    if not isinstance(frozen['study'], str) or not frozen['study'] or frozen['purpose'] != 'confirmation':
        _deny('new named study and exact confirmation purpose required')
    roles = frozen['population_roles']
    if (type(roles) is not list or not roles or any(type(r) is not str for r in roles)
            or roles != sorted(set(roles)) or not set(roles) <= _ROLES):
        _deny('only explicitly authorized new confirmation roles are eligible')
    if not isinstance(manifest, DataManifest) or not manifest.records:
        _deny('nonempty exact population required')
    if _sha(frozen['population_sha256']) != manifest.digest() or set(manifest.roles.values()) != set(roles):
        _deny('exact population/role identity mismatch')
    if manifest.exposure_path != frozen['exposure_ledger']:
        _deny('population exposure ledger binding mismatch')
    for record in manifest.records:
        _sha(record.text_sha256)
    items = frozen['method_artifacts']
    if type(items) is not list or not items: _deny('frozen method/artifact set required')
    references = [frozen['protocol'], frozen['analysis']]
    for item in items:
        if type(item) is not dict or set(item) != {'method_id','artifact_id','manifest'}:
            _deny('unknown method/artifact schema')
        references.append(item['manifest'])
    _metadata_paths_only(manifest, references)
    _read_ref(frozen['protocol']); _read_ref(frozen['analysis'])
    pairs = []
    for item in items:
        if type(item) is not dict or set(item) != {'method_id','artifact_id','manifest'}:
            _deny('unknown method/artifact schema')
        _sha(item['method_id']); _sha(item['artifact_id'])
        _read_ref(item['manifest'])
        if item['artifact_id'] != item['manifest']['sha256']:
            _deny('artifact ID must equal its frozen manifest hash')
        pairs.append((item['method_id'],item['artifact_id']))
    context = EvaluationContext(frozen['study'], frozen['protocol']['sha256'],
                                frozen['analysis']['sha256'], tuple(pairs))
    _path(frozen['exposure_ledger'])
    return auth, frozen, context


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _write_new(path, value):
    with path.open('xb') as f:
        f.write(canonical_json(value)); f.flush(); os.fsync(f.fileno())
    _fsync_directory(path.parent)


def _inode(path):
    s = path.stat()
    return [s.st_dev, s.st_ino]


@contextmanager
def _existing_lock(path, *, expected_inode=None):
    _path(str(path))
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(fd, 'r+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                s = os.fstat(stream.fileno())
                if (s.st_nlink != 1 or [s.st_dev,s.st_ino] != _inode(_path(str(path)))
                        or expected_inode is not None and [s.st_dev,s.st_ino] != list(expected_inode)):
                    _deny('audit file changed while claiming access')
                stream.seek(0)
                yield stream
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise AccessDenied('audit journal unavailable; access cannot be reset') from exc


def issue_evaluation_permit(*, authorization: dict, freeze: dict, manifest: DataManifest) -> EvaluationPermit:
    """Verify metadata only and issue one nonreusable local evaluation capability.

    A supplied authorization is an input from the authorized confirmation workflow,
    never evidence of platform permission. No text is read by this function.
    """
    auth, frozen, _ = _bindings(authorization, freeze, manifest)
    exposure = _path(frozen['exposure_ledger'])
    with _existing_lock(exposure) as f:
        if f.read(): _deny('population already exposed or has incomplete access')
        exposure_inode = _inode(exposure)
    directory = _path(auth['permit_directory'], missing=True)
    metadata_paths = {authorization['path'],freeze['path'],frozen['protocol']['path'],frozen['analysis']['path']}
    metadata_paths.update(item['manifest']['path'] for item in frozen['method_artifacts'])
    if str(exposure) in metadata_paths or any(Path(p).is_relative_to(directory) for p in metadata_paths):
        _deny('audit outputs overlap frozen inputs')
    try: directory.mkdir(mode=0o700)
    except OSError as exc: raise AccessDenied('exclusive issuance failed; retained state cannot be overwritten') from exc
    _fsync_directory(directory.parent)
    # Create the journal inode before binding it into the immutable grant.
    journal = directory/'journal.jsonl'
    with journal.open('xb') as f:
        f.flush(); os.fsync(f.fileno())
    _fsync_directory(directory)
    journal_inode = _inode(journal)
    grant = {'schema_version':1, 'authorization':authorization, 'freeze':freeze,
             'population_sha256':manifest.digest(), 'exposure_inode':exposure_inode,
             'journal_inode':journal_inode, 'nonce':uuid4().hex}
    path = directory/'grant.json'
    _write_new(path, grant)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with _existing_lock(journal, expected_inode=journal_inode) as f:
        write_row(f, {'event_id':1,'timestamp':timestamp(),'event':'issued',
                      'grant_sha256':digest})
    permit = object.__new__(EvaluationPermit)
    object.__setattr__(permit, '_grant_path', str(path))
    object.__setattr__(permit, '_grant_sha256', digest)
    object.__setattr__(permit, '_nonce', grant['nonce'])
    object.__setattr__(permit, '_grant_inode', tuple(_inode(path)))
    object.__setattr__(permit, '_journal_inode', tuple(journal_inode))
    return permit


def _admit(manifest, permit, evaluation):
    if type(permit) is not EvaluationPermit or type(evaluation) is not EvaluationContext:
        _deny('confirmation requires issued permit and explicit frozen caller context')
    try:
        reference = {'path':permit._grant_path, 'sha256':permit._grant_sha256}
        grant_inode = permit._grant_inode
        journal_inode = permit._journal_inode
    except AttributeError as exc:
        raise AccessDenied('unissued permit') from exc
    _metadata_paths_only(manifest, (reference,))
    if tuple(_inode(_path(permit._grant_path))) != grant_inode:
        _deny('issued grant file was replaced')
    grant = _object(reference, {'schema_version','authorization','freeze','population_sha256','exposure_inode','journal_inode','nonce'})
    if tuple(grant['journal_inode']) != journal_inode:
        _deny('issued journal binding differs')
    if grant['nonce'] != permit._nonce or grant['population_sha256'] != manifest.digest():
        _deny('permit population/nonce mismatch')
    auth, frozen, expected = _bindings(grant['authorization'], grant['freeze'], manifest)
    if evaluation != expected or str(Path(permit._grant_path).parent) != auth['permit_directory']:
        _deny('caller does not match frozen study/analysis/method/artifact identities')
    exposure = _path(frozen['exposure_ledger'])
    if _inode(exposure) != grant['exposure_inode']:
        _deny('population audit ledger was replaced')
    _path(auth['permit_directory'], directory=True)
    journal = Path(auth['permit_directory'])/'journal.jsonl'
    if tuple(_inode(_path(str(journal)))) != journal_inode:
        _deny('issued consumption journal was replaced')
    return grant, journal, exposure


def open_confirmation_records(manifest, *, permit, evaluation):
    """Materialize one whole protected population after durable fail-closed claims."""
    grant, journal, exposure = _admit(manifest, permit, evaluation)
    with _existing_lock(journal, expected_inode=permit._journal_inode) as stream:
        try: events = read_rows(stream)
        except ValueError as exc: raise AccessDenied('corrupt consumption journal') from exc
        if (len(events) != 1 or events[0].get('event') != 'issued'
                or events[0].get('grant_sha256') != permit._grant_sha256):
            _deny('confirmation grant is consumed, interrupted or corrupt')
        _admit(manifest, permit, evaluation)
        intent_id = uuid4().hex
        try:
            # Population-wide durable intent comes first. An abrupt exit after
            # any claim row blocks every grant, even before this grant starts.
            with _existing_lock(exposure, expected_inode=grant['exposure_inode']) as exposure_stream:
                if exposure_stream.read(): _deny('population already consumed through another grant or reader')
                # Append under the SAME population lock as the fresh-state check.
                for index, record in enumerate(manifest.records, 1):
                    write_row(exposure_stream, {'event_id':index,'timestamp':timestamp(),
                        'event':'exposure_intent','purpose':'confirmation','intent_id':intent_id,
                        'manifest_sha256':manifest.digest(),'document_id':record.document_id,
                        'component_id':manifest.membership[record.document_id],
                        'confirmation_grant_sha256':permit._grant_sha256})
            write_row(stream, {'event_id':2,'timestamp':timestamp(),'event':'started',
                               'grant_sha256':permit._grant_sha256,'intent_id':intent_id})
            _admit(manifest, permit, evaluation)
            materialized = tuple(replace(r, text=r.read_text()) for r in manifest.records)
            # A change detected after I/O yields no text and still consumes the attempt.
            _admit(manifest, permit, evaluation)
            ExposureLedger(exposure).record(replace(manifest, records=materialized),
                [r.document_id for r in materialized], purpose='confirmation',
                intent_manifest_sha256=manifest.digest(), intent_id=intent_id)
        except BaseException as exc:
            stream.seek(0)
            failed_event_id = len(read_rows(stream)) + 1
            write_row(stream, {'event_id':failed_event_id,'timestamp':timestamp(),'event':'failed',
                               'grant_sha256':permit._grant_sha256,'intent_id':intent_id,
                               'exception_type':type(exc).__name__})
            raise
        write_row(stream, {'event_id':3,'timestamp':timestamp(),'event':'completed',
                           'grant_sha256':permit._grant_sha256,'intent_id':intent_id,
                           'documents':len(materialized),'components':len(set(manifest.membership.values()))})
    yield from materialized

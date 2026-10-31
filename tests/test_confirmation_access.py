"""Owned fixtures only: no real protected population is opened by these tests."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from poc.data.access import AccessDenied, EvaluationPermit, open_records
from poc.data.manifest import DataManifest, DocumentRecord
from poc.records import canonical_json


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, data):
    path.write_bytes(canonical_json(data))
    return ref(path)


@pytest.fixture
def case(tmp_path):
    # Import within fixture so missing implementation is an actual RED.
    from poc.data.confirmation import EvaluationContext, issue_evaluation_permit
    text = tmp_path / 'owned.txt'
    text.write_text('Owned fixture for frozen access.\n')
    exposure = tmp_path / 'exposure.jsonl'
    exposure.touch()
    r = DocumentRecord('owned-a', 'owned', 'a'*40, 'fixture', 0,
                       text_path=str(text), text_sha256=ref(text)['sha256'])
    manifest = DataManifest((r,), {'owned-a': 'confirm_A'}, {'owned-a': 'owned-component'},
                            'new-owned-study', exposure_path=str(exposure))
    protocol = tmp_path / 'protocol.json'
    analysis = tmp_path / 'analysis.py'
    artifact = tmp_path / 'artifact-manifest.json'
    save(protocol, {'primary_contrast': 'owned fixture only', 'study': 'owned-confirmation-v1'})
    analysis.write_text('result = "owned fixture only"\n')
    save(artifact, {'scope': 'owned fixture; no model'})
    freeze_data = {'schema_version': 1, 'study': 'owned-confirmation-v1',
                   'purpose': 'confirmation', 'population_sha256': manifest.digest(),
                   'population_roles': ['confirm_A'], 'exposure_ledger': str(exposure),
                   'protocol': ref(protocol), 'analysis': ref(analysis),
                   'method_artifacts': [{'method_id': 'b'*64,
                                         'artifact_id': ref(artifact)['sha256'],
                                         'manifest': ref(artifact)}]}
    frozen = save(tmp_path / 'freeze.json', freeze_data)
    auth_data = {'schema_version': 1, 'study': 'owned-confirmation-v1',
                 'purpose': 'confirmation', 'population_sha256': manifest.digest(),
                 'population_roles': ['confirm_A'], 'exposure_ledger': str(exposure),
                 'freeze': frozen, 'permit_directory': str(tmp_path / 'grant'),
                 'population_origin': 'new_confirmation'}
    authorization = save(tmp_path / 'authorization.json', auth_data)
    context = EvaluationContext(study=freeze_data['study'],
                               protocol_sha256=ref(protocol)['sha256'],
                               analysis_sha256=ref(analysis)['sha256'],
                               method_artifacts=(('b'*64, ref(artifact)['sha256']),))
    return dict(locals())


def issue(c):
    return c['issue_evaluation_permit'](authorization=c['authorization'],
                                       freeze=c['frozen'], manifest=c['manifest'])


def read(c, permit, **kwargs):
    return list(open_records(c['manifest'], purpose='confirmation', permit=permit,
                             evaluation=kwargs.get('evaluation', c['context'])))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_owned_frozen_access_succeeds_once_and_preserves_lineage(case):
    c = case; p = issue(c)
    assert read(c, p)[0].text == c['text'].read_text()
    events = rows(c['tmp_path'] / 'grant' / 'journal.jsonl')
    assert [r['event'] for r in events] == ['issued', 'started', 'completed']
    exposed = rows(c['exposure'])
    assert exposed[0]['event'] == 'exposure_intent'
    assert exposed[1]['normalized_sha256']
    with pytest.raises(AccessDenied): read(c, p)
    with pytest.raises(AccessDenied): issue(c)
    assert 'Owned fixture' not in (c['tmp_path'] / 'grant' / 'journal.jsonl').read_text()
    with pytest.raises(TypeError): EvaluationPermit()
    with pytest.raises((AttributeError, TypeError)): p._nonce = 'changed'


def test_issue_never_reads_text(case, monkeypatch):
    def forbidden(_): raise AssertionError('issuer accessed protected text')
    monkeypatch.setattr(DocumentRecord, 'read_text', forbidden)
    issue(case)


@pytest.mark.parametrize('field', ['study', 'protocol_sha256', 'analysis_sha256', 'method_artifacts'])
def test_wrong_caller_context_denied_before_io(case, monkeypatch, field):
    c = case; p = issue(c)
    value = (('d'*64, 'e'*64),) if field == 'method_artifacts' else ('other-study' if field == 'study' else 'f'*64)
    ctx = replace(c['context'], **{field: value})
    monkeypatch.setattr(DocumentRecord, 'read_text', lambda _: pytest.fail('unexpected I/O'))
    with pytest.raises((AccessDenied, ValueError)): read(c, p, evaluation=ctx)


@pytest.mark.parametrize('name', ['authorization', 'frozen', 'protocol', 'analysis', 'artifact'])
def test_changed_frozen_file_after_issuance_denied(case, monkeypatch, name):
    c = case; p = issue(c)
    path = Path(c[name]['path']) if name in {'authorization', 'frozen'} else c[name]
    path.write_text('changed\n')
    monkeypatch.setattr(DocumentRecord, 'read_text', lambda _: pytest.fail('unexpected I/O'))
    with pytest.raises(AccessDenied): read(c, p)


@pytest.mark.parametrize('name', ['journal.jsonl', 'grant.json'])
def test_missing_or_corrupt_grant_files_fail_closed(case, monkeypatch, name):
    c = case; p = issue(c)
    (c['tmp_path'] / 'grant' / name).unlink()
    monkeypatch.setattr(DocumentRecord, 'read_text', lambda _: pytest.fail('unexpected I/O'))
    with pytest.raises(AccessDenied): read(c, p)
    with pytest.raises(AccessDenied): issue(c)


def test_corrupted_journal_denied_without_reset(case, monkeypatch):
    c = case; p = issue(c)
    (c['tmp_path'] / 'grant' / 'journal.jsonl').write_text('{"event_id":1}\n')
    monkeypatch.setattr(DocumentRecord, 'read_text', lambda _: pytest.fail('unexpected I/O'))
    with pytest.raises(AccessDenied): read(c, p)


def test_durable_intent_precedes_first_read_and_failure_consumes(case, monkeypatch):
    c = case; p = issue(c)
    def interrupted(_):
        assert [r['event'] for r in rows(c['tmp_path'] / 'grant' / 'journal.jsonl')] == ['issued', 'started']
        assert rows(c['exposure'])[0]['event'] == 'exposure_intent'
        raise KeyboardInterrupt('owned interruption')
    monkeypatch.setattr(DocumentRecord, 'read_text', interrupted)
    with pytest.raises(KeyboardInterrupt): read(c, p)
    assert rows(c['tmp_path'] / 'grant' / 'journal.jsonl')[-1]['event'] == 'failed'
    with pytest.raises(AccessDenied): read(c, p)
    with pytest.raises(AccessDenied): issue(c)


def test_abrupt_interruption_state_cannot_be_reused(case):
    c = case; p = issue(c)
    # A durable started row also blocks reuse if no failure handler ran.
    from poc.records import locked_ledger, write_row, timestamp
    path = c['tmp_path'] / 'grant' / 'journal.jsonl'
    with locked_ledger(path) as f:
        first = rows(path)[0]
        write_row(f, {'event_id': 2, 'timestamp': timestamp(), 'event': 'started',
                      'grant_sha256': first['grant_sha256']})
    with pytest.raises(AccessDenied): read(c, p)


@pytest.mark.parametrize('purpose', ['fit','calibration','counterexample','ranking','selection','analysis','prepare','unknown'])
def test_permit_never_widens_exploratory_purposes(case, purpose, monkeypatch):
    c = case; p = issue(c)
    monkeypatch.setattr(DocumentRecord, 'read_text', lambda _: pytest.fail('unexpected I/O'))
    with pytest.raises(AccessDenied): list(open_records(c['manifest'], purpose=purpose, permit=p,
                                                      evaluation=c['context']))


def test_confirmation_requires_both_permit_and_explicit_context(case):
    c = case; p = issue(c)
    with pytest.raises(AccessDenied): read(c, None)
    with pytest.raises(AccessDenied): read(c, p, evaluation=None)
    with pytest.raises(AccessDenied): read(c, object())


@pytest.mark.parametrize('mutation', ['role','population','ledger','text_hash','subset'])
def test_manifest_bindings_are_exact(case, mutation, monkeypatch):
    c = case; p = issue(c); m = c['manifest']
    if mutation == 'role': m = replace(m, roles={'owned-a':'dev'})
    elif mutation == 'population': m = replace(m, salt='other')
    elif mutation == 'ledger': m = replace(m, exposure_path=str(c['tmp_path']/'other'))
    elif mutation == 'text_hash': m = replace(m, records=(replace(m.records[0], text_sha256='f'*64),))
    else: m = replace(m, records=(), roles={}, membership={})
    monkeypatch.setattr(DocumentRecord, 'read_text', lambda _: pytest.fail('unexpected I/O'))
    with pytest.raises(AccessDenied): list(open_records(m, purpose='confirmation', permit=p,
                                                      evaluation=c['context']))


def test_previously_populated_or_missing_exposure_ledger_denies(case):
    c = case; c['exposure'].write_text('previous exposure\n')
    with pytest.raises(AccessDenied): issue(c)
    c['exposure'].unlink()
    with pytest.raises(AccessDenied): issue(c)


@pytest.mark.parametrize('action', ['delete','replace','symlink','hardlink'])
def test_ledger_replacement_or_alias_after_issue_denies(case, action):
    import os
    c = case; p = issue(c); c['exposure'].unlink()
    other = c['tmp_path'] / 'other-empty'; other.touch()
    if action == 'replace': c['exposure'].touch()
    elif action == 'symlink': c['exposure'].symlink_to(other)
    elif action == 'hardlink': os.link(other, c['exposure'])
    with pytest.raises(AccessDenied): read(c, p)


def test_second_authorization_for_same_population_cannot_read_twice(case):
    c = case; first = issue(c)
    auth = {**c['auth_data'], 'permit_directory': str(c['tmp_path']/'grant-two')}
    second_ref = save(c['tmp_path']/'authorization-two.json', auth)
    second = c['issue_evaluation_permit'](authorization=second_ref, freeze=c['frozen'], manifest=c['manifest'])
    assert len(read(c, first)) == 1
    with pytest.raises(AccessDenied): read(c, second)


@pytest.mark.parametrize('which', ['authorization','freeze'])
def test_unknown_schema_fields_rejected(case, which):
    c = case
    data = c['auth_data'] if which == 'authorization' else c['freeze_data']
    key = 'authorization' if which == 'authorization' else 'frozen'
    c[key] = save(c['tmp_path']/f'bad-{which}.json', {**data, 'extra': True})
    if which == 'freeze':
        c['authorization'] = save(c['tmp_path']/'bad-auth.json', {**c['auth_data'], 'freeze': c['frozen']})
    with pytest.raises(AccessDenied): issue(c)


def test_change_during_materialization_yields_nothing_and_consumes(case, monkeypatch):
    c = case; p = issue(c); original = DocumentRecord.read_text
    def change(record):
        value = original(record)
        c['analysis'].write_text('changed after first read\n')
        return value
    monkeypatch.setattr(DocumentRecord, 'read_text', change)
    with pytest.raises(AccessDenied): read(c, p)
    assert rows(c['tmp_path']/'grant'/'journal.jsonl')[-1]['event'] == 'failed'
    with pytest.raises(AccessDenied): read(c, p)


def test_two_grants_race_on_same_population_only_one_reads(case):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    c = case; first = issue(c)
    second_ref = save(c['tmp_path']/'authorization-two.json',
                      {**c['auth_data'], 'permit_directory': str(c['tmp_path']/'grant-two')})
    second = c['issue_evaluation_permit'](authorization=second_ref, freeze=c['frozen'], manifest=c['manifest'])
    barrier = threading.Barrier(2)
    def attempt(p):
        barrier.wait(timeout=5)
        try: return len(read(c, p))
        except AccessDenied: return 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (first, second)))
    assert sorted(results) == [0, 1]
    assert sum(r.get('event') == 'exposure_intent' for r in rows(c['exposure'])) == 1


@pytest.mark.parametrize('alias', ['symlink','hardlink'])
def test_aliased_frozen_metadata_rejected(case, alias):
    import os
    c = case; target = c['tmp_path']/'alias-authorization.json'
    if alias == 'symlink': target.symlink_to(c['authorization']['path'])
    else: os.link(c['authorization']['path'], target)
    with pytest.raises(AccessDenied):
        c['issue_evaluation_permit'](authorization=ref(target), freeze=c['frozen'], manifest=c['manifest'])


@pytest.mark.parametrize('role', ['fit','dev','probe_score','protected_benchmark'])
def test_relabelled_authorization_cannot_grant_exploratory_or_historical_roles(case, role):
    c = case; c['manifest'] = replace(c['manifest'], roles={'owned-a': role})
    data = {**c['freeze_data'], 'population_roles': [role], 'population_sha256': c['manifest'].digest()}
    c['frozen'] = save(c['tmp_path']/'role-freeze.json', data)
    c['authorization'] = save(c['tmp_path']/'role-authorization.json',
        {**c['auth_data'], 'freeze': c['frozen'], 'population_roles': [role],
         'population_sha256': c['manifest'].digest()})
    with pytest.raises(AccessDenied): issue(c)


def test_missing_text_identity_rejected_at_issuance(case):
    c = case; c['manifest'] = replace(c['manifest'], records=(replace(c['manifest'].records[0], text_sha256=None),))
    c['frozen'] = save(c['tmp_path']/'hashless-freeze.json',
                        {**c['freeze_data'], 'population_sha256': c['manifest'].digest()})
    c['authorization'] = save(c['tmp_path']/'hashless-authorization.json',
        {**c['auth_data'], 'freeze':c['frozen'], 'population_sha256':c['manifest'].digest()})
    with pytest.raises(AccessDenied): issue(c)


def test_context_freezes_mutable_input_lists(case):
    from poc.data.confirmation import EvaluationContext
    c = case; values = [list(c['context'].method_artifacts[0])]
    context = EvaluationContext(c['context'].study, c['context'].protocol_sha256,
                                c['context'].analysis_sha256, values)
    values[0][0] = 'c'*64
    assert context == c['context']


def test_abrupt_exit_after_started_consumes_population_across_grants(case):
    import multiprocessing
    import os
    import poc.data.confirmation as gate
    c = case; first = issue(c)
    second_ref = save(c['tmp_path']/'authorization-two.json',
        {**c['auth_data'], 'permit_directory':str(c['tmp_path']/'grant-two')})
    second = c['issue_evaluation_permit'](authorization=second_ref, freeze=c['frozen'], manifest=c['manifest'])
    def child():
        original = gate.write_row
        def die(stream, row):
            result = original(stream, row)
            if row.get('event') == 'started': os._exit(73)
            return result
        gate.write_row = die
        read(c, first)
        os._exit(74)
    p = multiprocessing.get_context('fork').Process(target=child)
    p.start();p.join(10)
    if p.is_alive():p.kill();p.join();pytest.fail('owned interruption subprocess timed out')
    assert p.exitcode == 73
    with pytest.raises(AccessDenied): read(c, second)


def test_replaced_exposure_after_admission_is_denied_before_any_content(case, monkeypatch):
    from contextlib import contextmanager
    import poc.data.confirmation as gate
    c = case; p = issue(c); original = gate._existing_lock; changed = False
    @contextmanager
    def replacing(path, **kwargs):
        nonlocal changed
        if Path(path) == c['exposure'] and not changed:
            other = c['tmp_path']/'new-exposure';other.touch();other.replace(c['exposure']);changed = True
        with original(path, **kwargs) as stream: yield stream
    monkeypatch.setattr(gate, '_existing_lock', replacing)
    calls=[]
    original_read=DocumentRecord.read_text
    def reading(record):calls.append(record.document_id);return original_read(record)
    monkeypatch.setattr(DocumentRecord,'read_text',reading)
    with pytest.raises(AccessDenied):read(c,p)
    assert calls == []


def test_replaced_exposure_after_started_is_denied_before_any_content(case, monkeypatch):
    import poc.data.confirmation as gate
    c=case;p=issue(c);original=gate.write_row
    def replacing(stream,row):
        result=original(stream,row)
        if row.get('event')=='started':
            other=c['tmp_path']/'new-exposure';other.touch();other.replace(c['exposure'])
        return result
    monkeypatch.setattr(gate,'write_row',replacing)
    calls=[];original_read=DocumentRecord.read_text
    def reading(record):calls.append(record.document_id);return original_read(record)
    monkeypatch.setattr(DocumentRecord,'read_text',reading)
    with pytest.raises(AccessDenied):read(c,p)
    assert calls == []


@pytest.mark.parametrize('field',['protocol','analysis','artifact','authorization','freeze'])
def test_issuer_rejects_content_as_metadata_before_actual_file_read(case,monkeypatch,field):
    c=case;data=dict(c['freeze_data'])
    if field in ('protocol','analysis'):data[field]=ref(c['text'])
    elif field=='artifact':data['method_artifacts']=[{'method_id':'b'*64,'artifact_id':ref(c['text'])['sha256'],'manifest':ref(c['text'])}]
    c['frozen']=save(c['tmp_path']/'content-freeze.json',data)
    c['authorization']=save(c['tmp_path']/'content-auth.json',{**c['auth_data'],'freeze':c['frozen']})
    if field=='freeze':
        c['frozen']=ref(c['text']);c['authorization']=save(c['tmp_path']/'content-auth-two.json',{**c['auth_data'],'freeze':c['frozen']})
    elif field=='authorization':c['authorization']=ref(c['text'])
    original=Path.read_bytes;calls=[]
    def reading(path):
        if path.resolve()==c['text'].resolve():calls.append(str(path))
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',reading)
    with pytest.raises(AccessDenied):issue(c)
    assert calls==[]
    assert c['exposure'].read_bytes()==b''


@pytest.mark.parametrize('name',['grant.json','journal.jsonl'])
def test_byte_identical_audit_file_replacement_denied_before_content(case,monkeypatch,name):
    c=case;p=issue(c);target=c['tmp_path']/'grant'/name
    replacement=c['tmp_path']/'identical';replacement.write_bytes(target.read_bytes());replacement.replace(target)
    calls=[];original=DocumentRecord.read_text
    def reading(record):calls.append(record.document_id);return original(record)
    monkeypatch.setattr(DocumentRecord,'read_text',reading)
    with pytest.raises(AccessDenied):read(c,p)
    assert calls==[]

from dataclasses import replace
import pytest
from poc.data.manifest import DocumentRecord, DataManifest, ExposureLedger
from poc.data.access import open_records, AccessDenied, EvaluationPermit


@pytest.mark.parametrize('purpose', ['fit','calibration','counterexample','ranking','selection','analysis','prepare','unknown'])
def test_all_exploratory_purposes_deny_before_any_text_read(tmp_path, purpose):
    # First entry is allowed to some purposes: preflight the ENTIRE manifest before reading it.
    records = (DocumentRecord('a','owned','a'*40,'fixture',0,text_path=str(tmp_path/'missing')),
               DocumentRecord('b','owned','a'*40,'fixture',1,text_path=str(tmp_path/'protected-missing')))
    m = DataManifest(records, {'a':'fit','b':'confirm_A'}, {'a':'ca','b':'cb'}, 'salt')
    with pytest.raises(AccessDenied): list(open_records(m,purpose=purpose))


def test_reader_returns_original_text_and_records_exposure(tmp_path):
    path=tmp_path/'text'; path.write_text('Original Ａ text\n')
    d=DocumentRecord('a','owned','a'*40,'fixture',0,text_path=str(path))
    ledger=ExposureLedger(tmp_path/'exposure.jsonl')
    m=DataManifest((d,),{'a':'fit'},{'a':'c'},'salt',exposure_path=str(ledger.path))
    result=list(open_records(m,purpose='fit'))
    assert result[0].text == 'Original Ａ text\n'
    assert ledger.path.exists()


def test_no_public_permit_issuer_and_protected_reader_fails_closed(tmp_path):
    d=DocumentRecord('a','owned','a'*40,'fixture',0,text='secret')
    m=DataManifest((d,),{'a':'protected_benchmark'},{'a':'c'},'salt')
    with pytest.raises(AccessDenied): list(open_records(m,purpose='confirmation'))
    with pytest.raises(TypeError): EvaluationPermit()


def test_original_crlf_bytes_survive_reader(tmp_path):
    import hashlib
    data=b'Original\r\nsecond line\r\n'
    path=tmp_path/'text';path.write_bytes(data)
    r=DocumentRecord('a','owned','a'*40,'fixture',0,text_path=str(path),text_sha256=hashlib.sha256(data).hexdigest())
    assert r.read_text().encode()==data


def test_load_manifest_checks_digest_then_delivers_only_named_role(tmp_path):
    import hashlib
    from dataclasses import asdict
    from poc.records import canonical_json
    from poc.data.manifest import load_manifest
    records=(DocumentRecord('a','owned','a'*40,'fixture',0,text='owned'),DocumentRecord('b','owned','a'*40,'fixture',1,text='protected'))
    m=DataManifest(records,{'a':'fit','b':'confirm_A'},{'a':'ca','b':'cb'},'salt')
    path=tmp_path/'manifest.json';data=canonical_json(asdict(m));path.write_bytes(data)
    with pytest.raises(ValueError,match='hash'):load_manifest(path,expected_sha256='0'*64,roles={'fit'})
    selected=load_manifest(path,expected_sha256=hashlib.sha256(data).hexdigest(),roles={'fit'})
    assert [r.document_id for r in selected.records]==['a']
    assert list(selected.roles.values())==['fit']


def test_reader_persists_exposure_intent_before_first_text_io(tmp_path, monkeypatch):
    ledger=tmp_path/'exposure.jsonl'
    d=DocumentRecord('a','owned','a'*40,'fixture',0,text='original')
    m=DataManifest((d,),{'a':'dev'},{'a':'c'},'salt',exposure_path=str(ledger))
    original=DocumentRecord.read_text
    def read(record):
        assert ledger.exists(), 'exposure must survive a crash at first read'
        assert '"document_id":"a"' in ledger.read_text()
        return original(record)
    monkeypatch.setattr(DocumentRecord,'read_text',read)
    assert list(open_records(m,purpose='selection'))[0].text=='original'


def test_interrupted_first_read_intent_blocks_rebuild(tmp_path):
    from poc.data.manifest import ExposureError
    from poc.data.dedup import DedupPolicy
    ledger=tmp_path/'exposures.jsonl'
    r=DocumentRecord('a','owned','a'*40,'fixture',0,text_path=str(tmp_path/'missing'))
    manifest=DataManifest((r,),{'a':'dev'},{'a':'c'},'salt',exposure_path=str(ledger))
    with pytest.raises(FileNotFoundError): list(open_records(manifest,purpose='selection'))
    rebuilt=DataManifest((replace(r,text='different'),),{'a':'confirm_A'},{'a':'c'},'other')
    with pytest.raises(ExposureError,match='incomplete'): ExposureLedger(ledger).validate_rebuild(rebuilt,DedupPolicy())


def test_completed_reader_intent_allows_nonoverlapping_rebuild(tmp_path):
    from poc.data.dedup import DedupPolicy
    ledger=tmp_path/'exposure.jsonl'
    original=tmp_path/'original.txt'; original.write_text('old text')
    r=DocumentRecord('a','owned','a'*40,'fixture',0,text_path=str(original))
    manifest=DataManifest((r,),{'a':'dev'},{'a':'c'},'salt',exposure_path=str(ledger))
    list(open_records(manifest,purpose='selection'))
    other=DocumentRecord('b','owned','a'*40,'fixture',1,text='unrelated')
    rebuilt=DataManifest((other,),{'b':'confirm_A'},{'b':'cb'},'other')
    ExposureLedger(ledger).validate_rebuild(rebuilt,DedupPolicy())


@pytest.mark.parametrize("legacy",[False,True])
def test_interrupted_repeat_read_cannot_reuse_an_earlier_completion(tmp_path,legacy):
    from poc.data.manifest import ExposureError
    from poc.data.dedup import DedupPolicy
    ledger=tmp_path/'exposure.jsonl';text=tmp_path/'original.txt';text.write_text('old text')
    r=DocumentRecord('a','owned','a'*40,'fixture',0,text_path=str(text))
    manifest=DataManifest((r,),{'a':'dev'},{'a':'c'},'salt',exposure_path=str(ledger))
    list(open_records(manifest,purpose='selection'))
    text.unlink()
    with pytest.raises(FileNotFoundError): list(open_records(manifest,purpose='selection'))
    if legacy:
        import json
        rows=[json.loads(line) for line in ledger.read_text().splitlines()]
        for row in rows: row.pop('intent_id',None)
        ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    other=DocumentRecord('b','owned','a'*40,'fixture',1,text='unrelated')
    rebuilt=DataManifest((other,),{'b':'confirm_A'},{'b':'cb'},'other')
    with pytest.raises(ExposureError,match='incomplete'): ExposureLedger(ledger).validate_rebuild(rebuilt,DedupPolicy())

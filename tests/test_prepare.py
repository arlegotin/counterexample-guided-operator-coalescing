from dataclasses import asdict
import json
from pathlib import Path
import pytest
from poc.config import load_config, validate_config, ConfigError


def test_prepare_requires_explicit_resolved_limits():
    c=load_config(Path('configs/pilot.yaml'),phase='exploratory')
    from poc.data.prepare import run_prepare
    with pytest.raises(ConfigError,match='preparation'): run_prepare(c)


def test_sample_rows_are_seeded_across_row_groups_not_first_n():
    from poc.data.prepare import sampled_rows
    a=sampled_rows([10,10,10,10],seed=17,group_count=2,rows_per_group=3)
    assert a == sampled_rows([10,10,10,10],seed=17,group_count=2,rows_per_group=3)
    assert len(a)==2 and all(len(rows)==3 for _,rows in a)
    assert any(rows != [0,1,2] for _,rows in a)
    assert all(rows==sorted(set(rows)) for _,rows in a)


def test_provenance_gate_precedes_output_or_network(tmp_path):
    from poc.data.prepare import check_source_gate
    p=tmp_path/'sources.json'
    p.write_text(json.dumps({'source_gate':{'status':'blocked'}}))
    with pytest.raises(ValueError,match='source gate'): check_source_gate(p)


def test_prepare_cli_calls_real_preparer_without_teacher_execution(monkeypatch, capsys):
    from poc.cli import main
    import poc.data.prepare
    calls=[]
    def fake(config):
        calls.append(config)
        return {'status':'prepared_outcome_blind','counts':{'FineWeb':3}}
    monkeypatch.setattr(poc.data.prepare,'run_prepare',fake)
    assert main(['prepare','--config','configs/135m.yaml'])==0
    assert calls and 'prepared_outcome_blind' in capsys.readouterr().out


def test_parent_preparation_exposure_blocks_fresh_pool_before_reassignment(tmp_path):
    import hashlib
    from poc.data.prepare import check_parent_lineage
    parent=tmp_path/'preparation.json'
    parent.write_text(json.dumps({'status':'prepared_outcome_blind','protected_predictions':0,'training_updates':0}))
    digest=hashlib.sha256(parent.read_bytes()).hexdigest()
    assert check_parent_lineage(parent,digest)['parent_manifest_sha256']==digest
    (tmp_path/'exposures.jsonl').write_text('{"exploratory":"access occurred"}\n')
    with pytest.raises(ValueError,match='exposure'):check_parent_lineage(parent,digest)


def test_fresh_output_cannot_hide_exposures_in_another_preparation(tmp_path):
    from poc.data.prepare import check_exposure_history
    historical=tmp_path/'data'/'old-preparation';historical.mkdir(parents=True)
    (historical/'exposures.jsonl').write_text('{"prior":"exploratory exposure"}\n')
    with pytest.raises(ValueError,match='exposure'):check_exposure_history(tmp_path)


def test_fresh_prepare_preflights_all_history_before_output_or_network(tmp_path,monkeypatch):
    from dataclasses import replace
    import poc.data.prepare as prepare
    from poc.assets import AssetError
    c=load_config(Path('configs/135m.yaml'),phase='exploratory')
    c=replace(c,preparation=replace(c.preparation,output_dir=tmp_path/'fresh',parent_preparation_manifest=None,parent_manifest_sha256=None))
    def deny(_root): raise AssetError('historical exposure blocks fresh pool')
    monkeypatch.setattr(prepare,'check_exposure_history',deny,raising=False)
    def forbidden_fetch(*args,**kwargs):pytest.fail('download attempted before history preflight')
    monkeypatch.setattr(prepare,'fetch_asset',forbidden_fetch)
    monkeypatch.setattr(prepare,'fetch_git_blob',forbidden_fetch)
    with pytest.raises(AssetError,match='historical exposure'):prepare.run_prepare(c)
    assert not (tmp_path/'fresh').exists()


def test_incremental_text_token_and_manifest_writes_preflight_disk(tmp_path):
    from poc.assets import LogicalDiskBudget,AssetError
    from poc.data.prepare import _write
    disk=LogicalDiskBudget(tmp_path,30)
    disk.write_bytes(tmp_path/'original.txt',b'original\r\n')
    _write(tmp_path/'tokens.json',{'ids':[1,2]},disk_budget=disk)
    before=sum(p.stat().st_size for p in tmp_path.iterdir())
    with pytest.raises(AssetError,match='disk'):_write(tmp_path/'manifest.json',{'owned':'too much'},disk_budget=disk)
    assert not (tmp_path/'manifest.json').exists()
    assert disk.bytes==before<=30


@pytest.mark.parametrize('slack',[0,-1])
def test_final_receipt_encoded_bytes_are_included_without_self_reference(tmp_path,slack):
    from poc.assets import LogicalDiskBudget,AssetError
    from poc.data.prepare import _write
    from poc.records import canonical_json
    initial={'owned':'data'};receipt={'output_files':{'data.json':'a'*64},'status':'prepared_outcome_blind'}
    cap=len(canonical_json(initial))+len(canonical_json(receipt))+slack
    disk=LogicalDiskBudget(tmp_path,cap)
    _write(tmp_path/'data.json',initial,disk_budget=disk)
    if slack<0:
        with pytest.raises(AssetError,match='disk'):_write(tmp_path/'preparation.json',receipt,disk_budget=disk)
        assert not (tmp_path/'preparation.json').exists()
    else:
        _write(tmp_path/'preparation.json',receipt,disk_budget=disk)
        assert disk.bytes==sum(p.stat().st_size for p in tmp_path.iterdir())==cap
        assert json.loads((tmp_path/'preparation.json').read_text())==receipt


def test_existing_hardlinked_assets_count_as_full_logical_paths(tmp_path):
    import os
    from poc.assets import LogicalDiskBudget,AssetError
    (tmp_path/'asset').write_bytes(b'owned')
    os.link(tmp_path/'asset',tmp_path/'reused')
    assert LogicalDiskBudget(tmp_path,10).bytes==10
    with pytest.raises(AssetError,match='disk'):LogicalDiskBudget(tmp_path,9)


def test_prepare_counts_teacher_and_ranges_before_each_http_request(tmp_path,monkeypatch):
    """Owned fake parquet reads reproduce the reviewed 29-byte/25-byte case."""
    import hashlib,io,sys,types
    from dataclasses import replace
    import poc.config as configuration
    import poc.data.prepare as prepare
    from poc.assets import AssetError,RangeAsset,fetch_asset
    original=load_config(Path('configs/135m.yaml'),phase='exploratory')
    revision=original.teacher.revision
    teacher_payloads={'tokenizer.json':b'{}','tokenizer_config.json':b'{"eos_token":"x"}'}
    teacher_bytes=sum(map(len,teacher_payloads.values()))
    assert teacher_bytes==19
    def entry(path,payload):return {'path':path,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    def source(files):return {'revision':revision,'license':'owned-fixture','terms_url':'https://example.invalid/terms','files':files}
    sources={repo:source([]) for repo in [prepare.TEACHER,prepare.FINEWEB,prepare.WIKITEXT,prepare.ARC,prepare.BLIMP]}
    sources[prepare.TEACHER]=source([entry(k,v) for k,v in teacher_payloads.items()])
    sources[prepare.FINEWEB]=source([entry('sample/10BT/owned.parquet',b'0'*100)])
    manifest=tmp_path/'sources.json';manifest.write_text(json.dumps({'source_gate':{'status':'passed_local_preparation'},'sources':sources,'terms_evidence':[{'owned':'fixture'}]}))
    provenance=tmp_path/'provenance';provenance.mkdir()
    (provenance/'qualification-2026-10-27.json').write_text('{"compute_path_status":"passed"}')
    (provenance/'component-validation-2026-10-27.json').write_text('{"status":"passed"}')
    monkeypatch.setattr(configuration,'PROJECT_ROOT',tmp_path);monkeypatch.setattr(prepare,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(prepare,'_execution_identity',lambda _config:{'owned':'fixture'})
    config=replace(original,teacher=replace(original.teacher,weights_sha256=None),preparation=replace(original.preparation,
        source_manifest=manifest,output_dir=tmp_path/'prepared',parent_preparation_manifest=None,parent_manifest_sha256=None,
        max_download_bytes=25,max_range_bytes=20,fineweb_shards=1))
    validate_config(config,command='prepare')
    http=[];readers=[]
    def teacher_open(url,**_kwargs):
        path=url.rsplit('/',1)[-1];http.append(('teacher',len(teacher_payloads[path])))
        return io.BytesIO(teacher_payloads[path])
    monkeypatch.setattr(prepare,'fetch_asset',lambda source,destination,**kwargs:fetch_asset(source,destination,opener=teacher_open,**kwargs))
    def range_open(req,**_kwargs):
        a,b=map(int,req.get_header('Range').removeprefix('bytes=').split('-'))
        http.append(('range',b-a+1));r=io.BytesIO(b'0'*(b-a+1));r.status=206;r.headers={'Content-Range':f'bytes {a}-{b}/100'}
        return r
    def make_range(*args,**kwargs):
        reader=RangeAsset(*args,opener=range_open,**kwargs);readers.append(reader);return reader
    monkeypatch.setattr(prepare,'RangeAsset',make_range)
    tokenizer=types.ModuleType('tokenizers')
    tokenizer.Tokenizer=types.SimpleNamespace(from_file=lambda _path:types.SimpleNamespace(token_to_id=lambda _token:0,encode=lambda *_a,**_k:types.SimpleNamespace(ids=[1])))
    monkeypatch.setitem(sys.modules,'tokenizers',tokenizer)
    parquet=types.ModuleType('pyarrow.parquet');arrow=types.ModuleType('pyarrow');arrow.parquet=parquet
    def parquet_file(reader):
        reader.read(5)  # 19 teacher + 5 range = 24 bytes, permitted
        reader.read(5)  # 29 would exceed 25: no second range HTTP request may start
        pytest.fail('global cap did not stop the second parquet read')
    parquet.ParquetFile=parquet_file
    monkeypatch.setitem(sys.modules,'pyarrow',arrow);monkeypatch.setitem(sys.modules,'pyarrow.parquet',parquet)
    with pytest.raises(AssetError,match='transfer byte cap'):prepare.run_prepare(config)
    assert sum(size for _,size in http)==24 and [kind for kind,_ in http].count('range')==1
    assert readers[0].global_budget.bytes==24 and readers[0].budget.bytes==5
    assert not (config.preparation.output_dir/'preparation.json').exists()

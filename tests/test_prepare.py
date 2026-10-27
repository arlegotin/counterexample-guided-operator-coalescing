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

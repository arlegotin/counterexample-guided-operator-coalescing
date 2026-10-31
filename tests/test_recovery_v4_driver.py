"""Opt-in profile budgets preserve historical defaults and distinct ledger units."""
import importlib.util
from pathlib import Path
from dataclasses import replace,asdict

import pytest

from poc.contracts import Budget


def driver():
    path=Path(__file__).resolve().parents[1]/'provenance/recovery_diagnosis.py'
    spec=importlib.util.spec_from_file_location('v4_budget_driver',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_explicit_v4_profile_limits_cover_setup_initialization_and_one_update():
    module=driver();assert hasattr(module,'profile_limits'),'versioned profile resource dispatch missing'
    budget=Budget(1,4000000,2000000,100000,100000,10,900,900,32*1024**3,6*1024**3)
    default=replace(budget,disk_bytes=8*1024**3,memory_bytes=12*1024**3)
    config={'schema_version':4,'profile_resource_budget':budget.to_dict()}
    assert module.profile_limits(config,'arm',default)==budget
    setup=module.profile_limits(config,'setup',default)
    assert setup.updates==setup.valid_target_presentations==setup.teacher_forward_calls==setup.search_proposals==0
    initial=module.profile_limits(config,'initialization',default)
    assert initial.updates==initial.student_backward_calls==initial.suffix_backward_calls==0
    assert initial.teacher_forward_calls==budget.teacher_forward_calls and initial.search_proposals==10
    for b in (setup,initial):
        assert b.disk_bytes==budget.disk_bytes and b.memory_bytes==budget.memory_bytes and b.wall_seconds==900
    for version in (1,2,3):assert module.profile_limits({'schema_version':version},'arm',default) is default
    with pytest.raises(ValueError):module.profile_limits(config,'unknown',default)


def test_v4_profile_run_and_interrupted_resume_use_real_common_apis(monkeypatch,tmp_path):
    import test_recovery_study as original
    from poc.recovery_initializers import RecoveryInitializerSpec
    module=original.plan_api();load=module.load_study_config
    def version4(path):
        config=load(path)
        config.update(schema_version=4,observation_policy='optimizer_plus_tuning_v2',initialization_checkpoint=None,
            initializer=asdict(RecoveryInitializerSpec(method='parent_svd',calibration_components=2)),
            profile_resource_budget=Budget(1,4000000,2000000,100000,100000,10,900,900,2*1024**3,4*1024**3).to_dict())
        return config
    monkeypatch.setattr(module,'load_study_config',version4)
    original.test_driver_profile_and_run_complete_owned_campaign(monkeypatch,tmp_path)


from test_recovery_clone import source


def test_driver_clones_once_without_reselecting_parent_and_binds_source(source,monkeypatch,tmp_path):
    from poc.recovery_initializers import RecoveryInitializerSpec
    from poc.data.manifest import DataManifest,DocumentRecord
    from poc.recovery_study import NaturalWindowPlan
    from test_training import batch,MemoryRecorder,budget
    import poc.merge
    module=driver();config={**source.config,'schema_version':4,'study':'owned-v4-clone',
        'parent_fit_components':1,'initializer':asdict(RecoveryInitializerSpec(method='parent_svd',calibration_components=1)),
        'checkpoint_every':16,'owner_scope':'affected_group','observations':[0,1,2],
        'numerical_profile':{'device':'cpu','kl_floor_per_target':2e-6},'resource_budget':None}
    fit=DataManifest((DocumentRecord('fit-doc','owned','a'*40,'owned',0,text='owned'),),
        {'fit-doc':'fit'},{'fit-doc':'fit-component'},'owned')
    def forbidden(*args,**kwargs):raise AssertionError('verified clone must not repeat parent selection')
    monkeypatch.setattr(poc.merge,'initialize_merge',forbidden)
    rec=MemoryRecorder();trial=module.initialize(config,source.teacher,fit,(batch(),),23,rec)
    assert trial.initialized_digest==source.state.candidate_digest
    assert trial.initializer_record['initialized_checkpoint_clone']['source_run_id']==trial.clone_provenance['source_run_id']
    assert rec.totals.teacher_forward_calls==0 and rec.totals.search_proposals==0
    for n,p in trial.model.named_parameters():
        import torch
        assert torch.equal(p,source.state.model[n])
    monkeypatch.setattr(module,'output_root',lambda c:tmp_path)
    monkeypatch.setattr(module,'ASSET_SHA',source.teacher.tokenizer_digest)
    (tmp_path/'controls.json').write_text('{}')
    plan=source.plan
    cfg,obj=module.make_spec(config,'full_student',23,plan,trial.model,{'commit':'a'*40},{'owned':True},budget(),initializer_record=trial.initializer_record)
    assert cfg.run_spec.input_manifests['initialization_checkpoint']==config['initialization_checkpoint']['checkpoint_manifest_sha256']
    assert cfg.run_spec.input_manifests['raw_plan']==source.state.source_hash
    trial.initializer_record['initialized_checkpoint_clone']['raw_plan_sha256']='0'*64
    with pytest.raises(ValueError,match='clone plan'):
        module.make_spec(config,'full_student',23,plan,trial.model,{'commit':'a'*40},{'owned':True},budget(),initializer_record=trial.initializer_record)


@pytest.mark.parametrize('mutation',['missing','tampered'])
def test_v4_validate_checks_checkpoint_metadata_before_population_reads(source,monkeypatch,tmp_path,mutation):
    module=driver()
    config={**source.config,'schema_version':4,'asset_root':None,'resource_budget':None,'study':'owned-v4-clone',
        'loss_scopes':['full_student'],'checkpoint_every':16,'observations':[0,1,2]}
    config['populations']={k:{**v,'path':None} for k,v in config['populations'].items()}
    monkeypatch.setattr(module,'output_root',lambda c:tmp_path)
    if mutation=='missing':
        source.spec_path.unlink()
        value=module.preflight(config)
        assert any('initialization_checkpoint' in v for v in value['unresolved'])
    else:
        source.record_path.write_bytes(source.record_path.read_bytes()+b' ')
        with pytest.raises(ValueError,match='checkpoint'):
            module.preflight(config)


def test_profile_forecast_counts_retained_dependency_bytes_once(tmp_path):
    from types import SimpleNamespace
    module=driver();assert hasattr(module,'family_disk_forecast'),'retained dependency forecast missing'
    child=tmp_path/'child';ancestor=tmp_path/'ancestor';child.mkdir();ancestor.mkdir()
    (child/'new.bin').write_bytes(b'1234');(ancestor/'old.bin').write_bytes(b'12345678')
    recorder=SimpleNamespace(disk=lambda:sum(p.stat().st_size for root in (child,ancestor) for p in root.rglob('*') if p.is_file()))
    # Forecast20 already includes the child's current4bytes. Add only the8
    # retained dependency bytes, so a24byte budget correctly fails preflight.
    total,dependency=module.family_disk_forecast({'schema_version':4},child,recorder,20)
    assert dependency==8 and total==28 and total>24
    assert module.family_disk_forecast({'schema_version':3},child,recorder,20)==(20,0)


from test_recovery_clone import horizon_source


def test_driver_new_horizon_uses_original_windows_without_parent_reselection(horizon_source,monkeypatch,tmp_path):
    from poc.recovery_initializers import RecoveryInitializerSpec
    h=horizon_source;s=h.source;module=driver()
    config={**h.config,'schema_version':4,'parent_fit_components':1,
            'initializer':asdict(RecoveryInitializerSpec(calibration_components=1))}
    from poc.data.manifest import load_manifest
    ref=config['populations']['fit']
    fit=load_manifest(Path(ref['path']),expected_sha256=ref['sha256'],roles={'fit'})
    import poc.merge
    def forbidden(*args,**kwargs):raise AssertionError('reselected original parent')
    monkeypatch.setattr(poc.merge,'initialize_merge',forbidden)
    trial=module.initialize(config,s.teacher,fit,s.fit_batches,23,None)
    assert trial.model.manifest().checkpoint_digest==s.state.candidate_digest
    assert trial.initializer_record['initialized_checkpoint_clone']['source_updates']==2
    assert trial.initializer_record['initialized_checkpoint_clone']['target_updates']==12
    config.update(study='owned-horizon',checkpoint_every=16,owner_scope='affected_group',observations=[0,1,12],
        numerical_profile={'device':'cpu','kl_floor_per_target':2e-6},resource_budget=None)
    monkeypatch.setattr(module,'output_root',lambda c:tmp_path)
    monkeypatch.setattr(module,'ASSET_SHA',s.teacher.tokenizer_digest)
    (tmp_path/'controls.json').write_text('{}')
    from poc.recovery_study import NaturalWindowPlan
    from test_training import budget
    plan=NaturalWindowPlan.build(s.fit_batches,updates=12,seed=23,
        manifest_hashes={'fit':config['populations']['fit']['sha256'],'tokenizer':s.teacher.tokenizer_digest})
    cfg,obj=module.make_spec(config,'full_student',23,plan,trial.model,{'commit':'a'*40},{'owned':True},budget(),initializer_record=trial.initializer_record)
    assert cfg.max_updates==12 and cfg.run_spec.input_manifests['raw_plan']==plan.digest
    assert cfg.run_spec.input_manifests['initialization_source_raw_plan']==s.plan.digest
    assert cfg.run_spec.input_manifests['initialization_binding']==module.hashed(config['initialization_checkpoint'])
    assert obj.raw_plan_sha256==plan.digest!=s.plan.digest
    trial.initializer_record['initialized_checkpoint_clone']['target_raw_plan_sha256']='0'*64
    with pytest.raises(ValueError,match='clone plan'):
        module.make_spec(config,'full_student',23,plan,trial.model,{'commit':'a'*40},{'owned':True},budget(),initializer_record=trial.initializer_record)


def test_preflight_rejects_v2_lineage_before_population_reader(horizon_source,monkeypatch,tmp_path):
    import json
    from poc.records import canonical_json
    import hashlib
    h=horizon_source;module=driver()
    config={**h.config,'schema_version':4,'asset_root':None,'resource_budget':None,'study':'owned-horizon',
        'loss_scopes':['full_student'],'checkpoint_every':16,'observations':[0,1,12]}
    ref=config['populations']['fit'];path=Path(ref['path']);raw=json.loads(path.read_text())
    raw['lineage_sha256']='0'*64;path.write_bytes(canonical_json(raw));ref['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module,'output_root',lambda c:tmp_path)
    with pytest.raises(ValueError,match='lineage'):
        module.preflight(config)


def test_actual_input_loader_checks_v2_lineage_before_asset_or_raw_data_io(horizon_source,monkeypatch):
    import json,hashlib
    from poc.records import canonical_json
    from poc.teacher import AssetManifest
    h=horizon_source;module=driver();config={**h.config,'schema_version':4,'asset_root':'/owned/assets'}
    ref=config['populations']['fit'];path=Path(ref['path']);raw=json.loads(path.read_text())
    raw['lineage_sha256']='0'*64;path.write_bytes(canonical_json(raw));ref['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    def forbidden(*args,**kwargs):raise AssertionError('assets/raw data opened before lineage denial')
    monkeypatch.setattr(AssetManifest,'load',forbidden)
    with pytest.raises(ValueError,match='lineage'):
        module.load_inputs(config)

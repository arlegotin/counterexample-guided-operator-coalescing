"""Owned source/target plan lineage for opt-in reconstruction schema2."""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace

import pytest,yaml

from poc.contracts import thaw
from poc.data.manifest import DataManifest,DocumentRecord
from poc.records import canonical_json
from poc.reconstruction_study import load_reconstruction_study,recovery_config
from poc.reconstruction_executor import build_phase_plans,phase_config
from test_reconstruction_study import draft,limits
from test_training import batch as original_batch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver():
    path=Path(__file__).resolve().parents[1]/'provenance/reconstruction_recovery.py'
    spec=importlib.util.spec_from_file_location('owned_lineage_reconstruction',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def source(tmp_path,monkeypatch):
    import test_recovery_clone as original
    old_models,old_make=original.models,original.make_bank
    def models():
        teacher,unused=old_models();teacher.config.max_position_embeddings=1024
        return teacher,unused
    monkeypatch.setattr(original,'models',models)
    monkeypatch.setattr(original,'make_bank',lambda t,p,r,s:old_make(t,p,8,s))
    monkeypatch.setattr(original,'batch',lambda:original_batch(512))
    root=tmp_path/'source';root.mkdir()
    value=original.source.__wrapped__(root);value.config['rank']=8
    # Make fit/tuning components genuinely separate in this metadata fixture.
    ref=value.population_refs['tuning'];path=Path(ref['path'])
    manifest=DataManifest((DocumentRecord('tune-doc','owned','a'*40,'owned',1,text='distinct owned tuning'),),
        {'tune-doc':'dev'},{'tune-doc':'tune-component'},'owned',str(root/'old-tuning.jsonl'))
    path.write_bytes(canonical_json(asdict(manifest)));ref['sha256']=sha(path)
    value.config['populations']['tuning']=deepcopy(ref)
    value.publish(updated_spec=replace(value.spec,input_manifests={**thaw(value.spec.input_manifests),'tuning':ref['sha256']}))
    return value


@pytest.fixture
def study(tmp_path,source):
    c=draft();c.update(schema_version=2,study='owned-reconstruction-lineage-v2',out='artifacts/owned-reconstruction-lineage-v2',
        pair=[0,1],seed=23,calibration_components=1,warm_updates=2,recovery_updates=12,
        observations={'warm':[0,2],'recovery':[0,1,12]},numerical_profile={**c['numerical_profile'],'device':'cpu'})
    c['profile_resource_budget']['warm']=limits(2)
    c['retained_dependency_roots']=[str(tmp_path/'source')]
    assets=tmp_path/'assets';assets.mkdir();(assets/'teacher-assets.json').write_text('{}');c['asset_root']=str(assets)
    binding=deepcopy(source.config['initialization_checkpoint'])
    binding.update(schema_version=2,source_updates=2,source_populations=deepcopy(source.population_refs))
    c['initialization_checkpoint']=binding
    for role,ref in source.population_refs.items():
        raw=json.loads(Path(ref['path']).read_text());body=dict(raw);records=body.pop('records')
        manifest=DataManifest(records=tuple(DocumentRecord(**r) for r in records),**body)
        raw['lineage_sha256']=manifest.digest();raw['exposure_path']=str(tmp_path/('new-'+role+'.jsonl'))
        path=tmp_path/(role+'-target.json');path.write_bytes(canonical_json(raw))
        c['populations'][role]={'path':str(path),'sha256':sha(path),'components':1}
    return c


def load(tmp_path,study):
    path=tmp_path/'study.yaml';path.write_text(yaml.safe_dump(study,sort_keys=False))
    return load_reconstruction_study(path)


def test_schema2_requires_explicit_new_lineage_binding_and_preserves_legacy_strictness(tmp_path,study):
    config=load(tmp_path,study)
    assert config['schema_version']==2
    derived=recovery_config(config)
    assert derived['schema_version']==4 and derived['updates']==12
    assert derived['initialization_checkpoint']['source_updates']==2
    with pytest.raises(TypeError):config['initialization_checkpoint']['source_updates']=12
    legacy=deepcopy(study);legacy['schema_version']=1
    with pytest.raises(ValueError):load(tmp_path,legacy)
    missing=deepcopy(study);del missing['initialization_checkpoint']['source_updates']
    with pytest.raises(ValueError):load(tmp_path,missing)


def test_validate_rebuilds_source_plan_separately_from_target_plans(tmp_path,study,source,monkeypatch):
    config=load(tmp_path,study);module=driver()
    from poc.data.manifest import load_manifest
    ref=study['populations']['fit'];fit=load_manifest(Path(ref['path']),expected_sha256=ref['sha256'],roles={'fit'})
    tuning=replace(original_batch(512),document_ids=('tune-doc',),cluster_ids=('tune-component',))
    monkeypatch.setattr(module.driver,'ASSET_SHA',source.teacher.tokenizer_digest)
    monkeypatch.setattr(module.driver,'load_inputs',lambda c:(None,fit,source.fit_batches,(tuning,),{}))
    report=module.validate(config)
    assert report['missing']==['resource_budget']
    assert report['initialization_lineage']['source_raw_plan_sha256']==source.plan.digest
    assert report['raw_plans']['recovery']['plan_sha256']!=source.plan.digest
    assert report['initialization_lineage']['source_updates']==2
    assert report['initialization_lineage']['target_updates']==12


def test_derived_loader_denies_old_journal_before_assets_or_raw_reads(tmp_path,study,source,monkeypatch):
    from poc.teacher import AssetManifest
    module=driver();config=load(tmp_path,study);derived=recovery_config(config)
    ref=derived['populations']['fit'];path=Path(ref['path']);raw=json.loads(path.read_text())
    original=json.loads(Path(study['initialization_checkpoint']['source_populations']['fit']['path']).read_text())
    raw['exposure_path']=original['exposure_path'];path.write_bytes(canonical_json(raw));ref['sha256']=sha(path)
    def forbidden(*a,**k):raise AssertionError('asset/raw input read before old journal denial')
    monkeypatch.setattr(AssetManifest,'load',forbidden)
    with pytest.raises(ValueError,match='journal'):module.driver.load_inputs(derived)


def test_phase_specs_bind_original_initialization_and_new_target_plans(tmp_path,study,source):
    from poc.recovery_clone import clone_initialization_checkpoint
    config=load(tmp_path,study);derived=recovery_config(config)
    trial=clone_initialization_checkpoint(source.teacher,derived,23,fit_batches=source.fit_batches)
    plans=build_phase_plans(config,source.fit_batches,tokenizer_digest=source.teacher.tokenizer_digest)
    specs=[]
    for phase in ('warm','recovery'):
        cfg,obj=phase_config(config,phase,plans[phase],trial.model,source={'commit':'a'*40},environment={'owned':True},
            budget=source.spec.budgets,controls_sha256='c'*64)
        specs.append(cfg.run_spec)
        assert cfg.run_spec.input_manifests['initialization_source_raw_plan']==source.plan.digest
        assert cfg.run_spec.input_manifests['initialization_binding']==hashlib.sha256(canonical_json(derived['initialization_checkpoint'])).hexdigest()
        assert cfg.run_spec.input_manifests['raw_plan']==plans[phase].digest
        assert cfg.run_spec.resolved_config['initialization_lineage']['source_updates']==2
        assert cfg.run_spec.resolved_config['initialization_lineage']['target_updates']==12
    assert specs[0].input_manifests['owner_names']==specs[1].input_manifests['owner_names']
    assert specs[0].input_manifests['initialized_state']==specs[1].input_manifests['initialized_state']==source.state.candidate_digest


@pytest.mark.parametrize('mode',['profile','run'])
def test_actual_execution_denies_source_horizon_metadata_before_output_or_data_io(tmp_path,study,monkeypatch,mode):
    study['initialization_checkpoint']['source_updates']=3
    if mode=='run':
        study['resource_budget']={'frozen_profile_sha256':'a'*64,
            'per_phase':{'warm':limits(2),'recovery':limits(12)},
            'campaign':{'wall_seconds':3600,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}}
    config=load(tmp_path,study);module=driver()
    def forbidden(*a,**k):raise AssertionError('output/data admitted before known source horizon mismatch')
    monkeypatch.setattr(module,'admit_output',forbidden)
    with pytest.raises(ValueError,match='source.*horizon|metadata'):
        module.execute(config,tmp_path/'study.yaml',mode)


@pytest.mark.parametrize('problem',['record_hash','manifest_hash','record_state','record_owners','record_parent','manifest_run_id','record_alias','directory_alias','state_partition'])
@pytest.mark.parametrize('mode',['validate','profile','run'])
def test_complete_source_metadata_denied_before_output_or_raw_admission(tmp_path,study,source,monkeypatch,problem,mode):
    binding=study['initialization_checkpoint']
    if problem=='record_hash':binding['initialization_record_sha256']='0'*64
    elif problem=='manifest_hash':binding['checkpoint_manifest_sha256']='0'*64
    elif problem in ('record_state','record_owners','record_parent'):
        path=Path(binding['initialization_record_path']);raw=json.loads(path.read_text())
        if problem=='record_state':raw['initialized_digest']='0'*64
        elif problem=='record_owners':raw['owners']=[]
        else:raw['selected_parent']=2
        path.write_bytes(canonical_json(raw));binding['initialization_record_sha256']=sha(path)
    elif problem=='manifest_run_id':
        path=Path(binding['checkpoint_directory'])/'manifest.json';raw=json.loads(path.read_text())
        raw['run_id']='0'*64;path.write_bytes(canonical_json(raw));binding['checkpoint_manifest_sha256']=sha(path)
    elif problem=='record_alias':
        path=tmp_path/'record-alias.json';path.symlink_to(binding['initialization_record_path']);binding['initialization_record_path']=str(path)
    elif problem=='directory_alias':
        path=tmp_path/'checkpoint-alias';path.symlink_to(binding['checkpoint_directory'],target_is_directory=True);binding['checkpoint_directory']=str(path)
    else:
        from poc.checkpoint import _spec
        original_spec=_spec(json.loads(source.spec_path.read_text()))
        source.publish(updated_state=replace(source.state,spec=original_spec,partition=((0,),(1,2))))
        binding['checkpoint_directory']=source.config['initialization_checkpoint']['checkpoint_directory']
        binding['checkpoint_manifest_sha256']=source.config['initialization_checkpoint']['checkpoint_manifest_sha256']
    if mode=='run':
        study['resource_budget']={'frozen_profile_sha256':'a'*64,'per_phase':{'warm':limits(2),'recovery':limits(12)},
            'campaign':{'wall_seconds':3600,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}}
    config=load(tmp_path,study);module=driver()
    monkeypatch.setattr(module.driver,'ASSET_SHA',source.teacher.tokenizer_digest)
    def forbidden(*a,**k):raise AssertionError('known bad source admitted to output/raw I/O')
    monkeypatch.setattr(module,'admit_output',forbidden)
    monkeypatch.setattr(module.driver,'load_inputs',forbidden)
    with pytest.raises(ValueError):
        if mode=='validate':module.validate(config)
        else:module.execute(config,tmp_path/'study.yaml',mode)

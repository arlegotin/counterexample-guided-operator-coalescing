"""Owned CPU STEP0 checkpoint cloning; actual fit-only parent selection."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib
from types import SimpleNamespace

import pytest
import torch

from test_training import batch, models, budget, MemoryRecorder
from test_records import spec as run_spec
from poc.bank import make_bank, canonicalize_partition
from poc.checkpoint import save_checkpoint
from poc.contracts import thaw
from poc.data.manifest import DataManifest, DocumentRecord
from poc.merge import initialize_merge
from poc.records import canonical_json
from poc.recovery_objectives import RecoveryObjectiveSpec, owner_manifest, owner_names_hash
from poc.recovery_study import NaturalWindowPlan
from poc.train import (TrainingConfig, TrainingSources, configure_trainable_owners,
                       _optimizer, _snapshot, _schedule_state)


def hashed(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def api():
    return importlib.import_module('poc.recovery_clone')


@pytest.fixture
def source(tmp_path):
    teacher, _ = models()
    teacher.tokenizer_digest = 'a'*64
    fit = DataManifest((DocumentRecord('fit-doc','owned','a'*40,'owned',0,text='owned'),),
                       {'fit-doc':'fit'}, {'fit-doc':'fit-component'}, 'owned')
    current = make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),1,23)
    trial = initialize_merge(current,teacher,(0,1),fit,23,batches=(batch(),))
    student = trial.model
    population_refs={}
    for role,api_role in (('fit','fit'),('tuning','dev')):
        manifest=replace(fit,roles={'fit-doc':api_role},exposure_path=str(tmp_path/('old-'+role+'.jsonl')))
        path=tmp_path/(role+'-source.json');path.write_bytes(canonical_json(asdict(manifest)))
        population_refs[role]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    plan = NaturalWindowPlan.build((batch(),),updates=2,raw_targets_per_update=8192,seed=23,
                                   manifest_hashes={'fit':population_refs['fit']['sha256'],'tokenizer':teacher.tokenizer_digest})
    obj = RecoveryObjectiveSpec('local_sum','affected_group',(0,1),plan.digest,owner_names_hash(student,(0,1)))
    cfg = TrainingConfig(max_updates=2,alpha=0,affected_depths=(0,1))
    inputs = dict(assets=teacher.tokenizer_digest,fit=population_refs['fit']['sha256'],tuning=population_refs['tuning']['sha256'],raw_plan=plan.digest,
                  owner_names=obj.owner_names_sha256,initialized_state=trial.initialized_digest,
                  initialized_owners=hashed({'owners':owner_manifest(student,(0,1))}))
    spec = run_spec(seeds={'fit':23},budgets=budget(),input_manifests=inputs,
                    resolved_config={'training':cfg.identity(),'recovery_objective':obj.identity()})
    cfg = replace(cfg,run_spec=spec)
    owners = configure_trainable_owners(student,cfg,objective_spec=obj)
    rec = MemoryRecorder()
    state = _snapshot(student,_optimizer(owners,cfg),_schedule_state(0,2),cfg,TrainingSources(raw_plan=plan),
                      0,0,0,[],SimpleNamespace(recorder=rec,total=rec.totals),
                      teacher.manifest().checkpoint_digest,(),objective_spec=obj)
    record = dict(selected_parent=trial.selected_parent,parent_scores=trial.medoid_scores,
                  initialized_digest=trial.initialized_digest,owners=owner_manifest(student,(0,1)))
    spec_path=tmp_path/'spec.json';spec_path.write_bytes(canonical_json(spec.to_dict()))
    record_path=tmp_path/'initialization.json';record_path.write_bytes(canonical_json(record))
    config=dict(pair=[0,1],rank=1,seeds=[23],updates=2,raw_targets_per_update=8192,
                populations=deepcopy(population_refs))
    def publish(updated_state=None,updated_record=None,updated_spec=None):
        nonlocal state,spec,record
        if updated_spec is not None:
            spec=updated_spec;state=replace(state,spec=spec)
            spec_path.write_bytes(canonical_json(spec.to_dict()))
        if updated_state is not None:state=updated_state
        if updated_record is not None:
            record=updated_record;record_path.write_bytes(canonical_json(record))
        ref=save_checkpoint(state,tmp_path/'checkpoints')
        config['initialization_checkpoint']=dict(checkpoint_directory=ref.directory,
            checkpoint_manifest_sha256=ref.manifest_sha256,run_spec_path=str(spec_path),
            run_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            initialization_record_path=str(record_path),
            initialization_record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest())
        return ref
    publish()
    return SimpleNamespace(teacher=teacher,student=student,state=state,spec=spec,record=record,
                           config=config,publish=publish,spec_path=spec_path,record_path=record_path,
                           population_refs=population_refs,fit_batches=(batch(),),plan=plan)


def test_clone_copies_every_owner_exactly_with_independent_storage_and_fresh_optimizer(source):
    before = source.teacher.manifest().checkpoint_digest
    trial = api().clone_initialization_checkpoint(source.teacher,source.config,23)
    assert trial.initialized_digest == source.state.candidate_digest
    assert trial.model.manifest().checkpoint_digest == trial.initialized_digest
    assert trial.selected_parent == source.record['selected_parent']
    assert trial.medoid_scores == tuple(source.record['parent_scores'])
    for name,p in trial.model.named_parameters():
        assert torch.equal(p,source.state.model[name])
        assert p.data_ptr() != source.state.model[name].data_ptr()
        assert p.data_ptr() != dict(source.student.named_parameters())[name].data_ptr()
    obj=RecoveryObjectiveSpec('full_student','affected_group',(0,1),source.spec.input_manifests['raw_plan'],
                             owner_names_hash(trial.model,(0,1)))
    cfg=TrainingConfig(max_updates=2,alpha=0,affected_depths=(0,1))
    assert not _optimizer(configure_trainable_owners(trial.model,cfg,objective_spec=obj),cfg).state
    assert source.teacher.manifest().checkpoint_digest == before
    assert all(not p.requires_grad and p.grad is None for p in source.teacher.parameters())
    assert trial.clone_provenance['source_run_id'] == hashed(source.spec.to_dict())
    assert trial.clone_provenance['raw_plan_sha256'] == source.spec.input_manifests['raw_plan']
    assert trial.clone_provenance['initialized_owners'] == source.spec.input_manifests['initialized_owners']
    assert trial.clone_provenance['step'] == 0


@pytest.mark.parametrize('change',[
    {'step':1},{'target_cursor':8192},{'probe_cursor':1},{'visited':('fit-doc',)},
    {'events':[{'kind':'update','step':0}]},
    {'optimizer':{'state':{'bank.0.gate_proj.weight':{'step':torch.tensor(1.)}},'groups':[]}},
    {'teacher_digest':'0'*64},{'partition':((0,),(1,2))},
])
def test_rejects_noninitial_or_mismatched_state_before_constructing_candidate(source,change,monkeypatch):
    source.publish(updated_state=replace(source.state,**change))
    module=api()
    def forbidden(*args,**kwargs):raise AssertionError('candidate constructed before validation')
    monkeypatch.setattr(module,'BankDecoder',forbidden)
    with pytest.raises(ValueError):module.clone_initialization_checkpoint(source.teacher,source.config,23)


@pytest.mark.parametrize('field,value',[
    ('fit','0'*64),('tuning','0'*64),('assets','0'*64),('owner_names','0'*64),
    ('initialized_state','0'*64),('initialized_owners','0'*64),('raw_plan','0'*64),
])
def test_rejects_inconsistent_source_bindings(source,field,value):
    source.publish(updated_spec=replace(source.spec,input_manifests={**thaw(source.spec.input_manifests),field:value}))
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)


@pytest.mark.parametrize('field,value',[
    ('selected_parent',2),('initialized_digest','0'*64),('owners',[]),('parent_scores',[]),
])
def test_rejects_inconsistent_initialization_record(source,field,value):
    source.publish(updated_record={**source.record,field:value})
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)


@pytest.mark.parametrize('field,value',[
    ('pair',[0,2]),('rank',2),('updates',3),('raw_targets_per_update',4096),('seeds',[17]),
])
def test_rejects_target_contract_mismatch(source,field,value):
    source.config[field]=value
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)


def test_rejects_wrong_source_seed(source):
    source.publish(updated_spec=replace(source.spec,seeds={'fit':17}))
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)


@pytest.mark.parametrize('field',['checkpoint_manifest_sha256','run_spec_sha256','initialization_record_sha256'])
def test_rejects_detached_hash_mismatch(source,field):
    source.config['initialization_checkpoint'][field]='0'*64
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)


def test_rejects_missing_or_extra_checkpoint_binding_keys(source):
    source.config['initialization_checkpoint']['unreviewed']=True
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)


@pytest.mark.parametrize('mutation', ['names','shape','dtype','digest','schema_rank','schema_groups','schema_owner'])
def test_rejects_corrupt_canonical_model_before_copy(source,mutation):
    state=replace(source.state,model=deepcopy(source.state.model),model_schema=deepcopy(source.state.model_schema))
    name=next(iter(state.model))
    if mutation=='names':state.model['unexpected.owner']=state.model.pop(name)
    elif mutation=='shape':state.model[name]=state.model[name].flatten()
    elif mutation=='dtype':state.model[name]=state.model[name].float()
    elif mutation=='digest':state.model[name].add_(.25)
    elif mutation=='schema_rank':state.model_schema['rank']=2
    elif mutation=='schema_groups':state.model_schema['groups']=[[0],[1,2]]
    elif mutation=='schema_owner':state.model_schema['owners'][0]['shape']=[999]
    source.publish(updated_state=state)
    before=source.teacher.manifest().checkpoint_digest
    with pytest.raises(ValueError):api().clone_initialization_checkpoint(source.teacher,source.config,23)
    assert source.teacher.manifest().checkpoint_digest==before


def test_accepts_fully_empty_optimizer_mapping_and_no_source_storage_aliases(source):
    source.publish(updated_state=replace(source.state,optimizer={'state':{},'groups':[]}))
    trial=api().clone_initialization_checkpoint(source.teacher,source.config,23)
    assert trial.clone_provenance['initialized_owners_sha256']==source.spec.input_manifests['initialized_owners']
    original=source.teacher.embed_tokens.weight.detach().clone()
    with torch.no_grad():trial.model.embed_tokens.weight.add_(1)
    assert torch.equal(source.teacher.embed_tokens.weight,original)
    assert torch.equal(source.student.embed_tokens.weight,original)


def test_rejects_trainable_teacher_without_mutating_it(source):
    source.teacher.embed_tokens.weight.requires_grad_(True)
    before=source.teacher.manifest().checkpoint_digest
    with pytest.raises(ValueError,match='frozen'):api().clone_initialization_checkpoint(source.teacher,source.config,23)
    assert source.teacher.manifest().checkpoint_digest==before


@pytest.fixture
def horizon_source(source,tmp_path):
    import json
    config=deepcopy(source.config)
    config['updates']=12
    binding=config['initialization_checkpoint']
    binding.update(schema_version=2,source_updates=2,source_populations=deepcopy(source.population_refs))
    for role,ref in source.population_refs.items():
        original=json.loads(__import__('pathlib').Path(ref['path']).read_text())
        original['lineage_sha256']=hashed(original)
        original['exposure_path']=str(tmp_path/('new-'+role+'.jsonl'))
        path=tmp_path/(role+'-target.json');path.write_bytes(canonical_json(original))
        config['populations'][role]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return SimpleNamespace(source=source,config=config)


def test_explicit_v2_new_horizon_keeps_exact_step0_weights_and_separates_journals(horizon_source):
    h=horizon_source;s=h.source
    before=s.student.manifest().checkpoint_digest
    trial=api().clone_initialization_checkpoint(s.teacher,h.config,23,fit_batches=s.fit_batches)
    assert trial.initialized_digest==before==s.state.candidate_digest
    assert trial.clone_provenance['schema_version']==2
    assert trial.clone_provenance['source_updates']==2
    assert trial.clone_provenance['target_updates']==12
    assert trial.clone_provenance['source_raw_plan_sha256']==s.plan.digest
    assert trial.clone_provenance['population_lineage']['fit']['source_sha256']==s.population_refs['fit']['sha256']
    for name,p in trial.model.named_parameters():
        assert torch.equal(p,dict(s.student.named_parameters())[name])
        assert p.data_ptr()!=dict(s.student.named_parameters())[name].data_ptr()
    assert s.student.manifest().checkpoint_digest==before


def test_v2_requires_exact_source_plan_batches_before_allocating_candidate(horizon_source,monkeypatch):
    h=horizon_source;module=api()
    def forbidden(*args,**kwargs):raise AssertionError('allocated candidate before plan checks')
    monkeypatch.setattr(module,'BankDecoder',forbidden)
    with pytest.raises(ValueError,match='batch|plan'):
        module.clone_initialization_checkpoint(h.source.teacher,h.config,23)
    changed=replace(batch(),input_ids=batch().input_ids.clone())
    changed.input_ids[0,0]+=1
    with pytest.raises(ValueError,match='plan'):
        module.clone_initialization_checkpoint(h.source.teacher,h.config,23,fit_batches=(changed,))


@pytest.mark.parametrize('mutation',['records','roles','membership','salt','quarantine','lineage','journal','journal_alias','target_hash','source_hash','source_updates','unknown_binding','unknown_source_ref'])
def test_v2_rejects_population_horizon_and_closed_schema_mismatches(horizon_source,mutation,tmp_path):
    import json
    from pathlib import Path
    h=horizon_source;config=h.config;binding=config['initialization_checkpoint']
    ref=config['populations']['fit'];path=Path(ref['path']);raw=json.loads(path.read_text())
    if mutation=='records':raw['records'][0]['text']='different'
    elif mutation=='roles':raw['roles']['fit-doc']='dev'
    elif mutation=='membership':raw['membership']['fit-doc']='different-component'
    elif mutation=='salt':raw['salt']='changed'
    elif mutation=='quarantine':raw['quarantine']={'fit-doc':'excluded'}
    elif mutation=='lineage':raw['lineage_sha256']='0'*64
    elif mutation=='journal':raw['exposure_path']=str(tmp_path/'old-fit.jsonl')
    elif mutation=='journal_alias':
        old=tmp_path/'old-fit.jsonl';old.write_text('immutable old exposure')
        Path(raw['exposure_path']).symlink_to(old)
    elif mutation=='source_updates':binding['source_updates']=3
    elif mutation=='unknown_binding':binding['unpriced']=1
    elif mutation=='unknown_source_ref':binding['source_populations']['fit']['unexpected']=True
    elif mutation=='target_hash':ref['sha256']='0'*64
    elif mutation=='source_hash':binding['source_populations']['fit']['sha256']='0'*64
    if mutation in ('records','roles','membership','salt','quarantine','lineage','journal','journal_alias'):
        path.write_bytes(canonical_json(raw));ref['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        api().clone_initialization_checkpoint(h.source.teacher,config,23,fit_batches=h.source.fit_batches)


def test_v2_does_not_relax_untrained_checkpoint_gate(horizon_source):
    h=horizon_source;s=h.source
    s.publish(updated_state=replace(s.state,step=1))
    h.config['initialization_checkpoint'].update(s.config['initialization_checkpoint'])
    with pytest.raises(ValueError,match='STEP0'):
        api().clone_initialization_checkpoint(s.teacher,h.config,23,fit_batches=s.fit_batches)

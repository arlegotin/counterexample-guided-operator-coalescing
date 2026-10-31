"""Completed-source joint recovery controls; owned models, no pretrained data."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib,json

import pytest
import torch

from test_training import batch,budget,MemoryRecorder
from test_recovery_study import setup,bound_config
from poc.checkpoint import save_checkpoint
from poc.records import canonical_json,run_id
from poc.train import TrainingSources,fit_model
from poc.data.manifest import DataManifest,DocumentRecord


def api():
    assert importlib.util.find_spec('poc.recovery_composition'),'completed composition recovery initializer missing'
    return importlib.import_module('poc.recovery_composition')


@pytest.fixture
def source(tmp_path):
    teacher,student,plan,obj=setup(updates=2)
    # setup expands context after its original bank construction; construct the
    # source against the final teacher config before creating its lineage.
    from poc.bank import make_bank,PROJECTIONS
    generator=torch.Generator().manual_seed(991)
    with torch.no_grad():
        teacher.embed_tokens.weight.copy_(torch.randn(teacher.embed_tokens.weight.shape,generator=generator,dtype=teacher.dtype)*.4)
        for layer in teacher.layers:
            for name in PROJECTIONS:
                weight=getattr(layer.mlp,name).weight
                weight.copy_(torch.randn(weight.shape,generator=generator,dtype=teacher.dtype)*.2)
    student=make_bank(teacher,student.partition,student.rank,17)
    cfg=bound_config(obj,2)
    inputs=dict(cfg.run_spec.input_manifests,assets=teacher.tokenizer_digest,fit='b'*64,tuning='c'*64)
    identity=replace(cfg.run_spec,input_manifests=inputs,seeds={'fit':17})
    cfg=replace(cfg,run_spec=identity)
    result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=obj)
    ref=save_checkpoint(result.state,tmp_path/'safe')
    spec=tmp_path/'source-spec.json';spec.write_bytes(canonical_json(identity.to_dict()))
    binding=dict(schema_version=1,source_run_spec_path=str(spec),source_run_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),checkpoint_directory=ref.directory,checkpoint_manifest_sha256=ref.manifest_sha256,merge_pair=[0,2],forced_parent=0,reference='current_continuation',expected_partition=[[0,1,2]])
    config=dict(schema_version=5,composition=binding,pair=[0,1,2],rank=1,seeds=[17],parent_fit_components=1,populations={'fit':{'sha256':'b'*64},'tuning':{'sha256':'c'*64}})
    fit=DataManifest((DocumentRecord('fit-doc','owned','a'*40,'owned',0,text='owned'),),{'fit-doc':'fit'},{'fit-doc':'fit-component'},'owned')
    return SimpleNamespace(teacher=teacher,student=student,state=result.state,spec=identity,spec_path=spec,ref=ref,config=config,fit=fit,plan=plan)


def test_completed_composition_source_is_independent_and_changes_only_new_group(source):
    before=source.student.manifest().checkpoint_digest;rec=MemoryRecorder()
    trial=api().initialize_composition_recovery(source.teacher,source.config,source.fit,(batch(),),17,recorder=rec)
    assert trial.affected_depths==(0,1,2) and trial.selected_parent==0
    assert trial.model.partition.groups==((0,1,2),)
    assert source.student.manifest().checkpoint_digest==before
    assert trial.initializer_record['kind']=='completed_source_composition_v1'
    assert trial.initializer_record['source_run_id']==run_id(source.spec)
    assert trial.initializer_record['reference']=='current_continuation'
    assert trial.initializer_record['optimizer_policy']=='fresh; source optimizer/RNG/cursors/spending are not imported'
    from poc.recovery_objectives import affected_owners
    assert len(affected_owners(trial.model,(0,1,2)))==21
    assert rec.totals.search_proposals==2 and rec.totals.valid_target_presentations==6*8
    assert rec.totals.raw_corpus_target_presentations==2*8
    old=dict(source.student.named_parameters())
    for name,p in trial.model.named_parameters():
        if not name.startswith(('bank.','layers.0.mlp.','layers.1.mlp.','layers.2.mlp.')):
            assert torch.equal(p,old[name])


@pytest.mark.parametrize('mutation',['spec_hash','checkpoint_hash','teacher','seed','rank','population','partition','affected','parent','incomplete'])
def test_composition_rejects_mismatched_source_before_parent_scoring(source,tmp_path,mutation):
    c=source.config
    if mutation=='spec_hash':c['composition']['source_run_spec_sha256']='0'*64
    elif mutation=='checkpoint_hash':c['composition']['checkpoint_manifest_sha256']='0'*64
    elif mutation=='teacher':
        with torch.no_grad():source.teacher.embed_tokens.weight.add_(.001)
    elif mutation=='seed':c['seeds']=[23]
    elif mutation=='rank':c['rank']=2
    elif mutation=='population':c['populations']['fit']['sha256']='d'*64
    elif mutation=='partition':c['composition']['expected_partition']=[[0,1],[2]]
    elif mutation=='affected':c['pair']=[0,2]
    elif mutation=='parent':c['composition']['forced_parent']=1
    else:
        state=replace(source.state,step=1)
        ref=save_checkpoint(state,tmp_path/'changed')
        c['composition']['checkpoint_directory']=ref.directory;c['composition']['checkpoint_manifest_sha256']=ref.manifest_sha256
    rec=MemoryRecorder()
    with pytest.raises(ValueError):api().initialize_composition_recovery(source.teacher,c,source.fit,(batch(),),17,recorder=rec)
    assert rec.totals.search_proposals==0


def test_composition_binding_is_closed_and_reference_is_explicit(source):
    a=api();s=a.CompositionSourceSpec(**source.config['composition'])
    assert s.reference=='current_continuation'
    for changes in ({'forced_parent':True},{'reference':'svd'},{'merge_pair':[2,0]},{'schema_version':2},{'source_run_spec_sha256':'bad'}):
        with pytest.raises(ValueError):a.CompositionSourceSpec(**dict(source.config['composition'],**changes))
    with pytest.raises((TypeError,ValueError)):a.CompositionSourceSpec(**dict(source.config['composition'],extra=1))


def test_three_use_joint_recovery_has_fresh_optimizer_exact_counts_and_resume(source,tmp_path):
    from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash
    from poc.recovery_study import NaturalWindowPlan
    from poc.checkpoint import load_checkpoint
    from copy import deepcopy
    trial=api().initialize_composition_recovery(source.teacher,source.config,source.fit,(batch(),),17)
    model=trial.model;whole=deepcopy(model)
    plan=NaturalWindowPlan.build((batch(512),),updates=2,seed=17,manifest_hashes={'owned':'a'*64})
    obj=RecoveryObjectiveSpec('full_student','affected_group',(0,1,2),plan.digest,owner_names_hash(model,(0,1,2)))
    cfg=replace(bound_config(obj,2),affected_depths=(0,1,2))
    cfg=replace(cfg,run_spec=replace(cfg.run_spec,resolved_config={'training':cfg.identity(),'recovery_objective':obj.identity()},parent_run_ids=(run_id(source.spec),)))
    rec=MemoryRecorder();first=fit_model(model,source.teacher,TrainingSources(raw_plan=plan),replace(cfg,stop_after_updates=1),budget(),rec,objective_spec=obj)
    assert first.state.step==1 and first.state.target_cursor==8192
    assert first.state.optimizer['state'] and all(int(v['step'])==1 for v in first.state.optimizer['state'].values())
    ref=save_checkpoint(first.state,tmp_path/'joint');resumed=fit_model(model,source.teacher,TrainingSources(raw_plan=plan),replace(cfg,resume=load_checkpoint(ref,cfg.run_spec)),budget(),rec,objective_spec=obj)
    direct=fit_model(whole,source.teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=obj)
    assert resumed.state.events==direct.state.events
    assert rec.totals.raw_corpus_target_presentations==rec.totals.valid_target_presentations==16384
    assert all(torch.equal(p,dict(whole.named_parameters())[n]) for n,p in model.named_parameters())


def test_nested_composition_work_counts_wall_time_once(source,monkeypatch):
    from poc.contracts import CostRecord
    module=api();original=module.paired_composition_initializations
    def instrument(*args,recorder=None,**kwargs):
        recorder.record(CostRecord(wall_seconds=10))
        return original(*args,recorder=recorder,**kwargs)
    monkeypatch.setattr(module,'paired_composition_initializations',instrument)
    rec=MemoryRecorder()
    module.initialize_composition_recovery(source.teacher,source.config,source.fit,(batch(),),17,recorder=rec)
    assert rec.totals.wall_seconds<10,'nested wall measurements must be replaced by the outer elapsed interval'
    assert rec.totals.search_proposals==2


@pytest.mark.mps
def test_mps_completed_source_then_three_use_joint_update_keeps_exact_lineage(tmp_path):
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    from poc.bank import make_bank,canonicalize_partition,PROJECTIONS
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.recovery_study import NaturalWindowPlan
    from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash,affected_owners
    from poc.train import TrainingConfig
    from test_records import spec as run_spec
    teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024),seed=3,dtype=torch.float32)
    g=torch.Generator().manual_seed(991)
    with torch.no_grad():
        teacher.embed_tokens.weight.copy_(torch.randn(teacher.embed_tokens.weight.shape,generator=g)*.4)
        for layer in teacher.layers:
            for name in PROJECTIONS:
                weight=getattr(layer.mlp,name).weight;weight.copy_(torch.randn(weight.shape,generator=g)*.2)
    teacher=teacher.to('mps');teacher.tokenizer_digest='a'*64
    student=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,17)
    plan=NaturalWindowPlan.build((batch(512),),updates=1,seed=17,manifest_hashes={'owned':'a'*64})
    objective=RecoveryObjectiveSpec('full_student','affected_group',(0,1),plan.digest,owner_names_hash(student,(0,1)))
    cfg=TrainingConfig(max_updates=1,alpha=0,affected_depths=(0,1),kl_floor_per_target=2e-6,numerical_profile='owned-mps-fp32')
    identity=run_spec(budgets=budget(),resolved_config={'training':cfg.identity(),'recovery_objective':objective.identity()},input_manifests={'assets':'a'*64,'fit':'b'*64,'tuning':'c'*64,'raw_plan':plan.digest,'owner_names':objective.owner_names_sha256},seeds={'fit':17},hardware_policy={'backend':'mps','dtype':'float32'})
    cfg=replace(cfg,run_spec=identity)
    result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=objective)
    ref=save_checkpoint(result.state,tmp_path/'source');spec=tmp_path/'spec.json';spec.write_bytes(canonical_json(identity.to_dict()))
    c=dict(pair=[0,1,2],rank=1,seeds=[17],parent_fit_components=1,populations={'fit':{'sha256':'b'*64},'tuning':{'sha256':'c'*64}},composition=dict(schema_version=1,source_run_spec_path=str(spec),source_run_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),checkpoint_directory=ref.directory,checkpoint_manifest_sha256=ref.manifest_sha256,merge_pair=[0,2],forced_parent=0,reference='current_continuation',expected_partition=[[0,1,2]]))
    fit=DataManifest((DocumentRecord('fit-doc','owned','a'*40,'owned',0,text='owned'),),{'fit-doc':'fit'},{'fit-doc':'fit-component'},'owned')
    trial=api().initialize_composition_recovery(teacher,c,fit,(batch(512),),17)
    assert trial.initializer_record['source_candidate_digest']==result.state.candidate_digest
    joint=trial.model;owners=affected_owners(joint,(0,1,2));assert len(owners)==21
    frozen={n:p.detach().cpu().clone() for n,p in joint.named_parameters() if n not in owners}
    obj=replace(objective,affected_depths=(0,1,2),owner_names_sha256=owner_names_hash(joint,(0,1,2)))
    jc=replace(cfg,affected_depths=(0,1,2));js=replace(identity,resolved_config={'training':jc.identity(),'recovery_objective':obj.identity()},input_manifests=dict(identity.input_manifests,owner_names=obj.owner_names_sha256),parent_run_ids=(run_id(identity),));jc=replace(jc,run_spec=js)
    rec=MemoryRecorder();done=fit_model(joint,teacher,TrainingSources(raw_plan=plan),jc,budget(),rec,objective_spec=obj)
    assert done.state.step==1 and rec.totals.raw_corpus_target_presentations==8192
    assert rec.totals.valid_target_presentations==8192
    for n,p in joint.named_parameters():
        if n in frozen:assert torch.equal(p.detach().cpu(),frozen[n])
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())

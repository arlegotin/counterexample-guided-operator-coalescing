"""Study5 is an explicit complete-group continuation, not a legacy pair clone."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib,json

import pytest,yaml,torch

from test_recovery_v4 import config as v4_config,load
from test_recovery_v4_driver import driver
from test_recovery_composition import source
from test_training import batch,budget,MemoryRecorder


@pytest.fixture
def config(v4_config):
    c=deepcopy(v4_config)
    c.update(retained_dependency_roots=[],schema_version=5,study='owned-composition-v5',pair=[0,15,20],seeds=[17],initializer=None,
        composition=dict(schema_version=1,source_run_spec_path=None,source_run_spec_sha256=None,
            checkpoint_directory=None,checkpoint_manifest_sha256=None,merge_pair=[0,20],forced_parent=0,
            reference='current_continuation',expected_partition=[[0,15,20],*[[i] for i in range(1,30) if i not in (15,20)]]))
    c['profile_resource_budget']['valid_target_presentations']=122752
    return c


def test_schema5_accepts_explicit_three_use_group_and_unresolved_source(tmp_path,config):
    assert load(tmp_path,config)==config


@pytest.mark.parametrize('scope,minimum',[('full_student',106368),('local_sum',122752),('local_mean',122752)])
def test_schema5_profile_budget_counts_three_local_uses(tmp_path,config,scope,minimum):
    config['loss_scopes']=[scope];config['profile_resource_budget']['valid_target_presentations']=minimum
    assert load(tmp_path,config)==config
    config['profile_resource_budget']['valid_target_presentations']-=1
    with pytest.raises(ValueError,match='profile.*budget|budget.*profile'):load(tmp_path,config)


@pytest.mark.parametrize('mutation',['unknown','missing','initializer','clone','seeds','pair','partition','partial','reference','parent'])
def test_schema5_rejects_ambiguous_or_partial_group_contract(tmp_path,config,mutation):
    if mutation=='unknown':config['composition']['extra']=1
    elif mutation=='missing':del config['composition']
    elif mutation=='initializer':config['initializer']={'method':'parent_svd'}
    elif mutation=='clone':config['initialization_checkpoint']={'checkpoint_directory':'/old'}
    elif mutation=='seeds':config['seeds']=[17,23]
    elif mutation=='pair':config['pair']=[0,20,15]
    elif mutation=='partition':config['composition']['expected_partition']=[[0,15],[20],*[[i] for i in range(1,30) if i not in (15,20)]]
    elif mutation=='partial':config['pair']=[0,20]
    elif mutation=='reference':config['composition']['reference']='teacher_svd'
    elif mutation=='parent':config['composition']['forced_parent']=True
    with pytest.raises((ValueError,TypeError)):load(tmp_path,config)


def test_schema5_preflight_has_correct_K28_bytes_and_unresolved_source(config,monkeypatch):
    m=driver();monkeypatch.setattr(m,'output_root',lambda c:Path('/owned/new-study'))
    result=m.preflight(config)
    assert 'composition.source_run_spec' in result['unresolved']
    assert 'composition.checkpoint_manifest' in result['unresolved']
    assert result['byte_preflight']['K']==28
    assert result['byte_preflight']['fp32_owner_tensor_bytes']==4*(54888768+2654208*28+190080*8)


def test_schema5_real_shape_gate_reports_composition_gap_instead_of_requiring_scalar_parity(source):
    m=driver();from poc.recovery_composition import initialize_composition_recovery
    trial=initialize_composition_recovery(source.teacher,source.config,source.fit,(batch(),),17)
    c={'schema_version':5,'pair':[0,1,2],'rank':1,'seeds':[17],
       'numerical_profile':{'kl_floor_per_target':2e-6,'decision_resolution':2e-5}}
    from poc.losses import kl_sum
    b=batch();target=source.teacher.logits(b)[:,:-1]
    full=float((kl_sum(target,trial.model.logits(b)[:,:-1],b.target_mask)/8).detach())
    local=0.
    for depth in (0,1,2):
        boundary=source.teacher.prefix_boundary(b,depth)
        t=source.teacher.suffix_logits(source.teacher.ff_residual(boundary),boundary)[:,:-1]
        value=source.teacher.suffix_logits(source.teacher.ff_residual(boundary,trial.model.layers[depth].mlp),boundary)[:,:-1]
        local+=float((kl_sum(t,value,b.target_mask)/8).detach())
    assert abs(local-full)>c['numerical_profile']['decision_resolution'],'control must exhibit an actual composition gap'
    result=m.real_shape_controls(c,source.teacher,trial.model,b,MemoryRecorder())
    assert result['passed'] and result['initial_gap_interpretation']=='composition diagnostic; equality is not a numerical correctness requirement'
    assert result['initial_local_full_gap_abs'] is None,'resource profiles must not disclose candidate composition quality'


def test_schema5_initialize_and_runspec_bind_completed_source_and_fresh_owner_scope(source,tmp_path,monkeypatch):
    from poc.recovery_study import NaturalWindowPlan
    m=driver();c=dict(source.config,study='owned-v5',checkpoint_every=16,owner_scope='affected_group',observations=[0,1,2],updates=2,
        numerical_profile={'device':'cpu','kl_floor_per_target':2e-6},resource_budget=None)
    trial=m.initialize(c,source.teacher,source.fit,(batch(),),17,MemoryRecorder())
    assert trial.affected_depths==(0,1,2)
    monkeypatch.setattr(m,'output_root',lambda c:tmp_path);monkeypatch.setattr(m,'ASSET_SHA',source.teacher.tokenizer_digest)
    (tmp_path/'controls.json').write_text('{}')
    plan=NaturalWindowPlan.build((batch(512),),updates=2,seed=17,manifest_hashes={'owned':'a'*64})
    cfg,obj=m.make_spec(c,'full_student',17,plan,trial.model,{'commit':'a'*40},{'owned':True},budget(),initializer_record=trial.initializer_record)
    assert cfg.run_spec.parent_run_ids==(trial.initializer_record['source_run_id'],)
    assert cfg.run_spec.input_manifests['composition_checkpoint']==c['composition']['checkpoint_manifest_sha256']
    assert obj.affected_depths==(0,1,2)
    trial.initializer_record['reference']='teacher_reset'
    with pytest.raises(ValueError,match='composition'):m.make_spec(c,'full_student',17,plan,trial.model,{'commit':'a'*40},{'owned':True},budget(),initializer_record=trial.initializer_record)


def test_schema5_numerical_gate_rejects_large_negative_candidate_kl(source,monkeypatch):
    import poc.losses
    from poc.recovery_composition import initialize_composition_recovery
    m=driver();trial=initialize_composition_recovery(source.teacher,source.config,source.fit,(batch(),),17)
    original=poc.losses.stream_metrics;calls=0
    def broken(*args,**kwargs):
        nonlocal calls
        value=original(*args,**kwargs);calls+=1
        return replace(value,kl=-value.valid_targets) if calls==2 else value
    monkeypatch.setattr(poc.losses,'stream_metrics',broken)
    c={'schema_version':5,'pair':[0,1,2],'rank':1,'seeds':[17],
       'numerical_profile':{'kl_floor_per_target':2e-6,'decision_resolution':2e-5}}
    with pytest.raises(ValueError,match='numerical|floor|finite'):
        m.real_shape_controls(c,source.teacher,trial.model,batch(),MemoryRecorder())


def test_schema5_profile_run_and_safe_resume_use_real_common_pipeline(tmp_path,monkeypatch):
    from copy import deepcopy
    from poc.bank import make_bank,canonicalize_partition
    from poc.checkpoint import save_checkpoint
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash
    from poc.recovery_study import NaturalWindowPlan
    from poc.train import TrainingConfig,TrainingSources,fit_model
    from poc.records import canonical_json,run_id
    from poc.data.manifest import DataManifest,DocumentRecord
    from test_records import spec as run_spec
    import poc.teacher
    m=driver();monkeypatch.setattr(m,'ASSET_SHA','a'*64)
    teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024),seed=3,dtype=torch.float32)
    teacher.tokenizer_digest='a'*64
    windows=tuple(replace(batch(512),document_ids=(f'fit{i}',),cluster_ids=(f'component{i}',)) for i in range(2))
    tuning=(replace(batch(512),document_ids=('dev',),cluster_ids=('dev-component',)),)
    fit=DataManifest(tuple(DocumentRecord(f'fit{i}','owned','a'*40,'owned',i,text='owned') for i in range(2)),{f'fit{i}':'fit' for i in range(2)},{f'fit{i}':f'component{i}' for i in range(2)},'owned')
    plan=NaturalWindowPlan.build(windows,updates=1,seed=17,manifest_hashes={'owned':'a'*64})
    student=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,17)
    obj=RecoveryObjectiveSpec('full_student','affected_group',(0,1),plan.digest,owner_names_hash(student,(0,1)))
    cfg=TrainingConfig(max_updates=1,alpha=0,affected_depths=(0,1),kl_floor_per_target=2e-6,numerical_profile='owned-cpu-fp32')
    identity=run_spec(budgets=budget(),resolved_config={'training':cfg.identity(),'recovery_objective':obj.identity()},input_manifests={'assets':'a'*64,'fit':'b'*64,'tuning':'c'*64,'raw_plan':plan.digest,'owner_names':obj.owner_names_sha256},seeds={'fit':17})
    cfg=replace(cfg,run_spec=identity)
    result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=obj)
    dependency=tmp_path/'closed-source';dependency.mkdir()
    retained_receipt=dependency/'retained-cost-receipt.json';retained_receipt.write_bytes(b'owned retained source receipt')
    ref=save_checkpoint(result.state,dependency/'resume');spec_path=dependency/'spec.json';spec_path.write_bytes(canonical_json(identity.to_dict()))
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/composition-recovery-v1.yaml').read_text())
    c.update(asset_root=str(tmp_path),rank=1,pair=[0,1,2],parent_fit_components=2,updates=1,observations=[0,1],seeds=[17],retained_dependency_roots=[str(dependency)],populations={'fit':{'path':str(tmp_path/'fit'),'sha256':'b'*64,'components':2},'tuning':{'path':str(tmp_path/'dev'),'sha256':'c'*64,'components':1}})
    c['numerical_profile']['device']='cpu'
    c['composition']=dict(schema_version=1,source_run_spec_path=str(spec_path),source_run_spec_sha256=m.sha(spec_path),checkpoint_directory=ref.directory,checkpoint_manifest_sha256=ref.manifest_sha256,merge_pair=[0,2],forced_parent=0,reference='current_continuation',expected_partition=[[0,1,2]])
    c['profile_resource_budget'].update(memory_bytes=4*1024**3,disk_bytes=2*1024**3)
    monkeypatch.setattr(m,'load_inputs',lambda c:(object(),fit,windows,tuning,{17:plan}))
    monkeypatch.setattr(poc.teacher,'load_teacher',lambda a:deepcopy(teacher))
    out=tmp_path/'campaign';out.mkdir();monkeypatch.setattr(m,'output_root',lambda c:out)
    path=tmp_path/'config.json';path.write_text(json.dumps(c));sources=m.executed_sources(path)
    from poc.recovery_controls import qualify_backend
    m.write_new(out/'controls.json',{'backend':qualify_backend('cpu'),'implementation_sha256':sources['implementation_sha256'],'environment_sha256':m.hashed(m.environment())})
    profile=m.execute(c,path,'profile')
    assert profile['retained_family']['dependencies']==[str(dependency)]
    assert profile['retained_dependency_bytes']==sum(p.stat().st_size for p in dependency.rglob('*') if p.is_file())
    assert profile['arms']['full_student']['numerical_controls']['passed']
    assert not list(out.glob('profile-attempt-*/**/resume/generation-*'))
    b=budget().to_dict();b.update(updates=1,wall_seconds=3000,valid_target_presentations=200000,memory_bytes=4*1024**3,disk_bytes=2*1024**3)
    c['resource_budget']={'frozen_profile_sha256':m.sha(out/'profile.json'),'per_arm':{'full_student':b}};path.write_text(json.dumps(c))
    retained_receipt.unlink()
    original_inputs=m.load_inputs
    def forbidden_inputs(*args,**kwargs):raise AssertionError('dependency mismatch must reject before data/model I/O')
    monkeypatch.setattr(m,'load_inputs',forbidden_inputs)
    with pytest.raises(ValueError,match='retained.*family|dependency.*inventory'):
        m.execute(c,path,'run')
    monkeypatch.setattr(m,'load_inputs',original_inputs)
    retained_receipt.write_bytes(b'owned retained source receipt')
    observe=m.observation;failed=False
    def interrupt(*args,**kwargs):
        nonlocal failed
        if args[5]==1 and not failed:failed=True;raise InterruptedError('owned post-checkpoint interruption')
        return observe(*args,**kwargs)
    monkeypatch.setattr(m,'observation',interrupt)
    with pytest.raises(InterruptedError):m.execute(c,path,'run')
    monkeypatch.setattr(m,'observation',observe);m.execute(c,path,'run',resume=True)
    arm=out/'runs/full_student-seed17';final=json.loads((arm/'result.json').read_text())
    assert final['step']==1 and final['raw_target_cursor']==8192
    assert (arm/'observations/0000.json').exists() and (arm/'observations/0001.json').exists()
    saved=json.loads((arm/'spec.json').read_text())
    assert saved['parent_run_ids']==[run_id(identity)]
    assert saved['input_manifests']['composition_checkpoint']==ref.manifest_sha256

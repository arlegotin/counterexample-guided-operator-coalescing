"""Owned source→disjoint merge→joint fitting lifecycle; no pretrained assets."""
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import hashlib,json,importlib.util
import pytest,torch
from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
from poc.bank import make_bank,canonicalize_partition,PROJECTIONS
from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash,affected_owners
from poc.recovery_study import NaturalWindowPlan
from poc.train import TrainingConfig,TrainingSources,fit_model,configure_trainable_owners
from poc.records import canonical_json,run_id
from poc.checkpoint import save_checkpoint,load_checkpoint,CheckpointRef,_spec
from poc.data.manifest import DataManifest,DocumentRecord
from test_training import batch,budget,MemoryRecorder
from test_records import spec as owned_spec

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def driver():
 s=importlib.util.spec_from_file_location('joint_lifecycle_driver',ROOT/'provenance/recovery_diagnosis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.fixture
def source(tmp_path):
 teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024),seed=3,dtype=torch.float32);teacher.tokenizer_digest=driver().ASSET_SHA
 g=torch.Generator().manual_seed(991)
 with torch.no_grad():
  teacher.embed_tokens.weight.copy_(torch.randn(teacher.embed_tokens.weight.shape,generator=g)*.4)
  for layer in teacher.layers:
   for name in PROJECTIONS:
    weight=getattr(layer.mlp,name).weight;weight.copy_(torch.randn(weight.shape,generator=g)*.2)
 partition=canonicalize_partition([[0,1],[2],[3],[4],[5]],6);student=make_bank(teacher,partition,1,17);b=replace(batch(512),document_ids=('fit-doc',),cluster_ids=('fit-component',));plan=NaturalWindowPlan.build((b,),updates=2,seed=17,manifest_hashes={'fit':'b'*64,'tokenizer':teacher.tokenizer_digest})
 obj=RecoveryObjectiveSpec('full_student','affected_group',(0,1),plan.digest,owner_names_hash(student,(0,1)))
 cfg=TrainingConfig(max_updates=2,alpha=0,affected_depths=(0,1),record_norms=True,kl_floor_per_target=2e-6,numerical_profile='recovery-diagnosis-fp32-v1')
 spec=replace(owned_spec(budgets=budget()),resolved_config={'training':cfg.identity(),'recovery_objective':obj.identity()},input_manifests={'assets':teacher.tokenizer_digest,'fit':'b'*64,'tuning':'c'*64,'raw_plan':plan.digest,'owner_names':obj.owner_names_sha256},seeds={'fit':17});cfg=replace(cfg,run_spec=spec)
 result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=obj)
 dep=tmp_path/'completed-source';dep.mkdir();ref=save_checkpoint(result.state,dep/'safe');sp=dep/'spec.json';sp.write_bytes(canonical_json(spec.to_dict()))
 binding={'schema_version':1,'source_run_spec_path':str(sp),'source_run_spec_sha256':sha(sp),'checkpoint_directory':ref.directory,'checkpoint_manifest_sha256':ref.manifest_sha256,'merge_pair':[2,3],'forced_parent':2,'reference':'current_continuation','expected_partition':[[0,1],[2,3],[4],[5]]}
 config={'schema_version':6,'study':'owned-joint-source','pair':[2,3],'trainable_depths':[0,1,2,3],'owner_scope':'affected_groups','rank':1,'seeds':[17],'parent_fit_components':1,'composition':binding,'populations':{'fit':{'sha256':'b'*64},'tuning':{'sha256':'c'*64}}}
 fit=DataManifest((DocumentRecord('fit-doc','owned','a'*40,'owned',0,text='owned'),),{'fit-doc':'fit'},{'fit-doc':'fit-component'},'owned')
 return SimpleNamespace(teacher=teacher,student=student,state=result.state,spec=spec,config=config,fit=fit,batch=b,plan=plan,dependency=dep)

def initialize(source):
 from poc.recovery_composition import initialize_composition_recovery
 return initialize_composition_recovery(source.teacher,source.config,source.fit,(source.batch,),17)

@pytest.mark.parametrize('reference',['teacher_reset','current_continuation'])
def test_new_merge_record_separates_initialization_and_recovery_sites(source,reference):
 s=source;s.config['composition']['reference']=reference;before=s.student.manifest().checkpoint_digest;trial=initialize(s);r=trial.initializer_record
 assert trial.affected_depths==(2,3) and r['affected_depths']==[2,3]
 assert r['schema_version']==2 and r['trainable_depths']==[0,1,2,3] and r['owner_scope']=='affected_groups'
 assert r['owner_names_sha256']==owner_names_hash(trial.model,(0,1,2,3),owner_scope='affected_groups')
 assert s.student.manifest().checkpoint_digest==before
 old=dict(s.student.named_parameters())
 for n,p in trial.model.named_parameters():
  if n.startswith('bank.') or n.startswith(('layers.2.mlp.','layers.3.mlp.')):continue
  assert torch.equal(p,old[n])

@pytest.mark.parametrize('bad',['partial','missing_new','singleton','unknown','wrong_scope'])
def test_invalid_recovery_scope_denied_before_parent_scoring(source,bad):
 from poc.recovery_composition import initialize_composition_recovery
 s=source
 if bad=='partial':s.config['trainable_depths']=[0,2,3]
 elif bad=='missing_new':s.config['trainable_depths']=[0,1]
 elif bad=='singleton':s.config['trainable_depths']=[0,1,2,3,4]
 elif bad=='unknown':s.config['trainable_depths']=[0,1,2,3,29]
 else:s.config['owner_scope']='affected_group'
 rec=MemoryRecorder()
 with pytest.raises(ValueError):initialize_composition_recovery(s.teacher,s.config,s.fit,(s.batch,),17,recorder=rec)
 assert rec.totals.search_proposals==0


def test_same_initial_tensor_state_for_joint_and_new_only_ownership(source):
 s=source;joint=initialize(s);s.config.update(owner_scope='affected_group',trainable_depths=[2,3]);single=initialize(s)
 assert single.initialized_digest==joint.initialized_digest
 assert len(affected_owners(joint.model,(0,1,2,3),owner_scope='affected_groups'))==30
 assert len(affected_owners(single.model,(2,3)))==15
 assert single.initializer_record['owner_names_sha256']!=joint.initializer_record['owner_names_sha256']

@pytest.mark.parametrize('reference,scope',[('current_continuation','affected_groups'),('teacher_reset','affected_groups'),('current_continuation','affected_group')])
def test_full_driver_profile_run_interruption_resume_and_frozen_old_group(source,tmp_path,monkeypatch,reference,scope):
 import yaml,poc.teacher
 from poc.recovery_study import load_study_config
 s=source;d=driver();out=tmp_path/'recovery';cfg=load_study_config(ROOT/'configs/composition-recovery-v1.yaml');cfg.update(s.config);cfg.update(study='owned-joint-'+reference+'-'+scope,out=str(out),asset_root=str(tmp_path),updates=2,observations=[0,1,2],checkpoint_every=1,observation_policy='optimizer_plus_tuning_v2',resource_budget=None,retained_dependency_roots=[str(s.dependency)],profile_resource_budget=budget(updates=1,wall_seconds=600,valid_target_presentations=4000000,memory_bytes=4*1024**3,disk_bytes=2*1024**3,search_proposals=10).to_dict());cfg['composition']=dict(s.config['composition'],reference=reference);cfg['owner_scope']=scope;cfg['trainable_depths']=[0,1,2,3] if scope=='affected_groups' else [2,3];cfg['populations']={'fit':{'path':'owned','sha256':'b'*64,'components':1},'tuning':{'path':'owned-dev','sha256':'c'*64,'components':1}};cfg['numerical_profile']['device']='cpu';dev=replace(s.batch,document_ids=('dev',),cluster_ids=('dev-component',))
 monkeypatch.setattr(d,'load_inputs',lambda c:(object(),s.fit,(s.batch,),(dev,),{17:s.plan}));monkeypatch.setattr(poc.teacher,'load_teacher',lambda a:deepcopy(s.teacher));monkeypatch.setattr(d,'output_root',lambda c:out);out.mkdir();path=tmp_path/'study.yaml';path.write_text(yaml.safe_dump(cfg));code=d.executed_sources(path);d.write_new(out/'controls.json',{'backend':{'passed':True,'device':'cpu'},'implementation_sha256':code['implementation_sha256'],'environment_sha256':d.hashed(d.environment())})
 profile=d.execute(cfg,path,'profile');assert profile['arms']['full_student']['planned_updates']==2
 cap=budget(updates=2,wall_seconds=3000,valid_target_presentations=4000000,memory_bytes=4*1024**3,disk_bytes=2*1024**3,search_proposals=10).to_dict();cfg['resource_budget']={'frozen_profile_sha256':sha(out/'profile.json'),'per_arm':{'full_student':cap}};path.write_text(yaml.safe_dump(cfg));observe=d.observation;called={'once':False}
 def interrupt(*a,**k):
  if a[5]==1 and not called['once']:called['once']=True;raise InterruptedError('owned safe-boundary interruption')
  return observe(*a,**k)
 monkeypatch.setattr(d,'observation',interrupt)
 with pytest.raises(InterruptedError):d.execute(cfg,path,'run')
 monkeypatch.setattr(d,'observation',observe);d.execute(cfg,path,'run',resume=True)
 arm=out/'runs/full_student-seed17';value=json.loads((arm/'result.json').read_text());identity=_spec(json.loads((arm/'spec.json').read_text()));state=load_checkpoint(CheckpointRef(**value['checkpoint']),identity);assert state.step==2 and state.target_cursor==16384
 expected=30 if scope=='affected_groups' else 15;assert len(state.optimizer['state'])==expected and all(int(v['step'])==2 for v in state.optimizer['state'].values())
 delta=json.loads((arm/'deltas/0002/manifest.json').read_text());assert len(delta['owners'])==expected
 observation=json.loads((arm/'observations/0002.json').read_text());assert all(len(row['local_by_depth'])==len(cfg['trainable_depths']) for row in observation['populations']['optimizer_windows'])
 # The same initialized weights and complete raw plan reproduce the resumed
 # endpoint through the common fitter, without the observer or imported moments.
 trial=__import__('poc.recovery_composition',fromlist=['initialize_composition_recovery']).initialize_composition_recovery(s.teacher,cfg,s.fit,(s.batch,),17)
 train,obj=d.make_spec(cfg,'full_student',17,s.plan,trial.model,d.executed_sources(path),d.environment(),__import__('poc.contracts',fromlist=['Budget']).Budget(**cap),initializer_record=trial.initializer_record)
 direct=fit_model(trial.model,s.teacher,TrainingSources(raw_plan=s.plan),train,__import__('poc.contracts',fromlist=['Budget']).Budget(**cap),MemoryRecorder(),objective_spec=obj)
 assert state.events==direct.state.events and all(torch.equal(v,direct.state.model[n]) for n,v in state.model.items())
 if scope=='affected_group':
  for n in ('bank.0.gate_proj.weight','layers.0.mlp.residuals.gate_proj.A'):
   assert torch.equal(state.model[n],s.state.model[n])
 else:assert not torch.equal(state.model['bank.0.gate_proj.weight'],s.state.model['bank.0.gate_proj.weight'])

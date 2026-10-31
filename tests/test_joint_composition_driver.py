"""The study driver binds recovery groups separately from merge initialization."""
from pathlib import Path
from dataclasses import replace
import importlib.util,json
from types import SimpleNamespace
import pytest,torch
from test_joint_recovery import setup
from test_training import budget,MemoryRecorder,batch
from poc.recovery_objectives import affected_owners,owner_names_hash
from poc.records import canonical_json

ROOT=Path(__file__).resolve().parents[1]

def driver():
 s=importlib.util.spec_from_file_location('owned_joint_composition_driver',ROOT/'provenance/recovery_diagnosis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.fixture
def case(tmp_path):
 teacher,student,plan,obj,cfg=setup();m=driver();out=tmp_path/'study';out.mkdir();(out/'controls.json').write_text('owned source-control stub; only make_spec under test');m.output_root=lambda config:out
 selected=(0,1,2,3);owner_scope='affected_groups'
 source={'source_run_spec_path':'/owned/completed-spec','source_run_spec_sha256':'a'*64,'checkpoint_directory':'/owned/safe','checkpoint_manifest_sha256':'b'*64,'merge_pair':[2,3],'forced_parent':2,'reference':'current_continuation','expected_partition':[[0,1],[2,3],[4],[5]],'schema_version':1}
 config={'schema_version':6,'study':'owned-joint-composition','out':str(out),'pair':[2,3],'trainable_depths':list(selected),'owner_scope':owner_scope,'rank':1,'seeds':[17],'updates':2,'alpha':0,'raw_targets_per_update':8192,'observations':[0,1,2],'checkpoint_every':1,'loss_scopes':['full_student'],'parent_fit_components':1,'populations':{'fit':{'sha256':'d'*64,'components':1},'tuning':{'sha256':'e'*64,'components':1}},'numerical_profile':{'device':'cpu','dtype':'float32','chunk_size':32,'kl_floor_per_target':2e-6,'decision_resolution':2e-5},'resource_budget':None,'composition':source,'retained_dependency_roots':['/owned']}
 record={'schema_version':2,'kind':'completed_source_composition_v1','after_digest':student.manifest().checkpoint_digest,'rank':1,'seed':17,'teacher_digest':student.base_digest,'affected_depths':[2,3],'trainable_depths':list(selected),'owner_scope':owner_scope,'owner_names_sha256':owner_names_hash(student,selected,owner_scope=owner_scope),'reference':source['reference'],'source_binding':source,'source_checkpoint_manifest_sha256':'b'*64,'source_spec_sha256':'a'*64,'source_run_id':'a'*64,'source_candidate_digest':'c'*64,'composition':{'source_current_digest':'c'*64}}
 return SimpleNamespace(teacher=teacher,student=student,plan=plan,config=config,record=record,driver=m,out=out)

def make(case):
 c=case
 return c.driver.make_spec(c.config,'full_student',17,c.plan,c.student,{'commit':'f'*40},{'owned':True},budget(),initializer_record=c.record)

def test_make_spec_selects_both_complete_groups_and_binds_source(case):
 c=case;cfg,obj=make(c);assert obj.schema_version==3 and obj.owner_scope=='affected_groups' and obj.affected_depths==(0,1,2,3)
 assert cfg.affected_depths==(0,1,2,3) and obj.raw_plan_sha256==c.plan.digest
 assert cfg.run_spec.input_manifests['owner_names']==owner_names_hash(c.student,(0,1,2,3),owner_scope='affected_groups')
 assert cfg.run_spec.parent_run_ids==('a'*64,) and cfg.run_spec.input_manifests['composition_source_spec']=='a'*64
 from poc.train import configure_trainable_owners
 selected=configure_trainable_owners(c.student,cfg,objective_spec=obj);assert len(selected)==30
 assert all(p.requires_grad==(n in selected) for n,p in c.student.named_parameters())

def test_new_merge_only_control_has_15_owners_and_distinct_identity(case):
 c=case;full_cfg,_=make(c);c.config.update(trainable_depths=[2,3],owner_scope='affected_group');c.record.update(trainable_depths=[2,3],owner_scope='affected_group',owner_names_sha256=owner_names_hash(c.student,(2,3)))
 cfg,obj=make(c);assert obj.schema_version==1 and obj.affected_depths==(2,3)
 assert len(affected_owners(c.student,obj.affected_depths))==15 and c.config['pair']==[2,3]
 assert cfg.run_spec!=full_cfg.run_spec and cfg.run_spec.input_manifests['owner_names']!=full_cfg.run_spec.input_manifests['owner_names']

@pytest.mark.parametrize('bad',['depths','scope','hash','missing'])
def test_stale_or_missing_initializer_recovery_binding_rejected(case,bad):
 c=case
 if bad=='depths':c.record['trainable_depths']=[2,3]
 elif bad=='scope':c.record['owner_scope']='affected_group'
 elif bad=='hash':c.record['owner_names_sha256']='0'*64
 else:c.record.pop('trainable_depths')
 with pytest.raises(ValueError,match='owner|recovery|composition'):make(c)

def test_compact_observation_delta_contains_all30_selected_owners(case,tmp_path):
 c=case;cfg,obj=make(c);path=tmp_path/'delta';c.driver.save_recovery_delta(c.student,'a'*64,cfg.run_spec,path,MemoryRecorder())
 from safetensors.torch import load_file
 values=load_file(str(path/'owners.safetensors'));assert set(values)==set(affected_owners(c.student,(0,1,2,3),owner_scope='affected_groups'))
 assert len(values)==30

@pytest.mark.parametrize('bad',[None,'negative','nonfinite'])
def test_composition_numeric_controls_keep_noop_gates_without_scalar_equality(case,monkeypatch,bad):
 import poc.losses,poc.probes
 c=case;n=8;calls={'n':0}
 def metric(*args):
  calls['n']+=1;value=(0 if calls['n']==1 else .01)
  if bad=='negative' and calls['n']==1:value=-1
  if bad=='nonfinite' and calls['n']==2:value=float('nan')
  return SimpleNamespace(kl=value*n)
 monkeypatch.setattr(poc.losses,'stream_metrics',metric);monkeypatch.setattr(poc.probes,'local_discrepancy',lambda *a,**k:SimpleNamespace(kl=.1*n))
 if bad:
  with pytest.raises(ValueError):c.driver.real_shape_controls(c.config,c.teacher,c.student,batch(9),MemoryRecorder())
 else:
  r=c.driver.real_shape_controls(c.config,c.teacher,c.student,batch(9),MemoryRecorder());assert r['passed'] and r['initial_local_full_gap_abs'] is None
  assert calls['n']==2

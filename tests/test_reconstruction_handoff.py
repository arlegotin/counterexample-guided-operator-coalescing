"""A completed MSE phase may initialize KL, without importing warmed moments."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import pytest
import torch
from poc.records import run_id
from poc.train import TrainingSources,fit_model
from poc.checkpoint import save_checkpoint
from test_reconstruction_warmstart import warm_setup
from test_recovery_study import setup,bound_config
from test_training import budget,MemoryRecorder
pytestmark=pytest.mark.cpu

def api():
 assert importlib.util.find_spec('poc.recovery_phases'),'verified reconstruction handoff missing'
 from poc import recovery_phases
 return recovery_phases

@pytest.fixture
def phase(tmp_path):
 a=api();teacher,student,plan,obj=warm_setup(updates=2)
 # setup adjusts the owned teacher context/tokenizer after bank construction.
 student.base_digest=teacher.manifest().checkpoint_digest
 initial=deepcopy(student)
 cfg=bound_config(obj,2);inputs=dict(cfg.run_spec.input_manifests,assets=teacher.tokenizer_digest,fit='b'*64,tuning='c'*64,initialized_state=initial.manifest().checkpoint_digest)
 cfg=replace(cfg,run_spec=replace(cfg.run_spec,input_manifests=inputs,seeds={'fit':17}))
 result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=obj);ref=save_checkpoint(result.state,tmp_path/'warm')
 _,_,recovery_plan,recovery_obj=setup('full_student',updates=1);recovery_cfg=bound_config(recovery_obj,1)
 recovery_cfg=replace(recovery_cfg,run_spec=replace(recovery_cfg.run_spec,seeds={'fit':17},input_manifests=dict(recovery_cfg.run_spec.input_manifests,assets=teacher.tokenizer_digest,fit='b'*64,tuning='c'*64,initialized_state=initial.manifest().checkpoint_digest)))
 spec=a.ReconstructionHandoffSpec(warm_run_id=run_id(cfg.run_spec),checkpoint_manifest_sha256=ref.manifest_sha256,warm_raw_plan_sha256=plan.digest,recovery_raw_plan_sha256=recovery_plan.digest,teacher_digest=teacher.manifest().checkpoint_digest,owner_names_sha256=obj.owner_names_sha256,initial_state_digest=initial.manifest().checkpoint_digest,warm_updates=2,seed=17)
 return dict(teacher=teacher,initial=initial,checkpoint=ref,warm_run_spec=cfg.run_spec,warm_plan=plan,recovery_config=recovery_cfg,recovery_objective=recovery_obj,handoff_spec=spec),student,recovery_plan,result

def test_completed_phase_handoff_matches_weights_and_kl_starts_fresh(phase):
 args,warmed,plan,warm_result=phase;a=api();before=args['initial'].manifest().checkpoint_digest
 handoff=a.prepare_reconstruction_handoff(**args)
 assert handoff.student is not args['initial'] and args['initial'].manifest().checkpoint_digest==before
 for name,p in warmed.named_parameters():torch.testing.assert_close(p,dict(handoff.student.named_parameters())[name],rtol=0,atol=0)
 assert handoff.config.resume is None
 assert handoff.config.run_spec.parent_run_ids==(run_id(args['warm_run_spec']),)
 assert handoff.record['retained_warm_raw_targets']==16384 and handoff.record['retained_warm_scored_targets']==32768
 assert handoff.record['optimizer_policy']=='fresh; no warm optimizer/RNG/cursors/spending imported'
 assert handoff.config.run_spec.input_manifests['initialized_state']==handoff.student.manifest().checkpoint_digest
 result=fit_model(handoff.student,args['teacher'],TrainingSources(raw_plan=plan),handoff.config,budget(),MemoryRecorder(),objective_spec=handoff.objective)
 assert result.state.step==1 and result.state.target_cursor==8192
 warm_steps=[int(v['step']) for v in warm_result.state.optimizer['state'].values() if 'step' in v]
 kl_steps=[int(v['step']) for v in result.state.optimizer['state'].values() if 'step' in v]
 assert warm_steps and set(warm_steps)=={2};assert kl_steps and set(kl_steps)=={1}

@pytest.mark.parametrize('field,value',[('warm_updates',3),('warm_raw_plan_sha256','e'*64),('recovery_raw_plan_sha256','e'*64),('owner_names_sha256','e'*64),('initial_state_digest','e'*64),('teacher_digest','e'*64),('seed',23),('warm_run_id','e'*64)])
def test_handoff_rejects_mismatched_phase_bindings_without_mutation(phase,field,value):
 args,_,_,_=phase;before=args['initial'].manifest().checkpoint_digest
 args=dict(args,handoff_spec=replace(args['handoff_spec'],**{field:value}))
 with pytest.raises(ValueError):api().prepare_reconstruction_handoff(**args)
 assert args['initial'].manifest().checkpoint_digest==before

def test_handoff_rejects_non_mse_source_and_nonkl_target(phase):
 args,_,_,_=phase
 source=replace(args['warm_run_spec'],resolved_config={'training':{},'recovery_objective':{'loss_scope':'full_student'}})
 with pytest.raises(ValueError):api().prepare_reconstruction_handoff(**dict(args,warm_run_spec=source))
 with pytest.raises(ValueError):api().prepare_reconstruction_handoff(**dict(args,recovery_objective=replace(args['recovery_objective'],schema_version=2,loss_scope='ffn_mse_sum')))

@pytest.mark.parametrize('change',['frozen_owner','norm_scalar'])
def test_handoff_rejects_valid_checkpoint_with_changed_frozen_state(phase,tmp_path,change):
 from poc.train import _saved_digest
 from poc.checkpoint import checkpoint_decoder
 args,warmed,_,result=phase
 state=replace(result.state,model={n:t.clone() for n,t in result.state.model.items()},model_schema=deepcopy(result.state.model_schema))
 digest_model=deepcopy(warmed)
 if change=='frozen_owner':state.model['bank.1.gate_proj.weight'].add_(.01)
 else:
  state.model_schema['norm_epsilons']['norm']*=2
  digest_model.norm.variance_epsilon=state.model_schema['norm_epsilons']['norm']
 state.candidate_digest=_saved_digest(digest_model,state.model)
 checkpoint_decoder(state) # a valid, self-consistent checkpoint, not a broken hash
 ref=save_checkpoint(state,tmp_path/'changed')
 spec=replace(args['handoff_spec'],checkpoint_manifest_sha256=ref.manifest_sha256)
 with pytest.raises(ValueError,match='frozen owner|scalar|schema'):
  api().prepare_reconstruction_handoff(**dict(args,checkpoint=ref,handoff_spec=spec))


def test_handoff_rejects_incomplete_safe_phase(phase,tmp_path):
 from poc.train import _schedule_state
 args,_,_,result=phase
 state=replace(result.state,step=1,target_cursor=8192,scheduler=_schedule_state(1,2))
 ref=save_checkpoint(state,tmp_path/'partial');spec=replace(args['handoff_spec'],checkpoint_manifest_sha256=ref.manifest_sha256)
 with pytest.raises(ValueError,match='incomplete'):
  api().prepare_reconstruction_handoff(**dict(args,checkpoint=ref,handoff_spec=spec))


def test_kl_phase_after_handoff_resumes_exactly(phase,tmp_path):
 from poc.checkpoint import load_checkpoint
 args,_,_,_=phase
 _,_,plan,obj=setup('full_student',updates=2);cfg=bound_config(obj,2)
 cfg=replace(cfg,run_spec=replace(cfg.run_spec,seeds={'fit':17},input_manifests=dict(cfg.run_spec.input_manifests,assets='a'*64,fit='b'*64,tuning='c'*64,initialized_state=args['initial'].manifest().checkpoint_digest)))
 args=dict(args,recovery_config=cfg,recovery_objective=obj,handoff_spec=replace(args['handoff_spec'],recovery_raw_plan_sha256=plan.digest))
 handoff=api().prepare_reconstruction_handoff(**args);whole=api().prepare_reconstruction_handoff(**args);data=TrainingSources(raw_plan=plan)
 rec=MemoryRecorder();first=fit_model(handoff.student,args['teacher'],data,replace(handoff.config,stop_after_updates=1),budget(),rec,objective_spec=obj)
 ref=save_checkpoint(first.state,tmp_path/'kl');resumed=fit_model(handoff.student,args['teacher'],data,replace(handoff.config,resume=load_checkpoint(ref,handoff.config.run_spec)),budget(),rec,objective_spec=obj)
 uninterrupted=fit_model(whole.student,args['teacher'],data,whole.config,budget(),MemoryRecorder(),objective_spec=obj)
 assert resumed.state.events==uninterrupted.state.events and resumed.state.target_cursor==16384
 for name,t in handoff.student.named_parameters():assert torch.equal(t,dict(whole.student.named_parameters())[name])


def test_handoff_spec_is_closed_and_frozen():
 a=api()
 with pytest.raises(ValueError):a.ReconstructionHandoffSpec(**dict.fromkeys(('warm_run_id','checkpoint_manifest_sha256','warm_raw_plan_sha256','recovery_raw_plan_sha256','teacher_digest','owner_names_sha256','initial_state_digest'),'a'*64),warm_updates=0,seed=17)


def test_handoff_rejects_changed_owner_declarations(phase,tmp_path):
 from poc.checkpoint import checkpoint_decoder
 args,_,_,result=phase
 state=replace(result.state,model_schema=deepcopy(result.state.model_schema))
 next(o for o in state.model_schema['owners'] if o['name']=='bank.1.gate_proj.weight')['trainable']=True
 checkpoint_decoder(state) # tensor identity alone cannot detect this metadata change
 ref=save_checkpoint(state,tmp_path/'owner-metadata');spec=replace(args['handoff_spec'],checkpoint_manifest_sha256=ref.manifest_sha256)
 with pytest.raises(ValueError,match='owner'):
  api().prepare_reconstruction_handoff(**dict(args,checkpoint=ref,handoff_spec=spec))


@pytest.mark.mps
def test_actual_mps_completed_mse_checkpoint_handoff_and_kl(tmp_path):
 from poc.recovery_controls import qualify_reconstruction_backend,qualify_backend
 from poc.recovery_study import NaturalWindowPlan
 assert torch.backends.mps.is_available()
 assert qualify_reconstruction_backend('mps')['passed'] and qualify_backend('mps')['passed']
 a=api();teacher,student,plan,warm_obj=warm_setup(updates=1)
 teacher=teacher.float().to('mps');student=student.float().to('mps');student.base_digest=teacher.manifest().checkpoint_digest
 initial=deepcopy(student)
 def config(obj):
  cfg=replace(bound_config(obj,1),numerical_profile='owned-mps-handoff-v1',kl_floor_per_target=2e-6)
  identity=replace(cfg.run_spec,resolved_config={'training':cfg.identity(),'recovery_objective':obj.identity()},
   hardware_policy={'backend':'mps','dtype':'float32'},seeds={'fit':17},
   input_manifests=dict(cfg.run_spec.input_manifests,assets='a'*64,fit='b'*64,tuning='c'*64,initialized_state=initial.manifest().checkpoint_digest))
  return replace(cfg,run_spec=identity)
 warm_cfg=config(warm_obj);result=fit_model(student,teacher,TrainingSources(raw_plan=plan),warm_cfg,budget(),MemoryRecorder(),objective_spec=warm_obj)
 ref=save_checkpoint(result.state,tmp_path/'warm-mps');kl_obj=replace(warm_obj,schema_version=1,loss_scope='full_student')
 spec=a.ReconstructionHandoffSpec(run_id(warm_cfg.run_spec),ref.manifest_sha256,plan.digest,plan.digest,teacher.manifest().checkpoint_digest,warm_obj.owner_names_sha256,initial.manifest().checkpoint_digest,1,17)
 handoff=a.prepare_reconstruction_handoff(teacher=teacher,initial=initial,checkpoint=ref,warm_run_spec=warm_cfg.run_spec,warm_plan=plan,recovery_config=config(kl_obj),recovery_objective=kl_obj,handoff_spec=spec)
 assert handoff.student.manifest().checkpoint_digest==result.state.candidate_digest
 for name,t in student.named_parameters():assert torch.equal(t.cpu(),dict(handoff.student.named_parameters())[name].cpu())
 final=fit_model(handoff.student,teacher,TrainingSources(raw_plan=plan),handoff.config,budget(),MemoryRecorder(),objective_spec=kl_obj)
 assert final.state.step==1 and final.state.target_cursor==8192

"""Price extra-natural warm training without relabeling it nonlinear MSE."""
from copy import deepcopy
from dataclasses import asdict,replace
import pytest
import torch
from poc import recovery_phases as api
from poc.checkpoint import save_checkpoint,load_checkpoint
from poc.records import run_id
from poc.train import fit_model,TrainingSources
from test_reconstruction_handoff import phase
from test_recovery_study import bound_config
from test_training import budget,MemoryRecorder
pytestmark=pytest.mark.cpu


def generic_spec(old,scope):
 return api.OptimizationHandoffSpec(**asdict(old),warm_loss_scope=scope)


def test_generic_mse_matches_legacy_weights_with_an_explicit_new_identity(phase):
 args,_,_,_=phase
 old=api.prepare_reconstruction_handoff(**args)
 new=api.prepare_optimization_handoff(**dict(args,handoff_spec=generic_spec(args['handoff_spec'],'ffn_mse_sum')))
 for name,p in old.student.named_parameters():assert torch.equal(p,dict(new.student.named_parameters())[name])
 assert new.record['kind']=='named_optimization_phase_handoff_v1'
 assert new.record['warm_loss_scope']=='ffn_mse_sum'
 assert new.record['retained_warm_scored_targets']==32768
 assert new.config.run_spec.input_manifests['optimization_phase_handoff']
 assert new.config.run_spec!=old.config.run_spec


@pytest.fixture
def kl_phase(phase,tmp_path):
 args,_,recovery_plan,_=phase;initial=args['initial'];teacher=args['teacher'];warm_plan=args['warm_plan']
 source_obj=replace(args['recovery_objective'],raw_plan_sha256=warm_plan.digest)
 cfg=bound_config(source_obj,2);cfg=replace(cfg,run_spec=replace(cfg.run_spec,seeds={'fit':17},input_manifests=dict(cfg.run_spec.input_manifests,assets=teacher.tokenizer_digest,fit='b'*64,tuning='c'*64,initialized_state=initial.manifest().checkpoint_digest)))
 model=deepcopy(initial);result=fit_model(model,teacher,TrainingSources(raw_plan=warm_plan),cfg,budget(),MemoryRecorder(),objective_spec=source_obj);ref=save_checkpoint(result.state,tmp_path/'kl-warm')
 spec=replace(args['handoff_spec'],warm_run_id=run_id(cfg.run_spec),checkpoint_manifest_sha256=ref.manifest_sha256)
 return dict(args,checkpoint=ref,warm_run_spec=cfg.run_spec,handoff_spec=generic_spec(spec,'full_student')),model,recovery_plan,result


def test_extra_kl_warm_source_retains_correct_units_and_starts_fresh_optimizer(kl_phase):
 args,warmed,plan,result=kl_phase;before=args['initial'].manifest().checkpoint_digest
 handoff=api.prepare_optimization_handoff(**args)
 assert args['initial'].manifest().checkpoint_digest==before
 assert handoff.config.resume is None and handoff.record['warm_loss_scope']=='full_student'
 assert handoff.record['retained_warm_raw_targets']==16384
 assert handoff.record['retained_warm_scored_targets']==16384
 for name,p in warmed.named_parameters():assert torch.equal(p,dict(handoff.student.named_parameters())[name])
 final=fit_model(handoff.student,args['teacher'],TrainingSources(raw_plan=plan),handoff.config,budget(),MemoryRecorder(),objective_spec=handoff.objective)
 assert final.state.step==1 and final.state.target_cursor==8192
 assert set(int(v['step']) for v in final.state.optimizer['state'].values() if 'step' in v)=={1}
 assert set(int(v['step']) for v in result.state.optimizer['state'].values() if 'step' in v)=={2}


def test_legacy_mse_api_still_refuses_actual_kl_warm_state(kl_phase):
 args,_,_,_=kl_phase
 with pytest.raises(ValueError):api.prepare_reconstruction_handoff(**args)


@pytest.mark.parametrize('scope',['local_sum','donor',None])
def test_generic_warm_scope_is_explicit_and_closed(phase,scope):
 args,_,_,_=phase
 with pytest.raises(ValueError):generic_spec(args['handoff_spec'],scope)


def test_declared_warm_scope_must_match_actual_training_state(kl_phase):
 args,_,_,_=kl_phase
 with pytest.raises(ValueError):api.prepare_optimization_handoff(**dict(args,handoff_spec=replace(args['handoff_spec'],warm_loss_scope='ffn_mse_sum')))


def test_extra_kl_handoff_refuses_an_incomplete_safe_checkpoint(kl_phase,tmp_path):
 from poc.train import _schedule_state
 args,_,_,result=kl_phase
 state=replace(result.state,step=1,target_cursor=8192,scheduler=_schedule_state(1,2));ref=save_checkpoint(state,tmp_path/'incomplete-kl-warm')
 with pytest.raises(ValueError,match='incomplete'):
  api.prepare_optimization_handoff(**dict(args,checkpoint=ref,handoff_spec=replace(args['handoff_spec'],checkpoint_manifest_sha256=ref.manifest_sha256)))


def test_extra_kl_phase_resume_matches_uninterrupted(kl_phase,tmp_path):
 from poc.recovery_study import NaturalWindowPlan
 from test_training import batch
 args,_,_,_=kl_phase
 plan=NaturalWindowPlan.build((batch(512),),updates=2,seed=17,manifest_hashes={'owned':'a'*64});obj=replace(args['recovery_objective'],raw_plan_sha256=plan.digest);cfg=bound_config(obj,2)
 cfg=replace(cfg,run_spec=replace(cfg.run_spec,seeds={'fit':17},input_manifests=dict(cfg.run_spec.input_manifests,assets=args['teacher'].tokenizer_digest,fit='b'*64,tuning='c'*64,initialized_state=args['initial'].manifest().checkpoint_digest)))
 args=dict(args,recovery_config=cfg,recovery_objective=obj,handoff_spec=replace(args['handoff_spec'],recovery_raw_plan_sha256=plan.digest));first=api.prepare_optimization_handoff(**args);whole=api.prepare_optimization_handoff(**args);data=TrainingSources(raw_plan=plan);rec=MemoryRecorder()
 stopped=fit_model(first.student,args['teacher'],data,replace(first.config,stop_after_updates=1),budget(),rec,objective_spec=obj);ref=save_checkpoint(stopped.state,tmp_path/'recovery')
 resumed=fit_model(first.student,args['teacher'],data,replace(first.config,resume=load_checkpoint(ref,first.config.run_spec)),budget(),rec,objective_spec=obj);fit_model(whole.student,args['teacher'],data,whole.config,budget(),MemoryRecorder(),objective_spec=obj)
 assert resumed.state.step==2
 for name,p in first.student.named_parameters():assert torch.equal(p,dict(whole.student.named_parameters())[name])

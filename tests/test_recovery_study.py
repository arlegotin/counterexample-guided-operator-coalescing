"""Owned CPU controls for opt-in paired recovery; no pretrained inference."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from test_training import batch, models, budget, MemoryRecorder
from poc.train import TrainingConfig, TrainingSources, configure_trainable_owners, backward_update, fit_model


def api():
    assert importlib.util.find_spec('poc.recovery_objectives'), 'versioned recovery objective missing'
    return importlib.import_module('poc.recovery_objectives')


def plan_api():
    assert importlib.util.find_spec('poc.recovery_study'), 'immutable natural plan missing'
    return importlib.import_module('poc.recovery_study')


def setup(scope='full_student', updates=2):
    a=api();p=plan_api();teacher,student=models()
    teacher.tokenizer_digest=student.tokenizer_digest="a"*64
    teacher.config.max_position_embeddings=student.config.max_position_embeddings=1024
    plan=p.NaturalWindowPlan.build((batch(512),),updates=updates,raw_targets_per_update=8192,seed=17,
                                   manifest_hashes={'owned':'a'*64})
    objective=a.RecoveryObjectiveSpec(scope,'affected_group',(0,1),plan.digest,
                                    a.owner_names_hash(student,(0,1)))
    return teacher,student,plan,objective


def test_plan_exact_shuffled_cycles_and_mask_slicing_are_immutable():
    p=plan_api();b=batch(6);other=replace(batch(8),document_ids=('other',),cluster_ids=('other-component',))
    plan=p.NaturalWindowPlan.build((b,other),updates=3,raw_targets_per_update=8192,seed=17,
                                  manifest_hashes={'owned':'a'*64})
    h=plan.digest;b.input_ids.fill_(0)
    assert plan.digest==h
    assert plan.metadata['raw_target_presentations']==24576
    assert plan.metadata['unique_target_positions']==12
    updates=list(plan.iter_updates('cpu'))
    assert len(updates)==3 and all(u.valid_targets==8192 for u in updates)
    assert all(b.input_ids.shape[1] in (6,8) for u in updates for b in u.batches)
    assert plan.digest==p.NaturalWindowPlan.build((batch(6),other),updates=3,
        raw_targets_per_update=8192,seed=17,manifest_hashes={'owned':'a'*64}).digest
    assert any(not torch.equal(b.input_ids,torch.zeros_like(b.input_ids)) for b in updates[0].batches)
    with pytest.raises((TypeError,AttributeError)):
        plan.metadata['updates']=7


@pytest.mark.parametrize('scope',['local_sum','local_mean','full_student'])
def test_objective_gradient_matches_direct_autograd_and_directional_difference(scope):
    a=api();teacher,student,plan,spec=setup(scope)
    cfg=TrainingConfig(max_updates=2,alpha=0,affected_depths=(0,1))
    owners=configure_trainable_owners(student,cfg,objective_spec=spec)
    reference=deepcopy(student)
    from poc.losses import kl_sum
    b=batch(9);u=next(__import__('poc.train',fromlist=['iter_update_targets']).iter_update_targets((b,),8192))
    def loss(model):
        if scope=='full_student':return kl_sum(teacher.logits(b)[:,:-1],model.logits(b)[:,:-1],b.target_mask)/8
        total=0
        for d in (0,1):
            boundary=teacher.prefix_boundary(b,d)
            target=teacher.suffix_logits(teacher.ff_residual(boundary),boundary)[:,:-1]
            actual=teacher.suffix_logits(teacher.ff_residual(boundary,model.layers[d].mlp),boundary)[:,:-1]
            total=total+kl_sum(target,actual,b.target_mask)/8
        return total/(2 if scope=='local_mean' else 1)
    loss(reference).backward()
    rec=MemoryRecorder()
    backward_update(student,teacher,u,cfg,rec,objective_spec=spec,raw_templates=(b,))
    for n,p in owners.items():
        expected=dict(reference.named_parameters())[n].grad
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected,atol=2e-14,rtol=2e-10)
    n,p=next((n,p) for n,p in owners.items() if p.grad is not None and p.grad.abs().max()>1e-10)
    direction=torch.randn_like(p);analytic=(p.grad*direction).sum()
    original=p.detach().clone();h=1e-5
    with torch.no_grad():p.copy_(original+h*direction)
    plus=loss(student).item()
    with torch.no_grad():p.copy_(original-h*direction)
    minus=loss(student).item()
    with torch.no_grad():p.copy_(original)
    torch.testing.assert_close(analytic,torch.tensor((plus-minus)/(2*h),dtype=p.dtype),atol=2e-9,rtol=2e-5)
    assert rec.totals.valid_target_presentations==(8 if scope=='full_student' else 16)
    assert rec.totals.raw_corpus_target_presentations==8
    assert len(owners)==15
    assert all(p.grad is None for n,p in student.named_parameters() if n not in owners)
    assert all(p.grad is None for p in teacher.parameters())


def test_reject_partial_bank_and_stale_owner_binding_before_mutation():
    a=api();teacher,student,plan,spec=setup()
    flags=[p.requires_grad for p in student.parameters()]
    for bad in (replace(spec,affected_depths=(0,)),replace(spec,owner_names_sha256='0'*64)):
        with pytest.raises(ValueError):configure_trainable_owners(student,TrainingConfig(max_updates=1),objective_spec=bad)
        assert flags==[p.requires_grad for p in student.parameters()]


def test_legacy_training_identity_has_no_recovery_fields():
    cfg=TrainingConfig(max_updates=1)
    assert 'objective_spec' not in cfg.identity() and 'checkpoint_every' not in cfg.identity()


def bound_config(spec,updates,**kwargs):
    from test_records import spec as runspec
    cfg=TrainingConfig(max_updates=updates,alpha=0,affected_depths=(0,1),record_norms=True,**kwargs)
    identity=replace(runspec(budgets=budget()),resolved_config={'training':cfg.identity(),'recovery_objective':spec.identity()},
        input_manifests={'raw_plan':spec.raw_plan_sha256,'owner_names':spec.owner_names_sha256})
    return replace(cfg,run_spec=identity)


@pytest.mark.parametrize('scope',['local_sum','full_student'])
def test_finite_plan_exact_counters_resume_and_frozen_owners(scope,tmp_path):
    teacher,student,plan,spec=setup(scope,updates=2)
    cfg=bound_config(spec,2)
    sources=TrainingSources(raw_plan=plan)
    frozen={n:p.detach().clone() for n,p in student.named_parameters() if n not in api().affected_owners(student,(0,1))}
    rec=MemoryRecorder()
    first=fit_model(student,teacher,sources,replace(cfg,stop_after_updates=1),budget(),rec,objective_spec=spec)
    from poc.checkpoint import save_checkpoint,load_checkpoint
    ref=save_checkpoint(first.state,tmp_path)
    resumed=deepcopy(student)
    restored=fit_model(resumed,teacher,sources,replace(cfg,resume=load_checkpoint(ref,cfg.run_spec)),budget(),rec,objective_spec=spec)
    _,whole,_,_=setup(scope,updates=2)
    result=fit_model(whole,teacher,sources,cfg,budget(),MemoryRecorder(),objective_spec=spec)
    assert restored.state.events==result.state.events
    assert restored.state.target_cursor==16384
    assert rec.totals.raw_corpus_target_presentations==16384
    assert rec.totals.valid_target_presentations==16384*spec.scored_multiplier
    assert all(e['raw_target_presentations']==8192 and e['scored_target_presentations']==8192*spec.scored_multiplier for e in result.metrics)
    for n,p in whole.named_parameters():
        assert torch.equal(p,dict(resumed.named_parameters())[n])
        if n in frozen:assert torch.equal(p,frozen[n])
    with pytest.raises(ValueError,match='identity|lineage|RunSpec'):
        fit_model(resumed,teacher,sources,replace(cfg,resume=first.state),budget(),rec,
                  objective_spec=replace(spec,loss_scope='local_mean'))


def test_opt_in_rejects_plan_and_runspec_mismatch_before_stepping():
    teacher,student,plan,spec=setup()
    before=student.manifest().checkpoint_digest
    for sources,cfg in ((TrainingSources(natural=(batch(),)),bound_config(spec,2)),
                        (TrainingSources(raw_plan=plan),TrainingConfig(max_updates=2,alpha=0))):
        with pytest.raises(ValueError):fit_model(student,teacher,sources,cfg,budget(),MemoryRecorder(),objective_spec=spec)
        assert student.manifest().checkpoint_digest==before


def test_invalid_optimizer_step_restores_verified_safe_state_and_charges_lost_work(monkeypatch):
    teacher,student,plan,spec=setup(updates=1)
    cfg=bound_config(spec,1);before=student.manifest().checkpoint_digest;rec=MemoryRecorder()
    original=torch.optim.AdamW.step
    def corrupt(optimizer,*args,**kwargs):
        original(optimizer,*args,**kwargs)
        with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(torch.optim.AdamW,'step',corrupt)
    with pytest.raises(FloatingPointError):
        fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),rec,objective_spec=spec)
    assert student.manifest().checkpoint_digest==before
    assert rec.totals.updates==1 and rec.totals.raw_corpus_target_presentations==8192
    assert all(p.grad is None for p in student.parameters())


def test_plan_rejects_forged_order_digest_and_reports_only_consumed_unique_targets():
    p=plan_api()
    windows=tuple(replace(batch(512),document_ids=(f'd{i}',),cluster_ids=(f'c{i}',)) for i in range(32))
    plan=p.NaturalWindowPlan.build(windows,updates=1,seed=17,manifest_hashes={'owned':'a'*64})
    assert plan.metadata['unique_target_positions']==8192
    assert plan.metadata['inventory_unique_target_positions']==32*511
    with pytest.raises(ValueError,match='plan|digest|order'):
        replace(plan,order=plan.order[::-1])


def test_strict_study_config_rejects_unknown_and_unfrozen_run(tmp_path):
    p=plan_api()
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    cfg=p.load_study_config(root/'configs/recovery-diagnosis-v1.yaml')
    assert cfg['updates']==256 and cfg['raw_targets_per_update']==8192
    assert p.expected_counts(cfg,'local_sum')['scored_target_presentations']==4194304
    assert p.expected_counts(cfg,'full_student')['scored_target_presentations']==2097152
    import yaml
    raw=dict(cfg);raw['silent_new_setting']=True
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='unknown'):p.load_study_config(path)
    with pytest.raises(ValueError,match='budget|unresolved|population'):p.require_run_ready(cfg)


def test_driver_preflight_lists_unresolved_inputs_and_run_fails_before_models(tmp_path):
    import subprocess,sys,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    driver=root/'provenance/recovery_diagnosis.py'
    assert driver.is_file(),'explicit study driver missing'
    command=[sys.executable,str(driver),'--config',str(root/'configs/recovery-diagnosis-v1.yaml')]
    validation=subprocess.run(command+['--mode','validate'],cwd=root,capture_output=True,text=True)
    assert validation.returncode==0,validation.stderr
    result=json.loads(validation.stdout)
    assert result['status']=='blocked_prerequisites'
    assert 'resource_budget' in result['unresolved']
    run=subprocess.run(command+['--mode','run'],cwd=root,capture_output=True,text=True)
    assert run.returncode!=0 and 'unresolved' in run.stderr


def test_owned_backend_qualification_does_not_use_pretrained_assets():
    assert importlib.util.find_spec('poc.recovery_controls'),'backend qualification missing'
    from poc.recovery_controls import qualify_backend
    result=qualify_backend('cpu')
    assert result['passed']
    assert set(result['objectives'])=={'local_sum','local_mean','full_student'}
    assert result['full_rank_kl_abs']<2e-5


def test_driver_profile_and_run_complete_owned_campaign(monkeypatch,tmp_path):
    """Exercise real initialization/fitter/observation/checkpoint APIs end to end."""
    import importlib.util,json
    from pathlib import Path
    from poc.data.manifest import DataManifest,DocumentRecord
    from poc.contracts import Budget
    root=Path(__file__).resolve().parents[1]
    module_spec=importlib.util.spec_from_file_location('recovery_driver',root/'provenance/recovery_diagnosis.py')
    driver=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(driver)
    cfg=plan_api().load_study_config(root/'configs/recovery-diagnosis-v1.yaml')
    cfg.update(asset_root=str(tmp_path),updates=1,observations=[0,1],seeds=[17],pair=[0,1])
    cfg['numerical_profile']['device']='cpu'
    cfg['populations']={'fit':{'path':str(tmp_path/'fit'),'sha256':'a'*64,'components':2},
                        'tuning':{'path':str(tmp_path/'dev'),'sha256':'b'*64,'components':1}}
    teacher,_,_,_=setup(updates=1)
    windows=tuple(replace(batch(512),document_ids=(f'fit{i}',),cluster_ids=(f'component{i}',)) for i in range(2))
    tuning=(replace(batch(512),document_ids=('dev',),cluster_ids=('dev-component',)),)
    fit=DataManifest(tuple(DocumentRecord(f'fit{i}','owned','a'*40,'owned',i,text='owned') for i in range(2)),
        {f'fit{i}':'fit' for i in range(2)},{f'fit{i}':f'component{i}' for i in range(2)},'owned')
    plan=plan_api().NaturalWindowPlan.build(windows,updates=1,seed=17,manifest_hashes={'owned':'a'*64})
    monkeypatch.setattr(driver,'load_inputs',lambda c:(object(),fit,windows,tuning,{17:plan}))
    import poc.teacher
    monkeypatch.setattr(poc.teacher,'load_teacher',lambda a:deepcopy(teacher))
    out=tmp_path/'campaign';out.mkdir()
    monkeypatch.setattr(driver,'output_root',lambda c:out)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    source=driver.executed_sources(path)
    driver.write_new(out/'controls.json',{'backend':{'passed':True,'device':'cpu'},'implementation_sha256':source['implementation_sha256'],'environment_sha256':driver.hashed(driver.environment())})
    profile=driver.execute(cfg,path,'profile')
    assert set(profile['arms'])=={'local_sum','full_student'}
    assert profile['environment_sha256']==driver.hashed(driver.environment())
    assert all(a['numerical_controls']['passed'] for a in profile['arms'].values())
    assert not list(out.glob('profile-attempt-*/**/resume/generation-*'))
    assert all(a['forecast_counters']['valid_target_presentations']>8192 for a in profile['arms'].values())
    assert all(a['cost']['memory_bytes'] is not None for a in profile['arms'].values())
    assert all('quality' in a and a['planned_updates']==1 for a in profile['arms'].values())
    b=budget().to_dict();b.update(updates=1,wall_seconds=3000,valid_target_presentations=200000,memory_bytes=4*1024**3,disk_bytes=2*1024**3)
    cfg['resource_budget']={'frozen_profile_sha256':driver.sha(out/'profile.json'),'per_arm':{s:b for s in cfg['loss_scopes']}}
    path.write_text(json.dumps(cfg))
    observe=driver.observation
    failed={'value':False}
    def interrupted(*args,**kwargs):
        if args[5]==1 and not failed['value']:
            failed['value']=True
            raise InterruptedError('owned interruption after safe checkpoint before observation')
        return observe(*args,**kwargs)
    monkeypatch.setattr(driver,'observation',interrupted)
    with pytest.raises(InterruptedError):driver.execute(cfg,path,'run')
    monkeypatch.setattr(driver,'observation',observe)
    driver.execute(cfg,path,'run',resume=True)
    for scope in cfg['loss_scopes']:
        arm=out/'runs'/f'{scope}-seed17'
        result=json.loads((arm/'result.json').read_text())
        assert result['step']==1 and result['raw_target_cursor']==8192
        assert (arm/'observations'/'0000.json').exists() and (arm/'observations'/'0001.json').exists()
        assert (arm/'resume'/'current.json').exists()
        assert (arm/'resume'/'anchor'/'current.json').exists()
        assert (arm/'deltas'/'0001'/'manifest.json').exists()


def test_plan_snapshots_mutable_scalar_containers_and_counts_consumed_components():
    p=plan_api()
    windows=tuple(replace(batch(512),document_ids=(f'd{i}',),cluster_ids=(f'c{i}',)) for i in range(32))
    plan=p.NaturalWindowPlan.build(windows,updates=1,seed=17,manifest_hashes={'owned':'a'*64})
    tokens=list(plan.templates[0].input_ids)
    window=replace(plan.templates[0],input_ids=tokens)
    order=list(plan.order)
    copied=replace(plan,templates=(window,)+plan.templates[1:],order=order)
    tokens[1]=0;order.reverse()
    assert copied.templates[0].input_ids[1]==1 and copied.order==plan.order
    assert copied.metadata['unique_components']==17


def test_raw_counter_charges_only_attempted_contexts_after_partial_backward_failure(monkeypatch):
    a=api();teacher,student,plan,spec=setup('full_student',updates=1)
    cfg=TrainingConfig(max_updates=1,alpha=0,affected_depths=(0,1))
    configure_trainable_owners(student,cfg,objective_spec=spec)
    import poc.losses
    def fail(*args,**kwargs):raise FloatingPointError('first context failure')
    monkeypatch.setattr(poc.losses,'backward_distillation',fail)
    rec=MemoryRecorder()
    with pytest.raises(FloatingPointError):
        backward_update(student,teacher,next(plan.iter_updates('cpu')),cfg,rec,objective_spec=spec,raw_templates=plan)
    assert rec.totals.raw_corpus_target_presentations==511


def test_exact_work_plan_forecast_matches_scored_ledger():
    p=plan_api()
    teacher,student,plan,spec=setup('local_sum',updates=2)
    assert hasattr(p,'planned_training_work'),'exact plan work accounting missing'
    predicted=p.planned_training_work(plan,'local_sum',(0,1))
    cfg=TrainingConfig(max_updates=2,alpha=0,affected_depths=(0,1),kl_floor_per_target=2e-6,numerical_profile='owned')
    configure_trainable_owners(student,cfg,objective_spec=spec)
    rec=MemoryRecorder()
    for update in plan.iter_updates('cpu'):
        student.zero_grad(set_to_none=True)
        backward_update(student,teacher,update,cfg,rec,objective_spec=spec,raw_templates=plan)
    for key in ('valid_target_presentations','raw_corpus_target_presentations','teacher_forward_calls','student_forward_calls','student_backward_calls','suffix_backward_calls'):
        assert predicted[key]==getattr(rec.totals,key),key


def test_delta_recovers_partial_publication_and_reserves_campaign_bytes(tmp_path):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    ms=importlib.util.spec_from_file_location('delta_driver',root/'provenance/recovery_diagnosis.py')
    d=importlib.util.module_from_spec(ms);ms.loader.exec_module(d)
    _,student,_,obj=setup(updates=1)
    cfg=bound_config(obj,1);spec=replace(cfg.run_spec,resolved_config={**dict(cfg.run_spec.resolved_config),'study_contract_sha256':'a'*64})
    path=tmp_path/'arm'/'deltas'/'0001';path.mkdir(parents=True)
    (path/'owners.safetensors').write_bytes(b'owned interrupted partial')
    class Recorder(MemoryRecorder):
        def __init__(self):super().__init__();self.checks=[]
        def disk(self):return 1000000
        def check(self,cost):self.checks.append(cost)
    rec=Recorder()
    d.save_recovery_delta(student,student.manifest().checkpoint_digest,spec,path,rec)
    assert (path/'manifest.json').is_file()
    assert any(p.read_bytes()==b'owned interrupted partial' for p in (tmp_path/'arm').rglob('owners.safetensors') if p!=path/'owners.safetensors')
    assert rec.checks[0].disk_bytes>1000000


def test_observation_v2_scores_exact_optimizer_masks_and_full_tuning_only():
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    ms=importlib.util.spec_from_file_location('v2_driver',root/'provenance/recovery_diagnosis.py')
    d=importlib.util.module_from_spec(ms);ms.loader.exec_module(d)
    assert (root/'configs/recovery-diagnosis-v2.yaml').is_file(),'versioned bounded observation policy missing'
    cfg=plan_api().load_study_config(root/'configs/recovery-diagnosis-v2.yaml');cfg['pair']=[0,1]
    teacher,student,plan,_=setup(updates=1)
    fit=(batch(512),);tuning=(replace(batch(512),document_ids=('tune',),cluster_ids=('tune-component',)),)
    rec=MemoryRecorder();value=d.observation(teacher,student,fit,tuning,plan,0,rec,cfg)
    assert set(value['populations'])=={'optimizer_windows','tuning'}
    assert sum(r['full']['valid_targets'] for r in value['populations']['optimizer_windows'])==8192
    assert sum(r['full']['valid_targets'] for r in value['populations']['tuning'])==511
    assert all(len(r['local_by_depth'])==2 for r in value['populations']['optimizer_windows'])
    assert all(r['local_by_depth']==[] for r in value['populations']['tuning'])
    predicted=plan_api().planned_observation_work(plan,[0],fit,tuning,(0,1),policy=cfg['observation_policy'])
    for key in ('teacher_forward_calls','student_forward_calls','valid_target_presentations'):
        assert predicted[key]==getattr(rec.totals,key),key
    assert rec.totals.valid_target_presentations==25087

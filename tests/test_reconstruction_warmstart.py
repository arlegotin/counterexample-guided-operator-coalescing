"""Qualify an explicitly named FFN-output MSE objective through the common loop."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from poc.train import (TrainingSources, configure_trainable_owners, backward_update,
                       fit_model, iter_update_targets)
from test_training import batch,budget,MemoryRecorder
from test_recovery_study import setup,bound_config


def warm_setup(updates=2):
    teacher,student,plan,spec=setup('local_sum',updates=updates)
    return teacher,student,plan,replace(spec,schema_version=2,loss_scope='ffn_mse_sum')


def reconstruction_sum(teacher,student,b,depths=(0,1)):
    total=0
    for depth in depths:
        with torch.no_grad():
            boundary=teacher.prefix_boundary(b,depth)
            x=teacher._rms(boundary.u,teacher.layers[depth].post_attention_layernorm)
            target=teacher.layers[depth].mlp(x)
        error=(student.layers[depth].mlp(x)-target)[:,:-1][b.target_mask]
        total=total+error.double().square().mean(-1).sum()
    return total


def test_mse_objective_matches_direct_gradient_and_directional_derivative():
    teacher,student,plan,spec=warm_setup()
    cfg=bound_config(spec,2);owners=configure_trainable_owners(student,cfg,objective_spec=spec)
    reference=deepcopy(student);b=batch(9);update=next(iter_update_targets((b,),8192));rec=MemoryRecorder()
    direct=reconstruction_sum(teacher,reference,b)/8;direct.backward()
    sums=backward_update(student,teacher,update,cfg,rec,objective_spec=spec,raw_templates=(b,))
    assert sum(m.valid_targets for m in sums)==16
    assert sum(m.squared_error_sum for m in sums)==pytest.approx(float(direct.detach())*8)
    for name,p in owners.items():
        expected=dict(reference.named_parameters())[name].grad
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected,atol=2e-13,rtol=2e-10)
    p=next(p for p in owners.values() if p.grad is not None and p.grad.abs().max()>1e-10)
    direction=torch.randn_like(p);analytic=(p.grad*direction).sum();old=p.detach().clone();h=1e-5
    with torch.no_grad():p.copy_(old+h*direction)
    plus=reconstruction_sum(teacher,student,b).item()/8
    with torch.no_grad():p.copy_(old-h*direction)
    minus=reconstruction_sum(teacher,student,b).item()/8
    with torch.no_grad():p.copy_(old)
    torch.testing.assert_close(analytic,torch.tensor((plus-minus)/(2*h),dtype=p.dtype),atol=2e-9,rtol=2e-5)
    assert all(p.grad is None for p in teacher.parameters())
    assert rec.totals.raw_corpus_target_presentations==8
    assert rec.totals.valid_target_presentations==16
    assert rec.totals.suffix_backward_calls==0 and rec.totals.student_backward_calls==2


def test_common_loop_labels_mse_without_reporting_unmeasured_kl():
    teacher,student,plan,spec=warm_setup(updates=1);cfg=bound_config(spec,1);rec=MemoryRecorder()
    frozen={n:p.detach().clone() for n,p in student.named_parameters()
            if n not in __import__('poc.recovery_objectives',fromlist=['affected_owners']).affected_owners(student,(0,1))}
    result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),rec,objective_spec=spec)
    row=result.metrics[0]
    assert row['objective_name']=='ffn_output_mse_sum'
    assert row['kl_sum'] is None and row['nll_sum'] is None
    assert row['reconstruction_sum']>0 and row['weighted_objective']==row['reconstruction_sum']/8192
    assert 'source_kl_sums' not in row
    assert set(row['reconstruction_by_depth'])=={'0','1'}
    assert sum(v['valid_targets'] for v in row['reconstruction_by_depth'].values())==16384
    assert rec.totals.raw_corpus_target_presentations==8192
    assert rec.totals.valid_target_presentations==16384
    for n,p in student.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n])


def test_mse_common_loop_resume_matches_uninterrupted(tmp_path):
    from poc.checkpoint import save_checkpoint,load_checkpoint
    teacher,student,plan,spec=warm_setup();cfg=bound_config(spec,2);source=TrainingSources(raw_plan=plan);rec=MemoryRecorder()
    first=fit_model(student,teacher,source,replace(cfg,stop_after_updates=1),budget(),rec,objective_spec=spec)
    checkpoint=save_checkpoint(first.state,tmp_path)
    resumed=fit_model(student,teacher,source,replace(cfg,resume=load_checkpoint(checkpoint,cfg.run_spec)),budget(),rec,objective_spec=spec)
    _,whole,_,_=warm_setup()
    uninterrupted=fit_model(whole,teacher,source,cfg,budget(),MemoryRecorder(),objective_spec=spec)
    assert resumed.state.events==uninterrupted.state.events
    assert resumed.state.target_cursor==16384
    for n,p in whole.named_parameters():assert torch.equal(p,dict(student.named_parameters())[n])


def test_original_objective_version_remains_closed_to_mse():
    from poc.recovery_objectives import RecoveryObjectiveSpec
    with pytest.raises(ValueError):RecoveryObjectiveSpec('ffn_mse_sum','affected_group',(0,1),'a'*64,'b'*64)


def test_owned_representable_window_recovers_under_completed_mse_schedule():
    from poc.bank import PROJECTIONS,make_bank,canonicalize_partition
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash
    from poc.recovery_study import NaturalWindowPlan
    from poc.train import TrainingConfig
    from test_records import spec as run_spec
    teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024),seed=3,dtype=torch.float64)
    teacher.tokenizer_digest='a'*64
    g=torch.Generator().manual_seed(17)
    with torch.no_grad():
        for q in PROJECTIONS:
            base=getattr(teacher.layers[0].mlp,q).weight
            base.copy_(torch.randn(base.shape,generator=g,dtype=torch.float64)*.2)
            d=torch.randn(base.shape[0],1,generator=g,dtype=torch.float64)*.08
            a=torch.randn(1,base.shape[1],generator=g,dtype=torch.float64)*.08
            getattr(teacher.layers[1].mlp,q).weight.copy_(base+d@a)
    student=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,17)
    b=batch(512)
    assert reconstruction_sum(teacher,student,b).item()<1e-20
    with torch.no_grad():
        for factor in student.layers[1].mlp.residuals.values():
            factor.B.add_(torch.randn(factor.B.shape,generator=g,dtype=torch.float64)*.003)
    initial=reconstruction_sum(teacher,student,b).item()
    plan=NaturalWindowPlan.build((b,),updates=64,seed=17,manifest_hashes={'owned':'a'*64})
    objective=RecoveryObjectiveSpec('ffn_mse_sum','affected_group',(0,1),plan.digest,owner_names_hash(student,(0,1)),schema_version=2)
    cfg=TrainingConfig(max_updates=64,alpha=0,affected_depths=(0,1))
    allocated=budget(updates=64,valid_target_presentations=1100000,teacher_forward_calls=20000,
                     student_backward_calls=3000,suffix_backward_calls=0)
    identity=replace(run_spec(budgets=allocated),resolved_config={'training':cfg.identity(),'recovery_objective':objective.identity()},
                     input_manifests={'raw_plan':plan.digest,'owner_names':objective.owner_names_sha256})
    cfg=replace(cfg,run_spec=identity)
    result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,allocated,MemoryRecorder(),objective_spec=objective)
    final=reconstruction_sum(teacher,student,b).item()
    assert result.state.step==64 and result.state.target_cursor==524288
    assert final<initial*.1,(initial,final)


def test_nonfinite_reconstruction_clears_partial_gradients_without_stepping(monkeypatch):
    teacher,student,plan,spec=warm_setup(updates=1);cfg=bound_config(spec,1);rec=MemoryRecorder()
    before={n:p.detach().clone() for n,p in student.named_parameters()}
    original=student.layers[1].mlp.forward
    monkeypatch.setattr(student.layers[1].mlp,'forward',lambda x:original(x)*float('inf'))
    with pytest.raises(FloatingPointError,match='nonfinite'):
        fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),rec,objective_spec=spec)
    assert rec.totals.updates==0
    assert rec.totals.raw_corpus_target_presentations==511
    assert rec.totals.valid_target_presentations==1022
    for name,p in student.named_parameters():
        assert torch.equal(p,before[name]) and p.grad is None


def test_single_pass_teacher_boundaries_match_independent_prefixes_and_reduce_work():
    import poc.reconstruction as reconstruction
    assert hasattr(reconstruction,'collect_teacher_boundaries'),'single-pass collector missing'
    teacher,student,plan,spec=warm_setup();b=batch(9);visited=[]
    handles=[layer.mlp.register_forward_hook(lambda module,args,out,d=d:visited.append(d))
             for d,layer in enumerate(teacher.layers)]
    expected=[teacher.prefix_boundary(b,d) for d in (0,1,2)]
    independent_visits=len(visited);visited.clear();rec=MemoryRecorder()
    actual=reconstruction.collect_teacher_boundaries(teacher,b,(0,1,2),recorder=rec)
    for a,e in zip(actual,expected,strict=True):
        torch.testing.assert_close(a.u,e.u,atol=0,rtol=0)
        assert a.depth==e.depth and a.model_digest==e.model_digest and not a.u.requires_grad
    assert len(visited)<independent_visits
    assert rec.totals.teacher_forward_calls==1
    for h in handles:h.remove()


@pytest.mark.parametrize('depths',[(),(1,0),(0,0),(-1,),(3,),(True,)])
def test_single_pass_rejects_invalid_depths_before_teacher_queries(depths):
    import poc.reconstruction as reconstruction
    assert hasattr(reconstruction,'collect_teacher_boundaries'),'single-pass collector missing'
    teacher,student,plan,spec=warm_setup();rec=MemoryRecorder()
    with pytest.raises(ValueError):reconstruction.collect_teacher_boundaries(teacher,batch(9),depths,recorder=rec)
    assert rec.totals.teacher_forward_calls==0


def test_reconstruction_work_forecast_matches_actual_dispatch_units():
    from poc.recovery_study import planned_training_work
    teacher,student,plan,spec=warm_setup(updates=1)
    cfg=bound_config(spec,1);configure_trainable_owners(student,cfg,objective_spec=spec)
    predicted=planned_training_work(plan,'ffn_mse_sum',(0,1));rec=MemoryRecorder()
    for update in plan.iter_updates('cpu'):
        student.zero_grad(set_to_none=True)
        backward_update(student,teacher,update,cfg,rec,objective_spec=spec,raw_templates=plan)
    for key in ('valid_target_presentations','raw_corpus_target_presentations','teacher_forward_calls',
                'student_forward_calls','student_backward_calls','suffix_backward_calls','processed_inputs','processed_tokens'):
        assert predicted[key]==getattr(rec.totals,key),key


def test_training_work_forecast_rejects_unknown_objective():
    from poc.recovery_study import planned_training_work
    _,_,plan,_=warm_setup(updates=1)
    with pytest.raises(ValueError,match='scope'):
        planned_training_work(plan,'unknown',(0,1))


def test_owned_fp32_reconstruction_backend_qualification():
    import poc.recovery_controls as controls
    assert hasattr(controls,'qualify_reconstruction_backend'),'reconstruction backend gate missing'
    result=controls.qualify_reconstruction_backend('cpu')
    assert result['passed'] and result['device']=='cpu'
    assert result['gradient_max_abs_error']<=result['gradient_max_abs_tolerance']
    assert result['value_error']<=result['value_tolerance']
    assert result['reference_value']>1e-6
    assert result['reference_gradient_max_abs']>1e-5


@pytest.mark.mps
def test_owned_mps_reconstruction_backend_qualification():
    import poc.recovery_controls as controls
    assert hasattr(controls,'qualify_reconstruction_backend'),'reconstruction backend gate missing'
    assert torch.backends.mps.is_available(),'explicit MPS reconstruction qualification requires Metal'
    result=controls.qualify_reconstruction_backend('mps')
    assert result['passed'] and result['device']=='mps'


@pytest.mark.mps
def test_mps_reconstruction_preserves_nonzero_cpu_accumulation_and_gradients():
    assert torch.backends.mps.is_available(),'explicit regression requires Metal'
    teacher,student,plan,spec=warm_setup()
    teacher=teacher.float();student=student.float();reference_teacher=deepcopy(teacher).double();reference=deepcopy(student).double()
    cfg=bound_config(spec,2);configure_trainable_owners(reference,cfg,objective_spec=spec)
    b=batch(9);expected=reconstruction_sum(reference_teacher,reference,b)/8;expected.backward()
    assert float(expected.detach())>0
    teacher=teacher.to('mps');student=student.to('mps')
    owners=configure_trainable_owners(student,cfg,objective_spec=spec)
    mb=replace(b,**{n:getattr(b,n).to('mps') for n in ('input_ids','attention_mask','target_mask','positions')})
    rec=MemoryRecorder();actual=backward_update(student,teacher,next(iter_update_targets((mb,),8192)),cfg,rec,objective_spec=spec,raw_templates=(mb,))
    value=sum(r.squared_error_sum for r in actual)/8
    assert value>0,'cross-device FP64 accumulation lost the actual loss signal'
    assert value==pytest.approx(float(expected.detach()),rel=1e-4,abs=1e-12)
    for name,p in owners.items():
        gradient=dict(reference.named_parameters())[name].grad
        if gradient is not None:torch.testing.assert_close(p.grad.cpu().double(),gradient,atol=1e-9,rtol=1e-4)

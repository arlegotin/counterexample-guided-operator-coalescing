"""Owned numerical qualification for each opted-in recovery objective."""
from copy import deepcopy
from dataclasses import replace
import math

import torch

from poc.bank import make_bank,canonicalize_partition
from poc.contracts import Batch,CostRecord
from poc.losses import kl_sum,stream_metrics
from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash
from poc.runtime.torch_decoder import make_tiny_decoder,TinyConfig
from poc.train import TrainingConfig,configure_trainable_owners,iter_update_targets,backward_update


class ControlRecorder:
    def __init__(self):self.rows=[]
    def check(self,cost):pass
    def record(self,cost):self.rows.append(cost)


def qualify_backend(device):
    """Fixed tolerances are chosen before pretrained candidate outcomes.

    Compare actual chunk32 VJPs against direct, complete-vocabulary CPU FP64
    autograd on identical rounded weights. This qualifies owned shapes only;
    real-shape no-op/initial-loss checks remain required by the profile driver.
    """
    if device not in ('cpu','mps') or device=='mps' and not torch.backends.mps.is_available():
        raise ValueError('requested recovery backend unavailable')
    torch.set_num_threads(1)
    teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,
        num_attention_heads=2,num_key_value_heads=1),seed=3,dtype=torch.float64)
    teacher.tokenizer_digest='a'*64
    generator=torch.Generator().manual_seed(991)
    with torch.no_grad():
        for layer in teacher.layers:
            for projection in ('gate_proj','up_proj','down_proj'):
                weight=getattr(layer.mlp,projection).weight
                weight.copy_(torch.randn(weight.shape,generator=generator)*.2)
    student=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,7)
    teacher=teacher.to(device=device,dtype=torch.float32);student=student.to(device=device,dtype=torch.float32)
    reference_teacher=deepcopy(teacher).cpu().double()
    ids=torch.tensor([[1,4,7,2,3,6,1,0,2]],device=device)
    b=Batch(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,8),device=device,dtype=torch.bool),
            torch.arange(9,device=device)[None],('owned-numerics',),('owned-component',))
    rb=replace(b,**{n:getattr(b,n).cpu() for n in ('input_ids','attention_mask','target_mask','positions')})
    result={'profile':'owned-kl-fp32-reference-v2','device':device,'dtype':'float32','value_tolerance':2e-5,'gradient_max_abs_tolerance':2e-5,'objectives':{}}
    local_initial=full_initial=None
    for scope in ('local_sum','local_mean','full_student'):
        model=deepcopy(student);reference=deepcopy(student).cpu().double()
        spec=RecoveryObjectiveSpec(scope,'affected_group',(0,1),'b'*64,owner_names_hash(model,(0,1)))
        cfg=TrainingConfig(max_updates=1,alpha=0,affected_depths=(0,1),kl_floor_per_target=2e-6,numerical_profile='owned-fp32-v1')
        owners=configure_trainable_owners(model,cfg,objective_spec=spec)
        configure_trainable_owners(reference,cfg,objective_spec=spec)
        def loss(m,t,batch):
            if scope=='full_student':return kl_sum(t.logits(batch)[:,:-1],m.logits(batch)[:,:-1],batch.target_mask)/8
            total=0
            for d in (0,1):
                boundary=t.prefix_boundary(batch,d)
                target=t.suffix_logits(t.ff_residual(boundary),boundary)[:,:-1]
                logits=t.suffix_logits(t.ff_residual(boundary,m.layers[d].mlp),boundary)[:,:-1]
                total=total+kl_sum(target,logits,batch.target_mask)/8
            return total*spec.scale
        value=float(loss(model,teacher,b).detach().cpu());ref_loss=loss(reference,reference_teacher,rb);ref_loss.backward()
        update=next(iter_update_targets((b,),8192));rec=ControlRecorder()
        metrics=backward_update(model,teacher,update,cfg,rec,objective_spec=spec,raw_templates=(b,))
        errs=[]
        for n,p in owners.items():
            expected=dict(reference.named_parameters())[n].grad
            if expected is None:
                if p.grad is not None and p.grad.abs().max()>2e-5:raise ValueError('unexpected backend gradient')
            else:errs.append(float((p.grad.detach().cpu().double()-expected).abs().max()))
        streamed=math.fsum(m.kl for m in metrics)*spec.scale/8
        row={'value_error':abs(value-float(ref_loss.detach())),
             'chunk_value_error':abs(streamed-value),'gradient_max_abs_error':max(errs)}
        if any(not math.isfinite(v) or v>2e-5 for v in row.values()):raise ValueError(f'owned backend parity failed: {scope}: {row}')
        reference_gradient=max(float(p.grad.abs().max()) for p in reference.parameters() if p.grad is not None)
        if float(ref_loss.detach())<=1e-6 or reference_gradient<=1e-5:
            raise ValueError('owned backend parity control signal is too small')
        row.update(reference_value=float(ref_loss.detach()),reference_gradient_max_abs=reference_gradient)
        result['objectives'][scope]=row
        if scope=='local_sum':local_initial=value
        if scope=='full_student':full_initial=value
    fullrank=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),8,7)
    result['full_rank_kl_abs']=abs(stream_metrics(teacher,fullrank,b,32).kl/8)
    result['initial_local_full_gap']=abs(local_initial-full_initial)
    if max(result['full_rank_kl_abs'],result['initial_local_full_gap'])>2e-5:
        raise ValueError('owned full-rank/initial scalar parity failed')
    result['passed']=True
    result['scope']='owned shapes only; not pretrained recovery, native MLX, or capability qualification'
    return result


def qualify_reconstruction_backend(device):
    """Qualify named MSE values/VJPs before any pretrained warm-start outcome.

    Fixed2e-5 absolute envelopes compare actual FP32 execution with direct CPU
    FP64 autograd on identical rounded weights. This is an owned-shape numerical
    gate, separate from KL resolution and from real-shape resource profiling.
    """
    from poc.reconstruction import collect_teacher_boundaries
    if device not in ('cpu','mps') or device=='mps' and not torch.backends.mps.is_available():
        raise ValueError('requested reconstruction backend unavailable')
    torch.set_num_threads(1)
    teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,
        num_attention_heads=2,num_key_value_heads=1),seed=3,dtype=torch.float32)
    teacher.tokenizer_digest='a'*64
    # A measured parity gate must exercise a signal well above its envelope.
    generator=torch.Generator().manual_seed(991)
    with torch.no_grad():
        for layer in teacher.layers:
            for projection in ('gate_proj','up_proj','down_proj'):
                weight=getattr(layer.mlp,projection).weight
                weight.copy_(torch.randn(weight.shape,generator=generator)*.2)
    student=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,7)
    with torch.no_grad():
        for depth in (0,1):
            for factors in student.layers[depth].mlp.residuals.values():factors.B.add_(.03)
    teacher=teacher.to(device=device);student=student.to(device=device)
    rt=deepcopy(teacher).cpu().double()
    rs=deepcopy(student).cpu().double()
    ids=torch.tensor([[1,4,7,2,3,6,1,0,2]],device=device)
    b=Batch(ids,torch.ones_like(ids,dtype=torch.bool),torch.tensor([[True,False,True,True,True,False,True,True]],device=device),
            torch.arange(9,device=device)[None],('owned-reconstruction',),('owned-component',))
    # Provenance retains full canonical eligibility; dispatch alone masks targets.
    original=replace(b,target_mask=torch.ones_like(b.target_mask))
    rb=replace(b,**{n:getattr(b,n).cpu() for n in ('input_ids','attention_mask','target_mask','positions')})
    spec=RecoveryObjectiveSpec('ffn_mse_sum','affected_group',(0,1),'b'*64,
                               owner_names_hash(student,(0,1)),schema_version=2)
    cfg=TrainingConfig(max_updates=1,alpha=0,affected_depths=(0,1),numerical_profile='owned-ffn-mse-fp32-reference-v2')
    owners=configure_trainable_owners(student,cfg,objective_spec=spec)
    configure_trainable_owners(rs,cfg,objective_spec=spec)
    n=int(b.target_mask.sum())
    def direct(t,m,batch):
        total=0
        for d in (0,1):
            with torch.no_grad():
                boundary=t.prefix_boundary(batch,d)
                x=t._rms(boundary.u,t.layers[d].post_attention_layernorm)
                target=t.layers[d].mlp(x)
            error=m.layers[d].mlp(x)[:,:-1][batch.target_mask].cpu().double()-target[:,:-1][batch.target_mask].cpu().double()
            total=total+error.square().mean(-1).sum()/n
        return total
    reference=direct(rt,rs,rb);reference.backward()
    rec=ControlRecorder()
    metrics=backward_update(student,teacher,next(iter_update_targets((b,),8192)),cfg,rec,
                            objective_spec=spec,raw_templates=(original,))
    actual=math.fsum(x.squared_error_sum for x in metrics)/n
    errors=[]
    for name,p in owners.items():
        expected=dict(rs.named_parameters())[name].grad
        if expected is None:
            if p.grad is not None:errors.append(float(p.grad.detach().abs().max().cpu()))
        elif p.grad is None:raise ValueError('missing reconstruction gradient')
        else:errors.append(float((p.grad.detach().cpu().double()-expected).abs().max()))
    boundaries=collect_teacher_boundaries(teacher,original,(0,1),recorder=rec)
    boundary_error=max(float((x.u-teacher.prefix_boundary(original,x.depth).u).abs().max().cpu()) for x in boundaries)
    fullrank=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),8,7)
    with torch.no_grad():fullrank_error=float(direct(teacher,fullrank,b))
    result={'schema_version':1,'device':device,'dtype':'float32','profile':'owned-ffn-mse-fp32-reference-v2',
        'value_tolerance':2e-5,'gradient_max_abs_tolerance':2e-5,
        'actual_value':actual,'reference_value':float(reference.detach()),
        'reference_gradient_max_abs':max(float(p.grad.abs().max()) for p in rs.parameters() if p.grad is not None),
        'value_error':abs(actual-float(reference.detach())),
        'gradient_max_abs_error':max(errors),'single_pass_boundary_max_abs_error':boundary_error,
        'full_rank_reconstruction_value_abs':abs(fullrank_error),
        'cost_records':[r.to_dict() for r in rec.rows],
        'scope':'owned reconstruction shapes only; not pretrained/native/capability qualification'}
    if any(not math.isfinite(x) or x>2e-5 for x in (result['value_error'],result['gradient_max_abs_error'],boundary_error,abs(fullrank_error))):
        raise ValueError('owned reconstruction backend parity failed: '+str({k:v for k,v in result.items() if k!='cost_records'}))
    if result['reference_value']<=1e-6 or result['reference_gradient_max_abs']<=1e-5:
        raise ValueError('reconstruction control signal is too small for its numerical envelope')
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('teacher received reconstruction gradients')
    result['passed']=True
    return result

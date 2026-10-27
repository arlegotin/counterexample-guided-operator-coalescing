"""Whole-update accumulation and primary owner policy."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from poc.bank import make_bank, canonicalize_partition
from poc.contracts import Batch, Budget, CostRecord
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder


def api():
    assert importlib.util.find_spec('poc.train'), 'common fitter missing'
    return importlib.import_module('poc.train')


def batch(n=9, mask=None):
    ids = torch.arange(n)[None] % 11
    return Batch(ids, torch.ones_like(ids, dtype=torch.bool),
                 torch.ones((1,n-1), dtype=torch.bool) if mask is None else torch.tensor([mask]),
                 torch.arange(n)[None], ('fit-doc',), ('fit-component',))


def models():
    teacher = make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,
        num_attention_heads=2,num_key_value_heads=1),seed=3,dtype=torch.float64)
    return teacher, make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,7)


def budget(**changes):
    return Budget(**(dict(updates=100,valid_target_presentations=100000,teacher_forward_calls=10000,
        suffix_backward_calls=1000,student_backward_calls=1000,search_proposals=0,search_seconds=0,
        wall_seconds=300,disk_bytes=10**8,memory_bytes=10**9) | changes))


class MemoryRecorder:
    def __init__(self,initial=None): self.rows=[] if initial is None else [CostRecord(**initial)]
    @property
    def totals(self):
        from poc.budget import BudgetTracker
        return BudgetTracker._total(self,[{"cost":r.to_dict()} for r in self.rows])
    def check(self, cost): pass
    def record(self,cost): self.rows.append(cost)


def test_awkward_8200_targets_are_8192_plus_8_without_changing_context():
    t=api(); source=[batch(411) for _ in range(20)]
    updates=list(t.iter_update_targets(source,8192))
    assert [u.valid_targets for u in updates]==[8192,8]
    assert sum(len(u.batches) for u in updates)==21
    for u in updates:
        for b in u.batches:
            assert b.input_ids.shape==(1,411) and b.positions.tolist()==[list(range(411))]
    counts=torch.stack([b.target_mask.to(torch.int32) for u in updates for b in u.batches[-1:]])
    assert counts.sum().item()==410


def test_ragged_masks_accumulate_same_gradient_as_whole_population():
    t=api(); teacher,student=models(); reference=deepcopy(student)
    batches=[batch(9,[True,False,True,True,False,True,True,True]),batch(4)]
    update=list(t.iter_update_targets(batches,8192))[0]
    from poc.losses import kl_sum,nll_sum
    owners=t.configure_trainable_owners(student,t.TrainingConfig(max_updates=1))
    for b in batches:
        logits=reference.logits(b)[:,:-1]; target=teacher.logits(b)[:,:-1]
        ((kl_sum(target,logits,b.target_mask)+.1*nll_sum(logits,b.input_ids[:,1:],b.target_mask))/9).backward()
    t.backward_update(student,teacher,update,t.TrainingConfig(max_updates=1),MemoryRecorder())
    for name,p in owners.items():
        torch.testing.assert_close(p.grad,dict(reference.named_parameters())[name].grad,atol=2e-15,rtol=1e-12)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for n,p in student.named_parameters() if n not in owners)


def test_empty_population_never_steps_and_is_not_success():
    t=api(); teacher,student=models(); rec=MemoryRecorder()
    result=t.fit_model(student,teacher,t.TrainingSources(natural=(batch(3,[False,False]),)),
                       t.TrainingConfig(max_updates=2),budget(),rec)
    assert result.outcome=='invalid' and result.state.step==0
    assert any(e['kind']=='skipped' for e in result.state.events)
    assert sum(r.updates for r in rec.rows)==0


def test_local_owners_freeze_unaffected_bank_and_residuals():
    t=api(); teacher,student=models()
    owners=t.configure_trainable_owners(student,t.TrainingConfig(max_updates=2,phase='local',affected_depths=(0,1)))
    assert set(owners)=={n for n,_ in student.named_parameters() if n.startswith('bank.0.') or n.startswith('layers.0.mlp.residuals.') or n.startswith('layers.1.mlp.residuals.')}
    assert len({id(p) for p in owners.values()})==len(owners)


def test_tiny_global_learning_and_schedule():
    t=api(); teacher,student=models(); rec=MemoryRecorder()
    sources=t.TrainingSources(natural=(batch(9),batch(5)))
    from poc.losses import stream_metrics
    def loss():
        values=[stream_metrics(teacher,student,b,32) for b in sources.natural]
        return sum(v.kl+.1*v.nll for v in values)/12
    before=loss()
    result=t.fit_model(student,teacher,sources,t.TrainingConfig(max_updates=12,epochs=12),budget(),rec)
    assert result.state.step==12 and result.outcome=='exhausted'
    assert loss()<before
    assert result.state.scheduler['completed_updates']==12
    assert sum(r.valid_target_presentations for r in rec.rows)==144


def test_failed_backward_resets_partial_gradients_and_charges_work(monkeypatch):
    t=api(); teacher,student=models(); rec=MemoryRecorder()
    original=t.backward_distillation
    def failing(*args,**kwargs):
        original(*args,**kwargs)
        raise FloatingPointError('late nonfinite chunk')
    monkeypatch.setattr(t,'backward_distillation',failing)
    before={n:p.clone() for n,p in student.named_parameters()}
    with pytest.raises(FloatingPointError):
        t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),t.TrainingConfig(max_updates=2),budget(),rec)
    assert all(p.grad is None for p in student.parameters())
    assert all(torch.equal(p,before[n]) for n,p in student.named_parameters())
    assert sum(r.valid_target_presentations for r in rec.rows)==8
    assert sum(r.student_backward_calls for r in rec.rows)==1


def local_sources(teacher,student):
    t=api()
    from test_patching import record
    from poc.probes import prefix_for_probe,apply_probe
    templates=[]
    for source in ('natural','student','donor'):
        tokens=(1,4,7,2,3); mask=(True,)*4 if source!='donor' else (False,False,True,True)
        b=replace(batch(5),input_ids=torch.tensor([tokens]),document_ids=('d0',),cluster_ids=('c0',),target_mask=torch.tensor([mask]))
        def build(student,teacher,recorder,source=source,b=b):
            producer=student if source=='student' else teacher
            recipient=replace(b,target_mask=torch.ones_like(b.target_mask))
            bound=prefix_for_probe(producer,recipient,1,source='candidate' if source=='student' else 'teacher',recorder=recorder)
            r=record(source=source,patch_positions=(2,) if source=='donor' else (),target_mask=tuple(b.target_mask[0].tolist()),
                prefix_source='candidate' if source=='student' else 'teacher',teacher_digest=teacher.manifest().checkpoint_digest,
                candidate_digest=student.manifest().checkpoint_digest)
            if source=='donor':
                donor=replace(recipient,input_ids=torch.tensor([r.donor_tokens]),document_ids=(r.donor_id,))
                bound=apply_probe(bound,prefix_for_probe(teacher,donor,1,source='teacher',recorder=recorder),r)
            return bound,r
        templates.append(t.ProbeTrainingSource(b,source,'probe_fit',source,build))
    return t.TrainingSources(probes=tuple(templates))


def test_local_required_mixture_uses_source_denominators_and_learns():
    t=api();teacher,student=models();sources=local_sources(teacher,student)
    cfg=t.TrainingConfig(max_updates=5,epochs=5,phase='local',affected_depths=(0,1),
                         mixture_weights={'natural':.5,'student':.25,'donor':.25})
    frozen={n:p.clone() for n,p in student.named_parameters() if n.startswith('bank.1.') or not ('mlp.residuals' in n or n.startswith('bank.'))}
    result=t.fit_model(student,teacher,sources,cfg,budget(),MemoryRecorder())
    assert result.state.probe_cursor==15 and result.state.target_cursor==50
    assert all(m['source_counts']=={'natural':4,'student':4,'donor':2} for m in result.metrics)
    assert result.metrics[-1]['kl_sum']<result.metrics[0]['kl_sum']
    assert all(torch.equal(p,frozen[n]) for n,p in student.named_parameters() if n in frozen)
    assert all(p.grad is None for p in teacher.parameters())
    with pytest.raises(ValueError,match='mixture source absent'):
        t.fit_model(student,teacher,replace(sources,probes=sources.probes[:1]),cfg,budget(),MemoryRecorder())


def test_probe_subset_mask_preserves_eligible_record_and_matches_full_gradient():
    t=api();teacher,student=models();item=local_sources(teacher,student).probes[2]
    from poc.probes import backward_local_discrepancy
    boundary,record=item.build(student,teacher,MemoryRecorder())
    original=boundary.batch.target_mask.clone()
    grads=[]
    for selected in (original, torch.tensor([[False,False,True,False]]),torch.tensor([[False,False,False,True]])):
        student.zero_grad(set_to_none=True)
        backward_local_discrepancy(teacher,student.layers[1].mlp,boundary,chunk_size=32,denominator=2,record=record,loss_mask=selected)
        grads.append({n:p.grad.clone() for n,p in student.named_parameters() if p.grad is not None})
    for n in grads[0]:torch.testing.assert_close(grads[0][n],grads[1][n]+grads[2][n],atol=1e-15,rtol=1e-11)
    assert torch.equal(boundary.batch.target_mask,original) and record.target_mask==(False,False,True,True)
    with pytest.raises(ValueError,match='subset'):
        backward_local_discrepancy(teacher,student.layers[1].mlp,boundary,chunk_size=32,denominator=4,record=record,loss_mask=torch.ones_like(original))


def test_runspec_cannot_silently_omit_or_misstate_resolved_training_config():
    from test_records import spec
    t=api();teacher,student=models()
    with pytest.raises(ValueError,match='resolved training'):
        t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),
            t.TrainingConfig(max_updates=1,run_spec=spec(budgets=budget(),resolved_config={'training':{'max_updates':99}})),budget(),MemoryRecorder())

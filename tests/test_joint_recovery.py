"""Owned joint-recovery scope controls for multiple complete shared banks."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from poc.bank import make_bank,canonicalize_partition
from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
from poc.recovery_objectives import RecoveryObjectiveSpec,affected_owners,owner_names_hash
from poc.recovery_study import NaturalWindowPlan
from poc.train import TrainingConfig,TrainingSources,configure_trainable_owners,backward_update,iter_update_targets,fit_model
from test_training import batch,budget,MemoryRecorder
from test_records import spec as run_spec


def setup(scope='full_student'):
    teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,
        num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024),seed=3,dtype=torch.float64)
    teacher.tokenizer_digest='a'*64
    student=make_bank(teacher,canonicalize_partition([[0,1],[2,3],[4],[5]],6),1,17)
    plan=NaturalWindowPlan.build((batch(512),),updates=2,seed=17,manifest_hashes={'owned':'a'*64})
    depths=(0,1,2,3)
    identity=owner_names_hash(student,depths,owner_scope='affected_groups')
    spec=RecoveryObjectiveSpec(scope,'affected_groups',depths,plan.digest,identity,schema_version=3)
    cfg=TrainingConfig(max_updates=2,alpha=0,affected_depths=depths,record_norms=True)
    bound=replace(run_spec(budgets=budget()),resolved_config={'training':cfg.identity(),'recovery_objective':spec.identity()},
        input_manifests={'raw_plan':plan.digest,'owner_names':identity})
    return teacher,student,plan,spec,replace(cfg,run_spec=bound)


@pytest.mark.parametrize('scope',['local_sum','local_mean','full_student'])
def test_joint_scope_has_exact_owners_and_matches_direct_gradient(scope):
    from poc.losses import kl_sum
    teacher,student,plan,spec,cfg=setup(scope)
    owners=configure_trainable_owners(student,cfg,objective_spec=spec)
    assert len(owners)==30
    assert all(n.startswith(('bank.0.','bank.1.','layers.0.mlp.residuals.','layers.1.mlp.residuals.',
                             'layers.2.mlp.residuals.','layers.3.mlp.residuals.')) for n in owners)
    reference=deepcopy(student);b=batch(9)
    if scope=='full_student':
        loss=kl_sum(teacher.logits(b)[:,:-1],reference.logits(b)[:,:-1],b.target_mask)/8
    else:
        loss=0
        for d in spec.affected_depths:
            boundary=teacher.prefix_boundary(b,d)
            target=teacher.suffix_logits(teacher.ff_residual(boundary),boundary)[:,:-1]
            actual=teacher.suffix_logits(teacher.ff_residual(boundary,reference.layers[d].mlp),boundary)[:,:-1]
            loss=loss+kl_sum(target,actual,b.target_mask)/8
        loss=loss*spec.scale
    loss.backward();rec=MemoryRecorder();update=next(iter_update_targets((b,),8192))
    backward_update(student,teacher,update,cfg,rec,objective_spec=spec,raw_templates=(b,))
    for name,p in owners.items():
        expected=dict(reference.named_parameters())[name].grad
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected,atol=2e-14,rtol=2e-10)
    assert all(p.grad is None for n,p in student.named_parameters() if n not in owners)
    assert all(p.grad is None for p in teacher.parameters())
    assert rec.totals.raw_corpus_target_presentations==8
    assert rec.totals.valid_target_presentations==8*spec.scored_multiplier


def test_joint_scope_rejects_partial_use_and_preserves_old_version():
    teacher,student,plan,spec,cfg=setup()
    before=[p.requires_grad for p in student.parameters()]
    with pytest.raises(ValueError):affected_owners(student,(0,1,2),owner_scope='affected_groups')
    with pytest.raises(ValueError):affected_owners(student,(0,1),owner_scope='affected_groups')
    with pytest.raises(ValueError):affected_owners(student,(0,1,2,3))
    with pytest.raises(ValueError):replace(spec,schema_version=1)
    with pytest.raises(ValueError):replace(spec,owner_scope='affected_group')
    with pytest.raises(ValueError):configure_trainable_owners(student,cfg,objective_spec=replace(spec,owner_names_sha256='0'*64))
    assert before==[p.requires_grad for p in student.parameters()]


def test_joint_common_loop_preserves_frozen_owners_and_resume(tmp_path):
    from poc.checkpoint import save_checkpoint,load_checkpoint
    teacher,student,plan,spec,cfg=setup();source=TrainingSources(raw_plan=plan);rec=MemoryRecorder()
    frozen={n:p.detach().clone() for n,p in student.named_parameters()
            if n not in affected_owners(student,spec.affected_depths,owner_scope='affected_groups')}
    first=fit_model(student,teacher,source,replace(cfg,stop_after_updates=1),budget(),rec,objective_spec=spec)
    checkpoint=save_checkpoint(first.state,tmp_path)
    resumed=fit_model(student,teacher,source,replace(cfg,resume=load_checkpoint(checkpoint,cfg.run_spec)),budget(),rec,objective_spec=spec)
    _,whole,_,_,_=setup()
    uninterrupted=fit_model(whole,teacher,source,cfg,budget(),MemoryRecorder(),objective_spec=spec)
    assert resumed.state.events==uninterrupted.state.events
    assert resumed.state.target_cursor==16384
    assert rec.totals.raw_corpus_target_presentations==16384
    for n,p in student.named_parameters():
        assert torch.equal(p,dict(whole.named_parameters())[n])
        if n in frozen:assert torch.equal(p,frozen[n])

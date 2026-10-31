"""Composition controls: preserve current effective operators and correct nonzero baselines."""
import importlib
import importlib.util
import hashlib
from dataclasses import replace

import pytest
import torch

from poc.bank import make_bank,canonicalize_partition
from poc.data.manifest import DataManifest,DocumentRecord
from poc.losses import kl_sum
from test_bank import tiny
from test_model_parity import batch


def api():
    assert importlib.util.find_spec('poc.composition') is not None,'composition implementation missing'
    return importlib.import_module('poc.composition')


def fit():
    return DataManifest((DocumentRecord('d0','owned','v1','fixture',0),),{'d0':'fit'},{'d0':'c0'},'owned')


def test_full_rank_current_continuation_preserves_function_and_fixed_parent():
    teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),8,17)
    with torch.no_grad():
        current.bank[0].gate_proj.weight.add_(.01)
        current.layers[0].mlp.residuals['gate_proj'].B.add_(.1)
    before=current.manifest().checkpoint_digest
    branches=api().paired_composition_initializations(current,teacher,(1,2),fit(),17,batches=(batch(),))
    assert current.manifest().checkpoint_digest==before
    assert branches.teacher_reset.model.partition==branches.current_continuation.model.partition
    assert branches.teacher_reset.selected_parent==branches.current_continuation.selected_parent
    assert branches.teacher_reset.medoid_scores==branches.current_continuation.medoid_scores
    assert branches.current_continuation.affected_depths==(0,1,2)
    torch.testing.assert_close(branches.current_continuation.model.logits(batch()),current.logits(batch()),atol=2e-11,rtol=2e-11)
    assert (branches.teacher_reset.model.logits(batch())-current.logits(batch())).abs().max()>1e-8
    assert branches.record['optimizer_state_policy']=='fresh_optimizer_for_both_branches'
    assert branches.record['source_current_digest']==before


def test_low_rank_continuation_records_unrepresentable_matrix_error():
    teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),1,17)
    branches=api().paired_composition_initializations(current,teacher,(1,2),fit(),17,batches=(batch(),))
    assert any(r['delta_reconstruction_mse']>1e-8 for r in branches.record['projection_records'])
    assert branches.current_continuation.model.rank==branches.teacher_reset.model.rank==1


def test_unaffected_owners_and_input_models_are_unchanged():
    teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
    before=current.manifest().checkpoint_digest
    branches=api().paired_composition_initializations(current,teacher,(0,1),fit(),17,batches=(batch(),))
    reset=dict(branches.teacher_reset.model.named_parameters())
    for name,p in branches.current_continuation.model.named_parameters():
        if not name.startswith(('layers.0.mlp.residuals.','layers.1.mlp.residuals.')):
            torch.testing.assert_close(p,reset[name],atol=0,rtol=0)
    assert current.manifest().checkpoint_digest==before
    for branch in (branches.teacher_reset,branches.current_continuation):
        assert branch.fitted_digest is None
        assert branch.initialized_digest==branch.model.manifest().checkpoint_digest


def observation(a,b,model):
    states=((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.))
    logits=torch.tensor([[a*i+b*j for i,j in states]],dtype=torch.float64).repeat(3,1)
    raw=float(kl_sum(torch.zeros_like(logits),logits,torch.ones(3,dtype=torch.bool)))
    return api().KLObservation(model,'a'*64,'b'*64,raw,3)


def interaction(rows,**changes):
    args=dict(base_state_digest='0'*64,intervention_ids=('1'*64,'2'*64),study_contract_sha256='c'*64,
              composable=True,kl_floor_per_target=1e-12)
    args.update(changes)
    return api().baseline_corrected_interaction(rows,**args)


def additive_observations():
    # Two independent Bernoulli heads represented jointly by a4-class softmax.
    return {'base':observation(.5,0.,'0'*64),'A':observation(.8,0.,'3'*64),
            'B':observation(.5,.4,'4'*64),'AB':observation(.8,.4,'5'*64)}


def test_additive_actual_kl_fixture_requires_nonzero_baseline_correction():
    rows=additive_observations();result=interaction(rows)
    assert rows['base'].kl_sum/3>.1
    assert abs(result['interaction_kl_per_target'])<1e-12
    wrong=(rows['AB'].kl_sum-rows['A'].kl_sum-rows['B'].kl_sum)/3
    assert wrong<-.1


@pytest.mark.parametrize('change',['contexts','teachers','counts','baseline','recipe'])
def test_interaction_rejects_unmatched_or_noncomposable_comparisons(change):
    rows=additive_observations();kwargs={}
    if change=='contexts':rows['A']=replace(rows['A'],context_digest='d'*64)
    if change=='teachers':rows['B']=replace(rows['B'],teacher_digest='d'*64)
    if change=='counts':rows['AB']=replace(rows['AB'],valid_targets=2)
    if change=='baseline':kwargs['base_state_digest']='d'*64
    if change=='recipe':kwargs['composable']=False
    with pytest.raises(ValueError):interaction(rows,**kwargs)


def test_interaction_retains_tiny_negative_raw_estimates_but_rejects_floor_violation():
    rows=additive_observations();rows={k:replace(v,kl_sum=0.) for k,v in rows.items()}
    rows['base']=replace(rows['base'],kl_sum=-1e-13)
    assert interaction(rows)['interaction_kl_per_target']<0
    rows['base']=replace(rows['base'],kl_sum=-1.)
    with pytest.raises(FloatingPointError):interaction(rows)

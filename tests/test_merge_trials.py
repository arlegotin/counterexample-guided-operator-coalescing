"""Catch one-parent initialization and changed singleton state/rank."""
import importlib
import importlib.util
from dataclasses import replace
from copy import deepcopy
import pytest
import torch
from poc.bank import make_bank,canonicalize_partition
from poc.data.manifest import DataManifest,DocumentRecord
from test_bank import tiny
from test_model_parity import batch


def api():
    assert importlib.util.find_spec('poc.merge'), 'controlled merge initialization missing'
    return importlib.import_module('poc.merge')


def fit_manifest():
    return DataManifest((DocumentRecord('d0','fixture','v1','s',0,text='fixture'),),{'d0':'fit'},{'d0':'c0'},'test')


def test_both_parent_medoids_use_identical_fit_inputs_and_preserve_singletons():
    a=api(); t=tiny(); c=make_bank(t,canonicalize_partition([[0],[1],[2]],3),2,17)
    before=c.manifest().checkpoint_digest
    trial=a.initialize_merge(c,t,(0,2),fit_manifest(),17,batches=(batch(),))
    assert tuple(row['parent'] for row in trial.medoid_scores)==(0,2)
    assert len({row['fit_digest'] for row in trial.medoid_scores})==1
    assert trial.model.partition.groups==((0,2),(1,)) and trial.model.rank==2
    assert c.manifest().checkpoint_digest==before
    torch.testing.assert_close(trial.model.layers[1].mlp(batch().input_ids.new_ones((1,5,8)).double()),c.layers[1].mlp(batch().input_ids.new_ones((1,5,8)).double()),atol=0,rtol=0)
    assert trial.selected_parent==min(trial.medoid_scores,key=lambda row:(row['mse'],row['parent']))['parent']
    assert trial.initialized_digest==trial.model.manifest().checkpoint_digest


def test_equal_parent_ties_follow_canonical_group_and_zero_delta_can_learn():
    a=api();t=tiny();t.layers[2].mlp=deepcopy(t.layers[0].mlp)
    c=make_bank(t,canonicalize_partition([[0],[1],[2]],3),2,17)
    trial=a.initialize_merge(c,t,(0,2),fit_manifest(),17,batches=(batch(),))
    assert trial.selected_parent==0
    for depth in (0,2):
        assert torch.count_nonzero(trial.model.layers[depth].mlp.residuals['gate_proj'].A)>0
        assert torch.count_nonzero(trial.model.layers[depth].mlp.residuals['gate_proj'].B)==0
    torch.testing.assert_close(trial.model.logits(batch()),t.logits(batch()),atol=0,rtol=0)


def test_invalid_or_nonfit_population_fails_before_mutation():
    a=api();t=tiny();c=make_bank(t,canonicalize_partition([[0],[1],[2]],3),2,17)
    for pair in ((0,0),(2,0),(0,3)):
        with pytest.raises(ValueError):a.initialize_merge(c,t,pair,fit_manifest(),17,batches=(batch(),))
    with pytest.raises(PermissionError):a.initialize_merge(c,t,(0,2),replace(fit_manifest(),roles={'d0':'dev'}),17,batches=(batch(),))


def test_centered_cka_and_weight_distance_limiting_cases():
    a=api();x=torch.tensor([[1.,2.],[2.,1.],[4.,0.]],dtype=torch.float64)
    assert a.centered_linear_cka(x,2*x+8)==pytest.approx(1.)
    with pytest.raises(ValueError):a.centered_linear_cka(torch.ones_like(x),x)

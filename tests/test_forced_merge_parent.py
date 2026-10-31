"""Bounded parent ablation must change only representative choice."""
from pathlib import Path
import pytest
import torch
from poc.bank import make_bank,canonicalize_partition
from poc.merge import initialize_merge
from poc.composition import paired_composition_initializations
from test_bank import tiny
from test_model_parity import batch
from test_merge_trials import fit_manifest
from test_training import MemoryRecorder
pytestmark=pytest.mark.cpu

def test_forcing_other_parent_preserves_scores_capacity_and_default_winner():
 teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17);before=current.manifest().checkpoint_digest
 ordinary=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),))
 alternative=2 if ordinary.selected_parent==0 else 0
 forced=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),),forced_parent=alternative)
 assert forced.selected_parent==alternative and forced.medoid_scores==ordinary.medoid_scores
 assert forced.model.partition==ordinary.model.partition and forced.model.rank==ordinary.model.rank
 winner=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),),forced_parent=ordinary.selected_parent)
 assert winner.initialized_digest==ordinary.initialized_digest
 assert current.manifest().checkpoint_digest==before
 torch.testing.assert_close(forced.model.layers[1].mlp(torch.ones(1,5,8,dtype=torch.float64)),current.layers[1].mlp(torch.ones(1,5,8,dtype=torch.float64)),rtol=0,atol=0)

@pytest.mark.parametrize('forced',[True,-1,1,3,0.0])
def test_forced_parent_requires_one_canonical_group_representative_before_work(forced):
 teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),2,17);before=current.manifest().checkpoint_digest;rec=MemoryRecorder()
 with pytest.raises(ValueError,match='parent'):
  initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),),forced_parent=forced,recorder=rec)
 assert rec.totals.search_proposals==rec.totals.teacher_forward_calls==0
 assert current.manifest().checkpoint_digest==before

@pytest.mark.parametrize('parent',[0,2])
def test_composition_recipes_share_explicit_forced_parent_and_record_policy(parent):
 teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),2,17)
 branches=paired_composition_initializations(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),),forced_parent=parent)
 assert branches.teacher_reset.selected_parent==branches.current_continuation.selected_parent==parent
 assert branches.record['schema_version']==2 and branches.record['forced_parent']==parent
 assert branches.record['parent_policy']=='forced_current_group_representative'
 assert branches.teacher_reset.model.partition==branches.current_continuation.model.partition


def test_forced_policy_survives_trial_delta_and_partial_metadata_is_rejected(tmp_path):
 import json
 from poc.merge import save_trial_delta,restore_trial_delta
 teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
 ordinary=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),));parent=2 if ordinary.selected_parent==0 else 0
 forced=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),),forced_parent=parent);forced.fitted_digest=forced.model.manifest().checkpoint_digest
 save_trial_delta(forced,current,tmp_path/'delta');restored=restore_trial_delta(current,teacher,tmp_path/'delta')
 assert restored.parent_policy==forced.parent_policy and restored.parent_policy['parent']==parent
 assert restored.fitted_digest==forced.fitted_digest
 path=tmp_path/'delta/metadata.json';metadata=json.loads(path.read_text());metadata.pop('initialization_policy');path.write_text(json.dumps(metadata))
 with pytest.raises(ValueError,match='policy'):restore_trial_delta(current,teacher,tmp_path/'delta')

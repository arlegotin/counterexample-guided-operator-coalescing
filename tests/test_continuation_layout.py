"""Canonical factors must survive both serialization paths without changing owners."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from poc.bank import PROJECTIONS, initialize_residual, make_bank, canonicalize_partition, owner_census
from poc.bundle import write_bank_state, load_bank_state, write_bundle, load_bundle
from poc.merge import initialize_merge, save_trial_delta, restore_trial_delta
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from test_bank import tiny
from test_model_parity import batch
from test_merge_trials import fit_manifest
from test_export import runtime, model


def assert_factors(model):
    for layer in model.layers:
        for factors in layer.mlp.residuals.values():
            assert factors.A.is_contiguous() and factors.B.is_contiguous()
    assert model.lm_head.weight is model.embed_tokens.weight
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
    for d, layer in enumerate(model.layers):
        assert layer.mlp._owners_ref() is model.bank
        assert layer.mlp.index == model.partition.reuse_map[d]


def assert_roundtrip(source, restored):
    assert_factors(source)
    assert_factors(restored)
    assert owner_census(source) == owner_census(restored)
    assert source.manifest().checkpoint_digest == restored.manifest().checkpoint_digest
    for (n, p), (m, q) in zip(source.named_parameters(), restored.named_parameters(), strict=True):
        assert n == m and torch.equal(p, q)
        assert p is not q
    torch.testing.assert_close(source.logits(batch()), restored.logits(batch()), atol=1e-12, rtol=0)


@pytest.mark.parametrize('rank', [0, 2, 8])
def test_initialization_layout_and_external_state_roundtrip(tmp_path, rank):
    delta = torch.arange(96, dtype=torch.float64).reshape(12, 8).sin()
    a, b = initialize_residual(delta, rank, 17)
    assert a.is_contiguous() and b.is_contiguous()
    teacher = tiny()
    source = make_bank(teacher, canonicalize_partition([[0, 2], [1]], 3), rank, 17)
    restored = load_bank_state(write_bank_state(source, tmp_path/'state'), teacher)
    assert_roundtrip(source, restored)


def test_merge_compact_delta_layout_values_and_ownership(tmp_path):
    teacher = tiny()
    current = make_bank(teacher, canonicalize_partition([[0], [1], [2]], 3), 2, 17)
    trial = initialize_merge(current, teacher, (0, 2), fit_manifest(), 17, batches=(batch(),))
    trial.fitted_digest = trial.model.manifest().checkpoint_digest
    save_trial_delta(trial, current, tmp_path/'delta')
    restored = restore_trial_delta(current, teacher, tmp_path/'delta')
    assert_roundtrip(trial.model, restored.model)


def test_incompatible_teacher_lineage_rejected_before_merge():
    teacher = tiny()
    current = make_bank(teacher, canonicalize_partition([[0], [1], [2]], 3), 2, 17)
    current.base_digest = '0'*64
    with pytest.raises(ValueError, match='lineage'):
        initialize_merge(current, teacher, (0, 2), fit_manifest(), 17, batches=(batch(),))


def test_old_profile_bundle_rejected_and_current_metadata_bound(tmp_path):
    ref = write_bundle(model('2'), tmp_path/'bundle', runtime(tmp_path))
    path = tmp_path/'bundle/manifest.json'
    raw = json.loads(path.read_text())
    assert raw['execution_profile'] == EXECUTION_PROFILE
    raw['execution_profile'] = 'eager-llama-hf-fp32-reciprocal-true-fp64-v2'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='profile'):
        load_bundle(tmp_path/'bundle', backend='torch')


def test_external_state_rejects_old_profile_even_with_rehashed_manifest(tmp_path):
    teacher = tiny()
    source = make_bank(teacher, canonicalize_partition([[0, 2], [1]], 3), 2, 17)
    ref = write_bank_state(source, tmp_path/'state')
    path = tmp_path/'state/state.json'
    raw = json.loads(path.read_text())
    assert raw['execution_profile'] == EXECUTION_PROFILE
    raw['execution_profile'] = 'eager-llama-hf-fp32-reciprocal-true-fp64-v2'
    path.write_text(json.dumps(raw))
    ref = replace(ref, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='profile'):
        load_bank_state(ref, teacher)


def test_zero_product_has_live_b_gradient():
    a, b = initialize_residual(torch.zeros(12, 8, dtype=torch.float64), 2, 17)
    assert a.count_nonzero() and not b.count_nonzero()
    a.requires_grad_(); b.requires_grad_()
    (b @ a).square().sub(2*(b @ a)).sum().backward()
    assert b.grad.count_nonzero() and torch.isfinite(b.grad).all()


def test_compact_delta_retains_local_owner_trainability_and_rejects_old_profile(tmp_path):
    from poc.train import configure_trainable_owners, TrainingConfig
    teacher=tiny()
    current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
    trial=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(batch(),))
    configure_trainable_owners(trial.model,TrainingConfig(max_updates=1,phase='local',affected_depths=(0,2)))
    trial.fitted_digest=trial.model.manifest().checkpoint_digest
    save_trial_delta(trial,current,tmp_path/'delta')
    restored=restore_trial_delta(current,teacher,tmp_path/'delta')
    assert owner_census(restored.model)==owner_census(trial.model)
    path=tmp_path/'delta/metadata.json'
    meta=json.loads(path.read_text());meta['execution_profile']='old'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='profile'):restore_trial_delta(current,teacher,tmp_path/'delta')


def test_checkpoint_loader_rejects_old_profile_with_matching_expected_spec(tmp_path):
    from poc.train import fit_model,TrainingConfig,TrainingSources
    from poc.checkpoint import save_checkpoint,load_checkpoint
    from test_training import models,MemoryRecorder,budget
    from test_records import spec
    teacher,student=models()
    cfg=TrainingConfig(max_updates=1)
    cfg=replace(cfg,run_spec=spec(budgets=budget(),resolved_config={'training':cfg.identity()}))
    result=fit_model(student,teacher,TrainingSources(natural=(batch(),)),cfg,budget(),MemoryRecorder())
    # Save a current checkpoint, then emulate a historical state with its own
    # valid manifest. save_checkpoint verifies by calling load_checkpoint.
    result.state.model_schema['execution_profile']='eager-llama-hf-fp32-reciprocal-true-fp64-v2'
    with pytest.raises(ValueError,match='profile'):save_checkpoint(result.state,tmp_path)


def test_reload_verification_charges_both_forwards():
    from poc import continuation_controls as controls
    from test_training import MemoryRecorder
    assert hasattr(controls,'reload_difference'), 'charged reload helper missing'
    rec=MemoryRecorder();teacher=tiny()
    assert controls.reload_difference(teacher,teacher,batch(),rec)==0
    assert sum(row.student_forward_calls for row in rec.rows)==2

"""Real-context planning, fixed-parent ablation and pre-clip instrumentation."""
from dataclasses import replace
import importlib.util
import math

import pytest
import torch

from poc.train import ProbeTrainingSource, TrainingConfig, TrainingSources, iter_update_targets, fit_model
from test_training import models, MemoryRecorder, budget, batch


def api():
    assert importlib.util.find_spec('poc.recovery_plan'), 'variable-context recovery plan missing'
    return __import__('poc.recovery_plan',fromlist=['*'])


def test_variable_masks_fill_finite_updates_without_slicing_full_builders():
    a=api();items=[];depths={}
    for source,n in [('natural',512),('student',512),('donor',257)]:
        for depth in (0,2):
            item=ProbeTrainingSource(batch(n),source,'fit',f'{source}-{depth}',lambda *_:None)
            items.append(item);depths[item.identity]=depth
    plan=a.finite_plan(items,depths,updates=3,weights={'natural':.5,'student':.25,'donor':.25},affected_depths=(0,2))
    updates=list(iter_update_targets([p.template for p in plan.sources.probes],8192))
    assert all(u.valid_targets==8192 for u in updates[:3])
    assert sum(u.valid_targets for u in updates[3:])==plan.metadata['unconsumed_tail_targets']
    assert plan.metadata['logical_targets']==24576
    assert all(all(n>0 for n in row['source_counts'].values()) for row in plan.metadata['updates'])
    assert all(row['depths']==[0,2] for row in plan.metadata['updates'])
    assert all(any(p.template is item.template for item in items) for p in plan.sources.probes)
    assert len(plan.metadata['presentations'])==len(plan.sources.probes)


def test_fixed_parent_zero_product_keeps_bank_and_original_state_isolated():
    a=api()
    from poc.merge import initialize_merge
    from poc.bank import make_bank,canonicalize_partition
    from test_bank import tiny
    from test_merge_trials import fit_manifest
    from test_model_parity import batch as small_batch
    teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
    trial=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(small_batch(),))
    before=trial.model.manifest().checkpoint_digest
    zero=a.fixed_parent_copy(trial,zero_product=True,seed=17)
    assert trial.model.manifest().checkpoint_digest==before
    assert zero.selected_parent==trial.selected_parent
    for p,q in zip(zero.model.bank.parameters(),trial.model.bank.parameters(),strict=True):
        assert torch.equal(p,q) and p is not q
    for d in zero.affected_depths:
        for f in zero.model.layers[d].mlp.residuals.values():
            assert f.A.count_nonzero() and not f.B.count_nonzero()


def test_training_reports_gradient_before_clipping_and_actual_update_norm():
    assert 'record_norms' in TrainingConfig.__dataclass_fields__, 'optimizer instrumentation missing'
    teacher,student=models()
    result=fit_model(student,teacher,TrainingSources(natural=(batch(),)),
        TrainingConfig(max_updates=1,record_norms=True),budget(),MemoryRecorder())
    row=result.metrics[0]
    assert math.isfinite(row['gradient_norm_pre_clip']) and row['gradient_norm_pre_clip']>0
    assert math.isfinite(row['parameter_update_norm']) and row['parameter_update_norm']>0
    assert row['source_kl_sums']=={'natural':row['kl_sum']}
    assert row['weighted_objective']==pytest.approx((row['kl_sum']+.1*row['nll_sum'])/row['valid_targets'])


def test_reader_denies_protected_metadata_before_any_content_open(tmp_path):
    assert importlib.util.find_spec('poc.recovery_data'), 'recovery reader missing'
    from poc.recovery_data import restricted_population
    from poc.data.manifest import DataManifest,DocumentRecord
    m=DataManifest((DocumentRecord('secret','fixture','v1','s',0,text_path='/must-not-open'),),
        {'secret':'confirm_A'},{'secret':'component'},'fixture',str(tmp_path/'exposure'))
    with pytest.raises(PermissionError):restricted_population(m,['secret'],role='fit')


def test_child_checkpoint_disk_preflight_uses_complete_parent_tree(tmp_path):
    from poc.continuation_budget import PilotLedger
    from test_records import spec
    ledger=PilotLedger(spec(),tmp_path)
    ledger.tracker.disk=lambda:7654
    with ledger.arm(spec(method_variant='disk-child')) as recorder:
        assert hasattr(recorder,'disk'), 'child does not expose aggregate checkpoint disk'
        assert recorder.disk()==7654


def test_shared_nested_work_wall_is_counted_once(tmp_path,monkeypatch):
    from poc.continuation_budget import PilotLedger
    import poc.continuation_budget as module
    from poc.contracts import CostRecord
    from test_records import spec
    ledger=PilotLedger(spec(),tmp_path)
    ticks=iter((0.,10.));monkeypatch.setattr(module.time,'monotonic',lambda:next(ticks))
    ledger.shared('nested',CostRecord(),lambda:ledger.tracker.record(CostRecord(wall_seconds=4)))
    assert ledger.tracker.totals.wall_seconds==10


def test_final_cost_overrun_cannot_leave_success_status(tmp_path):
    from provenance import recovery_pilot as runner
    from poc.continuation_budget import PilotLedger
    from test_records import spec,budget as small_budget
    ledger=PilotLedger(spec(budgets=replace(small_budget(),wall_seconds=1)),tmp_path)
    result={'status':'complete_exploratory'}
    assert hasattr(runner,'finalize_costs'), 'final accounting guard missing'
    runner.finalize_costs(result,ledger.tracker,elapsed=2)
    assert result['status']=='failed_accounting'
    assert ledger.tracker.totals.wall_seconds==2


def test_local_weighted_objective_uses_each_source_own_denominator():
    from test_training import local_sources
    teacher,student=models()
    cfg=TrainingConfig(max_updates=1,phase='local',affected_depths=(0,1),
        mixture_weights={'natural':.5,'student':.25,'donor':.25})
    row=fit_model(student,teacher,local_sources(teacher,student),cfg,budget(),MemoryRecorder()).metrics[0]
    assert row['source_counts']=={'natural':4,'student':4,'donor':2}
    sums=row['source_kl_sums']
    assert row['weighted_objective']==pytest.approx(.5*sums['natural']/4+.25*sums['student']/4+.25*sums['donor']/2)


def test_full_rank_floor_reconstructs_actual_operator_on_owned_inputs():
    from provenance import recovery_pilot as runner
    from poc.merge import initialize_merge
    from poc.bank import make_bank,canonicalize_partition
    from test_bank import tiny
    from test_merge_trials import fit_manifest
    from test_model_parity import batch as small_batch
    teacher=tiny();current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
    trial=initialize_merge(current,teacher,(0,2),fit_manifest(),17,batches=(small_batch(),))
    assert hasattr(runner,'full_rank_floor'), 'real-shape full-rank control missing'
    bound=teacher.prefix_boundary(small_batch(),2)
    value=runner.full_rank_floor(teacher,trial,bound,MemoryRecorder())
    assert abs(value)<1e-12

"""Fail closed on budgets, incomplete plans and unsafe score exports."""
from copy import deepcopy
from dataclasses import replace
import importlib.util

import pytest
import torch

from poc.contracts import CostRecord
from test_records import spec, budget
from test_model_parity import batch


def module(name):
    assert importlib.util.find_spec(name), f'{name} missing'
    return __import__(name, fromlist=['*'])


def test_parent_budget_denies_next_arm_and_retains_failures_and_replay(tmp_path):
    api = module('poc.continuation_budget')
    from poc.budget import BudgetExceeded
    parent_spec = spec(budgets=replace(budget(), teacher_forward_calls=3))
    ledger = api.PilotLedger(parent_spec, tmp_path)
    ledger.shared('prepare', CostRecord(teacher_forward_calls=1), lambda: 7)
    ledger.shared('prepare', CostRecord(teacher_forward_calls=1), lambda: pytest.fail('duplicate shared charge'))
    child = spec(method_variant='arm-A', budgets=replace(budget(), teacher_forward_calls=3))
    with pytest.raises(RuntimeError, match='deliberate'):
        with ledger.arm(child, dependencies=('prepare',)) as recorder:
            recorder.check(CostRecord(teacher_forward_calls=1))
            recorder.record(CostRecord(teacher_forward_calls=1))
            raise RuntimeError('deliberate')
    with ledger.arm(child, dependencies=('prepare',)) as recorder:
        recorder.record(CostRecord(teacher_forward_calls=1))
    restored = api.PilotLedger(parent_spec, tmp_path)
    assert restored.tracker.totals.teacher_forward_calls == 3
    with pytest.raises(BudgetExceeded):
        with restored.arm(spec(method_variant='arm-B'), forecast=CostRecord(teacher_forward_calls=1)):
            pytest.fail('exhausted arm must not dispatch')
    assert restored.tracker.totals.teacher_forward_calls == 3
    assert [r['status'] for r in restored.graph() if r['kind']=='finish'] == ['complete', 'failed', 'complete']


def test_plan_full_updates_balance_and_preserve_full_builders():
    api = module('poc.continuation_plan')
    from poc.train import ProbeTrainingSource, iter_update_targets
    # 4 eligible targets/template: 2048 presentations/update. Whole templates
    # retain original masks and builders; no synthetic completion of epoch tails.
    items = []
    depths = {}
    for source in ('natural', 'student', 'donor'):
        for depth in (0, 2):
            identity = f'{source}-{depth}'
            b = batch()
            items.append(ProbeTrainingSource(b, source, 'fit', identity, lambda *_: None))
            depths[identity] = depth
    plan = api.build_target_plan(tuple(items), depths, updates=2,
        weights={'natural': .5, 'student': .25, 'donor': .25}, affected_depths=(0, 2))
    updates = tuple(iter_update_targets(tuple(p.template for p in plan.sources.probes), 8192))
    assert [u.valid_targets for u in updates] == [8192, 8192]
    for u in updates:
        counts = {s: 0 for s in ('natural', 'student', 'donor')}
        for b, (i, _, _) in zip(u.batches, u.slices, strict=True):
            counts[plan.sources.probes[i].source] += int(b.target_mask.sum())
        assert counts == {'natural':4096, 'student':2048, 'donor':2048}
    assert all(any(p.template is original.template and p.identity==original.identity
        and p.build.__wrapped__ is original.build for original in items) for p in plan.sources.probes)
    assert plan.metadata['logical_targets'] == 16384
    assert plan.metadata['unconsumed_tail_targets'] == 0
    assert plan.metadata['unique_target_positions'] == 4
    assert plan.digest != api.build_target_plan(tuple(items), depths, updates=1,
        weights={'natural':.5,'student':.25,'donor':.25}, affected_depths=(0,2)).digest
    with pytest.raises(ValueError, match='source|coverage'):
        api.build_target_plan(tuple(items[:-1]), depths, updates=1,
            weights={'natural':.5,'student':.25,'donor':.25}, affected_depths=(0,2))


def numeric_fixture():
    def h(n): return f'{n:064x}'
    candidates = []
    for index, pair in enumerate(((0,1),(0,2))):
        observations = []
        for ci in (10, 11):
            start=len(observations)
            for source, value in [('natural',2+index),('student',4),('random',6),('active',8),('active',10)]:
                observations.append(dict(component_sha256=h(ci), window_sha256=h(ci+100),
                    group_sha256=h(ci+200), probe_sha256=h(len(observations)+1000),
                    source=source, kl_sum=value*2., valid_targets=2))
            observations.append({**observations[start+2],'source':'active'})
        candidates.append(dict(pair=list(pair), inclusion_probability=1., status='valid',
            failure_code=None, initialized_digest=h(30+index), fitted_digest=h(40+index),
            observations=observations, damage=[dict(component_sha256=h(50),window_sha256=h(150),
                kl_sum=float(index+1),valid_targets=1)],query_node_ids=[h(60)]))
    return dict(schema_version=1, scope='owned_fixture', decision_resolution=1e-12,
        candidate_universe=[[0,1],[0,2]], score_components=[h(10),h(11)], damage_components=[h(50)],
        candidates=candidates, query_graph=[dict(node_id=h(60),dependencies=[],cost=CostRecord().to_dict())])


def test_numeric_export_reconstructs_every_fixed_formula_selection_and_ties():
    api = module('poc.continuation_evidence')
    result = api.export_numeric(numeric_fixture())
    # Hand-derived: N=2, S=4, R=6, A=max(8,10)=10.
    assert {k:v[0][0] for k,v in result['score_matrices'].items()} == dict(
        natural_kl=2., student_only=4., random_donor=7., active_1=9., active_01=2.7)
    assert all(v['selected_pair']==[0,1] for v in result['selections'].values())
    assert result['damage_matrix'] == [[1.],[2.]]
    assert result['primary_active_minus_natural_damage'] == 0


@pytest.mark.parametrize('attack',['candidate','denominator','private','component','group','graph'])
def test_numeric_export_rejects_incomplete_or_unsafe_evidence(attack):
    api = module('poc.continuation_evidence')
    value = numeric_fixture()
    if attack=='candidate':value['candidates'].pop()
    elif attack=='denominator':value['candidates'][0]['observations'][0]['valid_targets']=0
    elif attack=='private':value['candidates'][0]['text']='secret'
    elif attack=='component':value['candidates'][0]['observations'].pop(0)
    elif attack=='group':value['candidates'][0]['observations'][1]['group_sha256']='f'*64
    else:value['query_graph'][0]['dependencies']=['f'*64]
    with pytest.raises(ValueError):api.export_numeric(value)


def test_schema_rejects_runtime_profile_horizon_and_unknown_fields():
    api = module('poc.continuation')
    from pathlib import Path
    cfg = api.load_config(Path('configs/continuation-recovery-v1.yaml'))
    resolved = api.resolve_config(cfg)
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    assert resolved['numerical']['execution_profile'] == EXECUTION_PROFILE
    for section, key, value in [('numerical','execution_profile','old'),
            ('training','observation_steps',[0,1]),('training','target_count',8191)]:
        bad=deepcopy(cfg);bad[section][key]=value
        with pytest.raises(ValueError):api.resolve_config(bad)
    cfg['unknown']=True
    with pytest.raises(ValueError):api.resolve_config(cfg)


@pytest.mark.parametrize('field,value',[('initialized_digest','PRIVATE TEXT'),('fitted_digest',{'tokens':[1,2]})])
def test_failed_rows_cannot_export_private_payloads(field,value):
    api=module('poc.continuation_evidence');data=numeric_fixture()
    c=data['candidates'][0]
    c.update(status='failed',failure_code='budget',observations=[],damage=[])
    c[field]=value
    with pytest.raises(ValueError):api.export_numeric(data)


def test_damage_windows_and_denominators_must_be_common():
    api=module('poc.continuation_evidence');data=numeric_fixture()
    data['candidates'][1]['damage'][0]['window_sha256']='f'*64
    data['candidates'][1]['damage'][0]['valid_targets']=17
    with pytest.raises(ValueError):api.export_numeric(data)


@pytest.mark.parametrize('kind',['shared','arm'])
def test_accounting_overrun_is_failed_and_not_reusable(tmp_path,monkeypatch,kind):
    api=module('poc.continuation_budget')
    from poc.budget import BudgetExceeded
    ledger=api.PilotLedger(spec(budgets=replace(budget(),wall_seconds=1)),tmp_path)
    times=iter((0.,2.));monkeypatch.setattr(api.time,'monotonic',lambda:next(times))
    with pytest.raises(BudgetExceeded):
        if kind=='shared':ledger.shared('calibration',CostRecord(),lambda:None)
        else:
            with ledger.arm(spec()):pass
    assert ledger.graph()[-1]['status']=='failed'
    assert ledger.tracker.totals.wall_seconds==2
    if kind=='shared':
        with pytest.raises(BudgetExceeded):ledger.shared('calibration',CostRecord(),lambda:pytest.fail('reused overrun'))


def test_random_donor_must_be_a_member_of_the_active_pool():
    api=module('poc.continuation_evidence');data=numeric_fixture()
    for candidate in data['candidates']:
        for row in candidate['observations']:
            if row['source']=='random':row['probe_sha256']='f'*64
    with pytest.raises(ValueError):api.export_numeric(data)


def test_planned_depth_is_checked_against_actual_full_record_before_backward():
    api=module('poc.continuation_plan')
    from test_training import models,local_sources,MemoryRecorder
    teacher,student=models()
    natural=local_sources(teacher,student).probes[0]  # actual depth 1
    alias=replace(natural,identity='other')
    plan=api.build_target_plan((natural,alias),{natural.identity:0,alias.identity:1},
        updates=1,weights={'natural':1.},affected_depths=(0,1))
    with pytest.raises(ValueError,match='declared depth'):
        plan.sources.probes[0].build(student,teacher,MemoryRecorder())


def test_live_peak_observation_enforces_cap_and_retains_overrun(tmp_path,monkeypatch):
    api=module('poc.continuation_budget')
    from poc.budget import BudgetExceeded
    assert hasattr(api,'process_peak_bytes'), 'CPU high-water observer missing'
    monkeypatch.setattr(api,'process_peak_bytes',lambda:20000)
    ledger=api.PilotLedger(spec(),tmp_path,observe_resources=True)
    with pytest.raises(BudgetExceeded):ledger.tracker.record(CostRecord())
    assert ledger.tracker.totals.memory_bytes==20000


def test_safe_export_is_an_immutable_snapshot_of_caller_inputs():
    api=module('poc.continuation_evidence');data=numeric_fixture()
    result=api.export_numeric(data)
    data['candidates'][0]['text']='private mutation'
    assert 'text' not in result['raw']['candidates'][0]


def test_formula_overflow_cannot_produce_nonfinite_numeric_export():
    api=module('poc.continuation_evidence');data=numeric_fixture()
    for c in data['candidates']:
        for row in c['observations']:row['kl_sum']=1.7e308
    with pytest.raises((ValueError,OverflowError)):api.export_numeric(data)

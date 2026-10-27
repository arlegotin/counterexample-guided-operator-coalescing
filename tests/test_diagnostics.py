"""Selection must compare shared trials and resample source components, not pairs."""
import importlib
import importlib.util
from dataclasses import replace
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('poc.diagnostics'), 'single-merge diagnostics missing'
    return importlib.import_module('poc.diagnostics')


def table(method='natural_kl', values=((0., 0.), (1., 1.), (2., 2.))):
    a=api()
    ids=tuple(a.TrialIdentity(pair, str(i)*64, str(i+3)*64) for i,pair in enumerate(((0,1),(0,2),(1,2))))
    return a.ScoreTable(method,ids,('s1','s2'),tuple(map(tuple,values)), 'score-manifest',0,0.)


def test_full435_population_and_every_depth_separation_stratum_is_sampled():
    a=api(); policy=a.PairSamplingPolicy(seed=17,per_stratum=2)
    population=a.pair_population(30)
    assert len(population)==435 and len(set(population))==435
    sample=a.sample_pairs(30,policy)
    assert sample==a.sample_pairs(30,policy) and len(sample)==18
    assert {a.pair_stratum(p,30) for p in sample}=={a.pair_stratum(p,30) for p in population}


def test_shortlist_excluded_coverage_never_removes_original_sample():
    a=api(); sample=a.sample_pairs(30,a.PairSamplingPolicy(17,2))
    supplemented=a.ensure_excluded_coverage(sample,sample,30,17,minimum=2)
    assert set(sample)<=set(supplemented) and len(set(supplemented)-set(sample))==2


def test_known_regret_and_deterministic_tie():
    a=api(); scores=table(values=((1.,1.),(0.,0.),(0.,0.)))
    damage=a.DamageTable(scores.trials,('h1','h2'),((.2,.2),(.5,.5),(.1,.1)),'damage-manifest')
    result=a.selected_merge_regret(scores,damage)
    assert result.selected_pair==(0,2) and result.oracle_pair==(1,2)
    assert result.regret==pytest.approx(.4)


def test_empty_invalid_overlap_and_different_trials_are_not_zero_regret():
    a=api(); s=table(); d=a.DamageTable(s.trials,('h1','h2'),((0.,0.),)*3,'damage')
    for bad in [replace(s,trials=(),values=()),replace(s,values=((float('nan'),0.),)*3),replace(s,trials=s.trials[::-1])]:
        with pytest.raises(ValueError):a.selected_merge_regret(bad,d)
    with pytest.raises(ValueError,match='disjoint'):a.selected_merge_regret(s,replace(d,components=('s1','h2')))


def test_document_resampling_recomputes_selection_and_uses_shared_draws():
    a=api(); s=table(values=((0.,4.),(3.,0.),(10.,10.)))
    d=a.DamageTable(s.trials,('h1','h2'),((0.,0.),(2.,2.),(5.,5.)),'damage')
    result=a.bootstrap_regret([s,replace(s,method='active')],d,seed=8,resamples=300)
    assert result['unit']=='source_component' and result['selection_recomputed']
    assert set(result['methods']['natural_kl']['selected_indices'])=={0,1}
    assert result['methods']['natural_kl']['regret_draws']==result['methods']['active']['regret_draws']
    assert result['contrasts']['active-minus-natural_kl']['interval']==[0.,0.]


def test_all_score_methods_receive_identical_state_population_and_never_refit():
    a=api(); s=table(); rows=[]
    for index,identity in enumerate(s.trials):
        rows.append(a.TrialRecord(identity, {'weight_distance':{'s1':index,'s2':index},'natural_kl':{'s1':index+1,'s2':index+1}},'score-manifest', {'weight_distance':(0,0.),'natural_kl':(4,.2)}))
    for method in ('weight_distance','natural_kl'):
        result=a.score_trials(rows,method,records='score-manifest')
        assert result.trials==s.trials and result.components==s.components
    assert all(not hasattr(row,'model') and not hasattr(row,'fit') for row in rows)
    with pytest.raises(ValueError):a.score_trials(rows,'missing',records='score-manifest')


def test_pairwise_success_does_not_establish_joint_success():
    a=api(); controls=a.run_controls()
    assert controls['joint_failure']=={'first_only':0.,'second_only':0.,'joint':1.}
    assert controls['planted_selected_regret']==0 and controls['bad_selected_regret']>0
    assert controls['null_regret']==0


def test_fixed_score_formulas_are_finite_and_not_tuned_to_damage():
    a=api();v=a.fixed_scores(natural=2.,student=4.,random_donor=6.,active_donor=10.)
    assert v=={'natural_kl':2.,'random_donor':7.,'active_1':9.,'active_01':2.7}
    with pytest.raises(ValueError):a.fixed_scores(natural=2.,student=4.,random_donor=float('nan'),active_donor=10.)


def test_single_merge_cli_routes_profile_status_without_claiming_full_completion(monkeypatch,capsys):
    from poc import cli
    a=api()
    assert hasattr(a,'DiagnosticResult'), 'typed diagnostic result missing'
    monkeypatch.setattr(a,'run_single_merge',lambda config:a.DiagnosticResult('profile_complete','fixture/result.json'))
    code=cli.main(['diagnose','--kind','single_merge','--config','configs/pilot.yaml'])
    assert code==0 and 'profile_complete' in capsys.readouterr().out


def test_measured_driver_mean_accepts_generator_after_durable_trial_logging():
    from provenance.measure_single_merge import _mean
    assert _mean(x for x in [1.,2.,6.])==3.
    with pytest.raises(ValueError):_mean(iter(()))


def test_source_amendment_accepts_only_exact_old_and_new_source_maps():
    from provenance import measure_single_merge as driver
    assert hasattr(driver,'source_compatible'), 'explicit source amendment contract missing'
    old={'poc/merge.py':'a'*64,'driver':'b'*64};new={'poc/merge.py':'a'*64,'driver':'c'*64}
    amendment={'old_source_sha256':old,'new_source_sha256':new}
    assert driver.source_compatible(old,new,amendment)
    assert driver.source_compatible(new,new,None)
    assert not driver.source_compatible(old,new,None)
    assert not driver.source_compatible(old,new|{'poc/merge.py':'d'*64},amendment)
    assert not driver.source_compatible(old|{'extra':'e'*64},new,amendment)


def test_shortlist_miss_estimate_uses_observed_excluded_pairs_and_damage_components():
    a=api();s=table();d=a.DamageTable(s.trials,('h1','h2'),((.4,.4),(.1,.1),(.6,.6)),'damage')
    assert hasattr(a,'shortlist_misses'), 'excluded-pair miss estimate missing'
    result=a.shortlist_misses(d,[(0,1)],seed=17,resamples=100)
    assert result['observed_excluded_better_fraction']==.5
    assert result['restriction_regret']==pytest.approx(.3)
    assert result['restriction_regret_interval']==pytest.approx([.3,.3])
    assert result['scope']=='sampled excluded pairs against sampled shortlist members; not all435'
    with pytest.raises(ValueError):a.shortlist_misses(d,[],seed=17,resamples=100)


def test_primary_and_sensitivity_attempts_share_one_aggregate_budget(tmp_path):
    from provenance.measure_single_merge import DiagnosticRecorder
    from poc.records import start_attempt
    from poc.budget import BudgetTracker,BudgetExceeded
    from poc.contracts import CostRecord
    from test_records import spec
    base=spec();limits=replace(base.budgets,updates=2,wall_seconds=100.)
    config={'plan_sha256':'a'*64,'carry_in':{'audit_wall_seconds':3.,'spec_paths':[],'evidence':{}}}
    primary=start_attempt(replace(base,method_variant='task12_single_merge_score_only',budgets=limits,resolved_config=config),tmp_path/'runs')
    tracker=BudgetTracker(limits,primary);tracker.record(CostRecord(updates=1,wall_seconds=4.))
    sensitivity=start_attempt(replace(base,method_variant='task12_single_merge_sensitivity_refit',budgets=limits,resolved_config=config),tmp_path/'runs')
    second=BudgetTracker(limits,sensitivity);second.record(CostRecord(updates=1,wall_seconds=5.))
    recorder=DiagnosticRecorder(second,100)
    assert recorder.aggregate().updates==2 and recorder.aggregate().wall_seconds==12.
    with pytest.raises(BudgetExceeded):recorder.check_aggregate(CostRecord(updates=1))

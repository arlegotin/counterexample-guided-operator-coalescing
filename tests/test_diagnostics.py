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

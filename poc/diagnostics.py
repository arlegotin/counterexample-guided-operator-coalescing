"""Controlled score-only comparisons over immutable shared trial observations.

Pairs are the finite alternatives, not independent inferential units. Resamples
operate on source components and rerun selection within the sampled alternatives.
"""
from dataclasses import dataclass
import itertools
import math
import random
from collections.abc import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class PairSamplingPolicy:
    seed: int
    per_stratum: int


def pair_population(depth: int) -> list[tuple[int,int]]:
    if type(depth) is not int or depth<2: raise ValueError('depth must allow a pair')
    return list(itertools.combinations(range(depth),2))


def pair_stratum(pair,depth):
    a,b=pair
    if not 0<=a<b<depth: raise ValueError('invalid canonical pair')
    return min(2,(a+b)*3//(2*depth)),0 if b-a<=3 else 1 if b-a<=9 else 2


def sample_pairs(depth: int, policy: PairSamplingPolicy) -> list[tuple[int,int]]:
    if type(policy.seed) is not int or policy.seed<0 or type(policy.per_stratum) is not int or policy.per_stratum<1:
        raise ValueError('explicit positive per-stratum count and seed required')
    strata={}
    for p in pair_population(depth):strata.setdefault(pair_stratum(p,depth),[]).append(p)
    rng=random.Random(policy.seed)
    return sorted(p for key,rows in sorted(strata.items()) for p in rng.sample(rows,min(len(rows),policy.per_stratum)))


def ensure_excluded_coverage(sample,shortlist,depth,seed,*,minimum=2):
    population=set(pair_population(depth));sample=set(sample);shortlist=set(shortlist)
    if not sample or not sample<=population or not shortlist<=population or minimum<0:raise ValueError('invalid pair population')
    needed=max(0,minimum-len(sample-shortlist));available=sorted(population-shortlist-sample)
    if needed>len(available):raise ValueError('excluded coverage impossible')
    return sorted(sample|set(random.Random(seed).sample(available,needed)))


@dataclass(frozen=True)
class TrialIdentity:
    pair: tuple[int,int]
    initialized_digest: str
    fitted_digest: str


@dataclass(frozen=True)
class TrialRecord:
    identity: TrialIdentity
    scores: Mapping[str,Mapping[str,float]]
    score_manifest: str
    costs: Mapping[str,tuple[int,float]]


@dataclass(frozen=True)
class ScoreTable:
    method: str
    trials: tuple[TrialIdentity,...]
    components: tuple[str,...]
    values: tuple[tuple[float,...],...]
    manifest: str
    queries: int
    seconds: float


@dataclass(frozen=True)
class DamageTable:
    trials: tuple[TrialIdentity,...]
    components: tuple[str,...]
    values: tuple[tuple[float,...],...]
    manifest: str


@dataclass(frozen=True)
class RegretResult:
    selected_pair: tuple[int,int]
    oracle_pair: tuple[int,int]
    selected_damage: float
    oracle_damage: float
    regret: float


def _matrix(table):
    if not table.trials or not table.components or len(set(table.components))!=len(table.components):
        raise ValueError('nonempty unique trial and component populations required')
    pairs=[t.pair for t in table.trials]
    if len(set(pairs))!=len(pairs) or pairs!=sorted(pairs):raise ValueError('canonical unique trial order required')
    for t in table.trials:
        if len(t.pair)!=2 or not 0<=t.pair[0]<t.pair[1]:raise ValueError('invalid trial pair')
        for digest in (t.initialized_digest,t.fitted_digest):
            if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):raise ValueError('invalid state digest')
    array=np.asarray(table.values,dtype=float)
    if array.shape!=(len(table.trials),len(table.components)) or not np.isfinite(array).all():raise ValueError('complete finite component observations required')
    return array


def score_trials(trials: Sequence[TrialRecord], method: str, records) -> ScoreTable:
    """Pure reduction: no model, optimizer, callbacks or refit access here."""
    manifest=records.digest() if hasattr(records,'digest') else records
    if not trials or any(t.score_manifest!=manifest or method not in t.scores or method not in t.costs for t in trials):
        raise ValueError('missing shared trial scores or mismatched score manifest')
    components=tuple(sorted(trials[0].scores[method]))
    if any(set(t.scores[method])!=set(components) for t in trials):raise ValueError('score component population differs')
    result=ScoreTable(method,tuple(t.identity for t in trials),components,
        tuple(tuple(float(t.scores[method][c]) for c in components) for t in trials),manifest,
        sum(t.costs[method][0] for t in trials),sum(t.costs[method][1] for t in trials))
    _matrix(result)
    return result


def _aligned(scores,damage):
    s,d=_matrix(scores),_matrix(damage)
    if scores.trials!=damage.trials:raise ValueError('all methods/damage require identical initialized and fitted trials')
    if set(scores.components)&set(damage.components):raise ValueError('score/damage source components must be disjoint')
    return s,d


def selected_merge_regret(scores: ScoreTable, heldout_damage: DamageTable) -> RegretResult:
    s,d=_aligned(scores,heldout_damage);selection=int(s.mean(axis=1).argmin());means=d.mean(axis=1);oracle=int(means.argmin())
    return RegretResult(scores.trials[selection].pair,scores.trials[oracle].pair,
        float(means[selection]),float(means[oracle]),float(means[selection]-means[oracle]))


def bootstrap_regret(scores: Sequence[ScoreTable],damage: DamageTable,*,seed:int,resamples:int):
    if not scores or type(resamples) is not int or resamples<2:raise ValueError('scores and at least two resamples required')
    matrices=[]
    for s in scores:
        matrix,d=_aligned(s,damage)
        if s.trials!=scores[0].trials or s.components!=scores[0].components:raise ValueError('methods require the same trials and score components')
        matrices.append(matrix)
    if len({s.method for s in scores})!=len(scores):raise ValueError('duplicate method')
    rng=np.random.default_rng(seed)
    # A component's entire row vector across every overlapping pair moves together.
    score_draws=rng.integers(0,len(scores[0].components),(resamples,len(scores[0].components)))
    damage_draws=rng.integers(0,len(damage.components),(resamples,len(damage.components)))
    damage_means=d[:,damage_draws].mean(axis=2).T
    methods={};all_regrets={}
    for s,m in zip(scores,matrices,strict=True):
        selected=m[:,score_draws].mean(axis=2).argmin(axis=0)
        regrets=damage_means[np.arange(resamples),selected]-damage_means.min(axis=1)
        all_regrets[s.method]=regrets
        methods[s.method]={'interval':np.quantile(regrets,[.025,.975]).tolist(),'regret_draws':regrets.tolist(),'selected_indices':selected.tolist()}
    reference=scores[0].method
    contrasts={f'{s.method}-minus-{reference}':{'interval':np.quantile(all_regrets[s.method]-all_regrets[reference],[.025,.975]).tolist()} for s in scores[1:]}
    return dict(unit='source_component',selection_recomputed=True,seed=seed,resamples=resamples,methods=methods,contrasts=contrasts)


def run_controls():
    from poc.synthetic import make_causal_controls
    c=make_causal_controls();ids=tuple(TrialIdentity(p,str(i)*64,str(i+3)*64) for i,p in enumerate(((0,1),(0,2),(1,2))))
    s=ScoreTable('planted',ids,('s',),((0.,),(1.,),(2.,)),'fixture-score',0,0.)
    d=DamageTable(ids,('d',),((0.,),(1.,),(2.,)),'fixture-damage')
    bad=ScoreTable('bad',ids,('s',),((2.,),(1.,),(0.,)),'fixture-score',0,0.)
    null=DamageTable(ids,('d',),((1.,),)*3,'fixture-null')
    return dict(planted_selected_regret=selected_merge_regret(s,d).regret,bad_selected_regret=selected_merge_regret(bad,d).regret,
        null_regret=selected_merge_regret(s,null).regret,joint_failure={'first_only':c.joint_suffix(True,False),'second_only':c.joint_suffix(False,True),'joint':c.joint_suffix(True,True)})


def fixed_scores(*,natural,student,random_donor,active_donor):
    if not all(math.isfinite(v) for v in (natural,student,random_donor,active_donor)):raise ValueError('all fixed score sources must be finite')
    return dict(natural_kl=natural,random_donor=natural+.5*(student+random_donor),active_1=natural+.5*(student+active_donor),active_01=natural+.05*(student+active_donor))


@dataclass(frozen=True)
class DiagnosticResult:
    status: str
    result_path: str


def run_single_merge(config) -> DiagnosticResult:
    from provenance.measure_single_merge import run
    result=run(config)
    return DiagnosticResult(result['status'],result['attempt']+'/result.json')

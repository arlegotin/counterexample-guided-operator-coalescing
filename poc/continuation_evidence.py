"""Strict, content-free numeric exports. Historical omitted arrays stay omitted."""
import math
import re
from copy import deepcopy

from poc.contracts import CostRecord
from poc.diagnostics import fixed_scores


def keys(value, expected):
    if type(value) is not dict or set(value)!=set(expected):raise ValueError('unsafe or missing numeric evidence fields')


def sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('hash identity required')


def pair(value):
    if (type(value) is not list or len(value)!=2 or any(type(x) is not int for x in value)
            or not 0<=value[0]<value[1]):raise ValueError('canonical pair required')
    return tuple(value)


def metric(row, expected):
    keys(row,expected)
    for k,v in row.items():
        if k.endswith('sha256'):sha(v)
    if type(row['valid_targets']) is not int or row['valid_targets']<=0:raise ValueError('positive denominator required')
    if type(row['kl_sum']) not in (int,float) or not math.isfinite(row['kl_sum']):raise ValueError('finite KL sum required')
    if row['kl_sum']<0:raise ValueError('v1 numeric export requires nonnegative qualified KL sums')
    return row['kl_sum']/row['valid_targets']


def mean(values):
    if not values:raise ValueError('missing component observations')
    return math.fsum(values)/len(values)


def export_numeric(evidence):
    """Return raw sums/counts, fixed matrices and deterministic selections.

    Natural/damage use token-weighted sums within components. S/R average group
    means, A averages within-group maxima, matching the declared component/depth
    convention. Components receive equal weight for selection. Conditional KL
    never becomes full-model damage. The group is a recipient window/depth;
    R identifies the seeded sampled donor and A contains the complete donor pool.
    """
    evidence=deepcopy(evidence)
    keys(evidence,('schema_version','scope','decision_resolution','candidate_universe',
        'score_components','damage_components','candidates','query_graph'))
    if evidence['schema_version']!=1 or evidence['scope'] not in ('owned_fixture','exploratory'):
        raise ValueError('unsupported evidence scope/version')
    resolution=evidence['decision_resolution']
    if type(resolution) not in (int,float) or not math.isfinite(resolution) or resolution<=0:
        raise ValueError('frozen positive decision resolution required')
    universe=[pair(p) for p in evidence['candidate_universe']]
    if not universe or universe!=sorted(set(universe)):raise ValueError('complete canonical candidate universe required')
    score=evidence['score_components'];damage=evidence['damage_components']
    for components in (score,damage):
        if not components or len(set(components))!=len(components):raise ValueError('nonempty unique components required')
        for c in components:sha(c)
    if set(score)&set(damage):raise ValueError('score and damage identities overlap')
    nodes=set()
    for node in evidence['query_graph']:
        keys(node,('node_id','dependencies','cost'));sha(node['node_id'])
        if node['node_id'] in nodes or not set(node['dependencies'])<=nodes:raise ValueError('query graph must be a unique ordered DAG')
        # CostRecord permits free text reasons; public exports constrain these.
        cost=CostRecord(**node['cost'])
        if cost.estimation_method not in (None,'counted_dispatches_v1') or cost.memory_unavailable_reason not in (None,'not measured'):
            raise ValueError('unsafe cost annotation')
        nodes.add(node['node_id'])
    candidates=evidence['candidates']
    pairs=[pair(c.get('pair')) for c in candidates]
    if pairs!=sorted(set(pairs)) or not set(pairs)<=set(universe):raise ValueError('canonical sampled population required')
    # The candidate_universe is the evaluated candidate set for v1. Excluded
    # population members belong in a separately frozen sampling inventory.
    if pairs!=universe:raise ValueError('missing candidate from common population')
    methods=('natural_kl','student_only','random_donor','active_1','active_01')
    matrices={k:[] for k in methods};damage_matrix=[];failures=[]
    reference_coverage=None;reference_damage_coverage=None
    for candidate in candidates:
        keys(candidate,('pair','inclusion_probability','status','failure_code','initialized_digest',
            'fitted_digest','observations','damage','query_node_ids'))
        probability=candidate['inclusion_probability']
        if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<probability<=1:
            raise ValueError('sample inclusion probability required')
        if not candidate['query_node_ids'] or not set(candidate['query_node_ids'])<=nodes:
            raise ValueError('missing candidate query cost dependencies')
        status=candidate['status']
        if status not in ('valid','invalid','failed','oversized'):raise ValueError('unknown candidate outcome')
        for field in ('initialized_digest','fitted_digest'):
            if candidate[field] is not None:sha(candidate[field])
        if status!='valid':
            if candidate['failure_code'] not in ('correctness','budget','nonfinite','oversized','missing_input'):
                raise ValueError('bounded failure code required')
            if candidate['observations']!=[] or candidate['damage']!=[]:raise ValueError('failed rows cannot enter comparison')
            failures.append(dict(pair=candidate['pair'],status=status,failure_code=candidate['failure_code']))
            for m in methods:matrices[m].append(None)
            damage_matrix.append(None)
            continue
        if candidate['failure_code'] is not None:raise ValueError('valid candidate has failure')
        sha(candidate['initialized_digest']);sha(candidate['fitted_digest'])
        observed={c:{s:[] for s in ('natural','student','random','active')} for c in score}
        coverage=[];seen=set()
        for row in candidate['observations']:
            value=metric(row,('component_sha256','window_sha256','group_sha256','probe_sha256',
                'source','kl_sum','valid_targets'))
            if row['component_sha256'] not in observed or row['source'] not in observed[row['component_sha256']]:
                raise ValueError('unfrozen score component/source')
            key=(row['component_sha256'],row['source'],row['probe_sha256'])
            if key in seen:raise ValueError('duplicate probe observation')
            seen.add(key)
            observed[row['component_sha256']][row['source']].append(row)
            coverage.append((row['component_sha256'],row['window_sha256'],row['group_sha256'],row['source'],row['probe_sha256'],row['valid_targets']))
        if reference_coverage is None:reference_coverage=sorted(coverage)
        elif reference_coverage!=sorted(coverage):raise ValueError('candidate probe population differs')
        output={k:[] for k in methods}
        for component in score:
            sources=observed[component]
            if any(not rows for rows in sources.values()):raise ValueError('missing score source/component')
            natural=math.fsum(r['kl_sum'] for r in sources['natural'])/sum(r['valid_targets'] for r in sources['natural'])
            grouped={}
            for s in ('student','random','active'):
                groups={}
                for r in sources[s]:groups.setdefault(r['group_sha256'],[]).append(r['kl_sum']/r['valid_targets'])
                grouped[s]=groups
            if not set(grouped['student'])==set(grouped['random'])==set(grouped['active']):
                raise ValueError('conditional group coverage differs')
            for group in grouped['random']:
                random_rows=[r for r in sources['random'] if r['group_sha256']==group]
                active_rows=[r for r in sources['active'] if r['group_sha256']==group]
                for row in random_rows:
                    if not any(all(row[k]==a[k] for k in ('probe_sha256','window_sha256','kl_sum','valid_targets'))
                               for a in active_rows):
                        raise ValueError('seeded random donor absent from common active pool')
            if any(len(v)!=1 for s in ('student','random') for v in grouped[s].values()):
                raise ValueError('exactly one student/seeded-random row per group required')
            student=mean([mean(v) for v in grouped['student'].values()])
            random=mean([mean(v) for v in grouped['random'].values()])
            active=mean([max(v) for v in grouped['active'].values()])
            values=fixed_scores(natural=natural,student=student,random_donor=random,active_donor=active)
            values['student_only']=natural+.5*student
            if not all(math.isfinite(v) for v in values.values()):raise ValueError('nonfinite derived score')
            for m in methods:output[m].append(values[m])
        for m in methods:matrices[m].append(output[m])
        ds={c:[] for c in damage};damage_coverage=[];damage_seen=set()
        for row in candidate['damage']:
            metric(row,('component_sha256','window_sha256','kl_sum','valid_targets'))
            if row['component_sha256'] not in ds:raise ValueError('unfrozen damage component')
            key=(row['component_sha256'],row['window_sha256'])
            if key in damage_seen:raise ValueError('duplicate damage window')
            damage_seen.add(key)
            damage_coverage.append((*key,row['valid_targets']))
            ds[row['component_sha256']].append(row)
        if any(not rows for rows in ds.values()):raise ValueError('missing damage component')
        if reference_damage_coverage is None:reference_damage_coverage=sorted(damage_coverage)
        elif reference_damage_coverage!=sorted(damage_coverage):raise ValueError('damage population differs across candidates')
        damage_matrix.append([math.fsum(r['kl_sum'] for r in ds[c])/sum(r['valid_targets'] for r in ds[c]) for c in damage])
    selections={}
    if not failures:
        for method,matrix in matrices.items():
            means=[mean(row) for row in matrix];best=min(means);index=means.index(best)
            selections[method]=dict(selected_pair=list(pairs[index]),
                exact_ties=[list(p) for p,v in zip(pairs,means,strict=True) if v==best],
                near_ties=[list(p) for p,v in zip(pairs,means,strict=True) if v-best<=resolution])
        indices={k:pairs.index(tuple(v['selected_pair'])) for k,v in selections.items()}
        contrast=mean(damage_matrix[indices['active_1']])-mean(damage_matrix[indices['natural_kl']])
    else:contrast=None
    # Strict validation above means the raw input is safe to retain verbatim.
    return dict(raw=evidence,formulas={'natural_kl':'N','student_only':'N+0.5*S',
        'random_donor':'N+0.5*(S+R)','active_1':'N+0.5*(S+Amax)','active_01':'N+0.05*(S+Amax)'},
        aggregation='token weighted within natural/damage component; mean group conditional; mean components selection',
        score_matrices=matrices,damage_matrix=damage_matrix,selections=selections,failures=failures,
        comparison_status='BLOCKED' if failures else 'complete_numeric_only',
        primary_active_minus_natural_damage=contrast)

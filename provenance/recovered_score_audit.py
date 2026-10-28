"""Independently reduce recovered historical numeric records; never read corpus/weights."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase(root,label):
    original=label=='original'
    directory='artifacts/diagnostics/task12/trials-final' if original else 'artifacts/diagnostics/task12/precision-v1/trials'
    receipt_path=root/('provenance/single-merge-original-135m-2026-10-27.json' if original else 'provenance/single-merge-precision-135m-2026-10-27.json')
    receipt=json.loads(receipt_path.read_text());paths=sorted((root/directory).glob('*.json'))
    if len(paths)!=20:raise ValueError('complete frozen20 candidate population required')
    records=[];raw=[];score_components=damage_components=None;identities=[];arrays=defaultdict(list);damage=[];max_error=0
    for path in paths:
        if sha(path)!=receipt['artifact_sha256'][str(path.relative_to(root))]:raise ValueError('trial differs from pinned receipt')
        row=json.loads(path.read_text());identity=row['identity'];identities.append(identity)
        sc=sorted(row['scores']['natural_kl']);dc=sorted(row['natural_damage'])
        if set(sc)&set(dc):raise ValueError('score/damage overlap')
        if score_components is None:score_components,damage_components=sc,dc
        if sc!=score_components or dc!=damage_components:raise ValueError('candidate population differs')
        grouped=defaultdict(lambda:{'student':[],'donor':[]});safe=[]
        for p in row['probe_score_rows']:
            r=p['record'];n=p['valid_targets']
            if n!=sum(r['target_mask']) or n<=0 or r['candidate_digest']!=identity['fitted_digest']:
                raise ValueError('probe target/state provenance mismatch')
            if p['boundary_producer']!=(r['candidate_digest'] if r['prefix_source']=='candidate' else r['teacher_digest']):
                raise ValueError('boundary producer mismatch')
            if r['recipient_role']!='probe_score' or r['recipient_component'] not in sc:raise ValueError('unexpected score role')
            grouped[r['recipient_component'],r['depth']][p['source']].append(p)
            safe.append(dict(component_id=r['recipient_component'],depth=r['depth'],source=p['source'],
                probe_id=r['probe_id'],window_sha256=r['recipient_full_tokens_digest'],kl_sum=p['kl_sum'],valid_targets=n))
        student={};random_values={};active={}
        for component in sc:
            ss=[];rr=[];aa=[]
            for (c,depth),values in sorted(grouped.items()):
                if c!=component:continue
                if len(values['student'])!=1 or not values['donor']:raise ValueError('missing source observations')
                donors=sorted(values['donor'],key=lambda p:p['record']['probe_id'])
                chosen=random.Random(17+depth+int(component[:8],16)).randrange(len(donors))
                ss.append(values['student'][0]['kl_sum']/values['student'][0]['valid_targets'])
                rr.append(donors[chosen]['kl_sum']/donors[chosen]['valid_targets'])
                aa.append(max(p['kl_sum']/p['valid_targets'] for p in donors))
            student[component]=float(np.mean(ss));random_values[component]=float(np.mean(rr));active[component]=float(np.mean(aa))
        for method in ('natural_kl','random_donor','active_1','active_01'):
            values=[]
            for c in sc:
                n=row['scores']['natural_kl'][c];s=student[c];r=random_values[c];a=active[c]
                expected={'natural_kl':n,'random_donor':n+.5*(s+r),'active_1':n+.5*(s+a),'active_01':n+.05*(s+a)}[method]
                observed=row['scores'][method][c];max_error=max(max_error,abs(expected-observed))
                if abs(expected-observed)>1e-12:raise ValueError('fixed formula does not reconstruct')
                values.append(observed)
            arrays[method].append(values)
        arrays['student_only'].append([row['scores']['natural_kl'][c]+.5*student[c] for c in sc])
        damage.append([row['natural_damage'][c] for c in dc])
        raw.append(dict(identity=identity,source_record=str(path.relative_to(root)),source_sha256=sha(path),
            natural_score_rows=[{k:p[k] for k in ('component','document','kl_sum','valid_targets')} for p in row['natural_score_rows']],
            conditional_probes=safe,natural_damage_rows=[{k:p[k] for k in ('component','document','kl_sum','valid_targets')} for p in row['natural_damage_rows']]))
    pairs=[i['pair'] for i in identities]
    if pairs!=sorted(pairs) or len({tuple(p) for p in pairs})!=20:raise ValueError('canonical candidate universe required')
    means=np.asarray(damage).mean(axis=1);oracle=int(means.argmin());points={};boot={}
    rng=np.random.default_rng(17);sd=rng.integers(0,len(score_components),(2000,len(score_components)))
    dd=rng.integers(0,len(damage_components),(2000,len(damage_components)))
    dm=np.asarray(damage)[:,dd].mean(axis=2).T
    choices={}
    for method,values in arrays.items():
        matrix=np.asarray(values);choice=int(matrix.mean(axis=1).argmin())
        points[method]=dict(selected_pair=pairs[choice],sampled_oracle_pair=pairs[oracle],
            damage=float(means[choice]),sampled_regret=float(means[choice]-means[oracle]))
        selected=matrix[:,sd].mean(axis=2).argmin(axis=0);choices[method]=selected
        bootstrap_damage=dm[np.arange(2000),selected]
        contrast=bootstrap_damage-dm[np.arange(2000),choices['natural_kl']] if method!='natural_kl' else np.zeros(2000)
        boot[method]=dict(active_or_method_minus_natural_interval=np.quantile(contrast,[.025,.975]).tolist(),
            same_choice_fraction=float(np.mean(selected==choices['natural_kl'])),
            favorable_draws=int(np.sum(contrast<0)),adverse_draws=int(np.sum(contrast>0)),
            zero_contrast_fraction=float(np.mean(contrast==0)),contrast_draws=contrast.tolist())
    return dict(label=label,source_receipt_sha256=sha(receipt_path),pairs=pairs,score_components=score_components,
        damage_components=damage_components,identities=identities,score_matrices=dict(arrays),damage_matrix=damage,
        raw=raw,formula_max_abs_error=max_error,point_estimates=points,bootstrap=boot,
        original_measured_aggregate_cost=receipt['aggregate_cost'],
        uncertainty_scope='descriptive conditional component bootstrap on one fixed fitted population; not episode replication or equivalence proof')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    if args.out.exists():raise FileExistsError(args.out)
    result=dict(schema_version=1,scope='recovered private numeric arrays with public receipt hash verification; no new model/data predictions',
        original=phase(args.asset_root,'original'),fresh_exploratory=phase(args.asset_root,'fresh'))
    if result['original']['identities']!=result['fresh_exploratory']['identities']:raise ValueError('original/fresh fitted states differ')
    if abs(result['original']['point_estimates']['natural_kl']['sampled_regret']-0.0001156671144)>1e-12:raise ValueError('natural original reduction mismatch')
    if abs(result['fresh_exploratory']['point_estimates']['natural_kl']['sampled_regret']-0.009830384477)>1e-11:raise ValueError('natural fresh reduction mismatch')
    result['wall_seconds']=time.monotonic()-start
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({label:r['point_estimates'] for label,r in result.items() if isinstance(r,dict) and 'point_estimates' in r},indent=2))


if __name__=='__main__':main()

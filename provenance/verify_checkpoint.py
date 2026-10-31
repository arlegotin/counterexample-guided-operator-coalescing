"""Verify the v0.2.0 public numeric checkpoint without author-specific assets.

The default path uses only the standard library. --bootstrap additionally needs
NumPy and regenerates the retained paired component resamples. Neither mode
loads a model, reads a corpus, follows external receipt paths or reruns inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'provenance/continuation-checkpoint-2026-10-31'
INDEX=ROOT/'provenance/releases/v0.2.0/checkpoint-sha256.json'
STEPS=(0,1,4,16,32,64,128,256,512,1024)
SCOPES=('local_sum','full_student')
LABELS=('dense_gptq2_embedding4','dense_gptq4','dense_rtn2','dense_rtn2_embedding4','dense_rtn4','dense_rtn8',
        'shared_float','shared_gptq2_embedding4','shared_gptq4','shared_rtn2_embedding4','shared_rtn4','teacher_noop')
COPIES={'recovery-acceptance.json','recovery-analysis.json','native-acceptance.json','native-admission.json','native-frontier.json'}
COPIES|={f'recovery-{scope}-observation-{step}.json' for scope in SCOPES for step in STEPS}
COPIES|={f'native-{label}.json' for label in LABELS}
FILES=COPIES|{'manifest.json','summary.json','assemble-receipts-v1.py.txt','completion.json','local-commit-audit-v1.json'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def regular(path):
    p=Path(path)
    require(not p.is_symlink() and p.is_file() and p.stat().st_nlink==1,'regular nonaliased evidence file required')
    return p


def read(path):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'duplicate JSON key');out[key]=value
        return out
    def invalid(value):raise ValueError('nonfinite JSON value')
    return json.loads(regular(path).read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def number(value,*,nonnegative=False):
    require(type(value) in (int,float) and math.isfinite(value) and (not nonnegative or value>=0),'finite numeric metric required')
    return value


def close(a,b,*,tolerance=1e-14):
    require(abs(number(a)-number(b))<=tolerance,'numeric reduction mismatch')


def identity(row):
    keys=('document_id','component_id','window_sha256')
    require(all(type(row.get(k)) is str and re.fullmatch('[0-9a-f]{64}',row[k]) for k in keys),'invalid observation identity')
    return tuple(row[k] for k in keys)


def count(value):
    require(type(value) is int and value>0,'valid target denominator required');return value


def sums(rows,*,recovery=False):
    require(type(rows) is list and rows,'nonempty complete component rows required')
    seen=set();identities=[];kl=[];nll=[];sizes=[]
    for row in rows:
        ident=identity(row);require(ident[1] not in seen,'duplicate component');seen.add(ident[1]);identities.append(ident)
        metric=row['full'] if recovery else row;n=count(metric['valid_targets'])
        if recovery:
            positions=row['target_positions'];require(positions==list(range(511)) and n==len(positions),'complete next-token mask/denominator required')
            require(metric['status']=='ok' and metric['reason'] is None,'failed metric cannot be promoted')
        k=number(metric['kl'] if recovery else metric['kl_sum']);l=number(metric['nll'] if recovery else metric['nll_sum'],nonnegative=True)
        require(k>=-2e-6*n,'KL below qualified numerical floor')
        require(type(metric['disagreements']) is int and 0<=metric['disagreements']<=n,'invalid disagreement denominator')
        sizes.append(n);kl.append(k);nll.append(l)
    return {'identities':identities,'sizes':sizes,'kl_sums':kl,'nll_sums':nll,'targets':sum(sizes),
        'kl':math.fsum(kl)/sum(sizes),'nll':math.fsum(nll)/sum(sizes)}


def percentile(values,p):
    ordered=sorted(number(v) for v in values);x=(len(ordered)-1)*p;i=int(x)
    return ordered[i] if i==len(ordered)-1 else ordered[i]+(ordered[i+1]-ordered[i])*(x-i)


def verify_numeric(base,*,bootstrap=False):
    load=lambda name:read(base/name)
    summary=load('summary.json');er=load('recovery-analysis.json');ea=load('recovery-acceptance.json');done=load('completion.json')
    require(ea['status']=='accepted_exploratory_counterpart_comparison; original local pipeline resource failed','original local failure scope changed')
    require(er['original_local_failure']['original_pipeline_status']=='resource_failed','original failed pipeline cannot become successful')
    require(er['checks']['local_sum']['ledger']['budget_violations']==['memory_bytes'],'original memory overrun must remain visible')
    require(done['native_execution_qualified'] is False and done['new_scientific_model_runs']==0
        and done['native_artifact_recovery_horizon']==256 and done['long1024_native_exports_measured'] is False,'native result horizon or qualification claim changed')
    claims={'recovery_established_in_tested_regime':True,'sharing_beats_dense_quantization':False,
        'counterexample_advantage_established':False,'capability_acceptability_established':False,
        'protected_confirmation':False,'production_speedup_established':False}
    require(summary['claims']==claims,'unsupported capability/compression/counterexample claim')
    require(summary['recovery']['seed']==17 and summary['recovery']['updates']==1024
        and summary['recovery']['raw_presentations_per_arm']==8388608 and summary['recovery']['unique_fit_positions']==261632,'recovery exposure or seed scope changed')
    endpoint={};tuning_ids=None
    for scope in SCOPES:
        for step in STEPS:
            name=f'recovery-{scope}-observation-{step}.json';obs=load(name);curve=er['curves'][scope][str(step)]
            require(type(obs['step']) is int and obs['step']==step and obs['candidate_digest']==curve['candidate_digest'],'observation candidate/step mismatch')
            result=sums(obs['populations']['tuning'],recovery=True)
            require(len(result['identities'])==128 and result['targets']==65408,'complete128component tuning population required')
            if tuning_ids is None:tuning_ids=result['identities']
            require(result['identities']==tuning_ids,'tuning context/component identity changed')
            for key in ('kl','nll'):close(result[key],curve['tuning'][key])
            require(curve['observation']['sha256']==sha(base/name),'observation differs from admitted analysis')
            if step==1024:endpoint[scope]=result
        for key in ('kl','nll'):close(endpoint[scope][key],er['arms'][scope][key]);close(endpoint[scope][key],summary['recovery'][scope+'_kl'] if key=='kl' else er['arms'][scope][key])
    order=sorted(range(128),key=lambda i:tuning_ids[i][1]);a=endpoint['local_sum'];b=endpoint['full_student']
    diff=[b['kl_sums'][i]-a['kl_sums'][i] for i in order];sizes=[a['sizes'][i] for i in order]
    effect=math.fsum(diff)/sum(sizes);close(effect,er['full_minus_local']['kl_difference']);close(effect,summary['recovery']['full_minus_local_kl'])
    require(er['bootstrap']['seed']==17 and er['bootstrap']['draws']==10000,'fixed component resampling plan changed')
    stored=er['full_minus_local']['contrast_draws'];require(len(stored)==10000,'complete stored bootstrap required')
    for p,recorded in zip((.025,.975),er['full_minus_local']['component_percentile95'],strict=True):close(percentile(stored,p),recorded)
    native={};native_ids=None;teacher_nll=None;frontier=load('native-frontier.json');admission=load('native-admission.json')
    require(set(frontier['points'])==set(LABELS) and set(admission['artifacts'])==set(LABELS),'complete twelve-artifact frontier required')
    for label in LABELS:
        value=load(f'native-{label}.json');result=sums(value['rows'])
        require(len(result['identities'])==128 and result['targets']==65408,'complete128component native population required')
        require(value['mode']=='run' and value['counts']['valid_target_presentations']==result['targets'],'native completed target work changed')
        if native_ids is None:native_ids=result['identities']
        require(result['identities']==native_ids,'native artifact contexts/components differ')
        this_teacher=[number(row['teacher_nll_sum'],nonnegative=True) for row in value['rows']]
        if teacher_nll is None:teacher_nll=this_teacher
        require(this_teacher==teacher_nll,'native teacher targets differ')
        require(type(value['complete_bytes']) is int and value['complete_bytes']>0,'actual complete byte census required')
        for point in (frontier['points'][label],admission['artifacts'][label],summary['native_frontier']['points'][label]):
            require(point['complete_bytes']==value['complete_bytes'],'complete byte census mismatch')
            for key in ('kl','nll'):close(result[key],point[key],tolerance=1e-13)
        native[label]=result
    close(native['teacher_noop']['kl'],0,tolerance=2e-5)
    ndiff=[x-y for x,y in zip(native['shared_gptq4']['kl_sums'],native['dense_gptq4']['kl_sums'],strict=True)]
    nsizes=native['dense_gptq4']['sizes'];primary=admission['primary'];close(math.fsum(ndiff)/sum(nsizes),primary['kl_difference'])
    require(primary['better_components']==sum(x<0 for x in ndiff) and primary['worse_components']==sum(x>0 for x in ndiff),'native favorable/adverse counts changed')
    result={'schema_version':1,'status':'verified saved numeric evidence','recovery_observations':20,'native_artifacts':12,
        'valid_targets':65408,'training_rerun':False,'native_rerun':False,'original_local_pipeline':'resource_failed; saved states admitted',
        'bootstrap':{'regenerated':False,'recovery_draws':0,'native_interval_reproduced':False}}
    if bootstrap:
        try:import numpy as np
        except ImportError as exc:raise ValueError('--bootstrap requires NumPy; recorded stack uses2.5.4') from exc
        draw=np.random.default_rng(17).integers(0,128,(10000,128))
        actual=np.asarray(diff)[draw].sum(1)/np.asarray(sizes)[draw].sum(1)
        require(np.allclose(actual,stored,rtol=0,atol=1e-14),'independent recovery bootstrap differs')
        require(hashlib.sha256(draw.tobytes()).hexdigest()==er['bootstrap']['common_index_draws_sha256'],'bootstrap component draw identity changed')
        native_draw=np.random.default_rng(17).integers(0,128,(10000,128))
        nvalues=np.asarray(ndiff)[native_draw].sum(1)/np.asarray(nsizes)[native_draw].sum(1)
        require(np.allclose(np.quantile(nvalues,[.025,.975]),primary['paired_component_percentile_95'],rtol=0,atol=1e-14),'independent native interval differs')
        result['bootstrap']={'regenerated':True,'recovery_draws':10000,'native_interval_reproduced':True}
    return result


def verify_checkpoint(directory=BASE,*,index=INDEX,bootstrap=False):
    base=Path(directory).resolve();idx=read(Path(index))
    require(set(idx)=={'schema_version','release','evidence_directory','files'} and type(idx['schema_version']) is int
        and idx['schema_version']==1 and idx['release']=='0.2.0'
        and idx['evidence_directory']=='provenance/continuation-checkpoint-2026-10-31','closed release evidence index required')
    require(set(idx['files'])==FILES,'complete fixed release file coverage required')
    require({p.name for p in base.iterdir()}==FILES,'unlisted or missing evidence file')
    for name,record in idx['files'].items():
        require(Path(name).name==name and type(record) is dict and set(record)=={'sha256','bytes'},'closed safe evidence record required')
        require(type(record['bytes']) is int and record['bytes']>0 and type(record['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}',record['sha256']),'valid evidence size/hash required')
        p=regular(base/name);require(p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],'release file hash/size mismatch')
    result=verify_numeric(base,bootstrap=bootstrap)
    legacy=read(base/'manifest.json');require(set(legacy['copied_receipts'])==COPIES,'complete original receipt inventory required')
    for name,record in legacy['copied_receipts'].items():require(sha(base/name)==record['sha256'],'copied original receipt hash differs')
    for key,name in (('summary','summary.json'),('assembly_source','assemble-receipts-v1.py.txt')):
        require(sha(base/name)==legacy[key]['sha256'],'original summary/assembly hash differs')
    result['hashed_files']=len(FILES);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=BASE)
    parser.add_argument('--index',type=Path,default=INDEX);parser.add_argument('--bootstrap',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(verify_checkpoint(args.directory,index=args.index,bootstrap=args.bootstrap),sort_keys=True))


if __name__=='__main__':main()

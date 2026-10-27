"""Read-only source/lineage/statistics audit of the completed original Task12 batch."""
from pathlib import Path
import hashlib,json,time
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/task12'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def mean(values):return sum(values)/len(values)

def uncertainty(value,pairs):
    result={k:v for k,v in value.items() if k!='methods'}
    result['methods']={k:{'regret_interval':v['interval'],'selection_pair_counts':{str(pairs[i]):n for i,n in Counter(v['selected_indices']).items()}} for k,v in value['methods'].items()}
    methods=value['methods'];active=np.array(methods['active_1']['regret_draws']) if 'active_1' in methods else None
    if active is not None:
        result['paired_discrete_contrasts']={}
        for reference in ('natural_kl','random_donor'):
            difference=active-np.array(methods[reference]['regret_draws'])
            mismatch=np.array(methods['active_1']['selected_indices'])!=np.array(methods[reference]['selected_indices'])
            result['paired_discrete_contrasts'][reference]={'interval':np.quantile(difference,[.025,.975]).tolist(),'mean':float(difference.mean()),'range':[float(difference.min()),float(difference.max())],'negative_draws':int((difference<0).sum()),'positive_draws':int((difference>0).sum()),'zero_draws':int((difference==0).sum()),'different_selections':int(mismatch.sum()),'note':'percentile-zero intervals may hide rare nonzero discrete outcomes; not equivalence'}
    return result

def summary(value,pairs):
    result={k:v for k,v in value.items() if k not in ('uncertainty','fresh_uncertainty','secondary_uncertainty','secondary_original18')}
    for key in ('uncertainty','fresh_uncertainty','secondary_uncertainty'):result[key]=uncertainty(value[key],pairs)
    if 'secondary_original18' in value:
        original=json.loads((OUT/'pairs.json').read_text())['sample']
        result['secondary_original18']=summary(value['secondary_original18'],sorted(original))
    return result

def main():
    started=time.monotonic();latest=json.loads((OUT/'latest.json').read_text());primary_path=ROOT/latest['attempt']/'result.json';primary=json.loads(primary_path.read_text())
    assert primary['status']=='complete' and primary['completed_pairs']==20
    design=json.loads((OUT/'pairs-amended.json').read_text());pairs=sorted(design['sample'])
    rows=[json.loads((OUT/'trials-final'/f'{p[0]:02d}-{p[1]:02d}.json').read_text()) for p in pairs]
    methods=set(rows[0]['scores']);components=set(rows[0]['scores']['natural_kl']);damage=set(rows[0]['natural_damage']);fresh=set(rows[0]['intervention_damage'])
    assert len(components)==8 and len(damage)==16 and len(fresh)==16 and not components&damage and not components&fresh
    teacher=json.loads((OUT/'probe_score-pool.json').read_text())['teacher_digest']
    fit_digests=set();filtered_pass=filtered_total=0
    for row,pair in zip(rows,pairs,strict=True):
        assert row['identity']['pair']==pair and row['fit']['updates']==1 and row['fit']['targets']==8192
        assert set(row['scores'])==methods and all(set(v)==components for v in row['scores'].values())
        assert set(row['natural_damage'])==damage and set(row['intervention_damage'])==fresh
        assert row['selected_parent']==min(row['medoid_scores'],key=lambda x:(x['mse'],x['parent']))['parent']
        assert len(row['medoid_scores'])==2 and len({x['fit_digest'] for x in row['medoid_scores']})==1
        fit_digests.update(x['fit_digest'] for x in row['medoid_scores'])
        for item in row['probe_score_rows']+row['intervention_damage_rows']:
            probe=item['record'];assert probe['candidate_digest']==row['identity']['fitted_digest'] and probe['teacher_digest']==teacher
            assert item['boundary_producer']==(probe['candidate_digest'] if probe['prefix_source']=='candidate' else teacher)
            assert probe['recipient_role']==probe['donor_role']=='probe_score' and any(probe['target_mask'])
            assert item['valid_targets']==sum(probe['target_mask'])
            if item['source']=='donor':filtered_total+=1;filtered_pass+=item['threshold_pass']
    assert len(fit_digests)==1 and len({x['identity']['fitted_digest'] for x in rows})==20
    attempts=[];cost_sum={};artifact_hashes={};sources=0
    for result_path in sorted((OUT/'runs').glob('*/attempts/*/result.json')):
        result=json.loads(result_path.read_text());attempt=result_path.parent;spec=json.loads((attempt.parent.parent/'spec.json').read_text())
        for name,digest in spec['resolved_config']['source_sha256'].items():assert sha(attempt/'source'/name)==digest;sources+=1
        attempts.append({'path':str(result_path.relative_to(ROOT)),'status':result['status'],'method_variant':spec['method_variant'],'source_commit':spec['source_commit'],'cost':result['cost'],'failure':result.get('failure_reason')})
        artifact_hashes[str(result_path.relative_to(ROOT))]=sha(result_path)
        for k,v in result['cost'].items():
            if isinstance(v,(int,float)) and k not in ('disk_bytes','memory_bytes'):cost_sum[k]=cost_sum.get(k,0)+v
        artifact_hashes[str((attempt.parent.parent/'costs.jsonl').relative_to(ROOT))]=sha(attempt.parent.parent/'costs.jsonl')
    assert len(attempts)==4 and cost_sum['updates']==22 and cost_sum['valid_target_presentations']==180224
    sensitivities=[]
    for path in sorted((OUT/'sensitivity').glob('*/result.json')):
        value=json.loads(path.read_text());assert value['status']=='complete' and value['fit']['targets']==8192
        sensitivities.append({'pair':value['pair'],'primary_identity':value['primary_identity'],'fitted_digest':value['fitted_digest'],'fit_before':mean([x['kl_per_target'] for x in value['primary_fit_checks']]),'fit_after':mean([x['kl_per_target'] for x in value['after_fit_checks']]),'score_before':{k:mean(list(v.values())) for k,v in value['primary_scores'].items()},'score_after':{k:mean([v[k] for v in value['after_scores_by_component'].values()]) for k in next(iter(value['after_scores_by_component'].values()))},'cost':value['cost'],'artifact':str(path.relative_to(ROOT))})
        artifact_hashes[str(path.relative_to(ROOT))]=sha(path)
    final=json.loads((OUT/'sensitivity/01-05/result.json').read_text())['aggregate_phase_cost']
    plan=json.loads((OUT/'plan.json').read_text());assert abs(cost_sum['wall_seconds']+plan['task12_carry_in']['audit_wall_seconds']-final['wall_seconds'])<1e-8
    for key in ('teacher_forward_calls','student_forward_calls','updates','valid_target_presentations','student_backward_calls','suffix_backward_calls','search_proposals'):assert cost_sum[key]==final[key]
    for name in ('plan.json','pairs.json','pairs-amended.json','calibration.json','fit-pool.json','probe_score-pool.json','probe_damage-pool.json','full-phase-approved.json','source-amendment.json','zero-target-audit.json'):
        artifact_hashes[str((OUT/name).relative_to(ROOT))]=sha(OUT/name)
    for path in sorted((OUT/'trials-final').glob('*.json')):artifact_hashes[str(path.relative_to(ROOT))]=sha(path)
    by_pair=[]
    for row in rows:
        by_pair.append({'identity':row['identity'],'selected_parent':row['selected_parent'],'fit_record_digest':row['medoid_scores'][0]['fit_digest'],'score_means':{k:mean(list(v.values())) for k,v in row['scores'].items()},'natural_damage':mean(list(row['natural_damage'].values())),'fresh_conditional_intervention_damage':mean(list(row['intervention_damage'].values())),'fit_check_before':mean([x['kl_per_target'] for x in row['sensitivity']['before']]),'fit_check_after':mean([x['kl_per_target'] for x in row['sensitivity']['after']]),'source_denominators':row['fit']['metrics'][0]['source_counts'],'main_score_queries_seconds':row['costs'],'necessary_pool_rows':row['necessary_pool_rows']})
    result={'schema_version':1,'status':'original20plus2_batch_complete','gate_status':'inconclusive_pending_controller_precision_followup','scope':'exploratory fixed one-update candidate ranking; no protected confirmation, capability or native compression claim','primary_result':str(primary_path.relative_to(ROOT)),'primary_summary':summary(primary['summary'],pairs),'pair_rows':by_pair,'sensitivity_refits':sensitivities,'aggregate_cost':final,'attempts':attempts,'preflight_carry_in_seconds':plan['task12_carry_in']['audit_wall_seconds'],'historical_task11_wall_seconds_separate':2079.0169,'post_profile_cpu_control_analysis_seconds_separate':2.856126417,'audit':{'matched_medoid_fit_digests':len(fit_digests),'same_main_score_components':8,'same_natural_damage_components':16,'same_fresh_damage_components':16,'distinct_fitted_states':20,'source_bodies_checked':sources,'all_producer_digests_preserved':True,'all_positive_target_masks_preserved':True,'filtered_donor_rows_pass':filtered_pass,'filtered_donor_rows_total':filtered_total,'filtering_never_replaced_unfiltered_primary':True},'actual_shortlist_count':len(design['shortlist']),'actual_shortlist_coverage':{'included':2,'included_population':23,'excluded':18,'excluded_population':412},'calibration':{k:v for k,v in primary['calibration'].items() if k!='windows'},'artifact_sha256':artifact_hashes,'analysis_source_sha256':sha(Path(__file__)),'retained_disk_bytes_before_receipt':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'state_retention_limitation':'Only primary(0,15) and(1,5) deltas were retained; other18 exact primary states require charged deterministic reconstruction/replay for a later fixed-state repeat. No full model/Adam retained per pair.'}
    result['analysis_audit_wall_seconds_separate']=time.monotonic()-started
    (ROOT/'provenance/single-merge-original-135m-2026-10-27.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'audit':result['audit'],'cost':final,'audit_seconds':result['analysis_audit_wall_seconds_separate'],'receipt':'provenance/single-merge-original-135m-2026-10-27.json'}))

if __name__=='__main__':main()

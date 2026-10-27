"""Read-only audit and compact receipt for the fixed fresh-component repeat."""
from pathlib import Path
import json,time
from collections import Counter
from provenance.summarize_single_merge import sha,mean,uncertainty

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/diagnostics/task12'
OUT=OLD/'precision-v1'


def main():
    started=time.monotonic();plan=json.loads((OUT/'plan.json').read_text());data=json.loads((OUT/'data.json').read_text())
    states=json.loads((OUT/'verified-states.json').read_text());analysis=json.loads((OUT/'analysis.json').read_text())
    assert states['all20_verified_before_new_predictions'] and len(states['states'])==20
    old_components=set()
    for path in (OLD/'trials-final').glob('*.json'):
        row=json.loads(path.read_text())
        for key in ('scores','secondary_scores'):
            for values in row[key].values():old_components.update(values)
        old_components.update(row['natural_damage']);old_components.update(row['intervention_damage'])
    attempts=[];sources=0;hashes={};latest=json.loads((OUT/'latest.json').read_text());final=json.loads((ROOT/latest['attempt']/'result.json').read_text())
    assert final['status']=='complete' and final['stage']=='analyze'
    for path in sorted((OUT/'runs').glob('*/attempts/*/result.json')):
        result=json.loads(path.read_text());spec=json.loads((path.parents[2]/'spec.json').read_text())
        for name,digest in spec['resolved_config']['source_sha256'].items():assert sha(path.parent/'source'/name)==digest;sources+=1
        attempts.append({'path':str(path.relative_to(ROOT)),'stage':result['stage'],'status':result['status'],'cost':result['cost'],'source_commit':result['source_commit'],'failure':result.get('failure_reason')})
        hashes[str(path.relative_to(ROOT))]=sha(path);hashes[str((path.parents[2]/'costs.jsonl').relative_to(ROOT))]=sha(path.parents[2]/'costs.jsonl')
    by_pair=[];score_components=None;damage_components=None;donors=passed=students=0
    for item in states['states']:
        path=ROOT/item['path'];assert sha(path/'metadata.json')==item['metadata_sha256'];assert sha(path/'weights.safetensors')==item['weights_sha256']
        row_path=OUT/'trials'/f"{item['pair'][0]:02d}-{item['pair'][1]:02d}.json";row=json.loads(row_path.read_text());assert row['identity']==item['identity'] and row['no_fitting']
        score=set(row['scores']['natural_kl']);damage=set(row['natural_damage'])
        assert len(score)==16 and len(damage)==32 and not score&damage and not (score|damage)&old_components
        if score_components is None:score_components=score;damage_components=damage
        assert score==score_components and damage==damage_components and all(set(v)==score for v in row['scores'].values())
        for probe in row['probe_score_rows']:
            r=probe['record'];assert r['candidate_digest']==row['identity']['fitted_digest'];assert probe['valid_targets']==sum(r['target_mask'])>0
            assert probe['boundary_producer']==(r['candidate_digest'] if r['prefix_source']=='candidate' else r['teacher_digest'])
            if probe['source']=='donor':donors+=1;passed+=probe['threshold_pass']
            else:students+=1
        by_pair.append({'identity':item['identity'],'score_means':{k:mean(list(v.values())) for k,v in row['scores'].items()},'natural_damage':mean(list(row['natural_damage'].values())),'logical_positions':row['logical_positions'],'wall_seconds':row['wall_seconds']})
        for artifact in (path/'metadata.json',path/'weights.safetensors',row_path):hashes[str(artifact.relative_to(ROOT))]=sha(artifact)
    summaries={}
    for key in ('fresh_repeat','fresh_original18_secondary','pooled_exploratory_secondary'):
        value=analysis[key];summaries[key]={k:v for k,v in value.items() if k!='uncertainty'};summaries[key]['uncertainty']=uncertainty(value['uncertainty'],value['pairs'])
    for name in ('plan.json','data.json','pool.json','windows.json','verified-states.json','logical-positions.jsonl','analysis.json'):
        path=OUT/name;hashes[str(path.relative_to(ROOT))]=sha(path)
    positions=sum(json.loads(line)['positions'] for line in (OUT/'logical-positions.jsonl').read_text().splitlines());assert positions<=2500000
    assert final['aggregate_phase_cost']['updates']==40 and final['aggregate_phase_cost']['valid_target_presentations']==327680
    result={'schema_version':1,'status':'fixed_precision_repeat_complete','scope':'exploratory; original study remains original primary; no protected confirmation or extension until significance','original_receipt_sha256':plan['parent_result_sha256'],'summaries':summaries,'pair_rows':by_pair,'aggregate_cost':final['aggregate_phase_cost'],'followup_cost':final['followup_phase_cost'],'attempts':attempts,'audit':{'all20_primary_fitted_digests_exact':True,'reconstruction_updates':18,'original_primary_candidate_updates':1,'fresh_score_components':16,'fresh_damage_components':32,'no_old_component_overlap':True,'source_bodies_checked':sources,'positive_target_and_producer_guards':True,'donor_rows':donors,'threshold_pass_rows':passed,'student_rows':students,'logical_evaluation_positions_charged':positions},'shared_query_note':'all scorers use same generation/evaluation pool; active requires all donor rows, random requires one per component/depth (32 per trial), natural needs only natural rows; no intrinsically max-selection random pipeline claim','new_intervention_damage':False,'artifact_sha256':hashes,'analysis_source_sha256':sha(Path(__file__)),'retained_disk_bytes':sum(p.stat().st_size for p in OLD.rglob('*') if p.is_file())}
    result['audit_wall_seconds_separate']=time.monotonic()-started
    result['overall_wall_seconds_including_this_audit']=result['aggregate_cost']['wall_seconds']+result['audit_wall_seconds_separate']
    result['followup_wall_seconds_including_this_audit']=result['followup_cost']['wall_seconds']+result['audit_wall_seconds_separate']
    destination=ROOT/'provenance/single-merge-precision-135m-2026-10-27.json';destination.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'receipt':str(destination.relative_to(ROOT)),'aggregate':result['aggregate_cost'],'followup':result['followup_cost'],'audit':result['audit'],'audit_seconds':result['audit_wall_seconds_separate']}))

if __name__=='__main__':main()

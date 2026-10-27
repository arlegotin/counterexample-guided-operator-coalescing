"""Persist the Task9 evidence index/eligibility without generating predictions."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.bundle import _sha,_canonical,measure_bundle,verify_bundle
from poc.contracts import thaw
from poc.methods import method_registry
from poc.quantization_stage import _prior_costs


def main(summary):
    summary=Path(summary);report=json.loads(summary.read_text());root=ROOT/'artifacts/quantization'
    caps=json.loads((ROOT/'protocol/quantization-pilot-caps.json').read_text())
    proofs=[p for p in summary.parent.glob('native-offline-proof*.json') if json.loads(p.read_text())['status']=='passed']
    if not proofs: raise RuntimeError('actual native offline proof missing')
    proof=max(proofs,key=lambda p:p.stat().st_mtime)
    attempts=[]
    for p in sorted(root.glob('attempt-*/summary.json')):
        raw=json.loads(p.read_text());attempts.append({'path':str(p),'sha256':_sha(p),'recorded_status':raw['status'],'recorded_costs':raw['costs'],'cost_ledger':raw.get('cost_ledger'),'reason':raw.get('reason'),'attribution_note':'attempt3 SIGTERM was swallowed as recoverable method failure; all subsequent cost retained; fixed before actual4' if p.parent.name=='attempt-0d83f2dd972948b18693e244895e0b06' else 'attempt2 snapshot failed after numpy scalar serialization; authoritative ledger retains final cost' if raw['status']=='preflight' else None})
    anchor=report['anchor'];verify_bundle(Path(anchor['artifact']))
    if measure_bundle(Path(anchor['artifact'])).payload_bytes!=anchor['payload_bytes']: raise RuntimeError('immutable B* mismatch')
    rows=[]
    for r in report['rows']:
        row={k:v for k,v in r.items() if k!='quantizer_evidence'}
        if 'quantizer_evidence' in r:
            q=r['quantizer_evidence'];row['quantizer']={k:v for k,v in q.items() if k not in ('calibration_component_ids','internal_train_component_ids','internal_valid_component_ids')}
            row['quantizer']['fit_component_count']=len(q['internal_train_component_ids']);row['quantizer']['fit_validation_component_count']=len(q['internal_valid_component_ids'])
        rows.append(row)
    snapshots=[]
    for p in sorted((root/'source-snapshots').glob('*.zip')):
        snapshots.append({'path':str(p),'sha256':_sha(p),'bytes':p.stat().st_size,'scope':'all33 owned poc source files exactly match original successful attempt source hashes; frozen parsed caps from immutable RunSpec; no third-party source'})
    first=min(json.loads(Path(json.loads(p.read_text())['run_record']).read_text())['created_at'] for p in root.glob('attempt-*/summary.json'))
    exposure=ROOT/'data/prepared-135m-v2/exposures.jsonl'
    events=[json.loads(line) for line in exposure.read_text().splitlines()];events=[e for e in events if e['timestamp']>=first]
    roles=json.loads((ROOT/'data/prepared-135m-v2/data-manifest.json').read_text())['roles']
    observed_roles=sorted({roles[e['document_id']] for e in events})
    if not set(observed_roles)<={'fit','dev'}: raise RuntimeError('unexpected Task9 data exposure role')
    amendments=json.loads((root/'cost-amendments.json').read_text())
    best=next(r for r in rows if r['variant']==report['baseline_star'])
    evidence={'schema_version':1,'scope':'WP03 native format and preliminary dense frontier only; no protected evaluation, sharing advantage, anchor quality or training-success claim','core_implementation_commit':'ca048cf','source_snapshot_attribution':snapshots,'raw_attempts':attempts,'final_raw_summary':str(summary),'final_raw_summary_sha256':_sha(summary),'plan_sha256':report['plan_sha256'],'dev_windows_sha256':report['dev_windows_sha256'],'native_profile':report['native_numerical_controls'],'teacher_identity':report['teacher_identity'],'anchor':anchor,'rows':rows,'baseline_star':best,'actual_cumulative_costs':{**_prior_costs(root,caps['synthetic_qualification']['prior_observed_wall_seconds']),'valid_target_presentations':50078},'counter_amendments':amendments,'final_validation_note':'additional native final/trace fixtures20.22seconds plus nested-trace RED/GREEN6.64seconds; final CPU363passed/1explicit-assets skip/32hardwaredeselected in12.90seconds, prior full runs15.82/15.03seconds and focused18pass4.10seconds reported separately; historical source-bound evidence is not relabeled','caps':caps,'stage_peaks':{'larger':json.loads((root/'attempt-ae0c12cb6341423cb64e7a1ab8a13215/summary.json').read_text())['peaks'],'mixed_embedding':report['peaks'],'scope':'per-process sampled RSS and process-cumulative MLX counters overlap on unified memory, never summed; fine-grained isolated conversion peaks unknown; these are full conversion/load/calibration/evaluation stage envelopes'},'offline_proof':{'path':str(proof),'sha256':_sha(proof),'result':json.loads(proof.read_text())},'exposure_audit':{'path':str(exposure),'sha256':_sha(exposure),'task9_events':len(events),'roles':observed_roles,'purposes':sorted({e['purpose'] for e in events}),'protected_reads':False,'new_corpus_fetches':0},'formats':{'backend':'mlx_metal','mlx':'0.32.4','mlx_lm':'0.32.2','compute_dtype':'float32','packed_storage':'uint32+FP32 scales/offsets; BF16 private residuals','matrix':report['formats'],'training':'packed weights not claimed trainable; dense DWQ scale/offset differentiation verified at2/4-bit,g64; shared learned adapters not implemented'},'sv_feasibility':report['sv_feasibility'],'shared_learned_status':report['shared_learned_status'],'dynamic_reproducibility':{'status':'original sensitivity/threshold cause unresolved; validated fixed artifacts retained','same_source_seed_calibration':True,'only_map_difference':'tied embed_tokens.weight4-bit for target3.5 versus2-bit for target3.6; exact7,077,888B delta','upstream_semantics':'for a fixed score vector/parameter count, installed threshold search is monotone; discrete changes/coarse tolerance alone do not explain inversion','historical_scores_thresholds':'not_saved; cannot retrospectively diagnose or claim deterministic allocation reproduction','future_trace_amendment':'subsequent calls now persist full sensitivity vector, threshold and map digest; tiny synthetic RED/GREEN only; no historical quality rerun'},'wp03':report['wp03'],'next_action':'Task10 single-merge diagnostic; strong Q baseline does not cancel it; Task15 full SV and shared learned refinements remain future work','archives':{'status':'not_run_budget','reason':'Task8 measured level9 cost; B* unarchived complete bytes; Task20 finalist archive work'}}
    output=ROOT/'provenance/quantization-135m-2026-10-27.json';output.write_bytes(_canonical(evidence));digest=_sha(output)
    methods={k:{f.name:thaw(getattr(v,f.name)) for f in fields(v)} for k,v in method_registry().items()}
    methods['Q'].update(status='eligible',stage_readiness='preliminary_frontier_32fit_8fitvalid_32dev',reason='Actual pinned RTN/GPTQ/AWQ/DWQ/dynamic packed native artifacts and complete byte frontier; best eligible preliminary dense DWQ2/embed4. Dynamic fixed artifacts qualified; original allocation-trace reproducibility unresolved.',evidence_digest=digest,allowed_comparisons=['native_fidelity','payload','development_selection_below_Bstar'],supported_formats=['dense_affine2_g64','dense_affine3_g64_RTN','dense_affine4_g32_g64','dense_affine8_g64','dense_base2_tied_embedding4_g64','native_dynamic2_4_and4_8'],cost_budget=caps['aggregate_caps'])
    methods['T']['stage_readiness']='reviewed_tasks7_8_floating_and_task9_native_reference'
    methods['SV']['stage_readiness']='independent_tiny_paper_algebra_only_task15_pending'
    for key in ('R','RC','CKA','BS','P','H','QCE','M'): methods[key]['supported_formats']=['architecture_format_only_floating_and_shared_RTN4_g64_BF16_residuals'] if key in ('R','RC','CKA','H','M') else []
    eligibility={'schema_version':1,'evidence_path':str(output.relative_to(ROOT)),'evidence_sha256':digest,'methods':methods,'shared_learned_status':report['shared_learned_status'],'fixed_artifact_reproducibility_limits':evidence['dynamic_reproducibility'],'next_action':evidence['next_action']}
    (ROOT/'protocol/baseline-eligibility.json').write_bytes(_canonical(eligibility))
    print(json.dumps({'qualification':str(output),'sha256':digest,'baseline_star':best['variant'],'bytes':best['payload_bytes'],'kl':best['development']['kl_nats_per_target'],'cumulative':evidence['actual_cumulative_costs']}))

if __name__=='__main__': main(sys.argv[1])

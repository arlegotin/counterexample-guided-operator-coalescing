"""Two predeclared fresh-Adam sensitivity refits, never main ranking alternatives."""
from dataclasses import asdict,replace
import hashlib,json,subprocess,time,gc
from pathlib import Path
import psutil,torch
from poc.bank import make_bank,canonicalize_partition
from poc.budget import BudgetTracker
from poc.contracts import ProbeRecord,RunSpec
from poc.data.manifest import load_manifest
from poc.merge import restore_trial_delta,save_trial_delta,fit_trial
from poc.records import start_attempt,append_event,canonical_json
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,sha256_file
from poc.teacher_validation import _batch,_write
from poc.train import TrainingConfig
from poc.diagnostics import fixed_scores
from provenance.measure_single_merge import (ROOT,OUT,DiagnosticRecorder,_documents,_canonical,_accepted,_mean,
    training_sources,natural_metrics,activation_scores,probe_observations)


def run(config):
    policy=json.loads((OUT/'full-phase-approved.json').read_text());p=json.loads((OUT/'plan.json').read_text());limits=config.budgets.resolve();results=[]
    for pair_raw in policy['sensitivity_pairs']:
        pair=tuple(pair_raw);destination=OUT/'sensitivity'/f'{pair[0]:02d}-{pair[1]:02d}'
        if (destination/'result.json').exists():
            saved=json.loads((destination/'result.json').read_text())
            if saved['status']=='complete':results.append(saved);continue
            raise ValueError('failed sensitivity attempt requires explicit recovery; no silent repeat')
        primary_path=OUT/'trials-final'/f'{pair[0]:02d}-{pair[1]:02d}.json';primary=json.loads(primary_path.read_text())
        delta=OUT/'profile-delta' if pair==(0,15) else OUT/f'sensitivity-delta-{pair[0]:02d}-{pair[1]:02d}'
        cfg=TrainingConfig(max_updates=1,phase='local',affected_depths=pair,mixture_weights={'natural':.5,'student':.25,'donor':.25},
            kl_floor_per_target=json.loads((ROOT/p['numerics_path']).read_text())['negative_kl_floor_per_token'],numerical_profile=EXECUTION_PROFILE)
        source_files=sorted([*ROOT.glob('poc/**/*.py'),ROOT/'provenance/measure_single_merge.py',Path(__file__),ROOT/'provenance/measure_training_pilot.py',ROOT/'provenance/measure_probe_pilot.py',ROOT/'provenance/prepare_single_merge.py'])
        sources={str(f.relative_to(ROOT)):sha256_file(f) for f in source_files}
        spec=RunSpec(method_id='M',method_variant='task12_single_merge_sensitivity_refit',resolved_config={'plan_sha256':sha256_file(OUT/'plan.json'),'source_sha256':sources,'carry_in':p['task12_carry_in'],'training':cfg.identity(),'pair':pair,'primary_trial_sha256':sha256_file(primary_path),'source_amendment_sha256':sha256_file(OUT/'source-amendment.json')},
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_patch_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'])).hexdigest(),dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
            input_manifests={'teacher':p['teacher_asset_sha256'],'data':p['data_manifest_sha256'],'primary_delta_metadata':sha256_file(delta/'metadata.json'),'primary_delta_weights':sha256_file(delta/'weights.safetensors')},hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},seeds={'sensitivity':17},parent_run_ids=(),budgets=limits)
        attempt=start_attempt(spec,OUT/'runs');rec=DiagnosticRecorder(BudgetTracker(limits,attempt),p['student_call_cap']);started=time.monotonic()
        for f in source_files:
            target=attempt.path/'source'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
        result={'status':'running','pair':pair,'attempt':str(attempt.path.relative_to(ROOT)),'source_sha256':sources,'source_commit':spec.source_commit,'scope':'one extra exact8192 fresh-Adam update from retained primary state; never substitutes into20trial ranking','primary_identity':primary['identity']}
        try:
            if psutil.virtual_memory().available<10*1024**3:raise ValueError('fresh headroom below8GiB cap policy')
            asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=p['teacher_asset_sha256'])
            teacher=rec.call(lambda:load_teacher(asset,config).to('mps'));current=rec.call(lambda:make_bank(teacher,canonicalize_partition([[i] for i in range(30)],30),8,17))
            restored=rec.call(lambda:restore_trial_delta(current,teacher,delta,recorder=rec))
            if restored.fitted_digest!=primary['identity']['fitted_digest']:raise ValueError('primary fitted digest changed before sensitivity')
            extra=replace(restored,initialized_digest=restored.fitted_digest,fitted_digest=None,sources=None,fit_summary={})
            manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=p['data_manifest_sha256'],roles={'fit'})
            fit_docs=_documents(manifest,p['selected']['medoid_and_fit'],rec);fit_batches=[_canonical(d,manifest.membership[d.document_id],rec)[0] for d in fit_docs]
            fit_pool=[ProbeRecord(**x) for x in json.loads((OUT/'fit-pool.json').read_text())['rows']]
            score_pool=[ProbeRecord(**x) for x in json.loads((OUT/'probe_score-pool.json').read_text())['rows']]
            extra.sources,templates=training_sources(fit_pool,extra);_write(attempt.path/'source-templates.json',templates)
            extra=fit_trial(extra,replace(cfg,run_spec=spec),limits)
            fit_after=natural_metrics(teacher,extra.model,fit_batches[:3],rec)
            score_by_id={}
            for row in _accepted(score_pool,pair):score_by_id.setdefault(row.recipient_id,row)
            batches=[_batch(r.recipient_tokens,device='mps',document_id=r.recipient_id,cluster_id=r.recipient_component) for _,r in sorted(score_by_id.items())]
            natural_after=natural_metrics(teacher,extra.model,batches,rec);activation_after=activation_scores(teacher,extra,batches,rec)
            calibration=json.loads((OUT/'calibration.json').read_text())['summary'];probes,rows=probe_observations(teacher,extra,score_pool,rec,calibration)
            score_after={}
            for row in natural_after:
                c=row['component'];observed=probes[c]
                score_after[c]=fixed_scores(natural=row['kl_per_target'],student=_mean(observed['student']),random_donor=_mean(observed['random']),active_donor=_mean(observed['active']))|activation_after[c]
            save_trial_delta(extra,current,destination/'delta')
            result.update(status='complete',fit=extra.fit_summary,primary_fit_checks=primary['sensitivity']['after'],after_fit_checks=fit_after,primary_scores=primary['scores'],after_scores_by_component=score_after,
                after_natural_score_rows=natural_after,after_probe_rows=rows,fitted_digest=extra.fitted_digest,endpoint_predictions=False)
            del extra,restored,current,teacher,fit_pool,score_pool;gc.collect();torch.mps.empty_cache()
        except Exception as exc:
            result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc));append_event(attempt,dict(kind='failed',reason=str(exc)))
        finally:
            rec.finish_attempt(result,elapsed=time.monotonic()-started);_write(attempt.path/'result.json',result);_write(destination/'result.json',result)
            print(json.dumps({'sensitivity_pair':pair,'status':result['status'],'aggregate_cost':result['aggregate_phase_cost'],'failure':result.get('failure_reason')}),flush=True)
        results.append(result)
        if result['status']!='complete':break
    return results

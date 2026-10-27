"""Separately capped post-stop observation; never completes missing training."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil
import torch

from poc.budget import BudgetTracker
from poc.checkpoint import resolve_checkpoint,resolve_fitted_run,checkpoint_decoder
from poc.contracts import Budget,CostRecord,RunSpec
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.losses import stream_metrics
from poc.quantization_stage import _prepared_window
from poc.records import canonical_json,start_attempt,append_event
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,sha256_file
from poc.config import load_config
from poc.teacher_validation import _restricted,_batch,_write
from poc.train import _Dispatch,_Counts
from provenance.measure_training_pilot import ROOT,GiB,TrainingPilotRecorder


def run(global_root,global_run,global_attempt,out):
    started=time.monotonic()
    if not torch.backends.mps.is_available():raise RuntimeError('real MPS required')
    source=Path(global_root)/global_run/'attempts'/global_attempt
    run_record=json.loads((source/'run.json').read_text())
    if run_record['status']!='failed' or run_record['failure_category']!='BudgetExceeded':raise ValueError('fallback only after explicit resource stop')
    ref=resolve_checkpoint(source/'checkpoints')
    metadata=json.loads((Path(ref.directory)/'state.json').read_text())['items']
    step=metadata['step']['value'];digest=metadata['candidate_digest']['value']
    source_plan=json.loads((Path(global_root).parent/'plan.json').read_text())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=dict(global_run=global_run,global_attempt=global_attempt,source_step=step,source_digest=digest,
        source_checkpoint=asdict(ref),source_window_stream_sha256=sha256_file(source/'window-stream.json'),fit_check_ids=source_plan['observation_ids'],dev_ids=source_plan['dev_ids'],
        natural_window_policy=source_plan['natural_window_policy'],saved_windows_sha256=source_plan['saved_windows_sha256'],
        scope='separate observation of latest verified partial-global checkpoint; zero extra training',
        carry_in={'spec_paths':[],'audit_wall_seconds':0,'evidence':{}},wall_seconds=90,evaluated_targets=5621)
    plan_path=out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=p:raise ValueError('immutable observation plan differs')
    if not plan_path.exists():_write(plan_path,p)
    receipt=json.loads((ROOT/'provenance/preparation-135m-2026-10-27.json').read_text())
    limits=Budget(updates=0,valid_target_presentations=5621,teacher_forward_calls=512,suffix_backward_calls=0,
        student_backward_calls=0,search_proposals=0,search_seconds=0,wall_seconds=90,disk_bytes=8*GiB,
        memory_bytes=min(6*GiB,int(.8*psutil.virtual_memory().available)))
    files=sorted([str(f.relative_to(ROOT)) for f in (ROOT/'poc').rglob('*.py')]+['provenance/observe_training_checkpoint.py','provenance/measure_training_pilot.py','provenance/measure_probe_pilot.py'])
    sources={f:sha256_file(ROOT/f) for f in files};patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    spec=RunSpec(method_id='M',method_variant='task11_partial_checkpoint_observation',
        resolved_config={'plan_sha256':sha256_file(plan_path),'source_sha256':sources,'carry_in':p['carry_in'],'student_forward_cap':512},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty_patch_sha256=hashlib.sha256(patch).hexdigest(),
        dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
        input_manifests={'data':receipt['manifests']['data-manifest.json']['sha256'],'teacher':receipt['teacher_asset_manifest_sha256'],
                         'checkpoint':ref.manifest_sha256,'saved_windows':p['saved_windows_sha256']},
        hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},seeds={'observation':17},parent_run_ids=(global_run,),budgets=limits)
    attempt=start_attempt(spec,out/'runs');rec=TrainingPilotRecorder(BudgetTracker(limits,attempt),512)
    for f in files:
        destination=attempt.path/'source'/f;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,destination)
    (attempt.path/'source/dirty.patch').write_bytes(patch)
    result=dict(schema_version=1,status='running',run_id=attempt.run_id,attempt_id=attempt.attempt_id,
        attempt=str(attempt.path.relative_to(ROOT)),source_step=step,source_digest=digest,training_targets=0,observations=[],source_sha256=sources)
    record=json.loads((attempt.path/'run.json').read_text());record['status']='running';_write(attempt.path/'run.json',record)
    try:
        asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=receipt['teacher_asset_manifest_sha256'])
        teacher=rec.call(lambda:load_teacher(asset,load_config(ROOT/'configs/135m.yaml',phase='exploratory')).to('mps'))
        state,verified_ref=rec.call(lambda:resolve_fitted_run(global_root,global_run,global_attempt))
        if verified_ref!=ref or state.step!=step or state.candidate_digest!=digest:raise ValueError('observation source changed')
        student=rec.call(lambda:checkpoint_decoder(state).to('mps'));del state
        manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=receipt['manifests']['data-manifest.json']['sha256'],roles={'fit','dev'})
        if sha256_file(ROOT/'data/prepared-135m-v2/windows.json')!=p['saved_windows_sha256']:raise ValueError('saved window lineage changed')
        for role,identities,purpose in [('fit_observation',p['fit_check_ids'],'fit'),('dev',p['dev_ids'],'selection')]:
            docs=rec.call(lambda:list(open_records(_restricted(manifest,[r for r in manifest.records if r.document_id in identities]),purpose=purpose)))
            by_id={r.document_id:r for r in docs};metrics=[];windows=[]
            for identity in identities:
                w=_prepared_window(by_id[identity],p['natural_window_policy']['seed'])
                if len(w.token_ids)!=512:raise ValueError('observation window short')
                windows.append(asdict(w));b=_batch(w.token_ids,device='mps',document_id=identity,cluster_id=manifest.membership[identity],targets=w.target_indices)
                metrics.append(rec.call(lambda:stream_metrics(_Dispatch(teacher,_Counts(rec),'teacher'),_Dispatch(student,_Counts(rec),'student'),b,32),
                    processed_inputs=1,processed_tokens=512,valid_target_presentations=511))
            row=dict(role=role,step=step,valid_targets=sum(m.valid_targets for m in metrics),kl_sum=sum(m.kl for m in metrics),nll_sum=sum(m.nll for m in metrics))
            row['kl_per_target']=row['kl_sum']/row['valid_targets'];row['objective_per_target']=(row['kl_sum']+.1*row['nll_sum'])/row['valid_targets']
            result['observations'].append(row);_write(attempt.path/(role+'-windows.json'),windows);append_event(attempt,dict(kind='observation',**row))
            print(json.dumps(row),flush=True)
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc));raise
    finally:
        overrun=rec.finish_attempt(result,elapsed=time.monotonic()-started);_write(attempt.path/'result.json',result)
        record.update(status='complete' if result['status']=='passed' else 'failed',failure_category=result.get('failure_type'),failure_details=result.get('failure_reason'),
            output_manifest_sha256=sha256_file(attempt.path/'result.json') if result['status']=='passed' else None)
        _write(attempt.path/'run.json',record);append_event(attempt,dict(kind='result',status=result['status'],cost=result['cost']))
        print(json.dumps({'status':result['status'],'result_path':str(attempt.path/'result.json'),'cost':result['cost']}),flush=True)
        if overrun:raise overrun
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--global-run-root',type=Path,required=True);parser.add_argument('--global-run',required=True)
    parser.add_argument('--global-attempt',required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();run(a.global_run_root,a.global_run,a.global_attempt,a.out)

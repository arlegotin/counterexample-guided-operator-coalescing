"""Export recovered historical K15 weights under pinned v2 runtime, then compare."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path('/Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing')
PINNED=Path('/private/tmp/cgoc-audit-1625796')
RELATIVE=Path('artifacts/training/task11/learn31-canonical-v1/runs/5406a10de5b202d4a737889b7eea4dfea17d71b591c33378dec2323c21670d73/attempts/c154bb372f7c431cbaa29c51c7220181/checkpoints')


def convert(out):
    import torch
    import mlx.core as mx
    import resource
    from dataclasses import asdict
    import poc
    from poc.bundle import write_bundle
    from poc.checkpoint import resolve_checkpoint,load_checkpoint,checkpoint_decoder,_spec
    from poc.contracts import RuntimeSpec
    from poc.quantize import QuantizationSpec,quantize_rtn
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    from poc.teacher import sha256_file
    if Path(poc.__file__).resolve().parents[1]!=PINNED or EXECUTION_PROFILE!='eager-llama-hf-fp32-reciprocal-true-fp64-v2':
        raise ValueError('historical checkpoint must use untouched pinned runtime')
    start=time.monotonic();out.mkdir(exist_ok=False)
    checkpoint=ASSETS/RELATIVE;spec_path=checkpoint.parents[2]/'spec.json'
    state=load_checkpoint(resolve_checkpoint(checkpoint),_spec(json.loads(spec_path.read_text())))
    if state.step!=29 or state.candidate_digest!='a5a431e4262453d39e7e4558031e13018ef807ebaef977597248fe73bc0cc4d0':
        raise ValueError('wrong historical trained state')
    model=checkpoint_decoder(state)
    if len(model.partition.groups)!=15 or model.rank!=8:raise ValueError('wrong historical K/rank')
    if any(not torch.equal(p,state.model[n]) for n,p in model.named_parameters()):raise ValueError('historical values changed')
    original=ASSETS/'artifacts/quantization/attempt-0d83f2dd972948b18693e244895e0b06/floating'
    metadata=json.loads((original/'manifest.json').read_text())
    runtime=RuntimeSpec({n:str(original/'tokenizer'/n) for n in metadata['tokenizer_files']},
        {n:str(original/'LICENSES'/n) for n in metadata['license_files']},str(original/'requirements.lock'),
        metadata['install_bytes'],metadata['install_unknown_reason'])
    floating=write_bundle(model,out/'historical-K15-floating',runtime)
    artifacts=[dict(label='historical-K15-step29-floating',path=floating.directory,manifest_sha256=floating.manifest_sha256)]
    del model,state
    for bits,embedding in ((2,4),(4,2)):
        name=f'historical-K15-rtn{bits}-embed{embedding}'
        result=quantize_rtn(floating,QuantizationSpec('rtn',bits,64,str(out/name),embedding_bits=embedding),None)
        artifacts.append(dict(label=name,path=result.bundle.directory,manifest_sha256=result.bundle.manifest_sha256))
    receipt=dict(status='complete',artifacts=artifacts,source_runtime=poc.__file__,execution_profile=EXECUTION_PROFILE,
        historical_checkpoint_ref=asdict(resolve_checkpoint(checkpoint)),source_fitted_digest='a5a431e4262453d39e7e4558031e13018ef807ebaef977597248fe73bc0cc4d0',
        exact_values_verified=True,new_training_updates=0,wall_seconds=time.monotonic()-start,
        memory={'process_peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            'torch_mps_driver_bytes':torch.mps.driver_allocated_memory(),'mlx_peak_bytes':mx.get_peak_memory()},
        shared_quantizer='existing native RTN; no learned shared quantization claim')
    (out/'construction.json').write_text(json.dumps(receipt,indent=2)+'\n')


def run(out_override=None,construction_only=False):
    from dataclasses import replace
    import random
    import torch
    from poc.checkpoint import _spec
    from poc.continuation_budget import PilotLedger
    from poc.contracts import CostRecord
    from poc.records import run_id
    from poc.teacher import sha256_file
    from poc.train import _call
    from provenance.recovery_pilot import NativeTracker,write,finalize_costs,raise_for_accounting_error,subprocess_memory_bound
    from poc.continuation_budget import process_peak_bytes
    started=time.monotonic();recovery=ROOT/'artifacts/continuation-v2/recovery';frontier=ROOT/'artifacts/continuation-v2/frontier'
    out=Path(out_override).resolve() if out_override else ROOT/'artifacts/continuation-v2/legacy-frontier'
    out.mkdir(exist_ok=False)
    pilot=json.loads((recovery/'result.json').read_text());parent=_spec(pilot['run_spec'])
    ledger=PilotLedger(parent,recovery/'ledger');ledger.tracker=NativeTracker(parent.budgets,ledger.attempt,ROOT/'artifacts/continuation-v2')
    old_ids=json.loads((ASSETS/'artifacts/training/task11/learn31-canonical-v1/plan.json').read_text())['dev_ids']
    if old_ids!=pilot['population']['validation_ids']:raise ValueError('historical canonical validation population differs')
    spec=replace(parent,method_variant='legacy-trained-K15-native-artifacts-v2'+('-construction-replay' if construction_only else ''),
        budgets=replace(parent.budgets,updates=0,valid_target_presentations=0,wall_seconds=120 if construction_only else 600),parent_run_ids=(run_id(parent),),
        resolved_config={'script_sha256':sha256_file(__file__),'worker_sha256':sha256_file(ROOT/'provenance/native_artifact_worker.py'),
            'historical_checkpoint_pointer_sha256':sha256_file(ASSETS/RELATIVE/'current.json'),
            'inputs_sha256':sha256_file(frontier/'private-inputs.json'),
            'variants':[[2,4],[4,2]],'teacher_reference':'same verified dense teacher; frozen artifact-owned runtime/profile',
            'scope':'new export/RTN of existing step29 weights; no continuation of historical training or protected evaluation'})
    result=dict(status='running',run_spec=spec.to_dict(),rows=[],commands=[])
    try:
      with ledger.arm(spec,forecast=CostRecord(wall_seconds=spec.budgets.wall_seconds,disk_bytes=ledger.tracker.disk()+800*1024**2)) as recorder:
        command=[sys.executable,str(Path(__file__).resolve()),'--convert','--out',str(out/'export')]
        env={**os.environ,'PYTHONPATH':str(PINNED)}
        with (out/'construction.log').open('wb') as log:
            execution=_call(recorder,lambda:subprocess.run(command,cwd=PINNED,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180))
        result['commands'].append(dict(argv=command,exit_code=execution.returncode))
        if execution.returncode:raise RuntimeError('pinned historical export failed')
        construction=json.loads((out/'export/construction.json').read_text());result['construction']=construction
        recorder.record(CostRecord(memory_bytes=subprocess_memory_bound(process_peak_bytes()+torch.mps.driver_allocated_memory(),construction['memory'])))
        if construction_only:
            expected=json.loads((ROOT/'artifacts/continuation-v2/legacy-frontier-retry1/export/construction.json').read_text())
            if [(a['label'],a['manifest_sha256']) for a in construction['artifacts']]!=[(a['label'],a['manifest_sha256']) for a in expected['artifacts']]:
                raise ValueError('construction replay artifact bytes differ')
            result.update(status='complete_construction_replay',byte_identical_manifest_and_all_files=True)
            return
        artifacts=construction['artifacts'];random.Random(17).shuffle(artifacts)
        write(out/'freeze.json',dict(artifacts=artifacts,inputs_sha256=spec.resolved_config['inputs_sha256'],
            frame='same8 known exploratory development components',expected_original_floating_kl=6.49408069095,
            original_training_stays_WP04='INCONCLUSIVE'))
        for artifact in artifacts:
            target=out/(artifact['label']+'.json')
            command=[sys.executable,str(ROOT/'provenance/native_artifact_worker.py'),'--artifact',artifact['path'],
                '--manifest-sha256',artifact['manifest_sha256'],'--inputs',str(frontier/'private-inputs.json'),
                '--inputs-sha256',spec.resolved_config['inputs_sha256'],'--teacher-manifest',str(ASSETS/'data/prepared-135m-v2/teacher-assets.json'),'--out',str(target)]
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            with (out/(artifact['label']+'.log')).open('wb') as log:
                execution=_call(recorder,lambda:subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,600-(time.monotonic()-started))),
                    teacher_forward_calls=136,student_forward_calls=238)
            result['commands'].append(dict(argv=command,exit_code=execution.returncode))
            if execution.returncode:raise RuntimeError('native historical worker failed: '+artifact['label'])
            row=json.loads(target.read_text())
            if row['valid_targets']!=4088 or row['teacher_forward_calls']!=136 or row['student_forward_calls']!=238:raise ValueError('native denominator/count mismatch')
            result['rows'].append({'label':artifact['label'],**row})
            recorder.record(CostRecord(memory_bytes=subprocess_memory_bound(process_peak_bytes()+torch.mps.driver_allocated_memory(),row['memory'])))
            print(json.dumps({'label':artifact['label'],'bytes':row['payload_bytes'],'KL':row['kl_per_target']}),flush=True)
            write(out/'result.json',result)
        floating=next(r for r in result['rows'] if r['label']=='historical-K15-step29-floating')
        if abs(floating['kl_per_target']-6.49408069095)>2e-5:raise ValueError('historical frozen float quality mismatch')
        result['status']='complete_exploratory'
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        finalize_costs(result,ledger.tracker,elapsed=time.monotonic()-started);write(out/'result.json',result)
        raise_for_accounting_error(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--convert',action='store_true');parser.add_argument('--out',type=Path)
    parser.add_argument('--construction-only',action='store_true')
    args=parser.parse_args()
    if args.convert:convert(args.out)
    else:run(args.out,args.construction_only)

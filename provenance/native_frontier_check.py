"""Bounded matched-artifact exploratory check following the recovery pilot."""
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

from poc.bundle import write_bundle,verify_bundle
from poc.checkpoint import CheckpointRef,load_checkpoint,checkpoint_decoder,_spec
from poc.continuation_budget import PilotLedger
from poc.contracts import CostRecord,RuntimeSpec
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.data.windows import TokenDocument,WindowPolicy,make_windows
from poc.recovery_data import restricted_population
from poc.continuation_plan import digest
from poc.records import run_id
from poc.teacher import sha256_file
from poc.train import _call
from provenance.recovery_pilot import NativeTracker,write,finalize_costs

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path('/Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing')


def main():
    started=time.monotonic();recovery=ROOT/'artifacts/continuation-v2/recovery'
    pilot=json.loads((recovery/'result.json').read_text())
    if pilot['status']!='complete_exploratory':raise ValueError('completed valid pilot required')
    out=ROOT/'artifacts/continuation-v2/frontier';out.mkdir(exist_ok=False)
    parent=_spec(pilot['run_spec']);ledger=PilotLedger(parent,recovery/'ledger')
    ledger.tracker=NativeTracker(parent.budgets,ledger.attempt,ROOT/'artifacts/continuation-v2')
    source={str(p.relative_to(ROOT)):sha256_file(p) for p in (Path(__file__),ROOT/'provenance/native_artifact_worker.py')}
    child=replace(parent,method_variant='exploratory-matched-native-frontier-v2',
        budgets=replace(parent.budgets,updates=0,valid_target_presentations=0,wall_seconds=480),
        parent_run_ids=(run_id(parent),),resolved_config={'source':source,'pilot_digest':sha256_file(recovery/'result.json'),
            'comparison':'fixed best natural-fit shared artifact versus three previously qualified dense points and floating no-op',
            'validation_ids':pilot['population']['validation_ids'],'quality_reduction':'CPUFP64 full-vocabulary KL on nativeFP32 logits',
            'native_profiles':'each artifact executes its own frozen runtime in a fresh process; no historical relabeling'})
    result=dict(status='running',run_spec=child.to_dict(),rows=[],commands=[])
    try:
      with ledger.arm(child,forecast=CostRecord(wall_seconds=480)) as recorder:
        quant=json.loads((ASSETS/'provenance/quantization-135m-2026-10-27.json').read_text())
        requested=('frontier-rtn-b8-g64','frontier-dwq-b4-g64','embedding-dwq-b2-g64-embed4')
        artifacts=[]
        for name in requested:
            row=next(r for r in quant['rows'] if r['variant']==name and r['status']=='eligible')
            artifacts.append(dict(label=name,path=row['artifact'],manifest_sha256=row['manifest_sha256'],
                original_construction_and_dev_seconds=row.get('construction_and_dev_seconds'),historical_cost='reused artifact, not newly constructed for free'))
        floating=ASSETS/'artifacts/quantization/attempt-0d83f2dd972948b18693e244895e0b06/floating'
        artifacts.append(dict(label='teacher-native-noop',path=str(floating),manifest_sha256=sha256_file(floating/'manifest.json')))
        # Fixed validation membership and role guards before token content.
        manifest=load_manifest(ASSETS/'data/prepared-135m-v2/data-manifest.json',expected_sha256=parent.input_manifests['data'],roles={'dev'})
        selected=restricted_population(manifest,pilot['population']['validation_ids'],role='dev')
        selected=replace(selected,exposure_path=str(recovery/'exposures.jsonl'),lineage_sha256=parent.input_manifests['original_exposure'])
        def prepare_inputs():
            windows=[]
            for doc in open_records(selected,purpose='selection'):
                if sha256_file(doc.token_path)!=doc.token_sha256:raise ValueError('validation token mismatch')
                raw=json.loads(Path(doc.token_path).read_text())
                w=make_windows(TokenDocument(doc.document_id,tuple(raw['fields']['text'])),WindowPolicy())[0]
                expected=next(r for r in pilot['population']['validation_windows'] if r['document_id']==doc.document_id)
                if digest(w.token_ids)!=expected['window_sha256'] or w.source_start!=expected['source_start']:raise ValueError('pilot validation window differs')
                windows.append(dict(document_id=doc.document_id,component_id=selected.membership[doc.document_id],
                    tokens=w.token_ids,target_mask=[i+1 in w.target_indices for i in range(len(w.token_ids)-1)],window_sha256=expected['window_sha256']))
            write(out/'private-inputs.json',{'windows':windows})
        _call(recorder,prepare_inputs)
        def export_shared():
            ref=CheckpointRef(**pilot['arms']['B']['checkpoint'])
            spec_path=recovery/'ledger'/run_id(parent)/'children'/pilot['arms']['B']['run_id']/'spec.json'
            state=load_checkpoint(ref,_spec(json.loads(spec_path.read_text())))
            model=checkpoint_decoder(state)
            if any(not torch.equal(p,state.model[n]) for n,p in model.named_parameters()):raise ValueError('checkpoint reconstruction values changed')
            raw=json.loads((floating/'manifest.json').read_text())
            runtime=RuntimeSpec({n:str(floating/'tokenizer'/n) for n in raw['tokenizer_files']},
                {n:str(floating/'LICENSES'/n) for n in raw['license_files']},str(floating/'requirements.lock'),
                raw['install_bytes'],raw['install_unknown_reason'])
            bundle=write_bundle(model,out/'shared-B-floating',runtime)
            artifacts.append(dict(label='shared-natural-step16',path=bundle.directory,manifest_sha256=bundle.manifest_sha256,
                source_checkpoint=pilot['arms']['B']['checkpoint'],source_fitted_digest=state.candidate_digest,
                exact_parameter_values_verified=True))
            del model,state
        _call(recorder,export_shared);gc.collect();torch.mps.empty_cache()
        random.Random(17).shuffle(artifacts)
        freeze=dict(artifacts=artifacts,inputs_sha256=sha256_file(out/'private-inputs.json'),source=source,
            policy='fixed artifact set chosen before these predictions, common known exploratory8components; no protected inference')
        write(out/'freeze.json',freeze)
        print(json.dumps({'event':'frontier_frozen','artifacts':[a['label'] for a in artifacts]}),flush=True)
        for artifact in artifacts:
            target=out/(artifact['label']+'.json')
            command=[sys.executable,str(ROOT/'provenance/native_artifact_worker.py'),'--artifact',artifact['path'],
                '--manifest-sha256',artifact['manifest_sha256'],'--inputs',str(out/'private-inputs.json'),
                '--inputs-sha256',freeze['inputs_sha256'],'--teacher-manifest',str(ASSETS/'data/prepared-135m-v2/teacher-assets.json'),'--out',str(target)]
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            remaining=480-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('matched artifact480s cap exhausted')
            with (out/(artifact['label']+'.log')).open('wb') as log:
                execution=_call(recorder,lambda:subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining),
                    teacher_forward_calls=136,student_forward_calls=238)
            result['commands'].append(dict(argv=command,exit_code=execution.returncode))
            if execution.returncode:raise RuntimeError('artifact worker failed: '+artifact['label'])
            row=json.loads(target.read_text())
            if row['teacher_forward_calls']!=136 or row['student_forward_calls']!=238:raise ValueError('native operation counts differ from frozen precharge')
            if row['valid_targets']!=4088:raise ValueError('common validation denominator differs')
            result['rows'].append({'label':artifact['label'],**row})
            memory=row['memory']['process_peak_rss_bytes']+row['memory']['torch_mps_driver_bytes']+row['memory']['mlx_peak_bytes']
            recorder.record(CostRecord(memory_bytes=memory))
            print(json.dumps({'event':'native_result','label':artifact['label'],'bytes':row['payload_bytes'],'KL':row['kl_per_target']}),flush=True)
            write(out/'result.json',result)
        noop=next(r for r in result['rows'] if r['label']=='teacher-native-noop')
        shared=next(r for r in result['rows'] if r['label']=='shared-natural-step16')
        if abs(noop['kl_per_target'])>2e-6:raise ValueError('cross-runtime teacher floor too large')
        if abs(shared['kl_per_target']-pilot['arms']['B']['observations'][-1]['validation']['kl_per_target'])>pilot['decision_resolution']:
            raise ValueError('native reload/scientific quality differs beyond decision resolution')
        result['status']='complete_exploratory';result['recovery_and_frontier_total_cost']=ledger.tracker.totals.to_dict()
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        finalize_costs(result,ledger.tracker,elapsed=time.monotonic()-started)
        write(out/'result.json',result)


if __name__=='__main__':main()

"""Bounded WP03 pilot: isolated fit calibration, native fidelity, immutable B*."""
from dataclasses import asdict,replace
from pathlib import Path
import fcntl
import gc
import hashlib
import json
import os
import shutil
import signal
import tempfile
import threading
import time
from uuid import uuid4

from poc.bundle import _canonical,_sha,verify_bundle,measure_bundle,load_bundle
from poc.contracts import BundleRef,RuntimeSpec,Budget,CostRecord,RunSpec,thaw
from poc.budget import BudgetTracker,BudgetExceeded
from poc.records import start_attempt,append_event

ROOT=Path(__file__).resolve().parents[1]


class PilotInterrupted(BaseException):
    """A requested termination always escapes per-method recoverable failures."""


def _failure_status(error):
    from poc.budget import BudgetExceeded
    if isinstance(error,BudgetExceeded): raise error
    if isinstance(error,ImportError): return 'unavailable_upstream_entrypoint'
    if isinstance(error,NotImplementedError): return 'not_yet_implemented'
    if 'unsupported native affine' in str(error): return 'unsupported_kernel'
    if 'fidelity' in str(error) or 'parity' in str(error): return 'unresolved_fidelity'
    return 'adapter_execution_failed'


def _write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(_canonical(value));temporary.replace(path)


def publish_anchor(ref,root):
    """Verify and publish a byte-identical independent copy under detached digest."""
    source=Path(ref.directory);verify_bundle(source)
    if _sha(source/'manifest.json')!=ref.manifest_sha256: raise ValueError('anchor digest mismatch')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/ref.manifest_sha256
    if target.exists():
        verify_bundle(target)
        if _sha(target/'manifest.json')!=ref.manifest_sha256: raise ValueError('durable anchor identity mismatch')
    else:
        staging=Path(tempfile.mkdtemp(prefix='.publish-',dir=root))
        try:
            shutil.copytree(source,staging,dirs_exist_ok=True)
            verify_bundle(staging)
            if _sha(staging/'manifest.json')!=ref.manifest_sha256: raise ValueError('published anchor differs')
            staging.rename(target)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    return BundleRef(str(target.resolve()),ref.manifest_sha256)


def _subset(manifest,count):
    selected=[];components=set()
    for r in sorted(manifest.records,key=lambda r:r.document_id):
        if r.source!='HuggingFaceFW/fineweb-edu' or r.metadata.get('token_count',0)<600: continue
        component=manifest.membership[r.document_id]
        if component not in components: selected.append(r);components.add(component)
        if len(selected)==count: break
    if len(selected)!=count: raise ValueError('insufficient predefined pilot components')
    return _restrict(manifest,selected)


def _restrict(manifest,selected):
    ids={r.document_id for r in selected}
    return replace(manifest,records=tuple(selected),roles={i:manifest.roles[i] for i in ids},membership={i:manifest.membership[i] for i in ids},quarantine={i:v for i,v in manifest.quarantine.items() if i in ids})


def _prepared_window(record,seed):
    from poc.data.windows import TokenDocument,WindowPolicy,make_windows
    if _sha(Path(record.token_path))!=record.token_sha256: raise ValueError('selected dev token hash mismatch')
    raw=json.loads(Path(record.token_path).read_text())
    if raw['document_id']!=record.document_id: raise ValueError('token document mismatch')
    return make_windows(TokenDocument(record.document_id,tuple(raw['fields']['text'])),WindowPolicy(seed=seed))[0]


def _evaluation_contract(source_root,teacher_bundle,windows_sha256,expected_source_hashes=None):
    """Content identity for executed metrics/teacher; excludes accounting-only code.

    Historical source is accepted only after its whole-file hash matches the
    original report. ASTs then explicitly compare the numerical evaluator and
    native runtime, while CostRecord/logging growth cannot relabel old results.
    """
    import ast
    dependencies={'poc/quantization_stage.py':{'_quality','_native_numerical_controls'},
                  'poc/quantize.py':{'_native_batch'},
                  'poc/runtime/mlx_decoder.py':None,
                  'poc/contracts.py':{'_number','freeze','thaw','_optional_reason','Batch','Boundary','CacheState','LayerCache','ModelManifest','ByteAccounting'}}
    evaluator={}
    for filename,names in dependencies.items():
        path=Path(source_root)/filename
        if expected_source_hashes is not None and _sha(path)!=expected_source_hashes.get(filename):
            raise ValueError('historical evaluator source identity mismatch: '+filename)
        tree=ast.parse(path.read_text())
        nodes=tree.body if names is None else [node for node in tree.body if getattr(node,'name',None) in names]
        if names is not None and {node.name for node in nodes}!=names: raise ValueError('incomplete evaluator dependency binding')
        evaluator[filename]=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
        if filename=='poc/quantization_stage.py':
            run=next(node for node in tree.body if getattr(node,'name',None)=='_run')
            critical={'native_teacher','batches','teacher_hidden','batch','candidate'}
            flow=[]
            for node in ast.walk(run):
                if not any(isinstance(item,ast.Name) and item.id in critical for item in ast.walk(node)): continue
                if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr)): flow.append(node)
                elif isinstance(node,ast.For): flow.append(ast.Tuple(elts=[node.target,node.iter],ctx=ast.Load()))
                elif isinstance(node,(ast.If,ast.While)): flow.append(node.test)
            for node in list(tree.body)+list(ast.walk(run)):
                if isinstance(node,ast.ImportFrom):
                    aliases=[a for a in node.names if (a.asname or a.name) in {'load_bundle','_native_batch','_quality','_native_numerical_controls','_prepared_window'}]
                    if aliases: flow.append(ast.ImportFrom(module=node.module,names=aliases,level=node.level))
            evaluator[filename+'::_run_numerical_flow']=hashlib.sha256(ast.dump(ast.Module(body=flow,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    manifest=verify_bundle(Path(teacher_bundle)).metadata
    teacher={name:thaw(manifest[name]) for name in ('source_checkpoint_digest','tokenizer_digest','execution_profile','config','head_dim','rope_theta','norm_epsilons','dependency_lock_sha256')}
    teacher['weight_files']={name:entry['sha256'] for name,entry in manifest['file_entries'].items() if name.startswith('weights/')}
    return {'schema_version':1,'evaluator':evaluator,'teacher':teacher,'windows':windows_sha256}


def _historical_contract(evidence,previous):
    if 'evaluation_contract' in evidence: return evidence['evaluation_contract']
    previous=Path(previous)
    snapshot=previous.parent.parent/'source-snapshots'/previous.parent.name
    if not snapshot.is_dir(): return None
    windows=previous.parent/'dev-windows.json'
    if not windows.is_file() or _sha(windows)!=evidence.get('dev_windows_sha256'): raise ValueError('historical evaluation window identity mismatch')
    return _evaluation_contract(snapshot,previous.parent/'floating',evidence['dev_windows_sha256'],evidence['source_sha256'])


def _reusable_rows(evidence,previous,contract):
    """Reuse exact artifacts under an explicitly equal evaluator/teacher contract."""
    original_contract=_historical_contract(evidence,previous)
    if original_contract!=contract: return []
    contracts={str(previous):original_contract}
    rows=[]
    for row in evidence.get('rows',[]):
        if row['status'] not in ('eligible','operation_smoke_only','unsupported_kernel'): continue
        origin=evidence;origin_path=Path(previous);original=row;visited=set()
        while original.get('reused_from_summary'):
            path=Path(original['reused_from_summary'])
            if str(path) in visited: raise ValueError('cyclic reuse attribution')
            visited.add(str(path))
            if not path.is_file() or _sha(path)!=original['reused_from_summary_sha256']: raise ValueError('historical reused summary identity mismatch')
            origin=json.loads(path.read_text());origin_path=path
            matches=[r for r in origin['rows'] if r['variant']==original['variant'] and r.get('manifest_sha256')==original.get('manifest_sha256')]
            if len(matches)!=1: raise ValueError('ambiguous original reuse attribution')
            original=matches[0]
        if str(origin_path) not in contracts: contracts[str(origin_path)]=_historical_contract(origin,origin_path)
        if contracts[str(origin_path)]!=contract: continue
        if row.get('artifact'):
            artifact=Path(row['artifact'])
            if _sha(artifact/'manifest.json')!=row['manifest_sha256']: raise ValueError('historical reused artifact identity mismatch')
            if measure_bundle(artifact).payload_bytes!=row['payload_bytes']: raise ValueError('historical reused payload byte identity mismatch')
        copy=dict(row)
        copy['reused_from_summary']=str(previous)
        copy['reused_from_summary_sha256']=_sha(Path(previous)) if Path(previous).exists() else None
        copy['reuse_audit']={'evaluation_contract_sha256':hashlib.sha256(_canonical(contract)).hexdigest(),
                             'original_summary':str(origin_path),'original_source_sha256':origin.get('source_sha256'),
                             'artifact_manifest_sha256':row.get('manifest_sha256'),'payload_bytes':row.get('payload_bytes'),
                             'scope':'exact detached artifact/bytes, source-audited numerical evaluator, canonical floating teacher values, dependency lock and frozen windows; original outcomes/source retained'}
        rows.append(copy)
    return rows


class PilotRecorder:
    """One serialized job; live snapshots survive interruption and retries."""
    def __init__(self,out,report,prior,caps,tracker):
        self.out,self.report,self.prior,self.caps=out,report,prior,caps
        self.tracker=tracker
        self.baseline=tracker.totals
        self.lock=threading.RLock()
        self.started=time.monotonic();self.stop=threading.Event();self.error=None
        self.costs={'teacher_forward_calls':0,'student_forward_calls':0,'student_backward_calls':0,'fit_updates':0,'conversion_attempts':0}
        self.peaks={'process_rss_bytes':0,'mlx_active_bytes':0,'mlx_peak_memory_bytes':0,'generated_disk_bytes':0}
    def snapshot(self):
        with self.lock: self._snapshot()
    def _snapshot(self):
        import psutil
        import sys
        self.peaks['process_rss_bytes']=max(self.peaks['process_rss_bytes'],psutil.Process().memory_info().rss)
        if 'mlx.core' in sys.modules:
            mx=sys.modules['mlx.core']
            self.peaks['mlx_active_bytes']=max(self.peaks['mlx_active_bytes'],mx.get_active_memory())
            self.peaks['mlx_peak_memory_bytes']=mx.get_peak_memory()
        spent=self.tracker.totals
        self.report['costs']={**self.costs,'teacher_forward_calls':spent.teacher_forward_calls-self.baseline.teacher_forward_calls,'student_forward_calls':spent.student_forward_calls-self.baseline.student_forward_calls,'student_backward_calls':spent.student_backward_calls-self.baseline.student_backward_calls,'fit_updates':spent.updates-self.baseline.updates,'wall_seconds':time.monotonic()-self.started}
        self.report['authoritative_cost_record']=spent.to_dict()
        self.report['peaks']=dict(self.peaks)
        _write(self.out/'summary.json',self.report)
    def _check_aggregate(self,next_cost):
        if self.error: raise BudgetExceeded(self.error)
        elapsed=time.monotonic()-self.started
        if self.prior['wall_seconds']+max(elapsed,self.tracker.totals.wall_seconds-self.baseline.wall_seconds)+next_cost.wall_seconds>self.caps['wall_seconds']:
            raise BudgetExceeded('aggregate pilot wall cap exhausted')
        spent=self.tracker.totals
        for key,limit in (('teacher_forward_calls','teacher_forward_calls'),('student_forward_calls','student_forward_calls'),('student_backward_calls','student_backward_calls'),('updates','fit_updates')):
            if self.prior.get(limit,0)+getattr(spent,key)-getattr(self.baseline,key)+getattr(next_cost,key)>self.caps[limit]:
                raise BudgetExceeded('aggregate cap exhausted: '+limit)

    def check(self,next_cost):
        self.tracker.check(next_cost)
        self._check_aggregate(next_cost)

    def record(self,actual_cost):
        # Already-spent work must survive both local and aggregate overruns.
        try: self.tracker.record(actual_cost)
        finally: self.snapshot()
        self._check_aggregate(CostRecord())

    def account(self,**increments):
        import psutil
        cost=CostRecord(teacher_forward_calls=increments.get('teacher_forward_calls',0),student_forward_calls=increments.get('student_forward_calls',0),student_backward_calls=increments.get('student_backward_calls',0),updates=increments.get('fit_updates',0))
        self.check(cost)
        conversion=increments.get('conversion_attempts',0)
        if self.prior.get('conversion_attempts',0)+self.costs['conversion_attempts']+conversion>self.caps['conversion_attempts']:
            raise BudgetExceeded('aggregate cap exhausted: conversion_attempts')
        if psutil.virtual_memory().available<1024**3: raise BudgetExceeded('less than1GiB incremental compute headroom')
        self.costs['conversion_attempts']+=conversion
        self.record(cost)
    def monitor(self):
        import psutil
        while not self.stop.wait(.5):
            rss=psutil.Process().memory_info().rss
            try:
                import mlx.core as mx
                active=mx.get_active_memory()
            except ImportError: active=0
            if rss>self.caps['process_rss_bytes'] or active>self.caps['mlx_active_bytes'] or self.prior['wall_seconds']+time.monotonic()-self.started>self.caps['wall_seconds']:
                self.error='live finite resource cap exceeded';os.kill(os.getpid(),signal.SIGTERM);return
            # Persist lightweight live cost, including a kill during one operation.
            self.snapshot()


def _quality(teacher,candidate,batches,recorder,teacher_hidden):
    import mlx.core as mx
    kl=nll=0.;targets=disagree=0
    for batch,ht in zip(batches,teacher_hidden,strict=True):
        if candidate is teacher: hs=ht
        else:
            recorder.account(student_forward_calls=1)
            hs=candidate.final_hidden(batch)
        for lo in range(0,batch.input_ids.shape[1]-1,32):
            hi=min(lo+32,batch.input_ids.shape[1]-1)
            t=teacher.readout(ht[:,lo:hi]);s=candidate.readout(hs[:,lo:hi])
            lt=t-mx.logsumexp(t,axis=-1,keepdims=True);ls=s-mx.logsumexp(s,axis=-1,keepdims=True)
            mask=batch.target_mask[:,lo:hi]
            tokenkl=(mx.exp(lt)*(lt-ls)).sum(axis=-1)
            selected=mx.take_along_axis(ls,batch.input_ids[:,lo+1:hi+1,None],axis=-1).squeeze(-1)
            mx.eval(tokenkl,selected)
            kl+=sum(mx.where(mask,tokenkl,0).reshape(-1).tolist());nll-=sum(mx.where(mask,selected,0).reshape(-1).tolist())
            disagree+=int(mx.sum(mask & (mx.argmax(t,axis=-1)!=mx.argmax(s,axis=-1))).item())
            targets+=int(mx.sum(mask).item())
        del ht,hs
    if not targets: raise ValueError('zero target development population')
    return {'kl_sum':kl,'nll_sum':nll,'valid_targets':targets,'disagreements':disagree,'kl_nats_per_target':kl/targets,'nll_nats_per_target':nll/targets,'context_tokens':512,'temperature':1.,'head_chunk':32,'same_common_native_boundary':True}


def _native_numerical_controls(teacher,hidden):
    import mlx.core as mx
    import torch
    import numpy as np
    rows=[];max_error=0.
    for lo,hi in ((0,32),(480,511)):
        logits=teacher.readout(hidden[:,lo:hi]);mx.eval(logits)
        lt=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
        ref=torch.from_numpy(np.asarray(logits).copy()).double().log_softmax(-1)
        for epsilon in (0.,1e-6,1e-4):
            changed=logits.at[...,0].add(epsilon)
            ls=changed-mx.logsumexp(changed,axis=-1,keepdims=True)
            native=float((mx.exp(lt)*(lt-ls)).sum().item())/(hi-lo)
            changed64=torch.from_numpy(np.asarray(changed).copy()).double().log_softmax(-1)
            exact=float((ref.exp()*(ref-changed64)).sum().item())/(hi-lo)
            max_error=max(max_error,abs(native-exact))
            rows.append({'head_positions':[lo,hi],'epsilon':epsilon,'raw_native_kl_per_target':native,'fp64_reference_kl_per_target':exact})
    return {'profile':'mlx-metal-fp32-logsumexp-head32-scalar-fp64-v1','temperature':1.,'rows':rows,'negative_kl_floor_per_target':float(8*max_error+np.finfo(np.float32).eps),'scope':'measured primary512 teacher logits, head32 and31; no MPS floor reused'}


def _run(config,out,report,recorder,policy):
    import mlx.core as mx
    import torch
    from poc.teacher import AssetManifest,load_teacher
    from poc.data.manifest import load_manifest
    from poc.data.access import open_records
    from poc.data.windows import TokenDocument,WindowPolicy,make_windows
    from poc.quantize import QuantizationSpec,quantize_rtn,quantize_dense,discover_quantization_capabilities,_native_batch
    from poc.export import make_payload_anchor
    from poc.methods import method_registry,sv_feasibility
    from poc.contracts import thaw
    from poc.bundle import write_bundle,measure_resident
    if not mx.metal.is_available(): raise RuntimeError('native Metal required; no CPU fallback')
    mx.set_default_device(mx.gpu);torch.set_num_threads(4)
    preparation=config.preparation.output_dir
    receipt=json.loads((ROOT/'provenance/preparation-135m-2026-10-27.json').read_text())
    asset=AssetManifest.load(preparation/'teacher-assets.json',expected_sha256=config.teacher.weights_sha256)
    expected=receipt['manifests']['data-manifest.json']['sha256']
    fit=_subset(load_manifest(preparation/'data-manifest.json',expected_sha256=expected,roles={'fit'}),40)
    dev=_subset(load_manifest(preparation/'data-manifest.json',expected_sha256=expected,roles={'dev'}),32)
    plan={'fit_ids':[r.document_id for r in fit.records],'fit_component_ids':[fit.membership[r.document_id] for r in fit.records],
          'internal_train_ids':[r.document_id for r in fit.records[:32]],'internal_validation_ids':[r.document_id for r in fit.records[32:]],
          'smoke_train_ids':[r.document_id for r in fit.records[:2]],'smoke_internal_valid_ids':[r.document_id for r in fit.records[32:34]],'smoke_dev_ids':[r.document_id for r in dev.records[:4]],
          'dev_ids':[r.document_id for r in dev.records],'fit_manifest_sha256':fit.digest(),'dev_manifest_sha256':dev.digest(),
          'selection_policy':policy['policy'],'protected_access':False,'windows_sha256':_sha(preparation/'windows.json')}
    if plan['windows_sha256']!=receipt['manifests']['windows.json']['sha256']: raise ValueError('immutable window manifest hash mismatch')
    _write(out/'plan.json',plan);report['plan_sha256']=_sha(out/'plan.json')
    report['formats']=[asdict(r) for r in discover_quantization_capabilities([(49152,576),(576,576),(192,576),(576,1536),(1536,576)])]
    recorder.snapshot()
    # Actual teacher and initializer, never the Task8 constant-factor fixture.
    source=load_teacher(asset)
    float_input=Path('/private/tmp/cgoc-task8-floating-attempt5/dense')
    old=verify_bundle(float_input)
    if _sha(float_input/'manifest.json')!='14f75968d5f1eb0c1401cb84d8816dad2f28a0a345dfd73f0c5e7eb09db030f0': raise ValueError('reviewed floating input differs')
    runtime=RuntimeSpec({n:str(float_input/'tokenizer'/n) for n in old.metadata['tokenizer_files']},{n:str(float_input/'LICENSES'/n) for n in old.metadata['license_files']},str(float_input/'requirements.lock'),old.metadata['install_bytes'],old.metadata['install_unknown_reason'])
    anchor_path=ROOT/'provenance/payload-anchor-135m.json'
    if anchor_path.exists():
        anchor_receipt=json.loads(anchor_path.read_text())
        anchor=BundleRef(anchor_receipt['artifact'],anchor_receipt['manifest_sha256']);verify_bundle(Path(anchor.directory))
        actualbytes=measure_bundle(Path(anchor.directory)).payload_bytes
        if _sha(Path(anchor.directory)/'manifest.json')!=anchor.manifest_sha256 or actualbytes!=anchor_receipt['payload_bytes']: raise ValueError('immutable existing anchor mismatch')
        report['anchor']=anchor_receipt
    else:
        recorder.account(conversion_attempts=1)
        begin=time.monotonic();anchor_staging=make_payload_anchor(source,runtime,recorder=recorder)
        anchor=publish_anchor(anchor_staging,ROOT/'artifacts/payload-anchors')
        actualbytes=measure_bundle(Path(anchor.directory)).payload_bytes
        anchor_receipt={'schema_version':1,'scope':'untrained prescribed format anchor only; no anchor quality or training-success claim','artifact':anchor.directory,'manifest_sha256':anchor.manifest_sha256,'payload_bytes':actualbytes,
            'initializer':'SEQUENCE consecutive pairs,first original depth medoid,CPU FP64 symmetric-square-root SVD,seed17,zero delta seeded A/zero B','K':15,'rank':8,'bits':4,'group_size':64,'residual_dtype':'bfloat16','residual_scale':1.,
            'conversion_publish_seconds':time.monotonic()-begin,'caps_sha256':_sha(ROOT/'protocol/quantization-pilot-caps.json'),'source_asset_manifest_sha256':asset.expected_sha256,'source_sha256':report['source_sha256']}
        anchor_path=ROOT/'provenance/payload-anchor-135m.json'
        if anchor_path.exists():
            frozen=json.loads(anchor_path.read_text())
            if frozen['manifest_sha256']!=anchor.manifest_sha256 or frozen['payload_bytes']!=actualbytes: raise ValueError('B* already immutable; explicit amendment required')
        else: _write(anchor_path,anchor_receipt)
        report['anchor']=anchor_receipt;print(json.dumps({'anchor':anchor_receipt}),flush=True);recorder.snapshot()
        shutil.rmtree(Path(anchor_staging.directory).parent)
    # Fresh current-code floating input gives every candidate its actual runtime bytes.
    dense_ref=write_bundle(source,out/'floating',runtime)
    del source;gc.collect();mx.clear_cache()
    native_teacher=load_bundle(Path(dense_ref.directory),backend='mlx')
    batches=[];windows=[]
    for r in open_records(dev,purpose='selection'):
        window=_prepared_window(r,config.preparation.seed)
        if len(window.token_ids)!=512: raise ValueError('primary dev window incomplete')
        batches.append(_native_batch([window.token_ids]));windows.append({'document_id':r.document_id,'source_start':window.source_start,'targets':window.target_indices,'tokens_sha256':hashlib.sha256(_canonical({'ids':window.token_ids})).hexdigest()})
    _write(out/'dev-windows.json',{'windows':windows,'policy':'existing prepared primary512 WindowPolicy first window, positions reset0, no padding'})
    report['dev_windows_sha256']=_sha(out/'dev-windows.json')
    teacher_hidden=[]
    for batch in batches:
        recorder.account(teacher_forward_calls=1)
        teacher_hidden.append(native_teacher.final_hidden(batch))
    report['native_numerical_controls']=_native_numerical_controls(native_teacher,teacher_hidden[0])
    report['teacher_identity']=_quality(native_teacher,native_teacher,batches,recorder,teacher_hidden)
    report['evaluation_contract']=_evaluation_contract(ROOT,Path(dense_ref.directory),report['dev_windows_sha256'])
    report['rows']=[]
    reused=[]
    for previous in sorted((ROOT/'artifacts/quantization').glob('attempt-*/summary.json')):
        if previous.parent==out: continue
        evidence=json.loads(previous.read_text())
        for copy in _reusable_rows(evidence,previous,report['evaluation_contract']):
            reused=[r for r in reused if r['variant']!=copy['variant']]+[copy]
    report['rows'].extend(reused)
    # Predeclared dense frontier; all actual points retained, never discard a
    # working dense method because a shared learned adapter is unavailable.
    smoke_requests=[('smoke',m,4,64) for m in ('gptq','awq','dwq','dynamic')]
    frontier_requests=[('frontier','rtn',b,g) for b,g in ((8,64),(4,64),(3,64),(2,64),(4,32),(4,128))]+[('frontier',m,b,64) for b in (4,2) for m in ('gptq','awq','dwq','dynamic')]
    frontier_requests=[r for r in frontier_requests if not any(old['variant']==f'frontier-{r[1]}-b{r[2]}-g{r[3]}' for old in reused)]
    frontier_requests.append(('frontier','dynamic',2,64))
    smoke_requests=[r for r in smoke_requests if not any(old['variant']==f'smoke-{r[1]}-b{r[2]}-g{r[3]}' for old in reused)]
    frontier_requests=[r for r in frontier_requests if r[1]!='dynamic' or r[2]!=2 or not any(old['variant'].startswith('frontier-dynamic-b2-g64-target') for old in reused)]
    requests=smoke_requests+frontier_requests+[('embedding',m,2,64) for m in ('rtn','gptq','awq','dwq') if not any(old['variant']==f'embedding-{m}-b2-g64-embed4' for old in reused)]
    dynamic_low_seen=0
    forecast_checked=False
    for phase,method,bits,group in requests:
        if phase in ('frontier','embedding') and not forecast_checked:
            smoke=[r for r in report['rows'] if r['phase']=='smoke' and r['status']=='operation_smoke_only']
            learned=sum(r['conversion_seconds'] for r in smoke if r['method']!='rtn')
            dev_seconds=sum(r['construction_and_dev_seconds']-r['conversion_seconds'] for r in smoke)
            forecast=32*learned+24*dev_seconds+60
            report['frontier_forecast']={'conservative_remaining_seconds':forecast,'measured_basis':'16x each2-sequence learned conversion times2 precisions;24x five4-dev evaluation/load/parity times;60sec fixed allowance','actual_spent_seconds':time.monotonic()-recorder.started,'smoke_methods_working':len(smoke)}
            recorder.snapshot();forecast_checked=True
            if forecast+recorder.prior['wall_seconds']+time.monotonic()-recorder.started>recorder.caps['wall_seconds']:
                report['frontier_status']='not_run_resource_forecast_exceeds_aggregate_wall_cap';break
        if phase in ('frontier','embedding') and method!='rtn' and not any(r['phase']=='smoke' and r['method']==method and r['status']=='operation_smoke_only' for r in report['rows']):
            report['rows'].append({'phase':phase,'variant':f'frontier-{method}-b{bits}-g{group}','method':method,'bits':bits,'group_size':group,'status':'not_run_smoke_failed','reason':'larger learned qualification requires a successful real primary512 smoke and measured forecast'})
            continue
        dynamic_target=6.
        if method=='dynamic' and bits==2:
            dynamic_target=3.5 if dynamic_low_seen==0 else 3.6
            dynamic_low_seen+=1
        key=f'{phase}-{method}-b{bits}-g{group}'+(f'-target{dynamic_target}' if method=='dynamic' and bits==2 else '-embed4' if phase=='embedding' else '')
        recorder.account(conversion_attempts=1)
        begin=time.monotonic();row={'phase':phase,'variant':key,'method':method,'bits':bits,'group_size':group,'status':'running'}
        report['rows'].append(row);recorder.snapshot();print(json.dumps({'starting':key}),flush=True)
        try:
            spec=QuantizationSpec(method,bits,group,str(out/key),calibration_count=2 if phase=='smoke' else 32,length=512,seed=17,internal_validation_count=2 if phase=='smoke' else 8,dynamic_high_bits=4 if method=='dynamic' and bits==2 else 8,dynamic_target_bpw=dynamic_target,embedding_bits=4 if phase=='embedding' else None)
            supplied_fit=_restrict(fit,fit.records[:2]+fit.records[32:34]) if phase=='smoke' else fit
            # Explicit source query and gradient/update reservations include each
            # repeated fit presentation. Per-layer AWQ searches are not updates.
            result=quantize_rtn(dense_ref,spec,recorder) if method=='rtn' else quantize_dense(dense_ref,spec,supplied_fit,recorder)
            row.update(status='converted',artifact=result.bundle.directory,manifest_sha256=result.bundle.manifest_sha256,payload_bytes=result.payload_bytes,conversion_seconds=time.monotonic()-begin,at_or_below_anchor=result.payload_bytes<=actualbytes)
            candidate=load_bundle(Path(result.bundle.directory),backend='mlx')
            evidence_path=Path(spec.destination+'.quantizer-evidence.json')
            if evidence_path.exists():
                row['quantizer_evidence']=json.loads(evidence_path.read_text());row['quantizer_evidence_sha256']=_sha(evidence_path)
            row['resident_bytes']=measure_resident(candidate).resident_bytes
            row['development']=_quality(native_teacher,candidate,batches[:4] if phase=='smoke' else batches,recorder,teacher_hidden[:4] if phase=='smoke' else teacher_hidden)
            if row['development']['kl_nats_per_target'] < -report['native_numerical_controls']['negative_kl_floor_per_target']: raise RuntimeError('native KL exceeds measured negative floor; unresolved fidelity')
            # Common runtime packed full/cache parity at nonzero original positions.
            recorder.account(student_forward_calls=3)
            test=batches[0];full=candidate.logits(test);mx.eval(full)
            cache=candidate.new_cache();parts=[]
            for lo,hi in ((0,7),(7,512)):
                part=replace(test,input_ids=test.input_ids[:,lo:hi],attention_mask=test.attention_mask[:,lo:hi],target_mask=mx.ones((1,hi-lo-1),dtype=mx.bool_),positions=test.positions[:,lo:hi])
                parts.append(candidate.logits(part,cache=cache))
            cache_error=mx.max(mx.abs(mx.concatenate(parts,axis=1)-full)).item()
            if cache_error>.005: raise RuntimeError(f'packed full/cache parity unresolved: {cache_error}')
            row['full_cache_max_abs_error']=cache_error;row['status']='operation_smoke_only' if phase=='smoke' else 'eligible';row['construction_and_dev_seconds']=time.monotonic()-begin
            del candidate,full,parts,cache
        except (ValueError,RuntimeError,ImportError) as exc:
            row.update(status=_failure_status(exc),reason=str(exc),construction_and_dev_seconds=time.monotonic()-begin)
            if recorder.error: raise
        gc.collect();mx.clear_cache();recorder.snapshot();print(json.dumps(row),flush=True)
    feasible=[r for r in report['rows'] if r['status']=='eligible' and r['at_or_below_anchor']]
    report['baseline_star']=min(feasible,key=lambda r:(r['development']['kl_nats_per_target'],r['payload_bytes'],r['construction_and_dev_seconds'],r['manifest_sha256']))['variant'] if feasible else None
    report['sv_feasibility']=sv_feasibility([(1536,576)]*2,64)
    report['shared_learned_status']={'gptq':'not_yet_implemented pooled canonical-use adapter','awq':'not_yet_implemented common scaling proof/adapter','dwq':'not_yet_implemented shared refinement; dense independently exercised'}
    report['wp03']='passed_format_anchor_and_dense_pilot' if feasible else 'format_anchor_passed_no_eligible_dense_point_under_Bstar'
    report['status']='passed'


def _prior_costs(root,fixture_wall):
    prior={'wall_seconds':fixture_wall}
    for p in root.glob('attempt-*/summary.json'):
        old=json.loads(p.read_text());costs=dict(old.get('costs',{}))
        ledger=old.get('cost_ledger')
        if ledger is None and old.get('run_record'):
            ledger=str(Path(old['run_record']).parents[2]/'costs.jsonl')
        if ledger and Path(ledger).exists():
            rows=[json.loads(line) for line in Path(ledger).read_text().splitlines()]
            rows=[r['cost'] for r in rows if r.get('attempt_id')==old.get('attempt_id')]
            for key in ('wall_seconds','teacher_forward_calls','student_forward_calls','student_backward_calls','updates'):
                value=sum(r.get(key,0) for r in rows)
                target='fit_updates' if key=='updates' else key
                costs[target]=max(costs.get(target,0),value)
        for key,value in costs.items(): prior[key]=prior.get(key,0)+value
    amendments=root/'cost-amendments.json'
    if amendments.exists():
        for entry in json.loads(amendments.read_text())['amendments']:
            for key,value in entry['cost'].items(): prior[key]=prior.get(key,0)+value
    return prior


def run_format_and_quant(config):
    from poc.config import validate_config
    import psutil
    validate_config(config,command='validate')
    if config.phase!='exploratory' or config.teacher.repository!='HuggingFaceTB/SmolLM2-135M' or config.training.context_tokens!=512: raise ValueError('Task9 pilot is explicit primary135m exploratory512 only')
    policy=json.loads((ROOT/'protocol/quantization-pilot-caps.json').read_text());caps=policy['aggregate_caps']
    root=ROOT/'artifacts/quantization';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.compute.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('one conversion/compute job already active')
    prior=_prior_costs(root,policy['synthetic_qualification']['prior_observed_wall_seconds'])
    out=root/('attempt-'+uuid4().hex);out.mkdir()
    report={'schema_version':1,'status':'preflight','scope':'Task9 actual native format/dense quantization pilot; no protected evaluation or compression/training-success claim','caps':caps,'prior_costs':prior,'available_memory_before_bytes':psutil.virtual_memory().available,'disk_free_before_bytes':shutil.disk_usage(root).free,
            'source_sha256':{str(p.relative_to(ROOT)):_sha(p) for p in sorted((ROOT/'poc').rglob('*.py'))},'source_transfers_bytes':0,'source_commit':os.popen('git rev-parse HEAD').read().strip()}
    # Immutable attempt identity and the established append-only run cost ledger.
    budget=Budget(updates=caps['fit_updates'],valid_target_presentations=131072,teacher_forward_calls=caps['teacher_forward_calls'],suffix_backward_calls=0,student_backward_calls=caps['student_backward_calls'],search_proposals=0,search_seconds=0,wall_seconds=caps['wall_seconds'],disk_bytes=caps['generated_disk_bytes'],memory_bytes=caps['process_rss_bytes'])
    spec=RunSpec('Q','Task9-format-and-quant-pilot',{'policy':policy},report['source_commit'],hashlib.sha256(_canonical(report['source_sha256'])).hexdigest(),_sha(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),{'pilot_caps':_sha(ROOT/'protocol/quantization-pilot-caps.json')},{'backend':'mlx_metal','one_compute_job':True},{'primary':17},(),budget)
    attempt=start_attempt(spec,root/'runs');tracker=BudgetTracker(budget,attempt)
    report['run_id']=attempt.run_id;report['attempt_id']=attempt.attempt_id;report['run_record']=str(attempt.path/'run.json');report['cost_ledger']=str(tracker.path)
    append_event(attempt,{'kind':'stage_started','output':str(out),'prior_aggregate_costs':prior})
    recorder=PilotRecorder(out,report,prior,caps,tracker);recorder.snapshot()
    if report['available_memory_before_bytes']<caps['process_rss_bytes'] or report['disk_free_before_bytes']<caps['generated_disk_bytes']: raise BudgetExceeded('predeclared aggregate memory/disk headroom unavailable')
    print(json.dumps({'preflight':caps,'prior_costs':prior,'memory_available':report['available_memory_before_bytes'],'out':str(out)}),flush=True)
    def interrupt(signum,frame): raise PilotInterrupted('finite resource cap/interruption; attempt retained')
    previous=signal.signal(signal.SIGTERM,interrupt)
    thread=threading.Thread(target=recorder.monitor,daemon=True);thread.start()
    try:
        _run(config,out,report,recorder,policy)
        return 0
    except BaseException as exc:
        report.update(status='failed',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        recorder.stop.set();thread.join();signal.signal(signal.SIGTERM,previous)
        recorder.peaks['generated_disk_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())+sum(p.stat().st_size for p in (ROOT/'artifacts/payload-anchors').rglob('*') if p.is_file())
        remaining=time.monotonic()-recorder.started-(tracker.totals.wall_seconds-recorder.baseline.wall_seconds)
        try:
            recorder.record(CostRecord(wall_seconds=max(0,remaining),disk_bytes=recorder.peaks['generated_disk_bytes'],memory_bytes=recorder.peaks['process_rss_bytes'],memory_unavailable_reason=None))
        except BudgetExceeded as exc:
            report.update(status='failed',reason=f'BudgetExceeded: {exc}')
            raise
        finally:
            append_event(attempt,{'kind':'stage_finished','status':report['status'],'summary':str(out/'summary.json')})
            recorder.snapshot();print(json.dumps({'status':report['status'],'report':str(out/'summary.json'),'costs':report['costs'],'peaks':report['peaks']}),flush=True)
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

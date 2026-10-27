"""Finite WP02 qualification: local original teacher and predeclared dev slices.

This is qualification, not a final training/search budget or protected evaluation.
The machine-readable plan is persisted before the first development prediction.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import torch

from poc.contracts import Batch
from poc.evaluate import (ARCItem, BlimpItem, DocumentMetric, PromptPolicy, GenerationPolicy,
    aggregate_documents, aggregate_tasks, evaluate_documents, evaluate_synthetic,
    score_arc, score_blimp, synthetic_cases, copy_guard_cases)
from poc.losses import kl_sum, nll_sum
from poc.runtime.torch_decoder import TorchDecoder
from poc.teacher import AssetManifest, load_reference, load_tokenizer, sha256_file

ROOT=Path(__file__).resolve().parents[1]
GiB=1024**3


def _write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'); temp.replace(path)


def _batch(ids, *, device='cpu', offset=0, document_id='qualification', cluster_id='qualification', targets=None):
    tensor=torch.tensor([ids],dtype=torch.int64,device=device)
    mask=torch.ones_like(tensor,dtype=torch.bool)
    target=torch.ones((1,len(ids)-1),dtype=torch.bool,device=device)
    if targets is not None:
        target.zero_()
        for index in targets: target[0,index-1]=True
    return Batch(tensor,mask,target,torch.arange(offset,offset+len(ids),device=device)[None],(document_id,),(cluster_id,))


def _sync(device):
    if device=='mps': torch.mps.synchronize()


def qualify_numerics(reference, *, device: str, lengths=(16,128,512)) -> dict:
    """Actual-shape HF FP32 parity; true FP64 norm/RoPE reference; finite costs."""
    import psutil
    import transformers
    import inspect
    from transformers.models.llama import modeling_llama
    if device not in {'cpu','mps'}: raise ValueError('CPU or explicitly available MPS required')
    if device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('explicit MPS unavailable; no silent fallback')
    started=time.perf_counter()
    cpu=TorchDecoder.from_transformers(reference)
    # One CPU FP64 reference copy; scientific CPU modules share original owners.
    ref64=deepcopy(cpu).double()
    scientific=cpu if device=='cpu' else deepcopy(cpu).to(device='mps',dtype=torch.float32)
    config=reference.config
    result=dict(source_sha256={'poc/runtime/torch_decoder.py':sha256_file(ROOT/'poc/runtime/torch_decoder.py'),
                              'installed_transformers_modeling_llama.py':sha256_file(Path(inspect.getfile(modeling_llama)))},
        software=dict(torch=torch.__version__,transformers=transformers.__version__),
        unique_parameters=sum(p.numel() for p in cpu.parameters()),
        config=config.to_dict(),tied=cpu.lm_head.weight is cpu.embed_tokens.weight,
        backend=f'torch_{device}',dtype='float32',cpu_reference_dtype='float64',
        numeric_profile='true FP64 RMS/RoPE/softmax; original HF FP32 compatibility',
        hf_fp32_max_abs_error=0.,fp64_logit_max_abs_error=0.,nll_per_token_max_abs_error=0.,
        identity_kl_per_token=0.,near_identity_controls=[],forward_costs=[],suffix_costs=[],
        available_memory_before_bytes=psutil.virtual_memory().available,
        memory_note='RSS high-water includes imports, original CPU FP32, CPU FP64 and scientific copy; not isolated training peak')
    # Profile baseline original digest and warm signature path separately.
    cpu.invalidate_state_snapshot(); start=time.perf_counter(); cpu._digest()
    result['digest_full_seconds']=time.perf_counter()-start
    start=time.perf_counter(); cpu._digest();result['digest_snapshot_seconds']=time.perf_counter()-start
    for length in lengths:
        ids=tuple((i*37+11)%config.vocab_size for i in range(length))
        b=_batch(ids); bm=_batch(ids,device=device)
        with torch.no_grad():
            start=time.perf_counter()
            original=reference(input_ids=b.input_ids,attention_mask=b.attention_mask,
                               position_ids=b.positions,use_cache=False).logits
            adapter=cpu.logits(b)
            hf_error=float((original-adapter).abs().max())
            result['hf_fp32_max_abs_error']=max(result['hf_fp32_max_abs_error'],hf_error)
            if hf_error!=0: raise RuntimeError(f'original HF FP32 mismatch: {hf_error}')
            del original,adapter
            h64=ref64.final_hidden(b)
            _sync(device); start=time.perf_counter(); hidden=scientific.final_hidden(bm); _sync(device)
            forward_seconds=time.perf_counter()-start
            error=0.;nll64=0.;nll32=0.;near=defaultdict(float)
            for lo in range(0,length-1,32):
                hi=min(lo+32,length-1)
                logits64=ref64.readout(h64[:,lo:hi]); logits=scientific.readout(hidden[:,lo:hi])
                error=max(error,float((logits.cpu().double()-logits64).abs().max()))
                mask=bm.target_mask[:,lo:hi]; targets=bm.input_ids[:,lo+1:hi+1]
                nll32+=float(nll_sum(logits,targets,mask).item())
                nll64+=float(nll_sum(logits64,b.input_ids[:,lo+1:hi+1],b.target_mask[:,lo:hi]).item())
                identity=float(kl_sum(logits,logits,mask).item())
                if identity!=0: raise RuntimeError('identity KL is nonzero')
                for epsilon in (1e-6,1e-4,1e-2):
                    candidate=logits.clone(); candidate[...,0]+=epsilon
                    candidate64=logits64.clone();candidate64[...,0]+=epsilon
                    near[(epsilon,'scientific')]+=float(kl_sum(logits,candidate,mask).item())
                    near[(epsilon,'fp64')]+=float(kl_sum(logits64,candidate64,b.target_mask[:,lo:hi]).item())
            n=length-1
            result['fp64_logit_max_abs_error']=max(result['fp64_logit_max_abs_error'],error)
            result['nll_per_token_max_abs_error']=max(result['nll_per_token_max_abs_error'],abs(nll32-nll64)/n)
            result['forward_costs'].append(dict(context_tokens=length,valid_targets=n,seconds=forward_seconds,
                max_abs_fp64_logit_error=error,scientific_nll_sum=nll32,fp64_nll_sum=nll64,
                rss_bytes=psutil.Process().memory_info().rss,
                mps_allocated_bytes=torch.mps.current_allocated_memory() if device=='mps' else None,
                mps_driver_bytes=torch.mps.driver_allocated_memory() if device=='mps' else None))
            for epsilon in (1e-6,1e-4,1e-2):
                result['near_identity_controls'].append(dict(context_tokens=length,epsilon=epsilon,
                    raw_kl_per_token=near[(epsilon,'scientific')]/n,fp64_kl_per_token=near[(epsilon,'fp64')]/n))
            del h64,hidden
    # Cache, causal edits, and differentiable early/middle/late input gradients on
    # a short actual-shape workload, bounded independently of max context 8192.
    ids=tuple((i*37+11)%config.vocab_size for i in range(min(16,min(lengths))))
    b=_batch(ids);bm=_batch(ids,device=device)
    with torch.no_grad():
        full=scientific.logits(bm); cache=scientific.new_cache(); parts=[]
        for lo,hi in ((0,len(ids)//2),(len(ids)//2,len(ids))):
            parts.append(scientific.logits(_batch(ids[lo:hi],device=device,offset=lo),cache=cache))
        result['cache_max_abs_error']=float((torch.cat(parts,1)-full).abs().max().item())
        changed=ids[:-2]+((ids[-2]+3)%config.vocab_size,(ids[-1]+7)%config.vocab_size)
        edited=scientific.logits(_batch(changed,device=device))
        result['future_edit_past_max_abs_error']=float((edited[:,:-2]-full[:,:-2]).abs().max().item())
        if result['future_edit_past_max_abs_error']!=0: raise RuntimeError('future edit affected past')
        del full,parts,edited,cache
    for depth in sorted({0,len(cpu.layers)//2,len(cpu.layers)-1}):
        gradients=[];cost={}
        for label,model,batch in [('cpu_fp64',ref64,b),('scientific',scientific,bm)]:
            _sync('mps' if model.device.type=='mps' else 'cpu');start=time.perf_counter()
            with torch.no_grad(): bound=model.prefix_boundary(batch,depth)
            z=model.ff_residual(bound).detach().requires_grad_(True)
            _sync('mps' if model.device.type=='mps' else 'cpu')
            prefix_seconds=time.perf_counter()-start; suffix_start=time.perf_counter()
            logits=model.readout(model.suffix_hidden(z,bound)[:,:4])
            work=logits if logits.dtype==torch.float64 else logits.float()
            loss=-work.log_softmax(-1).gather(-1,batch.input_ids[:,1:5,None]).mean()
            gradient=torch.autograd.grad(loss,z)[0]
            _sync('mps' if model.device.type=='mps' else 'cpu')
            gradients.append(gradient.detach().cpu().double())
            cost[label+'_seconds']=time.perf_counter()-start
            cost[label+'_prefix_seconds']=prefix_seconds
            cost[label+'_suffix_backward_seconds']=time.perf_counter()-suffix_start
            del bound,z,loss,gradient,logits,work
        cost.update(depth=depth,context_tokens=len(ids),input_gradient_max_abs_error=float((gradients[0]-gradients[1]).abs().max()))
        result['suffix_costs'].append(cost)
    # Measured envelope: four times worst local near-identity discrepancy. Raw
    # negatives remain in controls/results; downstream consumers reject beyond it.
    discrepancies=[abs(r['raw_kl_per_token']-r['fp64_kl_per_token']) for r in result['near_identity_controls']]
    negatives=[max(0.,-r['raw_kl_per_token']) for r in result['near_identity_controls']]
    result['negative_kl_floor_per_token']=4*max([1e-12,*discrepancies,*negatives])
    result['local_envelopes']=dict(logit_atol=4*max(result['fp64_logit_max_abs_error'],1e-12),
        gradient_atol=4*max(r['input_gradient_max_abs_error'] for r in result['suffix_costs']),
        applicability='this actual checkpoint/profile and tested finite shapes; not every candidate or perturbation')
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['wall_seconds']=time.perf_counter()-started
    return result


def _restricted(manifest, records):
    ids={r.document_id for r in records}
    return replace(manifest,records=tuple(records),roles={i:manifest.roles[i] for i in ids},
        membership={i:manifest.membership[i] for i in ids},quarantine={i:v for i,v in manifest.quarantine.items() if i in ids})


def _run_teacher_validation(*, config_path=ROOT/'configs/135m.yaml', device='mps', out=ROOT/'artifacts/validation/teacher_and_harness') -> int:
    from poc.config import load_config
    from poc.data.manifest import load_manifest
    from poc.data.access import open_records
    from poc.data.windows import TokenDocument, WindowPolicy, make_windows, rolling_windows
    import psutil
    started=time.perf_counter();out=Path(out)
    sources={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'poc').rglob('*.py'))}
    config=load_config(config_path,phase="exploratory")
    preparation=config.preparation.output_dir
    receipt=json.loads((ROOT/'provenance/preparation-135m-2026-10-27.json').read_text())
    # Source and preparation gate passed locally before teacher use. WP01 is an
    # asset-free prerequisite, separate from historical WP00 doctor status.
    component=json.loads((ROOT/'artifacts/validation/component.json').read_text())
    if component['status']!='passed' or receipt['source_gate_status']!='passed_local_preparation':
        raise RuntimeError('WP01 component/public source gate prerequisite unmet')
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS qualification requires explicit native execution; no CPU fallback')
    # Metadata only. Exclude ARC train/test and WikiText test BEFORE text reader.
    manifest=load_manifest(preparation/'data-manifest.json',expected_sha256=receipt['manifests']['data-manifest.json']['sha256'],roles={'dev'})
    populations=defaultdict(list)
    for r in manifest.records:
        if r.source=='HuggingFaceFW/fineweb-edu': populations['fineweb'].append(r)
        elif r.source=='Salesforce/wikitext' and r.metadata['official_split']=='validation': populations['wikitext_validation'].append(r)
        elif r.source=='allenai/ai2_arc' and r.metadata['official_split']=='validation': populations[r.metadata['config']].append(r)
        elif r.source=='alexwarstadt/blimp': populations['blimp:'+r.metadata['paradigm']].append(r)
    selected=[]; smoke_ids=[]
    for group,rows in sorted(populations.items()):
        ordered=sorted(rows,key=lambda r:hashlib.sha256(('wp02-v1:'+r.document_id).encode()).digest())
        smoke_count=2 if group.startswith('blimp:') else 32 if group.startswith('ARC-') else 8 if group=='fineweb' else 4
        count=20 if group.startswith('blimp:') else len(rows) if group.startswith('ARC-') else 32 if group=='fineweb' else 8
        selected.extend(ordered[:count]); smoke_ids.extend(r.document_id for r in ordered[:smoke_count])
    manifest=_restricted(manifest,selected)
    asset=AssetManifest.load(preparation/'teacher-assets.json',expected_sha256=receipt['teacher_asset_manifest_sha256'])
    tokenizer=load_tokenizer(asset)
    policy=PromptPolicy(tokenizer,context_tokens=512,tokenizer_digest=asset.expected_sha256)
    cases=synthetic_cases(seed=17,split='dev'); guards=copy_guard_cases(seed=17)
    plan=dict(schema_version=1,scope='finite exploratory WP02 qualification subset; not full published benchmarks',
        selection='outcome-blind sha256(wp02-v1:document_id) within groups',
        population_counts={k:len(v) for k,v in populations.items()},
        selected_ids=[r.document_id for r in selected],smoke_ids=smoke_ids,
        competence_screen='all ARC official validation, 20 BLiMP dev pairs per paradigm, 32 FineWeb dev, 8 WikiText validation',synthetic_cases=[asdict(c) for c in cases],copy_guards=[asdict(c) for c in guards],
        copy_guard_policy='contexts128/512; full payload and output token guard; no truncation; exact labels',
        context_lengths=[16,128,512],policy_digest=policy.digest,
        prompt_policy=dict(arc='Question: {question}\nAnswer: {full answer}',arc_eos=False,
            sentence_bos=False,sentence_eos=True,sentence_first_token='conditioning only',
            generation='greedy, first newline after nonwhitespace content or EOS; max 32 new tokens',
            no_truncation=True,reliability_bins=[i/10 for i in range(11)]),
        source_sha256=sources,asset_manifest_sha256=asset.expected_sha256,
        data_manifest_sha256=receipt['manifests']['data-manifest.json']['sha256'])
    _write(out/'qualification-plan.json',plan)
    # Durable exposure intent before text I/O; completed fingerprints before yield.
    records=list(open_records(manifest,purpose='selection'))
    reference=load_reference(asset,config)
    numeric=qualify_numerics(reference,device=device)
    if numeric['unique_parameters']!=134515008 or not numeric['tied']:
        raise RuntimeError('actual primary checkpoint census/tie mismatch')
    required=dict(hidden_size=576,intermediate_size=1536,num_hidden_layers=30,num_attention_heads=9,
                  num_key_value_heads=3,max_position_embeddings=8192,rms_norm_eps=1e-5,vocab_size=49152)
    if any(getattr(reference.config,k)!=v for k,v in required.items()) or TorchDecoder.from_transformers(reference).rope_theta!=100000:
        raise RuntimeError('actual pinned primary model configuration mismatch')
    _write(out/'numerics.json',numeric)
    model=TorchDecoder.from_transformers(reference,tokenizer_digest=asset.expected_sha256).to(device=device)
    del reference
    # Predeclared secondary context memory guard, calibrated only after primary.
    available=psutil.virtual_memory().available
    secondary=[]
    for length in (2048,8192):
        score_bytes=9*length*length*4
        if 8*score_bytes+2*GiB>available/2:
            secondary.append(dict(context_tokens=length,status='not_run',reason='conservative measured-available-memory guard',
                available_bytes=available,estimated_attention_working_bytes=8*score_bytes))
            continue
        ids=tuple((i*37+11)%model.config.vocab_size for i in range(length))
        _sync(device);start=time.perf_counter()
        with torch.no_grad(): model.final_hidden(_batch(ids,device=device))
        _sync(device)
        secondary.append(dict(context_tokens=length,status='passed_forward_only',seconds=time.perf_counter()-start))
    arc=[];blimp=[];naturals=[]
    for r in records:
        if r.source=='allenai/ai2_arc':
            raw=json.loads(r.text);labels=raw['choices']['label'];key=raw['answerKey']
            # Upstream labels are strings; retain original option order.
            label=labels.index(str(key))
            arc.append(ARCItem(r.document_id,raw['question'],tuple(raw['choices']['text']),label,r.metadata['config']))
        elif r.source=='alexwarstadt/blimp':
            raw=json.loads(r.text);blimp.append(BlimpItem(r.document_id,raw['sentence_good'],raw['sentence_bad'],r.metadata['paradigm']))
        else: naturals.append(r)
    resource_check=dict(available_memory_bytes=psutil.virtual_memory().available,
        measured_primary_forward_seconds=numeric['forward_costs'][-1]['seconds'],
        maximum_task_full_or_incremental_calls=sum(len(i.choices) for i in arc)+2*len(blimp)+32*len(cases),
        finite_workload='all frozen item IDs; no correctness-based reductions',
        note='forward projection is operational qualification planning, not a final search/training budget')
    resource_check['conservative_task_forward_projection_seconds']=(resource_check['maximum_task_full_or_incremental_calls']
        *resource_check['measured_primary_forward_seconds'])
    _write(out/'qualification-resource-check.json',resource_check)
    if resource_check['available_memory_bytes']<2*GiB: raise RuntimeError('insufficient measured memory for frozen development screen')
    arc_rows=score_arc(model,arc,policy);_write(out/'arc-rows.json',[asdict(r) for r in arc_rows])
    blimp_rows=score_blimp(model,blimp,policy);_write(out/'blimp-rows.json',[asdict(r) for r in blimp_rows])
    synthetic_rows=evaluate_synthetic(model,cases,GenerationPolicy(policy,max_new_tokens=32))
    _write(out/'synthetic-rows.json',[asdict(r) for r in synthetic_rows])
    guard_rows=[]
    for context in (128,512):
        for case in guards:
            output_tokens=len(tokenizer(case.answer,add_special_tokens=False)['input_ids'])+4
            guard_policy=GenerationPolicy(replace(policy,context_tokens=context),max_new_tokens=output_tokens)
            row=evaluate_synthetic(model,[case],guard_policy)[0]
            guard_rows.append(dict(context_tokens=context,payload_words=int(case.stratum),
                payload_token_count=output_tokens-4,**asdict(row)))
    _write(out/'copy-guard-rows.json',guard_rows)
    # Verified metadata windows are role-filtered by document identity before use.
    windows_path=preparation/'windows.json'
    if sha256_file(windows_path)!=receipt['manifests']['windows.json']['sha256']:
        raise ValueError('window manifest hash mismatch')
    # Hash the complete immutable window manifest without decoding protected IDs.
    # Reconstruct only selected, exposed dev windows from their verified token
    # fields using the exact saved preparation policy (seed17, 512/256 stride).
    natural_rows=[]
    for r in naturals:
        if sha256_file(Path(r.token_path))!=r.token_sha256:
            raise ValueError('selected dev token file hash mismatch')
        token_data=json.loads(Path(r.token_path).read_text())
        if token_data['document_id']!=r.document_id: raise ValueError('token document mismatch')
        document=TokenDocument(r.document_id,tuple(token_data['fields']['text']))
        windows=(make_windows(document,WindowPolicy(seed=config.preparation.seed))
                 if r.source=='HuggingFaceFW/fineweb-edu' else rolling_windows(document))
        for w in windows:
            batch=_batch(w.token_ids,device=device,document_id=r.document_id,
                cluster_id=manifest.membership[r.document_id],targets=w.target_indices)
            row=next(iter(evaluate_documents(model,model,[batch],negative_kl_floor=numeric['negative_kl_floor_per_token'])))
            natural_rows.append(replace(row,window_id=f'{r.document_id}:{w.source_start}',source=r.source))
    _write(out/'natural-rows.json',[asdict(r) for r in natural_rows])
    summary=dict(schema_version=1,gate='WP02',status='passed',timestamp=datetime.now(timezone.utc).isoformat(),
        scope=plan['scope'],backend=f'torch_{device}',dtype='float32',
        plan_sha256=sha256_file(out/'qualification-plan.json'),numerics_sha256=sha256_file(out/'numerics.json'),
        source_sha256=sources,config_sha256=sha256_file(config_path),
        assets_sha256=asset.expected_sha256,secondary_contexts=secondary,
        arc=aggregate_tasks(arc_rows),blimp=aggregate_tasks(blimp_rows),
        smoke=dict(arc=aggregate_tasks([r for r in arc_rows if r.item_id in smoke_ids]),
                   blimp=aggregate_tasks([r for r in blimp_rows if r.item_id in smoke_ids]),
                   natural={source:aggregate_documents([r for r in natural_rows if r.source==source and r.document_id in smoke_ids])
                            for source in sorted({r.source for r in natural_rows})}),
        copy_guards=guard_rows,
        synthetic={family:aggregate_tasks([r for r in synthetic_rows if r.task==family]) for family in sorted({r.task for r in synthetic_rows})},
        natural={source:aggregate_documents([r for r in natural_rows if r.source==source]) for source in sorted({r.source for r in natural_rows})},
        actual=dict(wall_seconds=time.perf_counter()-started,
            task_teacher_forward_calls=sum(len(r.token_counts) for r in arc_rows if r.status=='ok')
                +sum(len(r.token_counts) for r in blimp_rows if r.status=='ok')
                +sum(r.token_counts[0] for r in synthetic_rows if r.status=='ok')
                +sum(r['token_counts'][0] for r in guard_rows if r['status']=='ok'),
            natural_teacher_forward_calls=2*len(natural_rows),
            qualification_full_forward_calls=4*len(plan['context_lengths'])+4,
            qualification_prefix_calls=2*len(numeric['suffix_costs']),
            qualification_suffix_backward_calls=2*len(numeric['suffix_costs']),
            secondary_forward_calls=sum(r['status']=='passed_forward_only' for r in secondary),
            task_items=len(arc_rows)+len(blimp_rows)+len(synthetic_rows),
            natural_windows=len(natural_rows),unique_natural_documents=len(naturals),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            available_memory_after_bytes=psutil.virtual_memory().available),
        limitations=['finite development subsets only; unknown teacher pretraining contamination',
            'no candidate compression, final training/search budgets or protected evaluation authorized',
            'local numeric envelope only; secondary forwards do not qualify backward memory',
            'natural fidelity is identity control; NLL measures teacher competence'])
    if sources!={name:sha256_file(ROOT/name) for name in sources}: raise RuntimeError('sources changed during qualification')
    summary['artifact_sha256']={p.name:sha256_file(p) for p in sorted(out.glob('*.json')) if p.name!='summary.json'}
    _write(out/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],gate='WP02',backend=summary['backend'],
        arc=summary['arc']['accuracy'],blimp_macro=summary['blimp']['macro_accuracy'],record=str(out/'summary.json'))),flush=True)
    return 0


def run_teacher_validation(*, config_path=ROOT/'configs/135m.yaml', device='mps', out=ROOT/'artifacts/validation/teacher_and_harness') -> int:
    """Persist a failed WP02 result on any integrity/runtime exception."""
    out=Path(out)
    # Never leave an earlier successful summary masquerading as this attempt.
    _write(out/'summary.json',dict(schema_version=1,gate='WP02',status='running',device=device))
    try:
        return _run_teacher_validation(config_path=config_path,device=device,out=out)
    except Exception as exc:
        _write(out/'summary.json',dict(schema_version=1,gate='WP02',status='failed',device=device,
            error_type=type(exc).__name__,reason=str(exc),timestamp=datetime.now(timezone.utc).isoformat()))
        raise

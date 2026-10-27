"""Bounded Task10 operation proof; no candidate advantage or protected outcomes.

Run costcheck first, resolve finite main caps, then run pilot against its saved
outcome-blind plan. All source text/tokens, probe rows and failed work remain in
ignored artifacts; only the small summary receipt is intended for versioning.
"""
from collections import Counter
from dataclasses import asdict, fields, replace
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import psutil
import torch

from poc.budget import BudgetExceeded, BudgetTracker
from poc.config import load_config
from poc.contracts import Budget, CostRecord, ProbeRecord, RunSpec
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.probes import (PathologyObservation, ProbePolicy, ProbePool, ProbeCache,
    apply_probe, fit_pathology_thresholds, filter_pathologies, generate_donors,
    local_discrepancy, mine_probes, prefix_for_probe)
from poc.records import RecordError, append_event, canonical_json, locked_ledger, read_rows, start_attempt
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest, load_teacher, load_tokenizer, sha256_file
from poc.teacher_validation import _batch, _restricted, _write

ROOT=Path(__file__).resolve().parents[1]
GiB=1024**3


class PilotRecorder:
    """Shared durable counters plus explicit student-query and live memory caps."""
    def __init__(self, tracker, student_cap):
        self.tracker,self.student_cap=tracker,student_cap
        self.memory_highwater=dict(rss_bytes=0,mps_allocated_bytes=0,mps_driver_bytes=0)
        self.phase_root=tracker.attempt.path.parents[3]
        self.phase_key=tracker.attempt.spec.method_variant
        self.plan_sha256=tracker.attempt.spec.resolved_config['plan_sha256']
        caps=dict(budget=tracker.budget.to_dict(),student_forward_calls=student_cap,plan_sha256=self.plan_sha256)
        path=self.phase_root/('caps-'+self.phase_key+'.json')
        if path.exists():
            if json.loads(path.read_text())!=caps: raise ValueError('immutable aggregate pilot caps differ')
        else: _write(path,caps)
        self.phase_wall_at_start=self.aggregate().wall_seconds
        self.started=time.monotonic()

    def aggregate(self):
        values=CostRecord().to_dict()
        for spec_path in (self.phase_root/'runs').glob('*/spec.json'):
            spec=json.loads(spec_path.read_text())
            if spec['method_variant']!=self.phase_key: continue
            if spec['resolved_config'].get('plan_sha256')!=self.plan_sha256:
                raise RecordError('pilot plan lineage changed within phase')
            ledger=spec_path.parent/'costs.jsonl'
            if not ledger.exists(): continue
            with locked_ledger(ledger) as stream: rows=read_rows(stream)
            for row in rows:
                raw=row['cost']
                if 'student_forward_calls' not in raw:
                    raise RecordError('legacy missing student counter is unknown, not zero')
                cost=CostRecord(**raw)
                for field in fields(cost):
                    name,value=field.name,getattr(cost,field.name)
                    if name in ('estimation_method','memory_unavailable_reason'): continue
                    if name in ('memory_bytes','disk_bytes'):
                        if value is not None: values[name]=max(values[name] or 0,value)
                    else: values[name]+=value
        if values['memory_bytes'] is not None: values['memory_unavailable_reason']=None
        return CostRecord(**values)

    def check_aggregate(self,cost=None):
        total=self.aggregate();cost=cost or CostRecord()
        if max(total.wall_seconds,self.phase_wall_at_start+time.monotonic()-self.started)+cost.wall_seconds>self.tracker.budget.wall_seconds:
            raise BudgetExceeded('aggregate live pilot ceiling: wall_seconds')
        for field in fields(self.tracker.budget):
            name=field.name;a,b=getattr(total,name),getattr(cost,name)
            if a is None or b is None: continue
            value=max(a,b) if name in ('memory_bytes','disk_bytes') else a+b
            if value>getattr(self.tracker.budget,name):
                raise BudgetExceeded('aggregate pilot ceiling: '+name)
        if total.student_forward_calls+cost.student_forward_calls>self.student_cap:
            raise BudgetExceeded('aggregate pilot ceiling: student_forward_calls')
    def memory(self):
        values=dict(rss_bytes=psutil.Process().memory_info().rss,mps_allocated_bytes=torch.mps.current_allocated_memory(),
                    mps_driver_bytes=torch.mps.driver_allocated_memory())
        for key,value in values.items(): self.memory_highwater[key]=max(self.memory_highwater[key],value)
        return values['rss_bytes']+values['mps_driver_bytes']
    def disk(self):
        root=self.tracker.attempt.path.parents[3]
        return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    def check(self,cost):
        self.check_aggregate(cost)
        self.tracker.check(replace(cost,memory_bytes=self.memory(),memory_unavailable_reason=None,
                                   disk_bytes=max(cost.disk_bytes,self.disk())))
    def record(self,cost):
        start=time.monotonic(); torch.mps.synchronize()
        # Include actual Metal completion wait in each dispatch's elapsed cost.
        cost=replace(cost,wall_seconds=cost.wall_seconds+time.monotonic()-start,
                     memory_bytes=self.memory(),memory_unavailable_reason=None,disk_bytes=max(cost.disk_bytes,self.disk()))
        self.tracker.record(cost)
        self.check_aggregate()
    def call(self,fn,**counts):
        self.check(CostRecord(**counts)); start=time.monotonic()
        try: return fn()
        finally: self.record(CostRecord(wall_seconds=time.monotonic()-start,**counts))


def conditional_entropy(teacher,hidden,position,recorder):
    # Preserve the position axis: DecoderAdapter.readout accepts [...,N,d], N>=1.
    logits=recorder.call(lambda:teacher.readout(hidden[0,position:position+1]),teacher_forward_calls=1)
    lp=logits.float().log_softmax(-1)
    return float((-(lp.exp()*lp).sum(-1)).item())


def metadata_plan(receipt,out):
    manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',
        expected_sha256=receipt['manifests']['data-manifest.json']['sha256'],roles={'fit','probe_score'})
    selected={}
    for role in ('fit','probe_score'):
        rows=sorted((r for r in manifest.records if manifest.roles[r.document_id]==role
            and r.source=='HuggingFaceFW/fineweb-edu' and r.metadata.get('token_count',0)>=512),
            key=lambda r:hashlib.sha256(('task10-v1:'+r.document_id).encode()).digest())
        chosen=[]; seen=set()
        for r in rows:
            component=manifest.membership[r.document_id]
            if component not in seen: chosen.append(r.document_id);seen.add(component)
            if len(chosen)==8: break
        if len(chosen)!=8: raise ValueError('insufficient metadata-only component-distinct pilot population')
        selected[role]=chosen
    plan=dict(schema_version=1,scope='Task10 operation validation; no active advantage, full-depth diagnostic or final budget',
        selection='sha256(task10-v1:document_id), eight distinct components/role, FineWeb metadata token_count>=512',
        selected_ids=selected,context_tokens=512,context_policy='first_tokens_reset_0_v1',
        representative_depth_policy='0,floor((L-1)/2),L-1',seed=17,generated_edits_per_document=2,
        selected_per_document_cap=2,selection_count=12,random_component_fraction=.25,
        depth_region_balance=[1/3]*3,max_pool_records=96,max_cache_entries=8,max_cache_bytes=32*1024**2,
        pathology=dict(lower_quantile=.01,upper_quantile=.99,norm='L2 per valid hidden target position, per depth',
            entropy='full-vocabulary teacher natural entropy per valid target; reused depth labels are not independent',
            donor_filter='norm and conditional teacher entropy at earliest patched position',source='fit only'),
        mask_policy='cycle full,span,quarter0..3 by saved seed and emitted record count; one mask per edit-depth',
        data_manifest_sha256=receipt['manifests']['data-manifest.json']['sha256'],
        teacher_asset_sha256=receipt['teacher_asset_manifest_sha256'])
    path=out/'plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan: raise ValueError('saved pilot population/policy differs')
    else: _write(path,plan)
    return manifest,plan


def run(stage,out,*,teacher_cap,student_cap,wall_cap,memory_cap):
    if not torch.backends.mps.is_available(): raise RuntimeError('explicit Torch FP32 MPS unavailable; no fallback')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((ROOT/'provenance/preparation-135m-2026-10-27.json').read_text())
    manifest,plan=metadata_plan(receipt,out)
    budgets=Budget(updates=0,valid_target_presentations=0,teacher_forward_calls=teacher_cap,
        suffix_backward_calls=0,student_backward_calls=0,search_proposals=512,search_seconds=60,
        wall_seconds=wall_cap,disk_bytes=256*1024**2,memory_bytes=memory_cap)
    source_files=['poc/probes.py','poc/contracts.py','poc/losses.py','poc/runtime/torch_decoder.py',
                  'provenance/measure_probe_pilot.py']
    sources={f:sha256_file(ROOT/f) for f in source_files}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    spec=RunSpec(method_id='M',method_variant='task10_'+stage,
        resolved_config=dict(plan_sha256=sha256_file(out/'plan.json'),source_sha256=sources,student_forward_cap=student_cap),
        source_commit=commit,dirty_patch_sha256=hashlib.sha256(patch).hexdigest(),
        dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
        input_manifests={'teacher':plan['teacher_asset_sha256'],'data':plan['data_manifest_sha256']},
        hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},
        seeds={'probes':17},parent_run_ids=(),budgets=budgets)
    attempt=start_attempt(spec,out/'runs'); tracker=BudgetTracker(budgets,attempt)
    recorder=PilotRecorder(tracker,student_cap);start=time.monotonic()
    append_event(attempt,dict(kind='start',stage=stage,plan_sha256=sha256_file(out/'plan.json')))
    def save_row(name,value):
        # Every successful/rejected probe row survives any later failure.
        append_event(attempt,dict(kind=name,**value))
    result=dict(stage=stage,attempt=str(attempt.path.relative_to(ROOT)),source_sha256=sources,
        teacher_asset_sha256=plan['teacher_asset_sha256'],plan_sha256=sha256_file(out/'plan.json'),
        backend='torch_mps',dtype='float32',execution_profile=EXECUTION_PROFILE)
    try:
        asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=plan['teacher_asset_sha256'])
        tokenizer=recorder.call(lambda:load_tokenizer(asset))
        teacher=recorder.call(lambda:load_teacher(asset,load_config(ROOT/'configs/135m.yaml',phase='exploratory')).to(device='mps'))
        depths=(0,(len(teacher.layers)-1)//2,len(teacher.layers)-1)
        if len(set(depths))!=3: raise ValueError('three representative depths require L>=3')
        tmanifest=teacher.manifest();result['teacher_digest']=tmanifest.checkpoint_digest;result['depths']=depths
        fit_ids=plan['selected_ids']['fit'][:1] if stage=='costcheck' else plan['selected_ids']['fit']
        selected=[r for r in manifest.records if r.document_id in fit_ids]
        docs=recorder.call(lambda:list(open_records(_restricted(manifest,selected),purpose='counterexample')))
        docs.sort(key=lambda r:fit_ids.index(r.document_id))
        fit_observations=[]; identity_proofs=[]
        for doc in docs:
            raw=recorder.call(lambda:tokenizer(doc.text,add_special_tokens=False)['input_ids'],processed_inputs=1)
            recorder.record(CostRecord(processed_tokens=len(raw)))
            ids=tuple(raw[:512])
            if len(ids)!=512: raise ValueError('predeclared primary512 document is short; no population replacement')
            batch=_batch(ids,device='mps',document_id=doc.document_id,cluster_id=manifest.membership[doc.document_id])
            if stage=='costcheck':
                b=prefix_for_probe(teacher,batch,depths[1],source='teacher',recorder=recorder)
                metric=local_discrepancy(teacher,teacher.layers[b.depth].mlp,b,32,recorder=recorder)
                if metric.kl!=0: raise ValueError('identity local KL must be exact zero')
                result['identity_metric']=asdict(metric)
                continue
            with torch.no_grad():
                hidden=recorder.call(lambda:teacher.final_hidden(batch),teacher_forward_calls=1)
                entropy=[]
                for pos in range(0,511,32):
                    logits=recorder.call(lambda:teacher.readout(hidden[0,pos:min(pos+32,511)]),teacher_forward_calls=1)
                    logp=logits.float().log_softmax(-1)
                    entropy.extend((-(logp.exp()*logp).sum(-1)).cpu().tolist())
                for depth in depths:
                    b=prefix_for_probe(teacher,batch,depth,source='teacher',recorder=recorder)
                    if doc.document_id==fit_ids[0]:
                        z=recorder.call(lambda:teacher.ff_residual(b),teacher_forward_calls=1)
                        suffix=recorder.call(lambda:teacher.suffix_hidden(z,b),teacher_forward_calls=1)
                        exact=torch.equal(suffix,hidden)
                        identity_proofs.append(dict(depth=depth,full_sequence_hidden_exact=exact,valid_targets=511))
                        if not exact: raise ValueError('natural suffix identity proof failed; entropy cannot be reused')
                    norms=b.u[0,:-1].float().norm(dim=-1).cpu().tolist()
                    fit_observations.extend(PathologyObservation(doc.document_id,manifest.membership[doc.document_id],
                        'fit',norm,ent,depth) for norm,ent in zip(norms,entropy))
        if stage=='costcheck':
            result['costcheck_scope']='one predeclared fit document, context512, middle-depth identity target/candidate'
        else:
            thresholds=fit_pathology_thresholds(fit_observations,lower_quantile=.01,upper_quantile=.99)
            _write(attempt.path/'thresholds.json',asdict(thresholds))
            calibration=dict(threshold_digest=thresholds.digest,identity_proofs=identity_proofs,
                independent_documents=8,independent_entropy_positions=8*511,
                by_depth={str(d):dict(valid_positions=sum(x.depth==d for x in fit_observations),
                    norm=thresholds.interval('norm',d),entropy=thresholds.interval('entropy',d)) for d in depths})
            save_row('calibration',calibration);result['calibration']=calibration
            score_ids=plan['selected_ids']['probe_score']; selected=[r for r in manifest.records if r.document_id in score_ids]
            docs=recorder.call(lambda:list(open_records(_restricted(manifest,selected),purpose='ranking')))
            docs.sort(key=lambda r:score_ids.index(r.document_id))
            roles={d:manifest.roles[d] for d in score_ids};components={d:manifest.membership[d] for d in score_ids}
            policy=ProbePolicy(selection_mode='active',seed=17,per_document_cap=2,generated_edits_per_document=2,
                random_component_fraction=.25,depth_regions=tuple((d,) for d in depths),depth_region_balance=(1/3,)*3,
                selection_count=12,max_pool_records=96,max_cache_entries=8,max_cache_bytes=32*1024**2,
                thresholds=thresholds,roles=roles,components=components,allowed_roles=('probe_score',),use='rank',
                teacher_digest=tmanifest.checkpoint_digest,candidate_digest=tmanifest.checkpoint_digest,
                tokenizer_digest=tmanifest.tokenizer_digest,dtype=tmanifest.dtype,backend=tmanifest.backend,
                boundary_version='post_attention_v1',execution_profile=EXECUTION_PROFILE,context_tokens=512,recorder=recorder)
            rows=[]
            for row in generate_donors(docs,tokenizer,policy):
                save_row('donor',dict(record=asdict(row)));rows.append(row)
            pool=ProbePool(tuple(rows)); scores={}; pathology={};metrics={};future_proofs=0;cache_hits=0;null_controls=[];controlled_depths=set()
            cache=ProbeCache(max_entries=policy.max_cache_entries,max_bytes=policy.max_cache_bytes)
            for r in rows:
                if r.status!='accepted': continue
                if len(r.recipient_tokens)!=512: raise ValueError('ranking primary512 document short; no replacement')
                recipient=_batch(r.recipient_tokens,device='mps',document_id=r.recipient_id,cluster_id=r.recipient_component)
                donor_batch=_batch(r.donor_tokens,device='mps',document_id=r.donor_id,cluster_id=r.donor_component)
                b=prefix_for_probe(teacher,recipient,r.depth,source='teacher',recorder=recorder)
                donor=prefix_for_probe(teacher,donor_batch,r.depth,source='teacher',recorder=recorder)
                patched=apply_probe(b,donor,r)
                first=r.patch_positions[0]
                if not torch.equal(b.u[:,:first],donor.u[:,:first]): raise ValueError('future donor edit leaked into prefix')
                future_proofs+=1
                cache.put(r,patched,teacher=teacher,candidate=teacher)
                reused=cache.get(r,teacher=teacher,candidate=teacher)
                if reused is None or not torch.equal(reused.u,patched.u): raise ValueError('identical state cache did not reuse')
                cache_hits+=1
                metric=local_discrepancy(teacher,teacher.layers[r.depth].mlp,reused,32,recorder=recorder)
                if metric.kl!=0: raise ValueError('identity FFN mismatch on patched boundary')
                with torch.no_grad():
                    z=recorder.call(lambda:teacher.ff_residual(patched),teacher_forward_calls=1)
                    h=recorder.call(lambda:teacher.suffix_hidden(z,patched),teacher_forward_calls=1)
                    ent=conditional_entropy(teacher,h,first,recorder)
                pathology[r.probe_id]=(float(patched.u[0,first].float().norm().item()),ent)
                if r.depth not in controlled_depths:
                    controlled_depths.add(r.depth)
                    null=replace(r,patch_positions=(),target_mask=(True,)*511)
                    zero=apply_probe(b,donor,null)
                    equal=apply_probe(b,replace(donor,u=b.u),r)
                    if not torch.equal(zero.u,b.u) or not torch.equal(equal.u,b.u):
                        raise ValueError('zero/equal patch changed hidden values')
                    null_metric=local_discrepancy(teacher,teacher.layers[r.depth].mlp,zero,32,recorder=recorder)
                    z0=recorder.call(lambda:teacher.ff_residual(b),teacher_forward_calls=1)
                    with torch.no_grad():
                        unpatched=recorder.call(lambda:teacher.suffix_hidden(z0,b),teacher_forward_calls=1)
                    if not torch.equal(unpatched[:,:first],h[:,:first]):
                        raise ValueError('patch leaked backwards through teacher suffix')
                    if null_metric.kl!=0: raise ValueError('zero-mask identity discrepancy')
                    null_controls.append(dict(depth=r.depth,zero_mask_kl=null_metric.kl,equal_patch_exact=True,suffix_future_leakage=False))
                scores[r.probe_id]=metric.kl/metric.valid_targets;metrics[r.probe_id]=asdict(metric)
                save_row('score',dict(probe_id=r.probe_id,metric=asdict(metric),pathology=pathology[r.probe_id]))
            filtered=filter_pathologies(pool,pathology,thresholds)
            selections={}
            for name,population in [('unfiltered',pool),('filtered',filtered)]:
                for mode in ['active','seeded_random']:
                    selected_pool=mine_probes(population,scores,replace(policy,selection_mode=mode))
                    selections[name+'_'+mode]=dict(ids=selected_pool.selected_ids,status=selected_pool.selection_status,
                        reason=selected_pool.selection_reason,mean_kl=sum(scores[k] for k in selected_pool.selected_ids)/len(selected_pool.selected_ids)
                        if selected_pool.selected_ids else None)
            result.update(donor_attempt_rows=len(rows),accepted=sum(r.status=='accepted' for r in rows),
                rejected_reasons=dict(Counter(r.rejection_reason for r in rows if r.status=='rejected')),
                patch_modes=dict(Counter(r.patch_mode for r in rows if r.status=='accepted')),
                exclusions=dict(filtered.exclusions),selections=selections,future_prefix_proofs=future_proofs,
                identical_state_cache_hits=cache_hits,cache_final_entries=cache.entry_count,cache_final_bytes=cache.resident_bytes,
                null_equal_controls=null_controls,all_identity_kl_zero=all(m['kl']==0 for m in metrics.values()),
                scientific_scope='identity real-teacher operation proof; no active-selection advantage measurement')
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc))
        save_row('failure',dict(exception=type(exc).__name__,reason=str(exc)))
        raise
    finally:
        result['process_wall_seconds']=time.monotonic()-start
        # Include loading/role ledger/snapshot/orchestration overhead without double-counting dispatches.
        elapsed=result['process_wall_seconds'];accounted=tracker.totals.wall_seconds
        fatal_overrun=None
        if elapsed>accounted:
            try: recorder.record(CostRecord(wall_seconds=elapsed-accounted))
            except BudgetExceeded as exc:
                result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc))
                fatal_overrun=exc
        result['cost']=tracker.totals.to_dict()
        result['aggregate_phase_cost']=recorder.aggregate().to_dict()
        result['memory_observed_highwaters']=recorder.memory_highwater
        result['memory_guard_note']='RSS+driver is a conservative overlapping observation, not a unique physical-memory peak'
        result['unmeasured_cost_fields']=['unique_tokens','estimated_operations']
        _write(attempt.path/'result.json',result)
        append_event(attempt,dict(kind='result',status=result['status'],cost=result['cost']))
        print(json.dumps(result,sort_keys=True))
        if fatal_overrun is not None: raise fatal_overrun
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('costcheck','pilot'),required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/probes/task10')
    parser.add_argument('--teacher-cap',type=int,required=True);parser.add_argument('--student-cap',type=int,required=True)
    parser.add_argument('--wall-cap',type=float,required=True);parser.add_argument('--memory-cap',type=int,required=True)
    a=parser.parse_args();run(a.stage,a.out,teacher_cap=a.teacher_cap,student_cap=a.student_cap,
        wall_cap=a.wall_cap,memory_cap=a.memory_cap)

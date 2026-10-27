"""Finite real135M fitting measurements; no protected data or downloads.

Profile first. Learning stages require a separately persisted outcome-blind plan.
Sources and dirty patch are copied before execution, including untracked code.
"""
from dataclasses import asdict,replace
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil
import torch

from poc.bank import make_bank,canonicalize_partition,initialize_residual
from poc.budget import BudgetTracker
from poc.config import load_config
from poc.contracts import Budget,CostRecord,RunSpec
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.losses import backward_distillation,stream_metrics
from poc.probes import local_discrepancy,prefix_for_probe,backward_local_discrepancy
from poc.records import append_event,canonical_json,start_attempt
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,load_tokenizer,sha256_file
from poc.teacher_validation import _batch,_restricted,_write
from poc.train import TrainingConfig,TrainingSources,fit_model,_optimizer,configure_trainable_owners,_Dispatch,_Counts,_call
from provenance.measure_probe_pilot import PilotRecorder

ROOT=Path(__file__).resolve().parents[1]
GiB=1024**3


class TrainingPilotRecorder(PilotRecorder):
    """Same accounting policy with prefix-verified incremental run totals."""
    def __init__(self,tracker,student_cap):
        self._phase_trackers={}
        self.carry=tracker.attempt.spec.resolved_config['carry_in']
        super().__init__(tracker,student_cap)

    def aggregate(self):
        from poc.checkpoint import _spec
        from poc.records import AttemptRecord
        total=CostRecord(wall_seconds=self.carry['audit_wall_seconds'])
        prior_paths={str(ROOT/p) for p in self.carry['spec_paths']}
        paths=list((self.phase_root/'runs').glob('*/spec.json'))+[Path(p) for p in sorted(prior_paths)]
        for spec_path in paths:
            raw=json.loads(spec_path.read_text())
            if raw['method_variant']!=self.phase_key:continue
            if str(spec_path) not in prior_paths and raw['resolved_config'].get('plan_sha256')!=self.plan_sha256:raise ValueError('pilot plan lineage changed within phase')
            key=str(spec_path.parent)
            if key not in self._phase_trackers:
                identity=_spec(raw)
                if key==str(self.tracker.path.parent):tracker=self.tracker
                else:
                    tracker=BudgetTracker(identity.budgets,AttemptRecord(spec_path.parent.name,'aggregate_only',spec_path.parent/'attempts'/'aggregate_only',identity))
                if str(spec_path) in prior_paths:
                    evidence=self.carry['evidence'][spec_path.parent.name]
                    if sha256_file(spec_path)!=evidence['spec_sha256'] or sha256_file(tracker.path)!=evidence['cost_ledger_sha256']:raise ValueError('prior carry-in evidence changed')
                self._phase_trackers[key]=tracker
            observed=self._phase_trackers[key].totals
            total=observed if total is None else BudgetTracker.combine(total,observed)
        return total

    def disk(self):
        # Retained diagnostics, all generations and transient files share the cap.
        return sum(p.stat().st_size for p in (ROOT/'artifacts/training/task11').rglob('*') if p.is_file())


def plan(out,stage,wall,updates):
    receipt=json.loads((ROOT/'provenance/preparation-135m-2026-10-27.json').read_text())
    manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',
        expected_sha256=receipt['manifests']['data-manifest.json']['sha256'],roles={'fit','dev'})
    rows=sorted((r for r in manifest.records if manifest.roles[r.document_id]=='fit' and r.source=='HuggingFaceFW/fineweb-edu' and r.metadata.get('token_count',0)>=512),
        key=lambda r:hashlib.sha256(('task11-v1:'+r.document_id).encode()).digest())
    chosen=[];seen=set()
    for row in rows:
        component=manifest.membership[row.document_id]
        if component not in seen:chosen.append(row.document_id);seen.add(component)
        if len(chosen)==20:break
    if len(chosen)!=20:raise ValueError('insufficient component-distinct fit windows')
    observations=chosen[17:]
    excluded={manifest.membership[i] for i in observations}
    fit_ids=[];seen=set(excluded)
    for row in rows:
        component=manifest.membership[row.document_id]
        if component not in seen:fit_ids.append(row.document_id);seen.add(component)
        if len(fit_ids)==512:break
    if len(fit_ids)!=512:raise ValueError('512 component-distinct fit windows required')
    audit_path=ROOT/'artifacts/training/task11/canonical-window-audit-v1.json'
    audit=json.loads(audit_path.read_text())
    if audit['saved_windows_sha256']!=receipt['manifests']['windows.json']['sha256']:
        raise ValueError('canonical window audit lineage mismatch')
    short={r['document_id']:r['context_tokens'] for r in audit['short_windows']}
    if any(i not in fit_ids for i in short):raise ValueError('unexpected short observation/development window')
    original_fit_ids=list(fit_ids)
    fit_ids=[i for i in fit_ids if i not in short]
    if len(fit_ids)<497:raise ValueError('insufficient verified full windows for253952targets without repetition')
    development=[];seen=set()
    for row in sorted((r for r in manifest.records if manifest.roles[r.document_id]=='dev' and r.source=='HuggingFaceFW/fineweb-edu' and r.metadata.get('token_count',0)>=512),key=lambda r:hashlib.sha256(('task11-v1:'+r.document_id).encode()).digest()):
        component=manifest.membership[row.document_id]
        if component not in seen:development.append(row.document_id);seen.add(component)
        if len(development)==8:break
    if len(development)!=8:raise ValueError('insufficient component-distinct development windows')
    from poc.checkpoint import _spec
    from poc.records import AttemptRecord,run_id
    prior_specs=[];carry_cost=None;carry_evidence={}
    for name in ('learn31','learn31-diverse-v1','learn31-diverse-v2'):
        for spec_path in sorted((ROOT/'artifacts/training/task11'/name/'runs').glob('*/spec.json')):
            raw=json.loads(spec_path.read_text());prior_spec=_spec(raw)
            if run_id(prior_spec)!=spec_path.parent.name:raise ValueError('prior cost identity mismatch')
            tracker=BudgetTracker(prior_spec.budgets,AttemptRecord(spec_path.parent.name,'carry_only',spec_path.parent/'attempts'/'carry_only',prior_spec))
            observed=tracker.totals
            carry_cost=observed if carry_cost is None else BudgetTracker.combine(carry_cost,observed)
            prior_specs.append(str(spec_path.relative_to(ROOT)))
            carry_evidence[spec_path.parent.name]={'spec_sha256':sha256_file(spec_path),'cost_ledger_sha256':sha256_file(tracker.path)}
    carry_cost=carry_cost or CostRecord()
    carry=dict(spec_paths=prior_specs,verified_cost=carry_cost.to_dict(),evidence=carry_evidence,
        audit_wall_seconds=audit['wall_seconds'],audit_sha256=sha256_file(audit_path))
    available=psutil.virtual_memory().available
    memory=min(6*GiB,int(.8*available))
    p=dict(schema_version=1,stage=stage,selection='original512 metadata-ranked fit IDs, exclude4 canonicalshort windows without replacement;508 distinct full fit components, fixed3fitchecks',
        fit_ids=fit_ids,original_fit_ids=original_fit_ids,observation_ids=observations,dev_ids=development,excluded_short_windows=short,
        canonical_window_audit_sha256=sha256_file(audit_path),saved_windows_sha256=receipt['manifests']['windows.json']['sha256'],natural_window_policy={'context_tokens':512,'max_windows':4,'seed':17,'selection':'first regenerated saved-policy window per document'},
        observation_steps=[0]+[k for k in (2,8,16,31) if k<=updates],context_tokens=512,window='verifiedpreparedtokens;WindowPolicy512/max4/seed17 firstwindow;source_start saved;positions reset0;originaltarget_indices',
        head_chunk=32,bank_count=15,rank=8,seed=17,alpha=.1,lr_bank=1e-4,lr_residual=3e-4,
        max_updates=updates,wall_cap=wall,carry_in=carry,teacher_cap=2048 if stage=='profile' else 12000+carry_cost.teacher_forward_calls,
        student_cap=2048 if stage=='profile' else 12000+carry_cost.student_forward_calls,memory_cap=memory,headroom_at_plan_bytes=available,
        backward_cap=4 if stage=='profile' else 600+carry_cost.student_backward_calls,suffix_backward_cap=3 if stage=='profile' else 0,
        update_cap=1 if stage=='profile' else updates+carry_cost.updates,presentation_cap=2044 if stage=='profile' else updates*8192+carry_cost.valid_target_presentations,
        disk_cap=3*GiB if stage=='profile' else 8*GiB,data_manifest_sha256=receipt['manifests']['data-manifest.json']['sha256'],
        teacher_asset_sha256=receipt['teacher_asset_manifest_sha256'],scope='finite exploratory resource/fit evidence; final study seeds/budgets unchosen')
    path=out/'plan.json'
    if path.exists():
        existing=json.loads(path.read_text())
        # Fresh headroom is rechecked live; immutable historical resource ceilings remain.
        for key in ('headroom_at_plan_bytes','memory_cap'):p[key]=existing[key]
        if existing!=p:raise ValueError('existing plan differs; use new phase directory')
    else:_write(path,p)
    return manifest,p


def run(stage,out,wall,updates):
    if not torch.backends.mps.is_available():raise RuntimeError('realMPS required; no CPU fallback')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest,p=plan(out,stage,wall,updates)
    if psutil.virtual_memory().available<GiB:raise RuntimeError('insufficient fresh headroom')
    source_files=sorted([str(f.relative_to(ROOT)) for f in (ROOT/'poc').rglob('*.py')]+['provenance/measure_training_pilot.py','provenance/measure_probe_pilot.py'])
    sources={f:sha256_file(ROOT/f) for f in source_files}
    budgets=Budget(updates=p['update_cap'],valid_target_presentations=p['presentation_cap'],teacher_forward_calls=p['teacher_cap'],
        suffix_backward_calls=p['suffix_backward_cap'],student_backward_calls=p['backward_cap'],search_proposals=0,search_seconds=0,
        wall_seconds=wall,disk_bytes=p['disk_cap'],memory_bytes=p['memory_cap'])
    config=TrainingConfig(max_updates=max(1,updates),epochs=1,observation_steps=tuple(k for k in (2,8,16,31) if k<=updates),kl_floor_per_target=1.2814748543802652e-7,
        numerical_profile=EXECUTION_PROFILE+';teacher-qualification-torch_mps-context512')
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    spec=RunSpec(method_id='M',method_variant='task11_'+stage+'_natural_only',
        resolved_config=dict(plan_sha256=sha256_file(out/'plan.json'),source_sha256=sources,training=config.identity(),student_forward_cap=p['student_cap'],carry_in=p['carry_in']),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty_patch_sha256=hashlib.sha256(patch).hexdigest(),
        dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
        input_manifests={'teacher':p['teacher_asset_sha256'],'data':p['data_manifest_sha256'],'saved_windows':p['saved_windows_sha256'],'window_audit':p['canonical_window_audit_sha256']},
        hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},seeds={'pilot':17},parent_run_ids=tuple(p['carry_in']['evidence']),budgets=budgets)
    attempt=start_attempt(spec,out/'runs');tracker=BudgetTracker(budgets,attempt);rec=TrainingPilotRecorder(tracker,p['student_cap'])
    snap=attempt.path/'source';snap.mkdir()
    for f in source_files:
        destination=snap/f;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,destination)
    (snap/'dirty.patch').write_bytes(patch)
    result=dict(schema_version=1,stage=stage,run_id=attempt.run_id,attempt_id=attempt.attempt_id,attempt=str(attempt.path.relative_to(ROOT)),
        plan_sha256=sha256_file(out/'plan.json'),source_sha256=sources,measurements=[],status='running',
        memory_guard='RSS+MPSdriver conservative overlapping observation; separate counters also retained')
    started=time.monotonic()
    run_record=json.loads((attempt.path/'run.json').read_text());run_record['status']='running';_write(attempt.path/'run.json',run_record)
    append_event(attempt,dict(kind='attempt_status',status='running'))
    def measured(name,fn,*,nested=False,**counts):
        start=time.monotonic()
        value=fn() if nested else rec.call(fn,**counts)
        row=dict(name=name,wall_seconds=time.monotonic()-start,**rec.memory_highwater)
        result['measurements'].append(row);append_event(attempt,dict(kind='measurement',**row));print(json.dumps(row),flush=True)
        return value
    try:
        asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=p['teacher_asset_sha256'])
        tokenizer=measured('tokenizer_load',lambda:load_tokenizer(asset))
        teacher=measured('teacher_load',lambda:load_teacher(asset,load_config(ROOT/'configs/135m.yaml',phase='exploratory')).to('mps'))
        chosen=set(p['fit_ids']+p['observation_ids']);records=[r for r in manifest.records if r.document_id in chosen]
        docs=measured('role_reader',lambda:list(open_records(_restricted(manifest,records),purpose='fit')))
        if stage!='profile':
            dev_records=[r for r in manifest.records if r.document_id in p['dev_ids']]
            docs.extend(measured('development_role_reader',lambda:list(open_records(_restricted(manifest,dev_records),purpose='selection'))))
        by_id={d.document_id:d for d in docs};batches={};window_rows=[]
        if sha256_file(ROOT/'data/prepared-135m-v2/windows.json')!=p['saved_windows_sha256']:raise ValueError('full saved windows manifest hash mismatch')
        from poc.quantization_stage import _prepared_window
        for identity in p['fit_ids']+p['observation_ids']+(p['dev_ids'] if stage!='profile' else []):
            w=_prepared_window(by_id[identity],p['natural_window_policy']['seed'])
            if len(w.token_ids)!=512:raise ValueError('predeclared canonical512 window too short; no replacement')
            window_rows.append(dict(**asdict(w),role=manifest.roles[identity],component=manifest.membership[identity],token_file_sha256=by_id[identity].token_sha256))
            batches[identity]=_batch(w.token_ids,device='mps',document_id=identity,cluster_id=manifest.membership[identity],targets=w.target_indices)
        # Full fit stream and independent fit/dev observation windows are fixed
        # before any prediction. Text/tokens stay in ignored local artifacts.
        _write(attempt.path/'window-stream.json',{'policy':p['natural_window_policy'],'saved_windows_sha256':p['saved_windows_sha256'],'windows':window_rows})
        result['window_stream_sha256']=sha256_file(attempt.path/'window-stream.json')
        b=batches[p['fit_ids'][0]]
        if stage=='profile':
            hidden=measured('teacher_full_context512',lambda:teacher.final_hidden(b),teacher_forward_calls=1)
            measured('readout_32_full_vocab',lambda:teacher.readout(hidden[:,:32]),teacher_forward_calls=1)
            for depth in (0,14,29):
                bound=measured(f'prefix_{depth}',lambda:teacher.prefix_boundary(b,depth),teacher_forward_calls=1)
                measured(f'suffix_{depth}',lambda:teacher.suffix_hidden(teacher.ff_residual(bound),bound),teacher_forward_calls=2)
            delta=teacher.layers[1].mlp.gate_proj.weight.detach()-teacher.layers[0].mlp.gate_proj.weight.detach()
            measured('cpu_fp64_svd_1536x576_rank8',lambda:initialize_residual(delta,8,17))
        student=measured('bank_K15_rank8_initialize',lambda:make_bank(teacher,canonicalize_partition([list(range(i,i+2)) for i in range(0,30,2)],30),8,17))
        if stage=='profile':
            owners=configure_trainable_owners(student,config);optimizer=_optimizer(owners,config)
            # Disposable511-target allocation probe, explicitly not a fitted update.
            measured('global_backward_511_resource_only',lambda:backward_distillation(_Dispatch(teacher,_Counts(rec),'teacher'),_Dispatch(student,_Counts(rec),'student'),
                b,alpha=.1,denominator=511,chunk_size=32),valid_target_presentations=511,student_backward_calls=1)
            measured('adamw_step_resource_only_511_not_learning',optimizer.step,updates=1)
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            for depth in (0,14,29):
                bound=measured(f'local_prefix_{depth}',lambda:prefix_for_probe(teacher,b,depth,source='teacher',recorder=rec),nested=True)
                from poc.contracts import ProbeRecord
                tokens=tuple(b.input_ids[0].tolist());positions=tuple(range(512));digest=teacher.manifest().checkpoint_digest
                record=ProbeRecord(probe_id='resource-'+str(depth),recipient_id=b.document_ids[0],donor_id=b.document_ids[0],recipient_component=b.cluster_ids[0],donor_component=b.cluster_ids[0],
                    recipient_role='fit',donor_role='fit',use='fit',source='natural',recipient_tokens=tokens,donor_tokens=tokens,attention_mask=(True,)*512,
                    positions=positions,donor_positions=positions,target_mask=(True,)*511,patch_positions=(),patch_mode='full',channel_quarter=None,depth=depth,
                    teacher_digest=digest,candidate_digest=student.manifest().checkpoint_digest,tokenizer_digest=teacher.tokenizer_digest,dtype='float32',backend='torch_mps',
                    boundary_version='post_attention_v1',execution_profile=EXECUTION_PROFILE,seed=17,edit_kind='identity',source_span=(0,0),donor_span=(0,0),replacement='',prefix_source='teacher',
                    source_window=(0,512),recipient_length_before_crop=512,donor_length_before_crop=512,recipient_full_tokens_digest=hashlib.sha256(canonical_json({'tokens':tokens})).hexdigest(),
                    donor_full_tokens_digest=hashlib.sha256(canonical_json({'tokens':tokens})).hexdigest(),context_policy='first_tokens_reset_0_v1')
                measured(f'local_suffix_backward_{depth}_resource_only',lambda:backward_local_discrepancy(teacher,student.layers[depth].mlp,bound,chunk_size=32,denominator=511,record=record,recorder=rec),nested=True)
                student.zero_grad(set_to_none=True)
        else:
            def observe(step):
                for role,identities in [('fit_observation',p['observation_ids']),('dev',p['dev_ids'])]:
                    observations=[batches[i] for i in identities]
                    label=f'{role}_step{step}'
                    sums=measured(label,lambda:[stream_metrics(_Dispatch(teacher,_Counts(rec),'teacher'),_Dispatch(student,_Counts(rec),'student'),b,32) for b in observations],processed_inputs=len(observations),processed_tokens=512*len(observations))
                    row=dict(label=label,step=step,role=role,valid_targets=sum(s.valid_targets for s in sums),kl_sum=sum(s.kl for s in sums),nll_sum=sum(s.nll for s in sums))
                    row['kl_per_target']=row['kl_sum']/row['valid_targets'];row['objective_per_target']=(row['kl_sum']+.1*row['nll_sum'])/row['valid_targets']
                    append_event(attempt,dict(kind='observation',**row));result.setdefault('observations',[]).append(row)
            observe(0)
            # A single pass over512 distinct fit components; exact8192 splits
            # reuse full context only at update boundaries, never repeat epochs.
            fit_batches=[batches[i] for i in p['fit_ids']]
            stream=TrainingSources(natural=tuple(fit_batches),manifest_hashes={'data':p['data_manifest_sha256']})
            _write(attempt.path/'fit-target-stream.json',{str(i):dict(document=b.document_ids[0],target_mask=b.target_mask[0].tolist()) for i,b in enumerate(fit_batches)})
            checkpoint_root=attempt.path/'checkpoints'
            previous=0;phase_started=time.monotonic()
            def milestone(step):
                nonlocal previous,phase_started
                from poc.checkpoint import resolve_checkpoint
                phase_seconds=time.monotonic()-phase_started
                observe(step)
                ref=resolve_checkpoint(checkpoint_root)
                destination=attempt.path/'milestones'/str(step)
                measured(f'preserve_milestone_{step}',lambda:shutil.copytree(ref.directory,destination))
                result.setdefault('milestones',[]).append(dict(step=step,directory=str(destination),manifest_sha256=ref.manifest_sha256))
                elapsed=rec.phase_wall_at_start+time.monotonic()-started
                # Include evaluation/copy overhead in the measured continuation phase.
                phase_seconds=time.monotonic()-phase_started
                forecast=elapsed+(updates-step)*phase_seconds/(step-previous)
                row=dict(kind='continuation_forecast',step=step,elapsed_seconds=elapsed,remaining_updates=updates-step,
                    measured_phase_seconds=phase_seconds,forecast_total_seconds=forecast,wall_cap_seconds=wall,
                    policy='linear measured phase including checkpoint+evaluation+ledger; explicit heuristic, no throughput promise')
                append_event(attempt,row);result.setdefault('continuation_forecasts',[]).append(row);print(json.dumps(row),flush=True)
                if forecast>wall:
                    from poc.budget import BudgetExceeded
                    raise BudgetExceeded('measured continuation forecast exceeds fixed wall cap')
                previous=step;phase_started=time.monotonic()
            config=replace(config,run_spec=spec,checkpoint_root=checkpoint_root,on_update=milestone)
            fitted=measured('fit_exact8192_diverse_single_pass',lambda:fit_model(student,teacher,stream,config,budgets,rec),nested=True)
            result['fit']=dict(outcome=fitted.outcome,reason=fitted.reason,step=fitted.state.step,target_cursor=fitted.state.target_cursor,
                metrics=list(fitted.metrics),candidate_digest=fitted.state.candidate_digest,
                checkpoint=asdict(fitted.checkpoint) if fitted.checkpoint else None)
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc));raise
    finally:
        overrun=rec.finish_attempt(result,elapsed=time.monotonic()-started)
        _write(attempt.path/'result.json',result)
        run_record['status']='complete' if result['status']=='passed' else ('interrupted' if result.get('failure_type') in ('KeyboardInterrupt','InterruptedError') else 'failed')
        run_record['output_manifest_sha256']=sha256_file(attempt.path/'result.json') if run_record['status']=='complete' else None
        run_record['failure_category']=result.get('failure_type');run_record['failure_details']=result.get('failure_reason')
        _write(attempt.path/'run.json',run_record);append_event(attempt,dict(kind='attempt_status',status=run_record['status']))
        append_event(attempt,dict(kind='result',status=result['status'],cost=result['cost']))
        print(json.dumps(dict(result_path=str(attempt.path/'result.json'),status=result['status'],cost=result['cost'],memory=result['memory_observed_highwaters'])),flush=True)
        if overrun:raise overrun
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('profile','learn'),required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--wall-cap',type=float,required=True);parser.add_argument('--updates',type=int,default=2)
    args=parser.parse_args();run(args.stage,args.out,args.wall_cap,args.updates)

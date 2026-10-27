"""One finite genuine local-mixture update after the verified global pilot."""
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

from poc.budget import BudgetTracker
from poc.checkpoint import checkpoint_decoder,resolve_fitted_run,resolve_checkpoint
from poc.config import load_config
from poc.contracts import Budget,CostRecord,RunSpec
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.probes import (PathologyObservation,fit_pathology_thresholds,ProbePolicy,generate_donors,
    prefix_for_probe,apply_probe,local_discrepancy)
from poc.records import append_event,canonical_json,start_attempt
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,load_tokenizer,sha256_file
from poc.teacher_validation import _batch,_restricted,_write
from poc.train import TrainingConfig,TrainingSources,ProbeTrainingSource,configure_trainable_owners
from poc.train import fit_model
from provenance.measure_training_pilot import TrainingPilotRecorder,ROOT,GiB


def run(global_root,global_run,global_attempt,out):
    process_started=time.monotonic()
    if not torch.backends.mps.is_available():raise RuntimeError('actual MPS required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    global_path=Path(global_root)/global_run/'attempts'/global_attempt
    global_record=json.loads((global_path/'run.json').read_text())
    if global_record['status']!='complete' and not (global_record['status']=='failed' and global_record['failure_category']=='BudgetExceeded'):
        raise ValueError('local operation requires complete or explicitly resource-stopped global attempt')
    checkpoint_ref=resolve_checkpoint(global_path/'checkpoints')
    raw_state=json.loads((Path(checkpoint_ref.directory)/'state.json').read_text())['items']
    actual_global_step=raw_state['step']['value'];actual_global_digest=raw_state['candidate_digest']['value']
    if actual_global_step<1:raise ValueError('no verified global update available')
    global_plan=json.loads((Path(global_root).parent/'plan.json').read_text())
    ids=global_plan['original_fit_ids'][:9]
    p=dict(schema_version=1,global_run=global_run,global_attempt=global_attempt,fit_ids=ids,
        global_parent_status=global_record['status'],global_source_step=actual_global_step,global_source_digest=actual_global_digest,
        global_source_manifest_sha256=checkpoint_ref.manifest_sha256,initialization_policy='latest verified durable checkpoint; no best-by-loss selection',
        checkpoint_policy='operation-only; globalcheckpoint and ownedMPS replay qualify recovery; no local resumable artifact promised',
        context_policy='Task10 explicit first_tokens_reset_0_v1 probes; not global natural WindowPolicy',
        depth=15,affected_depths=[14,15],seed=17,per_document_cap=2,generated_edits_per_document=4,
        selected_alignment_rows=9,selection='first accepted positive-target full512 aligned rows, original9document order, cap2/document; no outcome scores',
        mixture_weights={'natural':.5,'student':.25,'donor':.25},calibration_positions=[0,256,510],
        calibration_scope='27fit positions for operation policy only; unfiltered pool/no active selection or threshold efficacy claim',
        target_count=8192,updates=1,wall_cap=180,memory_cap=min(6*GiB,int(.8*psutil.virtual_memory().available)),
        disk_cap=8*GiB,teacher_cap=4096,student_cap=256,student_backward_cap=32,suffix_backward_cap=32,
        edit_cap=36,search_seconds=30,carry_in={'spec_paths':[],'audit_wall_seconds':0,'evidence':{}},
        source_checkpoint_pointer_sha256=sha256_file(global_path/'checkpoints/current.json'))
    plan_path=out/'plan.json'
    if plan_path.exists():
        previous=json.loads(plan_path.read_text());p['memory_cap']=previous['memory_cap']
        if p!=previous:raise ValueError('local immutable plan differs')
    else:_write(plan_path,p)
    cfg=TrainingConfig(max_updates=1,alpha=0,phase='local',affected_depths=(14,15),mixture_weights=p['mixture_weights'],
        kl_floor_per_target=1.2814748543802652e-7,numerical_profile=EXECUTION_PROFILE+';qualified_MPS')
    limits=Budget(updates=1,valid_target_presentations=8192,teacher_forward_calls=4096,suffix_backward_calls=32,
        student_backward_calls=32,search_proposals=36,search_seconds=30,wall_seconds=180,disk_bytes=8*GiB,memory_bytes=p['memory_cap'])
    receipt=json.loads((ROOT/'provenance/preparation-135m-2026-10-27.json').read_text())
    source_files=sorted([str(f.relative_to(ROOT)) for f in (ROOT/'poc').rglob('*.py')]+[
        'provenance/measure_local_training_pilot.py','provenance/measure_training_pilot.py','provenance/measure_probe_pilot.py'])
    sources={f:sha256_file(ROOT/f) for f in source_files}
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    spec=RunSpec(method_id='M',method_variant='task11_local_unfiltered_operation' if actual_global_step==31 else 'task11_local_unfiltered_partial_global_operation',
        resolved_config={'training':cfg.identity(),'plan_sha256':sha256_file(plan_path),'source_sha256':sources,
                         'student_forward_cap':256,'carry_in':p['carry_in']},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty_patch_sha256=hashlib.sha256(patch).hexdigest(),
        dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
        input_manifests={'data':receipt['manifests']['data-manifest.json']['sha256'],'teacher':receipt['teacher_asset_manifest_sha256'],
                         'global_checkpoint_pointer':p['source_checkpoint_pointer_sha256']},
        hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},
        seeds={'local':17},parent_run_ids=(global_run,),budgets=limits)
    attempt=start_attempt(spec,out/'runs');rec=TrainingPilotRecorder(BudgetTracker(limits,attempt),256)
    source_root=attempt.path/'source'
    for f in source_files:
        target=source_root/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
    (source_root/'dirty.patch').write_bytes(patch)
    started=process_started;result=dict(schema_version=1,status='running',run_id=attempt.run_id,attempt_id=attempt.attempt_id,
        attempt=str(attempt.path.relative_to(ROOT)),source_sha256=sources,plan_sha256=sha256_file(plan_path),observations=[])
    record=json.loads((attempt.path/'run.json').read_text());record['status']='running';_write(attempt.path/'run.json',record)
    def measured(name,fn,*,nested=False,**counts):
        start=time.monotonic();value=fn() if nested else rec.call(fn,**counts)
        row=dict(kind='measurement',name=name,wall_seconds=time.monotonic()-start);append_event(attempt,row);print(json.dumps(row),flush=True)
        return value
    try:
        asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=receipt['teacher_asset_manifest_sha256'])
        tokenizer=measured('tokenizer_load',lambda:load_tokenizer(asset))
        teacher=measured('teacher_load',lambda:load_teacher(asset,load_config(ROOT/'configs/135m.yaml',phase='exploratory')).to('mps'))
        state,source_ref=measured('verified_global_checkpoint_load',lambda:resolve_fitted_run(global_root,global_run,global_attempt))
        if state.step!=p['global_source_step'] or state.candidate_digest!=p['global_source_digest']:raise ValueError('local initialization checkpoint changed')
        student=measured('global_candidate_reconstruction',lambda:checkpoint_decoder(state).to('mps'))
        result['source_checkpoint']=asdict(source_ref);del state
        teacher_parameters=dict(teacher.named_parameters())
        from poc.bank import owner_census
        backbone=[r.name for r in owner_census(student) if r.category=='non_ffn']
        def unchanged_backbone():
            return all(torch.equal(dict(student.named_parameters())[n],teacher_parameters[n]) for n in backbone)
        if not measured('global_frozen_backbone_value_check',unchanged_backbone):raise ValueError('global primary backbone changed')
        result['verified_global_frozen_owner_count']=len(backbone)
        manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=receipt['manifests']['data-manifest.json']['sha256'],roles={'fit'})
        docs=measured('fit_role_reader',lambda:list(open_records(_restricted(manifest,[r for r in manifest.records if r.document_id in ids]),purpose='counterexample')))
        docs.sort(key=lambda d:ids.index(d.document_id));batches={}
        for doc in docs:
            tokens=tokenizer(doc.text,add_special_tokens=False)['input_ids'][:512]
            if len(tokens)!=512:raise ValueError('local first512 probe source short; no replacement')
            batches[doc.document_id]=_batch(tokens,device='mps',document_id=doc.document_id,cluster_id=manifest.membership[doc.document_id])
        observations=[]
        with torch.no_grad():
            for doc in docs:
                b=batches[doc.document_id];boundary=prefix_for_probe(teacher,b,15,source='teacher',recorder=rec)
                hidden=rec.call(lambda:teacher.final_hidden(b),teacher_forward_calls=1)
                logits=rec.call(lambda:teacher.readout(hidden[:,p['calibration_positions']]),teacher_forward_calls=1)
                lp=logits.float().log_softmax(-1);entropy=-(lp.exp()*lp).sum(-1)
                for j,pos in enumerate(p['calibration_positions']):
                    observations.append(PathologyObservation(doc.document_id,manifest.membership[doc.document_id],'fit',
                        float(boundary.u[0,pos].float().norm().item()),float(entropy[0,j].item()),15))
        thresholds=fit_pathology_thresholds(observations,lower_quantile=.01,upper_quantile=.99)
        _write(attempt.path/'operation-calibration.json',asdict(thresholds))
        tm=teacher.manifest();policy=ProbePolicy(selection_mode='seeded_random',seed=17,per_document_cap=2,
            generated_edits_per_document=4,random_component_fraction=1,depth_regions=((15,),),depth_region_balance=(1,),
            selection_count=9,max_pool_records=36,max_cache_entries=1,max_cache_bytes=4*1024**2,thresholds=thresholds,
            roles={i:'fit' for i in ids},components={i:manifest.membership[i] for i in ids},allowed_roles=('fit',),use='fit',
            teacher_digest=tm.checkpoint_digest,candidate_digest=student.manifest().checkpoint_digest,tokenizer_digest=tm.tokenizer_digest,
            dtype='float32',backend='torch_mps',boundary_version='post_attention_v1',execution_profile=EXECUTION_PROFILE,context_tokens=512,recorder=rec)
        rows=measured('bounded_source_generation',lambda:list(generate_donors(docs,tokenizer,policy)),nested=True)
        _write(attempt.path/'all-source-attempts.json',[asdict(r) for r in rows])
        chosen=[];per_document={}
        for r in rows:
            if r.status!='accepted' or len(r.recipient_tokens)!=512 or not any(r.target_mask) or per_document.get(r.recipient_id,0)>=2:continue
            chosen.append(r);per_document[r.recipient_id]=per_document.get(r.recipient_id,0)+1
            if len(chosen)==9:break
        if len(chosen)!=9:raise ValueError(f'local alignment pool underfilled: {len(chosen)}/9; no replacement')
        templates=[];items=[]
        for base in chosen:
            for source in ('natural','student','donor'):
                r=replace(base,probe_id=hashlib.sha256(canonical_json({'base_probe_id':base.probe_id,'source':source})).hexdigest(),source=source,
                    prefix_source='candidate' if source=='student' else 'teacher',patch_positions=base.patch_positions if source=='donor' else (),
                    target_mask=base.target_mask if source=='donor' else (True,)*511,patch_mode=base.patch_mode if source=='donor' else 'full',
                    channel_quarter=base.channel_quarter if source=='donor' else None)
                templates.append(r)
                template=_batch(r.recipient_tokens,device='mps',document_id=r.recipient_id,cluster_id=r.recipient_component)
                template=replace(template,target_mask=torch.tensor([r.target_mask],dtype=torch.bool,device='mps'))
                def build(student,teacher,recorder,r=r):
                    current=replace(r,candidate_digest=student.manifest().checkpoint_digest)
                    recipient=_batch(current.recipient_tokens,device='mps',document_id=current.recipient_id,cluster_id=current.recipient_component)
                    producer=student if current.prefix_source=='candidate' else teacher
                    boundary=prefix_for_probe(producer,recipient,15,source=current.prefix_source,recorder=recorder)
                    if current.source=='donor':
                        donor=_batch(current.donor_tokens,device='mps',document_id=current.donor_id,cluster_id=current.recipient_component)
                        boundary=apply_probe(boundary,prefix_for_probe(teacher,donor,15,source='teacher',recorder=recorder),current)
                    return boundary,current
                items.append(ProbeTrainingSource(template,source,'fit',r.probe_id,build))
        _write(attempt.path/'source-templates.json',[asdict(r) for r in templates])
        data=TrainingSources(probes=tuple(items),manifest_hashes={'source_templates':sha256_file(attempt.path/'source-templates.json')})
        configure_trainable_owners(student,cfg)
        frozen_names=tuple(n for n,parameter in student.named_parameters() if not parameter.requires_grad)
        def frozen_digest():
            owners=dict(student.named_parameters());h=hashlib.sha256()
            for name in frozen_names:
                h.update(name.encode());h.update(owners[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        frozen_before=measured('frozen_owner_snapshot_digest',frozen_digest)
        def observe(label):
            sums={s:[] for s in p['mixture_weights']}
            for item in items:
                boundary,r=item.build(student,teacher,rec)
                sums[item.source].append(local_discrepancy(teacher,student.layers[15].mlp,boundary,32,recorder=rec))
            sources={s:dict(kl_sum=sum(m.kl for m in values),valid_targets=sum(m.valid_targets for m in values)) for s,values in sums.items()}
            for value in sources.values():value['kl_per_target']=value['kl_sum']/value['valid_targets']
            row=dict(kind='local_observation',label=label,metric='conditional local replacement through frozen teacher suffix KL; not full compressed-model natural KL',sources=sources,weighted_kl=sum(p['mixture_weights'][s]*v['kl_per_target'] for s,v in sources.items()),
                policy='same saved cases; natural/donor fixedteacherprefix, studentprefix refreshed at current candidate digest')
            result['observations'].append(row);append_event(attempt,row);print(json.dumps(row),flush=True)
        observe('before')
        cfg=replace(cfg,run_spec=spec)
        fitted=measured('one_local_exact8192_update',lambda:fit_model(student,teacher,data,cfg,limits,rec),nested=True)
        result['fit']=dict(step=fitted.state.step,target_cursor=fitted.state.target_cursor,metrics=list(fitted.metrics),outcome=fitted.outcome,
            checkpoint=asdict(fitted.checkpoint) if fitted.checkpoint else None,candidate_digest=fitted.state.candidate_digest)
        observe('after')
        if measured('frozen_owner_final_digest',frozen_digest)!=frozen_before:raise ValueError('local frozen owner changed')
        result['verified_local_frozen_owner_count']=len(frozen_names);result['source_rows']=len(rows);result['selected_rows']=len(chosen)
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc));raise
    finally:
        overrun=rec.finish_attempt(result,elapsed=time.monotonic()-started)
        _write(attempt.path/'result.json',result)
        record.update(status='complete' if result['status']=='passed' else 'failed',output_manifest_sha256=sha256_file(attempt.path/'result.json') if result['status']=='passed' else None,
            failure_category=result.get('failure_type'),failure_details=result.get('failure_reason'))
        _write(attempt.path/'run.json',record);append_event(attempt,dict(kind='result',status=result['status'],cost=result['cost']))
        print(json.dumps(dict(result_path=str(attempt.path/'result.json'),status=result['status'],cost=result['cost'])),flush=True)
        if overrun:raise overrun
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--global-run-root',type=Path,required=True);parser.add_argument('--global-run',required=True)
    parser.add_argument('--global-attempt',required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();run(a.global_run_root,a.global_run,a.global_attempt,a.out)

"""Bounded real SmolLM2 recovery on known exploratory data; no network/protected reads."""
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
import yaml

from poc.bank import make_bank,canonicalize_partition,PROJECTIONS,initialize_residual,_Factors
from poc.budget import BudgetTracker,BudgetExceeded
from poc.continuation_budget import PilotLedger,process_peak_bytes
from poc.continuation_plan import digest
from poc.contracts import Budget,CostRecord,RunSpec
from poc.losses import stream_metrics
from poc.merge import initialize_merge,save_trial_delta
from poc.probes import prefix_for_probe,apply_probe,local_discrepancy
from poc.records import canonical_json,run_id
from poc.recovery_data import prepare_populations,fit_templates
from poc.recovery_plan import finite_plan,fixed_parent_copy
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,load_tokenizer,sha256_file
from poc.teacher_validation import _batch
from poc.train import TrainingConfig,ProbeTrainingSource,fit_model,_call,_Dispatch,_Counts,_observe

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def finalize_costs(result,tracker,*,elapsed):
    try:
        extra=max(0,elapsed-tracker.attempt_cost.wall_seconds)
        tracker.record(CostRecord(wall_seconds=extra))
        result['aggregate_cost']=tracker.validate().to_dict()
        result['projected_transient_disk_peak_bytes']=getattr(tracker,'projected_disk_peak',None)
    except Exception as exc:
        result.update(status='failed_accounting',accounting_error=str(exc))
        result['aggregate_cost']=tracker.totals.to_dict()


def raise_for_accounting_error(result):
    if result.get('status')=='failed_accounting':
        raise BudgetExceeded(result['accounting_error'])


def subprocess_memory_bound(parent_bytes,child):
    """Concurrent processes add; their unified-memory counters may overlap."""
    return parent_bytes+sum(child[k] for k in ('process_peak_rss_bytes','torch_mps_driver_bytes','mlx_peak_bytes'))


class NativeTracker(BudgetTracker):
    def __init__(self,budget,attempt,out):
        super().__init__(budget,attempt);self.out=out;self.last_disk=0;self.disk_peak=0;self.memory_peak=0;self.projected_disk_peak=0
    def disk(self):return sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
    def check(self,cost):
        self.projected_disk_peak=max(self.projected_disk_peak,cost.disk_bytes)
        super().check(cost)
    def record(self,cost):
        # Conservative overlap: RSS + Metal driver bytes are NOT disjoint physical memory.
        memory=process_peak_bytes()+torch.mps.driver_allocated_memory()
        self.memory_peak=max(self.memory_peak,memory)
        now=time.monotonic()
        if cost.disk_bytes or now-self.last_disk>1:
            self.disk_peak=max(self.disk_peak,self.disk())
            self.last_disk=now
        super().record(replace(cost,memory_bytes=max(cost.memory_bytes or 0,self.memory_peak),
            memory_unavailable_reason=None,disk_bytes=max(cost.disk_bytes,self.disk_peak)))


class PrefixCache:
    def __init__(self,teacher):self.teacher=teacher;self.values={};self.hits=0;self.misses=0
    def get(self,batch,depth,recorder):
        key=digest({'teacher':self.teacher.manifest().checkpoint_digest,'depth':depth,
            'document':batch.document_ids,'tokens':batch.input_ids.cpu().tolist(),
            'positions':batch.positions.cpu().tolist(),'mask':batch.target_mask.cpu().tolist(),
            'profile':EXECUTION_PROFILE})
        if key not in self.values:
            if len(self.values)>=128:raise ValueError('bounded teacher prefix cache exhausted')
            self.values[key]=prefix_for_probe(self.teacher,batch,depth,source='teacher',recorder=recorder);self.misses+=1
        else:self.hits+=1
        return self.values[key]


def sources_for(bases,teacher,cache,mixture,device):
    sources=[];depths={}
    for base in bases:
        for source in sorted(mixture):
            identity=digest({'input_template':base.probe_id,'source':source,'profile':EXECUTION_PROFILE,
                'teacher':teacher.manifest().checkpoint_digest,'seed':17})
            record=replace(base,probe_id=identity,source=source,teacher_digest=teacher.manifest().checkpoint_digest,
                tokenizer_digest=teacher.tokenizer_digest,execution_profile=EXECUTION_PROFILE,backend='torch_'+device,
                prefix_source='candidate' if source=='student' else 'teacher',
                patch_positions=base.patch_positions if source=='donor' else (),
                target_mask=base.target_mask if source=='donor' else (True,)*511,
                patch_mode=base.patch_mode if source=='donor' else 'full',channel_quarter=base.channel_quarter if source=='donor' else None)
            template=_batch(record.recipient_tokens,device=device,document_id=record.recipient_id,cluster_id=record.recipient_component)
            template=replace(template,target_mask=torch.tensor([record.target_mask],device=device,dtype=torch.bool))
            def build(student,teacher,recorder,record=record,template=template):
                current=replace(record,candidate_digest=student.manifest().checkpoint_digest)
                full=replace(template,target_mask=torch.ones_like(template.target_mask))
                if current.source=='student':
                    boundary=prefix_for_probe(student,full,current.depth,source='candidate',recorder=recorder)
                else:boundary=cache.get(full,current.depth,recorder)
                if current.source=='donor':
                    donor=_batch(current.donor_tokens,device=device,document_id=current.donor_id,cluster_id=current.donor_component)
                    boundary=apply_probe(boundary,cache.get(donor,current.depth,recorder),current)
                return boundary,current
            sources.append(ProbeTrainingSource(template,source,'fit',identity,build));depths[identity]=record.depth
    return tuple(sources),depths


def metrics(teacher,student,batches,recorder):
    rows=[]
    for b in batches:
        m=_call(recorder,lambda:stream_metrics(_Dispatch(teacher,_Counts(recorder),'teacher'),
            _Dispatch(student,_Counts(recorder),'student'),b,32))
        if m.valid_targets<=0:raise ValueError('empty natural metric')
        rows.append(dict(component_id=b.cluster_ids[0],document_id=b.document_ids[0],
            kl_sum=m.kl,nll_sum=m.nll,valid_targets=m.valid_targets))
    targets=sum(r['valid_targets'] for r in rows)
    return dict(rows=rows,kl_per_target=sum(r['kl_sum'] for r in rows)/targets,
        nll_per_target=sum(r['nll_sum'] for r in rows)/targets,valid_targets=targets)


def full_rank_floor(teacher,trial,boundary,recorder):
    """Reconstruct the actual FFN at a teacher boundary; no whole-model claim."""
    operator=deepcopy(trial.model.layers[boundary.depth].mlp)
    parent=trial.model.bank[trial.model.partition.reuse_map[boundary.depth]]
    for index,name in enumerate(PROJECTIONS):
        delta=getattr(teacher.layers[boundary.depth].mlp,name).weight.detach()-getattr(parent,name).weight.detach()
        a,b=_call(recorder,lambda:initialize_residual(delta,min(delta.shape),17+index))
        operator.residuals[name]=_Factors(a,b)
    metric=local_discrepancy(teacher,operator,boundary,32,recorder=recorder)
    return metric.kl/metric.valid_targets


def run(config, *, profile_only=False):
    started=time.monotonic();cfg=yaml.safe_load(Path(config).read_text())
    if cfg['schema_version']!=2 or cfg['device']!='mps' or not torch.backends.mps.is_available():raise RuntimeError('qualified native MPS required')
    macos=subprocess.check_output(['sw_vers','-productVersion'],text=True).strip()
    if int(macos.split('.')[0])<26:raise RuntimeError('macOS product version26+ required')
    out=ROOT/cfg['out'];out.mkdir(parents=True,exist_ok=False)
    root=Path(cfg['asset_root']);pair=tuple(cfg['pair']);budget=Budget(**cfg['budget'])
    source_files=[*ROOT.glob('poc/**/*.py'),Path(__file__),ROOT/'protocol/recovery-pilot-v2.md',Path(config).resolve()]
    source={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(source_files)}
    identity=dict(config=cfg,source_files=source,source_sha256=digest(source),execution_profile=EXECUTION_PROFILE)
    parent=RunSpec(method_id='M',method_variant='exploratory-recovery-v2-parent',resolved_config=identity,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_patch_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest(),
        dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
        input_manifests={'teacher':'9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083',
            'data':'aa1e4e65f4b9b6170d74fea68bb57a11848bafb8767a2aca432b486fa12b2026',
            'pool':sha256_file(root/'artifacts/diagnostics/task12/fit-pool.json'),
            'historical_fit_plan':sha256_file(root/'artifacts/diagnostics/task12/plan.json'),
            'historical_development_plan':sha256_file(root/'artifacts/training/task11/learn31-diverse-v2/plan.json'),
            'original_exposure':sha256_file(root/'data/prepared-135m-v2/exposures.jsonl')},
        hardware_policy={'backend':'torch_mps','dtype':'float32','macOS':macos,'execution_profile':EXECUTION_PROFILE},
        seeds={'pilot':cfg['seed']},parent_run_ids=(),budgets=budget)
    ledger=PilotLedger(parent,out/'ledger');ledger.tracker=NativeTracker(budget,ledger.attempt,out)
    result=dict(schema_version=2,status='running',run_spec=parent.to_dict(),arms={},
        scope='historically exposed exploratory recovery, one sentinel/seed; no protected/capability/selector/acquisition/deployment claim',
        environment=dict(macOS=macos,Darwin=platform.release(),python=sys.version,torch=str(torch.__version__)),
        decision_resolution=cfg['decision_resolution'],practical_margin=None)
    write(out/'result.json',result)
    teacher=current=prototype=None
    try:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        torch.set_num_threads(4)
        prepared={}
        def prepare():
            asset=AssetManifest.load(root/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=parent.input_manifests['teacher'])
            prepared['teacher']=load_teacher(asset).to('mps');prepared['tokenizer']=load_tokenizer(asset)
            fit,docs,fb,db,lineage=prepare_populations(root,out,device='mps')
            bases,pool=fit_templates(root,fit,pair)
            by_id={d.document_id:d for d in docs}
            for base in bases:
                doc=by_id[base.recipient_id];tokenizer=prepared['tokenizer']
                if tuple(tokenizer(doc.text,add_special_tokens=False)['input_ids'][:512])!=base.recipient_tokens:
                    raise ValueError('historical recipient tokenizer mismatch')
                a,b=base.source_span;edited=doc.text[:a]+base.replacement+doc.text[b:]
                if tuple(tokenizer(edited,add_special_tokens=False)['input_ids'][:512])!=base.donor_tokens:
                    raise ValueError('historical donor tokenizer mismatch')
            prepared.update(fit=fit,fit_batches=fb,dev_batches=db,bases=bases,lineage=lineage,pool=pool)
            write(out/'population.json',lineage)
        ledger.shared('verified-assets-and-population',CostRecord(),prepare)
        teacher=prepared['teacher'];fit_batches=prepared['fit_batches'];dev_batches=prepared['dev_batches']
        result['population']=prepared['lineage'];result['pool']=prepared['pool']
        if prepared['lineage']['original_exposure_sha256']!=parent.input_manifests['original_exposure']:
            raise ValueError('exposure lineage changed during preparation')
        built={}
        def initialize():
            built['current']=make_bank(teacher,canonicalize_partition([[d] for d in range(len(teacher.layers))],len(teacher.layers)),cfg['rank'],cfg['seed'])
            built['trial']=initialize_merge(built['current'],teacher,pair,prepared['fit'],cfg['seed'],batches=fit_batches,recorder=ledger.tracker)
        ledger.shared('fit-only-parent-initialization',CostRecord(),initialize,dependencies=('verified-assets-and-population',))
        current=built['current'];prototype=built['trial'];prototype.recorder=None;cache=PrefixCache(teacher)
        result['selected_parent']=prototype.selected_parent;result['medoid_scores']=prototype.medoid_scores
        # Numerical qualification only on allowed fit input, before pilot quality.
        floors={}
        def numerical():
            b=fit_batches[0]
            value=metrics(teacher,current,(b,),ledger.tracker)
            floors['singleton_identity_kl_per_target']=value['kl_per_target']
            floors['negative_kl_floor_per_target']=cfg['kl_floor_per_target']
            floors['decision_resolution']=cfg['decision_resolution']
            if abs(value['kl_per_target'])>cfg['decision_resolution']/10:raise ValueError('identity floor above frozen resolution')
            hidden=_call(ledger.tracker,lambda:teacher.final_hidden(b),teacher_forward_calls=1)
            floors['suffix_hidden_errors']={}
            floors['full_rank_local_kl_per_target']={}
            for depth in pair:
                boundary=prefix_for_probe(teacher,b,depth,source='teacher',recorder=ledger.tracker)
                actual=_call(ledger.tracker,lambda:teacher.suffix_hidden(teacher.ff_residual(boundary),boundary),teacher_forward_calls=1)
                error=float((actual-hidden).abs().max().cpu())
                floors['suffix_hidden_errors'][str(depth)]=error
                if error>2e-5:raise ValueError('real context suffix identity floor failed')
                full_rank=full_rank_floor(teacher,prototype,boundary,ledger.tracker)
                floors['full_rank_local_kl_per_target'][str(depth)]=full_rank
                if abs(full_rank)>cfg['decision_resolution']/10:raise ValueError('full-rank floor above frozen resolution')
        ledger.shared('real-context-numerical-floor',CostRecord(),numerical,dependencies=('fit-only-parent-initialization',))
        result['numerics']=floors
        plans={};templates={}
        for name,arm in cfg['arms'].items():
            items,depths=sources_for(prepared['bases'],teacher,cache,arm['mixture'],'mps')
            plans[name]=finite_plan(items,depths,updates=cfg['updates'],weights=arm['mixture'],affected_depths=pair)
            templates[name]=items;write(out/f'target-plan-{name}.json',plans[name].metadata)
        result['target_plan_hashes']={name:p.digest for name,p in plans.items()}
        def child_spec(name,training,plan):
            limits=replace(budget,updates=1 if name=='profile' else cfg['updates'],
                valid_target_presentations=8192 if name=='profile' else cfg['updates']*8192)
            return replace(parent,method_variant='exploratory-recovery-v2-'+name,budgets=limits,
                parent_run_ids=(run_id(parent),),resolved_config={**identity,'training':training.identity(),
                    'target_plan_sha256':plan.digest,'selected_parent':prototype.selected_parent})
        def training_config(mixture):
            return TrainingConfig(max_updates=cfg['updates'],epochs=1,phase='local',affected_depths=pair,
                mixture_weights=mixture,alpha=0,record_norms=True,kl_floor_per_target=cfg['kl_floor_per_target'],
                numerical_profile=EXECUTION_PROFILE+';MPS-requalified-v2',observation_steps=tuple(cfg['observations']))
        # One independent A update; no pilot quality is printed or used for design.
        pcfg=training_config(cfg['arms']['A']['mixture']);pspec=child_spec('profile',pcfg,plans['A'])
        profile_start=time.monotonic()
        with ledger.arm(pspec,dependencies=('real-context-numerical-floor',),forecast=CostRecord(updates=1,valid_target_presentations=8192)) as recorder:
            trial=fixed_parent_copy(prototype,zero_product=False,seed=cfg['seed'])
            prof=fit_model(trial.model,teacher,plans['A'].sources,
                replace(pcfg,run_spec=pspec,stop_after_updates=1,checkpoint_root=out/'profile-checkpoint'),pspec.budgets,recorder)
            if prof.state.step!=1 or prof.state.target_cursor!=8192:raise ValueError('profile did not finish one full update')
            profile_seconds=time.monotonic()-profile_start
            result['profile']=dict(seconds=profile_seconds,cost=recorder.totals.to_dict(),quality_used_for_design=False,
                planned_horizon=cfg['updates'],actual_updates=1)
            del prof,trial
        gc.collect();torch.mps.empty_cache()
        # Explicit overhead allowance for15 full observations and48 durable saves.
        forecast=dict(update_seconds_upper=profile_seconds*2,updates=48,
            observations_and_checkpoint_allowance_seconds=600,
            projected_remaining_seconds=48*profile_seconds*2+600,
            measured_shared_seconds=ledger.tracker.totals.wall_seconds,time_margin_factor=2,
            quality_inspected=False,horizon=cfg['updates'])
        if forecast['projected_remaining_seconds']+time.monotonic()-started>budget.wall_seconds:
            result.update(status='BLOCKED_FORECAST',forecast=forecast);return result
        write(out/'frozen-forecast.json',forecast);result['forecast']=forecast
        print(json.dumps({'event':'forecast_frozen','seconds':forecast['projected_remaining_seconds'],
            'profile_update_seconds':profile_seconds,'parent':prototype.selected_parent}),flush=True)
        if profile_only:result['status']='profile_complete';return result
        for name,arm in cfg['arms'].items():
            plan=plans[name];training=training_config(arm['mixture']);spec=child_spec(name,training,plan)
            with ledger.arm(spec,dependencies=('real-context-numerical-floor',),
                    forecast=CostRecord(updates=cfg['updates'],valid_target_presentations=cfg['updates']*8192,
                        wall_seconds=cfg['updates']*profile_seconds*2)) as recorder:
                trial=fixed_parent_copy(prototype,zero_product=arm['zero_product'],seed=cfg['seed'])
                observations=[];arm_root=out/name;arm_root.mkdir()
                record=dict(pair=pair,rank=cfg['rank'],initialized_digest=trial.initialized_digest,selected_parent=trial.selected_parent,
                    target_plan_sha256=plan.digest,observations=observations,status='running',run_id=run_id(spec))
                result['arms'][name]=record
                def observe(step):
                    start=time.monotonic()
                    fit_metric=metrics(teacher,trial.model,fit_batches,recorder)
                    dev_metric=metrics(teacher,trial.model,dev_batches,recorder)
                    probes=[]
                    for item in templates[name]:
                        boundary,probe=item.build(trial.model,teacher,recorder)
                        m=local_discrepancy(teacher,trial.model.layers[probe.depth].mlp,boundary,32,recorder=recorder)
                        if not m.valid_targets or m.kl < -cfg['kl_floor_per_target']*m.valid_targets:raise ValueError('invalid local observation')
                        probes.append(dict(probe_id=probe.probe_id,component_id=probe.recipient_component,depth=probe.depth,
                            source=probe.source,kl_sum=m.kl,valid_targets=m.valid_targets))
                    trial.fitted_digest=trial.model.manifest().checkpoint_digest
                    _call(recorder,lambda:save_trial_delta(trial,current,arm_root/f'step-{step}-delta'))
                    observations.append(dict(step=step,fit=fit_metric,validation=dev_metric,local_probes=probes,
                        fitted_digest=trial.fitted_digest,seconds=time.monotonic()-start))
                    write(out/'result.json',result)
                    print(json.dumps({'event':'observation','arm':name,'step':step,
                        'fit_kl':fit_metric['kl_per_target'],'validation_kl':dev_metric['kl_per_target'],
                        'validation_nll':dev_metric['nll_per_target']}),flush=True)
                _observe(observe,0,trial.model,teacher)
                fitted=fit_model(trial.model,teacher,plan.sources,
                    replace(training,run_spec=spec,checkpoint_root=arm_root/'checkpoints',on_update=observe),spec.budgets,recorder)
                if fitted.state.step!=cfg['updates'] or fitted.state.target_cursor!=cfg['updates']*8192:
                    raise ValueError('short scientific fit is invalid')
                if any(p.grad is not None or p.requires_grad for p in teacher.parameters()) or teacher.training:
                    raise ValueError('teacher was not frozen')
                record.update(status='complete',updates=fitted.state.step,targets=fitted.state.target_cursor,
                    fit_metrics=fitted.metrics,fitted_digest=fitted.state.candidate_digest,
                    unique_parameter_bytes=sum(p.numel()*p.element_size() for p in trial.model.parameters()),
                    cost=recorder.totals.to_dict(),teacher_frozen=True,checkpoint=asdict(fitted.checkpoint))
                del fitted,trial
            write(out/'result.json',result);gc.collect();torch.mps.empty_cache()
        result.update(status='complete_exploratory',cache=dict(hits=cache.hits,misses=cache.misses))
        return result
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        elapsed=time.monotonic()-started
        finalize_costs(result,ledger.tracker,elapsed=elapsed)
        result['wall_seconds']=elapsed;result['query_cost_graph']=ledger.graph()
        write(out/'result.json',result)
        raise_for_accounting_error(result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/recovery-pilot-v2.yaml')
    parser.add_argument('--profile-only',action='store_true')
    args=parser.parse_args(argv)
    result=run(args.config,profile_only=args.profile_only)
    return 0 if result['status'] in ('complete_exploratory','profile_complete') else 3


if __name__=='__main__':raise SystemExit(main())

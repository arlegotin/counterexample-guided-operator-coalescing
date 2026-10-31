#!/usr/bin/env python3
"""Explicit WP1 executor. Unresolved inputs and unfrozen budgets fail closed."""
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from poc.records import canonical_json
from poc.recovery_study import load_study_config,expected_counts,require_run_ready,NaturalWindowPlan,recovery_depths

ASSET_SHA='9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashed(value):return hashlib.sha256(canonical_json(value)).hexdigest()


def design(config):return {k:v for k,v in config.items() if k!='resource_budget'}


def executed_sources(config_path):
    # Include new/untracked drivers and configs; a tracked diff is insufficient.
    paths=set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
    hashes={p:sha(ROOT/p) for p in sorted(paths) if (ROOT/p).is_file() and
        (p.endswith('.py') or p.endswith('.yaml') or p=='pyproject.toml')}
    code={p:h for p,h in hashes.items() if p.endswith('.py') or p=='pyproject.toml'}
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'tracked_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
        'files':hashes,'config_path':str(config_path),'config_sha256':sha(config_path),
        'implementation_sha256':hashed(code)}


def write_or_verify(path,value):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=canonical_json(value):raise ValueError('resume artifact/contract identity mismatch: '+str(path))
    else:write_new(path,value)


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/('.pending-'+path.name+'-'+uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical_json(value));stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)  # exclusive, atomic publication; never overwrite
    finally:
        if temporary.exists():temporary.unlink()



from poc.budget import BudgetTracker
from poc.continuation_budget import process_peak_bytes
from poc.contracts import CostRecord


class StudyTracker(BudgetTracker):
    """Campaign-wide disk and kernel process RSS high-water guards.

    Accelerator allocation is reported separately; unified-memory counters are
    never added. These are observed guards, not an allocator reservation or an
    exact physical unified-memory census.
    """
    def __init__(self,budget,attempt,root):
        super().__init__(budget,attempt);self.output_root=Path(root);self.device_peaks={}
    def disk(self):return sum(p.stat().st_size for p in self.output_root.rglob('*') if p.is_file())
    def observed(self,cost):
        import torch
        if torch.backends.mps.is_available():
            for key,fn in (('mps_active_bytes',torch.mps.current_allocated_memory),('mps_driver_bytes',torch.mps.driver_allocated_memory)):
                self.device_peaks[key]=max(self.device_peaks.get(key,0),int(fn()))
        return replace(cost,memory_bytes=max(cost.memory_bytes or 0,process_peak_bytes(),*self.device_peaks.values()),
            memory_unavailable_reason=None,disk_bytes=max(cost.disk_bytes,self.disk()))
    def check(self,cost):super().check(self.observed(cost))
    def record(self,cost):super().record(self.observed(cost))
    def write(self,path,value):
        payload=canonical_json(value);self.check(CostRecord(disk_bytes=self.disk()+len(payload)))
        started=time.monotonic()
        try:write_new(path,value)
        finally:self.record(CostRecord(wall_seconds=time.monotonic()-started))


def profile_limits(config,phase,default):
    """Explicit schema4 resource envelope; historical profiles retain defaults."""
    if phase not in ('setup','initialization','arm'):raise ValueError('unknown profile phase')
    if config.get('schema_version') not in (4,5,6):return default
    from poc.contracts import Budget
    budget=Budget(**config['profile_resource_budget'])
    if phase=='setup':
        return replace(budget,updates=0,valid_target_presentations=0,teacher_forward_calls=0,
            suffix_backward_calls=0,student_backward_calls=0,search_proposals=0,search_seconds=0)
    if phase=='initialization':return replace(budget,updates=0,suffix_backward_calls=0,student_backward_calls=0)
    return budget


def family_disk_forecast(config,out,recorder,owned_forecast):
    """Add fixed retained dependencies once; old profile semantics stay intact."""
    if config.get('schema_version') not in (4,5,6):return owned_forecast,0
    owned_now=sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())
    dependency_bytes=recorder.disk()-owned_now
    if dependency_bytes<0:raise ValueError('inconsistent retained-output census')
    return owned_forecast+dependency_bytes,dependency_bytes


def delta_cost(before,after):
    result={}
    for k,v in after.to_dict().items():
        if isinstance(v,(int,float)) and k not in ('memory_bytes','disk_bytes'):
            result[k]=v-before.to_dict()[k]
    return result


def real_shape_controls(config,teacher,student,batch,recorder):
    import torch
    from poc.bank import make_bank,canonicalize_partition
    from poc.losses import stream_metrics
    from poc.probes import local_discrepancy,prefix_for_probe
    from poc.train import _call,_Dispatch,_Counts
    batch=replace(batch,**{n:getattr(batch,n).to(teacher.device) for n in ('input_ids','attention_mask','target_mask','positions')})
    inner=_Counts(recorder);n=int(batch.target_mask.sum())
    def metric(model):
        return _call(recorder,lambda:stream_metrics(_Dispatch(teacher,inner,'teacher'),_Dispatch(model,inner,'student'),batch,32),
                     valid_target_presentations=n)
    with torch.no_grad():
        noop=make_bank(teacher,canonicalize_partition([[i] for i in range(len(teacher.layers))],len(teacher.layers)),config['rank'],config['seeds'][0])
        no_error=metric(noop).kl/n;del noop
        full=metric(student).kl/n
        local=[]
        for d in recovery_depths(config):
            boundary=prefix_for_probe(teacher,batch,d,source='teacher',recorder=recorder)
            local.append(_call(recorder,lambda:local_discrepancy(teacher,student.layers[d].mlp,boundary,32,recorder=inner),valid_target_presentations=n).kl/n)
    result={'no_sharing_kl_raw':no_error,'initial_local_full_gap_abs':abs(sum(local)-full),
            'qualified_floor':config['numerical_profile']['kl_floor_per_target'],
            'decision_resolution':config['numerical_profile']['decision_resolution']}
    if config.get('schema_version') in (5,6):
        result['initial_gap_interpretation']='composition diagnostic; equality is not a numerical correctness requirement'
        import math
        if any(not math.isfinite(value) or value < -result['qualified_floor'] for value in (no_error,full,*local)):
            raise ValueError('composition numerical control violates finite KL or qualified negative floor')
        result['initial_local_full_gap_abs']=None  # quality belongs only in budget-frozen observations
    if abs(no_error)>result['qualified_floor'] or (config.get('schema_version') not in (5,6) and result['initial_local_full_gap_abs']>result['decision_resolution']):
        raise ValueError('real-shape no-op or initial local/full scalar parity failed')
    result['passed']=True
    return result


def finish_attempt(attempt,status,*,reason):
    from poc.records import append_event
    append_event(attempt,{'kind':'attempt_finished','status':status,'reason':reason})
    path=attempt.path/'run.json';raw=json.loads(path.read_text());raw.update(status=status)
    if status=='failed':raw.update(failure_category='execution',failure_details=reason)
    temporary=path.with_suffix('.pending');temporary.write_bytes(canonical_json(raw));os.replace(temporary,path)


def save_recovery_delta(model,anchor_digest,spec,path,recorder):
    from safetensors.torch import save_file
    from poc.recovery_objectives import affected_owners
    from poc.train import _call
    from poc.contracts import CostRecord
    path=Path(path);objective=spec.resolved_config['recovery_objective']
    owners=affected_owners(model,tuple(objective['affected_depths']),owner_scope=objective['owner_scope'])
    if path.exists() and not (path/'manifest.json').is_file():
        failed=path.parent.parent/'failed-artifacts';failed.mkdir(exist_ok=True)
        path.rename(failed/(path.name+'-'+uuid4().hex))
    current=recorder.disk() if hasattr(recorder,'disk') else (sum(p.stat().st_size for p in path.parent.parent.rglob('*') if p.is_file()) if path.parent.parent.exists() else 0)
    recorder.check(CostRecord(disk_bytes=current+sum(p.numel()*p.element_size() for p in owners.values())+65536))
    def save():
        staging=path.parent/('.staging-'+path.name+'-'+uuid4().hex)
        staging.mkdir(parents=True,exist_ok=False)
        save_file({n:p.detach().cpu().contiguous() for n,p in owners.items()},str(staging/'owners.safetensors'))
        write_new(staging/'manifest.json',{'schema_version':1,'anchor_digest':anchor_digest,
            'candidate_digest':model.manifest().checkpoint_digest,'owners':list(owners),
            'weights_sha256':sha(staging/'owners.safetensors'),'run_spec_sha256':hashed(spec.to_dict()),
            'study_contract_sha256':spec.resolved_config['study_contract_sha256']})
        os.rename(staging,path)
    try:_call(recorder,save)
    finally:
        actual=sum(p.stat().st_size for p in path.parent.parent.rglob('*') if p.is_file())
        recorder.record(CostRecord(disk_bytes=actual))


def environment():
    import importlib.metadata
    import torch
    names=('torch','transformers','numpy','scipy','scikit-learn','datasets','safetensors','psutil','pyarrow','pytest','PyYAML','mlx')
    versions={}
    for n in names:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]=None
    import psutil
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'versions':versions,
        'hardware':{'machine':platform.machine(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'physical_memory_bytes':psutil.virtual_memory().total},
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'execution_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','HF_HUB_OFFLINE')},
        'mps_available':torch.backends.mps.is_available(),'lock_status':'observed installed environment; no claim of hash-locked recreation'}


def output_root(config):
    out=Path(config['out']);out=out if out.is_absolute() else ROOT/out
    out=out.resolve()
    if out.parent!=(ROOT/'artifacts').resolve() or out.name!=config['study']:
        raise ValueError('output must be artifacts/<new-study-id> in this worktree')
    if config['study'] in ('continuation-v1','continuation-v2','training','diagnostics','validation'):
        raise ValueError('historical output identities forbidden')
    return out


def preflight(config):
    unresolved=[]
    if config['asset_root'] is None:unresolved.append('asset_root')
    elif not (Path(config['asset_root'])/'teacher-assets.json').is_file():unresolved.append('asset_root/teacher-assets.json')
    for name,p in config['populations'].items():
        if p['path'] is None:unresolved.append('populations.'+name)
        elif not Path(p['path']).is_file() or sha(p['path'])!=p['sha256']:raise ValueError('population path/hash mismatch: '+name)
    if config['resource_budget'] is None:unresolved.append('resource_budget')
    if config.get('schema_version')==4 and config['initialization_checkpoint'] is not None:
        binding=config['initialization_checkpoint']
        if binding.get('schema_version')==2:
            from poc.recovery_clone import verify_population_continuation
            verify_population_continuation(config)
        references={'run_spec':(Path(binding['run_spec_path']),binding['run_spec_sha256']),
            'initialization_record':(Path(binding['initialization_record_path']),binding['initialization_record_sha256']),
            'manifest':(Path(binding['checkpoint_directory'])/'manifest.json',binding['checkpoint_manifest_sha256'])}
        for name,(path,expected) in references.items():
            if not path.is_file():unresolved.append('initialization_checkpoint.'+name)
            elif sha(path)!=expected:raise ValueError('checkpoint reference path/hash mismatch: '+name)
    if config.get('schema_version') in (5,6):
        binding=config['composition']
        if not config['retained_dependency_roots']:unresolved.append('retained_dependency_roots')
        for name,path,expected in (('source_run_spec',binding['source_run_spec_path'],binding['source_run_spec_sha256']),
                ('checkpoint_manifest',str(Path(binding['checkpoint_directory'])/'manifest.json') if binding['checkpoint_directory'] is not None else None,binding['checkpoint_manifest_sha256'])):
            if path is None or not Path(path).is_file():unresolved.append('composition.'+name)
            elif sha(path)!=expected:raise ValueError('composition source metadata hash mismatch: '+name)
    output_root(config)
    banks=len(config['composition']['expected_partition']) if config.get('schema_version') in (5,6) else 29
    result=dict(status='blocked_prerequisites' if unresolved else 'metadata_ready',unresolved=unresolved,
        design_sha256=hashed(design(config)),expected={s:expected_counts(config,s) for s in config['loss_scopes']},
        recovery_resolution_targets={'primary_nats':.01,'sensitivity_nats':[.002,.005],
            'interpretation':'diagnostic information resolution; no capability acceptability margin'},
        historical_ceiling_bytes=66064401,byte_preflight={'K':banks,'rank':config['rank'],
            'fp32_owner_tensor_bytes':4*(54888768+2654208*banks+190080*config['rank']),'complete_bytes':'requires actual export; tensors already exceed B*',
            'purpose':'above-ceiling recovery mechanism; no compression-success claim'})
    if config.get('schema_version')==4:
        result['initialization_checkpoint_verification']='metadata hashes only; initializer verifies tensors and STEP0 before fitting' if config['initialization_checkpoint'] is not None else 'not configured'
    if config.get('schema_version') in (5,6):
        result['composition_checkpoint_verification']='metadata hashes only; completed tensors, teacher/profile and all use-sites verified at initialization before fitting'
        result['retained_family_scope']='declared roots only; ancestor completeness is audited when freezing actual study inputs'
    return result


def load_inputs(config):
    from poc.teacher import AssetManifest
    from poc.recovery_population import load_natural_population
    if config.get('schema_version')==4 and config.get('initialization_checkpoint') is not None:
        from poc.recovery_clone import verify_population_continuation
        verify_population_continuation(config)
    if config.get('schema_version') in (3,4,5,6):
        # Metadata-only denial precedes text/token/asset I/O. Do not clear an
        # exclusion by restricting the fit population for calibration later.
        from poc.data.manifest import load_manifest
        selected=config['populations']['fit']
        fit_metadata=load_manifest(Path(selected['path']),expected_sha256=selected['sha256'],roles={'fit'})
        if fit_metadata.quarantine:raise ValueError('quarantined fit population requires an explicit rebuild before reading')
    asset=AssetManifest.load(Path(config['asset_root'])/'teacher-assets.json',expected_sha256=ASSET_SHA)
    fit_cfg=config['populations']['fit'];tune_cfg=config['populations']['tuning']
    fit,fit_batches=load_natural_population(fit_cfg['path'],fit_cfg['sha256'],role='fit',components=fit_cfg['components'],tokenizer_digest=ASSET_SHA)
    tuning,tuning_batches=load_natural_population(tune_cfg['path'],tune_cfg['sha256'],role='dev',components=tune_cfg['components'],tokenizer_digest=ASSET_SHA)
    if set(fit.membership.values())&set(tuning.membership.values()):raise ValueError('fit/tuning component overlap')
    if set(fit.roles)&set(tuning.roles):raise ValueError('fit/tuning document overlap')
    plans={seed:NaturalWindowPlan.build(fit_batches,updates=config['updates'],seed=seed,
        manifest_hashes={'fit':fit_cfg['sha256'],'tokenizer':ASSET_SHA}) for seed in config['seeds']}
    return asset,fit,fit_batches,tuning_batches,plans


def synchronized(device):
    if device=='mps':
        import torch
        torch.mps.synchronize()


def observation(teacher,student,fit_batches,tune_batches,plan,step,recorder,config):
    import torch
    from poc.losses import stream_metrics
    from poc.probes import local_discrepancy,prefix_for_probe
    from poc.train import _call,_Dispatch,_Counts
    from poc.contracts import CostRecord
    def moved(b):return replace(b,**{n:getattr(b,n).to(student.device) for n in ('input_ids','attention_mask','target_mask','positions')})
    optimizer_update=next(u for i,u in enumerate(plan.iter_updates(student.device)) if i==max(0,step-1))
    groups={'fit_inventory':tuple(moved(b) for b in fit_batches),
            'optimizer_windows':optimizer_update.batches,'tuning':tuple(moved(b) for b in tune_batches)}
    if config.get('observation_policy')=='optimizer_plus_tuning_v2':groups.pop('fit_inventory')
    result={'observation_policy':config.get('observation_policy','full_inventory_v1'),'step':step,'plan_sha256':plan.digest,'candidate_digest':student.manifest().checkpoint_digest,'populations':{}}
    with torch.no_grad():
        for name,batches in groups.items():
            rows=[]
            for b in batches:
                inner=_Counts(recorder)
                metric=_call(recorder,lambda:stream_metrics(_Dispatch(teacher,inner,'teacher'),_Dispatch(student,inner,'student'),b,32),
                    valid_target_presentations=int(b.target_mask.sum()),processed_inputs=1,processed_tokens=int(b.attention_mask.sum()))
                if metric.kl < -config['numerical_profile']['kl_floor_per_target']*metric.valid_targets:
                    raise ValueError('observation KL below qualified floor')
                local=[]
                for d in (recovery_depths(config) if config.get('observation_policy')!='optimizer_plus_tuning_v2' or name=='optimizer_windows' else ()):
                    boundary=prefix_for_probe(teacher,b,d,source='teacher',recorder=recorder)
                    local.append(_call(recorder,lambda:local_discrepancy(teacher,student.layers[d].mlp,boundary,32,recorder=inner),
                        valid_target_presentations=int(b.target_mask.sum())))
                rows.append({'document_id':b.document_ids[0],'component_id':b.cluster_ids[0],
                    'target_positions':b.target_mask[0].nonzero().flatten().cpu().tolist(),
                    'window_sha256':hashed({'tokens':b.input_ids[0].cpu().tolist(),'positions':b.positions[0].cpu().tolist()}),
                    'full':asdict(metric),'local_by_depth':[asdict(m) for m in local]})
            result['populations'][name]=rows
    return result


def make_spec(config,scope,seed,plan,student,sources,env,budget,*,initializer_record=None):
    from poc.contracts import RunSpec
    from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash
    from poc.train import TrainingConfig
    depths=recovery_depths(config);owner_scope=config['owner_scope']
    obj=RecoveryObjectiveSpec(scope,owner_scope,depths,plan.digest,
        owner_names_hash(student,depths,owner_scope=owner_scope),checkpoint_every=config['checkpoint_every'],
        schema_version=3 if owner_scope=='affected_groups' else 1)
    cfg=TrainingConfig(max_updates=config['updates'],alpha=0,affected_depths=depths,
        observation_steps=tuple(x for x in config['observations'] if x),record_norms=True,
        kl_floor_per_target=config['numerical_profile']['kl_floor_per_target'],numerical_profile='recovery-diagnosis-fp32-v1')
    inputs={'raw_plan':plan.digest,'owner_names':obj.owner_names_sha256,'assets':ASSET_SHA,
        'fit':config['populations']['fit']['sha256'],'tuning':config['populations']['tuning']['sha256'],
        'executed_source':hashed(sources),'controls':sha(output_root(config)/'controls.json'),
        'initialized_state':student.manifest().checkpoint_digest,'initialized_owners':hashed({'owners':__import__('poc.recovery_objectives',fromlist=['owner_manifest']).owner_manifest(student,depths,owner_scope=owner_scope)})}
    if config.get('schema_version') in (3,4):
        if initializer_record is None:raise ValueError('v3 requires recorded initializer/calibration bindings')
        from poc.recovery_initializers import RecoveryInitializerSpec
        if (initializer_record.get('spec_sha256')!=RecoveryInitializerSpec(**config['initializer']).digest()
                or initializer_record.get('after_digest')!=student.manifest().checkpoint_digest
                or initializer_record.get('seed')!=seed or initializer_record.get('rank')!=student.rank
                or tuple(initializer_record.get('affected_depths',()))!=tuple(config['pair'])
                or initializer_record.get('teacher_digest')!=student.base_digest):
            raise ValueError('stale or mismatched initializer state/spec/seed bindings')
        if config.get('schema_version')==4:
            clone=initializer_record.get('initialized_checkpoint_clone')
            binding=config['initialization_checkpoint']
            if (binding is None)!=(clone is None):raise ValueError('checkpoint initialization must be explicitly bound')
            if binding is not None:
                if binding.get('schema_version')==2:
                    lineage=clone.get('population_lineage',{})
                    if (clone.get('schema_version')!=2
                            or clone.get('source_updates')!=binding['source_updates']
                            or clone.get('target_updates')!=config['updates']
                            or clone.get('source_raw_plan_sha256')!=clone.get('raw_plan_sha256')
                            or any(lineage.get(role,{}).get('source_sha256')!=binding['source_populations'][role]['sha256']
                                or lineage.get(role,{}).get('target_sha256')!=config['populations'][role]['sha256']
                                for role in ('fit','tuning'))):
                        raise ValueError('checkpoint clone plan/population/horizon bindings differ')
                    target_plan=clone.get('target_raw_plan_sha256')
                    inputs['initialization_source_raw_plan']=clone['source_raw_plan_sha256']
                    inputs['initialization_binding']=hashed(binding)
                else:target_plan=clone.get('raw_plan_sha256')
                if (target_plan!=plan.digest or clone.get('owner_names_sha256')!=obj.owner_names_sha256
                        or clone.get('initialized_owners_sha256')!=inputs['initialized_owners']
                        or clone.get('checkpoint_manifest_sha256')!=binding['checkpoint_manifest_sha256']
                        or clone.get('run_spec_sha256')!=binding['run_spec_sha256']
                        or clone.get('initialization_record_sha256')!=binding['initialization_record_sha256']):
                    raise ValueError('checkpoint clone plan/owners/source bindings differ')
                inputs['initialization_checkpoint']=binding['checkpoint_manifest_sha256']
                inputs['initialization_source_spec']=binding['run_spec_sha256']
        inputs['initializer_spec']=hashed(config['initializer'])
        inputs['initializer_calibration']=hashed(initializer_record)
    parents=()
    if config.get('schema_version') in (5,6):
        binding=config['composition'];record=initializer_record
        if (not isinstance(record,dict) or record.get('kind')!='completed_source_composition_v1'
                or record.get('after_digest')!=student.manifest().checkpoint_digest
                or record.get('seed')!=seed or record.get('rank')!=student.rank
                or tuple(record.get('affected_depths',()))!=tuple(config['pair'])
                or record.get('teacher_digest')!=student.base_digest
                or record.get('reference')!=binding['reference']
                or hashed(record.get('source_binding'))!=hashed(binding)
                or record.get('source_checkpoint_manifest_sha256')!=binding['checkpoint_manifest_sha256']
                or record.get('source_spec_sha256')!=binding['source_run_spec_sha256']
                or record.get('composition',{}).get('source_current_digest')!=record.get('source_candidate_digest')
                or [list(g) for g in student.partition.groups]!=binding['expected_partition']):
            raise ValueError('composition source/reference/state bindings differ')
        inputs['composition_initialization']=hashed(record)
        inputs['composition_checkpoint']=binding['checkpoint_manifest_sha256']
        inputs['composition_source_spec']=binding['source_run_spec_sha256']
        if config.get('schema_version')==6 and (
                type(record.get('schema_version')) is not int or record['schema_version']!=2
                or record.get('trainable_depths')!=list(depths)
                or any(type(d) is not int for d in record.get('trainable_depths',()))
                or record.get('owner_scope')!=owner_scope
                or record.get('owner_names_sha256')!=obj.owner_names_sha256):
            raise ValueError('composition recovery owner binding differs from explicit plan')
        parents=(record['source_run_id'],)
    if config['resource_budget'] is not None:inputs['profile']=config['resource_budget']['frozen_profile_sha256']
    identity=RunSpec('R',config['study']+':'+scope,{'training':cfg.identity(),'recovery_objective':obj.identity(),
        'study_contract_sha256':hashed(config)},sources['commit'],hashed(sources),hashed(env),inputs,
        {'device':config['numerical_profile']['device'],'environment':env,'use_case':'offline English batch1 context512'},
        {'fit':seed},parents,budget)
    return replace(cfg,run_spec=identity),obj


def initialize(config,teacher,fit,fit_batches,seed,recorder):
    if config.get('schema_version') in (5,6):
        from poc.recovery_composition import initialize_composition_recovery
        return initialize_composition_recovery(teacher,config,fit,fit_batches,seed,recorder=recorder)
    from poc.bank import make_bank,canonicalize_partition
    from poc.merge import initialize_merge
    from poc.recovery_data import restricted_population
    if config.get('schema_version') in (3,4):
        if fit.quarantine:raise ValueError('quarantined fit documents require a rebuilt population before initialization')
        if any(role!='fit' for role in fit.roles.values()):raise PermissionError('v3 initialization requires fit-only population')
    if config.get('schema_version')==4 and config['initialization_checkpoint'] is not None:
        from poc.recovery_clone import clone_initialization_checkpoint
        trial=clone_initialization_checkpoint(teacher,config,seed,recorder=recorder,fit_batches=fit_batches)
    else:
        # Choose parent once from a prospective fit-only component subset. The rule
        # and its exact window identities are shared by both loss objectives.
        subset_ids={b.document_ids[0] for b in fit_batches[:config['parent_fit_components']]}
        subset=restricted_population(fit,sorted(subset_ids),role='fit')
        batches=tuple(replace(b,**{n:getattr(b,n).to(teacher.device) for n in ('input_ids','attention_mask','target_mask','positions')})
            for b in fit_batches if b.document_ids[0] in subset_ids)
        current=make_bank(teacher,canonicalize_partition([[i] for i in range(len(teacher.layers))],len(teacher.layers)),config['rank'],seed)
        trial=initialize_merge(current,teacher,tuple(config['pair']),subset,seed,batches=batches,recorder=recorder)
    if config.get('schema_version') in (3,4):
        from poc.recovery_initializers import RecoveryInitializerSpec,initialize_recovery_corrections
        init_spec=RecoveryInitializerSpec(**config['initializer'])
        calibration_ids={b.document_ids[0] for b in fit_batches[:init_spec.calibration_components]}
        calibration_fit=restricted_population(fit,sorted(calibration_ids),role='fit')
        calibration_batches=tuple(replace(b,**{n:getattr(b,n).to(teacher.device) for n in ('input_ids','attention_mask','target_mask','positions')})
            for b in fit_batches if b.document_ids[0] in calibration_ids)
        trial.initializer_record=initialize_recovery_corrections(trial.model,teacher,trial.affected_depths,
            calibration_fit,calibration_batches,seed,init_spec,recorder=recorder)
        if getattr(trial,'clone_provenance',None) is not None:
            trial.initializer_record['initialized_checkpoint_clone']=trial.clone_provenance
        trial.initialized_digest=trial.initializer_record['after_digest']
    return trial


def execute(config,config_path,mode,*,resume=False):
    import torch
    from poc.contracts import Budget
    from poc.budget import BudgetTracker
    from poc.records import start_attempt
    from poc.train import fit_model,TrainingSources
    from poc.teacher import load_teacher
    from poc.recovery_objectives import owner_manifest
    out=output_root(config);sources=executed_sources(config_path);env=environment()
    tracker_class=StudyTracker
    family=None
    if config.get('schema_version') in (5,6):
        from poc.recovery_family import RetainedFamily
        binding=config['composition']
        if binding['source_run_spec_path'] is None or binding['checkpoint_directory'] is None:
            raise ValueError('unresolved completed composition source before data access')
        family=RetainedFamily(str(out),config['retained_dependency_roots'],required_sources=[
            binding['source_run_spec_path'],str(Path(binding['checkpoint_directory'])/'manifest.json')])
        class FamilyTracker(StudyTracker):
            def disk(self):return family.disk()
        tracker_class=FamilyTracker
    if resume and mode!='run':raise ValueError('resume is only defined for a frozen run')
    if mode=='run':require_run_ready(config)
    elif config['asset_root'] is None or any(p['path'] is None for p in config['populations'].values()):
        raise ValueError('unresolved assets/populations; profile cannot invent inputs')
    controls_path=out/'controls.json'
    if not controls_path.is_file():raise ValueError('run controls on the actual planned backend before profile/run')
    controls=json.loads(controls_path.read_text())
    if not controls['backend']['passed'] or controls['backend']['device']!=config['numerical_profile']['device'] or controls['implementation_sha256']!=sources['implementation_sha256'] or controls.get('environment_sha256')!=hashed(env):
        raise ValueError('stale or mismatched backend controls')
    if mode=='run':
        profile_path=out/'profile.json'
        if not profile_path.is_file() or sha(profile_path)!=config['resource_budget']['frozen_profile_sha256']:
            raise ValueError('frozen measured profile mismatch')
        profile=json.loads(profile_path.read_text())
        if family is not None and profile.get('retained_family')!=family.identity():
            raise ValueError('retained family dependency inventory differs from frozen profile')
        if profile['design_sha256']!=hashed(design(config)) or profile['implementation_sha256']!=sources['implementation_sha256'] or profile.get('environment_sha256')!=hashed(env) or profile.get('controls_receipt_sha256')!=sha(controls_path):
            raise ValueError('profile design/source mismatch')
        for scope in config['loss_scopes']:
            b=config['resource_budget']['per_arm'][scope]
            forecast=profile['arms'][scope]
            for key in ('updates','valid_target_presentations','teacher_forward_calls','suffix_backward_calls','student_backward_calls','search_proposals','search_seconds'):
                if b[key]<forecast['forecast_counters'].get(key,0):raise ValueError('frozen budget cannot cover forecast including observations: '+key)
            if b['wall_seconds']<forecast['forecast_seconds_per_seed']:raise ValueError('frozen wall budget cannot cover measured forecast')
            if b['disk_bytes']<profile['forecast_campaign_disk_bytes']:raise ValueError('frozen disk budget cannot cover retained campaign')
            if b['memory_bytes']<forecast['cost']['memory_bytes']:raise ValueError('frozen memory budget below measured process high water')
    stage=out/('profile-attempt-'+uuid4().hex if mode=='profile' else 'runs')
    if resume and not stage.is_dir():raise ValueError('no existing frozen campaign to resume')
    stage.mkdir(parents=True,exist_ok=resume)
    for path,value in ((stage/'executed-source.json',sources),(stage/'environment.json',env),(stage/'contract.json',config)):
        write_or_verify(path,value)
    report={'mode':mode,'design_sha256':hashed(design(config)),'implementation_sha256':sources['implementation_sha256'],'environment_sha256':hashed(env),'controls_receipt_sha256':sha(controls_path),'arms':{}}
    if family is not None:report['retained_family']=family.identity()
    # Offline loading, plan construction and allowed-role reads are one-off
    # preparation, not free orchestration hidden outside the cost graph.
    from poc.contracts import RunSpec
    setup_budget=Budget(0,0,0,0,0,0,0,900,8*1024**3 if mode=='profile' else min(b['disk_bytes'] for b in config['resource_budget']['per_arm'].values()),12*1024**3)
    if mode=='profile' or config.get('schema_version') in (5,6):setup_budget=profile_limits(config,'setup',setup_budget)
    setup_spec=RunSpec('R',config['study']+':offline_setup',{'contract':config},sources['commit'],hashed(sources),hashed(env),
        {'assets':ASSET_SHA,'fit':config['populations']['fit']['sha256'],'tuning':config['populations']['tuning']['sha256']},
        {'device':config['numerical_profile']['device']},{'seed':config['seeds'][0]},(),setup_budget)
    setup_attempt=start_attempt(setup_spec,stage/'ledgers');setup_rec=tracker_class(setup_budget,setup_attempt,out);setup_started=time.monotonic()
    try:
        setup_rec.check(CostRecord())
        asset,fit,fit_batches,tune_batches,plans=load_inputs(config)
        device=config['numerical_profile']['device'];teacher=load_teacher(asset).to(device=device,dtype=torch.float32)
        synchronized(device)
        setup_rec.record(CostRecord(wall_seconds=time.monotonic()-setup_started))
        report['offline_setup']={'cost':setup_rec.totals.to_dict(),'allocation':'one-off shared across all trajectories; repeated setup on retry remains charged'}
        finish_attempt(setup_attempt,'exhausted',reason='verified offline assets/populations/plans loaded')
    except BaseException as exc:
        try:setup_rec.record(CostRecord(wall_seconds=time.monotonic()-setup_started))
        finally:finish_attempt(setup_attempt,'failed',reason=type(exc).__name__+': '+str(exc))
        raise
    seeds=config['seeds'][:1] if mode=='profile' else config['seeds']
    for seed in seeds:
        seed_started=time.monotonic()
        plan=plans[seed];write_or_verify(stage/f'plan-{seed}.json',dict(plan.metadata))
        # Initialization gets its own charged attempt, never free shared work.
        init_budget=Budget(0,100000,200000,0,0,10,900,900,
            8*1024**3 if mode=='profile' else min(b['disk_bytes'] for b in config['resource_budget']['per_arm'].values()),12*1024**3)
        if mode=='profile' or config.get('schema_version') in (5,6):init_budget=profile_limits(config,'initialization',init_budget)
        # The RunSpec needs owners, so use a separate explicitly named prep spec.
        from poc.contracts import RunSpec
        prep_spec=RunSpec('R',config['study']+':parent_preparation',{'contract':config,'plan':plan.digest},sources['commit'],hashed(sources),hashed(env),
            {'assets':ASSET_SHA,'raw_plan':plan.digest}, {'device':device}, {'seed':seed},(),init_budget)
        prep_attempt=start_attempt(prep_spec,stage/'ledgers');prep_rec=tracker_class(init_budget,prep_attempt,out)
        start=seed_started
        try:
            trial=initialize(config,teacher,fit,fit_batches,seed,prep_rec)
            numerical_controls=real_shape_controls(config,teacher,trial.model,fit_batches[0],prep_rec)
            synchronized(device);init_seconds=time.monotonic()-start
            from poc.train import finish_attempt_cost
            finish_attempt_cost(prep_rec,elapsed=init_seconds)
            finish_attempt(prep_attempt,'exhausted',reason='fit-only parent initialization complete')
        except BaseException:
            finish_attempt(prep_attempt,'failed',reason='parent initialization failed');raise
        if mode=='run':
            initialized={'selected_parent':trial.selected_parent,'parent_scores':trial.medoid_scores,
                'initialized_digest':trial.initialized_digest,'owners':owner_manifest(trial.model,recovery_depths(config),owner_scope=config['owner_scope']),
                'shared_cost_allocation':'one-off preparation; counted once, shared equally across two arms'}
            if config.get('schema_version') in (3,4,5,6):
                initialized['initializer_record']=trial.initializer_record
                initialized['shared_cost_allocation']=f"one-off parent/calibration preparation; counted once across {len(config['loss_scopes'])} named objectives; initializer variants are separately costed studies"
            write_or_verify(stage/f'initialization-{seed}.json',initialized)
        for scope in config['loss_scopes']:
            arm=stage/f'{scope}-seed{seed}';arm.mkdir(exist_ok=resume)
            arm_started=time.monotonic()
            model=deepcopy(trial.model)
            if model.manifest().checkpoint_digest!=trial.initialized_digest:raise ValueError('paired initialization clone changed')
            if mode=='profile':
                # Finite outcome-blind profiling caps are independent of quality.
                b=Budget(1,4000000,2000000,100000,100000,0,0,900,8*1024**3,12*1024**3)
                b=profile_limits(config,'arm',b)
            else:b=Budget(**config['resource_budget']['per_arm'][scope])
            cfg,obj=make_spec(config,scope,seed,plan,model,sources,env,b,initializer_record=getattr(trial,'initializer_record',None))
            cfg=replace(cfg,checkpoint_root=arm/'resume')
            attempt=start_attempt(cfg.run_spec,stage/'ledgers');rec=tracker_class(b,attempt,out)
            write_or_verify(arm/'spec.json',cfg.run_spec.to_dict())
            try:
                if mode=='run':
                    if (arm/'result.json').is_file():
                        previous=json.loads((arm/'result.json').read_text())
                        if previous['step']!=config['updates']:raise ValueError('incomplete result cannot be reused')
                        report['arms'][f'{scope}-seed{seed}']={'status':'reused_completed_frozen_run','result':str(arm/'result.json')}
                        finish_attempt(attempt,'exhausted',reason='reuse completed frozen run; no training replay')
                        del model
                        continue
                    def observe(step):
                        existing=arm/'observations'/f'{step:04d}.json'
                        delta=arm/'deltas'/f'{step:04d}'
                        if existing.is_file():
                            try:previous=json.loads(existing.read_text())
                            except (ValueError,UnicodeError):
                                failed=arm/'failed-artifacts';failed.mkdir(exist_ok=True)
                                existing.rename(failed/(existing.name+'-'+uuid4().hex))
                                previous=None
                            if previous is None:return observe(step)
                            if (previous.get('study_contract_sha256')!=hashed(config) or
                                    previous.get('candidate_digest')!=model.manifest().checkpoint_digest):
                                raise ValueError('resume observation state/contract mismatch')
                            if (delta/'manifest.json').is_file():
                                metadata=json.loads((delta/'manifest.json').read_text())
                                if metadata['candidate_digest']!=model.manifest().checkpoint_digest or metadata['weights_sha256']!=sha(delta/'owners.safetensors'):
                                    raise ValueError('resume observation delta integrity mismatch')
                            else:save_recovery_delta(model,trial.initialized_digest,cfg.run_spec,delta,rec)
                            return
                        value=observation(teacher,model,fit_batches,tune_batches,plan,step,rec,config)
                        value['study_contract_sha256']=hashed(config)
                        rec.write(arm/'observations'/f'{step:04d}.json',value)
                        save_recovery_delta(model,trial.initialized_digest,cfg.run_spec,arm/'deltas'/f'{step:04d}',rec)
                    cfg=replace(cfg,on_update=observe)
                    if resume:
                        from poc.checkpoint import resolve_checkpoint,load_checkpoint
                        checkpoint_root=arm/'resume'
                        if (checkpoint_root/'current.json').is_file():ref=resolve_checkpoint(checkpoint_root)
                        elif (checkpoint_root/'anchor'/'current.json').is_file():ref=resolve_checkpoint(checkpoint_root/'anchor')
                        else:ref=None
                        if ref is not None:
                            saved=load_checkpoint(ref,cfg.run_spec)
                            from poc.train import configure_trainable_owners,_optimizer,_restore
                            owners=configure_trainable_owners(model,cfg,objective_spec=obj);optimizer=_optimizer(owners,cfg)
                            _restore(saved,model,optimizer,cfg,TrainingSources(raw_plan=plan),teacher.manifest().checkpoint_digest,obj)
                            del optimizer
                            cfg=replace(cfg,resume=saved)
                            if saved.step in config['observations']:observe(saved.step)
                        else:observe(0)
                    else:
                        from poc.train import _observe
                        _observe(observe,0,model,teacher)
                else:cfg=replace(cfg,stop_after_updates=1)
                synchronized(device);arm_setup_seconds=time.monotonic()-arm_started;start=time.monotonic();before_update=rec.totals;disk_before_update=rec.disk()
                result=fit_model(model,teacher,TrainingSources(raw_plan=plan),cfg,b,rec,objective_spec=obj)
                synchronized(device);update_seconds=time.monotonic()-start;update_cost=delta_cost(before_update,rec.totals)
                update_ledger_bytes=max(0,rec.disk()-disk_before_update-sum(p.stat().st_size for p in (arm/'resume').rglob('*') if p.is_file()))
                if mode=='profile':
                    start=time.monotonic();before_observation=rec.totals;disk_before_observation=rec.disk()
                    # The exact same 128-component tuning observation plus fit
                    # inventory and optimizer-window diagnostics used in run.
                    measured_observation=observation(teacher,model,fit_batches,tune_batches,plan,1,rec,config)
                    observation_payload_bytes=len(canonical_json(measured_observation));del measured_observation
                    synchronized(device);observe_seconds=time.monotonic()-start;observe_cost=delta_cost(before_observation,rec.totals)
                    observation_ledger_bytes=max(0,rec.disk()-disk_before_observation)
                    from poc.checkpoint import save_checkpoint
                    start=time.monotonic();save_checkpoint(result.state,arm/'resume',recorder=rec)
                    synchronized(device);checkpoint_seconds=time.monotonic()-start
                    checkpoints=1+len(set(range(config['checkpoint_every'],config['updates']+1,config['checkpoint_every']))|set(config['observations'][1:]))
                    forecast=init_seconds/2+config['updates']*update_seconds+len(config['observations'])*observe_seconds+checkpoints*checkpoint_seconds
                    from poc.recovery_study import planned_training_work,planned_observation_work
                    planned=[(planned_training_work(p,scope,recovery_depths(config)),
                              planned_observation_work(p,config['observations'],fit_batches,tune_batches,recovery_depths(config),policy=config.get('observation_policy','full_inventory_v1'))) for p in plans.values()]
                    forecast_counters={k:max(t[k]+o[k] for t,o in planned) for k in planned[0][0]}
                    training_scale=max(1.,*(t[k]/(update_cost[k]*config['updates']) for t,_ in planned for k in t if update_cost.get(k,0)>0))
                    observation_scale=max(1.,*(o[k]/(observe_cost[k]*len(config['observations'])) for _,o in planned for k in o if observe_cost.get(k,0)>0))
                    forecast=arm_setup_seconds+init_seconds/len(config['loss_scopes'])+config['updates']*update_seconds*training_scale+len(config['observations'])*observe_seconds*observation_scale+checkpoints*checkpoint_seconds
                    report['arms'][scope]={'numerical_controls':numerical_controls,'forecast_counters':forecast_counters,'profile_updates':1,'planned_updates':config['updates'],'update_seconds_upper_bound_including_setup_and_checkpoints':update_seconds,
                        'observation_seconds':observe_seconds,'checkpoint_seconds':checkpoint_seconds,'planned_checkpoints':checkpoints,
                        'forecast_seconds_per_seed':forecast,'cost':rec.totals.to_dict(),'accelerator_observed_peaks':rec.device_peaks,
                        'quality':'not disclosed or used in resource design','counts':expected_counts(config,scope)}
                else:
                    if result.state.step!=config['updates']:raise ValueError('finite plan did not complete frozen horizon')
                    from poc.checkpoint import resolve_checkpoint
                    final_ref=result.checkpoint or resolve_checkpoint(arm/'resume')
                    rec.write(arm/'result.json',{'step':result.state.step,'raw_target_cursor':result.state.target_cursor,
                        'events':result.state.events,'cost':rec.totals.to_dict(),'final_digest':result.state.candidate_digest,
                        'checkpoint':asdict(final_ref),'scope':'exploratory fidelity recovery; no capability or deployment victory'})
                    report['arms'][f'{scope}-seed{seed}']={'status':'complete','result':str(arm/'result.json')}
                if mode=='profile':
                    # Verify actual checkpoint reload, retain only hashes/size
                    # census and spending, then remove quality-bearing states
                    # before the profile becomes eligible for budget freezing.
                    from poc.checkpoint import resolve_checkpoint,load_checkpoint
                    load_checkpoint(resolve_checkpoint(arm/'resume'),cfg.run_spec)
                    census={str(p.relative_to(arm)): {'bytes':p.stat().st_size,'sha256':sha(p)} for p in (arm/'resume').rglob('*') if p.is_file()}
                    rec.write(arm/'profile-state-census.json',{'files':census,'policy':'temporary profile tensors/events removed before resource freeze'})
                    shutil.rmtree(arm/'resume')
                    report['arms'][scope]['profile_checkpoint_peak_bytes']=sum(x['bytes'] for x in census.values())
                    from poc.recovery_objectives import affected_owners
                    delta_bytes=sum(p.numel()*p.element_size() for p in affected_owners(model,recovery_depths(config),owner_scope=config['owner_scope']).values())
                    report['arms'][scope]['forecast_retained_bytes_per_seed']=(
                        sum(x['bytes'] for x in census.values()) + config['updates']*update_ledger_bytes*training_scale +
                        len(config['observations'])*(observation_ledger_bytes+observation_payload_bytes+delta_bytes+65536)*observation_scale+64*1024**2)
                    report['arms'][scope]['process_rss_highwater_bytes']=process_peak_bytes()
                    report['arms'][scope]['memory_guard_policy']='maximum of observed process RSS high water and accelerator counters; separately reported, never summed; not exact unified physical peak'
                from poc.train import finish_attempt_cost
                finish_attempt_cost(rec,elapsed=time.monotonic()-arm_started)
                rec.record(CostRecord())
                if mode=='profile':report['arms'][scope]['cost']=rec.totals.to_dict()
                finish_attempt(attempt,'exhausted',reason='outcome-blind profile complete' if mode=='profile' else 'completed frozen recovery schedule')
            except BaseException as exc:
                if mode=='profile' and (arm/'resume').exists():
                    # Failed profiles cannot leave loss-bearing checkpoint events
                    # available to choose the next resource envelope either.
                    failed_census={str(p.relative_to(arm)):{'bytes':p.stat().st_size} for p in (arm/'resume').rglob('*') if p.is_file()}
                    try:write_new(arm/'failed-profile-state-census.json',{'files':failed_census,'failure':type(exc).__name__,'quality_state_removed':True})
                    finally:shutil.rmtree(arm/'resume')
                finish_attempt(attempt,'failed',reason=type(exc).__name__+': '+str(exc));raise
            del model,result
        del trial
    if mode=='profile':
        # Each arm retains initial anchor + latest/previous safe state and all
        # compact observations. Use measured checkpoint census conservatively.
        report['forecast_campaign_disk_bytes']=sum(a['forecast_retained_bytes_per_seed'] for a in report['arms'].values())*len(config['seeds'])+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        report['forecast_campaign_disk_bytes'],dependency_bytes=family_disk_forecast(config,out,setup_rec,report['forecast_campaign_disk_bytes'])
        if config.get('schema_version') in (4,5,6):report['retained_dependency_bytes']=dependency_bytes
    destination=out/('profile.json' if mode=='profile' else 'result.json')
    if resume and destination.exists():
        # A completed campaign is immutable; recovery does not overwrite its report.
        prior=json.loads(destination.read_text())
        if prior['design_sha256']!=report['design_sha256']:raise ValueError('completed campaign identity mismatch')
    else:write_new(destination,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('validate','prepare','controls','profile','run'),required=True)
    parser.add_argument('--resume',action='store_true',help='resume an identical frozen run from its verified safe checkpoint')
    args=parser.parse_args(argv);config=load_study_config(args.config)
    if args.resume and args.mode!='run':raise ValueError('--resume requires --mode run')
    if args.mode=='validate':
        result=preflight(config)
        if not any(x!='resource_budget' for x in result['unresolved']):
            _,_,_,_,plans=load_inputs(config)
            result['plan_sha256']={str(s):p.digest for s,p in plans.items()}
    elif args.mode=='prepare':
        from poc.recovery_population import read_fineweb_inventory,prepare_natural_records
        from poc.teacher import AssetManifest,load_tokenizer
        prep=config['preparation']
        if prep is None or config['asset_root'] is None:raise ValueError('unresolved natural preparation inventory/assets')
        target=Path(prep['out']);target=target if target.is_absolute() else ROOT/target;target=target.resolve()
        if target.parent!=(ROOT/'data').resolve() or target.name!=config['study']+'-population':
            raise ValueError('preparation output must be data/<study>-population')
        asset=AssetManifest.load(Path(config['asset_root'])/'teacher-assets.json',expected_sha256=ASSET_SHA)
        tokenizer=load_tokenizer(asset)
        target.mkdir(parents=True,exist_ok=False)
        from poc.data.manifest import load_manifest
        from poc.data.access import open_records
        prior_records=[];prior_hashes=[]
        for prior in prep['prior_manifests']:
            manifest=load_manifest(Path(prior['path']),expected_sha256=prior['sha256'],roles={'fit','dev'})
            prior_hashes.append(prior['sha256'])
            for role,purpose in (('fit','fit'),('dev','selection')):
                allowed=manifest.for_roles(role)
                records=tuple(r for r in allowed.records if r.source=='HuggingFaceFW/fineweb-edu')
                ids={r.document_id for r in records}
                allowed=replace(allowed,records=records,roles={i:role for i in ids},
                    membership={i:allowed.membership[i] for i in ids},quarantine={},exposure_path=str(target/'prior-exposures.jsonl'))
                prior_records.extend(open_records(allowed,purpose=purpose))
        records,acquisition=read_fineweb_inventory(prep['inventory_path'],prep['inventory_sha256'],target/'acquisition',
            source_manifest=ROOT/'provenance/assets-135m-sources.json',max_range_bytes=prep['max_range_bytes'],
            max_request_bytes=prep['max_range_request_bytes'],max_disk_bytes=prep['max_disk_bytes'],
            max_records=prep['max_records'],max_document_bytes=prep['max_document_bytes'])
        occupied=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
        result=prepare_natural_records(records,tokenizer,target/'prepared',tokenizer_digest=ASSET_SHA,
            inventory_sha256=prep['inventory_sha256'],seed=config['seeds'][0],max_document_bytes=prep['max_document_bytes'],
            max_records=prep['max_records'],max_tokens=prep['max_tokens'],max_disk_bytes=prep['max_disk_bytes']-occupied,
            prior_records=prior_records,prior_manifest_hashes=prior_hashes)
    elif args.mode=='controls':
        out=output_root(config);out.mkdir(parents=True,exist_ok=True)
        from poc.recovery_controls import qualify_backend
        source=executed_sources(args.config)
        result={'backend':qualify_backend(config['numerical_profile']['device']),
            'implementation_sha256':source['implementation_sha256'],'executed_source':source,'environment':environment(),'environment_sha256':hashed(environment())}
        history=out/'controls-history';history.mkdir(exist_ok=True)
        if (out/'controls.json').is_file():
            old=json.loads((out/'controls.json').read_text());archive=history/(hashed(old)+'.json')
            if not archive.exists():write_new(archive,old)
        write_new(history/(hashed(result)+'.json'),result)
        temporary=out/('.controls-'+uuid4().hex+'.json');write_new(temporary,result);os.replace(temporary,out/'controls.json')
    else:result=execute(config,args.config,args.mode,resume=args.resume)
    print(json.dumps(result,sort_keys=True,default=list))


if __name__=='__main__':
    try:main()
    except (ValueError,PermissionError,FileNotFoundError,FileExistsError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(2)

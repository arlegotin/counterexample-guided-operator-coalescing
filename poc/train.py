"""Bounded bank fitting; only complete, finite updates are resumable.

Epoch tails use their actual target denominator. The same epoch/order/final-tail
policy must be used by matched methods. Failed partial updates are never stepped
or checkpointed; their precharged work remains in the authoritative recorder.
"""
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
import time

import torch

from poc.bank import BankDecoder, owner_census
from poc.budget import BudgetExceeded, BudgetTracker
from poc.contracts import Batch, Boundary, Budget, CostRecord, DecoderAdapter, MetricSums, ProbeRecord, Recorder, RunSpec, thaw
from poc.losses import backward_distillation
from poc.records import append_event, canonical_json
from poc.probes import backward_local_discrepancy


@dataclass(frozen=True)
class UpdateBatch:
    batches: tuple[Batch, ...]
    valid_targets: int
    # Source index plus offset into its original eligible-target order.
    slices: tuple[tuple[int, int, int], ...] = ()


def iter_update_targets(batches: Iterable[Batch], target_count: int) -> Iterator[UpdateBatch]:
    """Split masks in row-major valid-target order, preserving complete context."""
    if type(target_count) is not int or target_count < 1:
        raise ValueError('target_count must be positive')
    pieces, slices, count = [], [], 0
    for index, batch in enumerate(batches):
        if batch.target_mask.dtype != torch.bool:
            raise ValueError('target masks must be boolean')
        indices = batch.target_mask.reshape(-1).nonzero().flatten()
        start = 0
        while start < indices.numel():
            take = min(target_count-count, indices.numel()-start)
            mask = torch.zeros_like(batch.target_mask).reshape(-1)
            mask[indices[start:start+take]] = True
            pieces.append(replace(batch,target_mask=mask.reshape_as(batch.target_mask)))
            slices.append((index,start,start+take)); count += take; start += take
            if count == target_count:
                yield UpdateBatch(tuple(pieces),count,tuple(slices))
                pieces,slices,count = [],[],0
    if count:
        yield UpdateBatch(tuple(pieces),count,tuple(slices))


@dataclass(frozen=True)
class ProbeTrainingSource:
    """Rebuild producer-keyed provenance after every optimizer change.

    template contains immutable eligible inputs/mask; build returns the *full*
    current boundary and original semantic ProbeRecord, never a sliced record.
    """
    template: Batch
    source: str
    role: str
    identity: str
    build: Callable[[BankDecoder, DecoderAdapter, Recorder], tuple[Boundary, ProbeRecord]]

    def __post_init__(self):
        if self.source not in ('natural','student','donor') or self.role not in ('fit','probe_fit') or not self.identity:
            raise ValueError('fit probes require explicit source, allowed role and identity')


@dataclass(frozen=True)
class TrainingSources:
    natural: tuple[Batch, ...] = ()
    probes: tuple[ProbeTrainingSource, ...] = ()
    natural_role: str = 'fit'
    manifest_hashes: Mapping = field(default_factory=dict)
    raw_plan: object | None = None

    def __post_init__(self):
        if self.raw_plan is not None:
            from poc.recovery_study import NaturalWindowPlan,NaturalWindowSegment
            if not isinstance(self.raw_plan,(NaturalWindowPlan,NaturalWindowSegment)) or self.natural or self.probes:
                raise ValueError('immutable raw plan requires exclusive natural sources')
        if self.natural_role != 'fit': raise PermissionError('natural fitting requires fit role')
        object.__setattr__(self,'natural',tuple(self.natural))
        object.__setattr__(self,'probes',tuple(self.probes))
        if any(not isinstance(p,ProbeTrainingSource) for p in self.probes): raise ValueError('typed probe sources required')

    def digest(self):
        if self.raw_plan is not None:
            if self.natural or self.probes:raise ValueError('raw plan cannot mix legacy sources')
            return self.raw_plan.digest
        h=hashlib.sha256(canonical_json({'manifests':dict(self.manifest_hashes),'natural_role':self.natural_role,
            'probes':[{'source':p.source,'role':p.role,'identity':p.identity} for p in self.probes]}))
        for b in (*self.natural,*(p.template for p in self.probes)):
            h.update(canonical_json({'documents':b.document_ids,'clusters':b.cluster_ids}))
            for x in (b.input_ids,b.attention_mask,b.target_mask,b.positions):
                h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    max_updates: int
    epochs: int = 1
    target_count: int = 8192
    chunk_size: int = 32
    alpha: float = .1
    bank_lr: float = 1e-4
    residual_lr: float = 3e-4
    weight_decay: float = 0
    lr_factor: float = 1
    phase: str = 'global'
    affected_depths: tuple[int,...] = ()
    mixture_weights: Mapping = field(default_factory=lambda:{'natural':1.0})
    kl_floor_per_target: float = 0.0
    numerical_profile: str = 'cpu_fp64_reference'
    run_spec: RunSpec | None = None
    checkpoint_root: Path | None = None
    resume: object | None = None
    stop_after_updates: int | None = None
    observation_steps: tuple[int,...] = ()
    on_update: Callable[[int],None] | None = None
    record_norms: bool = False

    def __post_init__(self):
        if type(self.record_norms) is not bool:raise ValueError('record_norms must be boolean')
        for name in ('max_updates','epochs','target_count','chunk_size'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError(name+' must be positive')
        if self.target_count != 8192: raise ValueError('primary fitting requires exactly8192 targets/update')
        if self.chunk_size != 32: raise ValueError('larger chunks require a separately qualified policy')
        if self.phase not in ('global','local','counterexample'): raise ValueError('unknown fit phase')
        if self.phase != 'global' and not self.affected_depths: raise ValueError('local fit needs affected depths')
        if self.weight_decay not in (0,.01) or self.lr_factor not in (1/3,1,3): raise ValueError('unqualified optimizer policy')
        for name in ('alpha','bank_lr','residual_lr','kl_floor_per_target'):
            v=getattr(self,name)
            if type(v) not in (int,float) or not math.isfinite(v) or v<0: raise ValueError('finite nonnegative '+name+' required')
        weights=dict(self.mixture_weights)
        if not weights or any(k not in ('natural','student','donor') or type(v) not in (int,float) or not math.isfinite(v) or v<=0 for k,v in weights.items()) or not math.isclose(sum(weights.values()),1):
            raise ValueError('explicit positive source weights must sum to one')
        if self.phase=='global' and weights!={'natural':1.0}: raise ValueError('global fitting uses natural text')
        if (tuple(sorted(set(self.observation_steps)))!=tuple(self.observation_steps) or any(type(k) is not int or not 1<=k<=self.max_updates for k in self.observation_steps)):
            raise ValueError('observation steps must be sorted unique update boundaries')
        if self.on_update is not None and not callable(self.on_update):raise ValueError('observation callback must be callable')
        if self.stop_after_updates is not None and (type(self.stop_after_updates) is not int or self.stop_after_updates<1): raise ValueError('invalid bounded stop')
        if self.kl_floor_per_target and self.numerical_profile=='cpu_fp64_reference': raise ValueError('nonzero floor needs qualified scientific profile')

    def identity(self):
        return dict(max_updates=self.max_updates,epochs=self.epochs,target_count=self.target_count,chunk_size=self.chunk_size,
            alpha=self.alpha,bank_lr=self.bank_lr,residual_lr=self.residual_lr,weight_decay=self.weight_decay,
            lr_factor=self.lr_factor,phase=self.phase,affected_depths=list(self.affected_depths),
            mixture_weights=dict(self.mixture_weights),kl_floor_per_target=self.kl_floor_per_target,
            numerical_profile=self.numerical_profile,epoch_tail_policy='actual_denominator_v1',observation_steps=list(self.observation_steps),record_norms=self.record_norms)


@dataclass
class TrainingState:
    spec: RunSpec | None
    model: dict
    model_schema: dict
    optimizer: dict
    scheduler: dict
    rng: dict
    step: int
    target_cursor: int
    probe_cursor: int
    order: tuple
    probe_order: tuple
    partition: tuple
    candidate_digest: str
    visited: tuple
    source_hash: str
    config_hash: str
    budget_totals: dict
    events: list
    teacher_digest: str
    safe_boundary: bool = True


@dataclass
class MethodResult:
    outcome: str
    reason: str
    state: TrainingState
    metrics: tuple[dict,...]
    checkpoint: object | None = None


def configure_trainable_owners(student: BankDecoder, config: TrainingConfig, *, apply=True, objective_spec=None) -> dict:
    if not isinstance(student,BankDecoder): raise ValueError('common fitter requires canonical BankDecoder')
    if getattr(student,'_deployment_inference_only',False):
        raise ValueError('inference-only elided payload cannot train or resume; use the scientific checkpoint')
    if objective_spec is not None:
        from poc.recovery_objectives import affected_owners, owner_names_hash
        owners=affected_owners(student,objective_spec.affected_depths,owner_scope=objective_spec.owner_scope)
        if owner_names_hash(student,objective_spec.affected_depths,owner_scope=objective_spec.owner_scope)!=objective_spec.owner_names_sha256:
            raise ValueError('recovery owner identity mismatch')
        if apply:
            for n,p in student.named_parameters():p.requires_grad_(n in owners);p.grad=None
            student.eval()
        return owners
    census=owner_census(student); parameters=dict(student.named_parameters())
    selected=set(config.affected_depths)
    if any(type(d) is not int or not 0<=d<len(student.layers) for d in selected): raise ValueError('affected depth outside model')
    if config.phase!='global':
        for group in student.partition.groups:
            if selected.intersection(group) and not set(group)<=selected:
                raise ValueError('local fit must include every use of affected canonical bank')
    owners={}
    for row in census:
        train=row.category in ('bank','residual')
        if config.phase!='global':
            if row.category=='bank': train=any(site.startswith(tuple(f'layers.{d}.' for d in selected)) for site in row.use_sites)
            elif row.category=='residual': train=any(row.name.startswith(f'layers.{d}.') for d in selected)
        p=parameters[row.name]
        if apply:p.requires_grad_(train);p.grad=None
        if train: owners[row.name]=p
    if not owners or len({id(p) for p in owners.values()})!=len(owners): raise ValueError('unique trainable owners required')
    if apply:student.eval()
    return owners


def _optimizer(owners,config):
    groups=[]
    for category,lr in [('bank',config.bank_lr),('residual',config.residual_lr)]:
        names=[n for n in owners if n.startswith('bank.')==(category=='bank')]
        if names: groups.append(dict(params=[owners[n] for n in names],owner_ids=names,base_lr=lr*config.lr_factor,lr=lr*config.lr_factor))
    return torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=config.weight_decay,foreach=False)


def _schedule_state(step,max_updates):
    warmup=max(1,math.ceil(.05*max_updates))
    # The upcoming update has a nonzero warmup LR. The last update reaches cosine zero.
    factor=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,max_updates-warmup)))
    return dict(completed_updates=step,planned_updates=max_updates,warmup_updates=warmup,next_factor=factor)


def _schedule(optimizer,step,max_updates):
    state=_schedule_state(step,max_updates)
    for group in optimizer.param_groups:group['lr']=group['base_lr']*state['next_factor']
    return state


class _Guard:
    def __init__(self,recorder,budget):
        self.recorder,self.budget=recorder,budget
        self.total=recorder.aggregate() if hasattr(recorder,'aggregate') else (recorder.totals if hasattr(recorder,'totals') else (recorder.tracker.totals if hasattr(recorder,'tracker') else CostRecord()))
    def _combined(self,cost):
        values=self.total.to_dict()
        for k,v in cost.to_dict().items():
            if k in ('estimation_method','memory_unavailable_reason'):continue
            if k in ('memory_bytes','disk_bytes'):
                if v is not None:values[k]=max(values[k] or 0,v)
            else:values[k]+=v
        return CostRecord(**values)
    def check(self,cost):
        self.recorder.check(cost)
        total=self._combined(cost)
        for k,limit in self.budget.to_dict().items():
            value=getattr(total,k)
            if value is not None and value>limit:raise BudgetExceeded('budget exhausted: '+k)
    def record(self,cost):
        self.total=self._combined(cost)
        self.recorder.record(cost)
        for k,limit in self.budget.to_dict().items():
            value=getattr(self.total,k)
            if value is not None and value>limit:raise BudgetExceeded('budget exhausted: '+k)


def _call(recorder,fn,**counts):
    # Persist counts *before* dispatch; even hard termination cannot give free retries.
    cost=CostRecord(**counts);recorder.check(cost);recorder.record(cost)
    start=time.monotonic()
    try:return fn()
    finally:recorder.record(CostRecord(wall_seconds=time.monotonic()-start))


class _Dispatch:
    def __init__(self,model,recorder,source):self.model,self.recorder,self.source=model,recorder,source
    def final_hidden(self,batch):
        return _call(self.recorder,lambda:self.model.final_hidden(batch),**{self.source+'_forward_calls':1})
    def readout(self,hidden):
        return _call(self.recorder,lambda:self.model.readout(hidden),**{self.source+'_forward_calls':1})


class _Counts:
    def __init__(self,recorder):self.recorder=recorder
    def check(self,cost):self.recorder.check(replace(cost,wall_seconds=0))
    def record(self,cost):self.recorder.record(replace(cost,wall_seconds=0))


def backward_update(student,teacher,update,config,recorder,*,probe_sources=None,objective_spec=None,raw_templates=None):
    """One zero/one step is the caller's responsibility; this only accumulates."""
    metrics=[]
    if objective_spec is not None:
        from poc.recovery_objectives import backward_recovery
        metrics=backward_recovery(student,teacher,update,config,recorder,objective_spec,raw_templates)
    elif config.phase=='global':
        for b in update.batches:
            count=int(b.target_mask.sum().item())
            counts=dict(valid_target_presentations=count,student_backward_calls=1,
                processed_inputs=b.input_ids.shape[0],processed_tokens=int(b.attention_mask.sum().item()))
            inner=_Counts(recorder)
            metrics.append(_call(recorder,lambda:backward_distillation(_Dispatch(teacher,inner,'teacher'),
                _Dispatch(student,inner,'student'),b,alpha=config.alpha,denominator=update.valid_targets,chunk_size=config.chunk_size),**counts))
    else:
        if probe_sources is None:raise ValueError('local fit requires typed probe sources')
        source_counts={k:0 for k in config.mixture_weights}
        for b,(i,_,_) in zip(update.batches,update.slices,strict=True):
            source=probe_sources[i].source
            if source not in source_counts:raise ValueError('unexpected local source')
            source_counts[source]+=int(b.target_mask.sum().item())
        if any(n==0 for n in source_counts.values()):raise ValueError('required mixture source has no update observations')
        for b,(i,_,_) in zip(update.batches,update.slices,strict=True):
            item=probe_sources[i];boundary,record=item.build(student,teacher,recorder)
            if record.source!=item.source or record.recipient_role!=item.role:raise ValueError('probe source/role mismatch')
            if record.candidate_digest!=student.manifest().checkpoint_digest:raise ValueError('stale candidate probe')
            if record.depth not in config.affected_depths:raise ValueError('probe depth is unaffected')
            # Validate build inputs/eligible mask against the saved source template.
            for name in ('input_ids','attention_mask','target_mask','positions'):
                if not torch.equal(getattr(boundary.batch,name),getattr(item.template,name)):raise ValueError('probe template changed')
            weight=config.mixture_weights[item.source]
            # Hooks scale only this backward contribution, before accumulation;
            # no full bank-sized gradient copies or subtraction are necessary.
            hooks=[p.register_hook(lambda gradient,w=weight:gradient*w) for p in student.parameters() if p.requires_grad]
            try:
                m=backward_local_discrepancy(teacher,student.layers[record.depth].mlp,boundary,chunk_size=config.chunk_size,
                    denominator=source_counts[item.source],record=record,recorder=recorder,loss_mask=b.target_mask)
            finally:
                for hook in hooks:hook.remove()
            metrics.append(m)
    for m in metrics:
        if objective_spec is not None and objective_spec.loss_scope=='ffn_mse_sum':
            if not math.isfinite(m.squared_error_sum) or m.squared_error_sum<0:
                raise FloatingPointError('invalid FFN reconstruction sum')
        elif m.kl < -config.kl_floor_per_target*m.valid_targets:raise FloatingPointError('KL exceeds qualified negative floor')
    return metrics


def _rng():
    import random,numpy as np
    result=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.random.get_rng_state())
    if torch.backends.mps.is_available():result['mps']=torch.mps.get_rng_state()
    return result


def _restore_rng(state):
    import random,numpy as np
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.random.set_rng_state(state['torch'])
    if 'mps' in state:
        if not torch.backends.mps.is_available():raise ValueError('saved MPS RNG requires MPS')
        torch.mps.set_rng_state(state['mps'])


def _model_schema(student):
    if getattr(student,'_deployment_inference_only',False):
        raise ValueError('inference-only elided payload cannot become a scientific checkpoint')
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    from dataclasses import asdict
    import json
    cfg=student.config.to_dict() if hasattr(student.config,'to_dict') else asdict(student.config)
    cfg=json.loads(json.dumps(cfg))
    return dict(config=cfg,head_dim=student.head_dim,rope_theta=student.rope_theta,tokenizer_digest=student.tokenizer_digest,
        groups=[list(g) for g in student.partition.groups],rank=student.rank,base_digest=student.base_digest,
        execution_profile=EXECUTION_PROFILE,backend='torch_'+student.device.type,device=str(student.device),
        norm_epsilons={'norm':student.norm.variance_epsilon}|{f'layers.{i}.{s}':getattr(layer,s).variance_epsilon for i,layer in enumerate(student.layers) for s in ('input_layernorm','post_attention_layernorm')},
        owners=[dict(name=r.name,shape=list(r.shape),trainable=r.trainable,category=r.category,use_sites=list(r.use_sites)) for r in owner_census(student)],
        tied=student.lm_head.weight is student.embed_tokens.weight)


def _training_hash(config,objective_spec):
    value={'training':config.identity()}
    if objective_spec is not None:value['recovery_objective']=objective_spec.identity()
    # Retain legacy byte identity exactly.
    return hashlib.sha256(canonical_json(config.identity() if objective_spec is None else value)).hexdigest()


def _source_order(sources):
    if sources.raw_plan is not None:
        from poc.recovery_study import NaturalWindowSegment
        kind='raw_plan_segment_v1' if isinstance(sources.raw_plan,NaturalWindowSegment) else 'raw_plan_v1'
        return (kind,sources.raw_plan.digest)
    return tuple(b.document_ids for b in sources.natural)


def _snapshot(student,optimizer,scheduler,config,sources,step,cursor,probe_cursor,events,recorder,teacher_digest,visited,objective_spec=None):
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    optimizer_state={'state':{},'groups':[]}
    for group in optimizer.param_groups:
        optimizer_state['groups'].append({k:cpu(v) for k,v in group.items() if k!='params'})
        for name,p in zip(group['owner_ids'],group['params'],strict=True):optimizer_state['state'][name]=cpu(optimizer.state[p])
    if hasattr(recorder.recorder,'aggregate'):recorder.total=recorder.recorder.aggregate()
    elif hasattr(recorder.recorder,'totals'):recorder.total=recorder.recorder.totals
    order=_source_order(sources)
    probes=tuple(p.identity for p in sources.probes)
    return TrainingState(config.run_spec,{n:cpu(p) for n,p in student.named_parameters()},_model_schema(student),
        optimizer_state,scheduler,_rng(),step,cursor,probe_cursor,order,probes,student.partition.groups,
        student.manifest().checkpoint_digest,tuple(visited),sources.digest(),_training_hash(config,objective_spec),
        recorder.total.to_dict(),list(events),teacher_digest)


def _saved_digest(student,values,*,device_override=None):
    """Compute executable identity from saved owners without mutating the model."""
    import json
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    config=json.dumps(student.config.to_dict(),sort_keys=True,default=str)
    norms=tuple((f'layers.{i}.{s}',getattr(layer,s).variance_epsilon) for i,layer in enumerate(student.layers)
                for s in ('input_layernorm','post_attention_layernorm'))+(('norm',student.norm.variance_epsilon),)
    h=hashlib.sha256(config.encode())
    h.update(json.dumps([EXECUTION_PROFILE,student.head_dim,student.rope_theta,device_override or str(student.device),norms]).encode())
    canonical={id(p):n for n,p in student.named_parameters()};aliases={}
    for n,p in student.named_parameters(remove_duplicate=False):aliases[n]=canonical[id(p)]
    for n in student.state_dict():
        v=values[aliases[n]].detach().cpu().contiguous()
        h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8).numpy().tobytes())
    owner_aliases={}
    for n,p in student.named_parameters(remove_duplicate=False):owner_aliases.setdefault(id(p),[]).append(n)
    h.update(json.dumps(list(owner_aliases.values()),sort_keys=True).encode())
    return hashlib.sha256(json.dumps((h.hexdigest(),student.partition.groups,student.rank,student.base_digest),sort_keys=True).encode()).hexdigest()


def _recovery_cursors(sources,config,step,objective_spec=None):
    if step==0:return 0,0
    if objective_spec is not None:
        if step>sources.raw_plan.updates:raise ValueError('recovery step exceeds finite plan')
        calls=0
        if objective_spec.loss_scope!='full_student':
            for i,u in enumerate(sources.raw_plan.iter_updates('cpu')):
                if i>=step:break
                calls+=len(u.batches)*len(objective_spec.affected_depths)
        return step*config.target_count,calls
    batches=sources.natural if config.phase=='global' else tuple(p.template for p in sources.probes)
    counts=[int(b.target_mask.sum().item()) for b in batches]
    completed=cursor=probes=0
    for _ in range(config.epochs):
        remaining=config.target_count
        for count in counts:
            while count:
                take=min(remaining,count);count-=take;remaining-=take;cursor+=take
                if config.phase!='global':probes+=1
                if remaining==0:
                    completed+=1
                    if completed==step:return cursor,probes
                    remaining=config.target_count
        if remaining<config.target_count:
            completed+=1
            if completed==step:return cursor,probes
    raise ValueError('recovery step exceeds saved source order')


def _restore(state,student,optimizer,config,sources,teacher_digest,objective_spec=None):
    if not state.safe_boundary:raise ValueError('only complete update boundaries can resume')
    if state.spec!=config.run_spec or state.config_hash!=_training_hash(config,objective_spec) or state.source_hash!=sources.digest():raise ValueError('resume lineage mismatch: config/data/source')
    if state.partition!=student.partition.groups or state.teacher_digest!=teacher_digest:raise ValueError('resume lineage mismatch: map/teacher')
    expected_schema=_model_schema(student)
    if 'device' not in state.model_schema:expected_schema.pop('device') # legacy owned v1
    desired={n for group in optimizer.param_groups for n in group['owner_ids']}
    for owner in expected_schema['owners']:owner['trainable']=owner['name'] in desired
    if state.model_schema!=expected_schema:raise ValueError('resume owner/schema mismatch')
    if type(state.step) is not int or not 0<=state.step<=config.max_updates:raise ValueError('invalid recovery step')
    if any(type(v) is not int or v<0 for v in (state.target_cursor,state.probe_cursor)):raise ValueError('invalid recovery cursor')
    if state.order!=_source_order(sources) or state.probe_order!=tuple(p.identity for p in sources.probes):raise ValueError('recovery order mismatch')
    if (state.target_cursor,state.probe_cursor)!=_recovery_cursors(sources,config,state.step,objective_spec):raise ValueError('recovery cursor mismatch')
    import re
    if (len(state.visited)!=state.step or any(type(d) is not str or not re.fullmatch('[0-9a-f]{64}',d) for d in state.visited)
            or state.step and state.visited[-1]!=state.candidate_digest):raise ValueError('recovery visited history mismatch')
    updates=[e for e in state.events if e.get('kind')=='update']
    if [e.get('step') for e in updates]!=list(range(1,state.step+1)) or sum(e.get('valid_targets',0) for e in updates)!=state.target_cursor:raise ValueError('recovery event cursor mismatch')
    if objective_spec is not None:
        if any(e.get('raw_target_presentations')!=config.target_count or
               e.get('scored_target_presentations')!=config.target_count*objective_spec.scored_multiplier for e in updates):
            raise ValueError('recovery raw/scored accounting mismatch')
        if state.budget_totals.get('raw_corpus_target_presentations',0)<state.target_cursor:
            raise ValueError('recovery raw ledger below completed cursor')
    if state.scheduler!=_schedule_state(state.step,config.max_updates):raise ValueError('recovery scheduler mismatch')
    CostRecord(**state.budget_totals)
    parameters=dict(student.named_parameters())
    if parameters.keys()!=state.model.keys():raise ValueError('resume owner set mismatch')
    # All validation, including owner-indexed moments and RNG schema, precedes
    # mutation of any caller tensor. A bad resume never half-restores a model.
    if len(optimizer.param_groups)!=len(state.optimizer['groups']):raise ValueError('optimizer group mismatch')
    expected_names={n for g in optimizer.param_groups for n in g['owner_ids']}
    if set(state.optimizer['state'])!=expected_names:raise ValueError('optimizer state owner mismatch')
    for group,saved in zip(optimizer.param_groups,state.optimizer['groups'],strict=True):
        if group['owner_ids']!=saved.get('owner_ids'):raise ValueError('optimizer canonical owners mismatch')
        if set(group)-{'params'}!=set(saved):raise ValueError('optimizer fields mismatch')
        for key in set(saved)-{'lr','owner_ids'}:
            if saved[key]!=group[key]:raise ValueError('optimizer hyperparameter mismatch')
        if saved['lr']!=group['base_lr']*state.scheduler['next_factor']:raise ValueError('optimizer scheduler LR mismatch')
        for n,p in zip(group['owner_ids'],group['params'],strict=True):
            moments=state.optimizer['state'][n]
            if not moments and config.phase=='global' and state.step:raise ValueError('missing optimizer moments for used global owner')
            if moments and set(moments)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('optimizer moment fields mismatch')
            for key,v in moments.items():
                if not isinstance(v,torch.Tensor) or not torch.isfinite(v).all().item():raise ValueError('invalid optimizer tensor')
                if key=='step':
                    if v.ndim!=0 or v.dtype not in (torch.float32,torch.float64):raise ValueError('invalid optimizer step scalar')
                    count=v.item()
                    if not float(count).is_integer() or not 1<=count<=state.step or config.phase=='global' and count!=state.step:
                        raise ValueError('optimizer step count disagrees with completed updates')
                elif v.shape!=p.shape or v.dtype!=p.dtype:raise ValueError('optimizer moment schema mismatch')
                elif key=='exp_avg_sq' and (v<0).any().item():raise ValueError('optimizer second moment must be nonnegative')
    for n,p in parameters.items():
        v=state.model[n]
        if v.shape!=p.shape or v.dtype!=p.dtype or not torch.isfinite(v).all().item():raise ValueError('invalid checkpoint owner')
    if _saved_digest(student,state.model)!=state.candidate_digest:raise ValueError('restored candidate digest mismatch')
    required_rng={'python','numpy','torch'}
    if (not isinstance(state.rng,Mapping) or not required_rng<=state.rng.keys()
            or state.rng.keys()-required_rng-{'mps'} or student.device.type=='mps' and 'mps' not in state.rng):
        raise ValueError('incomplete or unsupported recovery RNG schema')
    import random,numpy as np
    try:
        random.Random().setstate(state.rng['python']);np.random.RandomState().set_state(state.rng['numpy'])
        torch.Generator().set_state(state.rng['torch'])
        if 'mps' in state.rng:
            saved_rng=state.rng['mps']
            if (not torch.backends.mps.is_available() or not isinstance(saved_rng,torch.Tensor)
                    or saved_rng.dtype!=torch.uint8 or saved_rng.device.type!='cpu'):
                raise ValueError('invalid saved MPS RNG')
            previous_rng=torch.mps.get_rng_state()
            try:torch.mps.set_rng_state(saved_rng)
            finally:torch.mps.set_rng_state(previous_rng)
    except (TypeError,ValueError,RuntimeError,KeyError,IndexError,OverflowError) as exc:
        raise ValueError('invalid recovery RNG state') from exc
    with torch.no_grad():
        for n,p in parameters.items():p.copy_(state.model[n].to(p.device))
    for group,saved in zip(optimizer.param_groups,state.optimizer['groups'],strict=True):
        for key,value in saved.items():group[key]=value
        for n,p in zip(group['owner_ids'],group['params'],strict=True):
            # Adam step is deliberately CPU unless capturable; moments follow owner device.
            optimizer.state[p]={k:(v.clone() if k=='step' or v.device==p.device else v.to(p.device)) if isinstance(v,torch.Tensor) else v for k,v in state.optimizer['state'][n].items()}
    _restore_rng(state.rng)


def _observe(callback,step,student,teacher):
    """Read-only observation restores RNG/modes even when the observer fails."""
    rng=_rng()
    modules=tuple(student.modules())+tuple(teacher.modules())
    modes=tuple(module.training for module in modules)
    flags=tuple((p,p.requires_grad) for model in (student,teacher) for p in model.parameters())
    before=(student.manifest().checkpoint_digest,teacher.manifest().checkpoint_digest)
    try:
        with torch.no_grad():callback(step)
    finally:
        _restore_rng(rng)
        for module,mode in zip(modules,modes,strict=True):module.training=mode
        for parameter,flag in flags:parameter.requires_grad_(flag)
    if before!=(student.manifest().checkpoint_digest,teacher.manifest().checkpoint_digest):
        raise ValueError('observation mutated model state; fitting invalidated')


def finish_attempt_cost(recorder,*,elapsed):
    """Top up the current attempt only; prior same-RunSpec attempts stay charged."""
    if isinstance(recorder,BudgetTracker):
        extra=max(0,elapsed-recorder.attempt_cost.wall_seconds)
        if extra:recorder.record(CostRecord(wall_seconds=extra))


def _fit_model(student: BankDecoder, teacher: DecoderAdapter, data: TrainingSources,
              config: TrainingConfig, budget: Budget, recorder: Recorder, *, objective_spec=None) -> MethodResult:
    if config.run_spec is not None:
        if config.run_spec.budgets!=budget:raise ValueError('immutable RunSpec budget mismatch')
        if thaw(config.run_spec.resolved_config).get('training')!=config.identity():raise ValueError('RunSpec must bind the complete resolved training config')
    if objective_spec is not None:
        from poc.recovery_objectives import RecoveryObjectiveSpec
        if not isinstance(objective_spec,RecoveryObjectiveSpec):raise ValueError('typed objective spec required')
        if config.run_spec is None:raise ValueError('recovery requires a bound RunSpec')
        resolved=thaw(config.run_spec.resolved_config)
        if resolved.get('recovery_objective')!=__import__('json').loads(canonical_json(objective_spec.identity())):
            raise ValueError('RunSpec recovery objective identity mismatch')
        if (data.raw_plan is None or data.raw_plan.digest!=objective_spec.raw_plan_sha256
                or data.raw_plan.updates!=config.max_updates or data.raw_plan.raw_targets_per_update!=config.target_count
                or config.epochs!=1 or config.alpha!=0 or config.phase!='global'
                or config.affected_depths!=objective_spec.affected_depths or config.mixture_weights!={'natural':1.0}):
            raise ValueError('recovery finite raw plan/objective normalization mismatch')
        if (config.run_spec.input_manifests.get('raw_plan')!=objective_spec.raw_plan_sha256 or
                config.run_spec.input_manifests.get('owner_names')!=objective_spec.owner_names_sha256):
            raise ValueError('RunSpec must bind raw plan and exact owner names')
    elif data.raw_plan is not None:raise ValueError('raw plan requires explicit recovery objective')
    if any(p.requires_grad for p in teacher.parameters()):raise ValueError('teacher must remain frozen')
    if objective_spec is None and config.phase=='global' and data.probes:raise ValueError('global fit receives natural source only')
    if objective_spec is None and config.phase!='global' and data.natural:raise ValueError('local natural states require typed full probe provenance')
    if student.dtype not in (torch.float32,torch.float64):raise ValueError('scientific fitting requires FP32 or CPU FP64 reference')
    if student.dtype==torch.float64 and student.device.type!='cpu':raise ValueError('FP64 reference is CPU only')
    if list(student.named_buffers()):raise ValueError('unmodeled persistent buffers are unsupported')
    owners=configure_trainable_owners(student,config,apply=config.resume is None,objective_spec=objective_spec);optimizer=_optimizer(owners,config)
    guard=_Guard(recorder,budget);teacher_digest=teacher.manifest().checkpoint_digest
    step=cursor=probe_cursor=0;events=[];visited=[];metrics=[];checkpoint_ref=None
    if config.resume is not None:
        saved=config.resume
        # Authoritative recorder may contain interrupted work after this boundary.
        for k,v in saved.budget_totals.items():
            current=getattr(guard.total,k,None)
            if k not in ('memory_bytes','disk_bytes','memory_unavailable_reason','estimation_method') and isinstance(v,(int,float)) and current<v:
                raise ValueError('authoritative spending cannot roll back to checkpoint')
        _restore(saved,student,optimizer,config,data,teacher_digest,objective_spec)
        configure_trainable_owners(student,config,objective_spec=objective_spec)
        step,cursor,probe_cursor=saved.step,saved.target_cursor,saved.probe_cursor
        events=list(saved.events);visited=list(saved.visited)
    schedule=_schedule(optimizer,step,config.max_updates)
    batches=data.natural if config.phase=='global' else tuple(p.template for p in data.probes)
    if objective_spec is None and config.phase!='global' and set(p.source for p in data.probes)!=set(config.mixture_weights):raise ValueError('required mixture source absent')
    for i,b in enumerate(batches):
        if not b.target_mask.any().item() and not any(e.get('kind')=='skipped' and e.get('source_index')==i for e in events):
            events.append(dict(kind='skipped',source_index=i,reason='no_valid_targets'))
    safe_state=None
    if objective_spec is not None:
        safe_state=_snapshot(student,optimizer,schedule,config,data,step,cursor,probe_cursor,events,guard,teacher_digest,visited,objective_spec)
        if config.checkpoint_root is not None and config.resume is None:
            from poc.checkpoint import save_checkpoint
            checkpoint_ref=save_checkpoint(safe_state,config.checkpoint_root/'anchor',recorder=guard)
            safe_state=None
    started=time.monotonic();initial_step=step;sequence_index=0
    wall_baseline=recorder.attempt_cost.wall_seconds if isinstance(recorder,BudgetTracker) else 0
    try:
        for epoch in range(config.epochs):
            for update in (data.raw_plan.iter_updates(student.device) if objective_spec is not None else iter_update_targets(batches,config.target_count)):
                sequence_index+=1
                if sequence_index<=step:continue
                if step>=config.max_updates or (config.stop_after_updates is not None and step-initial_step>=config.stop_after_updates):break
                guard.check(CostRecord(updates=1,valid_target_presentations=update.valid_targets*(objective_spec.scored_multiplier if objective_spec is not None else 1)))
                optimizer.zero_grad(set_to_none=True)
                try:
                    sums=backward_update(student,teacher,update,config,guard,probe_sources=data.probes,objective_spec=objective_spec,raw_templates=data.raw_plan)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in owners.values()):raise FloatingPointError('nonfinite owner gradient')
                    if not any(p.grad is not None for p in owners.values()):raise ValueError('update has no trainable gradient')
                    if config.record_norms:
                        gradient_norm=math.sqrt(math.fsum(float(p.grad.detach().float().square().sum().item()) for p in owners.values() if p.grad is not None))
                        before_step={n:p.detach().clone() for n,p in owners.items()}
                    actual_lrs={('bank' if g['owner_ids'][0].startswith('bank.') else 'residual'):g['lr'] for g in optimizer.param_groups} if objective_spec is not None else None
                    torch.nn.utils.clip_grad_norm_(list(owners.values()),1,error_if_nonfinite=True)
                    _call(guard,optimizer.step,updates=1)
                    if config.record_norms:
                        update_norm=math.sqrt(math.fsum(float((p.detach()-before_step[n]).float().square().sum().item()) for n,p in owners.items()))
                        del before_step
                        if not math.isfinite(gradient_norm) or not math.isfinite(update_norm):raise FloatingPointError('nonfinite optimizer norm')
                    if any(not torch.isfinite(p).all().item() for p in owners.values()):raise FloatingPointError('nonfinite updated owner; no checkpoint')
                except BaseException:
                    optimizer.zero_grad(set_to_none=True)
                    raise
                optimizer.zero_grad(set_to_none=True)
                step+=1;cursor+=update.valid_targets
                if objective_spec is not None:
                    if objective_spec.loss_scope!='full_student':probe_cursor+=len(update.batches)*len(objective_spec.affected_depths)
                elif config.phase!='global':probe_cursor+=len(update.batches)
                schedule=_schedule(optimizer,step,config.max_updates)
                reconstruction=objective_spec is not None and objective_spec.loss_scope=='ffn_mse_sum'
                row=dict(kind='update',step=step,epoch=epoch,valid_targets=update.valid_targets,
                    kl_sum=None if reconstruction else math.fsum(s.kl for s in sums),
                    nll_sum=None if reconstruction else math.fsum(s.nll for s in sums),
                    source_counts={s:sum(int(b.target_mask.sum().item()) for b,(i,_,_) in zip(update.batches,update.slices,strict=True) if (s=='natural' if objective_spec is not None or config.phase=='global' else data.probes[i].source==s)) for s in config.mixture_weights})
                events.append(row);metrics.append(row)
                if objective_spec is not None:
                    row.update(raw_target_presentations=update.valid_targets,
                        scored_target_presentations=sum(m.valid_targets for m in sums),
                        source_counts={'natural':update.valid_targets},
                        weighted_objective=(math.fsum(s.squared_error_sum for s in sums) if reconstruction else row['kl_sum'])*objective_spec.scale/update.valid_targets,
                        raw_plan_sha256=objective_spec.raw_plan_sha256,learning_rates=actual_lrs)
                    if reconstruction:
                        row.update(objective_name='ffn_output_mse_sum',reconstruction_sum=math.fsum(s.squared_error_sum for s in sums),
                            reconstruction_by_depth={str(d):{'output_width':next(s.output_width for s in sums if s.depth==d),'squared_error_sum':math.fsum(s.squared_error_sum for s in sums if s.depth==d),
                                'valid_targets':sum(s.valid_targets for s in sums if s.depth==d)} for d in objective_spec.affected_depths})
                        row['source_reconstruction_sums']={'natural':row['reconstruction_sum']}
                    else:row['source_kl_sums']={'natural':row['kl_sum']}
                    if config.record_norms:row['clip_applied']=gradient_norm>1
                else:
                    source_sums={s:math.fsum(m.kl for m,(i,_,_) in zip(sums,update.slices,strict=True)
                        if (s=='natural' if config.phase=='global' else data.probes[i].source==s)) for s in config.mixture_weights}
                    row['source_kl_sums']=source_sums
                    row['weighted_objective']=((row['kl_sum']+config.alpha*row['nll_sum'])/update.valid_targets
                        if config.phase=='global' else math.fsum(config.mixture_weights[s]*source_sums[s]/row['source_counts'][s] for s in config.mixture_weights))
                if config.record_norms:row.update(gradient_norm_pre_clip=gradient_norm,parameter_update_norm=update_norm)
                visited.append(student.manifest().checkpoint_digest)
                if config.checkpoint_root is not None and (objective_spec is None or step%objective_spec.checkpoint_every==0
                        or step in config.observation_steps or step==config.max_updates
                        or config.stop_after_updates is not None and step-initial_step>=config.stop_after_updates):
                    from poc.checkpoint import save_checkpoint
                    state=_snapshot(student,optimizer,schedule,config,data,step,cursor,probe_cursor,events,guard,teacher_digest,visited,objective_spec)
                    checkpoint_ref=save_checkpoint(state,config.checkpoint_root,recorder=guard)
                    del state
                if step in config.observation_steps and config.on_update is not None:_observe(config.on_update,step,student,teacher)
                guard.check(CostRecord())
            if step>=config.max_updates or (config.stop_after_updates is not None and step-initial_step>=config.stop_after_updates):break
    except BaseException:
        if objective_spec is not None:
            from poc.checkpoint import load_checkpoint
            saved=load_checkpoint(checkpoint_ref,config.run_spec) if checkpoint_ref is not None else safe_state
            _restore(saved,student,optimizer,config,data,teacher_digest,objective_spec)
            configure_trainable_owners(student,config,objective_spec=objective_spec)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        # Nested dispatches measure only counted work, so fill orchestration time once.
        if isinstance(recorder,BudgetTracker):
            finish_attempt_cost(recorder,elapsed=wall_baseline+time.monotonic()-started)
            guard.total=recorder.totals
    state=_snapshot(student,optimizer,schedule,config,data,step,cursor,probe_cursor,events,guard,teacher_digest,visited,objective_spec)
    outcome='invalid' if not cursor else 'exhausted'
    reason='no_valid_targets' if not cursor else 'declared bounded fitting completed; compression target not evaluated'
    return MethodResult(outcome,reason,state,tuple(metrics),checkpoint_ref)


def fit_model(student: BankDecoder, teacher: DecoderAdapter, data: TrainingSources,
              config: TrainingConfig, budget: Budget, recorder: Recorder, *, objective_spec=None) -> MethodResult:
    """Charge the entire call, including setup failures and final snapshots."""
    started=time.monotonic()
    baseline=recorder.attempt_cost.wall_seconds if isinstance(recorder,BudgetTracker) else 0
    result=None
    try:
        result=_fit_model(student,teacher,data,config,budget,recorder,objective_spec=objective_spec)
        return result
    finally:
        if isinstance(recorder,BudgetTracker):
            finish_attempt_cost(recorder,elapsed=baseline+time.monotonic()-started)
            if result is not None:result.state.budget_totals=recorder.totals.to_dict()

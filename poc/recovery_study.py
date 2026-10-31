"""Immutable CPU window inventory and finite raw-target presentation plans.

Plans contain scalar tuples, never caller-owned tensors. Repeated contexts are
indices into one canonical inventory. Digests therefore remain valid without
rehashing expanded device tensors at every safe boundary.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import random
from types import MappingProxyType

import torch

from poc.contracts import Batch, freeze
from poc.records import canonical_json


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class NaturalWindow:
    input_ids: tuple
    attention_mask: tuple
    target_mask: tuple
    positions: tuple
    document_ids: tuple
    cluster_ids: tuple

    def __post_init__(self):
        for name in ('input_ids','attention_mask','target_mask','positions','document_ids','cluster_ids'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if any(type(v)!=int or v<0 for v in (*self.input_ids,*self.positions)):
            raise ValueError('natural tokens/positions must be nonnegative integers')
        if any(type(v)!=bool for v in (*self.attention_mask,*self.target_mask)):
            raise ValueError('natural masks must be boolean')

    @classmethod
    def from_batch(cls, batch):
        if batch.input_ids.shape[0]!=1 or batch.input_ids.shape[1]<2:
            raise ValueError('one complete context per natural template required')
        if batch.attention_mask.dtype!=torch.bool or batch.target_mask.dtype!=torch.bool:
            raise ValueError('boolean masks required')
        if batch.input_ids.dtype!=torch.int64 or batch.positions.dtype!=torch.int64:
            raise ValueError('int64 tokens/positions required')
        ids,att,mask,pos=(tuple(getattr(batch,n)[0].detach().cpu().tolist()) for n in
                          ('input_ids','attention_mask','target_mask','positions'))
        if pos!=tuple(range(len(ids))) or any(i<0 for i in ids):
            raise ValueError('nonnegative tokens and reset positions required')
        if any(mask[i] and not (att[i] and att[i+1]) for i in range(len(mask))):
            raise ValueError('loss target outside attention mask')
        if not any(mask):raise ValueError('empty natural window')
        # ProbeRecord's full original template has canonical valid-pair eligibility;
        # an update may select a subset only through its separate loss mask.
        if mask!=tuple(att[i] and att[i+1] for i in range(len(mask))):
            raise ValueError('natural template requires full valid-target eligibility')
        if any(not att[i] and any(att[i:]) for i in range(len(att))):
            raise ValueError('right padded attention required')
        return cls(ids,att,mask,pos,tuple(batch.document_ids),tuple(batch.cluster_ids))

    def batch(self, device='cpu'):
        return Batch(torch.tensor([self.input_ids],dtype=torch.int64,device=device),
            torch.tensor([self.attention_mask],dtype=torch.bool,device=device),
            torch.tensor([self.target_mask],dtype=torch.bool,device=device),
            torch.tensor([self.positions],dtype=torch.int64,device=device),self.document_ids,self.cluster_ids)


@dataclass(frozen=True)
class NaturalWindowPlan:
    templates: tuple[NaturalWindow,...]
    order: tuple[int,...]
    updates: int
    raw_targets_per_update: int
    seed: int
    manifests: tuple[tuple[str,str],...]
    digest: str

    def __post_init__(self):
        for name in ('templates','order','manifests'):
            object.__setattr__(self,name,tuple(tuple(v) if name=='manifests' else v for v in getattr(self,name)))
        body=dict(schema_version=1,templates=[asdict(w) for w in self.templates],order=self.order,updates=self.updates,
            raw_targets_per_update=self.raw_targets_per_update,seed=self.seed,manifests=dict(self.manifests),policy='shuffled_cycles_full_context_v1')
        if digest(body)!=self.digest:raise ValueError('natural plan digest/order mismatch')
        if not self.order or any(type(i)!=int or not 0<=i<len(self.templates) for i in self.order):
            raise ValueError('invalid natural plan order')
        counts=[sum(self.templates[i].target_mask) for i in self.order]
        required=self.updates*self.raw_targets_per_update
        if not sum(counts[:-1])<required<=sum(counts):raise ValueError('incomplete or overlong finite natural plan')

    @classmethod
    def build(cls,batches,*,updates,raw_targets_per_update=8192,seed,manifest_hashes):
        if any(type(v)!=int or v<1 for v in (updates,raw_targets_per_update)):
            raise ValueError('positive finite update/target plan required')
        if type(seed)!=int or seed<0:raise ValueError('nonnegative integer seed required')
        if raw_targets_per_update!=8192:raise ValueError('qualified plan requires 8192 raw targets/update')
        templates=tuple(NaturalWindow.from_batch(b) for b in batches)
        if not templates:raise ValueError('nonempty fit inventory required')
        ids=[w.document_ids[0] for w in templates]
        if len(set(ids))!=len(ids):
            # Multiple windows from the same document would change this study's
            # one-window/component design; do not silently deduplicate them.
            raise ValueError('unique document templates required')
        components=[w.cluster_ids[0] for w in templates]
        if len(set(components))!=len(components):raise ValueError('one window per component required')
        import re
        manifests=tuple(sorted(manifest_hashes.items()))
        if not manifests or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for _,v in manifests):
            raise ValueError('exact input manifest hashes required')
        order=[];total=0;rng=random.Random(seed)
        while total<updates*raw_targets_per_update:
            cycle=list(range(len(templates)));rng.shuffle(cycle)
            for index in cycle:
                order.append(index);total+=sum(templates[index].target_mask)
                if total>=updates*raw_targets_per_update:break
        body=dict(schema_version=1,templates=[asdict(w) for w in templates],order=order,updates=updates,
            raw_targets_per_update=raw_targets_per_update,seed=seed,manifests=dict(manifests),policy='shuffled_cycles_full_context_v1')
        return cls(templates,tuple(order),updates,raw_targets_per_update,seed,manifests,digest(body))

    @property
    def metadata(self):
        used=set();remaining=self.updates*self.raw_targets_per_update
        for i in self.order:
            positions=[j for j,m in enumerate(self.templates[i].target_mask) if m]
            take=min(len(positions),remaining);used.update((i,j) for j in positions[:take]);remaining-=take
        return freeze(dict(schema_version=1,updates=self.updates,raw_targets_per_update=self.raw_targets_per_update,
            raw_target_presentations=self.updates*self.raw_targets_per_update,
            unique_target_positions=len(used),inventory_unique_target_positions=sum(sum(w.target_mask) for w in self.templates),
            unique_components=len({i for i,_ in used}),unique_windows=len({i for i,_ in used}),inventory_unique_components=len(self.templates),
            template_hashes=[digest(asdict(w)) for w in self.templates],order=self.order,
            manifests=dict(self.manifests),seed=self.seed,plan_sha256=self.digest))

    def template_at(self,index,device='cpu'):
        return self.templates[self.order[index]].batch(device)

    def batches(self,device='cpu'):
        remaining=self.updates*self.raw_targets_per_update
        for i in range(len(self.order)):
            b=self.template_at(i,device);count=int(b.target_mask.sum())
            if count>remaining:
                indices=b.target_mask.flatten().nonzero().flatten()[:remaining]
                mask=torch.zeros_like(b.target_mask).flatten();mask[indices]=True
                b=replace(b,target_mask=mask.reshape_as(b.target_mask))
            yield b
            remaining-=min(count,remaining)

    def iter_updates(self,device='cpu'):
        from poc.train import iter_update_targets
        yield from iter_update_targets(self.batches(device),self.raw_targets_per_update)


def _segment_presentations(parent,start_update,end_update):
    if not isinstance(parent,NaturalWindowPlan):
        raise ValueError('segment requires one immutable original natural plan')
    if (type(start_update) is not int or type(end_update) is not int
            or not 0<=start_update<end_update<=parent.updates):
        raise ValueError('phase bounds must select complete updates inside the parent plan')
    lower=start_update*parent.raw_targets_per_update
    upper=end_update*parent.raw_targets_per_update
    cursor=0;rows=[]
    for index,template_index in enumerate(parent.order):
        count=sum(parent.templates[template_index].target_mask)
        begin=max(lower,cursor);end=min(upper,cursor+count)
        if begin<end:rows.append((index,begin-cursor,end-cursor))
        cursor+=count
        if cursor>=upper:break
    if sum(end-start for _,start,end in rows)!=upper-lower:
        raise ValueError('phase plan does not cover its explicit raw targets')
    return tuple(rows)


@dataclass(frozen=True)
class NaturalWindowSegment:
    """Versioned consecutive phase of a frozen stream, with full original contexts.

    Presentation tuples contain parent-order index and eligible-target offsets.
    They never retain device tensors or silently restart the parent shuffle.
    """
    parent: NaturalWindowPlan
    start_update: int
    end_update: int
    presentations: tuple[tuple[int,int,int],...]
    digest: str

    def __post_init__(self):
        object.__setattr__(self,'presentations',tuple(tuple(row) for row in self.presentations))
        expected=_segment_presentations(self.parent,self.start_update,self.end_update)
        if self.presentations!=expected or self.digest!=digest(self._identity()):
            raise ValueError('phase content/order identity mismatch')

    def _identity(self):
        return dict(schema_version=1,kind='natural_window_segment_v1',parent_plan_sha256=self.parent.digest,
            start_update=self.start_update,end_update=self.end_update,presentations=self.presentations)

    @classmethod
    def build(cls,parent,*,start_update,end_update):
        rows=_segment_presentations(parent,start_update,end_update)
        identity=dict(schema_version=1,kind='natural_window_segment_v1',parent_plan_sha256=parent.digest,
            start_update=start_update,end_update=end_update,presentations=rows)
        return cls(parent,start_update,end_update,rows,digest(identity))

    @property
    def updates(self):return self.end_update-self.start_update

    @property
    def raw_targets_per_update(self):return self.parent.raw_targets_per_update

    @property
    def metadata(self):
        used=set()
        for index,start,end in self.presentations:
            template_index=self.parent.order[index]
            positions=[i for i,m in enumerate(self.parent.templates[template_index].target_mask) if m]
            used.update((template_index,i) for i in positions[start:end])
        return freeze(dict(**self._identity(),plan_sha256=self.digest,updates=self.updates,
            raw_targets_per_update=self.raw_targets_per_update,
            raw_target_presentations=self.updates*self.raw_targets_per_update,
            unique_target_positions=len(used),unique_windows=len({i for i,_ in used}),
            unique_components=len({i for i,_ in used}),manifests=dict(self.parent.manifests)))

    def template_at(self,index,device='cpu'):
        if type(index) is not int or not 0<=index<len(self.presentations):
            raise ValueError('phase presentation index outside plan')
        return self.parent.template_at(self.presentations[index][0],device)

    def batches(self,device='cpu'):
        for index,(_,start,end) in enumerate(self.presentations):
            b=self.template_at(index,device)
            indices=b.target_mask.flatten().nonzero().flatten()[start:end]
            mask=torch.zeros_like(b.target_mask).flatten();mask[indices]=True
            yield replace(b,target_mask=mask.reshape_as(b.target_mask))

    def iter_updates(self,device='cpu'):
        from poc.train import iter_update_targets
        yield from iter_update_targets(self.batches(device),self.raw_targets_per_update)


_STUDY_FIELDS={'schema_version','study','asset_root','out','pair','rank','seeds','loss_scopes','owner_scope',
    'alpha','raw_targets_per_update','updates','observations','checkpoint_every','parent_fit_components',
    'populations','numerical_profile','resource_budget','preparation'}


def _fields(value,expected,name):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise ValueError(f'{name}: missing or unknown fields: {set(value) ^ set(expected) if isinstance(value,dict) else "not mapping"}')


def recovery_depths(config):
    """Explicit recovery sites; the new merge has its own unchanged pair field."""
    return tuple(config['trainable_depths'] if config.get('schema_version')==6 else config['pair'])


def _validate_recovery_groups(config,partition):
    depths=config['trainable_depths']
    if (not isinstance(depths,list) or not depths or any(type(d) is not int for d in depths)
            or depths!=sorted(set(depths)) or not 0<=depths[0]<=depths[-1]<len(partition.reuse_map)):
        raise ValueError('sorted unique recovery trainable_depths required')
    chosen=set(depths)
    groups=[g for g in partition.groups if chosen.intersection(g)]
    if (not set(config['pair'])<=chosen or any(len(g)<2 or not set(g)<=chosen for g in groups)):
        raise ValueError('recovery owners must cover the new merge and complete shared groups; singleton owners are unrelated')
    scope=config['owner_scope']
    if scope not in ('affected_group','affected_groups') or (len(groups)!=1 if scope=='affected_group' else len(groups)<2):
        raise ValueError('explicit recovery owner_scope disagrees with selected complete groups')


def load_study_config(path):
    """Strict versioned schemas. Unresolved prerequisites are legal for preflight only."""
    from pathlib import Path
    import re,yaml,math
    from poc.contracts import Budget
    raw=yaml.safe_load(Path(path).read_text())
    version=raw.get('schema_version') if isinstance(raw,dict) else None
    _fields(raw,_STUDY_FIELDS
        |({'observation_policy'} if version in (2,3,4,5,6) else set())
        |({'initializer'} if version in (3,4,5,6) else set())
        |({'profile_resource_budget','initialization_checkpoint'} if version in (4,5,6) else set())
        |({'composition','retained_dependency_roots'} if version in (5,6) else set())
        |({'trainable_depths'} if version==6 else set()),'study')
    if version not in (1,2,3,4,5,6):raise ValueError('unknown study version')
    if version in (2,3,4,5,6) and raw['observation_policy']!='optimizer_plus_tuning_v2':raise ValueError('unknown v2 observation policy')
    if not isinstance(raw['study'],str) or not re.fullmatch('[a-zA-Z0-9_-]+',raw['study']):raise ValueError('explicit study ID required')
    for name in ('updates','rank','checkpoint_every','parent_fit_components'):
        if type(raw[name])!=int or raw[name]<1:raise ValueError('positive '+name+' required')
    if version in (1,2):
        if raw['pair']!=[0,15] or raw['rank']!=8:raise ValueError('v1 is the fixed (0,15)/rank8 WP1 sentinel')
    else:
        pair=raw['pair']
        if (not isinstance(pair,list) or not pair or any(type(d)!=int for d in pair)
                or pair!=sorted(set(pair)) or not 0<=pair[0]<=pair[-1]<30
                or (len(pair)<(3 if version==5 else 2) if version in (5,6) else len(pair)!=2) or raw['rank'] not in (8,16,32,64)):
            raise ValueError('canonical depth pair/group and rank8/16/32/64 required')
        if version in (5,6):
            roots=raw['retained_dependency_roots']
            if not isinstance(roots,list) or any(not isinstance(p,str) or not p.strip() for p in roots):raise ValueError('explicit retained dependency root list required')
            from poc.recovery_composition import CompositionSourceSpec
            from poc.bank import canonicalize_partition
            if raw['initializer'] is not None or raw['initialization_checkpoint'] is not None:
                raise ValueError('composition reference must not be relabeled as a STEP0 clone or projection initializer')
            _fields(raw['composition'],set(CompositionSourceSpec.__dataclass_fields__),'composition')
            source=CompositionSourceSpec(**raw['composition'])
            partition=canonicalize_partition(source.expected_partition,30)
            if tuple(pair) not in partition.groups or not set(source.merge_pair)<=set(pair) or source.forced_parent not in pair:
                raise ValueError('composition must name one complete final group')
            if version==6:_validate_recovery_groups(raw,partition)
        else:
            from poc.recovery_initializers import RecoveryInitializerSpec
            _fields(raw['initializer'],set(RecoveryInitializerSpec.__dataclass_fields__),'initializer')
            RecoveryInitializerSpec(**raw['initializer'])
    if (version!=6 and raw['owner_scope']!='affected_group') or raw['alpha']!=0 or raw['raw_targets_per_update']!=8192:
        raise ValueError('v1 recovery owner/normalization mismatch')
    for name in ('seeds','observations'):
        v=raw[name]
        if not isinstance(v,list) or not v or any(type(x)!=int or x<0 for x in v) or len(set(v))!=len(v):
            raise ValueError('unique nonnegative '+name+' required')
    if version in (5,6) and len(raw['seeds'])!=1:raise ValueError('composition source requires exactly one paired seed')
    if raw['observations']!=sorted(raw['observations']) or raw['observations'][0]!=0 or raw['observations'][-1]!=raw['updates']:
        raise ValueError('observations must cover initialization and completed horizon')
    if version in (1,2):
        if raw['loss_scopes']!=['local_sum','full_student']:raise ValueError('v1 primary includes exactly two matched objective families')
    elif (not isinstance(raw['loss_scopes'],list) or not raw['loss_scopes']
          or len(set(raw['loss_scopes']))!=len(raw['loss_scopes'])
          or any(scope not in ('local_sum','local_mean','full_student') for scope in raw['loss_scopes'])):
        raise ValueError('v3 requires explicit unique objective families; local_mean is separately costed')
    _fields(raw['populations'],{'fit','tuning'},'populations')
    for role,n in (('fit',512),('tuning',128)):
        p=raw['populations'][role];_fields(p,{'path','sha256','components'},role)
        if p['components']!=n:raise ValueError('v1 component counts must be 512/128; amendments need new schema')
        if (p['path'] is None)!=(p['sha256'] is None):raise ValueError('population path/hash must resolve together')
        if p['sha256'] is not None and not re.fullmatch('[0-9a-f]{64}',p['sha256']):raise ValueError('population SHA256 required')
    if version in (3,4) and max(raw['parent_fit_components'],raw['initializer']['calibration_components'])>raw['populations']['fit']['components']:
        raise ValueError('initialization calibration cannot exceed declared fit population')
    if version in (5,6) and raw['parent_fit_components']>raw['populations']['fit']['components']:raise ValueError('composition parent fit count exceeds fit population')
    n=raw['numerical_profile'];_fields(n,{'device','dtype','chunk_size','kl_floor_per_target','decision_resolution'},'numerical profile')
    if n['device'] not in ('cpu','mps') or n['dtype']!='float32' or n['chunk_size']!=32:raise ValueError('unqualified numerical path')
    for key in ('kl_floor_per_target','decision_resolution'):
        if type(n[key]) not in (float,int) or not math.isfinite(n[key]) or n[key]<=0:raise ValueError('finite numerical resolution required')
    for name in ('asset_root','out'):
        if raw[name] is not None and (not isinstance(raw[name],str) or not raw[name].strip()):raise ValueError('invalid '+name)
    if not raw['out']:raise ValueError('isolated output required')
    if raw['resource_budget'] is not None:
        _fields(raw['resource_budget'],{'frozen_profile_sha256','per_arm'},'resource budget')
        if not re.fullmatch('[0-9a-f]{64}',raw['resource_budget']['frozen_profile_sha256']):raise ValueError('profile hash required')
        _fields(raw['resource_budget']['per_arm'],set(raw['loss_scopes']),'per-arm budget')
        for scope in raw['loss_scopes']:
            b=Budget(**raw['resource_budget']['per_arm'][scope]);expected=expected_counts(raw,scope)
            if b.updates<raw['updates'] or b.valid_target_presentations<expected['scored_target_presentations']:
                raise ValueError('budget cannot complete declared horizon')
    if version in (4,5,6):
        profile=raw['profile_resource_budget']
        _fields(profile,set(Budget.__dataclass_fields__),'profile resource budget')
        b=Budget(**profile)
        if b.updates!=1 or any(getattr(b,key)<=0 for key in ('wall_seconds','memory_bytes','disk_bytes')):
            raise ValueError('profile resource budget requires exactly one update and positive wall/memory/disk limits')
        # One update plus one optimizer-window/tuning observation. Local objectives
        # score each affected depth; the full-student objective scores once.
        minimum=(65408+8192*(1+len(recovery_depths(raw))))+8192*max(1 if scope=='full_student' else len(recovery_depths(raw)) for scope in raw['loss_scopes'])
        if b.valid_target_presentations<minimum:
            raise ValueError('profile resource budget cannot complete one update and observation')
        checkpoint=raw['initialization_checkpoint']
        if checkpoint is not None:
            from poc.recovery_clone import validate_initialization_binding
            validate_initialization_binding(checkpoint)
            if raw['initializer']['method']!='parent_svd' or len(raw['seeds'])!=1:
                raise ValueError('checkpoint clone requires parent_svd and exactly one seed')
    if raw['preparation'] is not None:
        _fields(raw['preparation'],{'inventory_path','inventory_sha256','out','max_document_bytes','max_records','max_tokens','max_disk_bytes','max_range_bytes','max_range_request_bytes','prior_manifests','max_wall_seconds','memory_bytes'},'preparation')
    return raw


def expected_counts(config,scope):
    if scope not in ('local_sum','local_mean','full_student'):raise ValueError('unknown scope')
    raw=config['updates']*config['raw_targets_per_update']
    return dict(updates=config['updates'],raw_target_presentations=raw,
        scored_target_presentations=raw*(1 if scope=='full_student' else len(recovery_depths(config))),
        observations=len(config['observations']),checkpoint_every=config['checkpoint_every'])


def require_run_ready(config):
    if config['asset_root'] is None or any(p['path'] is None for p in config['populations'].values()):
        raise ValueError('unresolved asset/population prerequisites')
    if config['resource_budget'] is None:raise ValueError('resource budget must be frozen from a measured profile')
    if config.get('schema_version') in (5,6) and (not config['retained_dependency_roots'] or any(config['composition'][key] is None for key in ('source_run_spec_path','checkpoint_directory'))):
        raise ValueError('unresolved composition checkpoint/source spec')


def _work_fields():
    return dict(valid_target_presentations=0,raw_corpus_target_presentations=0,
        teacher_forward_calls=0,student_forward_calls=0,student_backward_calls=0,
        suffix_backward_calls=0,processed_inputs=0,processed_tokens=0,updates=0,
        search_proposals=0,search_seconds=0)


def _batch_work(batch,scope,depth_count,*,training):
    count=int(batch.target_mask.sum());chunks=len(set((batch.target_mask.flatten().nonzero().flatten()//32).tolist()))
    result=_work_fields()
    if scope=='full_student':
        result.update(valid_target_presentations=count,teacher_forward_calls=1+chunks,
            student_forward_calls=1+chunks,student_backward_calls=int(training),
            processed_inputs=1,processed_tokens=int(batch.attention_mask.sum()))
    elif scope=='ffn_mse_sum':
        if not training:
            raise ValueError('reconstruction scope is training-only')
        result.update(valid_target_presentations=count*depth_count,
            teacher_forward_calls=1+2*depth_count,
            student_forward_calls=depth_count,student_backward_calls=depth_count,
            processed_inputs=1,processed_tokens=int(batch.attention_mask.sum()))
    elif scope in ('local_sum','local_mean'):
        result.update(valid_target_presentations=count*depth_count,teacher_forward_calls=depth_count*(4+2*chunks),
            student_forward_calls=depth_count,student_backward_calls=depth_count*int(training),
            suffix_backward_calls=depth_count*int(training),processed_inputs=depth_count,
            processed_tokens=depth_count*int(batch.attention_mask.sum()))
    else:
        raise ValueError('unknown objective scope in work forecast')
    if training:result['raw_corpus_target_presentations']=count
    return result


def planned_training_work(plan,scope,depths):
    """Exact dispatch units across all slices; update one is not a multiplier."""
    total=_work_fields()
    for update in plan.iter_updates('cpu'):
        total['updates']+=1
        for b in update.batches:
            for k,v in _batch_work(b,scope,len(depths),training=True).items():total[k]+=v
    return total


def planned_observation_work(plan,observations,fit_batches,tune_batches,depths,*,policy="full_inventory_v1"):
    total=_work_fields();indices={max(0,step-1) for step in observations}
    optimizer={i:u for i,u in enumerate(plan.iter_updates('cpu')) if i in indices}
    for step in observations:
        if policy=='optimizer_plus_tuning_v2':
            groups=((optimizer[max(0,step-1)].batches,('full_student','local_sum')),(tune_batches,('full_student',)))
        elif policy=='full_inventory_v1':
            groups=(((*fit_batches,*tune_batches,*optimizer[max(0,step-1)].batches),('full_student','local_sum')),)
        else:raise ValueError('unknown observation work policy')
        for batches,scopes in groups:
            for b in batches:
                for scope in scopes:
                    for k,v in _batch_work(b,scope,len(depths),training=False).items():total[k]+=v
    return total

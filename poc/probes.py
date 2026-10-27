"""Aligned, role-restricted counterexamples and bounded full-state cache keys.

Only materialized role-reader outputs enter donor generation. Every attempted
edit is returned, including alignment rejects. Numerical branches run through
one unchanged prefix boundary and the frozen teacher suffix; score selection
never changes the declared population or discards unfiltered stress records.
"""
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import random
import re
import time
from typing import Protocol

import torch

from poc.contracts import (Batch, Boundary, CostRecord, DecoderAdapter, FFNOperator,
                           MetricSums, ProbeRecord, Recorder, freeze)
from poc.data.manifest import DocumentRecord
from poc.losses import backward_distillation, stream_metrics
from poc.records import canonical_json


class Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs) -> Mapping: ...


@dataclass(frozen=True)
class PathologyObservation:
    document_id: str
    component_id: str
    role: str
    norm: float
    entropy: float
    depth: int

    def __post_init__(self):
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError('pathology observations require original depth')
        if not self.document_id or not self.component_id:
            raise ValueError('pathology observation requires document/component provenance')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in (self.norm,self.entropy)):
            raise ValueError('teacher norm/entropy observations must be finite and nonnegative')


@dataclass(frozen=True)
class PathologyThresholds:
    observations: tuple[PathologyObservation, ...]
    lower_quantile: float
    upper_quantile: float

    def __post_init__(self):
        object.__setattr__(self, 'observations', tuple(self.observations))
        if not self.observations or any(x.role != 'fit' for x in self.observations):
            raise ValueError('pathology thresholds require nonempty fit-only teacher observations')
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError('explicit ordered pathology quantiles required')

    @property
    def digest(self):
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()

    def interval(self, name, depth):
        values = sorted(getattr(x, name) for x in self.observations if x.depth == depth)
        if not values: raise ValueError('missing fit pathology observations for depth')
        def quantile(q):
            k = q*(len(values)-1); a = int(k); b = min(a+1,len(values)-1)
            return values[a]+(values[b]-values[a])*(k-a)
        return quantile(self.lower_quantile), quantile(self.upper_quantile)


def fit_pathology_thresholds(observations: Iterable[PathologyObservation], *, lower_quantile: float,
                              upper_quantile: float) -> PathologyThresholds:
    """Freeze teacher fit statistics before any candidate comparisons."""
    return PathologyThresholds(tuple(observations), lower_quantile, upper_quantile)


@dataclass(frozen=True)
class ProbePolicy:
    # No scientific or resource setting below has a guessed default.
    selection_mode: str
    seed: int
    per_document_cap: int
    generated_edits_per_document: int
    random_component_fraction: float
    depth_regions: tuple[tuple[int, ...], ...]
    depth_region_balance: tuple[float, ...]
    selection_count: int
    max_pool_records: int
    max_cache_entries: int
    max_cache_bytes: int
    thresholds: PathologyThresholds
    roles: Mapping[str, str]
    components: Mapping[str, str]
    allowed_roles: tuple[str, ...]
    use: str
    teacher_digest: str
    candidate_digest: str
    tokenizer_digest: str
    dtype: str
    backend: str
    boundary_version: str
    execution_profile: str
    context_tokens: int
    recorder: Recorder | None = None

    def __post_init__(self):
        if self.selection_mode not in ('active','seeded_random'):
            raise ValueError('selection_mode must be active or seeded_random')
        for name in ('per_document_cap','generated_edits_per_document','selection_count','max_pool_records','max_cache_entries','max_cache_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'explicit positive {name} required')
        if type(self.seed) is not int or self.seed < 0 or not 0 <= self.random_component_fraction <= 1:
            raise ValueError('explicit seed and random component fraction required')
        object.__setattr__(self,'depth_regions',tuple(tuple(x) for x in self.depth_regions))
        object.__setattr__(self,'depth_region_balance',tuple(self.depth_region_balance))
        flat = [x for region in self.depth_regions for x in region]
        if (not flat or len(flat) != len(set(flat)) or any(type(x) is not int or x < 0 for x in flat)
                or any(not region for region in self.depth_regions)
                or len(self.depth_region_balance) != len(self.depth_regions)
                or any(not math.isfinite(x) or x < 0 for x in self.depth_region_balance)
                or not math.isclose(math.fsum(self.depth_region_balance),1.,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('explicit disjoint depth regions and normalized balance required')
        if not isinstance(self.thresholds,PathologyThresholds):
            raise ValueError('fit-derived pathology thresholds must be resolved before selection')
        if not set(flat) <= {x.depth for x in self.thresholds.observations}:
            raise ValueError('policy includes uncalibrated original depths')
        roles = {'fit':{'fit','probe_fit'}, 'rank':{'probe_score'}}
        if self.use not in roles or not self.allowed_roles or not set(self.allowed_roles) <= roles[self.use]:
            raise ValueError('declared probe roles violate use matrix')
        object.__setattr__(self,'allowed_roles',tuple(self.allowed_roles))
        for name in ('roles','components'):
            object.__setattr__(self,name,freeze(getattr(self,name)))
        if set(self.roles) != set(self.components):
            raise ValueError('role/component manifest identities must match')
        if type(self.context_tokens) is not int or self.context_tokens < 2:
            raise ValueError('explicit context_tokens >=2 required')
        for name in ('teacher_digest','candidate_digest','tokenizer_digest'):
            if not re.fullmatch('[0-9a-f]{64}',getattr(self,name)):
                raise ValueError('exact model/tokenizer identities required')


@contextmanager
def _charged(recorder, **counts):
    """Preflight then durably charge dispatched work on success, error or interrupt."""
    if recorder is not None:
        recorder.check(CostRecord(**counts))
    start = time.monotonic()
    try:
        yield counts
    finally:
        if recorder is not None:
            elapsed = time.monotonic()-start
            recorder.record(CostRecord(wall_seconds=elapsed,search_seconds=elapsed if counts.get('search_proposals') else 0,**counts))


def _encode(tokenizer, text):
    raw = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = tuple(raw['input_ids']), tuple(tuple(x) for x in raw['offset_mapping'])
    if len(ids) != len(offsets) or any(len(x) != 2 or not 0 <= x[0] <= x[1] <= len(text) for x in offsets):
        raise ValueError('tokenizer must provide exact token character offsets')
    return ids,offsets


def _edits(text):
    # Closed auditable dictionaries avoid unlogged model-generated donor text.
    names = {'Alice':'Clara','Bob':'Tom','John':'Paul','Mary':'Jane'}
    plurals = {'cat':'cats','cats':'cat','dog':'dogs','dogs':'dog','book':'books','books':'book',
               'is':'are','are':'is','has':'have','have':'has'}
    for match in re.finditer(r'\b(?:[A-Za-z]+|[0-9]+)\b',text):
        word = match.group()
        if word in names: yield 'name',match.span(),names[word]
        elif word.isdecimal(): yield 'number',match.span(),str(int(word)+1)
        elif word in plurals: yield 'plural',match.span(),plurals[word]


def generate_donors(records: Iterable[DocumentRecord], tokenizer: Tokenizer,
                    policy: ProbePolicy) -> Iterator[ProbeRecord]:
    """Generate capped edit attempts; rejected alignments are population rows.

    A cap counts attempted text edits, not accepted edits. Each accepted edit
    cycles one token/span/quarter mask at each declared depth. The explicit
    compact-pool cap is a hard error, never a silent truncation.
    """
    emitted = 0
    for doc in records:
        role = policy.roles.get(doc.document_id)
        if role not in policy.allowed_roles:
            raise PermissionError('source role not permitted by resolved probe policy')
        if doc.text is None:
            raise ValueError('donors require materialized output of the role/exposure reader')
        with _charged(policy.recorder, processed_inputs=1) as charged:
            original, offsets = _encode(tokenizer,doc.text)
            charged['processed_tokens'] = len(original)
        n = min(len(original),policy.context_tokens)
        if n < 2:
            raise ValueError('donor document has no valid next-token population')
        original_length = len(original)
        original_digest = hashlib.sha256(canonical_json({'tokens':original})).hexdigest()
        all_original, all_offsets = original,offsets
        original = original[:n]; offsets = offsets[:n]
        # Stable source order makes caps auditable; seed governs mask selection/mining.
        for attempt,(kind,span,replacement) in enumerate(_edits(doc.text)):
            if attempt >= policy.generated_edits_per_document: break
            edited = doc.text[:span[0]]+replacement+doc.text[span[1]:]
            with _charged(policy.recorder,search_proposals=1,processed_inputs=1) as charged:
                donor_all,donor_offsets_all = _encode(tokenizer,edited)
                charged['processed_tokens'] = len(donor_all)
                donor = donor_all[:policy.context_tokens]
                donor_offsets = donor_offsets_all[:policy.context_tokens]
                donor_span = (span[0],span[0]+len(replacement))
                pp = tuple(i for i,(a,b) in enumerate(offsets) if a < span[1] and b > span[0])
                reason = None
                if len(donor_all) != original_length or len(donor) != n:
                    reason = 'token_length_mismatch'
                elif not pp: reason = 'edit_outside_context'
                else:
                    full_pp = tuple(i for i,(a,b) in enumerate(all_offsets) if a < span[1] and b > span[0])
                    dpp = tuple(i for i,(a,b) in enumerate(donor_offsets_all) if a < donor_span[1] and b > donor_span[0])
                    shift = len(replacement)-(span[1]-span[0])
                    def recipient_offset(x):
                        if x <= span[0]: return x
                        if x >= donor_span[1]: return x-shift
                        return x if shift == 0 else None
                    aligned_offsets = tuple((recipient_offset(a),recipient_offset(b)) for a,b in donor_offsets_all)
                    if (dpp != full_pp or aligned_offsets != all_offsets
                            or any(all_original[i] != donor_all[i] for i in range(original_length) if i not in full_pp)):
                        reason = 'token_position_mismatch'
                depths = [depth for region in policy.depth_regions for depth in region]
                rows = []
                masks = [('full',pp[:1],None),('span',pp,None)]+[('quarter',pp[:1],q) for q in range(4)]
                if reason: masks,depths = [('span',(),None)],depths[:1]
                for depth_index,depth in enumerate(depths):
                    # One stratified mask/depth per edit, bounded instead of Cartesian expansion.
                    choice = (policy.seed+emitted) % len(masks)
                    for mode,patch,quarter in [masks[choice]]:
                        identity = dict(document=doc.document_id,span=span,replacement=replacement,depth=depth,mode=mode,quarter=quarter)
                        key = hashlib.sha256(canonical_json(identity)).hexdigest()
                        row = ProbeRecord(probe_id=key,recipient_id=doc.document_id,donor_id=doc.document_id+':'+key,
                            recipient_component=policy.components[doc.document_id],donor_component=policy.components[doc.document_id],
                            recipient_role=role,donor_role=role,use=policy.use,source='donor',recipient_tokens=original,
                            donor_tokens=donor,attention_mask=(True,)*n,positions=tuple(range(n)),donor_positions=tuple(range(len(donor))),
                            target_mask=tuple(i >= (patch[0] if patch else 0) for i in range(n-1)),patch_positions=patch,
                            patch_mode=mode,channel_quarter=quarter,depth=depth,teacher_digest=policy.teacher_digest,
                            candidate_digest=policy.candidate_digest,tokenizer_digest=policy.tokenizer_digest,
                            dtype=policy.dtype,backend=policy.backend,boundary_version=policy.boundary_version,
                            execution_profile=policy.execution_profile,seed=policy.seed,edit_kind=kind,source_span=span,
                            donor_span=donor_span,replacement=replacement,prefix_source='teacher',source_window=(0,n),
                            recipient_length_before_crop=original_length,donor_length_before_crop=len(donor_all),
                            recipient_full_tokens_digest=original_digest,
                            donor_full_tokens_digest=hashlib.sha256(canonical_json({'tokens':donor_all})).hexdigest(),
                            context_policy='first_tokens_reset_0_v1',
                            status='rejected' if reason else 'accepted',rejection_reason=reason)
                        emitted += 1
                        if emitted > policy.max_pool_records:
                            raise ValueError('compact probe pool capacity exceeded; resolve a larger pilot cap')
                        rows.append(row)
            yield from rows


def _match_boundary(boundary, record, *, donor=False, patched=False):
    if record.status != 'accepted': raise ValueError('cannot execute a rejected alignment')
    tokens = record.donor_tokens if donor else record.recipient_tokens
    identity = record.donor_id if donor else record.recipient_id
    positions = record.donor_positions if donor else record.positions
    b = boundary.batch
    if boundary.depth != record.depth or tuple(boundary.u.shape[:2]) != (1,len(tokens)):
        raise ValueError('probe depth/shape must match full-sequence boundary')
    if (b.document_ids != (identity,) or b.cluster_ids != (record.recipient_component,)
            or b.input_ids.tolist() != [list(tokens)] or b.positions.tolist() != [list(positions)]
            or b.attention_mask.tolist() != [list(record.attention_mask)]):
        raise ValueError('probe token/position/mask/identity provenance mismatch')
    if str(boundary.u.dtype).removeprefix('torch.') != record.dtype or 'torch_'+boundary.u.device.type != record.backend:
        raise ValueError('probe dtype/backend mismatch')
    digest = record.candidate_digest if record.prefix_source == 'candidate' else record.teacher_digest
    if boundary.model_digest != digest:
        raise ValueError('stale or wrong prefix producer digest')
    if patched and b.target_mask.tolist() != [list(record.target_mask)]:
        raise ValueError('probe target mask mismatch')


def apply_probe(boundary: Boundary, donor: Boundary, record: ProbeRecord) -> Boundary:
    _match_boundary(boundary,record)
    _match_boundary(donor,record,donor=True)
    if donor.u.shape != boundary.u.shape: raise ValueError('donor channel width must match')
    u = boundary.u.detach().clone()
    d = donor.u.detach()
    if any(valid and not bool(boundary.batch.target_mask[0,i].item()) for i,valid in enumerate(record.target_mask)):
        raise ValueError('patch target mask cannot add invalid source targets')
    channels = slice(None)
    if record.patch_mode == 'quarter':
        width = u.shape[-1]
        if width % 4: raise ValueError('fixed channel quarters require a width divisible by four')
        start = record.channel_quarter*(width//4); channels = slice(start,start+width//4)
    for pos in record.patch_positions:
        u[0,pos,channels] = d[0,pos,channels]
    mask = torch.tensor([record.target_mask],device=u.device,dtype=torch.bool)
    return replace(boundary,u=u,batch=replace(boundary.batch,target_mask=mask))


class _LocalBranch:
    def __init__(self,teacher,operator,boundary,recorder):
        self.teacher,self.operator,self.boundary,self.recorder = teacher,operator,boundary,recorder

    def final_hidden(self,batch):
        b = self.boundary
        counts = {'student_forward_calls':1} if self.operator is not None else {'teacher_forward_calls':1}
        with _charged(self.recorder,**counts):
            z = self.teacher.ff_residual(b,self.operator)
        with _charged(self.recorder,teacher_forward_calls=1):
            return self.teacher.suffix_hidden(z,b)

    def readout(self,hidden):
        with _charged(self.recorder,teacher_forward_calls=1):
            return self.teacher.readout(hidden)


def _branches(teacher,candidate_operator,boundary,recorder):
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('local teacher FFN/suffix/readout parameters must remain frozen')
    detached = replace(boundary,u=boundary.u.detach())
    return _LocalBranch(teacher,None,detached,recorder),_LocalBranch(teacher,candidate_operator,detached,recorder)


def local_discrepancy(teacher: DecoderAdapter, candidate_operator: FFNOperator, boundary: Boundary,
                       chunk_size: int, *, recorder: Recorder | None = None) -> MetricSums:
    t,s = _branches(teacher,candidate_operator,boundary,recorder)
    return stream_metrics(t,s,boundary.batch,chunk_size)


class _CountsOnlyRecorder:
    # An outer backward operation measures the whole forward/VJP wall interval.
    # Nested dispatches still persist counters, without duplicating that time.
    def __init__(self,recorder): self.recorder = recorder
    def check(self,cost): self.recorder.check(cost)
    def record(self,cost): self.recorder.record(replace(cost,wall_seconds=0,search_seconds=0))


def prefix_for_probe(decoder: DecoderAdapter,batch: Batch,depth: int,*,source: str,
                       recorder: Recorder | None = None) -> Boundary:
    if source not in ('teacher','candidate'): raise ValueError('explicit prefix producer required')
    counts = {'teacher_forward_calls':1} if source == 'teacher' else {'student_forward_calls':1}
    with _charged(recorder,processed_inputs=batch.input_ids.shape[0],
                  processed_tokens=int(batch.attention_mask.sum().item()),**counts):
        with torch.no_grad():
            b = decoder.prefix_boundary(batch,depth)
        return replace(b,u=b.u.detach())


def backward_local_discrepancy(teacher: DecoderAdapter,candidate_operator: FFNOperator,boundary: Boundary,*,
                                chunk_size: int,denominator: int,record: ProbeRecord,recorder: Recorder | None = None) -> MetricSums:
    """Accumulate candidate-only gradients; caller discards update on any error."""
    if record.use != 'fit': raise PermissionError('gradient probes require fit provenance')
    _match_boundary(boundary,record,patched=True)
    if record.teacher_digest != teacher.manifest().checkpoint_digest:
        raise ValueError('local target teacher must match probe provenance')
    t,s = _branches(teacher,candidate_operator,boundary,_CountsOnlyRecorder(recorder) if recorder else None)
    if not boundary.batch.target_mask.any().item():
        return backward_distillation(t,s,boundary.batch,alpha=0,denominator=denominator,chunk_size=chunk_size)
    with _charged(recorder,suffix_backward_calls=1,student_backward_calls=1,
                  valid_target_presentations=int(boundary.batch.target_mask.sum().item())):
        return backward_distillation(t,s,boundary.batch,alpha=0,denominator=denominator,chunk_size=chunk_size)


@dataclass(frozen=True)
class ProbePool:
    records: tuple[ProbeRecord, ...]
    selected_ids: tuple[str, ...] = ()
    exclusions: Mapping[str, tuple[str, ...]] | None = None
    pathology_threshold_digest: str | None = None
    selection_scores: Mapping[str, float] | None = None
    selection_status: str = 'unselected'
    selection_reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self,'records',tuple(self.records))
        object.__setattr__(self,'selected_ids',tuple(self.selected_ids))
        object.__setattr__(self,'exclusions',freeze(self.exclusions or {}))
        object.__setattr__(self,'selection_scores',freeze(self.selection_scores or {}))
        ids = {x.probe_id for x in self.records}
        if len(ids) != len(self.records) or not set(self.selected_ids) <= ids or not set(self.exclusions) <= ids:
            raise ValueError('pool requires unique records and known selection/exclusion identities')
        component_roles = {}
        for r in self.records:
            if r.recipient_component in component_roles and component_roles[r.recipient_component] != r.recipient_role:
                raise ValueError('probe component cannot cross roles')
            component_roles[r.recipient_component] = r.recipient_role
        if len({x.use for x in self.records}) > 1:
            raise ValueError('fit and score probe pools must be disjoint')

    @property
    def selected_records(self):
        by_id = {r.probe_id:r for r in self.records}
        return tuple(replace(by_id[k],selection_score=self.selection_scores.get(k)) for k in self.selected_ids)


def filter_pathologies(pool: ProbePool, observations: Mapping[str,tuple[float,float]],
                        thresholds: PathologyThresholds) -> ProbePool:
    if pool.pathology_threshold_digest is not None and pool.pathology_threshold_digest != thresholds.digest:
        raise ValueError('already-filtered pool cannot use different pathology thresholds; filter original records')
    exclusions = dict(pool.exclusions)
    for r in pool.records:
        if r.status != 'accepted': continue
        if r.probe_id not in observations: raise ValueError('missing required teacher pathology observation')
        values = observations[r.probe_id]
        if len(values) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid teacher pathology observation')
        reasons = []
        for name,value in zip(('norm','entropy'),values):
            lo,hi = thresholds.interval(name,r.depth)
            if value < lo: reasons.append(name+'_below_fit_threshold')
            if value > hi: reasons.append(name+'_above_fit_threshold')
        if reasons: exclusions[r.probe_id] = tuple(reasons)
    return replace(pool,selected_ids=(),exclusions=exclusions,pathology_threshold_digest=thresholds.digest)


def _quotas(n,balance):
    raw = [n*x for x in balance]; result = [int(x) for x in raw]
    for i in sorted(range(len(raw)),key=lambda i:(-(raw[i]-result[i]),i))[:n-sum(result)]: result[i] += 1
    return result


def mine_probes(pool: ProbePool,scores: Mapping[str,float],policy: ProbePolicy) -> ProbePool:
    with _charged(policy.recorder,search_proposals=len(pool.records)):
        if not pool.records: raise ValueError('empty probe population')
        if len(pool.records) > policy.max_pool_records: raise ValueError('probe pool capacity exceeded')
        eligible = [r for r in pool.records if r.status == 'accepted' and r.probe_id not in pool.exclusions]
        if any(policy.roles.get(r.recipient_id) != r.recipient_role
               or policy.components.get(r.recipient_id) != r.recipient_component for r in pool.records):
            raise PermissionError('source role/component differs from declared manifest')
        if any(r.use != policy.use or r.recipient_role not in policy.allowed_roles for r in pool.records):
            raise ValueError('selection pool role/use violates policy')
        state_fields = ('teacher_digest','candidate_digest','tokenizer_digest','dtype','backend','boundary_version','execution_profile')
        if any(getattr(r,k) != getattr(policy,k) for r in pool.records for k in state_fields):
            raise ValueError('probe pool state differs from current resolved selection policy')
        if any(r.depth not in {d for region in policy.depth_regions for d in region} for r in pool.records):
            raise ValueError('probe depth is outside calibrated selection regions')
        if policy.selection_mode == 'active':
            if any(r.probe_id not in scores for r in eligible): raise ValueError('missing required probe scores')
            if any(not math.isfinite(scores[r.probe_id]) for r in eligible): raise ValueError('probe scores must be finite')
        rng = random.Random(policy.seed); selected = []; doc_counts = {}; deficits = []
        if pool.pathology_threshold_digest is not None and pool.pathology_threshold_digest != policy.thresholds.digest:
            raise ValueError('pathology filter differs from resolved policy')
        quotas = _quotas(policy.selection_count,policy.depth_region_balance)
        random_quotas = _quotas(round(policy.selection_count*policy.random_component_fraction),policy.depth_region_balance)
        for region,quota,random_quota in zip(policy.depth_regions,quotas,random_quotas):
            candidates = sorted((r for r in eligible if r.depth in region),key=lambda r:r.probe_id)
            rng.shuffle(candidates)
            random_count = quota if policy.selection_mode == 'seeded_random' else min(quota,random_quota)
            chosen = 0
            # Fill the random quota from the entire shuffled eligible region;
            # exhausted document caps must not silently turn it into active slots.
            for r in candidates:
                if chosen == random_count: break
                if doc_counts.get(r.recipient_id,0) >= policy.per_document_cap: continue
                selected.append(r.probe_id);doc_counts[r.recipient_id] = doc_counts.get(r.recipient_id,0)+1;chosen += 1
            remaining = [r for r in candidates if r.probe_id not in selected]
            if policy.selection_mode == 'active': remaining.sort(key=lambda r:(-scores[r.probe_id],r.probe_id))
            for r in remaining:
                if chosen == quota: break
                if doc_counts.get(r.recipient_id,0) >= policy.per_document_cap: continue
                selected.append(r.probe_id);doc_counts[r.recipient_id] = doc_counts.get(r.recipient_id,0)+1;chosen += 1
            if chosen < quota:
                deficits.append(dict(depths=list(region),requested=quota,available_under_caps=chosen))
        return replace(pool,selected_ids=tuple(selected),selection_scores={k:scores[k] for k in selected} if policy.selection_mode == 'active' else {},
                       selection_status='underfilled' if deficits else 'complete',
                       selection_reason=str(deficits) if deficits else None)


def mixture_discrepancy(sources: Mapping[str,MetricSums],weights: Mapping[str,float]) -> float:
    if (not weights or any(k not in ('natural','student','donor') or not math.isfinite(v) or v < 0 for k,v in weights.items())
            or not math.isclose(math.fsum(weights.values()),1.,rel_tol=0,abs_tol=1e-12)):
        raise ValueError('mixture weights must be nonnegative and sum to one')
    terms = []
    for source,weight in weights.items():
        if not weight: continue
        if source not in sources or sources[source].valid_targets == 0:
            raise ValueError('missing required source observations; cannot renormalize mixture')
        metric = sources[source]
        terms.append(weight*metric.kl/metric.valid_targets)
    return math.fsum(terms)


def probe_cache_key(record: ProbeRecord) -> str:
    return hashlib.sha256(canonical_json(asdict(record))).hexdigest()


def _copy_boundary(b):
    batch = replace(b.batch,**{name:getattr(b.batch,name).detach().clone() for name in
                              ('input_ids','attention_mask','target_mask','positions')})
    return replace(b,u=b.u.detach().clone(),batch=batch)


class ProbeCache:
    """LRU minibatches bounded by both actual tensor bytes and entry count."""
    def __init__(self,*,max_entries: int,max_bytes: int):
        if type(max_entries) is not int or max_entries < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('explicit positive cache bounds required')
        self.max_entries,self.max_bytes = max_entries,max_bytes
        self._entries = OrderedDict()

    @property
    def entry_count(self): return len(self._entries)

    @property
    def resident_bytes(self): return sum(size for _,size in self._entries.values())

    def _validate_current(self,r,teacher,candidate):
        t,c = teacher.manifest(),candidate.manifest()
        if (r.teacher_digest != t.checkpoint_digest or r.candidate_digest != c.checkpoint_digest
                or any(r.dtype != m.dtype or r.backend != m.backend for m in (t,c))
                or any(x.tokenizer_digest != r.tokenizer_digest for x in (t,c))):
            self._entries.clear()
            raise ValueError('stale probe state: record must match current teacher and candidate')

    def put(self,record,boundary,*,teacher,candidate):
        self._validate_current(record,teacher,candidate)
        _match_boundary(boundary,record,patched=True)
        tensors = [boundary.u,*[getattr(boundary.batch,n) for n in ('input_ids','attention_mask','target_mask','positions')]]
        size = sum(t.numel()*t.element_size() for t in tensors)
        if size > self.max_bytes: raise ValueError('boundary exceeds cache byte capacity')
        key = probe_cache_key(record)
        self._entries.pop(key,None)
        while self._entries and (len(self._entries) >= self.max_entries or self.resident_bytes+size > self.max_bytes):
            self._entries.popitem(last=False)
        self._entries[key] = (_copy_boundary(boundary),size)

    def get(self,record,*,teacher,candidate):
        self._validate_current(record,teacher,candidate)
        key = probe_cache_key(record)
        if key not in self._entries: return None
        b,size = self._entries[key]
        _match_boundary(b,record,patched=True)
        self._entries.move_to_end(key)
        return _copy_boundary(b)

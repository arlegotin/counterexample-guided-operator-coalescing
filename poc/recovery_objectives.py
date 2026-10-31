"""Versioned opt-in recovery losses, independent of canonical owner scope."""
from dataclasses import dataclass, asdict
import hashlib
import re

import torch

from poc.bank import owner_census
from poc.contracts import ProbeRecord, CostRecord
from poc.records import canonical_json
from poc.probes import prefix_for_probe, backward_local_discrepancy


def _hash(value):return hashlib.sha256(canonical_json({"value":value})).hexdigest()


@dataclass(frozen=True)
class RecoveryObjectiveSpec:
    loss_scope: str
    owner_scope: str
    affected_depths: tuple[int,...]
    raw_plan_sha256: str
    owner_names_sha256: str
    schema_version: int = 1
    checkpoint_every: int = 16

    def __post_init__(self):
        object.__setattr__(self,'affected_depths',tuple(self.affected_depths))
        if type(self.checkpoint_every)!=int or self.checkpoint_every<1:raise ValueError('positive checkpoint cadence required')
        allowed={1:('local_sum','local_mean','full_student'),2:('ffn_mse_sum',),
                 3:('local_sum','local_mean','full_student')}
        if type(self.schema_version)!=int or self.loss_scope not in allowed.get(self.schema_version,()):
            raise ValueError('unsupported recovery objective/version')
        expected_scope='affected_groups' if self.schema_version==3 else 'affected_group'
        if self.owner_scope!=expected_scope:raise ValueError('unsupported recovery owner scope/version')
        if (not self.affected_depths or self.affected_depths!=tuple(sorted(set(self.affected_depths)))
                or any(type(d)!=int or d<0 for d in self.affected_depths)):
            raise ValueError('sorted unique affected depths required')
        if any(not re.fullmatch('[0-9a-f]{64}',h) for h in (self.raw_plan_sha256,self.owner_names_sha256)):
            raise ValueError('plan and owner SHA256 required')

    def identity(self):return asdict(self)

    @property
    def scored_multiplier(self):return 1 if self.loss_scope=='full_student' else len(self.affected_depths)

    @property
    def scale(self):return 1/len(self.affected_depths) if self.loss_scope=='local_mean' else 1


def affected_owners(student,depths,*,owner_scope="affected_group"):
    """Select complete canonical groups; multiple groups require explicit opt-in."""
    selected=set(depths)
    if any(type(d)!=int or not 0<=d<len(student.layers) for d in selected):raise ValueError('depth outside model')
    groups=[g for g in student.partition.groups if selected.intersection(g)]
    if owner_scope not in ('affected_group','affected_groups'):
        raise ValueError('unsupported recovery owner scope')
    required_count=(len(groups)==1 if owner_scope=='affected_group' else len(groups)>=2)
    if not required_count or set().union(*(set(g) for g in groups))!=selected:
        raise ValueError('owner scope must include every use of the selected canonical banks')
    prefixes=tuple(f'layers.{d}.mlp.residuals.' for d in depths)
    rows=[r for r in owner_census(student) if r.category=='bank' and any(
        site.startswith(tuple(f'layers.{d}.' for d in depths)) for site in r.use_sites)
        or r.category=='residual' and r.name.startswith(prefixes)]
    return {r.name:dict(student.named_parameters())[r.name] for r in sorted(rows,key=lambda r:r.name)}


def owner_names_hash(student,depths,*,owner_scope="affected_group"):
    return _hash(sorted(affected_owners(student,depths,owner_scope=owner_scope)))


def owner_manifest(student,depths,*,owner_scope="affected_group"):
    return [dict(name=n,shape=list(p.shape),sha256=hashlib.sha256(
        p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for n,p in affected_owners(student,depths,owner_scope=owner_scope).items()]


def natural_record(teacher,student,batch,depth,seed=0):
    """Bind an unsliced natural template to actual executable identities."""
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    if batch.input_ids.shape[0]!=1:raise ValueError('one context per natural record')
    tm,sm=teacher.manifest(),student.manifest()
    if (tm.tokenizer_digest,tm.dtype,tm.backend)!=(sm.tokenizer_digest,sm.dtype,sm.backend):
        raise ValueError('teacher/student numerical and tokenizer profiles differ')
    tokens=tuple(batch.input_ids[0].tolist());positions=tuple(batch.positions[0].tolist())
    token_digest=_hash(list(tokens));doc=batch.document_ids[0];component=batch.cluster_ids[0]
    return ProbeRecord(probe_id=_hash([doc,tokens,depth,sm.checkpoint_digest]),recipient_id=doc,donor_id=doc,
        recipient_component=component,donor_component=component,recipient_role='fit',donor_role='fit',use='fit',
        source='natural',recipient_tokens=tokens,donor_tokens=tokens,
        attention_mask=tuple(batch.attention_mask[0].tolist()),positions=positions,donor_positions=positions,
        target_mask=tuple(batch.target_mask[0].tolist()),patch_positions=(),patch_mode='full',channel_quarter=None,
        depth=depth,teacher_digest=tm.checkpoint_digest,candidate_digest=sm.checkpoint_digest,
        tokenizer_digest=tm.tokenizer_digest,dtype=tm.dtype,backend=tm.backend,
        boundary_version='ffn_input_v1',execution_profile=EXECUTION_PROFILE,seed=seed,
        edit_kind='natural_identity',source_span=(0,0),donor_span=(0,0),replacement='',prefix_source='teacher',
        source_window=(0,len(tokens)),recipient_length_before_crop=len(tokens),donor_length_before_crop=len(tokens),
        recipient_full_tokens_digest=token_digest,donor_full_tokens_digest=token_digest,context_policy='first_tokens_reset_0_v1')


def backward_recovery(student,teacher,update,config,recorder,spec,raw_templates):
    from poc.train import _call, _Counts, _Dispatch
    from poc.losses import backward_distillation
    # Persist once before loss calls, including invalid/interrupted work. The
    # legacy presentation counter continues to count each scored loss target.
    if update.valid_targets!=sum(int(b.target_mask.sum()) for b in update.batches):raise ValueError('raw update denominator mismatch')
    metrics=[]
    for b,(i,_,_) in zip(update.batches,update.slices,strict=True):
        original=raw_templates.template_at(i,student.device) if hasattr(raw_templates,'template_at') else raw_templates[i]
        for name in ('input_ids','attention_mask','positions'):
            if not torch.equal(getattr(original,name),getattr(b,name)):raise ValueError('raw context differs from plan')
        if (b.target_mask & ~original.target_mask).any():raise ValueError('loss mask exceeds raw eligibility')
        recorder.record(CostRecord(raw_corpus_target_presentations=int(b.target_mask.sum())))
        if spec.loss_scope=='full_student':
            inner=_Counts(recorder)
            metrics.append(_call(recorder,lambda:backward_distillation(_Dispatch(teacher,inner,'teacher'),
                _Dispatch(student,inner,'student'),b,alpha=0,denominator=update.valid_targets,chunk_size=config.chunk_size),
                valid_target_presentations=int(b.target_mask.sum()),student_backward_calls=1,
                processed_inputs=1,processed_tokens=int(b.attention_mask.sum())))
        elif spec.loss_scope=='ffn_mse_sum':
            from poc.reconstruction import backward_ffn_reconstruction,collect_teacher_boundaries
            boundaries=collect_teacher_boundaries(teacher,original,spec.affected_depths,recorder=recorder)
            for depth,boundary in zip(spec.affected_depths,boundaries,strict=True):
                r=natural_record(teacher,student,original,depth)
                metrics.append(backward_ffn_reconstruction(teacher,student.layers[depth].mlp,boundary,
                    denominator=update.valid_targets,record=r,loss_mask=b.target_mask,recorder=recorder))
        else:
            hooks=[p.register_hook(lambda g:g*spec.scale) for p in student.parameters() if p.requires_grad] if spec.scale!=1 else []
            try:
                for depth in spec.affected_depths:
                    r=natural_record(teacher,student,original,depth)
                    boundary=prefix_for_probe(teacher,original,depth,source='teacher',recorder=recorder)
                    metrics.append(backward_local_discrepancy(teacher,student.layers[depth].mlp,boundary,
                        chunk_size=config.chunk_size,denominator=update.valid_targets,record=r,recorder=recorder,loss_mask=b.target_mask))
            finally:
                for h in hooks:h.remove()
    return metrics

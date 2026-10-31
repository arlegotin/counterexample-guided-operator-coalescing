"""Opt-in current-state merge initialization and fixed-intervention KL diagnostics."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import re

import torch

from poc.bank import PROJECTIONS, initialize_residual
from poc.contracts import freeze
from poc.merge import MergeTrial, _call, initialize_merge
from poc.records import canonical_json


def _sha(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _valid_hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class CompositionInitializations:
    teacher_reset: MergeTrial
    current_continuation: MergeTrial
    record: object


def paired_composition_initializations(current,teacher,pair,fit,seed,*,batches=None,recorder=None,forced_parent=None):
    """Choose a parent once, then vary only the projection reconstruction target.

    Both branches require a fresh optimizer; no moments are copied. Continuation
    reconstructs old bank+BA against the selected new bank with the same rank.
    This can discard capacity at truncated rank and preserves no quality guarantee.
    The original current model and every unaffected reference owner stay unchanged.
    """
    if fit.quarantine:
        raise ValueError('quarantined fit records cannot select composition parents')
    before=_call(recorder,lambda:current.manifest().checkpoint_digest)
    reset=initialize_merge(current,teacher,pair,fit,seed,batches=batches,recorder=recorder,forced_parent=forced_parent)
    model=_call(recorder,lambda:deepcopy(reset.model))
    staged={};rows=[]
    for depth in reset.affected_depths:
        old=current.layers[depth].mlp
        bank=current.bank[current.partition.reuse_map[depth]]
        new_bank=model.bank[model.partition.reuse_map[depth]]
        for qi,q in enumerate(PROJECTIONS):
            def old_effective():
                value=getattr(bank,q).weight.detach()
                if q in old.residuals:
                    factor=old.residuals[q]
                    value=value+factor.B.detach()@factor.A.detach()
                return value
            effective=_call(recorder,old_effective)
            delta=_call(recorder,lambda:effective-getattr(new_bank,q).weight.detach())
            a,b=_call(recorder,lambda:initialize_residual(delta,current.rank,(seed+depth*3+qi)%2**63))
            error=_call(recorder,lambda:delta.detach().cpu().double()-b.detach().cpu().double()@a.detach().cpu().double())
            rows.append({'depth':depth,'projection':q,'rank':current.rank,
                         'delta_reconstruction_mse':float(error.square().mean()),
                         'delta_reconstruction_max_abs':float(error.abs().max()),
                         'delta_sha256':hashlib.sha256(delta.detach().cpu().contiguous().numpy().tobytes()).hexdigest()})
            staged[(depth,q)]=(a,b)
    with torch.no_grad():
        for (depth,q),(a,b) in staged.items():
            if current.rank:
                model.layers[depth].mlp.residuals[q].A.copy_(a)
                model.layers[depth].mlp.residuals[q].B.copy_(b)
    digest=_call(recorder,lambda:model.manifest().checkpoint_digest)
    if current.manifest().checkpoint_digest!=before:
        raise ValueError('composition initialization mutated the input current model')
    continuation=MergeTrial(model,teacher,reset.pair,reset.affected_depths,reset.selected_parent,
                            reset.medoid_scores,digest,recorder=recorder,parent_policy=reset.parent_policy)
    record=freeze({'schema_version':1,'kind':'paired_composition_initialization_v1',
        'source_current_digest':before,'teacher_digest':teacher.manifest().checkpoint_digest,
        'pair':list(pair),'affected_depths':list(reset.affected_depths),'rank':current.rank,'seed':seed,
        'selected_parent':reset.selected_parent,'parent_scores':reset.medoid_scores,
        'final_partition':[list(g) for g in model.partition.groups],
        'teacher_reset_digest':reset.initialized_digest,'current_continuation_digest':digest,
        'optimizer_state_policy':'fresh_optimizer_for_both_branches',
        'effective_weight_arithmetic':'source dtype bank+BA, difference then CPU FP64 SVD; returned factors in source dtype',
        'projection_records':rows,'cost_allocation':'parent selection shared once for paired mechanism; continuation SVD extra work separately charged; independent pipelines must pay their own preparation'})
    if forced_parent is not None:
        record=freeze(dict(record,schema_version=2,parent_policy='forced_current_group_representative',forced_parent=forced_parent))
    return CompositionInitializations(reset,continuation,record)


@dataclass(frozen=True)
class KLObservation:
    model_digest: str
    teacher_digest: str
    context_digest: str
    kl_sum: float
    valid_targets: int

    def __post_init__(self):
        if not all(_valid_hash(x) for x in (self.model_digest,self.teacher_digest,self.context_digest)):
            raise ValueError('exact model/teacher/context identities required')
        if type(self.valid_targets) is not int or self.valid_targets<1:
            raise ValueError('positive valid target count required')
        if type(self.kl_sum) not in (int,float) or not math.isfinite(self.kl_sum):
            raise ValueError('finite raw KL sum required')


def baseline_corrected_interaction(observations,*,base_state_digest,intervention_ids,
                                  study_contract_sha256,composable,kl_floor_per_target):
    """D(AB)-D(A)-D(B)+D(base), on one fixed state and identical contexts/masks.

    The caller constructs the four actual fixed interventions. Noncomposable
    overlapping merge recipes require a separately labeled recipe comparison.
    This descriptive diagnostic supplies no transitivity or generalization bound.
    """
    if set(observations)!={'base','A','B','AB'} or any(not isinstance(x,KLObservation) for x in observations.values()):
        raise ValueError('four explicit fixed-intervention observations required')
    if type(composable) is not bool or not composable:
        raise ValueError('joint recipe is not a composable fixed operator intervention')
    if (not _valid_hash(base_state_digest) or not _valid_hash(study_contract_sha256)
            or len(intervention_ids)!=2 or not all(_valid_hash(x) for x in intervention_ids)):
        raise ValueError('baseline/intervention/study identities required')
    if type(kl_floor_per_target) not in (int,float) or not math.isfinite(kl_floor_per_target) or kl_floor_per_target<0:
        raise ValueError('qualified finite nonnegative KL floor required')
    base=observations['base']
    if base.model_digest!=base_state_digest:
        raise ValueError('baseline state differs from frozen intervention source')
    common={(x.teacher_digest,x.context_digest,x.valid_targets) for x in observations.values()}
    if len(common)!=1:
        raise ValueError('teacher, complete windows/masks and valid target counts must match')
    for value in observations.values():
        if value.kl_sum < -kl_floor_per_target*value.valid_targets:
            raise FloatingPointError('raw KL below qualified negative floor')
    raw=math.fsum((observations['AB'].kl_sum,-observations['A'].kl_sum,
                   -observations['B'].kl_sum,base.kl_sum))
    return {'schema_version':1,'kind':'fixed_operator_interaction',
            'study_contract_sha256':study_contract_sha256,'base_state_digest':base_state_digest,
            'intervention_ids':list(intervention_ids),'observations':{k:asdict(v) for k,v in observations.items()},
            'interaction_kl_sum':raw,'valid_targets':base.valid_targets,
            'interaction_kl_per_target':raw/base.valid_targets,'kl_floor_per_target':kl_floor_per_target}

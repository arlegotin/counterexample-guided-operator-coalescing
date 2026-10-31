"""Verified nonlinear reconstruction → fresh KL initialization handoff.

Both optimization stages use train.fit_model. This module imports only model
weights from a completed safe MSE checkpoint; it never resumes its optimizer.
"""
from dataclasses import dataclass,asdict,replace,field
import hashlib
import re

import torch

from poc.checkpoint import load_checkpoint,checkpoint_decoder
from poc.contracts import freeze,thaw
from poc.records import canonical_json,run_id
from poc.recovery_objectives import owner_names_hash,owner_manifest,affected_owners
from poc.train import TrainingSources,_recovery_cursors,_schedule_state,_model_schema


def _hash(value):return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ReconstructionHandoffSpec:
    warm_run_id: str
    checkpoint_manifest_sha256: str
    warm_raw_plan_sha256: str
    recovery_raw_plan_sha256: str
    teacher_digest: str
    owner_names_sha256: str
    initial_state_digest: str
    warm_updates: int
    seed: int
    schema_version: int = 1
    raw_targets_per_update: int = 8192

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1:
            raise ValueError('unsupported reconstruction handoff schema')
        for key in ('warm_run_id','checkpoint_manifest_sha256','warm_raw_plan_sha256',
                    'recovery_raw_plan_sha256','teacher_digest','owner_names_sha256','initial_state_digest'):
            if not isinstance(getattr(self,key),str) or not re.fullmatch('[0-9a-f]{64}',getattr(self,key)):
                raise ValueError('detached handoff SHA256 required: '+key)
        if (type(self.warm_updates) is not int or self.warm_updates<1 or
                type(self.seed) is not int or not 0<=self.seed<2**63 or
                type(self.raw_targets_per_update) is not int or self.raw_targets_per_update!=8192):
            raise ValueError('positive completed warm horizon, int64 seed and8192 targets required')

    def identity(self):return asdict(self)


@dataclass(frozen=True)
class ReconstructionHandoff:
    student: object
    config: object
    objective: object
    record: object


@dataclass(frozen=True)
class OptimizationHandoffSpec(ReconstructionHandoffSpec):
    """Explicit source objective for the priced extra-natural warm control."""
    warm_loss_scope: str = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.warm_loss_scope not in ('ffn_mse_sum','full_student'):
            raise ValueError('named MSE or full-student warm objective required')


@dataclass(frozen=True)
class OptimizationHandoff(ReconstructionHandoff):
    pass


def prepare_reconstruction_handoff(*,teacher,initial,checkpoint,warm_run_spec,warm_plan,
                                   recovery_config,recovery_objective,handoff_spec):
    """Preserve the original MSE-only schema and serialized record semantics."""
    return _prepare_handoff(teacher=teacher,initial=initial,checkpoint=checkpoint,
        warm_run_spec=warm_run_spec,warm_plan=warm_plan,recovery_config=recovery_config,
        recovery_objective=recovery_objective,handoff_spec=handoff_spec,
        warm_loss_scope='ffn_mse_sum',generic=False)


def prepare_optimization_handoff(*,teacher,initial,checkpoint,warm_run_spec,warm_plan,
                                recovery_config,recovery_objective,handoff_spec):
    """Transfer a verified completed named warm phase into fresh KL state."""
    if not isinstance(handoff_spec,OptimizationHandoffSpec):
        raise ValueError('explicit optimization-phase handoff spec required')
    return _prepare_handoff(teacher=teacher,initial=initial,checkpoint=checkpoint,
        warm_run_spec=warm_run_spec,warm_plan=warm_plan,recovery_config=recovery_config,
        recovery_objective=recovery_objective,handoff_spec=handoff_spec,
        warm_loss_scope=handoff_spec.warm_loss_scope,generic=True)


def _prepare_handoff(*,teacher,initial,checkpoint,warm_run_spec,warm_plan,
                     recovery_config,recovery_objective,handoff_spec,warm_loss_scope,generic):
    """Build a new KL RunSpec and independent model from a completed MSE phase.

    Extra retained warm presentations are explicit. Physical spending stays in
    the parent ledger; it is not imported into the new recovery attempt or
    inferred from a possibly stale checkpoint wall-cost snapshot.
    """
    spec=handoff_spec
    if not isinstance(spec,ReconstructionHandoffSpec):raise ValueError('explicit handoff spec required')
    if (checkpoint.manifest_sha256!=spec.checkpoint_manifest_sha256 or
            run_id(warm_run_spec)!=spec.warm_run_id or warm_plan.digest!=spec.warm_raw_plan_sha256 or
            warm_plan.updates!=spec.warm_updates or warm_plan.raw_targets_per_update!=8192):
        raise ValueError('warm checkpoint/run/plan/horizon binding mismatch')
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError('frozen teacher required for phase handoff')
    if teacher.manifest().checkpoint_digest!=spec.teacher_digest or initial.base_digest!=spec.teacher_digest:
        raise ValueError('teacher handoff identity mismatch')
    initial_digest=initial.manifest().checkpoint_digest
    if initial_digest!=spec.initial_state_digest:raise ValueError('original initial state differs')
    warm=thaw(warm_run_spec.resolved_config);training=warm.get('training',{});objective=warm.get('recovery_objective',{})
    if (objective.get('schema_version')!=(2 if warm_loss_scope=='ffn_mse_sum' else 1) or objective.get('loss_scope')!=warm_loss_scope or
            objective.get('owner_scope')!='affected_group' or training.get('alpha')!=0 or
            training.get('max_updates')!=spec.warm_updates or training.get('target_count')!=8192 or
            objective.get('raw_plan_sha256')!=spec.warm_raw_plan_sha256 or
            objective.get('owner_names_sha256')!=spec.owner_names_sha256):
        raise ValueError('completed named warm objective/training identity required' if generic else 'completed MSE objective/training identity required')
    target=recovery_objective
    if (target.loss_scope not in ('local_sum','local_mean','full_student') or target.owner_scope!='affected_group' or
            target.raw_plan_sha256!=spec.recovery_raw_plan_sha256 or target.owner_names_sha256!=spec.owner_names_sha256 or
            tuple(objective.get('affected_depths',()))!=target.affected_depths):
        raise ValueError('same-owner KL target objective and explicit recovery plan required')
    cfg=recovery_config;template=cfg.run_spec
    if template is None or cfg.resume is not None or cfg.alpha!=0 or cfg.target_count!=8192:
        raise ValueError('fresh explicitly identified KL run required')
    expected={'training':cfg.identity(),'recovery_objective':target.identity()}
    if any(_hash(thaw(template.resolved_config).get(k))!=_hash(v) for k,v in expected.items()):
        raise ValueError('recovery RunSpec objective/schedule mismatch')
    if template.input_manifests.get('raw_plan')!=target.raw_plan_sha256 or template.input_manifests.get('owner_names')!=target.owner_names_sha256:
        raise ValueError('recovery plan/owner inputs mismatch')
    if warm_run_spec.seeds.get('fit')!=spec.seed or template.seeds.get('fit')!=spec.seed:
        raise ValueError('phase seed mismatch')
    for role in ('assets','fit','tuning'):
        value=warm_run_spec.input_manifests.get(role)
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value) or template.input_manifests.get(role)!=value:
            raise ValueError('phase population/asset mismatch: '+role)
    if (warm_run_spec.input_manifests.get('assets')!=teacher.tokenizer_digest or
            warm_run_spec.input_manifests.get('initialized_state')!=initial_digest or
            warm_run_spec.input_manifests.get('raw_plan')!=warm_plan.digest or
            warm_run_spec.input_manifests.get('owner_names')!=spec.owner_names_sha256 or
            owner_names_hash(initial,target.affected_depths)!=spec.owner_names_sha256):
        raise ValueError('warm initial state, assets or canonical owners mismatch')
    state=load_checkpoint(checkpoint,warm_run_spec)
    raw=spec.warm_updates*8192
    if (state.step!=spec.warm_updates or state.target_cursor!=raw or not state.safe_boundary or
            state.source_hash!=warm_plan.digest or state.teacher_digest!=spec.teacher_digest or
            state.config_hash!=_hash({'training':training,'recovery_objective':objective}) or
            state.scheduler!=_schedule_state(spec.warm_updates,spec.warm_updates)):
        raise ValueError('incomplete or incompatible safe reconstruction phase')
    expected_schema=_model_schema(initial)
    for key in ('config','head_dim','rope_theta','tokenizer_digest','groups','rank',
                'base_digest','execution_profile','backend','device','norm_epsilons','tied'):
        if state.model_schema.get(key)!=expected_schema[key]:
            raise ValueError('reconstruction changed scalar/schema state: '+key)
    selected=set(affected_owners(initial,target.affected_depths))
    declared=state.model_schema.get('owners',[])
    expected_owners={o['name']:o for o in expected_schema['owners']}
    if (len(declared)!=len(expected_owners) or len({o.get('name') for o in declared})!=len(declared)):
        raise ValueError('phase owner declarations differ')
    for owner in declared:
        name=owner.get('name')
        if (name not in expected_owners or type(owner.get('trainable')) is not bool or
                owner['trainable']!=(name in selected) or
                {k:v for k,v in owner.items() if k!='trainable'}!=
                {k:v for k,v in expected_owners[name].items() if k!='trainable'}):
            raise ValueError('phase owner declaration/trainability mismatch')
    from poc.recovery_objectives import RecoveryObjectiveSpec
    from poc.train import TrainingConfig
    restored_config=dict(training);restored_config.pop('epoch_tail_policy',None)
    source_cfg=TrainingConfig(**restored_config)
    source_objective=RecoveryObjectiveSpec(**objective)
    expected_cursors=_recovery_cursors(TrainingSources(raw_plan=warm_plan),source_cfg,spec.warm_updates,source_objective)
    if (state.target_cursor,state.probe_cursor)!=expected_cursors:
        raise ValueError('warm raw/operator cursors differ from immutable plan')
    parameters=dict(initial.named_parameters())
    if set(state.model)!=set(parameters):raise ValueError('phase canonical tensor owners differ')
    for name,p in parameters.items():
        v=state.model[name]
        if v.shape!=p.shape or v.dtype!=p.dtype or not torch.isfinite(v).all().item():
            raise ValueError('invalid phase tensor shape/dtype/value')
        if name not in selected and not torch.equal(v,p.detach().cpu()):
            raise ValueError('reconstruction changed frozen owner: '+name)
    # checkpoint_decoder verifies actual schema, aliases, norm scalars, source
    # device and candidate digest before any new-device model is returned.
    model=checkpoint_decoder(state)
    if model.partition!=initial.partition or model.rank!=initial.rank:
        raise ValueError('phase partition/rank differs')
    model=model.to(device=initial.device)
    if owner_names_hash(model,target.affected_depths)!=spec.owner_names_sha256:
        raise ValueError('phase executable owner names differ')
    record=dict(schema_version=1,handoff_spec=spec.identity(),warm_run_id=spec.warm_run_id,
        warm_checkpoint_manifest_sha256=checkpoint.manifest_sha256,warm_source_candidate_digest=state.candidate_digest,
        original_initial_state=initial_digest,recovery_initial_state=model.manifest().checkpoint_digest,
        retained_warm_updates=spec.warm_updates,retained_warm_raw_targets=raw,
        retained_warm_scored_targets=raw*(len(target.affected_depths) if warm_loss_scope=='ffn_mse_sum' else 1),
        exposure_policy='additional warm presentations; not new unique text and not part of KL raw cursor',
        optimizer_policy='fresh; no warm optimizer/RNG/cursors/spending imported',
        physical_cost_policy='parent authoritative ledger retained separately; no snapshot wall-cost inference')
    if generic:
        record.update(kind='named_optimization_phase_handoff_v1',warm_loss_scope=warm_loss_scope,
                      warm_objective_schema_version=objective['schema_version'])
    slot='optimization_phase_handoff' if generic else 'reconstruction_handoff'
    inputs=dict(template.input_manifests,initialized_state=record['recovery_initial_state'],
                initialized_owners=_hash({'owners':owner_manifest(model,target.affected_depths)}),
                warm_checkpoint=checkpoint.manifest_sha256)
    inputs[slot]=_hash(record)
    resolved=dict(thaw(template.resolved_config));resolved[slot]=record
    identity=replace(template,input_manifests=inputs,resolved_config=resolved,
        parent_run_ids=tuple(dict.fromkeys((*template.parent_run_ids,spec.warm_run_id))))
    cls=OptimizationHandoff if generic else ReconstructionHandoff
    return cls(model,replace(cfg,run_spec=identity),target,freeze(record))

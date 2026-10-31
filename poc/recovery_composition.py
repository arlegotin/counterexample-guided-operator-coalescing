"""Explicit completed-source composition initialization for new recovery studies."""
from dataclasses import dataclass,asdict,replace
import hashlib,re

import torch

from poc.bank import BankDecoder,canonicalize_partition
from poc.checkpoint import CheckpointRef,_spec,load_checkpoint,checkpoint_decoder
from poc.composition import paired_composition_initializations
from poc.contracts import thaw
from poc.merge import _call
from poc.records import canonical_json,run_id
from poc.recovery_clone import _hashed_json
from poc.recovery_data import restricted_population
from poc.train import _model_schema,_schedule_state,_Counts


def _hash(value):return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class CompositionSourceSpec:
    source_run_spec_path: str | None
    source_run_spec_sha256: str | None
    checkpoint_directory: str | None
    checkpoint_manifest_sha256: str | None
    merge_pair: tuple[int,int]
    forced_parent: int
    reference: str
    expected_partition: tuple[tuple[int,...],...]
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1:raise ValueError('unknown composition source schema')
        pair=tuple(self.merge_pair)
        if len(pair)!=2 or any(type(d) is not int or d<0 for d in pair) or pair[0]>=pair[1]:
            raise ValueError('canonical composition merge_pair required')
        object.__setattr__(self,'merge_pair',pair)
        if type(self.forced_parent) is not int or self.forced_parent<0:raise ValueError('explicit canonical forced parent required')
        if self.reference not in ('teacher_reset','current_continuation'):raise ValueError('explicit composition reference required')
        groups=tuple(tuple(g) for g in self.expected_partition)
        flat=[d for g in groups for d in g]
        if not flat or any(type(d) is not int or d<0 for d in flat):raise ValueError('explicit expected partition required')
        partition=canonicalize_partition(groups,max(flat)+1)
        if partition.groups!=groups:raise ValueError('canonical expected partition required')
        object.__setattr__(self,'expected_partition',groups)
        for path_key,hash_key in (('source_run_spec_path','source_run_spec_sha256'),('checkpoint_directory','checkpoint_manifest_sha256')):
            path,digest=getattr(self,path_key),getattr(self,hash_key)
            if (path is None)!=(digest is None):raise ValueError('composition source paths/hashes must resolve together')
            if path is not None and (not isinstance(path,str) or not path.strip() or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)):
                raise ValueError('composition source requires nonempty paths and detached SHA256 identities')

    def identity(self):return asdict(self)


def initialize_composition_recovery(teacher,config,fit,fit_batches,seed,*,recorder=None):
    """Verify one completed source, construct the selected recipe, import no optimizer.

    Both references are constructed and their preparation charged. Separate recipe
    studies pay their own preparation; a common fitting loop handles all updates.
    No backend/profile migration, STEP0 cloning or old-study resumption occurs.
    """
    binding=CompositionSourceSpec(**config['composition'])
    if binding.source_run_spec_path is None or binding.checkpoint_directory is None:
        raise ValueError('unresolved completed composition source')
    if type(seed) is not int or config.get('seeds')!=[seed]:raise ValueError('composition source requires one explicitly paired seed')
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('frozen teacher required')
    if fit.quarantine or any(role!='fit' for role in fit.roles.values()):raise ValueError('unquarantined fit-only population required')
    teacher_digest=teacher.manifest().checkpoint_digest
    spec=_spec(_hashed_json(binding.source_run_spec_path,binding.source_run_spec_sha256))
    inputs=thaw(spec.input_manifests);resolved=thaw(spec.resolved_config)
    training=resolved.get('training',{});objective=resolved.get('recovery_objective',{})
    horizon=training.get('max_updates')
    if (spec.seeds.get('fit')!=seed or inputs.get('assets')!=teacher.tokenizer_digest
            or type(horizon) is not int or horizon<1 or training.get('target_count')!=8192
            or objective.get('schema_version')!=1 or objective.get('owner_scope')!='affected_group'
            or objective.get('loss_scope') not in ('local_sum','local_mean','full_student')):
        raise ValueError('completed KL source seed/assets/objective/horizon mismatch')
    for role in ('fit','tuning'):
        expected=config.get('populations',{}).get(role,{}).get('sha256')
        if not isinstance(expected,str) or inputs.get(role)!=expected:raise ValueError('composition source population mismatch: '+role)
    state=_call(recorder,lambda:load_checkpoint(CheckpointRef(binding.checkpoint_directory,binding.checkpoint_manifest_sha256),spec))
    if (not state.safe_boundary or state.step!=horizon or state.target_cursor!=horizon*8192
            or state.teacher_digest!=teacher_digest or state.source_hash!=inputs.get('raw_plan')
            or state.source_hash!=objective.get('raw_plan_sha256')
            or state.scheduler!=_schedule_state(horizon,horizon)
            or state.config_hash!=_hash({'training':training,'recovery_objective':objective})):
        raise ValueError('incomplete or incompatible safe composition source')
    rank=config.get('rank')
    if type(rank) is not int or rank<1 or state.model_schema.get('rank')!=rank:raise ValueError('composition source rank mismatch')
    n=len(teacher.layers);expected=canonicalize_partition(binding.expected_partition,n)
    groups=tuple(tuple(g) for g in state.partition)
    source_partition=canonicalize_partition(groups,n)
    pair=binding.merge_pair
    if pair[-1]>=n:raise ValueError('composition pair outside source model')
    indices={source_partition.reuse_map[d] for d in pair}
    affected=tuple(sorted(d for i in indices for d in groups[i]))
    if len(indices)!=2 or affected!=tuple(config['pair']):raise ValueError('composition must cover every use of both source groups')
    if binding.forced_parent not in {groups[i][0] for i in indices}:raise ValueError('forced parent is not a source canonical representative')
    merged=canonicalize_partition([g for i,g in enumerate(groups) if i not in indices]+[affected],n)
    if merged!=expected:raise ValueError('composition final partition differs from frozen expectation')
    if config.get('schema_version')==6:
        from poc.recovery_study import _validate_recovery_groups
        _validate_recovery_groups(config,expected)
    # This new study executes the source backend/profile; it does not rewrite its
    # teacher reference to bypass identity gates or inherit warm optimizer state.
    template=_call(recorder,lambda:BankDecoder(teacher,source_partition,rank,seed,initialize=False))
    schema=_model_schema(template);del template
    actual=dict(state.model_schema)
    for value in (schema,actual):value['owners']=[{k:v for k,v in row.items() if k!='trainable'} for row in value['owners']]
    if actual!=schema:raise ValueError('composition source teacher/backend/scalar/owner schema mismatch')
    current=_call(recorder,lambda:checkpoint_decoder(state).to(device=teacher.device))
    if current.manifest().checkpoint_digest!=state.candidate_digest:raise ValueError('composition source executable identity differs')
    from poc.recovery_objectives import owner_names_hash
    old_depths=tuple(objective.get('affected_depths',()))
    old_names=owner_names_hash(current,old_depths)
    if old_names!=inputs.get('owner_names') or old_names!=objective.get('owner_names_sha256'):
        raise ValueError('composition source trainable owner bindings differ')
    subset_ids={b.document_ids[0] for b in fit_batches[:config['parent_fit_components']]}
    subset=restricted_population(fit,sorted(subset_ids),role='fit')
    batches=tuple(b for b in fit_batches if b.document_ids[0] in subset_ids)
    if not batches or len(subset_ids)!=config['parent_fit_components']:raise ValueError('insufficient deterministic parent fit components')
    moved=tuple(replace(b,**{key:getattr(b,key).to(teacher.device) for key in ('input_ids','attention_mask','target_mask','positions')}) for b in batches)
    targets=sum(int(b.target_mask.sum()) for b in batches)
    branches=_call(recorder,lambda:paired_composition_initializations(current,teacher,pair,subset,seed,batches=moved,recorder=_Counts(recorder) if recorder is not None else None,forced_parent=binding.forced_parent),
        valid_target_presentations=2*len(affected)*targets,raw_corpus_target_presentations=2*targets,
        processed_inputs=2*len(affected)*len(batches),processed_tokens=2*len(affected)*sum(int(b.attention_mask.sum()) for b in batches))
    trial=getattr(branches,binding.reference)
    trial.recorder=recorder
    if trial.model.partition!=expected or trial.affected_depths!=affected:raise ValueError('initialized composition shape differs')
    trial.initializer_record=dict(schema_version=1,kind='completed_source_composition_v1',
        source_spec_sha256=binding.source_run_spec_sha256,source_run_id=run_id(spec),
        source_checkpoint_manifest_sha256=binding.checkpoint_manifest_sha256,
        source_candidate_digest=state.candidate_digest,source_completed_updates=horizon,
        source_raw_targets=state.target_cursor,source_raw_plan_sha256=state.source_hash,
        teacher_digest=teacher_digest,reference=binding.reference,seed=seed,rank=rank,
        affected_depths=list(affected),after_digest=trial.initialized_digest,
        composition=thaw(branches.record),source_binding=binding.identity(),
        optimizer_policy='fresh; source optimizer/RNG/cursors/spending are not imported',
        cost_policy='both reference initializations and both parent scores charged per study; source physical ledger retained separately, not inferred from checkpoint snapshots')
    if config.get('schema_version')==6:
        from poc.recovery_study import recovery_depths
        depths=recovery_depths(config)
        trial.initializer_record.update(schema_version=2,trainable_depths=list(depths),
            owner_scope=config['owner_scope'],owner_names_sha256=owner_names_hash(
                trial.model,depths,owner_scope=config['owner_scope']))
    return trial

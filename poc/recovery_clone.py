"""Clone a hash-bound, verified STEP0 initialization into independent owners.

This is an initialization source, never a resume: no optimizer, RNG or spending
state is imported. The usual fitter must construct a fresh optimizer afterward.
"""
import hashlib
import math
from pathlib import Path
import re

import torch

from poc.bank import BankDecoder, canonicalize_partition
from poc.checkpoint import CheckpointRef, _read_json, _spec, checkpoint_decoder, load_checkpoint
from poc.contracts import thaw
from poc.merge import MergeTrial, _call
from poc.records import canonical_json, run_id
from poc.recovery_objectives import owner_manifest, owner_names_hash
from poc.train import _model_schema, _schedule_state


_BINDINGS = {'checkpoint_directory', 'checkpoint_manifest_sha256', 'run_spec_path',
             'run_spec_sha256', 'initialization_record_path', 'initialization_record_sha256'}


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _hashed_json(path, expected):
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise ValueError('detached source SHA256 required')
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('initialization source requires regular unlinked files')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('initialization source content hash mismatch')
    return _read_json(path)


def validate_initialization_binding(binding):
    """Closed opt-in STEP0 bindings; the original six-field contract stays strict."""
    if not isinstance(binding,dict):
        raise ValueError('closed initialization_checkpoint bindings required')
    version=binding.get('schema_version',1)
    if type(version) is not int or version not in (1,2):
        raise ValueError('unknown initialization checkpoint binding schema')
    expected=_BINDINGS if version==1 else _BINDINGS|{'schema_version','source_updates','source_populations'}
    if set(binding)!=expected:
        raise ValueError('closed initialization_checkpoint bindings required')
    for name in _BINDINGS:
        value=binding[name]
        if not isinstance(value,str) or not value.strip():
            raise ValueError('initialization checkpoint requires nonempty paths and SHA256 identities')
        if name.endswith('_sha256') and not re.fullmatch('[0-9a-f]{64}',value):
            raise ValueError('initialization checkpoint SHA256 identity required')
    if version==2:
        if type(binding['source_updates']) is not int or binding['source_updates']<1:
            raise ValueError('positive source initialization horizon required')
        refs=binding['source_populations']
        if not isinstance(refs,dict) or set(refs)!={'fit','tuning'}:
            raise ValueError('exact source fit/tuning population references required')
        for ref in refs.values():
            if (not isinstance(ref,dict) or set(ref)!={'path','sha256'}
                    or not isinstance(ref['path'],str) or not ref['path'].strip()
                    or not isinstance(ref['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',ref['sha256'])):
                raise ValueError('closed source population path/SHA256 required')
    return version


def verify_population_continuation(config, *, source_inputs=None):
    """Metadata only: same whole populations, explicit lineage and new journals.

    No text/token files or exposure journals are read or written here. Both the
    driver's pre-read validation and the STEP0 loader independently enforce this.
    """
    from dataclasses import asdict
    from poc.data.manifest import DataManifest,DocumentRecord
    binding=config['initialization_checkpoint']
    if validate_initialization_binding(binding)!=2:return {}
    if source_inputs is None:
        source_inputs=thaw(_spec(_hashed_json(binding['run_spec_path'],binding['run_spec_sha256'])).input_manifests)
    lineage={};source_journals=set();target_journals=set()
    for role,allowed in (('fit','fit'),('tuning','dev')):
        original=binding['source_populations'][role]
        target=config.get('populations',{}).get(role,{})
        if source_inputs.get(role)!=original['sha256']:
            raise ValueError('initialization source '+role+' population mismatch')
        raw_original=_hashed_json(original['path'],original['sha256'])
        raw_target=_hashed_json(target.get('path',''),target.get('sha256'))
        def manifest(raw):
            if not isinstance(raw,dict):raise ValueError('population metadata object required')
            body=dict(raw);records=body.pop('records',None)
            if not isinstance(records,list):raise ValueError('population records required')
            return DataManifest(records=tuple(DocumentRecord(**row) for row in records),**body)
        old,new=manifest(raw_original),manifest(raw_target)
        if (not old.records or any(r!=allowed for r in old.roles.values())
                or any(r!=allowed for r in new.roles.values())):
            raise ValueError('initialization population role mismatch')
        for m in (old,new):
            if not isinstance(m.exposure_path,str) or not Path(m.exposure_path).is_absolute():
                raise ValueError('absolute population exposure journal required')
        old_journal=Path(old.exposure_path).resolve()
        new_path=Path(new.exposure_path)
        if new_path.is_symlink() or (new_path.exists() and (not new_path.is_file() or new_path.stat().st_nlink!=1)):
            raise ValueError('new exposure journal must be a regular unaliased file')
        new_journal=new_path.resolve()
        if new.lineage_sha256!=old.digest():
            raise ValueError('initialization population lineage mismatch')
        old_body,new_body=asdict(old),asdict(new)
        for key in ('exposure_path','lineage_sha256'):
            old_body.pop(key);new_body.pop(key)
        if old_body!=new_body:
            raise ValueError('initialization population content/role/component mismatch')
        source_journals.add(old_journal);target_journals.add(new_journal)
        lineage[role]=dict(source_sha256=original['sha256'],source_object_sha256=old.digest(),
            target_sha256=target['sha256'],source_exposure_path=str(old_journal),
            target_exposure_path=str(new_journal))
    if source_journals & target_journals or len(target_journals)!=2:
        raise ValueError('initialization continuation requires separate new exposure journals')
    return lineage


def clone_initialization_checkpoint(teacher, config, seed, *, recorder=None, fit_batches=None):
    """Return a MergeTrial cloned from one explicitly bound initial checkpoint.

    Validate source lineage, complete singleton-plus-pair ownership, fit/tuning
    population identities and STEP0 state before allocating the target model.
    Copy every canonical parameter, including frozen backbone and other banks.
    """
    binding = config.get('initialization_checkpoint')
    version=validate_initialization_binding(binding)
    pair = tuple(config.get('pair',()))
    depth = len(teacher.layers)
    if (len(pair) != 2 or any(type(d) is not int for d in pair)
            or not 0 <= pair[0] < pair[1] < depth):
        raise ValueError('canonical initialization pair required')
    rank = config.get('rank')
    if (type(rank) is not int or rank < 0 or type(seed) is not int or not 0 <= seed < 2**63
            or seed not in config.get('seeds',()) or type(config.get('updates')) is not int
            or config['updates'] < 1 or config.get('raw_targets_per_update') != 8192):
        raise ValueError('initialization seed/rank/horizon and 8192 raw targets required')
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError('initialization requires frozen teacher')
    teacher_digest = teacher.manifest().checkpoint_digest
    source_spec = _spec(_hashed_json(binding['run_spec_path'],binding['run_spec_sha256']))
    record = _hashed_json(binding['initialization_record_path'],binding['initialization_record_sha256'])
    state = load_checkpoint(CheckpointRef(binding['checkpoint_directory'],
                                        binding['checkpoint_manifest_sha256']),source_spec)
    if (state.step != 0 or state.target_cursor != 0 or state.probe_cursor != 0
            or state.visited or state.events or state.budget_totals.get('updates',0) != 0):
        raise ValueError('initialization checkpoint must be untrained STEP0 with no visited/updates')
    if (not isinstance(state.optimizer,dict) or set(state.optimizer) != {'state','groups'}
            or not isinstance(state.optimizer['state'],dict)
            or any(not isinstance(v,dict) or v for v in state.optimizer['state'].values())):
        raise ValueError('initialization cannot import warmed optimizer state')
    resolved = thaw(source_spec.resolved_config)
    training = resolved.get('training',{})
    objective = resolved.get('recovery_objective',{})
    inputs = thaw(source_spec.input_manifests)
    if (source_spec.seeds.get('fit') != seed or state.teacher_digest != teacher_digest
            or inputs.get('assets') != teacher.tokenizer_digest
            or state.model_schema.get('base_digest') != teacher_digest
            or state.model_schema.get('tokenizer_digest') != teacher.tokenizer_digest):
        raise ValueError('initialization seed/teacher/assets lineage mismatch')
    population_lineage=verify_population_continuation(config,source_inputs=inputs) if version==2 else {}
    population_refs=binding['source_populations'] if version==2 else config.get('populations',{})
    for role in ('fit','tuning'):
        expected = population_refs.get(role,{}).get('sha256')
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected) or inputs.get(role) != expected:
            raise ValueError('initialization '+role+' population mismatch')
    partition = canonicalize_partition([pair]+[(d,) for d in range(depth) if d not in pair],depth)
    groups = partition.groups
    if (state.partition != groups or state.model_schema.get('groups') != [list(g) for g in groups]
            or state.model_schema.get('rank') != rank
            or tuple(objective.get('affected_depths',())) != pair
            or objective.get('owner_scope') != 'affected_group'
            or objective.get('schema_version') != 1
            or objective.get('loss_scope') not in ('local_sum','local_mean','full_student')):
        raise ValueError('initialization pair/rank/partition/owner scope mismatch')
    source_updates=binding['source_updates'] if version==2 else config['updates']
    if (training.get('max_updates') != source_updates or training.get('target_count') != 8192
            or tuple(training.get('affected_depths',())) != pair
            or state.scheduler != _schedule_state(0,source_updates)):
        raise ValueError('initialization training horizon/raw-target schedule mismatch')
    if (inputs.get('raw_plan') != state.source_hash
            or objective.get('raw_plan_sha256') != state.source_hash
            or tuple(state.order) != ('raw_plan_v1',state.source_hash)
            or state.probe_order
            or state.config_hash != _hash({'training':training,'recovery_objective':objective})):
        raise ValueError('initialization raw plan/training identity mismatch')
    if version==2:
        if fit_batches is None:
            raise ValueError('source-plan verification requires exact fit batches')
        from poc.recovery_study import NaturalWindowPlan
        fit_batches=tuple(fit_batches)
        source_plan=NaturalWindowPlan.build(fit_batches,updates=source_updates,
            raw_targets_per_update=8192,seed=seed,
            manifest_hashes={'fit':inputs['fit'],'tokenizer':inputs['assets']})
        if source_plan.digest!=state.source_hash:
            raise ValueError('initialization source raw plan differs from supplied fit batches')
        target_plan=NaturalWindowPlan.build(fit_batches,updates=config['updates'],
            raw_targets_per_update=8192,seed=seed,manifest_hashes={
                'fit':config['populations']['fit']['sha256'],'tokenizer':inputs['assets']})
    if (inputs.get('initialized_state') != state.candidate_digest
            or not isinstance(record,dict) or record.get('initialized_digest') != state.candidate_digest
            or record.get('selected_parent') not in pair
            or type(record.get('selected_parent')) is not int
            or inputs.get('initialized_owners') != _hash({'owners':record.get('owners')})):
        raise ValueError('initialization record/state/owners mismatch')
    scores = record.get('parent_scores')
    if (not isinstance(scores,list) or len(scores) != 2
            or any(not isinstance(row,dict) for row in scores)
            or [row.get('parent') for row in scores] != list(pair)
            or any(type(row.get('mse')) not in (float,int) or not math.isfinite(row['mse'])
                   or row['mse'] < 0 for row in scores)
            or min(scores,key=lambda row:row['mse'])['parent'] != record['selected_parent']):
        raise ValueError('initialization parent scores mismatch')
    # Safe loader reconstructs the scientific schema on CPU and verifies the
    # saved executable digest using its original device identity.
    source = checkpoint_decoder(state)
    owners_identity = _hash({'owners':owner_manifest(source,pair)})
    owner_names = owner_names_hash(source,pair)
    if (owners_identity != inputs.get('initialized_owners')
            or owner_names != inputs.get('owner_names')
            or owner_names != objective.get('owner_names_sha256')):
        raise ValueError('initialization canonical owner identity mismatch')
    # No SVD is run; initialize=False allocates only the eventual owner tensors.
    candidate = _call(recorder,lambda:BankDecoder(teacher,partition,rank,seed,initialize=False))
    expected_schema = _model_schema(candidate)
    # Trainability flags are selected again by the target fitter. Every other
    # execution/owner field must match before a single parameter is copied.
    source_schema = dict(state.model_schema)
    source_schema['owners'] = [{k:v for k,v in row.items() if k != 'trainable'}
                              for row in source_schema['owners']]
    expected_schema['owners'] = [{k:v for k,v in row.items() if k != 'trainable'}
                                for row in expected_schema['owners']]
    if source_schema != expected_schema:
        raise ValueError('initialization executable schema mismatch')
    params = dict(candidate.named_parameters())
    if set(params) != set(state.model):
        raise ValueError('initialization canonical parameter names mismatch')
    for name,p in params.items():
        value = state.model[name]
        if (not isinstance(value,torch.Tensor) or value.device.type != 'cpu'
                or value.shape != p.shape or value.dtype != p.dtype):
            raise ValueError('initialization parameter shape/dtype mismatch: '+name)
    with torch.no_grad():
        for name,p in params.items():
            # Source is already CPU in its native dtype. Never combine a device
            # transfer and FP64 conversion (Torch 2.14.1 MPS silently zeroes it).
            p.copy_(state.model[name].to(device=p.device))
    if (candidate.manifest().checkpoint_digest != state.candidate_digest
            or _hash({'owners':owner_manifest(candidate,pair)}) != owners_identity
            or teacher.manifest().checkpoint_digest != teacher_digest):
        raise ValueError('initialization clone digest/owner verification failed')
    trial = MergeTrial(candidate,teacher,pair,pair,record['selected_parent'],tuple(scores),
                       state.candidate_digest,recorder=recorder)
    trial.clone_provenance = dict(schema_version=1,kind='step0_initialization_checkpoint_clone_v1',
        checkpoint_manifest_sha256=binding['checkpoint_manifest_sha256'],
        run_spec_sha256=binding['run_spec_sha256'],
        initialization_record_sha256=binding['initialization_record_sha256'],
        source_run_id=run_id(source_spec),raw_plan_sha256=state.source_hash,
        owner_names_sha256=owner_names,initialized_state=state.candidate_digest,
        initialized_owners=owners_identity,initialized_owners_sha256=owners_identity,
        seed=seed,rank=rank,pair=list(pair),
        step=0,target_cursor=0,probe_cursor=0,
        optimizer_policy='fresh_optimizer; no optimizer/RNG/cursors/spending imported')
    if version==2:
        trial.clone_provenance.update(schema_version=2,kind='step0_initialization_checkpoint_clone_v2',
            source_updates=source_updates,target_updates=config['updates'],
            source_raw_plan_sha256=state.source_hash,target_raw_plan_sha256=target_plan.digest,
            population_lineage=population_lineage,
            population_policy='identical whole populations; new exposure journals; source manifest lineage',
            schedule_policy='weights-only STEP0 clone; new finite plan and completed schedule')
    return trial

"""Closed contracts for separately costed warm and fresh-KL recovery phases."""
from pathlib import Path
import hashlib
import math
import re
import yaml
from poc.contracts import Budget,freeze,thaw
from poc.records import canonical_json
from poc.train import TrainingConfig


_FIELDS={'schema_version','study','asset_root','out','pair','rank','seed','calibration_components',
 'warm_loss_scope','warm_updates','recovery_updates','owner_scope','alpha','observations',
 'checkpoint_every','optimizer','populations','initialization_checkpoint','numerical_profile',
 'profile_resource_budget','resource_budget','retained_dependency_roots'}
RESERVED_STUDIES=frozenset(('continuation-v1','continuation-v2','training','diagnostics','validation'))
_CHECKPOINT={'checkpoint_directory','checkpoint_manifest_sha256','run_spec_path','run_spec_sha256',
 'initialization_record_path','initialization_record_sha256'}


def _closed(value,keys,name):
 if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('closed '+name+' fields required')


def _sha(value,name):
 if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('SHA256 required: '+name)


def _positive(value,name):
 if type(value) is not int or value<1:raise ValueError('positive integer required: '+name)


def _path(value,name,*,optional=False):
 if value is None and optional:return
 if not isinstance(value,str) or not value or not Path(value).is_absolute() or '..' in Path(value).parts:raise ValueError('absolute path required: '+name)


def _budget(raw,name):
 _closed(raw,Budget.__dataclass_fields__,name)
 result=Budget(**raw)
 if any(getattr(result,k)<=0 for k in ('wall_seconds','memory_bytes','disk_bytes')):raise ValueError('finite positive wall/memory/disk budget required: '+name)
 return result


def expected_phase_counts(config,phase):
 if phase not in ('warm','recovery'):raise ValueError('unknown optimization phase')
 updates=config['warm_updates' if phase=='warm' else 'recovery_updates'];raw=updates*8192
 factor=len(config['pair']) if phase=='warm' and config['warm_loss_scope']=='ffn_mse_sum' else 1
 unique_components=config['calibration_components'] if phase=='warm' else config['populations']['fit']['components']
 observation_targets=len(config['observations'][phase])*(8192*(1+len(config['pair']))+config['populations']['tuning']['components']*511)
 return {'updates':updates,'raw_targets':raw,'scored_training_targets':raw*factor,
         'observation_targets':observation_targets,'total_scored_targets':raw*factor+observation_targets,
         'observation_count':len(config['observations'][phase]),
         'distinct_source_targets':511*unique_components}


def study_digest(config):return hashlib.sha256(canonical_json(thaw(config))).hexdigest()


def load_reconstruction_study(path):
 raw=yaml.safe_load(Path(path).read_text());_closed(raw,_FIELDS,'reconstruction study')
 if type(raw['schema_version']) is not int or raw['schema_version'] not in (1,2):raise ValueError('unknown reconstruction study schema')
 name=raw['study']
 if isinstance(name,str) and name in RESERVED_STUDIES:raise ValueError('reserved historical study identity')
 if not isinstance(name,str) or re.fullmatch('[a-z0-9][a-z0-9-]{0,100}',name) is None or raw['out']!='artifacts/'+name:raise ValueError('isolated artifacts/study output required')
 _path(raw['asset_root'],'asset_root',optional=True)
 pair=raw['pair']
 if not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int for v in pair) or not 0<=pair[0]<pair[1]<30:raise ValueError('canonical two-depth pair required')
 if type(raw['rank']) is not int or raw['rank'] not in (8,16,32,64):raise ValueError('qualified rank ladder required')
 if type(raw['seed']) is not int or not 0<=raw['seed']<2**63:raise ValueError('int64 seed required')
 for k in ('warm_updates','recovery_updates','checkpoint_every','calibration_components'):_positive(raw[k],k)
 if raw['warm_loss_scope'] not in ('ffn_mse_sum','full_student') or raw['owner_scope']!='affected_group' or type(raw['alpha']) not in (int,float) or raw['alpha']!=0:raise ValueError('named warm objective, complete affected owners and alpha0 required')
 _closed(raw['observations'],{'warm','recovery'},'phase observations')
 for phase,horizon in [('warm',raw['warm_updates']),('recovery',raw['recovery_updates'])]:
  observations=raw['observations'][phase]
  if not isinstance(observations,list) or not observations or any(type(v) is not int for v in observations) or observations!=sorted(set(observations)) or observations[0]!=0 or observations[-1]!=horizon or any(v<0 or v>horizon for v in observations):raise ValueError('observations must include initial/final completed phase boundaries')
 defaults=TrainingConfig(max_updates=1)
 expected={k:getattr(defaults,k) for k in ('bank_lr','residual_lr','weight_decay','lr_factor')}
 _closed(raw['optimizer'],expected,'optimizer')
 if any(type(v) not in (int,float) or not math.isfinite(v) for v in raw['optimizer'].values()) or raw['optimizer']!=expected:raise ValueError('this no-HPO study preserves inherited optimizer defaults')
 _closed(raw['populations'],{'fit','tuning'},'populations')
 for role,pop in raw['populations'].items():
  _closed(pop,{'path','sha256','components'},role+' population');_positive(pop['components'],role+' components')
  if (pop['path'] is None)!=(pop['sha256'] is None):raise ValueError('population path/hash must resolve together')
  _path(pop['path'],role+' manifest',optional=True)
  if pop['sha256'] is not None:_sha(pop['sha256'],role)
 if raw['calibration_components']>raw['populations']['fit']['components']:raise ValueError('calibration exceeds fit population')
 source=raw['initialization_checkpoint']
 if source is not None:
  if raw['schema_version']==1:
   _closed(source,_CHECKPOINT,'STEP0 source')
   for k,v in source.items():
    if k.endswith('_sha256'):_sha(v,k)
    else:_path(v,k)
  else:
   from poc.recovery_clone import validate_initialization_binding
   if validate_initialization_binding(source)!=2:raise ValueError('reconstruction schema2 requires explicit initialization binding2')
 n=raw['numerical_profile'];_closed(n,{'device','dtype','chunk_size','kl_floor_per_target','decision_resolution'},'numerical profile')
 if n['device'] not in ('cpu','mps') or n['dtype']!='float32' or type(n['chunk_size']) is not int or n['chunk_size']!=32:raise ValueError('qualified FP32/chunk32 scientific backend required')
 if any(type(n[k]) not in (int,float) or not math.isfinite(n[k]) for k in ('kl_floor_per_target','decision_resolution')) or not 0<=n['kl_floor_per_target']<=n['decision_resolution'] or n['decision_resolution']<=0:raise ValueError('finite qualified numerical floor/resolution required')
 roots=raw['retained_dependency_roots']
 if not isinstance(roots,list) or len(roots)!=len(set(roots)):raise ValueError('explicit disjoint dependency roots required')
 for value in roots:_path(value,'retained dependency')
 for i,left in enumerate(map(Path,roots)):
  if any(left.is_relative_to(Path(right)) or Path(right).is_relative_to(left) for right in roots[i+1:]):raise ValueError('overlapping dependency roots')
 _closed(raw['profile_resource_budget'],{'setup','warm','recovery'},'profile budgets')
 for phase,expected_updates in [('setup',0),('warm',raw['warm_updates']),('recovery',1)]:
  b=_budget(raw['profile_resource_budget'][phase],phase+' profile')
  if b.updates!=expected_updates:raise ValueError('profile requires complete warm rehearsal and exactly one recovery update; setup has zero updates')
  if phase!='setup':
   scope=raw['warm_loss_scope'] if phase=='warm' else 'full_student';factor=len(pair) if scope=='ffn_mse_sum' else 1
   minimum=expected_updates*8192*factor+8192*(1+len(pair))+raw['populations']['tuning']['components']*511
   if b.valid_target_presentations<minimum:raise ValueError('profile cannot cover complete phase work and observation')
 if raw['resource_budget'] is not None:
  r=raw['resource_budget'];_closed(r,{'frozen_profile_sha256','per_phase','campaign'},'run resource budget');_sha(r['frozen_profile_sha256'],'profile');_closed(r['per_phase'],{'warm','recovery'},'phase budgets')
  for phase in ('warm','recovery'):
   b=_budget(r['per_phase'][phase],phase+' run');counts=expected_phase_counts(raw,phase)
   if b.updates<counts['updates'] or b.valid_target_presentations<counts['total_scored_targets']:raise ValueError('scored budget cannot complete phase plus observations')
  _closed(r['campaign'],{'wall_seconds','memory_bytes','disk_bytes'},'campaign budget')
  for k,v in r['campaign'].items():
   if type(v) not in (int,float) or not math.isfinite(v) or v<=0 or k!='wall_seconds' and type(v) is not int:raise ValueError('finite positive campaign budget required')
 return freeze(raw)


def unresolved(config):
 missing=[]
 if config['asset_root'] is None:missing.append('asset_root')
 if config['initialization_checkpoint'] is None:missing.append('initialization_checkpoint')
 for role in ('fit','tuning'):
  if config['populations'][role]['path'] is None:missing.append(role)
 if not config['retained_dependency_roots']:missing.append('retained_dependency_roots')
 if config['resource_budget'] is None:missing.append('resource_budget')
 return tuple(missing)


def require_run_ready(config):
 if unresolved(config):raise ValueError('unresolved study inputs or unfrozen resource budget: '+','.join(unresolved(config)))


def initialization_lineage(config):
 """Verify source files/scalars without deserializing tensors or reading text.

 Schema1 keeps its original exact-horizon checks at its existing call sites.
 Schema2 deliberately separates an original STEP0 source from new target plans.
 """
 c=thaw(config)
 if c['schema_version']!=2:raise ValueError('explicit reconstruction schema2 lineage required')
 from poc.recovery_clone import validate_initialization_binding,_hashed_json
 from poc.checkpoint import _spec,_verify,_decode,CheckpointRef
 from poc.records import run_id
 binding=c['initialization_checkpoint']
 if validate_initialization_binding(binding)!=2:raise ValueError('explicit initialization binding2 required')
 spec=_spec(_hashed_json(binding['run_spec_path'],binding['run_spec_sha256']))
 inputs=thaw(spec.input_manifests);resolved=thaw(spec.resolved_config)
 training=resolved.get('training',{});objective=resolved.get('recovery_objective',{})
 if (spec.seeds.get('fit')!=c['seed'] or training.get('max_updates')!=binding['source_updates']
     or training.get('target_count')!=8192 or tuple(training.get('affected_depths',()))!=tuple(c['pair'])
     or objective.get('schema_version')!=1 or objective.get('owner_scope')!='affected_group'
     or tuple(objective.get('affected_depths',()))!=tuple(c['pair'])
     or objective.get('loss_scope') not in ('local_sum','local_mean','full_student')
     or objective.get('raw_plan_sha256')!=inputs.get('raw_plan')
     or objective.get('owner_names_sha256')!=inputs.get('owner_names')
     or any(inputs.get(role)!=binding['source_populations'][role]['sha256'] for role in ('fit','tuning'))):
  raise ValueError('reconstruction source initialization metadata/plan/horizon/owner mismatch')
 for key in ('raw_plan','assets','owner_names','initialized_state','initialized_owners'):_sha(inputs.get(key),'source '+key)
 # Known invalid references must fail before output ownership or raw reads.
 # Hashing the safe tensor payload is an integrity read, not model loading.
 for key in ('checkpoint_directory','run_spec_path','initialization_record_path'):
  path=Path(binding[key])
  if not path.is_absolute() or path.resolve()!=path or path.is_symlink():
   raise ValueError('canonical unaliased source paths required')
 directory=Path(binding['checkpoint_directory'])
 _hashed_json(directory/'manifest.json',binding['checkpoint_manifest_sha256'])
 manifest=_verify(CheckpointRef(str(directory),binding['checkpoint_manifest_sha256']))
 if manifest['run_id']!=run_id(spec):raise ValueError('source checkpoint manifest/RunSpec mismatch')
 record=_hashed_json(binding['initialization_record_path'],binding['initialization_record_sha256'])
 if (not isinstance(record,dict) or record.get('initialized_digest')!=inputs['initialized_state']
     or hashlib.sha256(canonical_json({'owners':record.get('owners')})).hexdigest()!=inputs['initialized_owners']):
  raise ValueError('source initialization record state/owner mismatch')
 scores=record.get('parent_scores');pair=list(c['pair'])
 if (type(record.get('selected_parent')) is not int or record['selected_parent'] not in pair
     or not isinstance(scores,list) or len(scores)!=2 or any(not isinstance(row,dict) for row in scores)
     or [row.get('parent') for row in scores]!=pair
     or any(type(row.get('mse')) not in (int,float) or not math.isfinite(row['mse']) or row['mse']<0 for row in scores)
     or min(scores,key=lambda row:row['mse'])['parent']!=record['selected_parent']):
  raise ValueError('source initialization parent record mismatch')
 encoded=_hashed_json(directory/'state.json',manifest['files']['state.json'])
 if not isinstance(encoded,dict) or set(encoded)!={'tag','items'} or encoded['tag']!='dict' or not isinstance(encoded['items'],dict):
  raise ValueError('source scalar checkpoint metadata malformed')
 scalar_fields=('step','target_cursor','probe_cursor','visited','events','safe_boundary','candidate_digest',
  'teacher_digest','model_schema','partition','optimizer','scheduler','source_hash','config_hash','order','probe_order','budget_totals','spec')
 if any(key not in encoded['items'] for key in scalar_fields):raise ValueError('source scalar checkpoint metadata incomplete')
 # The qualified safe decoder rejects tensor references in these metadata
 # fields. Actual model/RNG tensors remain the full clone loader's job.
 state={key:_decode(encoded['items'][key],{},set()) for key in scalar_fields}
 from poc.train import _schedule_state
 optimizer=state['optimizer'];schema=state['model_schema']
 if (state['safe_boundary'] is not True or any(type(state[key]) is not int or state[key]!=0 for key in ('step','target_cursor','probe_cursor'))
     or state['visited'] or state['events'] or state['budget_totals'].get('updates',0)!=0
     or state['candidate_digest']!=inputs['initialized_state']
     or state['source_hash']!=inputs['raw_plan'] or tuple(state['order'])!=('raw_plan_v1',inputs['raw_plan'])
     or state['probe_order'] or state['scheduler']!=_schedule_state(0,binding['source_updates'])
     or state['config_hash']!=hashlib.sha256(canonical_json({'training':training,'recovery_objective':objective})).hexdigest()
     or canonical_json(state['spec'])!=canonical_json(spec.to_dict())
     or schema.get('base_digest')!=state['teacher_digest'] or schema.get('tokenizer_digest')!=inputs['assets']
     or schema.get('rank')!=c['rank']
     or not isinstance(optimizer,dict) or set(optimizer)!={'groups','state'} or not isinstance(optimizer['state'],dict)
     or any(not isinstance(v,dict) or v for v in optimizer['state'].values())):
  raise ValueError('source checkpoint is not consistent untrained STEP0 metadata')
 from poc.bank import canonicalize_partition
 depth=schema.get('config',{}).get('num_hidden_layers')
 if type(depth) is not int or depth<=max(pair):raise ValueError('source checkpoint depth metadata mismatch')
 groups=canonicalize_partition([pair]+[(d,) for d in range(depth) if d not in pair],depth).groups
 if tuple(tuple(group) for group in state['partition'])!=groups or schema.get('groups')!=[list(group) for group in groups]:raise ValueError('source complete-use partition metadata mismatch')
 return {'schema_version':1,'binding_schema_version':2,'binding_sha256':hashlib.sha256(canonical_json(binding)).hexdigest(),
  'source_updates':binding['source_updates'],'target_updates':c['recovery_updates'],
  'source_checkpoint_state_sha256':manifest['files']['state.json'],'source_teacher_digest':state['teacher_digest'],
  'source_raw_plan_sha256':inputs['raw_plan'],'source_assets_sha256':inputs['assets'],
  'source_initialized_state':inputs['initialized_state'],'source_initialized_owners':inputs['initialized_owners'],
  'source_owner_names_sha256':inputs['owner_names'],'source_fit_sha256':inputs['fit'],
  'source_tuning_sha256':inputs['tuning'],'source_run_spec_sha256':binding['run_spec_sha256'],
  'target_fit_sha256':c['populations']['fit']['sha256'],'target_tuning_sha256':c['populations']['tuning']['sha256'],
  'policy':'exact original STEP0 weights; new journals/lineage and fresh optimizer/RNG/cursors/schedule'}


def recovery_config(config):
 """Reuse the original full recovery plan/horizon when cloning STEP0."""
 c=thaw(config);budget=c['resource_budget']
 return {'schema_version':4,'study':c['study'],'asset_root':c['asset_root'],'out':c['out'],
  'pair':c['pair'],'rank':c['rank'],'seeds':[c['seed']],'loss_scopes':['full_student'],
  'owner_scope':c['owner_scope'],'alpha':0,'raw_targets_per_update':8192,
  'updates':c['recovery_updates'],'observations':c['observations']['recovery'],
  'checkpoint_every':c['checkpoint_every'],'parent_fit_components':c['calibration_components'],
  'populations':c['populations'],'numerical_profile':c['numerical_profile'],
  'resource_budget':None if budget is None else {'frozen_profile_sha256':budget['frozen_profile_sha256'],'per_arm':{'full_student':budget['per_phase']['recovery']}},
  'preparation':None,'observation_policy':'optimizer_plus_tuning_v2',
  'initializer':{'schema_version':1,'method':'parent_svd','calibration_components':c['calibration_components'],'ridge_relative':1e-6,'max_condition':1e8},
  'profile_resource_budget':c['profile_resource_budget']['recovery'],
  'initialization_checkpoint':c['initialization_checkpoint']}

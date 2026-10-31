"""Two named phases using the existing common optimizer/checkpoint loop.

The caller admits outputs and verifies controls/resources before loading inputs.
This module never opens a data population, downloads assets or changes budgets.
"""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib,json,time
from pathlib import Path

import torch
from poc.contracts import RunSpec,CostRecord,thaw
from poc.records import canonical_json,run_id,start_attempt
from poc.budget import BudgetTracker
from poc.checkpoint import load_checkpoint,resolve_checkpoint,CheckpointRef
from poc.recovery_study import NaturalWindowPlan
from poc.recovery_objectives import RecoveryObjectiveSpec,owner_names_hash,owner_manifest
from poc.reconstruction_study import study_digest,initialization_lineage
from poc.recovery_phases import OptimizationHandoffSpec,prepare_optimization_handoff
from poc.train import (TrainingConfig,TrainingSources,fit_model,_observe,finish_attempt_cost,
                       configure_trainable_owners,_optimizer,_restore,_schedule_state)


def _hash(value):return hashlib.sha256(canonical_json(value)).hexdigest()


def _driver():
 from provenance import recovery_diagnosis
 return recovery_diagnosis


def build_phase_plans(config,fit_batches,*,tokenizer_digest):
 """Freeze CPU templates once; subset selection precedes model outcomes."""
 batches=tuple(fit_batches)
 if len(batches)!=config['populations']['fit']['components']:raise ValueError('fit component count differs')
 for b in batches:
  if (b.input_ids.shape!=(1,512) or b.target_mask.shape!=(1,511) or
      int(b.target_mask.sum())!=511 or not bool(b.attention_mask.all()) or
      b.positions[0].tolist()!=list(range(512))):
   raise ValueError('eligible complete512-token context with511 targets required')
 subset=batches[:config['calibration_components']]
 # Full template and component identities bind the deterministic first-N rule.
 subset_hash=_hash({'rule':'first_components_in_frozen_manifest_order_v1',
  'windows':[asdict(__import__('poc.recovery_study',fromlist=['NaturalWindow']).NaturalWindow.from_batch(b)) for b in subset]})
 common={'fit':config['populations']['fit']['sha256'],'tokenizer':tokenizer_digest}
 return {'warm':NaturalWindowPlan.build(subset,updates=config['warm_updates'],seed=config['seed'],
          manifest_hashes=dict(common,calibration_subset=subset_hash)),
         'recovery':NaturalWindowPlan.build(batches,updates=config['recovery_updates'],seed=config['seed'],manifest_hashes=common)}


def phase_config(config,phase,plan,initial,*,source,environment,budget,controls_sha256,mode='run'):
 if phase not in ('warm','recovery') or mode not in ('run','profile'):raise ValueError('named phase/mode required')
 horizon=config['warm_updates' if phase=='warm' else 'recovery_updates']
 if plan.updates!=horizon or plan.seed!=config['seed']:raise ValueError('phase plan horizon/seed mismatch')
 scope=config['warm_loss_scope'] if phase=='warm' else 'full_student'
 obj=RecoveryObjectiveSpec(scope,'affected_group',tuple(config['pair']),plan.digest,
  owner_names_hash(initial,tuple(config['pair'])),schema_version=2 if scope=='ffn_mse_sum' else 1,
  checkpoint_every=config['checkpoint_every'])
 cfg=TrainingConfig(max_updates=horizon,alpha=0,affected_depths=tuple(config['pair']),
  observation_steps=tuple(x for x in config['observations'][phase] if x),record_norms=True,
  numerical_profile='nonlinear-reconstruction-fp32-v1',kl_floor_per_target=config['numerical_profile']['kl_floor_per_target'],
  **dict(config['optimizer']))
 inputs={'study_contract':study_digest(config),'raw_plan':plan.digest,'owner_names':obj.owner_names_sha256,
  'assets':initial.tokenizer_digest,'fit':config['populations']['fit']['sha256'],
  'tuning':config['populations']['tuning']['sha256'],'executed_source':_hash(source),
  'controls':controls_sha256,'initialized_state':initial.manifest().checkpoint_digest,
  'initialized_owners':_hash({'owners':owner_manifest(initial,tuple(config['pair']))})}
 if config['initialization_checkpoint'] is not None:
  binding=config['initialization_checkpoint'];inputs.update(initialization_checkpoint=binding['checkpoint_manifest_sha256'],initialization_source_spec=binding['run_spec_sha256'])
 resolved={'training':cfg.identity(),'recovery_objective':obj.identity(),'study_contract_sha256':study_digest(config),'phase':phase,'execution_mode':mode}
 if config['schema_version']==2:
  lineage=initialization_lineage(config)
  if (lineage['source_initialized_state']!=inputs['initialized_state']
      or lineage['source_initialized_owners']!=inputs['initialized_owners']
      or lineage['source_owner_names_sha256']!=obj.owner_names_sha256
      or lineage['source_assets_sha256']!=initial.tokenizer_digest
      or lineage['source_teacher_digest']!=initial.base_digest):
   raise ValueError('phase original initialization/owner/assets identity mismatch')
  inputs['initialization_source_raw_plan']=lineage['source_raw_plan_sha256']
  inputs['initialization_binding']=lineage['binding_sha256']
  resolved['initialization_lineage']=lineage
 spec=RunSpec('R',config['study']+':'+mode+':'+phase+':'+scope,
  resolved,
  source['commit'],_hash(source),_hash(environment),inputs,
  {'device':config['numerical_profile']['device'],'environment':environment,'use_case':'offline English batch1 context512'},
  {'fit':config['seed']},(),budget)
 return replace(cfg,run_spec=spec),obj


def _safe_state(arm,cfg):
 root=arm/'resume'
 if (root/'current.json').is_file():return load_checkpoint(resolve_checkpoint(root),cfg.run_spec)
 if (root/'anchor/current.json').is_file():return load_checkpoint(resolve_checkpoint(root/'anchor'),cfg.run_spec)
 return None


def _completed(arm,cfg,plan,obj):
 path=arm/'result.json'
 if not path.is_file():return None
 value=json.loads(path.read_text())
 if value.get('run_id')!=run_id(cfg.run_spec) or value.get('study_contract_sha256')!=cfg.run_spec.input_manifests['study_contract']:
  raise ValueError('completed phase identity mismatch')
 ref=CheckpointRef(**value['checkpoint']);state=load_checkpoint(ref,cfg.run_spec)
 if (not state.safe_boundary or state.step!=cfg.max_updates or state.target_cursor!=cfg.max_updates*8192 or
     state.scheduler!=_schedule_state(cfg.max_updates,cfg.max_updates) or state.source_hash!=plan.digest or
     state.candidate_digest!=value.get('final_digest') or value.get('step')!=state.step):
  raise ValueError('incomplete or forged completed phase checkpoint')
 from poc.train import _recovery_cursors,_training_hash
 if state.config_hash!=_training_hash(cfg,obj) or (state.target_cursor,state.probe_cursor)!=_recovery_cursors(TrainingSources(raw_plan=plan),cfg,state.step,obj):
  raise ValueError('completed phase plan/objective/cursors mismatch')
 if cfg.run_spec.resolved_config['execution_mode']=='run':
  expected={f'{step:04d}.json' for step in (0,*cfg.observation_steps)}
  observed={p.name:_driver().sha(p) for p in (arm/'observations').glob('*.json')}
  if set(observed)!=expected or value.get('observations')!=observed:raise ValueError('completed observation metric bytes or coverage changed')
 return value


def _validate_observations(arm,cfg):
 """Validate even earlier observations when resuming a later safe checkpoint."""
 d=_driver()
 for p in (arm/'observations').glob('*.json'):
  value=json.loads(p.read_text());step=value.get('step')
  if (type(step) is not int or p.stem!=f'{step:04d}' or step not in (0,*cfg.observation_steps) or
      value.get('study_contract_sha256')!=cfg.run_spec.input_manifests['study_contract'] or
      value.get('run_id')!=run_id(cfg.run_spec)):
   raise ValueError('observation identity mismatch')
  delta=arm/'deltas'/p.stem;manifest=json.loads((delta/'manifest.json').read_text())
  if (manifest['candidate_digest']!=value['candidate_digest'] or manifest['weights_sha256']!=d.sha(delta/'owners.safetensors') or
      manifest['run_spec_sha256']!=_hash(cfg.run_spec.to_dict())):
   raise ValueError('observation delta integrity mismatch')


def execute_phases(*,config,teacher,initial,plans,fit_batches,tune_batches,source,environment,
                   controls_sha256,stage,budgets,observer,recorder_factory=None,resume=False,
                   stop_phase=None,stop_after_updates=None,mode='run',ledger_root=None):
 """Complete warm before fresh KL. A safe stop never pretends completion.

 Profile executes the entire declared warm rehearsal and just one KL update.
 Run resume reuses verified completed warm state and cumulative spending.
 """
 if mode not in ('run','profile') or resume and mode!='run':raise ValueError('invalid phase execution mode/resume')
 if stop_phase not in (None,'warm','recovery') or stop_phase is None and stop_after_updates is not None:raise ValueError('explicit safe-stop phase required')
 if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('frozen teacher required')
 stage=Path(stage);stage.mkdir(parents=True,exist_ok=resume)
 d=_driver();d.write_or_verify(stage/'phase-contract.json',thaw(config))
 for phase,plan in plans.items():d.write_or_verify(stage/(phase+'-plan.json'),thaw(plan.metadata))
 report={'status':'complete','phases':{},'study_contract_sha256':study_digest(config),'mode':mode}
 warm_ref=None;warm_spec=None;recovery_model=None
 for phase in ('warm','recovery'):
  plan=plans[phase];b=budgets[phase]
  cfg,obj=phase_config(config,phase,plan,initial,source=source,environment=environment,budget=b,controls_sha256=controls_sha256,mode=mode)
  model=deepcopy(initial) if phase=='warm' else None
  if phase=='recovery':
   handoff=prepare_optimization_handoff(teacher=teacher,initial=initial,checkpoint=warm_ref,warm_run_spec=warm_spec,warm_plan=plans['warm'],
    recovery_config=cfg,recovery_objective=obj,handoff_spec=OptimizationHandoffSpec(
     run_id(warm_spec),warm_ref.manifest_sha256,plans['warm'].digest,plan.digest,teacher.manifest().checkpoint_digest,
     obj.owner_names_sha256,initial.manifest().checkpoint_digest,config['warm_updates'],config['seed'],warm_loss_scope=config['warm_loss_scope']))
   model=handoff.student;cfg=handoff.config;d.write_or_verify(stage/'handoff.json',thaw(handoff.record))
  arm=stage/phase;arm.mkdir(exist_ok=resume)
  cfg=replace(cfg,checkpoint_root=arm/'resume');d.write_or_verify(arm/'spec.json',cfg.run_spec.to_dict())
  _validate_observations(arm,cfg)
  previous=_completed(arm,cfg,plan,obj)
  if previous is not None:
   # Revalidate observations/candidate state before importing a completed phase.
   report['phases'][phase]=previous
   if phase=='warm':warm_ref=CheckpointRef(**previous['checkpoint']);warm_spec=cfg.run_spec
   continue
  attempt=start_attempt(cfg.run_spec,stage/'ledgers' if ledger_root is None else Path(ledger_root))
  rec=BudgetTracker(b,attempt) if recorder_factory is None else recorder_factory(b,attempt)
  begun=time.monotonic();initial_digest=model.manifest().checkpoint_digest
  try:
   rec.check(CostRecord())
   def observe(step):
    path=arm/'observations'/f'{step:04d}.json';delta=arm/'deltas'/f'{step:04d}'
    if path.exists():
     previous=json.loads(path.read_text())
     if previous['candidate_digest']!=model.manifest().checkpoint_digest:raise ValueError('resume observation state mismatch')
     return
    value=dict(observer(teacher,model,fit_batches,tune_batches,plan,step,rec,config))
    value.update(step=step,candidate_digest=model.manifest().checkpoint_digest,plan_sha256=plan.digest,
     study_contract_sha256=study_digest(config),run_id=run_id(cfg.run_spec),phase=phase)
    # Publish the delta first, so an observer crash cannot leave a metric with
    # no independently bound tensor state. Incomplete delta staging is retained.
    if (delta/'manifest.json').exists():
     m=json.loads((delta/'manifest.json').read_text())
     if m['candidate_digest']!=value['candidate_digest'] or m['weights_sha256']!=d.sha(delta/'owners.safetensors') or m['run_spec_sha256']!=_hash(cfg.run_spec.to_dict()):raise ValueError('observation pending delta mismatch')
    else:d.save_recovery_delta(model,initial_digest,cfg.run_spec,delta,rec)
    d.write_new(path,value)
   saved=_safe_state(arm,cfg) if resume else None
   if saved is not None:
    owners=configure_trainable_owners(model,cfg,objective_spec=obj);opt=_optimizer(owners,cfg)
    _restore(saved,model,opt,cfg,TrainingSources(raw_plan=plan),teacher.manifest().checkpoint_digest,obj);del opt
    cfg=replace(cfg,resume=saved)
   if mode=='run':
    step=0 if saved is None else saved.step
    if step in config['observations'][phase]:_observe(observe,step,model,teacher)
    cfg=replace(cfg,on_update=observe)
   else:cfg=replace(cfg,on_update=None)
   # Callback/stop controls do not alter the scientific schedule identity.
   limit=1 if mode=='profile' and phase=='recovery' else stop_after_updates if stop_phase==phase else None
   cfg=replace(cfg,stop_after_updates=limit)
   fit_before=rec.totals;fit_ledger_before=rec.path.stat().st_size if rec.path.exists() else 0
   result=fit_model(model,teacher,TrainingSources(raw_plan=plan),cfg,b,rec,objective_spec=obj)
   training_cost=d.delta_cost(fit_before,rec.totals);fit_ledger_bytes=rec.path.stat().st_size-fit_ledger_before
   ref=result.checkpoint or resolve_checkpoint(arm/'resume')
   if result.outcome=='invalid':raise ValueError('invalid bounded phase result: '+result.reason)
   complete=result.state.step==cfg.max_updates
   value={'run_id':run_id(cfg.run_spec),'study_contract_sha256':study_digest(config),'step':result.state.step,
    'raw_target_cursor':result.state.target_cursor,'final_digest':result.state.candidate_digest,'checkpoint':asdict(ref),
    'retained_plan':thaw(plan.metadata),'objective':obj.identity(),'completion':complete}
   # Profile timings are measured, never used as quality-selected checkpoints.
   if mode=='profile':
    d.synchronized(config['numerical_profile']['device']);fit_seconds=time.monotonic()-begun
    before=rec.totals;obs_ledger_before=rec.path.stat().st_size;start=time.monotonic()
    measured=observer(teacher,model,fit_batches,tune_batches,plan,result.state.step,rec,config)
    payload_bytes=len(canonical_json(measured));del measured
    d.synchronized(config['numerical_profile']['device']);observation_seconds=time.monotonic()-start
    observation_cost=d.delta_cost(before,rec.totals);observation_ledger_bytes=rec.path.stat().st_size-obs_ledger_before
    from poc.checkpoint import save_checkpoint
    start=time.monotonic();ref=save_checkpoint(result.state,arm/'resume',recorder=rec);value['checkpoint']=asdict(ref)
    d.synchronized(config['numerical_profile']['device']);checkpoint_seconds=time.monotonic()-start
    value['profile']={'fitted_updates':result.state.step,'fit_seconds_including_safe_checkpoints':fit_seconds,
     'observation_seconds':observation_seconds,'observation_cost':observation_cost,'observation_payload_bytes':payload_bytes,
     'fit_ledger_bytes':fit_ledger_bytes,'observation_ledger_bytes':observation_ledger_bytes,'training_cost':training_cost,
     'checkpoint_seconds':checkpoint_seconds,'quality':'not disclosed or used in resource design'}
   finish_attempt_cost(rec,elapsed=time.monotonic()-begun);value['cost']=rec.validate().to_dict()
   if mode=='run':value['observations']={p.name:d.sha(p) for p in (arm/'observations').glob('*.json')}
   if complete:d.write_new(arm/'result.json',value)
   else:d.write_new(arm/('safe-stop-'+attempt.attempt_id+'.json'),value)
   d.finish_attempt(attempt,'exhausted',reason='completed phase' if complete else 'safe bounded phase stop')
   report['phases'][phase]=value
   if phase=='warm':
    if not complete:return dict(report,status='stopped',phase=phase)
    warm_ref=ref;warm_spec=cfg.run_spec
   elif not complete and mode!='profile':return dict(report,status='stopped',phase=phase)
  except BaseException as exc:
   try:finish_attempt_cost(rec,elapsed=time.monotonic()-begun)
   finally:d.finish_attempt(attempt,'failed',reason=type(exc).__name__+': '+str(exc))
   # The common fitter restores its safe state. The caller discards this model.
   raise
  finally:
   del model
 report['final_digest']=report['phases']['recovery']['final_digest']
 return report

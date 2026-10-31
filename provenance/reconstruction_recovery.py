#!/usr/bin/env python3
"""Explicit bounded reconstruction/extra-natural warm → fresh-KL campaign."""
import argparse
from dataclasses import asdict,replace
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import recovery_diagnosis as driver
from poc.contracts import Budget,CostRecord,RunSpec,thaw
from poc.records import canonical_json,start_attempt
from poc.reconstruction_study import (load_reconstruction_study,study_digest,unresolved,
                                     require_run_ready,expected_phase_counts,recovery_config,initialization_lineage)
from poc.reconstruction_executor import build_phase_plans,execute_phases
from poc.recovery_study import planned_training_work,planned_observation_work
from poc.reconstruction_resources import StudyFamily
from poc.reconstruction_study import RESERVED_STUDIES
from poc.budget import BudgetTracker,BudgetExceeded


def design(config):return {k:v for k,v in thaw(config).items() if k!='resource_budget'}


def validate(config):
 missing=unresolved(config)
 report={'schema_version':1,'status':'blocked_prerequisites' if missing else 'ready_for_integrity_checks',
  'missing':list(missing),'study_contract_sha256':study_digest(config),
  'phase_counts':{p:expected_phase_counts(config,p) for p in ('warm','recovery')},
  'scope':'exploratory recovery; no protected access, capability tolerance or deployment claim'}
 if any(v!='resource_budget' for v in missing):return report
 # Metadata hash checks precede any role-bound text/token I/O. The initializer
 # still verifies every actual tensor, scalar and complete owner set before fit.
 if config['schema_version']==2:
  lineage=initialization_lineage(config)
  if lineage['source_assets_sha256']!=driver.ASSET_SHA:raise ValueError('source initialization asset identity mismatch')
  report['initialization_lineage']=lineage
 derived=recovery_config(config);meta=driver.preflight(derived)
 unresolved_paths=[v for v in meta['unresolved'] if v!='resource_budget']
 if unresolved_paths:return dict(report,status='blocked_prerequisites',missing=sorted(set((*missing,*unresolved_paths))))
 asset,fit,fit_batches,tune_batches,_=driver.load_inputs(derived)
 plans=build_phase_plans(config,fit_batches,tokenizer_digest=driver.ASSET_SHA)
 binding=config['initialization_checkpoint']
 for key,path in [('run_spec',binding['run_spec_path']),('initialization_record',binding['initialization_record_path']),('checkpoint_manifest',str(Path(binding['checkpoint_directory'])/'manifest.json'))]:
  expected=binding['checkpoint_manifest_sha256' if key=='checkpoint_manifest' else key+'_sha256']
  if driver.sha(path)!=expected:raise ValueError('source metadata hash mismatch: '+key)
 spec=json.loads(Path(binding['run_spec_path']).read_text());inputs=spec.get('input_manifests',{})
 objective=spec.get('resolved_config',{}).get('recovery_objective',{});training=spec.get('resolved_config',{}).get('training',{})
 record=json.loads(Path(binding['initialization_record_path']).read_text())
 if config['schema_version']==1:
  if (inputs.get('assets')!=driver.ASSET_SHA or inputs.get('fit')!=config['populations']['fit']['sha256'] or
      inputs.get('tuning')!=config['populations']['tuning']['sha256'] or inputs.get('raw_plan')!=plans['recovery'].digest or
      objective.get('raw_plan_sha256')!=plans['recovery'].digest or objective.get('owner_scope')!='affected_group' or
      objective.get('affected_depths')!=list(config['pair']) or training.get('max_updates')!=config['recovery_updates'] or
      spec.get('seeds',{}).get('fit')!=config['seed'] or inputs.get('initialized_state')!=record.get('initialized_digest')):
   raise ValueError('source initialization/data/owner/raw-plan metadata differs')
 else:
  from poc.recovery_study import NaturalWindowPlan
  original=NaturalWindowPlan.build(fit_batches,updates=lineage['source_updates'],seed=config['seed'],
   manifest_hashes={'fit':lineage['source_fit_sha256'],'tokenizer':driver.ASSET_SHA})
  if original.digest!=lineage['source_raw_plan_sha256'] or inputs.get('initialized_state')!=record.get('initialized_digest'):
   raise ValueError('source initialization windows/raw-plan metadata differs')
  report['initialization_lineage']['target_recovery_raw_plan_sha256']=plans['recovery'].digest
 report['raw_plans']={p:thaw(v.metadata) for p,v in plans.items()}
 report['source_verification']='metadata and raw-plan bindings; tensor/scalar/owner verification remains mandatory before fitting'
 return report


def admit_output(config,source,env,*,root=None):
 if config['study'] in RESERVED_STUDIES:raise ValueError('reserved historical study identity')
 root=ROOT if root is None else Path(root);out=root/config['out']
 if out.is_symlink() or out.resolve()!=out or not out.is_relative_to(root.resolve()/'artifacts'):
  raise ValueError('canonical isolated output required')
 marker=out/'.reconstruction-owner.json'
 value={'schema_version':1,'study':config['study'],'design_sha256':driver.hashed(design(config)),
  'implementation_sha256':source['implementation_sha256'],'environment_sha256':driver.hashed(env)}
 if out.exists():
  if not marker.is_file():raise ValueError('unowned output cannot be changed or reused')
  if marker.read_bytes()!=canonical_json(value):raise ValueError('output ownership identity differs')
 else:
  out.mkdir(parents=True,exist_ok=False);driver.write_new(marker,value)
 return out


def _family(config,out):
 b=config['initialization_checkpoint']
 files,mutable=input_metadata(config)
 return StudyFamily(str(out),list(config['retained_dependency_roots']),required_sources=[
  b['run_spec_path'],b['initialization_record_path'],str(Path(b['checkpoint_directory'])/'manifest.json')],
  input_files=files,mutable_ledgers=mutable)


def input_metadata(config):
 """Read only pinned asset metadata and role-approved population metadata."""
 from poc.data.manifest import load_manifest
 files=set();mutable=set();asset=Path(config['asset_root'])/'teacher-assets.json'
 raw=json.loads(asset.read_text());files.add(str(asset))
 for row in raw['assets']:files.add(str((asset.parent/row['local_path']).resolve()))
 for name,roles in [('fit',{'fit'}),('tuning',{'dev'})]:
  entry=config['populations'][name];p=Path(entry['path'])
  manifest=load_manifest(p,expected_sha256=entry['sha256'],roles=roles);files.add(str(p))
  if manifest.exposure_path is None:raise ValueError('role-bound exposure ledger required')
  mutable.add(manifest.exposure_path)
  for row in manifest.records:
   for path in (row.text_path,row.token_path):
    if path is not None:files.add(path)
 return sorted(files),sorted(mutable)


def _verify_exposure_history(out,family):
 for pattern in ('profile-attempt-*/exposure-prefixes-*.json','runs/exposure-prefixes-*.json'):
  for path in out.glob(pattern):family.verify_exposure_prefixes(json.loads(path.read_text())['prefixes'])


def _profile_costs(spending,*,exclude=None):
 from poc.records import locked_ledger,read_rows
 buckets={p:CostRecord() for p in ('setup','warm','recovery')}
 for path in (Path(spending)/'ledgers').glob('*/costs.jsonl'):
  if exclude is not None and path==exclude:continue
  spec=json.loads((path.parent/'spec.json').read_text());variant=spec['method_variant']
  if ':profile:offline_setup' in variant:phase='setup'
  elif ':profile:warm:' in variant:phase='warm'
  elif ':profile:recovery:' in variant:phase='recovery'
  else:raise ValueError('unexpected preparation spending identity')
  with locked_ledger(path) as stream:
   for row in read_rows(stream):buckets[phase]=BudgetTracker.combine(buckets[phase],CostRecord(**row['cost']))
 return buckets


def _profile_remaining(config,spending):
 costs=_profile_costs(spending)
 for phase in ('setup','warm','recovery'):
  counts=expected_phase_counts(config,phase) if phase!='setup' else None
  raw=8192*(config['warm_updates'] if phase=='warm' else 1)
  minimum=CostRecord() if phase=='setup' else CostRecord(updates=config['warm_updates'] if phase=='warm' else 1,
   valid_target_presentations=raw*(len(config['pair']) if phase=='warm' and config['warm_loss_scope']=='ffn_mse_sum' else 1)+8192*(1+len(config['pair']))+511*config['populations']['tuning']['components'])
  total=BudgetTracker.combine(costs[phase],minimum);budget=Budget(**config['profile_resource_budget'][phase])
  for key in budget.__dataclass_fields__:
   value=getattr(total,key)
   if value is not None and value>getattr(budget,key):raise BudgetExceeded('remaining cumulative profile allowance cannot complete '+phase+':'+key)


def _scientific_prerequisites(config):
 missing=[v for v in unresolved(config) if v!='resource_budget']
 if missing:raise ValueError('unresolved study inputs: '+','.join(missing))


def _ledger_wall(stage):
 """One live costly job; read complete append-locked cumulative ledgers."""
 from poc.records import locked_ledger,read_rows
 result=0.
 for path in (Path(stage)/'ledgers').glob('*/costs.jsonl'):
  with locked_ledger(path) as stream:
   for row in read_rows(stream):result+=CostRecord(**row['cost']).wall_seconds
 return result


def campaign_spent_wall(stage):
 return _ledger_wall(stage)+_ledger_wall(Path(stage)/'phases')


def campaign_tracker(family,out,caps,*,prior_wall,begun,profile_spending=None):
 """Use the ledger's overrun path so actual work survives cap violations."""
 from poc.budget import BudgetExceeded
 class Tracker(driver.StudyTracker):
  def __init__(self,budget,attempt):
   super().__init__(budget,attempt,out);self.other_profile=CostRecord()
   if profile_spending is not None:
    variant=attempt.spec.method_variant;bucket='setup' if ':offline_setup' in variant else 'warm' if ':warm:' in variant else 'recovery'
    self.other_profile=_profile_costs(profile_spending,exclude=self.path)[bucket]
  def disk(self):return family.disk()
  def _check_total(self,total):
   super()._check_total(total)
   combined=BudgetTracker.combine(total,self.other_profile)
   for key in self.budget.__dataclass_fields__:
    value=getattr(combined,key)
    if value is not None and value>getattr(self.budget,key):raise BudgetExceeded('cumulative preparation budget exhausted: '+key)
   if prior_wall+time.monotonic()-begun>caps['wall_seconds']:raise BudgetExceeded('cumulative campaign wall budget exhausted')
   if (total.memory_bytes or 0)>caps['memory_bytes'] or total.disk_bytes>caps['disk_bytes']:raise BudgetExceeded('campaign memory/disk budget exhausted')
 return Tracker


def forecast(config,plans,fit_batches,tune_batches,profile,*,setup_seconds,family_bytes):
 phases={};disk=family_bytes
 for phase in ('warm','recovery'):
  measured=profile['phases'][phase];p=measured['profile'];updates=config[phase+'_updates']
  scope=config['warm_loss_scope'] if phase=='warm' else 'full_student'
  train=planned_training_work(plans[phase],scope,tuple(config['pair']))
  obs=planned_observation_work(plans[phase],config['observations'][phase],fit_batches,tune_batches,tuple(config['pair']),policy='optimizer_plus_tuning_v2')
  counters={k:train[k]+obs[k] for k in train}
  checkpoints=1+len(set(range(config['checkpoint_every'],updates+1,config['checkpoint_every']))|set(config['observations'][phase][1:]))
  fit_scale=updates/p['fitted_updates']
  training_scale=max([1.]+[train[k]/(p['training_cost'][k]*fit_scale) for k in train if p['training_cost'].get(k,0)>0])
  training_seconds=p['fit_seconds_including_safe_checkpoints']*fit_scale*training_scale
  # Count exact slices/calls: a partial ending window may add a presentation.
  observed=p['observation_cost'];nobs=len(config['observations'][phase])
  obs_scale=max([1.]+[obs[k]/(observed[k]*nobs) for k in obs if observed.get(k,0)>0])
  seconds=training_seconds+nobs*p['observation_seconds']*obs_scale+checkpoints*p['checkpoint_seconds']
  retained=measured.get('profile_checkpoint_peak_bytes',0)+math.ceil(fit_scale*training_scale*p['fit_ledger_bytes'])+math.ceil(nobs*obs_scale*p['observation_ledger_bytes'])+nobs*(p['observation_payload_bytes']+measured.get('observation_delta_bytes',0)+65536)+64*1024**2
  disk+=retained
  phases[phase]={'planned_updates':updates,'profile_fitted_updates':p['fitted_updates'],
   'forecast_counters':counters,'forecast_seconds':seconds,'planned_checkpoints':checkpoints,
   'forecast_retained_bytes':retained,'measured_cost':measured['cost'],
   'quality':'not disclosed or used for resource design'}
 return {'phases':phases,'forecast_campaign_seconds':setup_seconds+sum(p['forecast_seconds'] for p in phases.values()),
  'forecast_campaign_disk_bytes':disk,'setup_seconds':setup_seconds,
  'scope':'measured extrapolation with explicit observations/checkpoints; not a runtime speed claim'}


def _check_profile(config,out,source,env,family,controls_path):
 r=config['resource_budget'];path=out/'profile.json'
 if not path.is_file() or driver.sha(path)!=r['frozen_profile_sha256']:raise ValueError('frozen measured profile mismatch')
 p=json.loads(path.read_text())
 if (p['design_sha256']!=driver.hashed(design(config)) or p['implementation_sha256']!=source['implementation_sha256'] or
     p['environment_sha256']!=driver.hashed(env) or p['controls_receipt_sha256']!=driver.sha(controls_path) or
     p['retained_family']!=family.identity()):raise ValueError('profile design/source/environment/family mismatch')
 family.verify_exposure_prefixes(p['exposure_prefixes'])
 for phase in ('warm','recovery'):
  b=r['per_phase'][phase];f=p['forecast']['phases'][phase]
  for key,value in f['forecast_counters'].items():
   if key in b and b[key]<value:raise ValueError('frozen phase budget below exact forecast: '+phase+':'+key)
  if b['wall_seconds']<f['forecast_seconds'] or b['memory_bytes']<(f['measured_cost']['memory_bytes'] or 0) or b['disk_bytes']<p['forecast']['forecast_campaign_disk_bytes']:
   raise ValueError('frozen phase wall/memory/disk budget below measured forecast')
 c=r['campaign']
 if c['wall_seconds']<p['forecast']['forecast_campaign_seconds'] or c['disk_bytes']<p['forecast']['forecast_campaign_disk_bytes'] or c['memory_bytes']<max((v['measured_cost']['memory_bytes'] or 0) for v in p['forecast']['phases'].values()):
  raise ValueError('campaign budget below measured forecast')
 return p


def _bindings(stage,contract):
 """Sidecar binds legacy cost rows without migrating old ledger schemas."""
 from poc.records import read_rows
 records=[]
 for path in sorted((stage/'ledgers').glob('*/costs.jsonl')):
  run=path.parent.name;spec_path=path.parent/'spec.json'
  if driver.sha(spec_path)!=run:raise ValueError('ledger directory/spec identity mismatch')
  for ordinal,line in enumerate(path.read_bytes().splitlines(),1):
   value=json.loads(line);attempt=json.loads((path.parent/'attempts'/value['attempt_id']/'run.json').read_text())
   if attempt['run_id']!=run:raise ValueError('cost attempt/run identity mismatch')
   records.append({'study_contract_sha256':contract,'run_id':run,'cost_ordinal':ordinal,
    'attempt_id':value['attempt_id'],'raw_cost_line_sha256':hashlib.sha256(line).hexdigest()})
 driver.write_new(stage/('cost-contract-bindings-'+uuid4().hex+'.json'),{'schema_version':1,'records':records})


def execute(config,config_path,mode,*,resume=False):
 if mode not in ('profile','run'):raise ValueError('explicit profile/run required')
 if resume and mode!='run':raise ValueError('resume requires frozen run')
 if mode=='run':require_run_ready(config)
 else:_scientific_prerequisites(config)
 if config['schema_version']==2:
  lineage=initialization_lineage(config)
  if lineage['source_assets_sha256']!=driver.ASSET_SHA:raise ValueError('source initialization asset identity mismatch')
  from poc.recovery_clone import verify_population_continuation
  verify_population_continuation(recovery_config(config))
 source=driver.executed_sources(config_path);env=driver.environment();out=admit_output(config,source,env)
 if mode=='profile' and (out/'profile.json').exists():raise ValueError('published profile already exists; refuse repeated rehearsal')
 spending=out/'profile-spending' if mode=='profile' else None
 if mode=='profile':_profile_remaining(config,spending)
 family=_family(config,out);_verify_exposure_history(out,family);controls_path=out/'controls.json'
 if not controls_path.is_file():raise ValueError('qualify planned backend before profile/run')
 controls=json.loads(controls_path.read_text())
 if (controls.get('implementation_sha256')!=source['implementation_sha256'] or controls.get('environment_sha256')!=driver.hashed(env) or
     controls.get('design_sha256')!=driver.hashed(design(config)) or
     any(not controls.get(k,{}).get('passed') or controls[k].get('device')!=config['numerical_profile']['device'] for k in ('backend','reconstruction_backend'))):
  raise ValueError('stale or mismatched numerical backend controls')
 if mode=='run':_check_profile(config,out,source,env,family,controls_path)
 stage=out/('profile-attempt-'+uuid4().hex if mode=='profile' else 'runs')
 if resume and not stage.is_dir():raise ValueError('no frozen campaign to resume')
 stage.mkdir(parents=True,exist_ok=resume)
 for path,value in [(stage/'executed-source.json',source),(stage/'environment.json',env),(stage/'contract.json',thaw(config))]:driver.write_or_verify(path,value)
 if mode=='run':caps=config['resource_budget']['campaign']
 else:
  bs=config['profile_resource_budget'];caps={'wall_seconds':sum(b['wall_seconds'] for b in bs.values()),'memory_bytes':min(b['memory_bytes'] for b in bs.values()),'disk_bytes':min(b['disk_bytes'] for b in bs.values())}
 prior_wall=_ledger_wall(spending) if mode=='profile' else campaign_spent_wall(stage);begun=time.monotonic()
 Tracker=campaign_tracker(family,out,caps,prior_wall=prior_wall,begun=begun,profile_spending=spending)
 # Offline setup/preparation has one cumulative RunSpec and ledger across retry.
 setup_budget=Budget(**config['profile_resource_budget']['setup'])
 setup_spec=RunSpec('R',config['study']+':'+mode+':offline_setup',{'study_contract_sha256':study_digest(config)},source['commit'],driver.hashed(source),driver.hashed(env),
  {'assets':driver.ASSET_SHA,'fit':config['populations']['fit']['sha256'],'tuning':config['populations']['tuning']['sha256'],'study_contract':study_digest(config)},
  {'device':config['numerical_profile']['device']},{'fit':config['seed']},(),setup_budget)
 attempt=start_attempt(setup_spec,(spending if mode=='profile' else stage)/'ledgers');rec=Tracker(setup_budget,attempt);status='failed';reason='interrupted setup/campaign'
 try:
  rec.check(CostRecord())
  import torch
  from poc.teacher import load_teacher
  derived=recovery_config(config)
  asset,fit,fit_batches,tune_batches,_=driver.load_inputs(derived)
  teacher=load_teacher(asset).to(device=config['numerical_profile']['device'],dtype=torch.float32)
  plans=build_phase_plans(config,fit_batches,tokenizer_digest=teacher.tokenizer_digest)
  trial=driver.initialize(derived,teacher,fit,fit_batches,config['seed'],rec)
  numerical=driver.real_shape_controls(derived,teacher,trial.model,fit_batches[0],rec)
  driver.write_or_verify(stage/'initialization.json',{'initialized_digest':trial.initialized_digest,'initializer_record':trial.initializer_record,'numerical_controls':numerical})
  driver.synchronized(config['numerical_profile']['device']);rec.record(CostRecord(wall_seconds=max(0,time.monotonic()-begun-rec.attempt_cost.wall_seconds)))
  budgets={p:Budget(**(config['profile_resource_budget'][p] if mode=='profile' else config['resource_budget']['per_phase'][p])) for p in ('warm','recovery')}
  result=execute_phases(config=config,teacher=teacher,initial=trial.model,plans=plans,fit_batches=fit_batches,tune_batches=tune_batches,
   source=source,environment=env,controls_sha256=driver.sha(controls_path),stage=stage/'phases',budgets=budgets,
   observer=lambda t,m,f,v,p,s,r,c:driver.observation(t,m,f,v,p,s,r,derived),recorder_factory=lambda b,a:Tracker(b,a),resume=resume,mode=mode,ledger_root=spending/'ledgers' if mode=='profile' else None)
  # The phase library has separate append-locked ledgers; include their prior
  # attempts when measuring orchestration gaps and campaign retry spending.
  if mode=='profile':
   from poc.recovery_objectives import affected_owners
   owner_bytes=sum(p.numel()*p.element_size() for p in affected_owners(trial.model,tuple(config['pair'])).values())
   for phase in ('warm','recovery'):
    folder=stage/'phases'/phase/'resume';census={str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':driver.sha(p)} for p in folder.rglob('*') if p.is_file()}
    result['phases'][phase]['profile_checkpoint_peak_bytes']=sum(v['bytes'] for v in census.values())
    result['phases'][phase]['observation_delta_bytes']=owner_bytes
    driver.write_new(stage/(phase+'-profile-state-census.json'),{'files':census,'policy':'temporary rehearsal tensors removed before resource freeze'})
    shutil.rmtree(folder)
   elapsed=time.monotonic()-begun
   phase_seconds=sum(p['cost']['wall_seconds'] for p in result['phases'].values())
   setup_seconds=max(0,elapsed-phase_seconds)
   report={'schema_version':1,'design_sha256':driver.hashed(design(config)),'implementation_sha256':source['implementation_sha256'],
    'environment_sha256':driver.hashed(env),'controls_receipt_sha256':driver.sha(controls_path),'retained_family':family.identity(),'exposure_prefixes':family.exposure_snapshot(),
    'forecast':forecast(config,plans,fit_batches,tune_batches,result,setup_seconds=setup_seconds,family_bytes=family.disk()),
    'scope':'full warm rehearsal plus actual post-warm KL1; no candidate quality disclosed; preparation separately charged'}
   driver.write_new(out/'profile.json',report)
  else:
   if result['status']!='complete':raise ValueError('frozen campaign did not complete both phases')
   report=result;driver.write_or_verify(out/'result.json',report)
  status='exhausted';reason='completed '+mode+' campaign'
  return report
 except BaseException as exc:
  reason=type(exc).__name__+': '+str(exc);raise
 finally:
  # Top up overhead once; physical work, including failed/retried phases, stays
  # in its original ledger and is never inferred from a checkpoint snapshot.
  spent=_ledger_wall(spending) if mode=='profile' else campaign_spent_wall(stage)
  gap=max(0,time.monotonic()-begun-(spent-prior_wall))
  try:
   if gap:rec.record(CostRecord(wall_seconds=gap))
  finally:
   driver.finish_attempt(attempt,status,reason=reason)
   driver.write_new(stage/('exposure-prefixes-'+attempt.attempt_id+'.json'),{'study_contract_sha256':study_digest(config),'prefixes':family.exposure_snapshot()})
   _bindings(spending if mode=='profile' else stage,study_digest(config))
   if (stage/'phases/ledgers').exists():_bindings(stage/'phases',study_digest(config))


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--mode',required=True,choices=('validate','controls','profile','run'));p.add_argument('--resume',action='store_true');args=p.parse_args(argv)
 c=load_reconstruction_study(args.config)
 if args.resume and args.mode!='run':p.error('--resume requires --mode run')
 if args.mode=='validate':report=validate(c)
 elif args.mode=='controls':
  source=driver.executed_sources(args.config);env=driver.environment();out=admit_output(c,source,env)
  if (out/'controls.json').exists():raise ValueError('controls receipt exists; refuse overwrite')
  from poc.recovery_controls import qualify_backend,qualify_reconstruction_backend
  report={'schema_version':1,'design_sha256':driver.hashed(design(c)),'implementation_sha256':source['implementation_sha256'],'environment_sha256':driver.hashed(env),
   'backend':qualify_backend(c['numerical_profile']['device']),
   'reconstruction_backend':qualify_reconstruction_backend(c['numerical_profile']['device'])}
  if not report['backend']['passed'] or not report['reconstruction_backend']['passed']:raise ValueError('numerical backend controls failed')
  driver.write_new(out/'controls.json',report)
 else:report=execute(c,args.config,args.mode,resume=args.resume)
 print(json.dumps(report,sort_keys=True))
 return 0


if __name__=='__main__':raise SystemExit(main())

"""Admission and resource forecasting precede scientific-content reads."""
import importlib.util,json
from pathlib import Path
from dataclasses import replace
import pytest
from poc.contracts import thaw
from test_reconstruction_study import draft,load
from test_reconstruction_executor import owned,arguments
pytestmark=pytest.mark.cpu


def api():
 assert importlib.util.find_spec('provenance.reconstruction_recovery'), 'explicit two-phase CLI missing'
 from provenance import reconstruction_recovery
 return reconstruction_recovery


def test_unresolved_template_validate_is_explicit_and_does_no_content_io(tmp_path,monkeypatch):
 a=api();c=load(tmp_path,draft());monkeypatch.setattr(a.driver,'load_inputs',lambda *_:pytest.fail('scientific content opened'))
 report=a.validate(c)
 assert report['status']=='blocked_prerequisites'
 assert 'initialization_checkpoint' in report['missing'] and 'resource_budget' in report['missing']
 assert report['phase_counts']['warm']['raw_targets']==131072


@pytest.mark.parametrize('mode',['profile','run'])
def test_unresolved_or_unfrozen_execution_rejected_before_loading(tmp_path,monkeypatch,mode):
 a=api();c=load(tmp_path,draft());monkeypatch.setattr(a.driver,'load_inputs',lambda *_:pytest.fail('content opened'))
 with pytest.raises(ValueError,match='unresolved|frozen'):a.execute(c,tmp_path/'study.yaml',mode)


def test_controls_cannot_replace_unowned_historical_output(tmp_path):
 a=api();c=load(tmp_path,draft());root=tmp_path/'worktree';out=root/c['out'];out.mkdir(parents=True);p=out/'controls.json';p.write_text('historical bytes')
 with pytest.raises(ValueError,match='unowned|ownership'):a.admit_output(c,{'implementation_sha256':'a'*64},{'owned':True},root=root)
 assert p.read_text()=='historical bytes'


def test_owned_output_accepts_budget_freeze_but_rejects_semantic_changes(tmp_path):
 a=api();c=load(tmp_path,draft());root=tmp_path/'worktree';source={'implementation_sha256':'a'*64};env={'owned':True}
 first=a.admit_output(c,source,env,root=root)
 assert first.is_dir() and a.admit_output(c,source,env,root=root)==first
 changed=thaw(c);changed['warm_loss_scope']='full_student'
 with pytest.raises(ValueError,match='ownership|identity'):a.admit_output(changed,source,env,root=root)
 assert not (first/'controls.json').exists()


def test_forecast_uses_completed_warm_actual_kl_and_exact_observation_units(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path)
 from poc.reconstruction_executor import execute_phases
 result=execute_phases(**args,mode='profile')
 forecast=a.forecast(args['config'],args['plans'],args['fit_batches'],args['tune_batches'],result,setup_seconds=2.,family_bytes=1234)
 warm=forecast['phases']['warm'];kl=forecast['phases']['recovery']
 assert warm['profile_fitted_updates']==2 and kl['profile_fitted_updates']==1
 assert warm['forecast_counters']['updates']==kl['forecast_counters']['updates']==2
 assert warm['forecast_counters']['raw_corpus_target_presentations']==16384
 assert warm['forecast_counters']['valid_target_presentations']==32768+2*(24576+511)
 assert kl['forecast_counters']['valid_target_presentations']==16384+3*(24576+511)
 assert forecast['forecast_campaign_seconds']>=sum(p['forecast_seconds'] for p in forecast['phases'].values())+2.
 assert forecast['forecast_campaign_disk_bytes']>=1234
 assert not any('kl_sum' in str(p) or 'nll_sum' in str(p) for p in forecast['phases'].values())


def test_campaign_retry_wall_includes_nested_phase_ledgers(tmp_path):
 a=api()
 from poc.records import start_attempt
 from poc.contracts import RunSpec,CostRecord
 from poc.budget import BudgetTracker
 from test_training import budget
 def spend(root,variant,value):
  spec=RunSpec('R',variant,{'owned':True},'a'*40,'b'*64,'c'*64,{'owned':'d'*64},{'cpu':True},{'fit':17},(),budget())
  rec=BudgetTracker(spec.budgets,start_attempt(spec,root/'ledgers'));rec.record(CostRecord(wall_seconds=value))
 spend(tmp_path,'setup',2.);spend(tmp_path/'phases','warm',3.);spend(tmp_path/'phases','kl',5.)
 assert a.campaign_spent_wall(tmp_path)==10.


def test_campaign_overrun_is_persisted_before_it_raises(tmp_path):
 a=api()
 from poc.records import start_attempt
 from poc.contracts import RunSpec,CostRecord
 from poc.budget import BudgetExceeded
 from poc.recovery_family import RetainedFamily
 from test_training import budget
 dep=tmp_path/'dep';dep.mkdir();(dep/'source').write_text('owned')
 family=RetainedFamily(str(tmp_path/'output'),[str(dep)])
 spec=RunSpec('R','owned',{'owned':True},'a'*40,'b'*64,'c'*64,{'owned':'d'*64},{'cpu':True},{'fit':17},(),budget(memory_bytes=6*1024**3))
 attempt=start_attempt(spec,tmp_path/'output/ledgers')
 import time
 tracker=a.campaign_tracker(family,tmp_path/'output',{'wall_seconds':1.,'memory_bytes':6*1024**3,'disk_bytes':10**8},prior_wall=2.,begun=time.monotonic())(spec.budgets,attempt)
 with pytest.raises(BudgetExceeded):tracker.record(CostRecord(wall_seconds=.01))
 assert tracker.totals.wall_seconds==.01
 assert 'overrun_reason' in json.loads(tracker.path.read_text())


def test_cli_profiles_real_owned_phases_then_freezes_and_runs(tmp_path,monkeypatch,owned):
 a=api();teacher,initial,fit,tune,c=owned;c=thaw(c)
 root=tmp_path/'workspace';root.mkdir();dep=tmp_path/'source';dep.mkdir()
 for name in ('spec.json','initialization.json','manifest.json'):(dep/name).write_text('{}')
 c['asset_root']=str(dep);c['retained_dependency_roots']=[str(dep)]
 c['initialization_checkpoint']={'checkpoint_directory':str(dep),'checkpoint_manifest_sha256':'f'*64,'run_spec_path':str(dep/'spec.json'),'run_spec_sha256':'f'*64,'initialization_record_path':str(dep/'initialization.json'),'initialization_record_sha256':'f'*64}
 monkeypatch.setattr(a,'ROOT',root)
 monkeypatch.setattr(a,'input_metadata',lambda c:([],[]))
 source={'commit':'a'*40,'implementation_sha256':'d'*64};env={'owned':'cpu'}
 monkeypatch.setattr(a.driver,'executed_sources',lambda path:source);monkeypatch.setattr(a.driver,'environment',lambda:env)
 monkeypatch.setattr(a.driver,'load_inputs',lambda config:(None,None,fit,tune,{}))
 teacher=teacher.float();initial=initial.float();initial.base_digest=teacher.manifest().checkpoint_digest
 import poc.teacher
 monkeypatch.setattr(poc.teacher,'load_teacher',lambda asset:teacher)
 from types import SimpleNamespace
 monkeypatch.setattr(a.driver,'initialize',lambda *args:SimpleNamespace(model=initial,initialized_digest=initial.manifest().checkpoint_digest,initializer_record={'owned':True}))
 out=a.admit_output(c,source,env)
 controls={'design_sha256':a.driver.hashed(a.design(c)),'implementation_sha256':source['implementation_sha256'],'environment_sha256':a.driver.hashed(env),'backend':{'passed':True,'device':'cpu'},'reconstruction_backend':{'passed':True,'device':'cpu'}}
 a.driver.write_new(out/'controls.json',controls)
 profile=a.execute(c,tmp_path/'study.yaml','profile')
 assert profile['forecast']['phases']['warm']['profile_fitted_updates']==2
 assert profile['forecast']['phases']['recovery']['profile_fitted_updates']==1
 assert not list(out.glob('profile-attempt-*/phases/*/resume'))
 from test_reconstruction_study import limits
 c['resource_budget']={'frozen_profile_sha256':a.driver.sha(out/'profile.json'),'per_phase':{'warm':limits(2),'recovery':limits(2)},'campaign':{'wall_seconds':1800,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}}
 result=a.execute(c,tmp_path/'study.yaml','run')
 assert result['status']=='complete'
 assert a.campaign_spent_wall(out/'runs')>0
 assert list((out/'runs').glob('cost-contract-bindings-*.json'))
 assert list((out/'runs/phases').glob('cost-contract-bindings-*.json'))
 repeated=a.execute(c,tmp_path/'study.yaml','run',resume=True)
 assert repeated['final_digest']==result['final_digest']


def test_resolved_validate_checks_actual_role_plans_and_bound_source_metadata(tmp_path,monkeypatch,owned):
 a=api();args=arguments(owned,tmp_path);c=thaw(args['config'])
 dep=tmp_path/'source';dep.mkdir();source_spec=dep/'spec.json';record=dep/'initialization.json';manifest=dep/'manifest.json'
 from poc.reconstruction_executor import phase_config
 cfg,obj=phase_config(c,'recovery',args['plans']['recovery'],args['initial'],source=args['source'],environment=args['environment'],budget=args['budgets']['recovery'],controls_sha256='e'*64)
 source_spec.write_text(json.dumps(cfg.run_spec.to_dict()));record.write_text(json.dumps({'initialized_digest':args['initial'].manifest().checkpoint_digest}));manifest.write_text('{}')
 c['asset_root']=str(dep);c['retained_dependency_roots']=[str(dep)]
 c['initialization_checkpoint']={'checkpoint_directory':str(dep),'checkpoint_manifest_sha256':a.driver.sha(manifest),'run_spec_path':str(source_spec),'run_spec_sha256':a.driver.sha(source_spec),'initialization_record_path':str(record),'initialization_record_sha256':a.driver.sha(record)}
 monkeypatch.setattr(a.driver,'preflight',lambda c:{'unresolved':['resource_budget']})
 monkeypatch.setattr(a.driver,'load_inputs',lambda c:(None,None,args['fit_batches'],args['tune_batches'],{}))
 monkeypatch.setattr(a.driver,'ASSET_SHA',args['teacher'].tokenizer_digest)
 report=a.validate(c)
 assert report['raw_plans']['recovery']['plan_sha256']==args['plans']['recovery'].digest
 assert report['source_verification']=='metadata and raw-plan bindings; tensor/scalar/owner verification remains mandatory before fitting'
 changed=json.loads(source_spec.read_text());changed['input_manifests']['raw_plan']='f'*64;source_spec.write_text(json.dumps(changed));c['initialization_checkpoint']['run_spec_sha256']=a.driver.sha(source_spec)
 with pytest.raises(ValueError,match='source|plan'):a.validate(c)


@pytest.mark.parametrize('name',['continuation-v1','continuation-v2','training','diagnostics','validation'])
def test_direct_admission_also_denies_absent_historical_output(tmp_path,name):
 a=api();c=draft();c.update(study=name,out='artifacts/'+name)
 with pytest.raises(ValueError,match='historical|reserved'):
  a.admit_output(c,{'implementation_sha256':'a'*64},{'owned':True},root=tmp_path)
 assert not (tmp_path/c['out']).exists()


def test_profile_replay_denied_before_content_io(tmp_path,monkeypatch,owned):
 a,c,out=ready_cli(tmp_path,monkeypatch,owned)
 (out/'profile.json').write_text('{}')
 monkeypatch.setattr(a.driver,'load_inputs',lambda *_:pytest.fail('published profile was needlessly rerun'))
 with pytest.raises(ValueError,match='profile.*exist|published'):a.execute(c,tmp_path/'config.yaml','profile')


def ready_cli(tmp_path,monkeypatch,owned):
 a=api();teacher,initial,fit,tune,c=owned;c=thaw(c);root=tmp_path/'workspace';root.mkdir();dep=tmp_path/'source';dep.mkdir()
 for name in ('spec.json','initialization.json','manifest.json'):(dep/name).write_text('{}')
 ledger=dep/'exposure.jsonl';ledger.write_text('first\n')
 c['asset_root']=str(dep);c['retained_dependency_roots']=[str(dep)]
 c['initialization_checkpoint']={'checkpoint_directory':str(dep),'checkpoint_manifest_sha256':'f'*64,'run_spec_path':str(dep/'spec.json'),'run_spec_sha256':'f'*64,'initialization_record_path':str(dep/'initialization.json'),'initialization_record_sha256':'f'*64}
 monkeypatch.setattr(a,'ROOT',root);source={'commit':'a'*40,'implementation_sha256':'d'*64};env={'owned':'cpu'}
 monkeypatch.setattr(a.driver,'executed_sources',lambda path:source);monkeypatch.setattr(a.driver,'environment',lambda:env)
 def inputs(*_):
  with ledger.open('a') as f:f.write('allowed read\n')
  return None,None,fit,tune,{}
 monkeypatch.setattr(a.driver,'load_inputs',inputs)
 # Only the metadata discovery is substituted; real phase fitting/handoff and
 # the mutable-footprint guard still execute on owned tensors and files.
 monkeypatch.setattr(a,'input_metadata',lambda c:([],[str(ledger)]),raising=False)
 teacher=teacher.float();initial=initial.float();initial.base_digest=teacher.manifest().checkpoint_digest
 import poc.teacher
 monkeypatch.setattr(poc.teacher,'load_teacher',lambda asset:teacher)
 from types import SimpleNamespace
 monkeypatch.setattr(a.driver,'initialize',lambda *args:SimpleNamespace(model=initial,initialized_digest=initial.manifest().checkpoint_digest,initializer_record={'owned':True}))
 monkeypatch.setattr(a.driver,'real_shape_controls',lambda *args:{'passed':True})
 monkeypatch.setattr(a.driver,'observation',lambda *args:{'owned':True})
 out=a.admit_output(c,source,env);a.driver.write_new(out/'controls.json',{'design_sha256':a.driver.hashed(a.design(c)),'implementation_sha256':source['implementation_sha256'],'environment_sha256':a.driver.hashed(env),'backend':{'passed':True,'device':'cpu'},'reconstruction_backend':{'passed':True,'device':'cpu'}})
 return a,c,out


def test_failed_profile_spending_prevents_a_second_full_rehearsal(tmp_path,monkeypatch,owned):
 a,c,out=ready_cli(tmp_path,monkeypatch,owned)
 monkeypatch.setattr(a.driver,'observation',lambda *args:(_ for _ in ()).throw(RuntimeError('owned profile observation failed')))
 with pytest.raises(RuntimeError,match='owned profile'):a.execute(c,tmp_path/'config.yaml','profile')
 assert not (out/'profile.json').exists()
 monkeypatch.setattr(a.driver,'load_inputs',lambda *_:pytest.fail('failed warm budget was reset'))
 from poc.budget import BudgetExceeded
 with pytest.raises(BudgetExceeded):a.execute(c,tmp_path/'different-path-same-config.yaml','profile')
 assert a.campaign_spent_wall(out/'profile-spending')>0


def test_mutable_exposure_profile_freeze_run_resume_counts_growth(tmp_path,monkeypatch,owned):
 a,c,out=ready_cli(tmp_path,monkeypatch,owned)
 profile=a.execute(c,tmp_path/'config.yaml','profile')
 ledger=Path(c['retained_dependency_roots'][0])/'exposure.jsonl';size=ledger.stat().st_size
 from test_reconstruction_study import limits
 c['resource_budget']={'frozen_profile_sha256':a.driver.sha(out/'profile.json'),'per_phase':{'warm':limits(2),'recovery':limits(2)},'campaign':{'wall_seconds':1800,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}}
 result=a.execute(c,tmp_path/'config.yaml','run')
 assert result['status']=='complete' and ledger.stat().st_size>size
 resumed=a.execute(c,tmp_path/'config.yaml','run',resume=True)
 assert resumed['final_digest']==result['final_digest']
 ledger.write_bytes(b'truncated')
 with pytest.raises(ValueError,match='prefix|truncat'):a.execute(c,tmp_path/'config.yaml','run',resume=True)

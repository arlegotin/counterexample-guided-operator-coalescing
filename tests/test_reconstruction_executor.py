"""Owned full lifecycle: completed warm, weights-only handoff, safe KL resume."""
from copy import deepcopy
from dataclasses import asdict,replace
import importlib.util,json
import pytest,torch
from poc.bank import make_bank,canonicalize_partition
from poc.contracts import thaw
from poc.records import run_id
from poc.checkpoint import load_checkpoint
from test_training import models,batch,budget
from test_reconstruction_study import draft,load
pytestmark=pytest.mark.cpu


def api():
 assert importlib.util.find_spec('poc.reconstruction_executor'), 'common-loop phase executor missing'
 from poc import reconstruction_executor
 return reconstruction_executor


@pytest.fixture
def owned(tmp_path):
 teacher,_=models();teacher.config.max_position_embeddings=1024;teacher.tokenizer_digest='a'*64
 initial=make_bank(teacher,canonicalize_partition([[0,1],[2]],3),8,17)
 with torch.no_grad():initial.bank[0].up_proj.weight.add_(.001)
 initial.base_digest=teacher.manifest().checkpoint_digest
 fit=(batch(512),replace(batch(512),document_ids=('second',),cluster_ids=('second-component',)))
 tune=(replace(batch(512),document_ids=('tuning',),cluster_ids=('tuning-component',)),)
 d=draft();d.update(pair=[0,1],calibration_components=1,warm_updates=2,recovery_updates=2,checkpoint_every=1)
 d['numerical_profile']['device']='cpu';d['observations']={'warm':[0,2],'recovery':[0,1,2]}
 d['populations']={'fit':{'path':'/owned/fit.json','sha256':'b'*64,'components':2},'tuning':{'path':'/owned/tuning.json','sha256':'c'*64,'components':1}}
 d['profile_resource_budget']['warm']['updates']=2
 c=load(tmp_path,d)
 return teacher,initial,fit,tune,c


def arguments(owned,tmp_path):
 teacher,initial,fit,tune,c=owned;a=api();plans=a.build_phase_plans(c,fit,tokenizer_digest=teacher.tokenizer_digest)
 return dict(config=c,teacher=teacher,initial=initial,plans=plans,fit_batches=fit,tune_batches=tune,
  source={'commit':'a'*40,'implementation_sha256':'d'*64},environment={'owned':'cpu'},controls_sha256='e'*64,
  stage=tmp_path/'run',budgets={'warm':budget(),'recovery':budget()},observer=lambda *args:{'owned':True})


def test_phase_plans_keep_full_contexts_and_exact_subset_lineage(owned):
 teacher,_,fit,_,c=owned;a=api();plans=a.build_phase_plans(c,fit,tokenizer_digest=teacher.tokenizer_digest)
 assert len(plans['warm'].templates)==1 and len(plans['recovery'].templates)==2
 assert plans['warm'].digest!=plans['recovery'].digest
 assert plans['warm'].metadata['raw_target_presentations']==16384
 assert plans['warm'].metadata['inventory_unique_target_positions']==511
 assert dict(plans['warm'].manifests)['fit']=='b'*64
 assert dict(plans['warm'].manifests)['calibration_subset']
 for u in plans['warm'].iter_updates():
  assert u.valid_targets==8192
  assert all(torch.equal(b.input_ids,fit[0].input_ids) for b in u.batches)
 bad=(fit[0],replace(fit[1],target_mask=fit[1].target_mask.clone()))
 bad[1].target_mask[0,0]=False
 with pytest.raises(ValueError,match='511|eligible|512'):a.build_phase_plans(c,bad,tokenizer_digest=teacher.tokenizer_digest)


@pytest.mark.parametrize('scope',['ffn_mse_sum','full_student'])
def test_real_two_phase_execution_has_fresh_kl_state_and_priced_warm_units(owned,tmp_path,scope):
 a=api();args=arguments(owned,tmp_path);c=thaw(args['config']);c['warm_loss_scope']=scope;args['config']=c
 result=a.execute_phases(**args)
 assert result['status']=='complete'
 warm=json.loads((args['stage']/'warm/result.json').read_text());kl=json.loads((args['stage']/'recovery/result.json').read_text())
 assert warm['step']==kl['step']==2 and warm['raw_target_cursor']==kl['raw_target_cursor']==16384
 assert warm['cost']['raw_corpus_target_presentations']==16384
 assert warm['cost']['valid_target_presentations']==(32768 if scope=='ffn_mse_sum' else 16384)
 assert kl['cost']['valid_target_presentations']==16384
 hand=json.loads((args['stage']/'handoff.json').read_text())
 assert hand['warm_loss_scope']==scope and hand['retained_warm_scored_targets']==warm['cost']['valid_target_presentations']
 spec=json.loads((args['stage']/'recovery/spec.json').read_text())
 assert spec['parent_run_ids']==[warm['run_id']]
 from poc.checkpoint import _spec as spec_from_dict
 from poc.checkpoint import CheckpointRef
 state=load_checkpoint(CheckpointRef(**kl['checkpoint']),spec_from_dict(spec))
 assert set(int(v['step']) for v in state.optimizer['state'].values() if 'step' in v)=={2}
 assert state.target_cursor==16384 and state.scheduler['completed_updates']==2
 assert sorted(p.name for p in (args['stage']/'recovery/observations').glob('*.json'))==['0000.json','0001.json','0002.json']


def test_interrupted_warm_cannot_handoff_then_resume_matches_whole(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path)
 first=a.execute_phases(**args,stop_phase='warm',stop_after_updates=1)
 assert first['status']=='stopped' and first['phase']=='warm'
 assert not (args['stage']/'handoff.json').exists() and not (args['stage']/'recovery').exists()
 resumed=a.execute_phases(**args,resume=True)
 whole=a.execute_phases(**dict(args,stage=tmp_path/'whole'))
 assert resumed['final_digest']==whole['final_digest']
 assert json.loads((args['stage']/'warm/result.json').read_text())['cost']['updates']==2


def test_recovery_interruption_retains_complete_warm_and_does_not_replay_it(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path)
 first=a.execute_phases(**args,stop_phase='recovery',stop_after_updates=1)
 assert first['status']=='stopped' and first['phase']=='recovery'
 warm=(args['stage']/'warm/result.json').read_bytes()
 resumed=a.execute_phases(**args,resume=True)
 assert warm==(args['stage']/'warm/result.json').read_bytes()
 assert resumed['final_digest']==a.execute_phases(**dict(args,stage=tmp_path/'whole'))['final_digest']


def test_resume_rejects_modified_observation_and_forged_completed_checkpoint(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path);a.execute_phases(**args,stop_phase='warm',stop_after_updates=1)
 p=args['stage']/'warm/observations/0000.json';raw=json.loads(p.read_text());raw['candidate_digest']='f'*64;p.write_text(json.dumps(raw))
 with pytest.raises(ValueError,match='observation|integrity'):a.execute_phases(**args,resume=True)


def test_profile_rehearses_entire_warm_before_one_actual_kl_update(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path);result=a.execute_phases(**args,mode='profile')
 assert result['phases']['warm']['step']==2
 assert result['phases']['warm']['completion'] is True
 assert result['phases']['recovery']['step']==1
 assert result['phases']['recovery']['completion'] is False
 from poc.checkpoint import CheckpointRef,_spec
 for name,record in result['phases'].items():
  spec=_spec(json.loads((args['stage']/name/'spec.json').read_text()))
  state=load_checkpoint(CheckpointRef(**record['checkpoint']),spec)
  assert state.step==record['step']
  assert record['profile']['observation_seconds']>=0
  assert record['profile']['fit_ledger_bytes']>0 and record['profile']['observation_ledger_bytes']>=0
  assert record['profile']['training_cost']['raw_corpus_target_presentations']==8192*record['step']
 assert not (args['stage']/'recovery/result.json').exists()


def test_observer_failure_retains_safe_state_and_lost_work_cost(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path);calls=[]
 def fail(*values):
  calls.append(values[5])
  if values[5]==1:raise RuntimeError('owned interrupted observation')
  return {'owned':True}
 # Make update1 an observation boundary in warm as well.
 c=thaw(args['config']);c['observations']['warm']=[0,1,2];args['config']=c
 with pytest.raises(RuntimeError,match='interrupted'):a.execute_phases(**dict(args,observer=fail))
 assert not (args['stage']/'warm/result.json').exists()
 result=a.execute_phases(**args,resume=True)
 assert result['status']=='complete'
 assert result['phases']['warm']['cost']['updates']==2
 assert result['final_digest']==a.execute_phases(**dict(args,stage=tmp_path/'whole'))['final_digest']


def test_completed_resume_verifies_exact_metric_bytes(owned,tmp_path):
 a=api();args=arguments(owned,tmp_path);a.execute_phases(**args)
 p=args['stage']/'warm/observations/0000.json';v=json.loads(p.read_text());v['owned']='changed metric payload';p.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='observation|metric'):a.execute_phases(**args,resume=True)


@pytest.mark.mps
@pytest.mark.parametrize('scope',['ffn_mse_sum','full_student'])
def test_actual_mps_named_warm_and_fresh_kl_lifecycle(owned,tmp_path,scope):
 from poc.recovery_controls import qualify_backend,qualify_reconstruction_backend
 assert torch.backends.mps.is_available()
 assert qualify_backend('mps')['passed'] and qualify_reconstruction_backend('mps')['passed']
 a=api();teacher,initial,fit,tune,c=owned
 teacher=teacher.float().to('mps');initial=initial.float().to('mps');initial.base_digest=teacher.manifest().checkpoint_digest
 c=thaw(c);c.update(warm_updates=1,recovery_updates=1,warm_loss_scope=scope)
 c['observations']={'warm':[0,1],'recovery':[0,1]};c['numerical_profile']['device']='mps'
 plans=a.build_phase_plans(c,fit,tokenizer_digest=teacher.tokenizer_digest)
 result=a.execute_phases(config=c,teacher=teacher,initial=initial,plans=plans,fit_batches=fit,tune_batches=tune,
  source={'commit':'a'*40,'implementation_sha256':'d'*64},environment={'owned':'mps'},controls_sha256='e'*64,
  stage=tmp_path/'metal-run',budgets={'warm':budget(),'recovery':budget()},observer=lambda *args:{'owned':True})
 assert result['status']=='complete'
 assert result['phases']['warm']['raw_target_cursor']==result['phases']['recovery']['raw_target_cursor']==8192
 assert result['phases']['warm']['cost']['valid_target_presentations']==(16384 if scope=='ffn_mse_sum' else 8192)
 assert result['phases']['recovery']['cost']['valid_target_presentations']==8192
 assert result['phases']['recovery']['final_digest']!=result['phases']['warm']['final_digest']

"""Closed two-phase contracts: finite warm work and independent KL recovery."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import pytest,yaml
from poc.contracts import Budget
pytestmark=pytest.mark.cpu


def api():
 assert importlib.util.find_spec('poc.reconstruction_study'), 'versioned reconstruction study missing'
 from poc import reconstruction_study
 return reconstruction_study


def limits(updates,targets=4000000):
 return asdict(Budget(updates,targets,2000000,100000,100000,0,0,1800,32*1024**3,6*1024**3))


def draft():
 return {'schema_version':1,'study':'nonlinear-reconstruction-v1','asset_root':None,'out':'artifacts/nonlinear-reconstruction-v1','pair':[0,15],'rank':8,'seed':17,'calibration_components':16,'warm_loss_scope':'ffn_mse_sum','warm_updates':16,'recovery_updates':256,'owner_scope':'affected_group','alpha':0,'observations':{'warm':[0,16],'recovery':[0,1,4,16,32,64,128,256]},'checkpoint_every':16,'optimizer':{'bank_lr':1e-4,'residual_lr':3e-4,'weight_decay':0,'lr_factor':1},'populations':{'fit':{'path':None,'sha256':None,'components':512},'tuning':{'path':None,'sha256':None,'components':128}},'initialization_checkpoint':None,'numerical_profile':{'device':'mps','dtype':'float32','chunk_size':32,'kl_floor_per_target':2e-6,'decision_resolution':2e-5},'profile_resource_budget':{'setup':limits(0),'warm':limits(16),'recovery':limits(1)},'resource_budget':None,'retained_dependency_roots':[]}


def load(tmp_path,value):
 p=tmp_path/'study.yaml';p.write_text(yaml.safe_dump(value,sort_keys=False));return api().load_reconstruction_study(p)


def test_contract_is_immutable_and_semantic_choices_have_distinct_hashes(tmp_path):
 c=load(tmp_path,draft());a=api();assert a.study_digest(c)
 with pytest.raises(TypeError):c['seed']=23
 with pytest.raises(TypeError):c['observations']['warm'][0]=1
 other=draft();other['warm_loss_scope']='full_student'
 assert a.study_digest(load(tmp_path,other))!=a.study_digest(c)


@pytest.mark.parametrize('change',['unknown','nested','owner','alpha','scope','chunk','seed','calibration','observations','optimizer'])
def test_invalid_scientific_contracts_are_rejected(tmp_path,change):
 d=draft()
 if change=='unknown':d['hidden_extra_updates']=16
 elif change=='nested':d['optimizer']['free_hpo']=3
 elif change=='owner':d['owner_scope']='all'
 elif change=='alpha':d['alpha']=.1
 elif change=='scope':d['warm_loss_scope']='local_sum'
 elif change=='chunk':d['numerical_profile']['chunk_size']=16
 elif change=='seed':d['seed']=True
 elif change=='calibration':d['calibration_components']=513
 elif change=='observations':d['observations']['warm']=[0,8]
 elif change=='optimizer':d['optimizer']['bank_lr']*=3
 with pytest.raises(ValueError):load(tmp_path,d)


def test_profile_requires_complete_warm_rehearsal_and_one_real_kl_update(tmp_path):
 d=draft();d['profile_resource_budget']['warm']['updates']=1
 with pytest.raises(ValueError,match='complete|warm'):load(tmp_path,d)
 d=draft();d['profile_resource_budget']['recovery']['updates']=2
 with pytest.raises(ValueError,match='one|recovery'):load(tmp_path,d)


def test_phase_counts_keep_raw_and_operator_units_distinct(tmp_path):
 c=load(tmp_path,draft());a=api();warm=a.expected_phase_counts(c,'warm');kl=a.expected_phase_counts(c,'recovery')
 assert warm['raw_targets']==131072 and warm['scored_training_targets']==262144
 assert warm['distinct_source_targets']==8176 and kl['distinct_source_targets']==261632
 assert warm['observation_targets']==179968
 assert kl['raw_targets']==2097152 and kl['scored_training_targets']==2097152
 assert kl['observation_targets']==719872
 d=draft();d['warm_loss_scope']='full_student';assert a.expected_phase_counts(load(tmp_path,d),'warm')['scored_training_targets']==131072


def test_unfrozen_run_rejected_before_content_io(tmp_path):
 c=load(tmp_path,draft());assert set(api().unresolved(c))=={'asset_root','initialization_checkpoint','fit','tuning','retained_dependency_roots','resource_budget'}
 with pytest.raises(ValueError,match='unresolved|frozen'):api().require_run_ready(c)


def test_recovery_clone_keeps_original_horizon_not_warm_horizon(tmp_path):
 c=load(tmp_path,draft());derived=api().recovery_config(c)
 assert derived['schema_version']==4 and derived['updates']==256
 assert derived['raw_targets_per_update']==8192 and derived['loss_scopes']==['full_student']
 assert derived['initializer']['method']=='parent_svd' and derived['seeds']==[17]
 assert derived['checkpoint_every']==16


def test_training_budget_includes_phase_observations_and_rejects_raw_only(tmp_path):
 d=draft();d['resource_budget']={'frozen_profile_sha256':'a'*64,'per_phase':{'warm':limits(16,131072),'recovery':limits(256,2097152)},'campaign':{'wall_seconds':30000,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}}
 with pytest.raises(ValueError,match='scored|observation|complete'):load(tmp_path,d)


def test_source_checkpoint_binding_is_closed(tmp_path):
 d=draft();d['initialization_checkpoint']={'path':'borrowed'}
 with pytest.raises(ValueError):load(tmp_path,d)


@pytest.mark.parametrize('name',['continuation-v1','continuation-v2','training','diagnostics','validation'])
def test_historical_study_identities_cannot_be_claimed(tmp_path,name):
 d=draft();d.update(study=name,out='artifacts/'+name)
 with pytest.raises(ValueError,match='historical|reserved'):load(tmp_path,d)

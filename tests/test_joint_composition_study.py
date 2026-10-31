"""Schema6 separates new-merge sites from complete recovery owner groups."""
from pathlib import Path
import copy
import pytest,yaml
from poc.recovery_study import load_study_config,expected_counts

ROOT=Path(__file__).resolve().parents[1]

def config():
 c=yaml.safe_load((ROOT/'configs/composition-recovery-v1.yaml').read_text())
 c.update(schema_version=6,study='owned-joint-composition-v1',out='artifacts/owned-joint-composition-v1',pair=[12,20],owner_scope='affected_groups',trainable_depths=[0,12,15,20])
 c['composition'].update(merge_pair=[12,20],forced_parent=12,expected_partition=[[0,15] if d==0 else [12,20] if d==12 else [d] for d in range(30) if d not in (15,20)])
 return c

def load(tmp_path,c):
 p=tmp_path/'study.yaml';p.write_text(yaml.safe_dump(c));return load_study_config(p)

def test_joint_schema_preserves_explicit_new_merge_and_all_recovery_sites(tmp_path):
 from poc.recovery_study import recovery_depths
 c=config();assert load(tmp_path,c)==c
 assert recovery_depths(c)==(0,12,15,20) and c['pair']==[12,20]
 old=yaml.safe_load((ROOT/'configs/composition-recovery-v1.yaml').read_text());assert recovery_depths(old)==(0,15,20)

@pytest.mark.parametrize('bad',['missing','extra','partial_old','missing_new','duplicate','unsorted','singleton','empty','bool','scope','one_as_many','many_as_one','partition'])
def test_rejects_ambiguous_partial_or_unrelated_owners_before_content(tmp_path,bad):
 c=config()
 if bad=='missing':del c['trainable_depths']
 elif bad=='extra':c['unpriced']=True
 elif bad=='partial_old':c['trainable_depths']=[0,12,20]
 elif bad=='missing_new':c['trainable_depths']=[0,15]
 elif bad=='duplicate':c['trainable_depths']=[0,12,15,20,20]
 elif bad=='unsorted':c['trainable_depths']=[12,0,15,20]
 elif bad=='singleton':c['trainable_depths']=[0,1,12,15,20]
 elif bad=='empty':c['trainable_depths']=[]
 elif bad=='bool':c['trainable_depths']=[False,12,15,20]
 elif bad=='scope':c['owner_scope']='global'
 elif bad=='one_as_many':c['trainable_depths']=[12,20]
 elif bad=='many_as_one':c['owner_scope']='affected_group'
 else:c['composition']['expected_partition']=[[0,15] if d==0 else [d] for d in range(30) if d!=15]
 with pytest.raises(ValueError):load(tmp_path,c)

def test_same_new_merge_can_recover_only_new_group_explicitly(tmp_path):
 from poc.recovery_study import recovery_depths
 c=config();c.update(owner_scope='affected_group',trainable_depths=[12,20]);assert load(tmp_path,c)==c
 assert recovery_depths(c)==(12,20)

@pytest.mark.parametrize('scope,multiplier',[('full_student',1),('local_sum',4),('local_mean',4)])
def test_raw_and_scored_counts_follow_recovery_owners_not_merge_pair(tmp_path,scope,multiplier):
 c=config();c['loss_scopes']=[scope];c=load(tmp_path,c);v=expected_counts(c,scope)
 assert v['raw_target_presentations']==2097152 and v['scored_target_presentations']==2097152*multiplier

@pytest.mark.parametrize('scope,minimum',[('full_student',114560),('local_sum',139136),('local_mean',139136)])
def test_profile_budget_covers_all_four_depth_diagnostics(tmp_path,scope,minimum):
 c=config();c['loss_scopes']=[scope];c['profile_resource_budget']['valid_target_presentations']=minimum
 assert load(tmp_path,c)==c
 c['profile_resource_budget']['valid_target_presentations']=minimum-1
 with pytest.raises(ValueError,match='profile.*budget'):load(tmp_path,c)

@pytest.mark.parametrize('version',[1,2,3,4,5])
def test_old_schema_rejects_trainable_depths_without_identity_change(tmp_path,version):
 names={1:'recovery-diagnosis-v1.yaml',2:'recovery-diagnosis-v2.yaml',3:'initializer-recovery-v3.yaml',4:'initializer-fixed-parent-r8-s17-parent-svd-v2.yaml',5:'composition-recovery-v1.yaml'}
 c=yaml.safe_load((ROOT/'configs'/names[version]).read_text());assert load(tmp_path,c)==c
 c['trainable_depths']=[0,15]
 with pytest.raises(ValueError,match='unknown|missing'):load(tmp_path,c)

def test_schema_validation_never_reads_roles_or_checkpoint_contents(tmp_path,monkeypatch):
 import poc.data.access,poc.checkpoint
 def denied(*a,**k):raise AssertionError('metadata schema validation must not read corpus or checkpoint')
 monkeypatch.setattr(poc.data.access,'open_records',denied);monkeypatch.setattr(poc.checkpoint,'load_checkpoint',denied)
 assert load(tmp_path,config())['schema_version']==6

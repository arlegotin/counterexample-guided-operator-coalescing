"""Version2 orchestration must fail closed before every native dispatch."""
from pathlib import Path
import importlib.util,sys,json
import pytest


def api():
 path=Path(__file__).parents[1]/'provenance/dense_frontier_campaign.py'
 assert path.is_file(),'explicit new version2 campaign driver missing'
 spec=importlib.util.spec_from_file_location('owned_dense_campaign',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def test_campaign_import_never_initializes_native_backend(monkeypatch):
 import builtins
 original=builtins.__import__
 def guarded(name,*args,**kwargs):
  if name=='mlx' or name.startswith(('mlx.','mlx_lm')):pytest.fail('campaign import initialized Metal')
  return original(name,*args,**kwargs)
 monkeypatch.setattr(builtins,'__import__',guarded)
 m=api();assert callable(m.main)


def test_campaign_missing_source_qualification_blocks_before_serial_or_native_setup(tmp_path,monkeypatch):
 m=api();c={'schema_version':2,'study':'owned','out':str(tmp_path/'out')};config=tmp_path/'proposal.yaml'
 monkeypatch.setattr(m,'read_contract',lambda path:c)
 monkeypatch.setattr(m,'require_dense_qualification',lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('owned qualification absent')))
 monkeypatch.setattr(m,'run_controls',lambda *args,**kwargs:pytest.fail('native setup before qualification'))
 with pytest.raises(ValueError,match='qualification'):m.main(['--config',str(config),'--mode','controls'])
 assert not Path(c['out']).exists()


@pytest.mark.parametrize('mode',['controls','profile','run','analyze'])
def test_public_modes_reach_explicit_existing_dispatch_only_after_source_admission(tmp_path,monkeypatch,mode,capsys):
 m=api();c={'schema_version':2,'study':'owned','out':str(tmp_path/'out')};config=tmp_path/'proposal.yaml';calls=[]
 monkeypatch.setattr(m,'read_contract',lambda path:c)
 monkeypatch.setattr(m,'require_dense_qualification',lambda *args,**kwargs:[{'path':'/private/tmp/source','sha256':'a'*64}])
 for name in ('run_controls','profile_campaign','run_campaign','analyze_campaign'):
  monkeypatch.setattr(m,name,lambda *args,_name=name,**kwargs:calls.append(_name) or {'status':'owned_dispatch'})
 assert m.main(['--config',str(config),'--mode',mode])==0
 assert calls==[{'controls':'run_controls','profile':'profile_campaign','run':'run_campaign','analyze':'analyze_campaign'}[mode]]


def test_v1_contract_cannot_enter_new_native_campaign(tmp_path):
 m=api();p=tmp_path/'old.yaml';p.write_text('schema_version: 1\nstudy: old\n')
 with pytest.raises(ValueError,match='version2'):m.read_contract(p)


def test_run_cannot_start_without_frozen_resource_capsule(tmp_path):
 m=api();c={'resource_freeze':None,'out':str(tmp_path/'out')}
 with pytest.raises(ValueError,match='frozen|resource'):m.run_campaign(c,tmp_path/'config',[])
 assert not Path(c['out']).exists()


def test_profile_cannot_overwrite_an_attempt_or_contain_candidate_quality(tmp_path):
 m=api();out=tmp_path/'out';out.mkdir();(out/'campaign-intent.json').write_text('{}')
 with pytest.raises(FileExistsError):m.profile_campaign({'out':str(out)},tmp_path/'config',[])


def test_review_r2_full_dispatch_does_not_require_future_freeze_in_past_profile(tmp_path,monkeypatch):
 from poc.dense_frontier import dense_recipes
 m=api();c={'study':'owned','out':str(tmp_path/'out')};config=tmp_path/'proposal.yaml';config.write_text('owned')
 source=tmp_path/'source.py';source.write_text('owned');future=tmp_path/'future-freeze.json';future.write_text('{}')
 old_ref=m.reference(source);future_ref=m.reference(future)
 monkeypatch.setattr(m,'preparation_input_refs',lambda c:[])
 monkeypatch.setattr(m,'preparation_profile_closure',lambda *a,**k:[old_ref],raising=False)
 class Observed(Exception):pass
 def saved(c,recipe,*,config_path,required_refs,admissions):
  assert required_refs==[old_ref],'a future freeze was incorrectly required in the already completed profile guard'
  raise Observed()
 monkeypatch.setattr(m,'admit_preparation_freeze',saved)
 with pytest.raises(Observed):m.dispatch_prepare(c,config,dense_recipes()[0],profile=False,
  resource={'wall_seconds':300,'memory_bytes':8*1024**3,'disk_bytes':12*1024**3},refs=[old_ref,future_ref],admissions={})

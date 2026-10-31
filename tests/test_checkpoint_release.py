"""Portable saved-evidence checks: no teacher, corpus, network or native backend."""
from pathlib import Path
import hashlib,importlib.util,json,shutil
import pytest
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'provenance/continuation-checkpoint-2026-10-31'


def api():
 path=ROOT/'provenance/verify_checkpoint.py'
 assert path.is_file(),'portable release verifier missing'
 spec=importlib.util.spec_from_file_location('public_checkpoint_verifier',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def index_for(base,path):
 files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in base.iterdir() if p.is_file()}
 path.write_text(json.dumps({'schema_version':1,'release':'0.2.0','evidence_directory':'provenance/continuation-checkpoint-2026-10-31','files':files}))
 return path


@pytest.fixture
def packet(tmp_path):
 base=tmp_path/'evidence';shutil.copytree(BASE,base);index=index_for(base,tmp_path/'index.json');return base,index


def test_release_numeric_evidence_is_portable_and_complete(packet,tmp_path,monkeypatch):
 base,index=packet;m=api();original=Path.open
 def guarded(path,*a,**k):
  assert path.resolve().is_relative_to(tmp_path),'verifier followed author-specific external path'
  return original(path,*a,**k)
 monkeypatch.setattr(Path,'open',guarded)
 result=m.verify_checkpoint(base,index=index)
 assert result['hashed_files']==42 and result['native_artifacts']==12 and result['recovery_observations']==20
 assert result['valid_targets']==65408 and result['training_rerun'] is False and result['native_rerun'] is False


def test_optional_bootstrap_reconstructs_recorded_recovery_and_native_intervals(packet):
 base,index=packet;result=api().verify_checkpoint(base,index=index,bootstrap=True)
 assert result['bootstrap']['recovery_draws']==10000 and result['bootstrap']['native_interval_reproduced'] is True


@pytest.mark.parametrize('change',['hash','missing','extra','symlink','hardlink','traversal','wrong_version','unknown_field','nonfinite'])
def test_release_integrity_rejections(packet,tmp_path,change):
 base,index=packet;m=api();p=base/'native-dense_gptq4.json';source_links=(BASE/p.name).stat().st_nlink
 if change=='hash':p.write_text('{}')
 elif change=='missing':p.unlink()
 elif change=='extra':(base/'unlisted.json').write_text('{}')
 elif change=='symlink':p.unlink();p.symlink_to(BASE/p.name)
 elif change=='hardlink':
  owned=tmp_path/'owned-original.json';owned.write_bytes(p.read_bytes());p.unlink();__import__('os').link(owned,p)
 elif change=='nonfinite':p.write_text('{"value":NaN}');index_for(base,index)
 else:
  v=json.loads(index.read_text())
  if change=='traversal':v['files']['../../outside.json']=v['files'].pop(p.name)
  elif change=='wrong_version':v['release']='0.3.0'
  else:v['trusted']=True
  index.write_text(json.dumps(v))
 assert (BASE/p.name).stat().st_nlink==source_links,'negative fixture aliased immutable source evidence'
 with pytest.raises((ValueError,FileNotFoundError)):m.verify_checkpoint(base,index=index)


@pytest.mark.parametrize('change',['denominator','component','window','nonfinite_metric','complete_bytes','contrast','pipeline_success','capability_claim','native_horizon'])
def test_rehashed_numerical_or_scope_tampering_is_rejected(packet,change):
 base,index=packet
 name='native-dense_gptq4.json';v=json.loads((base/name).read_text())
 if change=='denominator':v['rows'][0]['valid_targets']=0
 elif change=='component':v['rows'][0]['component_id']='f'*64
 elif change=='window':v['rows'][0]['window_sha256']='f'*64
 elif change=='nonfinite_metric':v['rows'][0]['kl_sum']=float('inf')
 elif change=='complete_bytes':v['complete_bytes']+=1
 elif change in ('contrast','pipeline_success'):
  name='recovery-analysis.json';v=json.loads((base/name).read_text())
  if change=='contrast':v['full_minus_local']['kl_difference']+=.01
  else:v['original_local_failure']['original_pipeline_status']='complete'
 else:
  name='summary.json';v=json.loads((base/name).read_text())
  if change=='capability_claim':v['claims']['capability_acceptability_established']=True
  else:
   name='completion.json';v=json.loads((base/name).read_text());v['native_artifact_recovery_horizon']=1024
 (base/name).write_text(json.dumps(v));index_for(base,index)
 with pytest.raises(ValueError):api().verify_checkpoint(base,index=index)

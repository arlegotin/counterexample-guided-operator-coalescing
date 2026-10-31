"""Owned real-collector persistence controls; no pretrained/native execution."""
import importlib,importlib.util,json,hashlib
from dataclasses import replace
from pathlib import Path
import pytest,torch
from test_shared_calibration import owned,bind
from poc.shared_calibration import collect_shared_inputs


def api():
 assert importlib.util.find_spec('poc.shared_calibration_bundle'),'verified calibration persistence is missing'
 return importlib.import_module('poc.shared_calibration_bundle')


@pytest.fixture
def measured(owned):
 return collect_shared_inputs(owned.layout,owned.layout,owned.population,owned.windows,bind(owned))


def save(m,result,path):
 return m.save_calibration(result,path,expected_result_sha256=result.digest(),study_contract_sha256='c'*64)


def rebind(m,ref,change):
 path=Path(ref.directory)/'manifest.json';data=json.loads(path.read_text());change(data);path.write_text(json.dumps(data));return replace(ref,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_every_actual_owner_use_window_and_fp64_value_reloads_exactly(measured,tmp_path):
 m=api();ref=save(m,measured,tmp_path/'stats');loaded=m.load_calibration(ref,expected_contract_sha256='c'*64)
 assert loaded.digest()==measured.digest() and loaded.identity()==measured.identity()
 assert len(loaded.owners)==9 and loaded.owners['bank.0.down_proj.weight'].count==2044
 assert loaded.owners['bank.0.down_proj.weight'].use_counts=={0:1022,2:1022}
 assert loaded.owners['bank.0.down_proj.weight'].moment_sum.shape==(12,12)
 for key,stats in measured.owners.items():
  for name in ('moment_sum','abs_sum','max_abs'):
   value=getattr(loaded.owners[key],name);assert value.dtype==torch.float64 and value.device.type=='cpu'
   torch.testing.assert_close(value,getattr(stats,name),rtol=0,atol=0)
 assert set(p.name for p in Path(ref.directory).iterdir())=={'manifest.json','statistics.safetensors'}


def test_mutated_result_cannot_replace_the_recorded_collection_digest(measured,tmp_path):
 m=api();expected=measured.digest();measured.owners['bank.0.gate_proj.weight'].moment_sum[0,0]+=1
 with pytest.raises(ValueError):m.save_calibration(measured,tmp_path/'stats',expected_result_sha256=expected,study_contract_sha256='c'*64)
 assert not (tmp_path/'stats').exists()


@pytest.mark.parametrize('bad',['dtype','shape','nan','count','use_count','negative_abs','owner_key','contract','expected_digest'])
def test_writer_rejects_invalid_statistics_even_with_fresh_digest(measured,tmp_path,bad):
 m=api();stats=measured.owners['bank.0.down_proj.weight'];contract='c'*64;expected=None
 if bad=='dtype':stats.max_abs=stats.max_abs.float()
 elif bad=='shape':stats.moment_sum=stats.moment_sum[:-1]
 elif bad=='nan':stats.abs_sum[0]=float('nan')
 elif bad=='count':stats.count=1
 elif bad=='use_count':stats.use_counts[0]-=1
 elif bad=='negative_abs':stats.max_abs[0]=-1
 elif bad=='owner_key':measured.owners['bad']=measured.owners.pop('bank.0.down_proj.weight')
 elif bad=='contract':contract='bad'
 elif bad=='expected_digest':expected='0'*64
 with pytest.raises(ValueError):m.save_calibration(measured,tmp_path/'stats',expected_result_sha256=expected or measured.digest(),study_contract_sha256=contract)
 assert not (tmp_path/'stats').exists()


def test_existing_destination_is_never_overwritten(measured,tmp_path):
 m=api();target=tmp_path/'stats';target.mkdir();(target/'keep').write_text('original')
 with pytest.raises(FileExistsError):save(m,measured,target)
 assert (target/'keep').read_text()=='original'


@pytest.mark.parametrize('bad',['tensor_bytes','manifest_bytes','schema','extra_field','contract','extra_file','symlink','hardlink','result_digest','declared_shape'])
def test_loader_rejects_corruption_and_unbound_references(measured,tmp_path,bad):
 m=api();ref=save(m,measured,tmp_path/'stats');root=Path(ref.directory);expected='c'*64
 if bad=='tensor_bytes':(root/'statistics.safetensors').write_bytes(b'owned corruption')
 elif bad=='manifest_bytes':(root/'manifest.json').write_text('{}')
 elif bad=='schema':ref=rebind(m,ref,lambda d:d.update(schema_version=2))
 elif bad=='extra_field':ref=rebind(m,ref,lambda d:d.update(unknown=True))
 elif bad=='contract':expected='d'*64
 elif bad=='extra_file':(root/'omitted.bin').write_bytes(b'cost')
 elif bad=='symlink':p=root/'statistics.safetensors';outside=tmp_path/'elsewhere';p.rename(outside);p.symlink_to(outside)
 elif bad=='hardlink':(tmp_path/'alias').hardlink_to(root/'statistics.safetensors')
 elif bad=='result_digest':ref=rebind(m,ref,lambda d:d.update(result_sha256='f'*64))
 elif bad=='declared_shape':ref=rebind(m,ref,lambda d:d['result_identity']['owner_layout']['bank.0.down_proj.weight'].update(shape=[8,13]))
 with pytest.raises(ValueError):m.load_calibration(ref,expected_contract_sha256=expected)


def test_failed_serialization_retains_partial_work_without_an_accepted_ref(measured,tmp_path,monkeypatch):
 import safetensors.torch
 m=api()
 def fail(tensors,path,*a,**k):Path(path).write_bytes(b'owned partial');raise OSError('owned serialization failure')
 monkeypatch.setattr(safetensors.torch,'save_file',fail)
 with pytest.raises(OSError,match='owned serialization'):save(m,measured,tmp_path/'stats')
 assert not (tmp_path/'stats').exists()
 failed=list(tmp_path.glob('stats.failed-*'));assert len(failed)==1 and (failed[0]/'failure.json').is_file()
 assert (failed[0]/'statistics.safetensors').read_bytes()==b'owned partial'


@pytest.mark.parametrize('operation',['save','load'])
def test_symlinked_ancestor_is_denied_before_statistics_io(measured,tmp_path,operation):
 m=api();real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
 if operation=='save':
  with pytest.raises(ValueError):save(m,measured,alias/'stats')
  assert not (real/'stats').exists()
 else:
  ref=save(m,measured,real/'stats');bad=replace(ref,directory=str(alias/'stats'))
  with pytest.raises(ValueError):m.load_calibration(bad,expected_contract_sha256='c'*64)

"""Explicit mutable exposure prefixes; all input bytes remain in the census."""
import importlib.util,json
import pytest
pytestmark=pytest.mark.cpu


def api():
 assert importlib.util.find_spec('poc.reconstruction_resources'), 'explicit study input/exposure footprint missing'
 from poc.reconstruction_resources import StudyFamily
 return StudyFamily


def test_exposure_append_keeps_identity_and_increases_disk_inside_or_outside_roots(tmp_path):
 dep=tmp_path/'dependency';dep.mkdir();source=dep/'model';source.write_bytes(b'immutable')
 inside=dep/'exposure.jsonl';inside.write_bytes(b'first\n');outside=tmp_path/'outside.jsonl';outside.write_bytes(b'first\n')
 data=tmp_path/'tokens';data.write_bytes(b'input bytes')
 cls=api();args=dict(output=str(tmp_path/'out'),dependencies=[str(dep)],required_sources=[str(source)],input_files=[str(data)],mutable_ledgers=[str(inside),str(outside)])
 a=cls(**args);identity=a.identity();before=a.disk();prefix=a.exposure_snapshot()
 for p in (inside,outside):
  with p.open('ab') as f:f.write(b'second\n')
 b=cls(**args)
 assert b.identity()==identity and b.disk()==before+14
 b.verify_exposure_prefixes(prefix)
 inside.write_bytes(b'changed\n')
 with pytest.raises(ValueError,match='prefix|truncat'):b.verify_exposure_prefixes(prefix)


def test_optional_new_exposure_ledger_has_stable_identity_without_omitting_bytes(tmp_path):
 dep=tmp_path/'dependency';dep.mkdir();source=dep/'model';source.write_bytes(b'immutable');ledger=dep/'new.jsonl'
 cls=api();args=dict(output=str(tmp_path/'out'),dependencies=[str(dep)],required_sources=[str(source)],input_files=[str(source)],mutable_ledgers=[str(ledger)])
 a=cls(**args);prefix=a.exposure_snapshot();before=a.disk();ledger.write_bytes(b'created\n');b=cls(**args)
 assert a.identity()==b.identity() and b.disk()==before+8
 b.verify_exposure_prefixes(prefix)
 snapshot=b.exposure_snapshot();ledger.unlink()
 with pytest.raises(ValueError,match='prefix|missing|truncat'):cls(**args).verify_exposure_prefixes(snapshot)


def test_source_tensor_cannot_be_declared_a_mutable_exposure_ledger(tmp_path):
 dep=tmp_path/'dependency';dep.mkdir();source=dep/'model';source.write_bytes(b'immutable')
 with pytest.raises(ValueError,match='immutable|source|overlap'):
  api()(output=str(tmp_path/'out'),dependencies=[str(dep)],required_sources=[str(source)],input_files=[],mutable_ledgers=[str(source)])

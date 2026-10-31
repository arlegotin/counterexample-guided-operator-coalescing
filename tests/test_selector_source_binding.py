"""Owned integration metadata; never touches real corpus/model/protected data."""
import hashlib,json
from pathlib import Path
import pytest
from poc.records import canonical_json

ROOT=Path(__file__).resolve().parents[1]

def reference(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def make_binding(tmp_path, *, change=None):
    table=json.loads((ROOT/'provenance/continuation-integration-v1/import-v1.json').read_text())['files']
    table={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in table}
    # New isolated studies can add executable helpers beyond the parent table.
    table.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'poc').rglob('*.py')})
    # Include this newly executed test, too.
    table['tests/test_selector_source_binding.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Schema/admission fixture only. Actual predecessor receipts are frozen
    # separately by the integration supervisor, never invented by these tests.
    parent=tmp_path/'synthetic-feature-qualification.json'
    parent.write_bytes(canonical_json({'schema_version':1,'status':'qualified_owned_controls_passed',
        'scope':'synthetic owned fixture qualification metadata; no real program or model qualification claimed'}))
    value={'schema_version':2,'files':table,'parent_qualifications':[reference(parent)]}
    if change:change(value)
    p=tmp_path/'source-binding-v2.json';p.write_bytes(canonical_json(value));return reference(p)


def test_explicit_source_binding_runs_actual_owned_control(tmp_path):
    from poc.selector_controls import run_owned_selector_control
    binding=make_binding(tmp_path)
    result=run_owned_selector_control(tmp_path/'out',regime='hidden_failure',seed=17,source_binding=binding)
    assert result['schema_version']==2
    assert result['source_binding']==binding
    assert result['natural_choice']!=result['active_choice']==result['damage_choice']
    spec=json.loads(next((tmp_path/'out').glob('ledgers/*/spec.json')).read_text())
    assert spec['input_manifests']['source_binding_v2']==binding['sha256']
    assert spec['method_variant']=='owned_tiny_bank_selector_control_v2'


@pytest.mark.parametrize('mutation',['source','omit','unknown','parent','hash','alias'])
def test_invalid_v2_binding_denied_before_model_allocation(tmp_path,monkeypatch,mutation):
    import poc.selector_controls as module
    def change(value):
        if mutation=='source':value['files']['poc/train.py']='0'*64
        elif mutation=='omit':value['files'].pop('poc/train.py')
        elif mutation=='unknown':value['extra']=True
        elif mutation=='parent':value['parent_qualifications'][0]['sha256']='0'*64
    binding=make_binding(tmp_path,change=change)
    if mutation=='hash':binding['sha256']='0'*64
    if mutation=='alias':
        alias=tmp_path/'alias.json';alias.symlink_to(binding['path']);binding['path']=str(alias)
    monkeypatch.setattr(module,'planted_models',lambda *a,**k:pytest.fail('allocated before source binding admission'))
    with pytest.raises((ValueError,FileNotFoundError)):
        module.run_owned_selector_control(tmp_path/'out',regime='hidden_failure',seed=17,source_binding=binding)


def test_legacy_none_does_not_silently_use_integrated_binding(tmp_path):
    from poc.selector_controls import run_owned_selector_control
    with pytest.raises(FileNotFoundError):run_owned_selector_control(tmp_path/'out',regime='hidden_failure',seed=17)


def test_omitted_executed_binding_helper_is_denied_before_model_io(tmp_path,monkeypatch):
    import poc.selector_controls as module
    binding=make_binding(tmp_path,change=lambda value:value['files'].pop('tests/test_selector_source_binding.py'))
    monkeypatch.setattr(module,'planted_models',lambda *a,**k:pytest.fail('model allocation after omitted executed helper'))
    with pytest.raises(ValueError):module.run_owned_selector_control(tmp_path/'out',regime='hidden_failure',seed=17,source_binding=binding)


def test_binding_fixture_requires_no_original_author_worktree(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(sys.modules[__name__],'O',tmp_path/'absent-original-worktree',raising=False)
    binding=make_binding(tmp_path)
    raw=json.loads(Path(binding['path']).read_text())
    assert raw['parent_qualifications']
    assert all(Path(ref['path']).is_relative_to(tmp_path) for ref in raw['parent_qualifications'])
    assert json.loads(Path(raw['parent_qualifications'][0]['path']).read_text())['scope'].startswith('synthetic owned fixture')

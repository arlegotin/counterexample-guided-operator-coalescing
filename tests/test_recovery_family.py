"""Canonical retained-output accounting; no teacher or corpus I/O."""
from pathlib import Path
import importlib
import pytest


def api():
    assert importlib.util.find_spec('poc.recovery_family'),'retained family census missing'
    return importlib.import_module('poc.recovery_family')


def roots(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';out=tmp_path/'out'
    for p in (a,b,out):p.mkdir()
    (a/'spec.json').write_bytes(b'a'*17);(b/'manifest.json').write_bytes(b'b'*23);(out/'ledger').write_bytes(b'c'*5)
    return a,b,out


def test_family_counts_all_files_once_and_does_not_cache_mutable_sizes(tmp_path):
    a,b,out=roots(tmp_path)
    family=api().RetainedFamily(str(out),[str(a),str(b)],required_sources=[str(a/'spec.json'),str(b/'manifest.json')])
    assert family.disk()==45
    (b/'new-checkpoint').write_bytes(b'x'*11)
    assert family.disk()==56
    assert family.identity()['dependencies']==[str(a),str(b)]


@pytest.mark.parametrize('kind',['duplicate','nested','output','parent_output','relative','missing','symlink','uncovered'])
def test_family_rejects_aliases_overlap_and_omitted_source_roots(tmp_path,kind,monkeypatch):
    a,b,out=roots(tmp_path);deps=[str(a),str(b)];required=[str(a/'spec.json'),str(b/'manifest.json')]
    if kind=='duplicate':deps=[str(a),str(a)]
    elif kind=='nested':deps=[str(tmp_path),str(a)]
    elif kind=='output':deps=[str(a),str(out)]
    elif kind=='parent_output':deps=[str(tmp_path)]
    elif kind=='relative':monkeypatch.chdir(tmp_path);deps=['a',str(b)]
    elif kind=='missing':deps=[str(a),str(tmp_path/'missing')]
    elif kind=='symlink':
        alias=tmp_path/'alias';alias.symlink_to(b,target_is_directory=True);deps=[str(a),str(alias)]
    elif kind=='uncovered':deps=[str(a)]
    with pytest.raises(ValueError):api().RetainedFamily(str(out),deps,required_sources=required)


def test_family_rejects_nested_file_symlinks_at_census(tmp_path):
    a,b,out=roots(tmp_path);family=api().RetainedFamily(str(out),[str(a),str(b)])
    (a/'alias').symlink_to(b/'manifest.json')
    with pytest.raises(ValueError,match='symlink'):family.disk()


def test_family_detects_root_replacement_and_empty_dependency_declaration(tmp_path):
    a,b,out=roots(tmp_path)
    with pytest.raises(ValueError):api().RetainedFamily(str(out),[])
    family=api().RetainedFamily(str(out),[str(a),str(b)])
    moved=tmp_path/'moved';a.rename(moved);a.symlink_to(moved,target_is_directory=True)
    with pytest.raises(ValueError):family.disk()


def test_required_source_disappearance_never_silently_reduces_retained_bytes(tmp_path):
    a,b,out=roots(tmp_path);source=a/'spec.json'
    family=api().RetainedFamily(str(out),[str(a),str(b)],required_sources=[str(source)])
    source.unlink()
    with pytest.raises(ValueError,match='required|retained|disappear'):family.disk()


def test_original_checkpoint_payloads_remain_required_even_when_not_named_separately(tmp_path):
    a,b,out=roots(tmp_path);payload=b/'tensors.safetensors';payload.write_bytes(b'owned payload')
    family=api().RetainedFamily(str(out),[str(a),str(b)],required_sources=[str(b/'manifest.json')])
    payload.unlink()
    with pytest.raises(ValueError,match='required|retained|disappear'):family.disk()

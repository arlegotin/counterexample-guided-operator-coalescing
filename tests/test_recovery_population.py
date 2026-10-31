"""Natural-only preparation uses owned text; no network or protected files."""
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('poc.recovery_population'),'natural-only population builder missing'
    from poc import recovery_population
    return recovery_population


class OwnedTokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):
        assert not add_special_tokens
        return [2+ord(x)%7 for x in text]


def records():
    from poc.data.manifest import DocumentRecord
    return [DocumentRecord(f'd{i}','owned-natural','a'*40,'owned.jsonl',i,
        text=(chr(65+i)+' unique '+str(i)+' ')*80) for i in range(8)]


def test_natural_preparation_roles_precede_windows_and_no_historical_donors(tmp_path):
    a=api();out=tmp_path/'population'
    result=a.prepare_natural_records(records(),OwnedTokenizer(),out,tokenizer_digest='b'*64,
        inventory_sha256='c'*64,fit_components=3,tuning_components=2,seed=17,
        max_document_bytes=10000,max_records=8,max_tokens=100000,max_disk_bytes=1000000)
    fit,fit_batches=a.load_natural_population(result['fit']['path'],result['fit']['sha256'],role='fit',components=3,tokenizer_digest='b'*64)
    tune,tune_batches=a.load_natural_population(result['tuning']['path'],result['tuning']['sha256'],role='dev',components=2,tokenizer_digest='b'*64)
    assert len(fit_batches)==3 and len(tune_batches)==2
    assert not set(fit.membership.values())&set(tune.membership.values())
    assert all(b.input_ids.shape==(1,512) and int(b.target_mask.sum())==511 for b in fit_batches+tune_batches)
    assert result['historical_exposure']=='unknown; exploratory only'
    with pytest.raises(FileExistsError):
        a.prepare_natural_records(records(),OwnedTokenizer(),out,tokenizer_digest='b'*64,
            inventory_sha256='c'*64,fit_components=3,tuning_components=2,seed=17,
            max_document_bytes=10000,max_records=8,max_tokens=100000,max_disk_bytes=1000000)
    with pytest.raises(PermissionError):
        a.load_natural_population(result['tuning']['path'],result['tuning']['sha256'],role='fit',components=2,tokenizer_digest='b'*64)


def test_natural_shortage_is_retained_without_replacement_after_outcomes(tmp_path):
    a=api()
    with pytest.raises(ValueError,match='shortage'):
        a.prepare_natural_records(records()[:1],OwnedTokenizer(),tmp_path/'short',tokenizer_digest='b'*64,
            inventory_sha256='c'*64,fit_components=3,tuning_components=2,seed=17,
            max_document_bytes=10000,max_records=8,max_tokens=100000,max_disk_bytes=1000000)
    assert (tmp_path/'short'/'preparation.json').exists()


def test_explicit_fineweb_inventory_refuses_revision_and_duplicate_rows_before_network(tmp_path):
    a=api()
    from poc.records import canonical_json
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    inventory={'schema_version':1,'repository':'HuggingFaceFW/fineweb-edu','revision':'0'*40,
        'shards':[{'path':'sample/10BT/000_00000.parquet','row_groups':[{'index':0,'rows':[1,2]}]}]}
    path=tmp_path/'inventory.json';path.write_bytes(canonical_json(inventory))
    with pytest.raises(ValueError,match='revision'):
        a.read_fineweb_inventory(path,hashlib.sha256(path.read_bytes()).hexdigest(),tmp_path/'ranges',
            source_manifest=root/'provenance/assets-135m-sources.json',max_range_bytes=1000000,
            max_request_bytes=100000,max_disk_bytes=1000000,max_records=8,max_document_bytes=10000)
    assert not (tmp_path/'ranges').exists()
    inventory['revision']='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
    inventory['shards'][0]['row_groups'][0]['rows']=[1,1]
    path.write_bytes(canonical_json(inventory))
    with pytest.raises(ValueError,match='duplicate|unique'):
        a.read_fineweb_inventory(path,hashlib.sha256(path.read_bytes()).hexdigest(),tmp_path/'ranges',
            source_manifest=root/'provenance/assets-135m-sources.json',max_range_bytes=1000000,
            max_request_bytes=100000,max_disk_bytes=1000000,max_records=8,max_document_bytes=10000)
    assert not (tmp_path/'ranges').exists()


def test_known_prior_text_overlap_is_excluded_before_role_assignment(tmp_path):
    from dataclasses import replace
    a=api();prior=replace(records()[0],document_id='historically-exposed')
    result=a.prepare_natural_records(records(),OwnedTokenizer(),tmp_path/'new',tokenizer_digest='b'*64,
        inventory_sha256='c'*64,fit_components=3,tuning_components=2,seed=17,
        max_document_bytes=10000,max_records=8,max_tokens=100000,max_disk_bytes=1000000,
        prior_records=(prior,),prior_manifest_hashes=('d'*64,))
    assert result['known_prior_records']==1
    assert any(r['document_id']=='d0' and r['reason']=='known_prior_overlap' for r in result['exclusions'])

"""Safe local teacher loading rejects corruption before any model construction."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import pytest
import torch

pytestmark=pytest.mark.cpu


def api():
    assert importlib.util.find_spec('poc.teacher'), 'safe local teacher loader is missing'
    return importlib.import_module('poc.teacher')


@pytest.fixture
def local_asset(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    model=LlamaForCausalLM(LlamaConfig(vocab_size=8,hidden_size=8,intermediate_size=12,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=64,tie_word_embeddings=True)).eval()
    model.save_pretrained(tmp_path,safe_serialization=True)
    token=Tokenizer(WordLevel({'<eos>':0,'<unk>':1,'x':2},unk_token='<unk>'))
    PreTrainedTokenizerFast(tokenizer_object=token,eos_token='<eos>',unk_token='<unk>').save_pretrained(tmp_path)
    files=[]
    for p in sorted(tmp_path.iterdir()):
        files.append(dict(local_path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                          source=dict(repository='owned/tiny',revision='a'*40,path=p.name,terms_permitted=True)))
    raw=json.dumps(dict(schema_version=1,assets=files)).encode(); manifest=tmp_path/'assets.json';manifest.write_bytes(raw)
    return manifest,hashlib.sha256(raw).hexdigest()


def test_local_safetensors_load_preserves_shapes_tie_and_frozen_ownership(local_asset):
    m=api(); asset=m.AssetManifest.load(local_asset[0],expected_sha256=local_asset[1])
    teacher=m.load_teacher(asset)
    assert teacher.config.hidden_size==8 and teacher.config.num_key_value_heads==1
    assert teacher.lm_head.weight is teacher.embed_tokens.weight
    assert not any(p.requires_grad for p in teacher.parameters())
    assert m.load_tokenizer(asset).eos_token_id==0


def test_modified_source_and_manifest_escape_fail_closed_before_load(local_asset):
    m=api(); manifest,digest=local_asset
    asset=m.AssetManifest.load(manifest,expected_sha256=digest)
    (manifest.parent/'model.safetensors').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash|size'): m.load_teacher(asset)
    with pytest.raises(ValueError,match='hash'): m.AssetManifest.load(manifest,expected_sha256='0'*64)


def test_snapshot_digest_rechecks_versions_config_and_dtype():
    api()
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    model=make_tiny_decoder(TinyConfig(),seed=1,dtype=torch.float32)
    before=model.manifest().checkpoint_digest
    assert model.manifest().checkpoint_digest==before
    with torch.no_grad(): model.layers[0].mlp.down_proj.weight.add_(.001)
    after=model.manifest().checkpoint_digest
    assert after!=before
    model.config.rms_norm_eps*=2
    assert model.manifest().checkpoint_digest!=after
    configured=model.manifest().checkpoint_digest
    model.double()
    assert model.manifest().dtype=='float64'
    assert model.manifest().checkpoint_digest!=configured


def test_bank_private_mapping_mutation_cannot_reuse_digest_or_cache():
    api()
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.bank import BankDecoder,canonicalize_partition
    teacher=make_tiny_decoder(TinyConfig(),seed=1,dtype=torch.float32)
    bank=BankDecoder(teacher,canonicalize_partition(((0,1),(2,)),3),0,17)
    bank.manifest(); cache=bank.new_cache()
    bank.layers[0].mlp.index=1
    with pytest.raises(ValueError,match='mapping|owner|partition'): bank.manifest()
    with pytest.raises(ValueError,match='mapping|owner|partition'): bank.new_cache()


def test_original_hf_parity_helper_records_actual_errors_and_costs(local_asset):
    m=api()
    assert importlib.util.find_spec('poc.teacher_validation'), 'teacher qualification is missing'
    from poc.teacher_validation import qualify_numerics
    asset=m.AssetManifest.load(local_asset[0],expected_sha256=local_asset[1])
    reference=m.load_reference(asset)
    result=qualify_numerics(reference,device='cpu',lengths=(8,32))
    assert result['unique_parameters']>0
    assert result['hf_fp32_max_abs_error']==0
    assert result['identity_kl_per_token']==0
    assert result['negative_kl_floor_per_token']>=0
    assert len(result['suffix_costs'])==2
    assert result['peak_rss_bytes']>0


def test_same_pinned_teacher_content_has_location_independent_executable_identity(local_asset,tmp_path):
    import shutil
    m=api();path,digest=local_asset
    second=tmp_path/'second-location';second.mkdir()
    for source in list(path.parent.iterdir()):
        if source.is_file(): shutil.copy2(source,second/source.name)
    first=m.load_teacher(m.AssetManifest.load(path,expected_sha256=digest))
    other=m.load_teacher(m.AssetManifest.load(second/path.name,expected_sha256=digest))
    assert first.manifest().checkpoint_digest==other.manifest().checkpoint_digest


def test_teacher_qualification_invalid_config_publishes_failed_record(tmp_path):
    from poc.teacher_validation import run_teacher_validation
    out=tmp_path/'qualification'
    with pytest.raises(FileNotFoundError):
        run_teacher_validation(config_path=tmp_path/'missing.yaml',device='cpu',out=out)
    result=json.loads((out/'summary.json').read_text())
    assert result['status']=='failed' and result['gate']=='WP02'


def test_external_storage_writer_explicit_invalidation_rehashes_bytes():
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    model=make_tiny_decoder(TinyConfig(),seed=1,dtype=torch.float32)
    before=model.manifest().checkpoint_digest
    model.embed_tokens.weight.data.add_(.01)
    model.invalidate_state_snapshot()
    assert model.manifest().checkpoint_digest!=before


@pytest.mark.mps
def test_device_change_invalidates_model_snapshot_and_prior_cache():
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from test_model_parity import batch
    assert torch.backends.mps.is_available()
    model=make_tiny_decoder(TinyConfig(),seed=1,dtype=torch.float32)
    cache=model.new_cache(); before=model.manifest().checkpoint_digest
    model.to('mps')
    assert model.manifest().checkpoint_digest!=before
    with pytest.raises(ValueError,match='state'): model.logits(batch(device='mps'),cache=cache)

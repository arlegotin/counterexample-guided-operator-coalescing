"""Owned Metal parity for explicit absent-factor deployment metadata."""
from pathlib import Path
import pytest
import torch
pytestmark=pytest.mark.mlx

@pytest.mark.parametrize('packed',[False,True])
def test_native_elision_matches_identical_unelided_format_and_cached_decode(tmp_path,packed):
 import mlx.core as mx
 import numpy as np
 from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
 from poc.bank import BankDecoder,canonicalize_partition
 from poc.bundle import write_bundle,load_bundle,verify_bundle,measure_bundle
 from poc.quantize import quantize_rtn,QuantizationSpec
 from test_export import runtime
 from test_native_parity import native_batch
 from test_model_parity import batch
 from test_offline_bundle import run_offline_bundle
 assert mx.metal.is_available()
 dense=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2),seed=4,dtype=torch.float32)
 source=BankDecoder(dense,canonicalize_partition([[0,2],[1]],3),8,17);rt=runtime(tmp_path)
 refs=[write_bundle(source,tmp_path/name,rt,elide_zero_residuals=flag) for name,flag in [('original',False),('elided',True)]]
 if packed:refs=[quantize_rtn(ref,QuantizationSpec('rtn',4,64,str(tmp_path/f'packed-{i}')),None).bundle for i,ref in enumerate(refs)]
 assert measure_bundle(Path(refs[1].directory)).payload_bytes<measure_bundle(Path(refs[0].directory)).payload_bytes
 old,new=[load_bundle(Path(ref.directory),backend='mlx') for ref in refs]
 b=native_batch(batch());a=old.logits(b);z=new.logits(b);mx.eval(a,z)
 np.testing.assert_array_equal(np.array(a),np.array(z))
 cache=new.new_cache();pieces=[new.logits(native_batch(batch([[t]],offset=i)),cache=cache) for i,t in enumerate([1,4,7,2,3])]
 np.testing.assert_allclose(np.array(mx.concatenate(pieces,axis=1)),np.array(z),atol=1e-4,rtol=0)
 assert old.owner('embed_tokens.weight') is old.owner('lm_head.weight') and new.owner('embed_tokens.weight') is new.owner('lm_head.weight')
 raw=verify_bundle(Path(refs[1].directory)).metadata
 assert len([o for o in raw['owners'] if o['category']=='bank'])==6
 assert all(o['dtype']=='bfloat16' for o in raw['owners'] if o['category']=='residual') if packed else True
 assert run_offline_bundle(refs[1].directory,tmp_path,backend='mlx')['download_denied']


@pytest.mark.parametrize('consumer',['mlx','quantization'])
def test_native_paths_reject_incorrect_floating_deployment_digest(tmp_path,consumer):
 import hashlib,json
 from poc.bundle import write_bundle,load_bundle
 from poc.contracts import BundleRef
 from poc.quantize import quantize_rtn,QuantizationSpec
 from test_export import model,runtime
 ref=write_bundle(model('2'),tmp_path/'elided',runtime(tmp_path),elide_zero_residuals=True)
 root=Path(ref.directory);path=root/'manifest.json';raw=json.loads(path.read_text());raw['deployment_checkpoint_digest']='0'*64;path.write_text(json.dumps(raw))
 with pytest.raises(ValueError,match='deployment checkpoint digest'):
  if consumer=='mlx':load_bundle(root,backend='mlx')
  else:quantize_rtn(BundleRef(str(root),hashlib.sha256(path.read_bytes()).hexdigest()),QuantizationSpec('rtn',4,64,str(tmp_path/'packed')),None)

"""Inference-only zero factor elision must preserve function and source state."""
import json
from pathlib import Path
import pytest
import torch
from test_export import model,runtime
from test_model_parity import batch
pytestmark=pytest.mark.cpu

def test_exact_zero_elision_roundtrip_smaller_and_source_unchanged(tmp_path):
 from poc.bundle import write_bundle,load_bundle,verify_bundle,measure_bundle
 source=model('2');before=source.manifest().checkpoint_digest
 params={n:p.detach().clone() for n,p in source.named_parameters()};grad={n:p.requires_grad for n,p in source.named_parameters()};rt=runtime(tmp_path)
 ordinary=write_bundle(source,tmp_path/'ordinary',rt)
 ref=write_bundle(source,tmp_path/'elided',rt,elide_zero_residuals=True)
 raw=verify_bundle(Path(ref.directory)).metadata
 assert raw['schema_version']==2 and raw['inference_only'] is True
 assert raw['factor_presence'][0]==dict.fromkeys(('gate_proj','up_proj','down_proj'),False)
 assert raw['factor_presence'][1]==dict.fromkeys(('gate_proj','up_proj','down_proj'),False)
 assert raw['factor_presence'][2]==dict.fromkeys(('gate_proj','up_proj','down_proj'),True)
 assert measure_bundle(Path(ref.directory)).payload_bytes<measure_bundle(Path(ordinary.directory)).payload_bytes
 loaded=load_bundle(Path(ref.directory),backend='torch')
 torch.testing.assert_close(loaded.logits(batch()),source.logits(batch()),rtol=0,atol=0)
 cache=loaded.new_cache();pieces=[loaded.logits(batch([[t]],offset=i),cache=cache) for i,t in enumerate([1,4,7,2,3])]
 torch.testing.assert_close(torch.cat(pieces,1),loaded.logits(batch()),rtol=0,atol=2e-6)
 assert all(not p.requires_grad for p in loaded.parameters())
 assert source.manifest().checkpoint_digest==before
 assert all(torch.equal(p,params[n]) and p.requires_grad==grad[n] for n,p in source.named_parameters())
 assert not loaded.layers[0].mlp.residuals
 assert raw['deployment_checkpoint_digest']==loaded.manifest().checkpoint_digest

@pytest.mark.parametrize('zero_factor',['A','B'])
def test_partial_projection_presence_and_no_tolerance(tmp_path,zero_factor):
 from poc.bundle import write_bundle,load_bundle,verify_bundle
 source=model('2')
 with torch.no_grad():
  site=source.layers[2].mlp.residuals;getattr(site['gate_proj'],zero_factor).zero_()
  source.layers[0].mlp.residuals['up_proj'].B.fill_(1e-30)
 ref=write_bundle(source,tmp_path/'elided',runtime(tmp_path),elide_zero_residuals=True)
 raw=verify_bundle(Path(ref.directory)).metadata
 assert raw['factor_presence'][2]['gate_proj'] is False
 assert raw['factor_presence'][2]['up_proj'] is True
 assert raw['factor_presence'][0]['up_proj'] is True
 loaded=load_bundle(Path(ref.directory),backend='torch')
 torch.testing.assert_close(loaded.logits(batch()),source.logits(batch()),rtol=0,atol=0)

@pytest.mark.parametrize('attack',['missing_site','extra_site','nonbool','owner_missing','deployment_digest','not_frozen'])
def test_schema2_rejects_presence_and_identity_attacks(tmp_path,attack):
 from poc.bundle import write_bundle,verify_bundle,load_bundle
 ref=write_bundle(model('2'),tmp_path/'elided',runtime(tmp_path),elide_zero_residuals=True)
 root=Path(ref.directory);raw=json.loads((root/'manifest.json').read_text())
 if attack=='missing_site':raw['factor_presence'].pop()
 if attack=='extra_site':raw['factor_presence'][0]['other']=False
 if attack=='nonbool':raw['factor_presence'][0]['gate_proj']=0
 if attack=='owner_missing':raw['factor_presence'][2]['gate_proj']=False
 if attack=='deployment_digest':raw['deployment_checkpoint_digest']='0'*64
 if attack=='not_frozen':raw['owners'][0]['trainable']=True
 (root/'manifest.json').write_text(json.dumps(raw))
 with pytest.raises(ValueError):load_bundle(root,backend='torch')

def test_nonfinite_zero_product_rejected_and_dense_default_unchanged(tmp_path):
 from poc.bundle import write_bundle,verify_bundle
 source=model('2')
 with torch.no_grad():source.layers[0].mlp.residuals['gate_proj'].A[0,0]=float('nan')
 with pytest.raises(ValueError,match='nonfinite'):write_bundle(source,tmp_path/'bad',runtime(tmp_path),elide_zero_residuals=True)
 dense=model('dense');rt=runtime(tmp_path)
 with pytest.raises(ValueError):write_bundle(dense,tmp_path/'dense-elision',rt,elide_zero_residuals=True)
 ref=write_bundle(dense,tmp_path/'dense',rt);assert verify_bundle(Path(ref.directory)).metadata['schema_version']==1


def test_elided_bundle_fresh_process_and_explicit_reexport(tmp_path):
 from poc.bundle import write_bundle,load_bundle
 from test_offline_bundle import run_offline_bundle
 rt=runtime(tmp_path);ref=write_bundle(model('2'),tmp_path/'first',rt,elide_zero_residuals=True)
 assert run_offline_bundle(ref.directory,tmp_path)['download_denied']
 loaded=load_bundle(Path(ref.directory),backend='torch')
 again=write_bundle(loaded,tmp_path/'again',rt,elide_zero_residuals=True)
 torch.testing.assert_close(load_bundle(Path(again.directory),backend='torch').logits(batch()),loaded.logits(batch()),rtol=0,atol=0)


def test_floating_verifier_rejects_incorrect_deployment_digest(tmp_path):
 from poc.bundle import write_bundle,verify_bundle
 ref=write_bundle(model('2'),tmp_path/'elided',runtime(tmp_path),elide_zero_residuals=True)
 root=Path(ref.directory);raw=json.loads((root/'manifest.json').read_text());raw['deployment_checkpoint_digest']='0'*64
 (root/'manifest.json').write_text(json.dumps(raw))
 with pytest.raises(ValueError,match='deployment checkpoint digest'):verify_bundle(root)


@pytest.mark.parametrize('entry',['configure','checkpoint_schema','bank_state'])
def test_inference_elision_cannot_silently_reenter_training_checkpoint_flow(tmp_path,entry):
 from poc.bundle import write_bundle,load_bundle,write_bank_state
 from poc.train import TrainingConfig,configure_trainable_owners,_model_schema
 ref=write_bundle(model('2'),tmp_path/'elided',runtime(tmp_path),elide_zero_residuals=True)
 loaded=load_bundle(Path(ref.directory),backend='torch');before=loaded.manifest().checkpoint_digest
 with pytest.raises(ValueError,match='inference.only'):
  if entry=='configure':configure_trainable_owners(loaded,TrainingConfig(max_updates=1))
  elif entry=='checkpoint_schema':_model_schema(loaded)
  else:write_bank_state(loaded,tmp_path/'scientific-state')
 assert loaded.manifest().checkpoint_digest==before and all(not p.requires_grad for p in loaded.parameters())
 assert not (tmp_path/'scientific-state').exists()

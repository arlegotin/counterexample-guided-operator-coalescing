"""Canonical shared GPTQ export; owned CPU models and explicit native controls."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import importlib,importlib.util,hashlib
import pytest,torch


def api():
 assert importlib.util.find_spec('poc.shared_quantization'),'canonical shared quantization adapter missing'
 return importlib.import_module('poc.shared_quantization')


@pytest.fixture
def case(tmp_path):
 from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
 from poc.bank import make_bank,canonicalize_partition,PROJECTIONS
 from poc.bundle import write_bundle,load_bundle,verify_bundle
 from poc.contracts import Batch,thaw
 from poc.data.manifest import DataManifest,DocumentRecord
 from poc.shared_calibration import SharedCalibrationSpec,collect_shared_inputs,window_plan_sha256,_model_sha
 from poc.shared_calibration_bundle import save_calibration
 from test_export import runtime
 dense=make_tiny_decoder(TinyConfig(vocab_size=41,hidden_size=64,intermediate_size=128,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024),seed=3,dtype=torch.float32);dense.tokenizer_digest='a'*64
 layout=make_bank(dense,canonicalize_partition([[0,2],[1]],3),1,17)
 with torch.no_grad():
  for depth in (0,2):
   for projection in PROJECTIONS:
    f=layout.layers[depth].mlp.residuals[projection];f.A.fill_(.03);f.B.fill_(.02)
 source=write_bundle(layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True);model=load_bundle(Path(source.directory),backend='torch')
 doc=DocumentRecord('doc','owned','b'*40,'owned',0,text='owned');population=DataManifest((doc,),{'doc':'fit'},{'doc':'component'},'owned')
 ids=torch.arange(512)[None]%41;batch=Batch(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,511),dtype=torch.bool),torch.arange(512)[None],('doc',),('component',))
 sha=_model_sha(model);spec=SharedCalibrationSpec(layout_sha256=sha,source_sha256=sha,population_sha256=population.digest(),window_plan_sha256=window_plan_sha256((batch,)),tokenizer_sha256='a'*64)
 stats=collect_shared_inputs(model,model,population,(batch,),spec);stats_ref=save_calibration(stats,tmp_path/'statistics',expected_result_sha256=stats.digest(),study_contract_sha256='c'*64)
 return SimpleNamespace(source=source,population=population,stats=stats,stats_ref=stats_ref,metadata=thaw(verify_bundle(Path(source.directory)).metadata),model=model)


def call(m,c,destination,**changes):
 from poc.shared_gptq import SharedGPTQSpec
 values={'bundle':c.source,'stats':c.stats_ref,'spec':SharedGPTQSpec(bits=4),'destination':destination,'fallback_bits':4,'embedding_bits':4,'calibration':c.population,'calibration_contract_sha256':'c'*64,'recorder':None};values.update(changes)
 return m.quantize_shared_gptq(**values)


@pytest.mark.parametrize('bad',['role','stats_hash','contract','population','source_hash','fallback','embedding','existing'])
def test_invalid_or_owned_inputs_fail_before_projection_quantization(case,tmp_path,monkeypatch,bad):
 from poc.contracts import BundleRef
 import poc.shared_gptq
 m=api();changes={};dest=tmp_path/'packed'
 if bad=='role':changes['calibration']=replace(case.population,roles={'doc':'dev'})
 elif bad=='stats_hash':changes['stats']=replace(case.stats_ref,manifest_sha256='f'*64)
 elif bad=='contract':changes['calibration_contract_sha256']='d'*64
 elif bad=='population':changes['calibration']=replace(case.population,salt='different')
 elif bad=='source_hash':changes['bundle']=BundleRef(case.source.directory,'f'*64)
 elif bad=='fallback':changes['fallback_bits']=3
 elif bad=='embedding':changes['embedding_bits']=True
 else:dest.mkdir();(dest/'keep').write_text('retain')
 monkeypatch.setattr(poc.shared_gptq,'quantize_projection',lambda *a,**k:pytest.fail('invalid input must precede quantization'))
 with pytest.raises((ValueError,FileExistsError)):call(m,case,dest,**changes)
 if bad=='existing':assert (dest/'keep').read_text()=='retain'
 else:assert not dest.exists()


@pytest.mark.mlx
def test_packed_export_keeps_one_bank_and_matches_actual_dequantized_model(case,tmp_path):
 import mlx.core as mx
 import numpy as np
 from poc.bundle import load_bundle,verify_bundle,measure_bundle
 from test_model_parity import batch
 from test_native_parity import native_batch
 m=api();before={str(p.relative_to(case.source.directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(case.source.directory).rglob('*') if p.is_file()}
 result=call(m,case,tmp_path/'packed');root=Path(result.bundle.directory);manifest=verify_bundle(root);native=load_bundle(root,backend='mlx')
 assert result.method=='gptq' and result.calibration_sha256==case.population.digest()
 assert result.payload_bytes==measure_bundle(root).payload_bytes==sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
 assert len([o for o in manifest.metadata['owners'] if o['category']=='bank'])==6
 assert manifest.metadata['groups']==((0,2),(1,)) and native.owner('embed_tokens.weight') is native.owner('lm_head.weight')
 assert all(o['dtype']=='bfloat16' for o in manifest.metadata['owners'] if o['category']=='residual')
 values={}
 for file in {o['file'] for o in manifest.metadata['owners']}:values.update(mx.load(str(root/file)))
 expected=load_bundle(Path(case.source.directory),backend='torch');parameters=dict(expected.named_parameters(remove_duplicate=False));formats=__import__('json').loads((root/'quantization.json').read_text())['owners']
 with torch.no_grad():
  for owner in manifest.metadata['owners']:
   name=owner['name'];value=mx.dequantize(values[name+'.packed'],values[name+'.scales'],values[name+'.biases'],bits=formats[name]['bits'],group_size=formats[name]['group_size']) if name in formats else values[name].astype(mx.float32)
   parameters[name].copy_(torch.from_numpy(np.array(value,copy=True)))
 b=batch([[1,4,7,2,3]]);actual=native.logits(native_batch(b));mx.eval(actual)
 torch.testing.assert_close(torch.from_numpy(np.array(actual,copy=True)),expected.logits(b),atol=2e-4,rtol=0)
 assert before=={str(p.relative_to(case.source.directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(case.source.directory).rglob('*') if p.is_file()}
 from test_offline_bundle import run_offline_bundle
 assert run_offline_bundle(root,tmp_path,backend='mlx')['download_denied']


@pytest.mark.mlx
def test_failed_shared_export_retains_partial_pack_without_accepted_destination(case,tmp_path,monkeypatch):
 import mlx.core as mx
 m=api();original=mx.save_safetensors;calls=[]
 def fail(path,tensors,*a,**kw):
  original(path,tensors,*a,**kw);calls.append(str(path));raise InterruptedError('owned interrupted packing')
 monkeypatch.setattr(mx,'save_safetensors',fail);dest=tmp_path/'packed'
 with pytest.raises(InterruptedError):call(m,case,dest)
 assert calls and not dest.exists();failed=list(tmp_path.glob('packed.failed-*'))
 assert len(failed)==1 and (failed[0]/'failure.json').is_file()
 assert any(p.suffix=='.safetensors' for p in failed[0].rglob('*'))


@pytest.mark.mlx
def test_dense_opt_in_accounts_both_native_controls_and_raw_contexts(tmp_path):
 from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
 from poc.bundle import write_bundle
 from poc.data.manifest import DataManifest,DocumentRecord
 from poc.quantize import quantize_dense,QuantizationSpec
 from poc.contracts import CostRecord
 from test_export import runtime
 model=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2),seed=19,dtype=torch.float32)
 source=write_bundle(model,tmp_path/'source',runtime(tmp_path));records=tuple(DocumentRecord(str(i),'owned','r','s',i,text=' '.join(['t1','t4','t7','t2']*6)) for i in range(2))
 pop=DataManifest(records,{str(i):'fit' for i in range(2)},{str(i):str(i) for i in range(2)},'s',str(tmp_path/'exposure.jsonl'))
 class Recorder:
  def __init__(self):self.rows=[]
  def check(self,c):pass
  def record(self,c):self.rows.append(c)
 rec=Recorder();result=quantize_dense(source,QuantizationSpec('gptq',4,64,str(tmp_path/'packed'),calibration_count=2,length=16),pop,rec,accounting_profile='raw-contexts-v1')
 assert result.method=='gptq'
 assert sum(c.teacher_forward_calls for c in rec.rows)==4
 assert sum(c.student_forward_calls for c in rec.rows)==2
 assert sum(c.raw_corpus_target_presentations for c in rec.rows)==90


def test_existing_sidecar_fails_before_projection_and_publication(case,tmp_path,monkeypatch):
 import poc.shared_gptq
 m=api();dest=tmp_path/'packed';sidecar=Path(str(dest)+'.gptq-evidence.json');sidecar.write_text('retain')
 monkeypatch.setattr(poc.shared_gptq,'quantize_projection',lambda *a,**k:pytest.fail('existing evidence must fail before quantization'))
 with pytest.raises(FileExistsError):call(m,case,dest)
 assert not dest.exists() and sidecar.read_text()=='retain'


@pytest.mark.mlx
def test_postpublication_evidence_failure_quarantines_bundle(case,tmp_path,monkeypatch):
 m=api();dest=tmp_path/'packed';sidecar=Path(str(dest)+'.gptq-evidence.json');original=Path.open
 def interrupted(self,*a,**kw):
  if self==sidecar and a and a[0]=='xb':raise InterruptedError('owned evidence interruption')
  return original(self,*a,**kw)
 monkeypatch.setattr(Path,'open',interrupted)
 with pytest.raises(InterruptedError):call(m,case,dest)
 assert not dest.exists();failed=list(tmp_path.glob('packed.failed-*'));assert len(failed)==1 and (failed[0]/'failure.json').is_file()


@pytest.mark.mlx
def test_postpublication_final_cost_failure_quarantines_bundle(case,tmp_path):
 m=api();dest=tmp_path/'packed'
 class Recorder:
  def check(self,c):pass
  def record(self,c):
   if c.wall_seconds:raise InterruptedError('owned terminal accounting interruption')
 with pytest.raises(InterruptedError):call(m,case,dest,recorder=Recorder())
 assert not dest.exists();assert len(list(tmp_path.glob('packed.failed-*')))==1

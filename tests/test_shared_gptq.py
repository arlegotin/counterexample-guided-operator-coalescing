"""Moment-bound quantization controls; pretrained/native work is separate."""
from pathlib import Path
from dataclasses import replace,FrozenInstanceError
from types import SimpleNamespace
import importlib,importlib.util
import pytest,torch
from test_shared_calibration import owned,bind


def api():
 assert importlib.util.find_spec('poc.shared_gptq'),'moment-bound shared GPTQ module missing'
 return importlib.import_module('poc.shared_gptq')


def test_spec_closed_immutable_and_identity_bound():
 m=api();spec=m.SharedGPTQSpec(bits=4)
 assert spec.group_size==64 and spec.damp_fraction==.01 and spec.numerical_profile=='shared-gptq-fp32-v1'
 assert m.SharedGPTQSpec.from_mapping(spec.identity())==spec and len(spec.digest())==64
 with pytest.raises(FrozenInstanceError):spec.bits=2
 with pytest.raises((ValueError,TypeError)):m.SharedGPTQSpec.from_mapping(dict(spec.identity(),extra=True))
 assert replace(spec,bits=2).digest()!=spec.digest()


@pytest.mark.parametrize('change',[{'bits':3},{'bits':True},{'group_size':32},{'group_size':True},{'damp_fraction':0},{'damp_fraction':float('nan')},{'damp_fraction':float('inf')},{'damp_fraction':True},{'damp_fraction':2.},{'numerical_profile':'unqualified'},{'schema_version':True},{'schema_version':2}])
def test_spec_refuses_unqualified_numerical_or_format_options(change):
 m=api()
 with pytest.raises(ValueError):m.SharedGPTQSpec(**dict({'bits':4},**change))


@pytest.fixture
def bound(owned,tmp_path):
 from poc.bundle import write_bundle,load_bundle,verify_bundle
 from poc.contracts import thaw
 from poc.shared_calibration import collect_shared_inputs
 from test_export import runtime
 ref=write_bundle(owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True)
 model=load_bundle(Path(ref.directory),backend='torch');source=SimpleNamespace(layout=model,teacher=owned.teacher,population=owned.population,windows=owned.windows)
 result=collect_shared_inputs(model,model,source.population,source.windows,bind(source))
 return source,thaw(verify_bundle(Path(ref.directory)).metadata),result


def test_every_bank_uses_all_declared_depths_and_true_input_width_without_forward(bound,monkeypatch):
 source,metadata,result=bound;m=api();before=result.digest();model_sha=source.layout.manifest().checkpoint_digest
 monkeypatch.setattr(source.layout,'final_hidden',lambda *a:pytest.fail('validation cannot run model'))
 packet=m.validate_shared_moments(metadata,result,source.population)
 assert len(packet['owners'])==9 and packet['owners']['bank.0.down_proj.weight']['shape']==[8,12]
 assert packet['owners']['bank.0.down_proj.weight']['use_sites']==[0,2] and packet['owners']['bank.0.down_proj.weight']['count']==2044
 assert packet['owners']['bank.1.gate_proj.weight']['count']==1022 and packet['raw_target_presentations']==1022
 assert packet['source_cpu_sha256']==metadata['deployment_checkpoint_digest']
 assert packet['statistics_sha256']==before==result.digest() and source.layout.manifest().checkpoint_digest==model_sha


@pytest.mark.parametrize('bad',['source','kind','dtype','groups','owner_shape','role','quarantine','population','nonfinite','stat_dtype','counts','teacher_inputs'])
def test_wrong_source_role_or_mutated_moments_fail_before_quantization(bound,bad):
 source,metadata,result=bound;m=api();population=source.population
 if bad=='source':metadata['deployment_checkpoint_digest']='f'*64
 elif bad=='kind':metadata['kind']='dense'
 elif bad=='dtype':metadata['dtype']='float64'
 elif bad=='groups':metadata['groups']=[[0,3],[1],[2]]
 elif bad=='owner_shape':next(o for o in metadata['owners'] if o['category']=='bank')['shape']=[1,1]
 elif bad=='role':population=replace(population,roles={k:'dev' for k in population.roles})
 elif bad=='quarantine':population=replace(population,quarantine={population.records[0].document_id:'owned'})
 elif bad=='population':population=replace(population,salt='different')
 elif bad=='nonfinite':next(iter(result.owners.values())).moment_sum[0,0]=float('nan')
 elif bad=='stat_dtype':next(iter(result.owners.values())).moment_sum=next(iter(result.owners.values())).moment_sum.float()
 elif bad=='counts':next(iter(result.owners.values())).count-=1
 else:
  changed=replace(result.spec,source='teacher');result=replace(result,spec=changed,spec_sha256=changed.digest())
 with pytest.raises(ValueError):m.validate_shared_moments(metadata,result,population)


@pytest.mark.parametrize('bad',['weight_dtype','moment_dtype','shape','width','nonfinite_w','nonfinite_h','asymmetric','negative_diagonal'])
def test_invalid_projection_is_denied_before_native_dispatch(monkeypatch,bad):
 m=api();w=torch.arange(128,dtype=torch.float32).reshape(2,64);h=torch.eye(64,dtype=torch.float64)
 if bad=='weight_dtype':w=w.double()
 elif bad=='moment_dtype':h=h.float()
 elif bad=='shape':h=h[:-1]
 elif bad=='width':w=w[:,:63];h=h[:63,:63]
 elif bad=='nonfinite_w':w[0,0]=float('nan')
 elif bad=='nonfinite_h':h[0,0]=float('inf')
 elif bad=='asymmetric':h[0,1]=1
 else:h[0,0]=-1
 monkeypatch.setattr(m,'_quantize_projection_mlx',lambda *a:pytest.fail('invalid projection must not initialize native quantizer'),raising=False)
 with pytest.raises(ValueError):m.quantize_projection(w,h,m.SharedGPTQSpec(bits=2))


@pytest.mark.mlx
def test_exact_grid_packing_and_actual_native_math_preserve_inputs():
 import mlx.core as mx
 m=api();w=(torch.arange(128).reshape(2,64)%4).float();h=512*torch.eye(64,dtype=torch.float64);before_w=w.clone();before_h=h.clone();q=m.quantize_projection(w,h,m.SharedGPTQSpec(bits=2))
 restored=mx.dequantize(q.packed,q.scales,q.biases,group_size=64,bits=2);mx.eval(restored)
 assert torch.equal(torch.from_numpy(__import__('numpy').array(restored,copy=True)),w)
 assert tuple(q.packed.shape)==(2,4) and tuple(q.scales.shape)==(2,1)
 x=(torch.arange(192).reshape(3,64)%2).float();native=mx.quantized_matmul(mx.array(x.numpy()),q.packed,q.scales,q.biases,transpose=True,group_size=64,bits=2);mx.eval(native)
 assert torch.equal(torch.from_numpy(__import__('numpy').array(native,copy=True)),x@w.T)
 assert torch.equal(w,before_w) and torch.equal(h,before_h) and q.diagnostics['fallback_reason'] is None


@pytest.mark.mlx
def test_correlated_column_compensation_improves_planted_linear_reconstruction():
 import mlx.core as mx
 import numpy as np
 m=api();w=(torch.arange(64)%2).float()[None];w[0,:2]=.5;h=torch.eye(64,dtype=torch.float64);h[0,1]=h[1,0]=.99
 packed,scales,biases=mx.quantize(mx.array(w.numpy()),group_size=64,bits=2);rtn=torch.from_numpy(np.array(mx.dequantize(packed,scales,biases,group_size=64,bits=2),copy=True)).double()
 q=m.quantize_projection(w,h,m.SharedGPTQSpec(bits=2));actual=torch.from_numpy(np.array(mx.dequantize(q.packed,q.scales,q.biases,group_size=64,bits=2),copy=True)).double()
 error=lambda value:float(((w.double()-value)@h*(w.double()-value)).sum())
 assert error(actual)<error(rtn)*.1
 assert q.diagnostics['columns']==64 and q.diagnostics['groups']==1


@pytest.mark.mlx
def test_zero_energy_fallback_is_explicit_and_native_equivalent():
 import mlx.core as mx
 import numpy as np
 m=api();w=torch.linspace(-1,1,256).reshape(2,128);h=torch.zeros((128,128),dtype=torch.float64);q=m.quantize_projection(w,h,m.SharedGPTQSpec(bits=4));r=mx.quantize(mx.array(w.numpy()),group_size=64,bits=4)
 assert q.diagnostics['fallback_reason']=='zero_calibration_energy'
 assert all(np.array_equal(np.array(a),np.array(b)) for a,b in zip((q.packed,q.scales,q.biases),r))


@pytest.mark.mlx
def test_multigroup_projection_matches_pinned_upstream_gptq_on_same_fp32_hessian():
 import mlx.core as mx
 import mlx.nn as nn
 import numpy as np
 from mlx_lm.quant.gptq import gptq_quantize
 m=api();g=torch.Generator().manual_seed(29);w=torch.randn(4,128,generator=g)*.07;x=torch.randn(1,256,128,generator=g)
 native_x=mx.array(x.numpy());flat=mx.reshape(native_x,(-1,128));gram=flat.T@flat;mx.eval(gram)
 h=torch.from_numpy(np.array(gram,copy=True)).double()
 class OwnedLinear(nn.Module):
  def __init__(self):super().__init__();self.proj=nn.Linear(128,4,bias=False);self.proj.weight=mx.array(w.numpy())
  def __call__(self,inputs):return self.proj(inputs)
 reference,_=gptq_quantize(OwnedLinear(),native_x,4,64,4,64,batch_size=1)
 actual=m.quantize_projection(w,h,m.SharedGPTQSpec(bits=4));mx.eval(reference.parameters())
 assert np.array_equal(np.array(actual.packed),np.array(reference.proj.weight))
 assert np.array_equal(np.array(actual.scales),np.array(reference.proj.scales))
 assert np.array_equal(np.array(actual.biases),np.array(reference.proj.biases))


def test_bound_moment_packet_is_canonical_json_without_mutating_raw_counts(bound,tmp_path):
 from poc.records import canonical_json
 import json
 source,metadata,result=bound;before=result.digest();raw={key:dict(owner.use_counts) for key,owner in result.owners.items()}
 packet=api().validate_shared_moments(metadata,result,source.population)
 # The real runner persists this binding through canonical_json, whose keys
 # must be strings; ordinary json.dumps alone silently hid the interface bug.
 payload=canonical_json({'value':{'binding':packet}});saved=json.loads(payload)
 assert saved['value']['binding']['owners']['bank.0.gate_proj.weight']['use_counts']=={'0':1022,'2':1022}
 assert result.digest()==before and {key:dict(owner.use_counts) for key,owner in result.owners.items()}==raw

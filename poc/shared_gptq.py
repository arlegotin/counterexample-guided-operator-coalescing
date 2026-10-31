"""Versioned current-student moments for canonical-bank GPTQ.

Quantizer kernels and real-study admission are separate. Importing this module
requires no MLX device, model load, source document access, or quantization.
"""
from dataclasses import asdict,dataclass
import hashlib,math
from poc.records import canonical_json
from poc.contracts import thaw


@dataclass(frozen=True)
class SharedGPTQSpec:
 bits:int
 group_size:int=64
 damp_fraction:float=.01
 numerical_profile:str='shared-gptq-fp32-v1'
 schema_version:int=1
 def __post_init__(self):
  if (type(self.schema_version)is not int or self.schema_version!=1 or type(self.bits)is not int or self.bits not in (2,4,8)
      or type(self.group_size)is not int or self.group_size!=64
      or type(self.damp_fraction)not in (int,float) or not math.isfinite(self.damp_fraction) or not 0<self.damp_fraction<=1
      or self.numerical_profile!='shared-gptq-fp32-v1'):raise ValueError('qualified finite shared GPTQ format/numerical profile required')
 def identity(self):return asdict(self)
 def digest(self):return hashlib.sha256(canonical_json(self.identity())).hexdigest()
 @classmethod
 def from_mapping(cls,value):
  if set(value)!={'bits','group_size','damp_fraction','numerical_profile','schema_version'}:raise ValueError('closed GPTQ specification required')
  return cls(**value)


def validate_shared_moments(source_metadata,result,population):
 """Freshly bind exact source, fit roles, canonical owners and all use sites."""
 from poc.data.manifest import DataManifest
 from poc.shared_calibration_bundle import _validate
 from poc.runtime.torch_decoder import EXECUTION_PROFILE
 metadata=thaw(source_metadata)
 if not isinstance(population,DataManifest) or not population.records or any(role!='fit' for role in population.roles.values()) or population.quarantine:raise ValueError('unquarantined fit-only calibration population required')
 identity,spec=_validate(result)
 if (metadata.get('schema_version')!=2 or metadata.get('kind')!='compact' or metadata.get('dtype')!='float32'
     or metadata.get('execution_profile')!=EXECUTION_PROFILE or spec.source!='current_student'
     or spec.source_sha256!=metadata.get('deployment_checkpoint_digest') or spec.layout_sha256!=spec.source_sha256
     or spec.population_sha256!=population.digest() or spec.tokenizer_sha256!=metadata.get('tokenizer_digest')):
  raise ValueError('statistics/source/current-student/population identity differs')
 groups=metadata['groups'];depth=metadata['config']['num_hidden_layers']
 if sorted(d for group in groups for d in group)!=list(range(depth)):raise ValueError('source groups must partition all logical depths')
 expected={}
 for bank,uses in enumerate(groups):
  for projection in ('gate_proj','up_proj','down_proj'):
   key=f'bank.{bank}.{projection}.weight';owner=next((o for o in metadata['owners'] if o['name']==key),None)
   if owner is None or owner['category']!='bank' or owner['dtype']!='float32':raise ValueError('missing canonical floating source bank owner')
   wanted_sites=[f'layers.{d}.mlp.{projection}.weight' for d in uses]
   if owner['use_sites']!=wanted_sites or result.owner_layout.get(key)!={'use_sites':uses,'shape':owner['shape']}:raise ValueError('bank input shape or use-site layout differs')
   expected[key]={'shape':list(owner['shape']),'use_sites':list(uses),'count':result.owners[key].count,
                  'use_counts':{str(depth):count for depth,count in result.owners[key].use_counts.items()}}
 if set(expected)!=set(result.owners) or result.counters['unique_contexts']>len(population.records):raise ValueError('statistics omit/add canonical owners or source components')
 return {'schema_version':1,'source_cpu_sha256':spec.source_sha256,'statistics_sha256':result.digest(),
         'population_sha256':population.digest(),'calibration_spec_sha256':spec.digest(),
         'raw_target_presentations':result.counters['raw_target_presentations'],'owners':expected}


@dataclass(frozen=True)
class PackedProjection:
 packed:object
 scales:object
 biases:object
 shape:tuple[int,int]
 spec:SharedGPTQSpec
 diagnostics:dict


def quantize_projection(weight,moment_sum,spec):
 """Quantize one source bank matrix without mutating caller-owned tensors."""
 import torch
 import numpy as np
 if not isinstance(spec,SharedGPTQSpec):raise ValueError('typed shared GPTQ specification required')
 if (not isinstance(weight,torch.Tensor) or weight.device.type!='cpu' or weight.dtype!=torch.float32
     or weight.ndim!=2 or min(weight.shape)<=0 or weight.shape[1]%spec.group_size
     or not isinstance(moment_sum,torch.Tensor) or moment_sum.device.type!='cpu' or moment_sum.dtype!=torch.float64
     or tuple(moment_sum.shape)!=(weight.shape[1],weight.shape[1])
     or not torch.isfinite(weight).all().item() or not torch.isfinite(moment_sum).all().item()
     or (moment_sum.diag()<0).any().item() or not torch.allclose(moment_sum,moment_sum.T,rtol=1e-10,atol=1e-10)):
  raise ValueError('finite CPU FP32 weights/FP64 symmetric input moments with supported dimensions required')
 if not moment_sum.diag().sum().item() and torch.count_nonzero(moment_sum).item():raise ValueError('zero-energy moment matrix has nonzero correlations')
 private_w=np.array(weight.detach().numpy(),dtype=np.float32,copy=True,order='C')
 private_h=np.array(moment_sum.detach().numpy(),dtype=np.float32,copy=True,order='C')
 if not np.isfinite(private_h).all():raise ValueError('moments overflow the qualified FP32 trajectory')
 return _quantize_projection_mlx(private_w,private_h,spec)


def _quantize_projection_mlx(weight,moments,spec):
 """GPTQ error compensation adapted from pinned MLX-LM0.32.2.

 Copyright (c)2025 Apple Inc. Original implementation distributed under MIT;
 see LICENSES/mlx-lm-MIT.txt. This extension consumes a preverified canonical
 bank moment matrix rather than wrapping/de-aliasing model layers.
 """
 import mlx.core as mx
 import numpy as np
 W=mx.array(weight,dtype=mx.float32);width=weight.shape[1]
 diagnostics={'schema_version':1,'algorithm':spec.numerical_profile,'columns':width,'groups':width//spec.group_size,
   'moment_cast':'verified CPU FP64 sum to private FP32 trajectory','inverse_dtype':'float32','fallback_reason':None,
   'source_weight_sha256':hashlib.sha256(weight.tobytes()).hexdigest(),'trajectory_moment_sha256':hashlib.sha256(moments.tobytes()).hexdigest(),
   'spec_sha256':spec.digest()}
 if not np.any(np.diag(moments)):
  packed,scales,biases=mx.quantize(W,bits=spec.bits,group_size=spec.group_size);mx.eval(packed,scales,biases)
  diagnostics.update(fallback_reason='zero_calibration_energy',damping=0.)
  return PackedProjection(packed,scales,biases,tuple(weight.shape),spec,diagnostics)
 try:
  with mx.stream(mx.cpu):
   H=mx.array(moments,dtype=mx.float32);damp=spec.damp_fraction*mx.mean(mx.diag(H))
   H=H+mx.eye(width,dtype=mx.float32)*damp
   lower=mx.linalg.cholesky(H);inverse=mx.linalg.cholesky_inv(lower);Hinv=mx.linalg.cholesky(inverse,upper=True)
  mx.eval(Hinv,damp)
 except Exception as exc:raise ValueError('damped input Hessian could not be factorized') from exc
 if not mx.all(mx.isfinite(Hinv)).item() or not mx.all(mx.diag(Hinv)>0).item():raise ValueError('damped inverse Cholesky is nonfinite or nonpositive')
 diagnostics['damping']=float(damp.item());all_scales=[];all_biases=[]
 @mx.compile
 def error(w,d,scales,biases):
  safe=mx.where(scales!=0,scales,mx.ones_like(scales));q=mx.clip(mx.round((w-biases)/safe),0.,float(2**spec.bits-1))
  return (w-(scales*q+biases))/d
 for start in range(0,width,spec.group_size):
  stop=start+spec.group_size;local=W[...,start:stop];errors=mx.zeros_like(local)
  _,scales,biases=mx.quantize(local,bits=spec.bits,group_size=spec.group_size)
  all_scales.append(scales);all_biases.append(biases)
  for offset in range(spec.group_size):
   column=start+offset;e=error(W[...,column:column+1],Hinv[column,column],scales,biases)
   W[...,column:stop]-=e@Hinv[column:column+1,column:stop];errors[...,offset:offset+1]=e;mx.eval(errors,W)
  W[...,stop:]-=errors@Hinv[start:stop,stop:]
 scales=mx.concatenate(all_scales,axis=-1);biases=mx.concatenate(all_biases,axis=-1)
 grouped=mx.reshape(W,(weight.shape[0],width//spec.group_size,spec.group_size));safe=mx.where(scales!=0,scales,mx.ones_like(scales))
 bins=mx.clip(mx.round((grouped-biases[...,None])/safe[...,None]),0.,float(2**spec.bits-1)).astype(mx.uint32)
 per_word=32//spec.bits;bins=mx.reshape(bins,(weight.shape[0],-1,per_word));shifts=mx.arange(0,32,spec.bits,dtype=mx.uint32)
 packed=mx.sum(mx.left_shift(bins,shifts),axis=-1).astype(mx.uint32);mx.eval(packed,scales,biases)
 if not mx.all(mx.isfinite(scales)).item() or not mx.all(mx.isfinite(biases)).item():raise ValueError('nonfinite affine output')
 return PackedProjection(packed,scales,biases,tuple(weight.shape),spec,diagnostics)

"""Canonical shared-bank GPTQ export from exact fit-only input moments.

The dense learned adapter keeps its original sharing denial. This separate
entrypoint preserves shared owners, fixed factor storage, and explicit fallback
precisions. It performs no raw-corpus read or model forward.
"""
from pathlib import Path
import time,json,hashlib


def quantize_shared_gptq(bundle,stats,spec,destination,*,fallback_bits,embedding_bits,calibration,
                         calibration_contract_sha256,recorder):
 from poc.data.manifest import DataManifest
 from poc.shared_gptq import SharedGPTQSpec
 if not isinstance(calibration,DataManifest) or not calibration.records or any(role!='fit' for role in calibration.roles.values()) or calibration.quarantine:raise ValueError('unquarantined fit-only population required before source access')
 if not isinstance(spec,SharedGPTQSpec) or any(type(bits)is not int or bits not in (2,4,8) for bits in (fallback_bits,embedding_bits)):raise ValueError('qualified shared/fallback precision required')
 target=Path(destination)
 if not target.is_absolute() or target.resolve()!=target or target.is_symlink():raise ValueError('canonical explicit shared quantization destination required')
 if target.exists():raise FileExistsError(target)
 if Path(str(target)+'.gptq-evidence.json').exists():raise FileExistsError('quantizer evidence already exists')
 started=time.monotonic()
 if recorder is not None:
  from poc.contracts import CostRecord
  recorder.check(CostRecord());recorder.record(CostRecord())
 published=None
 try:
  published=_quantize_shared_impl(bundle,stats,spec,target,fallback_bits,embedding_bits,calibration,calibration_contract_sha256,recorder);return published
 finally:
  if recorder is not None:
   from poc.contracts import CostRecord
   from poc.continuation_budget import process_peak_bytes
   import mlx.core as mx
   try:
    mx.synchronize();recorder.record(CostRecord(wall_seconds=time.monotonic()-started,memory_bytes=max(process_peak_bytes(),int(mx.get_peak_memory())),memory_unavailable_reason=None))
   except BaseException as exc:
    if published is not None and target.exists():
     from poc.bundle import _canonical
     failed=target.with_name(target.name+'.failed-'+__import__('uuid').uuid4().hex);target.rename(failed);evidence=Path(str(target)+'.gptq-evidence.json')
     if evidence.exists():evidence.rename(failed/'quantizer-evidence.unaccepted.json')
     (failed/'failure.json').write_bytes(_canonical({'status':'failed','error':repr(exc),'intended_output':str(target),'scope':'terminal accounting failed after packing; unaccepted payload retained'}))
    raise


def _quantize_shared_impl(bundle,stats,spec,target,fallback_bits,embedding_bits,population,contract_sha,recorder):
 from poc.shared_calibration_bundle import load_calibration
 from poc.shared_gptq import validate_shared_moments,quantize_projection
 from poc.quantize import _verified_ref,packed_layout,QuantizationSpec,_save_packed
 from poc.contracts import thaw
 from poc.bundle import load_bundle,_canonical
 import torch,numpy as np
 # Complete source verification may reconstruct CPU weights; the caller's
 # durable attempt/finite guard must already exist. Wrong roles fail earlier.
 result=load_calibration(stats,expected_contract_sha256=contract_sha);source,manifest=_verified_ref(bundle)
 binding=validate_shared_moments(thaw(manifest.metadata),result,population);stats_digest=result.digest()
 for owner in manifest.metadata['owners']:
  if len(owner['shape'])==2 and owner['category']!='residual':
   bits=spec.bits if owner['category']=='bank' else embedding_bits if owner['name']=='embed_tokens.weight' else fallback_bits
   packed_layout(owner['shape'],QuantizationSpec('rtn',bits,spec.group_size,str(target)))
 import mlx.core as mx
 native=load_bundle(source,backend='mlx');values={};formats={};diagnostics={}
 for owner in manifest.metadata['owners']:
  name=owner['name'];value=native.weights[name]
  if owner['category']=='bank':
   private=torch.from_numpy(np.array(value,dtype=np.float32,copy=True))
   quantized=quantize_projection(private,result.owners[name].moment_sum,spec)
   values[name]=(quantized.packed,quantized.scales,quantized.biases);formats[name]={'bits':spec.bits,'group_size':spec.group_size};diagnostics[name]=quantized.diagnostics
  elif len(owner['shape'])==2 and owner['category']!='residual':
   bits=embedding_bits if name=='embed_tokens.weight' else fallback_bits
   values[name]=mx.quantize(value,bits=bits,group_size=spec.group_size);formats[name]={'bits':bits,'group_size':spec.group_size};mx.eval(*values[name])
  else:values[name]=value
 if len(diagnostics)!=len(binding['owners']) or result.digest()!=stats_digest:raise ValueError('canonical bank count or calibration changed during quantization')
 exported=_save_packed(source,manifest,str(target),values,formats,'gptq',population.digest(),retain_failed=True)
 evidence={'schema_version':1,'method':'shared-gptq-fp32-v1','spec':spec.identity(),'spec_sha256':spec.digest(),
   'source':{'directory':bundle.directory,'manifest_sha256':bundle.manifest_sha256},'calibration_ref':{'directory':stats.directory,'manifest_sha256':stats.manifest_sha256},
   'calibration_contract_sha256':contract_sha,'binding':binding,'fallback_bits':fallback_bits,'embedding_bits':embedding_bits,
   'canonical_bank_quantizations':len(diagnostics),'bank_diagnostics':diagnostics,'packed_bundle':{'directory':exported.bundle.directory,'manifest_sha256':exported.bundle.manifest_sha256},
   'scope':'all-use fixed-source GPTQ bank reconstruction; attention/embedding RTN fallback, BF16 private factors; no raw-corpus read, model forward, optimizer update or downstream quality claim'}
 evidence_path=Path(str(target)+'.gptq-evidence.json');created=False
 try:
  with evidence_path.open('xb') as f:
   created=True;f.write(_canonical(evidence));f.flush();__import__('os').fsync(f.fileno())
 except BaseException as exc:
  failed=target.with_name(target.name+'.failed-'+__import__('uuid').uuid4().hex);target.rename(failed)
  if created and evidence_path.exists():evidence_path.rename(failed/'quantizer-evidence.pending.json')
  (failed/'failure.json').write_bytes(_canonical({'status':'failed','error':repr(exc),'intended_output':str(target),'scope':'post-publication quantizer evidence failed; bundle retained without accepted destination'}))
  raise
 return exported

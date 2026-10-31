"""Pinned native quantizers, fit-only calibration and complete packed export.

No convenience CLI or upstream corpus loader is used. Learned dense algorithms
operate on an official native module adapter and export its actual transformed
owners back through the common deployment inference boundary.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import socket
from typing import Sequence

from poc.contracts import BundleRef, CostRecord, Recorder
from poc.data.manifest import DataManifest


@dataclass(frozen=True)
class QuantizationSpec:
    method: str
    bits: int
    group_size: int
    destination: str
    calibration_count: int = 4
    length: int = 512
    seed: int = 17
    internal_validation_count: int = 2
    embedding_bits: int | None = None
    dynamic_high_bits: int = 8
    dynamic_target_bpw: float = 6.

    def __post_init__(self):
        if self.method not in ('rtn','gptq','awq','dwq','dynamic'): raise ValueError('unknown pinned quantizer')
        if self.bits not in (2,3,4,8) or self.group_size not in (32,64,128): raise ValueError('unsupported bit/group request')
        if self.embedding_bits is not None and self.embedding_bits not in (2,3,4,8): raise ValueError('unsupported embedding precision')
        if not self.destination or self.calibration_count<1 or self.length<2: raise ValueError('explicit output/calibration limits required')


@dataclass(frozen=True)
class FormatCapability:
    shape: tuple[int,int]
    bits: int
    group_size: int
    status: str
    reason: str
    packed_bytes: int | None = None
    scale_bytes: int | None = None
    bias_bytes: int | None = None
    padding_elements: int = 0
    inference_max_abs_error: float | None = None
    training_status: str = 'not_qualified'


def packed_layout(shape, spec):
    from poc.bundle import _packed_layout
    return _packed_layout(shape,spec.bits,spec.group_size)


@dataclass(frozen=True)
class CalibrationBatchSource:
    tokens: tuple[tuple[int,...],...]
    component_ids: tuple[str,...]
    manifest_sha256: str


@dataclass(frozen=True)
class QuantizationResult:
    bundle: BundleRef
    payload_bytes: int
    method: str
    calibration_sha256: str | None


def project_calibration(tokenizer, manifest: DataManifest, count: int, length: int, *, tokenization_policy=None) -> CalibrationBatchSource:
    if tokenization_policy not in (None,'no-special-tokens-v2'):raise ValueError('unknown calibration tokenization policy')
    from dataclasses import replace
    from poc.data.access import open_records, AccessDenied
    if count<1 or length<2: raise ValueError('positive calibration count and length>=2 required')
    if not manifest.records or any(r!='fit' for r in manifest.roles.values()): raise AccessDenied('calibration requires nonempty fit-only manifest')
    # Select component-disjoint records before I/O; never expose the whole pool.
    records=[]; components=set()
    for record in sorted(manifest.records,key=lambda r:r.document_id):
        component=manifest.membership[record.document_id]
        if component not in components:
            records.append(record);components.add(component)
        if len(records)==count: break
    if len(records)<count: raise ValueError('not enough disjoint fit components')
    ids={r.document_id for r in records}
    selected=replace(manifest,records=tuple(records),roles={i:manifest.roles[i] for i in ids},membership={i:manifest.membership[i] for i in ids},quarantine={i:v for i,v in manifest.quarantine.items() if i in ids})
    rows=[]
    for record in open_records(selected,purpose='calibration'):
        encoded=tokenizer.encode(record.text,add_special_tokens=False) if tokenization_policy=='no-special-tokens-v2' else tokenizer.encode(record.text)
        tokens=encoded.ids if hasattr(encoded,'ids') else encoded
        if len(tokens)<length: raise ValueError('calibration component shorter than requested context; preselect eligible components')
        rows.append(tuple(tokens[:length]))
    return CalibrationBatchSource(tuple(rows),tuple(selected.membership[r.document_id] for r in selected.records),selected.digest())


@contextmanager
def isolated_quantizer(module):
    """Deny defaults at the actual imported binding, plus all socket connects."""
    def denied(*args,**kwargs): raise PermissionError('upstream default calibration/network denied; explicit project fit manifest required')
    patches=[]
    for obj,names in ((module,('load_data','load_dataset','load','load_tokenizer','sharded_load')),
                      (socket,('create_connection',)),(socket.socket,('connect','connect_ex'))):
        for name in names:
            if hasattr(obj,name):
                patches.append((obj,name,getattr(obj,name)));setattr(obj,name,denied)
    try: yield
    finally:
        for obj,name,value in reversed(patches): setattr(obj,name,value)


def discover_quantization_capabilities(shapes: Sequence[tuple[int,int]]) -> list[FormatCapability]:
    import mlx.core as mx
    if not mx.metal.is_available(): raise RuntimeError('actual Metal qualification required')
    mx.set_default_device(mx.gpu)
    rows=[]
    for shape in sorted(set(map(tuple,shapes))):
        for bits in (2,3,4,8):
            for group in (32,64,128):
                try:
                    layout=packed_layout(shape,QuantizationSpec('rtn',bits,group,'unused'))
                    w=mx.sin(mx.arange(shape[0]*shape[1],dtype=mx.float32).reshape(shape)*.007)
                    p,s,b=mx.quantize(w,bits=bits,group_size=group);mx.eval(p,s,b)
                    x=mx.cos(mx.arange(shape[1],dtype=mx.float32))[None]
                    y=mx.quantized_matmul(x,p,s,b,transpose=True,bits=bits,group_size=group)
                    reference=x@mx.dequantize(p,s,b,bits=bits,group_size=group).T
                    error=mx.max(mx.abs(y-reference)).item()
                    if not bool(mx.all(mx.isfinite(y)).item()) or error>.01: raise ValueError('packed inference mismatch')
                    if p.nbytes!=layout['packed_bytes'] or s.nbytes!=layout['scale_bytes'] or b.nbytes!=layout['bias_bytes']: raise ValueError('actual packed accounting mismatch')
                    rows.append(FormatCapability(shape,bits,group,'supported','actual mx.quantize/quantized_matmul versus dequantized reference',p.nbytes,s.nbytes,b.nbytes,0,error))
                except (ValueError,RuntimeError) as exc:
                    rows.append(FormatCapability(shape,bits,group,'unsupported_kernel',str(exc)))
    return rows


def _verified_ref(bundle):
    from poc.bundle import _sha,verify_bundle
    path=Path(bundle.directory)
    if _sha(path/'manifest.json')!=bundle.manifest_sha256: raise ValueError('input BundleRef digest mismatch')
    return path,verify_bundle(path)


def _save_packed(source, manifest, destination, values, formats, method, calibration_digest, *, retain_failed=False):
    import mlx.core as mx
    import shutil,tempfile
    from poc.bundle import _canonical,_sha,_regular_files,verify_bundle,measure_bundle
    from poc.contracts import thaw
    if type(retain_failed)is not bool: raise ValueError('explicit export failure policy required')
    target=Path(destination)
    if target.exists(): raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.quantized-',dir=target.parent));published=False;failure=None
    try:
        shutil.copytree(source,temporary,dirs_exist_ok=True)
        from poc.bundle import _write_runtime
        shutil.rmtree(temporary/'runtime');_write_runtime(temporary)
        raw=thaw(manifest.metadata)
        for filename in sorted({o['file'] for o in raw['owners']}):
            tensors={}
            for owner in raw['owners']:
                if owner['file']!=filename: continue
                name=owner['name'];value=values[name]
                if name in formats:
                    p,s,b=value
                    tensors.update({name+'.packed':p,name+'.scales':s,name+'.biases':b})
                    owner['dtype']='uint32'
                else:
                    if owner['category']=='residual': value=value.astype(mx.bfloat16)
                    tensors[name]=value;owner['dtype']='bfloat16' if value.dtype==mx.bfloat16 else 'float32'
            mx.eval(*tensors.values());mx.save_safetensors(str(temporary/filename),tensors)
        (temporary/'quantization.json').write_bytes(_canonical({'schema_version':1,'format':'mlx-affine','method':method,'owners':formats,'calibration_manifest_sha256':calibration_digest}))
        raw['file_entries']={name:{'bytes':p.stat().st_size,'sha256':_sha(p)} for name,p in sorted(_regular_files(temporary).items()) if name!='manifest.json'}
        (temporary/'manifest.json').write_bytes(_canonical(raw));verify_bundle(temporary)
        digest=_sha(temporary/'manifest.json');temporary.rename(target);published=True
        ref=BundleRef(str(target.resolve()),digest)
        return QuantizationResult(ref,measure_bundle(target).payload_bytes,method,calibration_digest)
    except BaseException as exc:
        failure=repr(exc)
        if retain_failed and published and target.exists():
            failed=target.with_name(target.name+'.failed-'+__import__('uuid').uuid4().hex);target.rename(failed)
            (failed/'failure.json').write_bytes(_canonical({'status':'failed','error':failure,'intended_output':str(target),'source_manifest_sha256':_sha(source/'manifest.json'),'scope':'unaccepted post-publication export retained'}))
        raise
    finally:
        if temporary.exists():
            if retain_failed:
                failed=target.with_name(target.name+'.failed-'+__import__('uuid').uuid4().hex);temporary.rename(failed)
                (failed/'failure.json').write_bytes(_canonical({'status':'failed','error':failure or 'unpublished export cleanup','intended_output':str(target),'source_manifest_sha256':_sha(source/'manifest.json'),'scope':'failed partial export retained; not accepted or independently loadable'}))
            else:shutil.rmtree(temporary)


def _costed(function):
    from functools import wraps
    @wraps(function)
    def run(bundle,spec,recorder):
        import time,psutil
        start=time.monotonic()
        if recorder is not None: recorder.check(CostRecord())
        try: return function(bundle,spec,recorder)
        finally:
            if recorder is not None:
                recorder.record(CostRecord(wall_seconds=time.monotonic()-start,memory_bytes=psutil.Process().memory_info().rss,memory_unavailable_reason=None))
    return run


@_costed
def quantize_rtn(bundle: BundleRef, spec: QuantizationSpec, recorder) -> QuantizationResult:
    import mlx.core as mx
    from poc.bundle import load_bundle
    if spec.method!='rtn': raise ValueError('RTN method required')
    source,manifest=_verified_ref(bundle);native=load_bundle(source,backend='mlx')
    values={};formats={}
    for owner in manifest.metadata['owners']:
        name=owner['name'];value=native.weights[name]
        if len(owner['shape'])==2 and owner['category']!='residual':
            from dataclasses import replace
            bits=spec.embedding_bits if name=='embed_tokens.weight' and spec.embedding_bits is not None else spec.bits
            packed_layout(owner['shape'],replace(spec,bits=bits))
            values[name]=mx.quantize(value,bits=bits,group_size=spec.group_size)
            formats[name]={'bits':bits,'group_size':spec.group_size}
            mx.eval(*values[name])
        else: values[name]=value
    return _save_packed(source,manifest,spec.destination,values,formats,'rtn',None)



def dense_dispatch_units(method,phase,*,input_tokens,batch_size,context_tokens):
    """Opt-in accounting units; eligible distributions and shifted pairs differ.

    DWQ's actual whole-model input drops the final token of the original512
    context. It still compares511teacher distributions. These units describe
    each context view; the caller separately records scored losses and updates.
    No legacy dispatch/accounting behavior is changed by this pure helper.
    """
    if (method not in ('gptq','awq','dwq') or phase not in ('bridge','roundtrip','calibration','train','valid')
            or type(input_tokens) is not int or type(batch_size) is not int or batch_size<1
            or type(context_tokens) is not int or context_tokens!=512
            or phase in ('train','valid') and method!='dwq'
            or phase=='calibration' and method=='dwq'):
        raise ValueError('explicit dense method/phase/original512context required')
    expected=511 if phase in ('train','valid') else 512
    if input_tokens!=expected:raise ValueError('declared original context and actual model input differ')
    return {'processed_inputs':batch_size,'processed_tokens':batch_size*input_tokens,
            'raw_corpus_target_presentations':batch_size*(context_tokens-1),
            'distribution_targets':batch_size*(context_tokens-1),'model_shifted_pairs':batch_size*(input_tokens-1)}


def dense_native_cost(method,phase,*,input_tokens,batch_size,context_tokens,teacher):
    """Native bridge/target/roundtrip work; scored loss belongs to the student."""
    if type(teacher) is not bool:raise ValueError('explicit native teacher/student identity required')
    units=dense_dispatch_units(method,phase,input_tokens=input_tokens,batch_size=batch_size,context_tokens=context_tokens)
    return CostRecord(teacher_forward_calls=int(teacher),student_forward_calls=int(not teacher),
        processed_inputs=units['processed_inputs'],processed_tokens=units['processed_tokens'],
        raw_corpus_target_presentations=units['raw_corpus_target_presentations'])


def dense_optimizer_cost(profile,*,context_tokens,loss_target_count):
    """New forward precharges loss targets; Adam records only its own work."""
    if profile not in (None,'raw-contexts-v1','dense-native-contexts-v2'):raise ValueError('unknown dense accounting profile')
    if type(context_tokens) is not int or context_tokens<2:raise ValueError('complete source context required')
    if profile=='dense-native-contexts-v2':
        if context_tokens!=512 or type(loss_target_count) is not int or loss_target_count!=511:
            raise ValueError('actual full teacher-distribution loss target count required')
        return CostRecord(updates=1,student_backward_calls=1),loss_target_count
    return CostRecord(updates=1,student_backward_calls=1,valid_target_presentations=context_tokens-1),0

def _official_model(native, recorder=None, method=None, accounting_profile=None, *, accounting_context=None):
    new_accounting=accounting_profile=='dense-native-contexts-v2'
    if new_accounting and (method not in ('gptq','awq','dwq') or type(accounting_context) is not int or accounting_context!=512):
        raise ValueError('new dense accounting requires explicit512context and eligible fixedmethod')
    import mlx.core as mx
    from mlx_lm.models.llama import Model,ModelArgs
    from poc.contracts import thaw
    raw=thaw(native.metadata['config'])
    raw['rope_theta']=native.rope_theta
    raw['head_dim']=native.head_dim
    state={'quantized':False,'model':None,'forwards':0,'teacher_forwards':0,'backwards':0,'updates':0}
    if new_accounting:state.update(phase='bridge',forward_raw_corpus_presentations=0,scored_loss_targets=0,last_loss_target_count=0,logical_optimizer_raw_target_presentations=0,native_raw_corpus_presentations=0,native_teacher_forward_calls=0,native_student_forward_calls=0)
    class CountedModel(Model):
        def __call__(self,inputs,*args,**kwargs):
            teacher=self is state['model'] and not state['quantized']
            backward=method=='dynamic' and self is not state['model']
            if new_accounting:
                if len(inputs.shape)!=2:raise ValueError('complete rank2context input required')
                units=dense_dispatch_units(method,state['phase'],input_tokens=int(inputs.shape[1]),batch_size=int(inputs.shape[0]),context_tokens=accounting_context)
                scored=units['distribution_targets'] if method=='dwq' and state['phase'] in ('train','valid') else 0
                cost=CostRecord(teacher_forward_calls=int(teacher),student_forward_calls=int(not teacher),student_backward_calls=int(backward),
                    processed_inputs=units['processed_inputs'],processed_tokens=units['processed_tokens'],
                    raw_corpus_target_presentations=units['raw_corpus_target_presentations'],valid_target_presentations=scored)
            else:
                cost=CostRecord(teacher_forward_calls=int(teacher),student_forward_calls=int(not teacher),student_backward_calls=int(backward),processed_inputs=inputs.shape[0],processed_tokens=inputs.size,raw_corpus_target_presentations=(int(inputs.shape[0])*(int(inputs.shape[1])-1) if accounting_profile=='raw-contexts-v1' else 0))
            if recorder is not None:
                if new_accounting and method=='dwq' and state['phase']=='train':
                    from poc.budget import BudgetTracker
                    recorder.check(BudgetTracker.combine(cost,CostRecord(updates=1,student_backward_calls=1)))
                else:recorder.check(cost)
                recorder.record(cost)
            if new_accounting:
                state['forward_raw_corpus_presentations']+=units['raw_corpus_target_presentations']
                state['scored_loss_targets']+=scored;state['last_loss_target_count']=scored
            state['forwards']+=1;state['teacher_forwards']+=int(teacher);state['backwards']+=int(backward)
            return super().__call__(inputs,*args,**kwargs)
    model=CountedModel(ModelArgs(**{k:v for k,v in raw.items() if k in ModelArgs.__dataclass_fields__}))
    state['model']=model

    weights=[(('model.'+name) if name!='lm_head.weight' else name,value) for name,value in native.weights.items()]
    model.load_weights(weights,strict=True)
    model.model.norm.eps=native.norm_epsilons['norm']
    for i,layer in enumerate(model.layers):
        for site in ('input_layernorm','post_attention_layernorm'): getattr(layer,site).eps=native.norm_epsilons[f'layers.{i}.{site}']
    model.eval();mx.eval(model.parameters())
    return model,state


def _native_batch(tokens):
    import mlx.core as mx
    from poc.contracts import Batch
    ids=mx.array(tokens,dtype=mx.int32);b,s=ids.shape
    return Batch(ids,mx.ones_like(ids,dtype=mx.bool_),mx.ones((b,s-1),dtype=mx.bool_),mx.broadcast_to(mx.arange(s)[None],(b,s)),tuple(f'calibration-{i}' for i in range(b)),tuple(f'calibration-{i}' for i in range(b)))


def _quantize_dense_impl(bundle: BundleRef, spec: QuantizationSpec, calibration: DataManifest, recorder, accounting_profile=None) -> QuantizationResult:
    import importlib
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from poc.bundle import load_bundle
    from tokenizers import Tokenizer
    if spec.method=='rtn': return quantize_rtn(bundle,spec,recorder)
    source,manifest=_verified_ref(bundle)
    if manifest.metadata['kind']!='dense': raise ValueError('shared learned quantization adapter not implemented; no per-depth duplication allowed')
    new_accounting=accounting_profile=='dense-native-contexts-v2'
    native=load_bundle(source,backend='mlx');model,state=_official_model(native,recorder,spec.method,accounting_profile,accounting_context=spec.length if new_accounting else None)
    def native_charge(phase,batch_size,input_tokens,*,teacher):
        if new_accounting:
            cost=dense_native_cost(spec.method,phase,input_tokens=input_tokens,batch_size=batch_size,context_tokens=spec.length,teacher=teacher)
        elif phase in ('train','valid'):
            cost=CostRecord(teacher_forward_calls=1)
        elif phase=='roundtrip' and not accounting_profile:
            return
        else:
            cost=CostRecord(teacher_forward_calls=int(teacher),student_forward_calls=int(not teacher),
                processed_inputs=batch_size if accounting_profile else 0,processed_tokens=batch_size*input_tokens if accounting_profile else 0,
                raw_corpus_target_presentations=batch_size*(spec.length-1) if accounting_profile else 0)
        if recorder is not None:recorder.check(cost);recorder.record(cost)
        if new_accounting:
            state['native_raw_corpus_presentations']+=cost.raw_corpus_target_presentations
            state['native_teacher_forward_calls']+=cost.teacher_forward_calls;state['native_student_forward_calls']+=cost.student_forward_calls
    tokenizer=Tokenizer.from_file(str(source/'tokenizer/tokenizer.json'))
    count=spec.calibration_count+(spec.internal_validation_count if spec.method=='dwq' else 0)
    cal=project_calibration(tokenizer,calibration,count,spec.length,tokenization_policy='no-special-tokens-v2' if new_accounting else None)
    data=mx.array(cal.tokens[:spec.calibration_count],dtype=mx.int32)
    # Qualify the official architecture bridge before calling any optimizer.
    native_charge('bridge',1,spec.length,teacher=True)
    reference=native.logits(_native_batch([cal.tokens[0]]));actual=model(data[:1]);mx.eval(reference,actual)
    bridge_error=mx.max(mx.abs(reference-actual)).item()
    if bridge_error>.005: raise RuntimeError(f'official dense bridge unresolved fidelity: {bridge_error}')
    del reference,actual
    module=importlib.import_module('mlx_lm.quant.'+('dynamic_quant' if spec.method=='dynamic' else spec.method))
    mx.random.seed(spec.seed)
    if new_accounting:state['phase']='calibration'
    with isolated_quantizer(module):
        # Controls exercise the actual imported binding rather than patching only
        # its original utility module. No upstream default can run in this scope.
        try: module.load_data(None,None,None)
        except PermissionError: pass
        else: raise RuntimeError('default-loader denial control failed')
        if spec.method=='gptq':
            if spec.bits not in (2,4,8): raise ValueError('official GPTQ supports only 2/4/8 bits')
            model,_=module.gptq_quantize(model,data,spec.bits,spec.group_size,spec.embedding_bits or spec.bits,spec.group_size,batch_size=1)
        elif spec.method=='awq':
            module.awq_quantize(model,data,module.llama_awq,group_size=spec.group_size,bits=spec.bits,embed_group_size=spec.group_size,embed_bits=spec.embedding_bits or spec.bits,n_grid=20)
        elif spec.method=='dynamic':
            sensitivities=dict(module.estimate_sensitivities(model,data,low_bits=spec.bits,low_group_size=spec.group_size,high_bits=spec.dynamic_high_bits,high_group_size=spec.group_size,batch_size=1,gradient_accum_dtype=mx.float32,gradient_checkpoint=False))
            threshold=module.estimate_threshold(model,sensitivities,target_bpw=spec.dynamic_target_bpw,low_bits=spec.bits,low_group_size=spec.group_size,high_bits=spec.dynamic_high_bits,high_group_size=spec.group_size)
            from poc.bundle import _canonical
            allocation_trace={'schema_version':1,'source_manifest_sha256':bundle.manifest_sha256,'calibration_manifest_sha256':cal.manifest_sha256,'seed':spec.seed,'low_bits':spec.bits,'high_bits':spec.dynamic_high_bits,'group_size':spec.group_size,'target_effective_bpw':spec.dynamic_target_bpw,'sensitivities':sensitivities,'threshold':float(threshold),'packed_map_sha256':None}
            Path(spec.destination).parent.mkdir(parents=True,exist_ok=True)
            Path(spec.destination+'.allocation-trace.json').write_bytes(_canonical(allocation_trace))
            def predicate(path,layer):
                if not hasattr(layer,'to_quantized'): return False
                return {'bits':spec.dynamic_high_bits if sensitivities[path]>threshold else spec.bits,'group_size':spec.group_size}
            nn.quantize(model,bits=spec.bits,group_size=spec.group_size,class_predicate=predicate)
            state['quantized']=True
        elif spec.method=='dwq':
            import mlx.optimizers as optimizers
            nn.quantize(model,bits=spec.bits,group_size=spec.group_size,class_predicate=lambda path,leaf: {'bits':spec.embedding_bits if path=='model.embed_tokens' and spec.embedding_bits is not None else spec.bits,'group_size':spec.group_size} if hasattr(leaf,'to_quantized') else False)
            state['quantized']=True
            if new_accounting:state['dwq_trainability']=configure_dense_dwq_trainability(model,accounting_profile)
            train=[(list(t),0) for t in cal.tokens[:spec.calibration_count]]
            valid=[(list(t),0) for t in cal.tokens[spec.calibration_count:]]
            class CountedAdam(optimizers.Adam):
                def apply_gradients(self,grads,params):
                    if new_accounting:validate_dense_dwq_trainability(model,state['dwq_trainability'],params=params)
                    cost,logical=dense_optimizer_cost(accounting_profile,context_tokens=spec.length,loss_target_count=state.get('last_loss_target_count',0))
                    if recorder is not None: recorder.check(cost);recorder.record(cost)
                    state['updates']+=1;state['backwards']+=1
                    if new_accounting:state['logical_optimizer_raw_target_presentations']+=logical
                    return super().apply_gradients(grads,params)
            def target_fn(inputs,index,split):
                if new_accounting:state['phase']=split
                native_charge(split,int(inputs.shape[0]),int(inputs.shape[1]),teacher=True)
                return mx.stop_gradient(native.logits(_native_batch(inputs.tolist())))
            module.dwq_quantize(model,target_fn,CountedAdam(learning_rate=1e-5),train,valid,batch_size=1,max_seq_length=spec.length,seed=spec.seed,dtype=mx.float32,gradient_checkpoint=False,temperature=1.)
            if new_accounting:validate_dense_dwq_trainability(model,state['dwq_trainability'])
    state['quantized']=True
    # Extract every transformed float/norm and packed owner from the actual model.
    leaves=dict(tree_flatten(model.leaf_modules(),is_leaf=nn.Module.is_module))
    values={};formats={}
    for owner in manifest.metadata['owners']:
        name=owner['name'];path='model.'+name.rsplit('.',1)[0] if name!='lm_head.weight' else 'lm_head'
        leaf=leaves[path]
        if isinstance(leaf,(nn.QuantizedLinear,nn.QuantizedEmbedding)):
            if leaf.mode!='affine': raise ValueError('only actual affine native owners supported')
            values[name]=(leaf.weight,leaf.scales.astype(mx.float32),leaf.biases.astype(mx.float32))
            formats[name]={'bits':leaf.bits,'group_size':leaf.group_size}
        else: values[name]=getattr(leaf,name.rsplit('.',1)[1]).astype(mx.float32)
    result=_save_packed(source,manifest,spec.destination,values,formats,spec.method,cal.manifest_sha256)
    if spec.method=='dynamic':
        allocation_trace['packed_map_sha256']=hashlib.sha256(_canonical(formats)).hexdigest()
        Path(spec.destination+'.allocation-trace.json').write_bytes(_canonical(allocation_trace))
    # Roundtrip the final actual transformed model against the common inference
    # boundary, after packing, for the same calibration-only prompt.
    restored=load_bundle(Path(result.bundle.directory),backend='mlx')
    if new_accounting:state['phase']='roundtrip'
    expected=model(data[:1])
    native_charge('roundtrip',1,spec.length,teacher=False)
    observed=restored.logits(_native_batch([cal.tokens[0]]));mx.eval(expected,observed)
    error=mx.max(mx.abs(expected-observed)).item()
    evidence={'method':spec.method,'initial_official_bridge_max_abs_error':bridge_error,'packed_common_runtime_max_abs_error':error,'actual_upstream_counts':{k:v for k,v in state.items() if k not in ('model','quantized')},'calibration_component_ids':cal.component_ids,'internal_train_component_ids':cal.component_ids[:spec.calibration_count],'internal_valid_component_ids':cal.component_ids[spec.calibration_count:],'temperature':1. if spec.method=='dwq' else None,'seed':spec.seed,'group_size':spec.group_size,'bits':spec.bits,'embedding_bits':spec.embedding_bits,'dynamic_high_bits':spec.dynamic_high_bits if spec.method=='dynamic' else None,'dynamic_target_effective_bpw':spec.dynamic_target_bpw if spec.method=='dynamic' else None,'default_loader_denial_at_imported_binding':True,'socket_connect_denied':True}
    if new_accounting:
        evidence.update(accounting_profile=accounting_profile,tokenization_policy='no-special-tokens-v2',original_context_tokens=spec.length,
            accounting_units={'raw_corpus_target_presentations':'each native/official model context view uses original511next-token positions; repeated teacher/student views counted separately',
            'valid_target_presentations':'each DWQ student loss target once, train and internal validation; optimizer does not doublecharge',
            'logical_optimizer_raw_target_presentations':'one original511target context per actual DWQ update; distinct from model context views'},
            total_model_context_raw_presentations=state['forward_raw_corpus_presentations']+state['native_raw_corpus_presentations'])
    from poc.bundle import _canonical
    Path(spec.destination+'.quantizer-evidence.json').write_bytes(_canonical(evidence))
    if error>.01: raise RuntimeError(f'packed official/common-runtime roundtrip unresolved fidelity: {error}')
    return result


# Calibration is a separate explicit argument; preserve the public signature.
def quantize_dense(bundle: BundleRef, spec: QuantizationSpec, calibration: DataManifest, recorder: Recorder, *, accounting_profile=None) -> QuantizationResult:
    import time,psutil
    if accounting_profile not in (None,'raw-contexts-v1','dense-native-contexts-v2'): raise ValueError('unknown dense quantization accounting profile')
    if accounting_profile=='dense-native-contexts-v2' and (spec.method not in ('gptq','awq','dwq') or spec.length!=512
            or spec.method=='gptq' and spec.bits not in (2,4,8)):
        raise ValueError('new accounting requires eligible learned dense method and complete512context')
    if spec.method=='rtn': return quantize_rtn(bundle,spec,recorder)
    from poc.data.access import AccessDenied
    if not calibration.records or any(role!='fit' for role in calibration.roles.values()): raise AccessDenied('calibration requires fit-only manifest')
    start=time.monotonic()
    if recorder is not None: recorder.check(CostRecord())
    try: return _quantize_dense_impl(bundle,spec,calibration,recorder,accounting_profile)
    finally:
        if recorder is not None: recorder.record(CostRecord(wall_seconds=time.monotonic()-start,memory_bytes=psutil.Process().memory_info().rss,memory_unavailable_reason=None))


def _dense_dwq_owners(tree,prefix=''):
    from collections.abc import Mapping
    result=[]
    if isinstance(tree,Mapping):
        for name,value in tree.items():result.extend(_dense_dwq_owners(value,(prefix+'.' if prefix else '')+str(name)))
    elif isinstance(tree,(list,tuple)):
        for index,value in enumerate(tree):result.extend(_dense_dwq_owners(value,(prefix+'.' if prefix else '')+str(index)))
    elif tree is not None:
        if not hasattr(tree,'shape') or not hasattr(tree,'dtype'):raise ValueError('trainable parameter tree contains a non-array owner')
        result.append({'name':prefix,'shape':list(tree.shape),'dtype':str(tree.dtype)})
    return sorted(result,key=lambda row:row['name'])


def configure_dense_dwq_trainability(model,accounting_profile):
    """Explicit v2 affine-only fitting; historical profiles keep their scope."""
    if accounting_profile not in (None,'raw-contexts-v1','dense-native-contexts-v2'):raise ValueError('unknown dense scope profile')
    if accounting_profile!='dense-native-contexts-v2':return None
    model.freeze();allowed=[]
    def select(name,leaf):
        if (hasattr(leaf,'bits') and hasattr(leaf,'group_size') and getattr(leaf,'mode',None)=='affine' and leaf.bits<8):
            leaf.unfreeze(keys=['scales','biases'],recurse=False)
            for key in ('scales','biases'):
                value=getattr(leaf,key)
                allowed.append({'name':(name+'.' if name else '')+key,'shape':list(value.shape),'dtype':str(value.dtype)})
    model.apply_to_modules(select);allowed.sort(key=lambda row:row['name'])
    if not allowed or len({row['name'] for row in allowed})!=len(allowed):raise ValueError('nonempty unique affine trainable owner inventory required')
    from poc.records import canonical_json
    scope={'schema_version':1,'scope':'affine_scales_biases_only_v2','owners':allowed,
           'owners_sha256':hashlib.sha256(canonical_json({'owners':allowed})).hexdigest()}
    validate_dense_dwq_trainability(model,scope);return scope


def validate_dense_dwq_trainability(model,scope,*,params=None):
    validate_dense_dwq_scope_record(scope)
    if (_dense_dwq_owners(model.trainable_parameters())!=scope['owners']
            or params is not None and _dense_dwq_owners(params)!=scope['owners']):
        raise ValueError('actual DWQ trainable owner scope differs from fixed affine-only inventory')


def validate_dense_dwq_scope_record(scope):
    from poc.records import canonical_json
    if (type(scope) is not dict or set(scope)!={'schema_version','scope','owners','owners_sha256'}
            or type(scope['schema_version']) is not int or scope['schema_version']!=1
            or scope['scope']!='affine_scales_biases_only_v2' or type(scope['owners']) is not list or not scope['owners']):
        raise ValueError('complete affine-only DWQ trainable owner scope record required')
    names=[]
    for row in scope['owners']:
        if (type(row) is not dict or set(row)!={'name','shape','dtype'} or type(row['name']) is not str
                or row['name'].rsplit('.',1)[-1] not in ('scales','biases') or type(row['shape']) is not list
                or not row['shape'] or any(type(n) is not int or n<1 for n in row['shape'])
                or type(row['dtype']) is not str or not row['dtype']):
            raise ValueError('invalid affine-only DWQ trainable owner entry')
        names.append(row['name'])
    if (names!=sorted(set(names)) or scope['owners_sha256']!=hashlib.sha256(canonical_json({'owners':scope['owners']})).hexdigest()):
        raise ValueError('exact sorted DWQ trainable owner inventory hash differs')

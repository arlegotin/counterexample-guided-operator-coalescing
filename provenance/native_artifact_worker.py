"""Fresh-process evaluation using an artifact's own frozen runtime/profile."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','inputs','teacher-manifest','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--inputs-sha256',required=True)
    args=p.parse_args();started=time.monotonic()
    if args.out.exists():raise FileExistsError(args.out)
    if sha(args.artifact/'manifest.json')!=args.manifest_sha256 or sha(args.inputs)!=args.inputs_sha256:
        raise ValueError('frozen artifact/input hash mismatch')
    meta=json.loads((args.artifact/'manifest.json').read_text())
    runtime=args.artifact/'runtime/poc-runtime.zip'
    if sha(runtime)!=meta['file_entries']['runtime/poc-runtime.zip']['sha256']:raise ValueError('runtime hash mismatch')
    if 'poc' in sys.modules:raise RuntimeError('artifact runtime must be selected before any project import')
    sys.path.insert(0,str(runtime))
    import poc
    import numpy as np
    import torch
    import mlx.core as mx
    import psutil
    import resource
    from transformers import AutoModelForCausalLM
    from poc.bundle import load_bundle,measure_bundle,measure_resident
    from poc.runtime.torch_decoder import TorchDecoder,EXECUTION_PROFILE
    from poc.contracts import Batch
    if not torch.backends.mps.is_available() or not mx.metal.is_available():raise RuntimeError('native devices unavailable')
    torch.set_num_threads(4)
    manifest=args.teacher_manifest
    if sha(manifest)!='9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083':raise ValueError('teacher manifest differs')
    assets=json.loads(manifest.read_text())['assets'];parents=set()
    for entry in assets:
        path=(manifest.parent/entry['local_path']).resolve()
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('teacher asset differs')
        parents.add(path.parent)
    if len(parents)!=1:raise ValueError('inconsistent teacher asset set')
    hf=AutoModelForCausalLM.from_pretrained(parents.pop(),local_files_only=True,trust_remote_code=False,
        use_safetensors=True,dtype=torch.float32,attn_implementation='eager')
    hf.config._name_or_path=''
    teacher=TorchDecoder.from_transformers(hf,tokenizer_digest='9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083').to('mps')
    del hf
    t=time.monotonic();model=load_bundle(args.artifact,backend='mlx');mx.synchronize();load_seconds=time.monotonic()-t
    entries=json.loads(args.inputs.read_text())['windows'];rows=[];teacher_calls=student_calls=0
    def batches(row):
        ids=torch.tensor([row['tokens']],dtype=torch.long)
        mask=torch.tensor([row['target_mask']],dtype=torch.bool)
        base=Batch(ids,torch.ones_like(ids,dtype=torch.bool),mask,torch.arange(ids.shape[1])[None],
            (row['document_id'],),(row['component_id'],))
        tb=Batch(*(v.to('mps') for v in (base.input_ids,base.attention_mask,base.target_mask,base.positions)),base.document_ids,base.cluster_ids)
        nb=Batch(*(mx.array(v.numpy()) for v in (base.input_ids,base.attention_mask,base.target_mask,base.positions)),base.document_ids,base.cluster_ids)
        return base,tb,nb
    for entry in entries:
        base,tb,nb=batches(entry)
        with torch.no_grad():th=teacher.final_hidden(tb)
        teacher_calls+=1
        sh=model.final_hidden(nb);mx.eval(sh);student_calls+=1
        kl=nll=0.;count=0
        for start in range(0,base.target_mask.shape[1],32):
            end=min(start+32,base.target_mask.shape[1]);valid=base.target_mask[:,start:end]
            with torch.no_grad():tlogits=teacher.readout(th[:,start:end]).cpu().double()
            teacher_calls+=1
            slogits=model.readout(sh[:,start:end]);mx.eval(slogits);student_calls+=1
            s=torch.from_numpy(np.array(slogits)).double();tl=tlogits.log_softmax(-1);sl=s.log_softmax(-1)
            kl+=float((tl.exp()*(tl-sl)).sum(-1)[valid].sum())
            labels=base.input_ids[:,start+1:end+1]
            nll+=float((-sl.gather(-1,labels[...,None]).squeeze(-1))[valid].sum());count+=int(valid.sum())
        rows.append(dict(document_id=entry['document_id'],component_id=entry['component_id'],
            window_sha256=entry['window_sha256'],kl_sum=kl,nll_sum=nll,valid_targets=count))
    timings=[]
    for entry in entries[:3]:
        _,_,nb=batches(entry);cache=model.new_cache();mx.synchronize();start=time.monotonic()
        hidden=model.final_hidden(nb,cache=cache);last=model.readout(hidden[:,-1:]);mx.eval(last);mx.synchronize();student_calls+=2
        prefill=time.monotonic()-start
        mx.synchronize();start=time.monotonic()
        for index in range(16):
            b=Batch(mx.array([[1]],dtype=mx.int64),mx.array([[True]]),mx.array([[]],dtype=mx.bool_),
                mx.array([[512+index]],dtype=mx.int64),nb.document_ids,nb.cluster_ids)
            value=model.logits(b,cache=cache);mx.eval(value);student_calls+=2
        mx.synchronize();decode=time.monotonic()-start
        timings.append(dict(prefill_512_next_token_seconds=prefill,decode_16_teacher_forced_seconds=decode))
    targets=sum(r['valid_targets'] for r in rows)
    result=dict(schema_version=1,status='complete',runtime_source=poc.__file__,execution_profile=EXECUTION_PROFILE,
        manifest_sha256=args.manifest_sha256,inputs_sha256=args.inputs_sha256,
        payload_bytes=measure_bundle(args.artifact).payload_bytes,resident=asdict_compatible(measure_resident(model)),
        native_load_seconds=load_seconds,rows=rows,kl_per_target=sum(r['kl_sum'] for r in rows)/targets,
        nll_per_target=sum(r['nll_sum'] for r in rows)/targets,valid_targets=targets,
        timing_samples=timings,timing_scope='3 warm source contexts, batch1/context512, fixed-token cached decode16; descriptive, no stable-tail claim',
        teacher_forward_calls=teacher_calls,student_forward_calls=student_calls,
        memory=dict(process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            torch_mps_driver_bytes=torch.mps.driver_allocated_memory(),mlx_active_bytes=mx.get_active_memory(),mlx_peak_bytes=mx.get_peak_memory(),
            note='unified-memory counters overlap; not summed into disjoint physical memory'),wall_seconds=time.monotonic()-started)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','payload_bytes','kl_per_target','nll_per_target','wall_seconds')}))


def asdict_compatible(value):
    from dataclasses import fields
    from poc.contracts import thaw
    return {field.name:thaw(getattr(value,field.name)) for field in fields(value)}


if __name__=='__main__':main()

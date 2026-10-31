"""Source-bound offline full-vocabulary fidelity worker; no timing benchmark.

Inputs must be compiled by the parent through the allowed dev/selection reader.
The worker consumes that hash-bound artifact, never an origin/protected corpus.
Each artifact and teacher reference execute one identical embedded runtime.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys
import time


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(value,expected,name):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise ValueError('missing or unknown '+name+' fields')


def read_windows(path,digest,*,population_sha256,tokenizer_digest,vocabulary_size):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('compiled input identity differs')
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('compiled input identity differs')
    data=json.loads(content)
    fields(data,{'schema_version','role','purpose','population_manifest_sha256','tokenizer_digest','components','windows'},'compiled inputs')
    if (type(data['schema_version']) is not int or data['schema_version']!=1
            or data['role']!='dev' or data['purpose']!='selection'
            or data['population_manifest_sha256']!=population_sha256
            or data['tokenizer_digest']!=tokenizer_digest):
        raise ValueError('compiled inputs must bind the declared exploratory dev/selection population')
    rows=data['windows']
    if (not isinstance(rows,list) or not rows or type(data['components']) is not int
            or len(rows)!=data['components']):raise ValueError('component inventory differs')
    seen=set();documents=set()
    for row in rows:
        fields(row,{'document_id','component_id','tokens','attention_mask','target_mask','positions','window_sha256'},'window')
        if not all(isinstance(row[k],str) and row[k] for k in ('document_id','component_id')):
            raise ValueError('document/component identities required')
        if row['component_id'] in seen or row['document_id'] in documents:raise ValueError('one deterministic window per document/component required')
        seen.add(row['component_id']);documents.add(row['document_id'])
        if (not isinstance(row['tokens'],list) or len(row['tokens'])!=512
                or any(type(t) is not int or not 0<=t<vocabulary_size for t in row['tokens'])
                or row['positions']!=list(range(512))):raise ValueError('complete512-token position-reset window required')
        for name,n in (('attention_mask',512),('target_mask',511)):
            mask=row[name]
            if not isinstance(mask,list) or len(mask)!=n or any(type(x) is not bool for x in mask):
                raise ValueError('explicit boolean masks required')
        if not any(row['target_mask']):raise ValueError('nonempty shifted targets required')
        if any(valid and not (row['attention_mask'][i] and row['attention_mask'][i+1]) for i,valid in enumerate(row['target_mask'])):
            raise ValueError('target mask includes unattended inputs')
        expected=hashlib.sha256(canonical({k:v for k,v in row.items() if k!='window_sha256'})).hexdigest()
        if row['window_sha256']!=expected:raise ValueError('exact window/mask identity differs')
    return rows


def design(protocol):
    return {k:v for k,v in protocol.items() if k not in ('mode','output','ledger','resource','profile_binding')}


def validate_resource(budget):
    fields(budget,{'wall_seconds','memory_bytes','output_disk_bytes','valid_target_presentations','teacher_forward_calls','student_forward_calls'},'resource budget')
    for name,value in budget.items():
        if (type(value) not in (int,float) or not math.isfinite(value) or value<=0
                or name!='wall_seconds' and type(value) is not int):
            raise ValueError('finite positive worker limits and integer byte/query/target units required')


def metric_chunk(teacher_logits,student_logits,labels,mask):
    import torch
    if (teacher_logits.shape!=student_logits.shape or teacher_logits.ndim!=3
            or labels.shape!=teacher_logits.shape[:-1] or mask.shape!=labels.shape
            or any(x.device.type!='cpu' for x in (teacher_logits,student_logits,labels,mask))
            or teacher_logits.dtype!=torch.float64 or student_logits.dtype!=torch.float64
            or labels.dtype!=torch.int64 or mask.dtype!=torch.bool):
        raise ValueError('detached CPU FP64 complete-vocabulary logits and aligned int64 labels/boolean mask required')
    tl=teacher_logits.log_softmax(-1);sl=student_logits.log_softmax(-1)
    if not torch.isfinite(tl).all() or not torch.isfinite(sl).all():raise FloatingPointError('nonfinite full-vocabulary log probabilities')
    return {'kl_sum':float((tl.exp()*(tl-sl)).sum(-1)[mask].sum()),
        'nll_sum':float((-sl.gather(-1,labels[...,None]).squeeze(-1))[mask].sum()),
        'teacher_nll_sum':float((-tl.gather(-1,labels[...,None]).squeeze(-1))[mask].sum()),
        'disagreements':int(((teacher_logits.argmax(-1)!=student_logits.argmax(-1))&mask).sum()),
        'valid_targets':int(mask.sum())}


def publish_result(output,value,final_budget_check):
    """Only a committed .json is eligible; failed pending files stay auditable."""
    import os
    from uuid import uuid4
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    pending=output.with_name(output.name+'.pending-'+uuid4().hex)
    with pending.open('xb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    final_budget_check()
    os.link(pending,output);pending.unlink()
    fd=os.open(output.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verified_snapshot(source,expected_manifest,destination,*,max_bytes,check=None):
    """Copy a pinned fileset through anchored directory FDs, then make it read-only.

    No hardlinks, symlinks or lazy reads from the mutable original are used.
    A failed copy retains incomplete work without a loadable manifest.
    """
    import os,stat
    from pathlib import PurePosixPath
    source=Path(source);destination=Path(destination)
    if type(max_bytes) is not int or max_bytes<1:raise ValueError('finite snapshot byte budget required')
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
    root_fd=os.open(source,flags)
    def open_file(relative):
        parts=PurePosixPath(relative).parts;parent=root_fd;opened=[]
        try:
            for part in parts[:-1]:
                parent=os.open(part,flags,dir_fd=parent);opened.append(parent)
            result=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
            info=os.fstat(result)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
                os.close(result);raise ValueError('snapshot source must contain unlinked regular files')
            return result
        finally:
            for fd in reversed(opened):os.close(fd)
    def inventory(fd,prefix=''):
        result=set()
        for name in os.listdir(fd):
            relative=prefix+name;info=os.stat(name,dir_fd=fd,follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child=os.open(name,flags,dir_fd=fd)
                try:result.update(inventory(child,relative+'/'))
                finally:os.close(child)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink==1:result.add(relative)
            else:raise ValueError('snapshot rejects symlinks and nonregular/linked entries')
        return result
    try:
        with os.fdopen(open_file('manifest.json'),'rb') as stream:manifest_bytes=stream.read()
        if hashlib.sha256(manifest_bytes).hexdigest()!=expected_manifest:raise ValueError('snapshot manifest identity differs')
        metadata=json.loads(manifest_bytes);entries=metadata.get('file_entries')
        if not isinstance(entries,dict) or not entries or 'manifest.json' in entries:raise ValueError('snapshot fileset missing')
        total=len(manifest_bytes)
        for name,entry in entries.items():
            relative=PurePosixPath(name)
            if (not isinstance(name,str) or not name or relative.is_absolute() or str(relative)!=name
                    or '..' in relative.parts or '\\' in name or not isinstance(entry,dict)
                    or type(entry.get('bytes')) is not int or entry['bytes']<0
                    or not isinstance(entry.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',entry['sha256'])):
                raise ValueError('invalid snapshot entry')
            total+=entry['bytes']
        if total>max_bytes:raise ValueError('snapshot exceeds remaining output budget')
        if inventory(root_fd)!=set(entries)|{'manifest.json'}:raise ValueError('unmanifested or missing snapshot source payload')
        destination.mkdir(parents=True,exist_ok=False,mode=0o700)
        for name,entry in entries.items():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            digest=hashlib.sha256();copied=0
            with os.fdopen(open_file(name),'rb') as incoming,target.open('xb') as outgoing:
                if os.fstat(incoming.fileno()).st_size!=entry['bytes']:raise ValueError('snapshot payload size differs')
                while True:
                    chunk=incoming.read(min(1024**2,entry['bytes']-copied+1))
                    if not chunk:break
                    copied+=len(chunk)
                    if copied>entry['bytes']:raise ValueError('snapshot payload grew')
                    digest.update(chunk);outgoing.write(chunk)
                    if check is not None:check()
                outgoing.flush();os.fsync(outgoing.fileno())
            if copied!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:raise ValueError('snapshot payload hash differs')
            target.chmod(0o400)
        with (destination/'manifest.json').open('xb') as stream:
            stream.write(manifest_bytes);stream.flush();os.fsync(stream.fileno())
        (destination/'manifest.json').chmod(0o400)
        for folder in sorted((x for x in destination.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):folder.chmod(0o500)
        destination.chmod(0o500)
        return metadata
    finally:os.close(root_fd)


def compatible_tokenizers(teacher,candidate):
    if (teacher.get('tokenizer_digest')!=candidate.get('tokenizer_digest')
            or teacher.get('config',{}).get('vocab_size')!=candidate.get('config',{}).get('vocab_size')):
        raise ValueError('candidate tokenizer/vocabulary identity differs from teacher')
    def files(metadata):
        return {name:metadata['file_entries']['tokenizer/'+name]['sha256'] for name in metadata['tokenizer_files']}
    if files(teacher)!=files(candidate):raise ValueError('candidate tokenizer file identities differ from teacher')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    protocol_bytes=args.protocol.read_bytes();protocol_sha=hashlib.sha256(protocol_bytes).hexdigest();p=json.loads(protocol_bytes)
    source_sha=sha(__file__)
    fields(p,{'schema_version','study','study_contract_sha256','mode','artifact','teacher','inputs','numerical','worker_sha256','runtime_sha256','blocked_source_roots','output','ledger','resource','profile_binding'},'worker protocol')
    if p['schema_version']!=1 or type(p['schema_version']) is not int or p['mode'] not in ('profile','run'):
        raise ValueError('unknown worker protocol/mode')
    if not isinstance(p['study_contract_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',p['study_contract_sha256']):raise ValueError('frozen study analysis contract required')
    if source_sha!=p['worker_sha256']:raise ValueError('worker source differs')
    for name in ('artifact','teacher'):
        expected={'path','manifest_sha256'}|({'mps_checkpoint_digest'} if name=='teacher' else set())
        fields(p[name],expected,name)
    fields(p['inputs'],{'path','sha256','population_manifest_sha256'},'input binding')
    fields(p['numerical'],{'chunk_size','temperature','negative_floor','decision_resolution'},'numerical profile')
    if p['numerical']['chunk_size']!=32 or p['numerical']['temperature']!=1:
        raise ValueError('position chunk32 and temperature1 required')
    for key in ('negative_floor','decision_resolution'):
        value=p['numerical'][key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('finite numerical profile required')
    design_sha=hashlib.sha256(canonical(design(p))).hexdigest()
    if p['mode']=='run':
        fields(p['profile_binding'],{'path','sha256'},'profile binding')
        profile_path=Path(p['profile_binding']['path'])
        if profile_path.suffix!='.json':raise ValueError('only committed profile JSON is eligible')
        profile_bytes=profile_path.read_bytes()
        if hashlib.sha256(profile_bytes).hexdigest()!=p['profile_binding']['sha256']:raise ValueError('frozen profile differs')
        profile=json.loads(profile_bytes)
        if profile['mode']!='profile' or profile['design_sha256']!=design_sha:raise ValueError('profile design differs')
    elif p['profile_binding'] is not None:raise ValueError('profile cannot import another quality result')
    output=Path(p['output']);ledger=Path(p['ledger'])
    if output.suffix!='.json' or ledger.suffix!='.jsonl' or output.exists() or ledger.exists() or output.parent!=ledger.parent:raise ValueError('new isolated output and ledger required')
    output.parent.mkdir(parents=True,exist_ok=True)
    budget=p['resource']
    validate_resource(budget)
    def early_limits():
        if time.monotonic()-started>budget['wall_seconds']:raise RuntimeError('worker snapshot wall budget exhausted')
        if sum(x.stat().st_size for x in output.parent.rglob('*') if x.is_file())>budget['output_disk_bytes']:
            raise RuntimeError('worker snapshot disk budget exhausted')
    snapshots={};metadata={};copy_started=time.monotonic()
    for name in ('teacher','artifact'):
        snapshots[name]=output.parent/'verified-artifacts'/name
        remaining=budget['output_disk_bytes']-sum(x.stat().st_size for x in output.parent.rglob('*') if x.is_file())
        metadata[name]=verified_snapshot(p[name]['path'],p[name]['manifest_sha256'],snapshots[name],max_bytes=remaining,check=early_limits)
        if metadata[name]['file_entries']['runtime/poc-runtime.zip']['sha256']!=p['runtime_sha256']:
            raise ValueError('teacher/artifact must use identical frozen embedded runtime')
    snapshot_seconds=time.monotonic()-copy_started
    compatible_tokenizers(metadata['teacher'],metadata['artifact'])
    if 'poc' in sys.modules:raise RuntimeError('select artifact runtime before project imports')
    sys.path.insert(0,str(snapshots['artifact']/'runtime/poc-runtime.zip'))
    import site,os,resource
    allowed=tuple(Path(x).resolve() for x in [str(snapshots['artifact']),str(snapshots['teacher']),str(Path(p['inputs']['path']).parent),str(output.parent),*site.getsitepackages()])
    blocked=tuple(Path(x).resolve() for x in p['blocked_source_roots'])
    def guard(event,values):
        if event.startswith('socket.'):raise RuntimeError('offline worker network access denied')
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(values[0])).resolve()
            if any(path.is_relative_to(root) for root in blocked) and not any(path.is_relative_to(root) for root in allowed):
                raise RuntimeError('source/training path denied: '+str(path))
    sys.addaudithook(guard)
    import numpy as np
    import torch
    import mlx.core as mx
    import psutil
    from poc.bundle import load_bundle,measure_bundle,measure_resident
    from poc.contracts import Batch,thaw
    from dataclasses import fields as dataclass_fields
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    if not torch.backends.mps.is_available() or not mx.metal.is_available():raise RuntimeError('qualified native hardware unavailable')
    torch.set_num_threads(1);mx.set_default_device(mx.gpu)
    import platform,importlib.metadata
    environment={'python':sys.version,'system':platform.system(),'release':platform.release(),
        'macos':platform.mac_ver()[0],'machine':platform.machine(),'torch':str(torch.__version__),
        'mlx':importlib.metadata.version('mlx'),'numpy':str(np.__version__),
        'safetensors':importlib.metadata.version('safetensors'),'threads':torch.get_num_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'matmul_precision':torch.get_float32_matmul_precision(),'mlx_default_device':str(mx.default_device()),
        'physical_memory_bytes':psutil.virtual_memory().total,
        'execution_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','HF_HUB_OFFLINE')},
        'qualification':'observed environment; not a hash-locked recreation'}
    environment_sha=hashlib.sha256(canonical(environment)).hexdigest()
    if p['mode']=='run' and profile.get('environment_sha256')!=environment_sha:
        raise ValueError('execution environment differs from qualified profile')
    current_window=None
    counts={'valid_target_presentations':0,'teacher_forward_calls':0,'student_forward_calls':0}
    def observe():
        host=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform!='darwin':host*=1024
        values={'process_peak_rss_bytes':max(host,psutil.Process().memory_info().rss),
            'torch_mps_driver_bytes':int(torch.mps.driver_allocated_memory()),'mlx_peak_bytes':int(mx.get_peak_memory())}
        if time.monotonic()-started>budget['wall_seconds']:raise RuntimeError('worker wall budget exhausted')
        if max(values.values())>budget['memory_bytes']:raise RuntimeError('worker observed memory budget exhausted')
        if sum(x.stat().st_size for x in output.parent.rglob('*') if x.is_file())>budget['output_disk_bytes']:
            raise RuntimeError('worker output disk budget exhausted')
        return values
    def charge(**work):
        for name,value in work.items():
            if counts[name]+value>budget[name]:raise RuntimeError('worker query/target budget exhausted: '+name)
        for name,value in work.items():counts[name]+=value
        with ledger.open('ab') as f:
            f.write(canonical({'study':p['study'],'study_contract_sha256':p['study_contract_sha256'],'protocol_sha256':protocol_sha,
                'window_sha256':current_window,'raw_corpus_target_presentations':work.get('valid_target_presentations',0),
                'elapsed_seconds':time.monotonic()-started,'work':work,'cumulative':dict(counts)}));f.flush();os.fsync(f.fileno())
        observe()
    observe();tick=time.monotonic();teacher=load_bundle(snapshots['teacher'],backend='torch').to('mps');torch.mps.synchronize();teacher_load=time.monotonic()-tick
    if teacher.manifest().checkpoint_digest!=p['teacher']['mps_checkpoint_digest']:
        raise ValueError('teacher executable identity differs from the scientific reference')
    tick=time.monotonic();model=load_bundle(snapshots['artifact'],backend='mlx');mx.synchronize();model_load=time.monotonic()-tick
    rows=read_windows(p['inputs']['path'],p['inputs']['sha256'],population_sha256=p['inputs']['population_manifest_sha256'],tokenizer_digest=teacher.tokenizer_digest,vocabulary_size=teacher.config.vocab_size)
    selected=rows[:1] if p['mode']=='profile' else rows
    observations=[];durations=[]
    for row in selected:
        current_window=row['window_sha256']
        tick=time.monotonic();n=sum(row['target_mask']);charge(valid_target_presentations=n)
        ids=torch.tensor([row['tokens']],dtype=torch.int64);attention=torch.tensor([row['attention_mask']],dtype=torch.bool)
        mask=torch.tensor([row['target_mask']],dtype=torch.bool);positions=torch.tensor([row['positions']],dtype=torch.int64)
        tb=Batch(*(v.to('mps') for v in (ids,attention,mask,positions)),(row['document_id'],),(row['component_id'],))
        nb=Batch(*(mx.array(v.numpy()) for v in (ids,attention,mask,positions)),(row['document_id'],),(row['component_id'],))
        charge(teacher_forward_calls=1)
        with torch.no_grad():th=teacher.final_hidden(tb)
        charge(student_forward_calls=1);sh=model.final_hidden(nb);mx.eval(sh)
        kl=[];nll=[];teacher_nll=[];disagreements=0
        for start in range(0,511,32):
            end=min(start+32,511);valid=mask[:,start:end];labels=ids[:,start+1:end+1]
            charge(teacher_forward_calls=1)
            with torch.no_grad():tlogits=teacher.readout(th[:,start:end]).cpu().double()
            charge(student_forward_calls=1);slogits=model.readout(sh[:,start:end]);mx.eval(slogits)
            # Device transfer and FP64 conversion remain separate operations.
            s=torch.from_numpy(np.array(slogits)).double()
            part=metric_chunk(tlogits,s,labels,valid)
            kl.append(part['kl_sum']);nll.append(part['nll_sum']);teacher_nll.append(part['teacher_nll_sum'])
            disagreements+=part['disagreements']
        raw_kl=math.fsum(kl)
        if raw_kl < -p['numerical']['negative_floor']*n:raise FloatingPointError('KL below qualified negative floor')
        observations.append({'study_contract_sha256':p['study_contract_sha256'],'protocol_sha256':protocol_sha,'document_id':row['document_id'],'component_id':row['component_id'],'window_sha256':row['window_sha256'],
            'kl_sum':raw_kl,'nll_sum':math.fsum(nll),'teacher_nll_sum':math.fsum(teacher_nll),'disagreements':disagreements,'valid_targets':n})
        torch.mps.synchronize();mx.synchronize();durations.append(time.monotonic()-tick)
    is_noop=p['artifact']['manifest_sha256']==p['teacher']['manifest_sha256']
    if is_noop and any(abs(row['kl_sum']/row['valid_targets'])>p['numerical']['negative_floor'] for row in observations):
        raise ValueError('real teacher-equivalent numerical qualification failed')
    accounting=measure_resident(model)
    result={'schema_version':1,'study':p['study'],'mode':p['mode'],'design_sha256':design_sha,'study_contract_sha256':p['study_contract_sha256'],'protocol_sha256':protocol_sha,
        'worker_sha256':source_sha,'environment':environment,'environment_sha256':environment_sha,'execution_profile':EXECUTION_PROFILE,'runtime_sha256':p['runtime_sha256'],
        'artifact_manifest_sha256':p['artifact']['manifest_sha256'],'teacher_manifest_sha256':p['teacher']['manifest_sha256'],
        'inputs_sha256':p['inputs']['sha256'],'teacher_mps_checkpoint_digest':teacher.manifest().checkpoint_digest,
        'artifact_snapshot':str(snapshots['artifact']),'teacher_snapshot':str(snapshots['teacher']),
        'snapshot_seconds':snapshot_seconds,'snapshot_policy':'verified private read-only copies retained; loading and byte accounting never revisit original paths',
        'complete_bytes':measure_bundle(snapshots['artifact']).payload_bytes,'canonical_owner_resident':{f.name:thaw(getattr(accounting,f.name)) for f in dataclass_fields(accounting)},
        'counts':counts,'raw_corpus_target_presentations':counts['valid_target_presentations'],'unique_evaluated_targets':sum(r['valid_targets'] for r in observations),
        'teacher_equivalent_control':is_noop,'teacher_equivalent_passed':True if is_noop else None,
        'teacher_load_seconds':teacher_load,'artifact_load_seconds':model_load,'context_metric_seconds':durations,
        'wall_seconds':time.monotonic()-started,'memory':observe(),
        'memory_policy':'overlapping unified-memory counters reported separately; never summed as physical peak',
        'measurement_scope':'exploratory fidelity and resource forecasting; concurrent hardware activity allowed; no production latency or capability-equivalence claim'}
    if p['mode']=='run':result['rows']=observations
    else:result['quality']='candidate KL/NLL/disagreement withheld; only teacher-equivalent pass/fail may qualify numerics'
    publish_result(output,result,observe)
    print(json.dumps({'mode':p['mode'],'result':str(output),'scored_targets':counts['valid_target_presentations'],'quality_disclosed':p['mode']=='run'}))


if __name__=='__main__':main()

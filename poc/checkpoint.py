"""Verified immutable safe-tensor generations with atomic current/previous pointer.

No pickle or live autograd graph is accepted. Publication faults leave staging
and orphan generations intact. Explicit previous fallback never hides corruption.
The spending ledger is external and authoritative; loading never rewinds it.
"""
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from poc.contracts import Budget, RunSpec
from poc.records import canonical_json, run_id


@dataclass(frozen=True)
class CheckpointRef:
    directory: str
    manifest_sha256: str

    def __post_init__(self):
        if not self.directory or not re.fullmatch('[0-9a-f]{64}',self.manifest_sha256):raise ValueError('checkpoint requires detached manifest hash')


def _hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _publication_boundary(stage):
    """Fault-injection seam at durable filesystem transaction boundaries."""


def _encode(value,tensors):
    if isinstance(value,torch.Tensor):
        key=f'tensor_{len(tensors):06d}'
        tensors[key]=value.detach().cpu().contiguous()
        return {'tag':'tensor','key':key}
    if isinstance(value,np.ndarray):
        key=f'tensor_{len(tensors):06d}'
        # NumPy MT state uses uint32; safe tensors supports it directly.
        tensors[key]=torch.from_numpy(value.copy())
        return {'tag':'numpy','key':key}
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('checkpoint keys require strings')
        return {'tag':'dict','items':{k:_encode(v,tensors) for k,v in value.items()}}
    if isinstance(value,(tuple,list)):
        return {'tag':'tuple' if isinstance(value,tuple) else 'list','items':[_encode(v,tensors) for v in value]}
    if value is None or type(value) in (str,bool,int,float):
        # canonical JSON validates finite floating scalar values.
        return {'tag':'scalar','value':value}
    raise ValueError('unsafe checkpoint state type: '+type(value).__name__)


def _decode(value,tensors,used):
    if not isinstance(value,dict) or 'tag' not in value:raise ValueError('malformed safe state')
    tag=value['tag']
    if tag in ('tensor','numpy'):
        if set(value)!= {'tag','key'} or value['key'] not in tensors:raise ValueError('missing checkpoint tensor')
        key=value['key'];used.add(key);tensor=tensors[key]
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():raise ValueError('nonfinite checkpoint tensor')
        return tensor.numpy().copy() if tag=='numpy' else tensor
    if tag=='dict' and set(value)=={'tag','items'}:
        return {k:_decode(v,tensors,used) for k,v in value['items'].items()}
    if tag in ('tuple','list') and set(value)=={'tag','items'}:
        result=[_decode(v,tensors,used) for v in value['items']]
        return tuple(result) if tag=='tuple' else result
    if tag=='scalar' and set(value)=={'tag','value'}:return value['value']
    raise ValueError('unknown safe checkpoint tag')


def _read_json(path):
    from poc.bundle import _read_json as strict
    if path.is_symlink() or not path.is_file():raise ValueError('checkpoint requires regular files')
    return strict(path)


def _verify(ref):
    directory=Path(ref.directory)
    if directory.is_symlink() or not directory.is_dir():raise ValueError('invalid checkpoint generation')
    path=directory/'manifest.json'
    if _hash(path)!=ref.manifest_sha256:raise ValueError('checkpoint manifest hash mismatch')
    raw=_read_json(path)
    if set(raw)!= {'schema_version','run_id','files'} or raw['schema_version']!=1 or set(raw['files'])!={'state.json','tensors.safetensors'}:raise ValueError('invalid checkpoint manifest')
    if {p.name for p in directory.iterdir()}!={'manifest.json','state.json','tensors.safetensors'}:raise ValueError('unexpected checkpoint files')
    for name,digest in raw['files'].items():
        file=directory/name
        if file.is_symlink() or not file.is_file() or file.stat().st_nlink!=1 or _hash(file)!=digest:raise ValueError('checkpoint file hash mismatch')
    return raw


def save_checkpoint(state,root: Path,*,recorder=None) -> CheckpointRef:
    from poc.train import TrainingState
    if not isinstance(state,TrainingState) or not state.safe_boundary or state.spec is None:
        raise ValueError('checkpoint requires RunSpec and complete safe update boundary')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    boundary_recorder=recorder
    while boundary_recorder is not None and not hasattr(boundary_recorder,'validate') and hasattr(boundary_recorder,'recorder'):boundary_recorder=boundary_recorder.recorder
    if boundary_recorder is not None and hasattr(boundary_recorder,'validate'):boundary_recorder.validate()
    import time
    from poc.contracts import CostRecord
    started=time.monotonic()
    existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    def tensor_bytes(v):
        if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
        if isinstance(v,np.ndarray):return v.nbytes
        if isinstance(v,dict):return sum(tensor_bytes(x) for x in v.values())
        if isinstance(v,(tuple,list)):return sum(tensor_bytes(x) for x in v)
        return 0
    # Metadata bound is deliberately conservative. Actual bytes are observed
    # after publication, and the transient third generation also needs headroom.
    observed=recorder
    while observed is not None and not hasattr(observed,'disk') and hasattr(observed,'recorder'):observed=observed.recorder
    current=observed.disk() if observed is not None and hasattr(observed,'disk') else existing
    estimate=current+tensor_bytes(state.model)+tensor_bytes(state.optimizer)+tensor_bytes(state.rng)+2*1024**2
    if recorder is not None:recorder.check(CostRecord(disk_bytes=estimate))
    try:
        return _save_generation(state,root)
    finally:
        if recorder is not None:
            actual=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            recorder.record(CostRecord(wall_seconds=time.monotonic()-started,disk_bytes=actual))


def _save_generation(state,root):
    previous=resolve_checkpoint(root) if (root/'current.json').exists() else None
    old_pointer=_read_json(root/'current.json') if previous else None
    obsolete=old_pointer['previous'] if old_pointer else None
    if obsolete is not None and (not isinstance(obsolete,dict) or set(obsolete)!={'generation','manifest_sha256'} or not re.fullmatch('generation-[0-9a-f]{32}',obsolete['generation'])):
        raise ValueError('unsafe previous checkpoint generation pointer')
    generation='generation-'+uuid4().hex
    temporary=root/('.staging-'+generation);temporary.mkdir()
    tensors={}
    raw={f.name:getattr(state,f.name) for f in fields(state) if f.name!='spec'}
    raw['spec']=state.spec.to_dict()
    encoded=_encode(raw,tensors)
    (temporary/'state.json').write_bytes(canonical_json(encoded))
    save_file(tensors,temporary/'tensors.safetensors')
    manifest={'schema_version':1,'run_id':run_id(state.spec),
        'files':{name:_hash(temporary/name) for name in ('state.json','tensors.safetensors')}}
    (temporary/'manifest.json').write_bytes(canonical_json(manifest))
    ref=CheckpointRef(str(temporary),_hash(temporary/'manifest.json'))
    # Read it back through the same verifier/decoder before durable publication.
    load_checkpoint(ref,state.spec)
    for name in ('state.json','tensors.safetensors','manifest.json'):_fsync(temporary/name)
    _fsync(temporary)
    _publication_boundary('before_generation_publish')
    destination=root/generation;os.rename(temporary,destination);_fsync(root)
    ref=CheckpointRef(str(destination),ref.manifest_sha256)
    _publication_boundary('after_generation_publish')
    pointer={'schema_version':1,'current':{'generation':generation,'manifest_sha256':ref.manifest_sha256},
        'previous':None if previous is None else {'generation':Path(previous.directory).name,'manifest_sha256':previous.manifest_sha256}}
    pointer_tmp=root/('.pointer-'+uuid4().hex+'.json')
    pointer_tmp.write_bytes(canonical_json(pointer));_fsync(pointer_tmp)
    _publication_boundary('before_pointer_replace')
    os.replace(pointer_tmp,root/'current.json');_fsync(root)
    _publication_boundary('after_pointer_replace')
    # Prune only the explicitly published old previous generation. Staging and
    # crash-orphan generations remain diagnostic evidence and count toward disk.
    if obsolete is not None:
        candidate=root/obsolete['generation']
        old_ref=CheckpointRef(str(candidate),obsolete['manifest_sha256'])
        _verify(old_ref)
        import shutil
        shutil.rmtree(candidate);_fsync(root)
    return ref


def resolve_checkpoint(root: Path,*,fallback_previous=False) -> CheckpointRef:
    root=Path(root);pointer=_read_json(root/'current.json')
    if set(pointer)!={'schema_version','current','previous'} or pointer['schema_version']!=1:raise ValueError('invalid checkpoint pointer')
    def resolve(entry):
        if not isinstance(entry,dict) or set(entry)!={'generation','manifest_sha256'} or not re.fullmatch('generation-[0-9a-f]{32}',entry['generation']):raise ValueError('unsafe checkpoint generation pointer')
        ref=CheckpointRef(str(root/entry['generation']),entry['manifest_sha256']);_verify(ref);return ref
    try:return resolve(pointer['current'])
    except (OSError,ValueError):
        if not fallback_previous or pointer['previous'] is None:raise
        return resolve(pointer['previous'])


def _spec(raw):
    values=dict(raw);values['budgets']=Budget(**values['budgets']);return RunSpec(**values)


def load_checkpoint(ref: CheckpointRef,expected: RunSpec):
    from poc.train import TrainingState
    manifest=_verify(ref)
    if manifest['run_id']!=run_id(expected):raise ValueError('checkpoint lineage mismatch; new RunSpec required')
    root=Path(ref.directory);tensors=load_file(root/'tensors.safetensors',device='cpu');used=set()
    raw=_decode(_read_json(root/'state.json'),tensors,used)
    if used!=set(tensors):raise ValueError('unreferenced checkpoint tensors')
    if set(raw)!={f.name for f in fields(TrainingState)}:raise ValueError('checkpoint state fields mismatch')
    identity=_spec(raw['spec'])
    if run_id(identity)!=run_id(expected):raise ValueError('checkpoint lineage mismatch')
    raw['spec']=identity;state=TrainingState(**raw)
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    if state.model_schema.get('execution_profile') != EXECUTION_PROFILE:
        raise ValueError('unsupported checkpoint execution profile')
    if not state.safe_boundary or type(state.step) is not int or state.step<0 or type(state.target_cursor) is not int or state.target_cursor<0:raise ValueError('invalid update-boundary state')
    if not state.model or not state.source_hash or not state.config_hash or not state.rng:raise ValueError('incomplete recovery state')
    return state


def resolve_fitted_run(root: Path,identity: str,attempt_id: str):
    """Resolve actual attempt identity + checkpoint; external source labels stay separate."""
    if not re.fullmatch('[0-9a-f]{64}',identity) or not re.fullmatch('[0-9a-f]{32}',attempt_id):raise ValueError('invalid fitted run/attempt identity')
    path=Path(root)/identity/'attempts'/attempt_id
    attempt=_read_json(path/'run.json');spec=_spec(attempt['spec'])
    if attempt['run_id']!=identity or attempt['attempt_id']!=attempt_id or run_id(spec)!=identity:raise ValueError('fitted attempt identity mismatch')
    ref=resolve_checkpoint(path/'checkpoints');state=load_checkpoint(ref,spec)
    return state,ref


def checkpoint_decoder(state):
    """Reconstruct canonical scientific weights without base assets or fresh SVD."""
    from types import SimpleNamespace
    from poc.bundle import _expected_schema
    from poc.export import _load_scientific
    from poc.train import _saved_digest
    schema=state.model_schema
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    if schema.get('execution_profile') != EXECUTION_PROFILE:
        raise ValueError('unsupported checkpoint execution profile')
    raw=dict(schema,kind='compact',source_backend=schema['backend'],source_checkpoint_digest=state.candidate_digest)
    expected=_expected_schema(raw)
    if set(expected)!=set(state.model):raise ValueError('checkpoint canonical owner set mismatch')
    if {o['name'] for o in schema['owners']}!=set(expected):raise ValueError('checkpoint owner schema mismatch')
    for record in schema['owners']:
        n=record['name'];v=state.model[n];shape,category=expected[n]
        if (tuple(v.shape)!=shape or list(v.shape)!=record['shape'] or record['category']!=category
                or v.dtype not in (torch.float32,torch.float64) or not torch.isfinite(v).all().item()):raise ValueError('invalid checkpoint owner')
    if len({v.dtype for v in state.model.values()})!=1:raise ValueError('checkpoint owner dtypes differ')
    decoder=_load_scientific(Path('.'),SimpleNamespace(metadata=raw),values=state.model)
    # Legacy owned v1 recorded the backend but omitted PyTorch's explicit MPS
    # index. Its only supported Metal device is0; verify that exact old digest.
    device=schema.get('device','mps:0' if schema['backend']=='torch_mps' else 'cpu')
    if torch.device(device).type!=schema['backend'].removeprefix('torch_'):raise ValueError('checkpoint source device/backend mismatch')
    if _saved_digest(decoder,state.model,device_override=device)!=state.candidate_digest:raise ValueError('checkpoint candidate digest mismatch')
    return decoder

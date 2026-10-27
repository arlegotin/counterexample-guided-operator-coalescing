"""Fresh-process generation using artifact code and explicitly installed dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_export import api, model, runtime

pytestmark = pytest.mark.cpu

# Qualification utility, deliberately outside production runtime classes.
OFFLINE_SCRIPT = r'''
import json, sys, socket, os, site
from pathlib import Path
root = Path(sys.argv[1]).resolve()
blocked = Path(sys.argv[2]).resolve()
backend = sys.argv[3]
os_denial = sys.argv[4] == '1'
extra_blocked = tuple(Path(p).resolve() for p in json.loads(sys.argv[5]))
sys.path.insert(0,str(root/'runtime/poc-runtime.zip'))
dependencies = tuple(Path(p).resolve() for p in site.getsitepackages())
def guard(event, args):
    if event.startswith('socket.'): raise RuntimeError('network denied')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        if (p.is_relative_to(blocked) or any(p.is_relative_to(d) for d in extra_blocked)) and not p.is_relative_to(root) and not any(p.is_relative_to(d) for d in dependencies):
            raise RuntimeError('repository/source/training paths denied: ' + str(p))
sys.addaudithook(guard)
try:
    open(blocked/'README.md').read()
except RuntimeError:
    pass
else:
    raise RuntimeError('source denial negative control failed')
if os_denial:
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.open(os.fsencode(blocked/'README.md'), 0)
    if fd >= 0:
        libc.close(fd)
        raise RuntimeError('OS source denial negative control failed')
import huggingface_hub
huggingface_hub.hf_hub_download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('download denied'))
huggingface_hub.snapshot_download = huggingface_hub.hf_hub_download
from poc.bundle import load_bundle, measure_resident
from poc.contracts import Batch
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(root/'tokenizer/tokenizer.json'))
ids = tok.encode('t1 t4 t7' if tok.token_to_id('t1') is not None else 'A small model reads a sentence.').ids
m = load_bundle(root, backend=backend, offline=True)
if backend == 'torch':
    import torch
    torch.set_num_threads(4)
    def batch(tokens, start):
        x = torch.tensor([tokens]); return Batch(x, torch.ones_like(x,dtype=torch.bool), torch.ones((1,len(tokens)-1),dtype=torch.bool),torch.arange(start,start+len(tokens))[None],('offline',),('offline',))
else:
    import mlx.core as mx
    def batch(tokens, start):
        x = mx.array([tokens],dtype=mx.int32); return Batch(x,mx.ones_like(x,dtype=mx.bool_),mx.ones((1,len(tokens)-1),dtype=mx.bool_),mx.arange(start,start+len(tokens))[None],('offline',),('offline',))
c = m.new_cache(); logits = m.logits(batch(ids,0),cache=c)
initial_ids = list(ids)
for _ in range(3):
    token = int(logits[0,-1].argmax().item()); ids.append(token)
    logits = m.logits(batch([token],c.length),cache=c)
print(json.dumps({'ids':ids,'initial_ids':initial_ids,'owners':measure_resident(m).owner_count,
                  'module':__import__('poc').__file__,'backend':m.manifest().backend,
                  'source_denied':True,'os_source_denied':os_denial,'download_denied':True,
                  'resident_bytes':measure_resident(m).resident_bytes,
                  'peak_rss_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
                  'mlx_peak_memory_bytes':mx.get_peak_memory() if backend=='mlx' else None}))
'''


def run_offline_bundle(directory, workdir, backend='torch', *, os_denial=False, blocked_source_dirs=()):
    repo = Path(__file__).resolve().parents[1]
    cache = Path(workdir)/('empty-cache-'+backend); cache.mkdir(exist_ok=True)
    assert not list(cache.iterdir()), 'offline proof must start with an empty model cache'
    env = dict(os.environ, HF_HOME=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               XDG_CACHE_HOME=str(cache), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable,'-I','-B','-c',OFFLINE_SCRIPT,str(directory),str(repo),backend,str(int(os_denial)),json.dumps(list(blocked_source_dirs))]
    if os_denial:
        # OS denial catches native fopen/mmap too; dependency environment is the
        # explicit install boundary and remains readable. No repository code/data.
        profile = '(version 1)(allow default)(deny network*)(deny file-read* (subpath '+json.dumps(str(repo))+'))'
        profile += '(allow file-read-metadata (subpath '+json.dumps(str(repo))+'))'
        profile += '(allow file-read* (subpath '+json.dumps(str(repo/'.venv'))+'))'
        for source in blocked_source_dirs: profile += '(deny file-read* (subpath '+json.dumps(str(source))+'))'
        profile += '(allow file-read* (subpath '+json.dumps(str(Path(directory).resolve()))+'))'
        command = ['/usr/bin/sandbox-exec','-p',profile,*command]
    result = subprocess.run(command,cwd=workdir,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output['module'].startswith(str(directory))
    assert output['source_denied'] and output['download_denied']
    if os_denial: assert output['os_source_denied']
    assert len(output['ids']) == len(output['initial_ids'])+3
    return output


def test_fresh_process_offline_bundle_generation(tmp_path):
    io = api(); source = model('2'); ref = io.write_bundle(source,tmp_path/'bundle',runtime(tmp_path))
    output = run_offline_bundle(ref.directory,tmp_path)
    assert output['initial_ids'] == [1,4,7]
    assert output['owners'] == len(list(source.parameters()))


def test_offline_denial_allows_only_artifact_when_stored_inside_repository(tmp_path,monkeypatch):
    import test_offline_bundle as helper
    fake=tmp_path/'repository';(fake/'tests').mkdir(parents=True);(fake/'README.md').write_text('source must be denied')
    monkeypatch.setattr(helper,'__file__',str(fake/'tests/test_offline_bundle.py'))
    ref=api().write_bundle(model('0'),fake/'artifacts/payload',runtime(tmp_path))
    assert helper.run_offline_bundle(ref.directory,tmp_path)['source_denied']

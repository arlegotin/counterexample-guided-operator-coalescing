"""Measure isolated readout-loss CPU process peaks; no full-model memory claim.

Reproduce: python provenance/measure_loss_memory.py --out <JSON path>
Each fresh child uses one CPU thread; full/chunked graphs, inputs and seeds match.
Every vocabulary tensor is <=32 MiB; the peak includes imports/allocator overhead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(mode, chunk_size):
    import torch
    from torch import nn
    from poc.contracts import Batch
    from poc.losses import backward_distillation
    torch.set_num_threads(1)
    class Projection(nn.Module):
        def __init__(self, seed):
            super().__init__()
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.embedding = nn.Embedding(16384, 16)
                self.upstream = nn.Parameter(torch.randn(16,16)*.05)
            self.max_readout_positions = 0
        def final_hidden(self, batch):
            return torch.tanh(self.embedding(batch.input_ids) @ self.upstream)
        def readout(self, h):
            self.max_readout_positions = max(self.max_readout_positions, h.numel()//h.shape[-1])
            return h @ self.embedding.weight.T
    teacher, student = Projection(1).requires_grad_(False), Projection(2)
    ids = torch.arange(513)[None]
    b = Batch(ids, torch.ones_like(ids,dtype=torch.bool), torch.ones((1,512),dtype=torch.bool),
              ids.clone(), ('stress',), ('stress',))
    units = 1 if sys.platform == 'darwin' else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*units
    if mode == 'full':
        with torch.no_grad():
            t = teacher.readout(teacher.final_hidden(b)[:,:-1]).log_softmax(-1)
        s = student.readout(student.final_hidden(b)[:,:-1]).log_softmax(-1)
        kl = (t.exp()*(t-s)).sum()
        nll = -s.gather(-1, ids[:,1:,None]).sum()
        ((kl+.1*nll)/512).backward()
        metrics = dict(kl=kl.item(),nll=nll.item())
    else:
        result = backward_distillation(teacher,student,b,alpha=.1,denominator=512,chunk_size=chunk_size)
        metrics = dict(kl=result.kl,nll=result.nll)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*units
    print(json.dumps(dict(mode=mode,chunk_size=chunk_size,peak_rss_bytes=peak,
        baseline_high_water_bytes=before,high_water_increase_bytes=max(0,peak-before),
        max_readout_positions=student.max_readout_positions,**metrics)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('full','chunked'))
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.chunk)
        return
    if args.out is None:
        parser.error('--out is required for the parent measurement')
    rows=[]
    for repeat in range(2):
        for mode,chunk in [('full',512),('chunked',32),('chunked',128),('chunked',512)]:
            child=subprocess.run([sys.executable,__file__,'--worker',mode,'--chunk',str(chunk)],
                                  check=True,capture_output=True,text=True,cwd=ROOT)
            rows.append(dict(repeat=repeat,**json.loads(child.stdout)))
    report=dict(schema_version=1,date='2026-10-27',scope='isolated CPU FP32 projection/tied-readout loss only; not full-model memory',
        measurement='fresh-child ru_maxrss process high-water bytes, two repeats, one CPU thread',
        fixture=dict(batch_size=1,prediction_positions=512,vocabulary=16384,hidden_width=16,alpha=.1),rows=rows,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('poc/losses.py','poc/contracts.py','provenance/measure_loss_memory.py')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()

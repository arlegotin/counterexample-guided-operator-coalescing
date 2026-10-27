"""Reproduce real-weight near-identity controls using only pinned local assets.

Example: .venv/bin/python provenance/measure_teacher_controls.py --device cpu
Outputs are private qualification artifacts; no training or protected data.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import time

import torch

from poc.evaluate import check_kl
from poc.losses import stream_metrics
from poc.teacher import AssetManifest,load_teacher,sha256_file
from poc.teacher_validation import _batch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','mps'),required=True)
    parser.add_argument('--assets',type=Path,default=Path('data/prepared-135m-v2/teacher-assets.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--numerics',type=Path,required=True)
    args=parser.parse_args()
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('explicit MPS unavailable; no fallback')
    asset=AssetManifest.load(args.assets,expected_sha256='9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083')
    base=load_teacher(asset)
    ref=deepcopy(base).double();ref_candidate=deepcopy(ref)
    base.to(args.device);candidate=deepcopy(base)
    with torch.no_grad():
        candidate.layers[15].mlp.down_proj.weight[0,0].add_(1e-4)
        ref_candidate.layers[15].mlp.down_proj.weight[0,0].add_(1e-4)
    numeric=json.loads(args.numerics.read_text());rows=[]
    for length in (128,512):
        ids=tuple((i*37+11)%base.config.vocab_size for i in range(length))
        started=time.perf_counter()
        reference=stream_metrics(ref,ref_candidate,_batch(ids),32)
        scientific=stream_metrics(base,candidate,_batch(ids,device=args.device),32)
        check_kl(scientific,negative_kl_floor=numeric['negative_kl_floor_per_token'])
        rows.append(dict(context_tokens=length,valid_targets=length-1,
            scientific_kl_sum=scientific.kl,cpu_fp64_kl_sum=reference.kl,
            scientific_raw_kl_per_token=scientific.kl/(length-1),
            fp64_kl_per_token=reference.kl/(length-1),
            kl_error_per_token=abs(scientific.kl-reference.kl)/(length-1),
            wall_seconds=time.perf_counter()-started))
    root=Path(__file__).resolve().parents[1]
    result=dict(schema_version=1,scope='actual pinned teacher plus single-weight candidate perturbation; no data selection',
        backend='torch_'+args.device,perturbation='layer15 down_proj[0,0] += 1e-4; frozen candidate, no optimizer',
        numerical_floor_per_token=numeric['negative_kl_floor_per_token'],
        asset_manifest_sha256=asset.expected_sha256,numerics_sha256=sha256_file(args.numerics),
        source_sha256={str(p.relative_to(root)):sha256_file(p) for p in [Path(__file__).resolve(),root/'poc/runtime/torch_decoder.py',root/'poc/losses.py']},
        teacher_forward_calls=8,rows=rows,status='passed')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result['rows']))


if __name__=='__main__': main()

"""Offline actual bank/tied-head loss calibration against independent FP64 math.

Reproduce on native Metal: python provenance/measure_loss_numerics.py --out <path>
Transfers all MPS tensors to CPU BEFORE double conversion. No asset download.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))


def main():
    import torch
    from poc.losses import backward_distillation
    from test_chunked_loss import batch, models, full
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert torch.backends.mps.is_available(), 'requires native Metal qualification'
    torch.set_num_threads(1)
    rows=[]
    for alpha in (0.,.1,1.):
        teacher,reference=models()
        inputs=[batch(),batch(5)]
        expected=full(teacher,reference,inputs,alpha)
        count=sum(int(b.target_mask.sum()) for b in inputs)
        gradients={name:p.grad.clone() for name,p in reference.named_parameters() if p.requires_grad}
        max_gradient=max(g.abs().max().item() for g in gradients.values())
        for chunk in (1,4,32):
            row=dict(alpha=alpha,chunk_size=chunk,valid_targets=count,
                     reference_kl_sum=expected[0],reference_nll_sum=expected[1],
                     reference_gradient_max_abs=max_gradient)
            for device in ('cpu','mps'):
                t,s=models(device=device,dtype=torch.float32)
                t.requires_grad_(True)
                sums=[backward_distillation(t,s,batch(length,device=device),alpha=alpha,
                      denominator=count,chunk_size=chunk) for length in (8,5)]
                result=sums[0]+sums[1]
                errors={name:(p.grad.detach().cpu().double()-gradients[name]).abs().max().item()
                        for name,p in s.named_parameters() if p.requires_grad}
                assert all(p.grad is None for p in t.parameters())
                row[device+'_fp32']=dict(kl_sum=result.kl,nll_sum=result.nll,
                    kl_abs_error=abs(result.kl-expected[0]),nll_abs_error=abs(result.nll-expected[1]),
                    max_trainable_gradient_abs_error=max(errors.values()),gradient_error_by_owner=errors)
            rows.append(row)
    eps=torch.finfo(torch.float32).eps
    tolerances=dict(
        kl_sum_atol=8*max(max(r['cpu_fp32']['kl_abs_error'],eps*abs(r['reference_kl_sum'])) for r in rows),
        nll_sum_atol=8*max(max(r['cpu_fp32']['nll_abs_error'],eps*abs(r['reference_nll_sum'])) for r in rows),
        gradient_atol=8*max(max(r['cpu_fp32']['max_trainable_gradient_abs_error'],eps*r['reference_gradient_max_abs']) for r in rows))
    for row in rows:
        assert row['mps_fp32']['kl_abs_error'] <= tolerances['kl_sum_atol']
        assert row['mps_fp32']['nll_abs_error'] <= tolerances['nll_sum_atol']
        assert row['mps_fp32']['max_trainable_gradient_abs_error'] <= tolerances['gradient_atol']
    report=dict(schema_version=1,date='2026-10-27',scope='3-depth width16/vocab13 bank rank2, two ragged microbatches, optional tied owner enabled; no pretrained qualification',
        torch_version=torch.__version__,decoder_qualification_reference='provenance/decoder-tiny-numerics-2026-10-27.json',
        procedure='independent full-logit CPU FP64 reference; scientific FP32; teacher target requires_grad tested but frozen by losses',
        tolerance_policy='8 times max(measured CPU FP32 error, FP32 epsilon times reference peak), across alpha0/.1/1 and chunks1/4/32; absolute only',
        tolerances=tolerances,rows=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('poc/losses.py','poc/contracts.py','poc/bank.py','poc/runtime/torch_decoder.py','tests/test_chunked_loss.py','provenance/measure_loss_numerics.py')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(tolerances=tolerances,maximum_mps_errors={key:max(r['mps_fp32'][key] for r in rows)
            for key in ('kl_abs_error','nll_abs_error','max_trainable_gradient_abs_error')}),indent=2))


if __name__=='__main__':
    main()

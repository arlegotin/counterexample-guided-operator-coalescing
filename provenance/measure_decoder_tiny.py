"""Offline reproducible numeric evidence; no pretrained weights or data."""
from dataclasses import asdict
import json
import torch
from transformers import LlamaForCausalLM
from poc.contracts import Batch
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder


def batch(device):
    ids = torch.tensor([[1,4,7,2,3]], device=device)
    return Batch(ids, torch.ones_like(ids,dtype=torch.bool), torch.ones((1,4),dtype=torch.bool,device=device),
                 torch.arange(7,12,device=device)[None], ('d',), ('c',))


def error(a,b):
    return (a.detach().cpu().double()-b.detach().cpu().double()).abs().max().item()


report = dict(schema_version=1, date='2026-10-27', scope='offline tiny eager Llama only; no assets or gate implied',
    normalization='true CPU FP64 RMS/RoPE/softmax reference; scientific FP32 follows Llama FP32 reductions',
    procedure='three seeds times three actual config shapes; three repeated forwards; full versus incremental; central finite difference step=1e-6',
    qualification_reference='provenance/qualification-2026-10-27.json', rows=[])
for cfg in [TinyConfig(), TinyConfig(hidden_size=24,num_attention_heads=3,num_key_value_heads=1,intermediate_size=31),
            TinyConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=8)]:
    for seed in (7,17,29):
        ref = make_tiny_decoder(cfg,seed=seed,dtype=torch.float64)
        cpu = make_tiny_decoder(cfg,seed=seed,dtype=torch.float64).float()
        mps = make_tiny_decoder(cfg,seed=seed,dtype=torch.float64).to(device='mps',dtype=torch.float32)
        x = batch('cpu')
        expected = ref.logits(x)
        hf = LlamaForCausalLM(cpu.config).eval()
        hf.model.embed_tokens, hf.model.layers, hf.model.norm, hf.lm_head = cpu.embed_tokens,cpu.layers,cpu.norm,cpu.lm_head
        with torch.no_grad():
            upstream = hf(x.input_ids,attention_mask=x.attention_mask,position_ids=x.positions,use_cache=False).logits
        row=dict(config=asdict(cfg),seed=seed,cpu_fp32_vs_upstream_max_abs_error=error(cpu.logits(x),upstream))
        results={}
        for name,m in [('cpu_fp64',ref),('cpu_fp32',cpu),('mps_fp32',mps)]:
            xb=batch(m.device)
            values=[m.logits(xb) for _ in range(3)]
            cached=m.new_cache(start_position=7)
            pieces=[]
            for i in range(5):
                b=Batch(xb.input_ids[:,i:i+1],xb.attention_mask[:,i:i+1],xb.target_mask[:,:0],xb.positions[:,i:i+1],('d',),('c',))
                pieces.append(m.logits(b,cache=cached))
            bound=m.prefix_boundary(xb,1)
            z=bound.u.detach().requires_grad_()
            objective=m.suffix_logits(z,bound).square().mean()
            grad=torch.autograd.grad(objective,z)[0]
            branch=m.suffix_logits(m.ff_residual(bound),bound)
            results[name]=dict(output=values[0],gradient=grad)
            row[name]=dict(forward_reference_error=error(values[0],expected),
                forward_repeat_error=max(error(value,values[0]) for value in values),
                full_cache_error=error(torch.cat(pieces,1),values[0]),
                boundary_reconstruction_error=error(branch,values[0]))
            if name=='cpu_fp64':
                perturb=torch.zeros_like(z);perturb[0,2,3]=1e-6
                fd=(m.suffix_logits(z.detach()+perturb,bound).square().mean()-m.suffix_logits(z.detach()-perturb,bound).square().mean())/2e-6
                row['finite_difference_abs_error']=abs(fd.item()-grad[0,2,3].item())
        for name in results:
            row[name]['gradient_reference_error']=error(results[name]['gradient'],results['cpu_fp64']['gradient'])
        row['reference_max_abs_output']=expected.abs().max().item()
        row['reference_max_abs_gradient']=results['cpu_fp64']['gradient'].abs().max().item()
        report['rows'].append(row)

eps=torch.finfo(torch.float32).eps
report['tolerance_policy']='8 times max(measured CPU FP32 vs true FP64 error, FP32 epsilon times reference peak), across nine preregistered tiny configs/seeds; absolute only; repeats/boundary reconstruction exact'
report['tolerances']={
    'forward_atol':8*max(max(r['cpu_fp32']['forward_reference_error'],eps*r['reference_max_abs_output']) for r in report['rows']),
    'gradient_atol':8*max(max(r['cpu_fp32']['gradient_reference_error'],eps*r['reference_max_abs_gradient']) for r in report['rows']),
    'upstream_fp32_atol':8*max(max(r['cpu_fp32_vs_upstream_max_abs_error'],eps*r['reference_max_abs_output']) for r in report['rows']),
}
for row in report['rows']:
    for name in ('cpu_fp32','mps_fp32'):
        assert row[name]['forward_reference_error'] <= report['tolerances']['forward_atol']
        assert row[name]['full_cache_error'] <= report['tolerances']['forward_atol']
        assert row[name]['gradient_reference_error'] <= report['tolerances']['gradient_atol']
        assert row[name]['forward_repeat_error']==0
        assert row[name]['boundary_reconstruction_error']==0
with open('provenance/decoder-tiny-numerics-2026-10-27.json','w') as out:
    json.dump(report,out,indent=2,sort_keys=True);out.write('\n')
print(json.dumps(dict(tolerances=report['tolerances'],max_mps_forward=max(r['mps_fp32']['forward_reference_error'] for r in report['rows']),
 max_mps_gradient=max(r['mps_fp32']['gradient_reference_error'] for r in report['rows']),max_finite_difference=max(r['finite_difference_abs_error'] for r in report['rows'])),indent=2))

"""Packed owners must remain compact, complete, and shared after deployment."""
from pathlib import Path
import pytest
from test_export import runtime


def api():
    from poc import quantize
    return quantize


@pytest.mark.cpu
def test_invalid_formats_and_exact_layout():
    q = api()
    spec = q.QuantizationSpec('rtn', 4, 64, '/tmp/unused')
    layout = q.packed_layout((128, 192), spec)
    assert layout == {'weight_shape': (128, 24), 'scale_shape': (128, 3), 'packed_bytes': 12288,
                      'scale_bytes': 1536, 'bias_bytes': 1536, 'padding_elements': 0}
    for bits, group, shape in [(1,64,(128,192)),(4,16,(128,192)),(4,64,(128,193)),(6,64,(128,192))]:
        with pytest.raises(ValueError): q.packed_layout(shape,q.QuantizationSpec('rtn',bits,group,'/tmp/unused'))


@pytest.mark.mlx
def test_actual_packing_reload_ties_and_bank_ownership(tmp_path):
    import torch
    import mlx.core as mx
    from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder
    from poc.bank import BankDecoder, canonicalize_partition
    from poc.bundle import write_bundle, verify_bundle, load_bundle, measure_bundle
    dense = make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2),seed=4,dtype=torch.float32)
    source = BankDecoder(dense,canonicalize_partition([[0,2],[1]],3),8,17)
    floating = write_bundle(source,tmp_path/'floating',runtime(tmp_path))
    result = api().quantize_rtn(floating,api().QuantizationSpec('rtn',4,64,str(tmp_path/'packed')),None)
    root=Path(result.bundle.directory); manifest=verify_bundle(root)
    assert result.payload_bytes == measure_bundle(root).payload_bytes == sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    native=load_bundle(root,backend='mlx')
    assert native.owner('embed_tokens.weight') is native.owner('lm_head.weight')
    assert len([o for o in manifest.metadata['owners'] if o['category']=='bank']) == 6
    assert all(o['dtype']=='bfloat16' for o in manifest.metadata['owners'] if o['category']=='residual')
    assert not any('layers.2.mlp.gate_proj.weight' in k for k in native.weights)
    assert native.resident_accounting().resident_bytes < sum(p.numel()*4 for p in source.parameters())
    from test_native_parity import native_batch
    from test_model_parity import batch
    b=native_batch(batch([[1,4,7,2,3]]))
    logits=native.logits(b); mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits)).item())
    cache=native.new_cache()
    parts=[native.logits(native_batch(batch([[t]],offset=i)),cache=cache) for i,t in enumerate([1,4,7,2,3])]
    assert mx.max(mx.abs(mx.concatenate(parts,axis=1)-logits)).item()<1e-4
    from test_offline_bundle import run_offline_bundle
    assert run_offline_bundle(root,tmp_path,backend='mlx')['download_denied']

@pytest.mark.mlx
@pytest.mark.parametrize('method',['gptq','awq','dynamic','dwq'])
def test_official_dense_adapters_export_transformed_packed_owners(tmp_path,method):
    import torch
    import mlx.core as mx
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.bundle import write_bundle,load_bundle,verify_bundle
    from poc.data.manifest import DataManifest,DocumentRecord
    dense=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2),seed=19,dtype=torch.float32)
    floating=write_bundle(dense,tmp_path/'source',runtime(tmp_path))
    record=tuple(DocumentRecord(str(i),'f','r','s',i,text=' '.join(['t1','t4','t7','t2']*6)) for i in range(4))
    fit=DataManifest(record,{str(i):'fit' for i in range(4)},{str(i):str(i) for i in range(4)},'s',str(tmp_path/'exposure.jsonl'))
    assert hasattr(api(),'quantize_dense'),'official dense calibration adapter missing'
    result=api().quantize_dense(floating,api().QuantizationSpec(method,4,64,str(tmp_path/'q'),calibration_count=2,length=16),fit,None)
    loaded=load_bundle(Path(result.bundle.directory),backend='mlx')
    assert loaded.owner('embed_tokens.weight') is loaded.owner('lm_head.weight')
    from test_native_parity import native_batch
    from test_model_parity import batch
    output=loaded.logits(native_batch(batch([[1,4,7,2]])));mx.eval(output)
    assert bool(mx.all(mx.isfinite(output)).item())
    assert verify_bundle(Path(result.bundle.directory)).metadata['kind']=='dense'


@pytest.mark.mlx
def test_payload_anchor_is_prescribed_sequence_svd_not_format_fixture(tmp_path):
    import torch
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.export import make_payload_anchor
    from poc.bundle import verify_bundle,measure_bundle,load_bundle
    dense=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=30,num_attention_heads=4,num_key_value_heads=2),seed=3,dtype=torch.float32)
    ref=make_payload_anchor(dense,runtime(tmp_path))
    metadata=verify_bundle(Path(ref.directory)).metadata
    assert tuple(tuple(g) for g in metadata['groups'])==tuple((i,i+1) for i in range(0,30,2))
    assert metadata['rank']==8
    assert len([o for o in metadata['owners'] if o['category']=='bank'])==45
    assert measure_bundle(Path(ref.directory)).payload_bytes==sum(p.stat().st_size for p in Path(ref.directory).rglob('*') if p.is_file())
    loaded=load_bundle(Path(ref.directory),backend='mlx')
    assert loaded.weights['layers.1.mlp.residuals.gate_proj.B'].dtype==loaded.mx.bfloat16
    assert loaded.mx.max(loaded.mx.abs(loaded.weights['layers.1.mlp.residuals.gate_proj.B'])).item()>0


@pytest.mark.mlx
def test_native_near_identity_controls_are_serializable_and_backend_specific():
    import json
    import mlx.core as mx
    from poc.quantization_stage import _native_numerical_controls
    class Teacher:
        def readout(self,hidden): return hidden
    hidden=mx.sin(mx.arange(512*41,dtype=mx.float32).reshape(1,512,41))
    controls=_native_numerical_controls(Teacher(),hidden)
    assert controls['negative_kl_floor_per_target']>0
    assert len(controls['rows'])==6
    assert all(r['raw_native_kl_per_target']==0 for r in controls['rows'] if r['epsilon']==0)
    assert 'mlx-metal' in controls['profile']
    json.dumps(controls,allow_nan=False)


@pytest.mark.mlx
def test_official_bridge_uses_actual_rope_parameters_from_hf5_config(tmp_path):
    import json,torch
    import mlx.core as mx
    from poc.bundle import write_bundle,load_bundle,_canonical,_sha
    from poc.quantize import _official_model,_native_batch
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    source=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2,rope_theta=100000,max_position_embeddings=1024),seed=3,dtype=torch.float32)
    ref=write_bundle(source,tmp_path/'source',runtime(tmp_path));root=Path(ref.directory)
    raw=json.loads((root/'manifest.json').read_text())
    raw['config'].pop('rope_theta',None);raw['config']['rope_parameters']={'rope_type':'default','rope_theta':100000}
    (root/'config.json').write_bytes(_canonical(raw['config']))
    raw['file_entries']['config.json']={'bytes':(root/'config.json').stat().st_size,'sha256':_sha(root/'config.json')}
    (root/'manifest.json').write_bytes(_canonical(raw))
    native=load_bundle(root,backend='mlx');official,_=_official_model(native)
    b=_native_batch([[1,4,7,2]*128]);left=native.logits(b);right=official(b.input_ids);mx.eval(left,right)
    assert mx.max(mx.abs(left-right)).item()<1e-6


@pytest.mark.mlx
def test_rtn_embedding_upgrade_counts_bytes_and_keeps_one_tied_owner(tmp_path):
    import torch
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.bundle import write_bundle,load_bundle
    dense=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2),seed=2,dtype=torch.float32)
    source=write_bundle(dense,tmp_path/'source',runtime(tmp_path))
    q=api()
    low=q.quantize_rtn(source,q.QuantizationSpec('rtn',2,64,str(tmp_path/'low')),None)
    high=q.quantize_rtn(source,q.QuantizationSpec('rtn',2,64,str(tmp_path/'high'),embedding_bits=4),None)
    native=load_bundle(Path(high.bundle.directory),backend='mlx')
    assert high.payload_bytes>low.payload_bytes
    assert native.owner('lm_head.weight') is native.owner('embed_tokens.weight')
    assert native.owner('lm_head.weight').bits==4
    assert native.owner('layers.0.mlp.gate_proj.weight').bits==2


@pytest.mark.cpu
def test_quantizer_failed_work_is_charged_once():
    from poc.quantize import _costed
    from poc.contracts import CostRecord
    class Recorder:
        def __init__(self): self.rows=[]
        def check(self,cost): assert isinstance(cost,CostRecord)
        def record(self,cost): self.rows.append(cost)
    recorder=Recorder()
    @_costed
    def failed(bundle,spec,recorder): raise ValueError('measured failed operation')
    with pytest.raises(ValueError): failed(None,None,recorder)
    assert len(recorder.rows)==1 and recorder.rows[0].wall_seconds>0
    assert recorder.rows[0].memory_bytes>0

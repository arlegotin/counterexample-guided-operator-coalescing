"""Actual native accounting/export controls; run only in the serial MLX slot."""
from pathlib import Path
import json
import pytest


def float_norm_owner_names(metadata):
    # The schema groups RMSNorm vectors under non_ffn, not a norm category.
    return [row['name'] for row in metadata['owners'] if len(row['shape'])==1]


@pytest.mark.mlx
@pytest.mark.parametrize('method,bits',[(method,bits) for method in ('gptq','awq','dwq') for bits in (4,2)])
def test_actual_v2_dense_complete_context_accounting_and_offline_export(tmp_path,method,bits):
    import torch
    import mlx.core as mx
    from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
    from poc.bundle import write_bundle,load_bundle,measure_bundle
    from poc.quantize import quantize_dense,QuantizationSpec
    from poc.data.manifest import DataManifest,DocumentRecord
    from test_export import runtime
    torch.set_num_threads(1)
    dense=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,num_attention_heads=4,
        num_key_value_heads=2,max_position_embeddings=1024),seed=19,dtype=torch.float32)
    floating=write_bundle(dense,tmp_path/'source',runtime(tmp_path))
    records=tuple(DocumentRecord(str(i),'owned','r','s',i,text=' '.join(['t1','t4','t7','t2']*128)) for i in range(4))
    fit=DataManifest(records,{r.document_id:'fit' for r in records},{r.document_id:'owned-'+r.document_id for r in records},'owned',str(tmp_path/'exposure.jsonl'))
    class Recorder:
        def __init__(self):self.rows=[]
        def check(self,cost):pass
        def record(self,cost):self.rows.append(cost)
    recorder=Recorder();target=tmp_path/'packed'
    result=quantize_dense(floating,QuantizationSpec(method,bits,64,str(target),calibration_count=2,
        internal_validation_count=2,length=512,seed=17,embedding_bits=4),fit,recorder,accounting_profile='dense-native-contexts-v2')
    evidence=json.loads(Path(str(target)+'.quantizer-evidence.json').read_text())
    assert evidence['accounting_profile']=='dense-native-contexts-v2' and evidence['original_context_tokens']==512
    assert evidence['tokenization_policy']=='no-special-tokens-v2'
    assert len(evidence['internal_train_component_ids'])==2
    assert evidence['initial_official_bridge_max_abs_error']<=.005 and evidence['packed_common_runtime_max_abs_error']<=.01
    assert evidence['default_loader_denial_at_imported_binding'] and evidence['socket_connect_denied']
    totals={key:sum(getattr(row,key) for row in recorder.rows) for key in ('updates','valid_target_presentations',
        'raw_corpus_target_presentations','teacher_forward_calls','student_forward_calls','student_backward_calls')}
    counts=evidence['actual_upstream_counts']
    if method=='dwq':
        assert totals=={'updates':2,'valid_target_presentations':6*511,'raw_corpus_target_presentations':16*511,
            'teacher_forward_calls':8,'student_forward_calls':8,'student_backward_calls':2}
        assert counts['logical_optimizer_raw_target_presentations']==1022
        scope=counts['dwq_trainability']
        assert scope['scope']=='affine_scales_biases_only_v2'
        assert scope['owners'] and all(row['name'].endswith(('.scales','.biases')) for row in scope['owners'])
        assert len(evidence['internal_valid_component_ids'])==2
    else:
        assert totals['updates']==totals['student_backward_calls']==totals['valid_target_presentations']==0
        assert evidence['internal_valid_component_ids']==[]
    assert totals['raw_corpus_target_presentations']==evidence['total_model_context_raw_presentations']
    assert result.payload_bytes==measure_bundle(target).payload_bytes
    model=load_bundle(target,backend='mlx')
    assert model.owner('embed_tokens.weight') is model.owner('lm_head.weight')
    if method=='dwq':
        original=load_bundle(Path(floating.directory),backend='mlx')
        from poc.bundle import verify_bundle
        norm_names=float_norm_owner_names(verify_bundle(Path(floating.directory)).metadata)
        assert norm_names
        for name in norm_names:
            assert bool(mx.array_equal(model.owner(name),original.owner(name)).item()),name
    from test_offline_bundle import run_offline_bundle
    proof_dir=tmp_path/'offline-proof';proof_dir.mkdir()
    proof=run_offline_bundle(target,proof_dir,backend='mlx',blocked_source_dirs=(floating.directory,))
    assert proof['download_denied'] and proof['source_denied']

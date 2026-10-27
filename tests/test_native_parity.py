"""Run explicitly on qualified Metal; never certify native execution with CPU fallback."""
from pathlib import Path
import numpy as np
import pytest
import torch
from test_export import api, model, runtime
from test_model_parity import batch

pytestmark = pytest.mark.mlx


def native_batch(b):
    import mlx.core as mx
    from poc.contracts import Batch
    return Batch(*(mx.array(t.numpy()) for t in (b.input_ids,b.attention_mask,b.target_mask,b.positions)),b.document_ids,b.cluster_ids)


@pytest.mark.parametrize('kind',['dense','0','2'])
def test_mlx_full_cached_continuations_and_resident_owners(tmp_path,kind):
    import importlib.util
    assert importlib.util.find_spec('poc.runtime.mlx_decoder') is not None, 'native canonical runtime is missing'
    import mlx.core as mx
    assert mx.metal.is_available(), 'qualified MLX Metal is required'
    io = api(); source = model(kind)
    ref = io.write_bundle(source,tmp_path/'bundle',runtime(tmp_path))
    native = io.load_bundle(Path(ref.directory),backend='mlx')
    b = batch(offset=11)
    expected = source.logits(b).detach().numpy()
    actual = native.logits(native_batch(b)); mx.eval(actual)
    np.testing.assert_allclose(np.array(actual),expected,atol=4e-5,rtol=0)
    cache = native.new_cache(start_position=11)
    parts = [native.logits(native_batch(batch([[token]],offset=11+i)),cache=cache) for i,token in enumerate([1,4,7,2,3])]
    np.testing.assert_allclose(np.array(mx.concatenate(parts,axis=1)),expected,atol=4e-5,rtol=0)
    assert io.measure_resident(native).owner_count == len(list(source.parameters()))
    assert native.weights['embed_tokens.weight'] is native.owner('lm_head.weight')
    from poc.contracts import ContinuationRequest
    requests = [ContinuationRequest((1,4),(7,2),'fixture-prompt-v1','a'),ContinuationRequest((1,4),(3,2),'fixture-prompt-v1','b')]
    torch_scores = source.score_continuations(requests); native_scores = native.score_continuations(requests)
    np.testing.assert_allclose([s.log_likelihood for s in native_scores],[s.log_likelihood for s in torch_scores],atol=3e-5,rtol=0)
    assert np.argmax([s.log_likelihood for s in native_scores]) == np.argmax([s.log_likelihood for s in torch_scores])
    boundary = native.prefix_boundary(native_batch(b),1)
    np.testing.assert_allclose(np.array(native.suffix_logits(native.ff_residual(boundary),boundary)),expected,atol=4e-5,rtol=0)
    native.weights['norm.weight'] = native.weights['norm.weight'] + .1
    with pytest.raises(ValueError,match='state|cache'): native.logits(native_batch(batch([[2]],offset=16)),cache=cache)


def test_native_fresh_process_offline_generation(tmp_path):
    from test_offline_bundle import run_offline_bundle
    io = api(); ref = io.write_bundle(model('2'),tmp_path/'bundle',runtime(tmp_path))
    output = run_offline_bundle(ref.directory,tmp_path,backend='mlx')
    assert output['backend'] == 'mlx_metal'
    assert output['initial_ids'] == [1,4,7]



def test_native_ordinary_import_preserves_payload_bytes(tmp_path):
    from test_export import ordinary_artifact_inference
    io = api(); ref = io.write_bundle(model('0'),tmp_path/'bundle',runtime(tmp_path))
    ordinary_artifact_inference(ref.directory,tmp_path,backend='mlx')


def test_native_mask_causality_and_inplace_cache_invalidation(tmp_path):
    import mlx.core as mx
    from poc.contracts import Batch
    io = api(); source = model('2'); ref = io.write_bundle(source,tmp_path/'bundle',runtime(tmp_path))
    native = io.load_bundle(Path(ref.directory),backend='mlx')
    b = batch([[1,4,7,2],[3,2,0,0]],offset=7)
    b.attention_mask[1,2:] = False; b.target_mask[1,1:] = False
    expected = source.logits(b).detach().numpy()
    np.testing.assert_allclose(np.array(native.logits(native_batch(b))),expected,atol=4e-5,rtol=0)
    edited = batch([[1,4,7,8],[3,2,0,0]],offset=7)
    edited.attention_mask[1,2:] = False; edited.target_mask[1,1:] = False
    np.testing.assert_array_equal(np.array(native.logits(native_batch(edited)))[:,:3],np.array(native.logits(native_batch(b)))[:,:3])
    cache = native.new_cache(start_position=7)
    native.logits(native_batch(batch([[1]],offset=7)),cache=cache)
    owner = native.weights['norm.weight']; identity = id(owner)
    owner[0] = owner[0]+.1
    assert id(native.weights['norm.weight']) == identity
    with pytest.raises(ValueError,match='state|cache'):
        native.logits(native_batch(batch([[4]],offset=8)),cache=cache)


def test_native_rejects_later_cpu_default_instead_of_substitution(tmp_path):
    import mlx.core as mx
    io = api(); ref = io.write_bundle(model('0'),tmp_path/'bundle',runtime(tmp_path))
    native = io.load_bundle(Path(ref.directory),backend='mlx')
    b = native_batch(batch())
    try:
        mx.set_default_device(mx.cpu)
        with pytest.raises(RuntimeError,match='GPU|CPU|Metal'):
            native.logits(b)
    finally:
        mx.set_default_device(mx.gpu)

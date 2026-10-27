"""Catch boundary ordering, cached RoPE offsets, and incorrect tied readout."""
from dataclasses import replace
import json
from pathlib import Path
import importlib
import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec('poc.runtime') is not None, 'tiny decoder runtime is missing'
    return importlib.import_module('poc.runtime.torch_decoder')


def numeric_tolerances():
    path = Path(__file__).resolve().parents[1] / 'provenance/decoder-tiny-numerics-2026-10-27.json'
    return json.loads(path.read_text())['tolerances']


def batch(ids=None, *, offset=0, device='cpu'):
    from poc.contracts import Batch
    ids = torch.tensor([[1, 4, 7, 2, 3]] if ids is None else ids, device=device)
    b, s = ids.shape
    return Batch(ids, torch.ones_like(ids, dtype=torch.bool),
                 torch.ones((b, s-1), dtype=torch.bool, device=device),
                 torch.arange(offset, offset+s, device=device).expand(b, -1),
                 tuple(f'd{i}' for i in range(b)), tuple(f'c{i}' for i in range(b)))


@pytest.mark.cpu
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_prefix_ffn_suffix_reconstructs_first_middle_last_depth(depth):
    m = api().make_tiny_decoder(api().TinyConfig(), seed=13, dtype=torch.float64)
    x = batch(offset=7)
    boundary = m.prefix_boundary(x, depth)
    torch.testing.assert_close(m.suffix_logits(m.ff_residual(boundary), boundary), m.logits(x), atol=0, rtol=0)
    assert m.lm_head.weight is m.embed_tokens.weight
    assert boundary.batch is x
    assert m.readout(m.final_hidden(x)[:, :2]).shape == (1, 2, m.config.vocab_size)


@pytest.mark.cpu
@pytest.mark.parametrize('offset', [0, 11])
def test_full_matches_multitoken_and_single_token_cache_at_offset(offset):
    m = api().make_tiny_decoder(api().TinyConfig(), seed=8, dtype=torch.float64)
    x = batch(offset=offset)
    expected = m.logits(x)
    cache = m.new_cache(start_position=offset)
    parts = []
    for start, end in [(0, 2), (2, 3), (3, 5)]:
        part = batch(x.input_ids[:, start:end].tolist(), offset=offset+start)
        parts.append(m.logits(part, cache=cache))
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=2e-15, rtol=2e-14)
    assert cache.length == 5 and cache.start_position == offset
    assert len({id(c) for c in cache.layers}) == m.config.num_hidden_layers
    assert all(c.key.shape[-2] == 5 for c in cache.layers)
    with pytest.raises(ValueError, match='position'):
        m.logits(batch([[3]], offset=offset), cache=cache)


@pytest.mark.cpu
@pytest.mark.parametrize('heads,kv,width', [(4, 2, 16), (3, 1, 24)])
def test_eager_decoder_matches_independent_transformers_llama(heads, kv, width):
    from transformers import LlamaConfig, LlamaForCausalLM
    with torch.random.fork_rng():
        torch.manual_seed(22)
        hf = LlamaForCausalLM(LlamaConfig(vocab_size=23, hidden_size=width, intermediate_size=31,
             num_hidden_layers=3, num_attention_heads=heads, num_key_value_heads=kv,
             max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.0,
             attn_implementation='eager')).eval()
    m = api().TorchDecoder.from_transformers(hf)
    x = batch(offset=9)
    with torch.no_grad():
        ref = hf(input_ids=x.input_ids, attention_mask=x.attention_mask,
                 position_ids=x.positions, use_cache=False).logits
    torch.testing.assert_close(m.logits(x), ref, atol=0, rtol=0)


@pytest.mark.cpu
def test_manifest_and_inference_cache_track_actual_weight_state():
    m = api().make_tiny_decoder(api().TinyConfig(), seed=5, dtype=torch.float64)
    cache = m.new_cache()
    manifest = m.manifest()
    assert manifest.backend == 'torch_cpu' and len(manifest.checkpoint_digest) == 64
    with torch.no_grad():
        m.layers[0].mlp.gate_proj.weight.add_(0.1)
    assert m.manifest().checkpoint_digest != manifest.checkpoint_digest
    with pytest.raises(ValueError, match='digest|state'):
        m.logits(batch(), cache=cache)


@pytest.mark.cpu
def test_planted_oracle_permutations_and_independent_control():
    from poc.synthetic import make_planted_teacher
    f = make_planted_teacher(29)
    assert f.reuse_map == (0, 1, 0, 0, 1, 1, 0, 1)
    x = batch()
    assert f.teacher.config.hidden_size == 32 and f.teacher.config.intermediate_size == 64
    for depth, owner in enumerate(f.reuse_map):
        u = f.teacher.prefix_boundary(x, depth)
        torch.testing.assert_close(f.teacher.ff_residual(u, f.canonical_operators[owner]),
                                   f.teacher.ff_residual(u), atol=2e-15, rtol=2e-14)
    torch.testing.assert_close(f.oracle.logits(x), f.teacher.logits(x), atol=3e-15, rtol=3e-14)
    assert (f.teacher.logits(x) - f.independent_teacher.logits(x)).abs().max() > 1e-4
    assert torch.linalg.matrix_rank(f.teacher.lm_head.weight) == 32


@pytest.mark.cpu
def test_continuations_score_only_continuation_tokens_with_shared_readout():
    from poc.contracts import ContinuationRequest
    m = api().make_tiny_decoder(api().TinyConfig(), seed=7, dtype=torch.float64)
    requests = [ContinuationRequest((1, 4), (2, 3), 'policy', 'a'),
                ContinuationRequest((1,), (2,), 'policy', 'b')]
    scores = m.score_continuations(requests)
    ref = m.logits(batch([[1,4,2,3]]))[0].log_softmax(-1)
    torch.testing.assert_close(torch.tensor(scores[0].log_likelihood, dtype=torch.float64), ref[1,2]+ref[2,3])
    assert [s.scored_tokens for s in scores] == [2, 1]


@pytest.mark.mps
def test_mps_model_forward_cache_and_suffix_gradients_match_cpu_reference():
    assert torch.backends.mps.is_available(), 'explicit MPS qualification requires real Metal'
    module = api()
    cpu = module.make_tiny_decoder(module.TinyConfig(), seed=17, dtype=torch.float64)
    mps = module.make_tiny_decoder(module.TinyConfig(), seed=17, dtype=torch.float64).to(dtype=torch.float32, device='mps')
    x, xm = batch(offset=3), batch(offset=3, device='mps')
    reference = cpu.logits(x)
    result = mps.logits(xm)
    # CPU-reference-derived envelope, measured across nine tiny config/seed pairs.
    atol = numeric_tolerances()['forward_atol']
    torch.testing.assert_close(result.cpu().double(), reference, atol=atol, rtol=0)
    torch.testing.assert_close(mps.logits(xm), result, atol=0, rtol=0)
    cache = mps.new_cache(start_position=3)
    parts = [mps.logits(batch([[token]], offset=i+3, device='mps'), cache=cache) for i, token in enumerate(x.input_ids[0].tolist())]
    torch.testing.assert_close(torch.cat(parts, 1), result, atol=atol, rtol=0)
    for depth in (0, 1, 2):
        gradients = []
        for m, b in [(cpu, x), (mps, xm)]:
            bound = m.prefix_boundary(b, depth)
            torch.testing.assert_close(m.suffix_logits(m.ff_residual(bound), bound), m.logits(b), atol=0, rtol=0)
            z = bound.u.detach().requires_grad_()
            gradients.append(torch.autograd.grad(m.suffix_logits(z, bound).square().mean(), z)[0].cpu().double())
        torch.testing.assert_close(gradients[1], gradients[0], atol=numeric_tolerances()['gradient_atol'], rtol=0)
        bound = mps.prefix_boundary(xm, depth)
        weight = torch.nn.Parameter(torch.eye(mps.config.hidden_size, dtype=torch.float32, device='mps')*.02)
        with torch.no_grad():
            target = mps.suffix_logits(mps.ff_residual(bound), bound)
        candidate = mps.suffix_logits(mps.ff_residual(bound, lambda x: x @ weight), bound)
        (candidate-target).square().mean().backward()
        assert weight.grad.abs().sum().item() > 0
        assert all(p.grad is None for p in mps.parameters())
    edited = batch([[1,4,7,11,12]], offset=3, device='mps')
    torch.testing.assert_close(mps.logits(edited)[:,:3], result[:,:3], atol=0, rtol=0)



@pytest.mark.cpu
def test_shared_contract_import_does_not_load_any_backend():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "import sys; import poc.contracts; assert not any(x in sys.modules for x in ('torch','mlx','transformers'))"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.cpu
def test_protocol_boundary_and_failed_cache_call_preserve_original_state():
    from poc.contracts import DecoderAdapter
    m = api().make_tiny_decoder(api().TinyConfig(), seed=9, dtype=torch.float64)
    assert isinstance(m, DecoderAdapter)
    cache = m.new_cache(start_position=2)
    m.logits(batch([[1,4]], offset=2), cache=cache)
    previous = tuple((layer.key.clone(), layer.value.clone()) for layer in cache.layers)
    with pytest.raises(ValueError):
        m.logits(batch([[2,3]], offset=3), cache=cache)
    assert cache.length == 2
    for layer, (key, value) in zip(cache.layers, previous):
        assert torch.equal(layer.key, key) and torch.equal(layer.value, value)
    bound = m.prefix_boundary(batch(), 1)
    with pytest.raises(ValueError, match='shape|width'):
        m.suffix_hidden(bound.u[..., :-1], bound)
    with pytest.raises(ValueError, match='preserve'):
        m.ff_residual(bound, lambda x: x[..., :1])


@pytest.mark.mps
def test_mps_continuation_scoring_uses_cpu_fp64_scalar_aggregation():
    from poc.contracts import ContinuationRequest
    assert torch.backends.mps.is_available()
    module = api()
    cpu = module.make_tiny_decoder(module.TinyConfig(), seed=7, dtype=torch.float64)
    mps = module.make_tiny_decoder(module.TinyConfig(), seed=7, dtype=torch.float64).to(device='mps', dtype=torch.float32)
    requests = [ContinuationRequest((1,4), (2,3), 'a'*64, 'a')]
    reference = cpu.score_continuations(requests)[0]
    actual = mps.score_continuations(requests)[0]
    assert abs(actual.log_likelihood - reference.log_likelihood) < 2 * actual.scored_tokens * numeric_tolerances()['forward_atol']


@pytest.mark.cpu
def test_inference_cache_rejects_wrong_tensor_dtype_before_mutation():
    m = api().make_tiny_decoder(api().TinyConfig(), seed=9, dtype=torch.float64)
    cache = m.new_cache()
    m.logits(batch([[1,4]]), cache=cache)
    cache.layers[0].key = cache.layers[0].key.float()
    with pytest.raises(ValueError, match='cache'):
        m.logits(batch([[2]], offset=2), cache=cache)
    assert cache.length == 2


@pytest.mark.cpu
@pytest.mark.parametrize('kwargs', [dict(hidden_size=15), dict(num_key_value_heads=3), dict(rms_norm_eps=float('nan'))])
def test_tiny_config_rejects_unexecutable_shapes_and_nonfinite_parameters(kwargs):
    with pytest.raises(ValueError):
        api().TinyConfig(**kwargs)


@pytest.mark.cpu
def test_unqualified_rope_modes_fail_explicitly():
    from transformers import LlamaConfig, LlamaForCausalLM
    hf = LlamaForCausalLM(LlamaConfig(vocab_size=23,hidden_size=16,intermediate_size=32,
             num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
             rope_parameters={'rope_type':'linear','factor':2.0,'rope_theta':10000.0}))
    with pytest.raises(ValueError, match='RoPE'):
        api().TorchDecoder.from_transformers(hf)

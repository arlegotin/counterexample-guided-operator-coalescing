"""Catch incomplete partitions, wrong compact FFNs and unsafe state reloads."""
from copy import deepcopy
import importlib
import importlib.util
import json

import pytest
import torch

from test_initialization import bank_api
from test_model_parity import batch
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder


def tiny():
    return make_tiny_decoder(TinyConfig(hidden_size=8, intermediate_size=12, num_attention_heads=2, num_key_value_heads=1), seed=2, dtype=torch.float64)


def test_partition_canonicalizes_order_and_rejects_duplicate_or_missing_depths():
    api = bank_api()
    p = api.canonicalize_partition([[2, 0], [1]], 3)
    assert p.groups == ((0, 2), (1,)) and p.reuse_map == (0, 1, 0)
    assert p == api.canonicalize_partition([[1], [0, 2]], 3)
    for groups in ([], [[0, 1]], [[0, 1], [1, 2]], [[0, 1, 2], []], [[0, 1, 3]], [[False, 1, 2]]):
        with pytest.raises(ValueError):
            api.canonicalize_partition(groups, 3)


@pytest.mark.parametrize('rank', [0, 2, 8])
def test_compact_expansion_matches_outputs_and_input_gradients(rank):
    api = bank_api(); teacher = tiny()
    bank = api.make_bank(teacher, api.canonicalize_partition([[0, 2], [1]], 3), rank, 11)
    expanded = api.expand_for_diagnostic(bank)
    x = batch()
    torch.testing.assert_close(bank.logits(x), expanded.logits(x), atol=2e-15, rtol=2e-14)
    for depth in range(3):
        z = torch.randn((2, 4, 8), dtype=torch.float64, requires_grad=True)
        y = z.detach().clone().requires_grad_()
        actual = bank.layers[depth].mlp(z); expected = expanded.layers[depth].mlp(y)
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-14)
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), z)[0], torch.autograd.grad(expected.square().sum(), y)[0], atol=2e-15, rtol=2e-14)
    if rank == 8:
        torch.testing.assert_close(bank.logits(x), teacher.logits(x), atol=2e-15, rtol=2e-14)
    assert teacher.lm_head.weight is teacher.embed_tokens.weight
    assert bank.lm_head.weight is bank.embed_tokens.weight
    assert all(not p.requires_grad for p in teacher.parameters())


def test_rank_zero_allocates_no_factors_and_singletons_keep_fixed_rank():
    api = bank_api(); teacher = tiny()
    p = api.canonicalize_partition([[0], [1], [2]], 3)
    rank_zero = api.make_bank(teacher, p, 0, 1)
    assert rank_zero.residual_owner_count == 0
    bank = api.make_bank(teacher, p, 2, 1)
    assert bank.residual_owner_count == 18
    residual = [r for r in api.owner_census(bank) if r.category == 'residual']
    assert sum(r.numel for r in residual) == 3 * 3 * 2 * (8 + 12)
    assert sum(p.numel() for p in bank.parameters()) == sum(p.numel() for p in teacher.parameters()) + 360
    torch.testing.assert_close(bank.logits(batch()), teacher.logits(batch()), atol=0, rtol=0)


def test_planted_permuted_neurons_have_exact_oracle_on_fresh_full_support():
    from poc.synthetic import make_planted_teacher
    api = bank_api(); fixture = make_planted_teacher(23)
    partition = api.canonicalize_partition([[0, 2, 3, 6], [1, 4, 5, 7]], 8)
    oracle = api.make_bank(fixture.oracle, partition, 0, 0)
    generator = torch.Generator().manual_seed(731)
    x = torch.cat([torch.eye(32, dtype=torch.float64), torch.randn((97, 32), generator=generator, dtype=torch.float64)])
    for depth in range(8):
        torch.testing.assert_close(oracle.layers[depth].mlp(x), fixture.teacher.layers[depth].mlp(x), atol=2e-15, rtol=2e-14)
    torch.testing.assert_close(oracle.logits(batch()), fixture.teacher.logits(batch()), atol=2e-15, rtol=2e-14)
    # A permuted whole medoid is still the same function. Unaligned truncated
    # weight-difference adapters need not preserve this functional equivalence.
    medoid = api.make_bank(fixture.teacher, partition, 0, 0)
    torch.testing.assert_close(medoid.logits(batch()), fixture.teacher.logits(batch()), atol=2e-15, rtol=2e-14)
    truncated = api.make_bank(fixture.teacher, partition, 2, 0)
    assert (truncated.layers[2].mlp(x) - fixture.teacher.layers[2].mlp(x)).abs().max() > 1e-5


@pytest.mark.parametrize('rank', [-1, 9, True])
def test_bank_rejects_rank_before_allocating(rank):
    api = bank_api()
    with pytest.raises(ValueError):
        api.make_bank(tiny(), api.canonicalize_partition([[0, 1, 2]], 3), rank, 0)


def test_bank_validates_actual_depth_and_all_projection_shapes():
    api = bank_api(); teacher = tiny()
    with pytest.raises(ValueError):
        api.make_bank(teacher, api.canonicalize_partition([[0, 1]], 2), 1, 0)
    teacher.layers[2].mlp.down_proj = torch.nn.Linear(13, 8, bias=False).double()
    with pytest.raises(ValueError, match='projection|shape'):
        api.make_bank(teacher, api.canonicalize_partition([[0, 1, 2]], 3), 1, 0)


def test_published_135m_arithmetic_is_only_a_fixture():
    # Literals from the blueprint; production bank dimensions come from loaded tensors.
    def residual_count_135m(rank): return 3 * 30 * rank * (576 + 1536)
    def parameter_count_135m(K, rank): return 54_888_768 + 3 * K * 576 * 1536 + residual_count_135m(rank)
    assert parameter_count_135m(K=15, rank=8) == 96_222_528
    assert residual_count_135m(rank=8) == 1_520_640
    assert parameter_count_135m(K=30, rank=8) == 136_035_648


def bundle_api():
    assert importlib.util.find_spec('poc.bundle') is not None, 'bank state serialization missing'
    return importlib.import_module('poc.bundle')


@pytest.mark.parametrize('rank', [0, 2])
def test_safe_state_roundtrip_preserves_trained_outputs_owners_and_base(tmp_path, rank):
    api = bank_api(); io = bundle_api(); teacher = tiny()
    bank = api.make_bank(teacher, api.canonicalize_partition([[0, 2], [1]], 3), rank, 8)
    # Persist values after a real update, not just reproducible initialization.
    optimizer = torch.optim.SGD([p for p in bank.parameters() if p.requires_grad], lr=.2)
    bank.logits(batch()).square().sum().backward(); optimizer.step()
    ref = io.write_bank_state(bank, tmp_path / 'state')
    loaded = io.load_bank_state(ref, teacher)
    torch.testing.assert_close(loaded.logits(batch()), bank.logits(batch()), atol=0, rtol=0)
    assert api.owner_census(loaded) == api.owner_census(bank)
    assert loaded.partition == bank.partition
    assert loaded.manifest().checkpoint_digest == bank.manifest().checkpoint_digest
    assert not (tmp_path / 'state' / 'non_ffn.safetensors').exists()
    assert (tmp_path / 'state' / 'residuals.safetensors').exists() == (rank > 0)
    with pytest.raises((FileExistsError, ValueError)):
        io.write_bank_state(bank, tmp_path / 'state')
    wrong = deepcopy(teacher)
    with torch.no_grad(): wrong.layers[0].self_attn.q_proj.weight.add_(1)
    with pytest.raises(ValueError, match='base'):
        io.load_bank_state(ref, wrong)
    path = tmp_path / 'state' / 'bank.safetensors'
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='hash|digest'):
        io.load_bank_state(ref, teacher)


def test_bank_seed_controls_initialization_without_changing_caller_rng():
    api = bank_api(); teacher = tiny()
    partition = api.canonicalize_partition([[0, 2], [1]], 3)
    before = torch.random.get_rng_state().clone()
    first = api.make_bank(teacher, partition, 2, 27)
    assert torch.equal(torch.random.get_rng_state(), before)
    second = api.make_bank(teacher, partition, 2, 27)
    assert first.manifest().checkpoint_digest == second.manifest().checkpoint_digest


def test_manifest_hash_and_rehashed_malformed_state_are_rejected(tmp_path):
    from dataclasses import replace
    import hashlib
    api = bank_api(); io = bundle_api(); teacher = tiny()
    bank = api.make_bank(teacher, api.canonicalize_partition([[0, 2], [1]], 3), 2, 8)
    ref = io.write_bank_state(bank, tmp_path / 'state')
    with pytest.raises(ValueError, match='hash'):
        io.load_bank_state(replace(ref, manifest_sha256='0' * 64), teacher)
    path = tmp_path / 'state' / 'state.json'
    original = json.loads(path.read_text())
    for changes in ({'groups': [[0, 2], [1, 2]]}, {'rank': 9}, {'files': {'../bank.safetensors': '0' * 64}}, {'schema': True}):
        path.write_text(json.dumps(original | changes))
        altered = replace(ref, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with pytest.raises(ValueError):
            io.load_bank_state(altered, teacher)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.mps)])
def test_scientific_fp32_bank_forward_cache_and_gradients_match_cpu_fp64(device):
    from test_model_parity import numeric_tolerances
    if device == 'mps':
        assert torch.backends.mps.is_available(), 'explicit MPS bank check requires real Metal'
    api = bank_api(); partition = api.canonicalize_partition([[0, 2], [1]], 3)
    reference = api.make_bank(tiny(), partition, 2, 13)
    scientific = api.make_bank(tiny().to(dtype=torch.float32, device=device), partition, 2, 13)
    expected = reference.logits(batch(offset=3))
    actual = scientific.logits(batch(offset=3, device=device))
    tolerances = numeric_tolerances()
    torch.testing.assert_close(actual.cpu().double(), expected, atol=tolerances['forward_atol'], rtol=0)
    expected.square().sum().backward(); actual.square().sum().backward()
    for (name, parameter), (other_name, other) in zip(reference.named_parameters(), scientific.named_parameters(), strict=True):
        assert name == other_name
        if parameter.requires_grad:
            assert parameter.grad is not None and other.grad is not None
            torch.testing.assert_close(other.grad.cpu().double(), parameter.grad, atol=tolerances['gradient_atol'], rtol=0)
        else:
            assert parameter.grad is None and other.grad is None
    cache = scientific.new_cache(start_position=3)
    parts = [scientific.logits(batch([[token]], offset=i+3, device=device), cache=cache)
             for i, token in enumerate(batch().input_ids[0].tolist())]
    torch.testing.assert_close(torch.cat(parts, dim=1), actual, atol=tolerances['forward_atol'], rtol=0)


@pytest.mark.parametrize(('change', 'reason'), [('keys', 'keys'), ('shape', 'shape'), ('dtype', 'dtype'), ('nan', 'finite'), ('values', 'checkpoint')])
def test_rehashed_safetensors_still_validate_owner_schema_and_checkpoint(tmp_path, change, reason):
    from dataclasses import replace
    import hashlib
    from safetensors.torch import load_file, save_file
    api = bank_api(); io = bundle_api(); teacher = tiny()
    bank = api.make_bank(teacher, api.canonicalize_partition([[0, 2], [1]], 3), 2, 8)
    ref = io.write_bank_state(bank, tmp_path / 'state')
    state_path = tmp_path / 'state' / 'bank.safetensors'
    values = load_file(state_path)
    key = next(iter(values))
    if change == 'keys': values.pop(key)
    elif change == 'shape': values[key] = values[key][:1]
    elif change == 'dtype': values[key] = values[key].float()
    elif change == 'nan': values[key].fill_(float('nan'))
    else: values[key].add_(1)
    save_file(values, state_path)
    manifest_path = tmp_path / 'state' / 'state.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['bank.safetensors'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    altered = replace(ref, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match=reason):
        io.load_bank_state(altered, teacher)

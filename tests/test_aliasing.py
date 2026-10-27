"""Catch duplicate parameter owners, lost shared gradients and mutable cache aliases."""
from copy import deepcopy
import pytest
import torch
from test_initialization import bank_api
from test_bank import tiny
from test_model_parity import batch


@pytest.mark.parametrize('prefill', [False, True])
def test_mutated_cache_aliases_rejected_before_execution(prefill):
    teacher = tiny(); cache = teacher.new_cache()
    if prefill: teacher.logits(batch([[1, 4]]), cache=cache)
    cache.layers = (cache.layers[0],) * 3
    old_length = cache.length
    with pytest.raises(ValueError, match='distinct|alias'):
        teacher.logits(batch([[7]], offset=old_length), cache=cache)
    assert cache.length == old_length


def test_actual_owners_unique_optimizer_and_freezing():
    api = bank_api(); teacher = tiny()
    bank = api.make_bank(teacher, api.canonicalize_partition([[0, 2], [1]], 3), 2, 9)
    records = api.owner_census(bank)
    shared = [r for r in records if r.category == 'bank']
    assert len(shared) == 6 and sorted(len(r.use_sites) for r in shared) == [1, 1, 1, 2, 2, 2]
    assert sum(r.numel for r in records) == sum(p.numel() for p in bank.parameters())
    assert all(r.trainable == (r.category in ('bank', 'residual')) for r in records)
    optimizer_parameters = [p for p in bank.parameters() if p.requires_grad]
    assert len(optimizer_parameters) == len({id(p) for p in optimizer_parameters}) == 24
    # Bank matrices appear once in registered state; no hidden per-depth dense matrices.
    for layer in bank.layers:
        assert sum(p.numel() for p in layer.mlp.parameters()) == 120
    assert len({id(layer.self_attn) for layer in bank.layers}) == 3
    for parameter in bank.parameters():
        assert all(parameter is not p for p in teacher.parameters())


def test_shared_gradients_equal_explicit_independent_use_sums():
    api = bank_api(); bank = api.make_bank(tiny(), api.canonicalize_partition([[0, 1, 2]], 3), 0, 4)
    dense = api.expand_for_diagnostic(bank)
    generator = torch.Generator().manual_seed(91)
    inputs = [torch.randn((2, 5, 8), generator=generator, dtype=torch.float64) for _ in range(3)]
    # Independent diagnostic copies are frozen by default; turn FFNs on explicitly.
    for layer in dense.layers: layer.mlp.requires_grad_(True)
    sum(bank.layers[i].mlp(x).square().sum() * (i + 1) for i, x in enumerate(inputs)).backward()
    sum(dense.layers[i].mlp(x).square().sum() * (i + 1) for i, x in enumerate(inputs)).backward()
    actual = {r.name: p for r, p in zip(api.owner_census(bank), bank.parameters(), strict=True) if r.category == 'bank'}
    for projection in ('gate_proj', 'up_proj', 'down_proj'):
        parameter = next(p for name, p in actual.items() if projection in name)
        expected = sum(getattr(layer.mlp, projection).weight.grad for layer in dense.layers)
        torch.testing.assert_close(parameter.grad, expected, atol=2e-15, rtol=2e-14)


def test_compact_cache_matches_full_with_distinct_depth_buffers():
    api = bank_api(); bank = api.make_bank(tiny(), api.canonicalize_partition([[0, 1, 2]], 3), 2, 4)
    cache = bank.new_cache(start_position=7)
    actual = torch.cat([bank.logits(batch([[1, 4]], offset=7), cache=cache), bank.logits(batch([[7, 2, 3]], offset=9), cache=cache)], dim=1)
    torch.testing.assert_close(actual, bank.logits(batch(offset=7)), atol=2e-15, rtol=2e-14)
    assert len({id(layer) for layer in cache.layers}) == 3


def test_copied_bank_routes_to_independent_owners_and_serialized_use_map():
    api = bank_api(); bank = api.make_bank(tiny(), api.canonicalize_partition([[0, 2], [1]], 3), 2, 9)
    copied = deepcopy(bank)
    # This catches wrappers retaining a weak reference to the source bank.
    api.owner_census(copied)
    before = bank.logits(batch()).detach().clone()
    with torch.no_grad(): copied.bank[0].down_proj.weight.add_(1)
    torch.testing.assert_close(bank.logits(batch()), before, atol=0, rtol=0)
    assert not torch.equal(copied.logits(batch()), before)
    assert copied.manifest().checkpoint_digest != bank.manifest().checkpoint_digest


def test_dense_teacher_census_distinguishes_ffns_and_embedding_tie():
    api = bank_api(); teacher = tiny()
    census = api.owner_census(teacher)
    ffns = [record for record in census if record.category == 'dense_ffn']
    assert len(ffns) == 9 and sum(record.numel for record in ffns) == 864
    embedding = next(record for record in census if record.name == 'embed_tokens.weight')
    assert embedding.use_sites == ('embed_tokens.weight', 'lm_head.weight')
    assert sum(record.numel for record in census) == sum(p.numel() for p in teacher.parameters())

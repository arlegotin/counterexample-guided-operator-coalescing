"""Full-state identities and bounded detached student cache ownership."""
from dataclasses import replace
import pytest
import torch
from test_patching import probes,record
from test_model_parity import api,batch


@pytest.mark.cpu
@pytest.mark.parametrize('change',[{'candidate_digest':'d'*64},{'teacher_digest':'e'*64},
    {'tokenizer_digest':'e'*64},{'dtype':'float32'},{'backend':'torch_mps'},
    {'boundary_version':'post_attention_v2'},{'execution_profile':'fixture_v2'}])
def test_every_state_change_invalidates_identity(change):
    r=record(); p=probes()
    assert p.probe_cache_key(r)!=p.probe_cache_key(replace(r,**change))
    assert p.probe_cache_key(r)==p.probe_cache_key(replace(r))


@pytest.mark.cpu
def test_bank_update_without_partition_change_invalidates_stored_student_state():
    from poc.bank import make_bank
    from poc.contracts import Partition
    p=probes(); t=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    t.tokenizer_digest='c'*64
    c=make_bank(t,Partition(((0,1),(2,)),3),rank=1,seed=17)
    b=c.prefix_boundary(batch(),1); tm,cm=t.manifest(),c.manifest()
    r=record(source='student',prefix_source='candidate',teacher_digest=tm.checkpoint_digest,
             candidate_digest=cm.checkpoint_digest,target_mask=(True,)*4,patch_positions=())
    cache=p.ProbeCache(max_entries=1,max_bytes=100000)
    cache.put(r,b,teacher=t,candidate=c)
    got=cache.get(r,teacher=t,candidate=c)
    assert got.model_digest==b.model_digest and not got.u.requires_grad
    got.u.zero_()
    assert torch.equal(cache.get(r,teacher=t,candidate=c).u,b.u)
    with torch.no_grad(): next(c.bank.parameters()).add_(.1)
    assert c.partition==Partition(((0,1),(2,)),3)
    with pytest.raises(ValueError,match='current|stale'): cache.get(r,teacher=t,candidate=c)
    assert cache.entry_count==0


@pytest.mark.cpu
def test_cache_limits_evict_entries_and_reject_oversize():
    p=probes(); t=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    t.tokenizer_digest='c'*64
    b=t.prefix_boundary(batch(),1); d=t.manifest().checkpoint_digest
    r=record(teacher_digest=d,candidate_digest=d,patch_positions=(),target_mask=(True,)*4)
    c=p.ProbeCache(max_entries=1,max_bytes=10000); c.put(r,b,teacher=t,candidate=t)
    r2=replace(r,probe_id='p2'); c.put(r2,b,teacher=t,candidate=t)
    assert c.entry_count==1 and c.get(r,teacher=t,candidate=t) is None and c.resident_bytes<=10000
    with pytest.raises(ValueError,match='capacity|bytes'): p.ProbeCache(max_entries=1,max_bytes=1).put(r,b,teacher=t,candidate=t)


@pytest.mark.cpu
def test_rank_one_bank_digest_handles_contiguous_singleton_strides():
    from poc.bank import make_bank
    from poc.contracts import Partition
    t=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    c=make_bank(t,Partition(((0,1),(2,)),3),rank=1,seed=17)
    # SVD rank-one A has shape [1,d] and is contiguous with stride [1,m].
    before=c.manifest().checkpoint_digest
    assert c.manifest().checkpoint_digest==before
    with torch.no_grad(): next(c.bank.parameters()).add_(.1)
    assert c.manifest().checkpoint_digest!=before


@pytest.mark.cpu
def test_missing_tokenizer_identity_and_changed_norm_or_private_map_fail_closed():
    from poc.bank import make_bank
    from poc.contracts import Partition
    p=probes(); t=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    b=t.prefix_boundary(batch(),1); d=t.manifest().checkpoint_digest
    r=record(teacher_digest=d,candidate_digest=d,patch_positions=(),target_mask=(True,)*4)
    cache=p.ProbeCache(max_entries=1,max_bytes=10000)
    with pytest.raises(ValueError,match='tokenizer|current'): cache.put(r,b,teacher=t,candidate=t)
    t.tokenizer_digest='c'*64
    c=make_bank(t,Partition(((0,1),(2,)),3),rank=1,seed=17)
    b=c.prefix_boundary(batch(),1)
    r=replace(r,source='student',prefix_source='candidate',candidate_digest=b.model_digest)
    cache.put(r,b,teacher=t,candidate=c)
    c.layers[0].post_attention_layernorm.variance_epsilon*=2
    with pytest.raises(ValueError,match='current|stale'): cache.get(r,teacher=t,candidate=c)
    assert cache.entry_count==0
    c.layers[0].mlp.index=1
    with pytest.raises(ValueError): cache.get(r,teacher=t,candidate=c)

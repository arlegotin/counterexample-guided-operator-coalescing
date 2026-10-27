"""Patches must reproduce declared coordinates and reject misalignment."""
from dataclasses import replace
import importlib
import importlib.util
import pytest
import torch
from poc.contracts import Boundary
from test_model_parity import batch


def probes():
    assert importlib.util.find_spec('poc.probes') is not None, 'aligned probe policy/logic missing'
    return importlib.import_module('poc.probes')


def record(**changes):
    base = dict(probe_id='p', recipient_id='d0', donor_id='edited', recipient_component='c0',
        donor_component='c0', recipient_role='probe_fit', donor_role='probe_fit', use='fit',
        source='donor', recipient_tokens=(1,4,7,2,3), donor_tokens=(1,4,8,2,3),
        attention_mask=(True,)*5, positions=(0,1,2,3,4), donor_positions=(0,1,2,3,4),
        target_mask=(False,False,True,True), patch_positions=(2,), patch_mode='full',
        channel_quarter=None, depth=1, teacher_digest='a'*64, candidate_digest='b'*64,
        tokenizer_digest='c'*64, dtype='float64', backend='torch_cpu',
        boundary_version='post_attention_v1', execution_profile='fixture_v1', seed=17,
        edit_kind='number', source_span=(2,3), donor_span=(2,3), replacement='8',
        prefix_source='teacher',source_window=(0,5),recipient_length_before_crop=5,donor_length_before_crop=5,
        recipient_full_tokens_digest='d'*64,donor_full_tokens_digest='e'*64,context_policy='first_tokens_reset_0_v1')
    return probes().ProbeRecord(**(base | changes))


def bounds(r):
    a=batch([list(r.recipient_tokens)])
    d=replace(a,input_ids=torch.tensor([r.donor_tokens]),document_ids=(r.donor_id,))
    u=torch.arange(80,dtype=torch.float64).reshape(1,5,16)
    return Boundary(u,a,r.depth,r.teacher_digest),Boundary(u+100,d,r.depth,r.teacher_digest)


@pytest.mark.cpu
@pytest.mark.parametrize('mode,positions,quarter', [('full',(2,),None),('span',(1,2,3),None),('quarter',(2,),2)])
def test_declared_masks_and_earliest_target(mode,positions,quarter):
    r=record(patch_mode=mode,patch_positions=positions,channel_quarter=quarter,
             target_mask=tuple(i>=min(positions) for i in range(4)))
    a,d=bounds(r); out=probes().apply_probe(a,d,r)
    expected=a.u.clone()
    for pos in positions:
        sl=slice(8,12) if quarter==2 else slice(None)
        expected[0,pos,sl]=d.u[0,pos,sl]
    assert torch.equal(out.u,expected)
    assert out.batch.target_mask.tolist()==[list(r.target_mask)]
    assert out.model_digest==a.model_digest
    assert torch.equal(a.u,torch.arange(80,dtype=torch.float64).reshape(1,5,16))


@pytest.mark.cpu
def test_null_and_equal_patches_and_future_rejection():
    r=record(); a,d=bounds(r)
    assert torch.equal(probes().apply_probe(a,replace(d,u=a.u),r).u,a.u)
    null=record(patch_positions=(),target_mask=(True,)*4)
    assert torch.equal(probes().apply_probe(a,d,null).u,a.u)
    with pytest.raises(ValueError,match='position|future'): record(donor_positions=(0,1,3,4,5))
    with pytest.raises(ValueError,match='target'): record(target_mask=(True,)*4)


@pytest.mark.cpu
def test_padding_and_wrong_token_or_depth_cannot_be_patched():
    r=record(); a,d=bounds(r)
    for bad in [replace(d,depth=0),replace(d,batch=replace(d.batch,input_ids=a.batch.input_ids))]:
        with pytest.raises(ValueError): probes().apply_probe(a,bad,r)
    with pytest.raises(ValueError,match='padding|valid'): record(attention_mask=(True,True,False,False,False))


@pytest.mark.cpu
def test_causal_teacher_suffix_preserves_earlier_logits_and_real_eos_target():
    from test_model_parity import api
    p=probes(); m=api().make_tiny_decoder(api().TinyConfig(),seed=19,dtype=torch.float64)
    m.tokenizer_digest='c'*64; d=m.manifest().checkpoint_digest
    r=record(teacher_digest=d,candidate_digest=d,donor_tokens=(1,4,8,2,3))
    recipient=batch([list(r.recipient_tokens)]); donor=batch([list(r.donor_tokens)])
    donor=replace(donor,document_ids=('edited',))
    b=m.prefix_boundary(recipient,1); donor_b=m.prefix_boundary(donor,1)
    patched=p.apply_probe(b,donor_b,r)
    before=m.suffix_logits(m.ff_residual(b),b)
    after=m.suffix_logits(m.ff_residual(patched),patched)
    assert torch.equal(before[:,:2],after[:,:2])
    assert not torch.equal(before[:,2:],after[:,2:])
    eos=record(recipient_tokens=(1,4,7,2,2),donor_tokens=(1,4,8,2,2),
        attention_mask=(True,True,True,True,False),target_mask=(False,False,True,False))
    a,d=bounds(eos)
    a=replace(a,batch=replace(a.batch,attention_mask=torch.tensor([eos.attention_mask])))
    d=replace(d,batch=replace(d.batch,attention_mask=torch.tensor([eos.attention_mask])))
    assert p.apply_probe(a,d,eos).batch.target_mask.tolist()==[[False,False,True,False]]

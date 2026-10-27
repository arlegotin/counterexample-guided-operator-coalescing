"""Role isolation, auditable edits, score-independent random pools and gradients."""
from dataclasses import replace
import pytest
import torch
from test_patching import probes, record
from test_model_parity import api,batch


def policy(**changes):
    p=probes()
    t=p.fit_pathology_thresholds([p.PathologyObservation(doc,doc+'_c','fit',v,v,depth)
        for depth in range(3) for doc,v in [('fit',1.),('fit2',3.)]],lower_quantile=0.,upper_quantile=1.)
    return p.ProbePolicy(**(dict(selection_mode='active',seed=17,per_document_cap=2,generated_edits_per_document=2,
        random_component_fraction=.25,depth_regions=((0,),(1,),(2,)),depth_region_balance=(1/3,)*3,
        selection_count=3,max_pool_records=96,max_cache_entries=8,max_cache_bytes=33554432,
        thresholds=t,roles={'d0':'probe_fit'},components={'d0':'c0'},allowed_roles=('probe_fit',),
        use='fit',teacher_digest='a'*64,candidate_digest='b'*64,tokenizer_digest='c'*64,
        dtype='float64',backend='torch_cpu',boundary_version='post_attention_v1',
        execution_profile='fixture_v1',context_tokens=512) | changes))


class CharacterTokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(x) for x in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}


@pytest.mark.cpu
def test_donors_are_deterministic_auditable_and_retain_alignment_rejections():
    from poc.data.manifest import DocumentRecord
    r=DocumentRecord('d0','fixture','v1','s',0,text='Alice 9 cats.')
    rows=list(probes().generate_donors([r],CharacterTokenizer(),policy()))
    assert rows==list(probes().generate_donors([r],CharacterTokenizer(),policy()))
    assert rows and any(x.status=='accepted' for x in rows)
    assert any(x.status=='rejected' and x.rejection_reason=='token_length_mismatch' for x in rows)
    assert all(x.source_span and x.edit_kind in ('name','number','plural') for x in rows)
    assert all(x.recipient_component==x.donor_component=='c0' for x in rows)


@pytest.mark.cpu
def test_denied_sources_and_unmaterialized_text_never_get_read():
    from poc.data.manifest import DocumentRecord
    r=DocumentRecord('d0','fixture','v1','s',0,text_path='/must-not-read')
    with pytest.raises(PermissionError): list(probes().generate_donors([r],CharacterTokenizer(),policy(roles={'d0':'dev'})))
    with pytest.raises(ValueError,match='materialized|reader'): list(probes().generate_donors([r],CharacterTokenizer(),policy()))
    with pytest.raises(ValueError,match='role|component'): record(donor_role='probe_score')


@pytest.mark.cpu
def test_seeded_random_ignores_scores_active_detects_planted_coordinate():
    p=probes(); rows=tuple(record(probe_id=f'p{i}',recipient_id=f'd{i}',recipient_component=f'c{i}',
         donor_component=f'c{i}',depth=i%3) for i in range(12))
    pool=p.ProbePool(rows)
    selection_manifest=dict(roles={r.recipient_id:r.recipient_role for r in rows},components={r.recipient_id:r.recipient_component for r in rows})
    a={r.probe_id:float(i) for i,r in enumerate(rows)}; b={k:-v for k,v in a.items()}
    random_policy=policy(selection_mode='seeded_random',**selection_manifest)
    assert p.mine_probes(pool,a,random_policy).selected_ids==p.mine_probes(pool,b,random_policy).selected_ids
    selected=p.mine_probes(pool,a,policy(**selection_manifest)).selected_ids
    assert 'p11' in selected and len(selected)==3
    assert p.mine_probes(pool,a,policy(**selection_manifest)).records==rows
    with pytest.raises(ValueError,match='missing'): p.mine_probes(pool,{},policy(**selection_manifest))


@pytest.mark.cpu
def test_local_same_student_boundary_detaches_prefix_and_preserves_suffix_gradient():
    p=probes(); m=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    student=api().make_tiny_decoder(api().TinyConfig(),seed=4,dtype=torch.float64).requires_grad_(True)
    b=student.prefix_boundary(batch(),1); original=b.model_digest
    weight=torch.nn.Parameter(torch.eye(m.config.hidden_size,dtype=torch.float64)*.03)
    r=record(source='student',prefix_source='candidate',teacher_digest=m.manifest().checkpoint_digest,
             candidate_digest=original,patch_positions=(),target_mask=(True,)*4)
    metric=p.backward_local_discrepancy(m,lambda x:x@weight,b,chunk_size=3,denominator=4,record=r)
    assert metric.kl>0 and metric.valid_targets==4 and weight.grad.abs().sum()>0
    assert all(x.grad is None for x in m.parameters()) and all(x.grad is None for x in student.parameters())
    assert b.model_digest==original!=m.manifest().checkpoint_digest
    with torch.no_grad():
        target=m.suffix_logits(m.ff_residual(replace(b,u=b.u.detach())),b)
        candidate=m.suffix_logits(m.ff_residual(replace(b,u=b.u.detach()),lambda x:x@weight),b)
    from poc.losses import kl_sum
    expected=kl_sum(target[:,:-1],candidate[:,:-1],b.batch.target_mask).item()
    assert abs(p.local_discrepancy(m,lambda x:x@weight,b,3).kl-expected)<1e-12


@pytest.mark.cpu
def test_mixture_and_fit_only_pathology_filters_do_not_hide_unfiltered_stress():
    p=probes(); sums={k:p.MetricSums(kl=v,valid_targets=2) for k,v in [('natural',2.),('student',4.),('donor',8.)]}
    assert p.mixture_discrepancy(sums,{'natural':.5,'student':.25,'donor':.25})==2.
    with pytest.raises(ValueError,match='missing'): p.mixture_discrepancy({}, {'donor':1.})
    with pytest.raises(ValueError,match='sum'): p.mixture_discrepancy(sums,{'donor':.25})
    with pytest.raises(ValueError,match='fit'): p.fit_pathology_thresholds([p.PathologyObservation('x','c','dev',1.,1.,0)],lower_quantile=0.,upper_quantile=1.)
    rows=tuple(record(probe_id=f'p{i}',depth=i) for i in range(3))
    pool=p.filter_pathologies(p.ProbePool(rows),{'p0':(2.,2.),'p1':(200.,2.),'p2':(2.,2.)},policy().thresholds)
    assert pool.records==rows and pool.exclusions=={'p1':('norm_above_fit_threshold',)}
    assert p.mine_probes(pool,{'p0':0.,'p1':999.,'p2':0.},policy(selection_count=2,depth_region_balance=(.5,0.,.5))).selected_ids==('p0','p2')


@pytest.mark.cpu
def test_depth_calibration_is_not_pooled_and_underfill_is_explicit():
    p=probes()
    thresholds=p.fit_pathology_thresholds([
        p.PathologyObservation('f','fc','fit',1.,1.,0),
        p.PathologyObservation('f','fc','fit',100.,2.,1)],lower_quantile=0.,upper_quantile=1.)
    rows=(record(probe_id='early',depth=0),record(probe_id='middle',depth=1))
    filtered=p.filter_pathologies(p.ProbePool(rows),{'early':(100.,1.),'middle':(100.,2.)},thresholds)
    assert set(filtered.exclusions)=={'early'}
    result=p.mine_probes(p.ProbePool(rows),{'early':1.,'middle':2.},policy())
    assert result.selection_status=='underfilled' and result.selection_reason
    assert len(result.selected_ids)==2


@pytest.mark.cpu
def test_generated_edit_cap_is_distinct_and_pool_does_not_expand_cartesian():
    from poc.data.manifest import DocumentRecord
    p=probes(); doc=DocumentRecord('d0','fixture','v1','s',0,text='Alice Bob 1 2 cats dogs')
    rows=list(p.generate_donors([doc],CharacterTokenizer(),policy(generated_edits_per_document=2,per_document_cap=1)))
    assert len({r.source_span for r in rows})==2
    assert len(rows)==6  # two edits, three depths; one mode per depth/edit
    assert {r.patch_mode for r in rows}=={'full','span','quarter'}


@pytest.mark.cpu
def test_planted_missing_coordinate_and_readout_nullspace_are_detected_by_suffix():
    from poc.contracts import Boundary
    p=probes()
    class OwnedControl(torch.nn.Module):
        def ff_residual(self,b,operator=None):
            return b.u if operator is None else b.u+operator(b.u)
        def suffix_hidden(self,z,b):
            return torch.stack((z[...,0]+z[...,1].square(),z[...,1]),-1)
        def readout(self,h):
            return torch.cat((h[...,:1],-h[...,:1],torch.zeros((*h.shape[:-1],39),dtype=h.dtype)),dim=-1)
    t=OwnedControl()
    b=Boundary(torch.tensor([[[1.,0.]]*5],dtype=torch.float64),batch(),1,'a'*64)
    def missing(x): return torch.stack((x[...,1].square(),torch.zeros_like(x[...,1])),-1)
    assert p.local_discrepancy(t,missing,b,3).kl==0
    donor=replace(b,u=b.u+torch.tensor([0.,1.],dtype=torch.float64))
    r=record(recipient_tokens=tuple(batch().input_ids[0].tolist()),donor_tokens=tuple(batch().input_ids[0].tolist()),patch_mode='span',patch_positions=(1,2),target_mask=(False,True,True,True))
    patched=p.apply_probe(b,replace(donor,batch=replace(donor.batch,document_ids=('edited',))),r)
    assert torch.equal(t.readout(b.u),t.readout(donor.u)) # direct head nullspace
    damage=p.local_discrepancy(t,missing,patched,3)
    assert damage.kl>0.01
    natural=replace(r,probe_id='natural',patch_positions=(),target_mask=(True,)*4)
    planted=replace(r,probe_id='planted')
    selected=p.mine_probes(p.ProbePool((natural,planted)),{'natural':0.,'planted':damage.kl/damage.valid_targets},
        policy(selection_count=1,depth_region_balance=(0.,1.,0.)))
    assert selected.selected_ids==('planted',)
    # Changing future states never changes earlier causal logits or target start.
    torch.testing.assert_close(t.suffix_hidden(patched.u,patched)[:,:1],t.suffix_hidden(b.u,b)[:,:1])


@pytest.mark.cpu
@pytest.mark.parametrize('exception',[RuntimeError('failed operator'),KeyboardInterrupt()])
def test_failed_or_interrupted_queries_persist_cost_and_budget_stops_are_fatal(tmp_path,exception):
    from poc.budget import BudgetExceeded,BudgetTracker
    from poc.records import start_attempt
    from test_records import spec,budget
    p=probes(); attempt=start_attempt(spec(budgets=replace(budget(),teacher_forward_calls=1)),tmp_path)
    tracker=BudgetTracker(attempt.spec.budgets,attempt)
    m=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    bound=m.prefix_boundary(batch(),1)
    def failed(x): raise exception
    # Target FFN uses the allowance; the next teacher suffix is denied fatally.
    with pytest.raises(BudgetExceeded): p.local_discrepancy(m,failed,bound,3,recorder=tracker)
    assert BudgetTracker(attempt.spec.budgets,attempt).totals.teacher_forward_calls==1
    attempt2=start_attempt(spec(budgets=replace(budget(),teacher_forward_calls=20)),tmp_path)
    tracker2=BudgetTracker(attempt2.spec.budgets,attempt2)
    with pytest.raises(type(exception)): p.local_discrepancy(m,failed,bound,3,recorder=tracker2)
    totals=BudgetTracker(attempt2.spec.budgets,attempt2).totals
    assert totals.student_forward_calls==1 and totals.teacher_forward_calls==2 and totals.wall_seconds>0


@pytest.mark.cpu
def test_tokenizer_failures_and_rejected_alignment_costs_are_durable(tmp_path):
    from poc.budget import BudgetTracker
    from poc.records import start_attempt
    from test_records import spec,budget
    from poc.data.manifest import DocumentRecord
    p=probes(); attempt=start_attempt(spec(budgets=replace(budget(),search_proposals=100)),tmp_path)
    tracker=BudgetTracker(attempt.spec.budgets,attempt)
    doc=DocumentRecord('d0','fixture','v1','s',0,text='Alice 9 cats.')
    rows=list(p.generate_donors([doc],CharacterTokenizer(),policy(recorder=tracker)))
    assert any(r.status=='rejected' for r in rows)
    assert tracker.totals.search_proposals==2 and tracker.totals.processed_inputs==3
    assert tracker.totals.processed_tokens==len(doc.text)*3+1 # 9 ->10 increases donor length
    class BrokenTokenizer:
        def __call__(self,*a,**k): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): list(p.generate_donors([doc],BrokenTokenizer(),policy(recorder=tracker)))
    assert BudgetTracker(attempt.spec.budgets,attempt).totals.processed_inputs==4


@pytest.mark.cpu
def test_backward_requires_fit_provenance_and_wall_cost_is_not_counted_twice(tmp_path):
    import time
    from poc.budget import BudgetTracker
    from poc.records import start_attempt
    from test_records import spec,budget
    p=probes(); attempt=start_attempt(spec(budgets=replace(budget(),teacher_forward_calls=20)),tmp_path)
    tracker=BudgetTracker(attempt.spec.budgets,attempt)
    t=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64)
    b=t.prefix_boundary(batch(),1); w=torch.nn.Parameter(torch.eye(16,dtype=torch.float64)*.02)
    r=record(teacher_digest=b.model_digest,candidate_digest=b.model_digest,patch_positions=(),target_mask=(True,)*4)
    rank=replace(r,use='rank',recipient_role='probe_score',donor_role='probe_score')
    with pytest.raises(PermissionError): p.backward_local_discrepancy(t,lambda x:x@w,b,chunk_size=3,denominator=4,record=rank)
    def expensive(x):
        time.sleep(.05)
        return x@w
    start=time.monotonic()
    p.backward_local_discrepancy(t,expensive,b,chunk_size=3,denominator=4,recorder=tracker,record=r)
    elapsed=time.monotonic()-start
    assert tracker.totals.wall_seconds<=elapsed*1.2
    assert tracker.totals.student_forward_calls==tracker.totals.student_backward_calls==tracker.totals.suffix_backward_calls==1
    assert tracker.totals.teacher_forward_calls==7


@pytest.mark.cpu
def test_manifest_roles_and_components_are_authoritative_for_selection():
    p=probes(); pool=p.ProbePool((record(),))
    with pytest.raises(PermissionError): p.mine_probes(pool,{'p':1.},policy(roles={'d0':'fit'}))


@pytest.mark.cpu
def test_length_change_before_primary_crop_cannot_hide_in_equal_512_lengths():
    from poc.data.manifest import DocumentRecord
    text='9 '+('x'*600)
    doc=DocumentRecord('d0','fixture','v1','s',0,text=text)
    rows=list(probes().generate_donors([doc],CharacterTokenizer(),policy(generated_edits_per_document=1)))
    assert len(rows)==1 and rows[0].status=='rejected'
    assert rows[0].rejection_reason=='token_length_mismatch'
    assert len(rows[0].recipient_tokens)==len(rows[0].donor_tokens)==512
    assert rows[0].source_window==(0,512)
    assert rows[0].recipient_length_before_crop==602 and rows[0].donor_length_before_crop==603


@pytest.mark.cpu
def test_calibrated_policy_rejects_uncalibrated_depths():
    with pytest.raises(ValueError,match='calibrat|depth'):
        policy(depth_regions=((0,),(1,),(3,)))


@pytest.mark.cpu
def test_selector_rejects_stale_candidate_and_unlisted_depth():
    p=probes(); pool=p.ProbePool((record(),))
    with pytest.raises(ValueError,match='state|current'): p.mine_probes(pool,{'p':1.},policy(candidate_digest='f'*64))
    with pytest.raises(ValueError,match='depth'): p.mine_probes(p.ProbePool((record(depth=9),)),{'p':1.},policy())


@pytest.mark.cpu
def test_actual_pilot_caps_include_source_changed_failed_run_ledgers(tmp_path,monkeypatch):
    from provenance.measure_probe_pilot import PilotRecorder
    from poc.budget import BudgetTracker,BudgetExceeded
    from poc.contracts import CostRecord
    from poc.records import start_attempt
    from test_records import spec,budget
    monkeypatch.setattr(torch.mps,'synchronize',lambda:None)
    monkeypatch.setattr(PilotRecorder,'memory',lambda self:0)
    first=start_attempt(spec(method_variant='task10_pilot',resolved_config={'plan_sha256':'a'*64}),tmp_path/'runs')
    tracker1=BudgetTracker(first.spec.budgets,first)
    tracker1.record(CostRecord(teacher_forward_calls=2,student_forward_calls=1))
    second=start_attempt(spec(method_variant='task10_pilot',resolved_config={'plan_sha256':'a'*64},source_commit='b'*40),tmp_path/'runs')
    tracker2=BudgetTracker(second.spec.budgets,second)
    recorder=PilotRecorder(tracker2,1)
    with pytest.raises(BudgetExceeded): recorder.check(CostRecord(teacher_forward_calls=2))
    with pytest.raises(BudgetExceeded): recorder.check(CostRecord(student_forward_calls=1))
    with pytest.raises(BudgetExceeded): recorder.record(CostRecord(student_forward_calls=1))
    assert BudgetTracker(second.spec.budgets,second).totals.student_forward_calls==1


@pytest.mark.cpu
def test_measurement_conditional_entropy_preserves_readout_position_axis():
    import provenance.measure_probe_pilot as pilot
    assert hasattr(pilot,'conditional_entropy'), 'position-preserving entropy helper missing'
    teacher=api().make_tiny_decoder(api().TinyConfig(),seed=5,dtype=torch.float64)
    b=teacher.prefix_boundary(batch(),1)
    h=teacher.suffix_hidden(teacher.ff_residual(b),b)
    class DirectRecorder:
        def call(self,fn,**counts): return fn()
    expected=teacher.readout(h[0]).float().log_softmax(-1)[2]
    entropy=pilot.conditional_entropy(teacher,h,2,DirectRecorder())
    assert abs(entropy-float(-(expected.exp()*expected).sum()))<1e-7


@pytest.mark.mps
def test_mps_local_candidate_gradient_matches_labeled_cpu_fp64_reference():
    import json
    from pathlib import Path
    assert torch.backends.mps.is_available(), 'explicit MPS qualification requires real Metal'
    p=probes(); gradients=[];metrics=[]
    for device,dtype in [('cpu',torch.float64),('mps',torch.float32)]:
        t=api().make_tiny_decoder(api().TinyConfig(),seed=3,dtype=torch.float64).to(device=device,dtype=dtype)
        s=api().make_tiny_decoder(api().TinyConfig(),seed=4,dtype=torch.float64).to(device=device,dtype=dtype).requires_grad_(True)
        b=s.prefix_boundary(batch(device=device),1)
        w=torch.nn.Parameter(torch.eye(16,device=device,dtype=dtype)*.03)
        r=record(source='student',prefix_source='candidate',teacher_digest=t.manifest().checkpoint_digest,
            candidate_digest=b.model_digest,backend='torch_'+device,dtype=str(dtype).removeprefix('torch.'),
            patch_positions=(),target_mask=(True,)*4)
        metrics.append(p.backward_local_discrepancy(t,lambda x:x@w,b,chunk_size=3,denominator=4,record=r))
        gradients.append(w.grad.cpu().double())
        assert w.grad.abs().sum().item()>0 and all(x.grad is None for x in s.parameters())
        assert all(x.grad is None for x in t.parameters())
    tolerances=json.loads((Path(__file__).resolve().parents[1]/'provenance/loss-tiny-numerics-2026-10-27.json').read_text())['tolerances']
    torch.testing.assert_close(gradients[1],gradients[0],atol=tolerances['gradient_atol'],rtol=0)
    assert abs(metrics[1].kl-metrics[0].kl)<=tolerances['kl_sum_atol']


@pytest.mark.cpu
def test_actual_bpe_length_mismatch_with_same_character_length_is_rejected():
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from poc.data.manifest import DocumentRecord
    chars=list(dict.fromkeys('[UNK]'+'AliceClar23'))
    vocab={'[UNK]':0}
    for char in chars:
        if char not in vocab: vocab[char]=len(vocab)
    for token in ['Al','Ali','Alic','Alice','Cl']: vocab[token]=len(vocab)
    engine=Tokenizer(models.BPE(vocab,merges=[('A','l'),('Al','i'),('Ali','c'),('Alic','e'),('C','l')],unk_token='[UNK]'))
    engine.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=engine,unk_token='[UNK]')
    # Equal-length name strings have different real BPE segmentations.
    assert len(tokenizer('Alice')['input_ids'])==1 and len(tokenizer('Clara')['input_ids'])==4
    doc=DocumentRecord('d0','fixture','v1','s',0,text='Alice 2')
    rows=list(probes().generate_donors([doc],tokenizer,policy(generated_edits_per_document=1)))
    assert len(rows)==1 and rows[0].rejection_reason=='token_length_mismatch'

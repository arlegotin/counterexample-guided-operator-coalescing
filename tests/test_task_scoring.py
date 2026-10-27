"""Analytic fixtures catch score normalization, boundary, EOS, and tie errors."""
import importlib
import importlib.util
import math
import pytest
from poc.contracts import ContinuationScore

pytestmark = pytest.mark.cpu


def api():
    assert importlib.util.find_spec('poc.evaluate'), 'task scoring is missing'
    return importlib.import_module('poc.evaluate')


class Tokenizer:
    eos_token_id = 0
    bos_token_id = 0
    def __call__(self, text, **kwargs):
        # Spaces belong to following tokens; a prefix ending inside a token must fail.
        if text == 'ab': return {'input_ids':[1], 'offset_mapping':[(0,2)]}
        return {'input_ids':[ord(c) for c in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids if i)


class UniformDecoder:
    def score_continuations(self, requests):
        return [ContinuationScore(r.item_id, -len(r.continuation_ids)*math.log(4), len(r.continuation_ids)) for r in requests]


def test_joint_tokenization_rejects_boundary_crossing():
    m=api(); p=m.PromptPolicy(Tokenizer(), context_tokens=128)
    with pytest.raises(ValueError, match='boundary'): p.request('a','b','cross')
    r=p.request('x','yz','ok')
    assert r.prompt_ids==(120,) and r.continuation_ids==(121,122)


def test_arc_lowest_index_ties_uniform_calibration_and_length_normalization():
    m=api(); p=m.PromptPolicy(Tokenizer(), context_tokens=128)
    row=m.score_arc(UniformDecoder(), [m.ARCItem('a','q',('x','long','z','last'),2,'easy')],p)[0]
    assert row.prediction==0 and row.tie and row.credit==0
    assert row.probabilities==pytest.approx((.25,)*4)
    assert row.log_loss==pytest.approx(math.log(4))
    assert row.brier==pytest.approx(.75)
    assert row.secondary_prediction==0


def test_empty_answer_and_context_overflow_are_rejected_without_truncation():
    m=api(); p=m.PromptPolicy(Tokenizer(),context_tokens=4)
    with pytest.raises(ValueError,match='empty'): p.request('x','','empty')
    with pytest.raises(ValueError,match='context'): p.request('abc','de','long')


def test_blimp_includes_real_eos_and_half_credit_for_exact_ties():
    m=api(); p=m.PromptPolicy(Tokenizer(),context_tokens=128)
    row=m.score_blimp(UniformDecoder(),[m.BlimpItem('a','abx','aby','paradigm')],p)[0]
    assert row.credit==.5 and row.tie
    assert row.token_counts==(3,3)  # first token conditions; two remaining + real EOS
    assert row.scores==pytest.approx((-3*math.log(4),)*2)


def test_macro_aggregation_weights_paradigms_equally_and_reports_micro():
    m=api(); rows=[m.TaskPrediction(str(i),'blimp',cluster,(),(),0,0,False,credit,'p')
                 for i,(cluster,credit) in enumerate([('a',1.),('b',0.),('b',0.),('b',0.)])]
    result=m.aggregate_tasks(rows)
    assert result['macro_accuracy']==.5 and result['accuracy']==.25


def test_synthetic_labels_are_algorithmic_and_templates_disjoint():
    m=api(); dev=m.synthetic_cases(seed=17,split='dev'); confirm=m.synthetic_cases(seed=17,split='confirm')
    assert {c.family for c in dev}=={'copy','lookup','arithmetic','composition'}
    assert {c.template for c in dev}.isdisjoint({c.template for c in confirm})
    assert {c.operands for c in dev if c.family=='arithmetic'}.isdisjoint({c.operands for c in confirm if c.family=='arithmetic'})
    for c in dev:
        if c.family=='arithmetic': assert c.answer==str(sum(c.operands))


def test_failed_arc_items_remain_visible_in_denominator():
    m=api();rows=m.score_arc(UniformDecoder(),[m.ARCItem('a','q',('','x'),0,'easy')],m.PromptPolicy(Tokenizer(),context_tokens=128))
    assert rows[0].status=='failed' and rows[0].reason=='empty answer'
    summary=m.aggregate_tasks(rows)
    assert summary['items']==1 and summary['failed']==1 and summary['accuracy']==0


def test_greedy_real_eos_stops_at_one_token():
    import torch
    from poc.runtime.torch_decoder import make_tiny_decoder,TinyConfig
    m=api();decoder=make_tiny_decoder(TinyConfig(vocab_size=256),seed=1,dtype=torch.float64)
    with torch.no_grad(): decoder.lm_head.weight.zero_()
    policy=m.GenerationPolicy(m.PromptPolicy(Tokenizer(),context_tokens=128),max_new_tokens=4)
    row=m.evaluate_synthetic(decoder,[m.SyntheticCase('eos','copy','x','x','owned')],policy)[0]
    assert row.token_counts==(1,) and row.prediction==''


def test_analytic_transition_likelihood_scores_continuation_only():
    import torch
    from poc.runtime.torch_decoder import TorchDecoder
    from poc.contracts import ContinuationRequest
    class TransitionDecoder:
        device=torch.device('cpu')
        def final_hidden(self,batch): return batch.input_ids[...,None]
        def readout(self,hidden):
            table=torch.tensor([[.1,.6,.3],[.2,.3,.5],[.25,.25,.5]],dtype=torch.float64).log()
            return table[hidden[...,0]]
    scores=TorchDecoder.score_continuations(TransitionDecoder(),[ContinuationRequest((2,0),(1,2),'owned','a')])
    assert scores[0].scored_tokens==2
    assert scores[0].log_likelihood==pytest.approx(math.log(.6)+math.log(.5))


def test_copy_guards_retain_full_labels_for_every_declared_length():
    m=api()
    assert hasattr(m,'copy_guard_cases'), 'copy capability guards are missing'
    rows=m.copy_guard_cases(seed=17)
    assert [len(c.answer.split()) for c in rows]==[128,512,2048,8192]
    assert all(c.answer in c.prompt for c in rows)

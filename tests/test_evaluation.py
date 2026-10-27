"""Real tiny decoders catch identity, corruption, token weighting and causal leakage."""
import importlib
import importlib.util
from copy import deepcopy
import pytest
import torch
from poc.contracts import MetricSums
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder
from test_model_parity import batch

pytestmark=pytest.mark.cpu

def api():
    assert importlib.util.find_spec('poc.evaluate'), 'document evaluation is missing'
    return importlib.import_module('poc.evaluate')


def test_identity_and_corrupt_candidate_fidelity():
    m=api(); teacher=make_tiny_decoder(TinyConfig(),seed=19,dtype=torch.float64)
    candidate=deepcopy(teacher)
    rows=list(m.evaluate_documents(teacher,candidate,[batch()],negative_kl_floor=1e-12))
    assert rows[0].metrics.kl==0 and rows[0].metrics.disagreements==0
    with torch.no_grad(): candidate.layers[0].mlp.down_proj.weight.add_(2)
    corrupt=list(m.evaluate_documents(teacher,candidate,[batch()],negative_kl_floor=1e-12))
    assert corrupt[0].metrics.kl>1e-5


def test_aggregation_is_token_weighted_and_refuses_empty_or_uncalibrated_negative():
    m=api()
    result=m.aggregate_documents([m.DocumentMetric('a','ca',MetricSums(0,2,1)),m.DocumentMetric('b','cb',MetricSums(0,9,9))])
    assert result['nll_nats_per_token']==1.1 and result['perplexity']==pytest.approx(3.0041660239464334)
    with pytest.raises(ValueError,match='zero'): m.aggregate_documents([])
    with pytest.raises(FloatingPointError,match='floor'): m.check_kl(MetricSums(-.1,2,2),negative_kl_floor=1e-5)
    assert m.check_kl(MetricSums(-1e-8,2,2),negative_kl_floor=1e-5).kl==-1e-8


def test_future_token_edit_does_not_change_past_predictions():
    m=api(); teacher=make_tiny_decoder(TinyConfig(),seed=19,dtype=torch.float64)
    torch.testing.assert_close(teacher.logits(batch())[:,:3],teacher.logits(batch([[1,4,7,9,8]]))[:,:3],atol=0,rtol=0)

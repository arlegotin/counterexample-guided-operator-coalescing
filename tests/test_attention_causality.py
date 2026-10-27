"""Catch future leakage, padding-as-EOS confusion, and invalid mask acceptance."""
from dataclasses import replace
import pytest
import torch
from test_model_parity import api, batch


@pytest.mark.cpu
def test_future_token_edit_cannot_change_past_logits():
    m = api().make_tiny_decoder(api().TinyConfig(), seed=2, dtype=torch.float64)
    before, after = batch(), batch([[1,4,7,11,12]])
    torch.testing.assert_close(m.logits(before)[:, :3], m.logits(after)[:, :3], atol=0, rtol=0)


@pytest.mark.cpu
def test_right_padding_retains_real_eos_and_matches_independent_prompt():
    m = api().make_tiny_decoder(api().TinyConfig(), seed=2, dtype=torch.float64)
    x = batch([[1,4,2,2,2], [1,4,7,3,2]])
    x = replace(x, attention_mask=torch.tensor([[1,1,1,0,0], [1,1,1,1,1]], dtype=torch.bool),
                target_mask=torch.tensor([[1,1,0,0], [1,1,1,1]], dtype=torch.bool))
    torch.testing.assert_close(m.logits(x)[0,:3], m.logits(batch([[1,4,2]]))[0], atol=2e-15, rtol=2e-14)
    assert x.target_mask[0].sum() == 2
    with pytest.raises(ValueError, match='cache|batch'):
        m.logits(x, cache=m.new_cache())


@pytest.mark.cpu
@pytest.mark.parametrize('change', ['left_pad', 'bad_target', 'positions', 'dtype'])
def test_invalid_batch_masks_positions_and_dtypes_fail_before_execution(change):
    m = api().make_tiny_decoder(api().TinyConfig(), seed=2, dtype=torch.float64)
    x = batch()
    if change == 'left_pad':
        changes = dict(attention_mask=torch.tensor([[0,1,1,1,1]], dtype=torch.bool))
    elif change == 'bad_target':
        changes = dict(attention_mask=torch.tensor([[1,1,1,0,0]], dtype=torch.bool))
    elif change == 'positions':
        changes = dict(positions=torch.tensor([[0,1,4,5,6]]))
    else:
        changes = dict(input_ids=x.input_ids.float())
    with pytest.raises(ValueError):
        m.logits(replace(x, **changes))

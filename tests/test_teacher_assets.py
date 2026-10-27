"""Opt-in fixture: no downloads, only an explicitly named pinned local manifest."""
import os
from pathlib import Path
import pytest
import torch
from poc.teacher import AssetManifest,load_reference
from poc.runtime.torch_decoder import TorchDecoder
from test_model_parity import batch

pytestmark=[pytest.mark.assets,pytest.mark.skipif(not os.environ.get('POC_LOCAL_TEACHER_MANIFEST'),reason='explicit local asset path required')]


def test_pinned_real_teacher_matches_original_hf_and_actual_owner_census():
    asset=AssetManifest.load(Path(os.environ['POC_LOCAL_TEACHER_MANIFEST']),
        expected_sha256='9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083')
    hf=load_reference(asset);model=TorchDecoder.from_transformers(hf)
    assert sum(p.numel() for p in model.parameters())==134515008
    assert model.lm_head.weight is model.embed_tokens.weight
    b=batch(offset=7)
    with torch.no_grad():
        expected=hf(input_ids=b.input_ids,attention_mask=b.attention_mask,position_ids=b.positions,use_cache=False).logits
        torch.testing.assert_close(model.logits(b),expected,atol=0,rtol=0)

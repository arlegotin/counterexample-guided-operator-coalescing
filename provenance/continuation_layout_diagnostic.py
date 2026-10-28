"""New owned layout diagnostic; does not reproduce the unavailable archive script."""
import argparse
import json
from pathlib import Path
import tempfile

import torch

from poc.bank import make_bank, canonicalize_partition, owner_census
from poc.bundle import write_bank_state, load_bank_state
from poc.contracts import Batch
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder, EXECUTION_PROFILE


def measure():
    teacher = make_tiny_decoder(TinyConfig(hidden_size=8, intermediate_size=12,
        num_attention_heads=2, num_key_value_heads=1), seed=2, dtype=torch.float64)
    source = make_bank(teacher, canonicalize_partition([[0, 2], [1]], 3), 2, 17)
    ids = torch.tensor([[1, 4, 7, 2, 3]])
    batch = Batch(ids, torch.ones_like(ids, dtype=torch.bool),
        torch.ones((1, 4), dtype=torch.bool), torch.arange(5)[None], ('owned',), ('owned',))
    with tempfile.TemporaryDirectory() as tmp:
        restored = load_bank_state(write_bank_state(source, Path(tmp)/'state'), teacher)
    left, right = dict(source.named_parameters()), dict(restored.named_parameters())
    with torch.no_grad():
        difference = (source.logits(batch)-restored.logits(batch)).abs()
    return dict(execution_profile=EXECUTION_PROFILE, dtype='float64', device='cpu',
        parameter_bytes_equal=all(torch.equal(p, right[n]) for n, p in left.items()),
        digest_equal=source.manifest().checkpoint_digest == restored.manifest().checkpoint_digest,
        census_equal=owner_census(source) == owner_census(restored),
        tied_head=restored.lm_head.weight is restored.embed_tokens.weight,
        strides={n:dict(before=list(p.stride()), after=list(right[n].stride()))
            for n, p in left.items() if '.residuals.' in n},
        max_logit_difference=float(difference.max()),
        bit_exact_logits=not bool(difference.count_nonzero()), tolerance=1e-12,
        qualified=bool(difference.max() <= 1e-12))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(measure(), indent=2, sort_keys=True)+'\n')

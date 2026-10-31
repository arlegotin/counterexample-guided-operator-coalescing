# Counterexample-guided operator coalescing

CGOC tests which feed-forward networks in a language model can share weights. Layers reuse a shared bank and keep small private low-rank corrections. The next question is whether deliberately difficult examples help choose safer merges than ordinary text does.

The repository includes shared SwiGLU operators, controlled recovery, counterexample probes, and complete offline Torch/MLX artifacts. All 30 transformer layers still execute. Storage savings come from sharing weights.

[Findings](docs/continuation-status-2026-10-31.md) · [Reproduce](docs/reproduction.md) · [Design](docs/blueprint.md) · [Citation](CITATION.cff)

## What the measurements say

On **SmolLM2-135M**, longer recovery repaired about **62% of the initial teacher disagreement** for one merged pair, rank 8, seed 17:

| Objective | Initial KL | Step-1,024 KL ↓ |
| --- | ---: | ---: |
| Local teacher-suffix loss | 0.183561 | **0.069087** |
| Full-student loss | 0.183561 | 0.069852 |

These are checked states on 128 exposed development components. The local pipeline failed final memory accounting; its saved state was independently admitted. [Raw observations and uncertainty](provenance/continuation-checkpoint-2026-10-31/manifest.json).

The native comparison uses an **earlier 256-update shared checkpoint**. Shared four-bit GPTQ used **86.43 MB** at KL **0.4311**; dense GPTQ used **87.88 MB** at **0.4112**. That small saving costs fidelity. The tested recipes haven't established a useful edge over strong dense quantization or added value from counterexample selection. No capability tolerance or production speedup is established.

## Reuse the machinery

- Compare recovery objectives with the same initialized weights, trainable owners and raw text exposure.
- Test initialization and merge composition; restore optimizer, schedule, RNG and data cursors together.
- Export canonical shared owners once, reload offline, and count the complete artifact—including runtime and tokenizer.

The [campaign guide](docs/dense-baseline-campaign.md) separates implemented controls from future native studies. Historical reports and failed attempts stay intact.

## Check the evidence

From a clone, with Python 3.12:

```sh
python3.12 provenance/verify_checkpoint.py
python3.12 provenance/verify_public_continuation.py
```

Both checks use only the standard library. No weights, private corpus or Apple hardware required. Add `--bootstrap` to the first command with NumPy installed to regenerate the retained component resamples. Training and native replay need the separately pinned assets in the [reproduction guide](docs/reproduction.md).

By **Artem Legotin**. **AGPL-3.0-only** software; **CC BY 4.0** writing. See [licensing](LICENSING.md) and [upstream notices](THIRD_PARTY_NOTICES.md). Cite [CITATION.cff](CITATION.cff) and record the commit used. Development used AI assistance; README editing used [write-like-me](https://github.com/arlegotin/write-like-me).

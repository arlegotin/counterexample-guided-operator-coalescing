# Counterexample-guided operator coalescing

CGOC tests whether a language model can reuse feed-forward weights across layers, and whether deliberately difficult examples help choose which layers to merge. The goal is a smaller stored model that keeps the original model's behavior.

Sharing learns back some of that behavior. **The hard-example selector hasn't beaten ordinary examples in this study.** Four functional scoring rules choose the same merge within each fixed analysis; fresh documents change the winner. The local study finishes inconclusive, with a working implementation, controlled comparisons and the evidence needed to inspect the result.

[Results](#results) · [How it works](#how-it-works) · [Quick check](#quick-check) · [Full report](REPORT.md) · [Reproduce](docs/reproduction.md) · [Citation](CITATION.cff)

## How it works

A transformer usually stores a separate feed-forward network (FFN) for every layer. CGOC gives several layers a shared FFN bank, then adds small low-rank corrections for each layer. Every layer still executes. The saving comes from storing fewer distinct weights.

The difficult part is deciding what can share. Two FFNs can look similar as matrices and still behave differently on the inputs that matter. CGOC compares their behavior on ordinary text and on counterexamples built to expose disagreement. If those stress examples reveal damage that ordinary text misses, they should help choose a safer merge.

The original model (the teacher) is a frozen **SmolLM2-135M** with 30 layers; the shared models use rank-8 corrections. The implementation includes teacher-equivalent initialization, merge fitting, counterexample probes, checkpoint recovery, offline Torch/MLX bundles and native payload accounting. Experiments ran locally on a 36-GiB M3 Max. [Design](docs/superpowers/specs/2026-10-27-cgoc-design.md) · [Prior art and blueprint review](docs/analysis/2026-10-27-blueprint-review.md)

## Results

### Sharing can recover behavior; choosing the merge remains unresolved

A 15-bank shared model reduced teacher-distribution KL from **13.06 to 6.49 nats per target** during the recorded global fit, measured on eight development documents outside fitting. Lower is better. This measures agreement with the original model's next-token distribution; substantial error remains and the learning curve hadn't saturated. [Training receipt](provenance/training-135m-2026-10-27.json)

The selector experiment compares **seven scoring rules on the same 20 fitted single-merge candidates**. Each candidate gets one fitting update. Ordinary natural-text KL, random-donor scoring and both active-counterexample settings make the same choice within each analysis:

| Analysis | Scoring / damage documents | Pair selected by all four functional rules | Excess damage over best sampled pair |
| --- | ---: | --- | ---: |
| Original, primary | 8 / 16 | (12, 20) | 0.000116 |
| Fresh documents, exploratory | 16 / 32 | (4, 6) | 0.009830 |
| Pooled, secondary | 24 / 48 | (12, 20) | 0 |

Excess damage is selection regret in nats per target, relative to the best of these 20 candidates. The rows use different document populations; pooling doesn't replace the primary result. The comparison isolates scoring on common fitted states and probe pools, so it doesn't establish the efficiency of a standalone random-only acquisition pipeline.

The uncertainty matters. In the fresh-document bootstrap, the active setting with counterexample weight 1 and ordinary scoring choose different pairs in **80 of 2,000** component resamples: active does better in 49 and worse in 31. Their percentile contrast interval still reads `[0, 0]` because most resamples tie. That isn't evidence of equivalence. Near-best uncertainty and one-update fitting limit what this diagnostic can decide. [Original evidence](provenance/single-merge-original-135m-2026-10-27.json) · [Fresh evaluation and uncertainty](provenance/single-merge-precision-135m-2026-10-27.json)

### A practical compression win is still unproven

The untrained shared-format anchor sets a **66,064,401-byte** ceiling. A preliminary dense quantization baseline already fits in **61,327,184 bytes**. There is no trained native shared model at the ceiling, matched quality win, measured speedup or protected capability result. The global-fit and quantization KL measurements use different populations and cannot establish a head-to-head win. [Payload anchor](provenance/payload-anchor-135m.json) · [Dense baseline evidence](provenance/quantization-135m-2026-10-27.json)

The study stopped at its selection diagnostic. The later compiler, protected evaluation and independent replication stages were not run. The result leaves the broader sharing idea open, but gives no reason yet to prefer active selection over the simpler functional score.

## What you can reuse

- **The sharing mechanism:** shared nonlinear FFN banks, per-layer low-rank corrections and teacher-equivalent controls.
- **A controlled selector comparison:** seven scores, common fitted candidates, whole-document uncertainty and a fixed fresh-document evaluation. Eighteen reconstructed fits matched their original initialized and fitted state digests exactly.
- **Storage and numerical checks:** real native payload bytes, dense quantization baselines, backend parity and offline bundle validation. Stored size, output fidelity and capability remain separate measurements.
- **An inspectable result:** 290 source-bound report claims, 2,278 measured metric rows, six explicitly unexecuted rows, and retained failures. Public report regeneration needs no weights or corpus access.

The [full report](REPORT.md) carries the qualifications and costs. The [machine summary](results/summary.json), [metric table](results/regressions.parquet) and [evidence index](provenance/report-evidence-2026-10-28.json) let you follow the numbers back to their receipts.

## Quick check

From a checkout, use **Python 3.12** to install just the reporting and test dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,report]'
python -m poc report --include-failures
python -m pytest tests/test_report.py -q
```

This regenerates the report from public evidence and checks reporting integrity. It doesn't train a model, fetch a corpus or reproduce the scientific experiments. [Report CI](https://github.com/arlegotin/counterexample-guided-operator-coalescing/actions/workflows/report.yml) checks the same path. The [reproduction guide](docs/reproduction.md) separates this quick check from the full native environment and its exact dependency lock.

For the research history, start with the [original blueprint](docs/blueprint.md), [implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md) and [decision appendix](docs/analysis/2026-10-28-decision-appendix.md). [Publication scope](docs/PUBLICATION.md) describes what the public evidence can reproduce.

## License and citation

By **Artem Legotin**. Original software is **AGPL-3.0-only**; original writing and result presentation are **CC BY 4.0**. See [licensing scope](LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

Please cite [CITATION.cff](CITATION.cff), link this repository and identify the commit you used. Development, experiments and reporting used AI assistance under the owner's direction. README editing used [write-like-me](https://github.com/arlegotin/write-like-me).

# Counterexample-guided operator coalescing

CGOC tests whether a language model can reuse feed-forward weights across layers, and whether deliberately difficult examples help choose which layers to merge. The goal is a smaller stored model that keeps the original model's behavior.

Sharing learns back some of that behavior. **The tested sharing recipes still have no established useful advantage over strong dense quantization.** A completed long diagnostic repaired about 62% of the initial teacher disagreement on one pair and seed. Counterexample selection has no demonstrated added benefit.

**Research checkpoint — 31 October 2026:** the current implementation and evidence package are closed for now, with further experiments listed separately. [Current results and limits](docs/continuation-status-2026-10-31.md) · [Dense baseline campaign guide](docs/dense-baseline-campaign.md) · [Numeric receipts](provenance/continuation-checkpoint-2026-10-31/manifest.json)

[Results](#results) · [How it works](#how-it-works) · [Quick check](#quick-check) · [Continuation report](provenance/continuation-v2/REPORT.md) · [Historical report](REPORT.md) · [Reproduce](docs/reproduction.md) · [Citation](CITATION.cff)

## How it works

A transformer usually stores a separate feed-forward network (FFN) for every layer. CGOC gives several layers a shared FFN bank, then adds small low-rank corrections for each layer. Every layer still executes. The saving comes from storing fewer distinct weights.

The difficult part is deciding what can share. Two FFNs can look similar as matrices and still behave differently on the inputs that matter. CGOC compares their behavior on ordinary text and on counterexamples built to expose disagreement. If those stress examples reveal damage that ordinary text misses, they should help choose a safer merge.

The original model (the teacher) is a frozen **SmolLM2-135M** with 30 layers; the shared models use rank-8 corrections. The implementation includes teacher-equivalent initialization, merge fitting, counterexample probes, checkpoint recovery, offline Torch/MLX bundles and native payload accounting. Native receipts record an M3 Max with “36 GB” memory. [Design](docs/superpowers/specs/2026-10-27-cgoc-design.md) · [Prior art and blueprint review](docs/analysis/2026-10-27-blueprint-review.md)

## Results

### Historical continuation v2: some recovery, a weak practical tradeoff

The earlier exploratory pilot compares three separate 16-update trajectories for pair **(0,15)**, rank 8, seed 17, with a frozen SmolLM2-135M teacher. All use the same chosen parent; the first two start with identical weights. Validation uses eight previously exposed development components outside the twelve fitting components.

| Fitting recipe | Initial KL | Final KL |
| --- | ---: | ---: |
| Parent-SVD, natural/student/donor mixture | 0.223229 | 0.192125 |
| Same parent-SVD, ordinary text only | 0.223229 | **0.189272** |
| Same parent, zero-product corrections, ordinary text only | 0.369757 | 0.338294 |

Lower KL means less disagreement with the original next-token distribution. The 15.2% reduction in the natural-only arm is a fidelity improvement, not an accuracy gain. All eight validation components improve, but the curves are still falling at step 16. One pair, one seed and previously used development data do not establish convergence or generalization. The mixture comparison changes source allocation at equal logical targets; it does not isolate donor harm.

Complete native exports were then evaluated on the same eight windows and 4,088 targets:

| Artifact | Complete size (decimal MB) | KL/target ↓ |
| --- | ---: | ---: |
| New shared model, natural fitting | 537.217 | 0.189272 |
| Dense RTN 8-bit | **155.110** | **0.002529** |
| Historical trained K15, shared RTN 2-bit / embedding 4-bit | 49.484 | 10.474141 |
| Historical trained K15, shared RTN 4-bit / embedding 2-bit | 58.995 | 9.233236 |
| Dense DWQ 2-bit / embedding 4-bit | 61.327 | 4.441507 |

The dense 8-bit artifact is about 3.46 times smaller than the new shared model, with about 75 times less KL. The smaller trained shared exports meet the historical 66,064,401-byte ceiling but have poor fidelity. The dense 61.3-MB point has better fidelity than those shared points while using more bytes; it does not strictly dominate every smaller artifact. No useful quality margin or capability result is established at the ceiling.

This continuation also recovers the original raw scoring arrays from hash-verified local records. Ordinary, random-donor, active and added student-only scores select the same point-estimate winner within each fixed analysis. The evidence supports ordinary natural KL as the default for further research, without establishing equivalence or independent acquisition efficiency.

[Full continuation report](provenance/continuation-v2/REPORT.md) · [Learning curves](provenance/continuation-v2/learning-curves.csv) · [Eight native points, memory and descriptive timings](provenance/continuation-v2/native-frontier.csv) · [Raw score/damage matrices](provenance/continuation-v2/recovered-scores.json) · [Claims and limits](provenance/continuation-v2/CLAIMS.json)

### Original study: sharing recovery and unresolved selection

A 15-bank shared model reduced teacher-distribution KL from **13.06 to 6.49 nats per target** during the recorded global fit, measured on eight development documents outside fitting. Lower is better. This measures agreement with the original model's next-token distribution; substantial error remains and the learning curve hadn't saturated. [Training receipt](provenance/training-135m-2026-10-27.json)

The selector experiment compares **seven scoring rules on the same 20 fitted single-merge candidates**. Each candidate gets one fitting update. Ordinary natural-text KL, random-donor scoring and both active-counterexample settings make the same choice within each analysis:

| Analysis | Scoring / damage documents | Pair selected by all four functional rules | Excess damage over best sampled pair |
| --- | ---: | --- | ---: |
| Original, primary | 8 / 16 | (12, 20) | 0.000116 |
| Fresh documents, exploratory | 16 / 32 | (4, 6) | 0.009830 |
| Pooled, secondary | 24 / 48 | (12, 20) | 0 |

Excess damage is selection regret in nats per target, relative to the best of these 20 candidates. The rows use different document populations; pooling doesn't replace the primary result. The comparison isolates scoring on common fitted states and probe pools, so it doesn't establish the efficiency of a standalone random-only acquisition pipeline.

The uncertainty matters. In the fresh-document bootstrap, the active setting with counterexample weight 1 and ordinary scoring choose different pairs in **80 of 2,000** component resamples: active does better in 49 and worse in 31. Their percentile contrast interval still reads `[0, 0]` because most resamples tie. That isn't evidence of equivalence. Near-best uncertainty and one-update fitting limit what this diagnostic can decide. [Original evidence](provenance/single-merge-original-135m-2026-10-27.json) · [Fresh evaluation and uncertainty](provenance/single-merge-precision-135m-2026-10-27.json)

### Original storage comparison and stopping point

At the original stop, the untrained shared-format anchor set a **66,064,401-byte** ceiling and a preliminary dense baseline fit in **61,327,184 bytes**. There was no trained native shared model at the ceiling or matched quality comparison. Those historical losses used different populations. The continuation above adds matched native measurements and trained shared exports, while preserving this original evidence. A useful compression win, speedup and protected capability result remain unproven. [Payload anchor](provenance/payload-anchor-135m.json) · [Dense baseline evidence](provenance/quantization-135m-2026-10-27.json)

The original study stopped at its selection diagnostic. Its later compiler, protected evaluation and independent replication stages were not run. The separately authorized continuation does not change that stop or resume those historical stages.

## What you can reuse

- **The sharing mechanism:** shared nonlinear FFN banks, per-layer low-rank corrections and teacher-equivalent controls.
- **A controlled selector comparison:** seven scores, common fitted candidates, whole-document uncertainty and a fixed fresh-document evaluation. Eighteen reconstructed fits matched their original initialized and fitted state digests exactly.
- **Storage and numerical checks:** real native payload bytes, dense quantization baselines, backend parity and offline bundle validation. Stored size, output fidelity and capability remain separate measurements.
- **An inspectable result:** 290 source-bound report claims, 2,278 measured metric rows, six explicitly unexecuted rows, and retained failures. Public report regeneration needs no weights or corpus access.

The [continuation report](provenance/continuation-v2/REPORT.md) and [numeric summary](provenance/continuation-v2/summary.json) carry the new results, costs and limitations. The original [historical report](REPORT.md), [machine summary](results/summary.json), [metric table](results/regressions.parquet) and [evidence index](provenance/report-evidence-2026-10-28.json) remain unchanged.

## Quick check

From the current checkout, verify continuation hashes, raw numeric reductions, all five scoring formulas and selected pairs with **Python 3.12**, using only its standard library:

```sh
python3.12 provenance/verify_public_continuation.py
```

This needs no model weights, private corpus, network or Apple hardware. It checks the published measurements; it does not rerun training or native inference.

For byte-identical regeneration of the original report, use the audited source:

```sh
git worktree add --detach ../cgoc-historical-report 162579679f8baa47d2cb4445d3d82f396e685baa
cd ../cgoc-historical-report
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,report]'
python -m poc report --include-failures
python -m pytest tests/test_report.py -q
```

This regenerates the historical report from public evidence and checks reporting integrity. [Report CI](https://github.com/arlegotin/counterexample-guided-operator-coalescing/actions/workflows/report.yml) checks historical replay at the audited commit and continuation metadata at the current commit separately. The [reproduction guide](docs/reproduction.md) explains the private-input and native-environment requirements for scientific runs.

For the research history, start with the [original blueprint](docs/blueprint.md), [implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md) and [decision appendix](docs/analysis/2026-10-28-decision-appendix.md). [Publication scope](docs/PUBLICATION.md) describes what the public evidence can reproduce.

## License and citation

By **Artem Legotin**. Original software is **AGPL-3.0-only**; original writing and result presentation are **CC BY 4.0**. See [licensing scope](LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

Please cite [CITATION.cff](CITATION.cff), link this repository and identify the commit you used. Development, experiments and reporting used AI assistance under the owner's direction. README editing used [write-like-me](https://github.com/arlegotin/write-like-me).

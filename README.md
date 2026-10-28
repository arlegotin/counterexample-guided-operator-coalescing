# Counterexample-guided operator coalescing

This local study asks whether counterexamples help choose which nonlinear FFNs in a pretrained language model can share an operator bank. Any advantage must survive common capability checks and measured storage and construction costs.

**Terminal outcome: diagnosed inconclusive at WP04.** Sharing recovers some teacher behavior with training, but active selection has no demonstrated advantage. The four functional scoring rules agree within each fixed analysis; the winner flips on fresh documents and near-best uncertainty remains. The original study is primary, the fresh repeat exploratory, and pooling secondary. No trained native shared model at B*, speedup, preserved capability, formal null or equivalence is claimed. Tasks13–24 remain unexecuted.

Read the [audited terminal report](REPORT.md), [machine-readable summary](results/summary.json), [raw metric Parquet](results/regressions.parquet), and [reproduction instructions](docs/reproduction.md). The [decision appendix](docs/analysis/2026-10-28-decision-appendix.md) preserves every controller ruling and its cost if wrong. The subsequent [Approved Task12 review](provenance/task12-review-2026-10-28.md) establishes current review status; the raw gate's pending flag is historical.

The prescribed untrained format anchor is 66,064,401 bytes. The preliminary dense frontier's best eligible point below it is DWQ2/embed4 at 61,327,184 bytes and KL4.27990 on 32 development components. The trained global learning curve uses a different eight-component population; these KL values are not matched comparisons. Teacher development screens, failed attempts, unknown archive sizes, unresolved dynamic allocation reproduction and all numerical/cost limits remain in the report and receipts.

Start with [current progress](PROGRESS.md) and [decisions](DECISIONS.md). The [specification](docs/superpowers/specs/2026-10-27-cgoc-design.md) defines the study; the [implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md) orders its tasks. The [original blueprint](docs/blueprint.md) is unchanged, and the [critical review](docs/analysis/2026-10-27-blueprint-review.md) explains its open questions.

Use the existing pinned Python environment to regenerate the public report without model weights, corpus reads or network access:

```sh
source .venv/bin/activate
python -m poc report --include-failures
python -m pytest tests/test_report.py -q
python -m pytest
```

The [dependency locks](provenance/dependency-locks/README.md) and [reproduction guide](docs/reproduction.md) describe the inherited native/asset gates and restricted-source retrieval. The default CPU suite does not requalify Metal. Scientific jobs are stopped at the terminal diagnostic; protected A/B and official tests remain sealed. Preserve all exposure lineage. Raw text, token IDs, weights and restricted upstream code are excluded from the handoff.

CLI exits are 0 for completed work, 2 for invalid configuration, 3 for an unmet implementation/prerequisite, and 4 for runtime or integrity failure. Completion never implies a positive scientific result. Work remains on the existing branch.

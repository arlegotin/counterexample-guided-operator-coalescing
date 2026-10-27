# Counterexample-guided operator coalescing

This local study asks whether counterexamples help choose which nonlinear FFNs in a pretrained language model can share an operator bank. Any advantage must survive common capability checks and measured storage and construction costs.

The 135M teacher and scoring harness are now qualified on the local Apple Silicon machine. Development results are uneven: ARC Easy 56.8%, ARC Challenge 32.4%, and BLiMP 79.7% on the declared subset; synthetic arithmetic and composition score 0/16 each under the frozen exact-match protocol. These outcomes remain part of the evidence. **No actual compression advantage has been demonstrated.**

The next step is deployment packaging and quantization baselines, followed by the single-merge diagnostic. Final training/search budgets, payload size, protected evaluation and preservation tolerances remain unresolved. Protected readers stay denied until the freeze-verifying evaluator exists.

Start with [current progress](PROGRESS.md) and [decisions](DECISIONS.md). The [specification](docs/superpowers/specs/2026-10-27-cgoc-design.md) defines the study; the [implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md) orders its tasks. The [original blueprint](docs/blueprint.md) is unchanged, and the [critical review](docs/analysis/2026-10-27-blueprint-review.md) explains its open questions.

Use the pinned native Python 3.12 environment described in [dependency locks](provenance/dependency-locks/README.md):

```sh
.venv/bin/python -m pytest
.venv/bin/python -m poc validate --suite component
.venv/bin/python -m poc validate --suite teacher_and_harness
```

The teacher command requires the pinned private local preparation and native MPS access. It freezes development IDs and policies before scoring, verifies local safetensors/config/tokenizer hashes, and records actual numerical and capability results. `--device cpu` explicitly requests CPU execution; unavailable MPS fails without a fallback. Use a fresh `--out` directory for each attempt.

[Teacher evidence](provenance/teacher-qualification-2026-10-27.json), [scoring contracts](protocol/task-scoring.md), and [preparation contracts](protocol/data-preparation.md) document measured limits and data access. Preserve the private exposure ledger: development content has now been accessed, so rebuilding or expanding the pool must validate its full history or fail closed. Raw data, weights and prediction rows are not committed.

CLI exits are 0 for completed work, 2 for invalid configuration, 3 for an unmet implementation/prerequisite, and 4 for runtime or integrity failure. Completion never implies a positive scientific result. Work remains on the existing branch.

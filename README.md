# Counterexample-guided operator coalescing

A proposed local study of whether candidate-specific counterexamples improve the selection of reusable nonlinear FFNs in a pretrained language model, at measured storage and construction cost.

**Current state:** Tasks 1–6 provide strict configuration and run records, native tiny-operation qualification, decoder/bank/loss controls, and pinned public assets with outcome-blind data preparation. The prepared natural pool contains 3,072 FineWeb records plus the named benchmark distributions. Actual teacher runtime qualification is next; no model training or compression-quality result is claimed.

Read in this order:

1. [Critical blueprint review](docs/analysis/2026-10-27-blueprint-review.md) — findings, independently checked arithmetic, source spot-checks, gaps, and proposed resolutions.
2. [Project specification](docs/superpowers/specs/2026-10-27-cgoc-design.md) — requirements R01–R26, interfaces, scientific controls, artifact contracts, and acceptance gates.
3. [Implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md) — 25 tasks with dependencies, tests, verification commands, commit boundaries, and traceability.
4. [Original blueprint](docs/blueprint.md) — preserved research proposal and bibliography.

[Progress](PROGRESS.md) records the current handoff; [decisions](DECISIONS.md) distinguishes inherited requirements, proposed engineering choices, and unresolved execution gates.

Use native Python 3.12 for this foundation:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/python -m poc --help
.venv/bin/python -m poc compile --config configs/pilot.yaml
```

The last command exits **2** because the pilot deliberately leaves budgets and asset pins unresolved. Commands awaiting implementation return **3**, with their unmet prerequisites. Exit **0** means the requested command completed; **4** means a runtime or data-integrity failure. A command returning 0 never implies a positive scientific result. Preparation and component validation execute their documented work; fitting and evaluation still require later implementations and gates.

`load_config(path, phase=...)` rejects unknown or duplicate keys, implicit type coercion, nonfinite numbers, invalid ranges, and manifest paths escaping the repository. Relative configuration and manifest paths resolve against the repository root. `validate_config(config, command=...)` checks command-specific blocking fields. The pilot preserves the blueprint §9.3 values; its additional `budgets` section records unresolved finite ceilings for updates, valid target presentations, teacher/suffix/student calls, search proposals, search and wall seconds, and disk/memory bytes. Counts use integer units; seconds use finite numbers; ceilings are nonnegative and have no unlimited defaults. These fields implement specification R24; WP00 and later gates must supply measured or explicitly approved values.

The primary study requires reported KL temperature **1.0**, primary evaluation context **512**, and residual scale **1.0** in every phase, including preparation. Configuration validation checks bank counts as positive integers; it does not certify compatibility with an unloaded teacher. Tasks 4/7 must use authoritative loaded shapes to enforce `1 <= K <= L`, residual-rank compatibility with projection dimensions, and complete partition coverage before constructing or fitting the bank. Exploratory bank-count sweeps remain possible; protected execution must freeze and reverify the selected valid structure. Repository names are never used as dimension tables.

`RunSpec` snapshots configuration, source/patch/lock/input hashes, hardware policy, seeds, parents, and finite budgets. Its full SHA-256 identity uses sorted UTF-8 JSON with compact separators and one trailing newline. Retries create separate `artifacts/runs/<run_id>/attempts/<attempt_id>/` directories. Events receive monotonically increasing IDs and UTC timestamps. A truncated ledger blocks further writes and remains available for diagnosis.

`BudgetTracker(spec.budgets, attempt).check(next_cost)` preflights each costly operation; `record(actual_cost)` durably records actual expenditure. The shared run ledger retains teacher work across retries and fresh processes. An atomic-operation overrun is recorded before the tracker raises. Disk and memory are high-water observations, while execution counts and seconds accumulate. Unmeasured memory carries a reason. The caller must count recomputation and cache generation again; unique-token increments must count newly seen tokens only. One costly job may be active at a time; checking a budget does not reserve work for concurrent jobs.

With the pinned scientific environment installed, run `.venv/bin/python -m poc prepare --config configs/135m.yaml` to verify the existing private preparation or create it from qualified public sources. The typed `preparation` section records routine byte/record/token/shard ceilings independently of unresolved final study budgets. [Preparation contracts](protocol/data-preparation.md) document role access, windows, exposure lineage and Task 7 interfaces; [retrieval evidence](provenance/preparation-135m-2026-10-27.json) records actual hashes, counts and cost. Protected readers remain denied until the Task 22 freeze-verifying issuer exists.

Actual teacher/harness qualification is next. The current branch is retained. A supported negative or inconclusive experimental outcome is a valid completion of the planned study.

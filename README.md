# Counterexample-guided operator coalescing

A proposed local study of whether candidate-specific counterexamples improve the selection of reusable nonlinear FFNs in a pretrained language model, at measured storage and construction cost.

**Current state:** research and planning documents only. No implementation, dependency environment, model training, or compression result exists in this repository yet.

Read in this order:

1. [Critical blueprint review](docs/analysis/2026-10-27-blueprint-review.md) — findings, independently checked arithmetic, source spot-checks, gaps, and proposed resolutions.
2. [Project specification](docs/superpowers/specs/2026-10-27-cgoc-design.md) — requirements R01–R26, interfaces, scientific controls, artifact contracts, and acceptance gates.
3. [Implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md) — 25 tasks with dependencies, tests, verification commands, commit boundaries, and traceability.
4. [Original blueprint](docs/blueprint.md) — preserved research proposal and bibliography.

[Progress](PROGRESS.md) records the current handoff; [decisions](DECISIONS.md) distinguishes inherited requirements, proposed engineering choices, and unresolved execution gates.

The first proposed implementation task is configuration and run bookkeeping, followed by hardware/operation qualification. Commands shown in the plan do not exist yet. The current branch is retained. A supported negative or inconclusive experimental outcome is a valid completion of the planned study.

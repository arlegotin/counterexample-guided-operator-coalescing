# Project progress

**Updated:** 2026-10-27. **Phase:** Task 2 native qualification. **WP00 compute path passed; full WP00 pending asset provenance.**

## Current handoff

- Original input: [blueprint](docs/blueprint.md), preserved unchanged in root commit `9846751`.
- Completed documentation: [critical analysis](docs/analysis/2026-10-27-blueprint-review.md), [design specification](docs/superpowers/specs/2026-10-27-cgoc-design.md), and [implementation plan](docs/superpowers/plans/2026-10-27-cgoc.md).
- Design builder check: fresh Superpowers reviewer reported “nothing important left.” This is a document review, not scientific validation.
- Plan self-review: requirement/hypothesis coverage, interfaces, failure cases, gate ordering, and task verification commands checked; tiny serialization moved before real-asset preparation to avoid a WP01 dependency cycle.
- Integrated review: identified early payload-anchor ownership, independent-window position semantics, and the full-pipeline random-donor control. Corrections are recorded in D17–D19; no critical issue or additional mathematical contradiction was reported.
- Implemented Task 1: typed strict YAML configuration, finite budget contracts, frozen run identity, isolated retry directories, append-only event/cost ledgers, and CLI prerequisites/status exits. The native Python 3.12 test environment contains only foundation dependencies; no scientific lock or backend qualification is claimed.
- Verification: 91 full-suite tests passed after Task 1 review fixes; CLI help exits 0, unresolved pilot compilation exits 2, and unimplemented preparation exits 3. Tests include teacher-cost persistence across attempts, corruption refusal, all budget ceilings, command-specific unresolved fields, and binding primary temperature/context/residual-scale invariants across phases.
- Review handoff: primary settings are now enforced during loading and direct API validation. Tasks 4/7 still own bank-count/rank/partition compatibility checks against authoritative loaded shapes before model work; positive configuration values alone do not establish asset compatibility.
- Task 2 compute evidence: [sanitized qualification summary](provenance/qualification-2026-10-27.json). Native Apple M3 Max, macOS 26.7.1, CPython 3.12.9 arm64; PyTorch 2.14.1 MPS scientific FP32, CPU FP64 reference and MLX 0.32.4 / MLX-LM 0.32.2 native deployment candidate passed tiny operations. All 72 backend/dtype/operation screens pass; reduced precision remains diagnostic only. Tiny Transformers/MLX-LM Llama and float64 analysis smoke tests ran offline.
- Reproducibility: [67 exact tested wheel artifacts](provenance/dependency-locks/README.md), independently verified SHA-256 hashes, platform metadata and core upstream commits recorded. `uv pip check` confirms 68 installed distributions compatible (including the editable project). A second isolated rebuild is pending and required before WP07.
- Verification after Task 2: full default CPU suite 103 passed with 2 hardware tests deselected; separate native hardware tests passed. Discovery preserves nulls/reasons, refuses Rosetta/native misclassification, isolates a native MLX child abort, records explicit CPU fallback and caps tiny tensor allocation from measured headroom. Raw identifying OS stdout is ignored.
- Last successful qualification: WP00 compute-path only. Full WP00 remains `pending_asset_provenance`; immutable model/data revisions and terms remain unresolved. Actual teacher shapes/numerics, quantizer formats, bank anchor, native export and later gates have not passed.
- Active run/attempt/checkpoint: none.
- Next implementation action: Task 2 independent review, then Task 3 mathematical tiny fixtures on the current branch; establish immutable asset provenance before asset-dependent execution.
- Available qualification command: `.venv/bin/python -m poc doctor --out artifacts/doctor.json`; exits 0 for a passing compute path, 3 for unmet qualification. Host discovery/Metal may require authorized escalation; sandbox denials are not hardware incompatibility.
- Remaining execution uncertainty: immutable assets/terms, actual loaded shapes, full-model tolerances and WP07 useful-scale calibration. Snapshot: 36 GiB RAM, ~8.53 GiB reclaimable-page estimate, ~20.76 GiB used swap, ~196 GiB free disk. Tiny tensors capped at 64 MiB; physical RAM is not presumed available.

## Boundaries

The user approved implementation with subagents and meaningful commits on the current branch. Task 2 installed the stable scientific, deployment and analysis dependencies in the ignored local environment and ran only deterministic tiny local fixtures/models. No pretrained weights/datasets were downloaded. Compute-path qualification is distinct from full WP00/asset/model/quantization/export gates; no study result or training budget is implied.

Future work must read this file and [DECISIONS.md](DECISIONS.md), preserve evidence-linked stop paths, and distinguish proposed settings from frozen experimental choices. Do not create a new branch/worktree. Execution authorization does not resolve scientific budgets, protected-analysis choices, or qualification gates.

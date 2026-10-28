# Continuation recovery implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** Complete the asset-free WP1 foundation without continuing historical science.

**Architecture:** Preserve an untouched pinned checkout. Reuse decoder, merge, fit,
checkpoint, bundle and BudgetTracker APIs; add small fail-closed continuation modules.

**Tech Stack:** Python 3.12, installed Torch 2.14.1 CPU, pytest, YAML, safetensors.

**Spec:** User's WP1 instruction; durable protocol: protocol/continuation-v1.md.

## Global constraints

Historical evidence remains unchanged. No corpus/network access in fixture modes.
Native scientific work is blocked on archive controls, assets, lineage, qualification
and an outcome-blind resource forecast. CPU FP64 equivalence tolerance is 1e-12.
Use 8192 logical targets/update, chunk 32, finite ordered plans, sequential arms.
Record failures/replay and shared costs, keep raw numeric evidence free of content.

## Review focus

Cross-profile loading must reject old artifacts, including compact state/delta.
Budget restarts must retain spending; failed operations must remain charged.
Incomplete target streams and missing mixture sources must never look complete.
Missing candidates/denominators and private content must fail numeric export.
Unverified lineage must block before predictions or corpus access.

### Task 1: Canonical layout and execution identity

- [ ] Add failing tests for contiguous initialization/state/delta, exact values and
  ownership, output tolerance, foreign base and obsolete execution profiles.
- [ ] Run regression tests red; save stride/value/output evidence before change.
- [ ] Apply only the proposed residual-return change; version the runtime profile
  and bind storage/loader identities. Reject incompatible merge lineage.
- [ ] Run focused numerical/serialization tests and preserve evidence.

### Task 2: Bounded contracts and safe evidence

- [ ] Add failing tests for strict schema, finite balanced target plans preserving
  ProbeTrainingSource objects/masks, parent budget exhaustion/replay and raw numeric
  formula reconstruction with complete candidate coverage.
- [ ] Implement poc/continuation.py, poc/continuation_plan.py,
  poc/continuation_budget.py and poc/continuation_evidence.py.
- [ ] Verify contract tests, including corrupt/missing inputs and failed operations.

### Task 3: Executable runner and owned controls

- [ ] Add failing CLI/control tests, including no corpus/network fixture execution.
- [ ] Implement provenance/continuation_recovery.py and owned controls using the
  actual decoder/merge/fit/checkpoint APIs. Keep unavailable archive gates blocked.
- [ ] Run --help, validate-config, owned-controls, bounded profile and blocked pilot.
- [ ] Freeze config/protocol, inventory, environment, commands, claim ledger and
  deferred native commands. Run full CPU suite and review the complete diff.

Implementation uses the user's authorized plan directly; no additional design
approval is needed. Worktrees live in /private/tmp because sibling paths are outside
the writable roots. Original environment is left bound to its original checkout;
each invocation runs from its intended checkout with explicit PYTHONPATH.

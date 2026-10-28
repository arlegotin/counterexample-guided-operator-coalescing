# WP1 continuation evidence

This package covers the new asset-free CPU foundation. Historical WP04 remains
INCONCLUSIVE. No pretrained recovery, acquisition, composition, protected quality
or native deployment result is claimed. The exact archive gates and all native
science remain BLOCKED; see [missing inputs](missing-inputs.json).

Source lives on local branch `continuation-recovery-v1` in
`/private/tmp/cgoc-continuation-v1`; the untouched pinned worktree is
`/private/tmp/cgoc-audit-1625796`. The original checkout was clean at the audited
commit and remains unchanged. Nothing was pushed or published.

| Evidence | Scope |
| --- | --- |
| [Historical reporting tests](cgoc-report-baseline.log), [regeneration](cgoc-report-regeneration.log) | 65 passes; 290 claims, 20 records; generated tracked artifacts unchanged |
| [Baseline CPU suite](cgoc-cpu-before.log) | 568 passed, 1 skipped, 39 deselected on this Darwin host |
| [Before](layout-before.json), [after](layout-after.json) | Same owned FP64 values/digests/census; canonical factor strides after patch; sampled max logit difference zero on this host both times |
| [Review](review.md), [red](review-red-complete.log), [focused final](focused-final.log) | Independent review, adversarial regressions and fixes |
| [Executed commands](verification/commands.json) | Exact argv, checkout, timestamps, exit code and log hashes for final source |
| [Owned controls](verification/owned.json) | Real local 8192-target update, full-probe finite plan, checkpoint/delta reload, source counts, two owned observations, aggregate cost and safe arithmetic matrices |
| [Profile](verification/profile.json) | One tiny owned CPU operation; cannot forecast a 135M pilot |
| [Pilot preflight](verification/pilot-blocked.json) | Deliberate BLOCKED/3, no corpus readers or network |
| [Claims](claims.json) | Provenance labels and explicit scientific limitations |

The new runtime constant changes actual execution identity. Bundles, checkpoints,
bank states, compact deltas and probes bind it; historical artifacts must replay
under pinned source. Parameters/aliases are checked exactly; numerical execution
agreement is scoped to the frozen 1e-12 CPU FP64 control tolerance. The Linux
reload mismatch supplied by the user was not rerun on Linux, and did not recur
on this host. New source files intentionally change a release census; reporting
regeneration belongs only in the untouched pinned worktree.

The final environment receipt includes Python/platform/package versions, an
observed requirements snapshot and its digest, plus references to the pinned
historical wheel/commit manifests. No new environment was installed, no installed
wheel hash was reconstructed from version numbers, and no hash-checked native
rebuild or independent second environment is claimed. CPU kernel RSS high-water
is process lifetime, including imports. Child ledgers and the final parent disk
measurement include private owned checkpoints. Test/review/development invocations
have separate command records; the single owned run ledger does not claim to be
the total cost of this engineering investigation.

The raw numeric matrices in the owned receipt are a hand-specified arithmetic
fixture, separately labeled from actual tiny-model fitting observations. They
exercise N, N+.5S, N+.5(S+R), N+.5(S+Amax), N+.05(S+Amax), canonical ties and
complete common score/damage populations. They are not historical arrays, active
bootstrap replication, acquisition controls or pretrained selector evidence.

Private owned states are reproducible with the exact source/environment/config:

```sh
PYTHONPATH=. /Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing/.venv/bin/python \
  provenance/continuation_recovery.py --config configs/continuation-recovery-v1.yaml \
  --mode owned-controls --out artifacts/continuation-v1/fresh-owned.json
```

Use a fresh output path; existing receipts/private state are never overwritten.
`fresh-owned-private/owned-fit` contains the safe checkpoint generation and delta.
These tiny owned weights are kept under ignored artifacts storage and are not
public metadata. There are no pretrained checkpoints or native byte/memory/timing
frontiers in this package. The candidate pilot recipe, finite proposal ceilings,
source/data sufficiency gates, decision branches and later uncertainty/stopping
requirements are frozen in [the protocol](../../protocol/continuation-v1.md).

The implementation and evidence commits, complete local patch and receipt hash
manifest are recorded in the final local handoff receipt. Content hashes in each
RunSpec bind the code/config/protocol even when an earlier receipt names the
audited parent plus a dirty patch. Historical omitted arrays remain unavailable.

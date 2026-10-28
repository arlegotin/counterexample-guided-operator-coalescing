# Continuation v2 review and corrections

Two fresh read-only reviews inspected code and safe numeric results. Reviewers
did not read corpus contents/weights, run new training, or publish anything.

## Before the real pilot

- A checkpoint disk forecast counted only the current arm, missing sibling arms
  and the transient third generation. Added child-to-parent disk delegation and
  aggregate preflight; prospectively raised the new campaign disk cap from4GB to
  6GiB before pilot quality. No historical cap was edited.
- Final accounting could leave a success status after an overrun. The pilot now
  records failed accounting; the later CLI guard correction also forces nonzero
  exits in artifact comparison helpers.
- Raw training rows lacked the actual mixture objective. Added per-source KL
  sums/counts and the source-weighted objective, verified independently.
- Shared nested work could double-count elapsed time. Changed it to a top-up
  after already-recorded nested work. Counts and failed spending remain durable.

## After completed results

The final reviewer found no numerical validity blocker. Independent checks
confirmed fit/dev disjointness, equal A/B initialized digests, all eight native
raw reductions and4088-target denominators, frozen worker/source hashes, and
reconstruction of original/fresh score/damage vectors.

Two execution findings were addressed with failing regressions:

- CLI helpers could return0 after final accounting set `failed_accounting`.
  `raise_for_accounting_error` now runs after the receipt is preserved.
- Subprocess memory used max(parent,child), although both coexist. Future code
  sums the concurrent process estimates. The original outputs remain preserved;
  they are not retrospectively relabeled as exact physical-peak measurements.
  The recorded worker-stage maxima can be conservatively doubled to bound the
  recorded parent/child counters:8,262,358,074 bytes for the frontier phase and
  7,385,029,690 for legacy native evaluation. Both are below8GiB. Unified/device
  counters overlap and sampling limitations remain.

The first legacy conversion omitted child resource counters. A second isolated
construction, with summed parent/child accounting, reproduced all three artifact
manifests and their referenced files byte-identically. It measured the conversion
process CPU high-water and Metal/MLX counters. It involved no new fitting or
corpus evaluation. This is construction replay in the same environment, not
independent training replication.

The reviewer also noted a preliminary cumulative cost snapshot before final
top-up. Reports use authoritative `aggregate_cost`, not the earlier snapshot.
The final new campaign cost is882.595389 seconds and49 updates, including one
profile update, failed metadata-path work, native comparisons and replay.

## Interpretation constraints retained

- A/B changes the allocation of natural/student/donor presentations at fixed
  total targets; it does not isolate donor harm alone.
- B/C's initial gap explains most of its final gap. Both repair roughly0.03
  nats/target; do not call the SVD advantage faster learning.
- Dense8-bit dominates the new single-merge point and old K15 floating point in
  measured bytes/KL. The smaller shared quantizations trade quality for bytes;
  dense61MB does not strictly dominate every smaller artifact.
- Bootstrap zero concentration is caused largely by identical selections,
  not equivalence or independent episode replication.
- Timing uses three warm fixed-token cached-decode contexts, including the
  guarded runtime's state validation. It is not a production throughput claim.
- Prefix caches were shared across profile/A/B/C and construction was charged
  when first encountered. The raw phase graph does not explicitly encode every
  cache reuse edge. Supplementary dependency interpretation: B/C reuse immutable
  teacher-prefix entries built during profile/A observations; A/B/C all depend
  on verified teacher/assets and fit-only initialization. No cold independent
  pipeline-efficiency claim is based on the per-arm cost totals.

The supplied archive resolved its old missing-fixture gates. macOS product
version26.7.1 resolved the false OS blocker; Darwin25.6 is the kernel version.
Old blocked receipts remain unchanged, with this correction recorded separately.

No minor numerical finding remains unresolved. Explicit limits remain: one
historically informed pair/seed, previously used development data, a short
schedule, no protected confirmation/capability, no real repeated merges, and no
second environment. Future work is a new bounded experiment, not an extension
until a favorable outcome.

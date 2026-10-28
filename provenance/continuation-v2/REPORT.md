# CGOC continuation v2: real recovery and native artifact checks

The sharing mechanism repairs some of its damage on a real model. The counterexample mixture did not help the natural validation endpoint at this budget. Ordinary dense quantization currently offers a much stronger practical point than the new single-merge artifact. Highly compressed shared artifacts can meet the byte ceiling, but their quality is poor.

These are exploratory results on previously exposed development data, not protected confirmation or a capability-retention claim. Historical WP04 remains INCONCLUSIVE; its stops, caps and receipts were not changed.

## What was actually run

On Apple M3 Max / macOS26.7.1, Python3.12.9, Torch2.14.1 FP32 MPS and MLX0.32.4, the exact supplied archive and all25 internal hashes verified. Both diagnostic controls passed on pinned and continuation sources; the metadata audit passed on pinned source. Final tests:611 CPU passed,1 skipped;39 MPS/MLX passed. The existing environment was reused; a second environment or hash-checked rebuild was not performed.

The earlier inference that Darwin25.6 meant macOS below26 was wrong. Product version26.7.1 satisfies the OS floor; native access was hidden by the sandbox and worked outside it. The earlier blocked receipts remain historical records and are superseded by these measured qualifications.

## Real-model recovery

All arms use the same fit-MSE selected parent0, pair(0,15), rank8, seed17 and16 continuous updates of8192 logical targets. A/B have identical initialized digests. C keeps that parent and uses seeded nonzero A/zero B corrections. Teacher, attention, normalization and tied embedding/head stay frozen. Fitting uses conditional teacher suffix KL; the validation below is actual full-model natural KL. Lower is better; zero means teacher equivalence.

| Recipe | Initial validation KL | After16 updates | Reduction | Improving components |
| --- | ---: | ---: | ---: | ---: |
| Parent-SVD, natural/student/donor mixture | 0.223229 | 0.192125 | 13.9% | 8/8 |
| Same parent-SVD, natural only | 0.223229 | 0.189272 | 15.2% | 8/8 |
| Same parent, zero-product corrections, natural only | 0.369757 | 0.338294 | 8.5% | 8/8 |

Natural-only fitting is a sensible default for the next bounded experiment. This is a source-allocation contrast at equal total logical targets: A allocates fewer natural targets and also uses student/donor states. It does not prove donor examples themselves are harmful. Per-arm wall/query totals include different diagnostic workloads and sequentially shared prefix caches, so they are not isolated fitting-efficiency measurements.

SVD initialization helps this real pair at the short tested horizon. Most of B–C’s final difference already existed before fitting; both repair roughly0.03 nats/target. The planted permutation failure remains valid mechanism evidence, but does not show that raw SVD is generally harmful on pretrained layers.

The16-update curves continue improving between8 and16. This demonstrates sensitivity to the allowance, not convergence or adequate fitting for selector-policy claims. A step1 snapshot from this16-update plan is not labeled as a reproduction of the historical one-update schedule.

Fit and validation contain12 and8 distinct components, respectively, with no overlap. Both are historically exposed exploratory populations. The original exposure history was copied before new guarded reads, with original roles preserved. No protected text or labels were opened. Finite plans log every repeated full probe, sliced eligible loss mask, source/depth counts and unconsumed tail. Full records are rebuilt with current candidate identities. [Learning curves](learning-curves.csv), [raw observations, norms and counts](recovery/result.json), [population](recovery/population.json).

## Complete native artifacts on the same data

Every row below uses the same eight windows and4088 valid targets, full vocabulary, temperature1, FP32 native logits and CPU FP64 loss reduction. Each artifact runs its own frozen packaged runtime in a fresh process. Old v2 artifacts/checkpoints are never forced under v3. The old trained K15 state was recovered by exact digest, exported under untouched pinned v2 source, and its native loss reproduces the reported6.49408069095 within6.4e-7. New compact quantizations use existing RTN with packed canonical owners and FP32 private corrections, not learned shared quantization or dense expansion.

| Artifact | Complete MB | KL/target | At66,064,401 bytes? |
| --- | ---: | ---: | --- |
| historical-K15-rtn2-embed4 | 49.484 | 10.474141 | yes |
| historical-K15-rtn4-embed2 | 58.995 | 9.2332356 | yes |
| embedding-dwq-b2-g64-embed4 | 61.327 | 4.4415075 | yes |
| frontier-dwq-b4-g64 | 87.870 | 0.22209896 | no |
| frontier-rtn-b8-g64 | 155.110 | 0.0025285248 | no |
| historical-K15-step29-floating | 388.569 | 6.4940801 | no |
| shared-natural-step16 | 537.217 | 0.18927192 | no |
| teacher-native-noop | 541.680 | 3.075276e-11 | no |

The dense8-bit point is about3.46 times smaller than the new shared FP32 point, with about75 times less teacher-disagreement loss. It dominates that particular sharing point in measured bytes and KL. The old K15 floating point is also dominated. This does not prove all sharing representations or future recovered models are dominated.

The new49.5MB and59.0MB trained shared quantizations meet the historical storage ceiling, but their KL is10.47 and9.23. The61.3MB dense point has KL4.44. These smaller shared points trade worse quality for fewer bytes; the61.3MB point does not strictly dominate every smaller point. No practical quality margin was specified, so this is a descriptive frontier with no binary practical-success claim. The dense point at the ceiling is also far from teacher equivalence.

The table is a frozen subset of previously eligible dense points, selected before these matched predictions, not a newly exhaustively tuned frontier. Dense construction/calibration costs are historical recorded costs, not zero work. Shared construction includes the new pilot’s profiling and recipe comparison, and the older K15 training costs remain separately reported. No equal-construction-budget superiority claim is made.

Payload includes weights, packed scales/biases, tokenizer, config, licenses, requirements and executable runtime. Common package installation is separate. Native B reload differs from its scientific KL by1.27e-7, below the frozen2e-5 decision resolution; native teacher no-op KL is3.08e-11. The real full-rank local control’s largest absolute KL/target is3.63e-9. These are numerical qualifications, not capability or useful-quality margins.

[Native bytes/resident/timing table](native-frontier.csv). Three warm contexts use batch1, context512 and16 fixed-token cached decode steps. Timings include the guarded runtime’s state checks and are descriptive, not stable tails, ordinary generation throughput or a production speed claim. All30 attention layers/KV states remain. Fresh processes contain memory effects; they do not prove the old repeated-reader/serialization memory-growth issue fixed.

## Historical score information recovered

The local original and fresh20-candidate records hash-match the public receipts. Full safe component matrices and per-probe sums/counts are now exported, rather than reconstructed from means. All four historical formulas reproduce exactly; the added N+0.5S rule uses actual student-state rows. Natural/random/active/student-only choose the same pair at each phase’s point estimate: (12,20) original, (4,6) fresh. Natural sampled regrets independently recompute to0.0001156671144 and0.009830384477. These are regrets over the evaluated20, not all435 pairs or the whole shortlist.

Bootstrap active selections match natural in95.45% of original draws and96% of fresh draws. A zero-concentrated interval does not establish equivalence. These reductions condition on the same fixed one-update states/pools; they do not repeat acquisition, fitting episodes or training seeds. Original analysis remains primary; fresh remains secondary. [Safe raw score/damage matrices and source hashes](recovered-scores.json).

## Cost, review and limits

The new native campaign used49 updates including one profiling update, 401,408 logical fitting targets and 882.6 measured execution seconds, including comparisons, the failed metadata-path attempt and construction replay. This is not total engineering time. Parent caps are3600s,49 updates,8GiB recorded memory and6GiB new disk. New disk high-water recorded5,882,417,038 bytes. [Cost references and dependency graph](ledgers/cost-ledger-references.json).

Review found an initial max(parent,child) memory-accounting gap in subprocess stages. Future code sums concurrent parent and child estimates and propagates final accounting failures to nonzero exit. A byte-identical second legacy construction captured its missing process/device counters. A loose reconciliation of recorded parent and child worker maxima is8.26GB, below8GiB=8.59GB. These remain overlapping/sampled device-counter bounds, not proof of exact physical peak memory. Per-process and raw counters remain in the records; no deployment memory-constraint certification is claimed.

Legacy zero FLOP/unique-token counters mean uninstrumented, not zero work. Actual coverage is in finite target plans. Prefix caches were shared sequentially; the raw graph’s phase edges are supplemented in the review record with their cache dependencies. No independent acquisition-efficiency comparison follows from these costs.

A fresh reviewer found no numerical validity blocker and independently checked A/B initialization, fit/dev disjointness, all native raw reductions/denominators, worker hashes and recovered matrices. Review findings, fixes and limits are preserved in [review.md](review.md). There is no second-environment scientific replication. No protected confirmation, competent-task capability result, real repeated-merge result or transfer result is claimed.

## What this changes

The project now has a working native recovery experiment, real learning curves, safe raw historical scoring vectors, verified historical trained weights, and actual compressed shared artifacts below the byte ceiling. The tested sharing recipes still fail to offer a compelling quality–storage tradeoff against ordinary quantization. Natural fitting and better representation/recovery are the justified research priorities; the current evidence does not justify a larger active-selector compiler.

Any next experiment should be separately bounded: test the natural recipe on the predeclared closer sentinel pairs, or a fit-only alignment/representation hypothesis, before composition and fresh confirmation. None of those follow-ups was silently executed here.

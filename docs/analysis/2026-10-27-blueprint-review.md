# Critical analysis of the CGOC blueprint

**Reviewed:** 2026-10-27. **Input:** all 12 sections of [docs/blueprint.md](../blueprint.md), 1,553 lines, preserved unchanged in `9846751`. **Output:** [project specification](../superpowers/specs/2026-10-27-cgoc-design.md) and [implementation plan](../superpowers/plans/2026-10-27-cgoc.md).

## Assessment

The blueprint is a rigorous research proposal with unusually strong controls, not an implemented project. Its defensible question is whether active counterexamples improve sharing decisions enough to justify their cost. Neither the shared-FFN architecture nor intervention supervision alone is a new contribution. The implementation should preserve that distinction all the way to method IDs, budget accounting, and reporting.

The right first deliverable is a trustworthy diagnostic harness with a real teacher and native quantization frontier. A complete compiler is conditional on the diagnostic's information value. This is not a conventional product roadmap where every feature is presumed necessary. Failure to find an effect can be a completed research outcome, provided controls pass and uncertainty is characterized.

There is no existing code, dependency lock, model asset, test suite, or measured local runtime to inherit. The checkout originally had an unborn `main` branch and only the untracked blueprint. The preserved source's numerical toy results describe a different Linux CPU planning environment; they are not tests performed in this checkout. This review performed source inspection and standard-library arithmetic only. It did not install packages, download models/datasets, run training, or benchmark Apple hardware.

## What is already well specified

- The primary model, FFN-bank parameterization, residual scale, ownership rules, depth-specific KV state, and frozen-suffix gradient requirement are explicit.
- The distinction between scientific attribution and practical deployment prevents misleading BF16-versus-BF16 comparisons.
- Natural text, labeled capabilities, teacher competence, synthetic controls, and protected confirmation cover different failure modes.
- Exact bytes, quantization overhead, reconstruction caches, installation, and construction cost have separate boundaries.
- Data roles, document clustering, seed replication, fixed-sample testing, and simultaneous contrasts constrain adaptive selection.
- The comparison set includes ordinary KL, fixed and similarity sharing, intervention training, text hard examples, shared bases, SharVeT, whole-block RRT, pruning/reconstruction, learned quantization, and empirical precision refinement.
- Every work package has an entry condition, output, and gate; unsuccessful outcomes are explicitly legitimate.

These strengths should survive implementation. Replacing them with a minimal train/export demo would not satisfy the research question.

## Arithmetic independently checked

Using the configuration fields, compute:

`ffn_per_depth = 3*d*m`

`attention_per_depth = 2*d*d + 2*d*Hkv*(d/Hq)`

`non_ffn = L*attention_per_depth + V*d + (2*L+1)*d`

`candidate = non_ffn + K*ffn_per_depth + 3*L*r*(d+m)`

| Quantity | 135M | 360M |
|---|---:|---:|
| Teacher unique parameters | 134,515,008 | 361,821,120 |
| FFN parameters | 79,626,240 | 235,929,600 |
| Non-FFN parameters | 54,888,768 | 125,891,520 |
| Rank-8 residual parameters | 1,520,640 | 2,703,360 |
| Half-depth rank-8 bank parameters | 96,222,528 | 246,559,680 |
| One-bank rank-zero equal-precision maximum ratio | 2.337644× | 2.715064× |
| BF16 KV bytes, batch 1/context 8192 | 188,743,680 | 335,544,320 |

The blueprint's table is consistent. The 135M half-bank ratio is approximately 1.398×, and the unchanged non-FFN floor dominates ambitious compression. These are analytical counts, not serialized sizes, memory peaks, speedups, or retained-quality results. The architecture fields were rechecked in the official [135M config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json) and [360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json). Actual tensor census remains mandatory after downloading immutable revisions.

## Consequential gaps and resolutions

| Finding | Why a wrong implementation matters | Resolution in spec |
|---|---|---|
| Boundary API omits an explicit target mask and chunkable suffix hidden output | Could score padding/wrong token or materialize the entire vocabulary tensor | R04–R05: Boundary embeds Batch; suffix_hidden/readout and index contract |
| Generic `logits` can mutate a decode cache twice if used with a separate hidden call | Incorrect position offsets and inflated cache lengths | R04: each cache-mutating call executes exactly once; full boundaries never use cache |
| Frozen parameters can be confused with no-grad execution | Local replacement receives no learning signal | R04/R08: separate target branch and differentiable candidate-through-suffix branch; gradient parity fixture |
| Chunking alone does not guarantee correct token/update weighting | Ragged chunks and microbatch lengths bias both training and evaluation | R08–R09: sums over exact valid-target denominator and tied-head two-stage gradients |
| Exact 8192-target accumulation is underspecified | Memory changes can silently alter optimization exposure | R09: divide target masks deterministically, carry cursor, count recomputation |
| Single-merge active scoring and active refitting are interleaved | Ranking advantage can actually be a better-trained candidate | R14: scoring-only diagnostic first, separate matched-refit experiment |
| Comparison metrics vary in information/search cost | Equal training tokens alone are not equal total construction effort | R09/R16: separate equal-query, equal-fit, and equal-time tracks with shared preparation allocation |
| Manifest self-hash/size would recurse | Bundle verification or exact byte accounting cannot be implemented consistently | R17: file entries exclude manifest; detached digest; actual payload includes manifest |
| Run identity includes mutable outcomes unless defined carefully | Retry/output writes can change the supposed run identity | R24: immutable RunSpec hash, separate unique attempt, output hashes added later |
| Checkpoint directory replacement is not safely specified for nonempty destinations | Crash can leave neither a usable current checkpoint nor coherent history | R24: immutable generations plus atomic pointer, previous generation retained |
| Topology digest and full state digest are different concepts | Equivalent partitions can loop; changed weights can reuse stale probes | R13/R15: separate canonical partition and state digests, explicit invalidation |
| B* refers to an architecture without fixing the defining artifact | Candidate-specific map/code metadata can shift the budget after selection | R19: deterministic SEQUENCE format anchor; all actual finalists still satisfy its bytes |
| Search byte savings can be confused with quantized payload savings | Topology can be scored with hypothetical bytes | R15/R19: actual canonical scientific-format bytes in search; separate native payload ceiling |
| Preparation must read data that later becomes protected | Absolute “never read protected data” would make deduplication impossible | R11: outcome-blind preparation then role-restricted access; honest local isolation boundary |
| BLiMP has no official train/dev/test partition in the blueprint | Smoke tests and protocol choices could leak protected answers | R12: explicit proposed 10% development/90% protected split within every paradigm, labeled subset reporting |
| Natural sampled windows lack a precise overlap/offset convention | Different contexts/duplicate targets can change KL denominators | R12: saved deterministic starts, no within-document overlap, explicit short-document behavior |
| ARC prompt/answer token boundary is ambiguous under BPE | Scoring answer tokens alone may fail to represent the stated continuation | R12: full concatenation offsets, validated stable boundary/separator, refusal on ambiguity |
| Near-duplicate components can transitively span roles or sources | Exact per-document hashing misses leakage | R10: connected components before splitting; cross-source preparation screen and limitations |
| confirm_B comes from the same web pool | Calling it independent-source evidence would overstate replication | R23: independent held-out components, fresh-source claims require separately untouched data |
| A crash after partial protected evaluation is still information exposure | Restart could allow tuning on already observed outcomes | R23: sealed/in_progress/consumed access ledger; only identical-freeze resume |
| Missing or unsupported baselines could disappear from tables | A deployment win against a weak subset becomes a broad superiority claim | R16/R25: reasoned eligibility matrix, claim limitations and terminal report support |
| CKA/correlation/standardization can divide by zero | Degenerate fixtures produce favorable NaNs or unstable groups | R16: explicit flagged fallbacks; no fabricated functional evidence |
| A planted bank is not necessarily uniquely identifiable | A correct heuristic could be rejected for not recovering one arbitrary labeling | R07: oracle equivalence and recoverable cases, not universal topology recovery |
| Adapter-bearing K=L is larger than the teacher | Identity baseline could be assigned the wrong byte cost | R06/R19: separate adapter-free teacher and matched rank-bearing identity |
| Fresh process does not imply cold OS page cache | “Cold load” timing can exaggerate or understate deployment behavior | R21: name process/cache boundaries separately |
| Mutable upstream APIs are cited as executable assumptions | Dependency drift can invalidate quantizer adapters and shape support | R02/R18: exact local resolution, imported-binding tests and capability matrix |

The BLiMP split, window policy, archive codec, anchor map, access-token design, data normalization, and module decomposition are author proposals. They are exposed in the spec for review. They are not claimed to have been experimentally validated or approved by the user.

## Numerical and scientific traps that deserve early tests

1. Reverse KL can look plausible while changing the objective; use an asymmetric analytic fixture.
2. Padding-ID equality cannot distinguish real EOS from padding; masks come from lengths.
3. Permuted FFN neurons preserve the function but defeat naive weight averaging; test the whole gate/up/down transformation.
4. Separate low-error replacements can have large joint error; every accepted merge needs a complete-model check.
5. Student-state boundaries must refresh after upstream changes, including weight-only changes with the same partition.
6. A tied embedding/head has two gradient paths and one stored owner; check both gradient sum and actual file contents.
7. Groupwise quantization can duplicate shared matrices or add shape padding that erases nominal savings.
8. Folded adapters may make inference fast by expanding storage; such an artifact cannot claim the compact bank's bytes.
9. Runtime enqueue timing and summed unified-memory counters are not actual completion time or unique memory.
10. Document tokens, candidate pairs sharing depths, and repeated seeds do not create independent samples.
11. No significant accuracy loss is not evidence of negligible loss; capability margins remain externally supplied.
12. Any result seen during a failed confirmation is still consumed evidence for future model selection.

## Source spot-checks performed on 2026-10-27

This review rechecked implementation-critical sources, not every paper, legal term, or numerical result in the original bibliography. The preserved blueprint's other source claims remain attributed to that earlier research. No cited implementation was reproduced here.

| Source inspected | What was confirmed | Limitation |
|---|---|---|
| Official [135M config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json), [360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json) | Llama shape fields, tied embeddings, context and RoPE settings underlying the arithmetic | Main-branch inspection is not immutable local asset resolution |
| [MLX learned quantization documentation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LEARNED_QUANTS.md) | DWQ, AWQ, dynamic quantization and GPTQ paths exist and use calibration | No local kernel/shape/gradient test or version lock |
| [MLX calibration loader](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/utils.py) | Default loader reads a home cache and may download a separate text file | Must replace the actual binding in the pinned version and test every quantizer |
| [MLX Llama implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/llama.py) | Separate attention/cache structure and grouped-query projection dimensions | Native bank implementation and full export parity are still future work |
| [SharVeT repository](https://github.com/snudatalab/SharVeT) | Declared PolyForm Noncommercial terms, similarity/rank/refinement organization | No permission for this repository is inferred; actual asset terms need preservation |
| [SharVeT similarity source](https://raw.githubusercontent.com/snudatalab/SharVeT/main/src/similarity.py) | A common embedding sample feeds projection similarity calculations | Down-projection shape issue is conditional on the configured path; not a blanket claim of broken upstream results |
| [SharVeT ACL record](https://aclanthology.org/2026.acl-long.2213/) | Paper identity and similarity-aware sharing/refinement contribution | Record/abstract check, not a fresh full-paper reproduction |
| [RRT record](https://arxiv.org/abs/2410.20672) | Prior recursive sharing with depth-wise LoRA | Published performance and throughput are not imported into local claims |
| [XMerge record](https://arxiv.org/abs/2609.02083) | Adjacent-block reconstruction and cross-axis selection are current comparison mechanisms | Local adaptation remains distinct from the paper |
| [reViT record](https://arxiv.org/abs/2610.12448) | Depth-programmed FFN banks and compact versus expanded deployment are prior concepts | Vision evidence does not establish language-model results |

## Scope and risk ordering

The largest early risks are invalid teacher/boundary semantics, unavailable local kernels, and an active diagnostic that cannot improve selection over ordinary natural KL. The largest later risks are composition drift, expensive weak baselines, sharing-destroying export, and underpowered or adaptively contaminated confirmation. Their ordering is the reason for the staged plan.

Do not turn WP05 into an immediate all-method sweep. Implement and prove interfaces on tiny fixtures, establish each comparator's feasibility early, then spend adaptation budgets only after the diagnostic gate. All mandatory baseline families remain in the specification; actual unsupported/infeasible implementations are recorded rather than quietly removed. The report must distinguish mechanism evidence, deployable artifact advantage, and comparative limitations.

Hardware memory, dependency pins, exact asset revisions, numerical tolerances, useful effect resolution, final sample/seed counts, construction-time ceilings, and capability tolerances are intentionally unresolved. The spec assigns each to a gate. None blocks writing an actionable staged plan; several must block execution or confirmation until measured.

## Blueprint coverage map

| Blueprint section | Design coverage | Planning consequence |
|---|---|---|
| 1 Executive assessment | R01, R19, R25–R26 | Diagnostic before compiler; no compression promise |
| 2 Problem/hypotheses | R01, R16, R22, R26 | Separate H1–H5 evidence and claim boundaries |
| 3 Landscape/novelty | R01, R16, source ledger | Baseline fidelity and prior-art update before final report |
| 4 Mechanism | R04–R09, R13–R15, R17–R18 | Boundary, gradients, ownership, probes, finite search, export |
| 5 Evaluation | R10–R12, R16, R19–R23 | Common data, complete comparisons, fixed populations and statistics |
| 6 Feasibility | R02, R08–R09, R17–R18, R21, R24 | Measured hardware and resource gates |
| 7 Work packages | R25 plus plan task/gate map | Dependencies and terminal paths remain explicit |
| 8 Failure program | R07, R13–R16, R20–R23 | Negative controls and attribution/robustness test matrix |
| 9 Repository/contracts | R03–R04, R10–R11, R20, R23–R24 | Named modules, schemas, CLI, records and resume |
| 10 Interpretation | R01, R22, R25–R26 | Negative/inconclusive completion and conditional scale |
| 11 Audits/toy checks | R07–R08, R25–R26 | Reimplement reference controls; do not relabel prior checks as current passes |
| 12 References/integrity | R02, R10, R16, R26 | Pin execution assets; preserve evidence limitations |

## Review record

The specification receives the Superpowers builder check before handoff. The implementation plan receives a separate self-review for requirement coverage, exact interfaces, task boundaries, negative tests, and feasible verification commands. Review findings and their resolutions belong in the decision log; a passed document review is not an experimental result.

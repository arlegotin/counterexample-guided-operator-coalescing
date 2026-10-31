# Whole-feature review — dense-baseline-expansion-v2

Reviewed checkout: `/Users/artemlegotin/.codex-a/worktrees/8d38/counterexample-guided-operator-coalescing` (W).
Base/HEAD: `80a8b8389ecf60b8e23eccd29971c2c1cec380fd`; qualified parent U is worktree29d7. Reviewed all 17 manifest paths, the qualified-parent diff, binding v2 specification amendment, plan, and ledger rulings. All 17 supplied file hashes matched when checked. This is the one fresh whole-feature review, not a post-fix re-review.

No checkout/index/HEAD/branch mutations; no native/MLX imports, model execution, weights, raw corpus files, or protected data were read. Only source, metadata, and the existing CPU log were inspected. This report is the only authored artifact.

## Findings

### R1 — Important: DWQ optimizes floating normalization parameters beyond the declared scales/biases-only recipe

Location: `poc/quantize.py:386` (DWQ setup and dispatch), `poc/quantize.py:313` (official model initialization); coverage gap in `tests/test_dense_native_controls.py:27`.

The newly constructed official model is never frozen. `nn.quantize` freezes the replaced quantized modules, but leaves RMSNorm weights trainable. The installed `mlx_lm/quant/dwq.py:89-99` only unfreezes affine scales/biases; it does not freeze the model. Its optimizer consumes `model.trainable_parameters()` at lines151-155. Consequently v2 DWQ updates floating normalization weights as well as scales and biases. The counters and current six native controls can all pass while executing a different optimization recipe from the binding specification.

Practical fix: for explicit `dense-native-contexts-v2` only, freeze the complete quantized model before the upstream DWQ routine unfreezes its supported affine parameters. Record and check the exact allowed trainable-owner inventory. Add an owned negative control and an actual native assertion that floating norm owners remain unchanged and that every trainable leaf is an allowed scale/bias. Preserve the historical None/raw-v1 behavior.

### R2 — Important: direct full-preparation admission does not enforce the measured freeze or the full live execution binding

Location: `provenance/dense_frontier_execution.py:81-110`, particularly lines93-109; `provenance/dense_frontier_execution.py:113-122`.

The direct entry verifier checks that the phase plan's resource equals the supervisor's resource, but never loads `preparation-freeze.json` or recomputes the resource forecast from the accepted short preparation profile. The subsequent check validates only the recipe/work plan. A valid generic finite supervisor can therefore dispatch a full preparation with a newly authored cap, without any accepted short profile, or with a cap different from the measured freeze. In addition, the return value of `require_dense_qualification` is ignored here: the verifier checks current qualified files but never requires that complete source/reference closure in the live supervisor's frozen-files table, and does not compare the live guard's execution environment. The normal orchestrator supplies these fields, but the advertised independent direct-entry gate does not enforce them.

Practical fix: at direct entry, derive the exact expected profile/full phase resource from immutable admitted preparation evidence; require the freeze/profile/acceptance/terminal references in the live guard; recompute the forecast; compare the complete required source/reference closure and exact `dense_execution_env` plus standard cleared environment settings. Reject missing or modified evidence before backend/content work. Add direct-entry controls for missing freeze, changed measured cap, omitted source ref, and changed environment.

### R3 — Important: restored native runs trust resource-profile JSON without re-admitting the measured profile operation

Location: `provenance/dense_frontier_campaign.py:263-289`.

The resource capsule saves `profile_acceptance` and `native_profiles`, but neither `require_freeze` nor `run_campaign` consumes their guarded operation or terminal cost evidence. Run only reads the profile JSON and feeds its timing fields back to `forecast_worker`. The run YAML/resource capsule are intentionally excluded from the static source/scientific identity, so updating their hashes and the frozen run protocol does not establish that replacement timing fields came from a successful measured profile. A stale, mismatched, or altered profile packet can therefore become an apparently measured frozen budget. The original in-process profile acceptance is not a sufficient restore-time gate.

Practical fix: implement one saved-profile admission routine that rechecks the exact profile protocol, copied worker, source/config/artifact/input bindings, successful guard command/environment/caps, result identity and counts, common phase packet, terminal ledger, and output hash. Compare the capsule's profile references to that accepted operation before deriving run budgets; enforce the same relationship during analysis. Test changed timings with correspondingly updated capsule/run hashes, changed profile stage identity, and missing terminal evidence.

### R4 — Important: global resource-planning failure loses the promised complete 18-outcome handoff

Location: `provenance/dense_frontier_campaign.py:209-210`, `:241`, and `:259-260`; related run loop `:287-289`.

Per-recipe failures are retained, but aggregate envelope checks are outside those catches. For example, six successful short profiles can each forecast 6000 seconds (within the 7200 second per-phase ceiling), while the 36000 second sum exceeds the 28800 second campaign ceiling. The controller then writes only `profile-failed.json` and exits before preparation/resource freeze, evaluation index, or an 18-point provisional report. The native-profile sum check has the same behavior, after more model work. A restore-time forecast mismatch also exits before recording a label outcome. These are expected finite-resource failure branches, not merely hypothetical process crashes, and they omit explicit terminal outcomes for the fixed grid.

Practical fix: reserve report/closing resources prospectively and produce a terminal complete-or-failed inventory on campaign infeasibility. Mark affected not-run points explicitly as resource/admission failure with no metrics; preserve all successful preliminary artifacts, profile costs, failures, and imported references. Do not launch work that cannot fit, truncate schedules, or retry. Add an owned orchestration control where every individual forecast is valid but their total exceeds the campaign envelope, and verify the final 18-outcome handoff and cost links.

### R5 — Important: imported cost ancestry stops at M's common phases

Location: `provenance/dense_frontier_execution.py:407-420`; `provenance/dense_frontier_campaign.py:317-325`.

The importer calls `admitted['phase_costs']` the upstream preparation/failure history, but that table contains only M's 38 common phases (input-validation, 12 prepare, 12 profile, 12 run, analysis). It does not contain the separately prepared shared-calibration statistics, the recovery work that created the shared floating parent, or those ancestors' retained failed work. M's pinned config explicitly references the shared-calibration packet/acceptance and the 256-step scientific observation. The calibration packet even exposes a separate `terminal_cost_path` and labels its embedded `cost` as preliminary. Those terminal/ancestor ledgers are not re-admitted or surfaced by the new cost report. References to the old config or frozen packet are provenance links, not current validation/reporting of the omitted terminal expense.

Practical fix: import a bounded, explicit ancestor-cost inventory from those existing admissions; verify final terminal ledgers and failed-attempt records, retain their original identities, and report them separately from M, new scientific phases, and qualification expense. Deduplicate shared costs instead of allocating them silently to every point. If an ancestor cannot be established, mark its cost unknown and block any claim that all upstream expense is present. This does not require repeating model work or altering old receipts. Add a metadata-only negative control for a missing or changed calibration/recovery terminal ledger.

### R6 — Important: the original 512-parent/first 80 selection proof is not enforced

Location: `provenance/dense_frontier_study.py:49-50`; `provenance/dense_frontier_execution.py:132-139`; `configs/dense-baseline-expansion-v2-proposal-v1.yaml:fit_manifest`.

Both metadata and direct preparation treat the already-reduced 80-record manifest as the source population. `calibration_plan` proves first 80 ordering within those 80 records, not derivation from the explicitly required original 512 pool. Its `lineage_sha256` is carried onward without verifying the parent manifest. The separate proposal-lineage record only binds a prior 80-record manifest and is not consumed by these gates.

Read-only metadata verification during this review found that the CURRENT child is correct: its 80 records, order, and component membership exactly match the first 80 distinct components sorted by document ID in `/Users/artemlegotin/.codex-a/worktrees/e789/counterexample-guided-operator-coalescing/data/recovery-diagnosis-v1-population/prepared/fit-manifest.json`, SHA256 `604eb83e032a2ed375751af797e55a8827c546eab70d07b1ca78af6d6d3817ad`, containing 512 fit records. Therefore this is a missing admission invariant, not evidence of current population overlap or outcome-based selection.

Practical fix: bind the original parent manifest explicitly, verify its exact identity and lineage digest, derive the 80 selection from it, and compare the complete child record/membership/role inventory before content or gradients. Freeze that derivation evidence in source/phase admissions. Add controls for a valid 80-record child selected from the wrong positions, changed membership, and fabricated lineage.

## Evidence and successful boundaries

- Existing `/private/tmp/cgoc-dense-controller-cpu-v1.log` reports 174 passed, 11 deselected in 4.89s. I did not rerun tests or treat this log as final source qualification.
- All 17 review-manifest hashes matched.
- Source reconstruction of the deterministic inference runtime archive equals the historical teacher manifest's runtime SHA256: `26311d3baedba6b3483f943b4e765c1087ce7d48c73aa9a39bc7429fa699489d`. No artifact archive/weights were opened.
- The actual 80-record metadata selection is correct, as described in R6. No raw text/token file was opened.
- The v2 accounting distinguishes 511 DWQ distributions from 510 shifted pairs in the 511-token input; the optimizer allowance is positive and Adam avoids the new-profile loss double charge. Historical None/raw-v1 dispatch accounting remains separated.
- The original twelve measurement identities, same runtime/input requirement, fixed six recipes, failure-without-invented-quality representation, source archive verification, and explicit provisional final status are well represented in the implementation.

## Behavior deliberately not judged

1. Native algorithm feasibility, numerical fidelity, actual trainable updates on hardware, and the 17 native qualification cases: no hardware controls were executed, per the serial-slot restriction. Static evidence for R1 is sufficient to identify its scope mismatch, but its repair still needs the authorized native qualification.
2. Actual new 18-point frontier or practical/capability equivalence: the six new scientific points do not exist yet; application tolerance is absent. No scientific acceptance is inferred from code, metadata, or CPU tests.
3. Full weight payload/owner integrity of historical or future artifacts: I inspected code and metadata, not model binaries. Runtime identity was checked by source reconstruction versus pinned manifest metadata only.
4. Raw fit-token eligibility and source-text integrity: raw corpus access was prohibited. The source implements the intended 512-token checks, but actual eligibility remains an execution admission.
5. Current completion of Z and live accelerator availability: no neural work was started and no upstream run was independently accepted here. The reviewed serial code requires the stated terminal handoff; this review does not release the slot.
6. Independent saved-result admission: intentionally downstream and not implemented in this reviewed package. The campaign must remain provisional until that separate consumer validates final artifacts/raw results/uncertainty/cost ancestry.

## Verdict and readiness

RED for source freeze/execution readiness: six Important findings, zero Critical findings. Resolve the material findings in the authorized single fix pass and record their exact-source resolution; no second whole-feature review is requested. Then perform the required full CPU/public qualification and the source-bound 17 native controls only after the authorized serial release. This review grants no native/scientific acceptance, merge, push, or publication readiness.

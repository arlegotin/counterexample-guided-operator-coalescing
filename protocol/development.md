# Controlled single-merge development diagnostic

Task12 precedes compiler search. The outcome-blind metadata proposal is saved in
`artifacts/diagnostics/task12/proposal.json`; its exact component IDs, all435 pairs
and strata must be bound into every measured RunSpec before model work.

The current bank starts with30 singleton groups and rank8 residuals, exactly
teacher-equivalent. Each sampled pair produces one29-owner trial. Both existing
parent operators are tried, every affected-depth residual is initialized by
CPU FP64 SVD of original teacher minus proposed owner, and the medoid with the
smaller mean fit activation reconstruction MSE is selected; ties follow canonical
group order. Unaffected singleton residuals retain rank and bytes. This starts at
the specified search origin; the Task11 K15 partially trained checkpoint is not
silently substituted. Scientific FP32 search bytes never redefine native B*.

The proposal uses8 calibration fit components and12 separate medoid/fitting
components. Natural development score8 and damage16, intervention probe_score
score8 and damage16 are all source-component-disjoint. Natural text uses the
first canonical saved-policy512-token window per document (max4, seed17), with
source offsets, reset positions, masks and token hashes retained. Intervention
text uses the explicit first512/reset0 policy. Allowed exposure readers precede
text/token access. No protected A/B or official-test content is read.

Full30-depth fit-only calibration uses all511 valid positions per calibration
window. There are4088 **unique entropy positions within8 source components**;
they are not4088 statistically independent tokens. Reusing the natural entropy
at30depth labels does not multiply the observations. The historical Task10 raw
receipt's `independent_entropy_positions` wording is interpreted as unique
positions and left unchanged to preserve source binding. New receipts use
`unique_entropy_positions` and retain document/component counts. Norm/entropy
quantiles .01/.99 are frozen before candidate outcomes; unfiltered stress rows
are always retained, and candidate damage never determines an exclusion.

All435 unordered depth pairs are available. Sampling uses midpoint-depth thirds
crossed with separations1–3,4–9,10–29, with two seeded pairs per nonempty stratum.
The common shortlist is the union of8 weight-nearest,8 adjacent-original-depth,
and8 seeded remaining pairs. Save every pair's distance, stratum, inclusion and
shortlist membership; add at most2 random excluded pairs if the base sample has
fewer than2. Report observed excluded winners and coverage; avoid an unsupported
whole-population miss-rate claim from this small unequal-probability design.

One common local fit allowance is one8192-target update with fresh AdamW,
head32, full-vocabulary temperature1, natural/student/donor weights .5/.25/.25,
and complete affected-group ownership. Both parents receive identical medoid
fit records; score methods receive the same initialized/fitted trial digests.
No score method can invoke fitting. Any future active refit is a separately named
experiment with matched competing access; it is disabled here. Preserve original
full ProbeRecords when splitting training loss masks. Prefix-producer digests
remain unchanged when a compatible teacher FFN/suffix consumes the boundary.

Primary comparisons are weight distance, activation MSE, centered linear CKA,
natural downstream KL, random donor KL and active donor/student-state KL. Equal
query/data exposure and total construction cost are reported separately. Shared
initialization/fitting and pool-generation cost is charged to every method's
construction accounting; cached reuse does not become free computation.

Regret is selected damage minus minimum damage **within the sampled set**.
Empty, invalid, missing or nonfinite trial populations invalidate the comparison.
Source-component resamples rerun selection and regret, using shared draws across
methods; overlapping pairs are never bootstrapped as independent observations.
Final diagnostic damage is separated from score selection before outcomes.

Resource proposal: profile ceiling600s, total Task12 ceiling7200s including
profile/retries; maximum20updates/163840targets,300000teacher calls,
20000student calls,1000 each suffix/student backwards,10000 proposals,
900search seconds,2GiB retained diagnostic disk and8GiB conservative RSS+Metal
memory observation. Trials run sequentially, releasing models/optimizer snapshots
before the next. Forecast240s/trial starts from the measured91.7271s Task11 local
operation and adds full scoring/damage allowance,600s calibration/profile and
1200s overhead margin. The profile must justify proceeding. Task12 carry-in is0;
Task11's2079.0169s historical receipt remains separate and is never erased.

Planted/null selection arrays, a pairwise-success/joint-failure control and the
known neuron-permutation initializer sensitivity accompany real-model evidence.
Raw unaligned matrix SVD is not neuron alignment: an insensitive result may show
initializer/optimization failure rather than absent redundancy or active signal.
No capability, superiority, or equivalence tolerance has been supplied.

WP04 may continue only for observable active information or a concrete joint
composition question. A precise simpler-method tie can narrow/stop; low
sensitivity requires diagnosis. A legitimate terminal gate points to Task25 and
explicitly leaves Tasks13–24 unexecuted. Budget exhaustion alone is not a
scientific terminal result.

## Controller rulings before the first pretrained workload

The amended saved plan uses ordinary natural KL on the **same eight probe_score
recipient components and first512 contexts** as every main activation/intervention
score. Canonical development8 natural KL is a separately labeled practical
comparator. Before candidate predictions, try at most4 deterministic edits per
candidate document, using frozen disjoint candidate orders of24 score and48
intervention-damage components. Take the first8/16 with aligned nonidentity edits
inside context; preserve every unsuccessful attempt. Pool generation is shared
and charged once in the aggregate, with its construction/query allocation shown
for each method. This is an equal-pool score comparison; an efficient random-only
acquisition pipeline would need fewer queries and is not claimed measured here.

Fixed lower-is-better formulas: `natural_kl`; `natural_kl + .5*student_KL +
.5*seeded_random_donor_KL`; `natural_kl + .5*student_KL + .5*max_donor_KL`
(primary active); and `natural_kl + .05*student_KL + .05*max_donor_KL`
(fixed secondary). Each conditional term averages equally over affected depths
and source documents. The primary contrast is active1 versus ordinary natural
KL selected damage/regret. Weight distance, FFN-output MSE, and1-centered CKA
are separately reported. Final damage cannot tune these weights.

The controller amended caps to22 optimizer attempts/180224training targets,
350000teacher and30000student calls; all other aggregate caps above persist.
The extra2 optimizer attempts cover bounded profile/failed work; they do not
increase the maximum20 scientific candidate population. Fresh available RAM
was18.13GB, then14.90GB after data preflight; refresh before allocation and use
the8GiB cap only with at least10GiB available. RSS, Metal allocated and driver
observations are recorded separately and their overlap is explicit.

Preflight took57.89784004096873s, carried into Task12 aggregate accounting.
One metadata-selected natural-damage document had only508 actual tokens;
its frozen-order replacement occurred before any predictions, with original ID
and reason retained. The bounded600s profile includes this carry-in and retries.
It evaluates fit/main score/development8 secondary only. Final natural16 and
fresh-intervention16 predictions stay sealed until the protocol and full-phase
ruling are fixed. The profile's fitted changed-owner delta is retained with exact
base/initialized/fitted digests; delayed damage restores it without refitting.

## Frozen full-phase amendment

The numerical profile completed pair(0,15), one8192-target update and all score
observations, saving its changed-owner delta before a generator/list mismatch
failed a progress-print statement. Its raw status remains failed and attributed
to ed6193c. The source-bound derived profile receipt preserves exact raw hashes;
the logging fix materializes the iterable and has a failing-then-passing test.
Original calibration, profile trial, delta and failed result remain immutable;
deferred endpoint additions use new `trials-final` artifacts. Exact old/new source
maps and reused-artifact hashes are recorded in the source amendment.

The profile spent155.856894s including57.897840s preflight,4337teacher/555student
calls, one update8192targets and17backwards of each kind. The maximum observed
RSS+driver sum was6,712,950,784bytes. Separate RSS/driver/allocated maxima occur
at different times and must not be added into an invented simultaneous peak.
The fixed fit-check mean KL fell0.15911549 to0.15506479; this shows first-step
learning, not convergence. The source-bound profile receipt includes adverse
initializer and pairwise/joint controls.

All18 original stratified pairs happened to be outside the23-member deduplicated shortlist (the union cap was24).
Before any endpoint predictions, the controller approved two uniformly seeded
(seed18) shortlist additions: (1,5) and(1,2). Primary comparisons use all20
candidates. A frozen no-extra-query secondary analysis repeats selection/regret
on the original18 only. Neither population replaces the other according to its
outcomes. Coverage/miss estimates concern18 sampled excluded versus2 sampled
included pairs; they are neither a complete shortlist test nor a435-pair rate.

Within the unchanged22-update/180224-target cap, one further8192-target update
with fresh Adam is predeclared for(0,15) and(1,5). These are separately named
sensitivity-refit RunSpecs, initialized from each retained primary one-update
state, using the same fit cases and default mixture. Fixed fit/main-score checks
measure remaining optimization sensitivity. Their states never enter the
primary20-trial or secondary18-trial ranking. Primary and sensitivity variants
share one aggregate budget. No convergence or active-refit efficacy claim follows
from these two additional steps. Full phase was approved against a conservative
~3006s eighteen-trial forecast with ample margin under7200s; the20-trial amendment
and two extra updates remain within those existing caps.


## Fixed fresh-component precision repeat and terminal WP04

The preceding proposals and full-phase cap statements describe their historical
stages. Later authorization files version the changes below; historical plans,
RunSpecs, failures and receipts were not rewritten.

After original20+2 completion, the controller fixed one exploratory precision
repeat on the same20 primary one-update states. Freeze seed19 data order before
text eligibility: up to48 new probe_score candidates, at most4 deterministic edit
attempts per candidate, first16 eligible full512/reset0 recipients with positive
after-patch target masks;32 new canonical dev damage components, using a saved
64-candidate order. These populations exclude all prior selected score/damage/dev8
components. Preserve rejected/underfilled attempts without quality filtering.
All16 score components filled;33 dev eligibility checks yielded32 full windows
and one preserved508-token rejection. The frozen pool has1890 accepted and1
rejected rows. No new intervention-damage expansion was performed.

All formulas, temperature1, full vocabulary/head chunk32, per-component weights,
random donor policy and2000-resample component bootstrap stay fixed. Every
resample reselects candidates and recomputes oracle/regret. Original study stays
original primary; fresh20 repeat is separately labeled; original18-only remains
a fixed secondary in both studies; pooled24score/48damage is explicitly an
exploratory secondary. Report all seven main methods regardless of outcome.
Canonical dev8 remains an original secondary, not fresh data. Shared query
allocation is not the necessary cost of an efficient random-only pipeline.

Only2 original compact primary states had been retained. The controller explicitly
authorized18 fully charged exact reconstructions, first(12,20), with no retries
on digest mismatch: original initialized and fitted digests, data/masks/order,
seed/schedule/numerics must match. All18 matched, and all20 saved state identities
and40 metadata/weight hashes were verified before any fresh predictions. Compact
deltas now remain on disk. Replays are measurement/reproducibility work, not extra
primary learning. Primary exposure stays one update; two sensitivity states stay
separate. No new optimization was used for the precision comparison.

Cumulative caps changed22→40 physical optimizer operations and180224→327680
training presentations; student forwards30000→50000. Teacher350000,1000backwards
of each kind,2GiB generated disk,0fetch and overall7200s remain unchanged.
Follow-up authorization initially2400s then3600s including reconstruction, finally
4800s after measured accounting overhead, with<=160000additionalT/24000S,
<=2000new proposals/300search seconds and2.5M fresh logical evaluation positions
separate from training presentations. Explicit source/cap amendments are
`task12-precision-{amendment,durability-amendment,memory-amendment,evaluation-amendment,accounting-amendment}.json`.

The failed8GiB preparation attempt remains failed with observed RSS+driver
8,618,934,272bytes. Only subsequent authorized RunSpecs use9GiB, conditional on
fresh available RAM>=12GiB. RSS, MPS allocated and driver values remain separate;
the conservative RSS+driver sum may overlap allocations and is not physical
unique memory. Reuse saved edit pools/windows exactly; persist every eligibility
attempt before advancing. Fresh isolated evaluation uses frozen tokens/JSON and
no role readers. Repeated-reader/serialization growth remains unresolved.

The first fresh evaluation cost283.925990s and was retained. CPU-only diagnoses
identified exact disk census overhead. A tested direct scandir recursion preserves
old Path.rglob/is_file behavior, including file symlinks, hidden entries, broken
links and no following directory symlinks. No caching, guard skipping, numerical
or statistical change was introduced. One charged identical-inventory benchmark
showed2.37x census speedup. The next fixed pair qualified throughput178.355143s;
a one-process remaining18 forecast included20% trial margin and90s analysis,
and all remaining trials completed under hard guards. CPU diagnoses17.757186291s
are carried exactly once. Original failed printing was similarly repaired only
in logging, without repeating its completed neural work.

Final receipts report6411.902893s overall including1.145950s separate receipt
audit; follow-up4286.824123s;257236T/39858S;680backwards each;40physical operations;
327680training presentations;2233840fresh evaluation positions. Zero legacy
unique-token/operation counters mean unavailable, not zero work. All failed
attempts remain part of cost/history. Scientific floating merge bytes are not
native payload qualification, and B* remains66,064,401bytes.

The controller-adjudicated evidence-digested terminal gate is
[`WP04=INCONCLUSIVE`](../provenance/task12-wp04-2026-10-27.json). Fresh choices
flip relative to original, all four functional rules coincide within each point
analysis, and near-best regret intervals remain broad. Percentile-zero contrast
intervals hide both better and worse choices. One-update fitting and unaligned
raw-SVD limitations remain. No active superiority, precise equivalence, general
reuse-negative or capability/native claim follows. Next allowed action is Task25
honest reporting after Task12 independent review. Tasks13–24 remain unexecuted;
no third expansion, compiler continuation or protected confirmation is authorized.

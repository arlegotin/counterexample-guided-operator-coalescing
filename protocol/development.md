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

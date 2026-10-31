# Research checkpoint: 31 October 2026

CGOC has demonstrated recoverable FFN sharing in a narrow setting. It has **not demonstrated a useful compression advantage over strong dense quantization, an added benefit from counterexamples, or acceptable capability loss**. This checkpoint closes the current implementation and evidence package; further studies are separate work.

## What the measurements say

A completed 1,024-update diagnostic used pair (0,15), rank 8, seed 17, the same original initialized weights, 15 trainable owners, and 8,388,608 raw target presentations per arm. Those are repeated presentations of 261,632 distinct fit targets. Both arms repaired about 62% of the initial teacher disagreement on 128 already exposed tuning components.

| Objective | Initial KL | Final KL |
| --- | ---: | ---: |
| Sum of local teacher-suffix losses | 0.183560931 | 0.069087173 |
| Full-student loss | 0.183560931 | 0.069851784 |

Full minus local KL is **+0.000764611 nats/target**, with a paired component 95% interval of **[+0.000409527, +0.001088255]**. Local fitting finishes slightly better in this run. The shorter three-seed, 256-update comparison favored full-student fitting, so duration and recipe matter. This long result alone does not measure training-seed uncertainty or establish a generally better objective.

The original local pipeline exceeded its memory ceiling during final accounting after saving the complete trajectory. Its saved state and observations were independently admitted; **the pipeline remains failed**. The full-student counterpart completed under a different, explicitly qualified accounting implementation. Their wall times therefore do not isolate objective speed. The loss-scale diagnostic and additional long seeds are not included as completed results here.

The accepted native campaign evaluated twelve complete artifacts on the same 128 components and 65,408 valid targets. Selected rows are below; sizes are decimal bytes, and all twelve rows are retained in the linked evidence.

| Artifact | Complete bytes | KL/target |
| --- | ---: | ---: |
| Dense RTN8 | 155,122,704 | 0.002646381 |
| Dense GPTQ4 | 87,882,436 | 0.411150296 |
| Shared GPTQ4 | 86,427,212 | 0.431101169 |
| Dense GPTQ2, embedding4 | 61,340,018 | 7.422656065 |
| Shared GPTQ2, embedding4 | 60,548,353 | 9.663858076 |

Shared GPTQ4 saves 1,455,224 bytes against dense GPTQ4, with 0.019950873 higher KL—about 4.85% more KL, not a measured capability drop. This is a storage–fidelity tradeoff. The smaller two-bit points fit the named 66,064,401-byte ceiling but do not establish useful retained behavior. Dense RTN8 dominates the floating shared artifact. No application tolerance or production speed claim has been established.

These native observations use one fixed calibration recipe and seed. The shared learned adapter calibrates shared FFNs; attention and embedding use RTN fallback. Dense learned quantization covers more eligible matrices. Recovery and calibration expense differ, and the preparation costs are retained. This is a complete-artifact frontier, not an equal-preparation mechanism comparison.

[Compact summary](../provenance/continuation-checkpoint-2026-10-31/summary.json) · [Exact receipt manifest](../provenance/continuation-checkpoint-2026-10-31/manifest.json) · [Recovery analysis and bootstrap](../provenance/continuation-checkpoint-2026-10-31/recovery-analysis.json) · [Twelve native points](../provenance/continuation-checkpoint-2026-10-31/native-frontier.json)

The package contains original numeric receipt bytes, all twenty recovery observations, twelve native raw-result packets, and original admission records. Their arithmetic was checked again while assembling this checkpoint. It contains no model weights or raw corpus; retraining and inference replay still require the separately identified assets and compatible runtime. Receipt copying is not new scientific replication. The historical reports and continuation-v2 receipts remain unchanged.

## What is implemented

The new dense baseline campaign has a fixed six-recipe grid: GPTQ, AWQ and DWQ at two and four bits, group size 64, four-bit embeddings, and seed 17. It uses 64 fitting components; DWQ additionally uses 16 disjoint fit-internal components and completes 64 optimizer updates. It reuses the twelve admitted reference observations only after checking their exact artifact, runtime, population, raw-result and cost identities.

The implementation now enforces affine-only DWQ trainability, distinct raw/scored/operator counters, exact original-parent data lineage, measured preparation and native-profile restore, source/environment bindings, and complete failure reporting. If the individually feasible jobs together exceed the campaign budget, it writes all eighteen complete-or-failed outcomes and retains the expense, without inventing metrics or truncating schedules. Ancestor costs are streamed, deduplicated and explicitly marked unknown where the wider history cannot be established.

The final metadata preflight caught an inherited lineage-label mismatch. The 80 selected documents were correct; a new isolated manifest corrects their parent identity and uses a new journal/output binding. The earlier rejected input, source qualification and failed packaging attempt are preserved. No correctness check was loosened.

The source has one whole-feature review and one material fix pass. CPU/public qualification and native qualification are separate. See the final qualification receipt and [campaign guide](dense-baseline-campaign.md). **The six new baseline measurements and seventeen native controls have not been executed for this deliverable.**

## Future work

1. Independently analyze the already-running scale and long-seed replications, keeping the original resource failure and all costs visible.
2. After the serial slot and native controls qualify, profile and evaluate the six fixed dense baselines before testing another sharing frontier.
3. Advance composition, initialization or counterexample studies only where the resulting evidence motivates them. Freeze a promising claim and obtain fresh confirmation before making practical or capability claims.

The broader independent-family reader is an unshipped draft in a separate worktree. It is not part of this completed checkpoint. No new experimental sweep is launched by finishing this deliverable. No push, merge or publication is performed.

## Check the included receipt copies offline

This standard-library check verifies the packaged numeric evidence without following any external model or data paths:

```sh
python3.12 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('provenance/continuation-checkpoint-2026-10-31')
manifest = json.loads((root / 'manifest.json').read_text())
for name, record in manifest['copied_receipts'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == record['sha256'], name
for key in ('summary', 'assembly_source'):
    record = manifest[key]
    path = root / Path(record['path']).name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], key
print('Verified 37 copied receipts, summary and assembly source.')
PY
```

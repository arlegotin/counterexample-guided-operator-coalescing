# Public release scope

This repository publishes the original local CGOC diagnostic, a separately authorized bounded continuation, and their software. The original WP04 outcome remains **diagnosed inconclusive**. The continuation demonstrates scoped single-merge recovery but no compelling practical quality–storage win; active-counterexample selection has no demonstrated advantage. Compiler, protected capability evaluation and independent scientific replication remain unexecuted.

The [continuation report](../provenance/continuation-v2/REPORT.md), [raw numeric summary](../provenance/continuation-v2/summary.json), [claim ledger](../provenance/continuation-v2/CLAIMS.json) and [reconstruction instructions](../provenance/continuation-v2/REPRODUCE.md) are the current results. They preserve negative outcomes, a failed conversion attempt, accounting corrections and replay costs. Earlier continuation-v1 blocked and "not published" receipts describe their capture time; the continuation-v2 report records the resolved archive/OS gates. Publication was subsequently requested by the owner.

## Included and excluded material

The Git checkout contains source, tests, protocols, configuration, dependency provenance, numeric metric metadata, generated reports and historical reviews. It includes development history, failed attempts and superseded decisions so readers can inspect how the result was reached. Historical local paths identify assets that were used; those paths are not portable download locations. A receipt's hash can refer to an excluded artifact.

Downloaded corpus text, token arrays, teacher and fitted model weights, optimizer checkpoints, restricted upstream source and credentials are excluded. The native models and fitting states used in the experiments are not downloadable from this repository. The metrics retain sums, denominators and hashed document/component identities, not the source text or tokens.

The [historical release manifest](../artifacts/release_manifest.json) is a focused, hash-bound inventory at audited commit `162579679f8baa47d2cb4445d3d82f396e685baa`. Its source/document hashes describe that revision. The separate [continuation manifest](../provenance/continuation-v2/SHA256.json) binds all 92 continuation evidence files; the publication verifier and CI check it without private inputs. Neither manifest inventories every historical Git object or claims that all scientific artifacts are public.

## Levels of reproduction

1. **Inspect the result.** Read [REPORT.md](../REPORT.md), [results/summary.json](../results/summary.json) and the numeric [Parquet table](../results/regressions.parquet). No installation is required to read the Markdown or JSON.
2. **Verify or regenerate public metadata.** On current source, `python3.12 provenance/verify_public_continuation.py` checks continuation hashes, raw reductions and selections using only the standard library. To regenerate the original report, use the audited commit with Python 3.12 plus `.[test,report]`; CI pins that replay to its historical source. Neither check reruns model experiments.
3. **Reconstruct the scientific runs.** This requires separately acquiring licensed model/data assets, preserving split/exposure lineage, rebuilding the qualified native environment and satisfying the original numerical and resource gates. Selected public receipts describe those runs, but the excluded raw assets and fitted states prevent treating report regeneration as full raw-experiment replay. A clean native rebuild and independent replication were not completed in the study.

The [reproduction guide](reproduction.md) records both scopes. New scientific work requires its own bounded protocol and reconciled exposure lineage; publishing this continuation does not resume the stopped historical study. Models and corpora remain excluded. The new native bundles were measured locally and are described by manifests and numeric receipts, not distributed here.

## Licensing and attribution

Original software is AGPL-3.0-only; original writing and result presentation are CC-BY-4.0. This follows the owner's recent public research repositories, including [FAM-Jet](https://github.com/arlegotin/FAM-Jet) and [adaptive-tree-neurons](https://github.com/arlegotin/adaptive-tree-neurons). [LICENSING.md](../LICENSING.md) defines scope, [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) preserves upstream rights, and [CITATION.cff](../CITATION.cff) supplies a software citation without implying a paper or DOI.

Development, experiments and reporting used AI assistance under Artem Legotin's direction. Publication editing changes presentation and metadata; it does not turn an inconclusive diagnostic into positive compression evidence.

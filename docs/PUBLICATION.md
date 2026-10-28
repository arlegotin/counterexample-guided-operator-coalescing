# Public release scope

This repository publishes the completed local CGOC diagnostic and its original software. The scientific outcome is **diagnosed inconclusive**. Active-counterexample selection has no demonstrated advantage. Later compiler, protected capability evaluation and independent replication stages remain unexecuted.

## Included and excluded material

The Git checkout contains source, tests, protocols, configuration, dependency provenance, numeric metric metadata, generated reports and historical reviews. It includes development history, failed attempts and superseded decisions so readers can inspect how the result was reached. Historical local paths identify assets that were used; those paths are not portable download locations. A receipt's hash can refer to an excluded artifact.

Downloaded corpus text, token arrays, teacher and fitted model weights, optimizer checkpoints, restricted upstream source and credentials are excluded. The native models and fitting states used in the experiments are not downloadable from this repository. The metrics retain sums, denominators and hashed document/component identities, not the source text or tokens.

The [release manifest](../artifacts/release_manifest.json) is a focused, hash-bound report inventory. It is not an inventory of every historical Git object and does not claim that all original scientific artifacts are public. Subsequent review and verification records that hash that manifest stay outside its inventory to avoid a digest cycle.

## Levels of reproduction

1. **Inspect the result.** Read [REPORT.md](../REPORT.md), [results/summary.json](../results/summary.json) and the numeric [Parquet table](../results/regressions.parquet). No installation is required to read the Markdown or JSON.
2. **Regenerate the report.** Python 3.12 plus `.[test,report]` verifies the retained public source hashes and claim pointers and regenerates the report, summary, Parquet and inventory. This needs no model, corpus, network access after installation or Apple hardware. CI exercises this reporting path; it does not rerun model experiments.
3. **Reconstruct the scientific runs.** This requires separately acquiring licensed model/data assets, preserving split/exposure lineage, rebuilding the qualified native environment and satisfying the original numerical and resource gates. Selected public receipts describe those runs, but the excluded raw assets and fitted states prevent treating report regeneration as full raw-experiment replay. A clean native rebuild and independent replication were not completed in the study.

The [reproduction guide](reproduction.md) records the original machine, exact dependency lock, source revisions, qualification gates and limitations. A fresh experiment would require a new protocol; publishing the existing evidence does not resume the stopped study.

## Licensing and attribution

Original software is AGPL-3.0-only; original writing and result presentation are CC-BY-4.0. This follows the owner's recent public research repositories, including [FAM-Jet](https://github.com/arlegotin/FAM-Jet) and [adaptive-tree-neurons](https://github.com/arlegotin/adaptive-tree-neurons). [LICENSING.md](../LICENSING.md) defines scope, [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) preserves upstream rights, and [CITATION.cff](../CITATION.cff) supplies a software citation without implying a paper or DOI.

Development, experiments and reporting used AI assistance under Artem Legotin's direction. Publication editing changes presentation and metadata; it does not turn an inconclusive diagnostic into positive compression evidence.

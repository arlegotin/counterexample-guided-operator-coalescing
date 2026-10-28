# Continuation publication verification

The owner requested publication after the local experiment handoff. The measured evidence remains byte-identical at commit d72159d17aa9ef7794177e48131c937b83ff528c. Publication adds the current README and citation, separates historical pinned report CI from current metadata CI, and supplies a standard-library verifier plus rejection controls.

- [Exact checks, commands and source hashes](verification.json)
- [Read-only publication review](review.md)
- [Scientific continuation report](../continuation-v2/REPORT.md)
- [Public reconstruction boundaries](../../docs/reproduction.md)

The old report, numeric summary, Parquet and release inventory retain their audited bytes. Their source/document hashes describe the audited commit. Models and corpora remain local and excluded. GitHub CI and the published commit are identified by the repository's commit/run records; this prepublication receipt does not claim a push has already happened.

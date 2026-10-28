# Local reconstruction and evidence identities

The real recovery pilot executed commit
`22c1b8fd62e7dff13f7981089161c19f51e5ed99`.
The executed native comparison and historical numeric reduction scripts are
preserved in `e320583404c61532760b61cf76bd98f5307e2773`.
Later review corrections change accounting/error guards, not model outputs.
Final source/evidence commits and the full patch are recorded in the local handoff.

Original source remains at
`/Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing`
on audited commit162579679f8baa47d2cb4445d3d82f396e685baa.
Untouched pinned runtime is `/private/tmp/cgoc-audit-1625796`.
Continuation worktree is `/private/tmp/cgoc-continuation-v1`.
No push, publication, reset or merge was performed.

## Inputs

- Supplied archive:
  `/private/tmp/diagnostics/CGOC_EVIDENCE_1625796_2026-10-28.zip`,
  SHA256 `fdf89785dd4527217b6d9491ec7db4ec48a026f4fded319b3c6d1f38a76609df`.
  Extracted and25entries verified at
  `/private/tmp/cgoc-evidence-1625796-20261028`.
- Teacher asset manifest is9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083;
  raw model.safetensors is80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1.
  They are different identities. AssetManifest verifies the local full set before
  loading; trust_remote_code is false and loaders use local_files_only.
- Data metadata digest isaa1e4e65f4b9b6170d74fea68bb57a11848bafb8767a2aca432b486fa12b2026.
  Exact role/component/window identities are in recovery/population.json. Original
  exposure-history hash is bound in recovery/result.json RunSpec; new exposure
  history is retained privately under artifacts/continuation-v2/recovery.
- Historical donor input templates and original/fresh trial numeric records are
  private local recovered inputs. Their hashes are recorded. Missing private
  inputs cannot be invented from this metadata or treated as unused protected data.

## Commands executed

All source-dependent commands use explicit PYTHONPATH and Python3.12.9 at:
`/Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing/.venv/bin/python`.

```sh
CGOC_PY=/Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing/.venv/bin/python
PYTHONPATH=. "$CGOC_PY" provenance/recovery_pilot.py --config configs/recovery-pilot-v2.yaml
PYTHONPATH=. "$CGOC_PY" provenance/recovered_score_audit.py --asset-root /Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing --out artifacts/continuation-v2/recovered-scores.json
PYTHONPATH=. "$CGOC_PY" provenance/native_frontier_check.py
PYTHONPATH=. "$CGOC_PY" provenance/legacy_shared_frontier.py
PYTHONPATH=. "$CGOC_PY" provenance/legacy_shared_frontier.py --out artifacts/continuation-v2/legacy-frontier-retry1
PYTHONPATH=. "$CGOC_PY" provenance/legacy_shared_frontier.py --construction-only --out artifacts/continuation-v2/legacy-construction-replay
PYTHONPATH=. "$CGOC_PY" provenance/summarize_recovery_v2.py
PYTHONPATH=. "$CGOC_PY" -m pytest -q
PYTHONPATH=. "$CGOC_PY" -m pytest -o addopts='' -m 'mps or mlx' -q
```

The first legacy command failed a metadata path lookup before model reconstruction/predictions;
its failed receipt, exact source and spending remain preserved. The corrected
retry completed. The construction replay is byte-identical and adds resource
measurements. Native commands require host Metal access; sandbox CPU fallback
is never accepted. Fresh output paths are required; existing outputs are preserved.

The artifact worker commands, exact manifest/input hashes and runtime paths are
stored in frontier/result.json and legacy-frontier-retry1/result.json. Each worker
selects the artifact's own runtime zip before importing poc, preserving old v2
replay and new v3 execution. Private input token windows are not copied here.

## Retained states and artifacts

The ignored local tree `artifacts/continuation-v2/recovery` contains full rolling
model/optimizer/RNG checkpoints for A/B/C, the profile checkpoint, compact deltas
at0/1/4/8/16, complete target plans and exposure history. Their manifest references
and digests are in the safe result files. `frontier/shared-B-floating` is the
complete new trained native bundle. `legacy-frontier-retry1/export` contains the
new K15 floating/RTN bundles under pinned v2 runtime; construction replay contains
byte-identical copies. The reused dense bundles remain in the original ignored
artifacts/quantization tree. See artifacts-index.json for exact references.

These are legitimate local reconstruction/retrieval instructions. A public clone
alone does not contain private fitted weights, full source text or old exposure
history. Do not claim that public metadata reconstructs them. Weights, corpus
text, input token arrays and exposure fingerprints are excluded from this package.

Authoritative parent/child cost ledgers are referenced by path/hash/size under
ledgers/cost-ledger-references.json. Full operation dependency graph and immutable
RunSpecs are included; no legacy zero counter is interpreted as zero work.

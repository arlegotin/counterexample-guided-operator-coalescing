# Whole-feature release review: CGOC v0.2.0

Reviewed base `9d5d156242ae1d89d20ff61c47357393fd49c440` through candidate `53e411d5f8fe86f0347f064e3d14c3b8c23b615c` in the read-only ab01 checkout, including the supplied review package, binding spec, implementation plan, ledger and actual candidate assets. This is the one fresh whole-feature review, not a re-review. No checkout/index/HEAD edits, publication, network access, model/native execution or protected/raw-data reads were performed. Tests used an external temporary directory; the checkout remained clean.

## Strengths

- README and CFF consistently distinguish teacher-distribution KL recovery from practical usefulness. The one-pair, rank-8, seed-17, exposed-development scope and original local pipeline memory failure remain visible. The 1,024-update recovery states are explicitly separate from the earlier 256-update native comparison. No accuracy, protected capability, counterexample benefit, production speed or dense-quantization win is claimed.
- Original scientific modules and historical receipts are unchanged in the release diff. The campaign guide correctly labels the retained implementation/source qualification and leaves its native studies pending.
- The default verifier has no third-party dependency or author-filesystem requirement. Its fixed 42-file inventory, explicit hashes, identity/denominator/reduction checks and negative controls work. Optional NumPy replay independently regenerates all 10,000 recovery draws and the native paired interval.
- I reran all 103 release/public/report tests: 103 passed in 3.26 seconds. Both README public commands passed; checkpoint bootstrap replay also passed. The tests include execution of the evidence ZIP after extraction with isolated Python.
- All candidate SHA256SUMS entries match their bytes. The decompressed source tar is byte-for-byte equal to `git archive` of the reviewed candidate commit. The evidence ZIP is byte-for-byte equal to deterministic construction from that checkout. The source audit reports 2,005 regular files including three retained nested source archives and 41,394,967 uncompressed bytes.
- All three retained executed-source archives contain the exact repository MLX-LM MIT notice. Original software/writing license labels are consistent across README, CFF, pyproject and LICENSING; the release adds explicit retained-source attribution. The pinned account/README-skill reference record is present.
- I independently checked normalized member names in the actual source archive, all three nested archives and evidence ZIP: none has an extraction-path alias. No forbidden payload was detected by the release audit.

## Critical

None found.

## Important

None found in the actual release candidate.

## Minor

1. **Archive auditor accepts duplicate extraction destinations.** `provenance/build_release.py:34` examines `PurePosixPath.parts`, which has already removed `.` and repeated separators; `provenance/build_release.py:49` detects duplicates only by literal name. Two members named `cgoc-v0.2.0/poc/owned.py` and `cgoc-v0.2.0/poc/./owned.py` therefore pass the audit even though extraction overwrites the same destination. This makes the auditor's alias protection incomplete for future candidate inputs. The actual reviewed archives contain no such aliases, so this does not block these exact asset bytes.

   Concrete reproducer, from the checkout (in-memory only):

   ```python
   import importlib.util, io, tarfile
   spec = importlib.util.spec_from_file_location('builder', 'provenance/build_release.py')
   m = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(m)
   out = io.BytesIO()
   with tarfile.open(fileobj=out, mode='w') as t:
       for name, body in [
           ('cgoc-v0.2.0/poc/owned.py', b'first'),
           ('cgoc-v0.2.0/poc/./owned.py', b'replacement'),
       ]:
           member = tarfile.TarInfo(name)
           member.size = len(body)
           t.addfile(member, io.BytesIO(body))
   print(m.audit_tar(out.getvalue(), prefix='cgoc-v0.2.0/'))
   ```

   Actual result: `{'regular_files': 2, 'uncompressed_bytes': 16, 'nested_source_archives': 0, 'forbidden_payloads': 0}`. Reject noncanonical raw path components or compare canonical extraction destinations (including directory/file collisions), with negative controls for `./` and repeated separators.

## Declined-to-judge behaviors

- Remote main/tag/release metadata, CI completion and downloaded published bytes: publication has not occurred in this review and network/publication actions were prohibited; the root must perform the already planned final remote checks.
- Training, native model execution, capability preservation and protected confirmation: outside the release scope and not rerun; the release correctly makes no new execution claim.
- Author-specific absolute paths in historical receipts: retained as provenance, not treated as portable runtime inputs; the portable verifier's tests and standalone ZIP execution pass.
- Independent authenticity of historical raw observations: this review verifies the committed receipts and arithmetic, not an independently rerun scientific experiment.
- Fresh account-repository or upstream-license retrieval: no network access was allowed; reviewed the pinned reference record and bundled license consistency/notice presence instead.
- General legal sufficiency or ownership of upstream material: not a legal determination; reviewed the requested attribution and license consistency boundaries only.
- Exhaustive detection of arbitrary disguised secrets or corpus bytes: a pattern/name scanner is not such a proof; the actual archives pass the implemented audit and observed inventory checks.
- Historical source-bound qualification after release-only edits: retained qualification is explicitly scoped to its original implementation snapshot; no new native qualification was inferred.

## Readiness verdict

**Ready to publish the exact reviewed candidate assets, subject to the planned final push/CI/tag/release-download checks.** No Critical or Important release defect was found. The one Minor auditor alias gap is worth fixing in the allowed root fix pass; if fixed, rebuild and validate assets from the resulting final commit. Do not publish candidate assets while tagging a different commit. No further whole-feature re-review is requested.

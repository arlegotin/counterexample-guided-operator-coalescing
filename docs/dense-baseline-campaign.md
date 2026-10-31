# Dense baseline campaign

This is an explicitly versioned research driver for six dense GPTQ/AWQ/DWQ recipes. Its implementation and CPU checks are complete for this checkpoint. Native qualification and the six resulting scientific measurements are future execution gates.

Use Python 3.12 with the repository's test/science dependencies. Actual Apple execution additionally requires the qualified MLX/MLX-LM environment and pinned local assets. The supplied proposal records paths in the authorized research workspace; those files are not bundled here. A different workspace needs a new source-bound proposal and input preparation, not edits to frozen receipts.

## Check metadata

From the checkout:

```sh
python3.12 provenance/dense_frontier_campaign.py \
  --config configs/dense-baseline-expansion-v2-proposal-v1.yaml \
  --mode validate
```

This verifies source, reference and population metadata without loading a model, importing Metal, or separately reading text/token files. It checks the actual original 512-component parent and the exact first 80 selected components. Validation does not qualify a backend or grant protected-data access.

The public historical evidence check needs only the standard library:

```sh
python3.12 provenance/verify_public_continuation.py
```

Focused CPU controls:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3.12 -m pytest tests/test_dense_frontier.py \
  tests/test_dense_dispatch_accounting.py tests/test_dense_campaign.py \
  tests/test_dense_admission.py tests/test_dense_cost_ancestry.py -q
```

The normal pytest selection excludes MPS/MLX. CPU success cannot qualify native arithmetic, runtime or performance.

## Future native execution

The separate `controls`, `profile`, `run` and `analyze` modes enforce additional source, finite-budget, serial-slot and backend admissions. They are not shortcuts around those gates. No such new native campaign is started for this checkpoint.

The bounded controller is `provenance/dense-baseline-expansion-v2/execute-campaign-v1.py.txt`. It requires the exact completed upstream recovery family before obtaining the accelerator slot. It then runs seventeen native controls, completed short preparation/resource profiles, and freezes the full budgets before any new development quality is used. Failed profiles or recipes remain visible. An actual measured run configuration is generated; `run` rejects an unfrozen proposal. Reusing an attempted directory or changing the frozen plan is rejected.

DWQ's last distribution target is charged even when the official model sees a truncated 511-token input. Its two-depth/source units and raw corpus presentations remain separate. Trainable leaves are restricted to the declared quantizer scales and biases; floating normalization owners stay frozen. Shared aliases, BF16 stored corrections, complete bytes, offline runtime and independent reload remain part of artifact validation.

The common report stays provisional until a separate saved-result consumer admits its raw observations, artifacts and costs. Complete bytes, fidelity, capability and production runtime are distinct outcomes. No practical acceptability threshold has been supplied.

## Evidence and qualification

- [Resolved one-pass review](../provenance/dense-baseline-expansion-v2/review-resolution-v2.json)
- [Actual corrected-input metadata preflight](../provenance/dense-baseline-expansion-v2/accepted-metadata-preflight-v2.json)
- [Source/CPU/public qualification](../provenance/dense-baseline-expansion-v2/qualification-v1.json)
- [Preserved prior metadata rejection](../provenance/dense-baseline-expansion-v2/preserved-validation-failure-v1.json)
- [Input correction and unchanged population](../provenance/dense-baseline-expansion-v2/lineage-correction-v2.json)
- [Owned-fixture packaging failure and repair](../provenance/dense-baseline-expansion-v2/owned-fixture-closure-v2.json)

No qualification receipt should be inferred merely from a file name. Each consumer checks its source archive, exact commands, successful finite guards, environment and referenced bytes.

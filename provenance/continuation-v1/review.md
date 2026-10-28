# WP1 review record

An independent read-only reviewer examined the new continuation source against
162579679f8baa47d2cb4445d3d82f396e685baa. Review used owned tiny fixtures only;
no network, scientific assets or corpus readers. No Critical findings.

All seven Important findings were reproduced with failing tests and corrected:

| Finding | Correction and regression |
| --- | --- |
| Failed-row digest fields admitted private payloads | Validate every status; `test_failed_rows_cannot_export_private_payloads` |
| Damage populations could differ | Match windows/counts and reject duplicates; `test_damage_windows_and_denominators_must_be_common` |
| Final accounting overrun recorded as success | Record failed status and deny exhausted reuse; `test_accounting_overrun_is_failed_and_not_reusable` |
| Declared depth coverage could differ from builders | Validate record/boundary depth in wrapped full builder; `test_planned_depth_is_checked_against_actual_full_record_before_backward` |
| Checkpoint acceptance lacked profile rejection | Check loader and reconstruction; `test_checkpoint_loader_rejects_old_profile_with_matching_expected_spec` |
| Reload verification omitted two forwards | Charged helper; `test_reload_verification_charges_both_forwards` |
| Transient memory was not observed | Kernel CPU RSS high-water sampling; `test_live_peak_observation_enforces_cap_and_retains_overrun` |

The inventory-versus-presented coverage observation was corrected because the
user explicitly requires actual unique coverage. No minor findings remain deferred.
Additional author checks require the random donor to belong to the common active
pool, preserve local owner trainability across deltas, and hash template inventory
keys. `review-red-complete.log` preserves the failing regressions; `review-green.log`
records 31 focused passes. Later full command receipts are authoritative for the
final source. No second independent review or replication is claimed.

Execution failures retained: the first owned fixture exceeded its tiny decoder's
128-position limit with 129 input positions; the owned config now explicitly has
256 positions. The next run exposed lost local owner trainability after compact
delta restoration. Dependent owned checks were rerun after both corrections;
there were no dependent scientific comparisons.

Engineering rulings: use /private/tmp worktrees within writable roots; preserve
the user's existing environment and select each checkout through PYTHONPATH;
use the provided detailed WP1 design directly; leave scientific pilot execution
blocked pending a reviewed WP2 adapter rather than accepting hash references as
semantic qualification. Costs if wrong: worktree discoverability, import binding,
or an unqualified pilot. Source/command receipts expose the first two; the last
remains an explicit hard block.

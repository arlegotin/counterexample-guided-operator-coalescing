from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from poc.budget import BudgetExceeded, BudgetTracker
from poc.contracts import Budget, CostRecord, RunSpec
from poc.records import RecordError, append_event, canonical_json, run_id, start_attempt


def budget():
    return Budget(updates=10, valid_target_presentations=100, teacher_forward_calls=3,
                  suffix_backward_calls=3, student_backward_calls=10, search_proposals=3,
                  search_seconds=30, wall_seconds=60, disk_bytes=10000, memory_bytes=10000)


def spec(**changes):
    values = dict(method_id="M", method_variant="pilot", resolved_config={"b": 2, "a": 1},
                  source_commit="a" * 40, dirty_patch_sha256=None,
                  dependency_lock_sha256="b" * 64, input_manifests={"teacher": "c" * 64},
                  hardware_policy={"backend": "cpu", "dtype": "float32"},
                  seeds={"search": 17}, parent_run_ids=(), budgets=budget())
    values.update(changes)
    return RunSpec(**values)


def test_canonical_bytes_and_content_identity():
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    assert run_id(spec()) == run_id(spec(resolved_config={"a": 1, "b": 2}))
    assert run_id(spec()) != run_id(spec(seeds={"search": 18}))
    assert len(run_id(spec())) == 64
    assert run_id(spec()) == hashlib.sha256(canonical_json(spec().to_dict())).hexdigest()


@pytest.mark.parametrize("value", [{"x": float("nan")}, {"x": float("inf")}, {1: "bad"}])
def test_noncanonical_values_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        canonical_json(value)


def test_identity_snapshots_nested_inputs_and_is_immutable():
    source = {"list": [{"x": 1}]}
    identity = spec(resolved_config=source)
    before = run_id(identity)
    source["list"][0]["x"] = 2
    assert run_id(identity) == before
    with pytest.raises((TypeError, FrozenInstanceError)):
        identity.seeds["search"] = 42
    with pytest.raises((TypeError, FrozenInstanceError)):
        identity.resolved_config["list"][0]["x"] = 42


def test_retry_has_new_directory_and_keeps_previous_attempt(tmp_path):
    first = start_attempt(spec(), tmp_path)
    append_event(first, {"kind": "failure", "reason": "interrupted"})
    saved = (first.path / "events.jsonl").read_bytes()
    second = start_attempt(spec(), tmp_path)
    assert first.attempt_id != second.attempt_id
    assert first.run_id == second.run_id
    assert first.path != second.path
    assert first.path == tmp_path / first.run_id / "attempts" / first.attempt_id
    assert (first.path / "events.jsonl").read_bytes() == saved
    record = json.loads((second.path / "run.json").read_text())
    assert record["status"] == "planned"
    assert record["run_id"] == first.run_id


def test_events_are_monotonic_and_snapshot_payload(tmp_path):
    attempt = start_attempt(spec(), tmp_path)
    payload = {"kind": "trial", "metrics": {"loss": 0.5}}
    append_event(attempt, payload)
    payload["metrics"]["loss"] = 99
    append_event(attempt, {"kind": "end"})
    rows = [json.loads(line) for line in (attempt.path / "events.jsonl").read_text().splitlines()]
    assert [row["event_id"] for row in rows] == [1, 2]
    assert rows[0]["metrics"]["loss"] == 0.5
    assert rows[0]["timestamp"]
    with pytest.raises(RecordError):
        append_event(attempt, {"event_id": 100, "kind": "bad"})


def test_truncated_ledger_is_retained_and_blocks_append(tmp_path):
    attempt = start_attempt(spec(), tmp_path)
    ledger = attempt.path / "events.jsonl"
    ledger.write_bytes(b'{"event_id":1')
    with pytest.raises(RecordError, match="truncated"):
        append_event(attempt, {"kind": "end"})
    assert ledger.read_bytes() == b'{"event_id":1'


def test_teacher_cost_survives_retry_and_fresh_tracker(tmp_path):
    identity = spec()
    first = start_attempt(identity, tmp_path)
    tracker = BudgetTracker(identity.budgets, first)
    tracker.check(CostRecord(teacher_forward_calls=2, valid_target_presentations=40))
    tracker.record(CostRecord(teacher_forward_calls=2, valid_target_presentations=40))
    second = start_attempt(identity, tmp_path)
    retry = BudgetTracker(identity.budgets, second)
    with pytest.raises(BudgetExceeded, match="teacher_forward_calls"):
        retry.check(CostRecord(teacher_forward_calls=2))
    retry.check(CostRecord(teacher_forward_calls=1))
    retry.record(CostRecord(teacher_forward_calls=1))
    with pytest.raises(BudgetExceeded):
        tracker.check(CostRecord(teacher_forward_calls=1))
    assert retry.totals.teacher_forward_calls == 3
    assert retry.totals.valid_target_presentations == 40


def test_actual_atomic_overrun_is_recorded_not_deleted(tmp_path):
    identity = spec()
    tracker = BudgetTracker(identity.budgets, start_attempt(identity, tmp_path))
    with pytest.raises(BudgetExceeded):
        tracker.record(CostRecord(wall_seconds=61))
    assert tracker.totals.wall_seconds == 61


def test_memory_and_disk_are_high_water_counters(tmp_path):
    identity = spec()
    tracker = BudgetTracker(identity.budgets, start_attempt(identity, tmp_path))
    tracker.record(CostRecord(memory_bytes=6000, disk_bytes=6000))
    tracker.check(CostRecord(memory_bytes=6000, disk_bytes=6000))
    with pytest.raises(BudgetExceeded):
        tracker.check(CostRecord(memory_bytes=10001))


@pytest.mark.parametrize("field,value", [("wall_seconds", float("inf")), ("updates", -1),
                                         ("teacher_forward_calls", True), ("unique_tokens", 1.5)])
def test_invalid_costs_fail(field, value):
    with pytest.raises(ValueError):
        CostRecord(**{field: value})


def test_estimated_work_and_missing_memory_need_reasons():
    with pytest.raises(ValueError):
        CostRecord(estimated_operations=2)
    with pytest.raises(ValueError):
        CostRecord(memory_bytes=None, memory_unavailable_reason=None)


def test_budget_must_be_finite_and_cannot_change_on_retry(tmp_path):
    with pytest.raises(ValueError):
        replace(budget(), wall_seconds=float("inf"))
    attempt = start_attempt(spec(), tmp_path)
    with pytest.raises(ValueError):
        BudgetTracker(replace(budget(), teacher_forward_calls=100), attempt)


@pytest.mark.parametrize("field", list(budget().to_dict()))
def test_each_budget_ceiling_is_enforced(tmp_path, field):
    identity = spec()
    tracker = BudgetTracker(identity.budgets, start_attempt(identity, tmp_path))
    with pytest.raises(BudgetExceeded, match=field):
        tracker.check(CostRecord(**{field: getattr(identity.budgets, field) + 1}))


def test_cost_ledger_corruption_blocks_retry_without_reset(tmp_path):
    identity = spec()
    attempt = start_attempt(identity, tmp_path)
    tracker = BudgetTracker(identity.budgets, attempt)
    tracker.record(CostRecord(teacher_forward_calls=2))
    saved = tracker.path.read_bytes()
    with tracker.path.open("ab") as stream:
        stream.write(b'{"event_id":2')
    retry = BudgetTracker(identity.budgets, start_attempt(identity, tmp_path))
    with pytest.raises(RecordError, match="truncated"):
        retry.check(CostRecord(teacher_forward_calls=1))
    assert tracker.path.read_bytes() == saved + b'{"event_id":2'


def test_exhausted_check_has_no_spending_side_effect(tmp_path):
    identity = spec()
    tracker = BudgetTracker(identity.budgets, start_attempt(identity, tmp_path))
    with pytest.raises(BudgetExceeded):
        tracker.check(CostRecord(teacher_forward_calls=4))
    assert tracker.totals.teacher_forward_calls == 0


def test_nonmonotonic_events_block_append(tmp_path):
    attempt = start_attempt(spec(), tmp_path)
    ledger = attempt.path / "events.jsonl"
    ledger.write_bytes(b'{"event_id":2}\n')
    with pytest.raises(RecordError, match="nonmonotonic"):
        append_event(attempt, {"kind": "end"})


def test_identity_manifest_tampering_prevents_retry(tmp_path):
    identity = spec()
    attempt = start_attempt(identity, tmp_path)
    (attempt.path.parent.parent / "spec.json").write_bytes(b'{}\n')
    with pytest.raises(RecordError, match="mismatch"):
        start_attempt(identity, tmp_path)


def test_every_identity_component_changes_run_hash():
    original = spec()
    for changes in [dict(method_variant="other"), dict(source_commit="d" * 40),
                    dict(dirty_patch_sha256="e" * 64), dict(dependency_lock_sha256="f" * 64),
                    dict(input_manifests={"teacher": "e" * 64}), dict(hardware_policy={"backend": "torch_mps"}),
                    dict(parent_run_ids=("d" * 64,)), dict(budgets=replace(budget(), teacher_forward_calls=4))]:
        assert run_id(spec(**changes)) != run_id(original)


def test_student_forward_cost_is_compatible_with_legacy_rows_and_durable(tmp_path):
    identity=spec();attempt=start_attempt(identity,tmp_path)
    tracker=BudgetTracker(identity.budgets,attempt)
    tracker.record(CostRecord(teacher_forward_calls=1))
    # Simulate the genuine prior schema: absence means no typed increment was
    # recorded, not a measurement erasing externally attributed historical cost.
    rows=[json.loads(line) for line in tracker.path.read_text().splitlines()]
    rows[0]['cost'].pop('student_forward_calls')
    tracker.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    tracker.record(CostRecord(student_forward_calls=2))
    fresh=BudgetTracker(identity.budgets,attempt)
    assert fresh.totals.teacher_forward_calls==1 and fresh.totals.student_forward_calls==2
    with pytest.raises(ValueError): CostRecord(student_forward_calls=-1)

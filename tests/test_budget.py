"""Actual retries consume aggregate budget, independently of checkpoint rollback."""
from dataclasses import replace
import pytest
from poc.budget import BudgetTracker, BudgetExceeded
from poc.contracts import CostRecord
from poc.records import start_attempt
from test_training import api,batch,budget,models
from test_records import spec


def test_failed_work_survives_retry_and_stop_propagates(tmp_path,monkeypatch):
    t=api(); limits=budget(valid_target_presentations=8)
    identity=spec(budgets=limits,resolved_config={'training':t.TrainingConfig(max_updates=2).identity()});first=start_attempt(identity,tmp_path)
    tracker=BudgetTracker(limits,first); teacher,student=models()
    original=t.backward_distillation
    def fail(*args,**kwargs):
        original(*args,**kwargs); raise RuntimeError('after real backward')
    monkeypatch.setattr(t,'backward_distillation',fail)
    cfg=t.TrainingConfig(max_updates=2,run_spec=identity)
    with pytest.raises(RuntimeError): t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),cfg,limits,tracker)
    spent=tracker.totals.valid_target_presentations
    assert spent==8
    monkeypatch.setattr(t,'backward_distillation',original)
    retry=BudgetTracker(limits,start_attempt(identity,tmp_path))
    with pytest.raises(BudgetExceeded): t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),cfg,limits,retry)
    assert retry.totals.valid_target_presentations==8
    assert retry.attempt_cost.valid_target_presentations==0


def test_same_run_attempt_wall_topup_does_not_subtract_previous_attempt(tmp_path):
    t=api(); identity=spec(budgets=budget()); a=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path))
    a.record(CostRecord(wall_seconds=20))
    b=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path)); b.record(CostRecord(wall_seconds=2))
    t.finish_attempt_cost(b,elapsed=3)
    assert b.attempt_cost.wall_seconds==3
    assert b.totals.wall_seconds==23


def test_checkpoint_failure_retains_actual_disk_and_cost(tmp_path,monkeypatch):
    from poc.records import start_attempt
    from test_training import MemoryRecorder
    from test_resume import checkpoint
    t=api();c=checkpoint();teacher,student=models();identity=spec(budgets=budget(),resolved_config={'training':t.TrainingConfig(max_updates=1).identity()})
    state=t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),t.TrainingConfig(max_updates=1,run_spec=identity),budget(),MemoryRecorder()).state
    recorder=MemoryRecorder()
    def crash(stage):
        if stage=='after_generation_publish':raise InterruptedError('publish')
    monkeypatch.setattr(c,'_publication_boundary',crash)
    with pytest.raises(InterruptedError):c.save_checkpoint(state,tmp_path,recorder=recorder)
    assert recorder.totals.disk_bytes>0
    assert recorder.totals.wall_seconds>0


def test_nonfinite_gradients_never_step_and_are_cleared(monkeypatch):
    from test_training import MemoryRecorder
    t=api();teacher,student=models();original=t.backward_distillation;recorder=MemoryRecorder()
    def poison(*args,**kwargs):
        result=original(*args,**kwargs)
        next(p for p in student.parameters() if p.requires_grad).grad.fill_(float('nan'))
        return result
    monkeypatch.setattr(t,'backward_distillation',poison)
    with pytest.raises(FloatingPointError,match='gradient'):
        t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),t.TrainingConfig(max_updates=1),budget(),recorder)
    assert recorder.totals.updates==0 and recorder.totals.valid_target_presentations==8
    assert all(p.grad is None for p in student.parameters())


def test_incremental_ledger_totals_still_detect_prefix_edits_and_new_process_appends(tmp_path):
    import json
    from poc.records import RecordError,canonical_json
    identity=spec(budgets=budget());a=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path))
    a.record(CostRecord(wall_seconds=1,student_forward_calls=3));assert a.totals.student_forward_calls==3
    b=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path));b.record(CostRecord(wall_seconds=2,student_forward_calls=5))
    assert a.totals.wall_seconds==3 and a.totals.student_forward_calls==8
    rows=[json.loads(line) for line in a.path.read_text().splitlines()];rows[0]['cost']['student_forward_calls']=0
    a.path.write_bytes(b''.join(canonical_json(r) for r in rows))
    with pytest.raises(RecordError,match='prefix'):_=a.totals


def test_cached_forecasts_metadata_highwaters_and_overrun_are_exact(tmp_path):
    identity=spec(budgets=budget(wall_seconds=5));a=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path))
    costs=[CostRecord(wall_seconds=1,memory_bytes=400,memory_unavailable_reason=None,disk_bytes=10,estimated_operations=3,estimation_method='a'),
        CostRecord(wall_seconds=1,memory_bytes=None,memory_unavailable_reason='no sample',disk_bytes=4,estimated_operations=2,estimation_method='b'),
        CostRecord(wall_seconds=1,memory_bytes=300,memory_unavailable_reason=None,disk_bytes=20)]
    for c in costs:a.record(c);_=a.totals
    assert a.totals==a._total([{'cost':c.to_dict()} for c in costs])
    before=a.path.read_bytes();a.check(CostRecord(wall_seconds=1));assert a.path.read_bytes()==before
    with pytest.raises(BudgetExceeded):a.record(CostRecord(wall_seconds=3))
    assert a.totals.wall_seconds==6 and a.totals.memory_bytes==400 and a.totals.disk_bytes==20
    assert a.totals.estimation_method=='a; b' and a.totals.memory_unavailable_reason=='no sample; not measured'


def test_cached_ledger_accepts_other_process_append_and_reopen(tmp_path):
    import subprocess,sys,json
    identity=spec(budgets=budget());a=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path));a.record(CostRecord(wall_seconds=1))
    assert a.totals.wall_seconds==1
    code='''import sys
from pathlib import Path
from poc.records import locked_ledger,read_rows,write_row,timestamp
from poc.contracts import CostRecord
with locked_ledger(Path(sys.argv[1])) as stream:
    rows=read_rows(stream)
    write_row(stream,dict(event_id=len(rows)+1,timestamp=timestamp(),attempt_id='child',kind='cost',cost=CostRecord(wall_seconds=2).to_dict()))
'''
    subprocess.run([sys.executable,'-c',code,str(a.path)],check=True,capture_output=True,text=True)
    assert a.totals.wall_seconds==3
    reopened=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path));assert reopened.totals==a.totals
    assert reopened.attempt_cost.wall_seconds==0


@pytest.mark.parametrize('tail',[b'{',b'{"event_id":99}\n'])
def test_cached_ledger_rejects_malformed_partial_or_nonmonotonic_tail(tmp_path,tail):
    from poc.records import RecordError
    identity=spec(budgets=budget());a=BudgetTracker(identity.budgets,start_attempt(identity,tmp_path));a.record(CostRecord(wall_seconds=1));_=a.totals
    with a.path.open('ab') as stream:stream.write(tail)
    before=a.path.read_bytes()
    with pytest.raises(RecordError):_=a.totals
    assert a.path.read_bytes()==before

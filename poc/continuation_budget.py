"""Sequential parent/child accounting. Parent spending never rewinds on replay."""
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import resource
import sys
import time
from uuid import uuid4

from poc.budget import BudgetTracker
from poc.contracts import CostRecord
from poc.records import locked_ledger, read_rows, write_row, start_attempt, run_id


def process_peak_bytes():
    """Kernel CPU process-lifetime RSS high-water mark, including transient Adam."""
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


class _ObservedParent(BudgetTracker):
    def __init__(self,budget,attempt,root):
        super().__init__(budget,attempt)
        self.output_root=root

    def record(self,cost):
        disk=sum(p.stat().st_size for p in self.output_root.rglob('*') if p.is_file())
        super().record(replace(cost,memory_bytes=max(cost.memory_bytes or 0,process_peak_bytes()),
                               disk_bytes=max(cost.disk_bytes,disk)))


class _ChildTracker(BudgetTracker):
    def __init__(self, parent, spec, attempt):
        super().__init__(spec.budgets,attempt)
        self.parent = parent

    def check(self, cost):
        self.parent.check(cost)
        super().check(cost)

    def record(self, cost):
        # Parent first: a crash between appends cannot erase aggregate spending.
        try:self.parent.record(cost)
        finally:super().record(cost)


class PilotLedger:
    def __init__(self, spec, root, *, observe_resources=False):
        self.spec = spec
        self.attempt = start_attempt(spec,root)
        self.tracker = (_ObservedParent(spec.budgets,self.attempt,root) if observe_resources
                        else BudgetTracker(spec.budgets,self.attempt))
        self.root = self.attempt.path.parent.parent
        self.graph_path = self.root/'pilot-graph.jsonl'

    def graph(self):
        with locked_ledger(self.graph_path) as stream:return read_rows(stream)

    def _append(self, **row):
        with locked_ledger(self.graph_path) as stream:
            rows=read_rows(stream)
            write_row(stream,{'event_id':len(rows)+1,**row})

    @contextmanager
    def _serial(self):
        with (self.root/'pilot-operation.lock').open('a') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise ValueError('concurrent pilot operation denied') from exc
            try:
                rows=self.graph()
                begun={r['operation_id'] for r in rows if r['kind']=='begin'}
                finished={r['operation_id'] for r in rows if r['kind']=='finish'}
                if begun-finished:raise ValueError('unfinished operation; reconcile spending before replay')
                yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)

    def _dependencies(self, dependencies):
        complete={r['key'] for r in self.graph() if r['kind']=='finish' and r['status']=='complete'}
        if not set(dependencies)<=complete:raise ValueError('missing successful cost dependency')

    def shared(self, key, cost, operation, *, dependencies=()):
        """Dispatch fixed-count shared work once; reuse records dependency, not free work."""
        with self._serial():
            self._dependencies(dependencies)
            previous=[r for r in self.graph() if r['kind']=='finish' and r['key']==key and r['status']=='complete']
            if previous:
                self.tracker.check(CostRecord())
                self._append(kind='reuse',key=key,dependency_operation_id=previous[-1]['operation_id'])
                return None
            self.tracker.check(cost)
            op=uuid4().hex
            self._append(kind='begin',operation_id=op,key=key,dependencies=list(dependencies),child_run_id=None)
            started=time.monotonic();status='failed'
            try:
                self.tracker.record(replace(cost,wall_seconds=0))
                result=operation();status='complete';return result
            finally:
                try:self.tracker.record(CostRecord(wall_seconds=time.monotonic()-started))
                except BaseException:
                    status='failed'
                    raise
                finally:self._append(kind='finish',operation_id=op,key=key,status=status)

    @contextmanager
    def arm(self, spec, *, dependencies=(), forecast=CostRecord()):
        with self._serial():
            self._dependencies(dependencies)
            self.tracker.check(forecast)
            child_attempt=start_attempt(spec,self.root/'children')
            recorder=_ChildTracker(self.tracker,spec,child_attempt)
            recorder.check(forecast)
            op=uuid4().hex;key=run_id(spec)
            self._append(kind='begin',operation_id=op,key=key,dependencies=list(dependencies),
                         child_run_id=key,child_attempt_id=child_attempt.attempt_id)
            started=time.monotonic();status='failed'
            try:
                yield recorder
                status='complete'
            finally:
                try:
                    elapsed=time.monotonic()-started
                    extra=max(0,elapsed-recorder.attempt_cost.wall_seconds)
                    if extra:recorder.record(CostRecord(wall_seconds=extra))
                except BaseException:
                    status='failed'
                    raise
                finally:self._append(kind='finish',operation_id=op,key=key,status=status)

"""Research expenditure survives failed attempts and process restarts."""
from dataclasses import fields
import hashlib
import io

from poc.contracts import Budget, CostRecord
from poc.records import AttemptRecord, RecordError, locked_ledger, read_rows, timestamp, write_row


class BudgetExceeded(ValueError):
    """A finite ceiling would be exceeded (or actual work already exceeded it)."""


class BudgetTracker:
    """Preflight operations with check; persist actual work with record.

    One active costly job is the harness contract. check is not a reservation.
    record always retains actual spending, including an atomic-operation overrun,
    before raising BudgetExceeded. The authoritative ledger belongs to the run,
    not an attempt, and is reread under a process lock for every operation.
    """
    def __init__(self, budget: Budget, attempt: AttemptRecord):
        if budget != attempt.spec.budgets:
            raise ValueError("attempt budget differs from immutable RunSpec")
        self.budget = budget
        self.attempt = attempt
        self.path = attempt.path.parent.parent / "costs.jsonl"
        self._cached_bytes = 0
        self._cached_hash = hashlib.sha256(b'').digest()
        self._cached_count = 0
        self._cached_total = CostRecord()

    def _total(self, rows) -> CostRecord:
        values = CostRecord().to_dict()
        methods = set()
        missing_reasons = set()
        for row in rows:
            try:
                cost = CostRecord(**row["cost"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RecordError("invalid cost ledger entry") from exc
            for field in fields(cost):
                name, value = field.name, getattr(cost, field.name)
                if name == "estimation_method":
                    if value:
                        methods.add(value)
                elif name == "memory_unavailable_reason":
                    if cost.memory_bytes is None:
                        missing_reasons.add(value)
                elif name in {"memory_bytes", "disk_bytes"}:
                    if value is not None:
                        values[name] = max(values[name] or 0, value)
                else:
                    values[name] += value
        values["estimation_method"] = "; ".join(sorted(methods)) or None
        values["memory_unavailable_reason"] = "; ".join(sorted(missing_reasons)) or ("not measured" if not rows else None)
        return CostRecord(**values)

    @staticmethod
    def combine(first: CostRecord, second: CostRecord) -> CostRecord:
        """Combine already validated totals without revisiting historical rows."""
        values=first.to_dict()
        for field in fields(second):
            name,value=field.name,getattr(second,field.name)
            if name in ('estimation_method','memory_unavailable_reason'):
                parts=set()
                for text in (values[name],value):
                    if text:parts.update(text.split('; '))
                values[name]='; '.join(sorted(parts)) or None
            elif name in ('memory_bytes','disk_bytes'):
                if value is not None:values[name]=max(values[name] or 0,value)
            else:values[name]+=value
        return CostRecord(**values)

    def _read_total(self,stream):
        """Verify the exact cached prefix, parse only appended records under lock.

        The cache is an optimization, never the spending authority. Every access
        hashes the original ledger bytes; truncation/rewrites stop the run.
        Another process's append is validated with the next monotonic event ID.
        """
        data=stream.read()
        if len(data)<self._cached_bytes or hashlib.sha256(data[:self._cached_bytes]).digest()!=self._cached_hash:
            raise RecordError('cost ledger prefix changed; retained for diagnosis')
        rows=read_rows(io.BytesIO(data[self._cached_bytes:]),first_event_id=self._cached_count+1)
        total=self._cached_total
        for i,row in enumerate(rows):
            try:cost=CostRecord(**row['cost'])
            except (KeyError,TypeError,ValueError) as exc:raise RecordError('invalid cost ledger entry') from exc
            total=cost if self._cached_count+i==0 else self.combine(total,cost)
        self._cached_total=total
        self._cached_count+=len(rows)
        self._cached_bytes=len(data)
        self._cached_hash=hashlib.sha256(data).digest()
        return total

    @property
    def totals(self) -> CostRecord:
        with locked_ledger(self.path) as stream:
            return self._read_total(stream)

    @property
    def attempt_cost(self) -> CostRecord:
        """Current-attempt increments; aggregate run spending remains in totals."""
        with locked_ledger(self.path) as stream:
            rows = [row for row in read_rows(stream) if row['attempt_id'] == self.attempt.attempt_id]
            return self._total(rows)

    def _check_total(self, total: CostRecord) -> None:
        for field in fields(self.budget):
            measured = getattr(total, field.name)
            if measured is not None and measured > getattr(self.budget, field.name):
                raise BudgetExceeded(f"budget exhausted: {field.name} ({measured} > {getattr(self.budget, field.name)})")

    def check(self, next_cost: CostRecord) -> None:
        if not isinstance(next_cost, CostRecord):
            raise ValueError("next_cost must be a CostRecord")
        with locked_ledger(self.path) as stream:
            self._check_total(self.combine(self._read_total(stream),next_cost))

    def record(self, actual_cost: CostRecord) -> None:
        if not isinstance(actual_cost, CostRecord):
            raise ValueError("actual_cost must be a CostRecord")
        with locked_ledger(self.path) as stream:
            prior = self._read_total(stream)
            row = {"event_id": self._cached_count + 1, "timestamp": timestamp(), "attempt_id": self.attempt.attempt_id,
                   "kind": "cost", "cost": actual_cost.to_dict()}
            total = self.combine(prior,actual_cost)
            try:
                self._check_total(total)
            except BudgetExceeded as exc:
                row["overrun_reason"] = str(exc)
                write_row(stream, row)
                raise
            write_row(stream, row)

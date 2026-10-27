"""Research expenditure survives failed attempts and process restarts."""
from dataclasses import fields

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

    @property
    def totals(self) -> CostRecord:
        with locked_ledger(self.path) as stream:
            return self._total(read_rows(stream))

    def _check_total(self, total: CostRecord) -> None:
        for field in fields(self.budget):
            measured = getattr(total, field.name)
            if measured is not None and measured > getattr(self.budget, field.name):
                raise BudgetExceeded(f"budget exhausted: {field.name} ({measured} > {getattr(self.budget, field.name)})")

    def check(self, next_cost: CostRecord) -> None:
        if not isinstance(next_cost, CostRecord):
            raise ValueError("next_cost must be a CostRecord")
        with locked_ledger(self.path) as stream:
            rows = read_rows(stream)
            self._check_total(self._total([*rows, {"cost": next_cost.to_dict()}]))

    def record(self, actual_cost: CostRecord) -> None:
        if not isinstance(actual_cost, CostRecord):
            raise ValueError("actual_cost must be a CostRecord")
        with locked_ledger(self.path) as stream:
            rows = read_rows(stream)
            row = {"event_id": len(rows) + 1, "timestamp": timestamp(), "attempt_id": self.attempt.attempt_id,
                   "kind": "cost", "cost": actual_cost.to_dict()}
            total = self._total([*rows, row])
            try:
                self._check_total(total)
            except BudgetExceeded as exc:
                row["overrun_reason"] = str(exc)
                write_row(stream, row)
                raise
            write_row(stream, row)

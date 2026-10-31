"""Explicit resource-only child studies; ordinary resume identity remains strict.

Call only after the ancestor worker has stopped. A fork preserves its immutable
checkpoint and ledger, carries all spending (including work after the checkpoint),
and records a new identity. It is an exploratory protocol amendment, never a way
to relabel the ancestor as within budget or change its scientific schedule.
"""
from dataclasses import asdict,dataclass,replace
import hashlib
import io
import json
from pathlib import Path
import re

from poc.budget import BudgetTracker
from poc.checkpoint import load_checkpoint
from poc.contracts import CostRecord,freeze,thaw
from poc.records import canonical_json,locked_ledger,read_rows,run_id,append_event


def _sha(data):return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RecoveryResourceFork:
    parent_checkpoint_sha256: str
    parent_run_id: str
    parent_ledger_sha256: str
    parent_source_sha256: str
    child_source_sha256: str
    qualification_sha256: str
    profile_sha256: str
    new_study_contract_sha256: str
    schema_version: int = 1
    reason: str = 'wall_budget_completion'

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1 or self.reason!='wall_budget_completion':
            raise ValueError('unsupported resource-only fork contract')
        if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v)
               for k,v in asdict(self).items() if k not in ('schema_version','reason')):
            raise ValueError('complete checkpoint/run/ledger/source/qualification/profile/study hashes required')

    def identity(self):return asdict(self)


@dataclass(frozen=True)
class RecoveryForkState:
    state: object
    opening_balance: CostRecord
    record: object


def _receipt(path,expected):
    data=Path(path).read_bytes()
    if _sha(data)!=expected:raise ValueError('fork receipt content identity mismatch')
    value=json.loads(data)
    if not isinstance(value,dict) or value.get('passed') is not True:
        raise ValueError('fork requires a passed qualification/resource profile')
    return value


def fork_recovery_checkpoint(ref,ancestor_spec,child_spec,contract,*,ledger_path,qualification_path,profile_path):
    """Verify an explicit child protocol and load the ancestor safe checkpoint.

    Only the cumulative wall ceiling can increase. Data, optimizer, objective,
    owner names, schedule/horizon, seed, hardware and runtime policy stay bound.
    Qualification and resource-only profile receipts are workflow records; their
    author must actually execute/audit the bound code. They are not permissions
    to access protected data. Use start_resource_fork_attempt to import the opening balance under its own
    completed attempt before new timed work; the common fitter enforces no rollback.
    """
    if not isinstance(contract,RecoveryResourceFork):raise ValueError('typed resource fork required')
    if ref.manifest_sha256!=contract.parent_checkpoint_sha256 or run_id(ancestor_spec)!=contract.parent_run_id:
        raise ValueError('resource fork ancestor checkpoint/run differs')
    old,new=thaw(ancestor_spec.resolved_config),thaw(child_spec.resolved_config)
    if (set(new)!=set(old)|{'resource_fork'} or new.get('resource_fork')!=contract.identity()
            or new.get('study_contract_sha256')!=contract.new_study_contract_sha256
            or old.get('study_contract_sha256')==new.get('study_contract_sha256')
            or any(new[k]!=v for k,v in old.items() if k!='study_contract_sha256')):
        raise ValueError('resource fork must preserve all scientific configuration')
    if not all(k in old for k in ('training','recovery_objective','study_contract_sha256')):
        raise ValueError('only explicitly bound recovery studies can fork')
    for field in ('method_id','source_commit','dependency_lock_sha256','hardware_policy','seeds'):
        if getattr(child_spec,field)!=getattr(ancestor_spec,field):raise ValueError('fork changed '+field)
    if (child_spec.method_variant==ancestor_spec.method_variant
            or child_spec.parent_run_ids!=(*ancestor_spec.parent_run_ids,run_id(ancestor_spec))):
        raise ValueError('fork requires a new identity with its actual parent run')
    if (child_spec.budgets.wall_seconds<=ancestor_spec.budgets.wall_seconds
            or any(getattr(child_spec.budgets,k)!=v for k,v in ancestor_spec.budgets.to_dict().items() if k!='wall_seconds')):
        raise ValueError('schema1 permits only a larger cumulative wall ceiling')
    before,after=thaw(ancestor_spec.input_manifests),thaw(child_spec.input_manifests)
    changed={'executed_source','controls','profile'};added={'fork_checkpoint','fork_ledger'}
    if (set(after)!=set(before)|added or any(after[k]!=v for k,v in before.items() if k not in changed)
            or before.get('executed_source')!=contract.parent_source_sha256
            or after.get('executed_source')!=contract.child_source_sha256
            or after.get('controls')!=contract.qualification_sha256 or after.get('profile')!=contract.profile_sha256
            or after.get('fork_checkpoint')!=ref.manifest_sha256 or after.get('fork_ledger')!=contract.parent_ledger_sha256):
        raise ValueError('fork changed scientific inputs or receipt bindings')
    qualification=_receipt(qualification_path,contract.qualification_sha256)
    if (qualification.get('purpose')!='resource_fork_compatibility_v1'
            or qualification.get('parent_source_sha256')!=contract.parent_source_sha256
            or qualification.get('child_source_sha256')!=contract.child_source_sha256):
        raise ValueError('fork compatibility qualification has different sources')
    profile=_receipt(profile_path,contract.profile_sha256)
    if (profile.get('purpose')!='resource_only_completion_profile_v1'
            or profile.get('parent_ledger_sha256')!=contract.parent_ledger_sha256
            or type(profile.get('fixed_updates')) is not int
            or profile.get('fixed_updates')!=old['training']['max_updates']
            or type(profile.get('cumulative_wall_seconds')) not in (int,float)
            or profile.get('cumulative_wall_seconds')!=child_spec.budgets.wall_seconds):
        raise ValueError('resource profile must bind ancestor spending and unchanged completed horizon')
    ledger_path=Path(ledger_path)
    if ledger_path.name!='costs.jsonl' or ledger_path.is_symlink():
        raise ValueError('canonical ancestor costs.jsonl is required, never a checkpoint-era snapshot')
    ledger_path=ledger_path.resolve()
    if (not ledger_path.is_file() or ledger_path.parent.name!=run_id(ancestor_spec)
            or (ledger_path.parent/'spec.json').read_bytes()!=canonical_json(ancestor_spec.to_dict())):
        raise ValueError('ancestor spending ledger is not bound to its RunSpec')
    with locked_ledger(ledger_path) as stream:
        data=stream.read()
        if _sha(data)!=contract.parent_ledger_sha256:raise ValueError('ancestor spending changed from fork freeze')
        rows=read_rows(io.BytesIO(data))
        spending=BudgetTracker._total(None,rows)
    del rows,data
    state=load_checkpoint(ref,ancestor_spec)
    if (state.source_hash!=after.get('raw_plan') or state.step>old['training']['max_updates']
            or spending.raw_corpus_target_presentations<state.target_cursor):
        raise ValueError('fork checkpoint differs from its raw plan/completed horizon')
    for k,v in state.budget_totals.items():
        current=getattr(spending,k,None)
        if isinstance(v,(int,float)) and current is not None and current<v:
            raise ValueError('ancestor spending cannot roll back below its safe checkpoint')
    for k,limit in child_spec.budgets.to_dict().items():
        actual=getattr(spending,k)
        if actual is not None and actual>limit:raise ValueError('inherited spending already exceeds child budget')
    with locked_ledger(ledger_path) as stream:
        if _sha(stream.read())!=contract.parent_ledger_sha256:
            raise ValueError('ancestor worker is still changing spending; stop it before forking')
    child_state=replace(state,spec=child_spec,budget_totals=spending.to_dict())
    record=freeze(dict(schema_version=1,kind='resource_only_recovery_fork_v1',parent_run_id=run_id(ancestor_spec),
        child_run_id=run_id(child_spec),contract=contract.identity(),safe_step=state.step,
        committed_raw_cursor=state.target_cursor,inherited_spending=spending.to_dict(),
        schedule_policy='unchanged original completed horizon; optimizer/RNG/cursors retained',
        cost_policy='opening balance carried once; incremental child costs reported separately to avoid double counting ancestry',
        interpretation='new exploratory resource amendment; ancestor failure and original limits remain unchanged'))
    return RecoveryForkState(child_state,spending,record)


def _atomic_json(path,value,*,exclusive=False):
    import os
    from uuid import uuid4
    temporary=path.parent/('.pending-'+uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical_json(value));stream.flush();os.fsync(stream.fileno())
        if exclusive:os.link(temporary,path)
        else:os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def start_resource_fork_attempt(branch,root):
    """Import ancestry once under its own attempt, then start new timed work.

    A distinct import attempt is essential: finish_attempt_cost must not count
    inherited wall time as newly executed time and thereby miss child overhead.
    A durable intent precedes the cost append; retries inspect that exact row and
    never erase or import it twice. Concurrent training workers remain forbidden.
    """
    from poc.records import AttemptRecord,start_attempt
    if not isinstance(branch,RecoveryForkState) or CostRecord(**branch.state.budget_totals)!=branch.opening_balance:
        raise ValueError('verified fork state and exact inherited balance required')
    spec=branch.state.spec;identity=run_id(spec);run_path=Path(root).resolve()/identity
    run_path.mkdir(parents=True,exist_ok=True)
    body={'schema_version':1,'kind':'inherited_ancestor_balance_v1','child_run_id':identity,
          'parent_run_id':branch.record['parent_run_id'],'opening_balance':branch.opening_balance.to_dict(),
          'fork_record_sha256':_sha(canonical_json(thaw(branch.record)))}
    with locked_ledger(run_path/'.fork-opening.lock'):
        intent=run_path/'ancestor-opening.json';ledger_path=run_path/'costs.jsonl'
        if intent.exists():
            saved=json.loads(intent.read_text());import_id=saved.get('import_attempt_id')
            if ({k:v for k,v in saved.items() if k!='import_attempt_id'}!=body
                    or not isinstance(import_id,str) or not re.fullmatch('[0-9a-f]{32}',import_id)):
                raise ValueError('different or corrupt ancestor opening-balance intent')
            imported=AttemptRecord(identity,import_id,run_path/'attempts'/import_id,spec)
            metadata=json.loads((imported.path/'run.json').read_text())
            if (metadata.get('run_id')!=identity or metadata.get('attempt_id')!=import_id
                    or metadata.get('spec')!=spec.to_dict()
                    or (run_path/'spec.json').read_bytes()!=canonical_json(spec.to_dict())):
                raise ValueError('inherited balance attempt identity differs')
        else:
            if ledger_path.exists() and ledger_path.stat().st_size:
                raise ValueError('cannot add ancestry to an existing unbound spending ledger')
            imported=start_attempt(spec,Path(root))
            _atomic_json(intent,{**body,'import_attempt_id':imported.attempt_id},exclusive=True)
        with locked_ledger(ledger_path) as stream:rows=read_rows(stream)
        imports=[row for row in rows if row.get('attempt_id')==imported.attempt_id]
        if imports:
            if len(imports)!=1 or rows[0]!=imports[0] or imports[0].get('cost')!=branch.opening_balance.to_dict():
                raise ValueError('ancestor balance was modified or imported more than once')
        elif rows:
            raise ValueError('child work precedes its required ancestor balance')
        else:BudgetTracker(spec.budgets,imported).record(branch.opening_balance)
        metadata=json.loads((imported.path/'run.json').read_text())
        if metadata.get('status')!='exhausted':
            append_event(imported,{'kind':'ancestor_balance_imported','parent_run_id':branch.record['parent_run_id'],
                'executed_work':False,'cost_policy':'historical opening balance; exclude from incremental child attempt expense'})
            metadata['status']='exhausted';_atomic_json(imported.path/'run.json',metadata)
        return start_attempt(spec,Path(root))

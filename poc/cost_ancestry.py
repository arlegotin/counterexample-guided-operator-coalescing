"""Bounded saved-cost admission: final ledgers, explicit unknowns and aliases."""
from pathlib import Path
import json,math
from poc.contracts import CostRecord,Budget,RunSpec
from poc.budget import BudgetTracker
from poc.recovery_endpoints import finite_json,verify_refs,file_sha
from poc.records import canonical_json
import hashlib


def _read(ref):verify_refs([ref]);return finite_json(ref['path'])


def ledger_total(ref):
    """Stream immutable cost rows; do not materialize a large repeated ledger."""
    verify_refs([ref]);info={'rows':0,'overruns':0,'attempt_ids':set()}
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate cost record key')
            result[key]=value
        return result
    def rows():
        with Path(ref['path']).open('rb') as stream:
            while line:=stream.readline(1024*1024+1):
                if len(line)>1024*1024 or not line.endswith(b'\n'):raise ValueError('oversized/truncated cost row retained for diagnosis')
                row=json.loads(line,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite cost record')))
                info['rows']+=1
                if (type(row) is not dict or row.get('kind')!='cost' or type(row.get('event_id')) is not int
                        or row['event_id']!=info['rows'] or type(row.get('attempt_id')) is not str or not row['attempt_id']):
                    raise ValueError('cost ledger identity/order differs')
                CostRecord(**row['cost']);info['attempt_ids'].add(row['attempt_id']);info['overruns']+=int('overrun_reason' in row)
                yield row
    total=BudgetTracker._total(None,rows()).to_dict()
    if not info['rows']:raise ValueError('empty ledger supplies no measured final cost')
    verify_refs([ref]);info['attempt_ids']=sorted(info['attempt_ids'])
    return total,info


def read_final_cost(ref,*,expected_run_id,expected_contract_sha256=None):
    value=_read(ref)
    legacy={'schema_version','run_id','study_contract_sha256','ledger','totals','scope'}
    modern={'schema_version','run_id','scientific_contract_sha256','execution_contract_sha256','ledger','totals','resource','work_allowance','scope'}
    if (set(value) not in (legacy,modern) or type(value.get('schema_version')) is not int or value['schema_version']!=1
            or value.get('run_id')!=expected_run_id or Path(value['ledger']['path']).parent.name!=expected_run_id):
        raise ValueError('exact final terminal cost receipt required; preliminary snapshots are insufficient')
    if expected_contract_sha256 is not None and value.get('scientific_contract_sha256',value.get('study_contract_sha256'))!=expected_contract_sha256:
        raise ValueError('terminal scientific cost identity differs')
    total,info=ledger_total(value['ledger'])
    if total!=value['totals']:raise ValueError('terminal cost differs from complete immutable ledger')
    return {**value,'ledger_inspection':info,'terminal_reference':ref}


def _raw_node(node):
    spec=_read(node['spec']);verify_refs([node['ledger']])
    if (node['spec']['sha256']!=node['run_id'] or hashlib.sha256(canonical_json(spec)).hexdigest()!=node['run_id']
            or Path(node['ledger']['path']).parent.name!=node['run_id']):raise ValueError('raw ancestor run/spec/ledger identity differs')
    parsed=RunSpec(**{**spec,'budgets':Budget(**spec['budgets'])});total,info=ledger_total(node['ledger'])
    attempts={}
    for ref in node['attempts']:
        value=_read(ref)
        if value.get('run_id')!=node['run_id'] or type(value.get('attempt_id')) is not str:
            raise ValueError('ancestor attempt identity differs')
        attempts[value['attempt_id']]={'reference':ref,'status':value.get('status'),'failure':value.get('failure_details')}
    if not set(info['attempt_ids'])<=set(attempts):raise ValueError('ancestor ledger includes an undeclared attempt')
    if node.get('snapshot') is not None:
        snapshot=_read(node['snapshot']).get('cost')
        if type(snapshot) is not dict:raise ValueError('named ancestor snapshot lacks typed cost')
        prior=CostRecord(**snapshot).to_dict()
        for key,value in prior.items():
            if type(value) in (int,float) and total[key] is not None and total[key]<value:
                raise ValueError('final ancestor ledger omits cost already present at saved result')
    violations=[key for key,value in parsed.budgets.to_dict().items() if total.get(key) is not None and total[key]>value]
    return {'ledger':node['ledger'],'totals':total,'run_id':node['run_id'],'spec':node['spec'],
        'attempts':attempts,'ledger_inspection':info,'budget_violations':violations,
        'closed':all(v['status'] in ('exhausted','failed','complete') for v in attempts.values())}


def read_cost_ancestry(reference,*,expected_parent):
    d=_read(reference)
    if (type(d) is not dict or set(d)!={'schema_version','study','parent_config','nodes','scope'}
            or type(d['schema_version']) is not int or d['schema_version']!=1 or d['parent_config']!=expected_parent
            or type(d['nodes']) is not list):raise ValueError('closed named ancestor inventory/configuration required')
    verify_refs([expected_parent]);ledgers={};unknown=[];nodes=[];ids=set()
    for node in d['nodes']:
        base={'kind','id','classification'}
        if type(node) is not dict or not base<=set(node) or type(node['id']) is not str or not node['id'] or node['id'] in ids:
            raise ValueError('distinct explicit ancestor identities required')
        ids.add(node['id']);kind=node['kind']
        if kind=='unknown':
            if set(node)!=base|{'reason','references'} or type(node['reason']) is not str or not node['reason']:
                raise ValueError('unknown ancestor requires explicit reason and references')
            verify_refs(node['references']);unknown.append({**node,'cost':None});nodes.append({'id':node['id'],'status':'unknown'});continue
        required=base|({'run_id','terminal'} if kind=='terminal' else {'run_id','ledger','spec','attempts'} if kind=='raw_ledger' else set())
        if kind not in ('terminal','raw_ledger') or not required<=set(node) or set(node)-required-{'admission','snapshot'}:
            raise ValueError('only final terminal or complete raw-ledger ancestors accepted')
        if 'admission' in node:_read(node['admission'])
        value=read_final_cost(node['terminal'],expected_run_id=node['run_id']) if kind=='terminal' else _raw_node(node)
        ledger=value['ledger'];key=ledger['sha256']
        if key in ledgers:
            if ledgers[key]['totals']!=value['totals']:raise ValueError('shared ancestor ledger has contradictory totals')
            ledgers[key]['aliases'].append(node['id'])
        else:ledgers[key]={**value,'classification':node['classification'],'aliases':[node['id']]}
        nodes.append({'id':node['id'],'status':'verified_saved_cost','ledger_sha256':key,'classification':node['classification']})
        if value.get('closed') is False:unknown.append({'id':node['id']+':unclosed','classification':node['classification'],
            'reason':'retained spending prefix verified; one or more original attempts did not close','cost':None})
    total=CostRecord()
    for value in ledgers.values():total=BudgetTracker.combine(total,CostRecord(**value['totals']))
    return {'schema_version':1,'inventory':reference,'parent_config':expected_parent,'nodes':nodes,'ledgers':ledgers,
        'unknown_nodes':unknown,'known_totals':total.to_dict(),'complete_for_declared_inventory':not unknown,
        'all_upstream_expense_established':False,
        'scope':'bounded declared ancestor inventory; shared ledger aliases counted once, unknown costs never measured zeros; no complete lifetime research-expense claim',
        'units':'historical source-defined counters retained per node; aggregate charged seconds are not critical-path elapsed time; unique-token defaults are not measured coverage'}


def prior_calibration_inventory_reference(admission_contract,expected_path):
    refs=admission_contract.get('source_files')
    if type(refs) is not list:raise ValueError('original calibration admission source_files closure required')
    found=[r for r in refs if type(r) is dict and r.get('path')==str(expected_path)]
    if len(found)!=1:raise ValueError('prior calibration failure inventory is missing or ambiguous in original admission')
    verify_refs(found);return found[0]

"""Final parent costs are verified, deduplicated and never replaced by snapshots."""
from pathlib import Path
import json
import pytest
from poc.contracts import CostRecord
from poc.records import canonical_json
from poc.recovery_endpoints import file_sha


def reference(path):return {'path':str(path),'sha256':file_sha(path)}


@pytest.fixture
def terminal_packet(tmp_path):
 from poc.budget import BudgetTracker
 run_id='a'*64;root=tmp_path/run_id;root.mkdir()
 rows=[{'event_id':1,'kind':'cost','attempt_id':'owned','timestamp':'owned','cost':CostRecord(updates=2,valid_target_presentations=1022,raw_corpus_target_presentations=1022,memory_bytes=1024,memory_unavailable_reason=None).to_dict()}]
 ledger=root/'costs.jsonl';ledger.write_bytes(b''.join(canonical_json(row) for row in rows))
 terminal=tmp_path/'terminal.json';value={'schema_version':1,'run_id':run_id,'study_contract_sha256':'b'*64,
  'ledger':reference(ledger),'totals':BudgetTracker._total(None,rows).to_dict(),'scope':'owned final terminal'}
 terminal.write_bytes(canonical_json(value));parent=tmp_path/'parent.yaml';parent.write_text('owned: true\n')
 return terminal,value,reference(parent)


def test_review_r5_final_terminal_cost_recomputed_from_exact_ledger(terminal_packet):
 from poc.cost_ancestry import read_final_cost
 path,value,parent=terminal_packet
 assert read_final_cost(reference(path),expected_run_id=value['run_id'])['totals']==value['totals']


@pytest.mark.parametrize('change',['missing','changed','ledger_drift','preliminary'])
def test_review_r5_missing_changed_or_preliminary_parent_cost_denied(terminal_packet,change):
 from poc.cost_ancestry import read_final_cost
 path,value,parent=terminal_packet;frozen=reference(path)
 if change=='missing':path.unlink()
 elif change=='changed':path.write_text('{"changed":true}')
 elif change=='ledger_drift':Path(value['ledger']['path']).write_text('{"changed":true}\n')
 else:
  value['cost_snapshot_preliminary']=True;path.write_bytes(canonical_json(value));frozen=reference(path)
 with pytest.raises((ValueError,FileNotFoundError)):
  read_final_cost(frozen,expected_run_id=value['run_id'])


def test_review_r5_shared_ancestor_ledger_counted_once_with_all_aliases(terminal_packet,tmp_path):
 from poc.cost_ancestry import read_cost_ancestry
 path,value,parent=terminal_packet
 node={'kind':'terminal','classification':'shared_calibration','run_id':value['run_id'],'terminal':reference(path)}
 inventory={'schema_version':1,'study':'owned','parent_config':parent,'nodes':[dict(node,id='calibration-profile'),dict(node,id='shared-preparation-alias')],
  'scope':'bounded declared ancestor inventory; all cost identities retained'}
 desc=tmp_path/'ancestry.json';desc.write_bytes(canonical_json(inventory))
 result=read_cost_ancestry(reference(desc),expected_parent=parent)
 assert result['known_totals']['updates']==2 and len(result['ledgers'])==1
 assert sorted(next(iter(result['ledgers'].values()))['aliases'])==['calibration-profile','shared-preparation-alias']
 assert result['complete_for_declared_inventory'] is True


def test_review_r5_unknown_ancestor_is_not_a_measured_zero_or_complete_expense(terminal_packet,tmp_path):
 from poc.cost_ancestry import read_cost_ancestry
 path,value,parent=terminal_packet
 inventory={'schema_version':1,'study':'owned','parent_config':parent,'nodes':[{'kind':'unknown','id':'unavailable-export',
  'classification':'source_export','reason':'no authoritative terminal ledger available','references':[parent]}],
  'scope':'bounded declared ancestor inventory'}
 desc=tmp_path/'ancestry.json';desc.write_bytes(canonical_json(inventory))
 result=read_cost_ancestry(reference(desc),expected_parent=parent)
 assert result['complete_for_declared_inventory'] is False and result['all_upstream_expense_established'] is False
 assert result['unknown_nodes'][0]['cost'] is None


def test_review_r5_raw_recovery_cost_reconstructs_typed_run_spec_and_attempts(terminal_packet,tmp_path):
 from poc.cost_ancestry import read_cost_ancestry
 from poc.contracts import RunSpec,Budget
 from poc.records import start_attempt
 from poc.budget import BudgetTracker
 from provenance.recovery_diagnosis import finish_attempt
 _,_,parent=terminal_packet
 budget=Budget(2,1022,0,0,2,0,0,30,1024**3,8*1024**3)
 spec=RunSpec('R','owned_saved_cost',{'owned':True},'0'*40,'a'*64,'b'*64,{'owned':'c'*64},{'device':'cpu'},{'seed':17},(),budget)
 attempt=start_attempt(spec,tmp_path/'records');rec=BudgetTracker(budget,attempt)
 rec.record(CostRecord(updates=2,valid_target_presentations=1022,student_backward_calls=2,raw_corpus_target_presentations=1022,memory_bytes=1024,memory_unavailable_reason=None))
 finish_attempt(attempt,'exhausted',reason='owned completed fixture')
 spec_path=rec.path.parent/'spec.json'
 node={'kind':'raw_ledger','id':'source-recovery','classification':'source_recovery','run_id':attempt.run_id,
       'ledger':reference(rec.path),'spec':reference(spec_path),'attempts':[reference(attempt.path/'run.json')]}
 inventory={'schema_version':1,'study':'owned','parent_config':parent,'nodes':[node],'scope':'owned raw ancestor'}
 desc=tmp_path/'raw-ancestry.json';desc.write_bytes(canonical_json(inventory))
 result=read_cost_ancestry(reference(desc),expected_parent=parent)
 assert result['known_totals']['updates']==2 and result['complete_for_declared_inventory'] is True


def test_review_r5_prior_failure_inventory_uses_actual_admission_source_files(tmp_path):
 from poc.cost_ancestry import prior_calibration_inventory_reference
 path=tmp_path/'resource-amendment.json';path.write_text('{"prior_inventory":[]}')
 frozen=reference(path)
 assert prior_calibration_inventory_reference({'source_files':[frozen]},path)==frozen
 with pytest.raises(ValueError):prior_calibration_inventory_reference({'files':[frozen]},path)
 with pytest.raises(ValueError):prior_calibration_inventory_reference({'source_files':[]},path)

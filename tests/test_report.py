"""Authored summaries exercise terminal claims; fixtures are never study results."""
import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('poc.report'), 'terminal classification and traceability missing'
    return importlib.import_module('poc.report')


@pytest.mark.parametrize('fields,expected', [
    ({'deployment_improvement': True, 'attribution_improvement': True, 'guardrails_passed': True,
      'independent_replication': True, 'sensitivity_adequate': True}, 'confirmed_scoped_improvement'),
    ({'mechanism_benefit': True, 'sensitivity_adequate': True}, 'useful_mechanism_without_deployment_advantage'),
    ({'evidence_against': True, 'sensitivity_adequate': True}, 'evidence_against_tested_local_hypothesis'),
    ({}, 'diagnosed_inconclusive'),
])
def test_four_outcomes_require_their_evidence(fields, expected):
    a = api()
    assert a.classify_outcome([], a.AnalysisResult(**fields)).value == expected


@pytest.mark.parametrize('reason', ['invalid implementation', 'hardware limitation', 'null controls fail',
                                   'unsuitable proxy', 'inadequate sensitivity', 'budget limitation'])
def test_early_stops_override_apparent_positive_summary(reason):
    a = api()
    gate = a.GateRecord('fixture', 'stop', reason, ('receipt',))
    analysis = a.AnalysisResult(deployment_improvement=True, attribution_improvement=True,
                               guardrails_passed=True, independent_replication=True,
                               sensitivity_adequate=True)
    assert a.classify_outcome([gate], analysis).value == 'diagnosed_inconclusive'


def test_positive_deployment_without_attribution_is_scoped_deployment_result():
    a = api()
    result = a.AnalysisResult(deployment_improvement=True, guardrails_passed=True,
                             independent_replication=True, sensitivity_adequate=True)
    assert a.classify_outcome([], result).value == 'confirmed_scoped_improvement'
    assert result.attribution_improvement is False


def test_capability_regression_cannot_be_hidden_by_lower_kl():
    a = api()
    result = a.AnalysisResult(deployment_improvement=True, attribution_improvement=True,
                             guardrails_passed=False, sensitivity_adequate=True)
    assert a.classify_outcome([], result).value == 'evidence_against_tested_local_hypothesis'


@pytest.mark.parametrize('flag', ['unresolved_sv_comparison', 'budget_limited_curve', 'insufficient_sensitivity'])
def test_unresolved_comparison_or_learning_cannot_confirm_a_win(flag):
    a = api()
    result = a.AnalysisResult(deployment_improvement=True, attribution_improvement=True,
                             guardrails_passed=True, independent_replication=True,
                             sensitivity_adequate=True, **{flag: True})
    assert a.classify_outcome([], result).value == 'diagnosed_inconclusive'


def fixture(tmp_path):
    a = api()
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('{"measured": {"sum": 6.0, "targets": 3}}\n')
    return a.EvidenceIndex(root=tmp_path, records={'receipt': a.EvidenceRecord(
        'receipt.json', a.sha256_file(receipt))},
        claims=(a.Claim('sum', 6.0, 'receipt', '/measured/sum', 'tiny authored fixture'),),
        stages={'Task13': 'unexecuted'}, gates=(), analysis=None,
        sections=(('Fixture', (('Sum', 'sum'),)),), metrics=(), failures=('authored failure',))


def test_render_labels_unexecuted_and_retains_failures(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    report = a.render_report(evidence, include_failures=True)
    assert 'Task13: unexecuted' in report.markdown
    assert 'authored failure' in report.markdown
    assert '[claim:sum]' in report.markdown
    assert a.audit_claims(report, evidence).ok
    assert 'authored failure' not in a.render_report(evidence, include_failures=False).markdown


def test_claim_must_resolve_to_exact_number_and_immutable_record(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    report = a.render_report(evidence, include_failures=True)
    bad = replace(report, claims=(replace(report.claims[0], value=7.0),))
    assert not a.audit_claims(bad, evidence).ok
    missing = replace(report, claims=(replace(report.claims[0], record='missing'),))
    assert not a.audit_claims(missing, evidence).ok
    (tmp_path / 'receipt.json').write_text('{"measured": {"sum": 7.0}}')
    assert not a.audit_claims(report, evidence).ok


def test_pointer_escaping_and_unreferenced_claims_fail_closed(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    report = a.render_report(evidence, include_failures=True)
    assert not a.audit_claims(replace(report, markdown=report.markdown.replace('[claim:sum]', '')), evidence).ok
    assert not a.audit_claims(replace(report, markdown=report.markdown+' [claim:invented]'), evidence).ok
    assert a.resolve_pointer({'a/b': {'~': [9]}}, '/a~1b/~0/0') == 9


def test_metric_schema_preserves_sums_negative_kl_and_unavailable(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    row = {'metric_id': 'tiny', 'status': 'measured', 'study': 'fixture', 'scope': 'authored',
           'kl_sum': -1e-9, 'valid_targets': 3, 'nll_sum': 2.0, 'component_id': 'c1',
           'document_id': 'd1', 'record': 'receipt', 'pointer': '/measured', 'source_sha256': 'f'*64}
    path = tmp_path / 'metrics.parquet'
    a.write_metrics(path, (row, {'metric_id': 'unrun', 'status': 'unexecuted', 'scope': 'fixture'}))
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    assert rows[0]['kl_sum'] == -1e-9 and rows[0]['valid_targets'] == 3
    assert rows[1]['kl_sum'] is None and rows[1]['valid_targets'] is None
    with pytest.raises(ValueError, match='denominator'):
        a.write_metrics(path, ({**row, 'valid_targets': 0},))


def test_bundle_census_hashes_bytes_without_loading_model_and_rejects_tamper(tmp_path):
    a = api(); bundle = tmp_path / 'bundle'; bundle.mkdir()
    (bundle/'weights.bin').write_bytes(b'fixture')
    manifest = {'file_entries': [{'path': 'weights.bin', 'bytes': 7,
                                 'sha256': a.sha256_file(bundle/'weights.bin')}]}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    proof = a.inspect_existing_bundle(bundle, a.sha256_file(bundle/'manifest.json'))
    assert proof['payload_bytes'] == (bundle/'manifest.json').stat().st_size + 7
    (bundle/'weights.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        a.inspect_existing_bundle(bundle, a.sha256_file(bundle/'manifest.json'))


def test_metric_audit_rejects_invented_sum_or_missing_source(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    row = {'metric_id': 'fixture', 'status': 'measured', 'record': 'receipt',
           'pointer': '/measured', 'kl_sum': 6.0, 'valid_targets': 3,
           'source_sha256': evidence.records['receipt'].sha256, 'scope': 'authored fixture'}
    # Source must use the real metric schema, not a numerically similar unrelated key.
    bad = replace(evidence, metrics=(row,))
    assert not a.audit_claims(a.render_report(bad, include_failures=True), bad).ok


def test_report_cli_writes_portable_handoff_and_fails_on_tamper(tmp_path, capsys):
    a = api(); evidence = fixture(tmp_path)
    (tmp_path/'provenance').mkdir()
    payload = {'records': {'receipt': {'path': 'receipt.json', 'sha256': evidence.records['receipt'].sha256}},
               'claims': [{'id':'sum','value':6.0,'record':'receipt','pointer':'/measured/sum','scope':'authored'}],
               'stages': {'Task13':'unexecuted'}, 'gates': [], 'analysis': {},
               'sections': [['Fixture', [['Sum','sum']]]], 'metrics': [], 'failures': [],
               'narrative': 'Tiny authored fixture.', 'release_files': ['receipt.json']}
    (tmp_path/'provenance/report-evidence-2026-10-28.json').write_text(json.dumps(payload))
    assert a.report_command(include_failures=True, root=tmp_path) == 0
    assert json.loads((tmp_path/'results/summary.json').read_text())['outcome']=='diagnosed_inconclusive'
    manifest=json.loads((tmp_path/'artifacts/release_manifest.json').read_text())
    assert all(a.sha256_file(tmp_path/f['path'])==f['sha256'] for f in manifest['files'])
    (tmp_path/'receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        a.report_command(include_failures=True, root=tmp_path)


def test_unconfirmed_deployment_gain_is_not_mechanism_evidence():
    a = api()
    result = a.AnalysisResult(deployment_improvement=True, guardrails_passed=True, sensitivity_adequate=True)
    assert a.classify_outcome([], result).value == 'diagnosed_inconclusive'


def test_report_text_and_summary_cannot_disagree_with_registered_claims(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    report = a.render_report(evidence, include_failures=True)
    altered = replace(report, markdown=report.markdown.replace('6.0 [claim:sum]', '7.0 [claim:sum]'))
    assert not a.audit_claims(altered, evidence).ok
    altered_summary = {**report.summary, 'claims': []}
    assert not a.audit_claims(replace(report, summary=altered_summary), evidence).ok


def test_portable_actual_evidence_reduces_original_and_fresh_damage_from_sums():
    a = api(); evidence = a.load_evidence(Path(__file__).resolve().parents[1])
    assert a.audit_claims(a.render_report(evidence, include_failures=True), evidence).ok
    # Independently hand-selected recorded endpoints, not an expectation from rendering code.
    for study, pair, expected in [('original_primary','[12, 20]',0.1639481259622919),
                                  ('fresh_exploratory','[4, 6]',0.15121223012061968)]:
        rows=[r for r in evidence.metrics if r.get('study')==study and r.get('pair')==pair
              and r.get('source_pointer','').startswith('/natural_damage_rows/')]
        assert len(rows)==(16 if study=='original_primary' else 32)
        assert sum(r['kl_sum'] for r in rows)/sum(r['valid_targets'] for r in rows)==pytest.approx(expected,abs=1e-14)
    unavailable=[r for r in evidence.metrics if r['status']=='unexecuted']
    assert unavailable and all(r.get('valid_targets') is None and r.get('kl_sum') is None for r in unavailable)


def test_raw_metric_origin_sha_and_pointer_are_audited():
    a = api(); evidence = a.load_evidence(Path(__file__).resolve().parents[1])
    first = evidence.metrics[0]
    bad = replace(evidence,metrics=({**first,'source_sha256':'0'*64},*evidence.metrics[1:]))
    assert not a.audit_claims(a.render_report(bad,include_failures=True),bad).ok
    bad = replace(evidence,metrics=({**first,'source_pointer':'/invented'},*evidence.metrics[1:]))
    assert not a.audit_claims(a.render_report(bad,include_failures=True),bad).ok


def test_portable_metric_snapshot_excludes_private_payload_keys():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root/'provenance/report-metric-sources-2026-10-28.json').read_text())
    forbidden = {'text','raw_text','tokens','input_ids','recipient_tokens','donor_tokens','replacement','attention_mask','positions','mask','target_mask'}
    def check(node):
        if isinstance(node, dict):
            assert not forbidden.intersection(node)
            for child in node.values(): check(child)
        elif isinstance(node, list):
            for child in node: check(child)
    check(value)
    assert len({r['metric_id'] for r in value['metrics']}) == len(value['metrics'])


def test_compact_findings_table_uses_same_claim_as_detailed_ledger(tmp_path):
    a = api(); evidence = fixture(tmp_path)
    source=tmp_path/'receipt.json'
    source.write_text('{"point":{"selected_pair":[2,4],"regret":0.25}}')
    evidence=replace(evidence,records={'receipt':a.EvidenceRecord('receipt.json',a.sha256_file(source))},
                     claims=(a.Claim('point',{'selected_pair':[2,4],'regret':0.25},'receipt','/point','authored'),),
                     sections=(('Detail',(('Point','point'),)),),
                     overview=(('Authored fixture','point','point','point','point'),))
    report=a.render_report(evidence,include_failures=True)
    assert '| Authored fixture | (2, 4); regret 0.25' in report.markdown
    assert a.audit_claims(report,evidence).ok


def test_supported_scientific_negative_stop_requires_explicit_validity():
    a = api()
    gate = a.GateRecord('WP04', 'stop', 'authored negative', ('receipt',),
                        category='scientific_negative', record_id='negative')
    result = a.AnalysisResult(evidence_against=True, sensitivity_adequate=True, validity_confirmed=True)
    assert a.classify_outcome([gate], result).value == 'evidence_against_tested_local_hypothesis'
    assert a.classify_outcome([gate], replace(result, validity_confirmed=False)).value == 'diagnosed_inconclusive'
    assert a.classify_outcome([replace(gate,evidence_refs=())], result).value == 'diagnosed_inconclusive'


@pytest.mark.parametrize('decision', ['amend','stop','invalid','inconclusive'])
def test_explicitly_superseded_gate_history_does_not_veto_valid_assessment(decision):
    a = api()
    history = a.GateRecord('WP04', decision, 'historical', ('old',), category='control', record_id='old')
    current = a.GateRecord('WP04', 'pass', 'resolved with retained proof', ('new',),
                          category='assessment', record_id='new', supersedes=('old',))
    result = a.AnalysisResult(deployment_improvement=True, guardrails_passed=True,
                             sensitivity_adequate=True, independent_replication=True)
    assert a.classify_outcome([history,current], result).value == 'confirmed_scoped_improvement'
    assert history.decision == decision  # The provenance record remains present and unchanged.


@pytest.mark.parametrize('category', ['unknown','control','integrity','resource','sensitivity'])
@pytest.mark.parametrize('negative', [False,True])
def test_unresolved_gate_categories_block_favorable_and_negative_flags(category,negative):
    a=api()
    gate=a.GateRecord('WP04','stop','reason prose cannot authorize resolution',('proof',),category=category)
    result=a.AnalysisResult(evidence_against=negative,deployment_improvement=not negative,
                           guardrails_passed=True,independent_replication=True,sensitivity_adequate=True,
                           validity_confirmed=True)
    assert a.classify_outcome([gate],result).value=='diagnosed_inconclusive'


def test_gate_supersession_requires_ordered_unique_same_gate_evidence():
    a=api(); old=a.GateRecord('WP02','stop','failed',('old',),record_id='old')
    new=a.GateRecord('WP02','pass','fixed',('new',),record_id='new',supersedes=('old',))
    for gates in [[new,old], [old,replace(new,gate='WP04')], [old,replace(new,evidence_refs=())],
                  [old,replace(new,record_id='old')]]:
        with pytest.raises(ValueError,match='gate'):
            a.classify_outcome(gates,a.AnalysisResult())
    other=a.GateRecord('WP03','stop','still unresolved',('other',),category='resource')
    assert a.classify_outcome([old,new,other],a.AnalysisResult(evidence_against=True,sensitivity_adequate=True)).value=='diagnosed_inconclusive'


def test_current_wp04_and_low_sensitivity_remain_inconclusive_after_gate_fix():
    a=api(); actual=a.load_evidence(Path(__file__).resolve().parents[1])
    assert a.classify_outcome(actual.gates,actual.analysis).value=='diagnosed_inconclusive'
    negative=a.GateRecord('WP04','stop','authored',('proof',),category='scientific_negative')
    result=a.AnalysisResult(evidence_against=True,sensitivity_adequate=False,validity_confirmed=True)
    assert a.classify_outcome([negative],result).value=='diagnosed_inconclusive'


def index_payload():
    return {'records':{},'claims':[],'stages':{'Task13':'unexecuted'},'gates':[],
            'analysis':None,'sections':[],'metrics':[],'failures':[],
            'narrative':'Authored empty early result.','release_files':[]}


def write_index(root,payload):
    path=root/'provenance/report-evidence-2026-10-28.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize('payload', [
    {}, {'records':[]}, [],
    {**index_payload(),'records':{'bad':'not an entry'}},
    {**index_payload(),'records':{'bad':{'path':4,'sha256':'0'*64}}},
    {**index_payload(),'metrics_record':'missing'},
    {**index_payload(),'analysis':[]},
    {**index_payload(),'analysis':{'evidence_against':'true'}},
    {**index_payload(),'claims':[{'id':'x','value':1,'record':'missing','pointer':'/x','scope':'authored'}]},
    {**index_payload(),'sections':[['bad',[['row','missing']]]]},
    {**index_payload(),'gates':[{'gate':'WP04','decision':'stop','reason':'authored','unknown':True}]},
    {**index_payload(),'metrics':[{'metric_id':'x','status':'measured','scope':'authored','valid_targets':1,'kl_sum':0.,'record':'missing','pointer':'/x'}]},
])
def test_malformed_index_schema_raises_integrity_value_error(tmp_path,payload):
    a=api();write_index(tmp_path,payload)
    with pytest.raises(ValueError,match='evidence index'):
        a.load_evidence(tmp_path)


def test_index_schema_is_checked_before_linked_metric_file_io(tmp_path,monkeypatch):
    a=api(); payload={**index_payload(),'records':{'metrics':{'path':'missing.json','sha256':'0'*64}},
                      'metrics_record':'metrics','analysis':[]}
    write_index(tmp_path,payload)
    with pytest.raises(ValueError,match='evidence index.*analysis'):
        a.load_evidence(tmp_path)


def test_null_analysis_is_supported_early_inconclusive(tmp_path):
    a=api();write_index(tmp_path,index_payload())
    evidence=a.load_evidence(tmp_path)
    assert evidence.analysis is None
    assert a.classify_outcome(evidence.gates,evidence.analysis).value=='diagnosed_inconclusive'


def test_linked_metric_payload_schema_is_validated(tmp_path):
    a=api(); path=tmp_path/'metrics.json';path.write_text('{"metrics":[42]}')
    payload={**index_payload(),'records':{'metrics':{'path':'metrics.json','sha256':a.sha256_file(path)}},
             'metrics_record':'metrics'}
    write_index(tmp_path,payload)
    with pytest.raises(ValueError,match='evidence index.*metrics'):
        a.load_evidence(tmp_path)


def test_malformed_index_cli_returns_exit4_without_traceback(tmp_path,monkeypatch,capsys):
    a=api();write_index(tmp_path,{'records':[]})
    original=a.load_evidence
    monkeypatch.setattr(a,'load_evidence',lambda root:original(tmp_path))
    from poc.cli import main
    assert main(['report','--include-failures'])==4
    stderr=capsys.readouterr().err
    assert 'runtime/data-integrity failure:' in stderr and 'evidence index' in stderr
    assert 'Traceback' not in stderr


def test_internal_programmer_errors_are_not_swallowed_by_cli(monkeypatch):
    a=api()
    def crash(root):raise TypeError('authored programming error')
    monkeypatch.setattr(a,'load_evidence',crash)
    from poc.cli import main
    with pytest.raises(TypeError,match='authored programming error'):
        main(['report'])


def test_later_pass_without_explicit_supersession_does_not_resolve_history():
    a=api()
    gates=[a.GateRecord('WP04','amend','history',('old',),record_id='old'),
           a.GateRecord('WP04','pass','prose says resolved',('new',),record_id='new')]
    assert a.classify_outcome(gates,a.AnalysisResult(evidence_against=True,sensitivity_adequate=True)).value=='diagnosed_inconclusive'


def test_scientific_negative_cannot_bypass_other_guards():
    a=api()
    gate=a.GateRecord('WP04','stop','authored',('proof',),category='scientific_negative')
    result=a.AnalysisResult(evidence_against=True,sensitivity_adequate=True,validity_confirmed=True)
    for flag in ['budget_limited_curve','insufficient_sensitivity','unresolved_sv_comparison']:
        assert a.classify_outcome([gate],replace(result,**{flag:True})).value=='diagnosed_inconclusive'


def test_explicit_current_inconclusive_overrides_adequate_negative_flags():
    a=api()
    gate=a.GateRecord('WP04','inconclusive','authored current assessment',('proof',),category='assessment')
    result=a.AnalysisResult(evidence_against=True,sensitivity_adequate=True,validity_confirmed=True)
    assert a.classify_outcome([gate],result).value=='diagnosed_inconclusive'

"""Eligibility distinguishes missing implementations from actual infeasibility."""
import pytest
pytestmark=pytest.mark.cpu


def test_registry_covers_mandatory_families_without_false_unsupported_claims():
    from poc.methods import method_registry,MethodDescriptor
    registry=method_registry()
    assert set(registry)=={'T','Q','R','RC','CKA','BS','SV','P','H','QCE','M'}
    assert all(isinstance(v,MethodDescriptor) and len(v.evidence_digest)==64 and v.reason for v in registry.values())
    assert registry['SV'].status=='feasible_adaptation_pending'
    assert registry['M'].status=='not_yet_implemented'
    assert 'quality' not in registry['M'].allowed_comparisons


def test_paper_orientation_and_parameter_break_even():
    from poc.methods import sv_feasibility
    value=sv_feasibility([(6,4),(6,4)],4)
    assert value['dense_parameters']==48 and value['factor_parameters']==68
    assert value['full_rank_reconstruction_max_abs_error']<1e-12
    assert value['compression_feasible'] is False
    assert sv_feasibility([(1536,576)]*2,64)['compression_feasible'] is True


def test_format_stage_uses_explicit_pilot_policy_without_resolving_final_budgets(monkeypatch):
    from poc.cli import main
    import poc.quantization_stage as stage
    calls=[]
    monkeypatch.setattr(stage,'run_format_and_quant',lambda config: calls.append(config) or 0)
    assert main(['baselines','--stage','format_and_quant','--config','configs/135m.yaml'])==0
    assert len(calls)==1 and calls[0].budgets.wall_seconds is None


def test_anchor_publication_is_independent_complete_and_idempotent(tmp_path):
    from test_export import model,runtime
    from poc.bundle import write_bundle,measure_bundle
    from poc.quantization_stage import publish_anchor
    from pathlib import Path
    ref=write_bundle(model('0'),tmp_path/'source',runtime(tmp_path))
    published=publish_anchor(ref,tmp_path/'durable')
    assert Path(published.directory).name==ref.manifest_sha256
    assert publish_anchor(ref,tmp_path/'durable')==published
    assert measure_bundle(Path(published.directory)).payload_bytes==measure_bundle(Path(ref.directory)).payload_bytes
    (Path(ref.directory)/'config.json').write_text('destroyed staging')
    assert measure_bundle(Path(published.directory)).payload_bytes>0


def test_prepared_dev_window_respects_existing_token_envelope(tmp_path):
    import json,hashlib
    from poc.quantization_stage import _prepared_window
    from poc.data.manifest import DocumentRecord
    tokens=tmp_path/'tokens.json'
    tokens.write_text(json.dumps({'document_id':'abc','fields':{'text':list(range(700))}}))
    record=DocumentRecord('abc','HuggingFaceFW/fineweb-edu','r','s',0,token_path=str(tokens),token_sha256=hashlib.sha256(tokens.read_bytes()).hexdigest())
    window=_prepared_window(record,17)
    assert len(window.token_ids)==512 and window.token_ids[0]==window.source_start
    bad=DocumentRecord('wrong','f','r','s',0,token_path=str(tokens),token_sha256=record.token_sha256)
    with pytest.raises(ValueError,match='document'): _prepared_window(bad,17)


def test_retry_cost_uses_authoritative_ledger_when_summary_is_stale(tmp_path):
    import json
    from poc.quantization_stage import _prior_costs
    attempt=tmp_path/'attempt-fixture';attempt.mkdir()
    ledger=tmp_path/'costs.jsonl'
    ledger.write_text(json.dumps({'attempt_id':'a','cost':{'wall_seconds':5.,'teacher_forward_calls':7,'updates':2}})+'\n')
    (attempt/'summary.json').write_text(json.dumps({'attempt_id':'a','cost_ledger':str(ledger),'costs':{'wall_seconds':3.,'teacher_forward_calls':1,'conversion_attempts':1}}))
    values=_prior_costs(tmp_path,10.)
    assert values['wall_seconds']==15. and values['teacher_forward_calls']==7
    assert values['fit_updates']==2 and values['conversion_attempts']==1


@pytest.mark.parametrize('error,want',[(ImportError('missing official entry'),'unavailable_upstream_entrypoint'),(ValueError('unsupported native affine bit/group/shape combination'),'unsupported_kernel'),(NotImplementedError('shared pooled adapter'),'not_yet_implemented'),(RuntimeError('bridge unresolved fidelity'),'unresolved_fidelity')])
def test_method_failure_statuses_do_not_confuse_missing_adapters_and_hardware(error,want):
    from poc.quantization_stage import _failure_status
    assert _failure_status(error)==want


def test_resource_interruption_cannot_be_swallowed_as_method_failure():
    from poc.quantization_stage import PilotInterrupted,_failure_status
    from poc.budget import BudgetExceeded
    assert not issubclass(PilotInterrupted,(ValueError,RuntimeError,ImportError))
    with pytest.raises(BudgetExceeded): _failure_status(BudgetExceeded('cap'))


@pytest.mark.parametrize('kind',['interrupt','budget'])
def test_stage_stop_persists_actual_cost_and_prevents_next_operation(tmp_path,monkeypatch,kind):
    import json
    from pathlib import Path
    import poc.quantization_stage as stage
    from poc.budget import BudgetExceeded
    from poc.contracts import CostRecord
    from poc.config import load_config
    policy=json.loads((stage.ROOT/'protocol/quantization-pilot-caps.json').read_text())
    (tmp_path/'protocol').mkdir();(tmp_path/'protocol/quantization-pilot-caps.json').write_text(json.dumps(policy))
    dependency=tmp_path/'provenance/dependency-locks';dependency.mkdir(parents=True)
    (dependency/'qualified-darwin-arm64-py312.requirements.txt').write_text('fixture==1\n')
    (tmp_path/'poc').mkdir()
    calls=[]
    def stop(config,out,report,recorder,policy):
        calls.append('started')
        recorder.record(CostRecord(teacher_forward_calls=2))
        if kind=='interrupt': raise stage.PilotInterrupted('test stop')
        stage._failure_status(BudgetExceeded('test budget'))
        calls.append('must not run')
    config=load_config(Path('configs/135m.yaml'),phase='exploratory')
    monkeypatch.setattr(stage,'ROOT',tmp_path);monkeypatch.setattr(stage,'_run',stop)
    monkeypatch.setattr(stage.PilotRecorder,'monitor',lambda self:None)
    with pytest.raises(stage.PilotInterrupted if kind=='interrupt' else BudgetExceeded): stage.run_format_and_quant(config)
    assert calls==['started']
    report=json.loads(next((tmp_path/'artifacts/quantization').glob('attempt-*/summary.json')).read_text())
    assert report['status']=='failed' and report['costs']['teacher_forward_calls']==2
    assert report['authoritative_cost_record']['teacher_forward_calls']==2


def test_registry_reads_measured_eligibility_without_promoting_unimplemented_methods(tmp_path,monkeypatch):
    import json
    import poc.methods as methods
    path=tmp_path/'eligibility.json'
    path.write_text(json.dumps({'schema_version':1,'methods':{'Q':{'status':'eligible','reason':'measured primary512 native artifacts','evidence_digest':'a'*64,'allowed_comparisons':['native_fidelity','payload'],'supported_formats':['affine2-g64','affine4-g64'],'stage_readiness':'preliminary_frontier_32fit_32dev'}}}))
    monkeypatch.setattr(methods,'ELIGIBILITY_PATH',path)
    registry=methods.method_registry()
    assert registry['Q'].status=='eligible' and registry['Q'].supported_formats==('affine2-g64','affine4-g64')
    assert registry['M'].status=='not_yet_implemented'


def test_retry_cost_includes_attributed_supplemental_forward_counters(tmp_path):
    import json
    from poc.quantization_stage import _prior_costs
    (tmp_path/'cost-amendments.json').write_text(json.dumps({'amendments':[{'cost':{'student_forward_calls':213}},{'cost':{'student_forward_calls':15}}]}))
    assert _prior_costs(tmp_path,10.)['student_forward_calls']==228

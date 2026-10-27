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


def pilot_recorder(tmp_path, *, prior=None, limits=None):
    from dataclasses import replace
    from test_records import spec,budget
    from poc.records import start_attempt
    from poc.budget import BudgetTracker
    from poc.quantization_stage import PilotRecorder
    caps={'wall_seconds':3600,'teacher_forward_calls':100,'student_forward_calls':100,'student_backward_calls':100,'fit_updates':100,'conversion_attempts':100,'process_rss_bytes':10**12,'mlx_active_bytes':10**12,'generated_disk_bytes':10**12}
    caps.update(limits or {})
    b=replace(budget(),wall_seconds=3600,teacher_forward_calls=100,student_backward_calls=100,updates=100,memory_bytes=10**12,disk_bytes=10**12)
    identity=spec(budgets=b);attempt=start_attempt(identity,tmp_path/'runs')
    output=tmp_path/'output';output.mkdir()
    return PilotRecorder(output,{'status':'running'},{'wall_seconds':0,**(prior or {})},caps,BudgetTracker(b,attempt))


@pytest.mark.parametrize('field,increment', [('teacher_forward_calls',{'teacher_forward_calls':1}),('student_backward_calls',{'student_backward_calls':1}),('fit_updates',{'fit_updates':1}),('conversion_attempts',{'conversion_attempts':1}),('wall_seconds',{'conversion_attempts':1}),('headroom',{'teacher_forward_calls':1}),('live',{'teacher_forward_calls':1})])
def test_real_aggregate_ceiling_escapes_actual_method_failure_handler(tmp_path,monkeypatch,field,increment):
    from poc.quantization_stage import _failure_status
    from poc.budget import BudgetExceeded
    recorder=pilot_recorder(tmp_path,prior={field:101},limits={field:100})
    if field=='headroom':
        import psutil
        from types import SimpleNamespace
        monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    if field=='live': recorder.error='live resource cap exceeded'
    calls=[]
    with pytest.raises(BudgetExceeded):
        for method in ('first','must not run'):
            calls.append(method)
            try: recorder.account(**increment)
            except (ValueError,RuntimeError,ImportError) as error: _failure_status(error)
    assert calls==['first']
    assert recorder.tracker.totals.teacher_forward_calls==0


def test_actual_aggregate_overrun_is_durable_before_raising(tmp_path):
    from poc.contracts import CostRecord
    from poc.budget import BudgetExceeded,BudgetTracker
    recorder=pilot_recorder(tmp_path,limits={'teacher_forward_calls':1})
    with pytest.raises(BudgetExceeded): recorder.record(CostRecord(teacher_forward_calls=2))
    assert BudgetTracker(recorder.tracker.budget,recorder.tracker.attempt).totals.teacher_forward_calls==2


def test_student_forward_preflight_and_actual_overrun_use_shared_ledger(tmp_path):
    from poc.contracts import CostRecord
    from poc.budget import BudgetExceeded,BudgetTracker
    recorder=pilot_recorder(tmp_path,prior={'student_forward_calls':1},limits={'student_forward_calls':2})
    recorder.check(CostRecord(student_forward_calls=1));recorder.record(CostRecord(student_forward_calls=1))
    with pytest.raises(BudgetExceeded): recorder.check(CostRecord(student_forward_calls=1))
    assert recorder.tracker.totals.student_forward_calls==1
    with pytest.raises(BudgetExceeded): recorder.record(CostRecord(student_forward_calls=1))
    assert BudgetTracker(recorder.tracker.budget,recorder.tracker.attempt).totals.student_forward_calls==2


def test_reuse_rejects_valid_replacement_and_changed_evaluator_or_teacher(tmp_path):
    from pathlib import Path
    from test_export import model,runtime
    from poc.bundle import write_bundle,measure_bundle,_sha
    from poc.quantization_stage import _reusable_rows
    import shutil
    original=write_bundle(model('0'),tmp_path/'artifact',runtime(tmp_path))
    contract={'evaluator':'first','teacher':'original','windows':'fixed'}
    row={'artifact':original.directory,'manifest_sha256':original.manifest_sha256,'payload_bytes':measure_bundle(Path(original.directory)).payload_bytes,'status':'eligible','variant':'fixture','development':{'kl':0.1}}
    evidence={'evaluation_contract':contract,'rows':[row]}
    assert len(_reusable_rows(evidence,tmp_path/'previous.json',contract))==1
    assert _reusable_rows(evidence,tmp_path/'previous.json',{**contract,'evaluator':'changed'})==[]
    assert _reusable_rows(evidence,tmp_path/'previous.json',{**contract,'teacher':'changed'})==[]
    shutil.rmtree(original.directory)
    write_bundle(model('2'),Path(original.directory),runtime(tmp_path))
    with pytest.raises(ValueError,match='identity'): _reusable_rows(evidence,tmp_path/'previous.json',contract)


def test_evaluation_contract_binds_numeric_code_and_teacher_values_not_cost_logging(tmp_path):
    import shutil
    from pathlib import Path
    from test_export import model,runtime
    from poc.bundle import write_bundle
    from poc.quantization_stage import _evaluation_contract,ROOT
    source=tmp_path/'source';(source/'poc/runtime').mkdir(parents=True)
    for name in ('contracts.py','quantize.py','quantization_stage.py','runtime/mlx_decoder.py'):
        shutil.copyfile(ROOT/'poc'/name,source/'poc'/name)
    original=write_bundle(model('0'),tmp_path/'teacher',runtime(tmp_path))
    contract=_evaluation_contract(source,Path(original.directory),'fixed-windows')
    # Accounting-only source growth is explicitly outside numerical compatibility.
    p=source/'poc/contracts.py';p.write_text(p.read_text().replace('teacher_forward_calls: int = 0','teacher_forward_calls: int = 99'))
    assert _evaluation_contract(source,Path(original.directory),'fixed-windows')==contract
    p=source/'poc/quantization_stage.py';p.write_text(p.read_text().replace('range(0,batch.input_ids.shape[1]-1,32)','range(0,batch.input_ids.shape[1]-1,16)'))
    assert _evaluation_contract(source,Path(original.directory),'fixed-windows')!=contract
    # The orchestration passing teacher states into the evaluator is numerical
    # code too; definitions alone cannot bind this mutation.
    p.write_text((ROOT/'poc/quantization_stage.py').read_text().replace('teacher_hidden.append(native_teacher.final_hidden(batch))','teacher_hidden.append(native_teacher.final_hidden(batch)*0)'))
    assert _evaluation_contract(source,Path(original.directory),'fixed-windows')!=contract
    changed=model('0')
    import torch
    with torch.no_grad(): changed.norm.weight.add_(.01)
    other=write_bundle(changed,tmp_path/'other-teacher',runtime(tmp_path))
    assert _evaluation_contract(ROOT,Path(original.directory),'fixed-windows')['teacher']!=_evaluation_contract(ROOT,Path(other.directory),'fixed-windows')['teacher']


def test_legacy_known_student_spending_is_not_reset_by_new_zero_increments(tmp_path):
    import json
    from poc.quantization_stage import _prior_costs
    attempt=tmp_path/'attempt-fixture';attempt.mkdir()
    ledger=tmp_path/'costs.jsonl'
    ledger.write_text(json.dumps({'attempt_id':'a','cost':{'teacher_forward_calls':530}})+'\n'+json.dumps({'attempt_id':'a','cost':{'student_forward_calls':0,'wall_seconds':1}})+'\n')
    (attempt/'summary.json').write_text(json.dumps({'attempt_id':'a','cost_ledger':str(ledger),'costs':{'wall_seconds':1,'student_forward_calls':709}}))
    (tmp_path/'cost-amendments.json').write_text(json.dumps({'amendments':[{'cost':{'student_forward_calls':228}}]}))
    prior=_prior_costs(tmp_path,0)
    assert prior['student_forward_calls']==937 and prior['teacher_forward_calls']==530


def test_per_run_prior_cost_is_not_double_counted_on_same_identity_retry(tmp_path):
    from poc.contracts import CostRecord
    from poc.quantization_stage import PilotRecorder
    first=pilot_recorder(tmp_path,limits={'student_forward_calls':2})
    first.record(CostRecord(student_forward_calls=1))
    retry=PilotRecorder(first.out,{'status':'running'},{'wall_seconds':0,'student_forward_calls':1},first.caps,first.tracker)
    retry.check(CostRecord(student_forward_calls=1));retry.record(CostRecord(student_forward_calls=1))
    assert retry.tracker.totals.student_forward_calls==2
    assert retry.report['costs']['student_forward_calls']==1

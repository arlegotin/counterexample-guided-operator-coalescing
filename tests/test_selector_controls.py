"""Owned actual-pipeline sensitivity; no pretrained inference."""
import importlib,importlib.util
import pytest
from pathlib import Path


def api():
    assert importlib.util.find_spec('poc.selector_controls'),'owned end-to-end selector controls missing'
    return importlib.import_module('poc.selector_controls')


@pytest.fixture(autouse=True)
def explicit_integrated_source_binding(tmp_path_factory,monkeypatch):
    # This checkout composes independently reviewed features; opt in explicitly.
    helper=Path(__file__).with_name('test_selector_source_binding.py')
    spec=importlib.util.spec_from_file_location('owned_source_binding_fixture',helper)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    binding=loaded.make_binding(tmp_path_factory.mktemp("owned-selector-bindings"))
    module=api();original=module.run_owned_selector_control
    def bound(*args,**kwargs):
        kwargs.setdefault('source_binding',binding)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'run_owned_selector_control',bound)


@pytest.mark.parametrize('seed',[17,23,31])
def test_generated_counterexample_reverses_natural_decision_toward_heldout_damage(tmp_path,seed):
    result=api().run_owned_selector_control(tmp_path,regime='hidden_failure',seed=seed)
    assert result['scope']=='owned pipeline sensitivity; no pretrained or compression inference'
    assert result['natural_choice']!=result['damage_choice']==result['active_choice']
    assert result['raw_component_metrics'] and result['probe_metrics']
    assert result['proposal_counts']['rejected']>0
    assert result['proposal_counts']['accepted']>0
    assert result['cost']['teacher_forward_calls']>0 and result['cost']['student_forward_calls']>0
    assert result['cost']['search_proposals']>0
    assert result['unchanged_model_digests']
    assert not set(result['score_components']) & set(result['damage_components'])
    assert Path(result['source_exposure_journal']).is_file()


def test_trigger_disabled_and_no_information_controls_cannot_claim_benefit(tmp_path):
    disabled=api().run_owned_selector_control(tmp_path/'disabled',regime='hidden_failure',seed=17,allow_trigger=False)
    assert disabled['active_choice']==disabled['natural_choice']!=disabled['damage_choice']
    negative=api().run_owned_selector_control(tmp_path/'negative',regime='no_added_information',seed=17)
    assert negative['natural_choice']==negative['active_choice']==negative['damage_choice']
    assert not negative['selector_improved']


def test_generated_rejections_never_execute_or_enter_selected_set(tmp_path):
    result=api().run_owned_selector_control(tmp_path,regime='hidden_failure',seed=17)
    for pool in result['pools']:
        rejected={r['probe_id'] for r in pool['records'] if r['status']=='rejected'}
        assert rejected and not rejected & set(pool['selected_ids'])
        assert all(r['rejection_reason']=='token_length_mismatch' for r in pool['records'] if r['status']=='rejected')
        scored={row['probe']['probe_id'] for row in result['probe_metrics'] if row['candidate']==pool['candidate'] and row['mode']==pool['mode']}
        assert not rejected & scored
    assert result['population_manifest_sha256'].keys()=={'fit','score','damage'}
    assert result['executed_source_files']['poc/selector_controls.py']==result['source_sha256']


def test_output_reuse_is_denied_before_model_or_text_io(tmp_path,monkeypatch):
    module=api();(tmp_path/'old-result').write_text('keep')
    def forbidden(*args,**kwargs):raise AssertionError('allocated model before output denial')
    monkeypatch.setattr(module,'planted_models',forbidden)
    with pytest.raises(FileExistsError):module.run_owned_selector_control(tmp_path,regime='hidden_failure',seed=17)
    assert (tmp_path/'old-result').read_text()=='keep'


def test_owned_suite_has_fixed_positive_negative_and_disabled_cases(tmp_path):
    result=api().run_owned_selector_suite(tmp_path)
    assert result['status']=='owned_controls_passed'
    assert len(result['cases'])==5
    assert result['pretrained_model_calls']==0
    assert result['scope']=='owned pipeline sensitivity; no pretrained or compression inference'
    for case in result['cases'].values():
        assert case['path'] and len(case['sha256'])==64


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_owned_metrics_reject_nonfinite_or_below_qualified_negative_floor(value):
    from types import SimpleNamespace
    with pytest.raises(ValueError):api().qualified_metric(SimpleNamespace(kl=value,nll=3.,valid_targets=8))


def test_owned_metrics_retain_small_negative_raw_estimate():
    from types import SimpleNamespace
    metric=SimpleNamespace(kl=-1e-12,nll=3.,valid_targets=8)
    assert api().qualified_metric(metric) is metric and metric.kl==-1e-12


@pytest.mark.parametrize('original_failure',[False,True])
def test_budget_failure_persists_terminal_status_and_preserves_original_exception(tmp_path,monkeypatch,original_failure):
    from dataclasses import replace
    import json
    module=api();original_budget=module.Budget
    monkeypatch.setattr(module,'Budget',lambda *a,**k:replace(original_budget(*a,**k),wall_seconds=0))
    if original_failure:
        def fail(*args,**kwargs):raise RuntimeError('owned construction failure')
        monkeypatch.setattr(module,'planted_models',fail)
        expected=RuntimeError
    else:
        from poc.budget import BudgetExceeded
        expected=BudgetExceeded
    with pytest.raises(expected):module.run_owned_selector_control(tmp_path,regime='hidden_failure',seed=17)
    attempts=list(tmp_path.glob('ledgers/*/attempts/*/run.json'))
    assert len(attempts)==1
    record=json.loads(attempts[0].read_text())
    assert record['status']=='failed'
    events=[json.loads(line) for line in (attempts[0].parent/'events.jsonl').read_text().splitlines()]
    assert events[-1]['kind']=='attempt_finished' and events[-1]['status']=='failed'
    if original_failure:assert 'owned construction failure' in record['failure_details']
    ledger=next(tmp_path.glob('ledgers/*/costs.jsonl'))
    assert sum(json.loads(line)['cost']['wall_seconds'] for line in ledger.read_text().splitlines())>0

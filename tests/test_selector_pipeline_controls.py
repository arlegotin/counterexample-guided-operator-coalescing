"""Owned sensitivity controls must exercise generation through final decisions."""
import importlib
import importlib.util


def run_controls(tmp_path):
    assert importlib.util.find_spec('provenance.selector_pipeline_controls') is not None, 'end-to-end selector controls missing'
    return importlib.import_module('provenance.selector_pipeline_controls').run_owned_controls(tmp_path)


def test_generated_hidden_failure_reverses_decision_and_improves_independent_damage(tmp_path):
    result = run_controls(tmp_path)
    positive = result['ranking_reversal']
    assert positive['natural_choice'] != positive['active_choice']
    assert positive['natural_choice'] == 'hidden_failure'
    assert positive['active_choice'] == 'small_uniform_error'
    assert positive['damage'][positive['active_choice']] < positive['damage'][positive['natural_choice']]
    assert positive['selected_edit_kind']['hidden_failure'] == 'name'
    assert positive['active_probe_scores']['hidden_failure'] > 0
    assert set(positive['score_components']).isdisjoint(positive['damage_components'])


def test_no_added_information_control_and_actual_alignment_rejections_are_retained(tmp_path):
    result = run_controls(tmp_path)
    null = result['no_added_information']
    assert null['natural_choice'] == null['active_choice'] == 'hidden_failure'
    assert null['damage']['hidden_failure'] == 0
    positive = result['ranking_reversal']
    assert positive['accepted_proposals'] >= 2
    assert positive['rejected_proposals'] >= 1
    assert 'token_length_mismatch' in positive['rejection_reasons']
    assert positive['incompatible_boundary_rejected'] is True
    assert positive['cost']['search_proposals'] > positive['accepted_proposals']
    assert positive['cost']['teacher_forward_calls'] > 0
    assert positive['cost']['student_forward_calls'] > 0
    assert result['scope'] == 'owned sensitivity controls; no pretrained selection result'

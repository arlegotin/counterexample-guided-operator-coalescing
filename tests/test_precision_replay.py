"""Exact old states, unchanged original caps, and one fixed fresh-component repeat."""
import importlib,importlib.util
import pytest


def api():
    assert importlib.util.find_spec('provenance.measure_single_merge_precision'), 'charged precision/replay driver missing'
    return importlib.import_module('provenance.measure_single_merge_precision')


def test_replay_requires_both_exact_digests_without_tolerance_or_retry():
    m=api();expected={'initialized_digest':'a'*64,'fitted_digest':'b'*64}
    m.require_replay_identity(expected,'a'*64,'b'*64)
    for initial,fitted in [('c'*64,'b'*64),('a'*64,'c'*64)]:
        with pytest.raises(ValueError,match='exact'):m.require_replay_identity(expected,initial,fitted)


def test_phase_caps_include_replay_and_original_spending():
    m=api();caps={'wall_seconds':3600,'teacher_forward_calls':160000,'student_forward_calls':24000,'updates':18,'valid_target_presentations':147456,'search_proposals':2000,'search_seconds':300}
    m.enforce_phase_caps({k:0 for k in caps},caps)
    for key,value in caps.items():
        with pytest.raises(ValueError,match=key):m.enforce_phase_caps({**{k:0 for k in caps},key:value+1},caps)


def test_logical_position_bound_includes_activation_and_entropy_not_training():
    m=api()
    assert m.logical_position_count(score_documents=16,damage_documents=32,depths=2,donor_target_counts=[511]*128)==122768
    assert 20*122768<2500000

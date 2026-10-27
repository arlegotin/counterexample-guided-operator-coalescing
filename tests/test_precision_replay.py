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


def test_precision_analysis_preserves_fresh_and_pooled_populations(tmp_path,monkeypatch):
    import json
    m=api();out=tmp_path/'repeat';old=tmp_path/'old';(out/'trials').mkdir(parents=True);(old/'trials-final').mkdir(parents=True)
    monkeypatch.setattr(m,'OUT',out);monkeypatch.setattr(m,'ORIGINAL',old);monkeypatch.setattr(m,'ROOT',tmp_path)
    pairs=[(0,i) for i in range(1,21)];methods=('natural_kl','active_1','active_01','random_donor','activation_mse','cka','weight_distance')
    for index,pair in enumerate(pairs):
        identity={'pair':pair,'initialized_digest':f'{index:064x}','fitted_digest':f'{index+20:064x}'}
        for root,score_count,damage_count,prefix in [(out,16,32,'fresh'),(old,8,16,'old')]:
            row={'identity':identity,'scores':{k:{f'{prefix}-s-{c}':float(index+c/100) for c in range(score_count)} for k in methods},'natural_damage':{f'{prefix}-d-{c}':float(index+c/100) for c in range(damage_count)}}
            folder='trials' if root==out else 'trials-final';(root/folder/f'{pair[0]:02d}-{pair[1]:02d}.json').write_text(json.dumps(row))
    m.analyze({'original18':pairs[:18],'methods':methods})
    result=json.loads((out/'analysis.json').read_text())
    assert result['fresh_repeat']['score_components']==16
    assert result['fresh_repeat']['damage_components']==32
    assert len(result['fresh_original18_secondary']['pairs'])==18
    assert result['pooled_exploratory_secondary']['score_components']==24
    assert result['pooled_exploratory_secondary']['damage_components']==48
    assert result['original_result_remains_separate']
    changed=old/'trials-final/00-01.json';row=json.loads(changed.read_text());row['identity']['fitted_digest']='f'*64;changed.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='states differ'):m.analyze({'original18':pairs[:18],'methods':methods})


def test_precision_run_rejects_budget_drift_before_starting_attempt():
    from types import SimpleNamespace
    m=api();config=SimpleNamespace(budgets=SimpleNamespace(resolve=lambda:SimpleNamespace(to_dict=lambda:{'updates':41})))
    with pytest.raises(ValueError,match='frozen overall budgets'):
        m._start('replay_first',{'overall_budgets':{'updates':40}},config)


def test_saved_eligibility_recovery_keeps_rejections_and_fixed_order(tmp_path):
    import json
    m=api();entries=[{'document_id':str(i),'component_id':str(i),'role':'probe_score'} for i in range(3)]
    saved=tmp_path/'saved';saved.mkdir()
    for i,entry in enumerate(entries):
        (saved/f'{i}.json').write_text(json.dumps({'entry':entry,'check':{'eligible':i!=0,'document_id':str(i)},'rows':[{'status':'rejected' if i==0 else 'accepted','document_id':str(i)}]}))
    checks,selected=m.reuse_eligibility_pool(entries,saved,tmp_path/'pool.json',wanted=2)
    assert selected==entries[1:]
    assert len(checks)==3 and not checks[0]['eligible']
    assert [r['document_id'] for r in json.loads((tmp_path/'pool.json').read_text())]==['0','1','2']
    (saved/'1.json').unlink()
    with pytest.raises(ValueError,match='missing frozen candidate'):
        m.reuse_eligibility_pool(entries,saved,tmp_path/'missing.json',wanted=2)

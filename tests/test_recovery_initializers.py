"""Preserve parent choice, complete-bank scope and unrelated state during WP2 initialization."""
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from poc.bank import make_bank,canonicalize_partition
from poc.data.manifest import DataManifest,DocumentRecord
from poc.merge import initialize_merge
from test_bank import tiny
from test_model_parity import batch


def api():
    assert importlib.util.find_spec('poc.recovery_initializers') is not None, 'recovery initializer integration missing'
    return importlib.import_module('poc.recovery_initializers')


def fixture():
    teacher=tiny();teacher.tokenizer_digest='c'*64
    current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
    fit=DataManifest((DocumentRecord('d0','owned','v1','fixture',0),),{'d0':'fit'},{'d0':'c0'},'owned')
    b=batch();trial=initialize_merge(current,teacher,(0,2),fit,17,batches=(b,))
    return teacher,trial,fit,(b,)


def test_weighted_reinitialization_preserves_bank_and_unrelated_owners():
    teacher,trial,fit,batches=fixture();model=trial.model
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    spec=api().RecoveryInitializerSpec(method='weighted_delta_svd',calibration_components=1)
    report=api().initialize_recovery_corrections(model,teacher,(0,2),fit,batches,17,spec)
    assert report['spec_sha256']==spec.digest()
    assert report['before_digest']==trial.initialized_digest
    assert report['after_digest']==model.manifest().checkpoint_digest
    assert len(report['projection_records'])==6
    for n,p in model.named_parameters():
        if not n.startswith(('layers.0.mlp.residuals.','layers.2.mlp.residuals.')):
            torch.testing.assert_close(p,before[n],atol=0,rtol=0)
    down=[r for r in report['projection_records'] if r['projection']=='down_proj']
    assert all(r['input_width']==teacher.config.intermediate_size for r in down)
    assert all(r['valid_input_rows']==4 for r in report['projection_records'])
    assert all(p.grad is None for p in teacher.parameters())


def test_zero_product_is_live_at_both_uses_and_parent_reference_is_unchanged():
    teacher,trial,fit,batches=fixture();a=api()
    ref=a.initialize_recovery_corrections(trial.model,teacher,(0,2),fit,batches,17,
                        a.RecoveryInitializerSpec(method='parent_svd',calibration_components=1))
    assert ref['before_digest']==ref['after_digest']==trial.initialized_digest
    report=a.initialize_recovery_corrections(trial.model,teacher,(0,2),fit,batches,17,
                        a.RecoveryInitializerSpec(method='zero_product',calibration_components=1))
    assert report['after_digest']!=report['before_digest']
    for depth in (0,2):
        for factor in trial.model.layers[depth].mlp.residuals.values():
            assert factor.A.abs().max()>0 and torch.count_nonzero(factor.B)==0
            factor.B.grad=None
            factor(torch.ones(1,factor.A.shape[1],dtype=factor.A.dtype)).sum().backward()
            assert factor.B.grad.abs().max()>0


@pytest.mark.parametrize('bad', ['role','partial_bank','component','empty_mask'])
def test_invalid_calibration_rejected_before_model_mutation(bad):
    teacher,trial,fit,batches=fixture();a=api();depths=(0,2)
    if bad=='role':fit=replace(fit,roles={'d0':'dev'})
    if bad=='partial_bank':depths=(0,)
    if bad=='component':batches=(replace(batches[0],cluster_ids=('wrong',)),)
    if bad=='empty_mask':batches=(replace(batches[0],target_mask=torch.zeros_like(batches[0].target_mask)),)
    before=trial.model.manifest().checkpoint_digest
    with pytest.raises((ValueError,PermissionError)):
        a.initialize_recovery_corrections(trial.model,teacher,depths,fit,batches,17,
                                         a.RecoveryInitializerSpec(method='weighted_delta_svd',calibration_components=1))
    assert trial.model.manifest().checkpoint_digest==before


def test_spec_rejects_unknown_methods_and_content_hashes_choices():
    a=api()
    with pytest.raises(ValueError):a.RecoveryInitializerSpec(method='nonlinear_warmstart')
    with pytest.raises(ValueError):a.RecoveryInitializerSpec(calibration_components=0)
    x=a.RecoveryInitializerSpec(method='parent_svd')
    assert x.digest()!=replace(x,method='zero_product').digest()


def test_version_three_config_names_initializer_and_keeps_old_schema_closed(tmp_path):
    from dataclasses import asdict
    from pathlib import Path
    import yaml
    from poc.recovery_study import load_study_config
    raw=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/recovery-diagnosis-v2.yaml').read_text())
    raw.update(schema_version=3,study='owned-initializer-v3',out='artifacts/owned-initializer-v3',
               pair=[12,20],rank=16,loss_scopes=['full_student'],
               initializer=asdict(api().RecoveryInitializerSpec(method='weighted_delta_svd')))
    path=tmp_path/'v3.yaml';path.write_text(yaml.safe_dump(raw))
    config=load_study_config(path)
    assert config['initializer']['method']=='weighted_delta_svd'
    assert config['pair']==[12,20] and config['rank']==16
    raw['schema_version']=2;path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):load_study_config(path)
    raw['schema_version']=3;raw['initializer']['unpriced_extra_steps']=10;path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):load_study_config(path)


def test_driver_initialization_records_variant_after_one_parent_choice():
    from dataclasses import asdict
    from provenance.recovery_diagnosis import initialize
    teacher,trial,fit,batches=fixture()
    config={'schema_version':3,'pair':[0,2],'rank':2,'parent_fit_components':1,
            'initializer':asdict(api().RecoveryInitializerSpec(method='zero_product',calibration_components=1))}
    result=initialize(config,teacher,fit,batches,17,None)
    assert len(result.medoid_scores)==2
    assert result.selected_parent==trial.selected_parent
    assert result.initializer_record['spec']['method']=='zero_product'
    assert result.initializer_record['before_digest']==trial.initialized_digest
    assert result.initialized_digest==result.model.manifest().checkpoint_digest


def test_runspec_rejects_stale_initializer_before_optimization(tmp_path,monkeypatch):
    from dataclasses import asdict
    import json
    from pathlib import Path
    import yaml
    from provenance import recovery_diagnosis as driver
    from poc.recovery_study import NaturalWindowPlan
    from test_training import budget
    teacher,trial,fit,batches=fixture();a=api()
    spec=a.RecoveryInitializerSpec(method='zero_product',calibration_components=1)
    report=a.initialize_recovery_corrections(trial.model,teacher,(0,2),fit,batches,17,spec)
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'configs/recovery-diagnosis-v2.yaml').read_text())
    cfg.update(schema_version=3,study='owned-v3',out='artifacts/owned-v3',initializer=asdict(spec),pair=[0,2],rank=2,updates=1,observations=[0,1])
    cfg['populations']['fit']['sha256']='a'*64;cfg['populations']['tuning']['sha256']='b'*64
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    out=tmp_path/'artifacts/owned-v3';out.mkdir(parents=True);(out/'controls.json').write_text('{}')
    plan=NaturalWindowPlan.build(batches,updates=1,seed=17,manifest_hashes={'fit':'a'*64})
    args=(cfg,'full_student',17,plan,trial.model,{'commit':'a'*40},{},budget())
    training,_=driver.make_spec(*args,initializer_record=report)
    assert training.run_spec.input_manifests['initializer_spec']==spec.digest()
    for change in ({'after_digest':'0'*64},{'spec_sha256':'0'*64},{'seed':23}):
        with pytest.raises(ValueError,match='initializer'):
            driver.make_spec(*args,initializer_record={**report,**change})


def test_driver_rejects_quarantine_before_any_teacher_query(monkeypatch):
    from dataclasses import asdict
    from provenance.recovery_diagnosis import initialize
    teacher,trial,fit,batches=fixture()
    fit=replace(fit,quarantine={'d0':'excluded owned calibration'})
    def forbidden(*args,**kwargs):raise AssertionError('quarantined document reached teacher')
    monkeypatch.setattr(teacher,'prefix_boundary',forbidden)
    config={'schema_version':3,'pair':[0,2],'rank':2,'parent_fit_components':1,
            'initializer':asdict(api().RecoveryInitializerSpec(method='weighted_delta_svd',calibration_components=1))}
    with pytest.raises(ValueError,match='quarantin'):
        initialize(config,teacher,fit,batches,17,None)


def test_calibration_inputs_match_actual_teacher_ffn_on_fp64_reference():
    from poc.initializers import _digest_tensor
    teacher,trial,fit,batches=fixture();b=batches[0]
    captured=[]
    handle=teacher.layers[2].mlp.gate_proj.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    with torch.no_grad():teacher.ff_residual(teacher.prefix_boundary(b,2))
    handle.remove()
    expected=captured[0][:,:-1][b.target_mask]
    report=api().initialize_recovery_corrections(trial.model,teacher,(0,2),fit,batches,17,
             api().RecoveryInitializerSpec(method='weighted_delta_svd',calibration_components=1))
    row=next(r for r in report['projection_records'] if r['depth']==2 and r['projection']=='gate_proj')
    assert row['calibration']['documents'][0]['input_sha256']==_digest_tensor(expected.double())


def test_version_three_load_rejects_quarantine_before_content_or_assets(tmp_path,monkeypatch):
    from dataclasses import asdict
    import hashlib,json
    from provenance.recovery_diagnosis import load_inputs
    from poc import teacher as teacher_module
    from poc import recovery_population
    teacher,trial,fit,batches=fixture()
    fit=replace(fit,quarantine={'d0':'excluded before reading'},exposure_path=str(tmp_path/'exposure.jsonl'))
    path=tmp_path/'fit.json';payload=json.dumps(asdict(fit)).encode();path.write_bytes(payload)
    def forbidden(*args,**kwargs):raise AssertionError('content/assets opened before quarantine check')
    monkeypatch.setattr(teacher_module.AssetManifest,'load',forbidden)
    monkeypatch.setattr(recovery_population,'load_natural_population',forbidden)
    cfg={'schema_version':3,'asset_root':'/owned-do-not-open','populations':{'fit':{'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()}}}
    with pytest.raises(ValueError,match='quarantin'):load_inputs(cfg)
    assert not (tmp_path/'exposure.jsonl').exists()

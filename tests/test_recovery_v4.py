"""Schema 4 explicitly bounds profiling and checkpoint-clone provenance."""
from pathlib import Path

import pytest
import yaml

from poc.recovery_study import load_study_config


@pytest.fixture
def config():
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/initializer-recovery-v3.yaml').read_text())
    raw.update(schema_version=4, profile_resource_budget={
        'updates': 1, 'valid_target_presentations': 106368,
        'teacher_forward_calls': 10000, 'suffix_backward_calls': 0,
        'student_backward_calls': 10000, 'search_proposals': 0,
        'search_seconds': 0, 'wall_seconds': 3600,
        'disk_bytes': 2**30, 'memory_bytes': 2**30,
    }, initialization_checkpoint=None)
    return raw


def load(tmp_path, config):
    path = tmp_path / 'study.yaml'
    path.write_text(yaml.safe_dump(config))
    return load_study_config(path)


def checkpoint():
    return dict(checkpoint_directory='/owned/checkpoint', checkpoint_manifest_sha256='a'*64,
                run_spec_path='/owned/run-spec.json', run_spec_sha256='b'*64,
                initialization_record_path='/owned/initialization.json', initialization_record_sha256='c'*64)


def test_schema_four_accepts_explicit_finite_profile_and_preserves_payload(tmp_path, config):
    assert load(tmp_path, config) == config


@pytest.mark.parametrize('scope,minimum', [('local_sum',106368), ('local_mean',106368), ('full_student',98176)])
def test_profile_budget_counts_local_depths_and_observations(tmp_path, config, scope, minimum):
    config['loss_scopes'] = [scope]
    config['profile_resource_budget']['valid_target_presentations'] = minimum
    assert load(tmp_path, config)['profile_resource_budget']['valid_target_presentations'] == minimum
    config['profile_resource_budget']['valid_target_presentations'] -= 1
    with pytest.raises(ValueError, match='profile.*budget|budget.*profile'):
        load(tmp_path, config)


@pytest.mark.parametrize('name', ['profile_resource_budget', 'initialization_checkpoint'])
def test_schema_four_requires_new_fields(tmp_path, config, name):
    del config[name]
    with pytest.raises(ValueError, match='missing|unknown'):
        load(tmp_path, config)


@pytest.mark.parametrize('field,value', [
    ('updates', 0), ('updates', 2), ('updates', True),
    ('wall_seconds', 0), ('wall_seconds', float('inf')), ('wall_seconds', float('nan')),
    ('memory_bytes', 0), ('disk_bytes', 0), ('disk_bytes', 1.5),
    ('teacher_forward_calls', -1), ('suffix_backward_calls', True),
    ('student_backward_calls', -1), ('search_proposals', 0.5),
    ('search_seconds', -1), ('valid_target_presentations', True),
])
def test_schema_four_validates_every_profile_budget_dimension(tmp_path, config, field, value):
    config['profile_resource_budget'][field] = value
    with pytest.raises(ValueError):
        load(tmp_path, config)


@pytest.mark.parametrize('mutation', ['missing', 'unknown', 'null', 'not_mapping'])
def test_profile_budget_is_a_closed_complete_budget(tmp_path, config, mutation):
    if mutation == 'missing':
        del config['profile_resource_budget']['teacher_forward_calls']
    elif mutation == 'unknown':
        config['profile_resource_budget']['extra'] = 1
    else:
        config['profile_resource_budget'] = None if mutation == 'null' else []
    with pytest.raises(ValueError, match='profile.*budget|budget.*profile'):
        load(tmp_path, config)


def test_schema_four_clone_accepts_structural_identity_without_reading_artifacts(tmp_path, config):
    config.update(initialization_checkpoint=checkpoint(), seeds=[17])
    config['initializer']['method'] = 'parent_svd'
    assert load(tmp_path, config)['initialization_checkpoint'] == checkpoint()


@pytest.mark.parametrize('mutation', ['initializer', 'seeds', 'missing', 'unknown', 'not_mapping'])
def test_checkpoint_clone_is_closed_single_seed_parent_svd(tmp_path, config, mutation):
    config.update(initialization_checkpoint=checkpoint(), seeds=[17])
    config['initializer']['method'] = 'parent_svd'
    if mutation == 'initializer':
        config['initializer']['method'] = 'weighted_delta_svd'
    elif mutation == 'seeds':
        config['seeds'] = [17,23]
    elif mutation == 'missing':
        del config['initialization_checkpoint']['run_spec_path']
    elif mutation == 'unknown':
        config['initialization_checkpoint']['extra'] = True
    else:
        config['initialization_checkpoint'] = []
    with pytest.raises(ValueError, match='checkpoint|clone'):
        load(tmp_path, config)


@pytest.mark.parametrize('field,value', [
    (field,value) for field in checkpoint()
    for value in ('', ' ', None, 17, *(['not-a-hash', 'a'*63, 'g'*64] if field.endswith('_sha256') else []))
])
def test_checkpoint_identity_requires_nonempty_paths_and_sha256(tmp_path, config, field, value):
    config.update(initialization_checkpoint=checkpoint(), seeds=[17])
    config['initializer']['method'] = 'parent_svd'
    config['initialization_checkpoint'][field] = value
    with pytest.raises(ValueError, match='checkpoint|clone'):
        load(tmp_path, config)


@pytest.mark.parametrize('version', [1,2,3])
@pytest.mark.parametrize('new_field', ['profile_resource_budget', 'initialization_checkpoint'])
def test_legacy_schemas_reject_schema_four_fields(tmp_path, version, new_field):
    name = {1:'recovery-diagnosis-v1.yaml',2:'recovery-diagnosis-v2.yaml',3:'initializer-recovery-v3.yaml'}[version]
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs' / name).read_text())
    assert load(tmp_path, raw) == raw
    raw[new_field] = None
    with pytest.raises(ValueError, match='unknown'):
        load(tmp_path, raw)


@pytest.mark.parametrize('field,value', [('extra',1), ('pair',[15,0]), ('rank',7),
    ('loss_scopes',['other']), ('observation_policy','other'), ('parent_fit_components',513)])
def test_schema_four_keeps_schema_three_constraints(tmp_path, config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        load(tmp_path, config)


def test_schema_four_accepts_closed_explicit_new_horizon_binding(tmp_path,config):
    config.update(initialization_checkpoint={**checkpoint(),'schema_version':2,'source_updates':256,
        'source_populations':{role:{'path':'/owned/'+role+'.json','sha256':'d'*64} for role in ('fit','tuning')}},seeds=[17],updates=1024,
        observations=[0,1,4,16,32,64,128,256,512,1024])
    config['initializer']['method']='parent_svd'
    assert load(tmp_path,config)==config
    config['initialization_checkpoint']['source_populations']['fit']['unpriced']=True
    with pytest.raises(ValueError,match='population'):
        load(tmp_path,config)

"""Strict continuation configuration and resolved source identity, without assets."""
from copy import deepcopy
import hashlib
import importlib.metadata
from pathlib import Path
import re
import subprocess

import yaml

from poc.contracts import Budget, RunSpec
from poc.records import canonical_json
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.train import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
AUDITED = '162579679f8baa47d2cb4445d3d82f396e685baa'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keys(value, keys, name):
    if type(value) is not dict or set(value)!=set(keys):raise ValueError('invalid '+name+' fields')


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA256 required')


def load_config(path):
    class UniqueLoader(yaml.SafeLoader):pass
    def mapping(loader,node):
        pairs=loader.construct_pairs(node,deep=True);result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate config key')
            result[k]=v
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    return yaml.load(Path(path).read_text(), Loader=UniqueLoader)


def resolve_config(config):
    cfg=deepcopy(config)
    _keys(cfg,('schema_version','protocol','output_schema','audited_commit','seed','teacher',
        'numerical','pilot','training','data','gates','budgets','owned_caps'),'configuration')
    if cfg['schema_version']!=1 or cfg['output_schema']!='cgoc-continuation-v1' or cfg['audited_commit']!=AUDITED:
        raise ValueError('unsupported continuation version/base')
    if type(cfg['seed']) is not int or not 0<=cfg['seed']<2**63:raise ValueError('invalid seed')
    if cfg['protocol']!='protocol/continuation-v1.md':raise ValueError('unsupported protocol')
    _keys(cfg['teacher'],('repository','revision','trust_remote_code','asset_manifest_sha256','raw_weights_sha256'),'teacher')
    if (cfg['teacher']['repository']!='HuggingFaceTB/SmolLM2-135M'
            or cfg['teacher']['revision']!='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
            or cfg['teacher']['trust_remote_code'] is not False):raise ValueError('unqualified teacher source')
    for key in ('asset_manifest_sha256','raw_weights_sha256'):_hash(cfg['teacher'][key])
    if (cfg['teacher']['asset_manifest_sha256']!='9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083'
            or cfg['teacher']['raw_weights_sha256']!='80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'):
        raise ValueError('teacher identity differs from pinned source')
    n=cfg['numerical']
    _keys(n,('device','dtype','execution_profile','equivalence_atol','kl_floor_per_target','scientific_qualification'),'numerical')
    if (n['execution_profile']!=EXECUTION_PROFILE or n['device']!='cpu' or n['dtype']!='float64'
            or n['equivalence_atol']!=1e-12 or n['kl_floor_per_target']!=0):
        raise ValueError('unqualified numerical execution profile')
    p=cfg['pilot']
    _keys(p,('pairs','rank','arms','parent_policy','practical_margin'),'pilot')
    if (p['pairs']!=[[0,15]] or p['rank']!=8 or p['parent_policy']!='fit_only_two_parent_mse_once'
            or p['practical_margin'] is not None):raise ValueError('amend frozen pilot protocol explicitly')
    expected=[dict(id='A',initialization='parent_svd',mixture={'natural':.5,'student':.25,'donor':.25}),
              dict(id='B',initialization='parent_svd',mixture={'natural':1.}),
              dict(id='C',initialization='parent_zero_product',mixture={'natural':1.})]
    if p['arms']!=expected:raise ValueError('amend frozen pilot arms explicitly')
    t=cfg['training']
    _keys(t,('max_updates','epochs','target_count','chunk_size','observation_steps','step_zero','optimizer','scheduler'),'training')
    if (t['max_updates']!=16 or t['epochs']!=1 or t['observation_steps']!=[1,4,8,16]
            or t['step_zero']!='separately_charged_before_fit'
            or t['scheduler']!='existing_warmup_cosine_continuous'
            or t['optimizer']!=dict(name='AdamW',betas=[.9,.95],bank_lr=1e-4,residual_lr=3e-4,weight_decay=0)):
        raise ValueError('amend frozen horizon/optimizer explicitly')
    TrainingConfig(max_updates=t['max_updates'],epochs=t['epochs'],target_count=t['target_count'],
        chunk_size=t['chunk_size'],observation_steps=tuple(t['observation_steps']))
    _keys(cfg['data'],('fit','fit_calibration','pilot_validation','exposure_lineage','ordered_target_plan'),'data')
    _keys(cfg['gates'],('archive','mechanism_controls','acquisition_controls','teacher_assets','tokenizer',
        'lineage_review','numerical_floor','resource_forecast','native_qualification','pilot_executor_review'),'gates')
    refs=[*cfg['data'].values(),*cfg['gates'].values(),n['scientific_qualification']]
    for ref in refs:
        if ref is not None:
            _keys(ref,('path','sha256'),'metadata reference');_hash(ref['sha256'])
            if not isinstance(ref['path'],str) or not ref['path']:raise ValueError('metadata path required')
    Budget(**cfg['budgets']);Budget(**cfg['owned_caps'])
    # Explicit census of continuation sources, including untracked new modules.
    paths=sorted([*ROOT.glob('poc/**/*.py'),*ROOT.glob('tests/**/*.py'),*ROOT.glob('provenance/continuation*.py'),
        ROOT/cfg['protocol'],ROOT/'configs/continuation-recovery-v1.yaml',ROOT/'pyproject.toml'])
    files={str(path.relative_to(ROOT)):sha256(path) for path in paths}
    cfg['resolved']=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files=files,source_sha256=hashlib.sha256(canonical_json(files)).hexdigest(),
        protocol_sha256=files[cfg['protocol']],config_sha256=hashlib.sha256(canonical_json(config)).hexdigest())
    return cfg


def make_spec(resolved, *, mode, environment_sha256, plan_sha256=None, budget=None):
    _hash(environment_sha256)
    if plan_sha256 is not None:_hash(plan_sha256)
    sources=resolved['resolved']
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    return RunSpec(method_id='M',method_variant='continuation-v1-'+mode,
        resolved_config={'continuation':resolved,'ordered_target_plan_sha256':plan_sha256,
            'execution_profile':EXECUTION_PROFILE},source_commit=sources['source_commit'],
        dirty_patch_sha256=hashlib.sha256(patch).hexdigest() if patch else None,
        dependency_lock_sha256=hashlib.sha256(installed_requirements().encode()).hexdigest(),
        input_manifests={'source':sources['source_sha256'],'protocol':sources['protocol_sha256'],
            'config':sources['config_sha256'],'environment':environment_sha256,
            **({'target_plan':plan_sha256} if plan_sha256 else {})},
        hardware_policy=resolved['numerical'],seeds={'fixture':resolved['seed']},parent_run_ids=(),
        budgets=budget or Budget(**resolved['owned_caps']))


def installed_requirements():
    """Version snapshot, explicitly weaker than a hash-checked wheel rebuild."""
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    return ''.join(f'{name}=={packages[name]}\n' for name in sorted(packages,key=str.lower))


def pilot_blockers(resolved):
    """No artifact/corpus opens. Metadata review and execution qualification are separate gates."""
    missing=['data.'+k for k,v in resolved['data'].items() if v is None]
    missing+=['gates.'+k for k,v in resolved['gates'].items() if v is None]
    if resolved['numerical']['scientific_qualification'] is None:missing.append('numerical.scientific_qualification')
    # A hash reference alone is not proof of a reviewed native executor/lineage.
    # WP1 provides a fixture executor; the scientific adapter is deferred to WP2.
    missing.append('WP2 reviewed scientific executor and semantic verification of all receipts')
    return missing

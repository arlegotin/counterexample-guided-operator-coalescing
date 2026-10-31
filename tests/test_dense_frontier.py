"""Owned dense-baseline contracts; no MLX import, native or corpus I/O."""
from dataclasses import FrozenInstanceError,replace
from pathlib import Path
import sys
import pytest


def test_fixed_six_recipes_have_explicit_calibration_and_completed_updates():
    from poc.dense_frontier import dense_recipes
    recipes=dense_recipes()
    assert len(recipes)==6 and len({r.label for r in recipes})==6
    assert {r.method for r in recipes}=={'gptq','awq','dwq'}
    for r in recipes:
        assert r.calibration_components==64 and r.bits in (2,4) and r.embedding_bits==4
        assert r.group_size==64 and r.seed==17 and r.context==512
        assert r.internal_components==(16 if r.method=='dwq' else 0)
        assert r.completed_updates==(64 if r.method=='dwq' else 0)
        assert r.calibration_targets==64*511
        assert r.internal_target_inventory==(16*511 if r.method=='dwq' else 0)
        with pytest.raises(FrozenInstanceError):r.bits=8


@pytest.mark.parametrize('change',[{'method':'rtn'},{'bits':3},{'embedding_bits':2},{'seed':23},{'calibration_components':16},{'internal_components':0},{'context':511},{'group_size':32}])
def test_new_recipes_cannot_silently_change_scientific_design(change):
    from poc.dense_frontier import dense_recipes,DenseRecipe
    recipe=next(r for r in dense_recipes() if r.method=='dwq')
    with pytest.raises(ValueError):replace(recipe,**change)
    with pytest.raises(TypeError):DenseRecipe(**dict(recipe.identity(),unknown=True))


def population(n=84):
    from poc.data.manifest import DataManifest,DocumentRecord
    records=tuple(DocumentRecord(f'doc-{i:04d}','owned','a'*40,'owned',1,text='owned fixture only') for i in reversed(range(n)))
    return DataManifest(records,{r.document_id:'fit' for r in records},{r.document_id:'component-'+r.document_id for r in records},'owned','/owned/exposure.jsonl')


def test_component_plan_is_fixed_order_and_fit_internal_is_disjoint():
    from poc.dense_frontier import calibration_plan
    p=calibration_plan(population(),evaluation_components={'separate-dev'})
    assert p['train_document_ids']==[f'doc-{i:04d}' for i in range(64)]
    assert p['internal_document_ids']==[f'doc-{i:04d}' for i in range(64,80)]
    assert not set(p['train_component_ids'])&set(p['internal_component_ids'])
    assert p['selected_components']==80 and p['selected_target_inventory']==40880
    assert p['eligibility']=='requires role-bound exact512token checks before fitting'


@pytest.mark.parametrize('change',['shortage','dev_role','overlap','duplicate_component'])
def test_population_mismatch_blocks_calibration_before_content_access(change,monkeypatch):
    from poc.dense_frontier import calibration_plan
    from poc.data import access
    p=population(79 if change=='shortage' else 84)
    if change=='dev_role':p=replace(p,roles={**p.roles,'doc-0000':'dev'})
    if change=='duplicate_component':p=replace(p,membership={k:'same' for k in p.membership})
    monkeypatch.setattr(access,'open_records',lambda *a,**k:pytest.fail('content read during pure planning'))
    evaluation={'component-doc-0070'} if change=='overlap' else set()
    with pytest.raises(ValueError):calibration_plan(p,evaluation_components=evaluation)


def test_source_discovery_hashes_code_without_importing_mlx(tmp_path,monkeypatch):
    from poc.dense_frontier import installed_algorithm_sources
    root=tmp_path/'mlx_lm';(root/'quant').mkdir(parents=True);(root/'models').mkdir()
    for name in ('quant/gptq.py','quant/awq.py','quant/dwq.py','models/llama.py','__init__.py'):
        (root/name).write_text("raise AssertionError('source must never be imported')\n")
    import importlib.util
    monkeypatch.setattr(importlib.util,'find_spec',lambda *a,**k:pytest.fail('package find_spec can initialize Metal'))
    before=set(sys.modules)
    refs=installed_algorithm_sources(package_root=root)
    assert len(refs)==5 and all(Path(r['path']).is_relative_to(root) for r in refs)
    assert not (set(sys.modules)-before)&{'mlx','mlx.core','mlx_lm'}
    (root/'quant/aliased.py').symlink_to(root/'quant/gptq.py')
    with pytest.raises(ValueError):installed_algorithm_sources(package_root=root)


@pytest.mark.parametrize('phase,input_tokens,targets,pairs',[('train',511,511,510),('valid',511,511,510),('bridge',512,511,511)])
def test_dwq_target_units_preserve_last_prediction_position(phase,input_tokens,targets,pairs):
    from poc.dense_frontier import dwq_target_units
    value=dwq_target_units(phase,input_tokens=input_tokens,original_context_tokens=512)
    assert value['distribution_targets']==targets
    assert value['model_shifted_pairs']==pairs
    assert value['processed_tokens']==input_tokens
    assert value['original_corpus_targets']==511
    assert value['label_mask_source']=='original512context length; final EOS target preserved'


@pytest.mark.parametrize('phase,n',[('train',512),('valid',510),('bridge',511),('unknown',511)])
def test_dwq_unit_mismatch_is_rejected(phase,n):
    from poc.dense_frontier import dwq_target_units
    with pytest.raises(ValueError):dwq_target_units(phase,input_tokens=n,original_context_tokens=512)


@pytest.fixture
def contract(tmp_path):
    from poc.dense_frontier import dense_recipes,REFERENCE_LABELS,PREPARATION_LIMITS,NUMERICS
    root=tmp_path/'study';root.mkdir()
    file=lambda name:{'path':str(tmp_path/name),'sha256':'a'*64}
    return root,{'schema_version':1,'study':'dense-baseline-expansion-v1','out':str(root/'artifacts/dense-baseline-expansion-v1'),
        'reference_acceptance':file('accepted.json'),'reference_config':file('source.yaml'),
        'reference_artifacts':{label:{'directory':str(tmp_path/label),'manifest_sha256':'b'*64} for label in REFERENCE_LABELS},
        'fit_manifest':file('fit.json'),'inputs':dict(file('inputs.json'),population_manifest_sha256='c'*64),
        'worker':file('worker.py'),'recipes':[r.identity() for r in dense_recipes()],
        'calibration_order':'first80 disjoint components sorted by document_id; first64train next16fit_internal',
        'numerics':dict(NUMERICS),'preparation_limits':dict(PREPARATION_LIMITS),'resource_freeze':None}


def test_closed_prospective_contract_is_immutable_and_unprofiled(contract):
    from poc.dense_frontier import validate_contract_shape
    root,c=contract;frozen=validate_contract_shape(c,root=root)
    with pytest.raises(TypeError):frozen['study']='changed'
    assert frozen['resource_freeze'] is None
    assert len(frozen['reference_artifacts'])+len(frozen['recipes'])==18


@pytest.mark.parametrize('change',['unknown','version_bool','out_escape','out_alias','missing_reference','missing_recipe','extra_recipe','wrong_order','numerical_floor','memory','wall','nonfinite','unknown_budget','partial_freeze','relative_ref','bad_hash'])
def test_contract_drift_fails_before_any_source_or_model_read(contract,change):
    from copy import deepcopy
    from poc.dense_frontier import validate_contract_shape
    root,c=contract;c=deepcopy(c)
    if change=='unknown':c['optimizer_trials']=10
    elif change=='version_bool':c['schema_version']=True
    elif change=='out_escape':c['out']=str(root.parent/'outputs')
    elif change=='out_alias':c['out']=str(root)+'/artifacts/../outputs'
    elif change=='missing_reference':c['reference_artifacts'].pop(next(iter(c['reference_artifacts'])))
    elif change=='missing_recipe':c['recipes'].pop()
    elif change=='extra_recipe':c['recipes'].append(c['recipes'][0])
    elif change=='wrong_order':c['calibration_order']='best measured components'
    elif change=='numerical_floor':c['numerics']['negative_kl_floor_per_target']=.1
    elif change=='memory':c['preparation_limits']['memory_bytes']*=2
    elif change=='wall':c['preparation_limits']['campaign_wall_seconds']*=2
    elif change=='nonfinite':c['preparation_limits']['phase_wall_seconds']=float('inf')
    elif change=='unknown_budget':c['preparation_limits']['unlimited']=True
    elif change=='partial_freeze':c['resource_freeze']={'profiles':[]}
    elif change=='relative_ref':c['worker']['path']='relative.py'
    elif change=='bad_hash':c['fit_manifest']['sha256']='unknown'
    with pytest.raises(ValueError):validate_contract_shape(c,root=root)


def dense_driver():
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'provenance/dense_frontier_study.py'
    assert p.is_file(),'explicit metadata-only dense study driver missing'
    spec=importlib.util.spec_from_file_location('owned_dense_frontier_driver',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_driver_import_never_initializes_metal(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='mlx' or name.startswith(('mlx.','mlx_lm')):pytest.fail('metadata driver imported native backend')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    module=dense_driver();assert callable(module.metadata_validate)


def test_cli_rejects_unimplemented_model_modes_before_metadata(tmp_path,monkeypatch):
    module=dense_driver()
    monkeypatch.setattr(module,'metadata_validate',lambda *a,**k:pytest.fail('unimplemented mode touched metadata'))
    with pytest.raises(SystemExit) as error:module.main(['--config',str(tmp_path/'missing.yaml'),'--mode','run'])
    assert error.value.code==2


def test_metadata_composition_records_real_list_source_inventory(contract,tmp_path,monkeypatch):
    from dataclasses import asdict
    import yaml
    from poc.records import canonical_json
    from poc.recovery_endpoints import file_sha,hash_value
    root,c=contract;module=dense_driver();monkeypatch.setattr(module,'ROOT',root)
    def publish(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical_json(value));return {'path':str(path),'sha256':file_sha(path)}
    c['fit_manifest']=publish(tmp_path/'fit.json',asdict(population(80)))
    c['inputs']=dict(publish(tmp_path/'inputs.json',{'owned':'not decoded during metadata validation'}),population_manifest_sha256='c'*64)
    worker=Path(__file__).resolve().parents[1]/'provenance/dense_frontier_native_worker.py'
    c['worker']={'path':str(worker),'sha256':file_sha(worker)}
    scientific=publish(tmp_path/'scientific.json',{'populations':{'tuning':[{'component_id':f'eval-{i}','full':{'valid_targets':511}} for i in range(128)]}})
    prior=tmp_path/'prior';parent={'out':str(prior),'inputs':c['inputs'],'worker':c['worker'],'reference':{'scientific_observation':scientific}}
    config=tmp_path/'prior.yaml';config.write_text(yaml.safe_dump(parent));c['reference_config']={'path':str(config),'sha256':file_sha(config)}
    old={}
    for label in c['reference_artifacts']:
        folder=tmp_path/label;reference=publish(folder/'manifest.json',{'owned':'no weights/model in this fixture'})
        c['reference_artifacts'][label]={'directory':str(folder),'manifest_sha256':reference['sha256']}
        old[label]=dict(c['reference_artifacts'][label],complete_bytes=123)
    publish(prior/'preparation.json',{'study_contract_sha256':hash_value(parent),'artifacts':old})
    receipt=publish(tmp_path/'receipt.json',{'status':'accepted','configuration':c['reference_config'],
        'artifacts':{k:{key:v[key] for key in ('manifest_sha256','complete_bytes')} for k,v in old.items()}})
    independent=publish(tmp_path/'independent.json',{'config':c['reference_config']})
    c['reference_acceptance']=publish(tmp_path/'accepted.json',{'status':'accepted','result':receipt,'contract':independent})
    refs=[{'path':str(worker),'sha256':file_sha(worker)}]
    monkeypatch.setattr(module,'installed_algorithm_sources',lambda:refs)
    path=root/'contract.yaml';path.write_text(yaml.safe_dump(c))
    result=module.metadata_validate(path)
    assert result['algorithm_sources']==refs
    assert result['algorithm_sources_sha256']==hash_value({'files':refs})
    assert result['reference_points']==12 and result['new_points']==6
    assert result['separate_text_token_reads']==0 and result['native_imports'] is False


def test_owned_selector_binding_fixture_includes_new_poc_modules(tmp_path):
    from test_selector_source_binding import make_binding
    from poc.recovery_endpoints import finite_json
    from poc.selector_controls import verify_selector_source_binding
    binding=make_binding(tmp_path)
    assert 'poc/dense_frontier.py' in finite_json(binding['path'])['files']
    verify_selector_source_binding(Path(__file__).resolve().parents[1],binding)


def v2_contract(contract):
 from copy import deepcopy
 root,c=contract;c=deepcopy(c);c.update(schema_version=2,study='dense-baseline-expansion-v2',out=str(root/'artifacts/dense-baseline-expansion-v2'),
  accounting_profile='dense-native-contexts-v2',tokenization_policy='no-special-tokens-v2')
 return root,c


def test_v2_contract_binds_actual_context_accounting_and_tokenization(contract):
 from poc.dense_frontier import validate_contract_shape
 root,c=v2_contract(contract);f=validate_contract_shape(c,root=root)
 assert f['accounting_profile']=='dense-native-contexts-v2' and f['tokenization_policy']=='no-special-tokens-v2'


@pytest.mark.parametrize('change',['accounting','tokenization','missing','extra','old_identity','legacy_with_new_fields'])
def test_v2_semantic_or_identity_drift_is_rejected(contract,change):
 from poc.dense_frontier import validate_contract_shape
 root,c=v2_contract(contract)
 if change=='accounting':c['accounting_profile']='raw-contexts-v1'
 elif change=='tokenization':c['tokenization_policy']=None
 elif change=='missing':c.pop('accounting_profile')
 elif change=='extra':c['unpriced_teacher_queries']=True
 elif change=='old_identity':c['study']='dense-baseline-expansion-v1'
 elif change=='legacy_with_new_fields':c['schema_version']=1
 with pytest.raises(ValueError):validate_contract_shape(c,root=root)



def test_v2_scientific_identity_excludes_only_measured_resource_capsule(contract):
 from poc.dense_frontier import dense_contract_digest
 root,c=v2_contract(contract);before=dense_contract_digest(c)
 c['resource_freeze']={'path':str(root/'freeze.json'),'sha256':'a'*64}
 assert dense_contract_digest(c)==before
 c['recipes'][0]['seed']=23
 assert dense_contract_digest(c)!=before


@pytest.fixture
def native_packet(contract):
 from poc.dense_frontier import dense_contract_digest,dense_recipes,REFERENCE_LABELS
 from provenance.sharing_frontier_study import metric_work_plan
 root,c=v2_contract(contract);labels=list(REFERENCE_LABELS)+[r.label for r in dense_recipes()]
 rows=[{'document_id':str(i),'component_id':str(i+128),'tokens':[1]*512,'positions':list(range(512)),
     'attention_mask':[True]*512,'target_mask':[True]*511} for i in range(128)]
 prepared={'study_contract_sha256':dense_contract_digest(c),'inputs':c['inputs'],'teacher_mps_digest':'d'*64,
  'artifacts':{label:{'directory':str(root/'artifacts/points'/label),'manifest_sha256':'b'*64,'runtime_sha256':'c'*64} for label in labels},
  'blocked_source_roots':[str(root)]}
 counts=metric_work_plan(rows,mode='profile');counts.pop('components')
 resource={'wall_seconds':300,'memory_bytes':8*1024**3,'output_disk_bytes':12*1024**3,**counts}
 return c,prepared,rows,resource,root


def test_new18point_native_protocol_uses_exact_qualified_worker_schema(native_packet):
 from poc.dense_frontier import dense_native_protocol
 c,p,rows,r,root=native_packet
 packet=dense_native_protocol(c,p,'dense_dwq4_embedding4_cal64',rows,mode='profile',resource=r,output=root/'result.json',ledger=root/'costs.jsonl')
 assert packet['resource']['valid_target_presentations']==511 and packet['artifact']['path'].endswith('dense_dwq4_embedding4_cal64')
 assert packet['numerical']=={'chunk_size':32,'temperature':1,'negative_floor':2e-6,'decision_resolution':2e-5}
 assert packet['worker_sha256']==c['worker']['sha256']


@pytest.mark.parametrize('change',['partial_grid','runtime','count','unknown_resource','contract','missing_freeze'])
def test_native_protocol_drift_denied_before_model_or_input_io(native_packet,change):
 from poc.dense_frontier import dense_native_protocol
 c,p,rows,r,root=native_packet;mode='profile'
 if change=='partial_grid':p['artifacts'].pop('dense_rtn8')
 elif change=='runtime':p['artifacts']['dense_dwq4_embedding4_cal64']['runtime_sha256']='e'*64
 elif change=='count':r['valid_target_presentations']-=1
 elif change=='unknown_resource':r['unlimited']=True
 elif change=='contract':p['study_contract_sha256']='e'*64
 elif change=='missing_freeze':mode='run'
 with pytest.raises(ValueError):dense_native_protocol(c,p,'dense_dwq4_embedding4_cal64',rows,mode=mode,resource=r,output=root/'result.json',ledger=root/'costs.jsonl')



def test_run_protocol_preserves_profile_and_scientific_identity(native_packet):
 from poc.dense_frontier import dense_native_protocol,dense_contract_digest
 from provenance.sharing_frontier_study import metric_work_plan
 c,p,rows,r,root=native_packet
 c['resource_freeze']={'path':str(root/'freeze.json'),'sha256':'e'*64}
 r.update({k:v for k,v in metric_work_plan(rows,mode='run').items() if k!='components'})
 profile={'path':str(root/'profile.json'),'sha256':'f'*64}
 packet=dense_native_protocol(c,p,'dense_dwq4_embedding4_cal64',rows,mode='run',resource=r,output=root/'result.json',ledger=root/'costs.jsonl',profile_binding=profile)
 assert packet['profile_binding']==profile and packet['study_contract_sha256']==dense_contract_digest(c)
 assert packet['resource']['valid_target_presentations']==65408


@pytest.fixture
def frontier_packet(native_packet):
 from poc.dense_frontier import dense_contract_digest
 from poc.records import canonical_json
 import hashlib
 c,p,rows,r,root=native_packet
 for row in rows:row['window_sha256']=hashlib.sha256(canonical_json(row)).hexdigest()
 values={}
 for i,label in enumerate(p['artifacts']):
  nbytes=50000000+i*1000
  p['artifacts'][label].update(complete_bytes=nbytes,teacher_needed_at_inference=False)
  kl=0 if label=='teacher_noop' else (1 if 'cal64' not in label else 2)
  values[label]={'status':'complete','complete_bytes':nbytes,
   'rows':[{'document_id':w['document_id'],'component_id':w['component_id'],'window_sha256':w['window_sha256'],
    'valid_targets':511,'kl_sum':kl*511,'nll_sum':(3+kl)*511,'teacher_nll_sum':3*511,'disagreements':0} for w in rows]}
 return c,p,rows,values,root


def test_full18point_analysis_retains_costly_adverse_baselines_and_historical_ids(frontier_packet):
 from poc.dense_frontier import dense_frontier_summary,REFERENCE_LABELS
 c,p,rows,values,root=frontier_packet
 out=dense_frontier_summary(c,p,rows,values)
 assert len(out['points'])==18 and out['valid_targets']==65408
 point=out['points']['dense_dwq4_embedding4_cal64']
 assert point['status']=='complete' and point['kl']==2
 assert point['observation_origin']=='new_study'
 assert out['points']['dense_gptq4']['observation_origin']=='imported_reference_with_original_identity'
 assert len(out['contrasts'])==12
 assert all(v['kl_difference']==1 and v['paired_component_percentile_95']==[1,1] for v in out['contrasts'])
 assert out['scope']['capability_acceptability_established'] is False
 assert out['scope']['preparation_matched'] is False
 assert out['named_ceiling_bytes']==66064401


def test_declared_algorithm_failure_is_visible_without_invented_metrics(frontier_packet):
 from poc.dense_frontier import dense_frontier_summary
 c,p,rows,values,root=frontier_packet
 label='dense_awq2_embedding4_cal64'
 values[label]={'status':'failed','failure':{'path':str(root/'failed.json'),'sha256':'f'*64},'reason':'owned unsupported shape'}
 out=dense_frontier_summary(c,p,rows,values)
 assert out['points'][label]==values[label]
 assert len(out['contrasts'])==10 and out['failed_labels']==[label]
 assert label not in out['frontier_labels']


@pytest.mark.parametrize('change',['missing_point','missing_row','duplicate_component','wrong_mask','teacher_drift','negative_floor','nan','teacher_dependency','wrong_byte_census'])
def test_frontier_incomplete_or_incomparable_measurements_fail(frontier_packet,change):
 from poc.dense_frontier import dense_frontier_summary
 c,p,rows,values,root=frontier_packet;label='dense_dwq4_embedding4_cal64'
 if change=='missing_point':values.pop(label)
 elif change=='missing_row':values[label]['rows'].pop()
 elif change=='duplicate_component':values[label]['rows'][0]['component_id']=values[label]['rows'][1]['component_id']
 elif change=='wrong_mask':values[label]['rows'][0]['valid_targets']=510
 elif change=='teacher_drift':values[label]['rows'][0]['teacher_nll_sum']+=1
 elif change=='negative_floor':values[label]['rows'][0]['kl_sum']=-.01
 elif change=='nan':values[label]['rows'][0]['nll_sum']=float('nan')
 elif change=='teacher_dependency':p['artifacts'][label]['teacher_needed_at_inference']=True
 elif change=='wrong_byte_census':values[label]['complete_bytes']+=1
 with pytest.raises(ValueError):dense_frontier_summary(c,p,rows,values)



def test_preparation_plan_reserves_completed_dwq_updates_and_both_loss_splits():
 from poc.dense_frontier import dense_preparation_plan,dense_recipes
 recipe=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4)
 full=dense_preparation_plan(recipe,profile=False);short=dense_preparation_plan(recipe,profile=True)
 assert (full['train_components'],full['internal_components'],full['updates'])==(64,16,64)
 assert (short['train_components'],short['internal_components'],short['updates'])==(2,2,2)
 assert full['expected_scored_targets']==(64+2*16)*511
 assert short['expected_scored_targets']==6*511
 assert full['logical_optimizer_raw_targets']==64*511
 assert full['work']['updates']==64 and full['work']['student_backward_calls']==64
 assert full['work']['valid_target_presentations']==96*511
 assert full['expected_adapter_raw_targets']==196*511


@pytest.mark.parametrize('method',['gptq','awq'])
def test_noniterative_preparation_has_no_invented_updates(method):
 from poc.dense_frontier import dense_preparation_plan,dense_recipes
 p=dense_preparation_plan(next(r for r in dense_recipes() if r.method==method and r.bits==4),profile=False)
 assert p['updates']==0 and p['expected_scored_targets']==0
 assert p['train_components']==64 and p['internal_components']==0


@pytest.mark.parametrize('change',['nonfinite','updates','quality','over_ceiling','missing_context'])
def test_preparation_forecast_rejects_incomplete_or_quality_bearing_profile(change):
 from poc.dense_frontier import dense_preparation_forecast,dense_preparation_plan,dense_recipes
 r=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4);plan=dense_preparation_plan(r,profile=True)
 p={'schema_version':1,'recipe':r.identity(),'profile':True,'plan':plan,'wall_seconds':10.,'complete_bytes':60000000,
  'memory_peak_bytes':1000000000,'actual_work':dict(plan['work']),'candidate_quality_disclosed':False}
 if change=='nonfinite':p['wall_seconds']=float('nan')
 elif change=='updates':p['actual_work']['updates']=1
 elif change=='quality':p['tuning_kl']=.1
 elif change=='over_ceiling':p['wall_seconds']=1000.
 elif change=='missing_context':p['plan'].pop('train_components')
 with pytest.raises(ValueError):dense_preparation_forecast(r,p)


def test_preparation_forecast_prices_full_schedule_and_preserves_hard_caps():
 from poc.dense_frontier import dense_preparation_forecast,dense_preparation_plan,dense_recipes
 r=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4);plan=dense_preparation_plan(r,profile=True)
 p={'schema_version':1,'recipe':r.identity(),'profile':True,'plan':plan,'wall_seconds':10.,'complete_bytes':60000000,
  'memory_peak_bytes':1000000000,'actual_work':dict(plan['work']),'candidate_quality_disclosed':False}
 frozen=dense_preparation_forecast(r,p)
 assert frozen['resource']['wall_seconds']==780
 assert frozen['resource']['memory_bytes']==8*1024**3 and frozen['resource']['disk_bytes']==12*1024**3
 assert frozen['plan']['updates']==64 and frozen['plan']['logical_optimizer_raw_targets']==32704



def execution_driver():
 import importlib.util,sys
 path=Path(__file__).resolve().parents[1]/'provenance/dense_frontier_execution.py'
 assert path.is_file(),'explicit dense execution driver missing'
 spec=importlib.util.spec_from_file_location('owned_dense_execution_driver',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def test_new_phase_records_updates_raw_exposure_and_safe_failed_attempt(contract):
 from poc.contracts import CostRecord
 root,c=v2_contract(contract);m=execution_driver();source=root/'owned.py';source.write_text('owned')
 work={'updates':2,'valid_target_presentations':1022,'teacher_forward_calls':0,'student_forward_calls':0,
       'student_backward_calls':2,'raw_corpus_target_presentations':1022}
 resource={'wall_seconds':30,'memory_bytes':4*1024**3,'disk_bytes':128*1024**2}
 def operation(rec):
  rec.record(CostRecord(updates=2,valid_target_presentations=1022,student_backward_calls=2,raw_corpus_target_presentations=1022))
  return {'completed':2}
 value=m.record_dense_phase(c,'owned-completed',{str(source):m.file_sha(source)},operation,resource=resource,work=work)
 assert value['completed']==2
 import json
 phase=Path(c['out'])/'phases/owned-completed';terminal=json.loads((phase/'terminal-cost.json').read_text())
 assert terminal['totals']['updates']==2 and terminal['totals']['raw_corpus_target_presentations']==1022
 assert terminal['scientific_contract_sha256']==m.dense_contract_digest(c)
 with pytest.raises(FileExistsError):m.record_dense_phase(c,'owned-completed',{},operation,resource=resource,work=work)
 def interrupted(rec):
  rec.record(CostRecord(raw_corpus_target_presentations=511));raise InterruptedError('owned interruption')
 with pytest.raises(InterruptedError):m.record_dense_phase(c,'owned-failed',{},interrupted,resource=resource,work=work)
 failed=Path(c['out'])/'phases/owned-failed'
 assert (failed/'failed.json').is_file() and not (failed/'terminal-cost.json').exists()
 rows=list((failed/'records').rglob('costs.jsonl'))
 assert sum(json.loads(line)['cost']['raw_corpus_target_presentations'] for line in rows[0].read_text().splitlines())==511


@pytest.mark.parametrize('unit',['updates','raw_corpus_target_presentations','student_forward_calls','student_backward_calls','valid_target_presentations'])
def test_phase_rejects_unpriced_work_before_dispatch(contract,unit):
 from poc.contracts import CostRecord
 root,c=v2_contract(contract);m=execution_driver();work=dict(updates=0,valid_target_presentations=0,teacher_forward_calls=0,
 student_forward_calls=0,student_backward_calls=0,raw_corpus_target_presentations=0)
 def extra(rec):rec.check(CostRecord(**{unit:1}));pytest.fail('unpriced work reached dispatch')
 with pytest.raises(Exception,match='budget|allowance|cap'):
  m.record_dense_phase(c,'owned-'+unit,{},extra,resource={'wall_seconds':30,'memory_bytes':4*1024**3,'disk_bytes':128*1024**2},work=work)


def test_direct_native_preparation_requires_live_guard_before_backend_import(contract,monkeypatch):
 import builtins
 root,c=v2_contract(contract);m=execution_driver();original=builtins.__import__
 def guarded(name,*args,**kwargs):
  if name=='mlx' or name.startswith(('mlx.','mlx_lm')):pytest.fail('native setup before serial/source/guard checks')
  return original(name,*args,**kwargs)
 monkeypatch.setattr(builtins,'__import__',guarded)
 with pytest.raises(ValueError,match='guard'):
  m.prepare_dense_artifact(c,'dense_dwq4_embedding4_cal64',config_path=None,profile=True,sources={})
 assert not Path(c['out']).exists()



@pytest.fixture
def dense_fit_packet(contract):
 import json
 from dataclasses import asdict
 from tokenizers import Tokenizer,models,pre_tokenizers
 from poc.data.manifest import DataManifest,DocumentRecord
 from poc.records import canonical_json
 root,c=v2_contract(contract);m=execution_driver()
 records=tuple(DocumentRecord(f'{i:03}','owned','r','s',i,text=' '.join(['word']*512)) for i in range(80))
 parent_records=records+tuple(DocumentRecord(f'{i:03}','owned','r','s',i,text=' '.join(['word']*512)) for i in range(80,512))
 parent=DataManifest(parent_records,{r.document_id:'fit' for r in parent_records},{r.document_id:'component-'+r.document_id for r in parent_records},'owned',str(root/'prior-exposure.jsonl'))
 parent_path=root/'parent512.json';parent_path.write_bytes(canonical_json(asdict(parent)))
 pop=replace(parent,records=records,roles={r.document_id:'fit' for r in records},membership={r.document_id:'component-'+r.document_id for r in records},lineage_sha256=parent.digest(),exposure_path=str(root/'new-fit-exposure.jsonl'))
 path=root/'fit.json';path.write_bytes(canonical_json(asdict(pop)));c['fit_manifest']=m.reference(path)
 descriptor=root/'provenance'/c['study']/'fit-parent-derivation-v1.json';descriptor.parent.mkdir(parents=True)
 descriptor.write_bytes(canonical_json({'schema_version':1,'study':c['study'],'parent_manifest':m.reference(parent_path),'child_manifest':c['fit_manifest'],'rule':c['calibration_order']}))
 source=root/'teacher';(source/'tokenizer').mkdir(parents=True)
 tok=Tokenizer(models.WordLevel({'[UNK]':0,'word':1},unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(source/'tokenizer/tokenizer.json'))
 return c,pop,source,root,m


@pytest.mark.parametrize('profile,expected_train,expected_internal',[(True,2,2),(False,64,16)])
def test_dense_population_keeps_prespecified_fit_internal_and_original_windows(dense_fit_packet,profile,expected_train,expected_internal):
 from poc.dense_frontier import dense_recipes
 c,pop,source,root,m=dense_fit_packet;r=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4)
 selected,proof=m.selected_dense_population(c,r,source,profile=profile,phase='owned',evaluation_components={'separate'})
 assert len(selected.records)==expected_train+expected_internal
 assert proof['train_document_ids']==[f'{i:03}' for i in range(expected_train)]
 assert proof['internal_document_ids']==[f'{i:03}' for i in range(64,64+expected_internal)]
 assert all(w['input_tokens']==512 and w['valid_targets']==511 for w in proof['windows'])
 assert all(w['tokens']==[1]*512 for w in proof['windows'])
 assert selected.exposure_path!=pop.exposure_path and Path(selected.exposure_path).exists()
 assert not Path(pop.exposure_path).exists()


@pytest.mark.parametrize('change',['overlap','short','role'])
def test_dense_population_eligibility_failures_do_not_silently_replace(dense_fit_packet,change):
 from dataclasses import replace,asdict
 from poc.records import canonical_json
 from poc.dense_frontier import dense_recipes
 c,pop,source,root,m=dense_fit_packet
 if change=='short':pop=replace(pop,records=(replace(pop.records[0],text='word'),*pop.records[1:]))
 elif change=='role':pop=replace(pop,roles={**pop.roles,'000':'dev'})
 if change!='overlap':
  from poc.data.manifest import load_manifest
  import json
  descriptor=root/'provenance'/c['study']/'fit-parent-derivation-v1.json';d=json.loads(descriptor.read_text())
  if change=='short':
   parent=load_manifest(Path(d['parent_manifest']['path']),expected_sha256=d['parent_manifest']['sha256'],roles={'fit'})
   parent=replace(parent,records=(replace(parent.records[0],text='word'),*parent.records[1:]))
   parent_path=root/'changed-parent.json';parent_path.write_bytes(canonical_json(asdict(parent)))
   d['parent_manifest']=m.reference(parent_path);pop=replace(pop,lineage_sha256=parent.digest())
  path=root/'changed.json';path.write_bytes(canonical_json(asdict(pop)));c['fit_manifest']=m.reference(path)
  d['child_manifest']=c['fit_manifest'];descriptor.write_bytes(canonical_json(d))
 r=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4)
 with pytest.raises(ValueError):m.selected_dense_population(c,r,source,profile=True,phase='owned',evaluation_components={'component-000'} if change=='overlap' else {'separate'})



@pytest.fixture
def dense_quantizer_packet():
 from poc.dense_frontier import dense_recipes,dense_preparation_plan
 recipe=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4)
 plan=dense_preparation_plan(recipe,profile=False)
 proof={'train_component_ids':[f't{i}' for i in range(64)],'internal_component_ids':[f'v{i}' for i in range(16)]}
 evidence={'method':'dwq','bits':4,'embedding_bits':4,'group_size':64,'seed':17,'temperature':1.,
 'accounting_profile':'dense-native-contexts-v2','tokenization_policy':'no-special-tokens-v2','original_context_tokens':512,
 'default_loader_denial_at_imported_binding':True,'socket_connect_denied':True,
 'initial_official_bridge_max_abs_error':1e-6,'packed_common_runtime_max_abs_error':1e-6,
 'internal_train_component_ids':proof['train_component_ids'],'internal_valid_component_ids':proof['internal_component_ids'],
 'calibration_component_ids':proof['train_component_ids']+proof['internal_component_ids'],
 'actual_upstream_counts':{'updates':64,'backwards':64,'scored_loss_targets':49056,
  'logical_optimizer_raw_target_presentations':32704,'forward_raw_corpus_presentations':98*511,
  'native_raw_corpus_presentations':98*511},'total_model_context_raw_presentations':196*511}
 from test_dense_dispatch_accounting import OwnedDWQModel
 from poc.quantize import configure_dense_dwq_trainability
 evidence['actual_upstream_counts']['dwq_trainability']=configure_dense_dwq_trainability(OwnedDWQModel(),'dense-native-contexts-v2')
 return recipe,plan,proof,evidence


def test_exact_export_evidence_cannot_hide_partial_dwq_fitting(dense_quantizer_packet):
 from poc.dense_frontier import validate_dense_quantizer_evidence,dense_quantization_spec
 recipe,plan,proof,evidence=dense_quantizer_packet
 validate_dense_quantizer_evidence(recipe,plan,proof,evidence)
 spec=dense_quantization_spec(recipe,profile=False,destination='/private/tmp/owned-packed')
 assert spec.calibration_count==64 and spec.internal_validation_count==16 and spec.length==512 and spec.seed==17
 spec=dense_quantization_spec(recipe,profile=True,destination='/private/tmp/owned-packed')
 assert spec.calibration_count==spec.internal_validation_count==2


@pytest.mark.parametrize('change',['updates','loss_count','raw_views','logical_optimizer','train_ids','internal_ids','specials','bridge','roundtrip','network','temperature'])
def test_dense_export_evidence_mismatches_rejected(dense_quantizer_packet,change):
 from copy import deepcopy
 from poc.dense_frontier import validate_dense_quantizer_evidence
 recipe,plan,proof,evidence=dense_quantizer_packet;evidence=deepcopy(evidence)
 if change=='updates':evidence['actual_upstream_counts']['updates']=63
 elif change=='loss_count':evidence['actual_upstream_counts']['scored_loss_targets']-=511
 elif change=='raw_views':evidence['total_model_context_raw_presentations']-=511
 elif change=='logical_optimizer':evidence['actual_upstream_counts']['logical_optimizer_raw_target_presentations']=0
 elif change=='train_ids':evidence['internal_train_component_ids'].reverse()
 elif change=='internal_ids':evidence['internal_valid_component_ids'][0]='t0'
 elif change=='specials':evidence['tokenization_policy']=None
 elif change=='bridge':evidence['initial_official_bridge_max_abs_error']=.006
 elif change=='roundtrip':evidence['packed_common_runtime_max_abs_error']=float('nan')
 elif change=='network':evidence['socket_connect_denied']=False
 elif change=='temperature':evidence['temperature']=2
 with pytest.raises(ValueError):validate_dense_quantizer_evidence(recipe,plan,proof,evidence)



def test_source_qualification_requires_evidence_beyond_a_success_label(contract):
 import json
 root,c=v2_contract(contract);m=execution_driver();q=root/'qualification.json'
 q.write_text(json.dumps({'schema_version':1,'status':'qualified_dense_source_cpu','study':c['study'],'scientific_contract_sha256':m.dense_contract_digest(c)}))
 with pytest.raises(ValueError,match='qualification'):
  m.require_dense_qualification(c,proposal_path=root/'proposal.yaml',qualification=m.reference(q))


def test_native_qualification_cannot_be_a_self_asserted_status(contract):
 import json
 root,c=v2_contract(contract);m=execution_driver();q=root/'native.json'
 q.write_text(json.dumps({'status':'qualified_dense_native_controls'}))
 with pytest.raises(ValueError,match='native'):
  m.require_dense_native_qualification(c,qualification=m.reference(q),source_qualification={'path':str(root/'source.json'),'sha256':'a'*64})


def test_dense_source_inventory_remains_metadata_only_and_includes_new_drivers(contract,monkeypatch):
 import builtins
 root,c=v2_contract(contract);m=execution_driver();original=builtins.__import__
 def guarded(name,*args,**kwargs):
  if name=='mlx' or name.startswith(('mlx.','mlx_lm')):pytest.fail('source inventory imported Metal')
  return original(name,*args,**kwargs)
 monkeypatch.setattr(builtins,'__import__',guarded)
 path=m.ROOT/'configs/dense-baseline-expansion-v2-proposal-v1.yaml'
 files=m.dense_executed_sources(path)
 assert str(m.ROOT/'provenance/dense_frontier_execution.py') in files
 assert str(m.ROOT/'tests/test_dense_native_controls.py') in files
 assert str(m.ROOT/'provenance/continuation-integration-v1/owned-selector-source-binding-v2.json') in files
 assert str(m.ROOT/'configs/dense-baseline-expansion-v2-run-v1.yaml') not in files
 assert any(p.endswith('/mlx_lm/quant/dwq.py') for p in files)



def test_imported_reference_stage_must_freeze_exact_index_before_raw_metric_read(tmp_path):
 m=execution_driver();index=tmp_path/'index.json';index.write_text('{}')
 admitted={'status':'accepted','analysis_acceptance':m.reference(tmp_path/'accepted.json') if (tmp_path/'accepted.json').exists() else {'path':str(tmp_path/'accepted.json'),'sha256':'a'*64}}
 with pytest.raises(ValueError,match='reference|admission|analysis'):
  m.require_imported_index_binding(admitted,{'status':'accepted'},index_ref=m.reference(index),prepared_ref=m.reference(index),parent_config={'path':str(tmp_path/'config.yaml'),'sha256':'b'*64})


def test_imported_index_current_hash_cannot_replace_prior_guard_binding(tmp_path):
 m=execution_driver();root=tmp_path/'source';out=root/'artifacts/reference';out.mkdir(parents=True)
 worker=root/'provenance/shared_gptq_study.py';worker.parent.mkdir();worker.write_text('pass')
 config=root/'configs/reference.yaml';config.parent.mkdir()
 import yaml
 parent={'out':str(out),'resource':{'preparation_wall_seconds':30,'phase_memory_bytes':1024**3,'family_disk_bytes':1024**2}}
 config.write_text(yaml.safe_dump(parent));index=out/'evaluation-index.json';index.write_text('{}')
 prepared=out/'preparation.json';prepared.write_text('{}')
 from provenance.sharing_frontier_study import execute_stage
 command=[__import__('sys').executable,str(worker),'--config',str(config),'--internal-phase','analysis']
 caps={'wall_seconds':30,'memory_bytes':1024**3,'disk_bytes':1024**2}
 refs=[m.reference(p) for p in (worker,config,index,prepared)]
 env={'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'}
 stage=execute_stage(command,root=root,output=out,stage=out/'stages/analysis',caps=caps,frozen_files=refs,execution_env=env)
 accepted={'status':'accepted','stage':stage};ap=out/'analysis-acceptance.json';ap.write_text(__import__('json').dumps(accepted))
 admitted={'status':'accepted','analysis_acceptance':m.reference(ap)}
 m.require_imported_index_binding(admitted,accepted,index_ref=m.reference(index),prepared_ref=m.reference(prepared),parent_config=m.reference(config))
 index.write_text('{"changed":true}')
 with pytest.raises(ValueError,match='reference|analysis|binding|guard|source'):
  m.require_imported_index_binding(admitted,accepted,index_ref=m.reference(index),prepared_ref=m.reference(prepared),parent_config=m.reference(config))



def test_serial_slot_cannot_be_opened_by_a_status_only_record(contract):
 import json
 root,c=v2_contract(contract);m=execution_driver();path=root/'serial.json'
 path.write_text(json.dumps({'status':'accepted_serial_slot_after_cold_family'}))
 with pytest.raises(ValueError,match='serial'):
  m.require_dense_serial_admission(c,admission=m.reference(path))



@pytest.mark.parametrize('label,seed,scopes',[('scale-mean17',17,['local_mean']),('paired23',23,['local_sum','full_student']),('paired31',31,['local_sum','full_student'])])
def test_serial_member_horizons_come_from_exact_config_not_proposal_record(label,seed,scopes):
 m=execution_driver();member={'label':label,'seed':seed,'config':{'path':'/private/tmp/config.yaml','sha256':'a'*64}}
 config={'seeds':[seed],'loss_scopes':scopes,'updates':1024,'raw_targets_per_update':8192,'owner_scope':'affected_group','alpha':0}
 assert m.cold_member_expected_arms(member,config)=={scope+'-seed'+str(seed) for scope in scopes}
 config['updates']=256
 with pytest.raises(ValueError):m.cold_member_expected_arms(member,config)


def test_review_r1_quantizer_evidence_requires_exact_trainable_owner_record(dense_quantizer_packet):
 from poc.dense_frontier import validate_dense_quantizer_evidence
 recipe,plan,proof,evidence=dense_quantizer_packet
 evidence['actual_upstream_counts'].pop('dwq_trainability',None)
 with pytest.raises(ValueError,match='scope|owner|trainable'):validate_dense_quantizer_evidence(recipe,plan,proof,evidence)


def test_review_r1_native_control_identifies_rmsnorms_from_real_bundle_metadata(tmp_path):
 from test_dense_native_controls import float_norm_owner_names
 from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
 from poc.bundle import write_bundle,verify_bundle
 from test_export import runtime
 import torch
 dense=make_tiny_decoder(TinyConfig(),seed=3,dtype=torch.float32)
 source=write_bundle(dense,tmp_path/'source',runtime(tmp_path))
 metadata=verify_bundle(Path(source.directory)).metadata
 expected={'norm.weight'}|{f'layers.{d}.{site}.weight' for d in range(3) for site in ('input_layernorm','post_attention_layernorm')}
 assert set(float_norm_owner_names(metadata))==expected
 assert not any(row['category']=='norm' for row in metadata['owners'])


@pytest.fixture
def parent_child_population():
 parent=population(512);by_id={r.document_id:r for r in parent.records};ids=sorted(by_id)[:80]
 child=replace(parent,records=tuple(by_id[d] for d in ids),roles={d:'fit' for d in ids},
  membership={d:parent.membership[d] for d in ids},lineage_sha256=parent.digest(),exposure_path='/owned/new-exposure.jsonl')
 return parent,child


def test_review_r6_parent_derivation_checks_original512_before_content(parent_child_population,monkeypatch):
 from poc.dense_frontier import verify_dense_parent_selection
 from poc.data import access
 monkeypatch.setattr(access,'open_records',lambda *a,**k:pytest.fail('raw content during parent selection proof'))
 parent,child=parent_child_population
 proof=verify_dense_parent_selection(parent,child,evaluation_components={'separate'})
 assert proof['parent_components']==512 and proof['selected_components']==80
 assert proof['train_document_ids']==[f'doc-{i:04d}' for i in range(64)]


@pytest.mark.parametrize('change',['different80','membership','fabricated_lineage','parent_shortage'])
def test_review_r6_well_formed_child_from_wrong_parent_or_positions_rejected(parent_child_population,change):
 from poc.dense_frontier import verify_dense_parent_selection
 parent,child=parent_child_population
 if change=='different80':
  by_id={r.document_id:r for r in parent.records};ids=sorted(by_id)[80:160]
  child=replace(child,records=tuple(by_id[d] for d in ids),roles={d:'fit' for d in ids},membership={d:parent.membership[d] for d in ids})
 elif change=='membership':child=replace(child,membership={**child.membership,'doc-0000':'different-component'})
 elif change=='fabricated_lineage':child=replace(child,lineage_sha256='0'*64)
 else:parent=population(511)
 with pytest.raises(ValueError):verify_dense_parent_selection(parent,child,evaluation_components={'separate'})


def test_review_r4_infeasible_aggregate_retains_all18_outcomes_and_cost_links(frontier_packet,monkeypatch):
 from provenance import dense_frontier_campaign as m
 from poc.dense_frontier import dense_recipes,REFERENCE_LABELS,dense_contract_digest
 from poc.records import canonical_json
 import time
 c,p,rows,values,root=frontier_packet;out=Path(c['out']);out.mkdir(parents=True)
 config=root/'proposal.yaml';config.write_text('owned')
 m.write_once(out/'campaign-intent.json',{'scientific_contract_sha256':dense_contract_digest(c),
  'started_unix':time.time(),'campaign_wall_seconds':28800})
 ledger=out/'phases/owned-short/records/costs.jsonl';ledger.parent.mkdir(parents=True);ledger.write_text('owned retained cost prefix\n')
 imported={'artifacts':{k:p['artifacts'][k] for k in REFERENCE_LABELS},'prepared':p,'windows':rows,
  'measurements':{k:values[k] for k in REFERENCE_LABELS},'aggregate_reference_campaign_cost':{'wall_seconds':123},
  'phase_costs':{'owned_original_phase':{'wall_seconds':123}},'ancestor_costs':{'inventory':{'path':str(root/'ancestors.json'),'sha256':'a'*64}}}
 m.write_once(out/'imported-references.json',imported)
 monkeypatch.setattr(m,'campaign_ancestor_costs',lambda c,i:{'all_upstream_expense_established':False,'owned_fixture':True})
 monkeypatch.setattr(m,'native_job',lambda *a,**k:pytest.fail('infeasible campaign dispatched a native job'))
 # Every6000s recipe fits its7200s phase ceiling, but36000s cannot fit8h.
 result=m.complete_if_infeasible(c,config,[],imported,required_wall=6000*len(dense_recipes()),phase='aggregate-preparation',prepared=p)
 assert result['status']=='terminal resource failure; complete outcome inventory retained'
 report=m.finite_json(out/'analysis.json');assert len(report['points'])==18
 assert len(report['failed_labels'])==6 and report['campaign_status']=='resource_failed_before_completed_grid'
 assert report['original_reference_campaign_cost']=={'wall_seconds':123}
 assert report['new_phase_records']['owned-short']['ledgers']==[m.reference(ledger)]
 for recipe in dense_recipes():
  point=report['points'][recipe.label]
  assert point['status']=='failed' and 'kl' not in point and 'raw_sums' not in point
  failure=m.read_reference(point['failure']);assert failure['dispatch_status']=='not_run_resource_admission'
 completion=m.finite_json(out/'analysis-completion.json')
 assert completion['campaign_status']==report['campaign_status'] and completion['terminal_cost']
 assert (out/'phases/analysis/terminal-cost.json').is_file()
 assert not (out/'resource-freeze.json').exists()


def test_review_r4_success_preflight_does_not_create_terminal_report(frontier_packet):
 from provenance import dense_frontier_campaign as m
 from poc.dense_frontier import dense_contract_digest
 import time
 c,p,rows,values,root=frontier_packet;out=Path(c['out']);out.mkdir(parents=True)
 m.write_once(out/'campaign-intent.json',{'scientific_contract_sha256':dense_contract_digest(c),'started_unix':time.time(),'campaign_wall_seconds':28800})
 assert m.complete_if_infeasible(c,root/'proposal.yaml',[],{},required_wall=6000,phase='owned') is None
 assert not (out/'analysis.json').exists()


def test_review_r4_dispatch_reserves_time_for_terminal_accounting(frontier_packet,monkeypatch):
 from provenance import dense_frontier_campaign as m
 from poc.dense_frontier import dense_contract_digest
 c,p,rows,values,root=frontier_packet;out=Path(c['out']);out.mkdir(parents=True)
 m.write_once(out/'campaign-intent.json',{'scientific_contract_sha256':dense_contract_digest(c),'started_unix':0,'campaign_wall_seconds':28800})
 monkeypatch.setattr(m.time,'time',lambda:28200)
 with pytest.raises(ValueError,match='remaining|closing'):m.envelope(c,100)
 assert m.envelope(c,300,closing=True)['wall_seconds']==300


def test_review_r4_profile_mode_emits_full_terminal_report_instead_of_aborting(frontier_packet,monkeypatch):
 from provenance import dense_frontier_campaign as m
 from provenance import dense_frontier_execution as ex
 from poc.dense_frontier import dense_recipes,dense_preparation_plan,REFERENCE_LABELS
 c,p,rows,values,root=frontier_packet;out=Path(c['out']);config=root/'proposal.yaml';config.write_text('owned')
 imported={'artifacts':{k:p['artifacts'][k] for k in REFERENCE_LABELS},'prepared':p,'windows':rows,
  'measurements':{k:values[k] for k in REFERENCE_LABELS},'aggregate_reference_campaign_cost':{},'phase_costs':{},
  'ancestor_costs':{},'native_environment_sha256':'e'*64}
 admission=root/'owned-admission.json';admission.write_text('{}');admissions={k:m.reference(admission) for k in ('qualification','serial_admission','native_qualification')}
 monkeypatch.setattr(m,'native_admissions',lambda c,refs:(refs,admissions))
 monkeypatch.setattr(ex,'import_dense_references',lambda c,config_path:imported)
 monkeypatch.setattr(m,'selected_dense_population',lambda *a,**k:({},{}))
 monkeypatch.setattr(m,'campaign_ancestor_costs',lambda *a:{'all_upstream_expense_established':False})
 calls=[]
 def completed_short(c,config,recipe,*,profile,resource,refs,admissions):
  assert profile,'aggregate infeasibility dispatched a full quantizer'
  calls.append(recipe.label);d=out/'phases'/('preprofile-'+recipe.label);d.mkdir(parents=True)
  work=dense_preparation_plan(recipe,profile=True)['work']
  terminal=d/'terminal-cost.json';m.write_once(terminal,{'totals':dict(work,memory_bytes=1000)})
  acceptance=d/'acceptance.json';m.write_once(acceptance,{'owned':True})
  return {'value':{'complete_bytes':60000000},'terminal_cost':m.reference(terminal),
   'stage':{'attempt':{'wall_seconds':91,'sampled_process_tree_rss_bytes':1000}}}
 monkeypatch.setattr(m,'dispatch_prepare',completed_short)
 monkeypatch.setattr(m,'native_job',lambda *a,**k:pytest.fail('aggregate preflight dispatched native inference'))
 result=m.profile_campaign(c,config,[])
 assert len(calls)==6 and result['status']=='terminal resource failure; complete outcome inventory retained'
 report=m.finite_json(out/'analysis.json');assert len(report['points'])==18 and len(report['failed_labels'])==6
 assert (out/'analysis-completion.json').is_file() and not (out/'profile-failed.json').exists()
 assert len(report['new_phase_records'])==8 # reference import, eligibility, six short profiles


def test_review_r3_run_admission_failure_is_explicit_outcome_without_native_dispatch(frontier_packet,monkeypatch):
 from provenance import dense_frontier_campaign as m
 from poc.dense_frontier import dense_recipes
 c,p,rows,values,root=frontier_packet;out=Path(c['out']);out.mkdir(parents=True)
 config=root/'proposal.yaml';config.write_text('owned')
 marker=root/'admission.json';marker.write_text('{}');ref=m.reference(marker)
 recipes=dense_recipes();recipe=recipes[0]
 imported={'windows':rows};ip=out/'imported-references.json';m.write_once(ip,imported)
 pp=out/'preparation.json';m.write_once(pp,p)
 capsule={'reference_import':m.reference(ip),'preparation':m.reference(pp),'jobs':{recipe.label:{}},
  'failures':{r.label:{'status':'failed','failure':ref,'reason':'owned earlier failure'} for r in recipes[1:]}}
 c['resource_freeze']=ref
 monkeypatch.setattr(m,'require_freeze',lambda c:capsule)
 monkeypatch.setattr(m,'native_admissions',lambda c,refs:(refs,{}))
 monkeypatch.setattr(m,'admission_paths',lambda c:{'qualification':marker})
 monkeypatch.setattr(m,'native_profile_closure',lambda *a:[])
 monkeypatch.setattr(m,'admitted_native_job',lambda *a,**k:(_ for _ in ()).throw(ValueError('owned altered native profile')))
 monkeypatch.setattr(m,'require_dense_qualification',lambda *a,**k:[])
 monkeypatch.setattr(m,'native_job',lambda *a,**k:pytest.fail('native dispatch after failed profile admission'))
 monkeypatch.setattr(m,'analyze_campaign',lambda *a,**k:m.finite_json(out/'evaluation-index.json'))
 result=m.run_campaign(c,config,[])
 assert len(result)==6 and result[recipe.label]['status']=='failed'
 assert 'altered native profile' in m.read_reference(result[recipe.label]['failure'])['error']

"""Owned role-bound CPU study data; no private/pretrained/native execution."""
from pathlib import Path
import importlib.util,hashlib,json,sys
from dataclasses import asdict
import pytest
from poc.recovery_population import prepare_natural_records
from poc.data.manifest import DocumentRecord,load_manifest
from test_recovery_population import records,OwnedTokenizer
from test_shared_calibration import owned


def api():
 p=Path(__file__).resolve().parents[1]/'provenance/shared_calibration_study.py'
 assert p.is_file(),'explicit calibration study driver missing'
 s=importlib.util.spec_from_file_location('shared_calibration_study_driver',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m


@pytest.fixture
def population(tmp_path):
 result=prepare_natural_records(records(),OwnedTokenizer(),tmp_path/'population',tokenizer_digest='b'*64,inventory_sha256='c'*64,fit_components=3,tuning_components=2,seed=17,max_document_bytes=10000,max_records=8,max_tokens=100000,max_disk_bytes=1000000)
 c={'study':'owned-calibration','out':str(tmp_path/'study'),'fit_manifest':{k:v for k,v in result['fit'].items() if k!='components'},'source_components':3,'selected_components':2,'tokenizer_sha256':'b'*64,'vocabulary_size':11,'max_document_bytes':10000,'max_token_file_bytes':20000}
 return c,result


def test_subset_is_chosen_before_reads_and_preserves_old_exposure_lineage(population):
 c,_=population;m=api();parent=load_manifest(Path(c['fit_manifest']['path']),expected_sha256=c['fit_manifest']['sha256'],roles={'fit'});old=Path(parent.exposure_path);prior=old.read_bytes() if old.exists() else None
 subset,windows,provenance=m.prepare_windows(c)
 assert [r.document_id for r in subset.records]==sorted(r.document_id for r in parent.records)[:2]
 assert subset.lineage_sha256==parent.digest() and subset.exposure_path!=parent.exposure_path
 assert (old.read_bytes() if old.exists() else None)==prior
 assert Path(subset.exposure_path).is_file() and set(subset.roles.values())=={'fit'}
 assert len(windows)==2 and sum(int(b.target_mask.sum()) for b in windows)==1022
 assert all(tuple(b.input_ids.shape)==(1,512) and b.positions[0,0].item()==0 and b.positions[0,-1].item()==511 and b.target_mask.all().item() for b in windows)
 assert provenance['source_manifest_sha256']==c['fit_manifest']['sha256'] and provenance['subset_manifest_sha256']==subset.digest()
 assert provenance['guarded_documents']==2 and provenance['model_context_tokens']==1024


@pytest.mark.parametrize('bad',['role','quarantine','component_alias','wrong_tokenizer','shortage','bool_count','token_size','text_size'])
def test_data_denials_precede_any_content_materialization(population,tmp_path,monkeypatch,bad):
 c,result=population;m=api();raw=json.loads(Path(c['fit_manifest']['path']).read_text())
 if bad=='role':raw['roles']={k:'protected_benchmark' for k in raw['roles']}
 elif bad=='quarantine':raw['quarantine']={raw['records'][0]['document_id']:'owned exclusion'}
 elif bad=='component_alias':raw['membership']={k:'same' for k in raw['membership']}
 elif bad=='wrong_tokenizer':c['tokenizer_sha256']='f'*64
 elif bad=='shortage':c['selected_components']=4
 elif bad=='bool_count':c['selected_components']=True
 elif bad=='token_size':c['max_token_file_bytes']=1
 elif bad=='text_size':c['max_document_bytes']=1
 if bad in ('role','quarantine','component_alias'):
  path=tmp_path/'changed-manifest.json';path.write_text(json.dumps(raw));c['fit_manifest']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 monkeypatch.setattr(DocumentRecord,'read_text',lambda *a:pytest.fail('denial must precede content reads'))
 with pytest.raises((ValueError,PermissionError)):m.prepare_windows(c)
 assert not (Path(c['out'])/'inputs/exposure.jsonl').exists()


def test_changed_subset_plan_cannot_reuse_an_existing_population_identity(population,monkeypatch):
 c,_=population;m=api();m.prepare_windows(c);c['selected_components']=1
 monkeypatch.setattr(DocumentRecord,'read_text',lambda *a:pytest.fail('changed immutable input plan must fail before another read'))
 with pytest.raises(ValueError):m.prepare_windows(c)


def test_corrupt_token_file_is_not_silently_retokenized(population):
 c,_=population;m=api();raw=json.loads(Path(c['fit_manifest']['path']).read_text());selected=sorted(raw['records'],key=lambda r:r['document_id'])[0];Path(selected['token_path']).write_text('{}')
 with pytest.raises(ValueError):m.prepare_windows(c)


def config(tmp_path):
 ref=lambda name:{'path':str(tmp_path/name),'sha256':'a'*64}
 return {'schema_version':1,'study':'owned-calibration','out':str(tmp_path/'output'),
  'source_bundle':{'directory':str(tmp_path/'source-bundle'),'manifest_sha256':'b'*64},
  'parent_qualification':ref('parent-qualification.json'),'ancestor_config':ref('ancestor.yaml'),
  'fit_manifest':ref('population/fit-manifest.json'),'tokenizer_sha256':'b'*64,'vocabulary_size':11,
  'source_components':3,'selected_components':2,'max_document_bytes':10000,'max_token_file_bytes':20000,
  'expected_layout':{'hidden_size':8,'intermediate_size':12,'num_hidden_layers':4,'banks':3,'rank':1},
  'resource':{'profile':{'wall_seconds':900,'memory_bytes':4*1024**3,'output_disk_bytes':4*1024**3,'max_statistics_bytes':1024**3},
              'run':{'wall_ceiling':3600,'memory_bytes':4*1024**3,'family_disk_bytes':8*1024**3},'deadline_unix':1900000000},
  'serial_gate':{'directory':str(tmp_path/'prior-campaign'),'queue_freeze_sha256':'c'*64}}


def yaml_config(tmp_path,c):
 import yaml
 p=tmp_path/'study.yaml';p.write_text(yaml.safe_dump(c));return p


def test_study_config_is_closed_immutable_and_creates_no_outputs(tmp_path):
 m=api();c=config(tmp_path);loaded=m.load_contract(yaml_config(tmp_path,c))
 assert loaded['selected_components']==2 and not Path(c['out']).exists()
 with pytest.raises(TypeError):loaded['resource']['profile']['wall_seconds']=1


@pytest.mark.parametrize('bad',['unknown','version','bool_version','relative_output','relative_reference','bad_source_hash','unknown_profile','zero_memory','boolean_bound','statistics_infeasible','bank_count','rank','selected_count','source_overlap','bad_deadline'])
def test_config_rejects_invalid_science_or_resource_options_without_io(tmp_path,bad):
 m=api();c=config(tmp_path)
 if bad=='unknown':c['loss_scope']='teacher'
 elif bad=='version':c['schema_version']=2
 elif bad=='bool_version':c['schema_version']=True
 elif bad=='relative_output':c['out']='relative'
 elif bad=='relative_reference':c['ancestor_config']['path']='relative.yaml'
 elif bad=='bad_source_hash':c['source_bundle']['manifest_sha256']='wrong'
 elif bad=='unknown_profile':c['resource']['profile']['chunk_size']=16
 elif bad=='zero_memory':c['resource']['profile']['memory_bytes']=0
 elif bad=='boolean_bound':c['max_document_bytes']=True
 elif bad=='statistics_infeasible':c['resource']['profile']['max_statistics_bytes']=1
 elif bad=='bank_count':c['expected_layout']['banks']=5
 elif bad=='rank':c['expected_layout']['rank']=-1
 elif bad=='selected_count':c['selected_components']=4
 elif bad=='source_overlap':c['out']=c['source_bundle']['directory']+'/inside'
 elif bad=='bad_deadline':c['resource']['deadline_unix']=float('inf')
 with pytest.raises(ValueError):m.load_contract(yaml_config(tmp_path,c))
 assert not Path(config(tmp_path)['out']).exists()


def test_fit_inventory_inspection_never_opens_text_or_token_content(population,monkeypatch):
 c,_=population;m=api();read=Path.read_bytes
 def metadata_only(path):
  if path.parent.name in ('text','tokens'):pytest.fail('metadata validation must not expose corpus content')
  return read(path)
 monkeypatch.setattr(Path,'read_bytes',metadata_only)
 monkeypatch.setattr(DocumentRecord,'read_text',lambda *a:pytest.fail('metadata inspection must not materialize text'))
 inspected=m.inspect_fit_metadata(c)
 assert len(inspected['selected'])==2 and set(inspected['parent'].roles.values())=={'fit'}
 assert [r.document_id for r in inspected['selected']]==sorted(r.document_id for r in inspected['parent'].records)[:2]
 assert not Path(c['out']).exists()


def test_source_inspection_checks_complete_floating_layout_without_loading_model(owned,tmp_path,monkeypatch):
 from poc.bundle import write_bundle
 from test_export import runtime
 import poc.bundle
 m=api();ref=write_bundle(owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True);c=config(tmp_path);c['source_bundle']={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256};c['tokenizer_sha256']='a'*64
 monkeypatch.setattr(poc.bundle,'load_bundle',lambda *a,**k:pytest.fail('metadata inspection must not load/forward a model'))
 inspected=m.inspect_source_bundle(c)
 assert inspected['source_ref']==ref and inspected['complete_bytes']>0 and len(inspected['metadata']['groups'])==3
 assert not Path(c['out']).exists()


@pytest.mark.parametrize('bad',['manifest','dense','hidden_width','tokenizer','rank','context'])
def test_source_inspection_denies_changed_or_wrong_models_before_load(owned,tmp_path,monkeypatch,bad):
 from poc.bundle import write_bundle
 from test_export import runtime
 import poc.bundle
 m=api();ref=write_bundle(owned.teacher if bad=='dense' else owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=bad!='dense');c=config(tmp_path);c['source_bundle']={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256};c['tokenizer_sha256']='a'*64
 if bad=='manifest':c['source_bundle']['manifest_sha256']='f'*64
 elif bad=='hidden_width':c['expected_layout']['hidden_size']=16
 elif bad=='tokenizer':c['tokenizer_sha256']='f'*64
 elif bad=='rank':c['expected_layout']['rank']=2
 elif bad=='context':c['expected_layout']['num_hidden_layers']=5
 monkeypatch.setattr(poc.bundle,'load_bundle',lambda *a,**k:pytest.fail('unqualified model may not load'))
 with pytest.raises(ValueError):m.inspect_source_bundle(c)


@pytest.mark.parametrize('bad',['missing','hash','status','unqualified_code'])
def test_parent_context_denies_unqualified_sources_before_import(tmp_path,bad):
 m=api();c=config(tmp_path);root=tmp_path/'qualified-parent';qpath=root/'provenance/shared-rtn-frontier-v1/qualification.json';qpath.parent.mkdir(parents=True)
 c['parent_qualification']={'path':str(qpath),'sha256':'a'*64};c['serial_gate']['directory']=str(qpath.parent)
 if bad!='missing':
  qpath.write_text(json.dumps({'status':'failed' if bad=='status' else 'complete'}))
  if bad!='hash':c['parent_qualification']['sha256']=hashlib.sha256(qpath.read_bytes()).hexdigest()
 if bad=='unqualified_code':
  p=root/'provenance/sharing_frontier_study.py';p.write_text("raise AssertionError('unqualified code executed')")
 with pytest.raises(ValueError):m.verify_parent_context(c)
 assert not Path(c['out']).exists()


@pytest.mark.parametrize('mode',['profile','run','controls','analyze'])
def test_cli_cannot_report_success_for_unqualified_execution_inputs(tmp_path,mode):
 import subprocess,os
 c=config(tmp_path);path=yaml_config(tmp_path,c);driver=Path(__file__).resolve().parents[1]/'provenance/shared_calibration_study.py'
 result=subprocess.run([sys.executable,str(driver),'--config',str(path),'--mode',mode],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True)
 assert result.returncode!=0 and not Path(c['out']).exists()
 assert '"status": "complete"' not in result.stdout


def test_loaded_cpu_model_collects_and_saves_exact_costed_statistics(owned,tmp_path):
 from poc.recovery_controls import ControlRecorder
 from poc.shared_calibration_bundle import load_calibration
 m=api();rec=ControlRecorder();before=owned.layout.manifest().checkpoint_digest
 result=m.collect_and_save(owned.layout,owned.population,owned.windows,destination=tmp_path/'stats',study_contract_sha256='c'*64,max_statistics_bytes=1024**2,recorder=rec)
 loaded=load_calibration(result['statistics_ref'],expected_contract_sha256='c'*64)
 assert loaded.digest()==result['result_sha256'] and len(loaded.owners)==9
 assert loaded.counters['raw_target_presentations']==1022 and loaded.counters['projection_input_presentations']==12264
 assert sum(r.student_forward_calls for r in rec.rows)==2 and sum(r.raw_corpus_target_presentations for r in rec.rows)==1022
 assert sum(r.valid_target_presentations for r in rec.rows)==0 and sum(r.teacher_forward_calls for r in rec.rows)==0
 assert owned.layout.manifest().checkpoint_digest==before and all(p.grad is None for p in owned.layout.parameters())
 assert all(value>=0 for value in result['timings'].values())


def test_calibration_worker_rejects_wrong_role_before_forward(owned,tmp_path,monkeypatch):
 from dataclasses import replace
 from poc.recovery_controls import ControlRecorder
 m=api();rec=ControlRecorder();bad=replace(owned.population,roles={k:'dev' for k in owned.population.roles})
 monkeypatch.setattr(owned.layout,'final_hidden',lambda *a:pytest.fail('wrong role must not execute'))
 with pytest.raises(ValueError):m.collect_and_save(owned.layout,bad,owned.windows,destination=tmp_path/'stats',study_contract_sha256='c'*64,max_statistics_bytes=1024**2,recorder=rec)
 assert not (tmp_path/'stats').exists() and sum(r.student_forward_calls for r in rec.rows)==0


def test_interrupted_forward_is_precharged_and_leaves_no_accepted_stats(owned,tmp_path,monkeypatch):
 from poc.recovery_controls import ControlRecorder
 m=api();rec=ControlRecorder()
 def fail(*a):raise InterruptedError('owned partial forward')
 monkeypatch.setattr(owned.layout,'final_hidden',fail)
 with pytest.raises(InterruptedError):m.collect_and_save(owned.layout,owned.population,owned.windows,destination=tmp_path/'stats',study_contract_sha256='c'*64,max_statistics_bytes=1024**2,recorder=rec)
 assert sum(r.student_forward_calls for r in rec.rows)==1 and sum(r.raw_corpus_target_presentations for r in rec.rows)==511
 assert not (tmp_path/'stats').exists() and all('projection' not in layer.mlp.__dict__ for layer in owned.layout.layers)


def profile_packet(c):
 from poc.records import canonical_json
 return {'schema_version':1,'mode':'profile','study_contract_sha256':hashlib.sha256(canonical_json(c)).hexdigest(),
  'numerical_profile':'shared-inputs-cpu-fp32-moments-fp64-v1','canonical_owners':9,
  'counters':{'statistics_bytes':7872,'raw_target_presentations':511,'unique_context_target_positions':511,'unique_contexts':1,'processed_context_tokens':512,'operator_input_presentations':2044,'projection_input_presentations':6132},
  'queries':{'student_forward_calls':1,'teacher_forward_calls':0,'valid_target_presentations':0,'raw_corpus_target_presentations':511},
  'timings':{'setup_seconds':10.,'collection_seconds':20.,'save_seconds':3.,'reload_seconds':4.,'finalization_seconds':1.},
  'wall_seconds':38.,'process_peak_rss_bytes':100000000,'statistics_complete_bytes':10000,'retained_family_bytes':100000000}


def test_completed_calibration_schedule_forecast_prices_setup_every_context_and_reload(tmp_path):
 m=api();c=config(tmp_path);plan=m.forecast_calibration(profile_packet(c),c)
 assert plan['forecast_seconds']==58 and plan['resource']['wall_seconds']==240
 assert plan['queries']=={'student_forward_calls':2,'teacher_forward_calls':0,'valid_target_presentations':0,'raw_corpus_target_presentations':1022}
 assert plan['expected_counters']['operator_input_presentations']==4088 and plan['expected_counters']['projection_input_presentations']==12264
 assert plan['future_statistics_reserve_bytes']==10000+16*1024**2


@pytest.mark.parametrize('bad',['mode','contract','quality','nan','negative','wall_timing','query','scored','counts','stats_bytes','memory','wall_cap','disk_cap'])
def test_unqualified_or_infeasible_profile_cannot_freeze_a_calibration_run(tmp_path,bad):
 m=api();c=config(tmp_path);p=profile_packet(c)
 if bad=='mode':p['mode']='run'
 elif bad=='contract':p['study_contract_sha256']='f'*64
 elif bad=='quality':p['kl']=.2
 elif bad=='nan':p['timings']['collection_seconds']=float('nan')
 elif bad=='negative':p['timings']['reload_seconds']=-1
 elif bad=='wall_timing':p['wall_seconds']=1
 elif bad=='query':p['queries']['student_forward_calls']=2
 elif bad=='scored':p['queries']['valid_target_presentations']=511
 elif bad=='counts':p['counters']['projection_input_presentations']=511
 elif bad=='stats_bytes':p['statistics_complete_bytes']=1
 elif bad=='memory':p['process_peak_rss_bytes']=5*1024**3
 elif bad=='wall_cap':c['resource']['run']['wall_ceiling']=120
 elif bad=='disk_cap':c['resource']['run']['family_disk_bytes']=1000
 # Rebind the changed, still outcome-independent configuration for cap cases.
 if bad in ('wall_cap','disk_cap'):
  from poc.records import canonical_json
  p['study_contract_sha256']=hashlib.sha256(canonical_json(c)).hexdigest()
 with pytest.raises(ValueError):m.forecast_calibration(p,c)


def recorded_kwargs(tmp_path):
 c=config(tmp_path)
 return {'config':c,'mode':'profile','resource':{'wall_seconds':120,'memory_bytes':4*1024**3,'disk_bytes':32*1024**2},
         'source_commit':'a'*40,'source_files':{'owned-test-source':'b'*64},'configuration_sha256':'c'*64,'parent_run_id':'d'*64,'setup_seconds':.001}


def test_recorded_collection_persists_raw_and_operator_units_without_scored_loss(owned,tmp_path):
 from poc.shared_calibration_bundle import CalibrationRef,load_calibration
 m=api();kwargs=recorded_kwargs(tmp_path);packet=m.recorded_collection(owned.layout,owned.population,owned.windows[:1],**kwargs)
 assert packet['mode']=='profile' and packet['queries']=={'student_forward_calls':1,'teacher_forward_calls':0,'valid_target_presentations':0,'raw_corpus_target_presentations':511}
 assert packet['counters']['operator_input_presentations']==2044 and packet['counters']['projection_input_presentations']==6132
 assert packet['cost']['student_forward_calls']==1 and packet['cost']['raw_corpus_target_presentations']==511
 assert packet['cost']['student_backward_calls']==packet['cost']['teacher_forward_calls']==packet['cost']['valid_target_presentations']==0
 assert packet['statistics_complete_bytes']>=7872 and packet['process_peak_rss_bytes']>0
 assert load_calibration(CalibrationRef(**packet['statistics_ref']),expected_contract_sha256=packet['study_contract_sha256']).digest()==packet['result_sha256']
 assert (Path(kwargs['config']['out'])/'profile/packet.json').is_file()
 with pytest.raises(FileExistsError):m.recorded_collection(owned.layout,owned.population,owned.windows[:1],**kwargs)


@pytest.mark.parametrize('bad',['count','raw','mode','unbounded','nonfinite_setup'])
def test_recorded_collection_denials_precede_any_model_forward(owned,tmp_path,monkeypatch,bad):
 from dataclasses import replace
 import torch
 m=api();kwargs=recorded_kwargs(tmp_path);windows=owned.windows[:1]
 if bad=='count':windows=owned.windows
 elif bad=='raw':windows=(replace(windows[0],target_mask=torch.zeros_like(windows[0].target_mask)),)
 elif bad=='mode':kwargs['mode']='anything'
 elif bad=='unbounded':kwargs['resource']['wall_seconds']=float('inf')
 elif bad=='nonfinite_setup':kwargs['setup_seconds']=float('nan')
 monkeypatch.setattr(owned.layout,'final_hidden',lambda *a:pytest.fail('invalid plan must not forward'))
 with pytest.raises(ValueError):m.recorded_collection(owned.layout,owned.population,windows,**kwargs)
 assert not Path(kwargs['config']['out']).exists()


def test_recorded_interruption_retains_charged_attempt_and_no_accepted_packet(owned,tmp_path,monkeypatch):
 m=api();kwargs=recorded_kwargs(tmp_path)
 def fail(*a):raise InterruptedError('owned interrupted collection')
 monkeypatch.setattr(owned.layout,'final_hidden',fail)
 with pytest.raises(InterruptedError):m.recorded_collection(owned.layout,owned.population,owned.windows[:1],**kwargs)
 root=Path(kwargs['config']['out'])/'profile';assert not (root/'packet.json').exists()
 rows=[json.loads(line) for p in root.rglob('costs.jsonl') for line in p.read_text().splitlines()]
 assert sum(row['cost']['student_forward_calls'] for row in rows)==1
 assert sum(row['cost']['raw_corpus_target_presentations'] for row in rows)==511
 assert (root/'failed.json').is_file()


def test_cpu_source_copy_load_preserves_canonical_shared_outputs_without_forward_charges(owned,tmp_path):
 from poc.bundle import write_bundle,verify_bundle
 from poc.recovery_controls import ControlRecorder
 from test_export import runtime
 import torch
 m=api();ref=write_bundle(owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True);c=config(tmp_path);c['source_bundle']={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256};c['tokenizer_sha256']='a'*64
 rec=ControlRecorder();model,record=m.prepare_cpu_source(c,recorder=rec)
 assert record['source']==c['source_bundle'] and record['copied']['directory']==str(Path(c['out'])/'source-runtime-copy')
 assert verify_bundle(Path(record['copied']['directory'])).metadata['groups']==verify_bundle(Path(ref.directory)).metadata['groups']
 assert len(model.bank)==3 and model.layers[0].mlp._owners_ref() is model.bank and model.layers[2].mlp._owners_ref() is model.bank
 assert all(p.device.type=='cpu' and p.dtype==torch.float32 and p.grad is None for p in model.parameters())
 with torch.no_grad():assert torch.equal(model.final_hidden(owned.windows[0]),owned.layout.final_hidden(owned.windows[0]))
 assert sum(v.student_forward_calls+v.teacher_forward_calls for v in rec.rows)==0
 assert record['source_complete_bytes']>0 and record['copied_complete_bytes']>0


@pytest.mark.parametrize('bad',['changed_source','existing_copy'])
def test_cpu_source_stage_refuses_invalid_or_previously_owned_copy(owned,tmp_path,monkeypatch,bad):
 from poc.bundle import write_bundle
 from poc.recovery_controls import ControlRecorder
 from test_export import runtime
 import poc.bundle
 m=api();ref=write_bundle(owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True);c=config(tmp_path);c['source_bundle']={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256};c['tokenizer_sha256']='a'*64
 if bad=='changed_source':c['source_bundle']['manifest_sha256']='f'*64
 else:
  p=Path(c['out'])/'source-runtime-copy';p.mkdir(parents=True);(p/'keep').write_text('retain')
 monkeypatch.setattr(poc.bundle,'load_bundle',lambda *a,**k:pytest.fail('invalid source/copy must not load model'))
 with pytest.raises((ValueError,FileExistsError)):m.prepare_cpu_source(c,recorder=ControlRecorder())
 if bad=='existing_copy':assert (p/'keep').read_text()=='retain'


def cold_config(owned,population,tmp_path):
 from poc.bundle import write_bundle
 from test_export import runtime
 c,source=population;owned.layout.tokenizer_digest=c['tokenizer_sha256'];ref=write_bundle(owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True)
 full=config(tmp_path);full.update(c);full['source_bundle']={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256}
 return full


def cold_kwargs():
 return {'source_commit':'a'*40,'source_files':{'owned':'b'*64},'configuration_sha256':'c'*64,'parent_run_id':'d'*64}


def test_owned_cold_profile_and_completed_run_reuse_exact_source_and_full_data_plan(owned,population,tmp_path):
 m=api();c=cold_config(owned,population,tmp_path);profile=m.cold_cpu_collection(c,mode='profile',resource={'wall_seconds':120,'memory_bytes':4*1024**3,'disk_bytes':32*1024**2},**cold_kwargs())
 assert profile['data_provenance']['guarded_documents']==2 and profile['queries']['student_forward_calls']==1
 p=Path(c['out'])/'profile/packet.json';pref={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 plan=m.forecast_calibration(profile,c);result=m.cold_cpu_collection(c,mode='run',resource=plan['resource'],profile_reference=pref,**cold_kwargs())
 assert result['queries']['student_forward_calls']==2 and result['queries']['raw_corpus_target_presentations']==1022
 assert result['data_provenance']['window_plan_sha256']==profile['data_provenance']['window_plan_sha256']
 assert result['source_runtime_copy']==profile['source_runtime_copy'] and result['cpu_source_sha256']==profile['cpu_source_sha256']
 assert len(list(Path(c['out']).glob('source-runtime-copy')))==1


@pytest.mark.parametrize('bad',['missing_profile','wrong_resource','changed_profile'])
def test_cold_run_requires_frozen_matching_profile_before_data_or_model_load(tmp_path,monkeypatch,bad):
 m=api();c=config(tmp_path);resource={'wall_seconds':120,'memory_bytes':4*1024**3,'disk_bytes':32*1024**2};pref=None
 if bad!='missing_profile':
  p=tmp_path/'profile.json';p.write_text(json.dumps(profile_packet(c)));pref={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  if bad=='changed_profile':p.write_text('{}')
 monkeypatch.setattr(m,'prepare_windows',lambda *a:pytest.fail('invalid cold run must not read corpus'))
 monkeypatch.setattr(m,'prepare_cpu_source',lambda *a,**k:pytest.fail('invalid cold run must not load source'))
 with pytest.raises(ValueError):m.cold_cpu_collection(c,mode='run',resource=resource,profile_reference=pref,**cold_kwargs())
 assert not Path(c['out']).exists()


@pytest.mark.parametrize('bad',['missing','hash','status','contract','reader'])
def test_upstream_acceptance_denies_invalid_release_before_source_or_data_io(tmp_path,monkeypatch,bad):
 m=api();path=tmp_path/'acceptance.json';result=tmp_path/'result.json';reader=tmp_path/'verifier.py';reader.write_text('pass\n')
 reference={'path':str(path),'sha256':'a'*64};verifier={'path':str(reader),'sha256':hashlib.sha256(reader.read_bytes()).hexdigest()}
 if bad!='missing':
  value={'status':'failed' if bad=='status' else 'accepted','contract':{'path':str(tmp_path/'contract.json'),'sha256':'b'*64},'result':{'path':str(result),'sha256':'c'*64},'stage':{}}
  path.write_text(json.dumps(value))
  if bad!='hash':reference['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  if bad=='reader':reader.write_text('changed')
 monkeypatch.setattr(m,'prepare_windows',lambda *a:pytest.fail('no data access during release denial'))
 monkeypatch.setattr(m,'prepare_cpu_source',lambda *a,**k:pytest.fail('no model load during release denial'))
 with pytest.raises(ValueError):m.verify_upstream_acceptance(reference,expected_contract_sha256='d'*64,verifier_reference=verifier)


def test_execution_source_manifest_includes_untracked_driver_serializer_tests_and_config(tmp_path):
 m=api();c=config(tmp_path);path=yaml_config(tmp_path,c);sources=m.execution_sources(path)
 for name in ('provenance/shared_calibration_study.py','poc/shared_calibration_bundle.py','tests/test_shared_calibration_study.py','tests/test_shared_calibration_bundle.py'):
  p=Path(__file__).resolve().parents[1]/name;assert sources[str(p)]==hashlib.sha256(p.read_bytes()).hexdigest()
 assert sources[str(path)]==hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('payload',[{}, {'schema_version':1,'run_anyway':True}])
def test_unqualified_execution_registry_is_rejected_before_parent_or_corpus_reads(tmp_path,monkeypatch,payload):
 m=api();p=tmp_path/'execution.json';p.write_text(json.dumps(payload));c=config(tmp_path);config_path=yaml_config(tmp_path,c)
 monkeypatch.setattr(m,'verify_parent_context',lambda *a:pytest.fail('unqualified registry must not import parent'))
 monkeypatch.setattr(m,'prepare_windows',lambda *a:pytest.fail('unqualified registry must not read corpus'))
 with pytest.raises(ValueError):m.admit_execution_registry(p,config_path=config_path)
 assert not Path(c['out']).exists()


@pytest.mark.parametrize('bad',['missing','status','contract','sources'])
def test_own_qualification_cannot_be_replaced_with_partial_or_other_success(tmp_path,bad):
 m=api();c=config(tmp_path);config_path=yaml_config(tmp_path,c);p=tmp_path/'qualification.json';reference={'path':str(p),'sha256':'a'*64}
 if bad!='missing':
  value={'status':'failed' if bad=='status' else 'complete','contract_sha256':'f'*64,'source_files':{}}
  p.write_text(json.dumps(value));reference['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 with pytest.raises(ValueError):m.require_own_qualification(c,config_path=config_path,reference=reference)
 assert not Path(c['out']).exists()


def guard_context(tmp_path):
 m=api();c=config(tmp_path);return m,{'contract':c,'registry':{'provenance':str(tmp_path/'provenance')},'registry_ref':{'path':str(tmp_path/'execution.json'),'sha256':'a'*64},'refs':[],'config_path':tmp_path/'study.yaml'}


def test_direct_calibration_worker_refuses_missing_live_finite_guard(tmp_path):
 m,ctx=guard_context(tmp_path)
 with pytest.raises(ValueError):m.require_worker_guard(ctx,'profile')
 assert not Path(ctx['contract']['out']).exists()


def test_run_plan_requires_authenticated_freeze_before_following_profile_pointer(tmp_path,monkeypatch):
 m,ctx=guard_context(tmp_path);p=Path(ctx['contract']['out'])/'resource-freeze.json';p.parent.mkdir();p.write_text('{}')
 monkeypatch.setattr(m,'read_reference',lambda *a:pytest.fail('unbound plan must not be followed'))
 with pytest.raises(ValueError):m.read_frozen_plan(ctx,{})


def test_calibration_guard_accepts_real_kernel_python_parent_only(tmp_path):
 m=api();core=m.guard_core();stage=tmp_path/'stage';driver=Path(m.__file__)
 code="from pathlib import Path;import importlib.util;spec=importlib.util.spec_from_file_location('owned_driver',"+repr(str(driver))+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.verify_live_parent(Path("+repr(str(stage/'protocol.json'))+"));print('accepted')"
 record=core.execute_stage([sys.executable,'-c',code],root=driver.parents[1],output=tmp_path/'output',stage=stage,caps={'wall_seconds':30,'memory_bytes':1024**3,'disk_bytes':1024**2},frozen_files=[{'path':str(driver),'sha256':hashlib.sha256(driver.read_bytes()).hexdigest()}])
 core.require_stage_success(record);assert (stage/'execution.log').read_text().strip()=='accepted'


def test_matching_parent_json_without_real_supervisor_is_denied(tmp_path):
 import os
 m=api();p=tmp_path/'protocol.json';p.write_text('{}')
 with pytest.raises(ValueError):m.verify_live_parent(p)


def test_calibration_analysis_requires_bound_run_acceptance_before_stats_load(tmp_path,monkeypatch):
 import poc.shared_calibration_bundle
 m,ctx=guard_context(tmp_path);ctx['registry']['contract_sha256']='b'*64
 monkeypatch.setattr(poc.shared_calibration_bundle,'load_calibration',lambda *a,**k:pytest.fail('no unbound statistics read'))
 with pytest.raises(ValueError):m.analyze_collection(ctx,{})
 assert not Path(ctx['contract']['out']).exists()


def test_campaign_refuses_to_repeat_an_attempted_identity_before_worker_dispatch(tmp_path,monkeypatch):
 m,ctx=guard_context(tmp_path);p=Path(ctx['registry']['provenance']);p.mkdir();(p/'campaign-start.json').write_text('{}')
 monkeypatch.setattr(m,'guard_core',lambda *a:pytest.fail('existing campaign must fail before dispatch'))
 with pytest.raises(FileExistsError):m.run_campaign(ctx)
 assert not Path(ctx['contract']['out']).exists()


def test_source_metadata_inventory_never_reconstructs_schema_v2_model(owned,tmp_path,monkeypatch):
 from poc.bundle import write_bundle
 from test_export import runtime
 import poc.export
 m=api();ref=write_bundle(owned.layout,tmp_path/'source',runtime(tmp_path),elide_zero_residuals=True);c=config(tmp_path);c['source_bundle']={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256};c['tokenizer_sha256']='a'*64
 monkeypatch.setattr(poc.export,'_load_scientific',lambda *a,**k:pytest.fail('inventory must not reconstruct model'))
 value=m.inspect_source_bundle(c);assert value['complete_bytes']>0


def test_validate_rejects_bad_roles_before_any_asset_inventory(tmp_path,monkeypatch):
 m=api();c=config(tmp_path);path=yaml_config(tmp_path,c)
 monkeypatch.setattr(m,'verify_parent_context',lambda *a:{})
 def reject(*a):raise PermissionError('owned invalid role')
 monkeypatch.setattr(m,'inspect_fit_metadata',reject)
 monkeypatch.setattr(m,'inspect_source_bundle',lambda *a:pytest.fail('bad roles must precede asset work'))
 assert m.main(['--config',str(path),'--mode','validate'])==2
 assert not Path(c['out']).exists()


def test_interrupted_cold_setup_has_durable_common_attempt_and_costs(owned,population,tmp_path,monkeypatch):
 from poc.contracts import CostRecord
 import time
 m=api();c=cold_config(owned,population,tmp_path)
 def interrupt(*a,recorder,**k):
  recorder.record(CostRecord(wall_seconds=.01));raise InterruptedError('owned setup interruption after guarded document access')
 monkeypatch.setattr(m,'prepare_cpu_source',interrupt)
 with pytest.raises(InterruptedError):m.cold_cpu_collection(c,mode='profile',resource={'wall_seconds':120,'memory_bytes':4*1024**3,'disk_bytes':32*1024**2},**cold_kwargs())
 phase=Path(c['out'])/'profile';assert phase.is_dir() and (phase/'failed.json').is_file()
 rows=[json.loads(line) for p in phase.rglob('costs.jsonl') for line in p.read_text().splitlines()]
 assert rows and sum(row['cost']['wall_seconds'] for row in rows)>=.01
 assert sum(row['cost']['student_forward_calls'] for row in rows)==0
 assert (Path(c['out'])/'inputs/exposure.jsonl').is_file() and not (phase/'packet.json').exists()


def test_terminal_cost_is_bound_to_final_ledger_after_packet_writing(owned,tmp_path):
 m=api();kwargs=recorded_kwargs(tmp_path);packet=m.recorded_collection(owned.layout,owned.population,owned.windows[:1],**kwargs)
 cost=m.read_terminal_cost({'path':str(Path(kwargs['config']['out'])/'profile/terminal-cost.json'),'sha256':hashlib.sha256((Path(kwargs['config']['out'])/'profile/terminal-cost.json').read_bytes()).hexdigest()},expected_run_id=packet['run_id'])
 assert cost['totals']['wall_seconds']>=packet['cost']['wall_seconds']
 ledger=Path(cost['ledger']['path']);assert hashlib.sha256(ledger.read_bytes()).hexdigest()==cost['ledger']['sha256']
 assert cost['totals']['raw_corpus_target_presentations']==511


def test_owned_analysis_reports_terminal_ledgers_and_records_its_own_work(owned,population,tmp_path,monkeypatch):
 m=api();c=cold_config(owned,population,tmp_path);p=m.cold_cpu_collection(c,mode='profile',resource={'wall_seconds':120,'memory_bytes':4*1024**3,'disk_bytes':32*1024**2},**cold_kwargs());out=Path(c['out']);pref=m.file_reference(out/'profile/packet.json');plan=m.forecast_calibration(p,c)
 run=m.cold_cpu_collection(c,mode='run',resource=plan['resource'],profile_reference=pref,**cold_kwargs())
 ctx={'contract':c,'registry':{'contract_sha256':p['study_contract_sha256'],'config':{'sha256':'c'*64}},'registry_ref':{'path':str(tmp_path/'execution.json'),'sha256':'e'*64},'refs':[],'config_path':tmp_path/'study.yaml','sources':{'owned':'b'*64},'ancestry':{'source_run_id':'d'*64},'release':{'status':'owned fixture'}}
 core=m.guard_core();monkeypatch.setattr(core,'check_stage_binding',lambda *a,**k:None);monkeypatch.setattr(m,'guard_core',lambda:core)
 # Stage qualification is outside this owned analysis-unit test; statistics and ledgers are real.
 pcost=m.file_reference(out/'profile/terminal-cost.json');rcost=m.file_reference(out/'run/terminal-cost.json')
 freeze={'schema_version':1,'study_contract_sha256':p['study_contract_sha256'],'execution_freeze':ctx['registry_ref'],'profile':pref,'profile_stage':{},'profile_cost':pcost,'plan':plan};(out/'resource-freeze.json').write_text(json.dumps(freeze));fref=m.file_reference(out/'resource-freeze.json')
 acceptance={'schema_version':1,'execution_freeze':ctx['registry_ref'],'resource_freeze':fref,'packet':m.file_reference(out/'run/packet.json'),'terminal_cost':rcost,'stage':{}};(out/'run-acceptance.json').write_text(json.dumps(acceptance))
 refs=[pref,fref,pcost,rcost,m.file_reference(out/'run-acceptance.json'),acceptance['packet']]
 for r in (pcost,rcost):refs.append(json.loads(Path(r['path']).read_text())['ledger'])
 monkeypatch.setattr(m,'calibration_family',lambda *a:type('OwnedFamily',(),{'disk':lambda self:123456})())
 result=m.analyze_collection(ctx,{r['path']:r['sha256'] for r in refs})
 expected=m.read_terminal_cost(rcost,expected_run_id=run['run_id'])['totals'];assert result['statistics']['run']['cost']==expected
 assert result['analysis_cost_path']==str(out/'analysis/terminal-cost.json')
 analysis=json.loads((out/'analysis/terminal-cost.json').read_text());assert analysis['totals']['student_forward_calls']==analysis['totals']['raw_corpus_target_presentations']==0
 assert analysis['totals']['wall_seconds']>0

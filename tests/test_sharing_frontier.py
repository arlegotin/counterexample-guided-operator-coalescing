"""Owned complete-artifact controls for the new bounded RTN campaign."""
from dataclasses import replace
import hashlib,importlib.util,sys
from pathlib import Path
import pytest,torch
from test_export import model,runtime
from test_model_parity import batch
from poc.bundle import write_bundle,verify_bundle,load_bundle,measure_bundle


def api():
 path=Path(__file__).resolve().parents[1]/'provenance/sharing_frontier_study.py'
 assert path.is_file(),'explicit sharing frontier driver missing'
 spec=importlib.util.spec_from_file_location('sharing_frontier_driver',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def hashes(root):return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind',['dense','2'])
def test_refresh_keeps_every_nonruntime_byte_and_reloads_independently(tmp_path,kind):
 m=api();ref=write_bundle(model(kind),tmp_path/'source',runtime(tmp_path),elide_zero_residuals=kind=='2');root=Path(ref.directory);before=hashes(root)
 with torch.no_grad():expected=load_bundle(root,backend='torch').logits(batch()).clone()
 out=m.refresh_runtime(ref,tmp_path/'copy');target=Path(out.directory);after=hashes(target)
 assert hashes(root)==before and out.manifest_sha256==after['manifest.json']
 for name,h in before.items():
  if name not in ('manifest.json','runtime/poc-runtime.zip'):assert after[name]==h
 assert set(before)==set(after) and measure_bundle(target).payload_bytes==sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
 a,b=verify_bundle(root).metadata,verify_bundle(target).metadata
 assert {k:v for k,v in a.items() if k!='file_entries'}=={k:v for k,v in b.items() if k!='file_entries'}
 loaded=load_bundle(target,backend='torch');assert loaded.lm_head.weight is loaded.embed_tokens.weight
 with torch.no_grad():torch.testing.assert_close(loaded.logits(batch()),expected,rtol=0,atol=0)


@pytest.mark.parametrize('bad',['manifest_ref','source_byte','overwrite','child','ancestor'])
def test_repack_refuses_changed_sources_or_overlapping_outputs(tmp_path,bad):
 m=api();ref=write_bundle(model('dense'),tmp_path/'source',runtime(tmp_path));dest=tmp_path/'copy'
 if bad=='manifest_ref':ref=replace(ref,manifest_sha256='0'*64)
 elif bad=='source_byte':next((Path(ref.directory)/'weights').glob('*.safetensors')).write_bytes(b'owned corruption')
 elif bad=='overwrite':dest.mkdir();(dest/'marker').write_text('retain')
 elif bad=='child':dest=Path(ref.directory)/'inside'
 elif bad=='ancestor':dest=tmp_path
 with pytest.raises((ValueError,FileExistsError)):m.refresh_runtime(ref,dest)
 if bad=='overwrite':assert (dest/'marker').read_text()=='retain'
 assert not (tmp_path/'copy'/'manifest.json').exists()


def test_failed_copy_is_retained_and_never_published(tmp_path,monkeypatch):
 import shutil
 m=api();ref=write_bundle(model('dense'),tmp_path/'source',runtime(tmp_path));original=shutil.copytree
 def corrupt(source,destination,*args,**kwargs):
  value=original(source,destination,*args,**kwargs)
  if (Path(destination)/'manifest.json').exists():next((Path(destination)/'weights').glob('*.safetensors')).write_bytes(b'owned mixed copy')
  return value
 monkeypatch.setattr(shutil,'copytree',corrupt)
 with pytest.raises(ValueError):m.refresh_runtime(ref,tmp_path/'copy')
 assert not (tmp_path/'copy').exists() and list(tmp_path.glob('copy.failed-*'))


def test_each_precision_uses_its_original_float_source(tmp_path):
 m=api();rt=runtime(tmp_path);teacher=write_bundle(model('dense'),tmp_path/'teacher',rt);shared=write_bundle(model('2'),tmp_path/'shared',rt,elide_zero_residuals=True)
 jobs=m.rtn_jobs(teacher,shared,tmp_path/'packed')
 assert len(jobs)==6 and len({str(spec.destination) for _,_,spec in jobs})==6
 assert [(label,spec.bits) for label,_,spec in jobs]==[('dense_rtn8',8),('dense_rtn4',4),('dense_rtn2',2),('shared_rtn8',8),('shared_rtn4',4),('shared_rtn2',2)]
 for label,source,spec in jobs:
  assert source==(teacher if label.startswith('dense') else shared)
  assert spec.method=='rtn' and spec.group_size==64 and spec.embedding_bits==spec.bits
  assert Path(spec.destination).parent==tmp_path/'packed'


def windows():
 return [{'document_id':f'doc{i}','component_id':f'component{i}','tokens':[j%41 for j in range(512)],'attention_mask':[True]*512,'target_mask':[True]*511,'positions':list(range(512))} for i in range(128)]


@pytest.mark.parametrize('mode,components,targets,calls',[('profile',1,511,17),('run',128,65408,2176)])
def test_worker_budgets_count_hidden_plus_all_position_chunks(mode,components,targets,calls):
 counts=api().metric_work_plan(windows(),mode=mode)
 assert counts=={'components':components,'valid_target_presentations':targets,'teacher_forward_calls':calls,'student_forward_calls':calls}


def test_mask_holes_reduce_targets_but_not_full_context_processing():
 rows=windows();rows[0]['target_mask']=[j in (0,510) for j in range(511)]
 assert api().metric_work_plan(rows,mode='profile')=={'components':1,'valid_target_presentations':2,'teacher_forward_calls':17,'student_forward_calls':17}


@pytest.mark.parametrize('bad',['mode','population','duplicate_component','duplicate_doc','context','positions','mask_type','target_shape','empty_targets','padding'])
def test_invalid_metric_inventory_cannot_forecast_a_run(bad):
 rows=windows();mode='run'
 if bad=='mode':mode='best'
 elif bad=='population':rows.pop()
 elif bad=='duplicate_component':rows[1]['component_id']=rows[0]['component_id']
 elif bad=='duplicate_doc':rows[1]['document_id']=rows[0]['document_id']
 elif bad=='context':rows[0]['tokens'].pop()
 elif bad=='positions':rows[0]['positions'][0]=1
 elif bad=='mask_type':rows[0]['target_mask'][0]=1
 elif bad=='target_shape':rows[0]['target_mask'].append(True)
 elif bad=='empty_targets':rows[0]['target_mask']=[False]*511
 elif bad=='padding':rows[0]['attention_mask'][-1]=False
 with pytest.raises(ValueError):api().metric_work_plan(rows,mode=mode)


def profile():
 return {'mode':'profile','counts':{'valid_target_presentations':511,'teacher_forward_calls':17,'student_forward_calls':17},'context_metric_seconds':[2.0],'wall_seconds':10.0,'complete_bytes':1000,'memory':{'process_peak_rss_bytes':100,'torch_mps_driver_bytes':70,'mlx_peak_bytes':60}}


def test_completed_profile_schedule_includes_load_copy_and_every_component():
 result=api().forecast_worker(profile(),windows(),teacher_complete_bytes=1000,memory_bytes=110,output_disk_bytes=20*1024**2,wall_ceiling=1000)
 assert result['forecast_seconds']==264 and result['resource']=={'wall_seconds':660,'memory_bytes':110,'output_disk_bytes':20*1024**2,'valid_target_presentations':65408,'teacher_forward_calls':2176,'student_forward_calls':2176}
 assert result['required_copy_bytes']==2000+16*1024**2
 # The max of overlapping memory counters is100; adding them would be wrong.
 assert result['observed_memory_max_bytes']==100


@pytest.mark.parametrize('bad',['run_mode','quality_rows','profile_targets','profile_calls','negative_context','infinite_context','wall_before_context','insufficient_wall','insufficient_memory','insufficient_disk','boolean_cap','missing_memory'])
def test_unqualified_or_infeasible_profile_cannot_freeze_evaluation(bad):
 p=profile();kw=dict(teacher_complete_bytes=1000,memory_bytes=110,output_disk_bytes=20*1024**2,wall_ceiling=1000)
 if bad=='run_mode':p['mode']='run'
 elif bad=='quality_rows':p['rows']=[]
 elif bad=='profile_targets':p['counts']['valid_target_presentations']=510
 elif bad=='profile_calls':p['counts']['teacher_forward_calls']=16
 elif bad=='negative_context':p['context_metric_seconds']=[-1]
 elif bad=='infinite_context':p['context_metric_seconds']=[float('inf')]
 elif bad=='wall_before_context':p['wall_seconds']=1
 elif bad=='insufficient_wall':kw['wall_ceiling']=600
 elif bad=='insufficient_memory':kw['memory_bytes']=99
 elif bad=='insufficient_disk':kw['output_disk_bytes']=2000
 elif bad=='boolean_cap':kw['memory_bytes']=True
 elif bad=='missing_memory':p.pop('memory')
 with pytest.raises(ValueError):api().forecast_worker(p,windows(),**kw)


def contract(tmp_path):
 root=lambda name:str(tmp_path/name)
 digest='a'*64
 budget=dict(updates=0,valid_target_presentations=0,teacher_forward_calls=0,suffix_backward_calls=0,student_backward_calls=0,search_proposals=0,search_seconds=0,wall_seconds=900,disk_bytes=48*1024**3,memory_bytes=6*1024**3)
 return {'schema_version':1,'study':'owned-rtn-frontier','out':root('output'),
  'teacher':{'directory':root('teacher'),'manifest_sha256':digest},'shared':{'directory':root('shared'),'manifest_sha256':digest},
  'inputs':{'path':root('windows.json'),'sha256':digest,'population_manifest_sha256':digest},
  'qualified_source':{'path':root('qualification.json'),'sha256':digest},'worker':{'path':root('worker.py'),'sha256':digest},
  'reference':{'teacher_mps_digest':digest,'source_run_id':digest,'preparation':{'path':root('preparation.json'),'sha256':digest},'preparation_attempt':{'path':root('attempt.json'),'sha256':digest},'preparation_protocol':{'path':root('preparation-protocol.json'),'sha256':digest},'source_contract':{'path':root('source-contract.json'),'sha256':digest},'scientific_observation':{'path':root('0256.json'),'sha256':digest}},
  'serial_gate':{'directory':root('gate'),'registry_sha256':digest,'runner':root('runner.py'),'result_relative':'analysis/result.json','attempt_field':'analysis_attempt_sha256'},
  'resource':{'preparation':budget,'profile':{'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3},'evaluation':{'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'wall_ceiling':1800,'campaign_wall_seconds':7200,'family_disk_bytes':48*1024**3,'deadline_unix':2000000000}}}


def save_contract(tmp_path,value):
 import yaml
 path=tmp_path/'study.yaml';path.write_text(yaml.safe_dump(value));return path


def test_contract_is_closed_immutable_and_creates_no_output(tmp_path):
 c=contract(tmp_path);loaded=api().load_contract(save_contract(tmp_path,c))
 assert loaded['study']=='owned-rtn-frontier' and not (tmp_path/'output').exists()
 with pytest.raises(TypeError):loaded['resource']['profile']['wall_seconds']=1


@pytest.mark.parametrize('bad',['unknown','version','bool_version','relative_output','source_hash','unknown_input','unknown_resource','profile_bool','negative_wall','training_budget','zero_memory','reference_field','relative_gate','gate_escape'])
def test_closed_contract_rejects_unqualified_choices_before_io(tmp_path,bad):
 c=contract(tmp_path)
 if bad=='unknown':c['surprise']=True
 elif bad=='version':c['schema_version']=2
 elif bad=='bool_version':c['schema_version']=True
 elif bad=='relative_output':c['out']='relative/output'
 elif bad=='source_hash':c['shared']['manifest_sha256']='not-a-hash'
 elif bad=='unknown_input':c['inputs']['protected_override']=True
 elif bad=='unknown_resource':c['resource']['free_retry']=True
 elif bad=='profile_bool':c['resource']['profile']['wall_seconds']=True
 elif bad=='negative_wall':c['resource']['evaluation']['wall_ceiling']=-1
 elif bad=='training_budget':c['resource']['preparation']['updates']=1
 elif bad=='zero_memory':c['resource']['evaluation']['memory_bytes']=0
 elif bad=='reference_field':c['reference']['best_confirmation_checkpoint']=True
 elif bad=='relative_gate':c['serial_gate']['directory']='relative/gate'
 elif bad=='gate_escape':c['serial_gate']['result_relative']='../result.json'
 with pytest.raises(ValueError):api().load_contract(save_contract(tmp_path,c))
 assert not (tmp_path/'output').exists()


def accepted_source(tmp_path):
 """Owned metadata proof of a completed compiler; no pretrained claim."""
 import json,sys
 c=contract(tmp_path);source=tmp_path/'accepted';source.mkdir()
 c['teacher']['directory']=str(source/'artifacts/teacher_noop');c['shared']['directory']=str(source/'artifacts/shared_full17_float')
 def put(path,value):
  path.write_text(json.dumps(value,sort_keys=True));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 compiler=tmp_path/'owned-compiler.py';compiler.write_text('# Owned receipt-admission fixture only.\n')
 guard=tmp_path/'owned-guard.py';guard.write_text('# Owned receipt-admission fixture only.\n')
 files=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (compiler,guard)]
 source_contract={'study':'owned-source','population':{'role':'dev','purpose':'selection','components':128,'sha256':'a'*64},'teacher_mps_digest':'a'*64,'source_run_id':'a'*64,'frozen_files':files}
 c['reference']['source_contract']=put(tmp_path/'source-contract.json',source_contract)
 caps={'wall_seconds':10,'process_rss_bytes':10000,'family_disk_bytes':100000}
 protocol={'study':'owned-source:preparation','python':sys.executable,'worker':str(compiler),'root':str(tmp_path),'output':str(source),'provenance':str(tmp_path/'receipt'),'execution_env':{},'ancestors':[],'caps':caps,'frozen_files':files+[c['reference']['source_contract']]}
 c['reference']['preparation_protocol']=put(tmp_path/'preparation-protocol.json',protocol)
 attempt={'study':protocol['study'],'protocol_sha256':c['reference']['preparation_protocol']['sha256'],'supervisor_sha256':files[1]['sha256'],'caps':caps,'command':[sys.executable,str(compiler)],'exit_code':0,'failure':None,'elapsed_seconds':1,'sampled_peak_process_tree_rss_bytes':100,'sampled_family_output_peak_bytes':1000}
 (tmp_path/'receipt').mkdir();c['reference']['preparation_attempt']=put(tmp_path/'receipt/attempt.json',attempt)
 c['inputs']['path']=str(source/'inputs/windows.json')
 prep={'schema_version':1,'study_contract_sha256':c['reference']['source_contract']['sha256'],'components':128,'valid_targets':65408,'inputs':c['inputs'],'teacher_mps_digest':'a'*64,'artifacts':{'teacher_noop':dict(c['teacher'],complete_bytes=1000),'shared_full17_float':dict(c['shared'],complete_bytes=1000)}}
 c['reference']['preparation']=put(source/'preparation.json',prep)
 return c


def rewrite_reference(reference,change):
 import json
 p=Path(reference['path']);d=json.loads(p.read_text());change(d);p.write_text(json.dumps(d,sort_keys=True));reference['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()


def test_accepted_source_receipts_admit_declared_dev_inputs_without_opening_tokens(tmp_path):
 c=accepted_source(tmp_path)
 # Compiled token file intentionally does not exist. Admission checks metadata.
 proof=api().verify_source_receipts(c)
 assert proof['inputs']==c['inputs'] and proof['components']==128
 assert not Path(c['inputs']['path']).exists() and not (tmp_path/'output').exists()


@pytest.mark.parametrize('bad',['exit','cap','command','frozen_source','input_reference','prep_contract','artifact_reference','guard','role'])
def test_bad_source_receipts_fail_before_token_io(tmp_path,bad):
 c=accepted_source(tmp_path)
 if bad=='exit':rewrite_reference(c['reference']['preparation_attempt'],lambda d:d.update(exit_code=7))
 elif bad=='cap':rewrite_reference(c['reference']['preparation_attempt'],lambda d:d.update(elapsed_seconds=11))
 elif bad=='command':rewrite_reference(c['reference']['preparation_attempt'],lambda d:d.update(command=['wrong']))
 elif bad=='frozen_source':(tmp_path/'owned-compiler.py').write_text('changed')
 elif bad=='input_reference':c['inputs']['path']=str(tmp_path/'protected-token-file.json')
 elif bad=='prep_contract':rewrite_reference(c['reference']['preparation'],lambda d:d.update(study_contract_sha256='b'*64))
 elif bad=='artifact_reference':c['shared']['manifest_sha256']='c'*64
 elif bad=='guard':rewrite_reference(c['reference']['preparation_attempt'],lambda d:d.update(supervisor_sha256='d'*64))
 elif bad=='role':
  rewrite_reference(c['reference']['source_contract'],lambda d:d['population'].update(role='test'))
  rewrite_reference(c['reference']['preparation_protocol'],lambda d:d['frozen_files'][-1].update(c['reference']['source_contract']))
  rewrite_reference(c['reference']['preparation_attempt'],lambda d:d.update(protocol_sha256=c['reference']['preparation_protocol']['sha256']))
  rewrite_reference(c['reference']['preparation'],lambda d:d.update(study_contract_sha256=c['reference']['source_contract']['sha256']))
 with pytest.raises(ValueError):api().verify_source_receipts(c)
 assert not Path(c['inputs']['path']).exists()


def test_driver_uses_its_own_project_when_started_outside_checkout(tmp_path):
 import subprocess,os
 path=Path(__file__).resolve().parents[1]/'provenance/sharing_frontier_study.py'
 code="import importlib.util,pathlib; p="+repr(str(path))+"; s=importlib.util.spec_from_file_location('f',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(pathlib.Path(__import__('poc').__file__).resolve().parents[1])"
 r=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=''),text=True,capture_output=True)
 assert r.returncode==0 and r.stdout.strip()==str(path.parents[1]),r.stderr


def wired_inputs(tmp_path):
 import json
 from poc.runtime.torch_decoder import make_tiny_decoder,TinyConfig
 from poc.bank import make_bank,canonicalize_partition
 c=accepted_source(tmp_path);rt=runtime(tmp_path);teacher=make_tiny_decoder(TinyConfig(hidden_size=64,intermediate_size=128,max_position_embeddings=512),seed=3,dtype=torch.float32);teacher.tokenizer_digest='a'*64;shared=make_bank(teacher,canonicalize_partition([[0,2],[1]],3),8,17)
 for key,model_value in (('teacher',teacher),('shared',shared)):
  ref=write_bundle(model_value,Path(c[key]['directory']),rt,elide_zero_residuals=key=='shared');c[key]={'directory':ref.directory,'manifest_sha256':ref.manifest_sha256}
 rows=windows()
 for row in rows:row['window_sha256']=hashlib.sha256((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
 data={'schema_version':1,'role':'dev','purpose':'selection','population_manifest_sha256':'a'*64,'tokenizer_digest':'a'*64,'components':128,'windows':rows}
 path=Path(c['inputs']['path']);path.parent.mkdir();path.write_text(json.dumps(data));c['inputs']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 root=Path(__file__).resolve().parents[1];poc_files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'poc').rglob('*.py')}
 q=tmp_path/'qualification.json';q.write_text(json.dumps({'source_files':poc_files,'tests':{'default_cpu':{'passed':1,'guard':{'exit_code':0,'failure':None}}},'scope':'owned admission fixture, not pretrained qualification'}));c['qualified_source']={'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
 worker=root/'provenance/shared-rtn-frontier-v1/native-fidelity-worker-v1.py';c['worker']={'path':str(worker),'sha256':hashlib.sha256(worker.read_bytes()).hexdigest()}
 obs=tmp_path/'0256.json';obs.write_text('{}');c['reference']['scientific_observation']={'path':str(obs),'sha256':hashlib.sha256(obs.read_bytes()).hexdigest()}
 rewrite_reference(c['reference']['preparation'],lambda d:d.update(inputs=c['inputs'],artifacts={'teacher_noop':dict(c['teacher'],complete_bytes=measure_bundle(Path(c['teacher']['directory'])).payload_bytes),'shared_full17_float':dict(c['shared'],complete_bytes=measure_bundle(Path(c['shared']['directory'])).payload_bytes)}))
 return c


def test_source_inspection_checks_owned_bundles_and_exact_dev_inventory(tmp_path):
 c=wired_inputs(tmp_path);result=api().inspect_sources(c)
 assert len(result['windows'])==128 and result['run_work']['valid_target_presentations']==65408
 assert result['profile_work']['teacher_forward_calls']==17 and not (tmp_path/'output').exists()


@pytest.mark.parametrize('bad',['worker','qualification','compiled_hash','compiled_role','vocabulary','output_overlap'])
def test_source_inspection_denies_changed_or_wrong_role_inputs(tmp_path,bad):
 import json
 c=wired_inputs(tmp_path)
 if bad=='worker':c['worker']['sha256']='f'*64
 elif bad=='qualification':rewrite_reference(c['qualified_source'],lambda d:d['tests']['default_cpu']['guard'].update(exit_code=2))
 elif bad=='compiled_hash':Path(c['inputs']['path']).write_text('{}')
 elif bad in ('compiled_role','vocabulary'):
  p=Path(c['inputs']['path']);data=json.loads(p.read_text())
  if bad=='compiled_role':data['role']='test'
  else:data['windows'][0]['tokens'][0]=999
  p.write_text(json.dumps(data));c['inputs']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();rewrite_reference(c['reference']['preparation'],lambda d:d.update(inputs=c['inputs']))
 elif bad=='output_overlap':c['out']=c['teacher']['directory']
 with pytest.raises(ValueError):api().inspect_sources(c)


@pytest.mark.parametrize('exit_code',[0,7])
def test_guarded_owned_stage_retains_exact_protocol_and_exit(tmp_path,exit_code):
 command=[sys.executable,'-c',f'raise SystemExit({exit_code})']
 result=api().execute_stage(command,root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'stage',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[])
 assert result['exit_code']==exit_code and result['attempt']['exit_code']==exit_code
 assert (tmp_path/'stage/supervisor.log').is_file() and (tmp_path/'stage/protocol.json').is_file()
 assert result['attempt']['protocol_sha256']==hashlib.sha256((tmp_path/'stage/protocol.json').read_bytes()).hexdigest()
 with pytest.raises(FileExistsError):api().execute_stage(command,root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'stage',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[])


def test_changed_stage_source_cannot_execute(tmp_path):
 source=tmp_path/'source';source.write_text('before');ref={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()};source.write_text('after')
 with pytest.raises(ValueError):api().execute_stage([sys.executable,'-c',"raise RuntimeError('must not execute')"],root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'stage',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[ref])
 assert not (tmp_path/'stage/stage-start.json').exists()


def test_owned_stage_wall_failure_is_retained(tmp_path):
 result=api().execute_stage([sys.executable,'-c','import time; time.sleep(5)'],root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'stage',caps={'wall_seconds':.15,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[])
 assert result['exit_code']!=0 and result['attempt']['failure']=='stage_wall_cap'
 assert (tmp_path/'stage/attempt.json').exists()


def test_source_mutation_during_stage_is_rejected_after_receipt_is_retained(tmp_path):
 source=tmp_path/'source';source.write_text('before');ref={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 command=[sys.executable,'-c',"from pathlib import Path; Path("+repr(str(source))+").write_text('after')"]
 with pytest.raises(ValueError):api().execute_stage(command,root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'stage',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[ref])
 assert (tmp_path/'stage/attempt.json').exists()


def test_failed_guard_cannot_be_promoted_from_a_child_zero_exit(tmp_path):
 m=api();result=m.execute_stage([sys.executable,'-c','pass'],root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'stage',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[])
 assert m.require_stage_success(result) is None
 result['supervisor_exit_code']=120
 with pytest.raises(ValueError):m.require_stage_success(result)


def test_preparation_cannot_run_before_device_queue_gate(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path);before=set(sys.modules)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('no token/source I/O before serial admission'))
 with pytest.raises(ValueError):m.prepare_campaign(c)
 assert not Path(c['out']).exists() and ('mlx.core' not in sys.modules or 'mlx.core' in before)


def test_preparation_cannot_run_without_current_owned_controls(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path)
 monkeypatch.setattr(m,'require_serial_gate',lambda *a:None)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('controls must bind current code before preparing real artifacts'))
 with pytest.raises(ValueError):m.prepare_campaign(c)
 assert not Path(c['out']).exists()


def prepared_packet(c):
 labels=['teacher_noop','shared_float','dense_rtn8','dense_rtn4','dense_rtn2','shared_rtn8','shared_rtn4','shared_rtn2']
 return {'schema_version':1,'study_contract_sha256':api().contract_digest(c),'teacher_mps_digest':c['reference']['teacher_mps_digest'],'inputs':c['inputs'],'blocked_source_roots':[str(Path(__file__).resolve().parents[1])],
  'artifacts':{label:{'directory':str(Path(c['out'])/'artifacts'/label),'manifest_sha256':hashlib.sha256(label.encode()).hexdigest(),'runtime_sha256':'c'*64,'complete_bytes':1000,'teacher_needed_at_inference':False} for label in labels}}


def test_native_protocol_binds_original_sources_runtime_and_exact_work(tmp_path):
 m=api();c=contract(tmp_path);p=prepared_packet(c);resource={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':511,'teacher_forward_calls':17,'student_forward_calls':17}
 protocol=m.native_protocol(c,p,'shared_rtn4',windows(),mode='profile',resource=resource,output=tmp_path/'result.json',ledger=tmp_path/'costs.jsonl')
 assert protocol['artifact']['path'].endswith('/shared_rtn4') and protocol['teacher']['path'].endswith('/teacher_noop')
 assert protocol['numerical']=={'chunk_size':32,'temperature':1,'negative_floor':2e-6,'decision_resolution':2e-5}
 assert protocol['study_contract_sha256']==m.contract_digest(c) and protocol['resource']==resource and protocol['profile_binding'] is None


@pytest.mark.parametrize('bad',['runtime','contract','counts','profile_import','missing_run_profile','label'])
def test_native_protocol_cannot_mix_sources_or_change_estimand(tmp_path,bad):
 m=api();c=contract(tmp_path);p=prepared_packet(c);mode='profile';binding=None;label='shared_rtn4';resource={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':511,'teacher_forward_calls':17,'student_forward_calls':17}
 if bad=='runtime':p['artifacts'][label]['runtime_sha256']='d'*64
 elif bad=='contract':p['study_contract_sha256']='e'*64
 elif bad=='counts':resource['teacher_forward_calls']=16
 elif bad=='profile_import':binding={'path':str(tmp_path/'other.json'),'sha256':'a'*64}
 elif bad=='missing_run_profile':mode='run'
 elif bad=='label':label='selected_best'
 with pytest.raises(ValueError):m.native_protocol(c,p,label,windows(),mode=mode,resource=resource,output=tmp_path/'result.json',ledger=tmp_path/'costs.jsonl',profile_binding=binding)


def test_native_profile_acceptance_requires_exact_mode_counts_and_guard(tmp_path):
 m=api();c=contract(tmp_path);p=prepared_packet(c);r={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':511,'teacher_forward_calls':17,'student_forward_calls':17}
 protocol=m.native_protocol(c,p,'shared_rtn4',windows(),mode='profile',resource=r,output=tmp_path/'result.json',ledger=tmp_path/'ledger.jsonl')
 value={'schema_version':1,'protocol_sha256':m.contract_digest(protocol),'teacher_equivalent_control':protocol['artifact']['manifest_sha256']==protocol['teacher']['manifest_sha256'],'teacher_equivalent_passed':True if protocol['artifact']['manifest_sha256']==protocol['teacher']['manifest_sha256'] else None,'mode':'profile','study_contract_sha256':protocol['study_contract_sha256'],'artifact_manifest_sha256':protocol['artifact']['manifest_sha256'],'teacher_manifest_sha256':protocol['teacher']['manifest_sha256'],'runtime_sha256':protocol['runtime_sha256'],'worker_sha256':protocol['worker_sha256'],'inputs_sha256':protocol['inputs']['sha256'],'counts':{k:r[k] for k in ('valid_target_presentations','teacher_forward_calls','student_forward_calls')},'context_metric_seconds':[1.0],'wall_seconds':2.0,'complete_bytes':1000,'memory':{'process_peak_rss_bytes':100,'torch_mps_driver_bytes':0,'mlx_peak_bytes':10}}
 assert m.check_native_result(protocol,value) is None
 value['rows']=[]
 with pytest.raises(ValueError):m.check_native_result(protocol,value)


@pytest.mark.parametrize('bad',['manifest','runtime','counts','numerical_noop'])
def test_native_result_rejects_mixed_identity_or_incomplete_work(tmp_path,bad):
 m=api();c=contract(tmp_path);p=prepared_packet(c);r={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':511,'teacher_forward_calls':17,'student_forward_calls':17}
 protocol=m.native_protocol(c,p,'teacher_noop',windows(),mode='profile',resource=r,output=tmp_path/'result.json',ledger=tmp_path/'ledger.jsonl')
 value={'schema_version':1,'protocol_sha256':m.contract_digest(protocol),'teacher_equivalent_control':protocol['artifact']['manifest_sha256']==protocol['teacher']['manifest_sha256'],'teacher_equivalent_passed':True if protocol['artifact']['manifest_sha256']==protocol['teacher']['manifest_sha256'] else None,'mode':'profile','study_contract_sha256':protocol['study_contract_sha256'],'artifact_manifest_sha256':protocol['artifact']['manifest_sha256'],'teacher_manifest_sha256':protocol['teacher']['manifest_sha256'],'runtime_sha256':protocol['runtime_sha256'],'worker_sha256':protocol['worker_sha256'],'inputs_sha256':protocol['inputs']['sha256'],'counts':{k:r[k] for k in ('valid_target_presentations','teacher_forward_calls','student_forward_calls')},'teacher_equivalent_control':True,'teacher_equivalent_passed':True}
 if bad=='manifest':value['artifact_manifest_sha256']='d'*64
 elif bad=='runtime':value['runtime_sha256']='e'*64
 elif bad=='counts':value['counts']['valid_target_presentations']=510
 elif bad=='numerical_noop':value['teacher_equivalent_passed']=False
 with pytest.raises(ValueError):m.check_native_result(protocol,value)


@pytest.mark.parametrize('mode',['profile','run'])
def test_campaign_entry_does_not_advance_before_serial_admission(tmp_path,monkeypatch,mode):
 m=api();c=contract(tmp_path)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('serial admission must precede source reads'))
 with pytest.raises(ValueError):getattr(m,mode+'_campaign')(c,config_path=tmp_path/'study.yaml')
 assert not Path(c['out']).exists()


def test_run_refuses_missing_frozen_profiles_before_token_reads(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path)
 monkeypatch.setattr(m,'require_serial_gate',lambda *a:None);monkeypatch.setattr(m,'require_own_controls',lambda *a:None)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('no evaluation input reads without a profile freeze'))
 with pytest.raises(ValueError):m.run_campaign(c,config_path=tmp_path/'study.yaml')
 assert not Path(c['out']).exists()


def test_changed_frozen_evaluation_is_rejected_before_dispatch(tmp_path,monkeypatch):
 import json
 m=api();c=contract(tmp_path);p=prepared_packet(c);resource={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':65408,'teacher_forward_calls':2176,'student_forward_calls':2176};binding={'path':str(tmp_path/'profile.json'),'sha256':'a'*64}
 protocol=m.native_protocol(c,p,'shared_rtn4',windows(),mode='run',resource=resource,output=Path(c['out'])/'workers/run-shared_rtn4/result.json',ledger=Path(c['out'])/'workers/run-shared_rtn4/costs.jsonl',profile_binding=binding)
 protocol['artifact']['manifest_sha256']='e'*64;path=tmp_path/'planned.json';path.write_text(json.dumps(protocol));ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 monkeypatch.setattr(m,'execute_stage',lambda *a,**k:pytest.fail('must check frozen estimand before any native work'))
 with pytest.raises(ValueError):m.native_job(c,p,'shared_rtn4',windows(),mode='run',resource=resource,config_path=tmp_path/'study.yaml',profile_binding=binding,expected_protocol_ref=ref)
 assert not Path(c['out']).exists()


def test_validate_cli_is_operational_and_creates_no_study_output(tmp_path):
 import subprocess,os,json
 c=wired_inputs(tmp_path);path=save_contract(tmp_path,c);driver=Path(__file__).resolve().parents[1]/'provenance/sharing_frontier_study.py'
 result=subprocess.run([sys.executable,str(driver),'--config',str(path),'--mode','validate'],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 value=json.loads(result.stdout);assert value['mode']=='validate' and value['components']==128 and value['valid_targets']==65408 and value['native_work'] is False
 assert not Path(c['out']).exists()


def test_internal_prepare_cannot_be_called_without_its_guard(tmp_path):
 c=contract(tmp_path)
 with pytest.raises(ValueError):api().require_preparation_guard(c,tmp_path/'study.yaml')
 assert not Path(c['out']).exists()


def analysis_fixture(tmp_path):
 m=api();c=contract(tmp_path);prepared=prepared_packet(c);rows=windows()
 for i,row in enumerate(rows):row['window_sha256']=hashlib.sha256(str(i).encode()).hexdigest()
 protocols={};results={};scientific={'step':256,'populations':{'tuning':[]}}
 means={'teacher_noop':0.,'shared_float':.1,'dense_rtn8':.002,'dense_rtn4':.2,'dense_rtn2':3.,'shared_rtn8':.11,'shared_rtn4':.3,'shared_rtn2':4.}
 sizes={'teacher_noop':541000000,'shared_float':531000000,'dense_rtn8':155000000,'dense_rtn4':88000000,'dense_rtn2':61000000,'shared_rtn8':153000000,'shared_rtn4':86000000,'shared_rtn2':60000000}
 for label in m.POINT_LABELS:
  prepared['artifacts'][label]['complete_bytes']=sizes[label]
  r={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':65408,'teacher_forward_calls':2176,'student_forward_calls':2176}
  p=m.native_protocol(c,prepared,label,rows,mode='run',resource=r,output=tmp_path/(label+'.json'),ledger=tmp_path/(label+'.jsonl'),profile_binding={'path':str(tmp_path/'profile'),'sha256':'a'*64});protocols[label]=p
  value={'schema_version':1,'mode':'run','study_contract_sha256':p['study_contract_sha256'],'protocol_sha256':m.contract_digest(p),'artifact_manifest_sha256':p['artifact']['manifest_sha256'],'teacher_manifest_sha256':p['teacher']['manifest_sha256'],'runtime_sha256':p['runtime_sha256'],'worker_sha256':p['worker_sha256'],'inputs_sha256':p['inputs']['sha256'],'teacher_equivalent_control':label=='teacher_noop','teacher_equivalent_passed':True if label=='teacher_noop' else None,'counts':{k:r[k] for k in ('valid_target_presentations','teacher_forward_calls','student_forward_calls')},'complete_bytes':sizes[label],'rows':[]}
  for row in rows:
   value['rows'].append({**{k:row[k] for k in ('document_id','component_id','window_sha256')},'study_contract_sha256':p['study_contract_sha256'],'protocol_sha256':m.contract_digest(p),'valid_targets':511,'kl_sum':means[label]*511,'nll_sum':3.*511,'teacher_nll_sum':2.9*511,'disagreements':0 if label=='teacher_noop' else 10})
  results[label]=value
 for row in rows:scientific['populations']['tuning'].append({**{k:row[k] for k in ('document_id','component_id','window_sha256')},'target_positions':list(range(511)),'full':{'status':'ok','kl':.1*511,'nll':3.*511,'valid_targets':511}})
 return prepared,protocols,results,rows,scientific


def test_full_component_results_are_admitted_before_analysis(tmp_path):
 m=api();_,protocols,results,rows,_=analysis_fixture(tmp_path)
 assert m.check_native_result(protocols['shared_rtn4'],results['shared_rtn4'],rows=rows) is None


@pytest.mark.parametrize('bad',['missing_component','reorder','mask_count','negative_floor','nan','disagreement','source_window'])
def test_native_full_result_rejects_incomplete_or_changed_population(tmp_path,bad):
 m=api();_,protocols,results,rows,_=analysis_fixture(tmp_path);v=results['shared_rtn4']
 if bad=='missing_component':v['rows'].pop()
 elif bad=='reorder':v['rows'].reverse()
 elif bad=='mask_count':v['rows'][0]['valid_targets']=510
 elif bad=='negative_floor':v['rows'][0]['kl_sum']=-.002
 elif bad=='nan':v['rows'][0]['nll_sum']=float('nan')
 elif bad=='disagreement':v['rows'][0]['disagreements']=512
 elif bad=='source_window':v['rows'][0]['window_sha256']='d'*64
 with pytest.raises(ValueError):m.check_native_result(protocols['shared_rtn4'],v,rows=rows)


def test_frontier_retains_dominated_points_raws_and_fixed_adverse_primary(tmp_path):
 m=api();args=analysis_fixture(tmp_path);out=m.summarize_frontier(*args)
 assert set(out['points'])==set(m.POINT_LABELS)
 assert out['primary']['label']=='shared_rtn4_minus_dense_rtn4'
 assert out['primary']['kl_difference']==pytest.approx(.1)
 assert out['primary']['paired_component_percentile_95']==pytest.approx([.1,.1])
 assert out['primary']['better_components']==0 and out['primary']['worse_components']==128
 assert out['points']['shared_float']['dominated_by']==['dense_rtn8']
 assert out['points']['shared_rtn2']['within_named_ceiling'] is True
 assert out['points']['shared_float']['within_named_ceiling'] is False
 assert out['points']['shared_rtn4']['raw_sums']['kl_sum']==pytest.approx(19622.4)
 assert len(out['points']['shared_rtn4']['worst_components'])==10
 assert out['scope']['capability_acceptability_established'] is False
 assert out['scope']['learned_dense_quantization_qualified'] is False
 assert out['float_source_parity']['max_abs_component_kl_difference']==0


@pytest.mark.parametrize('bad',['missing_point','changed_bytes','teacher_targets','scientific_window','scientific_mask','float_parity'])
def test_frontier_refuses_missing_points_or_invalid_reload_parity(tmp_path,bad):
 m=api();p,protocols,v,rows,s=analysis_fixture(tmp_path)
 if bad=='missing_point':v.pop('shared_rtn2')
 elif bad=='changed_bytes':v['shared_rtn2']['complete_bytes']+=1
 elif bad=='teacher_targets':v['shared_rtn4']['rows'][0]['teacher_nll_sum']+=1
 elif bad=='scientific_window':s['populations']['tuning'][0]['window_sha256']='f'*64
 elif bad=='scientific_mask':s['populations']['tuning'][0]['target_positions'].pop()
 elif bad=='float_parity':v['shared_float']['rows'][0]['kl_sum']+=.1
 with pytest.raises(ValueError):m.summarize_frontier(p,protocols,v,rows,s)


def test_analysis_entry_refuses_unaccepted_evaluation_before_source_reads(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path)
 monkeypatch.setattr(m,'require_serial_gate',lambda *a:None);monkeypatch.setattr(m,'require_own_controls',lambda *a:None)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('no input reads before accepted evaluation'))
 with pytest.raises(ValueError):m.analyze_campaign(c,config_path=tmp_path/'study.yaml')
 assert not Path(c['out']).exists()


@pytest.mark.parametrize('bad',['deadline','failed_upstream','changed_registry'])
def test_serial_queue_rejects_failure_or_changed_identity_without_starting_native_work(tmp_path,bad):
 import json,time
 m=api();c=contract(tmp_path);gate=Path(c['serial_gate']['directory']);gate.mkdir();reg=gate/'registry.json';reg.write_text('{}');c['serial_gate']['registry_sha256']=hashlib.sha256(reg.read_bytes()).hexdigest()
 refs=[{'path':str(reg),'sha256':c['serial_gate']['registry_sha256']}];deadline=time.time()+10
 if bad=='deadline':deadline=time.time()-1
 elif bad=='failed_upstream':(gate/'watch-failed.json').write_text('{}')
 elif bad=='changed_registry':reg.write_text('{"altered":true}')
 with pytest.raises(ValueError):m.wait_for_serial(c,refs,deadline=deadline)
 assert not Path(c['out']).exists()


def test_queue_cannot_freeze_without_current_full_qualification(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path)
 monkeypatch.setattr(m,'require_own_controls',lambda *a:None)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('full qualification required before staging science'))
 with pytest.raises(ValueError):m.freeze_queue(c,config_path=tmp_path/'study.yaml')
 assert not Path(c['out']).exists()


import json,shutil,types
from poc.quantize import _save_packed


@pytest.mark.parametrize('phase',['prepare','profile','run','analyze'])
def test_direct_scientific_phase_requires_full_qualification(tmp_path,monkeypatch,phase):
 m=api();c=contract(tmp_path)
 monkeypatch.setattr(m,'require_serial_gate',lambda *a:None);monkeypatch.setattr(m,'require_own_controls',lambda *a:None)
 monkeypatch.setattr(m,'inspect_sources',lambda *a:pytest.fail('full qualification must precede scientific source I/O'))
 method=getattr(m,'prepare_campaign' if phase=='prepare' else phase+'_campaign')
 with pytest.raises(ValueError,match='qualification'):
  method(c,**({} if phase=='prepare' else {'config_path':tmp_path/'study.yaml'}))
 assert not Path(c['out']).exists()


def test_failed_real_serializer_copy_is_quarantined_before_cleanup(tmp_path,monkeypatch):
 import poc.bundle
 m=api();ref=write_bundle(model('dense'),tmp_path/'source',runtime(tmp_path));source=Path(ref.directory);manifest=verify_bundle(source)
 fake=types.ModuleType('mlx.core');monkeypatch.setitem(sys.modules,'mlx.core',fake)
 parent=types.ModuleType('mlx');parent.core=fake;monkeypatch.setitem(sys.modules,'mlx',parent)
 def fail(*a,**k):raise RuntimeError('owned injected packing failure')
 monkeypatch.setattr(poc.bundle,'_write_runtime',fail)
 original=shutil.rmtree;preexisting=tmp_path/'.quantized-preexisting';preexisting.mkdir();(preexisting/'marker').write_text('retain old')
 with pytest.raises(RuntimeError,match='owned injected'):
  with m.retain_packing_failures(tmp_path/'packed'):_save_packed(source,manifest,tmp_path/'packed',{}, {},'rtn',None)
 assert shutil.rmtree is original and (preexisting/'marker').read_text()=='retain old'
 retained=list(tmp_path.glob('packed.failed-*'));assert len(retained)==1
 assert (retained[0]/'manifest.json').is_file() and (retained[0]/'weights').is_dir()
 assert not (retained[0]/'runtime').exists() # Ordinary runtime deletion still executes.
 note=json.loads((retained[0]/'packing-failure.json').read_text());assert note['status']=='failed' and 'owned injected' in note['error']
 assert not (tmp_path/'packed').exists()


def test_packing_success_and_unrelated_cleanup_keep_normal_semantics(tmp_path):
 m=api();original=shutil.rmtree;unrelated=tmp_path/'unrelated';unrelated.mkdir()
 preexisting=tmp_path/'.quantized-existing';preexisting.mkdir()
 with m.retain_packing_failures(tmp_path/'packed'):
  stage=tmp_path/'.quantized-owned';stage.mkdir();(stage/'value').write_text('complete');stage.rename(tmp_path/'packed')
  shutil.rmtree(unrelated);shutil.rmtree(preexisting)
 assert shutil.rmtree is original and not unrelated.exists() and not preexisting.exists()
 assert (tmp_path/'packed/value').read_text()=='complete' and not list(tmp_path.glob('packed.failed-*'))


def test_preparation_rejects_unrelated_successful_guard(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path);out=Path(c['out']);out.mkdir();path=out/'preparation.json'
 ref=write_bundle(model('dense'),tmp_path/'tiny',runtime(tmp_path));prepared=prepared_packet(c)
 for row in prepared['artifacts'].values():row.update(directory=ref.directory,manifest_sha256=ref.manifest_sha256)
 m.write_exclusive(path,prepared)
 stage=m.execute_stage([sys.executable,'-c','pass'],root=tmp_path,output=tmp_path/'unrelated',stage=tmp_path/'guard',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[])
 m.write_exclusive(out/'preparation-acceptance.json',{'stage':stage,'output':{'path':str(path),'sha256':m.file_sha256(path)}})
 monkeypatch.setattr(m,'execution_references',lambda *a:[])
 with pytest.raises(ValueError):m.read_prepared(c)


def test_native_guard_must_execute_exact_indexed_protocol(tmp_path):
 m=api();inner=tmp_path/'native.json';inner.write_text('{}');ref={'path':str(inner),'sha256':m.file_sha256(inner)}
 stage=m.execute_stage([sys.executable,'-c','pass'],root=tmp_path,output=tmp_path/'scratch',stage=tmp_path/'guard',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},frozen_files=[ref])
 with pytest.raises(ValueError):m.check_stage_binding(stage,command=[sys.executable,'worker.py','--protocol',str(inner)],root=tmp_path,output=tmp_path/'scratch',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},required_refs=[ref])

 assert m.check_stage_binding(stage,command=[sys.executable,'-c','pass'],root=tmp_path,output=tmp_path/'scratch',caps={'wall_seconds':10,'memory_bytes':512*1024**2,'disk_bytes':1024**2},required_refs=[ref]) is None


def test_successful_real_serializer_publishes_without_retained_failure(tmp_path,monkeypatch):
 import torch
 from safetensors.torch import load_file,save_file
 m=api();ref=write_bundle(model('dense'),tmp_path/'source',runtime(tmp_path));source=Path(ref.directory);manifest=verify_bundle(source);values={}
 for filename in {o['file'] for o in manifest.metadata['owners']}:values.update(load_file(source/filename))
 fake=types.ModuleType('mlx.core');fake.bfloat16=torch.bfloat16;fake.eval=lambda *a:None;fake.save_safetensors=lambda path,tensors:save_file(tensors,path)
 monkeypatch.setitem(sys.modules,'mlx.core',fake);parent=types.ModuleType('mlx');parent.core=fake;monkeypatch.setitem(sys.modules,'mlx',parent)
 original=shutil.rmtree
 with m.retain_packing_failures(tmp_path/'packed'):result=_save_packed(source,manifest,tmp_path/'packed',values,{},'rtn',None)
 assert shutil.rmtree is original and Path(result.bundle.directory)==tmp_path/'packed'
 assert verify_bundle(tmp_path/'packed') and (tmp_path/'packed/runtime/poc-runtime.zip').is_file()
 assert not list(tmp_path.glob('packed.failed-*')) and not list(tmp_path.glob('.quantized-*'))


def test_retention_never_follows_symlinks_and_restores_cleanup(tmp_path):
 m=api();real=tmp_path/'important';real.mkdir();(real/'value').write_text('keep');link=tmp_path/'.quantized-link';original=shutil.rmtree
 with pytest.raises(OSError):
  with m.retain_packing_failures(tmp_path/'packed'):
   link.symlink_to(real,target_is_directory=True);shutil.rmtree(link)
 assert shutil.rmtree is original and (real/'value').read_text()=='keep' and link.is_symlink()


def test_retention_restores_cleanup_if_quarantine_itself_fails(tmp_path,monkeypatch):
 m=api();original=shutil.rmtree;rename=Path.rename
 def fail(path,target):
  if path.name=='.quantized-owned':raise OSError('owned quarantine failure')
  return rename(path,target)
 monkeypatch.setattr(Path,'rename',fail)
 with pytest.raises(OSError,match='owned quarantine'):
  with m.retain_packing_failures(tmp_path/'packed'):
   stage=tmp_path/'.quantized-owned';stage.mkdir();shutil.rmtree(stage)
 assert shutil.rmtree is original and stage.is_dir()


def test_native_acceptance_consumer_rejects_swapped_successful_guard(tmp_path,monkeypatch):
 m=api();c=contract(tmp_path);prepared=prepared_packet(c);out=Path(c['out']);case=out/'workers/run-shared_rtn4';protocol_path=out/'protocols/run-shared_rtn4.json'
 resource={'wall_seconds':300,'memory_bytes':6*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':65408,'teacher_forward_calls':2176,'student_forward_calls':2176}
 _,_,results,rows,_=analysis_fixture(tmp_path)
 protocol=m.native_protocol(c,prepared,'shared_rtn4',rows,mode='run',resource=resource,output=case/'result.json',ledger=case/'costs.jsonl',profile_binding={'path':str(tmp_path/'profile.json'),'sha256':'a'*64})
 m.write_exclusive(protocol_path,protocol);ref={'path':str(protocol_path),'sha256':m.file_sha256(protocol_path)};m.write_exclusive(out/'preparation.json',prepared)
 result=results['shared_rtn4'];result['protocol_sha256']=m.contract_digest(protocol)
 for row in result['rows']:row['protocol_sha256']=m.contract_digest(protocol)
 m.write_exclusive(case/'result.json',result)
 stage=m.execute_stage([sys.executable,'-c','pass'],root=Path('/private/tmp'),output=case,stage=tmp_path/'unrelated-stage',caps={'wall_seconds':330,'memory_bytes':6*1024**3,'disk_bytes':2*1024**3},frozen_files=[ref],execution_env={'PYTHONPATH':''})
 item={'stage':stage,'protocol':ref,'output':{'path':str(case/'result.json'),'sha256':m.file_sha256(case/'result.json')}}
 monkeypatch.setattr(m,'execution_references',lambda *a:[])
 with pytest.raises(ValueError):m.accept_native_job(c,prepared,item,'shared_rtn4',rows,mode='run',config_path=tmp_path/'study.yaml',expected_protocol_ref=ref)

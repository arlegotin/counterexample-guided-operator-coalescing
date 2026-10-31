"""Independent restore/direct-entry gates for measured dense preparation."""
from pathlib import Path
import pytest
from test_dense_frontier import execution_driver


@pytest.fixture
def live_packet():
 root='/private/tmp/owned-dense';output=root+'/artifacts/study';stage=output+'/stages/prepare-owned'
 caps={'wall_seconds':300,'memory_bytes':8*1024**3,'disk_bytes':12*1024**3}
 refs=[{'path':root+'/source.py','sha256':'a'*64},{'path':root+'/measured-profile.json','sha256':'b'*64}]
 env={'PYTHONPATH':root,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'}
 command=['/owned/python',root+'/campaign.py','--internal-phase','prepare-owned']
 protocol={'root':root,'output':output,'provenance':stage,'command':command,'caps':caps,'frozen_files':list(refs),
  'supervisor_sha256':'a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba',
  'execution_env':{'HF_HUB_OFFLINE':None,'PYTORCH_ENABLE_MPS_FALLBACK':None,**env}}
 return protocol,dict(command=command,root=Path(root),output=Path(output),provenance=Path(stage),caps=caps,required_refs=refs,execution_env=env)


def test_review_r2_live_guard_accepts_exact_source_measured_caps_and_environment(live_packet):
 m=execution_driver();p,kwargs=live_packet;m.validate_live_dense_guard(p,**kwargs)


@pytest.mark.parametrize('change',['source_omission','measured_ref_omission','environment','caps','provenance','supervisor'])
def test_review_r2_live_guard_rejects_incomplete_or_changed_closure(live_packet,change):
 m=execution_driver();p,kwargs=live_packet
 if change=='source_omission':p['frozen_files']=p['frozen_files'][1:]
 elif change=='measured_ref_omission':p['frozen_files']=p['frozen_files'][:1]
 elif change=='environment':p['execution_env']['PYTORCH_ENABLE_MPS_FALLBACK']='1'
 elif change=='caps':p['caps']={**p['caps'],'wall_seconds':301}
 elif change=='provenance':p['provenance']+='/other'
 else:p['supervisor_sha256']='0'*64
 with pytest.raises(ValueError):m.validate_live_dense_guard(p,**kwargs)


def test_review_r2_full_preparation_cannot_invent_a_resource_plan(tmp_path):
 from poc.dense_frontier import dense_recipes
 m=execution_driver();c={'out':str(tmp_path/'unattempted')}
 with pytest.raises((ValueError,FileNotFoundError)):
  m.admit_preparation_freeze(c,dense_recipes()[0],config_path=tmp_path/'proposal.yaml',required_refs=[],admissions={})
 assert not Path(c['out']).exists()


@pytest.fixture
def saved_preparation_profile(tmp_path,monkeypatch):
 from poc.dense_frontier import dense_recipes,dense_preparation_plan,NUMERICS,PREPARATION_LIMITS,REFERENCE_LABELS
 from poc.contracts import CostRecord
 from poc.records import canonical_json
 from provenance import sharing_frontier_study as core
 import sys,json
 m=execution_driver();root=tmp_path/'owned';root.mkdir();out=root/'artifacts/dense-baseline-expansion-v2'
 ref=lambda p:m.reference(p)
 dummy=root/'dummy.json';dummy.write_text('{}');r=ref(dummy)
 c={'schema_version':2,'study':'dense-baseline-expansion-v2','out':str(out),'resource_freeze':None,
    'reference_artifacts':{'teacher_noop':{'directory':str(root/'teacher'),'manifest_sha256':'a'*64}},
    'preparation_limits':dict(PREPARATION_LIMITS)}
 config=root/'proposal.yaml';config.write_text('owned metadata fixture')
 script=root/'provenance/dense_frontier_campaign.py';script.parent.mkdir();script.write_text('pass')
 recipe=next(x for x in dense_recipes() if x.method=='dwq' and x.bits==4);phase='preprofile-'+recipe.label
 admissions={'qualification':r,'serial_admission':r,'native_qualification':r}
 work=dense_preparation_plan(recipe,profile=True);caps={'wall_seconds':7200,'memory_bytes':8*1024**3,'disk_bytes':12*1024**3}
 plan={'schema_version':1,'scientific_contract_sha256':m.dense_contract_digest(c),'phase':phase,'resource':caps,'preparation':work,**admissions}
 pp=out/'phase-plans'/(phase+'.json');pp.parent.mkdir(parents=True);pp.write_bytes(canonical_json(plan))
 required=[ref(script),r];sources={v['path']:v['sha256'] for v in required+[ref(config),ref(pp)]}
 value={'scientific_contract_sha256':m.dense_contract_digest(c),'profile':True,'recipe':recipe.identity(),
    'preparation_plan':work,'teacher_needed_at_inference':False,'candidate_quality_disclosed':False,
    'directory':str(out/'artifacts/preprofile'/recipe.label),'complete_bytes':60000000}
 def operation(rec):rec.record(CostRecord(**work['work']));return value
 m.record_dense_phase(c,phase,sources,operation,resource=caps,work=work['work'])
 monkeypatch.setattr(m,'ROOT',root)
 command=[sys.executable,str(script),'--config',str(config),'--internal-phase',phase]
 stage=core.execute_stage(command,root=root,output=out,stage=out/'stages'/phase,caps=caps,
   frozen_files=required+[ref(config),ref(pp)],execution_env=m.dense_execution_env())
 directory=out/'phases'/phase
 item={'status':'complete','value':value,'packet':ref(directory/'packet.json'),'terminal_cost':ref(directory/'terminal-cost.json'),
       'stage':stage,'plan':ref(pp)}
 ap=directory/'acceptance.json';ap.write_bytes(canonical_json(item))
 return m,c,recipe,config,required,admissions,ref(ap),root


def test_review_r2_saved_short_operation_admitted_from_guard_and_final_ledger(saved_preparation_profile):
 m,c,recipe,config,refs,admissions,acceptance,root=saved_preparation_profile
 result=m.admit_preparation_profile(c,recipe,acceptance,config_path=config,required_refs=refs,admissions=admissions)
 assert result['measured']['actual_work']['updates']==2


@pytest.mark.parametrize('change',['changed_timing','changed_full_cap','missing_acceptance','missing_terminal'])
def test_review_r2_full_freeze_rejects_forged_caps_or_unadmitted_short_work(saved_preparation_profile,change):
 from poc.dense_frontier import dense_preparation_forecast,dense_recipes
 from poc.records import canonical_json
 m,c,recipe,config,refs,admissions,acceptance,root=saved_preparation_profile
 admitted=m.admit_preparation_profile(c,recipe,acceptance,config_path=config,required_refs=refs,admissions=admissions)
 measured=admitted['measured'];profile=root/'resource-profile.json';profile.write_bytes(canonical_json(measured))
 forecast=dense_preparation_forecast(recipe,measured)
 failures={r.label:{'status':'failed'} for r in dense_recipes() if r.label!=recipe.label}
 job={'profile':m.reference(profile),'acceptance':acceptance,'plan':forecast}
 if change=='changed_timing':
  measured['wall_seconds']+=1;profile.write_bytes(canonical_json(measured));job['profile']=m.reference(profile)
  job['plan']=dense_preparation_forecast(recipe,measured)
 elif change=='changed_full_cap':job['plan']['resource']['wall_seconds']+=30
 elif change=='missing_acceptance':job.pop('acceptance')
 else:
  Path(m.read_reference(acceptance)['terminal_cost']['path']).unlink()
 freeze={'schema_version':1,'scientific_contract_sha256':m.dense_contract_digest(c),'plans':{recipe.label:job},'failures':failures,'scope':'owned restore fixture'}
 (Path(c['out'])/'preparation-freeze.json').write_bytes(canonical_json(freeze))
 with pytest.raises((ValueError,FileNotFoundError)):
  m.admit_preparation_freeze(c,recipe,config_path=config,required_refs=refs,admissions=admissions)


def test_review_r2_full_budget_restores_exactly_from_admitted_short_operation(saved_preparation_profile):
 from poc.dense_frontier import dense_preparation_forecast,dense_recipes
 from poc.records import canonical_json
 m,c,recipe,config,refs,admissions,acceptance,root=saved_preparation_profile
 admitted=m.admit_preparation_profile(c,recipe,acceptance,config_path=config,required_refs=refs,admissions=admissions)
 measured=admitted['measured'];path=root/'resource-profile.json';path.write_bytes(canonical_json(measured))
 forecast=dense_preparation_forecast(recipe,measured)
 freeze={'schema_version':1,'scientific_contract_sha256':m.dense_contract_digest(c),
  'plans':{recipe.label:{'profile':m.reference(path),'acceptance':acceptance,'plan':forecast}},
  'failures':{r.label:{'status':'failed'} for r in dense_recipes() if r.label!=recipe.label},'scope':'owned complete budget restore'}
 (Path(c['out'])/'preparation-freeze.json').write_bytes(canonical_json(freeze))
 result=m.admit_preparation_freeze(c,recipe,config_path=config,required_refs=refs,admissions=admissions)
 assert result['resource']==forecast['resource']
 assert acceptance in result['required_refs'] and m.reference(path) in result['required_refs']


def test_review_r3_native_cost_intent_binds_protocol_before_dispatch(tmp_path):
 from poc.contracts import CostRecord
 from poc.recovery_endpoints import finite_json
 m=execution_driver();c={'study':'owned-native-intent','out':str(tmp_path/'out'),'reference_artifacts':{'teacher_noop':{'manifest_sha256':'a'*64}}}
 protocol={'schema_version':1,'study_contract_sha256':'b'*64,'artifact':{'path':'/owned/model','manifest_sha256':'c'*64}}
 zero=dict(updates=0,valid_target_presentations=0,teacher_forward_calls=0,student_forward_calls=0,student_backward_calls=0,raw_corpus_target_presentations=0)
 seen=[]
 def operation(rec):
  spec=finite_json(rec.path.parent/'spec.json');seen.append(spec['resolved_config']['declared_inputs']['protocol'])
  rec.record(CostRecord());return {'owned':True}
 m.record_dense_phase(c,'profile-owned',{},operation,resource={'wall_seconds':30,'memory_bytes':4*1024**3,'disk_bytes':128*1024**2},
  work=zero,declared_inputs={'protocol':protocol})
 assert seen==[protocol]


def test_review_r3_status_only_native_profile_cannot_authorize_frozen_budget(tmp_path):
 from poc.dense_frontier import dense_recipes
 m=execution_driver()
 with pytest.raises(ValueError,match='profile|admission'):
  m.admit_native_profile({'out':str(tmp_path/'out')},{},dense_recipes()[0],[],{'status':'accepted'},required_refs=[],config_path=tmp_path/'proposal.yaml')


@pytest.fixture
def saved_native_profile(tmp_path,monkeypatch):
 from poc.dense_frontier import dense_recipes,NUMERICS,PREPARATION_LIMITS,REFERENCE_LABELS,dense_native_protocol
 from poc.records import canonical_json
 from poc.contracts import CostRecord
 from provenance import sharing_frontier_study as core
 import sys
 m=execution_driver();root=tmp_path/'native-owned';root.mkdir();out=root/'artifacts/dense-baseline-expansion-v2'
 config=root/'proposal.yaml';config.write_text('owned source context')
 worker_source=root/'worker.py';worker_source.write_text('pass')
 r=lambda p:m.reference(p);dummy=root/'dummy.json';dummy.write_text('{}');dref=r(dummy)
 rows=[{'document_id':str(i),'component_id':'c'+str(i),'tokens':[1]*512,'positions':list(range(512)),
        'attention_mask':[True]*512,'target_mask':[True]*511} for i in range(128)]
 inputs=root/'inputs.json';inputs.write_bytes(canonical_json({'windows':rows,'population_manifest_sha256':'c'*64}))
 c={'schema_version':2,'study':'dense-baseline-expansion-v2','out':str(out),'reference_acceptance':dref,'reference_config':dref,
    'reference_artifacts':{label:{'directory':str(root/'references'/label),'manifest_sha256':'a'*64} for label in REFERENCE_LABELS},
    'fit_manifest':dref,'inputs':{**r(inputs),'population_manifest_sha256':'c'*64},'worker':r(worker_source),
    'recipes':[x.identity() for x in dense_recipes()],'calibration_order':'first80 disjoint components sorted by document_id; first64train next16fit_internal',
    'numerics':dict(NUMERICS),'preparation_limits':dict(PREPARATION_LIMITS),'resource_freeze':None,
    'accounting_profile':'dense-native-contexts-v2','tokenization_policy':'no-special-tokens-v2'}
 artifacts={}
 for label in list(REFERENCE_LABELS)+[x.label for x in dense_recipes()]:
  path=root/'points'/label;path.mkdir(parents=True);(path/'manifest.json').write_bytes(canonical_json({'owned_point':label}))
  artifacts[label]={'directory':str(path),'manifest_sha256':m.file_sha(path/'manifest.json'),'runtime_sha256':'b'*64,
    'complete_bytes':60000000,'teacher_needed_at_inference':False}
 prepared={'study_contract_sha256':m.dense_contract_digest(c),'artifacts':artifacts,'inputs':c['inputs'],
   'teacher_mps_digest':'d'*64,'blocked_source_roots':[str(root)],'reference_native_environment_sha256':'e'*64}
 out.mkdir(parents=True);(out/'preparation.json').write_bytes(canonical_json(prepared))
 recipe=next(x for x in dense_recipes() if x.method=='dwq' and x.bits==4);phase='profile-'+recipe.label;case=out/'workers'/phase;case.mkdir(parents=True)
 worker=case/'native-worker.py';worker.write_bytes(worker_source.read_bytes())
 counts={k:v for k,v in core.metric_work_plan(rows,mode='profile').items() if k!='components'}
 resource={'wall_seconds':300,'memory_bytes':8*1024**3,'output_disk_bytes':12*1024**3,**counts}
 protocol=dense_native_protocol(c,prepared,recipe.label,rows,mode='profile',resource=resource,output=case/'result.json',ledger=case/'costs.jsonl')
 pp=out/'protocols'/(phase+'.json');pp.parent.mkdir();pp.write_bytes(canonical_json(protocol))
 source_refs=[r(config),r(out/'preparation.json'),c['worker'],r(inputs)]+[r(Path(protocol[k]['path'])/'manifest.json') for k in ('teacher','artifact')]
 caps={'wall_seconds':330,'memory_bytes':8*1024**3,'disk_bytes':12*1024**3}
 stage=core.execute_stage([sys.executable,str(worker),'--protocol',str(pp)],root=Path('/private/tmp'),output=out,
  stage=out/'stages'/phase,caps=caps,frozen_files=source_refs+[r(pp),r(worker)],execution_env={'PYTHONPATH':''})
 design={k:v for k,v in protocol.items() if k not in ('mode','output','ledger','resource','profile_binding')}
 value={'schema_version':1,'mode':'profile','study_contract_sha256':protocol['study_contract_sha256'],
   'protocol_sha256':core.contract_digest(protocol),'artifact_manifest_sha256':protocol['artifact']['manifest_sha256'],
   'teacher_manifest_sha256':protocol['teacher']['manifest_sha256'],'runtime_sha256':protocol['runtime_sha256'],
   'worker_sha256':protocol['worker_sha256'],'inputs_sha256':protocol['inputs']['sha256'],'counts':counts,
   'teacher_equivalent_control':False,'teacher_equivalent_passed':None,'design_sha256':core.contract_digest(design),
   'environment_sha256':'e'*64,'complete_bytes':60000000,'wall_seconds':stage['attempt']['wall_seconds']*.9,
   'context_metric_seconds':[stage['attempt']['wall_seconds']*.4],
   'memory':{'process_peak_rss_bytes':1024,'torch_mps_driver_bytes':0,'mlx_peak_bytes':0}}
 output=case/'result.json';output.write_bytes(canonical_json(value));item={'protocol':r(pp),'output':r(output),'stage':stage}
 work=dict(updates=0,**counts,student_backward_calls=0,raw_corpus_target_presentations=511)
 def operation(rec):rec.record(CostRecord(**work));return item
 m.record_dense_phase(c,phase,{v['path']:v['sha256'] for v in source_refs},operation,resource=caps,work=work,declared_inputs={'protocol':protocol})
 directory=out/'phases'/phase;item={**item,'common_packet':r(directory/'packet.json'),'terminal_cost':r(directory/'terminal-cost.json')}
 return m,c,prepared,recipe,rows,item,source_refs,config


def test_review_r3_saved_native_operation_re_admitted_before_resource_restore(saved_native_profile):
 m,c,p,r,rows,item,refs,config=saved_native_profile
 value=m.admit_native_profile(c,p,r,rows,item,required_refs=refs,config_path=config)
 assert value['value']['counts']['valid_target_presentations']==511


@pytest.mark.parametrize('change',['changed_timing_with_new_hash','changed_stage','missing_terminal','changed_protocol','missing_source'])
def test_review_r3_rewritten_profile_or_incomplete_operation_cannot_supply_budget(saved_native_profile,change):
 from copy import deepcopy
 from poc.records import canonical_json
 m,c,p,r,rows,item,refs,config=saved_native_profile;item=deepcopy(item)
 if change=='changed_timing_with_new_hash':
  value=m.read_reference(item['output']);value['wall_seconds']+=10
  path=Path(item['output']['path']);path.write_bytes(canonical_json(value));item['output']=m.reference(path)
 elif change=='changed_stage':item['stage']['protocol_sha256']='0'*64
 elif change=='missing_terminal':Path(item['terminal_cost']['path']).unlink()
 elif change=='changed_protocol':
  value=m.read_reference(item['protocol']);value['resource']['wall_seconds']+=1
  path=Path(item['protocol']['path']);path.write_bytes(canonical_json(value));item['protocol']=m.reference(path)
 else:refs=refs+[{'path':'/private/tmp/omitted-source.py','sha256':'f'*64}]
 with pytest.raises((ValueError,FileNotFoundError)):
  m.admit_native_profile(c,p,r,rows,item,required_refs=refs,config_path=config)


def test_review_r3_restored_job_requires_actual_native_acceptance(saved_native_profile):
 from provenance.dense_frontier_campaign import admitted_native_job
 ex,c,prepared,recipe,windows,item,refs,config=saved_native_profile
 from provenance import sharing_frontier_study as core
 value=ex.read_reference(item['output'])
 forecast=core.forecast_worker(value,windows,teacher_complete_bytes=prepared['artifacts']['teacher_noop']['complete_bytes'],
  memory_bytes=c['preparation_limits']['memory_bytes'],output_disk_bytes=c['preparation_limits']['family_disk_bytes'],wall_ceiling=7200)
 capsule={'jobs':{recipe.label:{'profile':item['output'],'profile_acceptance':item,'plan':forecast}},'native_profiles':{recipe.label:item}}
 admitted=admitted_native_job(c,capsule,prepared,recipe,windows,required_refs=refs,config_path=config)
 assert admitted['plan']==forecast and item['terminal_cost'] in admitted['required_refs']


@pytest.mark.parametrize('mutation',['missing_acceptance','different_acceptance','changed_plan'])
def test_review_r3_restored_job_rejects_detached_or_rebudgeted_profile(saved_native_profile,mutation):
 from provenance.dense_frontier_campaign import admitted_native_job
 ex,c,prepared,recipe,windows,item,refs,config=saved_native_profile
 from provenance import sharing_frontier_study as core
 forecast=core.forecast_worker(ex.read_reference(item['output']),windows,
  teacher_complete_bytes=prepared['artifacts']['teacher_noop']['complete_bytes'],
  memory_bytes=c['preparation_limits']['memory_bytes'],output_disk_bytes=c['preparation_limits']['family_disk_bytes'],wall_ceiling=7200)
 import copy
 job={'profile':item['output'],'profile_acceptance':copy.deepcopy(item),'plan':copy.deepcopy(forecast)}
 if mutation=='missing_acceptance':del job['profile_acceptance']
 elif mutation=='different_acceptance':job['profile_acceptance']['terminal_cost']={'path':'/private/tmp/omitted-terminal.json','sha256':'a'*64}
 else:job['plan']['resource']['wall_seconds']+=1
 capsule={'jobs':{recipe.label:job},'native_profiles':{recipe.label:item}}
 with pytest.raises(ValueError,match='profile|admission|schedule'):
  admitted_native_job(c,capsule,prepared,recipe,windows,required_refs=refs,config_path=config)

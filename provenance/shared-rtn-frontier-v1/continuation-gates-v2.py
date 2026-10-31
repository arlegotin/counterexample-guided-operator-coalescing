"""Read-only source/receipt admission for explicitly continued finite studies."""
from pathlib import Path
import hashlib,json,re

def sha(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError('regular frozen file required: '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()

def verify_files(refs):
 seen={}
 for ref in refs:
  p=ref['path'];h=ref['sha256']
  if not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) or p in seen and seen[p]!=h:raise ValueError('conflicting or invalid frozen reference')
  if sha(p)!=h:raise ValueError('frozen source/receipt changed: '+p)
  seen[p]=h

def admit_analysis(directory,registry_sha256,runner,*,result_relative,attempt_field):
 root=Path(directory);rp=root/'registry.json';wp=root/'watch-result.json';result=root/result_relative;pp=root/'stage/protocol.json';ap=root/'stage/attempt.json'
 if sha(rp)!=registry_sha256:raise ValueError('upstream registry differs')
 registry=json.loads(rp.read_text());watch=json.loads(wp.read_text());attempt=json.loads(ap.read_text());protocol=json.loads(pp.read_text())
 if watch.get('status')!='complete' or watch.get('registry_sha256')!=registry_sha256 or watch.get('result_sha256')!=sha(result) or watch.get(attempt_field)!=sha(ap):raise ValueError('upstream endpoint lacks bound guarded acceptance')
 if attempt.get('exit_code')!=0 or attempt.get('failure','missing') is not None or attempt.get('protocol_sha256')!=sha(pp):raise ValueError('upstream endpoint guard failed or changed')
 if protocol.get('command')!=[registry['python'],runner,'--registry',str(rp),'--analyze'] or {'path':str(rp),'sha256':registry_sha256} not in protocol.get('frozen_files',[]):raise ValueError('upstream endpoint executable/registry binding differs')
 if 'root' in registry and protocol.get('root')!=registry['root']:raise ValueError('upstream endpoint root differs')
 expected_guard=registry.get('guard_sha256',registry.get('supervisor_sha256'))
 if expected_guard is not None and protocol.get('supervisor_sha256')!=expected_guard:raise ValueError('upstream endpoint supervisor differs')
 if 'run_outer_total_seconds' in registry and 'frozen_files' in registry:
  required=registry['frozen_files'];wall=900
 elif 'before_configs' in registry and 'static_frozen_files' in registry:
  required=registry['static_frozen_files'];wall=1200
 else:raise ValueError('unsupported prospective analysis registry type')
 expected_protocol={'root':registry['root'],'output':registry['output'],'provenance':str(root/'stage'),'supervisor_sha256':expected_guard,'execution_env':{'HF_HUB_OFFLINE':None,'PYTORCH_ENABLE_MPS_FALLBACK':None},'caps':{'wall_seconds':wall,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}}
 if any(protocol.get(k)!=v for k,v in expected_protocol.items()):raise ValueError('upstream complete analysis protocol contract differs')
 bound={v['path']:v['sha256'] for v in protocol['frozen_files']}
 if any(bound.get(v['path'])!=v['sha256'] for v in required):raise ValueError('upstream analysis omitted frozen registry sources')
 verify_files(protocol['frozen_files'])
 return [{'path':str(p),'sha256':sha(p)} for p in (rp,wp,result,pp,ap)]

def freeze_stage_outputs(protocol,refs):
 command=protocol['command'];mode=command[command.index('--mode')+1]
 if mode in ('controls','profile'):
  receipt=Path(protocol['output'])/(mode+'.json');refs.append({'path':str(receipt),'sha256':sha(receipt)})
 elif mode!='run':raise ValueError('unexpected scientific stage mode')
 verify_files(refs)

def hashed(value):return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')).hexdigest()

def expected_mps_fixture():
 return {'kind':'owned-six-layer-token-id-control','teacher_seed':3,'weight_seed':991,'training_seed':17,'vocabulary':11,'context':512,'unique_targets':511,'source_and_recovery_updates_each':2,'raw_targets_per_update':8192,'normalization':'temperature-one full-vocabulary forward KL, alpha0','source_and_recovery_rank':1,'full_rank_control':8,'parameter_resume_tolerance':{'atol':1e-6,'rtol':1e-5},'qualified_negative_floor':2e-6,'device':'mps','pretrained':False}

def admit_backend(directory,freeze_sha256,*,acceptance_record='queue-result.json'):
 import copy,math
 root=Path(directory);fp=root/'queue-freeze.json';qp=root/acceptance_record
 if sha(fp)!=freeze_sha256:raise ValueError('owned MPS execution freeze differs')
 freeze=json.loads(fp.read_text());watch=json.loads(qp.read_text());expected=copy.deepcopy(freeze['protocol']);expected['frozen_files'].append({'path':str(fp),'sha256':freeze_sha256})
 if 'prior_analysis' in freeze:
  up=freeze['prior_analysis'];expected['frozen_files'].extend(admit_analysis(up['directory'],up['registry_sha256'],up['runner'],result_relative=up['result_relative'],attempt_field=up['attempt_field']))
 pp=Path(expected['provenance'])/'protocol.json';ap=pp.parent/'attempt.json';protocol=json.loads(pp.read_text());attempt=json.loads(ap.read_text());out=Path(expected['command'][-1]);rp=out/'result.json';r=json.loads(rp.read_text())
 if watch.get('status')!='complete' or watch.get('queue_freeze_sha256')!=freeze_sha256 or watch.get('result_sha256')!=sha(rp) or watch.get('attempt_sha256')!=sha(ap) or protocol!=expected or attempt.get('exit_code')!=0 or attempt.get('failure','missing') is not None or attempt.get('protocol_sha256')!=sha(pp):raise ValueError('owned MPS guard/result linkage differs')
 verify_files(freeze['frozen_files']);verify_files(protocol['frozen_files'])
 if r.get('status')!='complete' or r.get('device')!='mps' or r.get('native_mps_qualified') is not True or r.get('pretrained_inference') is not False or r.get('frozen_owners_unchanged') is not True or any(r.get(k)!=v for k,v in {'source_steps':2,'joint_steps':2,'direct_steps':2,'trainable_owners':30,'retained_optimizer_raw_targets':49152,'unique_owned_target_positions':511}.items()):raise ValueError('actual owned MPS control incomplete')
 error=r.get('resume_max_abs_parameter_error')
 if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=1e-6:raise ValueError('MPS resume resolution failed')
 qualification=json.loads(Path(freeze['qualification']).read_text())
 if r.get('core_qualification_sha256')!=sha(freeze['qualification']) or r.get('source_sha256')!=hashed({'core':qualification['source_files'],'driver_sha256':sha(freeze['driver'])}) or r.get('fixture_contract_sha256')!=hashed(r.get('fixture_contract')):raise ValueError('actual owned backend source/contract differs')
 if r.get('fixture_contract')!=expected_mps_fixture():raise ValueError('owned MPS fixture differs from prospective contract')
 controls=r.get('numerical_controls')
 if not isinstance(controls,dict) or set(controls)!={'no_sharing','full_rank_two_groups'}:raise ValueError('both prospective numerical controls required')
 for row in controls.values():
  v=row.get('kl_raw_sum')
  if row.get('valid_targets')!=511 or row.get('qualified_floor_per_target')!=2e-6 or type(v) not in (int,float) or not math.isfinite(v) or abs(v/511)>2e-6:raise ValueError('owned MPS numerical floor failed')
 cost=r.get('cost_binding',{})
 if cost.get('path')!=str(out/'cost-contract-bindings.json') or cost.get('sha256')!=sha(out/'cost-contract-bindings.json'):raise ValueError('owned MPS cost binding differs')
 return [{'path':str(p),'sha256':sha(p)} for p in (fp,qp,rp,pp,ap)]

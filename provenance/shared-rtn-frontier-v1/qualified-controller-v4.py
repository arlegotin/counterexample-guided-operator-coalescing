"""Wait for accepted profiles, freeze resource-only budgets, run three fixed arms.

No candidate metrics are read to allocate resources. No retries, resume,
hyperparameter selection or schedule changes are performed automatically.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
import psutil,yaml
ROOT=Path('/Users/artemlegotin/.codex-a/worktrees/c81a/counterexample-guided-operator-coalescing')
PY='/Users/artemlegotin/Documents/git_alter/counterexample-guided-operator-coalescing/.venv/bin/python'
WORKER=Path('/private/tmp/cgoc_initializer_fixed_parent_worker_v6.py')
GUARD=Path('/private/tmp/cgoc_scientific_stage_supervisor_v1.py')
RULE=Path('/private/tmp/cgoc_initializer_budget_rule_v1.py')
METHODS=('parent-svd','weighted-svd','zero-product')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2)
def _cancel(signum,frame):raise InterruptedError('campaign cancelled by signal '+str(signum))

def run_stage(command,*,cwd):
 previous=signal.signal(signal.SIGTERM,_cancel);child=None
 try:
  child=subprocess.Popen(command,cwd=cwd,start_new_session=True)
  return child.wait()
 finally:
  try:
   if child is not None:
    # The group belongs to this one explicitly launched stage. It also covers
    # its worker if the supervisor exits first or fails to clean up.
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()
  finally:signal.signal(signal.SIGTERM,previous)

def admit_profiles(campaign,root):
 result_path=campaign/'profile-campaign-result.json';result=json.loads(result_path.read_text())
 rows=result.get('completed_stages',[]);expected=[(m,phase) for m in METHODS for phase in ('controls','profile')]
 if result.get('status')!='complete' or [(r.get('method'),r.get('mode')) for r in rows]!=expected:
  raise ValueError('exact six accepted stages required')
 refs={str(result_path):sha(result_path)}
 def retain(path,digest):
  path=str(path)
  if path in refs and refs[path]!=digest:raise ValueError('inconsistent accepted frozen reference')
  if sha(path)!=digest:raise ValueError('changed accepted stage source/receipt: '+path)
  refs[path]=digest
 for row,(method,mode) in zip(rows,expected,strict=True):
  study='initializer-fixed-parent-r8-s17-'+method+'-v7';stage=root/'provenance'/study/'stages'/mode
  attempt_path=stage/'attempt.json';protocol_path=stage/'protocol.json'
  if row.get('exit_code')!=0 or row.get('attempt_path')!=str(attempt_path):raise ValueError('stage summary differs')
  retain(attempt_path,row['attempt_sha256']);attempt=json.loads(attempt_path.read_text())
  if attempt.get('exit_code')!=0 or attempt.get('failure','missing') is not None:raise ValueError('stage failed')
  retain(protocol_path,attempt['protocol_sha256']);protocol=json.loads(protocol_path.read_text())
  inputs=root/'artifacts/initializer-resource-v7-inputs';cfg=inputs/(method+'.yaml');binding=inputs/'parent-binding.json'
  command=[PY,str(WORKER),'--config',str(cfg),'--binding',str(binding),'--mode',mode]
  caps={'wall_seconds':180 if mode=='controls' else 9120,'memory_bytes':6*1024**3,'disk_bytes':32*1024**3}
  if protocol.get('root')!=str(root) or protocol.get('output')!=str(root/'artifacts'/study) or protocol.get('command')!=command or protocol.get('caps')!=caps:
   raise ValueError('accepted stage protocol differs from fixed method/limits')
  stage_refs={r['path']:r['sha256'] for r in protocol['frozen_files']}
  if stage_refs.get(str(GUARD))!=protocol['supervisor_sha256'] or str(WORKER) not in stage_refs:
   raise ValueError('stage omitted qualified executable sources')
  for path,digest in stage_refs.items():retain(path,digest)
 qualification=campaign/'verification.json'
 if str(qualification) not in refs or json.loads(qualification.read_text())['worker_sha256']!=refs[str(WORKER)]:
  raise ValueError('accepted worker differs from qualified source')
 return [{'path':p,'sha256':h} for p,h in sorted(refs.items())]

def main():
 prov=ROOT/'provenance/initializer-run-campaign-v4'
 prov.mkdir(exist_ok=True)
 if any(prov.glob('controller-*.json')):raise ValueError('controller already owned; no implicit replay')
 started=time.monotonic();state={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'source_sha256':sha(__file__),'budget_rule_sha256':sha(RULE),'status':'waiting_for_profiles'}
 write(prov/'controller-start.json',state)
 campaign=ROOT/'provenance/initializer-resource-campaign-v7'
 refs=json.loads((ROOT/'provenance/reconstruction-study-v1/fixed-source.json').read_text())
 frozen=[{'path':str(ROOT/p),'sha256':h} for p,h in refs['files'].items()]
 frozen.extend({'path':str(p),'sha256':sha(p)} for p in (WORKER,GUARD,RULE,Path(__file__),campaign/'verification.json',ROOT/'provenance/initializer-endpoint-contract-v4/contract.json'))
 def verify():
  for r in frozen:
   if sha(r['path'])!=r['sha256']:raise ValueError('frozen source changed: '+r['path'])
 try:
  start=json.loads((campaign/'profile-campaign-start.json').read_text())
  origin=datetime.fromisoformat(start['utc']).timestamp();deadline=origin+3*(9120+180)+60
  while not (campaign/'profile-campaign-result.json').exists():
   if time.time()>deadline:raise ValueError('finite profile wait exhausted')
   try:process=psutil.Process(start['pid']);cmd=process.cmdline()
   except psutil.NoSuchProcess:raise ValueError('profile controller exited without final receipt') from None
   if not any('cgoc_initializer_profile_campaign_v7.py' in x for x in cmd):raise ValueError('profile controller identity changed')
   time.sleep(10)
  inherited=admit_profiles(campaign,ROOT);frozen.extend(inherited)
  verify();sys.path.insert(0,str(ROOT))
  from poc.records import canonical_json
  from poc.recovery_study import load_study_config
  spec=importlib.util.spec_from_file_location('budget_rule',RULE);rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
  inputs=ROOT/'artifacts/initializer-resource-v7-inputs';binding=inputs/'parent-binding.json'
  contract=json.loads((ROOT/'provenance/initializer-endpoint-contract-v4/contract.json').read_text())
  configs={};profiles={};profile_paths={};old_sha={}
  for method in METHODS:
   path=inputs/(method+'.yaml');old_sha[method]=sha(path)
   if old_sha[method]!=contract['config_paths_and_pre_budget_hashes'][method]['sha256']:raise ValueError('prospective scientific configuration changed')
   cfg=load_study_config(path);configs[method]=cfg
   output=ROOT/cfg['out'];profile_path=output/'profile.json';profile_paths[method]=profile_path
   a=json.loads((ROOT/'provenance'/cfg['study']/'stages/profile/attempt.json').read_text())
   if a['exit_code'] or a['failure']:raise ValueError('profile stage failed')
   profile=json.loads(profile_path.read_text());profiles[method]=profile
   design={k:v for k,v in cfg.items() if k!='resource_budget'}
   if hashlib.sha256(canonical_json(design)).hexdigest()!=profile['design_sha256']:raise ValueError('profile design differs')
  budgets=rule.derive(configs,profiles)
  proposal={};newtexts={}
  for method,cfg in configs.items():
   new=copy.deepcopy(cfg);new['resource_budget']={'frozen_profile_sha256':sha(profile_paths[method]),'per_arm':{'full_student':budgets[method]}}
   text=yaml.safe_dump(new,sort_keys=True);temp=prov/(method+'-frozen-config.txt');temp.write_text(text);load_study_config(temp)
   if {k:v for k,v in new.items() if k!='resource_budget'}!={k:v for k,v in cfg.items() if k!='resource_budget'}:raise ValueError('resource amendment altered science')
   proposal[method]={'before_sha256':old_sha[method],'after_sha256':hashlib.sha256(text.encode()).hexdigest(),'profile_sha256':sha(profile_paths[method]),'per_arm':budgets[method],'supervisor_wall_seconds':budgets[method]['wall_seconds']+2700};newtexts[method]=text
  write(prov/'resource-freeze-intent.json',{'schema_version':1,'rule_sha256':sha(RULE),'endpoint_contract_sha256':sha(ROOT/'provenance/initializer-endpoint-contract-v4/contract.json'),'profile_campaign_sha256':sha(campaign/'profile-campaign-result.json'),'methods':proposal,'run_outer_seconds_total':sum(p['supervisor_wall_seconds'] for p in proposal.values()),'global_limits':'at most24h across three run supervisors,6GiB each,32GiB declared disk family each; sum96GiB is conservative because shared inputs repeat','failed_cost_allocation':'retain and report all v3/v4/v5/v6 failed or completed preparations once, plus all v7 profiles and actual runs separately; no free failed retries'})
  for method,text in newtexts.items():
   path=inputs/(method+'.yaml')
   if sha(path)!=old_sha[method]:raise ValueError('configuration changed during resource freeze')
   temp=inputs/(method+'.freeze-tmp');temp.write_text(text);os.replace(temp,path)
  write(prov/'resource-freeze.json',{'status':'frozen','intent_sha256':sha(prov/'resource-freeze-intent.json'),'config_hashes':{m:sha(inputs/(m+'.yaml')) for m in METHODS},'inherited_profile_sources_and_receipts':inherited})
  replacements={str(inputs/(m+'.yaml')):proposal[m]['after_sha256'] for m in METHODS}
  frozen=[dict(r,sha256=replacements.get(r['path'],r['sha256'])) for r in frozen]
  frozen.extend({'path':str(p),'sha256':sha(p)} for p in (binding,prov/'resource-freeze.json',prov/'resource-freeze-intent.json',campaign/'profile-campaign-result.json'))
  rows=[]
  for method in METHODS:
   verify();cfg=configs[method];stage=ROOT/'provenance'/cfg['study']/'stages/run';out=ROOT/cfg['out'];config_path=inputs/(method+'.yaml')
   current=[*frozen,*({'path':str(p),'sha256':sha(p)} for p in (config_path,out/'controls.json',out/'profile.json'))]
   protocol={'schema_version':1,'root':str(ROOT),'provenance':str(stage),'supervisor_sha256':sha(GUARD),'frozen_files':current,'execution_env':{'HF_HUB_OFFLINE':None,'PYTORCH_ENABLE_MPS_FALLBACK':None},'output':str(out),'caps':{'wall_seconds':proposal[method]['supervisor_wall_seconds'],'memory_bytes':6*1024**3,'disk_bytes':32*1024**3},'command':[PY,str(WORKER),'--config',str(config_path),'--binding',str(binding),'--mode','run'],'scope':'exploratory fixed-parent/rank8/seed17 '+method+' complete256-update full-student recovery; no protected capability or deployment claim'}
   write(stage/'protocol.json',protocol)
   code=run_stage([PY,str(GUARD),'--protocol',str(stage/'protocol.json')],cwd=ROOT)
   attempt=stage/'attempt.json';row={'method':method,'exit_code':code,'attempt_path':str(attempt),'attempt_sha256':sha(attempt) if attempt.exists() else None};rows.append(row);write(prov/('controller-stage-'+str(len(rows))+'.json'),row)
   if code:raise ValueError('scientific run stopped; no automatic retry: '+method)
  write(prov/'controller-result.json',{'status':'complete','stages':rows,'wall_seconds':time.monotonic()-started,'next':'verify completed checkpoints, then apply frozen exploratory endpoint analysis; replicate other seeds/pair before broad claims'})
  return 0
 except BaseException as exc:
  write(prov/'controller-failed.json',{'status':'failed','error':repr(exc),'wall_seconds':time.monotonic()-started,'no_automatic_retry':True})
  raise
if __name__=='__main__':
 signal.signal(signal.SIGTERM,_cancel)
 raise SystemExit(main())

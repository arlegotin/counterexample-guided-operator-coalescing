"""Bounded learned/shared quantization campaign, separate from frozen RTN evidence.

Preparation is fit-only; profiles disclose resources only. Every actual model
phase runs in a finite source-bound supervisor. Successful artifacts are reused.
The fixed component analysis is exploratory and never certifies capability loss.
"""
from pathlib import Path
import sys,hashlib,json,math,importlib.util,time,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.records import canonical_json
from poc.contracts import thaw,BundleRef


def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


core=module('_learned_frontier_guard',ROOT/'provenance/sharing_frontier_study.py')
calcore=module('_learned_frontier_calibration',ROOT/'provenance/shared_calibration_study.py')
sha=core.file_sha256
write=core.write_exclusive


def digest(value):return hashlib.sha256(canonical_json(value)).hexdigest()
def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def read(reference):return core.read_json_reference(reference)
def closed(value,keys,label):
 if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('missing or unknown '+label+' fields')
def absolute(value):
 if type(value)is not str or not Path(value).is_absolute() or str(Path(value).resolve())!=value:raise ValueError('canonical absolute path required')
def hash_value(value):
 import re
 if type(value)is not str or re.fullmatch('[0-9a-f]{64}',value)is None:raise ValueError('exact SHA256 required')
def file_ref(value):
 closed(value,('path','sha256'),'file reference');absolute(value['path']);hash_value(value['sha256'])
def bundle_ref(value):
 closed(value,('directory','manifest_sha256'),'bundle reference');absolute(value['directory']);hash_value(value['manifest_sha256'])


def fixed_grid():
 rows=[('teacher_noop','dense','float',32,32),('shared_float','shared','float',32,32),
 ('dense_rtn8','dense','rtn',8,8),('dense_rtn4','dense','rtn',4,4),('dense_rtn2','dense','rtn',2,2),
 ('dense_rtn2_embedding4','dense','rtn',2,4),('dense_gptq4','dense','gptq',4,4),('dense_gptq2_embedding4','dense','gptq',2,4),
 ('shared_rtn4','shared','rtn',4,4),('shared_rtn2_embedding4','shared','rtn',2,4),('shared_gptq4','shared','gptq',4,4),('shared_gptq2_embedding4','shared','gptq',2,4)]
 return [dict(zip(('label','kind','method','bits','embedding_bits'),r)) for r in rows]


def load_contract(path):
 import yaml
 c=yaml.safe_load(Path(path).read_text())
 closed(c,('schema_version','study','out','teacher','shared','inputs','worker','calibration','reference','upstream_rtn_acceptance','grid','resource'),'learned frontier')
 import re
 if type(c['schema_version'])is not int or c['schema_version']!=1 or type(c['study'])is not str or re.fullmatch('[a-z0-9][a-z0-9-]{0,99}',c['study'])is None:raise ValueError('version1 named study required')
 absolute(c['out'])
 for k in ('teacher','shared'):bundle_ref(c[k])
 for k in ('worker','upstream_rtn_acceptance'):file_ref(c[k])
 closed(c['inputs'],('path','sha256','population_manifest_sha256'),'compiled input');file_ref({k:c['inputs'][k] for k in ('path','sha256')});hash_value(c['inputs']['population_manifest_sha256'])
 closed(c['calibration'],('packet','acceptance','fit_subset','statistics','contract_sha256'),'calibration')
 for k in ('packet','acceptance','fit_subset'):file_ref(c['calibration'][k])
 bundle_ref(c['calibration']['statistics']);hash_value(c['calibration']['contract_sha256'])
 closed(c['reference'],('teacher_mps_digest','source_run_id','scientific_observation'),'source reference')
 for k in ('teacher_mps_digest','source_run_id'):hash_value(c['reference'][k])
 file_ref(c['reference']['scientific_observation'])
 if c['grid']!=fixed_grid():raise ValueError('all12 prespecified points/order required')
 closed(c['resource'],('preparation_wall_seconds','phase_memory_bytes','phase_disk_bytes','profile_wall_seconds','evaluation_wall_ceiling','campaign_wall_seconds','family_disk_bytes','deadline_unix'),'resource')
 for k,v in c['resource'].items():
  if type(v)not in ((int,) if k.endswith('bytes') else (int,float)) or not math.isfinite(v) or v<=0:raise ValueError('finite positive resource budget required')
 return c


def native_protocol(c,p,label,rows,*,mode,resource,output,ledger,profile_binding=None):
 labels=[r['label'] for r in fixed_grid()]
 if p.get('study_contract_sha256')!=digest(c) or p.get('inputs')!=c['inputs'] or set(p.get('artifacts',{}))!=set(labels) or label not in labels or p.get('teacher_mps_digest')!=c['reference']['teacher_mps_digest']:raise ValueError('fixed prepared source/grid differs')
 work=core.metric_work_plan(rows,mode=mode);counts={k:v for k,v in work.items() if k!='components'}
 closed(resource,('wall_seconds','memory_bytes','output_disk_bytes',*counts),'native budget')
 for k,v in resource.items():
  if type(v)not in ((int,float) if k=='wall_seconds' else (int,)) or not math.isfinite(v) or v<=0:raise ValueError('finite native caps required')
 if any(resource[k]!=v for k,v in counts.items()):raise ValueError('exact raw/scored/query plan differs')
 r=c['resource']
 if mode=='profile':
  if profile_binding is not None or resource['wall_seconds']!=r['profile_wall_seconds']:raise ValueError('outcome-blind profile caps required')
 elif not isinstance(profile_binding,dict) or set(profile_binding)!={'path','sha256'} or resource['wall_seconds']>r['evaluation_wall_ceiling']:raise ValueError('frozen completed profile required')
 if resource['memory_bytes']!=r['phase_memory_bytes'] or resource['output_disk_bytes']!=r['phase_disk_bytes']:raise ValueError('native envelope differs')
 teacher=p['artifacts']['teacher_noop'];a=p['artifacts'][label]
 if a['runtime_sha256']!=teacher['runtime_sha256']:raise ValueError('common exact runtime required')
 return {'schema_version':1,'study':c['study'],'study_contract_sha256':digest(c),'mode':mode,
 'artifact':{'path':a['directory'],'manifest_sha256':a['manifest_sha256']},'teacher':{'path':teacher['directory'],'manifest_sha256':teacher['manifest_sha256'],'mps_checkpoint_digest':p['teacher_mps_digest']},
 'inputs':c['inputs'],'numerical':{'chunk_size':32,'temperature':1,'negative_floor':2e-6,'decision_resolution':2e-5},'worker_sha256':c['worker']['sha256'],'runtime_sha256':a['runtime_sha256'],
 'blocked_source_roots':p['blocked_source_roots'],'output':str(Path(output).resolve()),'ledger':str(Path(ledger).resolve()),'resource':dict(resource),'profile_binding':profile_binding}


def summarize(prepared,protocols,results,windows,*,check=core.check_native_result):
 import numpy as np
 labels=[r['label'] for r in fixed_grid()]
 if any(set(v)!=set(labels) for v in (prepared['artifacts'],protocols,results)):raise ValueError('all fixed points, including adverse outcomes, required')
 core.metric_work_plan(windows,mode='run');targets=np.asarray([sum(w['target_mask']) for w in windows]);n=int(targets.sum());points={}
 teacher=results['teacher_noop']['rows']
 for label in labels:
  value=results[label];a=prepared['artifacts'][label];check(protocols[label],value,rows=windows)
  if value['complete_bytes']!=a['complete_bytes'] or a['teacher_needed_at_inference']is not False:raise ValueError('independent actual artifact census differs')
  for row,t in zip(value['rows'],teacher,strict=True):
   if abs(row['teacher_nll_sum']-t['teacher_nll_sum'])>2e-5*row['valid_targets']:raise ValueError('teacher targets differ')
  sums={k:math.fsum(r[k] for r in value['rows']) for k in ('kl_sum','nll_sum','teacher_nll_sum')};sums.update(valid_targets=n,disagreements=sum(r['disagreements'] for r in value['rows']))
  values=[r['kl_sum']/r['valid_targets'] for r in value['rows']];worst=sorted(value['rows'],key=lambda r:r['kl_sum']/r['valid_targets'],reverse=True)[:10]
  points[label]={'complete_bytes':a['complete_bytes'],'within_named_ceiling':a['complete_bytes']<=66064401,'raw_sums':sums,'kl':sums['kl_sum']/n,'nll':sums['nll_sum']/n,'delta_nll':(sums['nll_sum']-sums['teacher_nll_sum'])/n,'perplexity':math.exp(sums['nll_sum']/n) if sums['nll_sum']/n<709 else None,'disagreement_rate':sums['disagreements']/n,'macro_component_kl':float(np.mean(values)),'component_kl_quantiles':dict(zip(('q50','q90','q95','q99'),np.quantile(values,[.5,.9,.95,.99]).tolist())),'worst_components':[{'component_id':r['component_id'],'valid_targets':r['valid_targets'],'kl':r['kl_sum']/r['valid_targets']} for r in worst]}
 for label,p in points.items():p['dominated_by']=[k for k,v in points.items() if k!=label and v['complete_bytes']<=p['complete_bytes'] and v['kl']<=p['kl'] and (v['complete_bytes']<p['complete_bytes'] or v['kl']<p['kl'])]
 draws=np.random.default_rng(17).integers(0,len(windows),(10000,len(windows)))
 def contrast(a,b):
  d=np.asarray([x['kl_sum']-y['kl_sum'] for x,y in zip(results[a]['rows'],results[b]['rows'],strict=True)]);v=d[draws].sum(1)/targets[draws].sum(1)
  return {'label':a+'_minus_'+b,'direction':'negative favors shared GPTQ','kl_difference':float(d.sum()/n),'paired_component_percentile_95':np.quantile(v,[.025,.975]).tolist(),'better_components':int((d<0).sum()),'worse_components':int((d>0).sum()),'tied_components':int((d==0).sum()),'bootstrap_draws':10000,'bootstrap_seed':17,'uncertainty':'paired source components conditional on fixed fitted artifacts; no training/acquisition-seed replication'}
 return {'schema_version':1,'study_contract_sha256':prepared['study_contract_sha256'],'named_ceiling_bytes':66064401,'components':len(windows),'valid_targets':n,'points':points,'primary':contrast('shared_gptq4','dense_gptq4'),'secondary':contrast('shared_gptq4','shared_rtn4'),'scope':{'population':'already exposed development; no protected confirmation','capability_acceptability_established':False,'production_runtime_established':False,'comparison':'complete pipelines: shared FFN GPTQ with RTN attention/embedding versus official dense GPTQ; all eligible attempted points retained','private_factors':'BF16 stored, FP32 computed'}}


def phase_command(config_path,phase):return [sys.executable,str(Path(__file__).resolve()),'--config',str(Path(config_path).resolve()),'--internal-phase',phase]


def require_phase_guard(c,phase,refs,config_path):
 stage=Path(c['out'])/'stages'/phase;pp=stage/'protocol.json';sp=stage/'stage-start.json'
 if config_path is None or not pp.is_file() or not sp.is_file():raise ValueError('internal phase requires exact live finite guard')
 calcore.verify_live_parent(pp);p=json.loads(pp.read_bytes());start=json.loads(sp.read_bytes());bound={r['path']:r['sha256'] for r in p['frozen_files']}
 if (start.get('pid')!=os.getppid() or start.get('protocol_sha256')!=sha(pp) or p['command']!=phase_command(config_path,phase) or p['root']!=str(ROOT) or p['output']!=c['out'] or p['caps']!=prepare_caps(c) or any(bound.get(r['path'])!=r['sha256'] for r in refs)):raise ValueError('phase guard/source/command/budget differs')
 for r in p['frozen_files']:
  if sha(r['path'])!=r['sha256']:raise ValueError('frozen phase source changed')
 return bound


def prepare_caps(c):return {'wall_seconds':c['resource']['preparation_wall_seconds'],'memory_bytes':c['resource']['phase_memory_bytes'],'disk_bytes':c['resource']['family_disk_bytes']}


def scientific_bridge(scientific,windows):
 rows=scientific.get('populations',{}).get('tuning',[])
 if scientific.get('step')!=256 or len(rows)!=len(windows):raise ValueError('complete scientific source observation required')
 bridge=[]
 for old,w in zip(rows,windows,strict=True):
  if (any(old.get(k)!=w[k] for k in ('document_id','component_id')) or old.get('target_positions')!=[i for i,v in enumerate(w['target_mask']) if v]
      or not all(w['attention_mask']) or old.get('window_sha256')!=digest({'tokens':w['tokens'],'positions':w['positions']})
      or w['window_sha256']!=digest({k:v for k,v in w.items() if k!='window_sha256'})
      or old['full'].get('status')!='ok' or old['full'].get('valid_targets')!=sum(w['target_mask'])):raise ValueError('unproven scientific/native exact context/mask binding')
  bridge.append({'document_id':w['document_id'],'scientific_sha256':old['window_sha256'],'native_sha256':w['window_sha256']})
 return bridge


def source_files(config_path):
 sources=calcore.execution_sources(config_path)
 # Include installed original algorithm/model code, rather than package version alone.
 import importlib.util
 for name in ('mlx_lm.quant.gptq','mlx_lm.models.llama'):
  path=Path(importlib.util.find_spec(name).origin).resolve();sources[str(path)]=sha(path)
 for path in (ROOT/'licenses/mlx-lm-MIT.txt',ROOT/'provenance/shared_gptq_study.py'):
  sources[str(path)]=sha(path)
 return dict(sorted(sources.items()))


def record_phase(c,phase,sources,operation,*,resource=None,work=None,declared_inputs=None):
 """Durable common attempt precedes every source/model/fit-content operation."""
 from dataclasses import replace
 from poc.contracts import Budget,CostRecord,RunSpec
 from poc.records import start_attempt
 from poc.train import finish_attempt_cost
 common=module('_learned_frontier_common',ROOT/'provenance/recovery_diagnosis.py');directory=Path(c['out'])/'phases'/phase
 if directory.exists():raise FileExistsError('phase already attempted; preserve all work')
 directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 counts=work or {'valid_target_presentations':0,'teacher_forward_calls':32,'student_forward_calls':16}
 limits=resource or {'wall_seconds':c['resource']['preparation_wall_seconds'],'memory_bytes':c['resource']['phase_memory_bytes'],'disk_bytes':c['resource']['family_disk_bytes']}
 budget=Budget(0,counts['valid_target_presentations'],counts['teacher_forward_calls'],0,0,0,0,limits['wall_seconds'],limits['disk_bytes'],limits['memory_bytes'])
 identity=RunSpec('Q',c['study']+':'+phase,{'contract':c,'phase':phase,'source_files':sources,'declared_inputs':declared_inputs,'work_allowance':counts,'scope':'versioned finite preparation, native fidelity or analysis phase'},
  __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),digest(sources),common.hashed(common.environment()),{'study_contract':digest(c),'source_teacher':c['teacher']['manifest_sha256'],'source_shared':c['shared']['manifest_sha256']},{'device':'MLX FP32; BF16 stored private factors; CPU FP64 fit moments'},{'seed':17},(c['reference']['source_run_id'],),budget)
 attempt=start_attempt(identity,directory/'records')
 class Tracker(common.StudyTracker):
  def observed(self,cost):
   cost=super().observed(cost)
   if 'mlx.core' in sys.modules:cost=replace(cost,memory_bytes=max(cost.memory_bytes or 0,int(sys.modules['mlx.core'].get_peak_memory())))
   return cost
  def check(self,cost):
   if self.totals.student_forward_calls+cost.student_forward_calls>counts['student_forward_calls']:raise ValueError('finite preparation student-query cap exceeded')
   super().check(cost)
 rec=Tracker(budget,attempt,Path(c['out']))
 try:
  rec.record(CostRecord());value=operation(rec)
  write(directory/'packet.json',{'study_contract_sha256':digest(c),'phase':phase,'run_id':attempt.run_id,'value':value,'terminal_cost_path':str(directory/'terminal-cost.json')})
  finish_attempt_cost(rec,elapsed=time.monotonic()-started);totals=rec.validate()
  write(directory/'terminal-cost.json',{'schema_version':1,'run_id':attempt.run_id,'study_contract_sha256':digest(c),'ledger':ref(rec.path),'totals':totals.to_dict(),'scope':'final common ledger through packet write/topup; closing receipt bookkeeping covered by finite supervisor'})
  common.finish_attempt(attempt,'exhausted',reason='finite phase completed');return value
 except BaseException as exc:
  try:finish_attempt_cost(rec,elapsed=time.monotonic()-started)
  finally:
   common.finish_attempt(attempt,'failed',reason=repr(exc));write(directory/'failed.json',{'status':'failed','error':repr(exc),'run_id':attempt.run_id,'scope':'all dispatched setup/model work retained; no automatic retry'})
  raise


def metadata_preflight(c):
 """Hash/role inventory only; no model construction, raw fit text, or gradients."""
 from poc.data.manifest import load_manifest
 for name in ('teacher','shared'):
  inspect_bundle_files(c[name],expected_kind='dense' if name=='teacher' else 'compact')
 packet=read(c['calibration']['packet']);acceptance=read(c['calibration']['acceptance']);upstream=read(c['upstream_rtn_acceptance']);verify_calibration_acceptance(c,packet)
 if acceptance.get('status')!='accepted' or upstream.get('status')!='accepted':raise ValueError('independent upstream acceptances required')
 population=load_manifest(Path(c['calibration']['fit_subset']['path']),expected_sha256=c['calibration']['fit_subset']['sha256'],roles={'fit'})
 if not population.records or len(population.records)!=16 or population.quarantine or set(population.roles.values())!={'fit'} or len(set(population.membership.values()))!=16:raise ValueError('exact16 disjoint fit calibration components required')
 cal=c['calibration']
 if packet.get('statistics_ref')!=cal['statistics'] or packet.get('study_contract_sha256')!=cal['contract_sha256'] or packet['source_runtime_copy']['source']!=c['shared'] or packet['data_provenance']['subset_manifest_sha256']!=cal['fit_subset']['sha256']:raise ValueError('accepted calibration population/source/contract differs')
 if sha(c['worker']['path'])!=c['worker']['sha256'] or c['worker']['sha256']!='7c062d1cd6686c247690a7367f021cdbad8a8a2b45aba358085d88c2507828b5':raise ValueError('qualified native worker changed')
 compiled=read({k:c['inputs'][k] for k in ('path','sha256')})
 worker=module('_learned_frontier_worker',Path(c['worker']['path']))
 rows=worker.read_windows(c['inputs']['path'],c['inputs']['sha256'],population_sha256=c['inputs']['population_manifest_sha256'],tokenizer_digest=packet['spec']['tokenizer_sha256'],vocabulary_size=49152)
 core.metric_work_plan(rows,mode='run')
 if set(population.membership.values())&{r['component_id'] for r in rows}:raise ValueError('fit and evaluation components overlap')
 scientific=read(c['reference']['scientific_observation']);bridge=scientific_bridge(scientific,rows)
 return {'population':population,'packet':packet,'windows':rows,'bridge':bridge,'scientific':scientific}


def execution_refs(c,config_path):
 files=source_files(config_path)
 refs=[{'path':p,'sha256':s} for p,s in files.items()]
 refs +=[dict(c['worker']),dict(c['upstream_rtn_acceptance']),dict(c['reference']['scientific_observation'])]
 refs +=[dict(c['calibration'][k]) for k in ('packet','acceptance','fit_subset')]
 refs +=[{'path':c['inputs']['path'],'sha256':c['inputs']['sha256']}]
 refs +=[{'path':str(Path(c[k]['directory'])/'manifest.json'),'sha256':c[k]['manifest_sha256']} for k in ('teacher','shared')]
 refs +=[{'path':str(Path(c['calibration']['statistics']['directory'])/'manifest.json'),'sha256':c['calibration']['statistics']['manifest_sha256']}]
 qualification=ROOT/'provenance'/c['study']/'qualification.json';q=read(ref(qualification))
 qualification_refs=require_qualification(c,config_path,q,files)
 refs.extend(statistics_refs(c,verify=True));refs.extend(qualification_refs);refs.append(ref(qualification));return refs


def inspect_bundle_files(source,*,expected_kind=None):
 from poc.bundle import _regular_files
 root=Path(source['directory']);absolute(str(root));path=root/'manifest.json'
 if sha(path)!=source['manifest_sha256']:raise ValueError('source manifest differs')
 metadata=json.loads(path.read_bytes());files=_regular_files(root);entries=metadata['file_entries']
 if set(files)!={'manifest.json',*entries}:raise ValueError('bundle file inventory differs')
 for name,p in files.items():
  if p.is_symlink() or p.stat().st_nlink!=1:raise ValueError('unlinked regular source files required')
  if name!='manifest.json' and (p.stat().st_size!=entries[name]['bytes'] or sha(p)!=entries[name]['sha256']):raise ValueError('bundle payload identity differs')
 if expected_kind is not None and metadata['kind']!=expected_kind:raise ValueError('source kind differs')
 return metadata,sum(p.stat().st_size for p in files.values())


def accepted_phase(c,label):
 phase='prepare-'+label;directory=Path(c['out'])/'phases'/phase;value=read(ref(directory/'packet.json'));terminal=calcore.read_terminal_cost(ref(directory/'terminal-cost.json'),expected_run_id=value['run_id'])
 if value.get('study_contract_sha256')!=digest(c) or value.get('phase')!=phase or terminal['study_contract_sha256']!=digest(c):raise ValueError('phase/terminal identity differs')
 acceptance=read(ref(directory/'acceptance.json'))
 if acceptance.get('packet')!=ref(directory/'packet.json') or acceptance.get('terminal_cost')!=ref(directory/'terminal-cost.json'):raise ValueError('phase handoff differs')
 config_path=ROOT/'configs'/(c['study']+'.yaml');refs=execution_refs(c,config_path)
 core.check_stage_binding(acceptance['stage'],command=phase_command(config_path,phase),root=ROOT,output=Path(c['out']),caps=prepare_caps(c),required_refs=refs,execution_env=calcore.worker_environment())
 artifact=value['value'];recipe=next(row for row in fixed_grid() if row['label']==label)
 if artifact.get('directory')!=str(Path(c['out'])/'artifacts'/label) or artifact.get('recipe')!=recipe or artifact.get('study_contract_sha256')!=digest(c):raise ValueError('prepared artifact recipe/directory/study differs')
 if recipe['method']!='float':
  for parent in ('teacher_noop','shared_float'):
   d=Path(c['out'])/'phases'/('prepare-'+parent);refs.extend(ref(d/name) for name in ('packet.json','terminal-cost.json','acceptance.json'))
  core.check_stage_binding(acceptance['stage'],command=phase_command(config_path,phase),root=ROOT,output=Path(c['out']),caps=prepare_caps(c),required_refs=refs,execution_env=calcore.worker_environment())
 return artifact


def offline_proof_module():
 # The test helper imports sibling fixtures. Fresh CLI processes do not inherit
 # pytest's test-directory import path; establish it before executing the file.
 test_path=str(ROOT/'tests')
 if test_path not in sys.path:sys.path.insert(0,test_path)
 return module('_learned_offline_proof',ROOT/'tests/test_offline_bundle.py')


def prepare_artifact(c,label,sources):
 """One isolated artifact recipe; all setup belongs to its durable attempt."""
 from poc.quantize import QuantizationSpec,quantize_rtn,quantize_dense
 from poc.shared_quantization import quantize_shared_gptq
 from poc.shared_gptq import SharedGPTQSpec
 from poc.shared_calibration_bundle import CalibrationRef
 from poc.data.manifest import load_manifest
 from poc.contracts import CostRecord
 from poc.train import _call
 g=next(r for r in fixed_grid() if r['label']==label);out=Path(c['out']);target=out/'artifacts'/label
 def operation(rec):
  import torch,mlx.core as mx
  torch.set_num_threads(1);mx.set_default_device(mx.gpu)
  if not mx.metal.is_available():raise RuntimeError('qualified Metal hardware required')
  if g['method']=='float':
   source=BundleRef(**c['teacher' if g['kind']=='dense' else 'shared']);artifact=core.refresh_runtime(source,target)
  else:
   source_row=accepted_phase(c,'teacher_noop' if g['kind']=='dense' else 'shared_float');source=BundleRef(source_row['directory'],source_row['manifest_sha256'])
   spec=QuantizationSpec(g['method'],g['bits'],64,str(target),embedding_bits=g['embedding_bits'],calibration_count=16,length=512,seed=17)
   if g['method']=='rtn':
    with core.retain_packing_failures(str(target)):artifact=quantize_rtn(source,spec,rec).bundle
   elif g['kind']=='shared':
    population=load_manifest(Path(c['calibration']['fit_subset']['path']),expected_sha256=c['calibration']['fit_subset']['sha256'],roles={'fit'})
    artifact=quantize_shared_gptq(source,CalibrationRef(**c['calibration']['statistics']),SharedGPTQSpec(bits=g['bits']),target,fallback_bits=g['bits'],embedding_bits=g['embedding_bits'],calibration=population,calibration_contract_sha256=c['calibration']['contract_sha256'],recorder=rec).bundle
   else:
    population,proof=dense_population(c,label,source)
    write(out/'phases'/('prepare-'+label)/'calibration-windows.json',proof)
    with core.retain_packing_failures(str(target)):artifact=quantize_dense(source,spec,population,rec,accounting_profile='raw-contexts-v1').bundle
  mx.synchronize();metadata,complete=inspect_bundle_files({'directory':artifact.directory,'manifest_sha256':artifact.manifest_sha256})
  tests=offline_proof_module()
  proofdir=out/'offline-proofs'/label;proofdir.mkdir(parents=True,exist_ok=False)
  proof=_call(rec,lambda:tests.run_offline_bundle(artifact.directory,proofdir,backend='mlx',blocked_source_dirs=(str(ROOT),c['teacher']['directory'],c['shared']['directory'])),student_forward_calls=8,processed_inputs=4)
  rec.record(CostRecord(processed_tokens=len(proof['initial_ids'])+3))
  return {'study_contract_sha256':digest(c),'recipe':g,'directory':artifact.directory,'manifest_sha256':artifact.manifest_sha256,'complete_bytes':complete,'runtime_sha256':sha(Path(artifact.directory)/'runtime/poc-runtime.zip'),'execution_profile':metadata['execution_profile'],'owners':len(metadata['owners']),'teacher_needed_at_inference':False,'offline_proof':proof}
 return record_phase(c,'prepare-'+label,sources,operation)


def dense_population(c,label,source):
 """New fit access journal; verify the exact16 tokenizer windows before fitting."""
 from dataclasses import replace,asdict
 from poc.data.manifest import load_manifest
 from poc.data.access import open_records
 from tokenizers import Tokenizer
 original=load_manifest(Path(c['calibration']['fit_subset']['path']),expected_sha256=c['calibration']['fit_subset']['sha256'],roles={'fit'})
 packet=read(c['calibration']['packet']);directory=Path(c['out'])/'calibration'/label;directory.mkdir(parents=True,exist_ok=False)
 population=replace(original,exposure_path=str(directory/'exposure.jsonl'),lineage_sha256=original.digest())
 tokenizer_path=Path(source.directory)/'tokenizer/tokenizer.json';raw=json.loads(tokenizer_path.read_bytes())
 if raw.get('post_processor')is not None:raise ValueError('implicit tokenizer postprocessor forbidden for this recipe')
 tokenizer=Tokenizer.from_file(str(tokenizer_path));expected={r['document_id']:r for r in packet['data_provenance']['records']};rows=[]
 for r in open_records(population,purpose='calibration'):
  ids=tokenizer.encode(r.text).ids[:512];prior=expected[r.document_id]
  from poc.teacher_validation import _batch
  from poc.shared_calibration import window_plan_sha256
  batch=_batch(ids,device='cpu',document_id=r.document_id,cluster_id=population.membership[r.document_id],targets=range(1,512))
  if len(ids)!=512 or window_plan_sha256((batch,))!=prior['window_sha256']:raise ValueError('dense/shared calibration token windows differ')
  rows.append({'document_id':r.document_id,'component_id':population.membership[r.document_id],'window_sha256':prior['window_sha256'],'tokens':512,'valid_targets':511})
 if len(rows)!=16:raise ValueError('exact16 eligible fit windows required')
 write(directory/'manifest.json',asdict(population))
 return population,{'population':ref(directory/'manifest.json'),'lineage_sha256':original.digest(),'windows':rows,'raw_target_inventory':8176,'unique_components':16,'read_policy':'one validity access plus official adapter calibration access, both journaled; no protected/evaluation text'}


def _native_job(c,p,label,rows,*,mode,resource,refs,profile_binding=None,expected=None,recorder):
 out=Path(c['out']);case=out/'workers'/(mode+'-'+label);case.mkdir(parents=True,exist_ok=False)
 worker=case/'native-worker.py'
 with worker.open('xb') as f:f.write(Path(c['worker']['path']).read_bytes())
 if sha(worker)!=c['worker']['sha256']:raise ValueError('private offline worker source changed')
 protocol=native_protocol(c,p,label,rows,mode=mode,resource=resource,output=case/'result.json',ledger=case/'costs.jsonl',profile_binding=profile_binding)
 if mode=='run' and (expected is None or read(expected)!=protocol):raise ValueError('native run differs from frozen completed schedule')
 pp=out/'protocols'/(mode+'-'+label+'.json');write(pp,protocol)
 required=refs+[ref(pp),ref(worker),ref(out/'preparation.json')]
 required +=[{'path':str(Path(protocol[k]['path'])/'manifest.json'),'sha256':protocol[k]['manifest_sha256']} for k in ('teacher','artifact')]
 if profile_binding:required.append(profile_binding)
 if expected:required.append(expected)
 command=[sys.executable,str(worker),'--protocol',str(pp)];caps=campaign_envelope(c,resource['wall_seconds']+30)
 stage=None
 try:
  stage=core.execute_stage(command,root=Path('/private/tmp'),output=out,stage=out/'stages'/(mode+'-'+label),caps=caps,frozen_files=required,execution_env={'PYTHONPATH':''})
 finally:
  reconcile_native_ledger(recorder,protocol,case/'costs.jsonl')
  if stage is not None:
   from poc.contracts import CostRecord
   recorder.record(CostRecord(memory_bytes=int(stage['attempt']['sampled_process_tree_rss_bytes']),memory_unavailable_reason=None,disk_bytes=int(stage['attempt']['sampled_scratch_peak_bytes'])))
 core.require_stage_success(stage);value=read(ref(case/'result.json'));core.check_native_result(protocol,value,rows=rows if mode=='run' else None)
 core.check_stage_binding(stage,command=command,root=Path('/private/tmp'),output=out,caps=caps,required_refs=required,execution_env={'PYTHONPATH':''})
 item={'protocol':ref(pp),'output':ref(case/'result.json'),'stage':stage};write(case/'acceptance.json',item);return item


def dispatch_prepare(c,label,config_path,refs):
 phase='prepare-'+label;out=Path(c['out']);command=phase_command(config_path,phase)
 # Earlier floating phases are explicit immutable dependencies, never implicit caches.
 extra=[]
 if label not in ('teacher_noop','shared_float'):
  for source_label in ('teacher_noop','shared_float'):
   directory=out/'phases'/('prepare-'+source_label)
   extra +=[ref(directory/name) for name in ('packet.json','terminal-cost.json','acceptance.json')]
 campaign_envelope(c,prepare_caps(c)['wall_seconds'])
 stage=core.execute_stage(command,root=ROOT,output=out,stage=out/'stages'/phase,caps=prepare_caps(c),frozen_files=refs+extra,execution_env=calcore.worker_environment())
 core.require_stage_success(stage);core.check_stage_binding(stage,command=command,root=ROOT,output=out,caps=prepare_caps(c),required_refs=refs+extra,execution_env=calcore.worker_environment())
 directory=out/'phases'/phase;packet=read(ref(directory/'packet.json'));terminal=calcore.read_terminal_cost(ref(directory/'terminal-cost.json'),expected_run_id=packet['run_id'])
 write(directory/'acceptance.json',{'packet':ref(directory/'packet.json'),'terminal_cost':ref(directory/'terminal-cost.json'),'stage':stage})
 return accepted_phase(c,label)


def campaign(c,config_path,*,profile_only=False):
 """No automatic retry: an existing study directory always blocks a new campaign."""
 started=time.monotonic();out=Path(c['out']);data=metadata_preflight(c);refs=execution_refs(c,config_path)
 if out.exists():raise FileExistsError('study already attempted; preserve successes and failures')
 out.mkdir(parents=True,exist_ok=False);write(out/'study-contract.json',c);write(out/'campaign-intent.json',{'started_unix':time.time(),'contract_sha256':digest(c),'campaign_wall_seconds':c['resource']['campaign_wall_seconds']})
 def check():
  if time.time()>c['resource']['deadline_unix'] or time.monotonic()-started>c['resource']['campaign_wall_seconds']:raise ValueError('finite campaign time exhausted')
  size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
  if size>c['resource']['family_disk_bytes']:raise ValueError('retained campaign disk cap exceeded')
  for r in refs:
   if sha(r['path'])!=r['sha256']:raise ValueError('qualified source changed')
 artifacts={}
 try:
  dispatch_input_validation(c,config_path,refs)
  for g in fixed_grid():check();artifacts[g['label']]=dispatch_prepare(c,g['label'],config_path,refs)
  runtimes={a['runtime_sha256'] for a in artifacts.values()}
  if len(runtimes)!=1:raise ValueError('all prepared artifacts must use common exact runtime')
  import subprocess
  blocked=sorted({os.fsdecode(row[9:]) for row in subprocess.check_output(['git','worktree','list','--porcelain','-z'],cwd=ROOT).split(b'\0') if row.startswith(b'worktree ')})
  p={'schema_version':1,'study_contract_sha256':digest(c),'artifacts':artifacts,'inputs':c['inputs'],'teacher_mps_digest':c['reference']['teacher_mps_digest'],'blocked_source_roots':blocked};write(out/'preparation.json',p)
  profiles={};work=core.metric_work_plan(data['windows'],mode='profile');counts={k:v for k,v in work.items() if k!='components'}
  r=c['resource'];profile_caps={'wall_seconds':r['profile_wall_seconds'],'memory_bytes':r['phase_memory_bytes'],'output_disk_bytes':r['phase_disk_bytes'],**counts}
  for g in fixed_grid():
   check();label=g['label'];profiles[label]=native_job(c,p,label,data['windows'],mode='profile',resource=profile_caps,refs=refs)
  write(out/'profiles.json',profiles);freeze={}
  for g in fixed_grid():
   label=g['label'];value=read(profiles[label]['output']);plan=core.forecast_worker(value,data['windows'],teacher_complete_bytes=artifacts['teacher_noop']['complete_bytes'],memory_bytes=r['phase_memory_bytes'],output_disk_bytes=r['phase_disk_bytes'],wall_ceiling=r['evaluation_wall_ceiling'])
   case=out/'workers'/('run-'+label);protocol=native_protocol(c,p,label,data['windows'],mode='run',resource=plan['resource'],output=case/'result.json',ledger=case/'costs.jsonl',profile_binding=profiles[label]['output'])
   path=out/'frozen-protocols'/(label+'.json');write(path,protocol);freeze[label]={'plan':plan,'protocol':ref(path),'profile':profiles[label]['output']}
  write(out/'evaluation-freeze.json',{'study_contract_sha256':digest(c),'order':[g['label'] for g in fixed_grid()],'jobs':freeze,'primary':'shared_gptq4_minus_dense_gptq4','secondary':'shared_gptq4_minus_shared_rtn4','scope':'frozen before any candidate quality outputs; exposed development only'})
  if profile_only:return {'status':'profiled','evaluation_freeze':ref(out/'evaluation-freeze.json'),'candidate_quality_disclosed':False}
  return run_from_freeze(c,config_path)
 except BaseException as exc:
  write(out/'campaign-failed.json',{'status':'failed','error':repr(exc),'elapsed_seconds':time.monotonic()-started,'scope':'no automatic repeat; all successful preparations, guards, failed partials and exposures retained'})
  raise


def accept_native(c,p,label,rows,item,*,mode,refs,frozen=None):
 out=Path(c['out']);case=out/'workers'/(mode+'-'+label);pp=out/'protocols'/(mode+'-'+label+'.json');worker=case/'native-worker.py'
 if item.get('protocol',{}).get('path')!=str(pp) or item.get('output',{}).get('path')!=str(case/'result.json'):raise ValueError('native result/protocol belongs to another case')
 protocol=read(item['protocol']);profile=None if mode=='profile' else frozen['profile']
 expected=native_protocol(c,p,label,rows,mode=mode,resource=protocol['resource'],output=case/'result.json',ledger=case/'costs.jsonl',profile_binding=profile)
 if protocol!=expected or mode=='run' and read(frozen['protocol'])!=protocol:raise ValueError('actual native protocol differs from frozen case')
 if sha(worker)!=c['worker']['sha256']:raise ValueError('executed private worker source differs')
 required=refs+[ref(pp),ref(worker),ref(out/'preparation.json')]
 required +=[{'path':str(Path(protocol[k]['path'])/'manifest.json'),'sha256':protocol[k]['manifest_sha256']} for k in ('teacher','artifact')]
 if mode=='run':required +=[profile,frozen['protocol']]
 caps={'wall_seconds':protocol['resource']['wall_seconds']+30,'memory_bytes':protocol['resource']['memory_bytes'],'disk_bytes':c['resource']['family_disk_bytes']}
 core.check_stage_binding(item['stage'],command=[sys.executable,str(worker),'--protocol',str(pp)],root=Path('/private/tmp'),output=out,caps=caps,required_refs=required,execution_env={'PYTHONPATH':''})
 value=read(item['output']);core.check_native_result(protocol,value,rows=rows if mode=='run' else None)
 common=read_native_common(c,mode+'-'+label,item,protocol)
 return protocol,value


def run_from_freeze(c,config_path):
 out=Path(c['out']);data=metadata_preflight(c);refs=execution_refs(c,config_path);p=read(ref(out/'preparation.json'));profiles=read(ref(out/'profiles.json'));frozen=read(ref(out/'evaluation-freeze.json'));intent=read(ref(out/'campaign-intent.json'))
 labels=[g['label'] for g in fixed_grid()]
 if (frozen.get('study_contract_sha256')!=digest(c) or frozen.get('order')!=labels or set(frozen.get('jobs',{}))!=set(labels)
     or intent.get('contract_sha256')!=digest(c) or intent.get('campaign_wall_seconds')!=c['resource']['campaign_wall_seconds']):raise ValueError('frozen campaign contract/grid differs')
 if (out/'evaluation-index.json').exists() or any((out/'workers'/('run-'+label)).exists() for label in labels):raise FileExistsError('evaluation already dispatched; retain partial or completed work')
 for label in labels:
  accepted_phase(c,label);_,value=accept_native(c,p,label,data['windows'],profiles[label],mode='profile',refs=refs)
  r=c['resource'];plan=core.forecast_worker(value,data['windows'],teacher_complete_bytes=p['artifacts']['teacher_noop']['complete_bytes'],memory_bytes=r['phase_memory_bytes'],output_disk_bytes=r['phase_disk_bytes'],wall_ceiling=r['evaluation_wall_ceiling'])
  if frozen['jobs'][label]['plan']!=plan or frozen['jobs'][label]['profile']!=profiles[label]['output']:raise ValueError('completed budget differs from admitted resource-only profile')
 campaign_envelope(c,sum(frozen['jobs'][label]['plan']['resource']['wall_seconds']+60 for label in labels)+prepare_caps(c)['wall_seconds'])
 results={}
 for label in labels:
  if time.time()>c['resource']['deadline_unix'] or time.time()-intent['started_unix']>c['resource']['campaign_wall_seconds']:raise ValueError('finite campaign time exhausted before dispatch')
  f=frozen['jobs'][label];results[label]=native_job(c,p,label,data['windows'],mode='run',resource=f['plan']['resource'],refs=refs+[ref(out/'evaluation-freeze.json')],profile_binding=f['profile'],expected=f['protocol'])
 write(out/'evaluation-index.json',results)
 return dispatch_analysis(c,config_path)


def _analyze(c,config_path):
 out=Path(c['out']);data=metadata_preflight(c);refs=execution_refs(c,config_path)
 p=read(ref(out/'preparation.json'));index=read(ref(out/'evaluation-index.json'));freeze=read(ref(out/'evaluation-freeze.json'));results={};protocols={}
 if freeze.get('study_contract_sha256')!=digest(c) or freeze.get('order')!=[g['label'] for g in fixed_grid()]:raise ValueError('frozen evaluation grid differs')
 for g in fixed_grid():
  label=g['label'];item=index[label];f=freeze['jobs'][label];protocol,value=accept_native(c,p,label,data['windows'],item,mode='run',refs=refs+[ref(out/'evaluation-freeze.json')],frozen=f)
  expected=native_protocol(c,p,label,data['windows'],mode='run',resource=f['plan']['resource'],output=protocol['output'],ledger=protocol['ledger'],profile_binding=f['profile'])
  if protocol!=expected or protocol!=read(f['protocol']):raise ValueError('measured result differs from frozen protocol')
  if accepted_phase(c,label)['manifest_sha256']!=p['artifacts'][label]['manifest_sha256']:raise ValueError('accepted preparation source differs')
  inspect_bundle_files({'directory':p['artifacts'][label]['directory'],'manifest_sha256':p['artifacts'][label]['manifest_sha256']})
  results[label]=value;protocols[label]=protocol
 result=summarize(p,protocols,results,data['windows']);errors=[]
 for s,n in zip(data['scientific']['populations']['tuning'],results['shared_float']['rows'],strict=True):
  error=max(abs(s['full']['kl']-n['kl_sum']),abs(s['full']['nll']-n['nll_sum']))/n['valid_targets'];errors.append(error)
  if error>2e-5:raise ValueError('floating scientific/native parity failed')
 result['float_source_parity']={'max_component_error':max(errors),'resolution':2e-5,'identity_bridge':data['bridge']}
 result['preparation_costs']={g['label']:read(ref(out/'phases'/('prepare-'+g['label'])/'terminal-cost.json')) for g in fixed_grid()}
 result['input_validation_cost']=read(ref(out/'phases/input-validation/terminal-cost.json'))
 result['native_profile_costs']={g['label']:read(ref(out/'phases'/('profile-'+g['label'])/'terminal-cost.json')) for g in fixed_grid()}
 result['native_run_costs']={g['label']:read(ref(out/'phases'/('run-'+g['label'])/'terminal-cost.json')) for g in fixed_grid()}
 result['upstream_calibration_costs']={'accepted':c['calibration']['acceptance'],'packet':c['calibration']['packet'],'scope':'shared preparation counts all profile/run and failed ancestor work separately; no unqualified amortization'}
 write(out/'analysis.json',result)
 lines=['# Learned shared quantization exploratory frontier','','Complete artifacts;128 exposed development components/65408targets. No capability acceptability or production runtime claim.','','|Artifact|Bytes|KL nats/target|NLL|','|---|---:|---:|---:|']
 for label,v in result['points'].items():lines.append(f"|{label}|{v['complete_bytes']}|{v['kl']:.10f}|{v['nll']:.10f}|")
 for name in ('primary','secondary'):
  v=result[name];lines+=['',f"{name}: {v['label']} = {v['kl_difference']:.10f}, paired component95% interval {v['paired_component_percentile_95']}; {v['better_components']}better/{v['worse_components']}worse components. Fixed-model component uncertainty, not independent training replication."]
 lines+=['','Shared adapter: canonical FFN-bank GPTQ, explicit RTN attention/embedding fallback; BF16 stored private factors. Dense adapter: official GPTQ on eligible dense linears with RTN embedding fallback. Complete pipeline contrast; extra shared recovery preparation is not free. Named ceiling66,064,401bytes; actual final census determines eligibility. All source/data/statistics/guards/costs retained.']
 with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 return result


def analyze(c,config_path):
 result=record_phase(c,'analysis',source_files(config_path),lambda rec:_analyze(c,config_path))
 write(Path(c['out'])/'campaign-complete.json',{'status':'complete','contract_sha256':digest(c),'analysis':ref(Path(c['out'])/'analysis.json'),'terminal_analysis_cost':ref(Path(c['out'])/'phases/analysis/terminal-cost.json'),'scope':'exploratory complete pipeline fidelity/byte comparison; no protected evaluation'})
 return result


def qualification_specs(work):
 tests=['tests/test_shared_gptq.py','tests/test_shared_quantization.py','tests/test_shared_gptq_study.py','tests/test_quantization.py','tests/test_export.py','tests/test_offline_bundle.py','tests/test_sharing_frontier.py']
 return [
 ('owned_cpu',['-m','pytest',*tests,'-q'],{'wall_seconds':300,'memory_bytes':8*1024**3,'disk_bytes':2*1024**3}),
 ('default_cpu',['-m','pytest','-q'],{'wall_seconds':1800,'memory_bytes':12*1024**3,'disk_bytes':8*1024**3}),
 ('owned_mlx',['-m','pytest','tests/test_shared_gptq.py','tests/test_shared_quantization.py','tests/test_shared_gptq_study.py','tests/test_quantization.py','-q','-m','mlx','-k','gptq or actual_rope or actual_packing or packed_export or failed_shared or embedding_upgrade or dense_opt_in or postpublication'],{'wall_seconds':300,'memory_bytes':8*1024**3,'disk_bytes':2*1024**3}),
 ('public',['provenance/verify_public_continuation.py'],{'wall_seconds':120,'memory_bytes':1024**3,'disk_bytes':1024**2})]


def require_qualification(c,config_path,q,files):
 import tarfile,platform
 fields=('schema_version','study','status','contract_sha256','config','sources','source_record','source_archive','review','environment','stages','created_utc','scope')
 closed(q,fields,'qualification')
 if (q['schema_version']!=1 or type(q['schema_version'])is not int or q['study']!=c['study'] or q['status']!='qualified' or q['sources']!=files or q['config']!=ref(config_path) or q['contract_sha256']!=digest(c)):raise ValueError('own exact source-bound qualification required')
 review=read(q['review']);record=read(q['source_record']);env=read(q['environment'])
 if review.get('unresolved_critical')!=0 or review.get('unresolved_important')!=0 or review.get('source_files')!={p:files[p] for p in review.get('source_files',{})} or not review.get('source_files'):raise ValueError('resolved source-bound whole-feature review required')
 if record.get('sources')!=files or record.get('archive')!=q['source_archive'] or record.get('review')!=q['review'] or record.get('environment')!=q['environment'] or record.get('contract_sha256')!=digest(c):raise ValueError('qualification source/archive/review bindings differ')
 if env.get('python')!=sys.version or env.get('executable')!=sys.executable or env.get('platform')!=platform.platform():raise ValueError('qualified execution environment differs')
 expected={str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else 'installed-source/'+str(Path(p)).lstrip('/'):h for p,h in files.items()}
 operation=record['operations'];file_ref(operation)
 if sha(operation['path'])!=operation['sha256']:raise ValueError('qualification operation source changed')
 expected['operations/'+Path(operation['path']).name]=operation['sha256']
 if sha(q['source_archive']['path'])!=q['source_archive']['sha256']:raise ValueError('source archive identity differs')
 with tarfile.open(q['source_archive']['path'],'r:gz') as tar:
  members=tar.getmembers()
  if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise ValueError('source archive inventory differs')
  for member in members:
   if not member.isfile() or hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=expected[member.name]:raise ValueError('source archive payload differs')
 base=[{'path':p,'sha256':h} for p,h in files.items()]+[q['source_archive'],q['review'],q['environment'],operation,q['source_record']]
 if set(q['stages'])!={'owned_cpu','default_cpu','owned_mlx','public'}:raise ValueError('all owned/full CPU/native/public qualification stages required')
 work=Path(q['stages']['default_cpu']['output']).parent
 if not work.is_absolute() or work.resolve()!=work or work.is_relative_to(ROOT):raise ValueError('external qualification scratch required')
 for name,args,caps in qualification_specs(work):
  item=q['stages'][name];output=work/name;command=[sys.executable,*args]
  if name!='public':command.append('--basetemp='+str(output/'pytest'))
  if item.get('command')!=command or item.get('caps')!=caps or item.get('output')!=str(output):raise ValueError('qualification command/caps/output differs')
  core.check_stage_binding(item['stage'],command=command,root=ROOT,output=output,caps=caps,required_refs=base,execution_env=calcore.worker_environment())
 return base+[ref(q['stages'][name]['stage'][key]) for name in q['stages'] for key in ('protocol_path','attempt_path')]


def statistics_refs(c,*,verify=False):
 stats=c['calibration']['statistics'];root=Path(stats['directory']);mp=root/'manifest.json'
 if sha(mp)!=stats['manifest_sha256']:raise ValueError('statistics manifest differs')
 metadata=json.loads(mp.read_bytes())
 if metadata.get('schema_version')!=1 or metadata.get('study_contract_sha256')!=c['calibration']['contract_sha256'] or set(metadata.get('files',{}))!={'statistics.safetensors'}:raise ValueError('statistics contract/inventory differs')
 entry=metadata['files']['statistics.safetensors'];payload=root/'statistics.safetensors'
 if payload.is_symlink() or not payload.is_file() or payload.stat().st_nlink!=1 or payload.stat().st_size!=entry['bytes']:raise ValueError('statistics payload census differs')
 if verify and sha(payload)!=entry['sha256']:raise ValueError('statistics payload hash differs')
 return [ref(mp),{'path':str(payload),'sha256':entry['sha256']}]


def check_calibration_binding(c,packet,result):
 run=result.get('results',{}).get('run',{})
 if (result.get('status')!='accepted' or run.get('packet')!=c['calibration']['packet'] or run.get('statistics')!=c['calibration']['statistics']
     or run.get('result_sha256')!=packet.get('result_sha256') or run.get('source_runtime_copy',{}).get('source')!=c['shared']
     or packet.get('study_contract_sha256')!=c['calibration']['contract_sha256']):raise ValueError('admitted calibration packet/statistics/source differs')
 return run


def verify_calibration_acceptance(c,packet):
 accepted=read(c['calibration']['acceptance']);result=read(accepted['result']);contract=read(accepted['contract']);run=check_calibration_binding(c,packet,result)
 registry=read(contract['registry'])
 if result.get('execution_freeze')!=contract['registry'] or registry.get('contract_sha256')!=c['calibration']['contract_sha256']:raise ValueError('calibration accepted registry/contract differs')
 verifier=result['verifier_source'];root=Path(registry['config']['path']).parents[1]
 if Path(verifier['path']).name!='cgoc_verify_calibration_campaign_v2.py' or verifier['sha256']!='3c28c63219cf9bbe20dda14c9e94dd8866b81462f209d374534100012a8e19e2':raise ValueError('qualified independent calibration verifier differs')
 command=[sys.executable,verifier['path'],'--root',str(root),'--registry-sha256',contract['registry']['sha256'],'--out',accepted['result']['path']]
 refs=contract['source_files']+[accepted['contract']]
 core.check_stage_binding(accepted['stage'],command=command,root=Path('/private/tmp'),output=Path(accepted['result']['path']).parent,caps=contract['caps'],required_refs=refs,execution_env={'PYTHONPATH':''})
 return run


def validate_input_statistics(c):
 from poc.shared_calibration_bundle import CalibrationRef,load_calibration
 from poc.shared_gptq import validate_shared_moments
 from poc.data.manifest import load_manifest
 statistics_refs(c,verify=True);metadata,_=inspect_bundle_files(c['shared'],expected_kind='compact');population=load_manifest(Path(c['calibration']['fit_subset']['path']),expected_sha256=c['calibration']['fit_subset']['sha256'],roles={'fit'})
 result=load_calibration(CalibrationRef(**c['calibration']['statistics']),expected_contract_sha256=c['calibration']['contract_sha256']);binding=validate_shared_moments(metadata,result,population)
 verify_calibration_acceptance(c,read(c['calibration']['packet']))
 return {'binding':binding,'statistics_files':statistics_refs(c,verify=True),'raw_corpus_reads':0,'neural_calls':0,'scope':'fresh FP64 payload/shape/finiteness/all-use/source validation; no model construction or raw text read'}


def dispatch_input_validation(c,config_path,refs):
 phase='input-validation';out=Path(c['out']);campaign_envelope(c,prepare_caps(c)['wall_seconds']);command=phase_command(config_path,phase)
 stage=core.execute_stage(command,root=ROOT,output=out,stage=out/'stages'/phase,caps=prepare_caps(c),frozen_files=refs,execution_env=calcore.worker_environment());core.require_stage_success(stage)
 core.check_stage_binding(stage,command=command,root=ROOT,output=out,caps=prepare_caps(c),required_refs=refs,execution_env=calcore.worker_environment())
 directory=out/'phases'/phase;packet=read(ref(directory/'packet.json'));calcore.read_terminal_cost(ref(directory/'terminal-cost.json'),expected_run_id=packet['run_id'])
 write(directory/'acceptance.json',{'status':'accepted','packet':ref(directory/'packet.json'),'terminal_cost':ref(directory/'terminal-cost.json'),'stage':stage})
 return ref(directory/'acceptance.json')

def campaign_envelope(c,required_wall):
 """Admit an entire completed phase; never truncate its scientific schedule."""
 if type(required_wall)not in (int,float) or not math.isfinite(required_wall) or required_wall<0:raise ValueError('finite completed wall allowance required')
 out=Path(c['out']);intent=read(ref(out/'campaign-intent.json'))
 if intent.get('contract_sha256')!=digest(c) or intent.get('campaign_wall_seconds')!=c['resource']['campaign_wall_seconds']:raise ValueError('campaign resource identity differs')
 now=time.time();remaining=min(c['resource']['deadline_unix']-now,c['resource']['campaign_wall_seconds']-(now-intent['started_unix']))
 if required_wall>remaining:raise ValueError('insufficient remaining campaign time for completed phase')
 disk=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
 if disk>c['resource']['family_disk_bytes']:raise ValueError('retained family disk cap exhausted')
 return {'wall_seconds':required_wall,'memory_bytes':c['resource']['phase_memory_bytes'],'disk_bytes':c['resource']['family_disk_bytes']}


def reconcile_native_ledger(recorder,protocol,path):
 """Mirror actual durable dispatched work, including a valid interrupted prefix."""
 from poc.contracts import CostRecord
 path=Path(path)
 if not path.exists():return {'valid_target_presentations':0,'teacher_forward_calls':0,'student_forward_calls':0}
 counts={'valid_target_presentations':0,'teacher_forward_calls':0,'student_forward_calls':0}
 for line in path.read_bytes().splitlines(keepends=True):
  if not line.endswith(b'\n'):raise ValueError('truncated native cost tail retained; completed prefix already charged')
  row=json.loads(line);work=row.get('work',{})
  if (row.get('study')!=protocol['study'] or row.get('study_contract_sha256')!=protocol['study_contract_sha256'] or row.get('protocol_sha256')!=digest(protocol)
      or not work or not set(work)<=set(counts) or any(type(v)is not int or v<0 for v in work.values())):raise ValueError('native cost identity/units differ')
  for k,v in work.items():counts[k]+=v
  if row.get('cumulative')!=counts or row.get('raw_corpus_target_presentations')!=work.get('valid_target_presentations',0):raise ValueError('native cumulative/raw cost differs')
  raw=work.get('valid_target_presentations',0)
  recorder.record(CostRecord(**work,raw_corpus_target_presentations=raw,processed_inputs=int(raw>0),processed_tokens=512 if raw else 0))
 return counts


def native_job(c,p,label,rows,*,mode,resource,refs,profile_binding=None,expected=None):
 phase=mode+'-'+label
 def operation(rec):return _native_job(c,p,label,rows,mode=mode,resource=resource,refs=refs,profile_binding=profile_binding,expected=expected,recorder=rec)
 counts={k:resource[k] for k in ('valid_target_presentations','teacher_forward_calls','student_forward_calls')} if resource else {}
 # The attempt is created before any case copy, source load or worker setup.
 limits={'wall_seconds':resource.get('wall_seconds',0)+60,'memory_bytes':c['resource']['phase_memory_bytes'],'disk_bytes':c['resource']['family_disk_bytes']}
 item=record_phase(c,phase,{r['path']:r['sha256'] for r in refs},operation,resource=limits,work=counts,declared_inputs={'prepared':p,'label':label,'mode':mode,'resource':resource,'profile':profile_binding,'frozen_protocol':expected})
 directory=Path(c['out'])/'phases'/phase
 return {**item,'common_packet':ref(directory/'packet.json'),'terminal_cost':ref(directory/'terminal-cost.json')}


def read_native_common(c,phase,item,protocol):
 directory=Path(c['out'])/'phases'/phase
 if item.get('common_packet')!=ref(directory/'packet.json') or item.get('terminal_cost')!=ref(directory/'terminal-cost.json'):raise ValueError('native common cost handoff differs')
 packet=read(item['common_packet']);terminal=calcore.read_terminal_cost(item['terminal_cost'],expected_run_id=packet['run_id'])
 if packet['phase']!=phase or packet['study_contract_sha256']!=digest(c) or any(packet['value'].get(k)!=item[k] for k in ('protocol','output','stage')):raise ValueError('native common attempt/result binding differs')
 for k in ('valid_target_presentations','teacher_forward_calls','student_forward_calls'):
  if terminal['totals'][k]!=protocol['resource'][k]:raise ValueError('native common query/target count differs')
 if terminal['totals']['raw_corpus_target_presentations']!=protocol['resource']['valid_target_presentations']:raise ValueError('native raw target cost differs')
 return terminal


def dispatch_analysis(c,config_path):
 out=Path(c['out']);refs=execution_refs(c,config_path);phase='analysis';command=phase_command(config_path,phase)
 analysis_refs=refs+[ref(out/name) for name in ('evaluation-index.json','evaluation-freeze.json','profiles.json','preparation.json')]
 caps=campaign_envelope(c,prepare_caps(c)['wall_seconds'])
 stage=core.execute_stage(command,root=ROOT,output=out,stage=out/'stages'/phase,caps=caps,frozen_files=analysis_refs,execution_env=calcore.worker_environment())
 core.require_stage_success(stage);core.check_stage_binding(stage,command=command,root=ROOT,output=out,caps=caps,required_refs=analysis_refs,execution_env=calcore.worker_environment())
 campaign_envelope(c,0)
 write(out/'analysis-acceptance.json',{'status':'accepted','analysis':ref(out/'analysis.json'),'terminal_cost':ref(out/'phases/analysis/terminal-cost.json'),'stage':stage})
 result=read(ref(out/'analysis.json'));return {'status':'complete','primary':result['primary'],'analysis':ref(out/'analysis.json')}

def main(argv=None):
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--mode',choices=('validate','controls','profile','run','analyze','campaign'));parser.add_argument('--internal-phase')
 args=parser.parse_args(argv);c=load_contract(args.config)
 if args.internal_phase:
  refs=execution_refs(c,args.config);require_phase_guard(c,args.internal_phase,refs,args.config)
  if args.internal_phase=='input-validation':
   record_phase(c,'input-validation',source_files(args.config),lambda rec:validate_input_statistics(c));print(json.dumps({'phase':'input-validation','status':'complete'}));return
  if args.internal_phase=='analysis':
   analyze(c,args.config);print(json.dumps({'phase':'analysis','status':'complete'}));return
  if not args.internal_phase.startswith('prepare-') or args.internal_phase[8:] not in [r['label'] for r in fixed_grid()]:raise ValueError('unknown finite preparation phase')
  prepare_artifact(c,args.internal_phase[8:],source_files(args.config));print(json.dumps({'phase':args.internal_phase,'status':'complete'}));return
 if args.mode=='validate':
  metadata_preflight(c);print(json.dumps({'status':'valid','study':c['study'],'points':len(fixed_grid()),'model_loaded':False,'neural_work':False}));return
 if args.mode=='controls':
  import subprocess
  subprocess.run([sys.executable,'-m','pytest','tests/test_shared_gptq.py','tests/test_shared_quantization.py','tests/test_shared_gptq_study.py','-q'],cwd=ROOT,check=True);return
 if args.mode=='profile':
  print(json.dumps(campaign(c,args.config,profile_only=True)));return
 if args.mode=='run':
  print(json.dumps(run_from_freeze(c,args.config)));return
 if args.mode=='campaign':
  print(json.dumps(campaign(c,args.config)));return
 elif args.mode=='analyze':
  print(json.dumps(dispatch_analysis(c,args.config)));return
 else:raise ValueError('explicit mode required')
 print(json.dumps({'status':'complete','primary':result['primary'],'analysis':ref(Path(c['out'])/'analysis.json')}))


if __name__=='__main__':main()

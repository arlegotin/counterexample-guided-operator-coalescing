"""Explicit CPU all-use calibration study with finite qualified execution gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib,json
from dataclasses import asdict,replace
from poc.contracts import thaw
from poc.data.manifest import load_manifest
from poc.data.access import open_records
from poc.records import canonical_json


def _digest(value):
 import re
 if type(value)is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA256 required')
 return value


def _positive(value):
 if type(value)is not int or value<1:raise ValueError('positive integer bound required')
 return value


def _regular(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file() or p.stat().st_nlink!=1:raise ValueError('regular unlinked study input required')
 return p


def inspect_fit_metadata(contract):
 """Role/count/identity/size inventory only; never open corpus content."""
 c=thaw(contract);ref=c['fit_manifest'];_digest(ref['sha256']);_digest(c['tokenizer_sha256'])
 if set(ref)!={'path','sha256'}:raise ValueError('closed fit manifest reference required')
 source_count=_positive(c['source_components']);count=_positive(c['selected_components'])
 vocabulary=_positive(c['vocabulary_size']);text_cap=_positive(c['max_document_bytes']);token_cap=_positive(c['max_token_file_bytes'])
 if count>source_count:raise ValueError('selected fit components exceed source inventory')
 path=_regular(ref['path'])
 if path.stat().st_size>32*1024**2:raise ValueError('fit metadata exceeds its finite bound')
 payload=path.read_bytes()
 if hashlib.sha256(payload).hexdigest()!=ref['sha256']:raise ValueError('fit manifest content changed')
 raw=json.loads(payload)
 if not raw.get('roles') or set(raw['roles'].values())!={'fit'}:raise PermissionError('calibration requires a wholly fit-only source manifest')
 parent=load_manifest(path,expected_sha256=ref['sha256'],roles={'fit'})
 if (parent.quarantine or not parent.exposure_path or len(parent.records)!=source_count
     or len(set(parent.membership.values()))!=source_count):raise ValueError('exact unquarantined component-disjoint fit source required')
 out=Path(c['out']).resolve();source_root=path.resolve().parent
 if out.is_relative_to(source_root) or source_root.is_relative_to(out):raise ValueError('study data output may not overlap frozen source population')
 selected=tuple(sorted(parent.records,key=lambda r:r.document_id)[:count]);source_sizes={}
 for record in selected:
  if record.metadata.get('tokenizer_digest')!=c['tokenizer_sha256']:raise ValueError('fit tokenizer identity differs')
  if record.text is not None or not record.text_path or not record.text_sha256 or not record.token_path or not record.token_sha256:raise ValueError('deferred hash-bound text and token records required')
  text=_regular(record.text_path);tokens=_regular(record.token_path)
  if text.stat().st_size>text_cap or tokens.stat().st_size>token_cap:raise ValueError('selected document/token file exceeds its finite read bound')
  _digest(record.text_sha256);_digest(record.token_sha256)
  source_sizes[record.document_id]={'text_bytes':text.stat().st_size,'token_bytes':tokens.stat().st_size}
 return {'parent':parent,'selected':selected,'source_sizes':source_sizes,'manifest_path':path,'out':out}


def prepare_windows(contract):
 """Select roles/components before content; append only a new study journal.

 Metadata/window construction is not a neural presentation. The returned
 provenance distinguishes document access, full token files and model contexts.
 """
 from poc.teacher_validation import _batch
 from poc.recovery_study import NaturalWindow
 from poc.shared_calibration import window_plan_sha256
 c=thaw(contract);ref=c['fit_manifest'];metadata=inspect_fit_metadata(c)
 parent=metadata['parent'];selected=metadata['selected'];source_sizes=metadata['source_sizes']
 path=metadata['manifest_path'];out=metadata['out'];token_cap=c['max_token_file_bytes'];vocabulary=c['vocabulary_size']
 ids={r.document_id for r in selected};inputs=out/'inputs';journal=inputs/'exposure.jsonl'
 if journal.resolve()==Path(parent.exposure_path).resolve():raise ValueError('new calibration exposure journal required')
 subset=replace(parent,records=selected,roles={i:'fit' for i in ids},membership={i:parent.membership[i] for i in ids},
                salt=parent.salt+':'+c['study'],exposure_path=str(journal),lineage_sha256=parent.digest(),quarantine={})
 subset_payload=canonical_json(asdict(subset));subset_path=inputs/'fit-subset-manifest.json'
 if subset_path.exists():
  if _regular(subset_path).read_bytes()!=subset_payload:raise ValueError('immutable calibration subset/input plan changed')
 else:
  inputs.mkdir(parents=True,exist_ok=True)
  with subset_path.open('xb') as f:f.write(subset_payload)
 batches=[];records=[];token_ids_read=0
 for record in open_records(subset,purpose='calibration'):
  token_path=_regular(record.token_path);token_payload=token_path.read_bytes()
  if len(token_payload)>token_cap or hashlib.sha256(token_payload).hexdigest()!=record.token_sha256:raise ValueError('natural token content changed')
  row=json.loads(token_payload)
  if (set(row)!={'document_id','fields','tokenizer_digest'} or row['document_id']!=record.document_id
      or row['tokenizer_digest']!=c['tokenizer_sha256'] or set(row['fields'])!={'text'}):raise ValueError('token source/tokenizer/document schema differs')
  tokens=row['fields']['text']
  if not isinstance(tokens,list) or len(tokens)<512 or any(type(i)is not int or not 0<=i<vocabulary for i in tokens):raise ValueError('eligible canonical token context required')
  batch=_batch(tokens[:512],device='cpu',document_id=record.document_id,cluster_id=subset.membership[record.document_id],targets=range(1,512))
  NaturalWindow.from_batch(batch);batches.append(batch);token_ids_read+=len(tokens)
  records.append({'document_id':record.document_id,'component_id':subset.membership[record.document_id],
                  'token_path':record.token_path,'token_sha256':record.token_sha256,'text_sha256':record.text_sha256,
                  'window_sha256':window_plan_sha256((batch,)),**source_sizes[record.document_id]})
 if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('source fit metadata changed during access')
 provenance={'source_manifest_sha256':ref['sha256'],'source_manifest_semantic_sha256':parent.digest(),
             'subset_manifest_sha256':subset.digest(),'subset_manifest_path':str(subset_path),'source_exposure_path':parent.exposure_path,
             'new_exposure_path':str(journal),'guarded_documents':len(records),'unique_components':len(set(subset.membership.values())),
             'full_source_token_ids_read':token_ids_read,'model_context_tokens':len(batches)*512,'planned_model_raw_targets':len(batches)*511,
             'window_plan_sha256':window_plan_sha256(batches),'records':records,
             'scope':'existing fit development; source lineage preserved in a new journal, not fresh confirmation; document accesses distinct from model target presentations'}
 return subset,tuple(batches),provenance


def load_contract(path):
 """Closed immutable structural contract; never grants execution permission."""
 import yaml,math,re
 from poc.contracts import freeze
 try:raw=yaml.safe_load(Path(path).read_text())
 except yaml.YAMLError as exc:raise ValueError('invalid calibration study YAML') from exc
 def fields(value,names,label):
  if not isinstance(value,dict) or set(value)!=set(names):raise ValueError('closed '+label+' fields required')
 def absolute(value):
  if type(value)is not str or not value or not Path(value).is_absolute():raise ValueError('absolute explicit study path required')
 def reference(value):
  fields(value,('path','sha256'),'file reference');absolute(value['path']);_digest(value['sha256'])
 def seconds(value):
  if type(value)not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('finite positive resource time required')
 fields(raw,('schema_version','study','out','source_bundle','parent_qualification','ancestor_config','fit_manifest',
             'tokenizer_sha256','vocabulary_size','source_components','selected_components','max_document_bytes',
             'max_token_file_bytes','expected_layout','resource','serial_gate'),'study')
 if type(raw['schema_version'])is not int or raw['schema_version']!=1:raise ValueError('only calibration study version1 qualified')
 if type(raw['study'])is not str or not re.fullmatch('[a-z0-9][a-z0-9-]{0,79}',raw['study']):raise ValueError('explicit isolated study identifier required')
 absolute(raw['out']);fields(raw['source_bundle'],('directory','manifest_sha256'),'source BundleRef')
 absolute(raw['source_bundle']['directory']);_digest(raw['source_bundle']['manifest_sha256'])
 for key in ('parent_qualification','ancestor_config','fit_manifest'):reference(raw[key])
 _digest(raw['tokenizer_sha256'])
 for key in ('vocabulary_size','source_components','selected_components','max_document_bytes','max_token_file_bytes'):_positive(raw[key])
 if raw['selected_components']>min(raw['source_components'],16):raise ValueError('bounded firstN<=16 fit selection required')
 source=Path(raw['source_bundle']['directory']).resolve();out=Path(raw['out']).resolve()
 if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('source bundle/output overlap forbidden')
 layout=raw['expected_layout'];fields(layout,('hidden_size','intermediate_size','num_hidden_layers','banks','rank'),'expected layout')
 for key in ('hidden_size','intermediate_size','num_hidden_layers','banks'):_positive(layout[key])
 if (layout['banks']>layout['num_hidden_layers'] or type(layout['rank'])is not int
     or not 0<=layout['rank']<=min(layout['hidden_size'],layout['intermediate_size'])):raise ValueError('invalid bank/rank representation')
 resource=raw['resource'];fields(resource,('profile','run','deadline_unix'),'resource')
 profile=resource['profile'];run=resource['run']
 fields(profile,('wall_seconds','memory_bytes','output_disk_bytes','max_statistics_bytes'),'profile resource')
 fields(run,('wall_ceiling','memory_bytes','family_disk_bytes'),'run resource ceilings')
 seconds(profile['wall_seconds']);seconds(run['wall_ceiling']);seconds(resource['deadline_unix'])
 for key in ('memory_bytes','output_disk_bytes','max_statistics_bytes'):_positive(profile[key])
 for key in ('memory_bytes','family_disk_bytes'):_positive(run[key])
 h=layout['hidden_size'];m=layout['intermediate_size'];estimated=8*layout['banks']*(2*(h*h+2*h)+m*m+2*m)
 if (estimated>profile['max_statistics_bytes'] or estimated>profile['memory_bytes']
     or estimated>run['memory_bytes'] or estimated>profile['output_disk_bytes']
     or 2*estimated>run['family_disk_bytes']):raise ValueError('statistics tensor storage is infeasible under frozen bounds')
 gate=raw['serial_gate'];fields(gate,('directory','queue_freeze_sha256'),'serial gate')
 absolute(gate['directory']);_digest(gate['queue_freeze_sha256'])
 return freeze(raw)


def inspect_source_bundle(contract):
 """Bounded file/hash inventory only; model verification belongs to setup.

 The exact manifest/content hashes bind the accepted source. This does not
 invoke verify_bundle/_verified_ref/measure_bundle: schema-v2 verification
 reconstructs a model, which must occur inside the durable cold setup stage.
 """
 from poc.contracts import BundleRef
 from poc.bundle import _regular_files
 from poc.runtime.torch_decoder import EXECUTION_PROFILE
 c=thaw(contract);source=c['source_bundle'];path=Path(source['directory'])
 if not path.is_absolute() or path.is_symlink() or path.resolve()!=path:raise ValueError('canonical source inventory path required')
 manifest=_regular(path/'manifest.json')
 if manifest.stat().st_size>16*1024**2:raise ValueError('source metadata exceeds finite inventory bound')
 payload=manifest.read_bytes()
 if hashlib.sha256(payload).hexdigest()!=source['manifest_sha256']:raise ValueError('source manifest byte identity differs')
 metadata=json.loads(payload);entries=metadata['file_entries'];files=_regular_files(path)
 if set(files)!={'manifest.json',*entries}:raise ValueError('source inventory contains missing/extra files')
 complete=0
 for name,file in files.items():
  _regular(file);size=file.stat().st_size;complete+=size
  if name=='manifest.json':continue
  entry=entries[name]
  if set(entry)!={'bytes','sha256'} or type(entry['bytes'])is not int or entry['bytes']!=size:raise ValueError('source file byte census differs')
  _digest(entry['sha256']);h=hashlib.sha256()
  with file.open('rb') as stream:
   while block:=stream.read(1024**2):h.update(block)
  if h.hexdigest()!=entry['sha256']:raise ValueError('source file content changed')
 layout=c['expected_layout'];config=metadata['config']
 if (metadata['kind']!='compact' or metadata['rank']!=layout['rank'] or len(metadata['groups'])!=layout['banks']
     or any(config[k]!=layout[k] for k in ('hidden_size','intermediate_size','num_hidden_layers'))
     or config['vocab_size']!=c['vocabulary_size'] or config['max_position_embeddings']<512
     or metadata['tokenizer_digest']!=c['tokenizer_sha256'] or metadata['execution_profile']!=EXECUTION_PROFILE):
  raise ValueError('declared source representation/tokenizer/profile differs')
 quantization=json.loads((path/'quantization.json').read_bytes())
 if quantization.get('format')!='floating' or any(owner['dtype']!='float32' for owner in metadata['owners']):raise ValueError('current-student calibration requires declared floating FP32 source')
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=source['manifest_sha256']:raise ValueError('source manifest changed during inventory')
 return {'source_ref':BundleRef(source['directory'],source['manifest_sha256']),'metadata':metadata,'complete_bytes':complete,
         'source_runtime_sha256':entries['runtime/poc-runtime.zip']['sha256'],'model_reconstruction_performed':False,
         'scope':'pinned source manifest/file-hash inventory only; complete schema/model reconstruction deferred to guarded durable setup; no model load or inference'}


def read_reference(ref):
 if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('closed immutable file reference required')
 _digest(ref['sha256']);path=_regular(ref['path'])
 if path.stat().st_size>64*1024**2:raise ValueError('metadata reference exceeds its finite bound')
 payload=path.read_bytes()
 if hashlib.sha256(payload).hexdigest()!=ref['sha256']:raise ValueError('referenced metadata changed')
 return payload


def verify_parent_context(contract):
 """Reuse the exact qualified parent reader for ancestry, never model work.

 The imported reader must be byte-identical to our inherited, frozen copy.
 Its source root remains the parent's root; our scientific package namespace
 and sys.path stay bound to this study. This is metadata admission, not the
 new executor's own qualification or its later serial release gate.
 """
 import importlib.util
 c=thaw(contract);parent_ref=c['parent_qualification'];parent=json.loads(read_reference(parent_ref))
 if parent.get('status')!='complete':raise ValueError('accepted parent implementation qualification required')
 qpath=Path(parent_ref['path']).resolve()
 if qpath.parent!=Path(c['serial_gate']['directory']).resolve():raise ValueError('parent qualification and serial campaign differ')
 parent_root=qpath.parents[2];reader=parent_root/'provenance/sharing_frontier_study.py';inherited=ROOT/'provenance/sharing_frontier_study.py'
 actual=_regular(reader).read_bytes()
 if hashlib.sha256(actual).digest()!=hashlib.sha256(_regular(inherited).read_bytes()).digest():raise ValueError('only the exact inherited qualified metadata reader may execute')
 ancestor_bytes=read_reference(c['ancestor_config'])
 saved_path=list(sys.path)
 try:
  spec=importlib.util.spec_from_file_location('_qualified_parent_frontier_metadata',reader)
  peer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=peer;spec.loader.exec_module(peer)
  ancestor=peer.load_contract(Path(c['ancestor_config']['path']))
  accepted=peer.require_current_qualification(ancestor)
  if canonical_json(accepted)!=canonical_json(parent):raise ValueError('declared parent qualification does not bind ancestor configuration')
  preparation=peer.verify_source_receipts(ancestor)
 finally:
  sys.path[:]=saved_path
 if ancestor['shared']!=c['source_bundle']:raise ValueError('calibration source differs from accepted recovery export')
 observation=Path(ancestor['reference']['scientific_observation']['path']);source_spec=observation.parents[1]/'spec.json'
 spec_ref={'path':str(source_spec),'sha256':ancestor['reference']['source_run_id']};source=json.loads(read_reference(spec_ref))
 if (source['input_manifests'].get('fit')!=c['fit_manifest']['sha256']
     or source['input_manifests'].get('assets')!=c['tokenizer_sha256']):raise ValueError('fit/tokenizer identities differ from frozen recovered source')
 return {'parent_qualification':parent_ref,'ancestor_config':c['ancestor_config'],'source_spec':spec_ref,
         'source_run_id':ancestor['reference']['source_run_id'],'source_preparation':thaw(ancestor['reference']['preparation']),
         'verified_source_bundle':dict(c['source_bundle']),'parent_source_count':len(parent['source_files']),
         'scope':'accepted source/fit/tokenizer ancestry only; no new model/data/quality execution or serial permission'}


def collect_and_save(layout,population,windows,*,destination,study_contract_sha256,max_statistics_bytes,recorder):
 """Costed library operation; actual study entrypoints need separate gates."""
 import time
 from dataclasses import asdict
 from poc.shared_calibration import SharedCalibrationSpec,collect_shared_inputs,window_plan_sha256,_model_sha
 from poc.shared_calibration_bundle import save_calibration,load_calibration
 from poc.train import _call
 _digest(study_contract_sha256);_positive(max_statistics_bytes)
 source_sha=_model_sha(layout);plan_sha=window_plan_sha256(windows)
 spec=SharedCalibrationSpec(layout_sha256=source_sha,source_sha256=source_sha,
       population_sha256=population.digest(),window_plan_sha256=plan_sha,tokenizer_sha256=layout.tokenizer_digest,
       source='current_student',max_windows=16,max_statistics_bytes=max_statistics_bytes)
 tick=time.monotonic();collected=collect_shared_inputs(layout,layout,population,windows,spec,recorder)
 collection_seconds=time.monotonic()-tick;result_sha=collected.digest()
 tick=time.monotonic()
 ref=_call(recorder,lambda:save_calibration(collected,Path(destination),expected_result_sha256=result_sha,
                  study_contract_sha256=study_contract_sha256))
 save_seconds=time.monotonic()-tick;tick=time.monotonic()
 loaded=_call(recorder,lambda:load_calibration(ref,expected_contract_sha256=study_contract_sha256))
 reload_seconds=time.monotonic()-tick
 if (loaded.digest()!=result_sha or _model_sha(layout)!=source_sha or window_plan_sha256(windows)!=plan_sha
     or population.digest()!=spec.population_sha256):raise ValueError('model/data/statistics changed during collection/persistence')
 return {'statistics_ref':ref,'result_sha256':result_sha,'cpu_source_sha256':source_sha,
         'window_plan_sha256':plan_sha,'spec':spec.identity(),'spec_sha256':spec.digest(),
         'counters':dict(collected.counters),'canonical_owners':len(collected.owners),
         'timings':{'collection_seconds':collection_seconds,'save_seconds':save_seconds,'reload_seconds':reload_seconds},
         'scope':'CPU current-student input statistics and independent reload; no KL/NLL, teacher query, optimizer update or quantization result'}


def forecast_calibration(profile,contract):
 """Measured finite16-or-fewer-context plan; no quality or optimizer schedule."""
 import math
 c=thaw(contract);p=profile
 required={'schema_version','mode','study_contract_sha256','numerical_profile','canonical_owners','counters','queries',
           'timings','wall_seconds','process_peak_rss_bytes','statistics_complete_bytes','retained_family_bytes'}
 optional={'statistics_ref','result_sha256','cpu_source_sha256','window_plan_sha256','spec','spec_sha256',
           'data_provenance','cost','run_id','source_runtime_copy','configuration_sha256','source_manifest_sha256',
           'source_files_sha256','scope','environment','ancestry','cost_snapshot_preliminary','terminal_cost_path'}
 if not isinstance(p,dict) or not required<=set(p) or set(p)-required-optional:raise ValueError('closed quality-free calibration profile required')
 identity=hashlib.sha256(canonical_json(c)).hexdigest()
 if (type(p['schema_version'])is not int or p['schema_version']!=1 or p['mode']!='profile'
     or p['study_contract_sha256']!=identity or p['numerical_profile']!='shared-inputs-cpu-fp32-moments-fp64-v1'):
  raise ValueError('profile source/configuration/numerical identity differs')
 layout=c['expected_layout'];h=layout['hidden_size'];m=layout['intermediate_size'];banks=layout['banks'];depth=layout['num_hidden_layers'];n=c['selected_components']
 statistics_bytes=8*banks*(2*(h*h+2*h)+m*m+2*m)
 expected={'statistics_bytes':statistics_bytes,'raw_target_presentations':511,'unique_context_target_positions':511,
           'unique_contexts':1,'processed_context_tokens':512,'operator_input_presentations':511*depth,
           'projection_input_presentations':511*depth*3}
 queries={'student_forward_calls':1,'teacher_forward_calls':0,'valid_target_presentations':0,'raw_corpus_target_presentations':511}
 if (p['counters']!=expected or any(type(v)is not int for v in p['counters'].values())
     or p['queries']!=queries or any(type(v)is not int for v in p['queries'].values())
     or type(p['canonical_owners'])is not int or p['canonical_owners']!=3*banks):raise ValueError('profile did not complete exact one-context/all-use work')
 times=p['timings'];names={'setup_seconds','collection_seconds','save_seconds','reload_seconds','finalization_seconds'}
 if not isinstance(times,dict) or set(times)!=names:raise ValueError('complete setup/collection/save/load/finalization timing required')
 for key,value in times.items():
  if type(value)not in (int,float) or not math.isfinite(value) or value<0 or key!='finalization_seconds' and value==0:raise ValueError('finite measured timing required')
 wall=p['wall_seconds']
 if type(wall)not in (int,float) or not math.isfinite(wall) or wall<sum(times.values()) or wall>c['resource']['profile']['wall_seconds']:raise ValueError('profile elapsed time is invalid or over budget')
 peak=_positive(p['process_peak_rss_bytes']);complete=_positive(p['statistics_complete_bytes']);family=_positive(p['retained_family_bytes'])
 if peak>min(c['resource']['profile']['memory_bytes'],c['resource']['run']['memory_bytes']):raise ValueError('observed profile peak does not fit run memory')
 if complete<statistics_bytes or complete>statistics_bytes+16*1024**2:raise ValueError('actual complete statistics bytes are inconsistent')
 forecast=times['setup_seconds']+n*times['collection_seconds']+times['save_seconds']+times['reload_seconds']+times['finalization_seconds']
 wall_budget=math.ceil((2*forecast+120)/30)*30;reserve=complete+16*1024**2
 if wall_budget>c['resource']['run']['wall_ceiling'] or family+reserve>c['resource']['run']['family_disk_bytes']:raise ValueError('completed calibration schedule is infeasible under frozen ceilings')
 full={**expected,'raw_target_presentations':511*n,'unique_context_target_positions':511*n,'unique_contexts':n,
       'processed_context_tokens':512*n,'operator_input_presentations':511*n*depth,'projection_input_presentations':511*n*depth*3}
 return {'schema_version':1,'study_contract_sha256':identity,'forecast_seconds':forecast,
         'resource':{'wall_seconds':wall_budget,'memory_bytes':c['resource']['run']['memory_bytes'],'disk_bytes':c['resource']['run']['family_disk_bytes']},
         'queries':{**queries,'student_forward_calls':n,'raw_corpus_target_presentations':511*n},'expected_counters':full,
         'canonical_owners':3*banks,'future_statistics_reserve_bytes':reserve,
         'scope':'completed prespecified finite context plan; conservative linear whole one-context collection cost includes repeated fixed collector overhead; twice forecast+120seconds rounded up30; no quality selection or teacher/backward work'}


def _start_collection_attempt(*,config,mode,resource,source_commit,source_files,configuration_sha256,parent_run_id,
                              setup_seconds=0.,retained_family=None,declared_inputs=None):
 """Durable intent and budget exist before corpus content or model setup."""
 import importlib.util,time,math
 from poc.contracts import Budget,CostRecord,RunSpec
 from poc.records import start_attempt
 from poc.budget import BudgetExceeded
 c=thaw(config);n=1 if mode=='profile' else c['selected_components'] if mode=='run' else 0
 if mode not in ('profile','run','analysis') or set(resource)!={'wall_seconds','memory_bytes','disk_bytes'}:raise ValueError('complete finite calibration phase required')
 if type(setup_seconds)not in (int,float) or not math.isfinite(setup_seconds) or setup_seconds<0:raise ValueError('finite nonnegative setup time required')
 budget=Budget(updates=0,valid_target_presentations=0,teacher_forward_calls=0,suffix_backward_calls=0,
    student_backward_calls=0,search_proposals=0,search_seconds=0,**resource)
 _digest(configuration_sha256);_digest(parent_run_id)
 if not source_files or any(_digest(v)!=v for v in source_files.values()):raise ValueError('exact source manifest required')
 path=ROOT/'provenance/recovery_diagnosis.py';spec=importlib.util.spec_from_file_location('_durable_calibration_common',path)
 common=importlib.util.module_from_spec(spec);sys.modules[spec.name]=common;spec.loader.exec_module(common)
 identity=hashlib.sha256(canonical_json(c)).hexdigest();phase=Path(c['out'])/mode
 if phase.exists():raise FileExistsError('calibration phase already attempted; retain it')
 intent=declared_inputs or {'source_bundle':c['source_bundle'],'fit_manifest':c['fit_manifest'],
   'selection':'first selected_components documents sorted by document_id before content access',
   'selected_components':c['selected_components'],'forward_contexts':n,'context_tokens':512,'shifted_targets_per_context':511,
   'window_policy':'first512tokens/fullattention/resetpositions/noimplicitBOS; token/text hashes and roles verified before dispatch'}
 intent_sha=hashlib.sha256(canonical_json(intent)).hexdigest()
 run=RunSpec('Q',c['study']+':'+mode,{'contract':c,'mode':mode,'source_files':source_files,'expected_raw_targets':n*511,
       'expected_student_calls':n,'declared_input_intent':intent},source_commit,hashlib.sha256(canonical_json(source_files)).hexdigest(),
       common.hashed(common.environment()),{'configuration':configuration_sha256,'contract':identity,
          'fit_manifest':c['fit_manifest']['sha256'],'source_bundle':c['source_bundle']['manifest_sha256'],'input_intent':intent_sha},
       {'device':'CPU FP32 forward; FP64 uncentered moments; no loss/readout/backward'},
       {'source_recovery':17},(parent_run_id,),budget)
 phase.mkdir(parents=True,exist_ok=False);attempt=start_attempt(run,phase/'records')
 class Tracker(common.StudyTracker):
  def disk(self):return retained_family.disk() if retained_family is not None else super().disk()
  def check(self,cost):
   total=self.combine(self.totals,cost)
   if total.student_forward_calls>n or total.raw_corpus_target_presentations>511*n:raise BudgetExceeded('finite calibration raw/context allowance exceeded')
   super().check(cost)
  def record(self,cost):
   super().record(cost)
   if self.totals.student_forward_calls>n or self.totals.raw_corpus_target_presentations>511*n:raise BudgetExceeded('recorded calibration raw/context overrun retained')
 life={'common':common,'recorder':Tracker(budget,attempt,Path(c['out'])),'attempt':attempt,'phase':phase,
       'started':time.monotonic()-setup_seconds,'identity':identity,'mode':mode,'closed':False}
 return life


def _fail_collection_attempt(life,exc):
 import time
 from poc.train import finish_attempt_cost
 if life['closed']:return
 try:finish_attempt_cost(life['recorder'],elapsed=time.monotonic()-life['started'])
 finally:
  life['common'].finish_attempt(life['attempt'],'failed',reason=repr(exc));life['closed']=True
  life['common'].write_new(life['phase']/'failed.json',{'status':'failed','error':repr(exc),'run_id':life['attempt'].run_id,
     'scope':'durable input/setup/collection work and exposures retained; no automatic retry or accepted phase'})


def _close_collection_attempt(life,reason):
 import time
 from poc.train import finish_attempt_cost
 rec=life['recorder'];finish_attempt_cost(rec,elapsed=time.monotonic()-life['started']);totals=rec.validate()
 terminal={'schema_version':1,'run_id':life['attempt'].run_id,'study_contract_sha256':life['identity'],
    'ledger':file_reference(rec.path),'totals':totals.to_dict(),
    'scope':'authoritative common ledger through all data/model/collection/statistics/packet work and final topup; terminal receipt and attempt-closing bookkeeping are covered by enclosing guard, never recursively rewritten'}
 life['common'].write_new(life['phase']/'terminal-cost.json',terminal)
 life['common'].finish_attempt(life['attempt'],'exhausted',reason=reason);life['closed']=True
 return terminal


def read_terminal_cost(reference,*,expected_run_id):
 from poc.budget import BudgetTracker
 value=json.loads(read_reference(reference))
 if (set(value)!={'schema_version','run_id','study_contract_sha256','ledger','totals','scope'} or value['schema_version']!=1
     or value['run_id']!=expected_run_id):raise ValueError('exact terminal calibration cost receipt required')
 payload=read_reference(value['ledger']);rows=[json.loads(line) for line in payload.splitlines()]
 if (not rows or any(row.get('kind')!='cost' or row.get('event_id')!=i+1 for i,row in enumerate(rows))
     or Path(value['ledger']['path']).parent.name!=expected_run_id):raise ValueError('terminal cost ledger identity/order differs')
 # This existing aggregation routine is stateless and validates every CostRecord.
 total=BudgetTracker._total(None,rows).to_dict()
 if total!=value['totals']:raise ValueError('terminal ledger totals differ from receipt')
 return value


def recorded_collection(layout,population,windows,*,config,mode,resource,source_commit,source_files,
                        configuration_sha256,parent_run_id,setup_seconds=0.,retained_family=None,setup_records=(),
                        data_provenance=None,source_runtime_copy=None,_lifecycle=None):
 """Same durable collector for owned loaded controls and the cold worker."""
 import math,time
 from dataclasses import asdict
 from poc.shared_calibration import _model_sha,window_plan_sha256
 from poc.train import finish_attempt_cost
 c=thaw(config);n=1 if mode=='profile' else c['selected_components'] if mode=='run' else 0
 if (not n or len(windows)!=n or any(int(b.target_mask.sum())!=511 for b in windows)
     or type(setup_seconds)not in (int,float) or not math.isfinite(setup_seconds) or setup_seconds<0):raise ValueError('exact finite collection context/setup contract required')
 source_sha=_model_sha(layout);plan_sha=window_plan_sha256(windows)
 life=_lifecycle or _start_collection_attempt(config=c,mode=mode,resource=resource,source_commit=source_commit,source_files=source_files,
      configuration_sha256=configuration_sha256,parent_run_id=parent_run_id,setup_seconds=setup_seconds,retained_family=retained_family,
      declared_inputs={'cpu_source_sha256':source_sha,'population_sha256':population.digest(),'window_plan_sha256':plan_sha})
 if life['identity']!=hashlib.sha256(canonical_json(c)).hexdigest() or life['mode']!=mode or life['closed']:raise ValueError('active durable collection intent differs')
 rec=life['recorder'];phase=life['phase'];identity=life['identity'];common=life['common']
 try:
  for row in setup_records:rec.record(row)
  rec.write(phase/'setup-complete.json',{'study_contract_sha256':identity,'cpu_source_sha256':source_sha,
      'window_plan_sha256':plan_sha,'population_sha256':population.digest(),'source_runtime_copy':source_runtime_copy,
      'data_provenance':data_provenance,'scope':'actual validated identities extend immutable declared source/input intent before forward dispatch'})
  value=collect_and_save(layout,population,windows,destination=phase/'statistics',study_contract_sha256=identity,
                        max_statistics_bytes=c['resource']['profile']['max_statistics_bytes'],recorder=rec)
  finalizing=time.monotonic();finish_attempt_cost(rec,elapsed=time.monotonic()-life['started']);totals=rec.totals
  queries={key:getattr(totals,key) for key in ('student_forward_calls','teacher_forward_calls','valid_target_presentations','raw_corpus_target_presentations')}
  if queries!={'student_forward_calls':n,'teacher_forward_calls':0,'valid_target_presentations':0,'raw_corpus_target_presentations':511*n}:raise ValueError('actual calibration units differ from immutable intent')
  ref=value['statistics_ref'];complete=sum(p.stat().st_size for p in Path(ref.directory).iterdir() if p.is_file())
  packet={'schema_version':1,'mode':mode,'study_contract_sha256':identity,'numerical_profile':'shared-inputs-cpu-fp32-moments-fp64-v1',
    **value,'statistics_ref':asdict(ref),'timings':{'setup_seconds':setup_seconds,**value['timings'],'finalization_seconds':time.monotonic()-finalizing},
    'queries':queries,'wall_seconds':time.monotonic()-life['started'],'process_peak_rss_bytes':common.process_peak_bytes(),
    'statistics_complete_bytes':complete,'retained_family_bytes':rec.disk(),'cost':totals.to_dict(),'cost_snapshot_preliminary':True,
    'terminal_cost_path':str(phase/'terminal-cost.json'),'run_id':life['attempt'].run_id,'configuration_sha256':configuration_sha256,
    'source_files_sha256':hashlib.sha256(canonical_json(source_files)).hexdigest()}
  if data_provenance is not None:packet['data_provenance']=data_provenance
  if source_runtime_copy is not None:packet['source_runtime_copy']=source_runtime_copy
  rec.write(phase/'packet.json',packet);_close_collection_attempt(life,'complete finite calibration and independently reloaded statistics')
  return packet
 except BaseException as exc:_fail_collection_attempt(life,exc);raise


def prepare_cpu_source(contract,*,recorder):
 """Caller-admitted, costed cold CPU source load; no neural dispatch.

 Copies complete weights/tokenizer/routing unchanged and binds the separately
 packaged current runtime. Recovery checkpoints and training owners are not
 modified. Actual CLI use still requires its own qualification/release gates.
 """
 import importlib.util
 import torch
 from dataclasses import asdict
 from poc.bundle import load_bundle,measure_bundle,verify_bundle
 from poc.train import _call
 from poc.shared_calibration import _model_sha
 c=thaw(contract);source=inspect_source_bundle(c);destination=Path(c['out'])/'source-runtime-copy'
 if destination.exists():raise FileExistsError('CPU source copy already owned; do not overwrite it')
 path=ROOT/'provenance/sharing_frontier_study.py';spec=importlib.util.spec_from_file_location('_calibration_inherited_runtime_copy',path)
 inherited=importlib.util.module_from_spec(spec);sys.modules[spec.name]=inherited;spec.loader.exec_module(inherited)
 copied=_call(recorder,lambda:inherited.refresh_runtime(source['source_ref'],destination))
 verified=verify_bundle(destination)
 if thaw(verified.metadata)['groups']!=source['metadata']['groups']:raise ValueError('copied canonical routing changed')
 model=_call(recorder,lambda:load_bundle(destination,backend='torch',offline=True))
 if any(p.device.type!='cpu' or p.dtype!=torch.float32 or p.grad is not None for p in model.parameters()):raise ValueError('CPU FP32 clean source owners required')
 if (len(model.bank)!=c['expected_layout']['banks'] or model.rank!=c['expected_layout']['rank']
     or len(model.layers)!=c['expected_layout']['num_hidden_layers'] or model.tokenizer_digest!=c['tokenizer_sha256']):raise ValueError('loaded CPU source representation differs')
 inspect_source_bundle(c)
 return model,{'source':dict(c['source_bundle']),'copied':asdict(copied),'cpu_source_sha256':_model_sha(model),
               'source_complete_bytes':source['complete_bytes'],'copied_complete_bytes':measure_bundle(destination).payload_bytes,
               'source_runtime_sha256':source['source_runtime_sha256'],
               'copied_runtime_sha256':hashlib.sha256((destination/'runtime/poc-runtime.zip').read_bytes()).hexdigest(),
               'scope':'verified complete CPU source/runtime copy only; no forward, teacher query, loss or optimizer update'}


def cold_cpu_collection(contract,*,mode,resource,source_commit,source_files,configuration_sha256,parent_run_id,
                        profile_reference=None,retained_family=None):
 """Durable source/input intent precedes every cold setup/content operation."""
 import time,torch
 from poc.contracts import BundleRef
 from poc.bundle import load_bundle,verify_bundle
 from poc.shared_calibration import _model_sha
 from poc.train import _call
 c=thaw(contract);profile=None
 if mode=='run':
  if profile_reference is None:raise ValueError('cold run requires exact accepted profile')
  profile=json.loads(read_reference(profile_reference));planned=forecast_calibration(profile,c)
  if resource!=planned['resource']:raise ValueError('cold run resource plan differs')
 elif mode!='profile' or profile_reference is not None:raise ValueError('explicit cold phase required')
 life=_start_collection_attempt(config=c,mode=mode,resource=resource,source_commit=source_commit,source_files=source_files,
     configuration_sha256=configuration_sha256,parent_run_id=parent_run_id,retained_family=retained_family)
 rec=life['recorder']
 try:
  population,windows,data=_call(rec,lambda:prepare_windows(c))
  if len(windows)!=c['selected_components']:raise ValueError('complete declared natural context plan required')
  if profile is None:model,source=prepare_cpu_source(c,recorder=rec)
  else:
   if profile['data_provenance']['window_plan_sha256']!=data['window_plan_sha256']:raise ValueError('profile/run input plan differs')
   source=profile['source_runtime_copy'];copied=source['copied']
   if source['source']!=c['source_bundle'] or copied['directory']!=str(Path(c['out'])/'source-runtime-copy'):raise ValueError('profile CPU source copy differs')
   ref=BundleRef(**copied);_call(rec,lambda:verify_bundle(Path(ref.directory)))
   if file_reference(Path(ref.directory)/'manifest.json')['sha256']!=ref.manifest_sha256:raise ValueError('profile source copy changed')
   if file_reference(Path(ref.directory)/'runtime/poc-runtime.zip')['sha256']!=source['copied_runtime_sha256']:raise ValueError('profile runtime changed')
   model=_call(rec,lambda:load_bundle(Path(ref.directory),backend='torch',offline=True))
   if _model_sha(model)!=source['cpu_source_sha256'] or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('profile/run CPU model differs')
  chosen=windows[:1] if mode=='profile' else windows
  return recorded_collection(model,population,chosen,config=c,mode=mode,resource=resource,source_commit=source_commit,source_files=source_files,
      configuration_sha256=configuration_sha256,parent_run_id=parent_run_id,setup_seconds=time.monotonic()-life['started'],
      retained_family=retained_family,data_provenance=data,source_runtime_copy=source,_lifecycle=life)
 except BaseException as exc:_fail_collection_attempt(life,exc);raise


def verify_upstream_acceptance(reference,*,expected_contract_sha256,verifier_reference):
 """Require the new completed-grid analysis; old failed controllers stay failed.

 This gate admits only an exact declared analysis reader/contract/result under
 its successful finite guard. It does not grant this implementation its own
 qualification, nor permission to read protected data.
 """
 import importlib.util
 _digest(expected_contract_sha256);read_reference(verifier_reference)
 accepted=json.loads(read_reference(reference))
 if accepted.get('status')!='accepted':raise ValueError('accepted new upstream analysis required')
 contract=json.loads(read_reference(accepted['contract']));result=json.loads(read_reference(accepted['result']))
 if (contract.get('schema_version')!=1 or contract.get('scientific_contract_sha256')!=expected_contract_sha256
     or result.get('status')!='accepted' or result.get('scientific_contract_sha256')!=expected_contract_sha256
     or result.get('complete_points')!=8 or result.get('execution_freeze')!=contract.get('execution_freeze')
     or result.get('verifier_source')!=verifier_reference
     or result.get('analysis',{}).get('components')!=128 or result.get('analysis',{}).get('valid_targets')!=65408):raise ValueError('upstream source/grid/reader identities differ')
 files=contract['files'];bound={v['path']:v['sha256'] for v in files}
 if bound.get(verifier_reference['path'])!=verifier_reference['sha256']:raise ValueError('upstream guard omitted exact analysis reader')
 for item in files:read_reference(item)
 reader=ROOT/'provenance/sharing_frontier_study.py';spec=importlib.util.spec_from_file_location('_calibration_upstream_guard_reader',reader)
 core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 registry=contract['execution_freeze'];command=[sys.executable,verifier_reference['path'],'--registry',registry['path'],
        '--expected-registry-sha256',registry['sha256'],'--out',accepted['result']['path']]
 core.check_stage_binding(accepted['stage'],command=command,root=Path('/private/tmp'),output=Path(accepted['result']['path']).parent,
       caps=contract['caps'],required_refs=[*files,accepted['contract']],execution_env={'PYTHONPATH':''})
 return {'status':'accepted','acceptance':reference,'analysis_contract':accepted['contract'],'result':accepted['result'],
         'verifier':verifier_reference,'scientific_contract_sha256':expected_contract_sha256,'completed_points':8,
         'scope':'accepted new RTN analysis releases serialization only; original controller failures preserved; no new model/data/quality execution or own implementation qualification'}


def execution_sources(config_path):
 """Hash actual code/config contents, including untracked study additions."""
 paths={p for group in ('poc','tests') for p in (ROOT/group).rglob('*.py')}
 paths.update((ROOT/'provenance').glob('*.py'))
 paths.update((ROOT/'provenance/shared-rtn-frontier-v1').glob('*.py'))
 paths.update((ROOT/'configs').glob('*.yaml'));paths.update((ROOT/'pyproject.toml',Path(config_path).resolve()))
 return {str(p.resolve()):hashlib.sha256(_regular(p).read_bytes()).hexdigest() for p in sorted(paths)}


def require_own_qualification(contract,*,config_path,reference):
 import importlib.util
 c=thaw(contract);q=json.loads(read_reference(reference));identity=hashlib.sha256(canonical_json(c)).hexdigest()
 fields={'schema_version','study','status','contract_sha256','source_files','source_archive','owned_controls','default_cpu','work_root','review','environment'}
 if (set(q)!=fields or q['schema_version']!=1 or type(q['schema_version'])is not int or q['study']!=c['study']
     or q['status']!='complete' or q['contract_sha256']!=identity or q['source_files']!=execution_sources(config_path)):
  raise ValueError('complete own source-bound CPU qualification required')
 work=Path(q['work_root'])
 if not work.is_absolute() or work.is_symlink() or work.resolve()!=work or work.is_relative_to(ROOT):raise ValueError('external canonical qualification scratch required')
 if q['review'].get('status')!='resolved':raise ValueError('whole-feature review must be resolved')
 review=json.loads(read_reference(q['review']['record']))
 if review.get('unresolved_critical')!=0 or review.get('unresolved_important')!=0:raise ValueError('unresolved review blocks execution')
 read_reference(q['source_archive']);read_reference(q['environment'])
 reader=ROOT/'provenance/sharing_frontier_study.py';spec=importlib.util.spec_from_file_location('_calibration_qualification_guard',reader)
 core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 refs=[{'path':p,'sha256':h} for p,h in q['source_files'].items()]+[q['source_archive'],q['environment'],q['review']['record']]
 environment={'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'}
 for name,tests,caps in (
    ('owned_controls',['tests/test_shared_calibration_study.py','tests/test_shared_calibration_bundle.py','tests/test_shared_calibration.py'],{'wall_seconds':300,'memory_bytes':4*1024**3,'disk_bytes':2*1024**3}),
    ('default_cpu',[],{'wall_seconds':1800,'memory_bytes':8*1024**3,'disk_bytes':8*1024**3})):
  output=work/name;command=[sys.executable,'-m','pytest',*tests,'-q','--basetemp='+str(output/'pytest')]
  core.check_stage_binding(q[name],command=command,root=ROOT,output=output,caps=caps,required_refs=refs,execution_env=environment)
 return q


def admit_execution_registry(path,*,config_path):
 """New own qualification and accepted upstream analysis precede any execution."""
 import math,time,importlib.util
 reference={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(_regular(path).read_bytes()).hexdigest()}
 r=json.loads(read_reference(reference));fields={'schema_version','study','config','contract_sha256','source_files','qualification',
        'upstream_acceptance','upstream_verifier','upstream_contract_sha256','out','provenance','deadline_unix'}
 if set(r)!=fields or type(r['schema_version'])is not int or r['schema_version']!=1:raise ValueError('closed explicit execution registry required')
 if r['config']['path']!=str(Path(config_path).resolve()):raise ValueError('execution configuration path differs')
 read_reference(r['config']);c=load_contract(config_path);identity=hashlib.sha256(canonical_json(thaw(c))).hexdigest()
 if (r['study']!=c['study'] or r['contract_sha256']!=identity or r['source_files']!=execution_sources(config_path)
     or r['out']!=c['out'] or Path(r['out'])!=ROOT/'artifacts'/c['study']
     or Path(r['provenance'])!=ROOT/'provenance'/c['study']
     or Path(reference['path'])!=Path(r['provenance'])/'execution-freeze.json'
     or type(r['deadline_unix'])not in (int,float) or not math.isfinite(r['deadline_unix'])
     or r['deadline_unix']!=c['resource']['deadline_unix'] or time.time()>r['deadline_unix']):raise ValueError('isolated finite exact execution contract required')
 q=require_own_qualification(c,config_path=config_path,reference=r['qualification'])
 ancestry=verify_parent_context(c)
 reader=ROOT/'provenance/sharing_frontier_study.py';spec=importlib.util.spec_from_file_location('_calibration_registry_parent',reader)
 core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 ancestor=core.load_contract(Path(c['ancestor_config']['path']))
 if core.contract_digest(ancestor)!=r['upstream_contract_sha256']:raise ValueError('upstream analysis does not refer to the qualified source campaign')
 release=verify_upstream_acceptance(r['upstream_acceptance'],expected_contract_sha256=r['upstream_contract_sha256'],verifier_reference=r['upstream_verifier'])
 refs=[{'path':p,'sha256':h} for p,h in r['source_files'].items()]+[reference,r['config'],r['qualification'],r['upstream_acceptance'],r['upstream_verifier'],release['analysis_contract'],release['result']]
 return {'contract':c,'registry':r,'registry_ref':reference,'sources':r['source_files'],'qualification':q,'ancestry':ancestry,'release':release,'refs':refs,'core':core,'config_path':Path(config_path).resolve()}


def file_reference(path):
 p=_regular(path).resolve()
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def guard_core():
 import importlib.util
 path=ROOT/'provenance/sharing_frontier_study.py';spec=importlib.util.spec_from_file_location('_calibration_finite_guard_core',path)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def worker_environment():return {'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'}


def phase_command(ctx,phase):
 return [sys.executable,str(Path(__file__).resolve()),'--config',str(ctx['config_path']),'--mode',phase,'--registry',ctx['registry_ref']['path']]


def phase_caps(ctx,phase,bound=None):
 c=ctx['contract']
 if phase=='profile':
  p=c['resource']['profile'];return {'wall_seconds':p['wall_seconds'],'memory_bytes':p['memory_bytes'],'disk_bytes':p['output_disk_bytes']}
 if phase=='run':return read_frozen_plan(ctx,bound or {})['plan']['resource']
 if phase=='analyze':return {'wall_seconds':180,'memory_bytes':c['resource']['run']['memory_bytes'],'disk_bytes':c['resource']['run']['family_disk_bytes']}
 raise ValueError('unknown finite calibration phase')


def read_frozen_plan(ctx,bound):
 path=Path(ctx['contract']['out'])/'resource-freeze.json'
 if str(path) not in bound:raise ValueError('run plan was not authenticated by parent guard')
 reference={'path':str(path),'sha256':bound[str(path)]};value=json.loads(read_reference(reference))
 fields={'schema_version','study_contract_sha256','execution_freeze','profile','profile_stage','profile_cost','plan'}
 if (set(value)!=fields or value['schema_version']!=1 or value['execution_freeze']!=ctx['registry_ref']
     or value['study_contract_sha256']!=ctx['registry']['contract_sha256']
     or value['profile']['path']!=str(Path(ctx['contract']['out'])/'profile/packet.json')
     or bound.get(value['profile']['path'])!=value['profile']['sha256']):raise ValueError('run plan/profile/execution identity differs')
 profile=json.loads(read_reference(value['profile']))
 cost_ref=value['profile_cost']
 if cost_ref['path']!=profile['terminal_cost_path'] or bound.get(cost_ref['path'])!=cost_ref['sha256']:raise ValueError('profile terminal cost was not frozen')
 cost=read_terminal_cost(cost_ref,expected_run_id=profile['run_id'])
 if bound.get(cost['ledger']['path'])!=cost['ledger']['sha256'] or cost['study_contract_sha256']!=ctx['registry']['contract_sha256']:raise ValueError('profile terminal ledger identity differs')
 if any(cost['totals'][k]!=v for k,v in profile['queries'].items()):raise ValueError('profile terminal query units differ')
 if forecast_calibration(profile,ctx['contract'])!=value['plan']:raise ValueError('completed measured resource plan differs')
 core=guard_core();core.check_stage_binding(value['profile_stage'],command=phase_command(ctx,'profile'),root=ROOT,
      output=Path(ctx['contract']['out']),caps=phase_caps(ctx,'profile'),required_refs=ctx['refs'],execution_env=worker_environment())
 return {**value,'reference':reference}


def verify_live_parent(protocol_path):
 import os,psutil
 guard=ROOT/'provenance/shared-rtn-frontier-v1/scientific-stage-supervisor-v1.py'
 try:
  parent=psutil.Process(os.getppid());actual=Path(psutil.Process().exe()).resolve();command=parent.cmdline()
  if (not parent.is_running() or Path(parent.exe()).resolve()!=actual or len(command)!=4
      or Path(command[0]).resolve()!=actual or Path(command[1]).resolve()!=guard.resolve()
      or command[2:]!=['--protocol',str(Path(protocol_path).resolve())] or parent.cwd()!=str(ROOT.resolve())
      or file_reference(guard)['sha256']!='a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba'):
   raise ValueError('live parent is not the exact finite calibration supervisor')
 except (psutil.Error,OSError) as exc:raise ValueError('cannot authenticate live calibration supervisor') from exc


def require_worker_guard(ctx,phase):
 import os
 stage=Path(ctx['registry']['provenance'])/'controller-stages'/phase;pp=stage/'protocol.json';sp=stage/'stage-start.json'
 if not pp.is_file() or not sp.is_file():raise ValueError('internal calibration worker requires its finite parent supervisor')
 protocol=json.loads(_regular(pp).read_bytes());start=json.loads(_regular(sp).read_bytes());verify_live_parent(pp)
 bound={item['path']:item['sha256'] for item in protocol['frozen_files']}
 if (start.get('pid')!=os.getppid() or start.get('protocol_sha256')!=file_reference(pp)['sha256']
     or protocol.get('command')!=phase_command(ctx,phase) or protocol.get('root')!=str(ROOT)
     or protocol.get('output')!=ctx['contract']['out'] or protocol.get('provenance')!=str(stage)
     or protocol.get('caps')!=phase_caps(ctx,phase,bound)
     or protocol.get('execution_env')!={'HF_HUB_OFFLINE':None,'PYTORCH_ENABLE_MPS_FALLBACK':None,**worker_environment()}
     or any(bound.get(item['path'])!=item['sha256'] for item in ctx['refs'])):raise ValueError('calibration worker differs from exact frozen command/source/resource guard')
 for item in protocol['frozen_files']:read_reference(item)
 return bound


def calibration_family(ctx):
 from poc.recovery_family import RetainedFamily
 c=ctx['contract'];return RetainedFamily(c['out'],[c['source_bundle']['directory'],str(ROOT/'provenance')],
      required_sources=[str(Path(c['source_bundle']['directory'])/'manifest.json')])


def analyze_collection(ctx,bound):
 import gc,time
 from dataclasses import asdict
 from poc.shared_calibration_bundle import CalibrationRef,load_calibration
 c=ctx['contract'];out=Path(c['out']);path=out/'run-acceptance.json'
 if str(path) not in bound:raise ValueError('analysis requires authenticated complete run acceptance')
 reference={'path':str(path),'sha256':bound[str(path)]};accepted=json.loads(read_reference(reference));frozen=read_frozen_plan(ctx,bound)
 if (set(accepted)!={'schema_version','execution_freeze','resource_freeze','packet','terminal_cost','stage'} or accepted['schema_version']!=1
     or accepted['execution_freeze']!=ctx['registry_ref'] or accepted['resource_freeze']!=frozen['reference']
     or accepted['packet']['path']!=str(out/'run/packet.json')
     or bound.get(accepted['packet']['path'])!=accepted['packet']['sha256']):raise ValueError('analysis run/plan/source identity differs')
 core=guard_core();core.check_stage_binding(accepted['stage'],command=phase_command(ctx,'run'),root=ROOT,output=out,
       caps=frozen['plan']['resource'],required_refs=[*ctx['refs'],frozen['reference'],frozen['profile']],execution_env=worker_environment())
 profile=json.loads(read_reference(frozen['profile']));run=json.loads(read_reference(accepted['packet']))
 if (run['mode']!='run' or run['study_contract_sha256']!=ctx['registry']['contract_sha256']
     or run['counters']!=frozen['plan']['expected_counters'] or run['queries']!=frozen['plan']['queries']
     or run['canonical_owners']!=frozen['plan']['canonical_owners'] or run['cpu_source_sha256']!=profile['cpu_source_sha256']
     or run['source_runtime_copy']!=profile['source_runtime_copy']
     or run['data_provenance']['window_plan_sha256']!=profile['data_provenance']['window_plan_sha256']):raise ValueError('complete calibration plan/counts/source differ')
 costs={}
 for name,packet,cost_ref in (('profile',profile,frozen['profile_cost']),('run',run,accepted['terminal_cost'])):
  if cost_ref['path']!=packet['terminal_cost_path'] or bound.get(cost_ref['path'])!=cost_ref['sha256']:raise ValueError('terminal cost receipt omitted from analysis freeze')
  costs[name]=read_terminal_cost(cost_ref,expected_run_id=packet['run_id'])
  if bound.get(costs[name]['ledger']['path'])!=costs[name]['ledger']['sha256']:raise ValueError('terminal ledger omitted from analysis freeze')
 import subprocess
 from poc.train import _call
 life=_start_collection_attempt(config=c,mode='analysis',resource=phase_caps(ctx,'analyze'),
      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_files=ctx['sources'],
      configuration_sha256=ctx['registry']['config']['sha256'],parent_run_id=ctx['ancestry']['source_run_id'],retained_family=calibration_family(ctx),
      declared_inputs={'profile_statistics':profile['statistics_ref'],'run_statistics':run['statistics_ref'],'resource_freeze':frozen['reference'],'run_acceptance':reference})
 rec=life['recorder']
 try:
  started=time.monotonic();verified={} 
  for name,packet in (('profile',profile),('run',run)):
   ref=CalibrationRef(**packet['statistics_ref']);loaded=_call(rec,lambda:load_calibration(ref,expected_contract_sha256=ctx['registry']['contract_sha256']))
   if loaded.digest()!=packet['result_sha256'] or loaded.counters!=packet['counters']:raise ValueError('independent persisted statistics differ')
   verified[name]={'reference':asdict(ref),'result_sha256':loaded.digest(),'counters':dict(loaded.counters),'owners':len(loaded.owners),
        'use_counts':{key:{str(k):v for k,v in owner.use_counts.items()} for key,owner in loaded.owners.items()},
        'cost':costs[name]['totals'],'terminal_cost':frozen['profile_cost'] if name=='profile' else accepted['terminal_cost'],'ledger':costs[name]['ledger'],'run_id':packet['run_id'],'data_provenance':packet['data_provenance']}
   del loaded;gc.collect()
  family=calibration_family(ctx).disk()
  if family>c['resource']['run']['family_disk_bytes']:raise ValueError('complete retained calibration family exceeds cap')
  result={'schema_version':1,'status':'accepted','study':c['study'],'study_contract_sha256':ctx['registry']['contract_sha256'],
        'execution_freeze':ctx['registry_ref'],'source':dict(c['source_bundle']),'upstream':ctx['release'],
        'resource_freeze':frozen['reference'],'run_acceptance':reference,'statistics':verified,'retained_family_bytes':family,
        'independent_reload_seconds':time.monotonic()-started,'neural_calls_in_analysis':0,'analysis_cost_path':str(life['phase']/'terminal-cost.json'),
        'scope':'CPU current-student input statistics from all shared logical uses; no fitted quantizer/model, fidelity result, capability certification or compression claim; source data already exposed fit development, not confirmation'}
  rec.write(out/'analysis.json',result)
  report=['# All-use shared calibration statistics','','Completed the fixed CPU calibration plan and independently reloaded its saved FP64 input statistics. This is calibration data, not a compressed model or a quality result.','',
          f"Run contexts: {run['counters']['unique_contexts']}; raw targets: {run['counters']['raw_target_presentations']}; operator inputs: {run['counters']['operator_input_presentations']}; projection inputs: {run['counters']['projection_input_presentations']}; canonical owners: {run['canonical_owners']}.",
          '', 'Profile and run each retain source/runtime identities, data-access journals, exact windows, common RunSpec/cost ledgers and failed-attempt records. Profile reads the declared fit subset but forwards one context; the run reopens the same subset and forwards the full finite plan. Teacher queries, readout loss and optimizer updates are zero.','',result['scope'],'']
  def write_report():
   with (out/'REPORT.md').open('x') as f:f.write('\n'.join(report))
  _call(rec,write_report);_close_collection_attempt(life,'independent statistics and terminal-ledger analysis complete')
  return result
 except BaseException as exc:_fail_collection_attempt(life,exc);raise


def run_campaign(ctx):
 import time,os
 from datetime import datetime,timezone
 prov=Path(ctx['registry']['provenance']);out=Path(ctx['contract']['out'])
 if any((prov/name).exists() for name in ('campaign-start.json','campaign-result.json','campaign-failed.json')):raise FileExistsError('calibration campaign already attempted; no automatic retry')
 core=guard_core();begun=time.monotonic();stages={};refs=list(ctx['refs'])
 core.write_exclusive(prov/'campaign-start.json',{'pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat(),'execution_freeze':ctx['registry_ref']})
 try:
  for phase in ('profile','run','analyze'):
   if time.time()>ctx['registry']['deadline_unix']:raise ValueError('calibration campaign deadline expired')
   for item in refs:read_reference(item)
   bound={x['path']:x['sha256'] for x in refs};caps=phase_caps(ctx,phase,bound)
   stage=core.execute_stage(phase_command(ctx,phase),root=ROOT,output=out,stage=prov/'controller-stages'/phase,
            caps=caps,frozen_files=refs,execution_env=worker_environment());stages[phase]=stage
   core.check_stage_binding(stage,command=phase_command(ctx,phase),root=ROOT,output=out,caps=caps,required_refs=refs,execution_env=worker_environment())
   if phase=='profile':
    profile_ref=file_reference(out/'profile/packet.json');profile=json.loads(read_reference(profile_ref));plan=forecast_calibration(profile,ctx['contract'])
    profile_cost=file_reference(out/'profile/terminal-cost.json');terminal=read_terminal_cost(profile_cost,expected_run_id=profile['run_id'])
    core.write_exclusive(out/'resource-freeze.json',{'schema_version':1,'study_contract_sha256':ctx['registry']['contract_sha256'],
           'execution_freeze':ctx['registry_ref'],'profile':profile_ref,'profile_stage':stage,'profile_cost':profile_cost,'plan':plan})
    refs += [file_reference(out/'resource-freeze.json'),profile_ref,profile_cost,terminal['ledger']]
   elif phase=='run':
    packet=file_reference(out/'run/packet.json');packet_value=json.loads(read_reference(packet));run_cost=file_reference(out/'run/terminal-cost.json');terminal=read_terminal_cost(run_cost,expected_run_id=packet_value['run_id']);core.write_exclusive(out/'run-acceptance.json',{'schema_version':1,
       'execution_freeze':ctx['registry_ref'],'resource_freeze':file_reference(out/'resource-freeze.json'),'packet':packet,'terminal_cost':run_cost,'stage':stage})
    refs += [file_reference(out/'run-acceptance.json'),packet,run_cost,terminal['ledger']]
  result=file_reference(out/'analysis.json')
  core.write_exclusive(prov/'campaign-result.json',{'status':'complete','execution_freeze':ctx['registry_ref'],'analysis':result,'analysis_terminal_cost':file_reference(out/'analysis/terminal-cost.json'),'stages':stages,
        'wall_seconds':time.monotonic()-begun,'scope':'completed bounded calibration statistics only; no model compression/capability claim'})
  return {'status':'complete','result':result}
 except BaseException as exc:
  core.write_exclusive(prov/'campaign-failed.json',{'status':'failed','execution_freeze':ctx['registry_ref'],'error':repr(exc),'stages':stages,
        'wall_seconds':time.monotonic()-begun,'scope':'failed work/data exposures retained; no accepted campaign or automatic retry'})
  raise


def run_owned_controls(config_path):
 import uuid
 c=load_contract(config_path);verify_parent_context(c);inspect_fit_metadata(c)
 core=guard_core();work=Path('/private/tmp')/('cgoc-calibration-controls-'+uuid.uuid4().hex);output=work/'owned_controls';output.mkdir(parents=True,exist_ok=False)
 sources=execution_sources(config_path);refs=[{'path':p,'sha256':h} for p,h in sources.items()]
 command=[sys.executable,'-m','pytest','tests/test_shared_calibration_study.py','tests/test_shared_calibration_bundle.py','tests/test_shared_calibration.py','-q','--basetemp='+str(output/'pytest')]
 stage=core.execute_stage(command,root=ROOT,output=output,stage=work/'stage',caps={'wall_seconds':300,'memory_bytes':4*1024**3,'disk_bytes':2*1024**3},frozen_files=refs,execution_env=worker_environment())
 core.require_stage_success(stage)
 value={'mode':'controls','status':'complete','stage':stage,'study_contract_sha256':hashlib.sha256(canonical_json(thaw(c))).hexdigest(),
        'scope':'owned implementation controls only; full own CPU qualification and resolved review remain required for real calibration'}
 core.write_exclusive(work/'result.json',value);return {'mode':'controls','status':'complete','result':file_reference(work/'result.json')}


def main(argv=None):
 import argparse,subprocess
 parser=argparse.ArgumentParser(description='All-use CPU calibration with frozen source, qualification and resource gates')
 parser.add_argument('--config',type=Path,required=True)
 parser.add_argument('--mode',choices=('validate','controls','profile','run','analyze','campaign'),required=True)
 parser.add_argument('--registry',type=Path)
 args=parser.parse_args(argv)
 try:
  if args.mode=='controls':summary=run_owned_controls(args.config)
  elif args.mode=='validate':
   c=load_contract(args.config);fit=inspect_fit_metadata(c);ancestry=verify_parent_context(c);source=inspect_source_bundle(c)
   summary={'mode':'validate','status':'metadata_validated','study':c['study'],
           'study_contract_sha256':hashlib.sha256(canonical_json(thaw(c))).hexdigest(),
           'parent_source_count':ancestry['parent_source_count'],'source_complete_bytes':source['complete_bytes'],
           'source_components':len(fit['parent'].records),'selected_components':len(fit['selected']),
           'planned_raw_targets':len(fit['selected'])*511,'model_calls':0,'corpus_materialized':False,
           'execution_ready':False,'remaining':'own qualified source registry and accepted upstream release before actual CPU calibration'}
   if args.registry is not None:admit_execution_registry(args.registry,config_path=args.config);summary['execution_ready']=True;summary['remaining']='finite guarded campaign dispatch'
  else:
   if args.registry is None:raise ValueError('real calibration requires its exact own qualified execution registry')
   ctx=admit_execution_registry(args.registry,config_path=args.config)
   if args.mode=='campaign':summary=run_campaign(ctx)
   else:
    bound=require_worker_guard(ctx,args.mode)
    if args.mode=='analyze':value=analyze_collection(ctx,bound)
    else:
     import torch
     torch.set_num_threads(1);plan=read_frozen_plan(ctx,bound) if args.mode=='run' else None
     value=cold_cpu_collection(ctx['contract'],mode=args.mode,resource=phase_caps(ctx,args.mode,bound),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_files=ctx['sources'],
        configuration_sha256=ctx['registry']['config']['sha256'],parent_run_id=ctx['ancestry']['source_run_id'],
        profile_reference=plan['profile'] if plan else None,retained_family=calibration_family(ctx))
    summary={'mode':args.mode,'status':'complete','study_contract_sha256':ctx['registry']['contract_sha256'],
             'execution_freeze':ctx['registry_ref'],'scope':'calibration input statistics only; no model quality or compression result'}
  print(json.dumps(summary,sort_keys=True));return 0
 except Exception as exc:
  print(json.dumps({'status':'failed','error':type(exc).__name__+': '+str(exc)},sort_keys=True),file=sys.stderr);return 2


# CLI_ENTRY
if __name__=='__main__':
 raise SystemExit(main())

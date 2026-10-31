"""Closed learned-frontier contracts and fixed-population analysis controls."""
import importlib.util,sys,json,copy
from pathlib import Path
import pytest


def api():
 path=Path(__file__).parents[1]/'provenance/shared_gptq_study.py'
 assert path.is_file(),'bounded shared GPTQ study driver missing'
 spec=importlib.util.spec_from_file_location('owned_shared_gptq_study',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def config(m,tmp_path):
 ref={'path':str(tmp_path/'input.json'),'sha256':'a'*64};bundle={'directory':str(tmp_path/'bundle'),'manifest_sha256':'b'*64}
 return {'schema_version':1,'study':'owned-gptq-v1','out':str(tmp_path/'out'),'teacher':dict(bundle),'shared':dict(bundle),
  'inputs':{**ref,'population_manifest_sha256':'c'*64},'worker':dict(ref),'calibration':{'packet':dict(ref),'acceptance':dict(ref),'fit_subset':dict(ref),'statistics':dict(bundle),'contract_sha256':'d'*64},
  'reference':{'teacher_mps_digest':'e'*64,'source_run_id':'f'*64,'scientific_observation':dict(ref)},'upstream_rtn_acceptance':dict(ref),
  'grid':m.fixed_grid(),'resource':{'preparation_wall_seconds':900,'phase_memory_bytes':8*1024**3,'phase_disk_bytes':2*1024**3,'profile_wall_seconds':300,'evaluation_wall_ceiling':1800,'campaign_wall_seconds':12000,'family_disk_bytes':64*1024**3,'deadline_unix':9999999999}}


def windows():
 return [{'document_id':str(i),'component_id':str(i),'tokens':[1]*512,'positions':list(range(512)),'attention_mask':[True]*512,'target_mask':[True]*511,'window_sha256':str(i)} for i in range(128)]


def test_closed_contract_and_full_precisions_grid(tmp_path):
 m=api();c=config(m,tmp_path);path=tmp_path/'config.json';path.write_text(json.dumps(c));assert m.load_contract(path)==c
 assert len(c['grid'])==12 and c['grid'][0]['label']=='teacher_noop'
 assert next(r for r in c['grid'] if r['label']=='dense_rtn2_embedding4')['embedding_bits']==4


@pytest.mark.parametrize('bad',['unknown','bool','nan','relative','grid','schema','negative'])
def test_bad_contract_fails_without_outputs(tmp_path,bad):
 m=api();c=config(m,tmp_path)
 if bad=='unknown':c['shortcut']=True
 elif bad=='bool':c['resource']['phase_memory_bytes']=True
 elif bad=='nan':c['resource']['preparation_wall_seconds']=float('nan')
 elif bad=='relative':c['out']='relative'
 elif bad=='grid':c['grid'].pop()
 elif bad=='schema':c['schema_version']=True
 else:c['resource']['campaign_wall_seconds']=-1
 path=tmp_path/'config.json';path.write_text(json.dumps(c))
 with pytest.raises(ValueError):m.load_contract(path)
 assert not (tmp_path/'out').exists()


def prepared(c):
 return {'study_contract_sha256':api().digest(c),'inputs':c['inputs'],'teacher_mps_digest':c['reference']['teacher_mps_digest'],
  'blocked_source_roots':[],'artifacts':{row['label']:{'directory':'/private/tmp/'+row['label'],'manifest_sha256':('1' if row['label']=='teacher_noop' else '2')*64,'runtime_sha256':'3'*64,'complete_bytes':100,'teacher_needed_at_inference':False} for row in c['grid']}}


def test_generic_protocol_preserves_counts_runtime_and_frozen_budget(tmp_path):
 m=api();c=config(m,tmp_path);p=prepared(c);w=windows();work=m.core.metric_work_plan(w,mode='profile');caps={'wall_seconds':300,'memory_bytes':8*1024**3,'output_disk_bytes':2*1024**3,**{k:v for k,v in work.items() if k!='components'}}
 protocol=m.native_protocol(c,p,'shared_gptq4',w,mode='profile',resource=caps,output=tmp_path/'result',ledger=tmp_path/'ledger')
 assert protocol['resource']['valid_target_presentations']==511 and protocol['resource']['student_forward_calls']==17
 p['artifacts']['shared_gptq4']['runtime_sha256']='4'*64
 with pytest.raises(ValueError):m.native_protocol(c,p,'shared_gptq4',w,mode='profile',resource=caps,output=tmp_path/'result',ledger=tmp_path/'ledger')


def test_wrong_grid_and_query_counts_rejected(tmp_path):
 m=api();c=config(m,tmp_path);p=prepared(c);w=windows();caps={'wall_seconds':300,'memory_bytes':8*1024**3,'output_disk_bytes':2*1024**3,'valid_target_presentations':510,'teacher_forward_calls':17,'student_forward_calls':17}
 with pytest.raises(ValueError):m.native_protocol(c,p,'shared_gptq4',w,mode='profile',resource=caps,output=tmp_path/'result',ledger=tmp_path/'ledger')
 p['artifacts'].pop('dense_gptq4')
 with pytest.raises(ValueError):m.native_protocol(c,p,'shared_gptq4',w,mode='profile',resource=caps,output=tmp_path/'result',ledger=tmp_path/'ledger')


def test_summary_preserves_adverse_points_and_component_uncertainty(tmp_path):
 m=api();c=config(m,tmp_path);p=prepared(c);w=windows();protocols={};results={}
 for j,g in enumerate(c['grid']):
  label=g['label'];p['artifacts'][label]['complete_bytes']=100+j;kl=0 if label=='teacher_noop' else 1 if label=='dense_gptq4' else 2
  protocols[label]={'label':label};results[label]={'complete_bytes':100+j,'rows':[{'document_id':r['document_id'],'component_id':r['component_id'],'window_sha256':r['window_sha256'],'valid_targets':511,'kl_sum':kl*511,'nll_sum':(3+kl)*511,'teacher_nll_sum':3*511,'disagreements':0} for r in w]}
 out=m.summarize(p,protocols,results,w,check=lambda *a,**k:None)
 assert len(out['points'])==12 and out['primary']['kl_difference']==1 and out['primary']['worse_components']==128
 assert out['primary']['paired_component_percentile_95']==[1,1]
 assert out['scope']['capability_acceptability_established'] is False
 assert out['secondary']['kl_difference']==0


def test_direct_phase_requires_live_guard_before_neural_setup(tmp_path):
 m=api();c=config(m,tmp_path)
 with pytest.raises(ValueError,match='guard'):m.require_phase_guard(c,'prepare-shared_gptq4',[],None)
 assert not Path(c['out']).exists()


def test_failed_setup_has_durable_common_attempt_and_cost(tmp_path):
 m=api();c=config(m,tmp_path);source=tmp_path/'owned.py';source.write_text('owned')
 from poc.contracts import CostRecord
 def fail(rec):
  rec.record(CostRecord(processed_inputs=3));raise InterruptedError('owned setup interruption')
 with pytest.raises(InterruptedError):m.record_phase(c,'owned-failed',{str(source):m.sha(source)},fail)
 phase=Path(c['out'])/'phases/owned-failed'
 assert (phase/'failed.json').is_file() and not (phase/'terminal-cost.json').exists()
 costs=list((phase/'records').rglob('costs.jsonl'));assert len(costs)==1
 assert sum(json.loads(line)['cost']['processed_inputs'] for line in costs[0].read_text().splitlines())==3
 with pytest.raises(FileExistsError):m.record_phase(c,'owned-failed',{str(source):m.sha(source)},fail)


def test_scientific_window_bridge_requires_exact_tokens_masks_and_ids():
 m=api();w=windows();rows=[]
 for x in w:
  x['window_sha256']=m.digest({k:v for k,v in x.items() if k!='window_sha256'})
  rows.append({'document_id':x['document_id'],'component_id':x['component_id'],'window_sha256':m.digest({'tokens':x['tokens'],'positions':x['positions']}),'target_positions':list(range(511)),'full':{'status':'ok','valid_targets':511,'kl':0,'nll':0}})
 scientific={'step':256,'populations':{'tuning':rows}}
 assert len(m.scientific_bridge(scientific,w))==128
 w[0]['tokens'][0]=2
 with pytest.raises(ValueError):m.scientific_bridge(scientific,w)


def test_dense_population_reopens_new_journal_and_matches_tensor_window_hash(tmp_path):
 m=api();c=config(m,tmp_path)
 from dataclasses import asdict
 from poc.data.manifest import DocumentRecord,DataManifest
 from poc.teacher_validation import _batch
 from poc.shared_calibration import window_plan_sha256
 from tokenizers import Tokenizer,models,pre_tokenizers
 source=tmp_path/'floating';(source/'tokenizer').mkdir(parents=True)
 tok=Tokenizer(models.WordLevel({'[UNK]':0,'t1':1},unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(source/'tokenizer/tokenizer.json'))
 records=tuple(DocumentRecord(str(i),'owned','r','s',i,text=' '.join(['t1']*512)) for i in range(16));pop=DataManifest(records,{r.document_id:'fit' for r in records},{r.document_id:r.document_id for r in records},'owned',str(tmp_path/'old-journal.jsonl'))
 manifest=tmp_path/'fit.json';manifest.write_bytes(m.canonical_json(asdict(pop)));c['calibration']['fit_subset']=m.ref(manifest)
 rows=[]
 for r in records:
  b=_batch([1]*512,device='cpu',document_id=r.document_id,cluster_id=r.document_id,targets=range(1,512));rows.append({'document_id':r.document_id,'window_sha256':window_plan_sha256((b,))})
 packet=tmp_path/'packet.json';packet.write_bytes(m.canonical_json({'data_provenance':{'records':rows}}));c['calibration']['packet']=m.ref(packet)
 result,proof=m.dense_population(c,'owned',m.BundleRef(str(source),'a'*64))
 assert proof['raw_target_inventory']==8176 and len(proof['windows'])==16
 assert result.exposure_path!=pop.exposure_path and not Path(pop.exposure_path).exists()
 assert Path(result.exposure_path).is_file()


@pytest.mark.parametrize('mode',['profile','run'])
def test_public_profile_and_run_modes_dispatch_implemented_stages(tmp_path,monkeypatch,capsys,mode):
 m=api();c=config(m,tmp_path);path=tmp_path/'config.json';path.write_text(json.dumps(c));calls=[]
 def profile(c,path,**kw):calls.append(('profile',kw));return {'status':'profiled'}
 def run(c,path):calls.append(('run',{}));return {'status':'complete','primary':{},'analysis':{}}
 monkeypatch.setattr(m,'campaign',profile);monkeypatch.setattr(m,'run_from_freeze',run,raising=False)
 m.main(['--config',str(path),'--mode',mode])
 assert calls==([('profile',{'profile_only':True})] if mode=='profile' else [('run',{})])


def test_review_qualification_rejects_self_asserted_record(tmp_path,monkeypatch):
 m=api();c=config(m,tmp_path);source=tmp_path/'owned.py';source.write_text('owned');sources={str(source):m.sha(source)};path=tmp_path/'config.json';path.write_text(json.dumps(c))
 monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_files',lambda p:sources)
 q=tmp_path/'provenance'/c['study']/'qualification.json';q.parent.mkdir(parents=True);q.write_bytes(m.canonical_json({'status':'qualified','sources':sources,'config':m.ref(path),'contract_sha256':m.digest(c)}))
 with pytest.raises(ValueError,match='qualification'):m.execution_refs(c,path)


def test_review_preparation_rejects_unrelated_successful_guard(tmp_path,monkeypatch):
 m=api();c=config(m,tmp_path);source=tmp_path/'owned.py';source.write_text('owned');label='teacher_noop'
 m.record_phase(c,'prepare-'+label,{str(source):m.sha(source)},lambda rec:{'directory':str(Path(c['out'])/'artifacts'/label),'recipe':m.fixed_grid()[0]})
 stage=m.core.execute_stage([sys.executable,'-c','pass'],root=tmp_path,output=tmp_path/'unrelated',stage=tmp_path/'unrelated-stage',caps={'wall_seconds':30,'memory_bytes':1024**3,'disk_bytes':1024**2},frozen_files=[m.ref(source)])
 d=Path(c['out'])/'phases'/('prepare-'+label);m.write(d/'acceptance.json',{'packet':m.ref(d/'packet.json'),'terminal_cost':m.ref(d/'terminal-cost.json'),'stage':stage})
 monkeypatch.setattr(m,'execution_refs',lambda *a:[])
 with pytest.raises(ValueError):m.accepted_phase(c,label)


def test_review_calibration_binding_rejects_different_admitted_packet(tmp_path):
 m=api();c=config(m,tmp_path);packet={'statistics_ref':c['calibration']['statistics'],'study_contract_sha256':c['calibration']['contract_sha256']}
 result={'status':'accepted','results':{'run':{'packet':{'path':'/private/tmp/other','sha256':'a'*64},'statistics':c['calibration']['statistics']}}}
 assert hasattr(m,'check_calibration_binding'),'upstream calibration admission binding missing'
 with pytest.raises(ValueError):m.check_calibration_binding(c,packet,result)


def test_review_statistics_payload_is_verified_before_dispatch(tmp_path):
 m=api();c=config(m,tmp_path);d=tmp_path/'statistics';d.mkdir();payload=d/'statistics.safetensors';payload.write_bytes(b'original');metadata={'schema_version':1,'study_contract_sha256':c['calibration']['contract_sha256'],'files':{'statistics.safetensors':{'bytes':8,'sha256':m.sha(payload)}},'result_sha256':'a'*64};m.write(d/'manifest.json',metadata);c['calibration']['statistics']={'directory':str(d),'manifest_sha256':m.sha(d/'manifest.json')};payload.write_bytes(b'altered!')
 assert hasattr(m,'statistics_refs'),'statistics payload preflight missing'
 with pytest.raises(ValueError):m.statistics_refs(c,verify=True)


def test_review_public_analysis_requires_supervised_dispatch(tmp_path,monkeypatch):
 m=api();c=config(m,tmp_path);path=tmp_path/'config.json';path.write_text(json.dumps(c));calls=[]
 monkeypatch.setattr(m,'analyze',lambda *a:pytest.fail('public analysis bypassed finite supervisor'))
 monkeypatch.setattr(m,'dispatch_analysis',lambda *a:{'status':'accepted','guarded':True},raising=False)
 m.main(['--config',str(path),'--mode','analyze'])


def test_review_native_attempt_precedes_case_setup(tmp_path,monkeypatch):
 m=api();c=config(m,tmp_path);p=prepared(c)
 class IntentStarted(Exception):pass
 def durable(*a,**kw):raise IntentStarted()
 monkeypatch.setattr(m,'record_phase',durable)
 with pytest.raises(IntentStarted):m.native_job(c,p,'shared_gptq4',windows(),mode='profile',resource={},refs=[])
 assert not (Path(c['out'])/'workers').exists()


def test_review_worker_cost_reconciliation_counts_completed_ledger_prefix(tmp_path):
 m=api();p={'study':'owned','study_contract_sha256':'a'*64};counts={'valid_target_presentations':0,'teacher_forward_calls':0,'student_forward_calls':0};rows=[]
 for work in ({'valid_target_presentations':511},{'teacher_forward_calls':1},{'student_forward_calls':1}):
  for k,v in work.items():counts[k]+=v
  rows.append({'study':'owned','study_contract_sha256':'a'*64,'protocol_sha256':m.digest(p),'window_sha256':'b'*64,'raw_corpus_target_presentations':work.get('valid_target_presentations',0),'elapsed_seconds':.1,'work':work,'cumulative':dict(counts)})
 path=tmp_path/'costs.jsonl';path.write_bytes(b''.join(m.canonical_json(r) for r in rows)+b'{incomplete')
 class Rec:
  def __init__(self):self.rows=[]
  def record(self,c):self.rows.append(c)
 rec=Rec();assert hasattr(m,'reconcile_native_ledger'),'native common cost reconciliation missing'
 with pytest.raises(ValueError):m.reconcile_native_ledger(rec,p,path)
 assert sum(r.teacher_forward_calls for r in rec.rows)==1 and sum(r.student_forward_calls for r in rec.rows)==1
 assert sum(r.raw_corpus_target_presentations for r in rec.rows)==511


def test_review_campaign_envelope_rejects_insufficient_time_and_monitors_family(tmp_path,monkeypatch):
 m=api();c=config(m,tmp_path);out=Path(c['out']);out.mkdir();m.write(out/'campaign-intent.json',{'started_unix':100.,'contract_sha256':m.digest(c),'campaign_wall_seconds':c['resource']['campaign_wall_seconds']});monkeypatch.setattr(m.time,'time',lambda:101.)
 assert hasattr(m,'campaign_envelope'),'campaign-wide stage envelope missing'
 caps=m.campaign_envelope(c,300);assert caps['disk_bytes']==c['resource']['family_disk_bytes']
 monkeypatch.setattr(m.time,'time',lambda:100.+c['resource']['campaign_wall_seconds']-1)
 with pytest.raises(ValueError):m.campaign_envelope(c,300)


def test_offline_proof_loader_works_in_fresh_cli_without_pytest_path(tmp_path):
 m=api();import subprocess
 script="import importlib.util,sys;from pathlib import Path;root=Path("+repr(str(Path(m.__file__).parents[1]))+");s=importlib.util.spec_from_file_location('fresh_cli',root/'provenance/shared_gptq_study.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);sys.path=[p for p in sys.path if str(Path(p).resolve())!=str(root/'tests')];assert callable(m.offline_proof_module().run_offline_bundle);print('fresh_cli_import_passed')"
 result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 assert 'fresh_cli_import_passed' in result.stdout


from test_shared_quantization import case


@pytest.mark.mlx
def test_owned_artifact_recipes_finish_common_loop_and_offline_reload(case,tmp_path,monkeypatch):
 """Exercise real owned setup/packing/proof/persistence, beyond leaf helpers."""
 m=api();c=config(m,tmp_path)
 import torch
 from dataclasses import asdict
 from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
 from poc.bundle import write_bundle,load_bundle,verify_bundle
 from poc.contracts import thaw
 from poc.data.manifest import DataManifest,DocumentRecord
 from poc.teacher_validation import _batch
 from poc.shared_calibration import SharedCalibrationSpec,collect_shared_inputs,window_plan_sha256,_model_sha
 from poc.shared_calibration_bundle import save_calibration
 from test_export import runtime
 dense=make_tiny_decoder(TinyConfig(vocab_size=41,hidden_size=64,intermediate_size=128,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024),seed=3,dtype=torch.float32);dense.tokenizer_digest='a'*64
 teacher=write_bundle(dense,tmp_path/'teacher',runtime(tmp_path));c['teacher']={'directory':teacher.directory,'manifest_sha256':teacher.manifest_sha256};c['shared']={'directory':case.source.directory,'manifest_sha256':case.source.manifest_sha256}
 records=tuple(DocumentRecord(str(i),'owned','r','s',i,text=' '.join(['t1','t4','t7','t2']*128)) for i in range(16));pop=DataManifest(records,{r.document_id:'fit' for r in records},{r.document_id:r.document_id for r in records},'owned',str(tmp_path/'original-exposure.jsonl'))
 windows=tuple(_batch([1,4,7,2]*128,device='cpu',document_id=r.document_id,cluster_id=r.document_id,targets=range(1,512)) for r in records)
 source=load_bundle(Path(case.source.directory),backend='torch');model_sha=_model_sha(source);spec=SharedCalibrationSpec(layout_sha256=model_sha,source_sha256=model_sha,population_sha256=pop.digest(),window_plan_sha256=window_plan_sha256(windows),tokenizer_sha256='a'*64)
 stats=collect_shared_inputs(source,source,pop,windows,spec);stats_ref=save_calibration(stats,tmp_path/'all-use-stats',expected_result_sha256=stats.digest(),study_contract_sha256='c'*64)
 fit=tmp_path/'fit.json';fit.write_bytes(m.canonical_json(asdict(pop)));packet=tmp_path/'packet.json';packet.write_bytes(m.canonical_json({'data_provenance':{'records':[{'document_id':r.document_id,'window_sha256':window_plan_sha256((w,))} for r,w in zip(records,windows,strict=True)]}}))
 c['calibration'].update(fit_subset=m.ref(fit),packet=m.ref(packet),statistics={'directory':stats_ref.directory,'manifest_sha256':stats_ref.manifest_sha256},contract_sha256='c'*64)
 values={};monkeypatch.setattr(m,'accepted_phase',lambda c,label:values[label]);sources={str(Path(m.__file__)):m.sha(m.__file__)}
 # Receipt admission has independent negative tests; these known owned outputs
 # supply dependencies so the actual model/packing/proof loop can be exercised.
 for label in ('teacher_noop','shared_float','dense_rtn4','shared_rtn4','dense_gptq4','shared_gptq4'):
  values[label]=m.prepare_artifact(c,label,sources)
  directory=Path(c['out'])/'phases'/('prepare-'+label);packet=m.read(m.ref(directory/'packet.json'));cost=m.calcore.read_terminal_cost(m.ref(directory/'terminal-cost.json'),expected_run_id=packet['run_id'])
  assert values[label]['offline_proof']['download_denied']
  assert cost['totals']['student_forward_calls']==(10 if label=='dense_gptq4' else 8)
  assert cost['totals']['teacher_forward_calls']==(18 if label=='dense_gptq4' else 0)
 assert len({v['runtime_sha256'] for v in values.values()})==1
 assert all(v['complete_bytes']>0 and v['teacher_needed_at_inference']is False for v in values.values())


@pytest.mark.mlx
def test_fresh_process_runs_owned_teacher_artifact_proof_without_pytest_path(tmp_path):
 m=api();c=config(m,tmp_path)
 import torch,subprocess
 from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
 from poc.bundle import write_bundle
 from test_export import runtime
 model=make_tiny_decoder(TinyConfig(vocab_size=41,hidden_size=64,intermediate_size=128,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024),seed=3,dtype=torch.float32);model.tokenizer_digest='a'*64
 teacher=write_bundle(model,tmp_path/'teacher',runtime(tmp_path));c['teacher']={'directory':teacher.directory,'manifest_sha256':teacher.manifest_sha256};path=tmp_path/'contract.json';path.write_text(json.dumps(c))
 root=Path(m.__file__).parents[1]
 script="import importlib.util,sys,json;from pathlib import Path;root=Path("+repr(str(root))+");s=importlib.util.spec_from_file_location('fresh_artifact_cli',root/'provenance/shared_gptq_study.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);sys.path=[p for p in sys.path if str(Path(p).resolve())!=str(root/'tests')];c=json.loads(Path("+repr(str(path))+").read_text());v=m.prepare_artifact(c,'teacher_noop',{str(Path(m.__file__)):m.sha(m.__file__)});assert v['offline_proof']['download_denied'];print('fresh_artifact_proof_passed')"
 result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 assert 'fresh_artifact_proof_passed' in result.stdout

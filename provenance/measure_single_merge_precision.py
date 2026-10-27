"""One fixed fresh-component repeat, with strictly charged exact-state replay."""
from dataclasses import asdict,replace
import argparse,gc,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import psutil,torch
from poc.bank import make_bank,canonicalize_partition
from poc.budget import BudgetExceeded,BudgetTracker
from poc.config import load_config
from poc.contracts import Budget,CostRecord,ProbeRecord,RunSpec
from poc.data.manifest import load_manifest
from poc.merge import initialize_merge,fit_trial,restore_trial_delta,save_trial_delta
from poc.probes import PathologyObservation,PathologyThresholds,ProbePolicy,generate_donors
from poc.records import AttemptRecord,append_event,canonical_json,start_attempt
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,load_tokenizer,sha256_file
from poc.teacher_validation import _batch,_restricted,_write
from poc.train import TrainingConfig
from poc.diagnostics import TrialIdentity,TrialRecord,DamageTable,score_trials,selected_merge_regret,bootstrap_regret,fixed_scores
from provenance.measure_training_pilot import TrainingPilotRecorder
from provenance.measure_single_merge import (_documents,_canonical,_accepted,_mean,training_sources,natural_metrics,activation_scores,probe_observations)

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'artifacts/diagnostics/task12'
OUT=ORIGINAL/'precision-v1'


def require_replay_identity(expected,initialized,fitted=None):
    if initialized!=expected['initialized_digest'] or (fitted is not None and fitted!=expected['fitted_digest']):
        raise ValueError('exact original initialized/fitted digest required; stop without retry or substitution')


def enforce_phase_caps(total,caps):
    for name,limit in caps.items():
        if total.get(name,0)>limit:raise BudgetExceeded('precision phase ceiling: '+name)


def logical_position_count(*,score_documents,damage_documents,depths,donor_target_counts):
    # KL/NLL share one logical position; activation MSE/CKA share one observation.
    return (score_documents+damage_documents+2*score_documents*depths)*511+sum(donor_target_counts)+len(donor_target_counts)


class PrecisionRecorder(TrainingPilotRecorder):
    def __init__(self,tracker,p):
        self.plan=p;self._pinned={};self.recovery=tracker.attempt.spec.resolved_config.get('recovery_authorization')
        super().__init__(tracker,p['overall_student_forward_calls'])
        self.followup_wall_at_start=self.phase_totals().wall_seconds
    def memory(self,label=None):
        values={'rss_bytes':psutil.Process().memory_info().rss,'mps_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()}
        for key,value in values.items():self.memory_highwater[key]=max(self.memory_highwater[key],value)
        combined=values['rss_bytes']+values['mps_driver_bytes'];now=time.monotonic()
        if self.phase_key.startswith('task12_precision_evaluate') and (label or now-getattr(self,'_last_memory_sample',0)>=5):
            with (self.tracker.attempt.path/'memory-samples.jsonl').open('a') as stream:stream.write(json.dumps(values|{'overlapping_rss_plus_driver':combined,'label':label,'elapsed_seconds':now-getattr(self,'started',now)})+'\n')
            self._last_memory_sample=now
        return combined
    def _pin(self,path,digest):
        stat=path.stat();stamp=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if self._pinned.get(str(path))!=stamp:
            if sha256_file(path)!=digest:raise ValueError('immutable parent evidence changed: '+str(path))
            self._pinned[str(path)]=stamp
    def _total_paths(self,paths):
        from poc.checkpoint import _spec
        total=CostRecord()
        for path in paths:
            key=str(path.parent)
            if key not in self._phase_trackers:
                raw=json.loads(path.read_text());identity=_spec(raw)
                self._phase_trackers[key]=self.tracker if key==str(self.tracker.path.parent) else BudgetTracker(identity.budgets,AttemptRecord(path.parent.name,'aggregate_only',path.parent/'attempts'/'aggregate_only',identity))
            total=BudgetTracker.combine(total,self._phase_trackers[key].totals)
        return total
    def phase_totals(self):
        total=self._total_paths(sorted((self.phase_root/'runs').glob('*/spec.json')))
        if self.recovery:total=BudgetTracker.combine(total,CostRecord(wall_seconds=self.recovery['diagnosis_wall_seconds']))
        return replace(total,memory_unavailable_reason=None) if total.memory_bytes is not None else total
    def aggregate(self):
        parent_paths=[]
        for item in self.plan['parent_runs']:
            path=ROOT/item['spec_path'];self._pin(path,item['spec_sha256']);self._pin(path.parent/'costs.jsonl',item['cost_sha256']);parent_paths.append(path)
        total=BudgetTracker.combine(self._total_paths(parent_paths),self.phase_totals())
        total=BudgetTracker.combine(total,CostRecord(wall_seconds=self.carry['audit_wall_seconds']))
        if total.memory_bytes is not None:total=replace(total,memory_unavailable_reason=None)
        return total
    def check_aggregate(self,cost=None):
        cost=cost or CostRecord();total=self.aggregate();phase=self.phase_totals()
        live=time.monotonic()-self.started
        if self.phase_key=='task12_precision_prepare_resume' and live+cost.wall_seconds>300:raise BudgetExceeded('isolated preparation300s ceiling')
        if self.phase_key.startswith('task12_precision_evaluate') and (self.tracker.totals.memory_bytes or 0)>int(8.5*1024**3):raise BudgetExceeded('evaluation approaching9GiB guard; stop at8.5GiB observed')
        if max(total.wall_seconds,self.phase_wall_at_start+live)+cost.wall_seconds>self.tracker.budget.wall_seconds:raise BudgetExceeded('Task12 overall live wall ceiling')
        phase_wall=max(phase.wall_seconds,getattr(self,'followup_wall_at_start',phase.wall_seconds)+live)+cost.wall_seconds
        combined=BudgetTracker.combine(total,cost)
        for name,limit in self.tracker.budget.to_dict().items():
            value=getattr(combined,name)
            if value is not None and value>limit:raise BudgetExceeded('Task12 overall ceiling: '+name)
        if combined.student_forward_calls>self.plan['overall_student_forward_calls']:raise BudgetExceeded('Task12 overall student ceiling')
        phase_values=BudgetTracker.combine(phase,cost).to_dict();phase_values['wall_seconds']=phase_wall
        enforce_phase_caps(phase_values,self.plan['phase_caps'])
    def disk(self):return sum(p.stat().st_size for p in ORIGINAL.rglob('*') if p.is_file())


def make_plan(config):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return json.loads((OUT/'plan.json').read_text())
    old=json.loads((ORIGINAL/'plan.json').read_text());receipt=json.loads((ROOT/'provenance/single-merge-original-135m-2026-10-27.json').read_text())
    manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=old['data_manifest_sha256'],roles={'probe_score','dev'})
    used={x['component_id'] for rows in old['selected'].values() for x in rows}
    for label in ('probe_score','probe_damage'):
        raw=json.loads((ORIGINAL/f'{label}-pool.json').read_text());used.update(r['component'] for r in raw['eligibility']['checks'])
    choices={}
    for label,role,count in [('score_candidates','probe_score',48),('damage_candidates','dev',64)]:
        eligible=sorted((r for r in manifest.records if manifest.roles[r.document_id]==role and manifest.membership[r.document_id] not in used and r.source=='HuggingFaceFW/fineweb-edu' and r.metadata.get('token_count',0)>=512),key=lambda r:hashlib.sha256(('task12-precision-seed19:'+r.document_id).encode()).digest())
        selected=[]
        for r in eligible:
            c=manifest.membership[r.document_id]
            if c in used:continue
            selected.append({'document_id':r.document_id,'component_id':c,'role':role,'token_sha256':r.token_sha256});used.add(c)
            if len(selected)==count:break
        if len(selected)!=count:raise ValueError('insufficient fixed metadata reserve')
        choices[label]=selected
    parents=[]
    for path in sorted((ORIGINAL/'runs').glob('*/spec.json')):
        parents.append({'spec_path':str(path.relative_to(ROOT)),'spec_sha256':sha256_file(path),'cost_sha256':sha256_file(path.parent/'costs.jsonl')})
    p=dict(schema_version=1,label='fixed fresh-component exploratory precision repeat; original study remains separately primary',
        parent_result_sha256=sha256_file(ROOT/'provenance/single-merge-original-135m-2026-10-27.json'),parent_plan_sha256=sha256_file(ORIGINAL/'plan.json'),parent_runs=parents,
        old_pretrained_wall_seconds=receipt['aggregate_cost']['wall_seconds'],old_separate_analysis_seconds=receipt['post_profile_cpu_control_analysis_seconds_separate']+receipt['analysis_audit_wall_seconds_separate'],
        carry_in={'audit_wall_seconds':old['task12_carry_in']['audit_wall_seconds']+receipt['post_profile_cpu_control_analysis_seconds_separate']+receipt['analysis_audit_wall_seconds_separate'],'spec_paths':[],'evidence':{}},
        data_manifest_sha256=old['data_manifest_sha256'],teacher_asset_sha256=old['teacher_asset_sha256'],numerics_path=old['numerics_path'],numerics_sha256=old['numerics_sha256'],
        primary_pairs=json.loads((ORIGINAL/'pairs-amended.json').read_text())['sample'],original18=json.loads((ORIGINAL/'pairs.json').read_text())['sample'],first_replay=[12,20],retained_primary_pairs=[[0,15],[1,5]],
        candidate_order=choices,new_score_components=16,new_natural_damage_components=32,seed_data_order=19,seed_model_and_probes=17,bootstrap_resamples=2000,bootstrap_seed=17,
        masks='unchanged first512 probe policies; canonical WindowPolicy512/max4/seed17 natural damage; positive target mask required',methods=old['methods'],
        analyses=['separate fresh16score/32damage repeat on same20primary states','fixed original18-only secondary','explicitly exploratory pooled24score/48damage secondary; never replaces original or fresh results'],
        replay='18 strict charged original one-update reconstructions; initialized and fitted digests must match exactly, no retries on mismatch; all20 verified before new predictions',
        overall_budgets=config.budgets.resolve().to_dict(),overall_student_forward_calls=50000,
        phase_caps={'wall_seconds':3600,'teacher_forward_calls':160000,'student_forward_calls':24000,'updates':18,'valid_target_presentations':147456,'search_proposals':2000,'search_seconds':300},
        evaluated_logical_positions_cap=2500000,logical_positions_upper_bound=20*logical_position_count(score_documents=16,damage_documents=32,depths=2,donor_target_counts=[511]*128),
        forecast={'replay_seconds_per_missing_state':30,'evaluation_seconds_per_state':100,'eligibility_loading_analysis_seconds':240,'total_followup_seconds':2780,'basis':'original54-84s fit+8score+dev8, ~25s damage; replay skips scoring, fresh scoring doubles8to16 and damage doubles16to32; first actual replay/evaluation must check forecast'},
        protected_reads=False,no_new_optimization_of_primary_states=True)
    _write(OUT/'plan.json',p);return p


def _start(stage,p,config):
    recovery=load_recovery_authorization()
    expected=recovery['amended_overall_budgets'] if recovery else p['overall_budgets']
    if config.budgets.resolve().to_dict()!=expected:raise ValueError('frozen overall budgets changed')
    sources=sorted([*ROOT.glob('poc/**/*.py'),*[(ROOT/'provenance'/name) for name in ('measure_single_merge.py','measure_single_merge_sensitivity.py','measure_training_pilot.py','measure_probe_pilot.py','prepare_single_merge.py','measure_single_merge_precision.py')]])
    hashes={str(f.relative_to(ROOT)):sha256_file(f) for f in sources}
    original=json.loads((ORIGINAL/'source-amendment.json').read_text())['new_source_sha256']
    for name,digest in original.items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('original numerical/helper source changed: '+name)
    parent_ids=tuple(sorted({Path(x['spec_path']).parent.name for x in p['parent_runs']}|{f.parent.name for f in (OUT/'runs').glob('*/spec.json')}))
    recovery=load_recovery_authorization()
    spec=RunSpec(method_id='M',method_variant='task12_precision_'+stage,resolved_config={'plan_sha256':sha256_file(OUT/'plan.json'),'source_sha256':hashes,'carry_in':p['carry_in'],'stage':stage,'original_candidate_training_updates':1,'replay_does_not_add_learning':True,'recovery_authorization':recovery},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_patch_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'])).hexdigest(),dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),input_manifests={'data':p['data_manifest_sha256'],'teacher':p['teacher_asset_sha256'],'parent_original_receipt':p['parent_result_sha256']},hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},seeds={'data_order':19,'model_and_probes':17,'bootstrap':17},parent_run_ids=parent_ids,budgets=config.budgets.resolve())
    attempt=start_attempt(spec,OUT/'runs');rec=PrecisionRecorder(BudgetTracker(spec.budgets,attempt),p)
    for f in sources:
        destination=attempt.path/'source'/f.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(f.read_bytes())
    (attempt.path/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD']))
    return attempt,rec,spec


def _load_models(p,config,rec):
    required=12 if config.budgets.resolve().memory_bytes>8*1024**3 else 10
    if psutil.virtual_memory().available<required*1024**3:raise ValueError('fresh available memory below approved reserve policy')
    asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=p['teacher_asset_sha256'])
    teacher=rec.call(lambda:load_teacher(asset,config).to('mps'));current=rec.call(lambda:make_bank(teacher,canonicalize_partition([[i] for i in range(30)],30),8,17))
    return teacher,current


def reconstruct(stage,p,config,rec,attempt):
    teacher,current=_load_models(p,config,rec);old=json.loads((ORIGINAL/'plan.json').read_text())
    manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=p['data_manifest_sha256'],roles={'fit'})
    docs=_documents(manifest,old['selected']['medoid_and_fit'],rec);batches=[_canonical(d,manifest.membership[d.document_id],rec)[0] for d in docs]
    pool=[ProbeRecord(**r) for r in json.loads((ORIGINAL/'fit-pool.json').read_text())['rows']]
    retained={tuple(x) for x in p['retained_primary_pairs']};first=tuple(p['first_replay'])
    pairs=[first] if stage=='replay_first' else [tuple(x) for x in p['primary_pairs'] if tuple(x) not in retained|{first}]
    if stage!='replay_first' and not (OUT/'states'/f'{first[0]:02d}-{first[1]:02d}'/'verified.json').exists():raise ValueError('first strict replay has not passed')
    rows=[]
    for pair in pairs:
        dest=OUT/'states'/f'{pair[0]:02d}-{pair[1]:02d}'
        if dest.exists():raise ValueError('replay destination exists; no silent retry')
        original_path=ORIGINAL/'trials-final'/f'{pair[0]:02d}-{pair[1]:02d}.json';original=json.loads(original_path.read_text());started=time.monotonic();before=rec.aggregate()
        trial=initialize_merge(current,teacher,pair,_restricted(manifest,docs),17,batches=batches,recorder=rec)
        require_replay_identity(original['identity'],trial.initialized_digest)
        if trial.medoid_scores!=tuple(original['medoid_scores']):raise ValueError('original medoid fit records/values differ before replay')
        trial.sources,templates=training_sources(pool,trial)
        parent=json.loads((ORIGINAL/'latest.json').read_text())['attempt'];saved_templates=json.loads((ROOT/parent/f'train-templates-{pair[0]}-{pair[1]}.json').read_text())
        if json.loads(json.dumps(templates))!=saved_templates:raise ValueError('original training token/mask/order identity changed')
        if sha256_file(ROOT/p['numerics_path'])!=p['numerics_sha256']:raise ValueError('numeric qualification receipt changed')
        cfg=TrainingConfig(max_updates=1,phase='local',affected_depths=trial.affected_depths,mixture_weights={'natural':.5,'student':.25,'donor':.25},kl_floor_per_target=json.loads((ROOT/p['numerics_path']).read_text())['negative_kl_floor_per_token'],numerical_profile=EXECUTION_PROFILE)
        trial=fit_trial(trial,cfg,config.budgets.resolve())
        require_replay_identity(original['identity'],trial.initialized_digest,trial.model.manifest().checkpoint_digest)
        save_trial_delta(trial,current,dest)
        row={'pair':pair,'initialized_digest':trial.initialized_digest,'fitted_digest':trial.fitted_digest,'original_trial_sha256':sha256_file(original_path),'source_templates_sha256':hashlib.sha256(canonical_json({'rows':templates})).hexdigest(),'training_identity':cfg.identity(),'fit':trial.fit_summary,'seconds':time.monotonic()-started,'cost_before':before.to_dict(),'cost_after':rec.aggregate().to_dict(),'status':'exact_match','not_additional_learning':True}
        _write(dest/'verified.json',row);append_event(attempt,dict(kind='strict_replay_verified',**row));rows.append(row)
        print(json.dumps({'replay_pair':pair,'exact':True,'seconds':row['seconds']}),flush=True)
        del trial;gc.collect();torch.mps.empty_cache()
    if stage=='replay_remaining':
        index=[]
        for pair in map(tuple,p['primary_pairs']):
            path=(ORIGINAL/'profile-delta' if pair==(0,15) else ORIGINAL/'sensitivity-delta-01-05') if pair in retained else OUT/'states'/f'{pair[0]:02d}-{pair[1]:02d}'
            restored=rec.call(lambda:restore_trial_delta(current,teacher,path,recorder=rec))
            original=json.loads((ORIGINAL/'trials-final'/f'{pair[0]:02d}-{pair[1]:02d}.json').read_text())
            require_replay_identity(original['identity'],restored.initialized_digest,restored.model.manifest().checkpoint_digest)
            index.append({'pair':pair,'path':str(path.relative_to(ROOT)),'metadata_sha256':sha256_file(path/'metadata.json'),'weights_sha256':sha256_file(path/'weights.safetensors'),'identity':original['identity']})
            del restored;gc.collect();torch.mps.empty_cache()
        _write(OUT/'verified-states.json',{'plan_sha256':sha256_file(OUT/'plan.json'),'states':index,'all20_verified_before_new_predictions':True})
    return {'replays':rows}


def prepare(p,rec,attempt):
    if (OUT/'data.json').exists():raise ValueError('fixed data preparation already exists')
    manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=p['data_manifest_sha256'],roles={'probe_score','dev'})
    asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=p['teacher_asset_sha256']);tokenizer=rec.call(lambda:load_tokenizer(asset))
    oldpool=json.loads((ORIGINAL/'probe_score-pool.json').read_text());base=next(r for r in oldpool['rows'] if r['status']=='accepted');cal=json.loads((ORIGINAL/'calibration.json').read_text());thresholds=PathologyThresholds(tuple(PathologyObservation(**x) for x in cal['thresholds']['observations']),.01,.99)
    pool=[];selected=[];eligibility=[]
    for entry in p['candidate_order']['score_candidates']:
        doc=_documents(manifest,[entry],rec)[0]
        policy=ProbePolicy(selection_mode='seeded_random',seed=17,per_document_cap=4,generated_edits_per_document=4,random_component_fraction=1.,depth_regions=(tuple(range(30)),),depth_region_balance=(1.,),selection_count=1,max_pool_records=120,max_cache_entries=1,max_cache_bytes=4*1024**2,thresholds=thresholds,roles={doc.document_id:'probe_score'},components={doc.document_id:entry['component_id']},allowed_roles=('probe_score',),use='rank',teacher_digest=base['teacher_digest'],candidate_digest=base['candidate_digest'],tokenizer_digest=base['tokenizer_digest'],dtype='float32',backend='torch_mps',boundary_version=base['boundary_version'],execution_profile=EXECUTION_PROFILE,context_tokens=512,recorder=rec)
        rows=list(generate_donors([doc],tokenizer,policy));good=_accepted(rows,tuple(range(30)));pool.extend(rows)
        eligibility.append({'document_id':doc.document_id,'component_id':entry['component_id'],'eligible':bool(good),'row_count':len(rows),'rejection_reasons':[r.rejection_reason or ('no_valid_targets' if not any(r.target_mask) else 'identity_or_short') for r in rows if r not in good]})
        _write(OUT/'eligibility-score'/f'{doc.document_id}.json',{'entry':entry,'check':eligibility[-1],'rows':[asdict(r) for r in rows]})
        if good:selected.append(entry)
        if len(selected)==16:break
    if len(selected)!=16:raise ValueError('fixed fresh score reserve underfilled; stop without population amendment')
    windows=[];damage_eligibility=[]
    for entry in p['candidate_order']['damage_candidates']:
        doc=_documents(manifest,[entry],rec)[0]
        raw=Path(doc.token_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=doc.token_sha256:raise ValueError('fresh canonical token hash mismatch')
        tokens=json.loads(raw)['fields']['text']
        if len(tokens)<512:
            damage_eligibility.append({'document_id':doc.document_id,'eligible':False,'reason':'verified_token_length_below512','length':len(tokens)})
            _write(OUT/'eligibility-damage'/f'{doc.document_id}.json',{'entry':entry,'check':damage_eligibility[-1]});continue
        batch,w=_canonical(doc,entry['component_id'],rec)
        windows.append({'entry':entry,'window':asdict(w)});damage_eligibility.append({'document_id':doc.document_id,'eligible':True})
        _write(OUT/'eligibility-damage'/f'{doc.document_id}.json',{'entry':entry,'check':damage_eligibility[-1],'window':asdict(w)})
        if len(windows)==32:break
    if len(windows)!=32:raise ValueError('fixed fresh damage reserve underfilled')
    _write(OUT/'pool.json',[asdict(r) for r in pool]);_write(OUT/'windows.json',windows)
    data={'plan_sha256':sha256_file(OUT/'plan.json'),'score_entries':selected,'score_eligibility':eligibility,'damage_eligibility':damage_eligibility,'pool_sha256':sha256_file(OUT/'pool.json'),'windows_sha256':sha256_file(OUT/'windows.json'),'no_candidate_predictions':True}
    _write(OUT/'data.json',data);return {'data':data}


def load_recovery_authorization():
    path=OUT/'recovery-authorization.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value['plan_sha256']!=sha256_file(OUT/'plan.json'):raise ValueError('recovery plan binding changed')
    for name,digest in (value['authorized_failures']|value['reused_artifact_sha256']).items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('recovery artifact binding changed: '+name)
    return value


def reuse_eligibility_pool(entries,directory,destination,*,wanted):
    """Stream exact saved rows, preserving failures without a second edit search."""
    checks=[];selected=[];first=True
    with destination.open('x') as stream:
        stream.write('[')
        for entry in entries:
            path=directory/(entry['document_id']+'.json')
            if not path.exists():raise ValueError('missing frozen candidate evidence')
            saved=json.loads(path.read_text())
            if saved['entry']!=entry:raise ValueError('saved eligibility candidate identity changed')
            checks.append(saved['check'])
            if saved['check']['eligible']:selected.append(entry)
            for row in saved['rows']:
                if not first:stream.write(',')
                json.dump(row,stream,sort_keys=True);first=False
            del saved;gc.collect()
            if len(selected)==wanted:break
        stream.write(']\n');stream.flush();__import__('os').fsync(stream.fileno())
    if len(selected)!=wanted:raise ValueError('saved eligibility population underfilled')
    return checks,selected


def resume_preparation(p,rec,attempt):
    if not rec.recovery:raise ValueError('explicit failed-preparation recovery required')
    if (OUT/'data.json').exists():raise ValueError('fixed data preparation already complete')
    # Reuse all saved edits and windows; never regenerate or alter their order.
    eligibility,selected=reuse_eligibility_pool(p['candidate_order']['score_candidates'],OUT/'eligibility-score',OUT/'pool.json',wanted=16)
    manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=p['data_manifest_sha256'],roles={'dev'})
    windows=[];damage_checks=[]
    for entry in p['candidate_order']['damage_candidates']:
        path=OUT/'eligibility-damage'/f"{entry['document_id']}.json"
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['entry']!=entry:raise ValueError('saved damage candidate identity changed')
            damage_checks.append(saved['check'])
            if saved['check']['eligible']:windows.append({'entry':entry,'window':saved['window']})
        else:
            gc.collect();rec.check(CostRecord())
            memory=rec.memory()
            with (OUT/'preparation-memory.jsonl').open('a') as log:log.write(json.dumps({'document_id':entry['document_id'],'before_read_bytes':memory,'seconds':time.monotonic()-rec.started})+'\n')
            if memory>4*1024**3:raise BudgetExceeded('isolated preparation4GiB growth stop')
            doc=_documents(manifest,[entry],rec)[0];raw=Path(doc.token_path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=doc.token_sha256:raise ValueError('fresh canonical token hash mismatch')
            tokens=json.loads(raw)['fields']['text'];check={'document_id':doc.document_id,'eligible':len(tokens)>=512}
            if not check['eligible']:check.update(reason='verified_token_length_below512',length=len(tokens));saved={'entry':entry,'check':check}
            else:
                batch,w=_canonical(doc,entry['component_id'],rec);saved={'entry':entry,'check':check,'window':asdict(w)};windows.append({'entry':entry,'window':asdict(w)});del batch,w
            _write(path,saved);damage_checks.append(check);del doc,raw,tokens,saved;gc.collect();rec.record(CostRecord())
            memory=rec.memory()
            with (OUT/'preparation-memory.jsonl').open('a') as log:log.write(json.dumps({'document_id':entry['document_id'],'after_gc_bytes':memory,'seconds':time.monotonic()-rec.started})+'\n')
            if memory>4*1024**3:raise BudgetExceeded('isolated preparation4GiB growth stop')
        if len(windows)==32:break
    if len(windows)!=32:raise ValueError('fixed fresh damage reserve underfilled')
    _write(OUT/'windows.json',windows)
    data={'plan_sha256':sha256_file(OUT/'plan.json'),'score_entries':selected,'score_eligibility':eligibility,'damage_eligibility':damage_checks,'pool_sha256':sha256_file(OUT/'pool.json'),'windows_sha256':sha256_file(OUT/'windows.json'),'no_candidate_predictions':True,'recovery_authorization_sha256':sha256_file(OUT/'recovery-authorization.json')}
    _write(OUT/'data.json',data);return {'data':data,'reused_score_candidates':len(selected),'no_edit_regeneration':True}


def _eval_total():
    path=OUT/'logical-positions.jsonl'
    return sum(json.loads(x)['positions'] for x in path.read_text().splitlines()) if path.exists() else 0


def evaluate(stage,p,config,rec,attempt):
    states=json.loads((OUT/'verified-states.json').read_text());data=json.loads((OUT/'data.json').read_text())
    if len(states['states'])!=20 or not states['all20_verified_before_new_predictions'] or states['plan_sha256']!=sha256_file(OUT/'plan.json'):raise ValueError('all original states must be verified before fresh predictions')
    for name in ('pool','windows'):
        if sha256_file(OUT/(name+'.json'))!=data[name+'_sha256']:raise ValueError('fresh frozen data changed')
    pool=[ProbeRecord(**r) for r in json.loads((OUT/'pool.json').read_text())];score_by_id={}
    for r in _accepted(pool,tuple(range(30))):score_by_id.setdefault(r.recipient_id,r)
    score_batches=[_batch(r.recipient_tokens,device='mps',document_id=r.recipient_id,cluster_id=r.recipient_component) for _,r in sorted(score_by_id.items())]
    damage_batches=[_batch(x['window']['token_ids'],device='mps',document_id=x['entry']['document_id'],cluster_id=x['entry']['component_id'],targets=x['window']['target_indices']) for x in json.loads((OUT/'windows.json').read_text())]
    teacher,current=_load_models(p,config,rec);calibration=json.loads((ORIGINAL/'calibration.json').read_text())['summary'];completed=[]
    for item in states['states']:
        pair=tuple(item['pair']);path=OUT/'trials'/f'{pair[0]:02d}-{pair[1]:02d}.json'
        if path.exists():continue
        state_path=ROOT/item['path']
        if sha256_file(state_path/'metadata.json')!=item['metadata_sha256'] or sha256_file(state_path/'weights.safetensors')!=item['weights_sha256']:raise ValueError('verified state artifact changed')
        baseline_memory=rec.memory(label='before_trial_restore');started=time.monotonic();trial=rec.call(lambda:restore_trial_delta(current,teacher,state_path,recorder=rec));require_replay_identity(item['identity'],trial.initialized_digest,trial.model.manifest().checkpoint_digest)
        donor_rows=_accepted(pool,pair);positions=logical_position_count(score_documents=16,damage_documents=32,depths=2,donor_target_counts=[sum(r.target_mask) for r in donor_rows])
        if _eval_total()+positions>p['evaluated_logical_positions_cap']:raise BudgetExceeded('fresh logical evaluation positions ceiling')
        with (OUT/'logical-positions.jsonl').open('a') as stream:stream.write(json.dumps({'pair':pair,'positions':positions,'scope':'precharged full trial KL/NLL, shared MSE/CKA positions and conditional entropy; failures retain charge'})+'\n');stream.flush();__import__('os').fsync(stream.fileno())
        natural=natural_metrics(teacher,trial.model,score_batches,rec);rec.memory(label='after_natural_score')
        activation=activation_scores(teacher,trial,score_batches,rec);rec.memory(label='after_activation_score')
        probes,probe_rows=probe_observations(teacher,trial,pool,rec,calibration);rec.memory(label='after_probe_score')
        original=json.loads((ORIGINAL/'trials-final'/f'{pair[0]:02d}-{pair[1]:02d}.json').read_text());weight=next(iter(original['scores']['weight_distance'].values()));scores={k:{} for k in p['methods']}
        for row in natural:
            c=row['component'];v=probes[c];values=fixed_scores(natural=row['kl_per_target'],student=_mean(v['student']),random_donor=_mean(v['random']),active_donor=_mean(v['active']))|activation[c]|{'weight_distance':weight}
            for name,value in values.items():scores[name][c]=value
        damage=natural_metrics(teacher,trial.model,damage_batches,rec);rec.memory(label='after_natural_damage')
        row={'identity':item['identity'],'scores':scores,'natural_damage':{r['component']:r['kl_per_target'] for r in damage},'natural_score_rows':natural,'probe_score_rows':probe_rows,'natural_damage_rows':damage,'logical_positions':positions,'wall_seconds':time.monotonic()-started,'plan_sha256':sha256_file(OUT/'plan.json'),'fixed_data_sha256':sha256_file(OUT/'data.json'),'no_fitting':True,'original_trial_sha256':sha256_file(ORIGINAL/'trials-final'/path.name),'unfiltered_primary':True}
        _write(path,row);completed.append({'pair':pair,'seconds':row['wall_seconds'],'logical_positions':positions});append_event(attempt,dict(kind='fresh_trial_complete',**completed[-1]));print(json.dumps(completed[-1]),flush=True)
        del trial;gc.collect();torch.mps.empty_cache()
        cleanup_memory=rec.memory(label='after_trial_cleanup');rec.record(CostRecord())
        completed[-1].update(memory_before_restore=baseline_memory,memory_after_cleanup=cleanup_memory,cleanup_growth_bytes=cleanup_memory-baseline_memory)
        append_event(attempt,{'kind':'trial_cleanup_memory','pair':pair,'before_restore':baseline_memory,'after_cleanup':cleanup_memory})
        if stage=='evaluate_first':break
    return {'completed':completed,'logical_positions_charged':_eval_total()}


def analyze(p):
    def tables(rows,label):
        rows=sorted(rows,key=lambda r:tuple(r['identity']['pair']));identities=tuple(TrialIdentity(tuple(r['identity']['pair']),r['identity']['initialized_digest'],r['identity']['fitted_digest']) for r in rows)
        records=[TrialRecord(i,r['scores'],label,{k:(0,0.) for k in r['scores']}) for i,r in zip(identities,rows,strict=True)]
        score_tables=[score_trials(records,k,records=label) for k in sorted(p['methods'],key=lambda k:(k!='natural_kl',k))]
        comps=tuple(sorted(rows[0]['natural_damage']));damage=DamageTable(identities,comps,tuple(tuple(r['natural_damage'][c] for c in comps) for r in rows),label+'-damage')
        return {'regret':{s.method:asdict(selected_merge_regret(s,damage)) for s in score_tables},'uncertainty':bootstrap_regret(score_tables,damage,seed=17,resamples=2000),'pairs':[list(i.pair) for i in identities],'score_components':len(score_tables[0].components),'damage_components':len(comps)}
    rows=[json.loads(path.read_text()) for path in sorted((OUT/'trials').glob('*.json'))]
    if len(rows)!=20:raise ValueError('entire fixed precision batch required before analysis')
    original18={tuple(x) for x in p['original18']};pooled=[]
    for row in rows:
        pair=row['identity']['pair'];old=json.loads((ORIGINAL/'trials-final'/f'{pair[0]:02d}-{pair[1]:02d}.json').read_text())
        if old['identity']!=row['identity']:raise ValueError('pooled states differ')
        if set(old['natural_damage'])&set(row['natural_damage']) or set(old['scores']['natural_kl'])&set(row['scores']['natural_kl']):raise ValueError('fresh component overlap')
        pooled.append({'identity':row['identity'],'scores':{k:old['scores'][k]|row['scores'][k] for k in p['methods']},'natural_damage':old['natural_damage']|row['natural_damage']})
    result={'fresh_repeat':tables(rows,'fresh16score32damage'),'fresh_original18_secondary':tables([r for r in rows if tuple(r['identity']['pair']) in original18],'fresh-original18'),'pooled_exploratory_secondary':tables(pooled,'pooled24score48damage'),'original_result_remains_separate':True,'no_further_sample_additions':True}
    _write(OUT/'analysis.json',result);return {'analysis_path':str((OUT/'analysis.json').relative_to(ROOT))}


def run(stage,config):
    started=time.monotonic();p=make_plan(config)
    if stage=='plan':print(json.dumps({'plan':str(OUT/'plan.json'),'forecast':p['forecast'],'new_score_components':16,'new_damage_components':32}));return 0
    recovery=load_recovery_authorization();allowed=recovery['authorized_failures'] if recovery else {}
    if any(json.loads(path.read_text()).get('status')=='failed' and allowed.get(str(path.relative_to(ROOT)))!=sha256_file(path) for path in (OUT/'runs').glob('*/attempts/*/result.json')):raise ValueError('a precision/replay failure exists; no retries without controller ruling')
    attempt,rec,spec=_start(stage,p,config);result={'status':'running','stage':stage,'attempt':str(attempt.path.relative_to(ROOT)),'source_commit':spec.source_commit,'source_sha256':dict(spec.resolved_config['source_sha256'])}
    try:
        if config.budgets.resolve().memory_bytes>8*1024**3 and psutil.virtual_memory().available<12*1024**3:raise ValueError('fresh available RAM below12GiB reserve')
        if stage.startswith('replay_'):result.update(reconstruct(stage,p,config,rec,attempt))
        elif stage=='prepare':result.update(prepare(p,rec,attempt))
        elif stage=='prepare_resume':result.update(resume_preparation(p,rec,attempt))
        elif stage.startswith('evaluate_'):result.update(evaluate(stage,p,config,rec,attempt))
        elif stage=='analyze':result.update(analyze(p))
        else:raise ValueError('unknown precision stage')
        result['status']='complete'
    except Exception as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc));append_event(attempt,dict(kind='failed',reason=str(exc)))
    finally:
        rec.finish_attempt(result,elapsed=time.monotonic()-started);result['followup_phase_cost']=rec.phase_totals().to_dict();_write(attempt.path/'result.json',result)
        _write(OUT/'latest.json',{'attempt':result['attempt'],'status':result['status'],'stage':stage,'result_sha256':sha256_file(attempt.path/'result.json')})
        print(json.dumps({'status':result['status'],'stage':stage,'failure':result.get('failure_reason'),'overall':result['aggregate_phase_cost'],'followup':result['followup_phase_cost']}),flush=True)
    return 0 if result['status']=='complete' else 4

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['plan','replay_first','replay_remaining','prepare','prepare_resume','evaluate_first','evaluate_remaining','analyze']);parser.add_argument('--config',type=Path,default=ROOT/'configs/pilot-precision.yaml');args=parser.parse_args()
    raise SystemExit(run(args.stage,load_config(args.config,phase='exploratory')))

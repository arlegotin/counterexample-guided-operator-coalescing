"""Source-bound, sequential Task12 workload; no protected readers or downloads."""
from dataclasses import asdict,replace
import gc,hashlib,json,random,subprocess,time
from pathlib import Path
import psutil
import torch
from poc.bank import make_bank,canonicalize_partition,PROJECTIONS
from poc.budget import BudgetExceeded,BudgetTracker
from poc.contracts import CostRecord,RunSpec
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.data.windows import TokenDocument,WindowPolicy,make_windows
from poc.diagnostics import (PairSamplingPolicy,pair_population,pair_stratum,sample_pairs,ensure_excluded_coverage,
    TrialIdentity,TrialRecord,DamageTable,ScoreTable,score_trials,selected_merge_regret,bootstrap_regret,fixed_scores,run_controls,shortlist_misses)
from poc.merge import initialize_merge,fit_trial,centered_linear_cka,initializer_sensitivity_control,save_trial_delta,restore_trial_delta
from poc.losses import stream_metrics
from poc.probes import PathologyObservation,PathologyThresholds,ProbePolicy,generate_donors,prefix_for_probe,apply_probe,local_discrepancy
from poc.records import append_event,canonical_json,start_attempt
from poc.runtime.torch_decoder import EXECUTION_PROFILE
from poc.teacher import AssetManifest,load_teacher,load_tokenizer,sha256_file
from poc.teacher_validation import _batch,_restricted,_write
from poc.train import TrainingConfig,TrainingSources,ProbeTrainingSource,_Dispatch,_Counts
from provenance.measure_training_pilot import TrainingPilotRecorder

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/task12'


class DiagnosticRecorder(TrainingPilotRecorder):
    def __init__(self,*args,profile=False,profile_cap=600,**kwargs):
        self.profile,self.profile_cap=profile,profile_cap
        super().__init__(*args,**kwargs)
    def aggregate(self):
        from poc.checkpoint import _spec
        from poc.records import AttemptRecord
        total=CostRecord(wall_seconds=self.carry['audit_wall_seconds'])
        for path in (self.phase_root/'runs').glob('*/spec.json'):
            raw=json.loads(path.read_text())
            if raw['method_variant'] not in ('task12_single_merge_score_only','task12_single_merge_sensitivity_refit'):continue
            if raw['resolved_config']['plan_sha256']!=self.plan_sha256:raise ValueError('Task12 aggregate plan changed')
            key=str(path.parent)
            if key not in self._phase_trackers:
                identity=_spec(raw)
                self._phase_trackers[key]=self.tracker if key==str(self.tracker.path.parent) else BudgetTracker(identity.budgets,AttemptRecord(path.parent.name,'aggregate_only',path.parent/'attempts'/'aggregate_only',identity))
            total=BudgetTracker.combine(total,self._phase_trackers[key].totals)
        if total.memory_bytes is not None:total=replace(total,memory_unavailable_reason=None)
        return total
    def disk(self):return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    def check_aggregate(self,cost=None):
        super().check_aggregate(cost)
        if self.profile and max(self.aggregate().wall_seconds,self.phase_wall_at_start+time.monotonic()-self.started)>self.profile_cap:
            raise BudgetExceeded('Task12 profile600s ceiling includes carry-in and retries')


def _cost_delta(after,before):
    result={}
    for key,value in after.to_dict().items():
        previous=getattr(before,key)
        result[key]=value-previous if isinstance(value,(int,float)) and key not in ('memory_bytes','disk_bytes') else value
    return result


def source_compatible(stored,current,amendment):
    if stored==current:return True
    return bool(amendment and stored==amendment.get('old_source_sha256') and current==amendment.get('new_source_sha256'))


def _identity(x):return TrialIdentity(tuple(x['pair']),x['initialized_digest'],x['fitted_digest'])
def _mean(values):
    values=tuple(values)
    if not values:raise ValueError('missing positive-weight source observations')
    return sum(values)/len(values)


def _documents(manifest,entries,rec):
    ids=[r['document_id'] for r in entries];rows=[r for r in manifest.records if r.document_id in ids]
    role=entries[0]['role'];purpose={'fit':'fit','dev':'selection','probe_score':'ranking'}[role]
    docs=rec.call(lambda:list(open_records(_restricted(manifest,rows),purpose=purpose)))
    return sorted(docs,key=lambda r:ids.index(r.document_id))


def _canonical(doc,component,rec):
    def load():
        raw=Path(doc.token_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=doc.token_sha256:raise ValueError('token hash mismatch')
        data=json.loads(raw)
        if data['document_id']!=doc.document_id:raise ValueError('token document mismatch')
        w=make_windows(TokenDocument(doc.document_id,tuple(data['fields']['text'])),WindowPolicy())[0]
        if len(w.token_ids)!=512:raise ValueError('unresolved pre-prediction canonical length')
        return w
    w=rec.call(load)
    return _batch(w.token_ids,device='mps',document_id=doc.document_id,cluster_id=component,targets=w.target_indices),w


def calibrate(teacher,docs,manifest,rec):
    observations=[];windows=[];identities=[]
    for doc in docs:
        batch,w=_canonical(doc,manifest.membership[doc.document_id],rec);windows.append(asdict(w))
        with torch.no_grad():
            hidden=rec.call(lambda:teacher.final_hidden(batch),teacher_forward_calls=1)
            entropy=[]
            for pos in range(0,511,32):
                logits=rec.call(lambda:teacher.readout(hidden[:,pos:min(511,pos+32)]),teacher_forward_calls=1)
                lp=logits.log_softmax(-1);entropy.extend((-(lp.exp()*lp).sum(-1))[0].cpu().tolist())
            for depth in range(30):
                boundary=prefix_for_probe(teacher,batch,depth,source='teacher',recorder=rec)
                if doc is docs[0]:
                    after=rec.call(lambda:teacher.ff_residual(boundary),teacher_forward_calls=1)
                    suffix=rec.call(lambda:teacher.suffix_hidden(after,boundary),teacher_forward_calls=1)
                    exact=torch.equal(hidden,suffix);identities.append(dict(depth=depth,hidden_exact=exact))
                    if not exact:raise ValueError('full-depth natural suffix identity failed')
                norms=boundary.u[0,:-1].norm(dim=-1).cpu().tolist()
                observations.extend(PathologyObservation(doc.document_id,manifest.membership[doc.document_id],'fit',n,e,depth) for n,e in zip(norms,entropy,strict=True))
    thresholds=PathologyThresholds(tuple(observations),.01,.99)
    return thresholds,dict(unique_entropy_positions=8*511,document_components=8,depths=list(range(30)),positions_per_depth=8*511,identity_proofs=identities,windows=windows,
        threshold_digest=thresholds.digest,by_depth={str(d):{'norm':thresholds.interval('norm',d),'entropy':thresholds.interval('entropy',d)} for d in range(30)},
        historical_wording='Task10 independent_entropy_positions means unique positions within8components, not independent tokens; original raw receipt unchanged')


def _policy(teacher,current,thresholds,docs,manifest,rec,role):
    return ProbePolicy(selection_mode='seeded_random',seed=17,per_document_cap=4,generated_edits_per_document=4,random_component_fraction=1.,
        depth_regions=(tuple(range(30)),),depth_region_balance=(1.,),selection_count=1,max_pool_records=max(120,len(docs)*120),max_cache_entries=1,max_cache_bytes=4*1024**2,
        thresholds=thresholds,roles={d.document_id:role for d in docs},components={d.document_id:manifest.membership[d.document_id] for d in docs},allowed_roles=(role,),
        use='fit' if role=='fit' else 'rank',teacher_digest=teacher.manifest().checkpoint_digest,candidate_digest=current.manifest().checkpoint_digest,
        tokenizer_digest=teacher.tokenizer_digest,dtype='float32',backend='torch_mps',boundary_version='post_attention_v1',execution_profile=EXECUTION_PROFILE,context_tokens=512,recorder=rec)


def prepare_pools(teacher,current,tokenizer,thresholds,manifest,fit_docs,p,rec,attempt):
    from poc.contracts import ProbeRecord
    pools={};eligibility={}
    for label in ('fit','probe_score','probe_damage'):
        path=OUT/(label+'-pool.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['plan_sha256']!=sha256_file(OUT/'plan.json') or saved['teacher_digest']!=teacher.manifest().checkpoint_digest:raise ValueError('pool binding mismatch')
            pools[label]=[ProbeRecord(**x) for x in saved['rows']];eligibility[label]=saved['eligibility'];continue
        selected=[];all_rows=[];checks=[];wanted=len(fit_docs) if label=='fit' else (8 if label=='probe_score' else 16)
        entries=p['selected']['medoid_and_fit'] if label=='fit' else p['probe_candidate_order'][label]
        for entry in entries:
            doc=next((d for d in fit_docs if d.document_id==entry['document_id']),None) if label=='fit' else None
            if doc is None:doc=_documents(manifest,[entry],rec)[0]
            rows=list(generate_donors([doc],tokenizer,_policy(teacher,current,thresholds,[doc],manifest,rec,entry['role'])))
            good=[r for r in rows if r.status=='accepted' and len(r.recipient_tokens)==512 and r.recipient_tokens!=r.donor_tokens and any(r.target_mask)]
            eligible=bool(good);checks.append(dict(document_id=doc.document_id,component=manifest.membership[doc.document_id],eligible=eligible,attempt_rows=len(rows),accepted_rows=len(good),rejections=[r.rejection_reason for r in rows if r.status!='accepted']))
            all_rows.extend(rows)
            if eligible:selected.append(doc.document_id)
            if label!='fit' and len(selected)==wanted:break
        if label!='fit' and len(selected)!=wanted:raise ValueError(f'{label} alignment underfilled:{len(selected)}/{wanted}; full-phase sensitivity ruling required')
        if label=='fit' and not selected:raise ValueError('fit donor source absent')
        saved=dict(plan_sha256=sha256_file(OUT/'plan.json'),teacher_digest=teacher.manifest().checkpoint_digest,rows=[asdict(r) for r in all_rows],eligibility={'selected_ids':selected,'checks':checks,'wanted':wanted})
        _write(path,saved);pools[label]=all_rows;eligibility[label]=saved['eligibility']
        append_event(attempt,dict(kind='pool_prepared',label=label,path=str(path.relative_to(ROOT)),sha256=sha256_file(path),eligibility=eligibility[label]))
    return pools,eligibility


def _accepted(pool,pair):
    return [r for r in pool if r.status=='accepted' and r.depth in pair and len(r.recipient_tokens)==512 and r.recipient_tokens!=r.donor_tokens and any(r.target_mask)]


def _record_source(base,source,candidate_digest):
    return replace(base,probe_id=hashlib.sha256(canonical_json({'base':base.probe_id,'source':source})).hexdigest(),source=source,candidate_digest=candidate_digest,
        prefix_source='candidate' if source=='student' else 'teacher',patch_positions=base.patch_positions if source=='donor' else (),
        target_mask=base.target_mask if source=='donor' else (True,)*511,patch_mode=base.patch_mode if source=='donor' else 'full',channel_quarter=base.channel_quarter if source=='donor' else None)


def _boundary(record,student,teacher,rec):
    recipient=_batch(record.recipient_tokens,device='mps',document_id=record.recipient_id,cluster_id=record.recipient_component)
    producer=student if record.prefix_source=='candidate' else teacher
    boundary=prefix_for_probe(producer,recipient,record.depth,source=record.prefix_source,recorder=rec)
    if record.source=='donor':
        donor=_batch(record.donor_tokens,device='mps',document_id=record.donor_id,cluster_id=record.donor_component)
        boundary=apply_probe(boundary,prefix_for_probe(teacher,donor,record.depth,source='teacher',recorder=rec),record)
    return boundary


def training_sources(pool,trial):
    by_key={}
    for r in _accepted(pool,trial.pair):by_key.setdefault((r.recipient_id,r.depth),r)
    items=[];templates=[]
    for key,base in sorted(by_key.items()):
        for source in ('natural','student','donor'):
            record=_record_source(base,source,trial.initialized_digest);templates.append(asdict(record))
            batch=_batch(record.recipient_tokens,device='mps',document_id=record.recipient_id,cluster_id=record.recipient_component)
            batch=replace(batch,target_mask=torch.tensor([record.target_mask],device='mps',dtype=torch.bool))
            def build(student,teacher,recorder,record=record):
                current=replace(record,candidate_digest=student.manifest().checkpoint_digest)
                return _boundary(current,student,teacher,recorder),current
            items.append(ProbeTrainingSource(batch,source,'fit',record.probe_id,build))
    if sum(int(x.template.target_mask.sum()) for x in items)<8192:raise ValueError('insufficient exact-update source targets; no undeclared repetition')
    return TrainingSources(probes=tuple(items),manifest_hashes={'templates':hashlib.sha256(canonical_json({'rows':templates})).hexdigest()}),templates


def natural_metrics(teacher,student,batches,rec):
    rows=[]
    for b in batches:
        metric=rec.call(lambda:stream_metrics(_Dispatch(teacher,_Counts(rec),'teacher'),_Dispatch(student,_Counts(rec),'student'),b,32))
        rows.append(dict(document=b.document_ids[0],component=b.cluster_ids[0],kl_sum=metric.kl,valid_targets=metric.valid_targets,nll_sum=metric.nll,kl_per_target=metric.kl/metric.valid_targets,mask=b.target_mask[0].tolist()))
    return rows


def probe_observations(teacher,trial,pool,rec,calibration,*,damage=False):
    grouped={}
    for r in _accepted(pool,trial.pair):grouped.setdefault((r.recipient_component,r.depth),[]).append(r)
    output=[];main={};digest=trial.fitted_digest
    for (component,depth),bases in sorted(grouped.items()):
        bases=sorted(bases,key=lambda x:x.probe_id)
        chosen=random.Random(17+depth+int(component[:8],16)).randrange(len(bases))
        if damage:bases=[bases[chosen]];chosen=0
        donor_values=[];filtered=[]
        for base in bases:
            r=replace(base,candidate_digest=digest);boundary=_boundary(r,trial.model,teacher,rec)
            metric=local_discrepancy(teacher,trial.model.layers[depth].mlp,boundary,32,recorder=rec)
            if not metric.valid_targets:raise ValueError('empty probe score')
            with torch.no_grad():
                after=rec.call(lambda:teacher.ff_residual(boundary),teacher_forward_calls=1)
                hidden=rec.call(lambda:teacher.suffix_hidden(after,boundary),teacher_forward_calls=1)
                pos=r.patch_positions[0]
                logits=rec.call(lambda:teacher.readout(hidden[:,pos:pos+1]),teacher_forward_calls=1)
                lp=logits.log_softmax(-1);entropy=float((-(lp.exp()*lp).sum(-1)).item());norm=float(boundary.u[0,pos].norm().item())
            limits=calibration['by_depth'][str(depth)];keep=limits['norm'][0]<=norm<=limits['norm'][1] and limits['entropy'][0]<=entropy<=limits['entropy'][1]
            value=metric.kl/metric.valid_targets;donor_values.append(value)
            if keep:filtered.append(value)
            output.append(dict(record=asdict(r),source='donor',kl_sum=metric.kl,valid_targets=metric.valid_targets,kl_per_target=value,threshold_pass=keep,norm=norm,conditional_entropy=entropy,boundary_producer=boundary.model_digest))
        entry=main.setdefault(component,{'random':[],'active':[],'student':[],'filtered_active':[]})
        entry['random'].append(donor_values[chosen]);entry['active'].append(max(donor_values))
        entry['filtered_active'].append(max(filtered) if filtered else None)
        if not damage:
            r=_record_source(bases[0],'student',digest);boundary=_boundary(r,trial.model,teacher,rec)
            metric=local_discrepancy(teacher,trial.model.layers[depth].mlp,boundary,32,recorder=rec)
            entry['student'].append(metric.kl/metric.valid_targets)
            output.append(dict(record=asdict(r),source='student',kl_sum=metric.kl,valid_targets=metric.valid_targets,kl_per_target=metric.kl/metric.valid_targets,boundary_producer=boundary.model_digest))
    return main,output


def activation_scores(teacher,trial,batches,rec):
    result={}
    for b in batches:
        mses=[];ckas=[]
        with torch.no_grad():
            for depth in trial.affected_depths:
                boundary=prefix_for_probe(teacher,b,depth,source='teacher',recorder=rec)
                expected=rec.call(lambda:teacher.ff_residual(boundary),teacher_forward_calls=1)-boundary.u
                actual=rec.call(lambda:teacher.ff_residual(boundary,trial.model.layers[depth].mlp),student_forward_calls=1)-boundary.u
                x=expected[:,:-1][b.target_mask].cpu().double();y=actual[:,:-1][b.target_mask].cpu().double()
                mses.append(float((x-y).square().mean()));ckas.append(1-centered_linear_cka(x,y))
        result[b.cluster_ids[0]]={'activation_mse':_mean(mses),'cka':_mean(ckas)}
    return result


def pair_design(teacher,rec):
    path=OUT/'pairs.json'
    amendment=OUT/'pairs-amended.json'
    if amendment.exists():
        changed=json.loads(amendment.read_text())
        if changed['original_pairs_sha256']!=sha256_file(path):raise ValueError('pair amendment original population changed')
        return changed
    if path.exists():return json.loads(path.read_text())
    distances={}
    started=time.monotonic()
    with torch.no_grad():
        for pair in pair_population(30):
            total=0.;count=0
            for name in PROJECTIONS:
                a=getattr(teacher.layers[pair[0]].mlp,name).weight;b=getattr(teacher.layers[pair[1]].mlp,name).weight
                total+=float((a-b).square().sum());count+=a.numel()
            distances[pair]=total/count
    rec.record(CostRecord(search_seconds=time.monotonic()-started,wall_seconds=time.monotonic()-started))
    cheap=sorted(distances,key=lambda x:(distances[x],x))[:8]
    adjacent=sorted((p for p in distances if p[1]==p[0]+1),key=lambda x:(distances[x],x))[:8]
    remaining=sorted(set(distances)-set(cheap)-set(adjacent));randoms=random.Random(17).sample(remaining,8)
    shortlist=set(cheap+adjacent+randoms);sample=ensure_excluded_coverage(sample_pairs(30,PairSamplingPolicy(17,2)),shortlist,30,17)
    result={'population':[dict(pair=p,stratum=pair_stratum(p,30),weight_distance=distances[p],shortlisted=p in shortlist,sampled=p in sample) for p in sorted(distances)],'sample':sample,'shortlist':sorted(shortlist),'teacher_digest':teacher.manifest().checkpoint_digest}
    _write(path,result);return result


def summarize(records,pair_info):
    records=sorted(records,key=lambda r:tuple(r['identity']['pair']))
    trials=tuple(_identity(r['identity']) for r in records)
    methods=list(records[0]['scores']);trial_records=[TrialRecord(_identity(r['identity']),r['scores'],r['score_manifest'],{k:tuple(v) for k,v in r['costs'].items()}) for r in records]
    tables=[score_trials(trial_records,k,records=records[0]['score_manifest']) for k in methods]
    # Baseline first makes each bootstrap contrast read active-minus-natural.
    tables.sort(key=lambda x:(x.method!='natural_kl',x.method))
    comps=tuple(sorted(records[0]['natural_damage']));damage=DamageTable(trials,comps,tuple(tuple(r['natural_damage'][c] for c in comps) for r in records),'natural-damage16')
    fresh_comps=tuple(sorted(records[0]['intervention_damage']));fresh=DamageTable(trials,fresh_comps,tuple(tuple(r['intervention_damage'][c] for c in fresh_comps) for r in records),'fresh-intervention16')
    regrets={t.method:asdict(selected_merge_regret(t,damage)) for t in tables}
    fresh_regrets={t.method:asdict(selected_merge_regret(t,fresh)) for t in tables}
    secondary_comps=tuple(sorted(records[0]['secondary_scores']));secondary=ScoreTable('canonical_dev8_kl',trials,secondary_comps,tuple(tuple(r['secondary_scores'][c] for c in secondary_comps) for r in records),'canonical-dev8',0,0.)
    shortlisted={tuple(p) for p in pair_info['shortlist']};means=[sum(r['natural_damage'].values())/len(comps) for r in records];best=min(range(len(records)),key=lambda i:(means[i],trials[i].pair))
    return dict(shortlist_misses=(shortlist_misses(damage,shortlisted,seed=17,resamples=2000) if any(t.pair in shortlisted for t in trials) and any(t.pair not in shortlisted for t in trials) else {'status':'not_estimable_without_both_included_and_excluded_pairs'}),regret=regrets,fresh_intervention_regret=fresh_regrets,uncertainty=bootstrap_regret(tables,damage,seed=17,resamples=2000),fresh_uncertainty=bootstrap_regret(tables,fresh,seed=17,resamples=2000),secondary=asdict(selected_merge_regret(secondary,damage)),secondary_uncertainty=bootstrap_regret([secondary],damage,seed=17,resamples=2000),
        coverage={'full_population':435,'sampled':len(records),'sampled_excluded':sum(t.pair not in shortlisted for t in trials),'sampled_best_excluded':trials[best].pair not in shortlisted,'sampled_best_pair':trials[best].pair,'strata':sorted({pair_stratum(t.pair,30) for t in trials})},controls=run_controls(),initializer_control=initializer_sensitivity_control())


def _finish_damage(record,teacher,trial,batches,pool,rec,calibration):
    before=rec.aggregate();started=time.monotonic()
    damage=natural_metrics(teacher,trial.model,batches,rec)
    fresh,fresh_rows=probe_observations(teacher,trial,pool,rec,calibration,damage=True)
    record.update(natural_damage={r['component']:r['kl_per_target'] for r in damage},intervention_damage={c:_mean(v['random']) for c,v in fresh.items()},
        natural_damage_rows=damage,intervention_damage_rows=fresh_rows,damage_status='observed_after_full_phase_protocol_freeze',damage_wall_seconds=time.monotonic()-started,
        damage_cost_before=before.to_dict(),damage_cost_after=rec.aggregate().to_dict())


def run(config):
    p=json.loads((OUT/'plan.json').read_text());plan_sha=sha256_file(OUT/'plan.json');limits=config.budgets.resolve()
    approval=OUT/'full-phase-approved.json';profile=not approval.exists()
    full_policy=json.loads(approval.read_text()) if approval.exists() else {}
    if approval.exists() and json.loads(approval.read_text())['plan_sha256']!=plan_sha:raise ValueError('full-phase approval belongs to another plan')
    amendment_path=OUT/'source-amendment.json'
    source_amendment=json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    source_files=sorted([*ROOT.glob('poc/**/*.py'),ROOT/'provenance/measure_single_merge.py',ROOT/'provenance/measure_training_pilot.py',ROOT/'provenance/measure_probe_pilot.py',ROOT/'provenance/prepare_single_merge.py',ROOT/'provenance/measure_single_merge_sensitivity.py'])
    sources={str(f.relative_to(ROOT)):sha256_file(f) for f in source_files}
    if source_amendment:
        if source_amendment['new_source_sha256']!=sources:raise ValueError('source amendment current source mismatch')
        for path,digest in source_amendment['reused_artifact_sha256'].items():
            if sha256_file(ROOT/path)!=digest:raise ValueError('reused profile artifact changed: '+path)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();patch=subprocess.check_output(['git','diff','--binary','HEAD'])
    spec=RunSpec(method_id='M',method_variant='task12_single_merge_score_only',resolved_config={'plan_sha256':plan_sha,'source_sha256':sources,'carry_in':p['task12_carry_in'],'fit':p['common_fit'],'methods':p['methods'],'student_forward_cap':p['student_call_cap'],'source_amendment_sha256':sha256_file(amendment_path) if amendment_path.exists() else None,'pair_amendment_sha256':sha256_file(OUT/'pairs-amended.json') if (OUT/'pairs-amended.json').exists() else None},source_commit=commit,dirty_patch_sha256=hashlib.sha256(patch).hexdigest(),dependency_lock_sha256=sha256_file(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),input_manifests={'data':p['data_manifest_sha256'],'teacher':p['teacher_asset_sha256'],'windows':p['saved_windows_sha256']},hardware_policy={'backend':'torch_mps','dtype':'float32','execution_profile':EXECUTION_PROFILE},seeds={'diagnostic':17},parent_run_ids=(),budgets=limits)
    attempt=start_attempt(spec,OUT/'runs');rec=DiagnosticRecorder(BudgetTracker(limits,attempt),p['student_call_cap'],profile=profile,profile_cap=p['profile_wall_cap']);started=time.monotonic()
    for f in source_files:
        dest=attempt.path/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    (attempt.path/'source.patch').write_bytes(patch)
    result={'status':'running','attempt':str(attempt.path.relative_to(ROOT)),'source_commit':commit,'source_sha256':sources,'plan_sha256':plan_sha,'profile':profile,'trials':[]}
    try:
        ram=psutil.virtual_memory();result['memory_before']={'available_bytes':ram.available,'total_bytes':ram.total}
        if limits.memory_bytes>6*1024**3 and ram.available<10*1024**3:raise ValueError('fresh headroom below approved8GiB policy')
        if not torch.backends.mps.is_available():raise RuntimeError('qualified MPS unavailable; no silent fallback')
        asset=AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=p['teacher_asset_sha256'])
        tokenizer=rec.call(lambda:load_tokenizer(asset));teacher=rec.call(lambda:load_teacher(asset,config).to('mps'))
        if len(teacher.layers)!=30:raise ValueError('planned435pair depth mismatch')
        current=rec.call(lambda:make_bank(teacher,canonicalize_partition([[i] for i in range(30)],30),8,17))
        manifest=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=p['data_manifest_sha256'],roles={'fit','dev','probe_score'})
        fit_docs=_documents(manifest,p['selected']['medoid_and_fit'],rec);fit_batches=[_canonical(d,manifest.membership[d.document_id],rec)[0] for d in fit_docs]
        calibration_path=OUT/'calibration.json'
        if calibration_path.exists():
            stored=json.loads(calibration_path.read_text())
            if stored['plan_sha256']!=plan_sha or not source_compatible(stored['source_sha256'],sources,source_amendment):raise ValueError('calibration source/plan binding mismatch')
            thresholds=PathologyThresholds(tuple(PathologyObservation(**x) for x in stored['thresholds']['observations']),.01,.99);calibration=stored['summary']
        else:
            thresholds,calibration=calibrate(teacher,_documents(manifest,p['selected']['calibration'],rec),manifest,rec)
            _write(calibration_path,{'plan_sha256':plan_sha,'source_sha256':sources,'thresholds':asdict(thresholds),'summary':calibration})
        result['calibration']=calibration
        pools,eligibility=prepare_pools(teacher,current,tokenizer,thresholds,manifest,fit_docs,p,rec,attempt)
        result['eligibility']=eligibility
        design=pair_design(teacher,rec)
        append_event(attempt,dict(kind='frozen_before_trial_predictions',eligibility=eligibility,pair_design_sha256=sha256_file(OUT/'pairs.json'),calibration_sha256=sha256_file(calibration_path)))
        natural_batches={}
        for label in (('natural_score',) if profile else ('natural_score','natural_damage')):
            docs=_documents(manifest,p['selected'][label],rec);natural_batches[label]=[_canonical(d,manifest.membership[d.document_id],rec)[0] for d in docs]
        # Same recipients and exact first512 contexts for every main scoring method.
        score_by_id={}
        for r in _accepted(pools['probe_score'],tuple(range(30))):score_by_id.setdefault(r.recipient_id,r)
        main_batches=[_batch(r.recipient_tokens,device='mps',document_id=r.recipient_id,cluster_id=r.recipient_component) for _,r in sorted(score_by_id.items())]
        score_manifest=hashlib.sha256(canonical_json({'components':[b.cluster_ids[0] for b in main_batches],'contexts':[b.input_ids[0].tolist() for b in main_batches]})).hexdigest()
        measured=0
        for pair_list in design['sample']:
            pair=tuple(pair_list);path=OUT/('trials' if profile else 'trials-final')/f'{pair[0]:02d}-{pair[1]:02d}.json'
            profile_path=OUT/'trials'/f'{pair[0]:02d}-{pair[1]:02d}.json'
            existing=path if path.exists() else profile_path
            if existing.exists():
                previous=json.loads(existing.read_text())
                if not source_compatible(previous['source_sha256'],sources,source_amendment) or previous['plan_sha256']!=plan_sha:raise ValueError('trial reuse source/plan binding mismatch')
                if 'natural_damage' not in previous and not profile:
                    trial=restore_trial_delta(current,teacher,OUT/'profile-delta',recorder=rec)
                    _finish_damage(previous,teacher,trial,natural_batches['natural_damage'],pools['probe_damage'],rec,calibration)
                    previous['damage_source_sha256']=sources;previous['profile_source_artifact_sha256']=sha256_file(existing)
                    _write(path,previous);del trial;gc.collect();torch.mps.empty_cache()
                result['trials'].append(previous);continue
            rec.check(CostRecord());trial_started=time.monotonic();before=rec.aggregate()
            trial=initialize_merge(current,teacher,pair,_restricted(manifest,fit_docs),17,batches=fit_batches,recorder=rec)
            sensitivity_before=natural_metrics(teacher,trial.model,fit_batches[:3],rec)
            trial.sources,templates=training_sources(pools['fit'],trial)
            _write(attempt.path/f'train-templates-{pair[0]}-{pair[1]}.json',templates)
            if sha256_file(ROOT/p['numerics_path'])!=p['numerics_sha256']:raise ValueError('qualified numeric receipt changed')
            floor=json.loads((ROOT/p['numerics_path']).read_text())['negative_kl_floor_per_token']
            schedule=TrainingConfig(max_updates=1,phase='local',affected_depths=trial.affected_depths,mixture_weights={'natural':.5,'student':.25,'donor':.25},kl_floor_per_target=floor,numerical_profile=EXECUTION_PROFILE)
            trial=fit_trial(trial,schedule,limits)
            sensitivity_after=natural_metrics(teacher,trial.model,fit_batches[:3],rec)
            shared_cost=_cost_delta(rec.aggregate(),before)
            score_start=time.monotonic();score_cost_before=rec.aggregate()
            natural=natural_metrics(teacher,trial.model,main_batches,rec);activations=activation_scores(teacher,trial,main_batches,rec)
            probes,probe_rows=probe_observations(teacher,trial,pools['probe_score'],rec,calibration)
            scores={k:{} for k in p['methods']}
            weight=next(r['weight_distance'] for r in design['population'] if tuple(r['pair'])==pair)
            for row in natural:
                component=row['component'];observed=probes[component]
                values=fixed_scores(natural=row['kl_per_target'],student=_mean(observed['student']),random_donor=_mean(observed['random']),active_donor=_mean(observed['active']))
                values.update(weight_distance=weight,**activations[component])
                for method,value in values.items():scores[method][component]=value
            score_elapsed=time.monotonic()-score_start;score_cost_after=rec.aggregate()
            queries=score_cost_after.teacher_forward_calls-score_cost_before.teacher_forward_calls
            secondary=natural_metrics(teacher,trial.model,natural_batches['natural_score'],rec)
            after=rec.aggregate()
            record=dict(identity=asdict(TrialIdentity(pair,trial.initialized_digest,trial.fitted_digest)),medoid_scores=trial.medoid_scores,selected_parent=trial.selected_parent,fit=trial.fit_summary,
                sensitivity={'before':sensitivity_before,'after':sensitivity_after},scores=scores,score_manifest=score_manifest,costs={k:(queries,score_elapsed) for k in scores},
                secondary_scores={r['component']:r['kl_per_target'] for r in secondary},
                natural_score_rows=natural,secondary_rows=secondary,probe_score_rows=probe_rows,
                filtered_probe_summary={c:v['filtered_active'] for c,v in probes.items()},unfiltered_primary=True,source_sha256=sources,plan_sha256=plan_sha,
                trial_wall_seconds=time.monotonic()-trial_started,cost_before=before.to_dict(),cost_after=after.to_dict(),shared_construction_through_fit=shared_cost,
                necessary_pool_rows={'random_donor':len(main_batches)*len(pair),'active_donor':sum(r['source']=='donor' for r in probe_rows),'student':sum(r['source']=='student' for r in probe_rows)},
                accounting='All methods receive shared generated pool/query results. Costs above are equal-pool comparison costs; efficient random-only acquisition requires fewer queries and is not claimed measured here.')
            if profile:
                save_trial_delta(trial,current,OUT/'profile-delta')
                record['damage_status']='sealed_pending_full_phase_protocol_freeze'
            else:
                if list(pair) in full_policy.get('sensitivity_pairs',[]):save_trial_delta(trial,current,OUT/f'sensitivity-delta-{pair[0]:02d}-{pair[1]:02d}')
                _finish_damage(record,teacher,trial,natural_batches['natural_damage'],pools['probe_damage'],rec,calibration)
            _write(path,record);append_event(attempt,dict(kind='trial_complete',pair=pair,path=str(path.relative_to(ROOT)),sha256=sha256_file(path),wall_seconds=record['trial_wall_seconds']))
            result['trials'].append(record);measured+=1
            print(json.dumps({'pair':pair,'trial_wall_seconds':record['trial_wall_seconds'],'fit':trial.fit_summary,'fit_before':_mean(x['kl_per_target'] for x in sensitivity_before),'fit_after':_mean(x['kl_per_target'] for x in sensitivity_after)}),flush=True)
            del trial;gc.collect();torch.mps.empty_cache()
            if profile and measured>=1:break
        result['status']='profile_complete' if profile else 'complete'
        if not profile:
            result['summary']=summarize(result['trials'],design)
            original=json.loads((OUT/'pairs.json').read_text());original_pairs={tuple(x) for x in original['sample']}
            result['summary']['secondary_original18']=summarize([r for r in result['trials'] if tuple(r['identity']['pair']) in original_pairs],original)
        else:
            result['profile_summary']={'damage_status':'not predicted; awaiting full-phase protocol freeze','controls':run_controls(),'initializer_control':initializer_sensitivity_control(),
                'trial_wall_seconds':result['trials'][0]['trial_wall_seconds'],'fit_sensitivity':result['trials'][0]['sensitivity'],'fit':result['trials'][0]['fit'],
                'score_queries':result['trials'][0]['costs'],'completed_source_sha256':sources}
        result['completed_pairs']=len(result['trials']);result['planned_pairs']=len(design['sample'])
    except Exception as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure_reason=str(exc))
        append_event(attempt,dict(kind='failed',failure_type=type(exc).__name__,reason=str(exc)))
    finally:
        rec.finish_attempt(result,elapsed=time.monotonic()-started)
        # Compact public result refers to immutable per-trial raw artifacts.
        result['trial_paths']=[str((OUT/('trials' if profile else 'trials-final')/f"{r['identity']['pair'][0]:02d}-{r['identity']['pair'][1]:02d}.json").relative_to(ROOT)) for r in result.pop('trials',[])]
        _write(attempt.path/'result.json',result);_write(OUT/'latest.json',{'attempt':result['attempt'],'result_sha256':sha256_file(attempt.path/'result.json'),'status':result['status']})
        print(json.dumps({'status':result['status'],'attempt':result['attempt'],'cost':result['aggregate_phase_cost'],'failure':result.get('failure_reason')}),flush=True)
    return result

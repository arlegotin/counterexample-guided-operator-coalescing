"""Durable execution boundaries for the new dense baseline campaign.

Model phases require a live finite supervisor plus source/backend/serial
admissions. Importing this module does not initialize a native backend.
"""
from pathlib import Path
import hashlib,json,math,os,subprocess,sys,time
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
from poc.dense_frontier import dense_contract_digest
from poc.recovery_endpoints import file_sha,finite_json,verify_refs,write_once
from poc.records import canonical_json
from poc.contracts import Budget,CostRecord,RunSpec
from poc.budget import BudgetTracker,BudgetExceeded
from poc.train import finish_attempt_cost
from poc.continuation_budget import process_peak_bytes
from provenance import recovery_diagnosis as common,sharing_frontier_study as core


def reference(path):return {'path':str(Path(path).resolve()),'sha256':file_sha(path)}


def record_dense_phase(c,phase,sources,operation,*,resource,work,declared_inputs=None):
    """Attempt/ledger precede setup; updates and raw work are distinct units."""
    fields={'updates','valid_target_presentations','teacher_forward_calls','student_forward_calls',
            'student_backward_calls','raw_corpus_target_presentations'}
    if (type(work) is not dict or set(work)!=fields or any(type(v) is not int or v<0 for v in work.values())
            or type(resource) is not dict or set(resource)!={'wall_seconds','memory_bytes','disk_bytes'}):
        raise ValueError('closed finite phase work/resource allowance required')
    budget=Budget(work['updates'],work['valid_target_presentations'],work['teacher_forward_calls'],0,
                  work['student_backward_calls'],0,0,resource['wall_seconds'],resource['disk_bytes'],resource['memory_bytes'])
    directory=Path(c['out'])/'phases'/phase
    if directory.exists():raise FileExistsError('phase already attempted; preserve all outcomes')
    directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source_hash=hashlib.sha256(canonical_json({'files':sources})).hexdigest()
    identity=RunSpec('Q',c['study']+':'+phase,
        {'scientific_contract_sha256':dense_contract_digest(c),'execution_contract':c,'phase':phase,'source_files':sources,'work_allowance':work,**({'declared_inputs':declared_inputs} if declared_inputs is not None else {})},
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hash,
        common.hashed(common.environment()),{'study_contract':dense_contract_digest(c),'teacher':c['reference_artifacts']['teacher_noop']['manifest_sha256']},
        {'device':'MLX FP32; actual packed offline evaluation separately qualified'},{'seed':17},(),budget)
    from poc.records import start_attempt
    attempt=start_attempt(identity,directory/'records')
    class DenseTracker(BudgetTracker):
        def disk(self):return sum(p.stat().st_size for p in Path(c['out']).rglob('*') if p.is_file())
        def observed(self,cost):
            peaks=[cost.memory_bytes or 0,process_peak_bytes()]
            if 'mlx.core' in sys.modules:peaks.append(int(sys.modules['mlx.core'].get_peak_memory()))
            return replace(cost,memory_bytes=max(peaks),memory_unavailable_reason=None,disk_bytes=max(cost.disk_bytes,self.disk()))
        def check(self,cost):
            for key in ('student_forward_calls','raw_corpus_target_presentations'):
                if getattr(self.totals,key)+getattr(cost,key)>work[key]:raise BudgetExceeded('finite phase '+key+' work cap exceeded')
            super().check(self.observed(cost))
        def record(self,cost):
            # Retain actual spending before raising, including an atomic overrun.
            try:super().record(self.observed(cost))
            finally:
                for key in ('student_forward_calls','raw_corpus_target_presentations'):
                    if getattr(self.totals,key)>work[key]:raise BudgetExceeded('actual phase '+key+' work cap exceeded')
    rec=DenseTracker(budget,attempt)
    try:
        rec.record(CostRecord());value=operation(rec)
        write_once(directory/'packet.json',{'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),
            'execution_contract_sha256':common.hashed(c),'phase':phase,'run_id':attempt.run_id,'value':value})
        finish_attempt_cost(rec,elapsed=time.monotonic()-started);totals=rec.validate()
        write_once(directory/'terminal-cost.json',{'schema_version':1,'run_id':attempt.run_id,
            'scientific_contract_sha256':dense_contract_digest(c),'execution_contract_sha256':common.hashed(c),
            'ledger':reference(rec.path),'totals':totals.to_dict(),'resource':resource,'work_allowance':work,
            'scope':'common durable phase ledger; supervisor additionally covers closing receipt bookkeeping'})
        common.finish_attempt(attempt,'exhausted',reason='fixed finite phase completed');return value
    except BaseException as exc:
        closing=None
        try:finish_attempt_cost(rec,elapsed=time.monotonic()-started)
        except BaseException as close_exc:closing=repr(close_exc)
        common.finish_attempt(attempt,'failed',reason=repr(exc))
        write_once(directory/'failed.json',{'status':'failed','phase':phase,'error':repr(exc),'closing_error':closing,
            'scientific_contract_sha256':dense_contract_digest(c),'run_id':attempt.run_id,'ledger':reference(rec.path),
            'scope':'all dispatched work and partial artifacts retained; no automatic scientific retry'})
        raise


def require_dense_phase_guard(c,phase,config_path):
    """Reject direct entry before any model/backend or fit-content operation."""
    stage=Path(c['out'])/'stages'/phase;pp=stage/'protocol.json';sp=stage/'stage-start.json'
    if config_path is None or not pp.is_file() or not sp.is_file():raise ValueError('model phase requires live finite source/serial guard')
    from provenance.shared_calibration_study import verify_live_parent
    verify_live_parent(pp);p=finite_json(pp);start=finite_json(sp)
    command=[sys.executable,str(ROOT/'provenance/dense_frontier_campaign.py'),'--config',str(Path(config_path).resolve()),'--internal-phase',phase]
    if start.get('pid')!=os.getppid() or start.get('protocol_sha256')!=file_sha(pp) or p.get('command')!=command or p.get('root')!=str(ROOT) or p.get('output')!=c['out']:
        raise ValueError('live phase guard command/parent/source binding differs')
    verify_refs(p['frozen_files']);bound={r['path']:r['sha256'] for r in p['frozen_files']}
    config=reference(config_path);plan_path=Path(c['out'])/'phase-plans'/(phase+'.json');plan_ref=reference(plan_path)
    if any(bound.get(r['path'])!=r['sha256'] for r in (config,plan_ref)):raise ValueError('guard omits exact config/resource plan')
    plan=finite_json(plan_path)
    if plan.get('scientific_contract_sha256')!=dense_contract_digest(c) or plan.get('phase')!=phase or plan.get('resource')!=p.get('caps'):
        raise ValueError('phase guard changes scientific identity or frozen resource cap')
    # Qualification and serial issuers are fixed to this study. Their complete
    # verification is part of the campaign admission before this phase plan.
    for key,name,status in (('qualification','qualification-v1.json','qualified_dense_source_cpu'),
                            ('serial_admission','serial-admission-v1.json','accepted_serial_slot_after_cold_family')):
        expected=ROOT/'provenance'/c['study']/name;r=reference(expected)
        if plan.get(key)!=r or bound.get(r['path'])!=r['sha256'] or finite_json(expected).get('status')!=status:
            raise ValueError('source/backend/serial admission missing from guard')
    native=ROOT/'provenance'/c['study']/'native-qualification-v1.json';nr=reference(native)
    if plan.get('native_qualification')!=nr or bound.get(nr['path'])!=nr['sha256'] or finite_json(native).get('status')!='qualified_dense_native_controls':
        raise ValueError('actual new accounting/backend controls required')
    qref=plan['qualification'];q=read_reference(qref)
    required=require_dense_qualification(c,proposal_path=q['proposal']['path'],qualification=qref)
    required+=require_dense_serial_admission(c,admission=plan['serial_admission'])
    required+=require_dense_native_qualification(c,qualification=nr,source_qualification=qref)
    required+=preparation_input_refs(c)
    from poc.dense_frontier import dense_recipes
    profile=phase.startswith('preprofile-');prefix='preprofile-' if profile else 'prepare-'
    recipe=next((r for r in dense_recipes() if phase==prefix+r.label),None)
    if recipe is None:raise ValueError('unknown live preparation recipe')
    if profile:
        expected_caps={'wall_seconds':7200,'memory_bytes':c['preparation_limits']['memory_bytes'],'disk_bytes':c['preparation_limits']['family_disk_bytes']}
    else:
        admitted=admit_preparation_freeze(c,recipe,config_path=config_path,required_refs=required,admissions={k:plan[k] for k in ('qualification','serial_admission','native_qualification')})
        expected_caps=admitted['resource'];required+=admitted['required_refs']
    required +=[config,plan_ref]
    validate_live_dense_guard(p,command=command,root=ROOT,output=Path(c['out']),provenance=stage,caps=expected_caps,required_refs=required,execution_env=dense_execution_env())
    return plan,required


def prepare_dense_artifact(c,label,*,config_path,profile,sources):
    phase=('preprofile-' if profile else 'prepare-')+label
    plan,verified_refs=require_dense_phase_guard(c,phase,config_path)
    sources={**sources,**{r['path']:r['sha256'] for r in verified_refs}}
    # The actual artifact operation is enabled only with the fully bound recipe
    # packet; it is kept separate from metadata and common ledger primitives.
    from poc.dense_frontier import dense_recipes,dense_preparation_plan
    recipe=next((r for r in dense_recipes() if r.label==label),None)
    if recipe is None or plan.get('preparation')!=dense_preparation_plan(recipe,profile=profile):
        raise ValueError('exact completed dense preparation recipe required')
    return _prepare_bound_artifact(c,recipe,plan,profile=profile,sources=sources)


def selected_dense_population(c,recipe,source,*,profile,phase,evaluation_components):
    """Select roles/components before reading fit content; never replace failures."""
    from dataclasses import asdict
    from poc.data.manifest import load_manifest
    from poc.data.access import open_records
    from poc.dense_frontier import calibration_plan,dense_preparation_plan
    from tokenizers import Tokenizer
    if c.get('schema_version')==2:
        from poc.dense_frontier import verify_dense_parent_derivation
        original,allocation=verify_dense_parent_derivation(c,evaluation_components=evaluation_components)
    else:
        original=load_manifest(Path(c['fit_manifest']['path']),expected_sha256=c['fit_manifest']['sha256'],roles={'fit'})
        allocation=calibration_plan(original,evaluation_components=evaluation_components)
    plan=dense_preparation_plan(recipe,profile=profile)
    train=allocation['train_document_ids'][:plan['train_components']]
    internal=allocation['internal_document_ids'][:plan['internal_components']]
    ids=train+internal;by_id={r.document_id:r for r in original.records}
    directory=Path(c['out'])/'calibration'/phase;directory.mkdir(parents=True,exist_ok=False)
    selected=replace(original,records=tuple(by_id[d] for d in ids),roles={d:'fit' for d in ids},
        membership={d:original.membership[d] for d in ids},quarantine={},
        lineage_sha256=original.digest(),exposure_path=str(directory/'exposure.jsonl'))
    manifest=directory/'manifest.json';write_once(manifest,asdict(selected))
    tokenizer_path=Path(source)/'tokenizer/tokenizer.json';tokenizer=Tokenizer.from_file(str(tokenizer_path));windows=[]
    for record in open_records(selected,purpose='calibration'):
        tokens=tokenizer.encode(record.text,add_special_tokens=False).ids[:512]
        if len(tokens)!=512:raise ValueError('prespecified calibration component lacks complete512context; no replacement')
        body={'tokens':tokens,'positions':list(range(512)),'attention_mask':[True]*512,'target_mask':[True]*511}
        windows.append({'document_id':record.document_id,'component_id':selected.membership[record.document_id],
            'tokens':tokens,'input_tokens':512,'valid_targets':511,'window_sha256':common.hashed(body)})
    if [w['document_id'] for w in windows]!=ids:raise ValueError('role-bound preparation order differs from prospective plan')
    proof={'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'phase':phase,'profile':profile,
        'population':reference(manifest),'lineage_sha256':original.digest(),'allocation':allocation,
        'train_document_ids':train,'internal_document_ids':internal,'windows':windows,
        'raw_target_inventory':len(windows)*511,'unique_components':len(windows),'tokenizer':reference(tokenizer_path),
        'scope':'fit-only eligibility read plus separately journaled quantizer calibration read; prior exposure retained; no protected text'}
    write_once(directory/'window-proof.json',proof)
    return selected,proof


def _prepare_bound_artifact(c,recipe,plan,*,profile,sources):
    """Actual original algorithm, common packed export and fresh offline reload."""
    from poc.dense_frontier import dense_quantization_spec,validate_dense_quantizer_evidence
    from poc.contracts import BundleRef
    phase=plan['phase'];out=Path(c['out']);target=out/'artifacts'/('preprofile' if profile else 'final')/recipe.label
    def operation(rec):
        from contextlib import redirect_stdout,redirect_stderr
        from poc.quantize import quantize_dense
        from poc.train import _call
        from provenance.shared_gptq_study import inspect_bundle_files,offline_proof_module
        import torch,mlx.core as mx
        torch.set_num_threads(1);mx.set_default_device(mx.gpu)
        if not mx.metal.is_available():raise RuntimeError('actual qualified Metal backend required')
        source=c['reference_artifacts']['teacher_noop'];source_metadata,_=inspect_bundle_files(source,expected_kind='dense')
        verify_refs([{k:c['inputs'][k] for k in ('path','sha256')}]);inputs=finite_json(c['inputs']['path'])
        if inputs.get('population_manifest_sha256')!=c['inputs']['population_manifest_sha256']:
            raise ValueError('evaluation population identity differs')
        windows=inputs['windows'];core.metric_work_plan(windows,mode='run')
        population,proof=selected_dense_population(c,recipe,source['directory'],profile=profile,phase=phase,
            evaluation_components={w['component_id'] for w in windows})
        spec=dense_quantization_spec(recipe,profile=profile,destination=str(target))
        log=out/'phases'/phase/'quantizer-output.log'
        # Original algorithms may print fit/internal losses. Preserve their exact
        # output without disclosing it through the outcome-blind resource packet.
        with log.open('x') as stream,redirect_stdout(stream),redirect_stderr(stream),core.retain_packing_failures(str(target)):
            result=quantize_dense(BundleRef(**source),spec,population,rec,accounting_profile='dense-native-contexts-v2')
        mx.synchronize();evidence_path=Path(str(target)+'.quantizer-evidence.json');evidence=finite_json(evidence_path)
        component_proof={key:[population.membership[d] for d in proof[ids]] for key,ids in
            (('train_component_ids','train_document_ids'),('internal_component_ids','internal_document_ids'))}
        validate_dense_quantizer_evidence(recipe,plan['preparation'],component_proof,evidence)
        metadata,complete=inspect_bundle_files({'directory':result.bundle.directory,'manifest_sha256':result.bundle.manifest_sha256},expected_kind='dense')
        runtime=file_sha(target/'runtime/poc-runtime.zip')
        if runtime!=file_sha(Path(source['directory'])/'runtime/poc-runtime.zip') or metadata['execution_profile']!=source_metadata['execution_profile']:
            raise ValueError('packed baseline changed the exact matched reference runtime/profile')
        proof_directory=out/'offline-proofs'/phase;proof_directory.mkdir(parents=True,exist_ok=False)
        offline=offline_proof_module()
        reloaded=_call(rec,lambda:offline.run_offline_bundle(result.bundle.directory,proof_directory,backend='mlx',
            blocked_source_dirs=(str(ROOT),source['directory'])),student_forward_calls=8,processed_inputs=4)
        rec.record(CostRecord(processed_tokens=len(reloaded['initial_ids'])+3))
        if evidence['total_model_context_raw_presentations']!=rec.totals.raw_corpus_target_presentations:
            raise ValueError('durable adapter raw exposure differs from actual quantizer evidence')
        return {'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'profile':profile,
            'recipe':recipe.identity(),'preparation_plan':plan['preparation'],'directory':result.bundle.directory,
            'manifest_sha256':result.bundle.manifest_sha256,'complete_bytes':complete,'runtime_sha256':runtime,
            'execution_profile':metadata['execution_profile'],'owners':len(metadata['owners']),'teacher_needed_at_inference':False,
            'quantizer_evidence':reference(evidence_path),'window_proof':reference(out/'calibration'/phase/'window-proof.json'),
            'algorithm_output':reference(log),'algorithm_output_policy':'fit-only printed losses retained unopened until all resource budgets frozen',
            'offline_proof':reloaded,'candidate_quality_disclosed':False}
    return record_dense_phase(c,phase,sources,operation,resource=plan['resource'],work=plan['preparation']['work'])


def dense_executed_sources(proposal_path):
    """Exact current code/config/installed-original bytes, without native import."""
    from provenance.shared_calibration_study import execution_sources
    from poc.dense_frontier import installed_algorithm_sources
    files=execution_sources(proposal_path)
    # Generated after profiling. Its exact full bytes are independently bound in
    # every execution protocol and immutable common RunSpec, never omitted there.
    files.pop(str(ROOT/'configs/dense-baseline-expansion-v2-run-v1.yaml'),None)
    for item in installed_algorithm_sources():files[item['path']]=item['sha256']
    extras=[ROOT/'licenses/mlx-lm-MIT.txt',ROOT/'configs/confirmation-access-implementation-v1.json',
        ROOT/'configs/continuation-integration-implementation-v1.json',
        ROOT/'provenance/continuation-integration-v1/import-v1.json',
        ROOT/'provenance/continuation-integration-v1/owned-selector-source-binding-v2.json']
    extras.append(ROOT/'provenance/dense-baseline-expansion-v2/serial-requirement-v1.json')
    extras.append(ROOT/'provenance/dense-baseline-expansion-v2/fit-parent-derivation-v1.json')
    extras.append(ROOT/'provenance/dense-baseline-expansion-v2/ancestor-cost-inventory-v1.json')
    extras+=list((ROOT/'provenance/dense-baseline-expansion-v2').glob('*.py.txt'))
    for path in extras:files[str(path)]=file_sha(path)
    return dict(sorted(files.items()))


def dense_qualification_specs(work):
    focused=['tests/test_dense_admission.py','tests/test_dense_cost_ancestry.py','tests/test_dense_campaign.py','tests/test_dense_frontier.py','tests/test_dense_dispatch_accounting.py','tests/test_quantization.py',
             'tests/test_data_access.py','tests/test_export.py','tests/test_offline_bundle.py','tests/test_budget.py']
    return [('focused_cpu',['-m','pytest',*focused,'-q'],{'wall_seconds':300,'memory_bytes':8*1024**3,'disk_bytes':2*1024**3}),
            ('default_cpu',['-m','pytest','-q'],{'wall_seconds':1800,'memory_bytes':12*1024**3,'disk_bytes':8*1024**3}),
            ('public',['provenance/verify_public_continuation.py'],{'wall_seconds':120,'memory_bytes':1024**3,'disk_bytes':1024**2})]


def dense_execution_env():return {'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'}


def require_dense_qualification(c,*,proposal_path,qualification):
    """Verify source/archive/review/environment and exact successful CPU stages."""
    import tarfile,platform,yaml
    verify_refs([qualification]);q=finite_json(qualification['path'])
    keys={'schema_version','status','study','scientific_contract_sha256','proposal','source_manifest','source_archive',
          'review','environment','operation','work_root','stages'}
    if (type(q) is not dict or set(q)!=keys or q['schema_version']!=1 or type(q['schema_version']) is not int
            or q['status']!='qualified_dense_source_cpu' or q['study']!=c['study']
            or q['scientific_contract_sha256']!=dense_contract_digest(c)):
        raise ValueError('complete source-bound dense CPU qualification required')
    verify_refs([q[k] for k in ('proposal','source_manifest','source_archive','review','environment','operation')])
    if q['proposal']!=reference(proposal_path):raise ValueError('qualification proposal path/content differs')
    proposal=yaml.safe_load(Path(proposal_path).read_text())
    if proposal.get('resource_freeze') is not None or dense_contract_digest(proposal)!=dense_contract_digest(c):
        raise ValueError('scientific contract changed after outcome-blind profiling')
    files=dense_executed_sources(proposal_path);source=finite_json(q['source_manifest']['path'])
    if source.get('files')!=files:raise ValueError('qualification executed source inventory changed or incomplete')
    verify_refs([{'path':p,'sha256':h} for p,h in files.items()])
    review=finite_json(q['review']['path']);env=finite_json(q['environment']['path'])
    if review.get('status')!='resolved' or review.get('unresolved_critical')!=0 or review.get('unresolved_important')!=0 or review.get('source_files')!=files:
        raise ValueError('resolved exact-source whole-feature review required')
    if env.get('python')!=sys.version or env.get('executable')!=sys.executable or env.get('platform')!=platform.platform():
        raise ValueError('qualified execution environment differs')
    expected={str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else 'installed-source/'+str(Path(p)).lstrip('/'):h for p,h in files.items()}
    if q['operation']['path'] not in files:expected['operations/'+Path(q['operation']['path']).name]=q['operation']['sha256']
    with tarfile.open(q['source_archive']['path'],'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise ValueError('qualified source archive inventory differs')
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=expected[member.name]:
                raise ValueError('qualified source archive bytes differ')
    work=Path(q['work_root'])
    if not work.is_absolute() or work.resolve()!=work or work.is_relative_to(ROOT):raise ValueError('external canonical qualification scratch required')
    specs=dense_qualification_specs(work)
    if type(q['stages']) is not dict or set(q['stages'])!={s[0] for s in specs}:raise ValueError('all CPU/public qualification stages required')
    refs=[{'path':p,'sha256':h} for p,h in files.items()]+[q[k] for k in ('proposal','source_manifest','source_archive','review','environment','operation')]
    for name,args,caps in specs:
        item=q['stages'][name];output=work/name;command=[sys.executable,*args]
        if name!='public':command.append('--basetemp='+str(output/'pytest'))
        if item.get('command')!=command or item.get('output')!=str(output) or item.get('caps')!=caps:
            raise ValueError('qualification command/output/caps differ')
        core.check_stage_binding(item['stage'],command=command,root=ROOT,output=output,caps=caps,required_refs=refs,execution_env=dense_execution_env())
    return refs+[qualification]


def require_dense_native_qualification(c,*,qualification,source_qualification):
    """A CPU receipt never stands for actual new-context MLX adapter controls."""
    import re
    verify_refs([qualification]);q=finite_json(qualification['path'])
    keys={'schema_version','status','study','scientific_contract_sha256','source_qualification','serial_admission',
          'command','output','caps','stage','new_owned_cases'}
    if (type(q) is not dict or set(q)!=keys or type(q.get('schema_version')) is not int or q['schema_version']!=1
            or q['status']!='qualified_dense_native_controls' or q['study']!=c['study']
            or q['scientific_contract_sha256']!=dense_contract_digest(c) or q['source_qualification']!=source_qualification
            or type(q['new_owned_cases']) is not int or q['new_owned_cases']!=6):
        raise ValueError('complete actual source-bound native qualification required')
    verify_refs([source_qualification,q['serial_admission']]);source=finite_json(source_qualification['path'])
    refs=require_dense_qualification(c,proposal_path=source['proposal']['path'],qualification=source_qualification)
    output=Path(q['output'])
    if not output.is_absolute() or output.resolve()!=output or output.is_relative_to(ROOT):raise ValueError('external native qualification scratch required')
    command=[sys.executable,'-m','pytest','tests/test_dense_native_controls.py','tests/test_quantization.py','-q','-m','mlx','--basetemp='+str(output/'pytest')]
    caps={'wall_seconds':600,'memory_bytes':8*1024**3,'disk_bytes':2*1024**3}
    if q['command']!=command or q['caps']!=caps:raise ValueError('actual new17case native command/caps differ')
    refs.extend(require_dense_serial_admission(c,admission=q['serial_admission']))
    core.check_stage_binding(q['stage'],command=command,root=ROOT,output=output,caps=caps,required_refs=refs,execution_env=dense_execution_env())
    log=Path(q['stage']['protocol_path']).parent/'execution.log'
    text=log.read_text()
    if re.search(r'\b17 passed\b',text) is None or re.search(r'\b[1-9][0-9]* (failed|error|errors)\b',text):
        raise ValueError('actual native qualification did not pass all17eligible cases')
    return refs+[qualification]


def read_reference(ref):
    verify_refs([ref]);return finite_json(ref['path'])


def require_imported_index_binding(admitted,accepted,*,index_ref,prepared_ref,parent_config):
    """Current raw paths/hashes cannot replace the originally guarded index."""
    import yaml
    if admitted.get('status')!='accepted' or accepted.get('status')!='accepted' or 'stage' not in accepted or 'analysis_acceptance' not in admitted:
        raise ValueError('independent reference admission and accepted analysis guard required')
    if read_reference(admitted['analysis_acceptance'])!=accepted:raise ValueError('accepted reference analysis bytes differ')
    verify_refs([parent_config]);parent=yaml.safe_load(Path(parent_config['path']).read_text())
    if type(parent) is not dict or type(parent.get('resource')) is not dict or type(parent.get('out')) is not str:
        raise ValueError('reference analysis execution configuration required')
    root=Path(parent_config['path']).parent.parent;out=Path(parent['out']);resource=parent['resource']
    if index_ref['path']!=str(out/'evaluation-index.json') or prepared_ref['path']!=str(out/'preparation.json'):
        raise ValueError('reference raw index/preparation paths differ')
    try:caps={'wall_seconds':resource['preparation_wall_seconds'],'memory_bytes':resource['phase_memory_bytes'],'disk_bytes':resource['family_disk_bytes']}
    except KeyError as exc:raise ValueError('reference analysis finite resource caps missing') from exc
    command=[sys.executable,str(root/'provenance/shared_gptq_study.py'),'--config',parent_config['path'],'--internal-phase','analysis']
    refs=[parent_config,index_ref,prepared_ref];env={'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'}
    core.check_stage_binding(accepted['stage'],command=command,root=root,output=out,caps=caps,required_refs=refs,execution_env=env)
    p=read_reference({'path':accepted['stage']['protocol_path'],'sha256':accepted['stage']['protocol_sha256']})
    verify_refs(p['frozen_files']);verify_refs(refs)
    return p['frozen_files']+[admitted['analysis_acceptance']]


def import_dense_references(c,*,config_path):
    """Re-admit12 old observations without a model, corpus read or relabeling."""
    import yaml
    from provenance.dense_frontier_study import metadata_validate
    from provenance.shared_gptq_study import inspect_bundle_files
    from provenance.shared_calibration_study import read_terminal_cost
    from poc.dense_frontier import REFERENCE_LABELS
    metadata_validate(config_path)
    if dense_contract_digest(yaml.safe_load(Path(config_path).read_text()))!=dense_contract_digest(c):
        raise ValueError('reference importer configuration differs from new scientific contract')
    acceptance=read_reference(c['reference_acceptance']);admitted=read_reference(acceptance['result'])
    parent_ref=c['reference_config'];verify_refs([parent_ref]);parent=yaml.safe_load(Path(parent_ref['path']).read_text());out=Path(parent['out'])
    accepted=read_reference(admitted['analysis_acceptance']);analysis=read_reference(admitted['analysis'])
    if (admitted.get('status')!='accepted' or admitted.get('configuration')!=parent_ref
            or accepted.get('analysis')!=admitted['analysis'] or analysis.get('study_contract_sha256')!=common.hashed(parent)):
        raise ValueError('independent reference analysis/configuration admission differs')
    index_ref=reference(out/'evaluation-index.json');prepared_ref=reference(out/'preparation.json')
    proof=require_imported_index_binding(admitted,accepted,index_ref=index_ref,prepared_ref=prepared_ref,parent_config=parent_ref)
    index=read_reference(index_ref);prepared=read_reference(prepared_ref);frozen=finite_json(out/'evaluation-freeze.json')
    verify_refs([{k:c['inputs'][k] for k in ('path','sha256')}]);compiled=finite_json(c['inputs']['path']);windows=compiled['windows']
    core.metric_work_plan(windows,mode='run')
    if (set(index)!=set(REFERENCE_LABELS) or set(prepared.get('artifacts',{}))!=set(REFERENCE_LABELS)
            or prepared.get('study_contract_sha256')!=common.hashed(parent) or prepared.get('inputs')!=c['inputs']
            or compiled.get('population_manifest_sha256')!=c['inputs']['population_manifest_sha256']
            or prepared.get('teacher_mps_digest')!=parent['reference']['teacher_mps_digest']):
        raise ValueError('complete reference grid/population/teacher binding differs')
    measurements={};protocols={};artifacts={};environments=set()
    for label in REFERENCE_LABELS:
        item=index[label];artifact=prepared['artifacts'][label];expected=c['reference_artifacts'][label]
        if ({k:artifact[k] for k in ('directory','manifest_sha256')}!=expected or artifact.get('teacher_needed_at_inference') is not False):
            raise ValueError('original reference artifact identity or inference dependency differs')
        manifest,size=inspect_bundle_files(expected)
        if size!=artifact['complete_bytes'] or size!=admitted['artifacts'][label]['complete_bytes']:
            raise ValueError('actual retained reference byte census changed')
        if file_sha(Path(expected['directory'])/'runtime/poc-runtime.zip')!=artifact['runtime_sha256']:
            raise ValueError('reference embedded runtime changed')
        protocol=read_reference(item['protocol']);value=read_reference(item['output'])
        teacher=prepared['artifacts']['teacher_noop']
        if (protocol.get('study')!=parent['study'] or protocol.get('study_contract_sha256')!=common.hashed(parent)
                or protocol.get('inputs')!=c['inputs'] or protocol.get('worker_sha256')!=c['worker']['sha256']
                or protocol.get('runtime_sha256')!=artifact['runtime_sha256'] or protocol.get('mode')!='run'
                or protocol.get('artifact')!={'path':artifact['directory'],'manifest_sha256':artifact['manifest_sha256']}
                or protocol.get('teacher')!={'path':teacher['directory'],'manifest_sha256':teacher['manifest_sha256'],'mps_checkpoint_digest':prepared['teacher_mps_digest']}):
            raise ValueError('original native observation protocol/runtime/input/teacher identity changed')
        core.check_native_result(protocol,value,rows=windows)
        case=out/'workers'/('run-'+label);worker=case/'native-worker.py'
        if file_sha(worker)!=c['worker']['sha256'] or read_reference(frozen['jobs'][label]['protocol'])!=protocol:
            raise ValueError('original executed worker or frozen completed native plan changed')
        required=[item['protocol'],reference(worker),prepared_ref,protocol['profile_binding'],frozen['jobs'][label]['protocol']]
        required +=[{'path':str(Path(protocol[k]['path'])/'manifest.json'),'sha256':protocol[k]['manifest_sha256']} for k in ('teacher','artifact')]
        caps={'wall_seconds':protocol['resource']['wall_seconds']+30,'memory_bytes':parent['resource']['phase_memory_bytes'],'disk_bytes':parent['resource']['family_disk_bytes']}
        core.check_stage_binding(item['stage'],command=[sys.executable,str(worker),'--protocol',item['protocol']['path']],
            root=Path('/private/tmp'),output=out,caps=caps,required_refs=required,execution_env={'PYTHONPATH':''})
        guard=read_reference({'path':item['stage']['protocol_path'],'sha256':item['stage']['protocol_sha256']});verify_refs(guard['frozen_files'])
        prior=admitted['phase_costs']['run-'+label]
        if item['terminal_cost']!=prior['terminal']:raise ValueError('reference original phase cost handoff changed')
        packet=read_reference(item['common_packet']);terminal=read_terminal_cost(item['terminal_cost'],expected_run_id=packet['run_id'])
        if terminal['ledger']!=prior['ledger'] or terminal['totals']!=prior['totals']:
            raise ValueError('reference terminal ledger or cumulative expense changed')
        n=sum(r['valid_targets'] for r in value['rows'])
        for metric in ('kl','nll'):
            observed=math.fsum(r[metric+'_sum'] for r in value['rows'])/n
            if abs(observed-admitted['artifacts'][label][metric])>1e-13:raise ValueError('reference raw metric sums differ from independent prior admission')
        environments.add(value['environment_sha256']);measurements[label]={'status':'complete',**value}
        protocols[label]=protocol;artifacts[label]=artifact
        proof +=[item[k] for k in ('protocol','output','common_packet','terminal_cost')]+[reference(case/'acceptance.json')]
    if len({a['runtime_sha256'] for a in artifacts.values()})!=1 or len(environments)!=1:
        raise ValueError('imported references lack one common qualified runtime/environment')
    # All original setup, failed/retained preparation, profiles and evaluation
    # costs stay visible, including work shared by multiple imported points.
    for phase,item in admitted['phase_costs'].items():
        terminal_header=read_reference(item['terminal']);terminal=read_terminal_cost(item['terminal'],expected_run_id=terminal_header['run_id'])
        if terminal['ledger']!=item['ledger'] or terminal['totals']!=item['totals']:
            raise ValueError('imported ancestor phase expense changed:'+phase)
        proof +=[item['terminal'],item['ledger']]
    from poc.cost_ancestry import read_cost_ancestry
    ancestry_ref=reference(ROOT/'provenance'/c['study']/'ancestor-cost-inventory-v1.json')
    ancestry=read_cost_ancestry(ancestry_ref,expected_parent=parent_ref)
    if parent['reference']['source_run_id'] not in {v.get('run_id') for v in ancestry['ledgers'].values()}:
        raise ValueError('named source recovery expense missing from admitted ancestor inventory')
    proof.append(ancestry_ref)
    return {'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'original_configuration':parent_ref,
        'original_independent_admission':c['reference_acceptance'],'original_independent_result':acceptance['result'],
        'prepared':prepared,'artifacts':artifacts,'protocols':protocols,'measurements':measurements,'windows':windows,
        'native_environment_sha256':next(iter(environments)),'phase_costs':admitted['phase_costs'],
        'aggregate_reference_campaign_cost':admitted['aggregate_common_cost'],'ancestor_costs':ancestry,'frozen_refs':proof,
        'observation_policy':'imported exact accepted raw observations; original study/protocol IDs and costs retained; no new inference',
        'cost_allocation':'original Mcommon phases and bounded calibration/recovery ancestors separately reported; shared ancestor ledger aliases counted once; unknown wider expense explicit; new import verification separately charged, no newly repeated neural calls',
        'scope':'already exposed development; no raw corpus/model/protected read; no capability or practical equivalence claim'}


def dense_serial_evidence(c):
    """Outcome-sign-blind release only after the preceding finite neural queue."""
    import yaml
    requirement=reference(ROOT/'provenance'/c['study']/'serial-requirement-v1.json');r=read_reference(requirement)
    keys={'schema_version','study','upstream_root','upstream_qualification','upstream_independent_admission',
          'upstream_proposal','upstream_worker','upstream_controller','outer_caps','policy','release_scope'}
    if set(r)!=keys or type(r['schema_version']) is not int or r['schema_version']!=1 or r['study']!=c['study']:
        raise ValueError('closed prospective serial requirement required')
    base=[r[k] for k in ('upstream_qualification','upstream_independent_admission','upstream_proposal','upstream_worker','upstream_controller')]
    verify_refs(base);q=read_reference(r['upstream_qualification']);prior=read_reference(r['upstream_independent_admission'])
    if q.get('status')!='qualified_cold_recovery1024_family_protocol' or q.get('proposal')!=r['upstream_proposal'] or prior.get('status')!='accepted_cold_family_source_and_metadata_only':
        raise ValueError('qualified upstream source/metadata serial dependency required')
    verify_refs(q['frozen_refs']);root=Path(r['upstream_root']);prov=root/'provenance/cold-recovery1024-family-v1'
    completion_ref=reference(prov/'family-complete-v1.json');completion=read_reference(completion_ref)
    worker_ref=reference(prov/'worker-complete-v1.json');worker=read_reference(worker_ref)
    if (completion.get('status')!='finite cold family common-driver complete; independent analysis pending'
            or completion.get('worker_completion')!=worker_ref or worker.get('status')!='common_driver_cold_family_complete; independent endpoints pending'
            or worker.get('proposal')!=r['upstream_proposal']):raise ValueError('cold neural family lacks completed serial handoff')
    family=read_reference(r['upstream_proposal']);upstream=completion['upstream_admission'];verify_refs([upstream,worker['freeze']])
    command=[sys.executable,r['upstream_worker']['path']]
    refs=q['frozen_refs']+[r['upstream_qualification']]
    core.check_stage_binding(completion['stage'],command=command,root=root,output=Path(family['output_family']['root']),
        caps=r['outer_caps'],required_refs=refs,execution_env={'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':'/private/tmp'})
    if set(worker.get('members',{}))!={m['label'] for m in family['members']}:raise ValueError('serial handoff omits a planned member')
    results=[];arms=0
    for member in family['members']:
        value=worker['members'][member['label']]
        verify_refs([value['configuration'],value['campaign'],*value['results'].values()]);core.require_stage_success(value['stage'])
        campaign=read_reference(value['campaign']);run_config=yaml.safe_load(Path(value['configuration']['path']).read_text())
        expected=cold_member_expected_arms(member,run_config)
        if set(value['results'])!=expected or set(campaign.get('arms',{}))!=expected or any(v.get('status')!='complete' for v in campaign['arms'].values()):
            raise ValueError('serial handoff did not finish all fixed member arms')
        for ref in value['results'].values():
            endpoint=read_reference(ref)
            if endpoint.get('step')!=1024 or endpoint.get('raw_target_cursor')!=8388608:
                raise ValueError('serial release requires exact completed horizons, regardless of quality')
            results.append(ref);arms+=1
    if arms!=5:raise ValueError('serial release requires all five predeclared trajectories')
    return {'schema_version':1,'status':'accepted_serial_slot_after_cold_family','study':c['study'],
        'requirement':requirement,'upstream_completion':completion_ref,'worker_completion':worker_ref,
        'member_results':results,'upstream_stage':completion['stage'],
        'scope':'serialization only; no new endpoint science, capability, compression, or counterexample claim'}


def require_dense_serial_admission(c,*,admission):
    value=read_reference(admission)
    keys={'schema_version','status','study','requirement','upstream_completion','worker_completion','member_results','upstream_stage','scope'}
    if type(value) is not dict or set(value)!=keys or value.get('status')!='accepted_serial_slot_after_cold_family':
        raise ValueError('complete verified serial release evidence required')
    if value!=dense_serial_evidence(c):raise ValueError('serial release producer or actual completion evidence changed')
    return [admission,value['requirement'],value['upstream_completion'],value['worker_completion'],*value['member_results']]


def cold_member_expected_arms(member,config):
    """Family proposal records bind configs; loss scopes live in those configs."""
    expected={'scale-mean17':(17,['local_mean']),'paired23':(23,['local_sum','full_student']),
              'paired31':(31,['local_sum','full_student'])}
    label=member.get('label')
    if label not in expected:raise ValueError('unknown prospective cold family member')
    seed,scopes=expected[label]
    if (type(member.get('seed')) is not int or member['seed']!=seed or config.get('seeds')!=[seed]
            or config.get('loss_scopes')!=scopes or type(config.get('updates')) is not int or config['updates']!=1024
            or type(config.get('raw_targets_per_update')) is not int or config['raw_targets_per_update']!=8192
            or config.get('owner_scope')!='affected_group' or config.get('alpha')!=0):
        raise ValueError('exact prospective cold seed/objective/owner/horizon contract differs')
    return {scope+'-seed'+str(seed) for scope in scopes}


def validate_live_dense_guard(protocol,*,command,root,output,provenance,caps,required_refs,execution_env):
    """Live analogue of successful-stage binding; no backend or content I/O."""
    if (protocol.get('command')!=command or protocol.get('root')!=str(Path(root)) or protocol.get('output')!=str(Path(output))
            or protocol.get('provenance')!=str(Path(provenance)) or protocol.get('caps')!=caps
            or protocol.get('supervisor_sha256')!='a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba'
            or protocol.get('execution_env')!={'HF_HUB_OFFLINE':None,'PYTORCH_ENABLE_MPS_FALLBACK':None,**execution_env}):
        raise ValueError('live preparation guard command/source/environment/measured caps differ')
    frozen=protocol.get('frozen_files')
    if type(frozen) is not list or any(type(r) is not dict or set(r)!={'path','sha256'} for r in frozen):
        raise ValueError('complete live frozen source/reference closure required')
    bound={r['path']:r['sha256'] for r in frozen}
    if len(bound)!=len(frozen) or any(bound.get(r['path'])!=r['sha256'] for r in required_refs):
        raise ValueError('live guard omits required exact source/profile/input reference')


def preparation_input_refs(c):
    """Role-bound metadata and exact deployment source identities, before I/O."""
    from poc.dense_frontier import verify_dense_parent_derivation
    verify_refs([{k:c['inputs'][k] for k in ('path','sha256')}]);compiled=finite_json(c['inputs']['path'])
    if compiled.get('population_manifest_sha256')!=c['inputs']['population_manifest_sha256']:
        raise ValueError('compiled evaluation population identity differs')
    windows=compiled['windows'];core.metric_work_plan(windows,mode='run')
    _,derivation=verify_dense_parent_derivation(c,evaluation_components={w['component_id'] for w in windows})
    teacher=c['reference_artifacts']['teacher_noop']
    refs=[derivation[k] for k in ('derivation','parent_manifest','child_manifest')]
    refs +=[{k:c['inputs'][k] for k in ('path','sha256')},c['worker'],c['reference_config'],c['reference_acceptance'],
        {'path':str(Path(teacher['directory'])/'manifest.json'),'sha256':teacher['manifest_sha256']}]
    verify_refs(refs);return refs


def admit_preparation_profile(c,recipe,acceptance,*,config_path,required_refs,admissions):
    """Re-admit the actual completed short fitting operation and final ledger."""
    from poc.dense_frontier import dense_preparation_plan
    from poc.cost_ancestry import read_final_cost
    out=Path(c['out']);phase='preprofile-'+recipe.label;directory=out/'phases'/phase
    if acceptance['path']!=str(directory/'acceptance.json'):raise ValueError('short preparation acceptance path differs')
    item=read_reference(acceptance)
    if set(item)!={'status','value','packet','terminal_cost','stage','plan'} or item['status']!='complete':
        raise ValueError('complete short preparation operation admission required')
    expected_paths={'packet':directory/'packet.json','terminal_cost':directory/'terminal-cost.json',
                    'plan':out/'phase-plans'/(phase+'.json')}
    if any(item[k]['path']!=str(v) for k,v in expected_paths.items()):raise ValueError('short operation handoff paths differ')
    packet=read_reference(item['packet']);plan=read_reference(item['plan'])
    expected_work=dense_preparation_plan(recipe,profile=True)
    caps={'wall_seconds':7200,'memory_bytes':c['preparation_limits']['memory_bytes'],'disk_bytes':c['preparation_limits']['family_disk_bytes']}
    expected_plan={'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'phase':phase,
                   'resource':caps,'preparation':expected_work,**admissions}
    if plan!=expected_plan:raise ValueError('short measured preparation recipe/source/resource plan differs')
    profile_contract={**c,'resource_freeze':None}
    if (packet.get('phase')!=phase or packet.get('scientific_contract_sha256')!=dense_contract_digest(c)
            or packet.get('execution_contract_sha256')!=common.hashed(profile_contract) or packet.get('value')!=item['value']):
        raise ValueError('short measured operation packet belongs to another execution')
    terminal=read_final_cost(item['terminal_cost'],expected_run_id=packet['run_id'],expected_contract_sha256=dense_contract_digest(c))
    if terminal.get('resource')!=caps or terminal.get('work_allowance')!=expected_work['work']:
        raise ValueError('short operation terminal work/caps differ')
    spec_ref=reference(Path(terminal['ledger']['path']).parent/'spec.json');spec=read_reference(spec_ref)
    if (spec_ref['sha256']!=packet['run_id'] or spec.get('resolved_config',{}).get('execution_contract')!=profile_contract
            or spec['resolved_config'].get('phase')!=phase or spec['resolved_config'].get('work_allowance')!=expected_work['work']):
        raise ValueError('short operation RunSpec/contract identity differs')
    refs=required_refs+[reference(config_path),item['plan']]
    sources=spec['resolved_config'].get('source_files',{})
    if any(sources.get(r['path'])!=r['sha256'] for r in refs):raise ValueError('short fitting RunSpec omits exact source/reference closure')
    command=[sys.executable,str(ROOT/'provenance/dense_frontier_campaign.py'),'--config',str(config_path),'--internal-phase',phase]
    core.check_stage_binding(item['stage'],command=command,root=ROOT,output=out,caps=caps,required_refs=refs,execution_env=dense_execution_env())
    value=item['value']
    if (value.get('scientific_contract_sha256')!=dense_contract_digest(c) or value.get('profile') is not True
            or value.get('recipe')!=recipe.identity() or value.get('preparation_plan')!=expected_work
            or value.get('teacher_needed_at_inference') is not False or value.get('candidate_quality_disclosed') is not False
            or value.get('directory')!=str(out/'artifacts/preprofile'/recipe.label)):
        raise ValueError('short fitting artifact/work recipe differs')
    from provenance.dense_frontier_campaign import preparation_profile
    measured=preparation_profile(recipe,item)
    # The resource packet is recomputed from admitted guard and final spending,
    # never accepted because its replacement JSON has a fresh hash.
    evidence=[acceptance,item['packet'],item['terminal_cost'],item['plan'],terminal['ledger'],spec_ref,
        {'path':item['stage']['protocol_path'],'sha256':item['stage']['protocol_sha256']},
        {'path':item['stage']['attempt_path'],'sha256':item['stage']['attempt_sha256']}]
    return {'measured':measured,'required_refs':evidence,'item':item}


def admit_preparation_freeze(c,recipe,*,config_path,required_refs,admissions):
    """Restore one full recipe only from its accepted measured short schedule."""
    from poc.dense_frontier import dense_preparation_forecast
    path=Path(c['out'])/'preparation-freeze.json';freeze_ref=reference(path);f=read_reference(freeze_ref)
    if (set(f)!={'schema_version','scientific_contract_sha256','plans','failures','scope'} or type(f['schema_version']) is not int
            or f['schema_version']!=1 or f['scientific_contract_sha256']!=dense_contract_digest(c)
            or set(f['plans'])|set(f['failures'])!={r.label for r in __import__('poc.dense_frontier',fromlist=['dense_recipes']).dense_recipes()}
            or set(f['plans'])&set(f['failures']) or recipe.label not in f['plans']):
        raise ValueError('complete immutable preparation resource freeze required')
    job=f['plans'][recipe.label]
    if set(job)!={'profile','acceptance','plan'}:raise ValueError('accepted measured preparation profile and resource plan required')
    admitted=admit_preparation_profile(c,recipe,job['acceptance'],config_path=config_path,required_refs=required_refs,admissions=admissions)
    measured=read_reference(job['profile'])
    if measured!=admitted['measured']:raise ValueError('frozen preparation timing/work differs from its actual admitted operation')
    forecast=dense_preparation_forecast(recipe,measured)
    if job['plan']!=forecast:raise ValueError('full preparation cap differs from admitted measured forecast')
    return {'resource':forecast['resource'],'required_refs':[freeze_ref,job['profile'],*admitted['required_refs']]}


def preparation_profile_closure(c,admissions):
    """Only facts that existed at short profiling; exclude later freeze files."""
    qref=admissions['qualification'];q=read_reference(qref)
    refs=require_dense_qualification(c,proposal_path=q['proposal']['path'],qualification=qref)
    refs+=require_dense_serial_admission(c,admission=admissions['serial_admission'])
    refs+=require_dense_native_qualification(c,qualification=admissions['native_qualification'],source_qualification=qref)
    return refs+preparation_input_refs(c)


def admit_native_profile(c,prepared,recipe,windows,item,*,required_refs,config_path):
    """Restore resource data only through its exact native operation and costs."""
    from poc.dense_frontier import dense_native_protocol
    from poc.cost_ancestry import read_final_cost
    expected_keys={'protocol','output','stage','common_packet','terminal_cost'}
    if type(item) is not dict or set(item)!=expected_keys:raise ValueError('complete native profile admission and terminal evidence required')
    out=Path(c['out']);phase='profile-'+recipe.label;case=out/'workers'/phase;directory=out/'phases'/phase
    paths={'protocol':out/'protocols'/(phase+'.json'),'output':case/'result.json',
           'common_packet':directory/'packet.json','terminal_cost':directory/'terminal-cost.json'}
    if any(item[k]['path']!=str(v) for k,v in paths.items()):raise ValueError('saved native profile case/handoff paths differ')
    counts={k:v for k,v in core.metric_work_plan(windows,mode='profile').items() if k!='components'}
    resource={'wall_seconds':300,'memory_bytes':c['preparation_limits']['memory_bytes'],
              'output_disk_bytes':c['preparation_limits']['family_disk_bytes'],**counts}
    profile_contract={**c,'resource_freeze':None}
    expected=dense_native_protocol(profile_contract,prepared,recipe.label,windows,mode='profile',resource=resource,
        output=case/'result.json',ledger=case/'costs.jsonl')
    protocol=read_reference(item['protocol']);value=read_reference(item['output'])
    if protocol!=expected:raise ValueError('saved native profile differs from exact qualified resource operation')
    worker=case/'native-worker.py'
    if file_sha(worker)!=c['worker']['sha256']:raise ValueError('copied native profile worker source differs')
    packet=read_reference(item['common_packet'])
    if (packet.get('scientific_contract_sha256')!=dense_contract_digest(c) or packet.get('phase')!=phase
            or packet.get('execution_contract_sha256')!=common.hashed(profile_contract)
            or packet.get('value')!={k:item[k] for k in ('protocol','output','stage')}):
        raise ValueError('native profile output/hash not bound to common phase packet')
    terminal=read_final_cost(item['terminal_cost'],expected_run_id=packet['run_id'],expected_contract_sha256=dense_contract_digest(c))
    expected_work=dict(updates=0,valid_target_presentations=counts['valid_target_presentations'],
        teacher_forward_calls=counts['teacher_forward_calls'],student_forward_calls=counts['student_forward_calls'],
        student_backward_calls=0,raw_corpus_target_presentations=counts['valid_target_presentations'])
    caps={'wall_seconds':330,'memory_bytes':c['preparation_limits']['memory_bytes'],'disk_bytes':c['preparation_limits']['family_disk_bytes']}
    if terminal.get('resource')!=caps or terminal.get('work_allowance')!=expected_work:
        raise ValueError('native profile terminal resource/work identity differs')
    if any(terminal['totals'][k]!=v for k,v in expected_work.items()):raise ValueError('native profile did not complete exact scored/raw/query work')
    spec_ref=reference(Path(terminal['ledger']['path']).parent/'spec.json');spec=read_reference(spec_ref)
    resolved=spec.get('resolved_config',{})
    if (spec_ref['sha256']!=packet['run_id'] or resolved.get('execution_contract')!=profile_contract
            or resolved.get('phase')!=phase or resolved.get('work_allowance')!=expected_work
            or resolved.get('declared_inputs')!={'protocol':expected}):
        raise ValueError('native profile pre-dispatch RunSpec/protocol identity differs')
    source_refs=required_refs+[reference(config_path),reference(out/'preparation.json'),c['worker']]
    source_refs +=[{'path':str(Path(expected[k]['path'])/'manifest.json'),'sha256':expected[k]['manifest_sha256']} for k in ('teacher','artifact')]
    if any(resolved.get('source_files',{}).get(r['path'])!=r['sha256'] for r in source_refs):
        raise ValueError('native profile RunSpec omits complete original source/input closure')
    guarded=source_refs+[item['protocol'],reference(worker)]
    core.check_stage_binding(item['stage'],command=[sys.executable,str(worker),'--protocol',item['protocol']['path']],
        root=Path('/private/tmp'),output=out,caps=caps,required_refs=guarded,execution_env={'PYTHONPATH':''})
    core.check_native_result(protocol,value)
    design={k:v for k,v in protocol.items() if k not in ('mode','output','ledger','resource','profile_binding')}
    if value.get('design_sha256')!=core.contract_digest(design):raise ValueError('native profile design identity differs')
    if value.get('environment_sha256')!=prepared.get('reference_native_environment_sha256'):
        raise ValueError('native profile environment differs from imported reference backend')
    if value.get('complete_bytes')!=prepared['artifacts'][recipe.label]['complete_bytes']:
        raise ValueError('native profile actual byte census differs')
    refs=[item[k] for k in ('protocol','output','common_packet','terminal_cost')]+[terminal['ledger'],spec_ref,reference(worker),
        {'path':item['stage']['protocol_path'],'sha256':item['stage']['protocol_sha256']},
        {'path':item['stage']['attempt_path'],'sha256':item['stage']['attempt_sha256']}]
    return {'value':value,'protocol':protocol,'terminal':terminal,'required_refs':refs}

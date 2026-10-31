#!/usr/bin/env python3
"""Explicit version2 dense campaign; source/native/serial gates precede models."""
from pathlib import Path
import argparse,json,math,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from poc.contracts import CostRecord
from poc.dense_frontier import (validate_contract_shape,dense_contract_digest,dense_recipes,dense_native_protocol,
    dense_preparation_plan,dense_preparation_forecast,dense_frontier_summary)
from poc.recovery_endpoints import file_sha,finite_json,verify_refs,write_once
from poc.contracts import thaw
from provenance import sharing_frontier_study as core
from provenance.dense_frontier_study import metadata_validate
from provenance.dense_frontier_execution import (reference,read_reference,record_dense_phase,dense_executed_sources,
    require_dense_qualification,require_dense_native_qualification,require_dense_serial_admission,dense_serial_evidence,
    dense_execution_env,prepare_dense_artifact,selected_dense_population,preparation_input_refs,admit_preparation_freeze,preparation_profile_closure,admit_native_profile)

PROPOSAL=ROOT/'configs/dense-baseline-expansion-v2-proposal-v1.yaml'
ZERO_WORK=dict(updates=0,valid_target_presentations=0,teacher_forward_calls=0,student_forward_calls=0,
               student_backward_calls=0,raw_corpus_target_presentations=0)


def read_contract(path):
    import yaml
    c=yaml.safe_load(Path(path).read_text())
    if type(c) is not dict or type(c.get('schema_version')) is not int or c['schema_version']!=2:
        raise ValueError('explicit new version2 campaign contract required')
    return thaw(validate_contract_shape(c,root=ROOT))


def admission_paths(c):
    prov=ROOT/'provenance'/c['study']
    return {'qualification':prov/'qualification-v1.json','serial_admission':prov/'serial-admission-v1.json',
            'native_qualification':prov/'native-qualification-v1.json'}


def native_admissions(c,refs):
    paths=admission_paths(c);q=reference(paths['qualification']);serial=reference(paths['serial_admission']);native=reference(paths['native_qualification'])
    more=require_dense_serial_admission(c,admission=serial)
    more+=require_dense_native_qualification(c,qualification=native,source_qualification=q)
    return refs+more,{k:reference(v) for k,v in paths.items()}


def run_controls(c,config_path,refs):
    paths=admission_paths(c);serial=paths['serial_admission']
    if serial.exists():require_dense_serial_admission(c,admission=reference(serial))
    else:write_once(serial,dense_serial_evidence(c))
    qref=reference(paths['qualification'])
    if paths['native_qualification'].exists():
        require_dense_native_qualification(c,qualification=reference(paths['native_qualification']),source_qualification=qref)
        return {'status':'already qualified native controls; no repeat','qualification':reference(paths['native_qualification'])}
    output=Path('/private/tmp/cgoc-dense-baseline-v2-native-controls-v1')
    if output.exists():raise FileExistsError('native controls already attempted; retain failures')
    output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','pytest','tests/test_dense_native_controls.py','tests/test_quantization.py','-q','-m','mlx','--basetemp='+str(output/'pytest')]
    caps={'wall_seconds':600,'memory_bytes':8*1024**3,'disk_bytes':2*1024**3}
    required=refs+require_dense_serial_admission(c,admission=reference(serial))
    stage=core.execute_stage(command,root=ROOT,output=output,stage=ROOT/'provenance'/c['study']/'native-controls-stage-v1',
        caps=caps,frozen_files=required,execution_env=dense_execution_env())
    core.require_stage_success(stage);core.check_stage_binding(stage,command=command,root=ROOT,output=output,caps=caps,required_refs=required,execution_env=dense_execution_env())
    value={'schema_version':1,'status':'qualified_dense_native_controls','study':c['study'],
        'scientific_contract_sha256':dense_contract_digest(c),'source_qualification':qref,'serial_admission':reference(serial),
        'command':command,'output':str(output),'caps':caps,'stage':stage,'new_owned_cases':6}
    write_once(paths['native_qualification'],value)
    require_dense_native_qualification(c,qualification=reference(paths['native_qualification']),source_qualification=qref)
    return {'status':'qualified native controls','qualification':reference(paths['native_qualification'])}


CLOSING_RESERVE_SECONDS=600
CLOSING_RESERVE_BYTES=8*1024**2


def envelope(c,required_wall,*,closing=False):
    out=Path(c['out']);intent=finite_json(out/'campaign-intent.json');limits=c['preparation_limits']
    if (intent.get('scientific_contract_sha256')!=dense_contract_digest(c) or intent.get('campaign_wall_seconds')!=limits['campaign_wall_seconds']):
        raise ValueError('campaign identity or finite wall contract changed')
    reserve=0 if closing else CLOSING_RESERVE_SECONDS
    if time.time()-intent['started_unix']+required_wall+reserve>limits['campaign_wall_seconds']:
        raise ValueError('remaining finite campaign time cannot supply completed phase')
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size+(0 if closing else CLOSING_RESERVE_BYTES)>limits['family_disk_bytes']:raise ValueError('new retained family disk cap exhausted')
    return {'wall_seconds':required_wall,'memory_bytes':limits['memory_bytes'],'disk_bytes':limits['family_disk_bytes']}



def campaign_ancestor_costs(c,imported):
    from poc.cost_ancestry import read_cost_ancestry
    return read_cost_ancestry(imported['ancestor_costs']['inventory'],expected_parent=c['reference_config'])


def attach_campaign_accounting(report,c,imported):
    out=Path(c['out'])
    report['original_reference_campaign_cost']=imported['aggregate_reference_campaign_cost']
    report['original_reference_phase_costs']=imported['phase_costs']
    report['ancestor_costs']=campaign_ancestor_costs(c,imported)
    report['all_upstream_expense_established']=report['ancestor_costs']['all_upstream_expense_established']
    report['reference_import']=reference(out/'imported-references.json')
    report['resource_freeze']=c['resource_freeze']
    report['qualification']={k:reference(v) for k,v in admission_paths(c).items() if v.is_file()}
    phases=out/'phases'
    report['new_phase_records']={p.name:{'packet':reference(p/'packet.json') if (p/'packet.json').exists() else None,
        'terminal':reference(p/'terminal-cost.json') if (p/'terminal-cost.json').exists() else None,
        'failure':reference(p/'failed.json') if (p/'failed.json').exists() else None,
        'ledgers':[reference(q) for q in p.rglob('costs.jsonl')]} for p in phases.iterdir() if p.is_dir() and p.name!='analysis'} if phases.exists() else {}
    report['new_retained_bytes_before_analysis']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    return report


def complete_if_infeasible(c,config_path,refs,imported,*,required_wall,phase,prepared=None,failures=None):
    """Keep the complete inventory when a completed grid cannot fit its cap.

    The prospective closing reserve pays for this metadata-only terminal phase;
    no shorter training schedule or unpriced retry is substituted.
    """
    try:envelope(c,required_wall)
    except ValueError as exc:
        out=Path(c['out']);failures=dict(failures or {});values=dict(imported['measurements'])
        for recipe in dense_recipes():
            if recipe.label not in failures:
                name='not-run-'+phase+'-'+recipe.label;path=out/'failures'/(name+'.json')
                write_once(path,{'status':'failed','label':recipe.label,'phase':phase,'error':repr(exc),
                    'scientific_contract_sha256':dense_contract_digest(c),'dispatch_status':'not_run_resource_admission',
                    'required_remaining_wall_seconds':required_wall,'closing_reserve_seconds':CLOSING_RESERVE_SECONDS,
                    'scope':'resource feasibility failure; no final inference dispatched; no scientific capacity or quality conclusion; no automatic retry'})
                failures[recipe.label]={'status':'failed','failure':reference(path),'reason':phase+' aggregate resource admission failed; not run'}
            values[recipe.label]=failures[recipe.label]
        p=prepared if prepared is not None else {'study_contract_sha256':dense_contract_digest(c),'inputs':c['inputs'],
            'artifacts':{**imported['artifacts'],**failures}}
        p={**p,'artifacts':{**p['artifacts'],**{r.label:p['artifacts'].get(r.label,failures[r.label]) for r in dense_recipes()}}}
        def operation(rec):
            report=dense_frontier_summary(c,p,imported['windows'],values)
            report['campaign_status']='resource_failed_before_completed_grid'
            report['resource_failure']={'phase':phase,'error':repr(exc),'required_remaining_wall_seconds':required_wall,
                'closing_reserve_seconds':CLOSING_RESERVE_SECONDS,'schedules_truncated':False,'automatic_retry':False}
            report['retained_preparation_index']=reference(out/'preparation.json') if (out/'preparation.json').is_file() else None
            attach_campaign_accounting(report,c,imported);write_once(out/'analysis.json',report)
            return {'analysis':reference(out/'analysis.json'),'campaign_status':report['campaign_status'],
                'scope':'complete18point outcome inventory; new grid resource-failed; no fabricated metric; independent admission pending'}
        value=record_dense_phase(c,'analysis',{r['path']:r['sha256'] for r in refs+[reference(config_path),reference(out/'imported-references.json')]},
            operation,resource=envelope(c,300,closing=True),work=ZERO_WORK)
        write_once(out/'evaluation-index.json',failures)
        write_once(out/'analysis-completion.json',{'status':'common-driver analysis complete; independent admission pending',**value,
            'terminal_cost':reference(out/'phases/analysis/terminal-cost.json')})
        return {'status':'terminal resource failure; complete outcome inventory retained',**value}
    return None


def dispatch_prepare(c,config_path,recipe,*,profile,resource,refs,admissions):
    phase=('preprofile-' if profile else 'prepare-')+recipe.label;out=Path(c['out'])
    preparation=dense_preparation_plan(recipe,profile=profile)
    plan={'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'phase':phase,
        'resource':resource,'preparation':preparation,**admissions}
    path=out/'phase-plans'/(phase+'.json');write_once(path,plan)
    required=refs+preparation_input_refs(c)
    if not profile:
        saved=admit_preparation_freeze(c,recipe,config_path=config_path,required_refs=preparation_profile_closure(c,admissions),admissions=admissions)
        if resource!=saved['resource']:raise ValueError('dispatch full-preparation resource differs from measured freeze')
        required+=saved['required_refs']
    required +=[reference(path),reference(config_path)]
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(config_path),'--internal-phase',phase]
    envelope(c,resource['wall_seconds'])
    stage=core.execute_stage(command,root=ROOT,output=out,stage=out/'stages'/phase,caps=resource,
        frozen_files=required,execution_env=dense_execution_env())
    core.require_stage_success(stage);core.check_stage_binding(stage,command=command,root=ROOT,output=out,caps=resource,required_refs=required,execution_env=dense_execution_env())
    directory=out/'phases'/phase;packet=finite_json(directory/'packet.json');terminal=finite_json(directory/'terminal-cost.json')
    if (packet.get('scientific_contract_sha256')!=dense_contract_digest(c) or terminal.get('scientific_contract_sha256')!=dense_contract_digest(c)
            or packet.get('phase')!=phase or terminal.get('run_id')!=packet.get('run_id')):
        raise ValueError('prepared phase common identity/cost handoff differs')
    verify_refs([terminal['ledger']]);value=packet['value']
    item={'status':'complete','value':value,'packet':reference(directory/'packet.json'),
          'terminal_cost':reference(directory/'terminal-cost.json'),'stage':stage,'plan':reference(path)}
    write_once(directory/'acceptance.json',item);return item


def failed_recipe(c,recipe,phase,exc):
    """A failed artifact stays visible; it supplies no fabricated quality value."""
    out=Path(c['out']);directory=out/'phases'/phase
    records=[reference(p) for p in directory.rglob('*') if p.is_file()] if directory.exists() else []
    stage=out/'stages'/phase/'attempt.json'
    if stage.exists():records.append(reference(stage))
    value={'status':'failed','label':recipe.label,'phase':phase,'error':repr(exc),'retained_records':records,
        'scope':'implementation/resource/algorithm failure; no method-quality or capacity conclusion; no automatic retry'}
    path=out/'failures'/(phase+'.json');write_once(path,value)
    return {'status':'failed','failure':reference(path),'reason':phase+' failed; all durable work retained'}


def preparation_profile(recipe,item):
    value=item['value'];terminal=read_reference(item['terminal_cost']);totals=terminal['totals']
    keys=dense_preparation_plan(recipe,profile=True)['work']
    return {'schema_version':1,'recipe':recipe.identity(),'profile':True,'plan':dense_preparation_plan(recipe,profile=True),
        'wall_seconds':item['stage']['attempt']['wall_seconds'],'complete_bytes':value['complete_bytes'],
        'memory_peak_bytes':max(totals['memory_bytes'] or 0,item['stage']['attempt']['sampled_process_tree_rss_bytes']),
        'actual_work':{k:totals[k] for k in keys},'candidate_quality_disclosed':False}


def mirror_native_ledger(rec,protocol,path):
    counts=dict(valid_target_presentations=0,teacher_forward_calls=0,student_forward_calls=0)
    if not path.exists():return counts
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):raise ValueError('truncated native cost tail retained; complete prefix charged')
            row=json.loads(line);work=row.get('work',{})
            if (row.get('study')!=protocol['study'] or row.get('study_contract_sha256')!=protocol['study_contract_sha256']
                    or row.get('protocol_sha256')!=core.contract_digest(protocol) or not work or not set(work)<=set(counts)
                    or any(type(v) is not int or v<0 for v in work.values())):
                raise ValueError('native ledger study/protocol/work units differ')
            for key,value in work.items():counts[key]+=value
            raw=work.get('valid_target_presentations',0)
            if row.get('cumulative')!=counts or row.get('raw_corpus_target_presentations')!=raw:
                raise ValueError('native cumulative or raw context count differs')
            rec.record(CostRecord(**work,raw_corpus_target_presentations=raw,processed_inputs=int(raw>0),processed_tokens=512 if raw else 0))
    return counts


def native_job(c,prepared,recipe,windows,*,mode,resource,refs,profile=None,frozen=None):
    refs,_=native_admissions(c,refs);out=Path(c['out']);label=recipe.label;phase=mode+'-'+label
    counts={k:resource[k] for k in ('valid_target_presentations','teacher_forward_calls','student_forward_calls')}
    work={**ZERO_WORK,**counts,'raw_corpus_target_presentations':counts['valid_target_presentations']}
    caps=envelope(c,resource['wall_seconds']+30)
    case=out/'workers'/phase
    protocol=dense_native_protocol(c,prepared,label,windows,mode=mode,resource=resource,output=case/'result.json',ledger=case/'costs.jsonl',profile_binding=profile)
    refs=refs+preparation_input_refs(c)+[reference(config_for(c)),reference(out/'preparation.json'),c['worker']]
    refs +=[{'path':str(Path(protocol[k]['path'])/'manifest.json'),'sha256':protocol[k]['manifest_sha256']} for k in ('teacher','artifact')]
    def operation(rec):
        case=out/'workers'/phase;case.mkdir(parents=True,exist_ok=False)
        worker=case/'native-worker.py';worker.write_bytes(Path(c['worker']['path']).read_bytes())
        if file_sha(worker)!=c['worker']['sha256']:raise ValueError('private native worker source differs')
        protocol=dense_native_protocol(c,prepared,label,windows,mode=mode,resource=resource,
            output=case/'result.json',ledger=case/'costs.jsonl',profile_binding=profile)
        if frozen is not None and read_reference(frozen)!=protocol:raise ValueError('run differs from outcome-blind frozen completed schedule')
        pp=out/'protocols'/(phase+'.json');write_once(pp,protocol)
        required=refs+[reference(config_for(c)),reference(pp),reference(worker),reference(out/'preparation.json')]
        required +=[{'path':str(Path(protocol[k]['path'])/'manifest.json'),'sha256':protocol[k]['manifest_sha256']} for k in ('teacher','artifact')]
        required +=[r for r in (profile,frozen) if r is not None]
        command=[sys.executable,str(worker),'--protocol',str(pp)];stage=None
        try:
            stage=core.execute_stage(command,root=Path('/private/tmp'),output=out,stage=out/'stages'/phase,
                caps=caps,frozen_files=required,execution_env={'PYTHONPATH':''})
        finally:
            mirror_native_ledger(rec,protocol,case/'costs.jsonl')
            if stage is not None:rec.record(CostRecord(memory_bytes=stage['attempt']['sampled_process_tree_rss_bytes'],memory_unavailable_reason=None,disk_bytes=stage['attempt']['sampled_scratch_peak_bytes']))
        core.require_stage_success(stage);core.check_stage_binding(stage,command=command,root=Path('/private/tmp'),output=out,caps=caps,required_refs=required,execution_env={'PYTHONPATH':''})
        value=read_reference(reference(case/'result.json'));core.check_native_result(protocol,value,rows=windows if mode=='run' else None)
        if value['environment_sha256']!=prepared['reference_native_environment_sha256']:
            raise ValueError('new native observation environment differs from imported references')
        item={'protocol':reference(pp),'output':reference(case/'result.json'),'stage':stage};write_once(case/'acceptance.json',item)
        return item
    item=record_dense_phase(c,phase,{r['path']:r['sha256'] for r in refs},operation,resource=caps,work=work,declared_inputs={'protocol':protocol})
    directory=out/'phases'/phase
    return {**item,'common_packet':reference(directory/'packet.json'),'terminal_cost':reference(directory/'terminal-cost.json')}


def config_for(c):return ROOT/'configs/dense-baseline-expansion-v2-run-v1.yaml' if c.get('resource_freeze') else PROPOSAL


def profile_campaign(c,config_path,refs):
    out=Path(c['out'])
    if out.exists():raise FileExistsError('campaign already attempted; preserve all evidence')
    refs,admissions=native_admissions(c,refs)
    if c['resource_freeze'] is not None:raise ValueError('profile starts from unprofiled proposal only')
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    write_once(out/'campaign-intent.json',{'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),
        'started_unix':started,'campaign_wall_seconds':c['preparation_limits']['campaign_wall_seconds'],
        'source_qualification':admissions['qualification'],'serial_admission':admissions['serial_admission'],
        'scope':'six new fixed recipes; imported12 retain original identities and all ancestor costs'})
    from provenance.dense_frontier_execution import import_dense_references
    try:
        sources={r['path']:r['sha256'] for r in refs}
        imported=record_dense_phase(c,'reference-import',sources,lambda rec:import_dense_references(c,config_path=config_path),
            resource=envelope(c,300),work=ZERO_WORK)
        import_path=out/'imported-references.json';write_once(import_path,imported);refs+= [reference(import_path)]
        # Validate every eligible512token fit/internal context before native fit.
        recipe=next(r for r in dense_recipes() if r.method=='dwq' and r.bits==4)
        record_dense_phase(c,'eligibility',sources,lambda rec:selected_dense_population(c,recipe,
            c['reference_artifacts']['teacher_noop']['directory'],profile=False,phase='eligibility',
            evaluation_components={w['component_id'] for w in imported['windows']})[1],resource=envelope(c,300),work=ZERO_WORK)
        profiles={};plans={};failures={}
        for recipe in dense_recipes():
            try:
                item=dispatch_prepare(c,config_path,recipe,profile=True,resource=envelope(c,7200),refs=refs,admissions=admissions)
                profiles[recipe.label]=item;measured=preparation_profile(recipe,item)
                path=out/'preparation-profiles'/(recipe.label+'.json');write_once(path,measured)
                plans[recipe.label]={'profile':reference(path),'acceptance':reference(out/'phases'/('preprofile-'+recipe.label)/'acceptance.json'),'plan':dense_preparation_forecast(recipe,measured)}
            except Exception as exc:
                require_dense_qualification(c,proposal_path=PROPOSAL,qualification=admissions['qualification'])
                failures[recipe.label]=failed_recipe(c,recipe,'preprofile-'+recipe.label,exc)
        required=sum(item['plan']['resource']['wall_seconds'] for item in plans.values())
        terminal=complete_if_infeasible(c,config_path,refs,imported,required_wall=required,phase='aggregate-preparation',failures=failures)
        if terminal is not None:return terminal
        write_once(out/'preparation-freeze.json',{'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'plans':plans,'failures':failures,
            'scope':'all full-preparation resource caps frozen before first final fitting; no development quality used'})
        refs.append(reference(out/'preparation-freeze.json'));artifacts=dict(imported['artifacts'])
        for recipe in dense_recipes():
            if recipe.label in failures:artifacts[recipe.label]=dict(failures[recipe.label]);continue
            try:
                item=dispatch_prepare(c,config_path,recipe,profile=False,resource=plans[recipe.label]['plan']['resource'],refs=refs,admissions=admissions)
                artifacts[recipe.label]=item['value']
            except Exception as exc:
                require_dense_qualification(c,proposal_path=PROPOSAL,qualification=admissions['qualification'])
                failures[recipe.label]=failed_recipe(c,recipe,'prepare-'+recipe.label,exc);artifacts[recipe.label]=dict(failures[recipe.label])
        blocked=sorted(set(imported['prepared']['blocked_source_roots'])|{str(ROOT)})
        prepared={'schema_version':1,'study_contract_sha256':dense_contract_digest(c),'inputs':c['inputs'],
            'teacher_mps_digest':imported['prepared']['teacher_mps_digest'],'artifacts':artifacts,'blocked_source_roots':blocked,
            'reference_native_environment_sha256':imported['native_environment_sha256'],'reference_import':reference(import_path)}
        write_once(out/'preparation.json',prepared)
        native_profiles={};jobs={};windows=imported['windows']
        work={k:v for k,v in core.metric_work_plan(windows,mode='profile').items() if k!='components'}
        resource={'wall_seconds':300,'memory_bytes':c['preparation_limits']['memory_bytes'],'output_disk_bytes':c['preparation_limits']['family_disk_bytes'],**work}
        for recipe in dense_recipes():
            if recipe.label in failures:continue
            try:
                item=native_job(c,prepared,recipe,windows,mode='profile',resource=resource,refs=refs)
                native_profiles[recipe.label]=item;value=read_reference(item['output'])
                forecast=core.forecast_worker(value,windows,teacher_complete_bytes=artifacts['teacher_noop']['complete_bytes'],
                    memory_bytes=resource['memory_bytes'],output_disk_bytes=resource['output_disk_bytes'],wall_ceiling=7200)
                jobs[recipe.label]={'profile':item['output'],'profile_acceptance':item,'plan':forecast}
            except Exception as exc:
                require_dense_qualification(c,proposal_path=PROPOSAL,qualification=admissions['qualification'])
                failures[recipe.label]=failed_recipe(c,recipe,'profile-'+recipe.label,exc)
        terminal=complete_if_infeasible(c,config_path,refs,imported,required_wall=sum(item['plan']['resource']['wall_seconds']+60 for item in jobs.values()),
            phase='aggregate-native-evaluation',prepared=prepared,failures=failures)
        if terminal is not None:return terminal
        capsule={'schema_version':1,'scientific_contract_sha256':dense_contract_digest(c),'preparation_freeze':reference(out/'preparation-freeze.json'),
            'preparation':reference(out/'preparation.json'),'reference_import':reference(import_path),'jobs':jobs,'failures':failures,
            'native_profiles':native_profiles,'candidate_quality_used':False,'order':[r.label for r in dense_recipes()],
            'cost_scope':'new phases plus separately reported complete original reference campaign and source/native qualification expenses'}
        path=out/'resource-freeze.json';write_once(path,capsule)
        run={**c,'resource_freeze':reference(path)};run_path=ROOT/'configs/dense-baseline-expansion-v2-run-v1.yaml'
        import yaml
        with run_path.open('x') as stream:yaml.safe_dump(run,stream,sort_keys=False)
        if read_contract(run_path)!=run:raise ValueError('frozen run configuration reload differs')
        for recipe in dense_recipes():
            if recipe.label not in jobs:continue
            job=jobs[recipe.label];case=out/'workers'/('run-'+recipe.label)
            protocol=dense_native_protocol(run,prepared,recipe.label,windows,mode='run',resource=job['plan']['resource'],
                output=case/'result.json',ledger=case/'costs.jsonl',profile_binding=job['profile'])
            write_once(out/'frozen-protocols'/(recipe.label+'.json'),protocol)
        return {'status':'profiled; final resources frozen before new development quality','run_configuration':reference(run_path),
                'resource_freeze':reference(path),'candidate_quality_disclosed':False,'failed_recipes':sorted(failures)}
    except BaseException as exc:
        write_once(out/'profile-failed.json',{'status':'failed','error':repr(exc),'no_automatic_retry':True});raise



def admitted_native_job(c,capsule,prepared,recipe,windows,*,required_refs,config_path):
    """Rebuild a completed schedule from the actual accepted native profile."""
    job=capsule['jobs'][recipe.label]
    if (type(job) is not dict or set(job)!={'profile','profile_acceptance','plan'}
            or job['profile_acceptance']!=capsule.get('native_profiles',{}).get(recipe.label)
            or job['profile']!=job['profile_acceptance'].get('output')):
        raise ValueError('frozen native job lacks the exact saved profile admission')
    admitted=admit_native_profile(c,prepared,recipe,windows,job['profile_acceptance'],required_refs=required_refs,config_path=config_path)
    expected=core.forecast_worker(admitted['value'],windows,
        teacher_complete_bytes=prepared['artifacts']['teacher_noop']['complete_bytes'],
        memory_bytes=c['preparation_limits']['memory_bytes'],output_disk_bytes=c['preparation_limits']['family_disk_bytes'],wall_ceiling=7200)
    if job['plan']!=expected:raise ValueError('native completed schedule differs from exact admitted measured profile')
    return {'plan':expected,'required_refs':admitted['required_refs']}


def native_profile_closure(c,capsule,admissions):
    # The generated run YAML and native freeze did not exist at profiling time.
    return preparation_profile_closure(c,admissions)+[capsule['reference_import'],capsule['preparation_freeze']]


def require_freeze(c):
    if c.get('resource_freeze') is None:raise ValueError('run requires exact frozen resource capsule')
    capsule=read_reference(c['resource_freeze']);out=Path(c['out'])
    if (set(capsule)!={'schema_version','scientific_contract_sha256','preparation_freeze','preparation','reference_import','jobs','failures',
                      'native_profiles','candidate_quality_used','order','cost_scope'}
            or capsule['schema_version']!=1 or capsule['scientific_contract_sha256']!=dense_contract_digest(c)
            or capsule['candidate_quality_used'] is not False or capsule['order']!=[r.label for r in dense_recipes()]
            or set(capsule['jobs'])|set(capsule['failures'])!={r.label for r in dense_recipes()}
            or set(capsule['jobs'])&set(capsule['failures'])
            or not set(capsule['jobs'])<=set(capsule['native_profiles'])
            or not set(capsule['native_profiles'])<={r.label for r in dense_recipes()}):raise ValueError('closed complete outcome-blind resource freeze differs')
    for key in ('preparation_freeze','preparation','reference_import'):read_reference(capsule[key])
    if capsule['preparation']!=reference(out/'preparation.json'):raise ValueError('frozen prepared artifact index differs')
    return capsule


def run_campaign(c,config_path,refs):
    capsule=require_freeze(c);refs,admissions=native_admissions(c,refs);out=Path(c['out'])
    if (out/'evaluation-index.json').exists() or any((out/'workers'/('run-'+r.label)).exists() for r in dense_recipes()):
        raise FileExistsError('native evaluation already attempted; preserve partial work')
    imported=read_reference(capsule['reference_import']);prepared=read_reference(capsule['preparation']);windows=imported['windows']
    refs +=[c['resource_freeze'],reference(config_path)]
    results=dict(capsule['failures'])
    for recipe in dense_recipes():
        if recipe.label not in capsule['jobs']:continue
        job=capsule['jobs'][recipe.label]
        try:
            admitted=admitted_native_job(c,capsule,prepared,recipe,windows,
                required_refs=native_profile_closure(c,capsule,admissions),config_path=PROPOSAL)
            results[recipe.label]={'status':'complete',**native_job(c,prepared,recipe,windows,mode='run',resource=admitted['plan']['resource'],
                refs=refs+admitted['required_refs'],profile=job['profile'],frozen=reference(out/'frozen-protocols'/(recipe.label+'.json')))}
        except Exception as exc:
            require_dense_qualification(c,proposal_path=PROPOSAL,qualification=reference(admission_paths(c)['qualification']))
            results[recipe.label]=failed_recipe(c,recipe,'run-'+recipe.label,exc)
    write_once(out/'evaluation-index.json',results)
    return analyze_campaign(c,config_path,refs)


def analyze_campaign(c,config_path,refs):
    capsule=require_freeze(c);refs,admissions=native_admissions(c,refs);out=Path(c['out'])
    if (out/'analysis.json').exists():raise FileExistsError('analysis already written; no overwrite')
    imported=read_reference(capsule['reference_import']);prepared=read_reference(capsule['preparation']);windows=imported['windows']
    index=finite_json(out/'evaluation-index.json')
    if set(index)!={r.label for r in dense_recipes()}:raise ValueError('all six terminal outcomes required')
    def operation(rec):
        values=dict(imported['measurements'])
        for recipe in dense_recipes():
            item=index[recipe.label]
            if item['status']=='failed':
                read_reference(item['failure']);values[recipe.label]=item;continue
            admitted_native_job(c,capsule,prepared,recipe,windows,
                required_refs=native_profile_closure(c,capsule,admissions),config_path=PROPOSAL)
            protocol=read_reference(item['protocol']);value=read_reference(item['output'])
            if protocol!=read_reference(reference(out/'frozen-protocols'/(recipe.label+'.json'))):raise ValueError('analysis native protocol differs from frozen schedule')
            core.require_stage_success(item['stage']);core.check_native_result(protocol,value,rows=windows)
            if value['environment_sha256']!=imported['native_environment_sha256']:raise ValueError('native observation environment differs')
            values[recipe.label]={'status':'complete',**value}
        report=dense_frontier_summary(c,prepared,windows,values)
        attach_campaign_accounting(report,c,imported)
        report['campaign_status']='complete_grid_with_failed_recipes' if report['failed_labels'] else 'complete_grid'
        write_once(out/'analysis.json',report)
        return {'analysis':reference(out/'analysis.json'),'scope':'exploratory frontier; independent final raw/cost/artifact admission pending'}
    value=record_dense_phase(c,'analysis',{r['path']:r['sha256'] for r in refs},operation,resource=envelope(c,300,closing=True),work=ZERO_WORK)
    write_once(out/'analysis-completion.json',{'status':'common-driver analysis complete; independent admission pending',**value,
        'terminal_cost':reference(out/'phases/analysis/terminal-cost.json')})
    return {'status':'complete; independent admission pending',**value}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--mode',choices=('validate','controls','profile','run','analyze'))
    modes.add_argument('--internal-phase');args=parser.parse_args(argv);path=args.config.resolve();c=read_contract(path)
    if args.mode=='validate':
        result=metadata_validate(path)
    else:
        qualification=reference(admission_paths(c)['qualification']) if admission_paths(c)['qualification'].exists() else {'path':str(admission_paths(c)['qualification']),'sha256':'0'*64}
        refs=require_dense_qualification(c,proposal_path=PROPOSAL,qualification=qualification)
        if args.internal_phase:
            name=args.internal_phase;profile=name.startswith('preprofile-');prefix='preprofile-' if profile else 'prepare-'
            if not name.startswith(prefix) or name[len(prefix):] not in {r.label for r in dense_recipes()}:raise ValueError('unknown internal finite phase')
            result=prepare_dense_artifact(c,name[len(prefix):],config_path=path,profile=profile,sources={r['path']:r['sha256'] for r in refs})
        else:result={'controls':run_controls,'profile':profile_campaign,'run':run_campaign,'analyze':analyze_campaign}[args.mode](c,path,refs)
    print(json.dumps({k:v for k,v in result.items() if k not in ('algorithm_sources','calibration_plan')},sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

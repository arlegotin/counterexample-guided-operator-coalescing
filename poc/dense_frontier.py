"""Prospective dense-baseline recipes and metadata-only calibration planning.

No MLX, teacher, optimizer, tokenization, raw text or protected reader is used.
Actual native dispatch and its accounting profile are separate qualified steps.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib


@dataclass(frozen=True)
class DenseRecipe:
    method:str
    bits:int
    embedding_bits:int=4
    calibration_components:int=64
    internal_components:int=0
    group_size:int=64
    seed:int=17
    context:int=512

    def __post_init__(self):
        if self.method not in ('gptq','awq','dwq'):raise ValueError('explicit eligible learned dense method required')
        expected={'embedding_bits':4,'calibration_components':64,'internal_components':16 if self.method=='dwq' else 0,
                  'group_size':64,'seed':17,'context':512}
        if type(self.bits) is not int or self.bits not in (2,4):raise ValueError('fixed2/4bit comparison required')
        for key,value in expected.items():
            if type(getattr(self,key)) is not int or getattr(self,key)!=value:raise ValueError('fixed dense baseline design differs:'+key)

    @property
    def label(self):return f'dense_{self.method}{self.bits}_embedding4_cal64'
    @property
    def completed_updates(self):return 64 if self.method=='dwq' else 0
    @property
    def calibration_targets(self):return self.calibration_components*(self.context-1)
    @property
    def internal_target_inventory(self):return self.internal_components*(self.context-1)
    def identity(self):return asdict(self)


def dense_recipes():
    return tuple(DenseRecipe(method,bits,internal_components=16 if method=='dwq' else 0)
                 for method in ('gptq','awq','dwq') for bits in (4,2))


def calibration_plan(manifest,*,evaluation_components):
    """Choose components before content reads; eligibility is checked later."""
    records=manifest.records;ids={r.document_id for r in records}
    if (len(records)!=len(ids) or set(manifest.roles)!=ids or set(manifest.membership)!=ids
            or set(manifest.roles.values())!={'fit'} or manifest.quarantine):
        raise ValueError('complete fit-only nonquarantined population required')
    components=set(manifest.membership.values())
    if any(type(c) is not str or not c for c in components) or components&set(evaluation_components):
        raise ValueError('fit/internal/evaluation source components overlap or lack identity')
    selected=[];seen=set()
    for record in sorted(records,key=lambda r:r.document_id):
        component=manifest.membership[record.document_id]
        if component in seen:continue
        selected.append(record.document_id);seen.add(component)
        if len(selected)==80:break
    if len(selected)!=80:raise ValueError('80 distinct fit components required; no silent replacement')
    train=selected[:64];internal=selected[64:]
    return {'schema_version':1,'rule':'first80 distinct components by sorted document_id; first64train next16fit_internal',
            'source_manifest_digest':manifest.digest(),'train_document_ids':train,'internal_document_ids':internal,
            'train_component_ids':[manifest.membership[d] for d in train],
            'internal_component_ids':[manifest.membership[d] for d in internal],
            'selected_components':80,'selected_target_inventory':80*511,
            'eligibility':'requires role-bound exact512token checks before fitting',
            'scope':'fit-internal validation; exposed development evaluation separate; no protected access'}


def installed_algorithm_sources(*,package_root=None):
    """Hash installed original code via distribution metadata, never find_spec.

    Importing a child of mlx_lm initializes Metal on this package version.
    Distribution metadata and ordinary source bytes remain portable CPU I/O.
    The explicit root is for owned fixtures and preverified execution metadata.
    """
    if package_root is None:
        from importlib.metadata import distribution
        package_root=Path(distribution('mlx-lm').locate_file('mlx_lm')).resolve()
    root=Path(package_root)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir() or root.is_symlink():
        raise ValueError('canonical installed package source root required')
    paths=sorted(root.rglob('*.py'))
    needed={'quant/gptq.py','quant/awq.py','quant/dwq.py','models/llama.py','__init__.py'}
    if not needed<={str(p.relative_to(root)) for p in paths}:raise ValueError('required original algorithm/model code missing')
    result=[]
    for path in paths:
        if path.resolve()!=path or path.is_symlink() or not path.is_file() or path.stat().st_nlink!=1:
            raise ValueError('canonical unaliased original source files required')
        before=path.stat();h=hashlib.sha256(path.read_bytes()).hexdigest();after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('installed source changed while hashing')
        result.append({'path':str(path),'sha256':h})
    return result


def dwq_target_units(phase,*,input_tokens,original_context_tokens):
    """Record full teacher-distribution targets separately from shifted labels."""
    if (phase not in ('train','valid','bridge') or type(original_context_tokens) is not int or original_context_tokens!=512
            or type(input_tokens) is not int or input_tokens!=(512 if phase=='bridge' else 511)):
        raise ValueError('declared original512context and exact named DWQ input required')
    return {'distribution_targets':511,'model_shifted_pairs':input_tokens-1,'processed_tokens':input_tokens,
            'original_corpus_targets':511,'label_mask_source':'original512context length; final EOS target preserved',
            'scope':'explicit complete source context; no padding or inferred token-ID masking'}


from types import MappingProxyType
REFERENCE_LABELS=('teacher_noop','shared_float','dense_rtn8','dense_rtn4','dense_rtn2','dense_rtn2_embedding4',
                  'dense_gptq4','dense_gptq2_embedding4','shared_rtn4','shared_rtn2_embedding4','shared_gptq4','shared_gptq2_embedding4')
PREPARATION_LIMITS=MappingProxyType({'phase_wall_seconds':7200,'memory_bytes':8*1024**3,
                                    'family_disk_bytes':12*1024**3,'campaign_wall_seconds':8*3600})
NUMERICS=MappingProxyType({'temperature':1,'chunk_size':32,'negative_kl_floor_per_target':2e-6,
                          'decision_resolution':2e-5,'accumulation':'cpu_float64','vocabulary':'complete',
                          'capability_tolerance':None})


def _canonical_path(value):
    if type(value) is not str or not value or '\x00' in value:raise ValueError('canonical absolute path required')
    p=Path(value)
    if not p.is_absolute() or str(p)!=value or p.resolve()!=p or '..' in p.parts:raise ValueError('canonical absolute path required')
    return p


def _reference(value,*,bundle=False,inputs=False):
    import re
    keys={'directory','manifest_sha256'} if bundle else {'path','sha256'}
    if inputs:keys=keys|{'population_manifest_sha256'}
    if type(value) is not dict or set(value)!=keys:raise ValueError('closed exact metadata reference required')
    _canonical_path(value['directory' if bundle else 'path'])
    for key in keys-{'path','directory'}:
        if type(value[key]) is not str or re.fullmatch('[0-9a-f]{64}',value[key]) is None:raise ValueError('exact SHA256 required')


def validate_contract_shape(contract,*,root):
    """Closed prospective schema, without opening a source, model or corpus."""
    from poc.contracts import freeze
    fields={'schema_version','study','out','reference_acceptance','reference_config','reference_artifacts','fit_manifest',
            'inputs','worker','recipes','calibration_order','numerics','preparation_limits','resource_freeze'}
    if type(contract) is not dict or type(contract.get('schema_version')) is not int or contract['schema_version'] not in (1,2):
        raise ValueError('closed named dense frontier version1/2 required')
    version=contract['schema_version']
    if version==2:fields|={'accounting_profile','tokenization_policy'}
    if set(contract)!=fields or contract['study']!='dense-baseline-expansion-v'+str(version):
        raise ValueError('closed exact versioned dense study identity required')
    if version==2 and (contract['accounting_profile']!='dense-native-contexts-v2' or contract['tokenization_policy']!='no-special-tokens-v2'):
        raise ValueError('explicit context-accounting/tokenization version2 required')
    root=_canonical_path(str(root));out=_canonical_path(contract['out'])
    if out!=root/('artifacts/dense-baseline-expansion-v'+str(version)):raise ValueError('isolated exact new-study output required')
    if contract['recipes']!=[r.identity() for r in dense_recipes()]:raise ValueError('six fixed eligible dense recipes required')
    if contract['calibration_order']!='first80 disjoint components sorted by document_id; first64train next16fit_internal':
        raise ValueError('prospective outcome-independent component allocation required')
    if type(contract['numerics']) is not dict or contract['numerics']!=dict(NUMERICS):raise ValueError('qualified numerical quantities changed')
    limits=contract['preparation_limits']
    if type(limits) is not dict or limits!=dict(PREPARATION_LIMITS) or any(type(v) is not int for v in limits.values()):
        raise ValueError('finite prospective preparation ceilings changed')
    for key in ('reference_acceptance','reference_config','fit_manifest','worker'):_reference(contract[key])
    _reference(contract['inputs'],inputs=True)
    refs=contract['reference_artifacts']
    if type(refs) is not dict or set(refs)!=set(REFERENCE_LABELS):raise ValueError('all12 original reference artifacts required')
    for value in refs.values():_reference(value,bundle=True)
    if contract['resource_freeze'] is not None:_reference(contract['resource_freeze'])
    return freeze(contract)



def dense_contract_digest(contract):
    """Scientific identity; v2 measured caps remain separately execution-bound.

    Only the outcome-blind resource capsule is excluded. Recipe, preparation
    ceilings, inputs, calibration allocation and numerical policy remain bound.
    Legacy v1 keeps its original complete-contract identity.
    """
    from poc.contracts import thaw
    from poc.records import canonical_json
    c=thaw(contract)
    if c.get('schema_version')==2:
        c={k:v for k,v in c.items() if k!='resource_freeze'}
    return hashlib.sha256(canonical_json(c)).hexdigest()


def dense_native_protocol(contract,prepared,label,rows,*,mode,resource,output,ledger,profile_binding=None):
    """Compile the unchanged qualified native worker for this explicit18grid.

    No source, model, corpus or profile content is opened here. The executing
    driver additionally verifies the exact referenced bytes and frozen budgets.
    """
    import math,re
    from poc.contracts import thaw
    from provenance.sharing_frontier_study import metric_work_plan
    c=thaw(contract)
    validate_contract_shape(c,root=Path(c['out']).parent.parent)
    labels=set(REFERENCE_LABELS)|{r.label for r in dense_recipes()}
    if (type(prepared) is not dict or prepared.get('study_contract_sha256')!=dense_contract_digest(c)
            or prepared.get('inputs')!=c['inputs'] or type(prepared.get('artifacts')) is not dict
            or set(prepared['artifacts'])!=labels or label not in labels):
        raise ValueError('complete prepared18point grid/input/study binding required')
    digest=prepared.get('teacher_mps_digest')
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}',digest) is None:
        raise ValueError('exact qualified teacher checkpoint digest required')
    teacher=prepared['artifacts']['teacher_noop'];artifact=prepared['artifacts'][label]
    for entry in (teacher,artifact):
        if type(entry) is not dict:raise ValueError('prepared canonical artifact required')
        _reference({k:entry[k] for k in ('directory','manifest_sha256')},bundle=True)
        runtime=entry.get('runtime_sha256')
        if type(runtime) is not str or re.fullmatch('[0-9a-f]{64}',runtime) is None:
            raise ValueError('exact embedded runtime digest required')
    if artifact['runtime_sha256']!=teacher['runtime_sha256']:
        raise ValueError('teacher and candidate embedded runtimes differ')
    blocked=prepared.get('blocked_source_roots')
    if type(blocked) is not list or not blocked or len(set(blocked))!=len(blocked):
        raise ValueError('explicit distinct blocked scientific source roots required')
    for path in blocked:_canonical_path(path)
    counts={k:v for k,v in metric_work_plan(rows,mode=mode).items() if k!='components'}
    if type(resource) is not dict or set(resource)!={'wall_seconds','memory_bytes','output_disk_bytes',*counts}:
        raise ValueError('closed exact native resource/work plan required')
    for key,value in resource.items():
        if (type(value) not in ((int,float) if key=='wall_seconds' else (int,))
                or not math.isfinite(value) or value<=0):
            raise ValueError('finite positive native resources required')
    if any(resource[k]!=v for k,v in counts.items()):raise ValueError('native raw target/query work differs')
    limits=c['preparation_limits']
    if (resource['wall_seconds']>limits['phase_wall_seconds'] or resource['memory_bytes']>limits['memory_bytes']
            or resource['output_disk_bytes']>limits['family_disk_bytes']):
        raise ValueError('native resource plan exceeds prospective ceilings')
    if mode=='profile':
        if profile_binding is not None:raise ValueError('outcome-blind profile cannot import quality')
    else:
        if c['resource_freeze'] is None:raise ValueError('run requires a separately frozen measured resource capsule')
        _reference(profile_binding)
    output=_canonical_path(str(output));ledger=_canonical_path(str(ledger))
    if output.suffix!='.json' or ledger.suffix!='.jsonl' or output.parent!=ledger.parent:
        raise ValueError('one isolated result/ledger directory required')
    return {'schema_version':1,'study':c['study'],'study_contract_sha256':dense_contract_digest(c),'mode':mode,
            'artifact':{'path':artifact['directory'],'manifest_sha256':artifact['manifest_sha256']},
            'teacher':{'path':teacher['directory'],'manifest_sha256':teacher['manifest_sha256'],'mps_checkpoint_digest':digest},
            'inputs':c['inputs'],'numerical':{'chunk_size':32,'temperature':1,'negative_floor':2e-6,'decision_resolution':2e-5},
            'worker_sha256':c['worker']['sha256'],'runtime_sha256':artifact['runtime_sha256'],
            'blocked_source_roots':list(blocked),'output':str(output),'ledger':str(ledger),
            'resource':dict(resource),'profile_binding':profile_binding}



def dense_frontier_summary(contract,prepared,windows,measurements):
    """Arithmetic on independently bound, fixed fitted artifact observations.

    The caller verifies protocol, source, stage and cost-file hashes first.
    Imported observations retain their original study identity and preparation
    expense; this function never relabels them as new executions.
    """
    import math
    import numpy as np
    from provenance.sharing_frontier_study import metric_work_plan
    from poc.contracts import thaw
    c=thaw(contract);labels=list(REFERENCE_LABELS)+[r.label for r in dense_recipes()]
    validate_contract_shape(c,root=Path(c['out']).parent.parent)
    if (prepared.get('study_contract_sha256')!=dense_contract_digest(c) or prepared.get('inputs')!=c['inputs']
            or set(prepared.get('artifacts',{}))!=set(labels) or type(measurements) is not dict or set(measurements)!=set(labels)):
        raise ValueError('complete fixed18point measurement inventory and input/study binding required')
    work=metric_work_plan(windows,mode='run');targets=np.asarray([sum(w['target_mask']) for w in windows],dtype=np.int64)
    n=work['valid_target_presentations'];points={};successful={};failed=[]
    for label in labels:
        value=measurements[label]
        if type(value) is not dict:raise ValueError('explicit completed or failed outcome required')
        if value.get('status')=='failed':
            if label in REFERENCE_LABELS or set(value)!={'status','failure','reason'} or type(value['reason']) is not str or not value['reason']:
                raise ValueError('new algorithm failures need explicit durable reason; admitted reference cannot disappear')
            _reference(value['failure']);points[label]=dict(value);failed.append(label);continue
        if value.get('status')!='complete':raise ValueError('every attempted point needs explicit terminal status')
        a=prepared['artifacts'][label];rows=value.get('rows')
        if (type(a.get('complete_bytes')) is not int or a['complete_bytes']<=0
                or value.get('complete_bytes')!=a['complete_bytes'] or a.get('teacher_needed_at_inference') is not False):
            raise ValueError('actual complete independently loadable artifact census required')
        if type(rows) is not list or len(rows)!=len(windows):raise ValueError('complete128component observations required')
        for w,row in zip(windows,rows,strict=True):
            count=sum(w['target_mask'])
            if (type(row) is not dict or any(row.get(k)!=w[k] for k in ('document_id','component_id','window_sha256'))
                    or type(row.get('valid_targets')) is not int or row['valid_targets']!=count
                    or type(row.get('disagreements')) is not int or not 0<=row['disagreements']<=count):
                raise ValueError('component/window/target/disagreement observation mismatch')
            for key in ('kl_sum','nll_sum','teacher_nll_sum'):
                number=row.get(key)
                if type(number) not in (int,float) or not math.isfinite(number):raise ValueError('finite raw metric sums required')
            if row['kl_sum'] < -2e-6*count or row['nll_sum']<0 or row['teacher_nll_sum']<0:
                raise ValueError('raw metric outside independently qualified numerical floor')
        sums={k:math.fsum(row[k] for row in rows) for k in ('kl_sum','nll_sum','teacher_nll_sum')}
        sums.update(valid_targets=n,disagreements=sum(row['disagreements'] for row in rows))
        values=np.asarray([row['kl_sum']/row['valid_targets'] for row in rows])
        worst=sorted(rows,key=lambda row:row['kl_sum']/row['valid_targets'],reverse=True)[:10]
        points[label]={'status':'complete','complete_bytes':a['complete_bytes'],'within_named_ceiling':a['complete_bytes']<=66064401,
            'raw_sums':sums,'kl':sums['kl_sum']/n,'nll':sums['nll_sum']/n,'delta_nll':(sums['nll_sum']-sums['teacher_nll_sum'])/n,
            'perplexity':math.exp(sums['nll_sum']/n) if sums['nll_sum']/n<709 else None,'disagreement_rate':sums['disagreements']/n,
            'macro_component_kl':float(values.mean()),'component_kl_quantiles':dict(zip(('q50','q90','q95','q99'),np.quantile(values,[.5,.9,.95,.99]).tolist())),
            'worst_components':[{'component_id':row['component_id'],'valid_targets':row['valid_targets'],'kl':row['kl_sum']/row['valid_targets']} for row in worst],
            'observation_origin':'imported_reference_with_original_identity' if label in REFERENCE_LABELS else 'new_study',
            'observation_contract':c['reference_config'] if label in REFERENCE_LABELS else {'scientific_contract_sha256':dense_contract_digest(c)}}
        successful[label]=rows
    teacher=successful['teacher_noop']
    if abs(points['teacher_noop']['kl'])>2e-5:raise ValueError('teacher-equivalent numerical qualification unresolved')
    for rows in successful.values():
        for row,reference in zip(rows,teacher,strict=True):
            if abs(row['teacher_nll_sum']-reference['teacher_nll_sum'])>2e-5*row['valid_targets']:
                raise ValueError('teacher targets differ across compared observations')
    for label in successful:
        point=points[label]
        point['dominated_by']=[other for other in successful if other!=label
            and points[other]['complete_bytes']<=point['complete_bytes'] and points[other]['kl']<=point['kl']
            and (points[other]['complete_bytes']<point['complete_bytes'] or points[other]['kl']<point['kl'])]
    draws=np.random.default_rng(17).integers(0,len(windows),(10000,len(windows)))
    contrasts=[]
    for recipe in dense_recipes():
        if recipe.label not in successful:continue
        for baseline in (('dense_gptq4','dense_rtn4') if recipe.bits==4 else ('dense_gptq2_embedding4','dense_rtn2_embedding4')):
            d=np.asarray([a['kl_sum']-b['kl_sum'] for a,b in zip(successful[recipe.label],successful[baseline],strict=True)])
            sample=d[draws].sum(axis=1)/targets[draws].sum(axis=1)
            contrasts.append({'label':recipe.label+'_minus_'+baseline,'direction':'negative favors new dense baseline',
                'kl_difference':float(d.sum()/n),'paired_component_percentile_95':np.quantile(sample,[.025,.975]).tolist(),
                'better_components':int((d<0).sum()),'worse_components':int((d>0).sum()),'tied_components':int((d==0).sum()),
                'bootstrap_draws':10000,'bootstrap_seed':17,
                'uncertainty':'paired source components conditional on fixed artifacts; exploratory multiple contrasts; no calibration/training-seed replication'})
    return {'schema_version':1,'study':c['study'],'study_contract_sha256':dense_contract_digest(c),
        'named_ceiling_bytes':66064401,'components':len(windows),'valid_targets':n,'points':points,'contrasts':contrasts,
        'failed_labels':failed,'frontier_labels':[label for label in successful if not points[label]['dominated_by']],
        'scope':{'population':'already exposed development; no protected confirmation','capability_acceptability_established':False,
            'production_runtime_established':False,'preparation_matched':False,
            'comparison':'six fixed new dense recipes with64fit calibration versus12 imported independently admitted matched-runtime/population observations; original preparation costs retained',
            'selection':'no outcome-based grid or checkpoint selection; all failures and adverse points retained',
            'private_factors':'BF16 stored, FP32 computed in imported packed shared references'}}



def dense_preparation_plan(recipe,*,profile):
    """Completed short or full original algorithm schedule, never a prefix."""
    if not isinstance(recipe,DenseRecipe) or type(profile) is not bool:raise ValueError('fixed recipe and explicit profile flag required')
    train=2 if profile else 64;internal=(2 if profile else 16) if recipe.method=='dwq' else 0
    updates=train if recipe.method=='dwq' else 0;losses=train+2*internal if updates else 0
    # Counted adapter includes teacher/official bridge and both roundtrip views.
    # Eight additional synthetic offline-loader calls are separately reserved.
    adapter_views=2*losses+4 if updates else None
    views_cap=adapter_views if updates else (train+4)*64
    teacher=losses+2 if updates else views_cap
    student=losses+2+8 if updates else views_cap+8
    work={'updates':updates,'valid_target_presentations':losses*511,'teacher_forward_calls':teacher,
          'student_forward_calls':student,'student_backward_calls':updates,
          'raw_corpus_target_presentations':views_cap*511}
    return {'schema_version':1,'profile':profile,'train_components':train,'internal_components':internal,'updates':updates,
            'expected_scored_targets':losses*511,'logical_optimizer_raw_targets':updates*511,
            'expected_adapter_raw_targets':None if adapter_views is None else adapter_views*511,'work':work,
            'query_policy':'DWQ exact completed update/loss counts; noniterative adapter invocation allowance conservative, all actual dispatches charged',
            'offline_policy':'eight synthetic loader/cache-parity calls reserved separately; no extra natural-corpus exposure'}


def dense_preparation_forecast(recipe,profile):
    """Conservative resource-only extrapolation, before final native outcomes."""
    import math
    keys={'schema_version','recipe','profile','plan','wall_seconds','complete_bytes','memory_peak_bytes','actual_work','candidate_quality_disclosed'}
    if (type(profile) is not dict or set(profile)!=keys or profile['schema_version']!=1 or profile['profile'] is not True
            or profile['recipe']!=recipe.identity() or profile['plan']!=dense_preparation_plan(recipe,profile=True)
            or profile['candidate_quality_disclosed'] is not False):raise ValueError('closed resource-only completed short profile required')
    for key in ('wall_seconds','complete_bytes','memory_peak_bytes'):
        value=profile[key]
        if type(value) not in ((int,float) if key=='wall_seconds' else (int,)) or not math.isfinite(value) or value<=0:
            raise ValueError('finite actual resource observations required')
    actual=profile['actual_work'];allowed=profile['plan']['work']
    if type(actual) is not dict or set(actual)!=set(allowed) or any(type(v) is not int or v<0 or v>allowed[k] for k,v in actual.items()):
        raise ValueError('actual complete profile work outside finite allowance')
    for key in ('updates','student_backward_calls','valid_target_presentations'):
        if actual[key]!=allowed[key]:raise ValueError('short completed schedule was censored or doublecharged')
    if recipe.method=='dwq' and actual['raw_corpus_target_presentations']!=profile['plan']['expected_adapter_raw_targets']:
        raise ValueError('DWQ native/official model context work incomplete')
    # Multiply the entire measured short phase, including setup/export, by32;
    # then apply2x and120s overhead. This intentionally overprices fixed setup.
    forecast=profile['wall_seconds']*32
    wall=int(math.ceil((2*forecast+120)/30)*30)
    if wall>PREPARATION_LIMITS['phase_wall_seconds'] or profile['memory_peak_bytes']>PREPARATION_LIMITS['memory_bytes']:
        raise ValueError('full completed preparation infeasible under prospective resource ceilings')
    return {'schema_version':1,'recipe':recipe.identity(),'plan':dense_preparation_plan(recipe,profile=False),
            'resource':{'wall_seconds':wall,'memory_bytes':PREPARATION_LIMITS['memory_bytes'],'disk_bytes':PREPARATION_LIMITS['family_disk_bytes']},
            'forecast_wall_seconds':forecast,'policy':'2x entire short completed phase scaled32, plus120s, roundedup30s; reject infeasibility, never truncate updates',
            'candidate_quality_used':False}


def dense_quantization_spec(recipe,*,profile,destination):
    from poc.quantize import QuantizationSpec
    plan=dense_preparation_plan(recipe,profile=profile);_canonical_path(str(destination))
    return QuantizationSpec(recipe.method,recipe.bits,recipe.group_size,str(destination),
        calibration_count=plan['train_components'],internal_validation_count=plan['internal_components'],
        length=512,seed=17,embedding_bits=4)


def validate_dense_quantizer_evidence(recipe,plan,proof,evidence):
    """Completed actual algorithm and export parity, before native scoring."""
    import math
    if type(plan) is not dict or plan!=dense_preparation_plan(recipe,profile=plan.get('profile')):
        raise ValueError('exact short/full completed preparation plan required')
    train=proof.get('train_component_ids');internal=proof.get('internal_component_ids')
    if (type(train) is not list or len(train)!=plan['train_components'] or type(internal) is not list
            or len(internal)!=plan['internal_components'] or len(set(train+internal))!=len(train+internal)):
        raise ValueError('exact distinct train/fit-internal components required')
    expected={'method':recipe.method,'bits':recipe.bits,'embedding_bits':4,'group_size':64,'seed':17,
        'accounting_profile':'dense-native-contexts-v2','tokenization_policy':'no-special-tokens-v2','original_context_tokens':512,
        'default_loader_denial_at_imported_binding':True,'socket_connect_denied':True,
        'internal_train_component_ids':train,'internal_valid_component_ids':internal,'calibration_component_ids':train+internal}
    if type(evidence) is not dict or any(evidence.get(k)!=v for k,v in expected.items()):
        raise ValueError('actual quantizer method/population/context/network evidence differs')
    for key,ceiling in (('initial_official_bridge_max_abs_error',.005),('packed_common_runtime_max_abs_error',.01)):
        value=evidence.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=ceiling:
            raise ValueError('actual official/native export parity unresolved')
    counts=evidence.get('actual_upstream_counts')
    if type(counts) is not dict:raise ValueError('actual native algorithm counters required')
    expected_counts={'updates':plan['updates'],'backwards':plan['updates'],'scored_loss_targets':plan['expected_scored_targets'],
                     'logical_optimizer_raw_target_presentations':plan['logical_optimizer_raw_targets']}
    if any(type(counts.get(k)) is not int or counts[k]!=v for k,v in expected_counts.items()):
        raise ValueError('completed update/loss/logical optimizer exposure differs')
    for key in ('forward_raw_corpus_presentations','native_raw_corpus_presentations'):
        if type(counts.get(key)) is not int or counts[key]<0:raise ValueError('actual native and official context exposure required')
    total=evidence.get('total_model_context_raw_presentations')
    if type(total) is not int or total!=counts['forward_raw_corpus_presentations']+counts['native_raw_corpus_presentations']:
        raise ValueError('native/official raw context accounting differs')
    if recipe.method=='dwq':
        from poc.quantize import validate_dense_dwq_scope_record
        validate_dense_dwq_scope_record(counts.get('dwq_trainability'))
        if type(evidence.get('temperature')) not in (int,float) or evidence['temperature']!=1. or total!=plan['expected_adapter_raw_targets']:
            raise ValueError('DWQ distribution temperature or completed context work differs')
    elif evidence.get('temperature') is not None:raise ValueError('noniterative method has undeclared temperature')


def verify_dense_parent_selection(parent,child,*,evaluation_components):
    """Prove the fixed80 cohort is derived from the named original512 pool."""
    from dataclasses import asdict
    if (len(parent.records)!=512 or len(set(parent.membership.values()))!=512 or set(parent.roles.values())!={'fit'}
            or parent.quarantine or child.quarantine or child.lineage_sha256!=parent.digest() or child.salt!=parent.salt
            or child.exposure_path==parent.exposure_path):
        raise ValueError('exact512fit parent and distinct journal/verified lineage required')
    plan=calibration_plan(parent,evaluation_components=evaluation_components)
    ids=plan['train_document_ids']+plan['internal_document_ids'];records={r.document_id:r for r in parent.records}
    if ([asdict(r) for r in child.records]!=[asdict(records[d]) for d in ids] or dict(child.roles)!={d:'fit' for d in ids}
            or dict(child.membership)!={d:parent.membership[d] for d in ids}):
        raise ValueError('complete child records/roles/membership differ from first80 of original512parent')
    return {**plan,'parent_components':512,'parent_manifest_digest':parent.digest(),'child_manifest_digest':child.digest(),
            'scope':'metadata-only original512-to-first80 proof; no raw content or token read'}


def verify_dense_parent_derivation(contract,*,evaluation_components):
    """Versioned source-bound derivation metadata, checked before fit content."""
    from poc.recovery_endpoints import finite_json,verify_refs,file_sha
    from poc.data.manifest import load_manifest
    root=Path(contract['out']).parent.parent;path=root/'provenance'/contract['study']/'fit-parent-derivation-v1.json'
    ref={'path':str(path),'sha256':file_sha(path)};verify_refs([ref]);d=finite_json(path)
    if (type(d) is not dict or set(d)!={'schema_version','study','parent_manifest','child_manifest','rule'}
            or type(d['schema_version']) is not int or d['schema_version']!=1 or d['study']!=contract['study']
            or d['child_manifest']!=contract['fit_manifest'] or d['rule']!=contract['calibration_order']):
        raise ValueError('exact named parent/child derivation descriptor required')
    _reference(d['parent_manifest']);_reference(d['child_manifest']);verify_refs([d['parent_manifest'],d['child_manifest']])
    parent=load_manifest(Path(d['parent_manifest']['path']),expected_sha256=d['parent_manifest']['sha256'],roles={'fit'})
    child=load_manifest(Path(d['child_manifest']['path']),expected_sha256=d['child_manifest']['sha256'],roles={'fit'})
    plan=verify_dense_parent_selection(parent,child,evaluation_components=evaluation_components)
    return child,{**plan,'derivation':ref,'parent_manifest':d['parent_manifest'],'child_manifest':d['child_manifest']}

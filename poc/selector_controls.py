"""Constructed end-to-end selector sensitivity; never pretrained evidence."""
from dataclasses import asdict,replace
import hashlib,json,math,os,subprocess,time,re
from pathlib import Path

import torch

from poc.bank import make_bank,canonicalize_partition,PROJECTIONS
from poc.budget import BudgetTracker
from poc.contracts import Batch,Budget,CostRecord,RunSpec
from poc.data.access import open_records
from poc.data.manifest import DataManifest,DocumentRecord
from poc.diagnostics import fixed_scores
from poc.losses import stream_metrics
from poc.probes import (ProbePolicy,ProbePool,PathologyObservation,fit_pathology_thresholds,
    generate_donors,filter_pathologies,prefix_for_probe,apply_probe,local_discrepancy,mine_probes)
from poc.records import canonical_json,start_attempt,append_event
from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder,EXECUTION_PROFILE
from poc.train import _Dispatch,_call,finish_attempt_cost
from poc.continuation_budget import process_peak_bytes


def finish_attempt(attempt,status,*,reason):
    append_event(attempt,{'kind':'attempt_finished','status':status,'reason':reason})
    path=attempt.path/'run.json';raw=json.loads(path.read_text());raw['status']=status
    if status=='failed':raw.update(failure_category='execution',failure_details=reason)
    temporary=path.with_suffix('.pending')
    with temporary.open('xb') as stream:
        stream.write(canonical_json(raw));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def hashed(value):return hashlib.sha256(canonical_json(value)).hexdigest()


class CharacterTokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}


def batch(tokens,document,component):
    ids=torch.tensor([tokens],dtype=torch.int64)
    return Batch(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,len(tokens)-1),dtype=torch.bool),
        torch.arange(len(tokens))[None],(document,),(component,))


def qualified_metric(metric):
    if (type(metric.valid_targets) is not int or metric.valid_targets<1
            or not math.isfinite(metric.kl) or not math.isfinite(metric.nll)
            or metric.kl < -1e-10*metric.valid_targets or metric.nll<0):
        raise ValueError('owned metric is nonfinite, empty or below qualified negative KL floor')
    return metric


def planted_models(regime,seed):
    teacher=make_tiny_decoder(TinyConfig(vocab_size=128,hidden_size=8,intermediate_size=12,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64),seed=seed,dtype=torch.float64)
    teacher.tokenizer_digest=hashed({'tokenizer':'owned-character-ascii-v1'})
    with torch.no_grad():
        for p in teacher.parameters():p.zero_()
        teacher.embed_tokens.weight[:,0]=1
        teacher.embed_tokens.weight[ord('9'),0]=0;teacher.embed_tokens.weight[ord('9'),1]=1
        teacher.norm.weight.fill_(1)
        for layer in teacher.layers:
            layer.input_layernorm.weight.fill_(1);layer.post_attention_layernorm.weight.fill_(1)
        rare=teacher.layers[0].mlp
        rare.gate_proj.weight[0,1]=1;rare.up_proj.weight[0,1]=1
        rare.down_proj.weight[0,0]=1 if regime=='hidden_failure' else 0
        ordinary=teacher.layers[2].mlp
        ordinary.gate_proj.weight[0,0]=1;ordinary.up_proj.weight[0,0]=1
        ordinary.down_proj.weight[1,0]=.01
    candidates={}
    for name,groups,parent in (('rare_failure',[[0,1],[2]],1),('ordinary_damage',[[0],[1,2]],1)):
        model=make_bank(teacher,canonicalize_partition(groups,3),0,seed)
        affected=(0,1) if name=='rare_failure' else (1,2)
        index=model.partition.reuse_map[affected[0]]
        with torch.no_grad():
            for proj in PROJECTIONS:
                getattr(model.bank[index],proj).weight.copy_(getattr(teacher.layers[parent].mlp,proj).weight)
        model.requires_grad_(False);candidates[name]=(model,affected)
    return teacher,candidates


def _source_reference(reference):
    if type(reference) is not dict or set(reference)!={'path','sha256'}:
        raise ValueError('closed source reference required')
    path=Path(reference['path'])
    if (not path.is_absolute() or str(path)!=reference['path'] or path.resolve()!=path
            or '..' in path.parts or not path.is_file() or path.stat().st_nlink!=1):
        raise ValueError('canonical single-link source file required')
    if not isinstance(reference['sha256'],str) or re.fullmatch('[0-9a-f]{64}',reference['sha256']) is None:
        raise ValueError('source reference SHA256 required')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=reference['sha256']:
        raise ValueError('frozen source reference changed')
    return data


def verify_selector_source_binding(root,reference):
    """Explicit v2 lineage for composed source; prior features are not a full qualification."""
    raw=json.loads(_source_reference(reference))
    if (type(raw) is not dict or set(raw)!={'schema_version','files','parent_qualifications'}
            or type(raw['schema_version']) is not int or raw['schema_version']!=2
            or type(raw['files']) is not dict or not raw['files']
            or type(raw['parent_qualifications']) is not list or not raw['parent_qualifications']):
        raise ValueError('closed source binding version2 required')
    required={p.relative_to(root).as_posix() for p in (root/'poc').rglob('*.py')}
    required.update(('poc/selector_controls.py','tests/test_selector_controls.py','tests/test_selector_source_binding.py','provenance/selector_sensitivity_controls.py'))
    if not required<=set(raw['files']):raise ValueError('source binding omits current Python dependency/driver')
    for name,expected in raw['files'].items():
        if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('source dependency escapes checkout')
        path=root/name
        if not path.resolve().is_relative_to(root):raise ValueError('source dependency escapes checkout')
        _source_reference({'path':str(path),'sha256':expected})
    seen=set()
    for parent in raw['parent_qualifications']:
        value=json.loads(_source_reference(parent))
        if parent['path'] in seen:raise ValueError('duplicate feature qualification')
        seen.add(parent['path'])
        status=value.get('status')
        if not isinstance(status,str) or not (status in {'qualified_owned_controls_passed','qualified_confirmation_access_implementation','qualified_replication_analysis'} or status.startswith('cpu_source_qualified;')):
            raise ValueError('unqualified feature ancestor')
    return dict(raw['files']),{'source_binding_v2':reference['sha256'],
        'feature_qualifications':hashed({'references':raw['parent_qualifications']})}


def run_owned_selector_control(out,*,regime,seed,allow_trigger=True,source_binding=None):
    if regime not in ('hidden_failure','no_added_information') or type(seed) is not int or seed<0 or type(allow_trigger) is not bool:
        raise ValueError('closed owned regime, nonnegative seed and boolean trigger policy required')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('owned control output is not empty')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(1)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root=Path(__file__).resolve().parents[1]
    if source_binding is None:
        imported=json.loads((root/'provenance/selector-pipeline-owned-v1/qualified-parent-import.json').read_text())
        source_files=dict(imported['files'])
        for name in ('poc/selector_controls.py','tests/test_selector_controls.py','provenance/selector_sensitivity_controls.py'):
            source_files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name,expected in source_files.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('owned control dependency source changed: '+name)
        parent=imported['parent_qualification']
        if hashlib.sha256(Path(parent['path']).read_bytes()).hexdigest()!=parent['sha256']:
            raise ValueError('owned control parent qualification changed')
        lineage={'parent_qualification':parent['sha256']}
    else:
        source_files,lineage=verify_selector_source_binding(root,source_binding)
    source_identity=hashed(source_files)
    contract={'schema_version':1 if source_binding is None else 2,'regime':regime,'seed':seed,'allow_trigger':allow_trigger,
        'dtype':'float64','device':'cpu','out':str(out.resolve()),'source_files_sha256':source_identity,'scope':'owned pipeline sensitivity; no pretrained or compression inference',
        'selection_formula':'N+0.5*(S+selected_donor_KL); S is mean affected-depth teacher-prefix discrepancy',
        'acquisition':'separately generated/scored active and seeded-random pools; one selected row per candidate',
        'numerical_resolution':1e-10,'negative_kl_floor_per_target':1e-10,'construction':'rare token9 neuron plus small ordinary response; rank0 actual canonical bank candidates'}
    limits=Budget(0,1000000,100000,0,0,100000,180,180,128*1024**2,2*1024**3)
    if source_binding is not None:contract['source_binding_v2']=source_binding['sha256']
    identity=RunSpec('R','owned_tiny_bank_selector_control' if source_binding is None else 'owned_tiny_bank_selector_control_v2',contract,subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        source_identity,hashed({'torch':str(torch.__version__),'cpu':True}),{'source':source_sha,'executed_sources':source_identity,**lineage},
        {'device':'cpu','dtype':'float64'},{'owned':seed},(),limits)
    attempt=start_attempt(identity,out/'ledgers');rec=BudgetTracker(limits,attempt)
    try:
        teacher,candidates=planted_models(regime,seed)
        initial={'teacher':teacher.manifest().checkpoint_digest,**{name:m.manifest().checkpoint_digest for name,(m,_) in candidates.items()}}
        tokenizer=CharacterTokenizer();populations={}
        texts={'fit':('Clara 8 dog.','Clara 9 dog.'),'score':('Alice 8 cat.','Bob 8 cat.'),
            'damage':('9 9 9 9 9.','9 9 9 9 9!')}
        for name,items in texts.items():
            api_role,purpose={'fit':('fit','fit'),'score':('probe_score','ranking'),'damage':('dev','selection')}[name]
            records=tuple(DocumentRecord(name+str(i),'owned','owned-v1','owned',i,text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest()) for i,text in enumerate(items))
            manifest=DataManifest(records,{r.document_id:api_role for r in records},
                {r.document_id:name+'-component'+str(i) for i,r in enumerate(records)},'owned-selector-v1',str(out/(name+'-exposure.jsonl')))
            metadata_dir=out/'populations';metadata_dir.mkdir(exist_ok=True)
            (metadata_dir/(name+'.json')).write_bytes(canonical_json(asdict(manifest)))
            materialized=tuple(open_records(manifest,purpose=purpose))
            rows=[]
            for r in materialized:
                encoded=_call(rec,lambda:tokenizer(r.text),processed_inputs=1,processed_tokens=len(r.text))
                rows.append(batch(encoded['input_ids'],r.document_id,manifest.membership[r.document_id]))
            populations[name]=(manifest,materialized,tuple(rows))
        # All calibration bounds are actual frozen-teacher per-position values
        # on a separate fit-role population, covering ordinary/rare inputs.
        observations=[]
        for b in populations['fit'][2]:
            logits=_call(rec,lambda:teacher.logits(b),teacher_forward_calls=1)
            logp=logits.log_softmax(-1);entropy=-(logp.exp()*logp).sum(-1)
            for depth in range(3):
                boundary=prefix_for_probe(teacher,b,depth,source='teacher',recorder=rec)
                for i in range(len(b.input_ids[0])-1):
                    observations.append(PathologyObservation(b.document_ids[0],b.cluster_ids[0],'fit',
                        float(boundary.u[0,i].norm()),float(entropy[0,i]),depth))
        thresholds=fit_pathology_thresholds(observations,lower_quantile=0.,upper_quantile=1.)
        metrics={};features={};probe_rows=[];proposals={'accepted':0,'rejected':0,'disabled_trigger':0};pools=[]
        def full_metric(model,b):
            n=int(b.target_mask.sum())
            rec.record(CostRecord(valid_target_presentations=n,raw_corpus_target_presentations=n,
                processed_inputs=1,processed_tokens=int(b.attention_mask.sum())))
            return qualified_metric(stream_metrics(_Dispatch(teacher,rec,'teacher'),_Dispatch(model,rec,'student'),b,32))
        equivalent=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),0,seed).requires_grad_(False)
        no_op=[full_metric(equivalent,b).kl/int(b.target_mask.sum()) for b in populations['fit'][2]]
        if any(abs(v)>1e-10 for v in no_op):raise ValueError('owned no-sharing numerical control failed')
        numerical={'teacher_equivalent_kl_raw_per_target':no_op,'floor_per_target':1e-10,'resolution':1e-10,'direct_chunk_errors':{}}
        from poc.losses import kl_sum
        reference_batch=populations['fit'][2][-1]
        for name,(model,_) in candidates.items():
            streamed=full_metric(model,reference_batch)
            target=_call(rec,lambda:teacher.logits(reference_batch),teacher_forward_calls=1)
            actual=_call(rec,lambda:model.logits(reference_batch),student_forward_calls=1)
            n=int(reference_batch.target_mask.sum());rec.record(CostRecord(valid_target_presentations=n))
            direct=float(kl_sum(target[:,:-1],actual[:,:-1],reference_batch.target_mask))
            error=abs(direct-streamed.kl)/n
            if not math.isfinite(error) or error>1e-10:raise ValueError('owned direct/chunked numerical parity failed')
            numerical['direct_chunk_errors'][name]=error
        for name,(model,depths) in candidates.items():
            metrics[name]={role:[asdict(full_metric(model,b)) for b in populations[role][2]] for role in ('score','damage')}
            natural=math.fsum(row['kl'] for row in metrics[name]['score'])/sum(row['valid_targets'] for row in metrics[name]['score'])
            local=[]
            for depth in depths:
                values=[]
                for b in populations['score'][2]:
                    boundary=prefix_for_probe(teacher,b,depth,source='teacher',recorder=rec)
                    rec.record(CostRecord(valid_target_presentations=int(b.target_mask.sum())))
                    values.append(local_discrepancy(teacher,model.layers[depth].mlp,boundary,32,recorder=rec))
                local.append(math.fsum(v.kl for v in values)/sum(v.valid_targets for v in values))
            selected={}
            for mode in ('active','seeded_random'):
                manifest,documents,_=populations['score']
                policy=ProbePolicy(mode,seed,1,3,0.,(tuple(depths),),(1.,),1,64,8,1024**2,thresholds,
                    manifest.roles,manifest.membership,('probe_score',),'rank',initial['teacher'],initial[name],
                    teacher.tokenizer_digest,'float64','torch_cpu','post_attention_v1',EXECUTION_PROFILE,64,rec)
                generated=tuple(generate_donors(documents,tokenizer,policy));pool=ProbePool(generated)
                scores={};pathologies={};disabled={}
                for row in generated:
                    proposals[row.status]+=1
                    if row.status!='accepted':continue
                    recipient=batch(row.recipient_tokens,row.recipient_id,row.recipient_component)
                    donor=batch(row.donor_tokens,row.donor_id,row.donor_component)
                    rb=prefix_for_probe(teacher,recipient,row.depth,source='teacher',recorder=rec)
                    db=prefix_for_probe(teacher,donor,row.depth,source='teacher',recorder=rec)
                    patched=apply_probe(rb,db,row)
                    rec.record(CostRecord(valid_target_presentations=int(patched.batch.target_mask.sum())))
                    value=qualified_metric(local_discrepancy(teacher,model.layers[row.depth].mlp,patched,32,recorder=rec))
                    scores[row.probe_id]=value.kl/value.valid_targets
                    logits=_call(rec,lambda:teacher.suffix_logits(teacher.ff_residual(patched),patched),teacher_forward_calls=1)
                    lp=logits.log_softmax(-1);ent=-(lp.exp()*lp).sum(-1)
                    pathologies[row.probe_id]=(float(patched.u[0,:-1].norm(dim=-1).mean()),float(ent[0,:-1].mean()))
                    probe_rows.append({'candidate':name,'mode':mode,'probe':asdict(row),'metric':asdict(value)})
                    if not allow_trigger and row.edit_kind=='number' and row.replacement=='9':
                        disabled[row.probe_id]=('predeclared_trigger_disabled',);proposals['disabled_trigger']+=1
                filtered=filter_pathologies(pool,pathologies,thresholds)
                filtered=replace(filtered,exclusions={**dict(filtered.exclusions),**disabled})
                mined=mine_probes(filtered,scores,policy)
                if mined.selection_status!='complete':raise ValueError('owned acquisition underfilled')
                selected[mode]=math.fsum(scores[k] for k in mined.selected_ids)/len(mined.selected_ids)
                pools.append({'candidate':name,'mode':mode,'records':[asdict(r) for r in generated],
                    'exclusions':dict(mined.exclusions),'selected_ids':list(mined.selected_ids)})
            features[name]={'N':natural,'S':math.fsum(local)/len(local),'A':selected['active'],'R':selected['seeded_random']}
        scores={name:fixed_scores(natural=f['N'],student=f['S'],active_donor=f['A'],random_donor=f['R']) for name,f in features.items()}
        choose=lambda key:min(sorted(scores),key=lambda name:scores[name][key])
        damage={name:math.fsum(r['kl'] for r in metrics[name]['damage'])/sum(r['valid_targets'] for r in metrics[name]['damage']) for name in metrics}
        damage_choice=min(sorted(damage),key=lambda name:damage[name]);natural_choice=choose('natural_kl');active_choice=choose('active_1')
        unchanged=(initial=={'teacher':teacher.manifest().checkpoint_digest,**{name:m.manifest().checkpoint_digest for name,(m,_) in candidates.items()}})
        if not unchanged:raise ValueError('owned scoring mutated a model')
        finish_attempt_cost(rec,elapsed=time.monotonic()-started)
        rec.record(CostRecord(memory_bytes=process_peak_bytes(),memory_unavailable_reason=None,
            disk_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))
        result={'schema_version':1,'scope':contract['scope'],'study_contract_sha256':hashed(contract),
            'source_sha256':source_sha,'executed_source_files':source_files,'population_manifest_sha256':{name:population[0].digest() for name,population in populations.items()},'initial_model_digests':initial,'unchanged_model_digests':unchanged,
            'natural_choice':natural_choice,'active_choice':active_choice,'random_choice':choose('random_donor'),
            'damage_choice':damage_choice,'selector_improved':damage[active_choice]<damage[natural_choice]-1e-10,
            'scores':scores,'features':features,'damage':damage,'raw_component_metrics':metrics,'probe_metrics':probe_rows,
            'pools':pools,'proposal_counts':proposals,'fit_thresholds_sha256':thresholds.digest,'numerical_qualification':numerical,
            'score_components':list(populations['score'][0].membership.values()),
            'damage_components':list(populations['damage'][0].membership.values()),
            'source_exposure_journal':populations['score'][0].exposure_path,'cost':rec.totals.to_dict(),
            'authoritative_cost_ledger':str(rec.path),'cost_snapshot_scope':'all neural work complete; before result/terminal metadata publication; external guard supplies complete wall/disk','cost_units':'raw counts are full-natural scored positions including repeats; local scored targets separately charged; call counts include prefix/suffix/readout calls, not equivalent full passes; generated edits plus examined mining rows are search proposals; no training updates'}
        if source_binding is not None:result.update(schema_version=2,source_binding=dict(source_binding))
        payload=canonical_json(result);rec.check(CostRecord(disk_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(payload)))
        (out/'result.json').write_bytes(payload)
        finish_attempt(attempt,'exhausted',reason='owned pipeline completed; not a pretrained result')
        return result
    except BaseException as exc:
        accounting_error=None
        try:finish_attempt_cost(rec,elapsed=time.monotonic()-started)
        except BaseException as final_cost_error:
            # BudgetTracker retains actual spending before raising. Terminal
            # metadata must still be published, preserving the original error.
            accounting_error=type(final_cost_error).__name__+': '+str(final_cost_error)
        reason=type(exc).__name__+': '+str(exc)
        if accounting_error:reason+='; final accounting: '+accounting_error
        try:finish_attempt(attempt,'failed',reason=reason)
        except BaseException as terminal_error:
            exc.add_note('Terminal publication also failed: '+repr(terminal_error))
        if accounting_error:exc.add_note('Final accounting: '+accounting_error)
        raise


def run_owned_selector_suite(out,*,source_binding=None):
    """Five fixed cases under an external180-second/2GiB/128MiB guard."""
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('owned suite output already exists')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    cases=[('positive-seed'+str(seed),'hidden_failure',seed,True) for seed in (17,23,31)]
    cases += [('trigger-disabled-seed17','hidden_failure',17,False),('negative-seed17','no_added_information',17,True)]
    contract={'schema_version':1 if source_binding is None else 2,'study':'selector-tiny-bank-controls-v1' if source_binding is None else 'selector-tiny-bank-controls-integrated-v2','cases':cases,
        'scope':'owned pipeline sensitivity; no pretrained or compression inference',
        'wall_seconds':180,'process_memory_bytes':2*1024**3,'output_disk_bytes':128*1024**2,
        'primary_gate':'active exposes eligible planted failure on all declared seeds; disabled and no-information controls do not claim benefit',
        'random_gate':'reported separately; planted sensitivity does not require or establish active superiority over random'}
    if source_binding is not None:
        verify_selector_source_binding(Path(__file__).resolve().parents[1],source_binding)
        contract['source_binding_v2']=source_binding['sha256']
    (out/'contract.json').write_bytes(canonical_json(contract));references={};random_successes=[]
    for name,regime,seed,trigger in cases:
        if time.monotonic()-started>180:raise ValueError('owned suite wall ceiling reached')
        if process_peak_bytes()>2*1024**3:raise ValueError('owned suite memory ceiling reached')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>128*1024**2:raise ValueError('owned suite disk ceiling reached')
        result=run_owned_selector_control(out/name,regime=regime,seed=seed,allow_trigger=trigger,**({} if source_binding is None else {'source_binding':source_binding}))
        if name.startswith('positive'):
            passed=result['natural_choice']!=result['damage_choice']==result['active_choice'] and result['selector_improved']
            if result['random_choice']==result['damage_choice']:random_successes.append(name)
        elif name.startswith('trigger-disabled'):
            passed=result['natural_choice']==result['active_choice']!=result['damage_choice'] and not result['selector_improved']
        else:passed=result['natural_choice']==result['active_choice']==result['damage_choice'] and not result['selector_improved']
        if not passed:raise ValueError('owned pipeline sensitivity gate failed: '+name)
        path=out/name/'result.json';references[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    summary={'schema_version':1,'status':'owned_controls_passed','scope':contract['scope'],
        'suite_contract_sha256':hashed(contract),'cases':references,'pretrained_model_calls':0,
        'random_also_exposes_failure':random_successes,'interpretation':'actual pipeline detects a planted failure; no active-over-random or pretrained decision claim',
        'wall_seconds_before_summary':time.monotonic()-started,'publication_cost_policy':'external finite guard supplies full process wall/memory/disk including final metadata; per-case ledgers retain all neural work'}
    if source_binding is not None:summary.update(schema_version=2,source_binding=dict(source_binding))
    (out/'result.json').write_bytes(canonical_json(summary))
    return summary

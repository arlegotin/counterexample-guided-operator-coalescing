"""Two-parent medoid trials with CPU SVD residuals and one common fitting path."""
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass,field
import hashlib
import weakref
import torch
from torch import nn
from poc.bank import BankDecoder,PROJECTIONS,canonicalize_partition,initialize_residual,_Factors
from poc.contracts import Batch,Budget,CostRecord,DecoderAdapter
from poc.data.manifest import DataManifest
from poc.train import TrainingConfig,TrainingSources,fit_model


@dataclass
class MergeTrial:
    model: BankDecoder
    teacher: DecoderAdapter
    pair: tuple[int,int]
    affected_depths: tuple[int,...]
    selected_parent: int
    medoid_scores: tuple[dict,...]
    initialized_digest: str
    fitted_digest: str|None=None
    sources: TrainingSources|None=None
    recorder: object|None=None
    fit_summary: dict=field(default_factory=dict)
    parent_policy: dict|None=None


def centered_linear_cka(x,y):
    x=x.detach().cpu().double();y=y.detach().cpu().double()
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or x.shape[0]<2:raise ValueError('CKA needs matched nondegenerate observations')
    x=x-x.mean(0);y=y-y.mean(0)
    numerator=(x.T@y).square().sum();denominator=(x.T@x).square().sum().sqrt()*(y.T@y).square().sum().sqrt()
    if not torch.isfinite(denominator) or denominator<=0:raise ValueError('CKA degenerate centered activation')
    return float((numerator/denominator).item())


def _call(recorder,fn,**counts):
    if recorder is None:return fn()
    from poc.train import _call as measured
    return measured(recorder,fn,**counts)


def _fit_batches(fit,teacher):
    """Natural windows require verified tokens after the allowed exposure reader."""
    import json
    from pathlib import Path
    from poc.data.access import open_records
    from poc.data.windows import TokenDocument,WindowPolicy,make_windows
    result=[]
    for row in open_records(fit,purpose='fit'):
        data=Path(row.token_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row.token_sha256:raise ValueError('fit token hash mismatch')
        raw=json.loads(data)
        if raw['document_id']!=row.document_id:raise ValueError('fit token document mismatch')
        tokens=raw['fields']['text']
        windows=make_windows(TokenDocument(row.document_id,tuple(tokens)),WindowPolicy())
        if not windows or not windows[0].target_indices:raise ValueError('empty fit window')
        from poc.teacher_validation import _batch
        w=windows[0]
        result.append(_batch(w.token_ids,device=teacher.device,document_id=row.document_id,cluster_id=fit.membership[row.document_id],targets=w.target_indices))
    return tuple(result)


def initialize_merge(current:BankDecoder,teacher:DecoderAdapter,pair:tuple[int,int],fit:DataManifest,seed:int,*,batches:Sequence[Batch]|None=None,recorder=None,forced_parent:int|None=None)->MergeTrial:
    if not isinstance(current,BankDecoder):raise ValueError('canonical bank required')
    if current.base_digest != teacher.manifest().checkpoint_digest:
        raise ValueError('merge teacher lineage differs from current base_digest')
    if any(role!='fit' for role in fit.roles.values()):raise PermissionError('medoid selection requires fit records')
    if not fit.records:raise ValueError('nonempty fit population required')
    if len(pair)!=2 or any(type(x) is not int for x in pair) or not 0<=pair[0]<pair[1]<len(current.layers):raise ValueError('canonical depth pair required')
    groups=current.partition.groups;indices=tuple(current.partition.reuse_map[d] for d in pair)
    if indices[0]==indices[1]:raise ValueError('pair already shares an operator')
    representatives={groups[i][0] for i in indices}
    if forced_parent is not None and (type(forced_parent) is not int or forced_parent not in representatives):
        raise ValueError('forced parent must be a canonical current group representative')
    affected=tuple(sorted(groups[indices[0]]+groups[indices[1]]))
    batches=tuple(batches) if batches is not None else _fit_batches(fit,teacher)
    if not batches or set(i for b in batches for i in b.document_ids)!=set(fit.roles):raise ValueError('exact fit document population required')
    h=hashlib.sha256(fit.digest().encode())
    for b in batches:
        if any(fit.membership[i]!=c for i,c in zip(b.document_ids,b.cluster_ids,strict=True)):raise ValueError('fit component identity mismatch')
        for tensor in (b.input_ids,b.attention_mask,b.target_mask,b.positions):h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    fit_digest=h.hexdigest();candidate=deepcopy(current)
    merged=canonicalize_partition([g for k,g in enumerate(groups) if k not in indices]+[affected],len(current.layers))
    old_owners={g:current.bank[k] for k,g in enumerate(groups)}
    candidate.bank=nn.ModuleList([deepcopy(current.bank[indices[0]] if g==affected else old_owners[g]) for g in merged.groups])
    candidate.partition=merged;merged_index=merged.reuse_map[pair[0]]
    for depth,layer in enumerate(candidate.layers):
        layer.mlp.index=merged.reuse_map[depth];layer.mlp._owners_ref=weakref.ref(candidate.bank)
    def install(parent_index):
        with torch.no_grad():
            for name in PROJECTIONS:getattr(candidate.bank[merged_index],name).weight.copy_(getattr(current.bank[parent_index],name).weight)
        for depth in affected:
            residuals={}
            for k,name in enumerate(PROJECTIONS):
                if current.rank:
                    delta=getattr(teacher.layers[depth].mlp,name).weight.detach()-getattr(candidate.bank[merged_index],name).weight.detach()
                    a,b=_call(recorder,lambda:initialize_residual(delta,current.rank,(seed+depth*3+k)%2**63))
                    residuals[name]=_Factors(a,b)
            candidate.layers[depth].mlp.residuals=nn.ModuleDict(residuals)
    rows=[];best=None;best_delta=None
    for index in sorted(indices,key=lambda k:groups[k]):
        _call(recorder,lambda:None,search_proposals=1)
        install(index);sums=[]
        with torch.no_grad():
            for depth in affected:
                for b in batches:
                    boundary=_call(recorder,lambda:teacher.prefix_boundary(b,depth),teacher_forward_calls=1)
                    target=_call(recorder,lambda:teacher.ff_residual(boundary),teacher_forward_calls=1)
                    actual=_call(recorder,lambda:teacher.ff_residual(boundary,candidate.layers[depth].mlp),student_forward_calls=1)
                    mask=b.target_mask
                    error=(actual[:,:-1]-target[:,:-1]).detach().cpu().double().square().mean(-1)[mask.cpu()]
                    if not error.numel():raise ValueError('nonempty fit activation mask required')
                    sums.append(float(error.mean().item()))
        score=sum(sums)/len(sums);parent=groups[index][0]
        if not torch.isfinite(torch.tensor(score)):raise ValueError('nonfinite medoid reconstruction')
        rows.append(dict(parent=parent,mse=score,fit_digest=fit_digest,components=len(set(fit.membership.values())),documents=len(fit.records)))
        keep=(best is None or score<best[0]) if forced_parent is None else parent==forced_parent
        if keep:
            best=(score,index,parent)
            # Only changed residuals, not a second model/optimizer snapshot.
            best_delta={d:{k:v.detach().cpu().clone() for k,v in candidate.layers[d].mlp.residuals.state_dict().items()} for d in affected}
    if best[1]!=index:
        with torch.no_grad():
            for name in PROJECTIONS:getattr(candidate.bank[merged_index],name).weight.copy_(getattr(current.bank[best[1]],name).weight)
        for d,values in best_delta.items():candidate.layers[d].mlp.residuals.load_state_dict(values)
    return MergeTrial(candidate,teacher,pair,affected,best[2],tuple(rows),candidate.manifest().checkpoint_digest,recorder=recorder,
        parent_policy=None if forced_parent is None else {'schema_version':1,'kind':'forced_current_group_representative','parent':forced_parent})


def fit_trial(trial:MergeTrial,schedule:TrainingConfig,budget:Budget)->MergeTrial:
    if trial.fitted_digest is not None:raise ValueError('score-only trial cannot be refitted')
    if schedule.phase!='local' or tuple(schedule.affected_depths)!=trial.affected_depths:raise ValueError('common local schedule must cover exact merged group')
    if trial.sources is None or trial.recorder is None:raise ValueError('auditable training sources and recorder required')
    result=fit_model(trial.model,trial.teacher,trial.sources,schedule,budget,trial.recorder)
    trial.recorder.record(CostRecord())  # observe transient full model/Adam snapshot while still live
    if result.state.step!=schedule.max_updates or result.state.target_cursor!=schedule.max_updates*8192:raise ValueError('incomplete common exact-target fitting allowance')
    trial.fitted_digest=result.state.candidate_digest
    trial.fit_summary=dict(updates=result.state.step,targets=result.state.target_cursor,metrics=result.metrics)
    return trial


def initializer_sensitivity_control():
    """A real nonlinear permutation fixture separates capacity from raw-SVD error."""
    from poc.synthetic import make_planted_teacher
    from poc.bank import make_bank
    fixture=make_planted_teacher(23)
    partition=canonicalize_partition([[0,2,3,6],[1,4,5,7]],8)
    x=torch.cat((torch.eye(32,dtype=torch.float64),torch.randn((64,32),generator=torch.Generator().manual_seed(731),dtype=torch.float64)))
    errors={}
    with torch.no_grad():
        for label,rank in [('whole_medoid',0),('truncated_svd',2),('full_rank',32)]:
            candidate=make_bank(fixture.teacher,partition,rank,17)
            errors[label+'_max_error']=max(float((candidate.layers[d].mlp(x)-fixture.teacher.layers[d].mlp(x)).abs().max()) for d in range(8))
            del candidate
    return errors|{'interpretation':'raw SVD is not neuron alignment','fixture_seed':23,'evaluation_seed':731,'evaluation_positions':96}



def _validate_parent_policy(policy,current,pair,parent):
    representatives={current.partition.groups[current.partition.reuse_map[d]][0] for d in pair}
    if (not isinstance(policy,dict) or set(policy)!={'schema_version','kind','parent'} or
            type(policy['schema_version']) is not int or policy['schema_version']!=1 or
            policy['kind']!='forced_current_group_representative' or
            type(policy['parent']) is not int or policy['parent'] not in representatives or type(parent) is not int or policy['parent']!=parent):
        raise ValueError('invalid forced initialization parent policy')


def save_trial_delta(trial,current,path):
    """Keep only changed owners for delayed damage evaluation, never Adam/full model."""
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    if trial.fitted_digest!=trial.model.manifest().checkpoint_digest:raise ValueError('fitted trial changed')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    index=trial.model.partition.reuse_map[trial.pair[0]]
    selected={n:p.detach().cpu().contiguous() for n,p in trial.model.named_parameters() if n.startswith(f'bank.{index}.') or any(n.startswith(f'layers.{d}.mlp.residuals.') for d in trial.affected_depths)}
    save_file(selected,str(path/'weights.safetensors'))
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    metadata=dict(execution_profile=EXECUTION_PROFILE,base_digest=current.manifest().checkpoint_digest,teacher_digest=trial.teacher.manifest().checkpoint_digest,pair=trial.pair,affected_depths=trial.affected_depths,selected_parent=trial.selected_parent,medoid_scores=trial.medoid_scores,initialized_digest=trial.initialized_digest,fitted_digest=trial.fitted_digest,fit_summary=trial.fit_summary,weights_sha256=hashlib.sha256((path/'weights.safetensors').read_bytes()).hexdigest())
    metadata['owner_trainability']={n:p.requires_grad for n,p in trial.model.named_parameters()}
    if trial.parent_policy is not None:
        _validate_parent_policy(trial.parent_policy,current,trial.pair,trial.selected_parent)
        metadata.update(initialization_schema_version=2,initialization_policy=trial.parent_policy)
    (path/'metadata.json').write_text(json.dumps(metadata,sort_keys=True)+'\n')


def restore_trial_delta(current,teacher,path,*,recorder=None):
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    path=Path(path);metadata=json.loads((path/'metadata.json').read_text())
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    if metadata.get('execution_profile') != EXECUTION_PROFILE:raise ValueError('unsupported delta execution profile')
    if metadata['base_digest']!=current.manifest().checkpoint_digest or metadata['teacher_digest']!=teacher.manifest().checkpoint_digest:raise ValueError('delta base identity changed')
    if hashlib.sha256((path/'weights.safetensors').read_bytes()).hexdigest()!=metadata['weights_sha256']:raise ValueError('delta weights hash mismatch')
    pair=tuple(metadata['pair']);affected=tuple(metadata['affected_depths']);indices={current.partition.reuse_map[d] for d in pair}
    policy=metadata.get('initialization_policy')
    if 'initialization_schema_version' in metadata or policy is not None:
        if type(metadata.get('initialization_schema_version')) is not int or metadata['initialization_schema_version']!=2:
            raise ValueError('unsupported initialization policy schema')
        _validate_parent_policy(policy,current,pair,metadata['selected_parent'])
    merged=canonicalize_partition([g for k,g in enumerate(current.partition.groups) if k not in indices]+[affected],len(current.layers))
    candidate=deepcopy(current);old={g:current.bank[k] for k,g in enumerate(current.partition.groups)}
    candidate.bank=nn.ModuleList([deepcopy(current.bank[current.partition.reuse_map[pair[0]]] if g==affected else old[g]) for g in merged.groups]);candidate.partition=merged
    for depth,layer in enumerate(candidate.layers):layer.mlp.index=merged.reuse_map[depth];layer.mlp._owners_ref=weakref.ref(candidate.bank)
    values=load_file(str(path/'weights.safetensors'));parameters=dict(candidate.named_parameters())
    flags=metadata.get('owner_trainability')
    if not isinstance(flags,dict) or set(flags)!=set(parameters) or any(type(v) is not bool for v in flags.values()):
        raise ValueError('delta owner trainability schema mismatch')
    expected={n for n in parameters if n.startswith(f'bank.{merged.reuse_map[pair[0]]}.') or any(n.startswith(f'layers.{d}.mlp.residuals.') for d in affected)}
    if set(values)!=expected:raise ValueError('delta changed-owner schema mismatch')
    with torch.no_grad():
        for n,v in values.items():parameters[n].copy_(v)
        for n,p in parameters.items():p.requires_grad_(flags[n])
    if candidate.manifest().checkpoint_digest!=metadata['fitted_digest']:raise ValueError('restored fitted state digest mismatch')
    return MergeTrial(candidate,teacher,pair,affected,metadata['selected_parent'],tuple(metadata['medoid_scores']),metadata['initialized_digest'],metadata['fitted_digest'],recorder=recorder,fit_summary=metadata['fit_summary'],parent_policy=policy)

"""New owned CPU controls; deliberately separate from inaccessible archive fixtures."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch

from poc.bank import make_bank, canonicalize_partition, initialize_residual, owner_census
from poc.checkpoint import resolve_checkpoint, load_checkpoint
from poc.continuation import make_spec
from poc.continuation_plan import build_target_plan, digest
from poc.contracts import Batch, CostRecord, ProbeRecord
from poc.data.manifest import DataManifest, DocumentRecord
from poc.diagnostics import run_controls
from poc.losses import stream_metrics
from poc.merge import initialize_merge, initializer_sensitivity_control, fit_trial, save_trial_delta, restore_trial_delta
from poc.probes import prefix_for_probe
from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder, EXECUTION_PROFILE
from poc.train import ProbeTrainingSource, TrainingConfig, _Dispatch, _Counts


def require(value, message):
    if not value:raise ValueError('owned control failed: '+message)


def mechanism_controls():
    values=initializer_sensitivity_control()
    require(values['whole_medoid_max_error']<1e-12,'permuted exact sharing')
    require(values['full_rank_max_error']<1e-12,'full rank')
    require(values['truncated_svd_max_error']>1e-5,'harmful truncated SVD detection')
    a,b=initialize_residual(torch.zeros((12,8),dtype=torch.float64),2,17)
    a.requires_grad_();b.requires_grad_()
    (b@a).sum().backward()
    require(bool(a.count_nonzero()) and not bool(b.count_nonzero()) and bool(b.grad.count_nonzero()),'zero-product live gradient')
    arrays=run_controls()
    require(arrays['planted_selected_regret']==0 and arrays['bad_selected_regret']>0 and arrays['null_regret']==0,'array selector controls')
    return dict(initializer=values,zero_product_live_b_gradient=True,
        array_only_selection_controls=arrays,scope='constructed mechanisms; acquisition not exercised')


def _source(batch,depth,teacher):
    tokens=tuple(batch.input_ids[0].tolist());positions=tuple(range(len(tokens)))
    identity=digest({'owned_window':digest(tokens),'depth':depth,'source':'natural',
        'execution_profile':EXECUTION_PROFILE,'teacher':teacher.manifest().checkpoint_digest,'seed':17})
    def build(student,teacher,recorder):
        record=ProbeRecord(probe_id=identity,recipient_id=batch.document_ids[0],donor_id=batch.document_ids[0],
            recipient_component=batch.cluster_ids[0],donor_component=batch.cluster_ids[0],
            recipient_role='fit',donor_role='fit',use='fit',source='natural',
            recipient_tokens=tokens,donor_tokens=tokens,attention_mask=(True,)*len(tokens),
            positions=positions,donor_positions=positions,target_mask=(True,)*(len(tokens)-1),
            patch_positions=(),patch_mode='full',channel_quarter=None,depth=depth,
            teacher_digest=teacher.manifest().checkpoint_digest,candidate_digest=student.manifest().checkpoint_digest,
            tokenizer_digest=digest({'owned_tokenizer':'integer_fixture_v1'}),dtype='float64',backend='torch_cpu',
            boundary_version='post_attention_v1',execution_profile=EXECUTION_PROFILE,seed=17,
            edit_kind='owned_identity',source_span=(0,0),donor_span=(0,0),replacement='',prefix_source='teacher',
            source_window=(0,len(tokens)),recipient_length_before_crop=len(tokens),donor_length_before_crop=len(tokens),
            recipient_full_tokens_digest=digest(tokens),donor_full_tokens_digest=digest(tokens),context_policy='first_tokens_reset_0_v1')
        return prefix_for_probe(teacher,batch,depth,source='teacher',recorder=recorder),record
    return ProbeTrainingSource(batch,'natural','fit',identity,build)


def owned_fitting(resolved, ledger, environment_hash, root):
    """One actual 8192-target local update, full provenance, checkpoint and delta."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    state={}
    def prepare():
        teacher=make_tiny_decoder(TinyConfig(hidden_size=8,intermediate_size=12,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=256),seed=2,dtype=torch.float64)
        current=make_bank(teacher,canonicalize_partition([[0],[1],[2]],3),2,17)
        ids=(torch.arange(129)%teacher.config.vocab_size)[None]
        b=Batch(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,128),dtype=torch.bool),
            torch.arange(129)[None],('owned-fit-window',),('owned-fit-component',))
        items=tuple(_source(b,d,teacher) for d in (0,2))
        plan=build_target_plan(items,{p.identity:d for p,d in zip(items,(0,2),strict=True)},
            updates=1,weights={'natural':1.},affected_depths=(0,2))
        state.update(teacher=teacher,current=current,batch=b,plan=plan)
    ledger.shared('owned-preparation',CostRecord(),prepare)
    require(bool(state),'owned output must use a fresh directory; shared cache reconstruction not implemented')
    teacher,current,b,plan=(state[k] for k in ('teacher','current','batch','plan'))
    fit=DataManifest((DocumentRecord(b.document_ids[0],'owned','v1','owned',0),),
        {b.document_ids[0]:'fit'},{b.document_ids[0]:b.cluster_ids[0]},'owned-v1')
    config=TrainingConfig(max_updates=1,epochs=1,phase='local',affected_depths=(0,2),
        mixture_weights={'natural':1.},observation_steps=(1,),numerical_profile=EXECUTION_PROFILE)
    spec=make_spec(resolved,mode='owned-fit',environment_sha256=environment_hash,plan_sha256=plan.digest)
    spec=replace(spec,resolved_config={**dict(spec.resolved_config),'training':config.identity(),
        'owned_fixture':{'pair':[0,2],'rank':2,'teacher_seed':2,'initialization_seed':17,
            'teacher_config':dict(teacher.manifest().config),'teacher_digest':teacher.manifest().checkpoint_digest,
            'current_digest':current.manifest().checkpoint_digest,'template_inventory_sha256':plan.sources.digest()}})
    observations=[]
    with ledger.arm(spec,dependencies=('owned-preparation',),forecast=CostRecord(updates=1,valid_target_presentations=8192)) as recorder:
        # initialize_merge uses the supplied owned batches: never _fit_batches/open_records.
        trial=initialize_merge(current,teacher,(0,2),fit,17,batches=(b,),recorder=recorder)
        trial.sources=plan.sources
        def observe(step):
            m=stream_metrics(_Dispatch(teacher,_Counts(recorder),'teacher'),
                _Dispatch(trial.model,_Counts(recorder),'student'),b,32)
            observations.append(dict(step=step,kl_sum=m.kl,nll_sum=m.nll,valid_targets=m.valid_targets,
                scope='owned-fit-window; not independent validation'))
        # Step zero is charged before fit_model and does not enter observation_steps.
        from poc.train import _call
        _call(recorder,lambda:observe(0))
        cfg=replace(config,run_spec=spec,checkpoint_root=root/'checkpoint',on_update=observe)
        trial.recorder=recorder
        fit_trial(trial,cfg,spec.budgets)
        saved=load_checkpoint(resolve_checkpoint(root/'checkpoint'),spec)
        require(saved.step==1 and saved.target_cursor==8192,'full update checkpoint')
        require(saved.model_schema['execution_profile']==EXECUTION_PROFILE,'checkpoint execution identity')
        require(all(not p.requires_grad and p.grad is None for p in teacher.parameters()) and not teacher.training,'teacher frozen')
        save_trial_delta(trial,current,root/'delta')
        restored=restore_trial_delta(current,teacher,root/'delta')
        require(owner_census(trial.model)==owner_census(restored.model),'owner census')
        difference=reload_difference(trial.model,restored.model,b,recorder)
        require(difference<=1e-12,'delta output tolerance')
        require(all(torch.equal(p,dict(restored.model.named_parameters())[n]) for n,p in trial.model.named_parameters()),'exact restored values')
        recorder.record(CostRecord(disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())))
    return dict(updates=saved.step,targets=saved.target_cursor,teacher_frozen=True,
        initialized_digest=trial.initialized_digest,fitted_digest=trial.fitted_digest,
        restored_digest_equal=restored.fitted_digest==trial.fitted_digest,max_reload_logit_difference=difference,
        observations=observations,fit_metrics=trial.fit_summary['metrics'],target_plan=plan.metadata,
        target_plan_sha256=plan.digest,child_run_id=ledger.graph()[-2]['key'],
        unavailable_metrics=['independent_validation','gradient_norm','update_norm','pretrained_quality'],
        checkpoint_retrieval='Private local owned artifacts beside receipt; regenerate with owned-controls in a fresh output directory')


def reload_difference(left,right,batch,recorder):
    from poc.train import _call
    with torch.no_grad():
        a=_call(recorder,lambda:left.logits(batch),student_forward_calls=1)
        b=_call(recorder,lambda:right.logits(batch),student_forward_calls=1)
        return float((a-b).abs().max())


def numeric_fixture():
    """Hand-specified arithmetic fixture, not neural score/damage measurements."""
    h=lambda n:f'{n:064x}'
    candidates=[]
    for i,pair in enumerate(([0,1],[0,2])):
        observations=[]
        for source,value in [('natural',2+i),('student',4),('random',6),('active',8),('active',10)]:
            observations.append(dict(component_sha256=h(10),window_sha256=h(11),group_sha256=h(12),
                probe_sha256=h(100+len(observations)),source=source,kl_sum=value*2.,valid_targets=2))
        observations.append({**observations[2],'source':'active'})
        candidates.append(dict(pair=pair,inclusion_probability=1.,status='valid',failure_code=None,
            initialized_digest=h(30+i),fitted_digest=h(40+i),observations=observations,
            damage=[dict(component_sha256=h(50),window_sha256=h(51),kl_sum=float(i+1),valid_targets=1)],query_node_ids=[h(60)]))
    return dict(schema_version=1,scope='owned_fixture',decision_resolution=1e-12,
        candidate_universe=[[0,1],[0,2]],score_components=[h(10)],damage_components=[h(50)],candidates=candidates,
        query_graph=[dict(node_id=h(60),dependencies=[],cost=CostRecord().to_dict())])

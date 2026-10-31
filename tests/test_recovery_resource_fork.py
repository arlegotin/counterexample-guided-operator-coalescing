"""Resource-only forks preserve scientific state and carry all ancestor spending."""
from dataclasses import replace
from pathlib import Path
import hashlib,importlib,json

import pytest
import torch

from poc.contracts import CostRecord
from poc.records import canonical_json,run_id,start_attempt
from poc.budget import BudgetTracker
from poc.checkpoint import save_checkpoint
from poc.train import fit_model,TrainingSources
from test_recovery_study import setup,bound_config
from test_training import budget,MemoryRecorder


def api():
    import importlib.util
    assert importlib.util.find_spec('poc.recovery_forks'),'explicit resource-fork gate missing'
    return importlib.import_module('poc.recovery_forks')


def prepare(tmp_path):
    teacher,student,plan,objective=setup('full_student',updates=2);cfg=bound_config(objective,2)
    cfg=replace(cfg,run_spec=replace(cfg.run_spec,resolved_config={**dict(cfg.run_spec.resolved_config),'study_contract_sha256':'a'*64},
        input_manifests={**dict(cfg.run_spec.input_manifests),'executed_source':'b'*64,'controls':'c'*64,'profile':'d'*64}))
    rec=MemoryRecorder();first=fit_model(student,teacher,TrainingSources(raw_plan=plan),replace(cfg,stop_after_updates=1),budget(),rec,objective_spec=objective)
    ref=save_checkpoint(first.state,tmp_path/'checkpoint')
    attempt=start_attempt(cfg.run_spec,tmp_path/'ledgers');ledger=BudgetTracker(budget(),attempt)
    ledger.record(rec.totals);ledger.record(CostRecord(wall_seconds=1,raw_corpus_target_presentations=7,valid_target_presentations=7))
    ledger_bytes=ledger.path.read_bytes()
    qualification=tmp_path/'qualification.json';qualification.write_bytes(canonical_json({'passed':True,'purpose':'resource_fork_compatibility_v1','parent_source_sha256':'b'*64,'child_source_sha256':'e'*64}))
    profile=tmp_path/'profile.json';profile.write_bytes(canonical_json({'purpose':'resource_only_completion_profile_v1','parent_ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),'fixed_updates':2,'cumulative_wall_seconds':600,'passed':True}))
    contract=api().RecoveryResourceFork(ref.manifest_sha256,run_id(cfg.run_spec),hashlib.sha256(ledger_bytes).hexdigest(),
        'b'*64,'e'*64,hashlib.sha256(qualification.read_bytes()).hexdigest(),hashlib.sha256(profile.read_bytes()).hexdigest(),'f'*64)
    child=replace(cfg.run_spec,method_variant='owned-resource-completion-v1',budgets=replace(budget(),wall_seconds=600),
        parent_run_ids=(*cfg.run_spec.parent_run_ids,run_id(cfg.run_spec)),
        resolved_config={**dict(cfg.run_spec.resolved_config),'study_contract_sha256':'f'*64,'resource_fork':contract.identity()},
        input_manifests={**dict(cfg.run_spec.input_manifests),'executed_source':'e'*64,'controls':contract.qualification_sha256,
                         'profile':contract.profile_sha256,'fork_checkpoint':ref.manifest_sha256,'fork_ledger':contract.parent_ledger_sha256})
    return teacher,student,plan,objective,cfg,ref,ledger,contract,child,qualification,profile


def fork(values):
    _,_,_,_,cfg,ref,ledger,contract,child,q,p=values
    return api().fork_recovery_checkpoint(ref,cfg.run_spec,child,contract,ledger_path=ledger.path,qualification_path=q,profile_path=p)


def test_resource_fork_preserves_optimizer_rng_schedule_and_full_spending(tmp_path):
    from poc.checkpoint import load_checkpoint
    values=prepare(tmp_path);teacher,student,plan,obj,cfg,ref,ledger,contract,child,q,p=values
    old=load_checkpoint(ref,cfg.run_spec);before={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(ref.directory).iterdir() if x.is_file()}
    branch=fork(values)
    assert branch.state.spec==child and branch.state.config_hash==old.config_hash
    assert branch.state.scheduler==old.scheduler and branch.state.target_cursor==8192
    assert branch.opening_balance==ledger.totals
    assert branch.state.budget_totals==ledger.totals.to_dict()
    assert branch.record['parent_run_id']==run_id(cfg.run_spec)
    for key,tensor in old.model.items():assert torch.equal(branch.state.model[key],tensor)
    newcfg=replace(cfg,run_spec=child,resume=branch.state)
    with pytest.raises(ValueError,match='spending cannot roll back'):
        fit_model(student,teacher,TrainingSources(raw_plan=plan),newcfg,child.budgets,MemoryRecorder(),objective_spec=obj)
    child_attempt=api().start_resource_fork_attempt(branch,tmp_path/'child-ledgers')
    rec=BudgetTracker(child.budgets,child_attempt)
    result=fit_model(student,teacher,TrainingSources(raw_plan=plan),newcfg,child.budgets,rec,objective_spec=obj)
    _,whole,_,_=setup('full_student',updates=2)
    full=fit_model(whole,teacher,TrainingSources(raw_plan=plan),cfg,budget(),MemoryRecorder(),objective_spec=obj)
    assert result.state.events==full.state.events
    for n,t in whole.named_parameters():assert torch.equal(t,dict(student.named_parameters())[n])
    assert rec.totals.raw_corpus_target_presentations==16384+7
    assert before=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(ref.directory).iterdir() if x.is_file()}


@pytest.mark.parametrize('change',['horizon','rate','objective','data','owners','hardware','budget_unit','same_identity','qualification','ledger','profile_budget'])
def test_resource_fork_rejects_scientific_or_unbound_changes(tmp_path,change):
    values=list(prepare(tmp_path));child=values[8]
    if change in ('horizon','rate'):
        config=dict(child.resolved_config);training=dict(config['training']);training['max_updates' if change=='horizon' else 'bank_lr']=3 if change=='horizon' else .123
        config['training']=training;child=replace(child,resolved_config=config)
    if change=='objective':
        config=dict(child.resolved_config);objective=dict(config['recovery_objective']);objective['loss_scope']='local_sum';config['recovery_objective']=objective;child=replace(child,resolved_config=config)
    if change in ('data','owners'):
        inputs=dict(child.input_manifests);inputs['raw_plan' if change=='data' else 'owner_names']='0'*64;child=replace(child,input_manifests=inputs)
    if change=='hardware':child=replace(child,hardware_policy={'backend':'other'})
    if change=='budget_unit':child=replace(child,budgets=replace(child.budgets,updates=101))
    if change=='profile_budget':child=replace(child,budgets=replace(child.budgets,wall_seconds=601))
    if change=='same_identity':child=replace(child,method_variant=values[4].run_spec.method_variant)
    if change=='qualification':values[9].write_text('{}')
    if change=='ledger':values[6].record(CostRecord(wall_seconds=1))
    values[8]=child
    with pytest.raises(ValueError):fork(values)


def test_resource_fork_rejects_checkpoint_era_ledger_snapshot(tmp_path):
    from types import SimpleNamespace
    values=list(prepare(tmp_path));ledger=values[6]
    snapshot=ledger.path.with_name('checkpoint-era.jsonl')
    snapshot.write_bytes(b''.join(ledger.path.read_bytes().splitlines(keepends=True)[:-1]))
    snapshot_sha=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    profile=values[10];profile_data=json.loads(profile.read_text());profile_data['parent_ledger_sha256']=snapshot_sha
    profile.write_bytes(canonical_json(profile_data))
    contract=replace(values[7],parent_ledger_sha256=snapshot_sha,profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
    child=values[8];values[8]=replace(child,resolved_config={**dict(child.resolved_config),'resource_fork':contract.identity()},
        input_manifests={**dict(child.input_manifests),'fork_ledger':snapshot_sha,'profile':contract.profile_sha256})
    values[6]=SimpleNamespace(path=snapshot);values[7]=contract
    with pytest.raises(ValueError,match='canonical'):
        fork(values)


def test_ancestor_balance_is_separate_from_new_attempt_elapsed(tmp_path):
    from poc.train import finish_attempt_cost
    values=prepare(tmp_path);branch=fork(values);root=tmp_path/'child-ledgers'
    assert hasattr(api(),'start_resource_fork_attempt'),'separate inherited-cost attempt missing'
    attempt=api().start_resource_fork_attempt(branch,root)
    rec=BudgetTracker(values[8].budgets,attempt)
    assert rec.totals==branch.opening_balance
    assert rec.attempt_cost.wall_seconds==0
    rec.record(CostRecord(wall_seconds=2))
    finish_attempt_cost(rec,elapsed=5)
    assert rec.attempt_cost.wall_seconds==5
    assert rec.totals.wall_seconds==pytest.approx(branch.opening_balance.wall_seconds+5)
    retry=api().start_resource_fork_attempt(branch,root);retried=BudgetTracker(values[8].budgets,retry)
    assert retried.totals==rec.totals and retried.attempt_cost.wall_seconds==0


@pytest.mark.parametrize('phase',['before_cost','after_cost'])
def test_opening_balance_interruption_cannot_erase_or_duplicate_inherited_cost(tmp_path,monkeypatch,phase):
    values=prepare(tmp_path);branch=fork(values);root=tmp_path/'child-ledgers';module=api()
    assert hasattr(module,'start_resource_fork_attempt'),'durable opening-balance import missing'
    def stop(*args,**kwargs):raise InterruptedError('owned opening-balance interruption')
    target=BudgetTracker if phase=='before_cost' else module
    name='record' if phase=='before_cost' else 'append_event'
    with monkeypatch.context() as patch:
        patch.setattr(target,name,stop)
        with pytest.raises(InterruptedError):module.start_resource_fork_attempt(branch,root)
    attempt=module.start_resource_fork_attempt(branch,root)
    rec=BudgetTracker(values[8].budgets,attempt)
    assert rec.totals==branch.opening_balance and rec.attempt_cost.wall_seconds==0

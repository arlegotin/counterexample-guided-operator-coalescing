"""Safe immutable generations and exact next-update replay."""
from dataclasses import replace
import importlib
import importlib.util
import json
import random

import numpy as np
import pytest
import torch

from test_training import api, batch, budget, models, MemoryRecorder
from test_records import spec


def bound_config(**changes):
    cfg=api().TrainingConfig(**changes)
    if cfg.run_spec is not None:
        cfg=replace(cfg,run_spec=replace(cfg.run_spec,resolved_config={'training':cfg.identity()}))
    return cfg


def checkpoint():
    assert importlib.util.find_spec('poc.checkpoint'), 'safe checkpoint missing'
    return importlib.import_module('poc.checkpoint')


def test_exact_next_update_replay_all_owned_state(tmp_path):
    t=api(); c=checkpoint(); teacher,student=models()
    cfg=bound_config(max_updates=4,epochs=4,run_spec=spec(budgets=budget()))
    sources=t.TrainingSources(natural=(batch(),))
    first=t.fit_model(student,teacher,sources,replace(cfg,stop_after_updates=2),budget(),MemoryRecorder())
    ref=c.save_checkpoint(first.state,tmp_path)
    saved=c.load_checkpoint(ref,cfg.run_spec)
    assert saved.target_cursor==16 and saved.step==2 and saved.probe_cursor==0
    next_random=(random.random(),float(np.random.random()),torch.rand(3))
    # Both branches restore from the same update boundary, including RNG.
    _,left=models(); _,right=models()
    a=t.fit_model(left,teacher,sources,replace(cfg,resume=saved),budget(),MemoryRecorder(saved.budget_totals))
    draws=(random.random(),float(np.random.random()),torch.rand(3))
    b=t.fit_model(right,teacher,sources,replace(cfg,resume=c.load_checkpoint(ref,cfg.run_spec)),budget(),MemoryRecorder(saved.budget_totals))
    assert random.random()==draws[0] and float(np.random.random())==draws[1]
    assert torch.equal(torch.rand(3),draws[2])
    for n,p in left.named_parameters(): assert torch.equal(p,dict(right.named_parameters())[n])
    assert a.state.target_cursor==b.state.target_cursor==32
    assert a.state.scheduler==b.state.scheduler
    assert a.state.events==b.state.events
    for n,values in a.state.optimizer['state'].items():
        for key,value in values.items():
            other=b.state.optimizer['state'][n][key]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
    assert saved.order==a.state.order and saved.probe_order==a.state.probe_order
    assert next_random[0]!=random.random()


@pytest.mark.parametrize('point',['before_generation_publish','after_generation_publish','before_pointer_replace','after_pointer_replace'])
def test_crash_publication_keeps_current_or_previous_verified_generation(tmp_path,monkeypatch,point):
    t=api();c=checkpoint();teacher,student=models(); identity=spec(budgets=budget())
    cfg=bound_config(max_updates=2,epochs=2,run_spec=identity);identity=cfg.run_spec
    state=t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),replace(cfg,stop_after_updates=1),budget(),MemoryRecorder()).state
    first=c.save_checkpoint(state,tmp_path)
    def crash(stage):
        if stage==point: raise InterruptedError(point)
    monkeypatch.setattr(c,'_publication_boundary',crash)
    with pytest.raises(InterruptedError): c.save_checkpoint(state,tmp_path)
    current=c.resolve_checkpoint(tmp_path)
    assert c.load_checkpoint(current,identity).step==1
    assert (tmp_path/'current.json').exists()
    assert c.load_checkpoint(first,identity).candidate_digest==state.candidate_digest


def test_corrupt_newest_explicit_fallback_and_changed_lineage(tmp_path):
    t=api();c=checkpoint();teacher,student=models(); identity=spec(budgets=budget())
    cfg=bound_config(max_updates=1,run_spec=identity);identity=cfg.run_spec
    state=t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),cfg,budget(),MemoryRecorder()).state
    old=c.save_checkpoint(state,tmp_path);new=c.save_checkpoint(state,tmp_path)
    tensor_path=__import__('pathlib').Path(new.directory)/'tensors.safetensors'
    tensor_path.write_bytes(tensor_path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash'): c.load_checkpoint(new,identity)
    with pytest.raises(ValueError): c.resolve_checkpoint(tmp_path)
    fallback=c.resolve_checkpoint(tmp_path,fallback_previous=True)
    assert fallback==old
    for mismatch in [replace(identity,dependency_lock_sha256='e'*64),replace(identity,input_manifests={'new':'f'*64}),replace(identity,resolved_config={'map':[0,0,0]})]:
        with pytest.raises(ValueError,match='lineage'): c.load_checkpoint(old,mismatch)
    assert not list(tmp_path.rglob('*.pt')) and not list(tmp_path.rglob('*.pkl'))


def test_restore_rejects_invalid_optimizer_without_mutating_model(tmp_path):
    from copy import deepcopy
    t=api();teacher,student=models();identity=spec(budgets=budget())
    cfg=bound_config(max_updates=2,epochs=2,run_spec=identity);identity=cfg.run_spec
    sources=t.TrainingSources(natural=(batch(),))
    state=t.fit_model(student,teacher,sources,replace(cfg,stop_after_updates=1),budget(),MemoryRecorder()).state
    _,fresh=models();before={n:p.clone() for n,p in fresh.named_parameters()}
    bad=replace(state,optimizer=deepcopy(state.optimizer));bad.optimizer['groups'][0]['owner_ids']=['wrong']
    with pytest.raises(ValueError,match='optimizer'):
        t.fit_model(fresh,teacher,sources,replace(cfg,resume=bad),budget(),MemoryRecorder(state.budget_totals))
    assert all(torch.equal(p,before[n]) for n,p in fresh.named_parameters())


def test_actual_fitted_checkpoint_export_resolves_identity_and_weights(tmp_path,capsys):
    from poc.cli import main
    from poc.records import start_attempt
    from poc.bundle import load_bundle
    from poc.contracts import thaw
    from test_export import runtime
    t=api();c=checkpoint();teacher,student=models();teacher=teacher.float();student=student.float()
    cfg=bound_config(max_updates=1,run_spec=spec(budgets=budget()),kl_floor_per_target=1e-6,numerical_profile='fixture_fp32')
    identity=cfg.run_spec;attempt=start_attempt(identity,tmp_path/'runs')
    fitted=t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),cfg,budget(),MemoryRecorder())
    ref=c.save_checkpoint(fitted.state,attempt.path/'checkpoints')
    run_runtime=runtime(tmp_path);runtime_path=tmp_path/'runtime.json'
    from dataclasses import asdict
    runtime_path.write_text(json.dumps({f.name:thaw(getattr(run_runtime,f.name)) for f in __import__('dataclasses').fields(run_runtime)}))
    assert main(['export','--run-root',str(tmp_path/'runs'),'--run',attempt.run_id,'--attempt',attempt.attempt_id,
        '--runtime-spec',str(runtime_path),'--out',str(tmp_path/'fitted-bundle')])==0
    receipt=json.loads((tmp_path/'fitted-bundle.receipt.json').read_text())
    assert receipt['fitted_checkpoint_resolved'] is True and receipt['run_id']==attempt.run_id
    assert receipt['checkpoint_manifest_sha256']==ref.manifest_sha256
    loaded=load_bundle(tmp_path/'fitted-bundle',backend='torch')
    torch.testing.assert_close(loaded.logits(batch()),student.logits(batch()),rtol=0,atol=0)
    assert main(['export','--run-root',str(tmp_path/'runs'),'--run','a'*64,'--attempt',attempt.attempt_id,
        '--runtime-spec',str(runtime_path),'--out',str(tmp_path/'wrong')])==4


def test_local_probe_cursor_and_live_candidate_provenance_replay(tmp_path):
    from test_training import local_sources
    t=api();c=checkpoint();teacher,student=models();identity=spec(budgets=budget())
    sources=local_sources(teacher,student)
    cfg=bound_config(max_updates=4,epochs=4,phase='local',affected_depths=(0,1),
        mixture_weights={'natural':.5,'student':.25,'donor':.25},run_spec=identity)
    identity=cfg.run_spec
    first=t.fit_model(student,teacher,sources,replace(cfg,stop_after_updates=2),budget(),MemoryRecorder())
    ref=c.save_checkpoint(first.state,tmp_path)
    _,resumed=models()
    restored=t.fit_model(resumed,teacher,local_sources(teacher,resumed),replace(cfg,resume=c.load_checkpoint(ref,identity)),budget(),MemoryRecorder(first.state.budget_totals))
    _,whole=models();uninterrupted=t.fit_model(whole,teacher,local_sources(teacher,whole),cfg,budget(),MemoryRecorder())
    assert restored.state.probe_cursor==uninterrupted.state.probe_cursor==12
    assert restored.state.target_cursor==uninterrupted.state.target_cursor==40
    for n,p in whole.named_parameters():assert torch.equal(p,dict(resumed.named_parameters())[n])
    assert restored.state.events==uninterrupted.state.events


def test_truncated_cost_tail_is_preserved_and_blocks_resume(tmp_path):
    from poc.records import start_attempt,RecordError
    from poc.budget import BudgetTracker
    t=api();teacher,student=models();identity=spec(budgets=budget());attempt=start_attempt(identity,tmp_path)
    tracker=BudgetTracker(identity.budgets,attempt);tracker.record(__import__('poc.contracts',fromlist=['CostRecord']).CostRecord(wall_seconds=1))
    with tracker.path.open('ab') as stream:stream.write(b'{"event_id":2')
    tail=tracker.path.read_bytes()
    with pytest.raises(RecordError,match='truncated'):
        t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),t.TrainingConfig(max_updates=1),budget(),tracker)
    assert tracker.path.read_bytes()==tail


def test_only_last_and_previous_published_generations_are_retained(tmp_path):
    t=api();c=checkpoint();teacher,student=models();identity=spec(budgets=budget())
    cfg=bound_config(max_updates=1,run_spec=identity);identity=cfg.run_spec
    state=t.fit_model(student,teacher,t.TrainingSources(natural=(batch(),)),cfg,budget(),MemoryRecorder()).state
    a=c.save_checkpoint(state,tmp_path);b=c.save_checkpoint(state,tmp_path);last=c.save_checkpoint(state,tmp_path)
    from pathlib import Path
    assert not Path(a.directory).exists()
    assert Path(b.directory).exists() and Path(last.directory).exists()
    assert len(list(tmp_path.glob('generation-*')))==2


def test_global_reference_replay_matches_uninterrupted_four_updates(tmp_path):
    t=api();c=checkpoint();teacher,whole=models();identity=spec(budgets=budget())
    cfg=bound_config(max_updates=4,epochs=4,run_spec=identity);identity=cfg.run_spec;sources=t.TrainingSources(natural=(batch(),))
    full=t.fit_model(whole,teacher,sources,cfg,budget(),MemoryRecorder())
    _,partial=models();first=t.fit_model(partial,teacher,sources,replace(cfg,stop_after_updates=2),budget(),MemoryRecorder())
    ref=c.save_checkpoint(first.state,tmp_path);_,resumed=models()
    after=t.fit_model(resumed,teacher,sources,replace(cfg,resume=c.load_checkpoint(ref,identity)),budget(),MemoryRecorder(first.state.budget_totals))
    for n,p in whole.named_parameters():assert torch.equal(p,dict(resumed.named_parameters())[n])
    assert full.state.events==after.state.events and full.state.visited==after.state.visited
    assert full.state.scheduler==after.state.scheduler

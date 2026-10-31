"""Recovery phases consume consecutive masks from one frozen raw stream."""
from dataclasses import replace
import importlib

import pytest
import torch

from poc.recovery_study import NaturalWindowPlan
from test_training import batch


def api():
    module=importlib.import_module('poc.recovery_study')
    assert hasattr(module,'NaturalWindowSegment'),'versioned phase plan missing'
    return module.NaturalWindowSegment


def parent():
    other=replace(batch(513),document_ids=('other',),cluster_ids=('other-component',))
    return NaturalWindowPlan.build((batch(512),other),updates=2,seed=17,manifest_hashes={'owned':'a'*64})


def test_segments_preserve_exact_stream_and_split_masks_without_cutting_context():
    p=parent();left=api().build(p,start_update=0,end_update=1);right=api().build(p,start_update=1,end_update=2)
    full=list(p.iter_updates());a=list(left.iter_updates());b=list(right.iter_updates())
    for expected,actual in zip(full,(*a,*b),strict=True):
        assert expected.valid_targets==actual.valid_targets==8192
        for x,y in zip(expected.batches,actual.batches,strict=True):
            for field in ('input_ids','attention_mask','positions','target_mask'):
                assert torch.equal(getattr(x,field),getattr(y,field))
            assert x.document_ids==y.document_ids and x.cluster_ids==y.cluster_ids
    tail=a[-1].batches[-1];head=b[0].batches[0]
    assert tail.document_ids==head.document_ids
    assert torch.equal(tail.input_ids,head.input_ids)
    assert not (tail.target_mask & head.target_mask).any()
    assert (tail.target_mask | head.target_mask).all()
    for segment in (left,right):
        for update in segment.iter_updates():
            for sliced,(index,_,_) in zip(update.batches,update.slices,strict=True):
                original=segment.template_at(index)
                assert original.target_mask.all()
                assert torch.equal(original.input_ids,sliced.input_ids)
        assert segment.metadata['raw_target_presentations']==8192
        assert segment.metadata['parent_plan_sha256']==p.digest
        assert len(segment.digest)==64
    assert left.digest!=right.digest
    changed=right.template_at(0);changed.input_ids.fill_(0)
    assert not torch.equal(changed.input_ids,right.template_at(0).input_ids)


@pytest.mark.parametrize('bounds',[(-1,1),(0,0),(1,0),(0,3),(True,2)])
def test_invalid_phase_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):api().build(parent(),start_update=bounds[0],end_update=bounds[1])


def test_segment_plan_runs_through_common_fitter_and_resumes(tmp_path):
    from copy import deepcopy
    from poc.train import fit_model,TrainingSources
    from poc.checkpoint import save_checkpoint,load_checkpoint
    from test_recovery_study import setup,bound_config
    from test_training import budget,MemoryRecorder
    teacher,student,p,spec=setup('full_student',updates=3)
    segment=api().build(p,start_update=1,end_update=3)
    spec=replace(spec,raw_plan_sha256=segment.digest);cfg=bound_config(spec,2)
    source=TrainingSources(raw_plan=segment);rec=MemoryRecorder();whole=deepcopy(student)
    first=fit_model(student,teacher,source,replace(cfg,stop_after_updates=1),budget(),rec,objective_spec=spec)
    checkpoint=save_checkpoint(first.state,tmp_path)
    final=fit_model(student,teacher,source,replace(cfg,resume=load_checkpoint(checkpoint,cfg.run_spec)),budget(),rec,objective_spec=spec)
    uninterrupted=fit_model(whole,teacher,source,cfg,budget(),MemoryRecorder(),objective_spec=spec)
    assert final.state.events==uninterrupted.state.events
    assert final.state.target_cursor==16384
    for name,p in student.named_parameters():assert torch.equal(p,dict(whole.named_parameters())[name])
    assert final.state.order==('raw_plan_segment_v1',segment.digest)

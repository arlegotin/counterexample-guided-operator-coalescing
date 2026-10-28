"""Finite local plans for full real-context probes with unequal eligible masks."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import wraps

import torch

from poc.bank import PROJECTIONS, initialize_residual
from poc.continuation_plan import TargetPlan, digest, window_metadata
from poc.train import TrainingConfig, TrainingSources, iter_update_targets


def finite_plan(items, depths, *, updates, weights, affected_depths):
    TrainingConfig(max_updates=updates,phase='local',affected_depths=affected_depths,mixture_weights=weights)
    if not items or len({p.identity for p in items})!=len(items):raise ValueError('unique full templates required')
    groups=defaultdict(lambda:defaultdict(list));bound={};inventory={}
    for item in items:
        depth=depths[item.identity]
        if item.source not in weights or depth not in affected_depths:raise ValueError('unexpected source/depth')
        count=int(item.template.target_mask.sum())
        if not 0<count<=8192:raise ValueError('invalid template target count')
        rows=window_metadata(item.template);components={r['component_sha256'] for r in rows}
        if len(components)!=1:raise ValueError('one component per template required')
        groups[item.source,depth][next(iter(components))].append(item)
        inventory[item.identity]=rows
        @wraps(item.build)
        def build(student,teacher,recorder,item=item,depth=depth):
            boundary,record=item.build(student,teacher,recorder)
            if record.depth!=depth or boundary.depth!=depth:raise ValueError('declared depth mismatch')
            return boundary,record
        bound[item.identity]=replace(item,build=build)
    keys=sorted((s,d) for s in weights for d in affected_depths)
    if set(groups)!=set(keys):raise ValueError('missing source/depth coverage')
    queues={k:[sorted(v[c],key=lambda p:p.identity) for c in sorted(v)] for k,v in groups.items()}
    cursor=defaultdict(int);spent=defaultdict(int);visits=defaultdict(int);selected=[];presentations=[];total=0
    required=updates*8192
    while total<required:
        key=min(keys,key=lambda k:(spent[k]/weights[k[0]],k))
        q=queues[key];i=cursor[key];cursor[key]+=1
        item=q[i%len(q)][(i//len(q))%len(q[i%len(q)])]
        count=int(item.template.target_mask.sum());visits[item.identity]+=1
        selected.append(bound[item.identity])
        presentations.append(dict(index=len(selected)-1,template_sha256=digest(item.identity),
            source=key[0],depth=key[1],occurrence=visits[item.identity],
            eligible_targets=count,consumed_targets=min(count,required-total),offset=total))
        total+=count;spent[key]+=count
    sources=TrainingSources(probes=tuple(selected),manifest_hashes={'inventory':digest(inventory)})
    actual=list(iter_update_targets([p.template for p in selected],8192));summaries=[];used=set()
    for index,update in enumerate(actual[:updates]):
        if update.valid_targets!=8192:raise ValueError('short input stream')
        counts={s:0 for s in weights};ds=set()
        for b,(i,start,end) in zip(update.batches,update.slices,strict=True):
            p=selected[i];counts[p.source]+=int(b.target_mask.sum());ds.add(depths[p.identity])
            for row in window_metadata(b):
                used.update((row['document_sha256'],row['window_sha256'],pos) for pos in row['target_positions'])
        if any(v==0 for v in counts.values()) or ds!=set(affected_depths):raise ValueError('incomplete update source/depth coverage')
        summaries.append(dict(step=index+1,targets=update.valid_targets,source_counts=counts,depths=sorted(ds)))
    metadata=dict(schema_version=2,policy='weighted_fair_whole_probe_v2',logical_targets=required,
        presentations=presentations,updates=summaries,unconsumed_tail_targets=total-required,
        unique_target_positions=len(used),unique_components=len({r['component_sha256'] for k,v in inventory.items() if visits[k] for r in v}),
        unique_windows=len({u[1] for u in used}),inventory={digest(k):v for k,v in inventory.items()},
        source_digest=sources.digest())
    return TargetPlan(sources,metadata,digest(metadata))


def fixed_parent_copy(trial, *, zero_product, seed):
    # Copy the student alone; keep the frozen teacher and live ledger outside.
    result=replace(trial,model=deepcopy(trial.model),recorder=None,sources=None,fit_summary={})
    if zero_product:
        with torch.no_grad():
            for depth in result.affected_depths:
                for index,name in enumerate(PROJECTIONS):
                    f=result.model.layers[depth].mlp.residuals[name]
                    a,b=initialize_residual(torch.zeros((f.B.shape[0],f.A.shape[1]),device=f.A.device,dtype=f.A.dtype),f.A.shape[0],seed+depth*3+index)
                    f.A.copy_(a);f.B.copy_(b)
    result.initialized_digest=result.model.manifest().checkpoint_digest
    result.fitted_digest=None
    return result

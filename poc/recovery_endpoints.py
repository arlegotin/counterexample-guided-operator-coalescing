"""Strict saved-evidence admission and paired component reductions.

No teacher, corpus reader, optimizer update or neural forward is used here.
The caller binds the actual finite design; these helpers also accept owned fixtures.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np
from poc.records import canonical_json


def hash_value(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def regular(path):
    p=Path(path)
    if not p.is_absolute() or p.resolve()!=p or p.is_symlink() or not p.is_file() or p.stat().st_nlink!=1:
        raise ValueError("canonical regular nonaliased file required: "+str(p))
    return p


def finite_json(path):
    def pairs(values):
        result={}
        for k,v in values:
            if k in result:raise ValueError("duplicate JSON key")
            result[k]=v
        return result
    def invalid(v):raise ValueError("nonfinite JSON constant: "+v)
    return json.loads(regular(path).read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def verify_refs(refs):
    seen={}
    for ref in refs:
        if set(ref)!={"path","sha256"} or not re.fullmatch("[0-9a-f]{64}",ref["sha256"]):
            raise ValueError("invalid detached file reference")
        p=regular(ref["path"])
        if str(p) in seen and seen[str(p)]!=ref["sha256"]:raise ValueError("conflicting reference")
        if file_sha(p)!=ref["sha256"]:raise ValueError("frozen file changed: "+str(p))
        seen[str(p)]=ref["sha256"]


def _number(value,*,nonnegative=False):
    if type(value) not in (int,float) or not math.isfinite(value) or nonnegative and value<0:
        raise ValueError("invalid finite numerical value")
    return value


def metric(value,targets,floor=2e-6):
    if set(value)!={"status","reason","valid_targets","kl","nll","disagreements"}:
        raise ValueError("metric schema differs")
    if value["status"]!="ok" or value["reason"] is not None:raise ValueError("failed/skipped metric")
    if type(targets)!=int or targets<1 or type(value["valid_targets"])!=int or value["valid_targets"]!=targets:
        raise ValueError("metric raw-target denominator differs")
    if _number(value["kl"]) < -floor*targets:raise ValueError("KL below qualified numerical floor")
    _number(value["nll"],nonnegative=True)
    if type(value["disagreements"])!=int or not 0<=value["disagreements"]<=targets:
        raise ValueError("disagreement denominator differs")
    return dict(value)


def row_identity(row):
    if set(row)!={"document_id","component_id","window_sha256","target_positions","full","local_by_depth"}:
        raise ValueError("observation row schema differs")
    for key in ("document_id","component_id","window_sha256"):
        if type(row[key])!=str or not re.fullmatch("[0-9a-f]{64}",row[key]):raise ValueError("invalid row identity")
    pos=row["target_positions"]
    if not isinstance(pos,list) or not pos or any(type(i)!=int or not 0<=i<511 for i in pos) or pos!=sorted(set(pos)):
        raise ValueError("invalid shifted target positions")
    return {k:row[k] for k in ("document_id","component_id","window_sha256","target_positions")}


def expected_update(order,documents,index,*,raw_targets=8192):
    if type(index)!=int or index<0 or type(raw_targets)!=int or raw_targets<1:
        raise ValueError("invalid finite update index")
    lower=index*raw_targets;upper=lower+raw_targets;rows=[]
    for j,i in enumerate(order):
        if type(i)!=int or not 0<=i<len(documents):raise ValueError("invalid plan template index")
        begin=max(lower,j*511);end=min(upper,(j+1)*511)
        if begin<end:
            d=documents[i]
            rows.append({"document_id":d["document_id"],"component_id":d["component_id"],"target_positions":list(range(begin-j*511,end-j*511))})
        if (j+1)*511>=upper:break
    if sum(len(r["target_positions"]) for r in rows)!=raw_targets:raise ValueError("finite plan exhausted")
    return rows


def _reduce(rows):
    sums={k:math.fsum(r["full"][k] for r in rows) for k in ("kl","nll","valid_targets","disagreements")}
    n=int(sums["valid_targets"])
    values=np.array([r["full"]["kl"]/r["full"]["valid_targets"] for r in rows],dtype=np.float64)
    return {"components":len(rows),"valid_targets":n,"raw_sums":sums,"kl":sums["kl"]/n,"nll":sums["nll"]/n,
            "disagreement_fraction":sums["disagreements"]/n,"perplexity":math.exp(sums["nll"]/n) if sums["nll"]/n<709 else None,
            "macro_component_kl":float(values.mean()),"component_q95_kl":float(np.quantile(values,.95)),"worst_component_kl":float(values.max())}


def validate_observation(obs,*,step,plan_sha256,study_sha256,tuning,optimizer,candidate_digest=None):
    if set(obs)!={"step","observation_policy","plan_sha256","candidate_digest","study_contract_sha256","populations"}:
        raise ValueError("observation schema differs")
    if type(obs["step"])!=int or obs["step"]!=step or obs["plan_sha256"]!=plan_sha256 or obs["study_contract_sha256"]!=study_sha256:
        raise ValueError("observation contract/plan/step differs")
    if obs["observation_policy"]!="optimizer_plus_tuning_v2" or not re.fullmatch("[0-9a-f]{64}",obs["candidate_digest"]):
        raise ValueError("observation policy/state differs")
    if candidate_digest is not None and obs["candidate_digest"]!=candidate_digest:raise ValueError("candidate linkage differs")
    if set(obs["populations"])!={"optimizer_windows","tuning"}:raise ValueError("diagnostic populations differ")
    result={}
    for name,expected in (("tuning",tuning),("optimizer_windows",optimizer)):
        rows=obs["populations"][name]
        if not isinstance(rows,list) or not rows or len(rows)!=len(expected):raise ValueError("diagnostic population size differs")
        seen=set()
        for row,want in zip(rows,expected,strict=True):
            ident=row_identity(row)
            if name=="tuning":
                if ident!=want or row["component_id"] in seen:raise ValueError("tuning component/context/mask differs")
                seen.add(row["component_id"])
            elif any(ident[k]!=v for k,v in want.items()):raise ValueError("optimizer update context/order/mask differs")
            n=len(ident["target_positions"]);metric(row["full"],n)
            if type(row["local_by_depth"])!=list or len(row["local_by_depth"])!=(0 if name=="tuning" else 2):
                raise ValueError("local depth discrepancy scope differs")
            for local in row["local_by_depth"]:metric(local,n)
        result[name]=_reduce(rows)
    n=result["optimizer_windows"]["valid_targets"]
    result["optimizer_local_sum_per_raw_target"]=math.fsum(m["kl"] for r in obs["populations"]["optimizer_windows"] for m in r["local_by_depth"])/n
    return result


def paired_summary(local,full,*,seed=17,draws=10000):
    if type(seed)!=int or seed<0 or type(draws)!=int or draws<1:raise ValueError("finite component bootstrap required")
    maps=[]
    for rows in (local,full):
        if not rows:raise ValueError("nonempty paired components required")
        by_id={}
        for row in rows:
            ident=row_identity(row);metric(row["full"],len(ident["target_positions"]))
            c=row["component_id"]
            if c in by_id:raise ValueError("duplicate component")
            by_id[c]=row
        maps.append(by_id)
    if maps[0].keys()!=maps[1].keys():raise ValueError("paired population differs")
    ids=sorted(maps[0]);a=[maps[0][i] for i in ids];b=[maps[1][i] for i in ids]
    if any(row_identity(x)!=row_identity(y) for x,y in zip(a,b,strict=True)):raise ValueError("paired contexts/masks differ")
    sizes=np.array([r["full"]["valid_targets"] for r in a],dtype=np.int64)
    sums=np.array([y["full"]["kl"]-x["full"]["kl"] for x,y in zip(a,b,strict=True)],dtype=np.float64)
    d=sums/sizes;rng=np.random.default_rng(seed);index=rng.integers(0,len(ids),(draws,len(ids)))
    # Keep every candidate column paired; population fidelity weights by target counts.
    values=sums[index].sum(axis=1)/sizes[index].sum(axis=1)
    interval=np.quantile(values,[.025,.975]).tolist();arms={"local_sum":_reduce(a),"full_student":_reduce(b)}
    return {"arms":arms,"component_ids":ids,"full_minus_local":{"kl_difference":float(sums.sum()/sizes.sum()),
      "nll_difference":arms["full_student"]["nll"]-arms["local_sum"]["nll"],"component_percentile95":interval,
      "better_components":int(np.sum(d<0)),"worse_components":int(np.sum(d>0)),"equal_components":int(np.sum(d==0)),
      "component_differences":d.tolist(),"contrast_draws":values.tolist()},
      "bootstrap":{"seed":seed,"draws":draws,"unit":"whole source/dedup component; fixed fitted states",
                   "common_index_draws_sha256":hashlib.sha256(index.tobytes()).hexdigest()},
      "diagnostic_resolution_targets":[.002,.005,.01],"numerics":{"negative_floor_per_target":2e-6,"decision_resolution":2e-5,"tiny_negative_raw_values":"retained"},
      "algorithmic_replication":"one pair/seed17; no algorithmic interval or confirmation","capability_tolerance":None,"compression_claim":False}


def validate_events(events,*,updates,scope,plan_sha256):
    if scope not in ("local_sum","full_student"):raise ValueError("unsupported paired objective")
    if len(events)!=updates or [e.get("step") for e in events]!=list(range(1,updates+1)) or any(e.get("kind")!="update" for e in events):
        raise ValueError("incomplete/noncanonical update history")
    multiplier=2 if scope=="local_sum" else 1
    for e in events:
        if any(type(e.get(k))!=int or e[k]!=v for k,v in {"step":e["step"],"valid_targets":8192,"raw_target_presentations":8192,"scored_target_presentations":8192*multiplier,"epoch":0}.items()):
            raise ValueError("raw/scored event denominator differs")
        if e.get("raw_plan_sha256")!=plan_sha256 or e.get("source_counts")!={"natural":8192}:raise ValueError("event plan/source differs")
        for k in ("gradient_norm_pre_clip","parameter_update_norm"):_number(e.get(k),nonnegative=True)
        if type(e.get("clip_applied"))!=bool or e["clip_applied"]!=(e["gradient_norm_pre_clip"]>1):raise ValueError("clip diagnostic differs")
        if set(e.get("learning_rates",{}))!={"bank","residual"}:raise ValueError("learning-rate owners differ")
        for v in e["learning_rates"].values():_number(v,nonnegative=True)
        if _number(e.get("kl_sum")) < -2e-6*8192*multiplier:raise ValueError("training KL below floor")
        _number(e.get("nll_sum"),nonnegative=True)
        if not math.isclose(_number(e.get("weighted_objective")),e["kl_sum"]/8192,rel_tol=1e-6,abs_tol=1e-9):
            raise ValueError("objective normalization differs")
        if e.get("source_kl_sums")!={"natural":e["kl_sum"]}:raise ValueError("source loss sums differ")
    return {"training_raw_targets":updates*8192,"training_scored_targets":updates*8192*multiplier,
            "clipped_updates":sum(e["clip_applied"] for e in events),"gradient_norms":[e["gradient_norm_pre_clip"] for e in events],
            "update_norms":[e["parameter_update_norm"] for e in events]}



def validate_completed_state(state,anchor,plan,training,objective):
    """Reconstruct a completed scientific decoder on CPU; never execute it."""
    import random
    import torch
    from poc.train import TrainingConfig,_optimizer,_schedule_state,_training_hash
    from poc.checkpoint import checkpoint_decoder
    from poc.recovery_objectives import affected_owners,owner_names_hash
    from poc.contracts import CostRecord
    cfg=dict(training)
    if cfg.pop("epoch_tail_policy",None)!="actual_denominator_v1":raise ValueError("unknown target-tail policy")
    for key in ("affected_depths","observation_steps"):cfg[key]=tuple(cfg[key])
    config=TrainingConfig(**cfg);updates=config.max_updates;raw=updates*8192
    if hash_value(config.identity())!=hash_value(training) or config.alpha!=0 or config.target_count!=8192:
        raise ValueError("noncanonical paired training configuration")
    if (state.spec!=anchor.spec or state.step!=updates or state.target_cursor!=raw or not state.safe_boundary
            or state.config_hash!=_training_hash(config,objective) or state.config_hash!=anchor.config_hash):
        raise ValueError("incomplete/changed recovery checkpoint")
    if (anchor.step!=0 or anchor.target_cursor or anchor.probe_cursor or anchor.visited or anchor.events
            or not anchor.safe_boundary or any(anchor.optimizer["state"].values())):
        raise ValueError("original STEP0 state required")
    if (state.source_hash!=objective.raw_plan_sha256 or state.source_hash!=anchor.source_hash
            or state.order!=("raw_plan_v1",state.source_hash) or state.order!=anchor.order or state.probe_order!=()):
        raise ValueError("checkpoint source/order differs")
    if (plan.get("updates")!=updates or plan.get("raw_targets_per_update")!=8192 or plan.get("raw_target_presentations")!=raw
            or plan.get("plan_sha256")!=state.source_hash):raise ValueError("finite-plan identity differs")
    if state.partition!=anchor.partition or state.model_schema!=anchor.model_schema or state.teacher_digest!=anchor.teacher_digest:
        raise ValueError("representation/teacher identity differs")
    multiplier=objective.scored_multiplier
    calls=sum(((i+1)*8192-1)//511-(i*8192)//511+1 for i in range(updates))*multiplier if objective.loss_scope!="full_student" else 0
    if state.probe_cursor!=calls:raise ValueError("local call cursor differs")
    if (len(state.visited)!=updates or state.visited[-1]!=state.candidate_digest
            or any(type(d)!=str or not re.fullmatch("[0-9a-f]{64}",d) for d in state.visited)):
        raise ValueError("candidate history differs")
    if state.scheduler!=_schedule_state(updates,updates) or anchor.scheduler!=_schedule_state(0,updates):
        raise ValueError("completed schedule differs")
    events=validate_events(state.events,updates=updates,scope=objective.loss_scope,plan_sha256=state.source_hash)
    for e in state.events:
        factor=_schedule_state(e["step"]-1,updates)["next_factor"]
        if e["learning_rates"]!={"bank":config.bank_lr*config.lr_factor*factor,"residual":config.residual_lr*config.lr_factor*factor}:
            raise ValueError("event learning rates differ from complete schedule")
    CostRecord(**state.budget_totals)
    if (state.budget_totals["updates"]!=updates or state.budget_totals["raw_corpus_target_presentations"]<raw
            or state.budget_totals["valid_target_presentations"]<raw*multiplier):
        raise ValueError("checkpoint spending below completed work")
    decoder=checkpoint_decoder(state)
    # The original anchor must itself reconstruct and match its stored digest.
    original=checkpoint_decoder(anchor)
    if decoder.manifest().tokenizer_digest!=original.manifest().tokenizer_digest:raise ValueError("tokenizer differs")
    owners=affected_owners(decoder,objective.affected_depths,owner_scope=objective.owner_scope)
    actual=sorted(r["name"] for r in state.model_schema["owners"] if r["trainable"])
    if sorted(owners)!=actual or owner_names_hash(decoder,objective.affected_depths,owner_scope=objective.owner_scope)!=objective.owner_names_sha256:
        raise ValueError("canonical trainable owner set differs")
    if state.model.keys()!=anchor.model.keys():raise ValueError("owner set changed")
    for name,value in state.model.items():
        if name not in owners and not torch.equal(value,anchor.model[name]):raise ValueError("frozen owner changed: "+name)
    optimizer=_optimizer(owners,config)
    if (set(anchor.optimizer)!={"groups","state"} or len(anchor.optimizer["groups"])!=len(optimizer.param_groups)
            or set(anchor.optimizer["state"])!=set(owners) or any(type(v)!=dict or v for v in anchor.optimizer["state"].values())):
        raise ValueError("anchor optimizer owner groups or empty moments differ")
    for actual_group,saved in zip(optimizer.param_groups,anchor.optimizer["groups"],strict=True):
        if set(actual_group)-{"params"}!=set(saved) or saved["owner_ids"]!=actual_group["owner_ids"]:
            raise ValueError("anchor optimizer canonical owner order/fields differ")
        if any(saved[k]!=actual_group[k] for k in set(saved)-{"lr","owner_ids"}):
            raise ValueError("anchor optimizer initial hyperparameters differ")
        if saved["lr"]!=actual_group["base_lr"]*anchor.scheduler["next_factor"]:
            raise ValueError("anchor optimizer initial scheduled LR differs")
    if set(state.optimizer)!={"groups","state"} or len(state.optimizer["groups"])!=len(optimizer.param_groups) or set(state.optimizer["state"])!=set(owners):
        raise ValueError("optimizer owner groups differ")
    for actual_group,saved in zip(optimizer.param_groups,state.optimizer["groups"],strict=True):
        if set(actual_group)-{"params"}!=set(saved) or saved["owner_ids"]!=actual_group["owner_ids"]:
            raise ValueError("optimizer owner order/fields differ")
        for key in set(saved)-{"lr","owner_ids"}:
            if saved[key]!=actual_group[key]:raise ValueError("optimizer hyperparameters differ")
        if saved["lr"]!=actual_group["base_lr"]*state.scheduler["next_factor"]:raise ValueError("optimizer final LR differs")
        for name,param in zip(actual_group["owner_ids"],actual_group["params"],strict=True):
            moments=state.optimizer["state"][name]
            if set(moments)!={"step","exp_avg","exp_avg_sq"}:raise ValueError("Adam moments missing")
            for key,value in moments.items():
                if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all().item():raise ValueError("nonfinite Adam moment")
                if key=="step":
                    if value.ndim!=0 or value.dtype not in (torch.float32,torch.float64) or value.item()!=updates:raise ValueError("Adam completed step differs")
                elif value.shape!=param.shape or value.dtype!=param.dtype:raise ValueError("Adam moment schema differs")
                elif key=="exp_avg_sq" and (value<0).any().item():raise ValueError("negative Adam second moment")
    for image in (anchor,state):
        if set(image.rng) not in ({"python","numpy","torch"},{"python","numpy","torch","mps"}):raise ValueError("checkpoint RNG schema differs")
        random.Random().setstate(image.rng["python"]);np.random.RandomState().set_state(image.rng["numpy"])
        image_torch=image.rng["torch"]
        if not isinstance(image_torch,torch.Tensor) or image_torch.dtype!=torch.uint8 or image_torch.ndim!=1 or image_torch.device.type!="cpu":
            raise ValueError("invalid CPU Torch RNG image")
        torch.Generator().set_state(image_torch)
        if image.model_schema["backend"]=="torch_mps" and "mps" not in image.rng:raise ValueError("missing source MPS RNG")
        if "mps" in image.rng:
            v=image.rng["mps"]
            # Qualified Torch2.14.1 serialization is44CPU bytes, established from
            # the hash-verified source STEP0 state. Never install an MPS RNG here.
            # A different Torch encoding needs its own profile and controls.
            if (str(torch.__version__).split("+")[0]!="2.14.1" or not isinstance(v,torch.Tensor)
                    or v.device.type!="cpu" or v.dtype!=torch.uint8 or v.ndim!=1 or v.numel()!=44):
                raise ValueError("invalid qualified Torch2.14.1 MPS RNG image; expected44CPU uint8 bytes")
    return dict(events,step=state.step,candidate_digest=state.candidate_digest,teacher_digest=state.teacher_digest,
                owner_names=actual,frozen_owners=len(state.model)-len(actual),probe_cursor=state.probe_cursor,
                scheduler=state.scheduler,cpu_decoder_reconstructed_without_teacher=True)



def write_once(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("xb") as stream:stream.write(canonical_json(value))


def require_terminal_pair(provenance):
    root=Path(provenance)
    if (root/"queue-failed-v2.json").exists():raise ValueError("paired run failed; no endpoint admission")
    if not (root/"queue-result-v2.json").is_file():raise ValueError("both successful terminal arms required")
    value=finite_json(root/"queue-result-v2.json")
    if value.get("status")!="completed_common_driver; independent endpoint acceptance pending" or set(value.get("results",{}))!={"local_sum","full_student"}:
        raise ValueError("incomplete terminal pair")
    verify_refs(list(value["results"].values()))
    return value

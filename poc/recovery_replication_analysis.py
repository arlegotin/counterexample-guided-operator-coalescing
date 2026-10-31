"""Version2 replication context and separately named uncertainty sources.

All inputs are saved evidence. No teacher, optimizer update or corpus reader.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import re
import sys
import numpy as np
from poc.recovery_endpoints import finite_json,verify_refs,file_sha,hash_value


@dataclass(frozen=True)
class RecoveryEndpointContext:
    seed: int
    stage: str
    command: tuple[str,...]
    schema_version: int = 2

    def __post_init__(self):
        object.__setattr__(self,"command",tuple(self.command))
        if type(self.schema_version)!=int or self.schema_version!=2 or type(self.seed)!=int or self.seed not in (23,31):
            raise ValueError("explicit replication seed23/31 and schema2 required")
        p=Path(self.stage)
        if not p.is_absolute() or p.resolve()!=p:raise ValueError("canonical analysis supervisor stage required")
        c=self.command
        if len(c)!=6 or c[0]!=sys.executable or not Path(c[1]).is_absolute() or not Path(c[3]).is_absolute() or c[2]!="--contract" or c[4:]!=("--mode","analyze"):
            raise ValueError("explicit frozen analysis command required")

    def digest(self):return hash_value(asdict(self))


def admit_replication_terminal(c,cfg,context):
    """Bind one successful Rqueue pair; a later seed failure cannot erase it."""
    if (not isinstance(context,RecoveryEndpointContext) or c.get("seed")!=context.seed or cfg.get("seeds")!=[context.seed]):
        raise ValueError("replication context/config seed differs")
    from provenance.sharing_frontier_study import require_stage_success,check_stage_binding
    seed=context.seed;root=Path(c["root"]);prov=root/"provenance/recovery-replications1024-v1"
    marker=prov/f"seed{seed}-common-driver-complete-v1.json"
    if not marker.is_file():raise ValueError("replication terminal pair required")
    done=finite_json(marker)
    if set(done)!={"stages","results","freeze"} or done["freeze"]!=c["freeze"] or set(done["results"])!={"local_sum","full_student"}:
        raise ValueError("replication terminal identity differs")
    verify_refs([done["freeze"],*done["results"].values()])
    freeze=finite_json(c["freeze"]["path"])
    if (freeze["seed"]!=seed or freeze["run_configuration"]!=c["run_config"] or freeze["study"]!=cfg["study"]
            or freeze["resource_budget"]!=cfg["resource_budget"] or freeze["quality_used_in_design"] is not False):
        raise ValueError("replication execution freeze differs")
    stage=done["stages"]["run"];require_stage_success(stage)
    caps={"wall_seconds":sum(b["wall_seconds"] for b in cfg["resource_budget"]["per_arm"].values())+1800,
          "memory_bytes":12*1024**3,"disk_bytes":24*1024**3}
    source=freeze["source_manifest"]
    required=[{"path":str(root/p),"sha256":h} for p,h in source["files"].items()]+[c["freeze"],c["run_config"],c["controller_source"]]
    check_stage_binding(stage,command=[sys.executable,"provenance/recovery_diagnosis.py","--config",c["run_config"]["path"],"--mode","run"],
        root=root,output=root/cfg["out"],caps=caps,required_refs=required,
        execution_env={"PYTHONPATH":str(root),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":"/private/tmp"})
    protocol=finite_json(stage["protocol_path"]);verify_refs(protocol["frozen_files"])
    for key,limit in (("wall_seconds","wall_seconds"),("sampled_process_tree_rss_bytes","memory_bytes"),("sampled_scratch_peak_bytes","disk_bytes")):
        value=stage["attempt"].get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=caps[limit]:raise ValueError("replication terminal resource measurement invalid")
    if any(done["results"][scope]["path"]!=str(root/cfg["out"]/"runs"/(scope+f"-seed{seed}")/"result.json") for scope in done["results"]):
        raise ValueError("replication result outside expected own arm")
    campaign=finite_json(root/cfg["out"]/"result.json")
    if (set(campaign["arms"])!={scope+f"-seed{seed}" for scope in cfg["loss_scopes"]}
            or any(v["status"]!="complete" for v in campaign["arms"].values())):raise ValueError("replication campaign incomplete")
    return done,stage


def aggregate_seeds(episodes,*,seed=17,draws=10000):
    """Text, optimization and crossed exploratory uncertainty kept separate."""
    if set(episodes)!={17,23,31} or any(type(s)!=int for s in episodes):raise ValueError("exact prespecified seeds17/23/31 required")
    if type(seed)!=int or seed<0 or type(draws)!=int or draws<1:raise ValueError("finite prespecified bootstrap required")
    component_ids=None;identity=None;matrix=[];by_seed={}
    expected_identity={"pair","rank","owner_names_sha256","teacher_digest","tuning_windows_sha256","raw_targets_per_arm","steps"}
    for s,e in sorted(episodes.items()):
        if (e.get("status")!="accepted_exploratory_endpoint" or type(e.get("seed"))!=int or e["seed"]!=s
                or e.get("capability_tolerance") is not None or e.get("compression_claim") is not False):
            raise ValueError("episode is not accepted under its actual seed/scope")
        ident=e["scientific_identity"]
        if (set(ident)!=expected_identity or ident["pair"]!=[0,15] or type(ident["rank"])!=int or ident["rank"]!=8
                or ident["steps"]!=1024 or ident["raw_targets_per_arm"]!=8388608):raise ValueError("prospective recovery identity differs")
        for k in ("owner_names_sha256","teacher_digest","tuning_windows_sha256"):
            if type(ident[k])!=str or not re.fullmatch("[0-9a-f]{64}",ident[k]):raise ValueError("missing scientific identity hash")
        if identity is None:identity=ident
        if identity!=ident:raise ValueError("paired scientific identities differ across seeds")
        ids=e["component_ids"];values=e["full_minus_local"]["component_differences"]
        if (type(ids)!=list or len(ids)<2 or len(ids)!=len(set(ids)) or len(ids)!=len(values)
                or any(type(i)!=str or not re.fullmatch("[0-9a-f]{64}",i) for i in ids)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
            raise ValueError("component identities/paired differences invalid")
        if component_ids is None:component_ids=sorted(ids)
        if set(ids)!=set(component_ids):raise ValueError("same complete source components required")
        vector=np.array([dict(zip(ids,values,strict=True))[i] for i in component_ids],dtype=np.float64)
        for name,arm in e["arms"].items():
            if name not in ("local_sum","full_student") or type(arm["components"])!=int or arm["components"]!=len(ids) or type(arm["valid_targets"])!=int or arm["valid_targets"]!=511*len(ids):
                raise ValueError("population target denominator differs")
            if any(type(arm[k]) not in (int,float) or not math.isfinite(arm[k]) for k in ("kl","nll")) or arm["kl"]<-2e-6 or arm["nll"]<0:
                raise ValueError("invalid episode population metric")
        if set(e["arms"])!={"local_sum","full_student"}:raise ValueError("both objective arms required")
        point=e["full_minus_local"]["kl_difference"]
        if (type(point) not in (int,float) or not math.isfinite(point) or abs(float(vector.mean())-point)>1e-14
                or abs(e["arms"]["full_student"]["kl"]-e["arms"]["local_sum"]["kl"]-point)>1e-14):
            raise ValueError("episode point and paired raw reductions differ")
        matrix.append(vector);by_seed[str(s)]={"kl_difference":point,"better_components":int((vector<0).sum()),
           "worse_components":int((vector>0).sum()),"equal_components":int((vector==0).sum()),"arms":e["arms"]}
    x=np.stack(matrix);rng=np.random.default_rng(seed)
    component_draws=rng.integers(0,x.shape[1],(draws,x.shape[1]));seed_draws=rng.integers(0,3,(draws,3))
    # One component index vector moves all seed/objective columns together.
    text=x[:,component_draws].mean(axis=(0,2));optimization=x[seed_draws,:].mean(axis=(1,2))
    crossed=np.array([x[np.ix_(s,c)].mean() for s,c in zip(seed_draws,component_draws,strict=True)])
    scopes={"components":"text uncertainty conditional on three fixed fitted pairs",
            "optimization_seeds":"paired optimization variation conditional on this fixed text; only3seeds",
            "crossed_sensitivity":"exploratory crossed seed/component resampling;3seed support is not powered confirmation"}
    uncertainty={name:{"contrast_draws":v.tolist(),"percentile95":np.quantile(v,[.025,.975]).tolist(),"scope":scopes[name]}
                 for name,v in (("components",text),("optimization_seeds",optimization),("crossed_sensitivity",crossed))}
    import hashlib
    return {"schema_version":2,"seed_ids":[17,23,31],"component_ids":component_ids,"scientific_identity":identity,
       "mean_full_minus_local":float(x.mean()),"optimization_seed_sd":float(x.mean(axis=1).std(ddof=1)),
       "per_seed":by_seed,"component_difference_matrix":x.tolist(),"uncertainty":uncertainty,
       "bootstrap":{"seed":seed,"draws":draws,"component_draws_sha256":hashlib.sha256(component_draws.tobytes()).hexdigest(),
                    "paired_seed_draws_sha256":hashlib.sha256(seed_draws.tobytes()).hexdigest()},
       "diagnostic_resolution_targets_nats":[.002,.005,.01],"capability_tolerance":None,
       "confirmation_claim":False,"equivalence_claim":False,"compression_claim":False,
       "scope":"one continuitypair/rank8/three exploratory seeds and exposeddevelopment; no applicationloss margin, practicalcompression or counterexample claim"}



def normalize_episode(seed,result,contract):
    """Bind oldv1 seed17 and explicitv2 seed23/31 to one aggregate schema."""
    if type(seed)!=int or seed not in (17,23,31) or result.get("status")!="accepted_exploratory_endpoint":raise ValueError("accepted explicit seed required")
    if seed!=17 and (result.get("seed")!=seed or result.get("schema_version")!=2):raise ValueError("version2 actual seed missing")
    names=None;teacher=None;owner_hash=None
    for scope in ("local_sum","full_student"):
        spec=contract["expected_specs"][scope];training=spec["resolved_config"]["training"];objective=spec["resolved_config"]["recovery_objective"]
        check=result["checkpoints"][scope]
        if (spec["seeds"]!={"fit":seed} or training["max_updates"]!=1024 or training["alpha"]!=0 or training["target_count"]!=8192
                or training["affected_depths"]!=[0,15] or objective["loss_scope"]!=scope or objective["owner_scope"]!="affected_group"
                or check["step"]!=1024 or check["training_raw_targets"]!=8388608):raise ValueError("episode seed/horizon/objective source differs")
        actual_names=check["owner_names"];actual_hash=hash_value({"value":actual_names})
        if actual_hash!=spec["input_manifests"]["owner_names"]:raise ValueError("episode owner-name identity differs")
        if names is None:names=actual_names;teacher=check["teacher_digest"];owner_hash=actual_hash
        if names!=actual_names or teacher!=check["teacher_digest"]:raise ValueError("paired owner/teacher source differs")
    identities=contract["tuning_identities"]
    if set(r["component_id"] for r in identities)!=set(result["component_ids"]) or len(identities)!=len(result["component_ids"]):raise ValueError("episode tuning component scope differs")
    normalized=dict(result,seed=seed,scientific_identity={"pair":[0,15],"rank":8,"owner_names_sha256":owner_hash,"teacher_digest":teacher,
        "tuning_windows_sha256":hash_value({"windows":sorted(identities,key=lambda r:r["component_id"])}),"raw_targets_per_arm":8388608,"steps":1024})
    return normalized



def pair_result_context(result,freeze,context):
    """Preserve legacy metadata; explicitly adapt the version2 producer schema."""
    if context is None:return result,freeze
    if not isinstance(context,RecoveryEndpointContext):raise ValueError("typed endpoint context required")
    result=dict(result,seed=context.seed,schema_version=2,endpoint_context_sha256=context.digest(),
        algorithmic_replication=f"one pair/seed{context.seed}; no algorithmic interval or confirmation")
    freeze={**freeze,"qualification":freeze["implementation_qualification_reused"],
            "accepted_profile_guard":freeze["accepted_profile_stage"],"predecessor_queue_failure":None}
    return result,freeze



def independent_pair_reductions(result,local_observation,full_observation):
    """Reconstruct raw reductions directly, without paired_summary/aggregate_seeds."""
    original={};ids=result["component_ids"]
    if ids!=sorted(set(ids)):raise ValueError("independent canonical component order differs")
    from poc.recovery_endpoints import row_identity,metric
    for name,obs in (("local_sum",local_observation),("full_student",full_observation)):
        rows=obs["populations"]["tuning"];mapped={r["component_id"]:r for r in rows}
        if len(mapped)!=len(rows) or set(mapped)!=set(ids):raise ValueError("independent component population differs")
        for r in rows:row_identity(r);metric(r["full"],511)
        sums={k:math.fsum(r["full"][k] for r in rows) for k in ("kl","nll","valid_targets","disagreements")}
        if sums["valid_targets"]!=511*len(ids) or any(abs(sums[k]-result["arms"][name]["raw_sums"][k])>1e-10 for k in sums):raise ValueError("independent raw metric sums differ")
        arm=result["arms"][name];n=sums["valid_targets"]
        for label,key in (("kl","kl"),("nll","nll"),("disagreement_fraction","disagreements")):
            if abs(arm[label]-sums[key]/n)>1e-14:raise ValueError("independent reported population mean differs")
        original[name]=mapped
    for i in ids:
        if row_identity(original["local_sum"][i])!=row_identity(original["full_student"][i]):raise ValueError("independent paired contexts differ")
    delta=np.array([original["full_student"][i]["full"]["kl"]/511-original["local_sum"][i]["full"]["kl"]/511 for i in sorted(ids)])
    draw_index=np.random.default_rng(17).integers(0,len(ids),(10000,len(ids)));values=delta[draw_index].mean(axis=1)
    contrast=result["full_minus_local"];recorded=np.asarray(contrast["contrast_draws"],dtype=np.float64)
    components=np.asarray(contrast["component_differences"],dtype=np.float64)
    if components.shape!=delta.shape or np.max(np.abs(components-delta))>1e-14:raise ValueError("independent paired component vector differs")
    if recorded.shape!=values.shape:raise ValueError("independent bootstrap size differs")
    error=float(np.max(np.abs(values-recorded)))
    if (error>1e-14 or abs(float(delta.mean())-contrast["kl_difference"])>1e-14
            or np.max(np.abs(np.quantile(values,[.025,.975])-np.asarray(contrast["component_percentile95"])))>1e-14):raise ValueError("independent paired bootstrap differs")
    if (contrast["better_components"],contrast["worse_components"],contrast["equal_components"])!=(int((delta<0).sum()),int((delta>0).sum()),int((delta==0).sum())):raise ValueError("independent component signs differ")
    return {"status":"independent_arithmetic_passed","components":len(ids),"valid_targets":511*len(ids),"max_draw_error":error,
            "numeric_error_bound":1e-14,"neural_calls":0,"corpus_reads":0,"scope":"original raw sums/paired contexts/10000whole-component bootstrap draws, independent estimator arithmetic"}



def require_analysis_qualification(root):
    path=Path(root)/"provenance/recovery-replication-analysis-v2/qualification-v1.json"
    if not path.is_file():raise ValueError("analysis source qualification required")
    q=finite_json(path)
    if q.get("status")!="qualified_replication_analysis" or set(q.get("stages",{}))!={"default_cpu","public","aggregate_validate"}:
        raise ValueError("analysis qualification is incomplete")
    verify_refs(q["frozen_refs"])
    from provenance.sharing_frontier_study import require_stage_success
    for stage in q["stages"].values():require_stage_success(stage["stage"])
    return q



def _study_input_snapshot(root):
    from poc.recovery_endpoints import regular
    root=Path(root)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir() or root.is_symlink():raise ValueError("inventory requires canonical study root")
    entries=sorted(root.rglob("*"));files={}
    if any(p.is_symlink() for p in entries):raise ValueError("inventory refuses linked input entries")
    for path in entries:
        if not path.is_file():continue
        regular(path);before=path.stat();h=file_sha(path);after=path.stat()
        signature=lambda s:(s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
        if signature(before)!=signature(after):raise ValueError("inventory input changed while hashing")
        files[str(path.relative_to(root))]={"sha256":h,"bytes":after.st_size}
    if not files:raise ValueError("inventory requires completed study inputs")
    now=sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    if now!=sorted(files):raise ValueError("inventory entries changed during snapshot")
    return files


def freeze_study_inputs(root,path):
    """Seal the entire consumed seed output, including profile/setup attempts."""
    from poc.recovery_endpoints import write_once
    root=Path(root);files=_study_input_snapshot(root)
    manifest={"schema_version":1,"kind":"cgoc_saved_study_input_inventory_v1","root":str(root),"files":files,
              "file_count":len(files),"total_bytes":sum(v["bytes"] for v in files.values())}
    write_once(path,manifest)
    ref={"path":str(Path(path)),"sha256":file_sha(path)}
    verify_study_inputs(ref,expected_root=root)
    return ref


def verify_study_inputs(reference,*,expected_root=None):
    verify_refs([reference]);manifest=finite_json(reference["path"])
    if (set(manifest)!={"schema_version","kind","root","files","file_count","total_bytes"}
            or type(manifest["schema_version"])!=int or manifest["schema_version"]!=1
            or manifest["kind"]!="cgoc_saved_study_input_inventory_v1"):raise ValueError("inventory schema differs")
    root=Path(manifest["root"])
    if expected_root is not None and root!=Path(expected_root):raise ValueError("inventory study root differs")
    if type(manifest["file_count"])!=int or type(manifest["total_bytes"])!=int:raise ValueError("inventory numerical schema differs")
    files=_study_input_snapshot(root)
    if files!=manifest["files"] or manifest["file_count"]!=len(files) or manifest["total_bytes"]!=sum(v["bytes"] for v in files.values()):
        raise ValueError("inventory files, attempts or bytes changed")
    return manifest


def require_context_study_inputs(c,cfg,context):
    """The dynamic inventory itself must be in this execution's frozen refs."""
    path=Path(context.stage).parent/f"seed{context.seed}-input-inventory-v1.json"
    pp=Path(context.stage)/"protocol.json"
    if not path.is_file() or not pp.is_file():raise ValueError("context input inventory required")
    p=finite_json(pp);bound={r["path"]:r["sha256"] for r in p["frozen_files"]}
    digest=bound.get(str(path))
    if digest is None:raise ValueError("context inventory not bound to supervisor")
    ref={"path":str(path),"sha256":digest}
    verify_study_inputs(ref,expected_root=Path(c["root"])/cfg["out"])
    return ref


def verify_episode_input_inventory(admission,contract,seed,worker):
    if type(seed)!=int or seed not in (23,31):raise ValueError("replication inventory seed differs")
    path=Path(worker)/f"provenance/recovery-replication-analysis-v2/seed{seed}-input-inventory-v1.json"
    stage=admission["stage"];verify_refs([{"path":stage["protocol_path"],"sha256":stage["protocol_sha256"]}])
    protocol=finite_json(stage["protocol_path"]);bound={r["path"]:r["sha256"] for r in protocol["frozen_files"]}
    expected={"path":str(path),"sha256":bound.get(str(path))}
    if expected["sha256"] is None or admission["input_inventory"]!=expected:raise ValueError("episode inventory differs from original stage")
    return verify_study_inputs(expected,expected_root=Path(contract["root"])/f"artifacts/recovery-1024-seed{seed}-v1")

"""Closed version2 metadata freeze, saved-pair admission and3seed analysis."""
from pathlib import Path
import argparse
import sys
import math
import json
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.recovery_endpoints import finite_json,file_sha,verify_refs,write_once,hash_value,row_identity
from poc.recovery_replication_analysis import RecoveryEndpointContext,normalize_episode,aggregate_seeds
from poc.contracts import Budget
from poc.checkpoint import _spec
from provenance import recovery_endpoint_analysis as common
from provenance.recovery_diagnosis import design
from poc.recovery_study import load_study_config


def context_for(seed,contract_path):
    return RecoveryEndpointContext(seed,str(ROOT/f"provenance/recovery-replication-analysis-v2/seed{seed}-final-analysis-stage-v1"),
       (sys.executable,str(ROOT/"provenance/recovery_replication_analysis.py"),"--contract",str(Path(contract_path).absolute()),"--mode","analyze"))


def require_profile_budgets(cfg,freeze,reference,*,producer_root=None,controller_source=None):
    if design(cfg)!=design(reference) or cfg["resource_budget"]!=freeze["resource_budget"]:raise ValueError("prospective design or resource freeze differs")
    verify_refs([freeze["profile"],freeze["accepted_profile_stage"],freeze["actual_controls"]])
    from provenance.sharing_frontier_study import require_stage_success,check_stage_binding
    if producer_root is None or controller_source is None:raise ValueError("explicit profile source frame required")
    root=Path(producer_root)
    if root.resolve()!=root or Path(freeze["profile"]["path"])!=root/reference["out"]/"profile.json":raise ValueError("profile source root/output differs")
    verify_refs([freeze["profile_configuration"],freeze["implementation_qualification_reused"],freeze["upstream_independent_acceptance"],controller_source])
    measured=finite_json(freeze["accepted_profile_stage"]["path"]);require_stage_success(measured)
    required=[{"path":str(root/p),"sha256":h} for p,h in freeze["source_manifest"]["files"].items() if p.endswith(".py") or p=="pyproject.toml"]
    required +=[freeze["profile_configuration"],freeze["implementation_qualification_reused"],freeze["upstream_independent_acceptance"],controller_source]
    check_stage_binding(measured,command=[sys.executable,"provenance/recovery_diagnosis.py","--config",freeze["profile_configuration"]["path"],"--mode","profile"],
        root=root,output=root/reference["out"],caps={"wall_seconds":3600,"memory_bytes":12*1024**3,"disk_bytes":24*1024**3},required_refs=required,
        execution_env={"PYTHONPATH":str(root),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":"/private/tmp"})
    profile=finite_json(freeze["profile"]["path"])
    controls=finite_json(freeze["actual_controls"]["path"])
    if (controls["backend"]["passed"] is not True or controls["backend"]["device"]!=reference["numerical_profile"]["device"]
            or controls["implementation_sha256"]!=profile["implementation_sha256"] or controls["environment_sha256"]!=profile["environment_sha256"]
            or profile["controls_receipt_sha256"]!=freeze["actual_controls"]["sha256"]):raise ValueError("actual backend controls/profile identity differs")
    if (profile["design_sha256"]!=hash_value(design(reference)) or profile["implementation_sha256"]!="9743af6ba20fc12b2c51cccc4b0df07cf508462b0ff37b9765e61676bc3d6fe3"
            or set(profile["arms"])!={"local_sum","full_student"} or cfg["resource_budget"]["frozen_profile_sha256"]!=freeze["profile"]["sha256"]):raise ValueError("per-seed actual profile source differs")
    for scope,row in profile["arms"].items():
        seconds=row["forecast_seconds_per_seed"]
        if (type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0 or row["planned_updates"]!=1024
                or row["profile_updates"]!=1 or row["cost"]["memory_bytes"]>8*1024**3):raise ValueError("per-seed profile incomplete")
        wall=math.ceil((1.5*seconds+300)/60)*60
        if wall>43200:raise ValueError("per-arm12h ceiling exceeded")
        counts=row["forecast_counters"];expected={k:(1024 if k=="updates" else math.ceil(1.2*counts.get(k,0))) for k in ("updates","valid_target_presentations","teacher_forward_calls","suffix_backward_calls","student_backward_calls","search_proposals","search_seconds")}
        expected.update(wall_seconds=wall,memory_bytes=8*1024**3,disk_bytes=24*1024**3)
        if cfg["resource_budget"]["per_arm"][scope]!=Budget(**expected).to_dict():raise ValueError("profile-derived finite budget differs")
    return profile


def load_contract(path):
    path=Path(path).absolute();c=finite_json(path)
    fields={"schema_version","study","seed","root","out","freeze","run_config","reference_configuration","controller_source","source_refs","analysis_refs","expected_specs","plan","initialization","tuning_identities","fit_documents","analysis","caps","context"}
    if set(c)!=fields or type(c["schema_version"])!=int or c["schema_version"]!=2:raise ValueError("unsupported replication analysis schema")
    seed=c["seed"]
    if type(seed)!=int or seed not in (23,31) or c["study"]!=f"recovery1024-seed{seed}-endpoint-v2":raise ValueError("prospective replication seed/study differs")
    ctx=context_for(seed,path)
    if hash_value(c["context"])!=ctx.digest():raise ValueError("analysis context source/command differs")
    expected_analysis={"seed":17,"draws":10000,"primary":"full_student_minus_local_sum_at1024","actual_seed":seed,"component_count":128,"capability_tolerance":None,"diagnostic_resolution_targets":[.002,.005,.01]}
    if c["analysis"]!=expected_analysis or c["caps"]!={"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2}:raise ValueError("prospective endpoint analysis changed")
    root=Path(c["root"])
    if root.resolve()!=root or c["out"]!=str(ROOT/f"artifacts/recovery-replication-analysis-v2/seed{seed}"):raise ValueError("analysis output/root isolation differs")
    verify_refs(c["source_refs"]+c["analysis_refs"]+[c[k] for k in ("freeze","run_config","reference_configuration","controller_source","plan","initialization")])
    if c["controller_source"]["sha256"]!="fb9851aaec65c857853762879a42f3ffc2848db9d6331a391f98f0abdce28036":raise ValueError("qualified replication controller changed")
    cfg=load_study_config(c["run_config"]["path"]);reference=load_study_config(c["reference_configuration"]["path"]);freeze=finite_json(c["freeze"]["path"])
    if (cfg["study"]!=f"recovery-1024-seed{seed}-v1" or cfg["out"]!=f"artifacts/recovery-1024-seed{seed}-v1" or cfg["seeds"]!=[seed]
            or cfg["pair"]!=[0,15] or cfg["rank"]!=8 or cfg["owner_scope"]!="affected_group" or cfg["alpha"]!=0
            or cfg["loss_scopes"]!=["local_sum","full_student"] or cfg["updates"]!=1024 or cfg["raw_targets_per_update"]!=8192
            or cfg["observations"]!=[0,1,4,16,32,64,128,256,512,1024] or cfg["checkpoint_every"]!=16
            or cfg["observation_policy"]!="optimizer_plus_tuning_v2" or cfg["initializer"]["method"]!="parent_svd"):
        raise ValueError("prospective replication design differs")
    if freeze["profile_configuration"]!=c["reference_configuration"]:raise ValueError("profile configuration reference differs")
    require_profile_budgets(cfg,freeze,reference,producer_root=root,controller_source=c["controller_source"])
    if freeze["seed"]!=seed or freeze["run_configuration"]!=c["run_config"] or freeze["quality_used_in_design"] is not False:raise ValueError("source freeze identity differs")
    if set(c["expected_specs"])!={"local_sum","full_student"}:raise ValueError("both exact RunSpecs required")
    plan=finite_json(c["plan"]["path"])
    if plan["seed"]!=seed or plan["updates"]!=1024 or plan["raw_target_presentations"]!=8388608:raise ValueError("source finite plan differs")
    for scope,raw in c["expected_specs"].items():
        spec=_spec(raw);training=spec.resolved_config["training"];obj=spec.resolved_config["recovery_objective"]
        if (spec.seeds!={"fit":seed} or spec.method_variant!=cfg["study"]+":"+scope or spec.budgets.to_dict()!=cfg["resource_budget"]["per_arm"][scope]
                or spec.resolved_config["study_contract_sha256"]!=hash_value(cfg) or training["max_updates"]!=1024 or training["alpha"]!=0
                or obj["loss_scope"]!=scope or obj["owner_scope"]!="affected_group" or obj["affected_depths"]!=(0,15)
                or spec.input_manifests["raw_plan"]!=plan["plan_sha256"] or spec.input_manifests["profile"]!=freeze["profile"]["sha256"]):
            raise ValueError("exact per-arm seed/objective/source identity differs")
    if len(c["tuning_identities"])!=128 or len(c["fit_documents"])!=512:raise ValueError("whole source populations required")
    seen=set()
    for ident in c["tuning_identities"]:
        row_identity(dict(ident,full={},local_by_depth=[]))
        if ident["target_positions"]!=list(range(511)) or ident["component_id"] in seen:raise ValueError("tuning masks/components differ")
        seen.add(ident["component_id"])
    if len({r["document_id"] for r in c["fit_documents"]})!=512 or len({r["component_id"] for r in c["fit_documents"]})!=512 or seen&{r["component_id"] for r in c["fit_documents"]}:raise ValueError("fit/tuning role-component identity differs")
    return c,cfg,ctx


def freeze_pair(root,seed,out_path):
    """Freeze metadata identities only; no metric quality is read for decisions."""
    if seed not in (23,31):raise ValueError("explicit seed23/31 required")
    root=Path(root).resolve();out_path=Path(out_path).absolute();prov=root/"provenance/recovery-replications1024-v1"
    preparation=finite_json(prov/"preparation-v1.json");freeze_path=prov/f"seed{seed}-execution-freeze-v1.json";freeze=finite_json(freeze_path)
    config_ref=freeze["run_configuration"];cfg=load_study_config(config_ref["path"]);runs=root/cfg["out"]/"runs"
    reference=preparation["replications"][str(seed)]["config"]
    initial=finite_json(runs/f"local_sum-seed{seed}/observations/0000.json")
    identities=[row_identity(r) for r in initial["populations"]["tuning"]]
    fit=finite_json(cfg["populations"]["fit"]["path"]);documents=[{"document_id":r["document_id"],"component_id":fit["membership"][r["document_id"]]} for r in fit["records"]]
    base=finite_json(runs/f"local_sum-seed{seed}/spec.json");specs={}
    for scope in cfg["loss_scopes"]:
        spec=deepcopy(base);spec["method_variant"]=cfg["study"]+":"+scope;spec["budgets"]=cfg["resource_budget"]["per_arm"][scope];spec["resolved_config"]["recovery_objective"]["loss_scope"]=scope;specs[scope]=spec
    ctx=context_for(seed,out_path)
    source_refs=[{"path":str(root/p),"sha256":h} for p,h in freeze["source_manifest"]["files"].items()]
    ref=lambda p:{"path":str(p),"sha256":file_sha(p)}
    source_refs +=[ref(prov/"preparation-v1.json"),ref(prov/"metadata-acceptance-v1.json"),ref(runs/"executed-source.json"),ref(runs/"contract.json"),ref(runs/"environment.json")]
    source_refs +=[{"path":r["path"],"sha256":r["sha256"]} for r in cfg["populations"].values()]
    from provenance.recovery_diagnosis import executed_sources
    analysis_sources=executed_sources(out_path) if out_path.exists() else executed_sources(reference["path"])
    analysis_refs=[{"path":str(ROOT/p),"sha256":h} for p,h in analysis_sources["files"].items()]
    c={"schema_version":2,"study":f"recovery1024-seed{seed}-endpoint-v2","seed":seed,"root":str(root),"out":str(ROOT/f"artifacts/recovery-replication-analysis-v2/seed{seed}"),
       "freeze":ref(freeze_path),"run_config":config_ref,"reference_configuration":reference,
       "controller_source":ref(Path('/private/tmp/cgoc_recovery1024_replications_queue_v1.py')),
       "source_refs":source_refs,"analysis_refs":analysis_refs,"expected_specs":specs,"plan":ref(runs/f"plan-{seed}.json"),"initialization":ref(runs/f"initialization-{seed}.json"),
       "tuning_identities":identities,"fit_documents":documents,"analysis":{"seed":17,"draws":10000,"primary":"full_student_minus_local_sum_at1024","actual_seed":seed,"component_count":128,"capability_tolerance":None,"diagnostic_resolution_targets":[.002,.005,.01]},
       "caps":{"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2},"context":__import__('dataclasses').asdict(ctx)}
    verify_refs(source_refs+analysis_refs);write_once(out_path,c);load_contract(out_path)
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True);p.add_argument("--mode",choices=("freeze","validate","analyze"),required=True);p.add_argument("--producer-root",type=Path);p.add_argument("--seed",type=int);a=p.parse_args();a.contract=a.contract.absolute()
    if a.mode=="freeze":
        if a.producer_root is None or a.seed is None:raise ValueError("freeze needs explicit producer root/seed")
        freeze_pair(a.producer_root,a.seed,a.contract)
    else:
        c,cfg,ctx=load_contract(a.contract)
        if a.mode=="analyze":
            from poc.recovery_replication_analysis import independent_pair_reductions,require_analysis_qualification
            require_analysis_qualification(ROOT)
            result=common.analyze(c,cfg,a.contract,context=ctx)
            runs=Path(c["root"])/cfg["out"]/"runs"
            left=finite_json(runs/f"local_sum-seed{ctx.seed}/observations/1024.json");right=finite_json(runs/f"full_student-seed{ctx.seed}/observations/1024.json")
            check=independent_pair_reductions(result,left,right)
            counters={}
            for scope,entry in result["checkpoints"].items():
                sums={k:0 for k in ("updates","raw_corpus_target_presentations","valid_target_presentations")}
                path=Path(entry["ledger"]["path"])
                if file_sha(path)!=entry["ledger"]["sha256"]:raise ValueError("independent ledger changed")
                with path.open() as stream:
                    for line in stream:
                        cost=json.loads(line)["cost"]
                        for key in sums:sums[key]+=cost.get(key,0)
                if any(sums[k]!=entry["ledger"]["totals"][k] for k in sums):raise ValueError("independent terminal integer work differs")
                counters[scope]=sums
            write_once(Path(c["out"])/"independent-arithmetic.json",dict(check,terminal_counters=counters,contract_sha256=file_sha(a.contract)))
        else:print(json.dumps({"status":"valid_replication_analysis","seed":ctx.seed,"contract_sha256":file_sha(a.contract),"neural_calls":0,"corpus_reads":0}))

if __name__=="__main__":main()

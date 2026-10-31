"""Fixed17/23/31 aggregate after independently accepted saved-evidence pairs."""
from pathlib import Path
import argparse
import sys
import json
import os
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.recovery_endpoints import finite_json,verify_refs,file_sha,write_once,hash_value
from poc.recovery_replication_analysis import normalize_episode,aggregate_seeds
from provenance.sharing_frontier_study import require_stage_success,check_stage_binding


def load_contract(path):
    c=finite_json(Path(path).absolute())
    fields={"schema_version","study","out","episodes","analysis_plan","source_refs","source_qualification_path","analysis","caps"}
    if set(c)!=fields or type(c["schema_version"])!=int or c["schema_version"]!=2:raise ValueError("unsupported aggregate contract schema")
    if (c["study"]!="recovery1024-seeds17-23-31-aggregate-v2" or c["out"]!=str(ROOT/"artifacts/recovery-replication-analysis-v2/aggregate")
            or c["analysis"]!={"seeds":[17,23,31],"draws":10000,"rng_seed":17,"primary":"mean_full_student_minus_local_sum_at_final1024","capability_tolerance":None}
            or c["caps"]!={"wall_seconds":180,"memory_bytes":2*1024**3,"disk_bytes":128*1024**2}):raise ValueError("prospective aggregate analysis changed")
    if set(c["episodes"])!={"17","23","31"}:raise ValueError("all three prespecified episodes required")
    if c["source_qualification_path"]!=str(ROOT/"provenance/recovery-replication-analysis-v2/qualification-v1.json"):raise ValueError("own source qualification path differs")
    verify_refs(c["source_refs"]+[c["analysis_plan"]])
    plan=finite_json(c["analysis_plan"]["path"])
    if plan["seed_ids"]!=[17,23,31] or plan["primary_contrast"]!="mean_across_fixed_seeds(full_student_KL_minus_local_sum_KL_at_final1024)" or plan["application_tolerance"] is not None:raise ValueError("original prospective aggregate plan differs")
    for seed,entry in c["episodes"].items():
        if set(entry)!={"admission_path","contract_path","result_path","worker_root","qualification_path"} or any(not Path(v).is_absolute() or Path(v).resolve()!=Path(v) for v in entry.values()):raise ValueError("fixed episode paths required")
        worker=Path(entry["worker_root"])
        if seed=="17":
            expected={"admission_path":str(worker/"provenance/recovery-endpoint-v1/independent-admission-v1.json"),"contract_path":str(worker/"configs/recovery-endpoint-seed17-v1.json"),"result_path":str(worker/"artifacts/recovery-endpoint-v1/result.json"),"qualification_path":str(worker/"provenance/recovery-endpoint-v1/qualification-v1.json")}
        else:
            if worker!=ROOT:raise ValueError("replication worker root differs")
            expected={"admission_path":str(ROOT/f"provenance/recovery-replication-analysis-v2/seed{seed}-independent-admission-v1.json"),"contract_path":str(ROOT/f"configs/recovery-seed{seed}-endpoint-v2.json"),"result_path":str(ROOT/f"artifacts/recovery-replication-analysis-v2/seed{seed}/result.json"),"qualification_path":c["source_qualification_path"]}
        if any(entry[k]!=v for k,v in expected.items()):raise ValueError("fixed episode artifact/contract paths differ")
    return c


def admit_episodes(c):
    episodes={};refs=[]
    for key in ("17","23","31"):
        e=c["episodes"][key];path=Path(e["admission_path"])
        if not path.is_file():raise ValueError("all three complete independently accepted episodes required")
        a=finite_json(path);expected_status="accepted_exploratory_paired1024" if key=="17" else "accepted_exploratory_replication1024"
        if a.get("status")!=expected_status or a.get("contract",{}).get("path")!=e["contract_path"] or a.get("result",{}).get("path")!=e["result_path"] or a.get("qualification",{}).get("path")!=e["qualification_path"]:raise ValueError("episode admission identity differs")
        verify_refs([a[k] for k in ("contract","result","qualification","terminal")])
        q=finite_json(a["qualification"]["path"])
        if key=="17" and q["status"]!="qualified_source_only; endpoints pending" or key!="17" and q["status"]!="qualified_replication_analysis":raise ValueError("episode source qualification absent")
        verify_refs(q["frozen_refs"])
        for stage in q["stages"].values():require_stage_success(stage["stage"])
        result=finite_json(a["result"]["path"]);contract=finite_json(a["contract"]["path"])
        if result["contract_sha256"]!=a["contract"]["sha256"]:raise ValueError("episode result contract differs")
        terminal=finite_json(a["terminal"]["path"])
        report=Path(e["result_path"]).parent/"REPORT.md"
        if (terminal["status"]!="complete" or terminal["result_sha256"]!=a["result"]["sha256"]
                or terminal["report_sha256"]!=file_sha(report)):raise ValueError("episode terminal result/report differs")
        worker=Path(e["worker_root"]);script="recovery_endpoint_analysis.py" if key=="17" else "recovery_replication_analysis.py"
        command=[sys.executable,str(worker/"provenance"/script),"--contract",e["contract_path"],"--mode","analyze"]
        required=contract["source_refs"]+contract["analysis_refs"]+[a["contract"],a["qualification"]]
        check_stage_binding(a["stage"],command=command,root=worker,output=Path(e["result_path"]).parent,caps=contract["caps"],required_refs=required,
            execution_env={"PYTHONPATH":str(worker),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":"/private/tmp"})
        if key!="17":
            verify_refs([a["independent_arithmetic"]]);ind=finite_json(a["independent_arithmetic"]["path"])
            if ind["status"]!="independent_arithmetic_passed" or ind["contract_sha256"]!=a["contract"]["sha256"]:raise ValueError("independent replication arithmetic absent")
            from poc.recovery_replication_analysis import verify_episode_input_inventory
            verify_episode_input_inventory(a,contract,int(key),worker)
            refs.extend([a["independent_arithmetic"],a["input_inventory"]])
        episodes[int(key)]=normalize_episode(int(key),result,contract)
        refs +=[{"path":str(path),"sha256":file_sha(path)}]
        refs +=[a[k] for k in ("contract","result","qualification","terminal")]
        refs.append({"path":str(report),"sha256":file_sha(report)})
    return episodes,refs


def require_guard(c,config_path,inputs):
    stage=ROOT/"provenance/recovery-replication-analysis-v2/aggregate-final-analysis-stage-v1"
    pp=stage/"protocol.json";sp=stage/"stage-start.json"
    if not pp.is_file() or not sp.is_file():raise ValueError("aggregate requires exact finite supervisor")
    p=finite_json(pp);start=finite_json(sp)
    command=[sys.executable,str(ROOT/"provenance/recovery_seed_aggregate.py"),"--config",str(Path(config_path).absolute()),"--mode","analyze"]
    if (start["pid"]!=os.getppid() or start["protocol_sha256"]!=file_sha(pp) or p["command"]!=command or p["root"]!=str(ROOT)
            or p["output"]!=c["out"] or p["provenance"]!=str(stage) or p["caps"]!=c["caps"]
            or p["supervisor_sha256"]!="a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba"):
        raise ValueError("aggregate supervisor/source/resource frame differs")
    verify_refs(p["frozen_files"]);bound={r["path"]:r["sha256"] for r in p["frozen_files"]}
    required=c["source_refs"]+[c["analysis_plan"],{"path":str(Path(config_path).absolute()),"sha256":file_sha(config_path)},*inputs]
    if any(bound.get(r["path"])!=r["sha256"] for r in required):raise ValueError("aggregate supervisor omitted frozen inputs")


def analyze(c,path):
    from poc.recovery_replication_analysis import require_analysis_qualification
    require_analysis_qualification(ROOT)
    episodes,refs=admit_episodes(c);require_guard(c,path,refs)
    out=Path(c["out"])
    if out.exists():raise FileExistsError("aggregate output already exists")
    out.mkdir(parents=True,exist_ok=False);write_once(out/"intent.json",{"status":"started","contract_sha256":file_sha(path)})
    try:
        result=aggregate_seeds(episodes,seed=c["analysis"]["rng_seed"],draws=c["analysis"]["draws"])
        result.update(status="accepted_exploratory_seed_aggregate",contract_sha256=file_sha(path),episode_inputs=refs,
            cost_scope="Each independently accepted pair retains its own raw/scored/diagnostic ledger, shared preparation and failures. Aggregate CPU stage reported separately; overlapping supervisor/ledger timings are not added; no fixed-total-work claim.",
            scale_sensitivity="Only local_sum/full_student are in this1024primary; prior256local_mean diagnostic is separate. No1024scale-only or matchedHPO contrast has been completed, so this is an objective/normalization recipe result, not sole evidence that full-model interactions are required.")
        verify_refs(c["source_refs"]+refs);write_once(out/"result.json",result)
        lines=["Completed1024 recovery across fixed seeds17,23,31, one continuitypair/rank8 and128alreadyexposed tuningcomponents.","", "| Seed | Full minus local KL |","|---|---:|"]
        for seed,v in result["per_seed"].items():lines.append(f"| {seed} | {v['kl_difference']:.10f} |")
        lines +=["",f"Mean full minus local: {result['mean_full_minus_local']:.10f}nats/target. Negative favors full_student.",""]
        for name,v in result["uncertainty"].items():lines.append(name+":95%percentile interval "+str(v["percentile95"])+"; "+v["scope"]+".")
        lines +=["","Paired source components move together across every objective/seed column. Complete paired optimization episodes are resampled separately; crossed resampling is an exploratory sensitivity with only three seeds. These intervals do not certify equivalence, capability preservation or generalization to other merges/populations.0.002/0.005/0.01nats remain diagnostic resolution targets, not acceptable loss margins. No counterexample or deployment-compression claim is made."]
        lines +=["",result["scale_sensitivity"]]
        (out/"REPORT.md").write_text("\n".join(lines)+"\n");write_once(out/"terminal.json",{"status":"complete","result_sha256":file_sha(out/"result.json"),"report_sha256":file_sha(out/"REPORT.md"),"contract_sha256":file_sha(path)})
    except BaseException as exc:
        try:write_once(out/"failed.json",{"status":"failed","error":repr(exc),"no_automatic_retry":True})
        except BaseException as publication:exc.add_note("failure publication also failed: "+repr(publication))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--mode",choices=("validate","analyze"),required=True);a=p.parse_args();a.config=a.config.absolute();c=load_contract(a.config)
    if a.mode=="analyze":analyze(c,a.config)
    else:print(json.dumps({"status":"valid_prospective_aggregate","missing_episode_inputs":[s for s,e in c["episodes"].items() if not Path(e["admission_path"]).is_file()],"neural_calls":0,"corpus_reads":0}))

if __name__=="__main__":main()

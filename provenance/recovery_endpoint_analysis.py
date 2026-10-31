"""Prospective paired1024 saved-evidence analysis; no model forwards or corpus I/O."""
from pathlib import Path
import argparse
import gc
import sys
import time
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.recovery_endpoints import (file_sha,hash_value,finite_json,regular,verify_refs,write_once,
    require_terminal_pair,expected_update,row_identity,validate_observation,paired_summary,validate_completed_state)


def load_contract(path):
    c=finite_json(path)
    fields={"schema_version","study","root","out","freeze","run_config","source_refs","analysis_refs","expected_specs","plan","initialization","tuning_identities","fit_documents","analysis","caps"}
    if set(c)!=fields or type(c["schema_version"])!=int or c["schema_version"]!=1:raise ValueError("unsupported endpoint contract schema")
    if (c["study"]!="recovery1024-endpoint-seed17-v1" or c["analysis"]!={"seed":17,"draws":10000,"primary":"full_student_minus_local_sum_at1024","diagnostic_resolution_targets":[.002,.005,.01],"component_count":128,"capability_tolerance":None}
            or c["caps"]!={"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2}):
        raise ValueError("prospective analysis changed")
    verify_refs(c["source_refs"]+c["analysis_refs"]+[c[k] for k in ("freeze","run_config","plan","initialization")])
    root=Path(c["root"]);out=Path(c["out"])
    if root.resolve()!=root or out.resolve()!=out or not out.is_relative_to(ROOT) or out.is_relative_to(root):raise ValueError("analysis output isolation differs")
    from poc.recovery_study import load_study_config
    cfg=load_study_config(c["run_config"]["path"])
    freeze=finite_json(c["freeze"]["path"])
    if freeze["run_configuration"]!=c["run_config"] or freeze["study"]!=cfg["study"]:raise ValueError("run freeze differs")
    if (cfg["updates"]!=1024 or cfg["seeds"]!=[17] or cfg["pair"]!=[0,15] or cfg["rank"]!=8 or cfg["loss_scopes"]!=["local_sum","full_student"]
            or cfg["owner_scope"]!="affected_group" or cfg["alpha"]!=0 or cfg["raw_targets_per_update"]!=8192
            or cfg["observations"]!=[0,1,4,16,32,64,128,256,512,1024]):raise ValueError("prospective paired experiment differs")
    if set(c["expected_specs"])!={"local_sum","full_student"}:raise ValueError("both frozen RunSpecs required")
    for scope,raw in c["expected_specs"].items():
        from poc.checkpoint import _spec
        spec=_spec(raw)
        if (spec.method_variant!=cfg["study"]+":"+scope or spec.seeds!={"fit":17} or spec.resolved_config["recovery_objective"]["loss_scope"]!=scope
                or spec.input_manifests["profile"]!=freeze["profile"]["sha256"]):raise ValueError("frozen arm identity differs")
    if len(c["tuning_identities"])!=128 or len(c["fit_documents"])!=512:raise ValueError("whole component population required")
    return c,cfg


def terminal(c,cfg):
    from provenance.sharing_frontier_study import check_stage_binding,require_stage_success
    root=Path(c["root"]);prov=root/"provenance/recovery-horizon-v1";done=require_terminal_pair(prov)
    stage=done["stages"]["run"];require_stage_success(stage)
    freeze=finite_json(c["freeze"]["path"])
    caps={"wall_seconds":sum(b["wall_seconds"] for b in cfg["resource_budget"]["per_arm"].values())+1800,"memory_bytes":12*1024**3,"disk_bytes":24*1024**3}
    required=[{"path":str(root/p),"sha256":h} for p,h in freeze["source_manifest"]["files"].items()]+[c["freeze"],c["run_config"]]
    command=[sys.executable,"provenance/recovery_diagnosis.py","--config",c["run_config"]["path"],"--mode","run"]
    check_stage_binding(stage,command=command,root=root,output=root/cfg["out"],caps=caps,required_refs=required,
        execution_env={"PYTHONPATH":str(root),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":"/private/tmp"})
    p=finite_json(stage["protocol_path"]);verify_refs(p["frozen_files"])
    attempt=finite_json(stage["attempt_path"])
    for actual,cap in (("wall_seconds","wall_seconds"),("sampled_process_tree_rss_bytes","memory_bytes"),("sampled_scratch_peak_bytes","disk_bytes")):
        if type(attempt.get(actual)) not in (int,float) or not math.isfinite(attempt[actual]) or not 0<=attempt[actual]<=caps[cap]:raise ValueError("terminal supervisor resource bound exceeded")
    expected={scope:str(root/cfg["out"]/("runs/"+scope+"-seed17/result.json")) for scope in cfg["loss_scopes"]}
    if any(done["results"][s]["path"]!=p for s,p in expected.items()):raise ValueError("terminal result path differs")
    campaign=finite_json(root/cfg["out"]/"result.json")
    if set(campaign["arms"])!={scope+"-seed17" for scope in cfg["loss_scopes"]} or any(v["status"]!="complete" for v in campaign["arms"].values()):raise ValueError("campaign incomplete")
    return done,stage


def ledger(path,spec,*,require_success=True):
    from poc.records import read_rows,run_id
    from poc.budget import BudgetTracker
    from poc.contracts import CostRecord
    import re
    p=Path(path);base=p.parent
    if p.exists():regular(p)
    elif require_success or not p.is_absolute() or p.resolve()!=p:raise ValueError("completed cost file required")
    if base.name!=run_id(spec) or finite_json(base/"spec.json")!=spec.to_dict():raise ValueError("ledger RunSpec differs")
    total=CostRecord();attempts={};rows=0
    for folder in sorted((base/"attempts").iterdir()):
        aid=folder.name
        if not re.fullmatch("[0-9a-f]{32}",aid) or not folder.is_dir() or folder.is_symlink():raise ValueError("invalid attempt directory")
        ap=folder/"run.json";a=finite_json(ap)
        if a["run_id"]!=run_id(spec) or a["attempt_id"]!=aid or a["spec"]!=spec.to_dict() or a["status"] not in ("exhausted","failed"):
            raise ValueError("nonterminal/mismatched attempt")
        attempts[aid]={"record":a,"path":str(ap),"sha256":file_sha(ap),"cost_rows":0}
    # Stream large common ledgers; enumerate attempts separately so failures
    # before the first charge remain visible even without a costs.jsonl file.
    if p.exists():
        with p.open("rb") as f:
            for line in f:
                row=read_rows(__import__("io").BytesIO(line),first_event_id=rows+1)[0];rows+=1
                if row.get("kind")!="cost" or row.get("attempt_id") not in attempts:raise ValueError("unknown cost event/attempt")
                attempts[row["attempt_id"]]["cost_rows"]+=1
                total=BudgetTracker.combine(total,CostRecord(**row["cost"]))
    if not attempts or require_success and (rows==0 or not any(a["record"]["status"]=="exhausted" for a in attempts.values())):
        raise ValueError("completed nonempty ledger required")
    for a in attempts.values():
        a["lost_work"]=("unavailable; zero instrumented rows do not prove zero executed work" if a["cost_rows"]==0 else
                        "instrumented work retained; any unrecorded interrupted work remains unavailable")
    violations=[]
    for key,value in spec.budgets.to_dict().items():
        actual=getattr(total,key)
        if actual is None or actual>value:violations.append(key)
    if require_success and violations:raise ValueError("instrumented budget exceeded: "+str(violations))
    return {"path":str(p),"sha256":file_sha(p) if p.exists() else None,"cost_file_available":p.exists(),
            "rows":rows,"totals":total.to_dict(),"attempts":attempts,"budget_violations":violations}


def require_analysis_guard(c,contract_path,*,context=None):
    """Admit only the direct child of this exact frozen finite supervisor."""
    import os
    if context is not None:
        from poc.recovery_replication_analysis import RecoveryEndpointContext
        if not isinstance(context,RecoveryEndpointContext):raise ValueError("typed version2 endpoint context required")
        if context.command[3]!=str(Path(contract_path).absolute()):raise ValueError("context command binds another contract")
    stage=(ROOT/"provenance/recovery-endpoint-v1/final-analysis-stage-v1" if context is None else Path(context.stage))
    pp=stage/"protocol.json";sp=stage/"stage-start.json"
    if not pp.is_file() or not sp.is_file():raise ValueError("analysis requires its verified finite supervisor")
    protocol=finite_json(pp);start=finite_json(sp)
    command=([sys.executable,str(ROOT/"provenance/recovery_endpoint_analysis.py"),"--contract",str(Path(contract_path).absolute()),"--mode","analyze"]
             if context is None else list(context.command))
    if (start.get("pid")!=os.getppid() or start.get("protocol_sha256")!=file_sha(pp)
            or protocol.get("command")!=command or protocol.get("root")!=str(ROOT) or protocol.get("output")!=c["out"]
            or protocol.get("provenance")!=str(stage) or protocol.get("caps")!=c["caps"]
            or protocol.get("supervisor_sha256")!="a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba"):
        raise ValueError("analysis is not under its exact source-bound resource supervisor")
    verify_refs(protocol["frozen_files"])
    bound={r["path"]:r["sha256"] for r in protocol["frozen_files"]}
    required=c["source_refs"]+c["analysis_refs"]+[{"path":str(Path(contract_path).absolute()),"sha256":file_sha(contract_path)}]
    if any(bound.get(r["path"])!=r["sha256"] for r in required):raise ValueError("analysis supervisor omitted frozen source/contract")


def analyze(c,cfg,contract_path,*,context=None):
    import torch
    from safetensors.torch import load_file
    from poc.checkpoint import _spec,CheckpointRef,load_checkpoint,resolve_checkpoint,checkpoint_decoder
    from poc.records import run_id
    from poc.train import _saved_digest
    from poc.recovery_objectives import RecoveryObjectiveSpec
    if context is None:
        actual_seed=17
        done,stage=terminal(c,cfg)
    else:
        from poc.recovery_replication_analysis import RecoveryEndpointContext,admit_replication_terminal
        if not isinstance(context,RecoveryEndpointContext):raise ValueError("typed version2 endpoint context required")
        actual_seed=context.seed;done,stage=admit_replication_terminal(c,cfg,context)
    # No output or checkpoint read before successful terminal pair and guard.
    require_analysis_guard(c,contract_path,context=context)
    if context is not None:
        from poc.recovery_replication_analysis import require_context_study_inputs
        require_context_study_inputs(c,cfg,context)
    out=Path(c["out"])
    if out.exists():raise FileExistsError("analysis output already exists; no overwrite")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();write_once(out/"intent.json",{"contract_sha256":file_sha(contract_path),"status":"started","scope":"CPU saved-evidence only; no neural forward/corpus/protected reads"})
    try:
        root=Path(c["root"]);runs=root/cfg["out"]/"runs";plan=finite_json(c["plan"]["path"]);initial=finite_json(c["initialization"]["path"])
        if (plan["updates"]!=1024 or plan["raw_target_presentations"]!=8388608 or plan["unique_target_positions"]!=261632
                or plan["inventory_unique_target_positions"]!=261632 or plan["unique_components"]!=512):raise ValueError("actual raw plan accounting differs")
        checks={};curves={};endpoints={};sources=finite_json(runs/"executed-source.json")
        if sources!=finite_json(c["freeze"]["path"])["source_manifest"]:raise ValueError("executed source differs from freeze")
        teacher_digests=set();anchor_owners=None
        for scope in cfg["loss_scopes"]:
            arm=runs/(scope+f"-seed{actual_seed}");raw=finite_json(arm/"spec.json")
            if raw!=c["expected_specs"][scope]:raise ValueError("RunSpec differs from prospective exact arm")
            spec=_spec(raw)
            if file_sha(arm/"spec.json")!=run_id(spec) or file_sha(runs/"executed-source.json")!=spec.input_manifests["executed_source"]:
                raise ValueError("noncanonical RunSpec/source identity")
            final=finite_json(arm/"result.json");ref=CheckpointRef(**final["checkpoint"])
            if Path(ref.directory).parent!=arm/"resume" or resolve_checkpoint(arm/"resume")!=ref:raise ValueError("final checkpoint differs from latest own safe state")
            state=load_checkpoint(ref,spec);anchor=load_checkpoint(resolve_checkpoint(arm/"resume/anchor"),spec)
            obj=RecoveryObjectiveSpec(**spec.resolved_config["recovery_objective"])
            check=validate_completed_state(state,anchor,plan,spec.resolved_config["training"],obj)
            if (len(check["owner_names"])!=15 or check["frozen_owners"]!=434 or state.partition!=tuple(tuple(g) for g in ([([0,15] if i==0 else [i]) for i in range(30) if i!=15]))
                    or state.model_schema["rank"]!=8 or anchor.candidate_digest!=initial["initialized_digest"] or anchor.candidate_digest!=spec.input_manifests["initialized_state"]):
                raise ValueError("sentinel representation/initialization differs")
            owner_values={n:file_sha_bytes(v) for n,v in anchor.model.items()}
            if anchor_owners is None:anchor_owners=owner_values
            if anchor_owners!=owner_values:raise ValueError("paired original initialized weights differ")
            teacher_digests.add(state.teacher_digest)
            if (final["step"]!=1024 or final["raw_target_cursor"]!=8388608 or final["final_digest"]!=state.candidate_digest or final["events"]!=state.events):
                raise ValueError("saved result/checkpoint histories differ")
            decoder=checkpoint_decoder(anchor);values=dict(anchor.model);curves[scope]={};window_ids={}
            expected_names=set(check["owner_names"])
            for step in cfg["observations"]:
                op=arm/"observations"/f"{step:04d}.json";obs=finite_json(op)
                want_digest=anchor.candidate_digest if step==0 else state.visited[step-1]
                expected=expected_update(plan["order"],c["fit_documents"],max(0,step-1))
                curves[scope][str(step)]=validate_observation(obs,step=step,plan_sha256=plan["plan_sha256"],study_sha256=spec.resolved_config["study_contract_sha256"],
                    tuning=c["tuning_identities"],optimizer=expected,candidate_digest=want_digest)
                for row in obs["populations"]["optimizer_windows"]:
                    ident=(row["component_id"],row["window_sha256"])
                    if row["document_id"] in window_ids and window_ids[row["document_id"]]!=ident:raise ValueError("optimizer context hash changed across masks")
                    window_ids[row["document_id"]]=ident
                delta=arm/"deltas"/f"{step:04d}";meta=finite_json(delta/"manifest.json")
                if (set(meta)!={"schema_version","anchor_digest","candidate_digest","owners","weights_sha256","run_spec_sha256","study_contract_sha256"}
                        or meta["schema_version"]!=1 or meta["anchor_digest"]!=anchor.candidate_digest or meta["candidate_digest"]!=want_digest
                        or meta["run_spec_sha256"]!=run_id(spec) or meta["study_contract_sha256"]!=spec.resolved_config["study_contract_sha256"]
                        or meta["owners"]!=check["owner_names"] or file_sha(regular(delta/"owners.safetensors"))!=meta["weights_sha256"]):raise ValueError("observation delta binding differs")
                weights=load_file(str(delta/"owners.safetensors"),device="cpu")
                if set(weights)!=expected_names:raise ValueError("delta owner set differs")
                for name,v in weights.items():
                    if v.shape!=anchor.model[name].shape or v.dtype!=anchor.model[name].dtype or not torch.isfinite(v).all().item():raise ValueError("delta owner schema/nonfinite")
                    if step==0 and not torch.equal(v,anchor.model[name]) or step==1024 and not torch.equal(v,state.model[name]):raise ValueError("delta/checkpoint weights differ")
                values.update(weights)
                if _saved_digest(decoder,values,device_override=state.model_schema["device"])!=want_digest:raise ValueError("delta executable candidate differs")
                curves[scope][str(step)]["observation_sha256"]=file_sha(op)
                curves[scope][str(step)]["candidate_digest"]=want_digest
                if step==1024:endpoints[scope]=obs["populations"]["tuning"]
            cost=ledger(runs/"ledgers"/run_id(spec)/"costs.jsonl",spec);totals=cost["totals"]
            expected_scored=8388608*obj.scored_multiplier+len(cfg["observations"])*(65408+3*8192)
            if totals["updates"]!=1024 or totals["raw_corpus_target_presentations"]!=8388608 or totals["valid_target_presentations"]!=expected_scored:
                raise ValueError("terminal raw/scored/diagnostic ledger differs from exact finite plan")
            for key in ("updates","raw_corpus_target_presentations","valid_target_presentations","teacher_forward_calls","student_forward_calls","student_backward_calls","suffix_backward_calls"):
                if final["cost"][key]!=totals[key]:raise ValueError("result/terminal work counts differ")
            check.update(run_id=run_id(spec),ledger=cost,checkpoint=final["checkpoint"],result_sha256=file_sha(arm/"result.json"),optimizer_context_identities=window_ids)
            checks[scope]=check
            del state,anchor,decoder,values,weights;gc.collect()
        if len(teacher_digests)!=1 or checks["local_sum"]["owner_names"]!=checks["full_student"]["owner_names"]:raise ValueError("paired teacher/owner identity differs")
        for step in cfg["observations"]:
            a=curves["local_sum"][str(step)];b=curves["full_student"][str(step)]
            if step==0 and (abs(a["tuning"]["kl"]-b["tuning"]["kl"])>2e-5 or a["candidate_digest"]!=b["candidate_digest"]):raise ValueError("paired initial state/measurement differs")
        for doc in checks["local_sum"]["optimizer_context_identities"]:
            if checks["local_sum"]["optimizer_context_identities"][doc]!=checks["full_student"]["optimizer_context_identities"].get(doc):raise ValueError("paired optimizer contexts differ")
        cost_context=[]
        for spec_path in sorted((root/cfg["out"]).glob("**/ledgers/*/spec.json")):
            path=spec_path.parent/"costs.jsonl"
            if any(path==Path(v["ledger"]["path"]) for v in checks.values()):continue
            other_spec=_spec(finite_json(spec_path))
            cost_context.append(ledger(path,other_spec,require_success=False))
        freeze=finite_json(c["freeze"]["path"])
        result=paired_summary(endpoints["local_sum"],endpoints["full_student"])
        from poc.recovery_replication_analysis import pair_result_context
        result,freeze=pair_result_context(result,freeze,context)
        result.update(schema_version=result.get("schema_version",1),status="accepted_exploratory_endpoint",contract_sha256=file_sha(contract_path),
            checkpoints=checks,curves=curves,training_guard=stage,analysis_seconds=time.monotonic()-started,
            preparation_context={"profile_and_setup_ledgers":cost_context,"qualification":freeze["qualification"],"accepted_profile_guard":freeze["accepted_profile_guard"],"profile":freeze["profile"],"predecessor_controller_failure":freeze["predecessor_queue_failure"],"shared_original_initialization":cfg["initialization_checkpoint"],"allocation":"Each new-study ledger once; arm ledgers separate. Supervisor views overlap instrumentation and are never added. Original acquisition/population/256-study STEP0 preparation remains a linked shared ancestor, not recharged or amortized. No total-preparation efficiency claim."},
            scope="fixed final completed1024 schedules, same original weights/owners/raw exposure; text uncertainty conditional on one fitted pair/seed; no capability/confirmation/compression/counterexample claim",
            window_proof_limit="metadata/order/masks and within/across-arm window hashes checked; raw tokens intentionally unopened, their content bound by qualified original source and manifest/plan identities")
        verify_refs(c["source_refs"]+c["analysis_refs"])
        if context is not None:require_context_study_inputs(c,cfg,context)
        write_once(out/"result.json",result)
        contrast=result["full_minus_local"]
        text=f"Paired seed{actual_seed} recovery completed1,024updates per objective on the same261,632distinct fit positions (8,388,608raw presentations each).\n\n"
        text+="| Objective | Final KL | Final NLL |\n|---|---:|---:|\n"
        for scope,a in result["arms"].items():text+=f"| {scope} | {a['kl']:.10f} | {a['nll']:.10f} |\n"
        text+=f"\nFull minus local final KL: {contrast['kl_difference']:.10f}nats/target; paired-component95%interval {contrast['component_percentile95']}.\n\n"
        text+="All observations, exact optimizer-window masks, candidate deltas, safe checkpoints, frozen owners, optimizer/RNG/schedule states, terminal costs and source identities were admitted without neural replay. Full natural metrics and local teacher-suffix discrepancies remain separate. The curves are fixed observations, with no best-checkpoint selection.\n\n"
        text+="One pair/seed and already exposed development text. The interval measures component uncertainty conditional on these states; other seeds and independent populations remain required.0.002/0.005/0.01nats are diagnostic resolution targets, not acceptable loss margins. This result cannot certify capability preservation, counterexample value or a storage advantage. CPU analysis overhead and prior acquisition/profile/failure/preparation costs remain separate; no fixed-time or total-preparation superiority is claimed.\n"
        (out/"REPORT.md").write_text(text)
        write_once(out/"terminal.json",{"status":"complete","result_sha256":file_sha(out/"result.json"),"report_sha256":file_sha(out/"REPORT.md"),"contract_sha256":file_sha(contract_path)})
        return result
    except BaseException as exc:
        try:write_once(out/"failed.json",{"status":"failed","error":repr(exc),"elapsed_seconds":time.monotonic()-started,"no_automatic_retry":True})
        except BaseException as publication_error:exc.add_note("Failure receipt publication also failed: "+repr(publication_error))
        raise


def file_sha_bytes(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True);p.add_argument("--mode",choices=("validate","analyze"),required=True);a=p.parse_args()
    a.contract=a.contract.absolute()
    c,cfg=load_contract(a.contract)
    if a.mode=="analyze":analyze(c,cfg,a.contract)
    else:print(__import__("json").dumps({"status":"valid_frozen_analysis","contract_sha256":file_sha(a.contract),"neural_calls":0,"corpus_reads":0}))

if __name__=="__main__":main()

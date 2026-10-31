"""Owned evidence mutations; no pretrained/corpus/backend execution."""
from copy import deepcopy
import hashlib
import math
import numpy as np
import pytest

from poc.recovery_endpoints import metric, expected_update, validate_observation, paired_summary, validate_events, finite_json, verify_refs

H=lambda s:hashlib.sha256(str(s).encode()).hexdigest()
def item(i,n=511,kl=.1):
 return {"document_id":H("doc"+str(i)),"component_id":H("component"+str(i)),"window_sha256":H("window"+str(i)),"target_positions":list(range(n)),"full":{"status":"ok","reason":None,"valid_targets":n,"kl":kl*n,"nll":3*n,"disagreements":3},"local_by_depth":[]}
def signature(row):return {k:deepcopy(row[k]) for k in ("document_id","component_id","window_sha256","target_positions")}
def observation():
 tune=[item(i) for i in range(128)]
 fit=[item(200+i,n) for i,n in enumerate((511,511,510))]
 for row in fit:row["local_by_depth"]=[deepcopy(row["full"]),deepcopy(row["full"])]
 return {"step":4,"observation_policy":"optimizer_plus_tuning_v2","plan_sha256":H("plan"),"candidate_digest":H("state"),"study_contract_sha256":H("contract"),"populations":{"tuning":tune,"optimizer_windows":fit}}
def validate(obs):
 good=observation()
 return validate_observation(obs,step=4,plan_sha256=H("plan"),study_sha256=H("contract"),tuning=[signature(r) for r in good["populations"]["tuning"]],optimizer=[{k:deepcopy(r[k]) for k in ("document_id","component_id","target_positions")} for r in good["populations"]["optimizer_windows"]],candidate_digest=H("state"))
def test_observation_ok_and_raw_counts():
 v=validate(observation());assert v["tuning"]["valid_targets"]==65408 and v["optimizer_windows"]["valid_targets"]==1532
 assert v["optimizer_local_sum_per_raw_target"]==pytest.approx(.2)
@pytest.mark.parametrize("mutation",["step","plan","contract","candidate","policy","population","duplicate","missing","doc","window","mask","optimizer_order","local_missing","tuning_local","nonfinite","floor","target_count","status","unknown_metric"])
def test_evidence_changes_are_rejected(mutation):
 o=observation();r=o["populations"]["tuning"][0]
 if mutation=="step":o["step"]=3
 elif mutation=="plan":o["plan_sha256"]=H("wrong")
 elif mutation=="contract":o["study_contract_sha256"]=H("wrong")
 elif mutation=="candidate":o["candidate_digest"]=H("wrong")
 elif mutation=="policy":o["observation_policy"]="legacy"
 elif mutation=="population":o["populations"]["fit_inventory"]=[]
 elif mutation=="duplicate":o["populations"]["tuning"][1]=deepcopy(r)
 elif mutation=="missing":o["populations"]["tuning"].pop()
 elif mutation=="doc":r["document_id"]=H("wrong")
 elif mutation=="window":r["window_sha256"]=H("wrong")
 elif mutation=="mask":r["target_positions"]=list(range(1,512))
 elif mutation=="optimizer_order":o["populations"]["optimizer_windows"].reverse()
 elif mutation=="local_missing":o["populations"]["optimizer_windows"][0]["local_by_depth"].pop()
 elif mutation=="tuning_local":r["local_by_depth"]=[r["full"]]
 elif mutation=="nonfinite":r["full"]["kl"]=float("nan")
 elif mutation=="floor":r["full"]["kl"]=-.1
 elif mutation=="target_count":r["full"]["valid_targets"]=True
 elif mutation=="status":r["full"]["status"]="failed"
 elif mutation=="unknown_metric":r["full"]["clipped_kl"]=0
 with pytest.raises(ValueError):validate(o)
def test_tiny_negative_is_retained():
 m=deepcopy(item(0)["full"]);m["kl"]=-1e-7
 assert metric(m,511)["kl"]==-1e-7
@pytest.mark.parametrize("field,value",[("nll",-1),("nll",float("inf")),("disagreements",512),("disagreements",True),("reason","ignored")])
def test_bad_metric(field,value):
 m=item(0)["full"];m[field]=value
 with pytest.raises(ValueError):metric(m,511)
def test_plan_cuts_mask_and_retains_context_identity():
 docs=[{"document_id":H(i),"component_id":H(100+i)} for i in range(3)]
 order=[2,0,1,2]
 rows=expected_update(order,docs,1,raw_targets=700)
 assert [r["document_id"] for r in rows]==[H(0),H(1)]
 assert rows[0]["target_positions"]==list(range(189,511))
 assert rows[1]["target_positions"]==list(range(378))
 assert sum(len(r["target_positions"]) for r in rows)==700
 with pytest.raises(ValueError):expected_update(order,docs,4,raw_targets=700)
def test_paired_component_draws_keep_candidates_together():
 a=[item(i,kl=.1+i/10000) for i in range(128)];b=deepcopy(a)
 for i,r in enumerate(b):r["full"]["kl"]+=(i%3-1)*.001*511
 s=paired_summary(a,b,draws=1000)
 ordered=sorted(zip(a,b),key=lambda xy:xy[0]["component_id"])
 d=np.array([(y["full"]["kl"]-x["full"]["kl"])/511 for x,y in ordered])
 draws=np.random.default_rng(17).integers(0,128,(1000,128));values=d[draws].mean(axis=1)
 assert np.allclose(s["full_minus_local"]["component_percentile95"],np.quantile(values,[.025,.975]),rtol=0,atol=1e-18)
 assert np.allclose(s["full_minus_local"]["contrast_draws"],values,rtol=0,atol=1e-18)
 assert s["full_minus_local"]["kl_difference"]==pytest.approx(d.mean(),abs=1e-16)
 assert s["capability_tolerance"] is None and s["compression_claim"] is False
 assert sorted(s["diagnostic_resolution_targets"])==[.002,.005,.01]
def test_pair_requires_identical_contexts_and_masks():
 a=[item(i) for i in range(128)];b=deepcopy(a);b[0]["window_sha256"]=H("other")
 with pytest.raises(ValueError):paired_summary(a,b)
def test_pair_is_order_invariant():
 a=[item(i) for i in range(128)];b=deepcopy(a);b.reverse()
 assert paired_summary(a,b,draws=10)["full_minus_local"]["kl_difference"]==0

def events(n=4):
 return [{"kind":"update","step":i,"valid_targets":8192,"raw_target_presentations":8192,"scored_target_presentations":16384,"raw_plan_sha256":H("plan"),"source_counts":{"natural":8192},"gradient_norm_pre_clip":.5,"parameter_update_norm":.1,"clip_applied":False,"learning_rates":{"bank":1e-4,"residual":3e-4},"weighted_objective":.1,"kl_sum":819.2,"nll_sum":1000.,"source_kl_sums":{"natural":819.2},"epoch":0} for i in range(1,n+1)]
def test_exact_event_units():
 e=events();v=validate_events(e,updates=4,scope="local_sum",plan_sha256=H("plan"))
 assert v["training_raw_targets"]==32768 and v["training_scored_targets"]==65536
@pytest.mark.parametrize("change",["early","wrong_step","raw","scored","plan","negative_norm","nonfinite_update","clip","source","loss"])
def test_event_mutations(change):
 e=events();r=e[-1]
 if change=="early":e.pop()
 elif change=="wrong_step":r["step"]=3
 elif change=="raw":r["raw_target_presentations"]=16384
 elif change=="scored":r["scored_target_presentations"]=8192
 elif change=="plan":r["raw_plan_sha256"]=H("wrong")
 elif change=="negative_norm":r["gradient_norm_pre_clip"]=-.1
 elif change=="nonfinite_update":r["parameter_update_norm"]=float("inf")
 elif change=="clip":r["clip_applied"]=True
 elif change=="source":r["source_counts"]={"donor":8192}
 elif change=="loss":r["weighted_objective"]=-1
 with pytest.raises(ValueError):validate_events(e,updates=4,scope="local_sum",plan_sha256=H("plan"))
def test_strict_json_ref_hash_and_no_alias(tmp_path):
 p=tmp_path/"a.json";p.write_text('{"x":1}')
 ref={"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
 verify_refs([ref]);p.write_text('{"x":2}')
 with pytest.raises(ValueError):verify_refs([ref])
 p.write_text('{"x":1,"x":2}')
 with pytest.raises(ValueError):finite_json(p)
 p.write_text('{"x":NaN}')
 with pytest.raises(ValueError):finite_json(p)
 p.write_text('{"x":1}');alias=tmp_path/"b.json";alias.symlink_to(p)
 with pytest.raises(ValueError):finite_json(alias)


@pytest.fixture(scope="module")
def completed_owned():
 from test_recovery_study import setup,bound_config
 from test_training import budget,MemoryRecorder
 from poc.train import TrainingSources,configure_trainable_owners,_optimizer,_snapshot,_schedule_state,fit_model
 from types import SimpleNamespace
 teacher,student,plan,obj=setup("full_student",updates=1)
 cfg=bound_config(obj,1);owners=configure_trainable_owners(student,cfg,objective_spec=obj);rec=MemoryRecorder()
 anchor=_snapshot(student,_optimizer(owners,cfg),_schedule_state(0,1),cfg,TrainingSources(raw_plan=plan),0,0,0,[],SimpleNamespace(recorder=rec,total=rec.totals),teacher.manifest().checkpoint_digest,(),objective_spec=obj)
 result=fit_model(student,teacher,TrainingSources(raw_plan=plan),cfg,budget(),rec,objective_spec=obj)
 return result.state,anchor,dict(plan.metadata),cfg,obj

def test_completed_checkpoint_reconstructed_without_teacher(completed_owned):
 from poc.recovery_endpoints import validate_completed_state
 state,anchor,plan,cfg,obj=completed_owned
 result=validate_completed_state(state,anchor,plan,cfg.identity(),obj)
 assert result["step"]==1 and result["training_raw_targets"]==8192
 assert result["cpu_decoder_reconstructed_without_teacher"] is True

@pytest.mark.parametrize("change",["step","cursor","probe_cursor","source","order","history","scheduler","config","owner","frozen","moment","moment_negative","moment_step","rng","safe","ledger"])
def test_completed_checkpoint_mutations_rejected(completed_owned,change):
 import torch
 from poc.recovery_endpoints import validate_completed_state
 original_state,original_anchor,plan,cfg,obj=completed_owned
 state,anchor=deepcopy((original_state,original_anchor),{id(original_state.spec):original_state.spec})
 if change=="step":state.step=0
 elif change=="cursor":state.target_cursor-=1
 elif change=="probe_cursor":state.probe_cursor=1
 elif change=="source":state.source_hash=H("wrong")
 elif change=="order":state.order=("legacy",state.source_hash)
 elif change=="history":state.visited=()
 elif change=="scheduler":state.scheduler["planned_updates"]=2
 elif change=="config":state.config_hash=H("wrong")
 elif change=="owner":state.model_schema["owners"][0]["trainable"]=not state.model_schema["owners"][0]["trainable"]
 elif change=="frozen":
  name=next(r["name"] for r in state.model_schema["owners"] if not r["trainable"]);state.model[name]=state.model[name]+1
 elif change=="moment":next(iter(state.optimizer["state"].values()))["exp_avg"][0,0]=float("nan")
 elif change=="moment_negative":next(iter(state.optimizer["state"].values()))["exp_avg_sq"][0,0]=-1
 elif change=="moment_step":next(iter(state.optimizer["state"].values()))["step"].fill_(0)
 elif change=="rng":state.rng["torch"]=torch.zeros(1)
 elif change=="safe":state.safe_boundary=False
 elif change=="ledger":state.budget_totals["raw_corpus_target_presentations"]=0
 with pytest.raises((ValueError,RuntimeError)):validate_completed_state(state,anchor,plan,cfg.identity(),obj)



def test_terminal_gate_refuses_absent_or_failed_pair_before_output(tmp_path):
 from poc.recovery_endpoints import require_terminal_pair,write_once
 root=tmp_path/"source";root.mkdir();output=tmp_path/"analysis"
 with pytest.raises(ValueError,match="terminal"):require_terminal_pair(root)
 assert not output.exists()
 (root/"queue-failed-v2.json").write_text('{"status":"failed"}')
 with pytest.raises(ValueError,match="failed"):require_terminal_pair(root)
 p=tmp_path/"result.json";write_once(p,{"ok":True})
 before=p.read_bytes()
 with pytest.raises(FileExistsError):write_once(p,{"ok":False})
 assert p.read_bytes()==before


def test_weighted_population_bootstrap_uses_counts():
 a=[item(0,n=10,kl=.1),item(1,n=100,kl=.1)];b=deepcopy(a)
 b[0]["full"]["kl"]+=10*.5;b[1]["full"]["kl"]-=100*.05
 r=paired_summary(a,b,draws=100)
 assert r["full_minus_local"]["kl_difference"]==pytest.approx(0.,abs=1e-16)
 assert r["arms"]["full_student"]["macro_component_kl"]>r["arms"]["local_sum"]["macro_component_kl"]



def test_streamed_costs_match_authoritative_ledger_and_retain_failed_attempts(tmp_path):
 from dataclasses import replace
 from test_records import spec
 from test_training import budget
 from poc.contracts import CostRecord
 from poc.records import start_attempt
 from provenance.recovery_diagnosis import finish_attempt
 from poc.budget import BudgetTracker,BudgetExceeded
 from provenance.recovery_endpoint_analysis import ledger
 s=spec(budgets=budget(valid_target_presentations=1));a=start_attempt(s,tmp_path/"ledgers");tracker=BudgetTracker(s.budgets,a)
 with pytest.raises(BudgetExceeded):tracker.record(CostRecord(valid_target_presentations=2,wall_seconds=.1,memory_bytes=1,memory_unavailable_reason=None))
 finish_attempt(a,"failed",reason="owned overrun")
 result=ledger(tracker.path,s,require_success=False)
 assert result["totals"]["valid_target_presentations"]==2
 assert result["budget_violations"]==["valid_target_presentations"]
 assert list(result["attempts"].values())[0]["record"]["status"]=="failed"
 with pytest.raises(ValueError):ledger(tracker.path,s)


def test_streamed_costs_require_terminal_attempt_and_exact_spec(tmp_path):
 from test_records import spec
 from test_training import budget
 from poc.contracts import CostRecord
 from poc.records import start_attempt
 from provenance.recovery_diagnosis import finish_attempt
 from poc.budget import BudgetTracker
 from provenance.recovery_endpoint_analysis import ledger
 s=spec(budgets=budget());a=start_attempt(s,tmp_path/"ledgers");tracker=BudgetTracker(s.budgets,a)
 tracker.record(CostRecord(valid_target_presentations=2,wall_seconds=.1,memory_bytes=1,memory_unavailable_reason=None))
 with pytest.raises(ValueError):ledger(tracker.path,s)
 finish_attempt(a,"exhausted",reason="owned complete")
 r=ledger(tracker.path,s)
 assert r["totals"]["valid_target_presentations"]==2 and not r["budget_violations"]
 with tracker.path.open("ab") as f:f.write(b"{broken")
 with pytest.raises(ValueError):ledger(tracker.path,s)


def test_real_analysis_entry_fails_before_output_without_terminal_pair(tmp_path):
 from provenance.recovery_endpoint_analysis import analyze,load_contract
 out=tmp_path/"analysis";root=tmp_path/"source";root.mkdir()
 with pytest.raises(ValueError,match="terminal"):analyze({"root":str(root),"out":str(out)}, {},tmp_path/"contract.json")
 assert not out.exists()
 c=tmp_path/"contract.json";c.write_text('{"schema_version":1,"unknown":1}')
 with pytest.raises(ValueError,match="schema"):load_contract(c)


@pytest.mark.parametrize("length",[0,1,43,45])
def test_malformed_mps_rng_serialization_rejected_without_mps(completed_owned,length):
 import torch
 from poc.checkpoint import checkpoint_decoder
 from poc.train import _saved_digest
 from poc.recovery_endpoints import validate_completed_state
 a,b,plan,cfg,obj=completed_owned
 state,anchor=deepcopy((a,b),{id(a.spec):a.spec})
 for image in (state,anchor):
  decoder=checkpoint_decoder(image)
  image.model_schema["backend"]="torch_mps";image.model_schema["device"]="mps:0"
  image.candidate_digest=_saved_digest(decoder,image.model,device_override="mps:0")
  image.rng["mps"]=torch.zeros(length,dtype=torch.uint8)
 state.visited=(state.candidate_digest,)
 with pytest.raises(ValueError,match="MPS RNG"):validate_completed_state(state,anchor,plan,cfg.identity(),obj)


@pytest.mark.parametrize("change",["empty","owners","order","lr","hyperparameters","extra_state"])
def test_original_step0_optimizer_groups_are_admitted(completed_owned,change):
 from poc.recovery_endpoints import validate_completed_state
 a,b,plan,cfg,obj=completed_owned
 state,anchor=deepcopy((a,b),{id(a.spec):a.spec})
 if change=="empty":anchor.optimizer={"groups":[],"state":{}}
 elif change=="owners":anchor.optimizer["groups"][0]["owner_ids"]=[]
 elif change=="order":anchor.optimizer["groups"].reverse()
 elif change=="lr":anchor.optimizer["groups"][0]["lr"]*=2
 elif change=="hyperparameters":anchor.optimizer["groups"][0]["eps"]=1.
 elif change=="extra_state":anchor.optimizer["state"]["unknown"]={}
 with pytest.raises(ValueError,match="anchor optimizer"):validate_completed_state(state,anchor,plan,cfg.identity(),obj)


def test_zero_cost_failed_attempts_are_retained_in_terminal_ledger(tmp_path):
 from test_records import spec
 from test_training import budget
 from poc.contracts import CostRecord
 from poc.records import start_attempt
 from provenance.recovery_diagnosis import finish_attempt
 from poc.budget import BudgetTracker
 from provenance.recovery_endpoint_analysis import ledger
 s=spec(budgets=budget());root=tmp_path/"ledgers"
 first=start_attempt(s,root);finish_attempt(first,"failed",reason="owned failure before first charge")
 second=start_attempt(s,root);rec=BudgetTracker(s.budgets,second)
 rec.record(CostRecord(wall_seconds=.1,memory_bytes=1,memory_unavailable_reason=None));finish_attempt(second,"exhausted",reason="owned later success")
 result=ledger(rec.path,s)
 assert set(result["attempts"])=={first.attempt_id,second.attempt_id}
 assert result["attempts"][first.attempt_id]["cost_rows"]==0
 assert result["attempts"][first.attempt_id]["lost_work"]=="unavailable; zero instrumented rows do not prove zero executed work"
 assert result["totals"]["wall_seconds"]==.1


def test_unsupervised_analysis_is_denied_before_output(tmp_path):
 from provenance.recovery_endpoint_analysis import require_analysis_guard
 out=tmp_path/"analysis"
 with pytest.raises(ValueError,match="supervisor"):require_analysis_guard({"out":str(out),"caps":{"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2}},tmp_path/"contract.json")
 assert not out.exists()



def test_whole_failed_run_without_any_cost_file_is_visible(tmp_path):
 from test_records import spec
 from test_training import budget
 from poc.records import start_attempt
 from provenance.recovery_diagnosis import finish_attempt
 from provenance.recovery_endpoint_analysis import ledger
 s=spec(budgets=budget());attempt=start_attempt(s,tmp_path/"ledgers")
 finish_attempt(attempt,"failed",reason="owned setup failed before first charge")
 path=attempt.path.parent.parent/"costs.jsonl"
 assert not path.exists()
 r=ledger(path,s,require_success=False)
 assert r["cost_file_available"] is False and r["sha256"] is None and r["rows"]==0
 assert r["attempts"][attempt.attempt_id]["cost_rows"]==0
 assert r["totals"]["memory_bytes"] is None


@pytest.mark.parametrize("mutation",[None,"parent","caps","command","missing_ref"])
def test_exact_analysis_supervisor_contract(monkeypatch,tmp_path,mutation):
 import os,sys
 import provenance.recovery_endpoint_analysis as d
 from poc.recovery_endpoints import write_once,file_sha
 monkeypatch.setattr(d,"ROOT",tmp_path)
 path=tmp_path/"contract.json";write_once(path,{"owned":True})
 ref={"path":str(path),"sha256":file_sha(path)}
 c={"out":str(tmp_path/"out"),"caps":{"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2},"source_refs":[],"analysis_refs":[]}
 stage=tmp_path/"provenance/recovery-endpoint-v1/final-analysis-stage-v1"
 protocol={"root":str(tmp_path),"output":c["out"],"provenance":str(stage),"caps":c["caps"],"supervisor_sha256":"a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba","frozen_files":[ref],"command":[sys.executable,str(tmp_path/"provenance/recovery_endpoint_analysis.py"),"--contract",str(path),"--mode","analyze"]}
 if mutation=="caps":protocol["caps"]={**c["caps"],"wall_seconds":1801}
 elif mutation=="command":protocol["command"][-1]="validate"
 elif mutation=="missing_ref":protocol["frozen_files"]=[]
 pp=stage/"protocol.json";write_once(pp,protocol)
 write_once(stage/"stage-start.json",{"pid":os.getppid()+(1 if mutation=="parent" else 0),"protocol_sha256":file_sha(pp)})
 if mutation is None:d.require_analysis_guard(c,path)
 else:
  with pytest.raises(ValueError):d.require_analysis_guard(c,path)

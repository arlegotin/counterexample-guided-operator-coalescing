"""Owned version2 controller, pairing and uncertainty controls; no model work."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
import pytest

from poc.recovery_replication_analysis import RecoveryEndpointContext,aggregate_seeds,admit_replication_terminal

H=lambda x:hashlib.sha256(str(x).encode()).hexdigest()
def context(tmp_path,seed=23):
 return RecoveryEndpointContext(seed,str(tmp_path/"stage"),(sys.executable,str(tmp_path/"driver.py"),"--contract",str(tmp_path/"contract.json"),"--mode","analyze"))

def test_context_is_closed_immutable_and_identity_bound(tmp_path):
 c=context(tmp_path)
 assert c.schema_version==2 and c.seed==23
 assert c.digest()!=context(tmp_path,31).digest()
 with pytest.raises(FrozenInstanceError):c.seed=31
 for seed in (17,True,7):
  with pytest.raises(ValueError):context(tmp_path,seed)
 with pytest.raises(ValueError):RecoveryEndpointContext(23,"relative",c.command)
 with pytest.raises(ValueError):RecoveryEndpointContext(23,c.stage,("python","driver","--contract","x","--mode","analyze"))
 with pytest.raises(ValueError):RecoveryEndpointContext(23,c.stage,c.command,schema_version=1)


def test_wrong_context_seed_fails_before_terminal_io(tmp_path):
 c=context(tmp_path)
 with pytest.raises(ValueError,match="seed"):admit_replication_terminal({"seed":31},{"seeds":[31]},c)
 assert not (tmp_path/"stage").exists()


def episode(seed,values,ids=None):
 ids=ids or [H(i) for i in range(len(values))]
 return {"schema_version":2,"status":"accepted_exploratory_endpoint","seed":seed,"component_ids":ids,
  "full_minus_local":{"component_differences":list(values),"kl_difference":float(np.mean(values))},
  "arms":{"local_sum":{"components":len(ids),"valid_targets":511*len(ids),"kl":.1,"nll":3.},"full_student":{"components":len(ids),"valid_targets":511*len(ids),"kl":.1+float(np.mean(values)),"nll":3.+float(np.mean(values))}},
  "scientific_identity":{"pair":[0,15],"rank":8,"owner_names_sha256":H("owners"),"teacher_digest":H("teacher"),"tuning_windows_sha256":H("windows"),"raw_targets_per_arm":8388608,"steps":1024},
  "capability_tolerance":None,"compression_claim":False}

def episodes():
 return {17:episode(17,[-.012,-.008,.002,.006]),23:episode(23,[-.009,-.005,.005,.009]),31:episode(31,[-.006,-.002,.008,.012])}

def test_independent_component_seed_and_crossed_uncertainty():
 e=episodes();r=aggregate_seeds(e,draws=1000)
 ids=sorted(e[17]["component_ids"])
 x=np.array([[dict(zip(v["component_ids"],v["full_minus_local"]["component_differences"]))[i] for i in ids] for _,v in sorted(e.items())])
 # Independent arithmetic reconstructs every retained draw from its named axis.
 rng=np.random.default_rng(17);component=rng.integers(0,4,(1000,4));seed=rng.integers(0,3,(1000,3))
 component_draws=np.array([sum(x[:,c].flat)/12 for c in component])
 seed_draws=np.array([sum(x[s,:].flat)/12 for s in seed])
 crossed=np.array([sum(x[np.ix_(s,c)].flat)/12 for s,c in zip(seed,component)])
 for label,want in (("components",component_draws),("optimization_seeds",seed_draws),("crossed_sensitivity",crossed)):
  assert np.allclose(r["uncertainty"][label]["contrast_draws"],want,rtol=0,atol=1e-17)
  assert np.allclose(r["uncertainty"][label]["percentile95"],np.quantile(want,[.025,.975]),rtol=0,atol=1e-17)
 assert r["mean_full_minus_local"]==pytest.approx(x.mean())
 assert r["optimization_seed_sd"]==pytest.approx(np.std(x.mean(axis=1),ddof=1))
 assert r["capability_tolerance"] is None and r["confirmation_claim"] is False
 assert r["bootstrap"]["seed"]==17 and r["seed_ids"]==[17,23,31]


def test_zero_added_information_has_zero_all_draws():
 e={s:episode(s,[0.,0.,0.,0.]) for s in (17,23,31)}
 r=aggregate_seeds(e,draws=100)
 assert r["mean_full_minus_local"]==r["optimization_seed_sd"]==0
 assert all(set(v["contrast_draws"])=={0.} for v in r["uncertainty"].values())
 assert r["equivalence_claim"] is False


@pytest.mark.parametrize("change",["missing","extra","seed_label","components","duplicate","contexts","owners","teacher","rank","pair","horizon","raw","nonfinite","point","targets","status","compression"])
def test_bad_episode_identity_or_pairing_rejected(change):
 e=episodes();v=e[23]
 if change=="missing":del e[31]
 elif change=="extra":e[7]=episode(7,[0.]*4)
 elif change=="seed_label":v["seed"]=31
 elif change=="components":v["component_ids"][0]=H("other")
 elif change=="duplicate":v["component_ids"][0]=v["component_ids"][1]
 elif change=="contexts":v["scientific_identity"]["tuning_windows_sha256"]=H("other")
 elif change=="owners":v["scientific_identity"]["owner_names_sha256"]=H("other")
 elif change=="teacher":v["scientific_identity"]["teacher_digest"]=H("other")
 elif change=="rank":v["scientific_identity"]["rank"]=16
 elif change=="pair":v["scientific_identity"]["pair"]=[1,15]
 elif change=="horizon":v["scientific_identity"]["steps"]=256
 elif change=="raw":v["scientific_identity"]["raw_targets_per_arm"]=1024
 elif change=="nonfinite":v["full_minus_local"]["component_differences"][0]=float("nan")
 elif change=="point":v["full_minus_local"]["kl_difference"]+=.01
 elif change=="targets":v["arms"]["full_student"]["valid_targets"]-=1
 elif change=="status":v["status"]="incomplete"
 elif change=="compression":v["compression_claim"]=True
 with pytest.raises(ValueError):aggregate_seeds(e,draws=10)


def test_component_order_does_not_change_pairing_or_draws():
 e=episodes();a=aggregate_seeds(e,draws=100)
 for v in e.values():
  v["component_ids"].reverse();v["full_minus_local"]["component_differences"].reverse()
 b=aggregate_seeds(e,draws=100)
 assert a==b



def test_common_analyzer_keeps_default_none_and_denies_wrong_context_before_io(monkeypatch,tmp_path):
 import provenance.recovery_endpoint_analysis as common
 class LegacyGate(Exception):pass
 def legacy(*args):raise LegacyGate()
 monkeypatch.setattr(common,"terminal",legacy)
 with pytest.raises(LegacyGate):common.analyze({}, {},tmp_path/"c.json")
 with pytest.raises(ValueError,match="seed"):common.analyze({"seed":31},{"seeds":[31]},tmp_path/"c.json",context=context(tmp_path,23))
 with pytest.raises(ValueError):common.analyze({}, {},tmp_path/"c.json",context={"seed":23})


def test_common_guard_uses_explicit_versioned_context_only(monkeypatch,tmp_path):
 import os
 import provenance.recovery_endpoint_analysis as common
 from poc.recovery_endpoints import write_once,file_sha
 monkeypatch.setattr(common,"ROOT",tmp_path)
 c={"out":str(tmp_path/"out"),"caps":{"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2},"source_refs":[],"analysis_refs":[]}
 cp=tmp_path/"contract.json";write_once(cp,{"owned":True});ctx=context(tmp_path)
 # context() chooses this same absolute contract path.
 protocol={"root":str(tmp_path),"output":c["out"],"provenance":ctx.stage,"caps":c["caps"],"supervisor_sha256":"a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba","frozen_files":[{"path":str(cp),"sha256":file_sha(cp)}],"command":list(ctx.command)}
 pp=Path(ctx.stage)/"protocol.json";write_once(pp,protocol);write_once(Path(ctx.stage)/"stage-start.json",{"pid":os.getppid(),"protocol_sha256":file_sha(pp)})
 common.require_analysis_guard(c,cp,context=ctx)
 with pytest.raises(ValueError):common.require_analysis_guard(c,cp)


@pytest.fixture
def terminal_fixture(tmp_path):
 from poc.recovery_endpoints import write_once,file_sha
 root=tmp_path/"producer";prov=root/"provenance/recovery-replications1024-v1";prov.mkdir(parents=True)
 cfg={"seeds":[23],"loss_scopes":["local_sum","full_student"],"study":"recovery-1024-seed23-v1","out":"artifacts/recovery-1024-seed23-v1","resource_budget":{"per_arm":{"local_sum":{"wall_seconds":20},"full_student":{"wall_seconds":10}}}}
 cfgpath=root/"run.yaml";write_once(cfgpath,cfg)
 src=root/"driver.py";src.write_text("#owned metadata only\n")
 ref=lambda p:{"path":str(p),"sha256":file_sha(p)}
 freeze={"seed":23,"run_configuration":ref(cfgpath),"study":cfg["study"],"resource_budget":cfg["resource_budget"],"quality_used_in_design":False,"source_manifest":{"files":{"driver.py":file_sha(src)}}}
 fp=prov/"seed23-execution-freeze-v1.json";write_once(fp,freeze)
 c={"seed":23,"root":str(root),"run_config":ref(cfgpath),"freeze":ref(fp),"controller_source":ref(src)}
 stage_dir=prov/"seed23-paired-run-stage-v1";stage_dir.mkdir()
 caps={"wall_seconds":1830,"memory_bytes":12*1024**3,"disk_bytes":24*1024**3}
 protocol={"command":[sys.executable,"provenance/recovery_diagnosis.py","--config",str(cfgpath),"--mode","run"],"root":str(root),"output":str(root/cfg["out"]),"provenance":str(stage_dir),"caps":caps,"supervisor_sha256":"a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba","frozen_files":[ref(src),ref(fp),ref(cfgpath)],"execution_env":{"HF_HUB_OFFLINE":None,"PYTORCH_ENABLE_MPS_FALLBACK":None,"PYTHONPATH":str(root),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":"/private/tmp"}}
 pp=stage_dir/"protocol.json";write_once(pp,protocol)
 attempt={"exit_code":0,"failure":None,"protocol_sha256":file_sha(pp),"wall_seconds":1.,"sampled_process_tree_rss_bytes":1024,"sampled_scratch_peak_bytes":1024}
 ap=stage_dir/"attempt.json";write_once(ap,attempt)
 stage={"exit_code":0,"supervisor_exit_code":0,"attempt":attempt,"attempt_path":str(ap),"attempt_sha256":file_sha(ap),"protocol_path":str(pp),"protocol_sha256":file_sha(pp)}
 results={}
 for scope in cfg["loss_scopes"]:
  p=root/cfg["out"]/"runs"/(scope+"-seed23")/"result.json";write_once(p,{"owned":True});results[scope]=ref(p)
 write_once(root/cfg["out"]/"result.json",{"arms":{s+"-seed23":{"status":"complete"} for s in cfg["loss_scopes"]}})
 marker=prov/"seed23-common-driver-complete-v1.json";write_once(marker,{"stages":{"run":stage},"results":results,"freeze":ref(fp)})
 return c,cfg,context(tmp_path),marker,protocol


def test_terminal_admits_exact_owned_metadata_and_retains_earlier_success(terminal_fixture):
 from poc.recovery_endpoints import write_once
 c,cfg,ctx,marker,_=terminal_fixture
 done,stage=admit_replication_terminal(c,cfg,ctx)
 assert stage["exit_code"]==0 and set(done["results"])=={"local_sum","full_student"}
 write_once(marker.parent/"queue-failed-v1.json",{"scope":"laterseed31failure"})
 assert admit_replication_terminal(c,cfg,ctx)[1]==stage


@pytest.mark.parametrize("change",["missing","wrong_seed","wrong_freeze","missing_result","failed_stage","wrong_command","wrong_caps","missing_source","wrong_env","nonfinite_peak"])
def test_terminal_binding_mutations_rejected(terminal_fixture,change):
 import json
 from poc.recovery_endpoints import file_sha,canonical_json
 c,cfg,ctx,marker,protocol=terminal_fixture;done=json.loads(marker.read_text());stage=done["stages"]["run"]
 if change=="missing":marker.unlink()
 elif change=="wrong_seed":cfg["seeds"]=[31]
 elif change=="wrong_freeze":done["freeze"]={**c["freeze"],"sha256":H("other")}
 elif change=="missing_result":del done["results"]["full_student"]
 elif change=="failed_stage":stage["exit_code"]=1
 else:
  pp=Path(stage["protocol_path"]);ap=Path(stage["attempt_path"]);attempt=json.loads(ap.read_text())
  if change=="wrong_command":protocol["command"][-1]="profile"
  elif change=="wrong_caps":protocol["caps"]["wall_seconds"]+=1
  elif change=="missing_source":protocol["frozen_files"].pop(0)
  elif change=="wrong_env":protocol["execution_env"]["PYTHONPATH"]="wrong"
  elif change=="nonfinite_peak":attempt["sampled_process_tree_rss_bytes"]=-1
  pp.write_bytes(canonical_json(protocol));stage["protocol_sha256"]=file_sha(pp);attempt["protocol_sha256"]=file_sha(pp);ap.write_bytes(canonical_json(attempt));stage["attempt_sha256"]=file_sha(ap);stage["attempt"]=attempt
 if change!="missing":marker.write_bytes(canonical_json(done))
 with pytest.raises(ValueError):admit_replication_terminal(c,cfg,ctx)



def test_normalizer_uses_actual_seed_specs_and_common_windows():
 from poc.recovery_replication_analysis import normalize_episode
 from poc.recovery_endpoints import hash_value
 e=episode(23,[-.01,.01]);e["checkpoints"]={s:{"step":1024,"training_raw_targets":8388608,"owner_names":["a","b"],"teacher_digest":H("teacher")} for s in ("local_sum","full_student")}
 identities=[{"component_id":i,"document_id":H("doc"+i),"window_sha256":H("window"+i),"target_positions":list(range(511))} for i in e["component_ids"]]
 c={"seed":23,"tuning_identities":identities,"expected_specs":{s:{"seeds":{"fit":23},"input_manifests":{"owner_names":hash_value({"value":["a","b"]})},"resolved_config":{"training":{"max_updates":1024,"alpha":0,"target_count":8192,"affected_depths":[0,15]},"recovery_objective":{"loss_scope":s,"owner_scope":"affected_group"}}} for s in ("local_sum","full_student")}}
 n=normalize_episode(23,e,c)
 assert n["seed"]==23 and n["scientific_identity"]["tuning_windows_sha256"]==hash_value({"windows":sorted(identities,key=lambda r:r["component_id"])})
 bad=deepcopy(c);bad["expected_specs"]["full_student"]["seeds"]={"fit":31}
 with pytest.raises(ValueError):normalize_episode(23,e,bad)


def test_v2_driver_denies_unknown_contract_without_io(tmp_path):
 import provenance.recovery_replication_analysis as d
 p=tmp_path/"bad.json";p.write_text('{"schema_version":2,"unknown":1}')
 with pytest.raises(ValueError,match="schema"):d.load_contract(p)
 assert not (tmp_path/"out").exists()


def test_context_command_cannot_bind_another_contract(monkeypatch,tmp_path):
 import os
 import provenance.recovery_endpoint_analysis as common
 from poc.recovery_endpoints import write_once,file_sha
 monkeypatch.setattr(common,"ROOT",tmp_path)
 actual=tmp_path/"actual.json";other=tmp_path/"contract.json";write_once(actual,{"owned":"actual"});write_once(other,{"owned":"other"})
 ctx=context(tmp_path);c={"out":str(tmp_path/"out"),"caps":{"wall_seconds":1800,"memory_bytes":8*1024**3,"disk_bytes":512*1024**2},"source_refs":[],"analysis_refs":[]}
 protocol={"root":str(tmp_path),"output":c["out"],"provenance":ctx.stage,"caps":c["caps"],"supervisor_sha256":"a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba","frozen_files":[{"path":str(p),"sha256":file_sha(p)} for p in (actual,other)],"command":list(ctx.command)}
 pp=Path(ctx.stage)/"protocol.json";write_once(pp,protocol);write_once(Path(ctx.stage)/"stage-start.json",{"pid":os.getppid(),"protocol_sha256":file_sha(pp)})
 with pytest.raises(ValueError,match="contract"):common.require_analysis_guard(c,actual,context=ctx)



def test_pair_result_context_preserves_legacy_and_binds_replication_metadata(tmp_path):
 from poc.recovery_replication_analysis import pair_result_context
 base={"algorithmic_replication":"one pair/seed17; no confirmation","arms":{}}
 oldfreeze={"qualification":{"path":"old"},"accepted_profile_guard":{"path":"guard"},"predecessor_queue_failure":None}
 a,b=pair_result_context(base,oldfreeze,None)
 assert a==base and b==oldfreeze
 ctx=context(tmp_path);newfreeze={"implementation_qualification_reused":{"path":"inherited"},"accepted_profile_stage":{"path":"actual"}}
 a,b=pair_result_context(base,newfreeze,ctx)
 assert a["seed"]==23 and a["schema_version"]==2 and a["endpoint_context_sha256"]==ctx.digest()
 assert b["qualification"]==newfreeze["implementation_qualification_reused"]
 assert b["accepted_profile_guard"]==newfreeze["accepted_profile_stage"]
 assert "seed" not in base and "qualification" not in newfreeze
 with pytest.raises(ValueError):pair_result_context(base,newfreeze,[])



@pytest.fixture
def profile_fixture(tmp_path):
 import provenance.recovery_replication_analysis as d
 from poc.recovery_endpoints import write_once,file_sha,hash_value
 from poc.contracts import Budget
 root=tmp_path/"producer";root.mkdir();src=root/"driver.py";src.write_text("#owned profile metadata only\n")
 reference={"study":"owned","seeds":[23],"updates":1024,"out":"artifacts/owned","numerical_profile":{"device":"mps"},"resource_budget":None}
 config_path=root/"configs/profile.yaml";write_once(config_path,reference)
 q=root/"qualified.json";write_once(q,{"owned":"qualification"});up=root/"upstream.json";write_once(up,{"owned":"accepted"})
 impl="9743af6ba20fc12b2c51cccc4b0df07cf508462b0ff37b9765e61676bc3d6fe3";environment=H("environment")
 ctrl=root/"controls.json";write_once(ctrl,{"backend":{"passed":True,"device":"mps"},"implementation_sha256":impl,"environment_sha256":environment})
 ref=lambda p:{"path":str(p),"sha256":file_sha(p)}
 counters={k:0 for k in ("updates","valid_target_presentations","teacher_forward_calls","suffix_backward_calls","student_backward_calls","search_proposals","search_seconds")};counters["updates"]=1024;counters["valid_target_presentations"]=100
 row={"forecast_seconds_per_seed":1000.,"planned_updates":1024,"profile_updates":1,"cost":{"memory_bytes":1},"forecast_counters":counters}
 profile={"design_sha256":hash_value(d.design(reference)),"implementation_sha256":impl,"environment_sha256":environment,"controls_receipt_sha256":file_sha(ctrl),"arms":{s:deepcopy(row) for s in ("local_sum","full_student")}}
 pp=root/"artifacts/owned/profile.json";write_once(pp,profile)
 stage=root/"provenance/profile-stage";caps={"wall_seconds":3600,"memory_bytes":12*1024**3,"disk_bytes":24*1024**3}
 proto={"caps":caps,"root":str(root),"output":str(pp.parent),"provenance":str(stage),"command":[sys.executable,"provenance/recovery_diagnosis.py","--config",str(config_path),"--mode","profile"],"supervisor_sha256":"a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba","execution_env":{"HF_HUB_OFFLINE":None,"PYTORCH_ENABLE_MPS_FALLBACK":None,"PYTHONPATH":str(root),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TMPDIR":"/private/tmp"},"frozen_files":[ref(src),ref(config_path),ref(q),ref(up)]}
 proto_path=stage/"protocol.json";write_once(proto_path,proto)
 attempt={"exit_code":0,"failure":None,"protocol_sha256":file_sha(proto_path),"wall_seconds":1.,"sampled_process_tree_rss_bytes":1,"sampled_scratch_peak_bytes":1};ap=stage/"attempt.json";write_once(ap,attempt)
 receipt={"exit_code":0,"supervisor_exit_code":0,"attempt":attempt,"attempt_path":str(ap),"attempt_sha256":file_sha(ap),"protocol_path":str(proto_path),"protocol_sha256":file_sha(proto_path)};rp=root/"accepted-profile.json";write_once(rp,receipt)
 b=Budget(updates=1024,valid_target_presentations=120,teacher_forward_calls=0,suffix_backward_calls=0,student_backward_calls=0,search_proposals=0,search_seconds=0,wall_seconds=1800,disk_bytes=24*1024**3,memory_bytes=8*1024**3).to_dict()
 cfg=deepcopy(reference);cfg["resource_budget"]={"frozen_profile_sha256":file_sha(pp),"per_arm":{s:deepcopy(b) for s in ("local_sum","full_student")}}
 freeze={"resource_budget":cfg["resource_budget"],"profile":ref(pp),"accepted_profile_stage":ref(rp),"actual_controls":ref(ctrl),"profile_configuration":ref(config_path),"implementation_qualification_reused":ref(q),"upstream_independent_acceptance":ref(up),"source_manifest":{"files":{"driver.py":file_sha(src)}}}
 return cfg,freeze,reference,ref(src),root,profile,proto,receipt


def test_profile_budgets_bind_own_seed_design_and_exact_formula(profile_fixture):
 import provenance.recovery_replication_analysis as d
 cfg,freeze,reference,source,root,profile,_,_=profile_fixture
 assert d.require_profile_budgets(cfg,freeze,reference,producer_root=root,controller_source=source)==profile
 bad=deepcopy(cfg);bad["resource_budget"]["per_arm"]["local_sum"]["wall_seconds"]=1799
 with pytest.raises(ValueError):d.require_profile_budgets(bad,{**freeze,"resource_budget":bad["resource_budget"]},reference,producer_root=root,controller_source=source)
 bad=deepcopy(cfg);bad["seeds"]=[31]
 with pytest.raises(ValueError):d.require_profile_budgets(bad,freeze,reference,producer_root=root,controller_source=source)


@pytest.mark.parametrize("change",["command","caps","root","output","source","environment"])
def test_wrong_profile_invocation_is_not_a_measured_resource_gate(profile_fixture,change):
 import provenance.recovery_replication_analysis as d
 from poc.recovery_endpoints import canonical_json,file_sha
 cfg,freeze,reference,source,root,profile,proto,receipt=profile_fixture
 if change=="command":proto["command"][-1]="run"
 elif change=="caps":proto["caps"]["wall_seconds"]+=1
 elif change=="root":proto["root"]=str(root/"other")
 elif change=="output":proto["output"]=str(root/"other")
 elif change=="source":proto["frozen_files"].pop(0)
 elif change=="environment":proto["execution_env"]["PYTHONPATH"]="wrong"
 pp=Path(receipt["protocol_path"]);pp.write_bytes(canonical_json(proto));receipt["protocol_sha256"]=file_sha(pp);receipt["attempt"]["protocol_sha256"]=file_sha(pp)
 ap=Path(receipt["attempt_path"]);ap.write_bytes(canonical_json(receipt["attempt"]));receipt["attempt_sha256"]=file_sha(ap)
 rp=Path(freeze["accepted_profile_stage"]["path"]);rp.write_bytes(canonical_json(receipt));freeze["accepted_profile_stage"]["sha256"]=file_sha(rp)
 with pytest.raises(ValueError):d.require_profile_budgets(cfg,freeze,reference,producer_root=root,controller_source=source)



def test_aggregate_driver_rejects_unknown_config_and_incomplete_episodes(tmp_path):
 import provenance.recovery_seed_aggregate as d
 from poc.recovery_endpoints import write_once
 config=tmp_path/"bad.json";write_once(config,{"schema_version":2,"unknown":True})
 with pytest.raises(ValueError,match="schema"):d.load_contract(config)
 c={"episodes":{str(s):{"admission_path":str(tmp_path/f"{s}-missing.json")} for s in (17,23,31)}}
 with pytest.raises(ValueError,match="complete"):d.admit_episodes(c)
 assert not (tmp_path/"out").exists()


def test_aggregate_supervisor_is_required_before_output(tmp_path):
 import provenance.recovery_seed_aggregate as d
 c={"out":str(tmp_path/"out"),"caps":{"wall_seconds":180,"memory_bytes":2*1024**3,"disk_bytes":128*1024**2},"source_refs":[]}
 with pytest.raises(ValueError,match="supervisor"):d.require_guard(c,tmp_path/"contract.json",[])
 assert not (tmp_path/"out").exists()



def test_independent_pair_reductions_reconstruct_raw_data_without_reusing_estimator():
 from test_recovery_endpoints import item
 from poc.recovery_endpoints import paired_summary
 from poc.recovery_replication_analysis import independent_pair_reductions
 local=[item(i,kl=.1+i/10000) for i in range(128)];full=deepcopy(local)
 for i,r in enumerate(full):r["full"]["kl"]+=(i%3-1)*.001*511
 result=paired_summary(local,full)
 a={"populations":{"tuning":local}};b={"populations":{"tuning":full}}
 check=independent_pair_reductions(result,a,b)
 assert check["components"]==128 and check["valid_targets"]==65408 and check["max_draw_error"]<1e-14
 bad=deepcopy(result);bad["full_minus_local"]["contrast_draws"][0]+=.01
 with pytest.raises(ValueError):independent_pair_reductions(bad,a,b)
 bad=deepcopy(result);bad["full_minus_local"]["component_differences"][0]+=.01
 with pytest.raises(ValueError):independent_pair_reductions(bad,a,b)
 bad=deepcopy(result);bad["arms"]["full_student"]["nll"]+=.01
 with pytest.raises(ValueError):independent_pair_reductions(bad,a,b)
 bad=deepcopy(a);bad["populations"]["tuning"][0]["window_sha256"]=H("other")
 with pytest.raises(ValueError):independent_pair_reductions(result,bad,b)


def test_new_feature_cannot_analyze_before_source_qualification(tmp_path):
 from poc.recovery_replication_analysis import require_analysis_qualification
 with pytest.raises(ValueError,match="qualification"):require_analysis_qualification(tmp_path)



def test_complete_study_inventory_covers_profile_cost_changes_and_added_failed_attempts(tmp_path):
 from poc.recovery_replication_analysis import freeze_study_inputs,verify_study_inputs
 root=tmp_path/"study";run=root/"runs/local_sum/result.json";run.parent.mkdir(parents=True);run.write_text('{"step":1024}')
 profile=root/"profile-attempt-owned/ledgers/run/costs.jsonl";profile.parent.mkdir(parents=True);profile.write_text('{"wall_seconds":0.1}\n')
 reference=freeze_study_inputs(root,tmp_path/"inventory.json")
 inventory=verify_study_inputs(reference,expected_root=root)
 assert set(inventory["files"])=={"runs/local_sum/result.json","profile-attempt-owned/ledgers/run/costs.jsonl"}
 profile.write_text('{"wall_seconds":0.2}\n')
 with pytest.raises(ValueError,match="inventory"):verify_study_inputs(reference,expected_root=root)
 profile.write_text('{"wall_seconds":0.1}\n');verify_study_inputs(reference,expected_root=root)
 failure=profile.parent/"attempts/failed/run.json";failure.parent.mkdir(parents=True);failure.write_text('{"status":"failed"}')
 with pytest.raises(ValueError,match="inventory"):verify_study_inputs(reference,expected_root=root)
 failure.unlink();verify_study_inputs(reference,expected_root=root)
 run.unlink()
 with pytest.raises(ValueError,match="inventory"):verify_study_inputs(reference,expected_root=root)


def test_context_inventory_is_required_and_bound_to_actual_study_root(monkeypatch,tmp_path):
 import provenance.recovery_endpoint_analysis as common
 from poc.recovery_replication_analysis import require_context_study_inputs,freeze_study_inputs
 from poc.recovery_endpoints import write_once,file_sha
 root=tmp_path/"producer";study=root/"artifacts/owned";study.mkdir(parents=True);(study/"result.json").write_text('{"owned":true}')
 ctx=context(tmp_path);c={"root":str(root)};cfg={"out":"artifacts/owned"}
 with pytest.raises(ValueError,match="inventory"):require_context_study_inputs(c,cfg,ctx)
 inv=Path(ctx.stage).parent/"seed23-input-inventory-v1.json";reference=freeze_study_inputs(study,inv)
 protocol=Path(ctx.stage)/"protocol.json";write_once(protocol,{"frozen_files":[]})
 with pytest.raises(ValueError,match="inventory"):require_context_study_inputs(c,cfg,ctx)
 protocol.write_text(__import__('json').dumps({"frozen_files":[reference]}))
 assert require_context_study_inputs(c,cfg,ctx)==reference
 bad={"out":"artifacts/other"}
 with pytest.raises(ValueError,match="inventory"):require_context_study_inputs(c,bad,ctx)


def test_aggregate_inventory_matches_its_original_stage_and_seed(tmp_path):
 from poc.recovery_replication_analysis import freeze_study_inputs,verify_episode_input_inventory
 from poc.recovery_endpoints import write_once,file_sha
 worker=tmp_path/"worker";root=tmp_path/"producer";case=root/"artifacts/recovery-1024-seed23-v1";case.mkdir(parents=True);(case/"result.json").write_text('{"owned":true}')
 inv=worker/"provenance/recovery-replication-analysis-v2/seed23-input-inventory-v1.json";reference=freeze_study_inputs(case,inv)
 pp=worker/"stage/protocol.json";write_once(pp,{"frozen_files":[reference]})
 admission={"input_inventory":reference,"stage":{"protocol_path":str(pp),"protocol_sha256":file_sha(pp)}}
 assert verify_episode_input_inventory(admission,{"root":str(root)},23,worker)["root"]==str(case)
 other=tmp_path/"other.json";other_ref=freeze_study_inputs(case,other)
 with pytest.raises(ValueError,match="inventory"):verify_episode_input_inventory({**admission,"input_inventory":other_ref},{"root":str(root)},23,worker)

"""All-use calibration contracts; owned CPU models and explicit fit windows."""
from dataclasses import replace,FrozenInstanceError
from copy import deepcopy
import importlib,importlib.util
from types import SimpleNamespace
import pytest,torch
from poc.bank import make_bank,canonicalize_partition,PROJECTIONS
from poc.runtime.torch_decoder import TinyConfig,make_tiny_decoder
from poc.contracts import Batch
from poc.data.manifest import DataManifest,DocumentRecord

def api():
 assert importlib.util.find_spec('poc.shared_calibration'),'all-use shared calibration module missing'
 return importlib.import_module('poc.shared_calibration')

@pytest.fixture
def owned():
 teacher=make_tiny_decoder(TinyConfig(vocab_size=11,hidden_size=8,intermediate_size=12,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024),seed=3,dtype=torch.float32);teacher.tokenizer_digest='a'*64;layout=make_bank(teacher,canonicalize_partition([[0,2],[1],[3]],4),1,17)
 with torch.no_grad():
  for depth in (0,2):
   for projection in PROJECTIONS:
    factors=layout.layers[depth].mlp.residuals[projection];factors.A.fill_(.17+depth*.01);factors.B.fill_(.08+depth*.02)
 records=tuple(DocumentRecord('doc'+str(i),'owned','b'*40,'owned',i,text='owned source '+str(i)) for i in range(2));population=DataManifest(records,{r.document_id:'fit' for r in records},{r.document_id:'component'+str(i) for i,r in enumerate(records)},'owned')
 windows=[]
 for i in range(2):
  ids=(torch.arange(512)[None]+i)%11;windows.append(Batch(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,511),dtype=torch.bool),torch.arange(512)[None],('doc'+str(i),),('component'+str(i),)))
 return SimpleNamespace(teacher=teacher,layout=layout,population=population,windows=tuple(windows))

def bind(o,*,source='current_student',windows=None,population=None,**changes):
 m=api();windows=o.windows if windows is None else windows;population=o.population if population is None else population;model=o.layout if source=='current_student' else o.teacher
 return m.SharedCalibrationSpec(layout_sha256=o.layout.manifest().checkpoint_digest,source_sha256=model.manifest().checkpoint_digest,population_sha256=population.digest(),window_plan_sha256=m.window_plan_sha256(windows),tokenizer_sha256='a'*64,source=source,**changes)

def test_spec_is_immutable_closed_and_roundtrips(owned):
 m=api();s=bind(owned);assert m.SharedCalibrationSpec.from_mapping(s.identity())==s and len(s.digest())==64
 with pytest.raises(FrozenInstanceError):s.source='teacher'
 with pytest.raises((ValueError,TypeError)):m.SharedCalibrationSpec.from_mapping(dict(s.identity(),surprise=True))

@pytest.mark.parametrize('change',[{'schema_version':2},{'schema_version':True},{'source':'all'},{'context_length':256},{'max_windows':0},{'max_windows':True},{'max_statistics_bytes':0},{'max_statistics_bytes':True},{'source_sha256':'x'},{'numerical_profile':'unqualified'}])
def test_spec_rejects_unqualified_settings(owned,change):
 s=bind(owned)
 with pytest.raises(ValueError):replace(s,**change)

@pytest.mark.parametrize('source',['current_student','teacher'])
def test_preflight_counts_every_canonical_owner_and_use_without_forward(owned,monkeypatch,source):
 o=owned;m=api();s=bind(o,source=source);model=o.layout if source=='current_student' else o.teacher;monkeypatch.setattr(model,'final_hidden',lambda *a,**k:pytest.fail('preflight must not forward'));p=m.calibration_preflight(o.layout,model,o.population,o.windows,s)
 assert len(p['owners'])==9 and p['raw_target_presentations']==1022 and p['projection_input_presentations']==1022*4*3
 assert p['owners']['bank.0.down_proj.weight']['use_sites']==[0,2] and p['owners']['bank.0.down_proj.weight']['shape']==[8,12]
 assert p['statistics_bytes']==3*(2*8*(8*8+2*8)+8*(12*12+2*12))

@pytest.mark.parametrize('bad',['dev','protected','quarantine','document','component','empty_mask','short','position_reset','windows','memory','source_kind','tokenizer','window_mutation','grad','existing_hook','projection_override','multiple_contexts'])
def test_preflight_denies_invalid_inputs_before_neural_work(owned,monkeypatch,bad):
 o=owned;m=api();model=o.layout;windows=o.windows;population=o.population;s=bind(o)
 if bad in ('dev','protected'):population=replace(population,roles={r.document_id:('dev' if bad=='dev' else 'test') for r in population.records})
 elif bad=='quarantine':population=replace(population,quarantine={'doc0':'owned exclusion'})
 elif bad=='document':windows=(replace(windows[0],document_ids=('other',)),windows[1])
 elif bad=='component':windows=(replace(windows[0],cluster_ids=('wrong',)),windows[1])
 elif bad=='empty_mask':windows=(replace(windows[0],target_mask=torch.zeros_like(windows[0].target_mask)),windows[1])
 elif bad=='short':
  b=windows[0];windows=(Batch(b.input_ids[:,:-1],b.attention_mask[:,:-1],b.target_mask[:,:-1],b.positions[:,:-1],b.document_ids,b.cluster_ids),windows[1])
 elif bad=='position_reset':windows=(replace(windows[0],positions=windows[0].positions+1),windows[1])
 elif bad=='windows':s=replace(s,max_windows=1)
 elif bad=='memory':s=replace(s,max_statistics_bytes=1)
 elif bad=='source_kind':model=o.teacher;s=replace(s,source_sha256=model.manifest().checkpoint_digest)
 elif bad=='tokenizer':s=replace(s,tokenizer_sha256='c'*64)
 elif bad=='window_mutation':windows[0].input_ids[0,1]=4
 elif bad=='grad':next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
 elif bad=='existing_hook':model.layers[1].mlp.register_forward_pre_hook(lambda *a:None)
 elif bad=='projection_override':model.layers[0].mlp.projection=lambda *a:None
 elif bad=='multiple_contexts':
  b=windows[0];windows=(b,replace(b,input_ids=(b.input_ids+1)%11))
 if bad not in ('windows','memory','source_kind','tokenizer','window_mutation','grad','existing_hook','projection_override'):s=replace(s,population_sha256=population.digest(),window_plan_sha256=m.window_plan_sha256(windows))
 monkeypatch.setattr(model,'final_hidden',lambda *a,**k:pytest.fail('denial must precede neural work'))
 with pytest.raises(ValueError):m.calibration_preflight(o.layout,model,population,windows,s)


def test_fp64_model_is_not_silently_treated_as_fp32_calibration(owned,monkeypatch):
 o=owned;o.layout.double();s=bind(o);monkeypatch.setattr(o.layout,'final_hidden',lambda *a,**k:pytest.fail('wrong profile must not forward'))
 with pytest.raises(ValueError):api().calibration_preflight(o.layout,o.layout,o.population,o.windows,s)


def reference_inputs(o,model,windows):
 """Independent prefix and explicit projection arithmetic; no observers."""
 from torch.nn import functional as F
 rows={}
 with torch.no_grad():
  for batch in windows:
   for depth,layer in enumerate(model.layers):
    boundary=model.prefix_boundary(batch,depth);x=model._rms(boundary.u,layer.post_attention_layernorm)
    def linear(name):
     if model is o.teacher:return F.linear(x,getattr(layer.mlp,name).weight)
     bank=getattr(model.bank[layer.mlp.index],name).weight;factors=layer.mlp.residuals[name]
     return F.linear(x,bank)+F.linear(F.linear(x,factors.A),factors.B)
    down=F.silu(linear('gate_proj'))*linear('up_proj')
    for name,value in (('gate_proj',x),('up_proj',x),('down_proj',down)):
     key=f'bank.{o.layout.partition.reuse_map[depth]}.{name}.weight';rows.setdefault(key,[]).append(value[:,:-1][batch.target_mask].double())
 return {key:torch.cat(values) for key,values in rows.items()}


@pytest.mark.parametrize('source',['current_student','teacher'])
def test_collects_real_all_use_inputs_and_preserves_full_hidden(owned,monkeypatch,source):
 o=owned;m=api();s=bind(o,source=source);model=o.layout if source=='current_student' else o.teacher
 expected=reference_inputs(o,model,o.windows)
 with torch.no_grad():hidden=[model.final_hidden(b).clone() for b in o.windows]
 original=model.final_hidden;calls=[]
 def counted(batch):calls.append(batch);return original(batch)
 monkeypatch.setattr(model,'final_hidden',counted)
 flags=[(module,module.training) for module in model.modules()];model.train();model.layers[1].eval();flags=[(module,module.training) for module in model.modules()]
 before={n:p.detach().clone() for n,p in model.named_parameters()};req={n:p.requires_grad for n,p in model.named_parameters()}
 result=m.collect_shared_inputs(o.layout,model,o.population,o.windows,s)
 assert len(calls)==2 and result.hidden_sha256==tuple(m.tensor_sha256(x) for x in hidden)
 assert result.spec_sha256==s.digest() and result.counters['raw_target_presentations']==1022
 assert result.counters['projection_input_presentations']==12264 and len(result.owners)==9
 for key,values in expected.items():
  stats=result.owners[key];assert stats.count==len(values)
  torch.testing.assert_close(stats.moment_sum,values.T@values,rtol=1e-12,atol=1e-12)
  torch.testing.assert_close(stats.abs_sum,values.abs().sum(0),rtol=1e-12,atol=1e-12)
  torch.testing.assert_close(stats.max_abs,values.abs().max(0).values,rtol=0,atol=0)
 assert result.owners['bank.0.down_proj.weight'].use_counts=={0:1022,2:1022}
 assert all(module.training==flag for module,flag in flags)
 for n,p in model.named_parameters():assert torch.equal(p,before[n]) and p.requires_grad==req[n] and p.grad is None
 assert all('projection' not in layer.mlp.__dict__ for layer in o.layout.layers)
 assert all(not module._forward_pre_hooks for module in model.modules())
 first=result.digest();result.owners['bank.0.down_proj.weight'].moment_sum[0,0]+=1;assert result.digest()!=first


@pytest.mark.parametrize('source',['current_student','teacher'])
def test_repeated_sliced_masks_keep_full_context_and_explicit_eos_targets(owned,source):
 o=owned;m=api();model=o.layout if source=='current_student' else o.teacher;b=o.windows[0]
 # Token2 serves as an ordinary EOS value and padded value. Only masks decide.
 ids=b.input_ids.clone();ids[0,3]=2;ids[0,510:]=2
 attention=b.attention_mask.clone();attention[0,510:]=False
 mask=torch.zeros_like(b.target_mask);mask[0,[0,2,508]]=True
 a=replace(b,input_ids=ids,attention_mask=attention,target_mask=mask)
 other=torch.zeros_like(mask);other[0,[2,5]]=True
 windows=(a,a,replace(a,target_mask=other));s=bind(o,source=source,windows=windows)
 expected=reference_inputs(o,model,windows);r=m.collect_shared_inputs(o.layout,model,o.population,windows,s)
 assert r.counters['raw_target_presentations']==8 and r.counters['unique_context_target_positions']==4
 assert r.counters['unique_contexts']==1 and r.counters['processed_context_tokens']==1536
 assert r.owners['bank.0.down_proj.weight'].count==16
 torch.testing.assert_close(r.owners['bank.0.down_proj.weight'].moment_sum,expected['bank.0.down_proj.weight'].T@expected['bank.0.down_proj.weight'],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('source',['current_student','teacher'])
@pytest.mark.parametrize('failure',['raise','missing','duplicate','nonfinite','window_mutation','weight_mutation'])
def test_failed_forward_is_rejected_and_instrumentation_restored(owned,monkeypatch,source,failure):
 o=owned;m=api();s=bind(o,source=source);model=o.layout if source=='current_student' else o.teacher;original=model.final_hidden
 model.train();model.layers[1].eval();flags=[(module,module.training) for module in model.modules()]
 def broken(batch):
  if failure=='missing':return torch.zeros((1,512,8))
  if failure=='nonfinite':
   if source=='current_student':model.layers[0].mlp.projection(torch.full((1,512,8),float('nan')),'gate_proj')
   else:model.layers[0].mlp.gate_proj(torch.full((1,512,8),float('nan')))
  value=original(batch)
  if failure=='duplicate':original(batch)
  elif failure=='window_mutation':batch.input_ids[0,0]=(batch.input_ids[0,0]+1)%11
  elif failure=='weight_mutation':
   with torch.no_grad():next(model.parameters()).view(-1)[0]+=1
  else:raise RuntimeError('owned interrupted dispatch')
  return value
 monkeypatch.setattr(model,'final_hidden',broken)
 with pytest.raises((ValueError,RuntimeError)):m.collect_shared_inputs(o.layout,model,o.population,o.windows,s)
 assert all(module.training==flag for module,flag in flags)
 assert all('projection' not in layer.mlp.__dict__ for layer in o.layout.layers)
 assert all(not module._forward_pre_hooks for module in model.modules())
 assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('source',['current_student','teacher'])
def test_costs_separate_raw_projection_and_scored_loss_targets(owned,source):
 from test_training import MemoryRecorder
 o=owned;m=api();s=bind(o,source=source);model=o.layout if source=='current_student' else o.teacher;rec=MemoryRecorder()
 r=m.collect_shared_inputs(o.layout,model,o.population,o.windows,s,recorder=rec);c=rec.totals
 assert c.raw_corpus_target_presentations==1022 and c.valid_target_presentations==0 and c.unique_tokens==0
 assert c.processed_inputs==2 and c.processed_tokens==1024 and c.teacher_forward_calls==(2 if source=='teacher' else 0)
 assert c.student_forward_calls==(2 if source=='current_student' else 0) and c.student_backward_calls==0 and c.suffix_backward_calls==0 and c.updates==0
 assert c.wall_seconds>0 and c.memory_bytes is None and c.memory_unavailable_reason
 assert len(r.window_records)==2
 for index,row in enumerate(r.window_records):
  assert row['spec_sha256']==s.digest() and row['window_plan_sha256']==s.window_plan_sha256
  assert row['presentation_index']==index and row['raw_targets']==511 and row['projection_inputs']==6132


@pytest.mark.parametrize('source',['current_student','teacher'])
def test_failed_dispatch_cost_is_retained(owned,monkeypatch,source):
 from test_training import MemoryRecorder
 o=owned;m=api();s=bind(o,source=source);model=o.layout if source=='current_student' else o.teacher;rec=MemoryRecorder()
 def fail(batch):raise RuntimeError('owned failed forward')
 monkeypatch.setattr(model,'final_hidden',fail)
 with pytest.raises(RuntimeError):m.collect_shared_inputs(o.layout,model,o.population,o.windows,s,recorder=rec)
 c=rec.totals;assert c.raw_corpus_target_presentations==511 and c.processed_tokens==512 and c.wall_seconds>0
 assert c.valid_target_presentations==0 and c.teacher_forward_calls+c.student_forward_calls==1


def test_budget_denial_prevents_forward_and_restores_observers(owned,monkeypatch):
 from test_training import MemoryRecorder
 from poc.budget import BudgetExceeded
 o=owned;m=api();s=bind(o)
 class Deny(MemoryRecorder):
  def check(self,cost):raise BudgetExceeded('owned exhausted budget')
 rec=Deny();monkeypatch.setattr(o.layout,'final_hidden',lambda *a:pytest.fail('no unbudgeted neural work'))
 with pytest.raises(BudgetExceeded):m.collect_shared_inputs(o.layout,o.layout,o.population,o.windows,s,recorder=rec)
 assert rec.rows==[] and all('projection' not in layer.mlp.__dict__ for layer in o.layout.layers)


def test_invalid_input_does_not_charge_a_forward(owned,monkeypatch):
 from test_training import MemoryRecorder
 o=owned;m=api();s=bind(o);rec=MemoryRecorder();o.windows[0].input_ids[0,0]=9
 monkeypatch.setattr(o.layout,'final_hidden',lambda *a:pytest.fail('no forward on invalid input'))
 with pytest.raises(ValueError):m.collect_shared_inputs(o.layout,o.layout,o.population,o.windows,s,recorder=rec)
 assert rec.rows==[]


@pytest.mark.parametrize('source',['current_student','teacher'])
@pytest.mark.parametrize('kind',['forward','pre_forward'])
def test_global_hooks_are_denied_without_removing_caller_hook(owned,source,kind):
 o=owned;m=api();model=o.layout if source=='current_student' else o.teacher;s=bind(o,source=source)
 module_api=torch.nn.modules.module
 if kind=='forward':handle=module_api.register_module_forward_hook(lambda module,args,out:out*2 if module is model.layers[0].mlp else None)
 else:handle=module_api.register_module_forward_pre_hook(lambda module,args:(args[0]*2,) if module is model.layers[0].mlp else None)
 try:
  hooks=module_api._global_forward_hooks if kind=='forward' else module_api._global_forward_pre_hooks
  with pytest.raises(ValueError):m.collect_shared_inputs(o.layout,model,o.population,o.windows,s)
  assert handle.id in hooks
 finally:handle.remove()


@pytest.mark.parametrize('source',['current_student','teacher'])
def test_tokenizer_change_during_forward_invalidates_result(owned,monkeypatch,source):
 o=owned;m=api();model=o.layout if source=='current_student' else o.teacher;s=bind(o,source=source);original=model.final_hidden
 def mutate(batch):
  value=original(batch);model.tokenizer_digest='c'*64;return value
 monkeypatch.setattr(model,'final_hidden',mutate)
 with pytest.raises(ValueError):m.collect_shared_inputs(o.layout,model,o.population,o.windows,s)
 assert all(not module._forward_pre_hooks for module in model.modules())
 assert all('projection' not in layer.mlp.__dict__ for layer in o.layout.layers)


@pytest.mark.parametrize('alias',['cross_bank','cross_projection'])
def test_undeclared_parameter_alias_is_denied_before_execution(owned,monkeypatch,alias):
 o=owned;m=api()
 if alias=='cross_bank':o.layout.bank[1].gate_proj.weight=o.layout.bank[0].gate_proj.weight
 else:o.layout.bank[0].up_proj.weight=o.layout.bank[0].gate_proj.weight
 s=bind(o)
 monkeypatch.setattr(o.layout,'final_hidden',lambda *a:pytest.fail('undeclared alias must fail before neural work'))
 with pytest.raises(ValueError):m.collect_shared_inputs(o.layout,o.layout,o.population,o.windows,s)


def test_unrouted_extra_bank_is_denied_before_execution(owned,monkeypatch):
 from copy import deepcopy
 o=owned;m=api();o.layout.bank.append(deepcopy(o.layout.bank[1]));s=bind(o)
 monkeypatch.setattr(o.layout,'final_hidden',lambda *a:pytest.fail('unrouted bank must not be silently omitted'))
 with pytest.raises(ValueError):m.collect_shared_inputs(o.layout,o.layout,o.population,o.windows,s)

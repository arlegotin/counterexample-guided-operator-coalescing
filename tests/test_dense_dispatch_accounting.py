"""Owned accounting units for complete-context dense calibration; no Metal import."""
import pytest

@pytest.mark.parametrize('phase',['train','valid'])
def test_dwq_full_distribution_targets_keep_original_final_target(phase):
 from poc.quantize import dense_dispatch_units
 units=dense_dispatch_units('dwq',phase,input_tokens=511,batch_size=1,context_tokens=512)
 assert units=={'processed_inputs':1,'processed_tokens':511,'raw_corpus_target_presentations':511,
  'distribution_targets':511,'model_shifted_pairs':510}

@pytest.mark.parametrize('method',['dwq','gptq','awq'])
@pytest.mark.parametrize('phase',['bridge','roundtrip'])
def test_full_context_bridge_and_roundtrip_keep_actual_input_work(method,phase):
 from poc.quantize import dense_dispatch_units
 u=dense_dispatch_units(method,phase,input_tokens=512,batch_size=1,context_tokens=512)
 assert u['processed_tokens']==512 and u['raw_corpus_target_presentations']==511 and u['model_shifted_pairs']==511

@pytest.mark.parametrize('method',['gptq','awq'])
def test_complete_calibration_forward_units(method):
 from poc.quantize import dense_dispatch_units
 u=dense_dispatch_units(method,'calibration',input_tokens=512,batch_size=2,context_tokens=512)
 assert u['processed_inputs']==2 and u['processed_tokens']==1024 and u['raw_corpus_target_presentations']==1022

@pytest.mark.parametrize('change',['method','phase','truncated','full_dwq','batch','context','bool'])
def test_unknown_or_ambiguous_context_units_fail(change):
 from poc.quantize import dense_dispatch_units
 args=dict(method='dwq',phase='train',input_tokens=511,batch_size=1,context_tokens=512)
 if change=='method':args['method']='dynamic'
 elif change=='phase':args['phase']='unknown'
 elif change=='truncated':args['input_tokens']=510
 elif change=='full_dwq':args['input_tokens']=512
 elif change=='batch':args['batch_size']=0
 elif change=='context':args['context_tokens']=513
 elif change=='bool':args['input_tokens']=True
 with pytest.raises(ValueError):dense_dispatch_units(**args)


@pytest.fixture
def official_stub(monkeypatch):
 import sys,types
 from dataclasses import dataclass
 from types import SimpleNamespace
 @dataclass
 class Args:pass
 class Model:
  calls=0
  def __init__(self,args):self.model=SimpleNamespace(norm=SimpleNamespace());self.layers=[]
  def load_weights(self,weights,strict):assert weights==[] and strict
  def eval(self):return self
  def parameters(self):return {}
  def __call__(self,inputs,*a,**k):Model.calls+=1;return inputs
 for name in ('mlx','mlx_lm','mlx_lm.models'):
  module=types.ModuleType(name);module.__path__=[];monkeypatch.setitem(sys.modules,name,module)
 mx=types.ModuleType('mlx.core');mx.eval=lambda *a:None;monkeypatch.setitem(sys.modules,'mlx.core',mx)
 llama=types.ModuleType('mlx_lm.models.llama');llama.Model=Model;llama.ModelArgs=Args;monkeypatch.setitem(sys.modules,'mlx_lm.models.llama',llama)
 native=SimpleNamespace(metadata={'config':{}},rope_theta=10000,head_dim=64,weights={},norm_epsilons={'norm':1e-5})
 class Recorder:
  def __init__(self):self.rows=[];self.checked=[];self.reject_update=False
  def check(self,cost):
   self.checked.append(cost)
   if self.reject_update and cost.updates:raise ValueError('owned update budget exhausted')
  def record(self,cost):self.rows.append(cost)
 return native,Model,Recorder,lambda b,s:SimpleNamespace(shape=(b,s),size=b*s)


@pytest.mark.parametrize('profile,raw',[(None,0),('raw-contexts-v1',510)])
def test_existing_official_counter_defaults_are_preserved(official_stub,profile,raw):
 from poc.quantize import _official_model
 n,base,R,inputs=official_stub;r=R();model,state=_official_model(n,r,'dwq',profile);state['quantized']=True
 model(inputs(1,511))
 assert len(r.rows)==1 and r.rows[0].raw_corpus_target_presentations==raw and r.rows[0].valid_target_presentations==0
 assert base.calls==1


@pytest.mark.parametrize('phase',['train','valid'])
def test_new_official_counter_charges_original_context_and_loss_targets(official_stub,phase):
 from poc.quantize import _official_model
 n,base,R,inputs=official_stub;r=R();model,state=_official_model(n,r,'dwq','dense-native-contexts-v2',accounting_context=512)
 state.update(quantized=True,phase=phase);model(inputs(1,511))
 cost=r.rows[0]
 assert cost.raw_corpus_target_presentations==511 and cost.processed_tokens==511 and cost.valid_target_presentations==511
 assert cost.student_forward_calls==1 and cost.updates==0 and base.calls==1


def test_new_dwq_counter_preflights_update_before_student_forward(official_stub):
 from poc.quantize import _official_model
 n,base,R,inputs=official_stub;r=R();r.reject_update=True
 model,state=_official_model(n,r,'dwq','dense-native-contexts-v2',accounting_context=512);state.update(quantized=True,phase='train')
 with pytest.raises(ValueError,match='budget exhausted'):model(inputs(1,511))
 assert base.calls==0 and r.rows==[]


def test_new_bridge_requires_complete_original_context_before_dispatch(official_stub):
 from poc.quantize import _official_model
 n,base,R,inputs=official_stub;r=R();model,state=_official_model(n,r,'dwq','dense-native-contexts-v2',accounting_context=512)
 with pytest.raises(ValueError):model(inputs(1,511))
 assert base.calls==0 and r.rows==[]


@pytest.mark.parametrize('phase,tokens',[('bridge',512),('train',511),('valid',511)])
def test_native_teacher_context_views_have_no_duplicate_scored_loss(phase,tokens):
 from poc.quantize import dense_native_cost
 c=dense_native_cost('dwq',phase,input_tokens=tokens,batch_size=1,context_tokens=512,teacher=True)
 assert c.teacher_forward_calls==1 and c.student_forward_calls==0 and c.raw_corpus_target_presentations==511
 assert c.processed_tokens==tokens and c.valid_target_presentations==0 and c.updates==0


def test_native_roundtrip_student_view_is_explicit_work():
 from poc.quantize import dense_native_cost
 c=dense_native_cost('dwq','roundtrip',input_tokens=512,batch_size=1,context_tokens=512,teacher=False)
 assert c.student_forward_calls==1 and c.teacher_forward_calls==0 and c.raw_corpus_target_presentations==511


@pytest.mark.parametrize('profile,scored,logical',[(None,511,0),('raw-contexts-v1',511,0),('dense-native-contexts-v2',0,511)])
def test_adam_update_keeps_legacy_and_avoids_new_doublecharge(profile,scored,logical):
 from poc.quantize import dense_optimizer_cost
 cost,exposure=dense_optimizer_cost(profile,context_tokens=512,loss_target_count=511)
 assert cost.updates==1 and cost.student_backward_calls==1 and cost.valid_target_presentations==scored and exposure==logical


def test_new_adam_update_rejects_missing_actual_loss_targets():
 from poc.quantize import dense_optimizer_cost
 with pytest.raises(ValueError):dense_optimizer_cost('dense-native-contexts-v2',context_tokens=512,loss_target_count=0)


@pytest.fixture
def calibration_fixture(tmp_path):
 from poc.data.manifest import DataManifest,DocumentRecord
 class Tokenizer:
  def __init__(self):self.calls=[]
  def encode(self,text,**kwargs):
   self.calls.append(kwargs)
   return list(range(512)) if kwargs.get('add_special_tokens') is False else [999]+list(range(512))
 record=DocumentRecord('owned','owned','r','s',0,text='owned calibration content')
 manifest=DataManifest((record,),{'owned':'fit'},{'owned':'component'},'owned',str(tmp_path/'exposure.jsonl'))
 return Tokenizer(),manifest


def test_new_dense_calibration_has_no_implicit_special_tokens(calibration_fixture):
 from poc.quantize import project_calibration
 tokenizer,manifest=calibration_fixture
 cal=project_calibration(tokenizer,manifest,1,512,tokenization_policy='no-special-tokens-v2')
 assert tokenizer.calls==[{'add_special_tokens':False}] and cal.tokens[0][0]==0 and cal.tokens[0][-1]==511


def test_default_dense_calibration_tokenization_is_preserved(calibration_fixture):
 from poc.quantize import project_calibration
 tokenizer,manifest=calibration_fixture;cal=project_calibration(tokenizer,manifest,1,512)
 assert tokenizer.calls==[{}] and cal.tokens[0][0]==999


def test_unknown_tokenization_policy_fails_before_content_io():
 from poc.quantize import project_calibration
 with pytest.raises(ValueError):project_calibration(None,None,1,512,tokenization_policy='unqualified')


class OwnedArray:
 shape=(2,3)
 dtype='float32'


class OwnedDWQLeaf:
 bits=4;group_size=64;mode='affine'
 def __init__(self):self.scales=OwnedArray();self.biases=OwnedArray();self.live=set()
 def unfreeze(self,*,keys,recurse):assert recurse is False;self.live.update(keys)


class OwnedDWQModel:
 def __init__(self):self.leaf=OwnedDWQLeaf();self.norm=OwnedArray();self.norm_live=True;self.frozen=False
 def freeze(self):self.frozen=True;self.norm_live=False;self.leaf.live.clear()
 def apply_to_modules(self,fn):fn('layer.q',self.leaf)
 def trainable_parameters(self):
  result={'layer':{'q':{k:getattr(self.leaf,k) for k in self.leaf.live}}}
  if self.norm_live:result['norm']={'weight':self.norm}
  return result


@pytest.mark.parametrize('legacy',[None,'raw-contexts-v1'])
def test_review_r1_preserves_legacy_dwq_trainability(legacy):
 from poc.quantize import configure_dense_dwq_trainability
 model=OwnedDWQModel();assert configure_dense_dwq_trainability(model,legacy) is None
 assert model.norm_live and not model.frozen


def test_review_r1_new_dwq_freezes_norms_and_records_only_affine_owners():
 from poc.quantize import configure_dense_dwq_trainability,validate_dense_dwq_trainability
 model=OwnedDWQModel();scope=configure_dense_dwq_trainability(model,'dense-native-contexts-v2')
 assert model.frozen and not model.norm_live
 assert [row['name'] for row in scope['owners']]==['layer.q.biases','layer.q.scales']
 validate_dense_dwq_trainability(model,scope)
 model.norm_live=True
 with pytest.raises(ValueError,match='trainable|scope|owner'):validate_dense_dwq_trainability(model,scope)


def test_review_r1_grad_parameter_tree_cannot_add_a_float_norm():
 from poc.quantize import configure_dense_dwq_trainability,validate_dense_dwq_trainability
 model=OwnedDWQModel();scope=configure_dense_dwq_trainability(model,'dense-native-contexts-v2')
 params=model.trainable_parameters();params['norm']={'weight':OwnedArray()}
 with pytest.raises(ValueError,match='trainable|scope|owner'):validate_dense_dwq_trainability(model,scope,params=params)

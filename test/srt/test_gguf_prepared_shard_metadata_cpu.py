"""Execute actual source AST with fake dispatch; no torch/GPU numerical claim."""
from pathlib import Path
import ast,json,statistics,time,types
PATH=Path(__file__).resolve().parents[2]/'python/sglang/srt/layers/quantization/gguf.py'

def method_type(text):
 tree=ast.parse(text);helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ordered_gguf_shard_ids')
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GGUFLinearMethod')
 names={'apply','_prepare_shard_exec_plan','_validated_shard_exec_plan'}
 body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_SHARD_EXEC_PLAN_ATTR' for t in n.targets)]
 reduced=ast.Module(body=[*ast.parse('from __future__ import annotations').body,helper,ast.ClassDef(name='ActualMethod',bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[])
 calls=[]
 class Output:
  def __init__(self,parts):self.parts=parts;self.bias_count=0
  def add_(self,bias):self.bias_count+=1;return self
 class Slice:
  def __init__(self,source,key):self.source=source;self.key=key
  def contiguous(self):return self
 def fused(x,w,kind):
  if isinstance(w,Slice):spec=(w.key[0].start,w.key[0].stop,w.key[1].stop,kind)
  else:spec=('whole',kind)
  calls.append(spec);return Output([spec])
 def cat(values,axis):assert axis==1;return Output([part for out in values for part in out.parts])
 ns={'torch':types.SimpleNamespace(cat=cat),'fused_mul_mat_gguf':fused}
 exec(compile(ast.fix_missing_locations(reduced),str(PATH),'exec'),ns)
 return ns['ActualMethod'],calls
class Weight:
 def __init__(self,ids,offsets):self.shard_id=ids;self.shard_offset_map=offsets
 def __getitem__(self,key):return types.SimpleNamespace() # overwritten per namespace below

def layer(ids,offsets,kinds):
 return types.SimpleNamespace(qweight=Weight(list(ids),dict(offsets)),qweight_type=types.SimpleNamespace(shard_weight_type=dict(kinds),weight_type=8))

def bind_weight_slices(klass):
 # Locate the exact Slice class used by fused(), avoiding real tensors.
 fused=klass.apply.__globals__['fused_mul_mat_gguf'];Slice=dict(zip(fused.__code__.co_freevars,[v.cell_contents for v in fused.__closure__]))['Slice']
 Weight.__getitem__=lambda self,key:Slice(self,key)

new,calls=method_type(PATH.read_text());tested=[]
def check(name,l,expected,mutation=None,bias=None):
 method=new();method._prepare_shard_exec_plan(l)
 if mutation:mutation(l)
 calls.clear();out=method.apply(l,object(),bias)
 assert out.parts==expected,(name,out.parts,expected)
 assert out.bias_count==(bias is not None),(name,'bias')
 tested.append(name)

bind_weight_slices(new)
check('integer-gateup-order',layer([1,0],{0:(0,4,8),1:(4,8,8)},{0:8,1:8}),[(0,4,8,8),(4,8,8,8)])
check('QKV-logical-order-and-padding',layer(['v','q','k'],{'q':(0,4,8),'k':(4,6,4),'v':(6,8,4)},{'q':8,'k':0,'v':8}),[(0,4,8,8),(4,6,4,0),(6,8,4,8)],bias=object())
check('custom-order-preserved',layer(['z','a'],{'z':(0,2,5),'a':(2,7,3)},{'z':8,'a':1}),[(0,2,5,8),(2,7,3,1)])
base=lambda:layer([0,1],{0:(0,4,8),1:(4,8,8)},{0:8,1:8})
check('qweight-replaced-fallback',base(),[(0,2,6,8),(2,6,6,8)],lambda l:setattr(l,'qweight',Weight([0,1],{0:(0,2,6),1:(2,6,6)})))
check('type-parameter-replaced-fallback',base(),[(0,4,8,0),(4,8,8,8)],lambda l:setattr(l,'qweight_type',types.SimpleNamespace(shard_weight_type={0:0,1:8},weight_type=8)))
check('offset-inplace-mutated-fallback',base(),[(0,3,6,8),(4,8,8,8)],lambda l:l.qweight.shard_offset_map.__setitem__(0,(0,3,6)))
check('type-inplace-mutated-fallback',base(),[(0,4,8,0),(4,8,8,8)],lambda l:l.qweight_type.shard_weight_type.__setitem__(0,0))
check('shard-list-mutated-fallback',base(),[(4,8,8,8)],lambda l:setattr(l.qweight,'shard_id',[1]))
l=base();method=new();method._prepare_shard_exec_plan(l);del l.qweight.shard_offset_map
assert method._validated_shard_exec_plan(l) is None;assert not hasattr(l,'_gguf_shard_exec_plan');tested.append('missing-offset-invalidates-plan')
l=base();method=new();method._prepare_shard_exec_plan(l);del l.qweight_type.shard_weight_type
assert method._validated_shard_exec_plan(l) is None;tested.append('missing-type-invalidates-plan')
l=layer([],{},{});out=new().apply(l,object(),object());assert out.parts==[('whole',8)] and out.bias_count==1;tested.append('unsharded-bias-once')
l=base();calls.clear();out=new().apply(l,object());assert out.parts==[(0,4,8,8),(4,8,8,8)];tested.append('uncached-original-fallback')
print(json.dumps({'status':'PASS','actual_source_dispatch_controls':tested,'count':len(tested),'limitations':'Actual source AST and fake dispatch only; not torch/CUDA numerical equivalence or end-to-end performance.'},indent=2))

"""Narrow stdlib controls against actual immutable 5b source, without CUDA import.

Core AST equality covers unchanged sampling/math branches. Fake greedy forward
checks trace off/on/off and logprob on/off. These are not GPU numeric or quality
evidence; original failed worker test is retained by the research Controller.
"""
import ast
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest

# This file lives in test/srt: the repository root is three parents up.
ROOT=Path(__file__).resolve().parents[2]
PATH='python/sglang/srt/layers/sampler.py'
BASE='5b89820b04cb26385d45915c484708796ba7df91'
def sources():
    root=Path.cwd()
    if not (root/PATH).is_file():root=ROOT
    baseline=subprocess.check_output(['git','-C',str(root),'show',BASE+':'+PATH],text=True)
    return baseline,(root/PATH).read_text()
def function(source,name):
    tree=ast.parse(source)
    if name=='forward':
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sampler')
        return copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
    return copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
class Core(ast.NodeTransformer):
    def visit_Expr(self,n):
        if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_trace_e2e_sampler':return None
        return self.generic_visit(n)
    def visit_Assign(self,n):
        if any(isinstance(t,ast.Name) and t.id=='trace_e2e' for t in n.targets):
            assert ast.unparse(n.value)=='_trace_e2e_sampler if envs.SGLANG_TRACE_SAMPLER_E2E.get() else None'
            return None
        return self.generic_visit(n)
    def visit_If(self,n):
        if ast.unparse(n.test)=='trace_e2e is not None':
            assert len(n.body)==1 and not n.orelse and isinstance(n.body[0],ast.Expr)
            assert isinstance(n.body[0].value,ast.Call) and ast.unparse(n.body[0].value.func)=='trace_e2e'
            return None
        return self.generic_visit(n)
def compile_fn(n,namespace):
    n.returns=None
    for arg in [*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs]:arg.annotation=None
    n.decorator_list=[]
    module=ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[]))
    exec(compile(module,'<extracted sampler>','exec'),namespace)
    return namespace[n.name]
def exercise(source):
    ops=[];shapes=[];calls=[];printed=[];flag=NS(enabled=False)
    class Tensor:
        def __init__(self,name):self.name=name
        @property
        def shape(self):shapes.append(self.name);return (2,4) if self.name=='logits' else (2,)
    def log_softmax(value,dim):ops.append('log_softmax');return Tensor('logprobs')
    def argmax(value,dim):ops.append('argmax');return Tensor('ids')
    def preprocess(value,info):ops.append('preprocess');return value
    def compute(*args):
        ops.append('compute_logprobs')
        return NS(write_output_to=lambda out:ops.append('write_output'))
    namespace={'envs':NS(SGLANG_TRACE_SAMPLER_E2E=NS(get=lambda:flag.enabled)),
      'torch':NS(argmax=argmax,nn=NS(functional=NS(log_softmax=log_softmax))),
      '_is_hip':False,'_use_aiter':False,'_disable_aiter_greedy_sample':True,
      'SGLANG_RETURN_ORIGINAL_LOGPROB':False,'SYNC_TOKEN_IDS_ACROSS_TP':False,
      'maybe_detect_nan':lambda *a:ops.append('nan_probe'),
      'get_parallel':lambda:NS(attn_dp_rank=0,tp_rank=0),
      'print':lambda *a,**kw:printed.append(a)}
    real_trace=compile_fn(function(source,'_trace_e2e_sampler'),namespace)
    def counted(stage,**fields):calls.append((stage,fields));return real_trace(stage,**fields)
    namespace['_trace_e2e_sampler']=counted
    forward=compile_fn(function(source,'forward'),namespace)
    self=NS(_preprocess_logits=preprocess,_sync_token_ids_across_tp=lambda *a:ops.append('sync'),
       output_logprob_processor=NS(compute_logprobs=compute))
    info=NS(is_all_greedy=True,sampling_mask_batch_indices=None,sampling_support_logprobs_capture_indices=None)
    rows=[]
    for enabled in (False,True,False):
        for with_logprob in (False,True):
            flag.enabled=enabled;ops.clear();shapes.clear();calls.clear();printed.clear()
            result=forward(self,NS(next_token_logits=Tensor('logits')),info,with_logprob,[],[],None)
            rows.append({'enabled':enabled,'logprob':with_logprob,'result':result.name,
                'ops':list(ops),'shapes':list(shapes),'calls':list(calls),'prints':list(printed)})
    return rows
class TraceControls(unittest.TestCase):
    def test_all_non_trace_sampling_branches_identical_to_pinned_baseline(self):
        baseline,candidate=sources()
        self.assertEqual(ast.dump(Core().visit(function(baseline,'forward')),include_attributes=False),
                         ast.dump(Core().visit(function(candidate,'forward')),include_attributes=False))
    def test_greedy_trace_toggle_and_logprob_preserve_ops_and_fields(self):
        baseline,candidate=sources();left=exercise(baseline);right=exercise(candidate)
        for old,new in zip(left,right,strict=True):
            with self.subTest(trace=new['enabled'],logprob=new['logprob']):
                self.assertEqual(old['result'],new['result']);self.assertEqual(old['ops'],new['ops'])
                self.assertEqual(old['prints'],new['prints'])
                if new['enabled']:
                    self.assertEqual(old['calls'],new['calls']);self.assertEqual(len(new['calls']),8)
                else:
                    self.assertEqual(len(old['calls']),8);self.assertEqual(new['calls'],[])
                    self.assertEqual(new['shapes'],[])
if __name__=='__main__':unittest.main(verbosity=2)

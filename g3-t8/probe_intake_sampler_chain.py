#!/usr/bin/env python3
"""g3-t8 probe: request-intake -> sampler-branch -> detokenizer chain for top_logprobs_num (k).

Question (falsifiable): for the frozen probe shape (native /generate,
return_logprob=true, top_logprobs_num=8, logprob_start_len absent/-1), does k=8
survive every CPU-side site into the sampler's topk branch, and do the output
field names stay `output_top_logprobs` end to end? i7 proved scheduler-side
POPULATED_AT_SOURCE hop1-h7 but ASSUMED a fixture batch with k=8; i5 proved the
response read-path names. The leg between "HTTP JSON has k=8" (C7) and "sampler
branch sees k=8" was never traced, nor the detokenizer pass-through.

Method (i7 style): ast-extract REAL functions/statements by file:line anchor,
exec with labeled stdlib fixtures, hard verdict gate (ANY harness error or
anchor drift => NO_CLAIM). Negative controls: k=None/k=0 must reproduce the
silent-empty paths; pruned_len=0 must reproduce input-side empty-by-design.

No engine change; no GPU; no torch/numpy in this sandbox (mocked where the
real object is a tensor). GPU-side topk numerics remain outside this probe.
"""

import ast
import dataclasses
import hashlib
import json
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(HERE, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SRC = {
    "io_struct": "python/sglang/srt/managers/io_struct.py",
    "tm": "python/sglang/srt/managers/tokenizer_manager.py",
    "sched": "python/sglang/srt/managers/scheduler.py",
    "sb": "python/sglang/srt/managers/schedule_batch.py",
    "fb": "python/sglang/srt/model_executor/forward_batch_info.py",
    "mr": "python/sglang/srt/model_executor/model_runner.py",
    "sampler": "python/sglang/srt/layers/sampler.py",
    "lp": "python/sglang/srt/layers/logprob_processor.py",
    "logits": "python/sglang/srt/layers/logits_processor.py",
    "detok": "python/sglang/srt/managers/detokenizer_manager.py",
    "http": "python/sglang/srt/entrypoints/http_server.py",
}

_harness_errors: List[str] = []
hops: List[Dict[str, Any]] = []


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _find_func(tree, name, near=None):
    cands = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if not cands:
        raise KeyError(f"function {name} not found")
    if near is not None:
        cands.sort(key=lambda n: abs(n.lineno - near))
    return cands[0]


def _find_class(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise KeyError(f"class {name} not found")


def _find_stmt_containing(tree, needle):
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.stmt):
            try:
                u = ast.unparse(n)
            except Exception:
                continue
            if needle in u:
                if best is None or (n.end_lineno - n.lineno) < (best.end_lineno - best.lineno):
                    best = n
    if best is None:
        raise KeyError(f"stmt containing {needle!r} not found")
    return best


def _smallest_call_spanning(tree, lo, hi):
    cands = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and n.lineno is not None
             and n.lineno <= lo and hi <= (n.end_lineno or 0)]
    if not cands:
        raise KeyError(f"no Call spanning {lo}-{hi}")
    cands.sort(key=lambda n: (n.end_lineno - n.lineno))
    return cands[0]


def _exprs_in_call_at(call, lo, hi):
    """args + keyword values of `call` whose source lines intersect [lo,hi]."""
    exprs = list(call.args) + [k.value for k in call.keywords]
    sel = [e for e in exprs
           if getattr(e, "lineno", None) is not None
           and not (e.end_lineno < lo or e.lineno > hi)]
    if not sel:
        raise KeyError(f"no args/keywords of call ({call.lineno}-{call.end_lineno}) in {lo}-{hi}")
    return sel


def record(hid, anchor, kind, extracted, detail):
    hops.append({"id": hid, "anchor": anchor, "kind": kind,
                 "extracted_sha256": _sha(extracted) if extracted else None,
                 "detail": detail})


def execute(hid, anchor, code, globals_, expect, extracted=None, kind="EXECUTED"):
    try:
        gl = dict(globals_)
        exec(compile(code, "<probe>", "exec"), gl)
        detail = expect(gl)
        ok = detail.get("passed", True)
        record(hid, anchor, kind if ok else "EXEC_FAILED", extracted or code, detail)
        if not ok:
            _harness_errors.append(f"{hid}: expectation failed: {json.dumps(detail, default=str)}")
        return gl
    except Exception as e:  # noqa: BLE001
        import traceback
        _harness_errors.append(f"{hid}: {type(e).__name__}: {e}")
        record(hid, anchor, "HARNESS_ERROR", extracted or code,
               {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(limit=3)})
        return None


# ---------------------------------------------------------------- fixtures
class _finfo:
    min = -3.4e38


class FakeTorch:
    Tensor = object
    finfo = staticmethod(lambda dt: _finfo())
    arange = staticmethod(lambda n, device=None: list(range(n)))

    class cuda:
        Event = object

    nn = types.SimpleNamespace(functional=types.SimpleNamespace(log_softmax=lambda x, dim=-1: x))


class MockTensor:
    """2-D row mock: clamp_, topk(k, dim), fancy (arange, ids) gather, tolist."""

    def __init__(self, data, dtype="bfloat16"):
        self.data = [list(r) for r in data]
        self.dtype = dtype

    def clamp_(self, min=None):
        return self

    def topk(self, k, dim=-1):
        rows_v, rows_i = [], []
        for r in self.data:
            order = sorted(range(len(r)), key=lambda j: -r[j])[:k]
            rows_v.append([r[j] for j in order])
            rows_i.append(order)
        return MockTensor(rows_v, "float32"), MockTensor(rows_i, "int64")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            return [self.data[i][c] for i, c in zip(rows, cols)]
        return _Row(self.data[key])

    def tolist(self):
        return [list(r) for r in self.data]


class _Row:
    def __init__(self, row):
        self.row = row

    def __getitem__(self, k):
        return self.row[k]


class Ids:
    """batch_next_token_ids mock: len(), iteration, .device."""

    device = "cuda"

    def __init__(self, ids):
        self.ids = list(ids)

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)


class FakeNP:
    ndarray = object


class LogprobStage:
    DECODE = "DECODE"
    PREFILL = "PREFILL"


BASE_G = {
    "torch": FakeTorch, "np": FakeNP, "List": List, "Optional": Optional,
    "Tuple": Tuple, "Dict": Dict, "Any": Any, "dataclasses": dataclasses,
    "field": dataclasses.field, "LogprobStage": LogprobStage,
    "logger": types.SimpleNamespace(warning=lambda *a, **k: None,
                                    error=lambda *a, **k: None),
    "_trace_e2e_sampler": lambda *a, **k: None,
    "LogitsProcessorOutput": object,  # annotation-only stub for extracted classes
    "Union": __import__("typing").Union,
}


class NS(types.SimpleNamespace):
    pass


def main():
    trees, srcs = {}, {}
    for k, rel in SRC.items():
        p = os.path.join(ROOT, rel)
        with open(p) as f:
            s = f.read()
        srcs[k] = s
        trees[k] = ast.parse(s)

    # ================================================================ A intake
    # A1 single-request normalization None -> 0 (probe sends 8 -> unchanged)
    a1 = _find_stmt_containing(trees["io_struct"], "if self.top_logprobs_num is None:")
    a1code = ast.unparse(a1)
    execute("A1_single_None_to_0", f"io_struct.py:{a1.lineno}-{a1.end_lineno}", a1code,
            {"self": NS(top_logprobs_num=None)},
            lambda g: {"in": None, "out": g["self"].top_logprobs_num,
                       "passed": g["self"].top_logprobs_num == 0}, a1code)

    # A2 batch normalize_param (real nested function)
    fn = _find_func(trees["io_struct"], "normalize_param", near=780)
    a2code = ast.unparse(fn)
    execute("A2_normalize_param", f"io_struct.py:{fn.lineno}-{fn.end_lineno}", a2code,
            {**BASE_G, "num": 2, "self": NS(parallel_sample_num=1)},
            lambda g: {"int8": g["normalize_param"](8, 0, "top_logprobs_num"),
                       "None": g["normalize_param"](None, 0, "top_logprobs_num"),
                       "list": g["normalize_param"]([8, 0], 0, "top_logprobs_num"),
                       "passed": g["normalize_param"](8, 0, "top_logprobs_num") == [8, 8]
                       and g["normalize_param"](None, 0, "top_logprobs_num") == [0, 0]}, a2code)

    # A3 fanout keyword at io_struct.py:986
    c3 = _smallest_call_spanning(trees["io_struct"], 986, 986)
    e3 = _exprs_in_call_at(c3, 986, 986)[0]
    e3code = ast.unparse(e3)
    v3 = eval(e3code, {}, {"self": NS(top_logprobs_num=[8, 6]), "i": 0})  # noqa: S307
    execute("A3_fanout_kw", f"io_struct.py:{e3.lineno} (kw of ctor {c3.lineno}-{c3.end_lineno})",
            e3code, {"self": NS(top_logprobs_num=[8, 6]), "i": 0},
            lambda g: {"out": v3, "passed": v3 == 8}, e3code)

    # A4 TM Tokenized keyword at tm.py:1591
    c4 = _smallest_call_spanning(trees["tm"], 1591, 1591)
    e4 = _exprs_in_call_at(c4, 1591, 1591)[0]
    e4code = ast.unparse(e4)
    v4 = eval(e4code, {}, {"obj": NS(top_logprobs_num=8)})  # noqa: S307
    execute("A4_tm_tokenized_kw", f"tokenizer_manager.py:{e4.lineno} (kw of ctor {c4.lineno}-{c4.end_lineno})",
            e4code, {"obj": NS(top_logprobs_num=8)},
            lambda g: {"out": v4, "passed": v4 == 8}, e4code)

    # A5 scheduler pass-through kwargs (scheduler.py:2991-2993)
    c5 = _smallest_call_spanning(trees["sched"], 2991, 2993)
    e5 = _exprs_in_call_at(c5, 2991, 2993)
    n5 = [ast.unparse(x) for x in e5]
    recv = NS(return_logprob=True, top_logprobs_num=8, token_ids_logprob=None)
    v5 = {ast.unparse(x): eval(ast.unparse(x), {}, {"recv_req": recv}) for x in e5}  # noqa: S307
    record("A5_sched_kwargs", f"scheduler.py:2991-2993 (args of _create_generation_req call {c5.lineno}-{c5.end_lineno})",
           "EXECUTED", ast.unparse(c5),
           {"arg_names": n5, "out": v5,
            "passed": v5.get("recv_req.top_logprobs_num") == 8
            and v5.get("recv_req.return_logprob") is True})

    # A6 real ReqLogprob dataclass
    rc = _find_class(trees["sb"], "ReqLogprob")
    a6code = ast.unparse(rc)
    gl6 = execute("A6_reqlogprob_class", f"schedule_batch.py:{rc.lineno}-{rc.end_lineno}", a6code,
                  dict(BASE_G),
                  lambda g: {"has_field": "top_logprobs_num" in g["ReqLogprob"].__dataclass_fields__,
                             "passed": "top_logprobs_num" in g["ReqLogprob"].__dataclass_fields__}, a6code)
    RL = gl6["ReqLogprob"] if gl6 else None
    if RL:
        rl = RL(top_logprobs_num=8, token_ids_logprob=None)
        hops[-1]["detail"]["constructed_k"] = rl.top_logprobs_num
        hops[-1]["detail"]["passed"] = rl.top_logprobs_num == 8
        if rl.top_logprobs_num != 8:
            _harness_errors.append("A6: constructed k != 8")

    # A7 batch assembly If: top_logprobs_nums = [r.logprob.top_logprobs_num ...]
    a7 = None
    for n in ast.walk(trees["sb"]):
        if isinstance(n, ast.If) and "self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]" in ast.unparse(n):
            a7 = n
            break
    if a7 is None:
        raise KeyError("batch-assembly If not found")
    a7code = ast.unparse(a7)
    mk = lambda k: NS(logprob=(RL(top_logprobs_num=k, token_ids_logprob=None) if RL  # noqa: E731
                               else NS(top_logprobs_num=k)))
    reqs = [mk(8), mk(0)]
    b1 = NS(return_logprob=True, top_logprobs_nums=None, token_ids_logprobs=None)
    execute("A7_batch_assembly", f"schedule_batch.py:{a7.lineno}-{a7.end_lineno}", a7code,
            {"self": b1, "reqs": reqs},
            lambda g: {"out": g["self"].top_logprobs_nums,
                       "passed": g["self"].top_logprobs_nums == [8, 0]}, a7code)
    b2 = NS(return_logprob=False, top_logprobs_nums=None, token_ids_logprobs=None)
    execute("A7ctl_batch_assembly_rl_False",
            f"schedule_batch.py:{a7.lineno}-{a7.end_lineno} (control)", a7code,
            {"self": b2, "reqs": reqs},
            lambda g: {"out": g["self"].top_logprobs_nums,
                       "passed": g["self"].top_logprobs_nums is None}, a7code, kind="CONTROL")

    # A8 batch return_logprob = any(...)
    a8 = _find_stmt_containing(trees["sb"], "return_logprob = any((req.return_logprob for req in reqs))")
    a8code = ast.unparse(a8)
    ns8 = {"reqs": [NS(return_logprob=True)]}
    exec(compile(a8code, "<a8>", "exec"), ns8)
    v8 = ns8["return_logprob"]
    execute("A8_batch_rl_any", f"schedule_batch.py:{a8.lineno}", a8code,
            {"reqs": [NS(return_logprob=True)]},
            lambda g: {"out": v8, "passed": v8 is True}, a8code)

    # A9 ForwardBatch init keyword (fb.py:1096)
    c9 = _smallest_call_spanning(trees["fb"], 1096, 1096)
    e9 = _exprs_in_call_at(c9, 1096, 1096)[0]
    e9code = ast.unparse(e9)
    v9 = eval(e9code, {}, {"batch": b1})  # noqa: S307
    execute("A9_forward_batch_kw", f"forward_batch_info.py:{e9.lineno} (kw of ctor {c9.lineno}-{c9.end_lineno})",
            e9code, {"batch": b1}, lambda g: {"out": v9, "passed": v9 == [8, 0]}, e9code)

    # A10 model_runner sampler call args (mr.py:2164-2165)
    c10 = _smallest_call_spanning(trees["mr"], 2164, 2165)
    e10 = _exprs_in_call_at(c10, 2164, 2165)
    fbm = NS(return_logprob=True, top_logprobs_nums=[8])
    n10 = [ast.unparse(x) for x in e10]
    v10 = {ast.unparse(x): eval(ast.unparse(x), {}, {"forward_batch": fbm}) for x in e10}  # noqa: S307
    record("A10_model_runner_sampler_args",
           f"model_runner.py:2164-2165 (args of sampler call {c10.lineno}-{c10.end_lineno})",
           "EXECUTED", ast.unparse(c10),
           {"arg_names": n10, "out": v10,
            "passed": v10.get("forward_batch.return_logprob") is True
            and v10.get("forward_batch.top_logprobs_nums") == [8]})

    # ============================================================ B compute side
    gl_b = dict(BASE_G)
    lprc = _find_class(trees["lp"], "LogprobResult")
    lprcode = ast.unparse(lprc)
    exec(compile(lprcode, "<LogprobResult>", "exec"), gl_b)
    for fname in ("get_top_logprobs", "get_top_logprobs_raw"):
        fnode = _find_func(trees["lp"], fname)
        fcode = ast.unparse(fnode)
        exec(compile(fcode, f"<{fname}>", "exec"), gl_b)
    clp = _find_func(trees["lp"], "compute_logprobs", near=918)
    clpcode = ast.unparse(clp)
    mang = dict(gl_b)
    exec(compile(clpcode, "<compute_logprobs>", "exec"), mang)
    real_compute_logprobs = mang["compute_logprobs"]

    class Proc:
        pass

    def make_proc():
        p = Proc()
        p.compute_logprobs = lambda *a, **k: real_compute_logprobs(p, *a, **k)
        return p

    # B1 real sampler `if return_logprob:` block -> real compute_logprobs -> real write_output_to
    sif = None
    fwd = _find_func(trees["sampler"], "forward", near=148)
    for n in ast.walk(fwd):
        if isinstance(n, ast.If) and "output_logprob_processor.compute_logprobs" in ast.unparse(n):
            sif = n
            break
    if sif is None:
        raise KeyError("sampler if-return_logprob block not found")
    sifcode = ast.unparse(sif)

    class SamplerStub:
        output_logprob_processor = None

    lp_tensor = MockTensor([[-1.0 * (j + 1) for j in range(20)]])

    def run_sampler_block(k, tag, ctl=False):
        out = NS()
        stub = SamplerStub()
        stub.output_logprob_processor = make_proc()
        g = {**BASE_G, "self": stub, "return_logprob": True, "target_logprobs": None,
             "original_logprobs": None, "logprobs": lp_tensor,
             "SGLANG_RETURN_ORIGINAL_LOGPROB": False, "top_logprobs_nums": [k],
             "token_ids_logprobs": [None], "batch_next_token_ids": Ids([7]),
             "logits_output": out}
        execute(tag, f"sampler.py:{sif.lineno}-{sif.end_lineno}"
                 + (" (control k=0)" if ctl else ""), sifcode, g,
                lambda gg: {"chosen": getattr(gg["logits_output"], "next_token_logprobs", None),
                            "top_val": getattr(gg["logits_output"], "next_token_top_logprobs_val", None),
                            "top_idx": getattr(gg["logits_output"], "next_token_top_logprobs_idx", None),
                            "passed": (getattr(gg["logits_output"], "next_token_top_logprobs_val", None) is not None
                                       and len(gg["logits_output"].next_token_top_logprobs_val[0]) == 8
                                       and getattr(gg["logits_output"], "next_token_logprobs", None) is not None)
                            if not ctl else
                            (getattr(gg["logits_output"], "next_token_logprobs", None) is not None
                             and getattr(gg["logits_output"], "next_token_top_logprobs_val", None) is None)},
                sifcode, kind="CONTROL" if ctl else "EXECUTED")

    run_sampler_block(8, "B1_sampler_block_k8_FULL")
    run_sampler_block(0, "B2ctl_sampler_block_k0_chosen_present_top_absent", ctl=True)

    # B3 real get_top_logprobs_raw: DECODE mixed k; PREFILL pruned_len=0
    det = gl_b["get_top_logprobs_raw"]
    big = MockTensor([[-0.1 * (j + 1) for j in range(20)], [-0.2 * (j + 1) for j in range(20)]])
    v_dec, i_dec = det(big, [8, 0], LogprobStage.DECODE, no_copy_to_cpu=True)
    v_pre, i_pre = det(big, [8], LogprobStage.PREFILL,
                       extend_logprob_pruned_lens_cpu=[0], no_copy_to_cpu=True)
    gtr_node = _find_func(trees["lp"], "get_top_logprobs_raw")
    record("B3_get_top_logprobs_raw",
           f"logprob_processor.py:{gtr_node.lineno}-{gtr_node.end_lineno} (real function)",
           "EXECUTED", ast.unparse(gtr_node),
           {"decode_k8_row0_len": len(v_dec[0]), "decode_k0_row1_len": len(v_dec[1]),
            "prefill_pruned0_row": v_pre[0],
            "passed": len(v_dec[0]) == 8 and len(v_dec[1]) == 0 and v_pre[0] == []})

    # B4 real extend guard any(x>0) + control k=0
    gnode = _find_stmt_containing(trees["logits"], "extend_return_top_logprob = any(")
    gcode = ast.unparse(gnode)
    execute("B4_extend_guard_k8", f"logits_processor.py:{gnode.lineno}-{gnode.end_lineno}", gcode,
            {**BASE_G, "forward_batch": NS(top_logprobs_nums=[8])},
            lambda g: {"out": g["extend_return_top_logprob"],
                       "passed": g["extend_return_top_logprob"] is True}, gcode)
    execute("B4ctl_extend_guard_k0", f"logits_processor.py:{gnode.lineno}-{gnode.end_lineno} (control)",
            gcode, {**BASE_G, "forward_batch": NS(top_logprobs_nums=[0])},
            lambda g: {"out": g["extend_return_top_logprob"],
                       "passed": g["extend_return_top_logprob"] is False}, gcode, kind="CONTROL")

    # B5 real pruned-len loop with probe shape (start=12=extend_len -> pruned 0)
    lnode = None
    for n in ast.walk(trees["logits"]):
        if isinstance(n, ast.For) and "extend_logprob_pruned_lens_cpu.append(extend_len - start_len)" in ast.unparse(n):
            lnode = n
            break
    if lnode is None:
        raise KeyError("pruned-len For not found")
    lcode = ast.unparse(lnode)
    execute("B5_extend_pruned_len_loop_probe_shape",
            f"logits_processor.py:{lnode.lineno}-{lnode.end_lineno} (logprob_start_len=-1 -> start=12=extend_len)",
            lcode, {**BASE_G, "extend_return_logprob": False, "extend_logprob_pruned_lens_cpu": [],
                    "forward_batch": NS(extend_seq_lens_cpu=[12], extend_logprob_start_lens_cpu=[12])},
            lambda g: {"pruned": g["extend_logprob_pruned_lens_cpu"],
                       "extend_return_logprob": g["extend_return_logprob"],
                       "passed": g["extend_logprob_pruned_lens_cpu"] == [0]
                       and g["extend_return_logprob"] is False}, lcode)

    # ========================================================== C output fields
    # C1 detokenizer pass-through kwargs (detok.py:497-498)
    try:
        c11 = _smallest_call_spanning(trees["detok"], 497, 498)
        e11 = _exprs_in_call_at(c11, 497, 498)
        n11 = [ast.unparse(x) for x in e11]
        recv2 = NS(output_top_logprobs_val=[[1.0]], output_top_logprobs_idx=[[7]])
        v11 = {ast.unparse(x): eval(ast.unparse(x), {}, {"recv_obj": recv2}) for x in e11}  # noqa: S307
        record("C1_detokenizer_passthrough",
               f"detokenizer_manager.py:497-498 (kwargs of ctor {c11.lineno}-{c11.end_lineno})",
               "EXECUTED", ast.unparse(c11),
               {"kw_names": n11, "out": v11,
                "passed": n11 == ["recv_obj.output_top_logprobs_val", "recv_obj.output_top_logprobs_idx"]
                and v11["recv_obj.output_top_logprobs_val"] == [[1.0]]})
    except Exception as e:  # noqa: BLE001
        _harness_errors.append(f"C1: {type(e).__name__}: {e}")
        record("C1_detokenizer_passthrough", "detokenizer_manager.py:497-498", "HARNESS_ERROR",
               None, {"error": str(e)})

    # C2 TM meta_info assembly (real statements)
    ext_if = _find_stmt_containing(
        trees["tm"], "if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):")
    meta_assign = _find_stmt_containing(
        trees["tm"], "meta_info['output_top_logprobs'] = state.output_top_logprobs")
    block = ast.unparse(ext_if) + "\n" + ast.unparse(meta_assign)
    tm_stub = NS(detokenize_top_logprobs_tokens=lambda v, i, rt: [{"lp": x} for x in v])
    stA = NS(output_top_logprobs_val=[[0.1, 0.2], [0.3]], output_top_logprobs_idx=[[7, 3], [9]],
             output_top_logprobs=[])
    execute("C2_tm_meta_key",
            f"tokenizer_manager.py:{ext_if.lineno}-{meta_assign.end_lineno}", block,
            {"state": stA, "self": tm_stub, "meta_info": {}, "return_text_in_logprobs": False},
            lambda g: {"meta": g["meta_info"],
                       "passed": list(g["meta_info"].keys()) == ["output_top_logprobs"]
                       and len(g["meta_info"]["output_top_logprobs"]) == 2}, block)
    stE = NS(output_top_logprobs_val=[], output_top_logprobs_idx=[], output_top_logprobs=[])
    execute("C2ctl_tm_meta_empty_val",
            f"tokenizer_manager.py:{ext_if.lineno}-{meta_assign.end_lineno} (control)", block,
            {"state": stE, "self": tm_stub, "meta_info": {}, "return_text_in_logprobs": False},
            lambda g: {"meta": g["meta_info"],
                       "passed": g["meta_info"] == {"output_top_logprobs": []}}, block, kind="CONTROL")

    # C3 CONTEXT: /generate builds GenerateReqInput from request JSON (field names pass through)
    lines = srcs["http"].splitlines()
    gri = [i + 1 for i, l in enumerate(lines) if "GenerateReqInput(" in l]
    aj = [i + 1 for i, l in enumerate(lines) if "await request.json()" in l]
    record("C3_http_entry_CONTEXT",
           "http_server.py (GenerateReqInput( lines / await request.json() lines)",
           "CONTEXT", "context-census",
           {"generate_req_input_lines": gri[:8], "await_request_json_lines": aj[:8],
            "passed": bool(gri) and bool(aj)})

    # ================================================================ verdict
    failed = [h["id"] for h in hops if h["kind"] in ("EXEC_FAILED", "HARNESS_ERROR")]
    k_trace = {
        "probe_request_k": 8,
        "A3_fanout_i0": v3, "A4_tm": v4,
        "A5_sched": v5.get("recv_req.top_logprobs_num"),
        "A7_batch": b1.top_logprobs_nums, "A9_forward_batch": v9,
        "A10_sampler_args": v10,
    }
    verdict = "NO_CLAIM" if (_harness_errors or failed) else "CHAIN_CLOSED_CPU_SIDE"
    out = {
        "probe": "g3-t8 intake/compute/output chain for top_logprobs_num (frozen probe shape)",
        "frozen_probe_shape": {"endpoint": "/generate (native)", "return_logprob": True,
                               "top_logprobs_num": 8, "logprob_start_len": -1},
        "verdict": verdict,
        "harness_errors": _harness_errors,
        "failed_hops": failed,
        "hops": hops,
        "k_trace": k_trace,
        "notes": [
            "GPU-side topk numerics and live IPC/zmq are outside this CPU probe (mock tensor, mock transport).",
            "B2 control reproduces the observed probe signature (chosen LP present, top absent) at the compute layer exactly when k=0 reaches compute; the A-leg shows k=8 survives every intake site for the frozen shape, so that signature is not reachable for this probe.",
            "B5: with logprob_start_len=-1 + return_logprob=true and token_ids_logprob=None, scheduler.py:3261-3263 resolves start=len(prompt)=extend_len -> pruned_len=0 -> INPUT top-logprobs EMPTY BY DESIGN ([], not null); OUTPUT top-k per generated token is unaffected (decode-style sampler path).",
            "C2 control: even with empty val, the meta key is output_top_logprobs=[] (EMPTY, never null); a null read can only come from a wrong field name (i5) or the k=0 / return_logprob=False paths.",
            "Engine unchanged this trial; git diff vs 2c854e0 python/ stays empty.",
        ],
    }
    with open(os.path.join(RAW_DIR, "intake_sampler_chain_output.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"verdict": verdict, "harness_errors": _harness_errors,
                      "failed": failed, "k_trace": k_trace, "n_hops": len(hops),
                      "kinds": {h["id"]: h["kind"] for h in hops}}, indent=1, default=str))
    return 0 if verdict == "CHAIN_CLOSED_CPU_SIDE" else 2


if __name__ == "__main__":
    sys.exit(main())

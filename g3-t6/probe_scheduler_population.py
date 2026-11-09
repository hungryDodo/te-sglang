#!/usr/bin/env python3
"""g3-t6 — Scheduler-side population falsifier for the numeric-coverage blocker.

Question (i5 next_proposal, verbatim): can `BatchStrOutput.output_top_logprobs_val`
be EMPTY AT SOURCE for the frozen probe's exact request shape
(return_logprob=true, top_logprobs_num=8, logprob_start_len absent)?

Method: locate the REAL guard/branch/assignment statements in the scheduler-side
chain by ast content matching (fails loudly if source drifts), then EXECUTE the
pure-Python ones against shape fixtures. torch-dependent compute (sampler topk)
is out of CPU scope and marked ASSUMED_GPU where relevant. Field-copy wiring
hops are ANCHORED (exact source text asserted), not executed.

Chain covered (hop -> file):
  1 request normalize      io_struct.py _normalize_logprob_params + TokenizedGenerateReqInput
  2 start-len resolution   scheduler.py _handle_generate_request branch
  3 batch flags            schedule_batch.py (extend + merge), scheduler.py (decode)
  4 sampler topk guard     logprob_processor.py OutputLogprobProcessor.compute_logprobs
                           + LogprobResult.write_output_to (both pure Python)
  5 extend 1st-token append logprob_result_processor.py add_logprob_return_values
  6 decode append          batch_result_processor.py _apply_decode_logprobs
  7 finish collection      output_streamer.py (slice + else-empty branch)
  8 IPC forwarding         detokenizer_manager.py + output_streamer.py:773 (ANCHORED)

Output: g3-t6/raw/scheduler_population_output.json
"""

import ast
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(exist_ok=True)

FAILURES = []


def src_of(rel):
    return (ROOT / rel).read_text()


def anchor(rel, needle, lo=None, hi=None):
    """Assert exact source text present (optionally within a line window)."""
    text = src_of(rel)
    lines = text.splitlines()
    hits = [i + 1 for i, ln in enumerate(lines) if needle in ln]
    if not hits:
        FAILURES.append(f"ANCHOR MISS {rel}: {needle!r}")
        return None
    if lo is not None and not any(lo <= h <= hi for h in hits):
        FAILURES.append(f"ANCHOR DRIFT {rel}: {needle!r} at {hits}, expected window {lo}-{hi}")
    return hits


def find_if_segment(rel, test_needle, body_needle=None):
    """Return source of the first ast.If whose test contains test_needle
    (and, if given, whose body source contains body_needle)."""
    src = src_of(rel)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            seg = ast.get_source_segment(src, node)
            test_seg = ast.get_source_segment(src, node.test) or ""
            if test_needle in test_seg and (body_needle is None or body_needle in seg):
                return seg
    FAILURES.append(f"IF NOT FOUND {rel}: test~{test_needle!r} body~{body_needle!r}")
    return None


def find_func_segment(rel, name, class_hint=None):
    src = src_of(rel)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if class_hint is None or class_hint in seg:
                return seg
    FAILURES.append(f"FUNC NOT FOUND {rel}: {name}")
    return None


def normalize_segment(seg):
    """get_source_segment strips only the FIRST line's indent; re-anchor the
    rest relative to it so else/elif clauses line up."""
    lines = seg.splitlines()
    if len(lines) == 1:
        return seg
    return lines[0] + "\n" + textwrap.dedent("\n".join(lines[1:]))


def run_segment(seg, ns, label):
    """Exec a statement block (single indent level) inside a wrapper fn."""
    code = "def _run():\n" + textwrap.indent(normalize_segment(seg), "    ")
    try:
        exec(compile(code, f"<{label}>", "exec"), ns)  # noqa: S102
        return ns["_run"](), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def ns(**kw):
    base = {"SimpleNamespace": SimpleNamespace, "len": len, "max": max, "any": any,
            "all": all, "isinstance": isinstance, "list": list}
    base.update(kw)
    return base


IO = "python/sglang/srt/managers/io_struct.py"
SCHED = "python/sglang/srt/managers/scheduler.py"
SB = "python/sglang/srt/managers/schedule_batch.py"
BRP = "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
LRP = "python/sglang/srt/managers/scheduler_components/logprob_result_processor.py"
OSTR = "python/sglang/srt/managers/scheduler_components/output_streamer.py"
DETK = "python/sglang/srt/managers/detokenizer_manager.py"
LPP = "python/sglang/srt/layers/logprob_processor.py"

# --- extract real segments (all anchors asserted before any execution) -------

# hop1a: GenerateReqInput normalization (nested def + the three calls, ast-located)
norm_param = find_func_segment(IO, "normalize_param")
_src_io = src_of(IO)
_io_tree = ast.parse(_src_io)
norm_calls_block = None
for _want, _default in (("self.logprob_start_len", "-1"), ("self.top_logprobs_num", "0")):
    for _node in ast.walk(_io_tree):
        if isinstance(_node, ast.Assign):
            _seg = ast.get_source_segment(_src_io, _node) or ""
            if "normalize_param(" in _seg and _want in _seg and f", {_default}," in _seg:
                norm_calls_block = (norm_calls_block + "\n" + _seg) if norm_calls_block else _seg
                break
    else:
        FAILURES.append(f"NORM CALL MISS io_struct.py: {_want}")
anchor(IO, "self.top_logprobs_num = normalize_param(")

# hop1b: TokenizedGenerateReqInput None-defaults (two sibling ifs)
tok_norm = find_if_segment(IO, "self.top_logprobs_num is None")
tok_norm_sl = find_if_segment(IO, "self.logprob_start_len is None")
anchor(IO, "self.logprob_start_len = -1", 560, 575)

# hop2: scheduler start-len resolution (two sibling ifs) + abort check
sched_if_a = find_if_segment(SCHED, "not recv_req.return_logprob and recv_req.logprob_start_len != -1")
sched_if_b = find_if_segment(SCHED, "recv_req.logprob_start_len == -1", "req.logprob_start_len = len(req.origin_input_ids)")
sched_if_c = find_if_segment(SCHED, "req.logprob_start_len > len(req.origin_input_ids)", "set_finish_with_abort")

# hop3: batch flags
flag_any = "return_logprob = any(req.return_logprob for req in reqs)"
anchor(SB, flag_any)
sb_extend_guard = find_if_segment(SB, "self.return_logprob", "self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]")
sched_decode_guard = find_if_segment(SCHED, "batch.return_logprob", "batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]")

# hop4: sampler-side topk guard + write-back (pure Python parts)
lp_guard = None
_src_lp = src_of(LPP)
for _node in ast.walk(ast.parse(_src_lp)):
    if isinstance(_node, ast.If):
        t = ast.get_source_segment(_src_lp, _node.test) or ""
        if "any(x > 0 for x in top_logprobs_nums)" in t:
            lp_guard = ast.unparse(_node.test)  # single-line, semantically identical
            break
if lp_guard is None:
    FAILURES.append("LP GUARD NOT FOUND logprob_processor.py any(x>0)")
write_output_to = find_func_segment(LPP, "write_output_to")

# hop5: extend first-token append guard
ext_guard = None
for _node in ast.walk(ast.parse(src_of(LRP))):
    if isinstance(_node, ast.If):
        t = ast.get_source_segment(src_of(LRP), _node.test) or ""
        if "req.logprob.top_logprobs_num > 0" in t and "output.next_token_top_logprobs_val is not None" in t:
            ext_guard = ast.unparse(_node.test)  # single-line, semantically identical
            break
if ext_guard is None:
    FAILURES.append("EXT GUARD NOT FOUND logprob_result_processor")

# hop6: decode append (guard + body with flat_idx)
dec_seg = find_if_segment(BRP, "req.logprob.top_logprobs_num > 0", "flat_idx = i * max_accept + j")

# hop7: streamer collection if/else
stream_seg = find_if_segment(OSTR, "req.return_logprob", "self.output_top_logprobs_val.append")

# hop8: IPC forwarding anchors (dataflow, not executed)
anchor(DETK, "output_top_logprobs_val=recv_obj.output_top_logprobs_val", 490, 500)
anchor(DETK, "output_top_logprobs_idx=recv_obj.output_top_logprobs_idx", 490, 500)
anchor(OSTR, "output_top_logprobs_val=self.output_top_logprobs_val", 765, 780)
anchor(SB, "output_top_logprobs_val: Optional[list] = None", 935, 945)

if FAILURES:
    print("EXTRACTION FAILED — source drifted; refusing to execute:")
    for f in FAILURES:
        print(" ", f)
    sys.exit(2)

# --- shape fixtures -----------------------------------------------------------

SHAPES = {
    "S_probe":   dict(return_logprob=True,  top_logprobs_num=8,    logprob_start_len=None),  # frozen probe (C7)
    "S_start0":  dict(return_logprob=True,  top_logprobs_num=8,    logprob_start_len=0),     # i4/i5 fixed shape
    "S_k0":      dict(return_logprob=True,  top_logprobs_num=None, logprob_start_len=None),  # refuted-by-C7 control
    "S_nologprob": dict(return_logprob=False, top_logprobs_num=8,  logprob_start_len=None),  # would lose chosen LP too
}

INPUT_IDS = list(range(100, 112))  # 12-token prompt fixture
GPU_TOPK_ROW = [[-0.12, 13007], [-1.7, 88], [-2.3, 4211], [-3.1, 9], [-4.0, 77],
                [-4.4, 512], [-5.2, 64], [-5.9, 1001]]  # stand-in for torch.topk (k=8)

rows = []

for name, sh in SHAPES.items():
    print("SHAPE:", name, file=sys.stderr)
    st = dict(shape=name, request=sh, steps={})

    # hop1a: exec normalize_param def + the three calls with num=1
    self1 = SimpleNamespace(return_logprob=sh["return_logprob"],
                            logprob_start_len=sh["logprob_start_len"],
                            top_logprobs_num=sh["top_logprobs_num"],
                            parallel_sample_num=1)
    n1 = ns(self=self1, num=1)
    exec(compile(norm_param, "<norm_param>", "exec"), n1)  # define real helper
    _, err1 = run_segment(norm_calls_block, n1, "hop1a")
    st["steps"]["hop1a_req_normalize"] = dict(
        evidence="EXECUTED",
        k=self1.top_logprobs_num, start_len=self1.logprob_start_len,
        return_logprob=self1.return_logprob, err=err1)

    # hop1b: TokenizedGenerateReqInput None-defaults (SINGLE-request path —
    # hop1a is the batch path; single /generate keeps scalars until here)
    tok = SimpleNamespace(top_logprobs_num=sh["top_logprobs_num"],
                          logprob_start_len=sh["logprob_start_len"])
    n1b = ns(self=tok)
    _, err1bs = run_segment(tok_norm_sl, n1b, "hop1b-sl")
    _, err1b = run_segment(tok_norm, n1b, "hop1b-k")
    st["steps"]["hop1b_tok_normalize"] = dict(
        evidence="EXECUTED", k=tok.top_logprobs_num, start_len=tok.logprob_start_len, err=err1b)

    # hop2: scheduler branch (recv_req after tokenizer carries ints)
    recv = SimpleNamespace(return_logprob=sh["return_logprob"],
                           logprob_start_len=tok.logprob_start_len,
                           token_ids_logprob=None)
    req = SimpleNamespace(logprob_start_len=0, origin_input_ids=INPUT_IDS,
                          is_prefill_only=False, aborted=None)
    sch = SimpleNamespace(_add_request_to_queue=lambda r: None)
    req.set_finish_with_abort = lambda m: setattr(req, "aborted", m)
    n2 = ns(recv_req=recv, req=req, self=sch)
    _, errA = run_segment(sched_if_a, n2, "hop2a")
    _, errB = run_segment(sched_if_b, n2, "hop2b")
    _, errC = run_segment(sched_if_c, n2, "hop2c")
    st["steps"]["hop2_start_len_resolution"] = dict(
        evidence="EXECUTED", resolved_start_len=req.logprob_start_len,
        aborted=req.aborted, errs=[errA, errB, errC])

    # hop3: batch flags (single-req ScheduleBatch; k is the scalar the
    # scheduler receives at scheduler.py:2992 recv_req.top_logprobs_num)
    k_req = tok.top_logprobs_num
    r0 = SimpleNamespace(return_logprob=sh["return_logprob"],
                         logprob=SimpleNamespace(top_logprobs_num=k_req, token_ids_logprob=None))
    b = SimpleNamespace(return_logprob=False, top_logprobs_nums=None, token_ids_logprobs=None)
    n3 = ns(reqs=[r0], self=b)
    exec(compile(flag_any, "<flag>", "exec"), n3)
    b.return_logprob = n3["return_logprob"]
    _, err3 = run_segment(sb_extend_guard, n3, "hop3")
    st["steps"]["hop3_batch_flags"] = dict(
        evidence="EXECUTED", batch_return_logprob=b.return_logprob,
        top_logprobs_nums=b.top_logprobs_nums, err=err3)

    # hop4: sampler guard + write-back semantics
    n4 = ns(top_logprobs_nums=b.top_logprobs_nums or [0])
    sampler_wants_topk = eval(lp_guard, n4)  # noqa: S307
    res = SimpleNamespace(token_logprobs=[-0.5], top_logprobs_val=None, top_logprobs_idx=None,
                          token_ids_logprobs_val=None, token_ids_logprobs_idx=None,
                          input_copy_done=None)
    if sampler_wants_topk:
        res.top_logprobs_val = [GPU_TOPK_ROW]
        res.top_logprobs_idx = [GPU_TOPK_ROW]  # shape stand-in; real idx are ids
    lout = SimpleNamespace(next_token_logprobs=None, next_token_top_logprobs_val=None,
                           next_token_top_logprobs_idx=None, next_token_token_ids_logprobs_val=None,
                           next_token_token_ids_logprobs_idx=None,
                           input_token_logprobs=None, input_top_logprobs_val=None,
                           input_top_logprobs_idx=None, input_token_ids_logprobs_val=None,
                           input_token_ids_logprobs_idx=None, input_logprobs_copy_done=None)
    n4b = ns(self=res, logits_output=lout, LogitsProcessorOutput=object)  # annotation-only name
    exec(compile(write_output_to, "<wot>", "exec"), n4b)
    err_wot = None
    try:
        n4b["write_output_to"](res, lout)
    except Exception as e:  # noqa: BLE001
        err_wot = f"{type(e).__name__}: {e}"
    st["steps"]["hop4_sampler_topk"] = dict(
        evidence="EXECUTED(guard+writeback) + ASSUMED_GPU(torch.topk rows are fixtures)",
        guard_any_k_gt0=sampler_wants_topk, err_writeback=err_wot,
        next_token_top_logprobs_val="POPULATED" if lout.next_token_top_logprobs_val is not None else None,
        chosen_next_token_logprobs="POPULATED" if lout.next_token_logprobs is not None else None)

    # hop5: extend first-token append (guarded, silent-skip capable)
    rq5 = SimpleNamespace(logprob=SimpleNamespace(top_logprobs_num=k_req,
                                                  output_top_logprobs_val=[], output_top_logprobs_idx=[]))
    out5 = SimpleNamespace(next_token_top_logprobs_val=lout.next_token_top_logprobs_val,
                           next_token_top_logprobs_idx=lout.next_token_top_logprobs_idx)
    n5 = ns(req=rq5, output=out5, i=0)
    take5 = eval(ext_guard, n5)  # noqa: S307
    if take5:
        rq5.logprob.output_top_logprobs_val.append(out5.next_token_top_logprobs_val[0])
    st["steps"]["hop5_extend_first_token_append"] = dict(
        evidence="EXECUTED(guard)", guard_passed=take5,
        silent_skip_possible=(k_req > 0 and out5.next_token_top_logprobs_val is None),
        req_output_top_logprobs_val_len=len(rq5.logprob.output_top_logprobs_val))

    # hop6: decode append — REAL guard executed (whole if, body included)
    rq6 = SimpleNamespace(logprob=SimpleNamespace(top_logprobs_num=k_req,
                                                  output_top_logprobs_val=[], output_top_logprobs_idx=[],
                                                  output_token_logprobs_val=[], output_token_logprobs_idx=[],
                                                  output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[]))
    out6 = SimpleNamespace(next_token_top_logprobs_val=lout.next_token_top_logprobs_val,
                           next_token_top_logprobs_idx=lout.next_token_top_logprobs_idx,
                           next_token_token_ids_logprobs_val=None,
                           next_token_token_ids_logprobs_idx=None)
    n6 = ns(req=rq6, logits_output=out6, i=0, j=0, max_accept=1, next_token_id=[7])
    _, err6 = run_segment(dec_seg, n6, "hop6")
    outcome6 = ("appended" if len(rq6.logprob.output_top_logprobs_val) == 1 else "no-append(guard false)")
    # counterfactual: force val=None with k>0 (unreachable via hop4, but shows the
    # failure mode is a CRASH, not a silent null)
    cf6 = ns(req=SimpleNamespace(logprob=SimpleNamespace(top_logprobs_num=8,
                                                         output_top_logprobs_val=[], output_top_logprobs_idx=[],
                                                         output_token_logprobs_val=[], output_token_logprobs_idx=[],
                                                         output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[])),
             logits_output=SimpleNamespace(next_token_top_logprobs_val=None, next_token_top_logprobs_idx=None),
             i=0, j=0, max_accept=1, next_token_id=[7])
    _, cf_err6 = run_segment(dec_seg, cf6, "hop6cf")
    st["steps"]["hop6_decode_append"] = dict(
        evidence="EXECUTED(if+body, real guard)",
        outcome=outcome6, err=err6,
        counterfactual_val_none_k8=("CRASH (TypeError on None index)" if cf_err6 and "TypeError" in cf_err6 else cf_err6))

    # hop7: streamer collection (non-stream finish path)
    st_req = SimpleNamespace(return_logprob=sh["return_logprob"], is_retracted=False,
                             send_output_token_logprobs_offset=0,
                             logprob=SimpleNamespace(
                                 output_token_logprobs_val=[-0.5, -0.7], output_token_logprobs_idx=[7, 8],
                                 output_top_logprobs_val=[GPU_TOPK_ROW, GPU_TOPK_ROW],
                                 output_top_logprobs_idx=[GPU_TOPK_ROW, GPU_TOPK_ROW],
                                 output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[]))
    stm = SimpleNamespace(output_token_logprobs_val=[], output_token_logprobs_idx=[],
                          output_top_logprobs_val=[], output_top_logprobs_idx=[],
                          output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[],
                          return_sampling_mask=False)
    n7 = ns(req=st_req, self=stm, output_ids_=[7, 8])
    _, err7 = run_segment(stream_seg, n7, "hop7")
    st["steps"]["hop7_streamer_collection"] = dict(
        evidence="EXECUTED", forwarded_rows=len(stm.output_top_logprobs_val[0]) if stm.output_top_logprobs_val else 0,
        err=err7)

    # hop8: anchors only
    st["steps"]["hop8_ipc_forwarding"] = dict(
        evidence="ANCHORED",
        detokenizer_copies="output_top_logprobs_val=recv_obj.output_top_logprobs_val (+idx) at detokenizer_manager.py:497-498",
        streamer_pass="output_top_logprobs_val=self.output_top_logprobs_val at output_streamer.py:773")

    # verdict
    final_rows = stm.output_top_logprobs_val[0] if stm.output_top_logprobs_val else []
    harness_errs = [e for e in (err1, err1b, err1bs, *filter(None, (errA, errB, errC)), err3, err_wot, err6, err7) if e]
    if k_req == 0:
        verdict = "EMPTY_AT_SOURCE_BY_REQUEST (k=0; also matches observed null signature)"
    elif not sh["return_logprob"]:
        verdict = "EMPTY_AT_SOURCE (return_logprob=False; but then chosen LP would be absent too — contradicts probe observation)"
    elif harness_errs or not final_rows:
        verdict = "HARNESS_ERROR — no claim (see steps errs)"
    else:
        verdict = ("POPULATED_AT_SOURCE (k>0 flows through every executed guard incl. streamer; "
                   "only GPU compute or the response read path can remove it)")
    st["verdict"] = verdict
    rows.append(st)

# --- discrimination table ------------------------------------------------------
disc = [
    dict(hypothesis="k=0 at request normalize",
         signature="chosen LP present, top null",
         status="EXCLUDED by C7 (frozen probe sends top_logprobs_num=8); reproduced here as S_k0"),
    dict(hypothesis="scheduler-side silent empty population",
         signature="chosen LP present, top null",
         status="EXCLUDED for probe shape by this trial: only POPULATED or CRASH(TypeError) reachable; the one silent-skip site (hop5) requires next_token_top_logprobs_val=None which hop4 cannot produce when any k>0"),
    dict(hypothesis="response read-path field mismatch (i5)",
         signature="chosen LP present, top null",
         status="STANDING — consistent with all executed evidence; probe read names are structurally absent on the native surface"),
    dict(hypothesis="logprob_start_len gating output top-k",
         signature="n/a",
         status="EXCLUDED by hop2 execution: start_len only scopes INPUT logprobs; output containers/appends are independent of it"),
]

out = dict(
    trial="g3-t6", worker="glm-b",
    question="Can BatchStrOutput.output_top_logprobs_val be empty at source for the frozen probe shape?",
    engine_tree="python/ identical to 2c854e0 (verified separately by git diff)",
    chain_files=[IO, SCHED, SB, BRP, LRP, OSTR, DETK, LPP],
    extraction_failures=FAILURES,
    shapes=rows,
    discrimination=disc,
    scope_limits=[
        "torch.topk rows in hop4/hop7 are fixtures (ASSUMED_GPU); the guard and write-back semantics are executed real source",
        "hop8 is anchored dataflow (IPC copies), not executed",
        "single-request batch; mixed-batch k-lists exercise the same any() guard",
        "speculative-decoding paths not exercised (frozen workload is non-spec)",
        "CPU-side evidence only; no GPU, no live server, no HTTP",
    ],
)
path = RAW / "scheduler_population_output.json"
path.write_text(json.dumps(out, indent=2))
print(f"wrote {path}")
for r in rows:
    print(f"  {r['shape']:12s} -> {r['verdict']}")

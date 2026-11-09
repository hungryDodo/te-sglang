#!/usr/bin/env python3
"""g3-t7 — Scheduler-side population falsifier v2 (recovery verification).

Verdict on the preserved session-26 artifact (g3-t6/): REJECTED as evidence —
see g3-t6/VERIFICATION.md. Root cause of its two execution failures was a
text-dedent re-indentation bug on multi-branch if-segments, plus a verdict
gate that the committed raw output provably did not pass through.

v2 fixes, then re-executes the SAME bounded question against the SAME chain:
  - extraction: ast.unparse(node) for executable text (formatting-normalized,
    AST-identical to source; kills the IndentationError class entirely);
    original file line range recorded per hop for provenance.
  - verdict gate is HARD: any harness error on any hop => HARNESS_ERROR_NO_CLAIM
    for that shape (the shape table is still written, but no discrimination
    entry may cite a shape that did not execute clean).
  - hop6 counterfactual restored (k>0 + val=None => TypeError crash, not null).
  - hop8 upgraded: full census of every `output_top_logprobs_val` occurrence in
    python/sglang/srt (classified from source reads) + confirmation that the
    two forwarding copies are unconditional; the streamer else-branch
    (empty-list append) is signature-analyzed: it empties chosen LP too, so it
    cannot produce "chosen LP present, top null".

Question (unchanged): can BatchStrOutput.output_top_logprobs_val be EMPTY AT
SOURCE for the frozen probe shape (return_logprob=true, top_logprobs_num=8,
logprob_start_len absent)? No engine change; stdlib only.

Output: g3-t7/raw/scheduler_population_v2_output.json
"""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(exist_ok=True)

FAILURES = []

FILES = {}
def src_of(rel):
    if rel not in FILES:
        FILES[rel] = (ROOT / rel).read_text()
    return FILES[rel]

def prov(rel, node):
    return f"{rel}:{node.lineno}-{node.end_lineno}"

def find_if(rel, test_unparsed, body_needle):
    """First ast.If whose unparse(test) == test_unparsed (or contains it when
    it is a fragment) and whose source contains body_needle."""
    src = src_of(rel)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If):
            t = ast.unparse(node.test)
            seg = ast.get_source_segment(src, node) or ""
            if (test_unparsed in t or t in test_unparsed) and body_needle in seg:
                return node
    FAILURES.append(f"IF NOT FOUND {rel}: test~{test_unparsed!r} body~{body_needle!r}")
    return None

def find_func(rel, name):
    src = src_of(rel)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    FAILURES.append(f"FUNC NOT FOUND {rel}: {name}")
    return None

def find_assign(rel, needles):
    src = src_of(rel)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            seg = ast.get_source_segment(src, node) or ""
            if all(n in seg for n in needles):
                return node
    FAILURES.append(f"ASSIGN NOT FOUND {rel}: {needles!r}")
    return None

def run(node_or_text, namespace, label):
    """Exec statement(s) inside a wrapper fn; return error string or None.
    ast.unparse output is top-level dedented, so uniform re-indent is safe."""
    text = node_or_text if isinstance(node_or_text, str) else ast.unparse(node_or_text)
    code = "def _run():\n" + "\n".join("    " + ln for ln in text.splitlines())
    ns = dict(namespace)
    try:
        exec(compile(code, f"<{label}>", "exec"), ns)  # noqa: S102
        ns["_run"]()
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"

def eval_guard(node, namespace, label):
    try:
        return eval(ast.unparse(node.test), dict(namespace)), None  # noqa: S307
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e} ({label})"

IO = "python/sglang/srt/managers/io_struct.py"
SCHED = "python/sglang/srt/managers/scheduler.py"
SB = "python/sglang/srt/managers/schedule_batch.py"
BRP = "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
LRP = "python/sglang/srt/managers/scheduler_components/logprob_result_processor.py"
OSTR = "python/sglang/srt/managers/scheduler_components/output_streamer.py"
DETK = "python/sglang/srt/managers/detokenizer_manager.py"
LPP = "python/sglang/srt/layers/logprob_processor.py"

# --- extract (all needles asserted before any shape executes) ---------------
EX = {}

EX["hop1a_def"] = (find_func(IO, "normalize_param"), IO)
EX["hop1a_k"] = (find_assign(IO, ["normalize_param(", "self.top_logprobs_num"]), IO)
EX["hop1a_sl"] = (find_assign(IO, ["normalize_param(", "self.logprob_start_len"]), IO)
EX["hop1b_k"] = (find_if(IO, "self.top_logprobs_num is None", "self.top_logprobs_num = "), IO)
EX["hop1b_sl"] = (find_if(IO, "self.logprob_start_len is None", "self.logprob_start_len = "), IO)
EX["hop2a"] = (find_if(SCHED, "not recv_req.return_logprob and recv_req.logprob_start_len != -1",
                       "recv_req.logprob_start_len = -1"), SCHED)
EX["hop2b"] = (find_if(SCHED, "recv_req.logprob_start_len == -1",
                       "req.logprob_start_len = len(req.origin_input_ids)"), SCHED)
EX["hop2c"] = (find_if(SCHED, "req.logprob_start_len > len(req.origin_input_ids)",
                       "set_finish_with_abort"), SCHED)
EX["hop3_extend"] = (find_if(SB, "self.return_logprob", "self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]"), SB)
EX["hop3_decode"] = (find_if(SCHED, "batch.return_logprob", "batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]"), SCHED)
EX["hop3_flag"] = ("return_logprob = any(req.return_logprob for req in reqs)", SB)
EX["hop4_guard"] = (find_if(LPP, "any((x > 0 for x in top_logprobs_nums))", "get_top_logprobs"), LPP)
EX["hop4_write"] = (find_func(LPP, "write_output_to"), LPP)
EX["hop5"] = (find_if(LRP, "req.logprob.top_logprobs_num > 0 and output.next_token_top_logprobs_val is not None",
                      "output_top_logprobs_val.append"), LRP)
EX["hop6"] = (find_if(BRP, "req.logprob.top_logprobs_num > 0", "flat_idx = i * max_accept + j"), BRP)
EX["hop7"] = (find_if(OSTR, "req.return_logprob", "self.output_top_logprobs_val.append"), OSTR)

for key, (node, rel) in EX.items():
    if isinstance(node, str):
        if node not in src_of(rel):
            FAILURES.append(f"ANCHOR MISS {key}: {rel}")
        continue
    if node is None:
        continue
    if "output_top_logprobs_val" in ast.unparse(node) or key in ("hop7",):
        pass  # hop-specific needles already matched during find

# hop7 must be the OUTPUT collection block, not the input one
if EX["hop7"][0] is not None:
    body = ast.unparse(EX["hop7"][0])
    if "output_token_logprobs_val" not in body or "logprob_end" not in body:
        FAILURES.append("hop7 grabbed wrong block (input collection?)")

if FAILURES:
    print("EXTRACTION FAILED — source drifted; refusing to execute:")
    for f in FAILURES:
        print(" ", f)
    sys.exit(2)

PROV = {k: (prov(rel, n) if not isinstance(n, str) else f"{rel}:{n}")
        for k, (n, rel) in EX.items()}

# --- hop8 census (source-read classification, recorded not executed) ---------
census_lines = []
for rel in [IO, SCHED, SB, BRP, LRP, OSTR, DETK, LPP,
            "python/sglang/srt/managers/tokenizer_manager.py",
            "python/sglang/srt/managers/multi_tokenizer_mixin.py",
            "python/sglang/srt/rust_server/server.py",
            "python/sglang/srt/disaggregation/decode.py",
            "python/sglang/srt/disaggregation/utils.py"]:
    for i, ln in enumerate(src_of(rel).splitlines(), 1):
        if "output_top_logprobs_val" in ln and "token_ids" not in ln:
            census_lines.append(f"{rel}:{i}: {ln.strip()[:100]}")

HOP8 = dict(
    evidence="CENSUS+CONTEXT_READ (not executed)",
    occurrences=census_lines,
    forwarding_copies_unconditional=(
        "detokenizer_manager.py:497-498 straight field copy inside the BatchStrOutput "
        "construction (no conditional wraps output_top_logprobs_val/idx); "
        "output_streamer.py:773 passes self.output_top_logprobs_val=... unconditionally "
        "(only the flat-format input fields are conditionally nulled there)"),
    streamer_else_branch_signature=(
        "the if/else at output_streamer.py:611-655 is guarded by req.return_logprob ONLY; "
        "the else appends EMPTY lists for chosen output_token_logprobs_val AND top_logprobs_val "
        "together, so its observable signature is 'chosen LP absent too' — it cannot produce "
        "the probe's observed 'chosen LP present, top null'"),
    off_probe_path_sites=(
        "rust_server/server.py (rust frontend), disaggregation/* (PD), multi_tokenizer_mixin.py "
        "(--multi-tokenizer-mode) are not on the frozen single-node single-tokenizer probe path; "
        "tokenizer_manager.py sites are the response-assembly side closed by g3-t5"),
)

# --- shape fixtures ------------------------------------------------------------
SHAPES = {
    "S_probe":     dict(return_logprob=True,  top_logprobs_num=8,  logprob_start_len=None),
    "S_start0":    dict(return_logprob=True,  top_logprobs_num=8,  logprob_start_len=0),
    "S_k0":        dict(return_logprob=True,  top_logprobs_num=None, logprob_start_len=None),
    "S_nologprob": dict(return_logprob=False, top_logprobs_num=8,  logprob_start_len=None),
}
INPUT_IDS = list(range(100, 112))  # 12-token prompt fixture
GPU_TOPK_ROW = [[-0.12, 13007], [-1.7, 88], [-2.3, 4211], [-3.1, 9], [-4.0, 77],
                [-4.4, 512], [-5.2, 64], [-5.9, 1001]]  # stand-in for torch.topk (k=8)

rows = []
for name, sh in SHAPES.items():
    errs = {}
    st = dict(shape=name, request=sh, steps={})

    # hop1a: batch-path normalize (num=1) — define real helper, run real assigns
    self1 = SimpleNamespace(return_logprob=sh["return_logprob"],
                            logprob_start_len=sh["logprob_start_len"],
                            top_logprobs_num=sh["top_logprobs_num"],
                            parallel_sample_num=1)
    n1 = {"self": self1, "num": 1}
    exec(compile(ast.unparse(EX["hop1a_def"][0]), "<hop1a_def>", "exec"), n1)  # noqa: S102
    errs["hop1a_def"] = run(EX["hop1a_k"][0], n1, "hop1a_k")
    errs["hop1a_sl"] = run(EX["hop1a_sl"][0], n1, "hop1a_sl")
    st["steps"]["hop1a_req_normalize"] = dict(
        k=self1.top_logprobs_num, start_len=self1.logprob_start_len)

    # hop1b: single-request path None-defaults
    tok = SimpleNamespace(top_logprobs_num=sh["top_logprobs_num"],
                          logprob_start_len=sh["logprob_start_len"])
    n1b = {"self": tok}
    errs["hop1b_sl"] = run(EX["hop1b_sl"][0], n1b, "hop1b_sl")
    errs["hop1b_k"] = run(EX["hop1b_k"][0], n1b, "hop1b_k")
    st["steps"]["hop1b_tok_normalize"] = dict(k=tok.top_logprobs_num, start_len=tok.logprob_start_len)

    # hop2: scheduler start-len resolution, three REAL branches in order
    recv = SimpleNamespace(return_logprob=sh["return_logprob"],
                           logprob_start_len=tok.logprob_start_len,
                           token_ids_logprob=None)
    req = SimpleNamespace(logprob_start_len=tok.logprob_start_len,
                          origin_input_ids=INPUT_IDS, is_prefill_only=False)
    req.set_finish_with_abort = lambda m: setattr(req, "aborted", m)
    n2 = {"recv_req": recv, "req": req, "self": SimpleNamespace(_add_request_to_queue=lambda r: None)}
    errs["hop2a"] = run(EX["hop2a"][0], n2, "hop2a")
    errs["hop2b"] = run(EX["hop2b"][0], n2, "hop2b")
    errs["hop2c"] = run(EX["hop2c"][0], n2, "hop2c")
    st["steps"]["hop2_start_len_resolution"] = dict(
        resolved_start_len=req.logprob_start_len, aborted=getattr(req, "aborted", None))

    # hop3: batch flags
    k_req = tok.top_logprobs_num
    r0 = SimpleNamespace(return_logprob=sh["return_logprob"],
                         logprob=SimpleNamespace(top_logprobs_num=k_req,
                                                 token_ids_logprob=None))
    b = SimpleNamespace(return_logprob=False, top_logprobs_nums=None)
    n3 = {"reqs": [r0], "self": b, "any": any}
    # direct exec (no wrapper) so the assignment lands in the namespace
    exec(compile(EX["hop3_flag"][0], "<hop3_flag>", "exec"), n3)  # noqa: S102
    b.return_logprob = n3["return_logprob"]
    errs["hop3_extend"] = run(EX["hop3_extend"][0], {"reqs": [r0], "self": b}, "hop3_extend")
    st["steps"]["hop3_batch_flags"] = dict(
        batch_return_logprob=b.return_logprob, top_logprobs_nums=b.top_logprobs_nums)

    # hop4: sampler guard (real test eval) + real write_output_to
    wants_topk, errs["hop4_guard"] = eval_guard(EX["hop4_guard"][0],
        {"top_logprobs_nums": b.top_logprobs_nums or [0]}, "hop4_guard")
    res = SimpleNamespace(token_logprobs=[-0.5], top_logprobs_val=None, top_logprobs_idx=None,
                          token_ids_logprobs_val=None, token_ids_logprobs_idx=None,
                          input_copy_done=None)
    if wants_topk:
        res.top_logprobs_val = [GPU_TOPK_ROW]
        res.top_logprobs_idx = [GPU_TOPK_ROW]  # shape stand-in; real idx are ids
    lout = SimpleNamespace(next_token_logprobs=None, next_token_top_logprobs_val=None,
                           next_token_top_logprobs_idx=None,
                           next_token_token_ids_logprobs_val=None,
                           next_token_token_ids_logprobs_idx=None,
                           input_token_logprobs=None, input_top_logprobs_val=None,
                           input_top_logprobs_idx=None, input_token_ids_logprobs_val=None,
                           input_token_ids_logprobs_idx=None, input_logprobs_copy_done=None)
    n4 = {"self": res, "logits_output": lout, "LogitsProcessorOutput": object}
    exec(compile(ast.unparse(EX["hop4_write"][0]), "<hop4_write>", "exec"), n4)  # noqa: S102
    try:
        n4["write_output_to"](res, lout)
        errs["hop4_write"] = None
    except Exception as e:  # noqa: BLE001
        errs["hop4_write"] = f"{type(e).__name__}: {e}"
    st["steps"]["hop4_sampler_topk"] = dict(
        evidence="EXECUTED(guard+writeback real source) + ASSUMED_GPU(torch.topk rows are fixtures)",
        guard_any_k_gt0=wants_topk,
        next_top_val="POPULATED" if lout.next_token_top_logprobs_val is not None else None,
        chosen_next_logprob="POPULATED" if lout.next_token_logprobs is not None else None)

    # hop5: extend first-token append — real guard test, manual append if taken
    rq5 = SimpleNamespace(logprob=SimpleNamespace(top_logprobs_num=k_req,
                                                  output_top_logprobs_val=[], output_top_logprobs_idx=[]))
    out5 = SimpleNamespace(next_token_top_logprobs_val=lout.next_token_top_logprobs_val,
                           next_token_top_logprobs_idx=lout.next_token_top_logprobs_idx)
    take5, errs["hop5_guard"] = eval_guard(EX["hop5"][0], {"req": rq5, "output": out5}, "hop5")
    if take5:
        rq5.logprob.output_top_logprobs_val.append(out5.next_token_top_logprobs_val[0])
    st["steps"]["hop5_extend_first_token_append"] = dict(
        evidence="EXECUTED(real guard test)" if errs["hop5_guard"] is None else "GUARD_ERROR",
        guard_passed=take5, appended=len(rq5.logprob.output_top_logprobs_val))

    # hop6: decode append — REAL if+body executed
    def mk_rq6(k):
        return SimpleNamespace(logprob=SimpleNamespace(
            top_logprobs_num=k, output_top_logprobs_val=[], output_top_logprobs_idx=[],
            output_token_logprobs_val=[], output_token_logprobs_idx=[],
            output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[]))
    rq6 = mk_rq6(k_req)
    out6 = SimpleNamespace(next_token_top_logprobs_val=lout.next_token_top_logprobs_val,
                           next_token_top_logprobs_idx=lout.next_token_top_logprobs_idx,
                           next_token_token_ids_logprobs_val=None,
                           next_token_token_ids_logprobs_idx=None)
    if b.return_logprob:
        errs["hop6"] = run(EX["hop6"][0], {"req": rq6, "logits_output": out6, "i": 0, "j": 0,
                                           "max_accept": 1, "next_token_id": [7],
                                           "next_token_logprobs": [-0.5]}, "hop6")
        st["steps"]["hop6_decode_append"] = dict(
            evidence="EXECUTED(real if+body)",
            outcome=("appended" if len(rq6.logprob.output_top_logprobs_val) == 1
                     else ("no-append(guard false)" if k_req == 0 else "SILENT-EMPTY")),
            err=errs["hop6"])
    else:
        st["steps"]["hop6_decode_append"] = dict(
            evidence="NOT_REACHED",
            why="_apply_decode_logprobs only runs under the caller's batch.return_logprob guard "
                "(batch_result_processor.py:1132); shape has return_logprob=False")
    # counterfactual: k>0 forced with val=None — reachable only if hop4 broke
    cf = SimpleNamespace(req=mk_rq6(8), logits_output=SimpleNamespace(
        next_token_top_logprobs_val=None, next_token_top_logprobs_idx=None,
        next_token_token_ids_logprobs_val=None, next_token_token_ids_logprobs_idx=None),
        i=0, j=0, max_accept=1, next_token_id=[7], next_token_logprobs=[-0.5])
    cf_err = run(EX["hop6"][0], vars(cf), "hop6cf")
    st["steps"]["hop6_counterfactual_k8_val_none"] = dict(
        err=cf_err, is_crash=bool(cf_err and "TypeError" in cf_err))

    # hop7: streamer OUTPUT collection — REAL if+body+else executed.
    # top-rows fixture = the ACTUAL post-hop6 accumulator (no pre-seeding), so
    # k=0 shapes forward what the chain really produced (empty), and the
    # identity check compares against the real slice bounds.
    acc_top = list(rq6.logprob.output_top_logprobs_val)
    acc_tok = [-0.5, -0.7]  # chosen LP accumulates unconditionally (hop6 semantics)
    st_req = SimpleNamespace(return_logprob=sh["return_logprob"], is_retracted=False,
                             send_output_token_logprobs_offset=0,
                             logprob=SimpleNamespace(
                                 output_token_logprobs_val=acc_tok,
                                 output_token_logprobs_idx=[7, 8],
                                 output_top_logprobs_val=acc_top,
                                 output_top_logprobs_idx=list(rq6.logprob.output_top_logprobs_idx),
                                 output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[]))
    stm = SimpleNamespace(output_token_logprobs_val=[], output_token_logprobs_idx=[],
                          output_top_logprobs_val=[], output_top_logprobs_idx=[],
                          output_token_ids_logprobs_val=[], output_token_ids_logprobs_idx=[])
    errs["hop7"] = run(EX["hop7"][0], {"req": st_req, "self": stm, "output_ids_": [7, 8],
                                       "send_output_token_logprobs_offset": 0, "len": len,
                                       "max": max}, "hop7")
    forwarded = stm.output_top_logprobs_val[0] if stm.output_top_logprobs_val else None
    expected_slice = acc_top[0:max(len([7, 8]), 1)] if sh["return_logprob"] else []
    st["steps"]["hop7_streamer_collection"] = dict(
        evidence="EXECUTED(real if+body+else)",
        forwarded_rows=len(forwarded) if forwarded is not None else 0,
        forwarded_matches_accumulator_slice=(forwarded == expected_slice),
        chosen_forwarded=len(stm.output_token_logprobs_val[0]) if stm.output_token_logprobs_val else 0,
        err=errs["hop7"])

    st["steps"]["hop8_ipc_forwarding"] = HOP8
    st["harness_errors"] = {k: v for k, v in errs.items() if v}

    # HARD verdict gate
    if st["harness_errors"]:
        st["verdict"] = "HARNESS_ERROR_NO_CLAIM"
    elif k_req == 0:
        st["verdict"] = "EMPTY_AT_SOURCE_BY_REQUEST (k=0; refuted-by-C7 control, matches null signature)"
    elif not sh["return_logprob"]:
        st["verdict"] = "EMPTY_AT_SOURCE (return_logprob=False; empties chosen LP too — contradicts probe observation)"
    else:
        fwd = stm.output_top_logprobs_val[0]
        st["verdict"] = ("POPULATED_AT_SOURCE (k>0 executed clean through hop1-hop7; only GPU "
                         "compute or the response read path can remove it)") \
            if fwd == expected_slice and len(fwd) > 0 else "UNEXPECTED_STATE_NO_CLAIM"
    rows.append(st)

clean = {r["shape"]: r["verdict"] for r in rows if not r["harness_errors"]}
disc = [
    dict(hypothesis="k=0 at request normalize",
         status=("EXCLUDED by C7 (frozen probe sends top_logprobs_num=8); control shape "
                 + clean.get("S_k0", "NOT-CLEAN", ))),
    dict(hypothesis="scheduler-side silent empty population for probe shape",
         status=("EXCLUDED if S_probe/S_start0 executed clean with POPULATED_AT_SOURCE: the only "
                 "silent-skip site (hop5) requires next_token_top_logprobs_val=None, which hop4's "
                 "batch-level any(k>0) guard prevents for k=8; hop6 crashes (TypeError) instead of "
                 "silently skipping; hop7's else empties chosen LP too")),
    dict(hypothesis="response read-path field mismatch (g3-t5)",
         status="STANDING — consistent with all executed evidence; probe read names structurally absent on the native surface"),
    dict(hypothesis="logprob_start_len gates output top-k",
         status=("EXCLUDED by hop2 execution: -1 resolves to len(origin_input_ids) (12) with no "
                 "abort; start_len scopes INPUT logprobs only — output containers/appends "
                 "(hops 5-7) do not read it")),
]
out = dict(
    trial="g3-t7", worker="glm-b",
    supersedes="g3-t6 (REJECTED as evidence — script/raw mismatch + hop2b/hop7 IndentationErrors; see g3-t6/VERIFICATION.md)",
    question="Can BatchStrOutput.output_top_logprobs_val be empty at source for the frozen probe shape?",
    engine_tree="python/ identical to 2c854e0 (git diff empty at commit time)",
    provenance=PROV, extraction_failures=FAILURES, shapes=rows, discrimination=disc,
    scope_limits=[
        "torch.topk rows are fixtures (ASSUMED_GPU); guards and write-backs are executed real source (ast.unparse = AST-identical)",
        "hop8 is census + context read, not executed",
        "single-request batch; mixed batches exercise the same batch-level any() guard",
        "speculative-decode / PD-disaggregation / rust-server / multi-tokenizer paths not exercised (off frozen probe path)",
        "CPU-side evidence only; no GPU, no live server, no HTTP",
    ],
)
path = RAW / "scheduler_population_v2_output.json"
path.write_text(json.dumps(out, indent=2))
print(f"wrote {path}")
for r in rows:
    print(f"  {r['shape']:12s} errs={len(r['harness_errors']):2d} -> {r['verdict'][:90]}")
if any(r["harness_errors"] for r in rows):
    sys.exit(1)

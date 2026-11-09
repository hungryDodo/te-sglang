#!/usr/bin/env python3
"""G3-t16: does `--cuda-graph-max-bs-decode 64` admit the observed C128 graph replay?

Two independent halves, hard-gated:

  A. TRACE (Controller packet, read-only): parse the real BF16 C128 decode trace
     and measure (i) every DECODE step span's host batch label, (ii) the
     cudaGraphLaunch count and its per-span placement, (iii) per-forward GPU
     geometry (attention launch count and grid, lm_head grid), and (iv) the
     byte-accounting bandwidth implied by the engine's own bs label vs the
     alternative bs=64 (a negative control: bs=64 must be numerically possible,
     so the test is not rigged by construction).

  B. SOURCE (this candidate tree, HEAD): AST-extract and execute the real
     `generate_decode_cuda_graph_batch_sizes` body (import shim only), then AST-
     assert the three consumers that turn its output into the replay predicate:
       memory_hook.py: bs materialized from decode.max_bs
       base_cuda_graph_runner.py: capture_bs = decode.bs ; max_bs = max(capture_bs)
       decode_cuda_graph_runner.py: is_bs_supported = cuda_graph_bs <= max_bs
     Sensitivity control: max_bs=256 must include 128 in the capture list.

Verdict is printed and written as JSON. No GPU, no engine import, no network.
"""
import argparse
import ast
import gzip
import hashlib
import json
import math
import statistics
import sys
from types import SimpleNamespace

TRACE_SHA = "09e8dcb9352c1809e05b10b1673420c543a12c5ebccdc57fc864c3ad4372e4b1"

# Spark2_5Config (python/sglang/srt/configs/spark2_5.py): defaults, then
# overridable by the checkpoint; the trace's kernel signatures pin them.
NUM_LAYERS = 28
NUM_KV_HEADS = 2
HEAD_DIM = 256
BYTES_PER_KV_ELEM = 2  # bfloat16
PEAK_GBS = 273.0  # GB10 LPDDR5X theoretical (cited, not measured here)
SEQ_LO, SEQ_HI = 513, 576  # prompt 512 + 1..64 generated tokens


def die(msg):
    print("HARNESS_ERROR:", msg)
    sys.exit(2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- A. trace
def analyze_trace(path):
    got = sha256(path)
    if got != TRACE_SHA:
        die(f"trace sha mismatch {got} != {TRACE_SHA}")
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"]

    kernels = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    steps = sorted(
        [e for e in ev if e.get("cat") == "user_annotation"
         and e["name"].startswith("step[DECODE")],
        key=lambda e: e["ts"],
    )
    launches = [e for e in ev if e.get("cat") == "cuda_runtime"
                and e["name"] == "cudaGraphLaunch"]
    if len(steps) != 16:
        die(f"expected 16 DECODE step spans, got {len(steps)}")
    if len(launches) != 16:
        die(f"expected 16 cudaGraphLaunch, got {len(launches)}")

    # one launch per span, and no span with 0 or >1
    per_span = []
    for s in steps:
        lo, hi = s["ts"], s["ts"] + s["dur"]
        inside = [e for e in launches if lo <= e["ts"] <= hi]
        per_span.append(len(inside))
    if per_span != [1] * len(steps):
        die(f"launch-per-span not all 1: {per_span}")

    labels = sorted({s["name"] for s in steps})
    if labels != ["step[DECODE bs=128]"]:
        die(f"unexpected step labels {labels}")

    # attention geometry: 28 layers * 16 forwards, one launch per layer-forward
    att = [e for e in kernels if "BatchPrefillWithPagedKVCacheKernel" in e["name"]]
    if len(att) != NUM_LAYERS * 16:
        die(f"attention kernels {len(att)} != {NUM_LAYERS*16}")
    att_grids = sorted({tuple(e["args"]["grid"]) for e in att})
    att_us = statistics.mean(e["dur"] for e in att)
    att_per_forward_ms = sum(e["dur"] for e in att) / 16.0 / 1000.0

    # lm_head GEMM: t15 identified it by the 208-wide vocab tile (133120/208=640)
    lm = [e for e in kernels if "128x208x64_2_32x104x64" in e["name"]]
    if len(lm) != 16:
        die(f"lm_head kernels {len(lm)} != 16 (one per forward)")
    lm_grids = sorted({tuple(e["args"]["grid"]) for e in lm})

    # Byte accounting: attention KV traffic per forward is the only class that
    # scales with batch. Compare engine label (M=128) against the alternative
    # M=64 that the 64-cap hypothesis would require. M=256 is a second control.
    kv_bytes_per_tok_layer = 2 * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_KV_ELEM
    bw = {}
    for m in (64, 128, 256):
        lo = m * SEQ_LO * NUM_LAYERS * kv_bytes_per_tok_layer / att_per_forward_ms / 1e6
        hi = m * SEQ_HI * NUM_LAYERS * kv_bytes_per_tok_layer / att_per_forward_ms / 1e6
        bw[f"M={m}"] = {"gbs_lo": round(lo, 1), "gbs_hi": round(hi, 1),
                        "feasible_under_273": hi <= PEAK_GBS,
                        "pct_of_peak_lo": round(100 * lo / PEAK_GBS, 1)}
    return {
        "trace_sha256": got,
        "step_spans": len(steps),
        "step_labels": labels,
        "cuda_graph_launches": len(launches),
        "launches_inside_each_step_span": per_span,
        "attention_launches": len(att),
        "attention_launches_per_forward": len(att) // 16,
        "attention_grids": [list(g) for g in att_grids],
        "attention_mean_us": round(att_us, 2),
        "attention_ms_per_forward": round(att_per_forward_ms, 3),
        "lm_head_launches": len(lm),
        "lm_head_grids": [list(g) for g in lm_grids],
        "kv_bytes_per_token_per_layer": kv_bytes_per_tok_layer,
        "implied_attention_bandwidth": bw,
    }


# --------------------------------------------------------------- B. source
def extract_function(path, name):
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node), src, tree
    die(f"{name} not found in {path}")


def exec_generator(cuda_graph_hook_path, max_bs, disable_padding=False, spec=False):
    seg, _, _ = extract_function(cuda_graph_hook_path,
                                 "generate_decode_cuda_graph_batch_sizes")
    from typing import Any
    ns = {"Any": Any, "resolving_view": lambda sa: SimpleNamespace(
        disable_cuda_graph_padding=disable_padding,
        speculative_algorithm=("ngram" if spec else None))}
    exec(compile(seg, cuda_graph_hook_path, "exec"), ns)
    return ns["generate_decode_cuda_graph_batch_sizes"](object(), max_bs)


def find_assign_expr(tree, filename, needle):
    """Return source segments of Assign nodes whose source contains needle."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            try:
                text = ast.unparse(node)
            except Exception:
                continue
            if needle in text:
                out.append(text)
    return out


def analyze_source(repo):
    import os
    hook = os.path.join(repo, "python/sglang/srt/arg_groups/cuda_graph_hook.py")
    mem = os.path.join(repo, "python/sglang/srt/arg_groups/memory_hook.py")
    base = os.path.join(repo, "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py")
    dec = os.path.join(repo, "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py")

    lists = {}
    for max_bs in (64, 128, 256):
        cap = exec_generator(hook, max_bs)
        if max(cap) != max_bs:
            die(f"generator max {max(cap)} != requested {max_bs}")
        lists[str(max_bs)] = cap

    # sensitivity controls: 64 must exclude 128; 256 must include it
    if 128 in lists["64"]:
        die("control failed: 128 in capture list for max_bs=64")
    if 128 not in lists["256"]:
        die("control failed: 128 missing from max_bs=256 list")
    if max(lists["64"]) != 64:
        die("max_bs=64 list exceeds 64")

    mem_tree = ast.parse(open(mem, encoding="utf-8").read())
    mem_gen = find_assign_expr(
        mem_tree, mem,
        "generate_decode_cuda_graph_batch_sizes(server_args, decode_cuda_graph_config.max_bs)")
    if not mem_gen:
        die("memory_hook: materialization from decode.max_bs not found")

    base_tree = ast.parse(open(base, encoding="utf-8").read())
    base_cap = find_assign_expr(
        base_tree, base, "capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)")
    if not base_cap:
        die("base runner: capture_bs source from decode.bs not found")

    dec_src = open(dec, encoding="utf-8").read()
    dec_tree = ast.parse(dec_src)
    max_assign = find_assign_expr(dec_tree, dec, "self.max_bs = max(self.capture_bs)")
    if not max_assign:
        die("decode runner: self.max_bs = max(capture_bs) not found")
    pred = "cuda_graph_bs <= self.max_bs"
    if pred not in dec_src:
        die("decode runner: replay predicate not found")
    # the predicate must sit in the not-disable_padding branch of is_bs_supported
    can_run = None
    for node in ast.walk(dec_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "can_run_graph":
            can_run = ast.get_source_segment(dec_src, node)
    if can_run is None or pred not in can_run:
        die("decode runner: predicate not inside can_run_graph")

    return {
        "capture_list_max_bs_64": lists["64"],
        "capture_list_max_bs_128_head": lists["128"][:8],
        "capture_list_max_bs_256_contains_128": 128 in lists["256"],
        "capture_list_max_bs_64_contains_128": 128 in lists["64"],
        "memory_hook_materialization": mem_gen[0],
        "base_runner_capture_source": base_cap[0],
        "decode_runner_max_bs": max_assign[0],
        "replay_predicate": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trace = analyze_trace(a.trace)
    source = analyze_source(a.repo)

    record = {"trace": trace, "source": source}
    # Verdict rule, pre-registered: the observed engine label (bs=128) plus one
    # graph launch per forward cannot occur under HEAD's max_bs=64 capture list;
    # M=64 must remain numerically feasible or the bandwidth control is invalid.
    contradiction = (
        trace["step_labels"] == ["step[DECODE bs=128]"]
        and trace["launches_inside_each_step_span"] == [1] * 16
        and source["capture_list_max_bs_64_contains_128"] is False
        and trace["implied_attention_bandwidth"]["M=64"]["feasible_under_273"] is True
    )
    record["verdict"] = (
        "CONTRADICTION_CONFIRMED: real trace replays a 128-row decode graph per "
        "step while HEAD's --cuda-graph-max-bs-decode 64 capture list tops out at "
        "64; the profiled tree's resolved decode-graph config is not HEAD's, so "
        "the trace cannot be used as HEAD's C128 graph behavior."
        if contradiction else "INCONCLUSIVE: pre-registered conditions not met"
    )
    with open(a.out, "w") as f:
        json.dump(record, f, indent=1)
    print(record["verdict"])
    print(json.dumps(record, indent=1))
    if not contradiction:
        die("verdict not reached")


if __name__ == "__main__":
    main()

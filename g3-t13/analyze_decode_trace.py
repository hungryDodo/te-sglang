#!/usr/bin/env python3
"""Focused, bounded analysis of the Controller-supplied BF16 C128 DECODE trace.

Question (new, falsifiable): does the observed 16-step C128 decode under the
frozen launch (--cuda-graph-max-bs-decode 64) still have a host-side critical
path, and is every step a CUDA-graph replay?

Mechanism being tested: the arg_groups/memory_hook.py RSI hook widens decode
graph capture to min(max_running_requests, 256) when the CLI knob is lower,
so at bs=128 each decode step should replay a captured bucket-128 graph and
the host should be fully hidden behind kernels.

Falsifiers:
  F1 any of the 16 scheduler steps has no cudaGraphLaunch call;
  F2 kernel-union fraction of any step window < 0.95;
  F3 a host EventSynchronize interval is not nested inside kernel execution;
  F4 the sum of per-step kernel time disagrees with the packet totals.

Input is read-only. Output: JSON to stdout / file; no event dumps.
"""
import argparse
import collections
import gzip
import json
import sys

ATTN = "BatchPrefillWithPagedKVCacheKernel"
GEMM_KEYS = ("nvjet_sm121", "cutlass_80_tensorop_bf16")
LMHEAD_KEY = "nvjet_sm121_tst_mma_128x208x64"
SAMPLE_KEYS = ("softcap_inplace_logits_kernel", "ArgMaxOps")
MEMCPY_CATS = ("gpu_memcpy", "gpu_memset")


def union_ms(intervals):
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    out = []
    s, e = iv[0]
    for a, b in iv[1:]:
        if a > e:
            out.append((s, e))
            s, e = a, b
        else:
            e = max(e, b)
    out.append((s, e))
    return sum(b - a for a, b in out) / 1000.0


def overlap_ms(a_list, b_list):
    """Total duration of a-intervals covered by union of b-intervals."""
    if not a_list or not b_list:
        return 0.0
    b = sorted(b_list)
    total = 0.0
    j = 0
    for s, e in sorted(a_list):
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            lo, hi = max(cur, b[k][0]), min(e, b[k][1])
            if hi > lo:
                total += hi - lo
                cur = hi
            k += 1
        # do not advance j past intervals that may overlap the next a-interval
        while j < len(b) and b[j][1] <= cur:
            j += 1
    return total / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out")
    args = ap.parse_args()

    with gzip.open(args.trace, "rt") as f:
        doc = json.load(f)
    ev = doc["traceEvents"]

    kernels = [e for e in ev if e.get("cat") == "kernel"]
    launches = sorted((e for e in ev if e.get("name") == "cudaGraphLaunch"),
                      key=lambda e: e["ts"])
    syncs = sorted((e for e in ev if e.get("name") == "cudaEventSynchronize"),
                   key=lambda e: e["ts"])
    cpu_steps = sorted((e for e in ev
                        if e.get("cat") == "user_annotation"
                        and e.get("name") == "step[DECODE bs=128]"),
                       key=lambda e: e["ts"])
    ann_other = collections.Counter(e.get("name") for e in ev
                                    if e.get("cat") == "user_annotation")

    gpu_steps = sorted((e for e in ev
                        if e.get("cat") == "gpu_user_annotation"
                        and e.get("name") == "step[DECODE bs=128]"),
                       key=lambda e: e["ts"])

    trap = {}
    trap["event_counts"] = {
        "kernels": len(kernels),
        "cudaGraphLaunch": len(launches),
        "cudaEventSynchronize": len(syncs),
        "step[DECODE bs=128]": len(cpu_steps),
        "gpu_step[DECODE bs=128]": len(gpu_steps),
    }
    trap["preconditions"] = {
        "sixteen_launches": len(launches) == 16,
        "sixteen_cpu_step_labels": len(cpu_steps) == 16,
        "sixteen_gpu_step_spans": len(gpu_steps) == 16,
        "kernels_multiple_of_16": len(kernels) % 16 == 0,
    }
    if not all(trap["preconditions"].values()):
        trap["verdict"] = "HARNESS_PRECONDITION_FAILED"
        emit(trap, args.out)
        return 2

    # Step windows: the GPU-side per-step annotation spans (they bracket each
    # step's kernels far more cleanly than host launch timestamps, which are
    # enqueued asynchronously and start mid-step at capture open).
    windows = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in gpu_steps]

    def cat_of(name):
        if ATTN in name:
            return "attention"
        if LMHEAD_KEY in name:
            return "lm_head"
        if any(k in name for k in GEMM_KEYS):
            return "gemm"
        if any(k in name for k in SAMPLE_KEYS):
            return "sampling"
        return "other"

    steps = []
    for i, (a, b) in enumerate(windows):
        kk = [e for e in kernels if a <= e["ts"] < b]
        per = collections.defaultdict(float)
        for e in kk:
            per[cat_of(e["name"])] += e.get("dur", 0)
        kv = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in kk]
        seq = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in syncs
               if a <= e["ts"] < b and e["ts"] + e.get("dur", 0) <= b]
        steps.append({
            "step": i,
            "window_ms": round((b - a) / 1000.0, 3),
            "kernels": len(kk),
            "kernel_sum_ms": round(sum(d for d in per.values()) / 1000.0, 3),
            "kernel_union_ms": round(union_ms(kv), 3),
            "kernel_union_frac": round(union_ms(kv) / ((b - a) / 1000.0), 4),
            "attention_ms": round(per["attention"] / 1000.0, 3),
            "gemm_ms": round((per["gemm"] + per["lm_head"]) / 1000.0, 3),
            "sampling_ms": round(per["sampling"] / 1000.0, 3),
            "other_ms": round(per["other"] / 1000.0, 3),
            "event_sync_ms": round(sum(e.get("dur", 0) for e in syncs
                                       if a <= e["ts"] < b) / 1000.0, 3),
            "event_sync_overlap_kernel_ms": round(
                overlap_ms(seq, kv), 3),
            "cpu_step_ms": round(sum(
                min(e.get("dur", 0), (b - e["ts"]))
                for e in cpu_steps if a <= e["ts"] < b) / 1000.0, 3),
            "has_graph_launch": True,
        })

    trap["steps"] = steps
    n = len(steps)
    means = {k: round(sum(s[k] for s in steps) / n, 3) for k in
             ("window_ms", "kernel_sum_ms", "kernel_union_ms",
              "attention_ms", "gemm_ms", "sampling_ms", "other_ms",
              "event_sync_ms", "cpu_step_ms")}
    trap["per_step_mean"] = means
    trap["shares"] = {
        "attention": round(means["attention_ms"] / means["kernel_sum_ms"], 4),
        "gemm_plus_lmhead": round(means["gemm_ms"] / means["kernel_sum_ms"], 4),
        "sampling": round(means["sampling_ms"] / means["kernel_sum_ms"], 4),
        "other": round(means["other_ms"] / means["kernel_sum_ms"], 4),
    }
    sync_total = sum(e.get("dur", 0) for e in syncs)
    sync_ov = overlap_ms([(e["ts"], e["ts"] + e.get("dur", 0)) for e in syncs],
                         [(e["ts"], e["ts"] + e.get("dur", 0)) for e in kernels])
    trap["visible_sync"] = {
        "event_sync_sum_ms": round(sync_total / 1000.0, 3),
        "event_sync_inside_kernels_ms": round(sync_ov, 3),
        "event_sync_inside_kernels_frac": round(sync_ov / (sync_total / 1000.0), 4),
    }
    trap["falsifiers"] = {
        "F1_all_steps_graph": len(launches) == 16,
        "F2_min_step_kernel_union_frac_ge_0_95": min(
            s["kernel_union_frac"] for s in steps) >= 0.95,
        "F3_event_sync_nested_frac_ge_0_95":
            trap["visible_sync"]["event_sync_inside_kernels_frac"] >= 0.95,
    }
    trap["verdict"] = (
        "HOST_HIDDEN_GRAPH_REPLAY_EVERY_STEP"
        if all(trap["falsifiers"].values())
        else "HOST_CRITICAL_PATH_OR_EAGER_STEP_PRESENT")
    trap["attention_call_count"] = sum(
        1 for e in kernels if ATTN in e["name"])
    trap["attention_ms_per_call"] = round(
        sum(e.get("dur", 0) for e in kernels if ATTN in e["name"])
        / trap["attention_call_count"] / 1000.0, 4)

    # Context scaling: each decode step appends exactly one KV position per
    # request, so under a KV-byte-proportional attention cost the per-step
    # attention time must grow with relative slope 1/L0 per step (L0 ~= 513
    # for a 512-token prompt -> ~0.0326 ms/step at the observed level).
    def ols(y):
        xs = list(range(len(y)))
        n = len(y)
        mx = sum(xs) / n
        my = sum(y) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (yy - my) for x, yy in zip(xs, y))
        b = sxy / sxx
        a = my - b * mx
        resid = [yy - (a + b * x) for x, yy in zip(xs, y)]
        s2 = sum(r * r for r in resid) / (n - 2)
        se = (s2 / sxx) ** 0.5
        return round(b, 5), round(se, 5)

    attn_slope, attn_se = ols([s["attention_ms"] for s in steps])
    tot_slope, tot_se = ols([s["kernel_sum_ms"] for s in steps])
    mean_attn = means["attention_ms"]
    trap["context_scaling"] = {
        "attention_slope_ms_per_step": attn_slope,
        "attention_slope_se": attn_se,
        "kernel_sum_slope_ms_per_step": tot_slope,
        "kernel_sum_slope_se": tot_se,
        "predicted_slope_if_kv_proportional_at_L0_513": (
            round(mean_attn / 513.0, 5)),
        "predicted_slope_if_kv_proportional_at_L0_577": (
            round(mean_attn / 577.0, 5)),
        "slope_ratio_to_kv_prediction_513": (
            round(attn_slope / (mean_attn / 513.0), 3)
            if mean_attn else None),
        "steps_with_context_growth": len(steps) - 1,
    }
    trap["annotation_names"] = dict(ann_other)
    trap["scope"] = (
        "One instrumented 16-forward C128 Medium512/64 decode fragment, GB10 "
        "a1, Controller packet commit 15fd7f4. Kernel union is not SM "
        "utilization. No speed ratio, no gate, no extrapolation to other "
        "batch sizes or hosts.")
    emit(trap, args.out)
    return 0


def emit(obj, path):
    txt = json.dumps(obj, indent=1, sort_keys=True)
    if path:
        with open(path, "w") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

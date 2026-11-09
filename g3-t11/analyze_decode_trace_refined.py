#!/usr/bin/env python3 -I
"""G3 t11 refined pass: idle-with-memcpy, leftover kernel identity, CPU runtime census.
Companion to analyze_decode_trace.py; same trace, additional slices."""
import gzip, json, sys
from collections import Counter, defaultdict

TRACE = sys.argv[1] if len(sys.argv) > 1 else "controller_evidence/continuous-20261109-01/20261110-decode-trace/decode.trace.json.gz"
d = json.load(gzip.open(TRACE))
ev = d["traceEvents"]
kern = [e for e in ev if e.get("cat") == "kernel"]
mc = [e for e in ev if e.get("cat") == "gpu_memcpy"]
ms = [e for e in ev if e.get("cat") == "gpu_memset"]
gpu = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation" and "DECODE" in e["name"]], key=lambda e: e["ts"])

def union(ivs):
    ivs = sorted(ivs); o = []
    for s, t in ivs:
        if o and s <= o[-1][1]: o[-1][1] = max(o[-1][1], t)
        else: o.append([s, t])
    return o

# leftover kernels outside all GPU step spans (assignment negative control)
out_k = [k for k in kern if not any(g["ts"] <= k["ts"] < g["ts"] + g["dur"] for g in gpu)]
leftover = {
    "n": len(out_k), "total_us": round(sum(k["dur"] for k in out_k), 1),
    "names": Counter(k["name"][:90] for k in out_k).most_common(),
    "first_ts_rel_to_first_span_us": round(out_k[0]["ts"] - gpu[0]["ts"], 1) if out_k else None,
    "all_before_first_span_end": bool(all(k["ts"] < gpu[0]["ts"] + gpu[0]["dur"] for k in out_k)),
}

# per-step idle with memcpy/memset included in coverage
allg = kern + mc + ms
rows = []
for g in gpu:
    s, t = g["ts"], g["ts"] + g["dur"]
    cov = sum(b - a for a, b in union([(e["ts"], e["ts"] + e["dur"]) for e in allg if s <= e["ts"] < t]))
    rows.append({"span_us": round(g["dur"], 1), "idle_us": round(g["dur"] - cov, 1)})
steady = rows[1:]
memcpy = {
    "n": len(mc), "total_us": round(sum(m["dur"] for m in mc), 1),
    "names": Counter(m["name"] for m in mc).most_common(),
    "n_memset": len(ms), "memset_total_us": round(sum(m["dur"] for m in ms), 1),
}
rt_cnt = Counter(e["name"] for e in ev if e.get("cat") == "cuda_runtime")
rt_tot = defaultdict(float)
for e in ev:
    if e.get("cat") == "cuda_runtime": rt_tot[e["name"]] += e["dur"]
gl = [e for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch"]
runtime = {
    "counts": dict(rt_cnt.most_common()),
    "total_us": {k: round(v, 1) for k, v in sorted(rt_tot.items(), key=lambda kv: -kv[1])},
    "cudaGraphLaunch_n": len(gl),
    "cudaGraphLaunch_mean_us": round(sum(e["dur"] for e in gl) / len(gl), 1) if gl else None,
}
res = {
    "trace": TRACE,
    "leftover_kernels_outside_spans": leftover,
    "per_step_idle_incl_memcpy": rows,
    "steady_state_mean_idle_us": round(sum(r["idle_us"] for r in steady) / len(steady), 1),
    "steady_state_mean_span_us": round(sum(r["span_us"] for r in steady) / len(steady), 1),
    "steady_state_idle_frac": round(sum(r["idle_us"] for r in steady) / sum(r["span_us"] for r in steady), 5),
    "memcpy": memcpy, "runtime_census": runtime,
    "interpretation": {
        "eventSynchronize_note": "cudaEventSynchronize 472169us total (~29.5ms/step) is the overlap-scheduler result-sync; GPU kernels are busy during it (README: overlaps kernels 469.531ms), so it is not host-caused GPU idle.",
        "step1_anomaly": "step 1 span idle 1205us is a profiler-start artifact; steps 2-16 are the steady state.",
    },
}
json.dump(res, open("g3-t11/raw/decode_trace_refined.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_step_idle_incl_memcpy", "runtime_census")}, indent=1)[:1200])
print("runtime totals:", runtime["total_us"], "graphLaunch", runtime["cudaGraphLaunch_n"], runtime["cudaGraphLaunch_mean_us"])

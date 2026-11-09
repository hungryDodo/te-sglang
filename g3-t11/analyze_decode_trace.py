#!/usr/bin/env python3 -I
"""G3 t11: focused analysis of the controller's real BF16 C128 decode trace.

Questions:
  Q1 host lever: GPU idle (kernel-union gaps) within/between decode steps.
  Q2 composition: kernel families as share of per-step GPU span.
Controls: recompute README totals; 16/16 pairing; assignment negative control.
"""
import gzip, json, sys, re
from collections import defaultdict

TRACE = sys.argv[1] if len(sys.argv) > 1 else "controller_evidence/continuous-20261109-01/20261110-decode-trace/decode.trace.json.gz"
REF = {  # README-stated totals (ms) for the independent recompute check
    "window_ms": 571.927153, "kernel_union_ms": 565.549025,
    "cpu_step_union_ms": 28.303983, "kernel_instances": 8560,
}

d = json.load(gzip.open(TRACE))
ev = d["traceEvents"]

kernels = [e for e in ev if e.get("cat") == "kernel"]
memcpys = [e for e in ev if e.get("cat") == "gpu_memcpy"]
memsets = [e for e in ev if e.get("cat") == "gpu_memset"]
cpu_steps = [e for e in ev if e.get("cat") == "user_annotation" and "DECODE" in (e.get("name") or "")]
gpu_steps = [e for e in ev if e.get("cat") == "gpu_user_annotation" and "DECODE" in (e.get("name") or "")]

def span(e): return (e["ts"], e["ts"] + e["dur"])

# --- control 0: instance counts
assert len(cpu_steps) == 16 and len(gpu_steps) == 16, (len(cpu_steps), len(gpu_steps))
n_k = len(kernels)

# --- interval union helper (us)
def union(ivs):
    ivs = sorted(ivs); out = []
    for s, t in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], t)
        else:
            out.append([s, t])
    return out

def total(u): return sum(t - s for s, t in u)

gpu_ivs = [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
env = (min(s for s, _ in gpu_ivs), max(t for _, t in gpu_ivs))
u_all = union(gpu_ivs)
union_ms = total(u_all) / 1000.0
win_ms = (env[1] - env[0]) / 1000.0
cpu_u = union([span(e) for e in cpu_steps])
cpu_union_ms = total(cpu_u) / 1000.0

check = {
    "kernel_instances": {"got": n_k, "ref": REF["kernel_instances"], "ok": n_k == REF["kernel_instances"]},
    "kernel_union_ms": {"got": round(union_ms, 6), "ref": REF["kernel_union_ms"],
                        "rel_diff": round(abs(union_ms - REF["kernel_union_ms"]) / REF["kernel_union_ms"], 8)},
    "window_ms": {"got": round(win_ms, 6), "ref": REF["window_ms"],
                  "rel_diff": round(abs(win_ms - REF["window_ms"]) / REF["window_ms"], 8)},
    "cpu_step_union_ms": {"got": round(cpu_union_ms, 6), "ref": REF["cpu_step_union_ms"],
                          "rel_diff": round(abs(cpu_union_ms - REF["cpu_step_union_ms"]) / REF["cpu_step_union_ms"], 8)},
}
checks_ok = check["kernel_instances"]["ok"] and all(v["rel_diff"] < 1e-3 for k, v in check.items() if k != "kernel_instances")

# --- pairing by External id
cpu_by_id = {e["args"]["External id"]: e for e in cpu_steps}
gpu_by_id = {e["args"]["External id"]: e for e in gpu_steps}
assert set(cpu_by_id) == set(gpu_by_id)
ids = sorted(gpu_by_id, key=lambda i: gpu_by_id[i]["ts"])

# --- assign kernels+memcpy to steps by containment in GPU step span (neg control: leftovers)
step_rows = []
for i in ids:
    g = gpu_by_id[i]; c = cpu_by_id[i]
    gs, ge = span(g)
    ks = [k for k in kernels if gs - 1 <= k["ts"] < ge]
    u = union([(k["ts"], k["ts"] + k["dur"]) for k in ks])
    cov = total(u)
    step_rows.append({
        "external_id": i,
        "gpu_span_us": round(g["dur"], 3),
        "cpu_label_us": round(c["dur"], 3),
        "n_kernels": len(ks),
        "kernel_sum_us": round(sum(k["dur"] for k in ks), 3),
        "kernel_union_us": round(cov, 3),
        "idle_in_span_us": round(g["dur"] - cov, 3),
        "idle_frac": round(1 - cov / g["dur"], 5),
    })

leftover = [k for k in kernels if not any(gpu_by_id[i]["ts"] <= k["ts"] < gpu_by_id[i]["ts"] + gpu_by_id[i]["dur"] for i in ids)]
leftover_ms = sum(k["dur"] for k in leftover) / 1000.0

# --- inter-step gaps (GPU annotation end -> next GPU annotation start)
gaps = []
for a, b in zip(ids, ids[1:]):
    ga, gb = gpu_by_id[a], gpu_by_id[b]
    s, t = ga["ts"] + ga["dur"], gb["ts"]
    if t <= s: continue
    cov = total(union([(k["ts"], k["ts"] + k["dur"]) for k in kernels if k["ts"] >= s and k["ts"] < t]))
    gaps.append({"between": [a, b], "gap_us": round(t - s, 3), "kernel_covered_us": round(cov, 3),
                 "uncovered_us": round(t - s - cov, 3)})

# --- composition: group kernel names
def fam(n):
    if "flashinfer" in n and ("Prefill" in n or "Decode" in n or "MergeStates" in n): return "attention_flashinfer"
    if "store_kvcache" in n: return "store_kv"
    if "rmsnorm" in n.lower() or "RMSNorm" in n: return "norm"
    if "nvjet" in n or "cutlass" in n or "cublas" in n or "gemm" in n.lower(): return "gemm"
    if "elementwise" in n or "copy" in n.lower() or "CatArrayBatched" in n: return "elementwise"
    if "reduce" in n.lower() or "Reduce" in n: return "reduce"
    return "other"

by_name = defaultdict(lambda: [0, 0.0])
for k in kernels:
    by_name[k["name"]][0] += 1; by_name[k["name"]][1] += k["dur"]
top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:15]
fam_tot = defaultdict(float)
for n, (c, t) in by_name.items(): fam_tot[fam(n)] += t
grand = sum(v for v in fam_tot.values())
mc = [{"name": m.get("name"), "us": round(m["dur"], 3)} for m in sorted(memcpys, key=lambda m: -m["dur"])[:10]]

out = {
    "trace": TRACE, "sha_note": "raw per MANIFEST 09e8dcb9...", "events": len(ev),
    "controls_recompute_vs_README": check, "controls_ok": bool(checks_ok),
    "pairing": "16/16 CPU<->GPU by External id",
    "assignment_negative_control": {"kernels_outside_all_gpu_spans": len(leftover), "their_total_ms": round(leftover_ms, 6),
                                    "frac_of_kernel_ms": round(leftover_ms / (union_ms if union_ms else 1), 6)},
    "window": {"envelope_ms": round(win_ms, 3), "kernel_union_ms": round(union_ms, 3),
               "global_gpu_uncovered_ms": round(win_ms - union_ms, 3),
               "global_uncovered_frac": round(1 - union_ms / win_ms, 5)},
    "steps": step_rows,
    "inter_step_gaps": gaps,
    "per_step_means": {
        "gpu_span_us": round(sum(r["gpu_span_us"] for r in step_rows) / 16, 3),
        "idle_in_span_us": round(sum(r["idle_in_span_us"] for r in step_rows) / 16, 3),
        "mean_idle_frac": round(sum(r["idle_frac"] for r in step_rows) / 16, 5),
        "kernel_count_minmax": [min(r["n_kernels"] for r in step_rows), max(r["n_kernels"] for r in step_rows)],
    },
    "family_share_of_kernel_time": {k: {"us": round(v, 1), "pct": round(100 * v / grand, 2)} for k, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])},
    "top_kernels_by_total_us": [{"name": n[:120], "count": c, "total_us": round(t, 1), "per_step": round(c / 16, 1), "mean_us": round(t / c, 2)} for n, (c, t) in top],
    "memcpy_top": mc, "n_memcpy": len(memcpys), "n_memset": len(memsets),
}
json.dump(out, open("g3-t11/raw/decode_trace_analysis.json", "w"), indent=1)
print("CONTROLS_OK", checks_ok, json.dumps(check))
print("GLOBAL uncovered %.3f ms of %.3f (%.2f%%)" % (win_ms - union_ms, win_ms, 100 * (1 - union_ms / win_ms)))
print("LEFTOVER outside spans:", len(leftover), "kernels", round(leftover_ms, 3), "ms")
print("PER-STEP idle us:", [r["idle_in_span_us"] for r in step_rows])
print("PER-STEP spans us:", [r["gpu_span_us"] for r in step_rows])
print("GAPS:", json.dumps(gaps))
print("FAMILIES:", json.dumps(out["family_share_of_kernel_time"]))
for t in out["top_kernels_by_total_us"][:10]:
    print("TOP %6.1fus x%3d (%.1f/step, mean %.2fus) %s" % (t["total_us"], t["count"], t["per_step"], t["mean_us"], t["name"][:95]))

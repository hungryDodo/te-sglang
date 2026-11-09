#!/usr/bin/env python3 -I
"""G3 trial-9 (glm-a): measured host-bubble census on the controller's real BF16
C128 decode trace (controller_evidence/continuous-20261109-01/20261110-decode-trace/
decode.trace.json.gz, raw SHA 09e8dcb9... per packet README; source commit 15fd7f4).

Question (H9, falsifiable): is host-side time NOT hidden under executing kernels
large enough (>=2% of marked wall) to justify an exact-runtime sync-removal /
SWA-host-bound Fast candidate on a1 for this workload?

Hard gate: the parser must first reproduce the packet README's stated anchors
within tolerance. Any anchor mismatch => verdict NO_CLAIM, exit 2, nothing else
is claimed. Self-tests of the interval machinery use hand-computed fixtures.
"""
import gzip, json, sys, os, hashlib

TRACE = sys.argv[1] if len(sys.argv) > 1 else \
    "controller_evidence/continuous-20261109-01/20261110-decode-trace/decode.trace.json.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "g3-t9/raw/trace_bubble_census_output.json"
TOL = 0.5  # us tolerance for anchor floats

# ---------- interval machinery ----------
def normalize(iv):
    """list of (ts,dur) -> sorted, merged list of (start,end)."""
    segs = sorted((ts, ts + dur) for ts, dur in iv if dur > 0)
    out = []
    for s, e in segs:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out

def ulen(u):
    return sum(e - s for s, e in u)

def overlap_len(u, s, e):
    """length of [s,e) covered by union u."""
    tot = 0.0
    for a, b in u:
        if b <= s:
            continue
        if a >= e:
            break
        tot += min(b, e) - max(a, s)
    return tot

def holes(u, lo, hi):
    """kernel-free gaps of union u within [lo,hi)."""
    out = []
    cur = lo
    for a, b in u:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out

# ---------- self-tests: hand-computed fixtures (independent expected values) ----------
def self_test():
    # fixtures are (ts, dur_us); hand-computed unions on the right
    assert normalize([(0, 10), (20, 30)]) == [(0, 10), (20, 50)]
    assert normalize([(0, 10), (10, 20), (5, 6)]) == [(0, 30)]
    assert normalize([(0, 10), (9, 1)]) == [(0, 10)]
    u = normalize([(0, 10), (20, 10)])          # [(0,10),(20,30)]
    assert overlap_len(u, 5, 25) == 10.0        # 5..10 + 20..25
    assert overlap_len(u, 0, 35) == 20.0        # total union
    assert overlap_len(u, 12, 18) == 0.0        # inside hole
    assert holes(u, 0, 40) == [(10, 20), (30, 40)]
    # planted-control arithmetic used later, by hand: step of dur 300 fully in a
    # hole of length >=300 must report not-hidden 300.0; step exactly covering
    # a kernel interval of length L must report 0.0.
    u2 = normalize([(100, 50), (400, 50)])      # [(100,150),(400,450)]
    assert overlap_len(u2, 200, 500) == 50.0    # only second kernel covered
    return True

# ---------- main ----------
def main():
    raw = open(TRACE, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    ev = json.loads(gzip.decompress(raw))["traceEvents"]

    kern = [e for e in ev if e.get("cat") == "kernel"]
    mcpy = [e for e in ev if e.get("cat") == "gpu_memcpy"]
    mset = [e for e in ev if e.get("cat") == "gpu_memset"]
    ua = [e for e in ev if e.get("cat") == "user_annotation"]
    gua = [e for e in ev if e.get("cat") == "gpu_user_annotation"]
    rt = [e for e in ev if e.get("cat") == "cuda_runtime"]
    trace = [e for e in ev if e.get("cat") == "Trace"][0]

    steps = [e for e in ua if e["name"] == "step[DECODE bs=128]"]
    gsteps = [e for e in gua if e["name"] == "step[DECODE bs=128]"]
    evsync = [e for e in rt if e["name"] == "cudaEventSynchronize"]
    dvsync = [e for e in rt if e["name"] == "cudaDeviceSynchronize"]

    ku = normalize([(e["ts"], e["dur"]) for e in kern])
    devu = normalize([(e["ts"], e["dur"]) for e in kern + mcpy + mset])
    stepu = normalize([(e["ts"], e["dur"]) for e in steps])
    hostu = normalize([(e["ts"], e["dur"]) for e in ua + rt])  # host-thread annotated+runtime

    w0, w1 = trace["ts"], trace["ts"] + trace["dur"]
    wall = w1 - w0

    # ---- anchors (packet README, us) ----
    anchors = {
        "trace_span_dur": (trace["dur"], 571927.153),
        "kernel_count": (len(kern), 8560),
        "kernel_union": (ulen(ku), 565549.025),
        "cpu_step_union": (ulen(stepu), 28303.983),
        "eventsync_sum": (sum(e["dur"] for e in evsync), 472169.159),
        "eventsync_kernel_overlap": (sum(overlap_len(ku, e["ts"], e["ts"] + e["dur"]) for e in evsync), 469530.690),
    }
    # largest host inter-step gap + its kernel overlap
    sst = sorted((e["ts"], e["ts"] + e["dur"]) for e in steps)
    gaps = [(sst[i + 1][0] - sst[i][1], sst[i][1], sst[i + 1][0]) for i in range(len(sst) - 1)]
    big_gap_dur, bg_s, bg_e = max(gaps)
    anchors["largest_interstep_gap"] = (big_gap_dur, 34676.268)
    anchors["largest_gap_kernel_overlap"] = (overlap_len(ku, bg_s, bg_e), 34417.705)

    ok = all(abs(a - b) <= TOL if isinstance(b, float) else a == b
             for a, b in anchors.values())
    anchor_rows = {k: {"got": round(a, 6), "expect": b,
                       "match": (a == b) if isinstance(b, int) else abs(a - b) <= TOL}
                   for k, (a, b) in anchors.items()}

    # ---- controls ----
    cpu_ids = sorted(e["args"].get("External id") for e in steps)
    gpu_ids = sorted(e["args"].get("External id") for e in gsteps)
    pairing = {"cpu_steps": len(steps), "gpu_steps": len(gsteps),
               "ext_id_sets_equal": cpu_ids == gpu_ids}

    # planted-gap negative control: synthetic 300.0us host step inside a real
    # device-free hole of length >= 400us -> not-hidden must be 300.0 +/- TOL.
    dev_holes = [h for h in holes(devu, w0, w1) if h[1] - h[0] >= 400.0]
    ctrl = {"device_holes_ge_400us": len(dev_holes)}
    if dev_holes:
        hs, he = dev_holes[0]
        plant_s = hs + 50.0
        plant_e = plant_s + 300.0
        not_hidden = 300.0 - overlap_len(devu, plant_s, plant_e)
        ctrl["planted_gap"] = {"hole": [round(hs - w0, 3), round(he - w0, 3)],
                               "expected_not_hidden": 300.0,
                               "got_not_hidden": round(not_hidden, 6),
                               "pass": abs(not_hidden - 300.0) <= TOL}
        # all-overlapped control: synthetic step exactly covering the longest kernel
        kmax = max(kern, key=lambda e: e["dur"])
        oh = overlap_len(devu, kmax["ts"], kmax["ts"] + kmax["dur"])
        ctrl["all_overlapped"] = {"kernel_dur": round(kmax["dur"], 3),
                                  "expected_not_hidden": 0.0,
                                  "got_not_hidden": round(kmax["dur"] - oh, 6),
                                  "pass": abs((kmax["dur"] - oh)) <= TOL}
        ctrl_pass = ctrl["planted_gap"]["pass"] and ctrl["all_overlapped"]["pass"]
    else:
        ctrl_pass = False

    verdict_gate = ok and pairing["ext_id_sets_equal"] and pairing["cpu_steps"] == 16 and ctrl_pass
    base = {"trace_sha256_first16": sha[:16], "trace_file": TRACE,
            "anchors": anchor_rows, "pairing": pairing, "controls": ctrl,
            "self_tests": "pass"}
    if not verdict_gate:
        base["verdict"] = "NO_CLAIM"
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(base, open(OUT, "w"), indent=1)
        print(json.dumps({"verdict": "NO_CLAIM", "anchors": anchor_rows,
                          "controls": ctrl}, indent=1))
        sys.exit(2)

    # ---- main derivations (only reached when gate passes) ----
    dev_busy = ulen(devu)
    # per-phase host decomposition: total dur, device-overlapped (hidden), active-not-hidden
    phases = {}
    for e in ua:
        n = e["name"]
        p = phases.setdefault(n, {"dur": 0.0, "hidden": 0.0, "n": 0})
        p["dur"] += e["dur"]
        p["hidden"] += overlap_len(devu, e["ts"], e["ts"] + e["dur"])
        p["n"] += 1
    for n, p in phases.items():
        p["active_not_hidden"] = p["dur"] - p["hidden"]
        p["not_hidden_pct_of_wall"] = round(100.0 * p["active_not_hidden"] / wall, 4)
        for k in ("dur", "hidden", "active_not_hidden"):
            p[k] = round(p[k], 3)

    # blocking syncs: blocked while device idle = hard bubble
    sync_rows = []
    for e in evsync + dvsync:
        ov = overlap_len(devu, e["ts"], e["ts"] + e["dur"])
        sync_rows.append({"name": e["name"], "dur": round(e["dur"], 3),
                          "hidden": round(ov, 3), "bubble": round(e["dur"] - ov, 3),
                          "t_rel": round(e["ts"] - w0, 3)})
    hard_bubble = sum(r["bubble"] for r in sync_rows)
    top_bubbles = sorted(sync_rows, key=lambda r: -r["bubble"])[:5]

    # step-level: attach device time + host-not-hidden to each CPU step
    per_step = []
    for e in sorted(steps, key=lambda x: x["ts"]):
        s, d = e["ts"], e["dur"]
        per_step.append({"t_rel": round(s - w0, 3), "cpu_dur": round(d, 3),
                         "hidden": round(overlap_len(devu, s, s + d), 3),
                         "not_hidden": round(d - overlap_len(devu, s, s + d), 3)})

    # wall decomposition (exclusive buckets, us)
    host_active_not_hidden = 0.0
    for e in ua + rt:
        host_active_not_hidden += e["dur"] - overlap_len(devu, e["ts"], e["ts"] + e["dur"])
    # cap at union level to avoid double counting nested annotations
    host_nh_union = ulen(hostu) - overlap_len(devu, w0, w1) if False else None
    dead = wall - dev_busy - (ulen(normalize([(e["ts"], e["dur"]) for e in rt])) -
                              overlap_len(devu, w0, w1))
    decomp = {
        "wall_us": round(wall, 3),
        "device_busy_us": round(dev_busy, 3),
        "device_busy_pct": round(100 * dev_busy / wall, 4),
        "runtime_calls_not_hidden_us": round(
            sum(e["dur"] - overlap_len(devu, e["ts"], e["ts"] + e["dur"]) for e in rt), 3),
        "annotations_summed_not_hidden_us": round(host_active_not_hidden, 3),
        "host_union_us": round(ulen(hostu), 3),
        "host_union_not_hidden_us": round(ulen(hostu) - overlap_len(devu, w0, w1), 3),
    }

    # largest gap anatomy: what runs inside it
    in_gap = [e for e in kern + mcpy + mset if e["ts"] >= bg_s - 1e-6 and e["ts"] + e["dur"] <= bg_e + 1e-6]
    gap_kernel_names = {}
    for e in in_gap:
        g = gap_kernel_names.setdefault(e["name"][:80], {"n": 0, "dur": 0.0, "cat": e["cat"]})
        g["n"] += 1
        g["dur"] += e["dur"]
    gap_kernels_sorted = sorted(((k, v) for k, v in gap_kernel_names.items() if v["cat"] == "kernel"),
                                key=lambda kv: -kv[1]["dur"])[:8]

    # kernel census (summed durations are NOT wall time)
    census = {}
    graph_kernels = 0
    for e in kern:
        c = census.setdefault(e["name"][:100], {"n": 0, "dur": 0.0})
        c["n"] += 1
        c["dur"] += e["dur"]
        if (e.get("args") or {}).get("graph id", 0):
            graph_kernels += 1
    ksum = sum(c["dur"] for c in census.values())
    top = sorted(census.items(), key=lambda kv: -kv[1]["dur"])[:14]
    census_rows = [{"name": k[:90], "n": v["n"], "sum_us": round(v["dur"], 1),
                    "pct_of_kernel_sum": round(100 * v["dur"] / ksum, 3)}
                   for k, v in top]

    # group-level split + attention detail (decode attention is dispatched through
    # BatchPrefillWithPagedKVCacheKernel = flashinfer tensor-core decode; no
    # BatchDecode kernel exists in this trace)
    att = [e for e in kern if "BatchPrefillWithPagedKVCache" in e["name"]]
    gemm = [e for e in kern if any(s in e["name"] for s in ("nvjet", "cutlass", "gemm", "Gemm"))
            and "flashinfer" not in e["name"]]
    att_sum = sum(e["dur"] for e in att)
    gemm_sum = sum(e["dur"] for e in gemm)
    att_sorted = sorted(e["dur"] for e in att)
    group_split = {
        "kernel_sum_us": round(ksum, 1), "kernel_union_us": round(ulen(ku), 1),
        "serialized_note": "sum~union => single-stream, summed shares are wall shares",
        "attention_us": round(att_sum, 1), "attention_pct": round(100 * att_sum / ksum, 2),
        "attention_instances": len(att),
        "attention_per_step_us": round(att_sum / 16, 1),
        "attention_single_dur_min_med_max_us": [round(att_sorted[0], 1),
                                                round(att_sorted[len(att_sorted) // 2], 1),
                                                round(att_sorted[-1], 1)],
        "gemm_us": round(gemm_sum, 1), "gemm_pct": round(100 * gemm_sum / ksum, 2),
        "other_pct": round(100 * (ksum - att_sum - gemm_sum) / ksum, 2),
        "decode_attention_kernels_present": sum(1 for e in kern if "BatchDecode" in e["name"]),
        # prior measured cross-check (G2-E2 on same node/model): fp8-e4m3 KV cut
        # decode TPOT by 8-9 ms/step; halving the measured attention time
        # predicts the same number. Comparison to prior data, NOT a new measurement.
        "e2_crosscheck": {"predicted_fp8kv_saving_us_per_step": round(att_sum / 32.0, 1),
                          "e2_measured_ms_per_step": "8-9",
                          "consistent": 8.0 <= att_sum / 32.0 / 1000.0 <= 9.0},
    }
    # per-GPU-step device totals
    gstep_rows = []
    for e in sorted(gsteps, key=lambda x: x["ts"]):
        s, d = e["ts"], e["dur"]
        kd = sum(k["dur"] for k in kern if s <= k["ts"] and k["ts"] + k["dur"] <= s + d)
        gstep_rows.append({"t_rel": round(s - w0, 3), "gpu_step_us": round(d, 1),
                           "kernel_sum_us": round(kd, 1), "kernels": sum(1 for k in kern if s <= k["ts"] and k["ts"] + k["dur"] <= s + d)})

    # ---- H9 verdict ----
    ceiling_us = decomp["annotations_summed_not_hidden_us"]  # generous upper bound
    ceiling_pct = 100.0 * ceiling_us / wall
    # stricter union-level bound (nested annotations double-count protection)
    union_ceiling_us = decomp["host_union_not_hidden_us"]
    union_ceiling_pct = 100.0 * union_ceiling_us / wall
    h9 = {
        "threshold_pct": 2.0,
        "summed_not_hidden_pct": round(ceiling_pct, 4),
        "union_not_hidden_pct": round(union_ceiling_pct, 4),
        "hard_bubble_sync_blocked_device_idle_us": round(hard_bubble, 3),
        "hard_bubble_pct": round(100 * hard_bubble / wall, 4),
        "refuted": union_ceiling_pct < 2.0,
    }

    out = dict(base)
    out.update({"verdict": "H9_REFUTED" if h9["refuted"] else "H9_SUPPORTED",
                "phases": phases, "sync_top_bubbles": top_bubbles,
                "sync_summary": {"n_eventsync": len(evsync), "n_devicesync": len(dvsync),
                                 "hard_bubble_us": round(hard_bubble, 3)},
                "per_step": per_step, "wall_decomposition": decomp,
                "largest_gap": {"span_rel": [round(bg_s - w0, 3), round(bg_e - w0, 3)],
                                "dur_us": round(big_gap_dur, 3),
                                "kernel_overlap_us": round(overlap_len(ku, bg_s, bg_e), 3),
                                "top_kernels_inside": [{"name": k[:80], "n": v["n"],
                                                        "sum_us": round(v["dur"], 1)} for k, v in gap_kernels_sorted],
                                "n_events_inside": len(in_gap)},
                "kernel_census_top": census_rows,
                "kernel_group_split": group_split,
                "graph_kernel_instances": graph_kernels,
                "per_gpu_step": gstep_rows,
                "h9": h9,
                "limits": ["16 instrumented forwards, single run, node a1, traced (perturbation possible)",
                           "kernel interval union != SM utilization or bandwidth; summed kernel durations are a census, not wall",
                           "bounds apply to Medium512/64 C128 decode class only; prefill-heavy/C256 not covered",
                           "not-hidden host time is an upper bound on removable cost, achievable only with zero kernel change"]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({"verdict": out["verdict"], "h9": h9,
                      "device_busy_pct": decomp["device_busy_pct"],
                      "union_ceiling_us": round(union_ceiling_us, 1),
                      "hard_bubble_us": round(hard_bubble, 1)}, indent=1))

if __name__ == "__main__":
    assert self_test()
    main()

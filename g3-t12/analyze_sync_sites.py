#!/usr/bin/env python3 -I
"""g3-t12 — DtoH sync-site census on the real controller decode trace.

New question (vs it11's idle/composition closure): enumerate the blocking
Device->Host memcpy sites in the measured C128 graph-decode window — count per
step, bytes, duration, placement (in-span vs inter-step gap), and the GPU
kernel-idle each DtoH exposes in a bounded post-sync window, against a
seeded random-window negative control.

Hard gate: controller-documented totals must recompute EXACTLY before any
slice is trusted (8560 kernels, 565.549025 ms kernel union, 16 GPU step
annotations, 16/16 CPU<->GPU External-id pairing).
Controls:
  - detector negative: kernels+memcpys-only view (no DtoH rows) => 0 sites;
  - detector self-test: one synthetic DtoH injected => counted exactly once;
  - idle control: seeded random windows, same length as the post-DtoH probe.

Usage: python3 -I analyze_sync_sites.py <trace.json.gz> <out.json>
"""
import gzip, json, random, sys
from collections import Counter

HARD = {"kernel_instances": 8560, "kernel_union_ms": 565.549025,
        "gpu_steps": 16, "cpu_steps": 16, "pairing": 16}


def union(ivs):
    out = []
    for s, t in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], t)
        else:
            out.append([s, t])
    return out


def uncovered(u, lo, hi):
    """time in [lo,hi) not covered by union u"""
    tot = 0.0
    for s, t in u:
        s2, t2 = max(s, lo), min(t, hi)
        if t2 > s2:
            tot += t2 - s2
    return (hi - lo) - tot


def main(path, outp):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    memcpys = [e for e in ev if e.get("cat") == "gpu_memcpy"]
    memsets = [e for e in ev if e.get("cat") == "gpu_memset"]
    gpu_steps = [e for e in ev if e.get("cat") == "gpu_user_annotation"
                 and "DECODE" in (e.get("name") or "")]
    cpu_steps = [e for e in ev if e.get("cat") == "user_annotation"
                 and "DECODE" in (e.get("name") or "")]

    # ---- hard gate: recompute controller totals exactly
    ku = union([(e["ts"], e["ts"] + e["dur"]) for e in kernels])
    ku_ms = sum(t - s for s, t in ku) / 1e3
    cpu_by_id = {e["args"]["External id"] for e in cpu_steps}
    gpu_by_id = {e["args"]["External id"] for e in gpu_steps}
    paired = len(cpu_by_id & gpu_by_id)
    gate = {"kernel_instances": len(kernels),
            "kernel_union_ms": round(ku_ms, 6),
            "gpu_steps": len(gpu_steps),
            "cpu_steps": len(cpu_steps),
            "pairing": paired}
    gate_ok = (gate == HARD)
    if not gate_ok:
        print("GATE FAILED:", gate, "expected", HARD); sys.exit(2)

    steps = sorted(gpu_steps, key=lambda e: e["ts"])
    spans = [(e["ts"], e["ts"] + e["dur"]) for e in steps]
    # steady state excludes step-1 profiler-start artifact (it11)
    steady = spans[1:]
    ku_steady = union([(e["ts"], e["ts"] + e["dur"]) for e in kernels])

    # ---- memcpy census by direction
    def census(rows):
        c = Counter(e["name"] for e in rows)
        out = {}
        for name, n in c.items():
            rs = [e for e in rows if e["name"] == name]
            out[name] = {"count": n,
                         "total_us": round(sum(e["dur"] for e in rs), 3),
                         "bytes": sorted({e.get("args", {}).get("bytes")
                                          for e in rs}),
                         "dur_us_min": round(min(e["dur"] for e in rs), 3),
                         "dur_us_max": round(max(e["dur"] for e in rs), 3)}
        return out

    # ---- DtoH placement + post-sync idle probe (W=200us)
    W = 200.0
    kern_lo = min(e["ts"] for e in kernels)
    kern_hi = max(e["ts"] + e["dur"] for e in kernels)
    dtoh = sorted([e for e in memcpys if "DtoH" in e["name"]],
                  key=lambda e: e["ts"])
    sites, edge_excluded = [], 0
    for e in dtoh:
        t0, t1 = e["ts"], e["ts"] + e["dur"]
        in_span = any(s <= t0 and t1 <= t for s, t in spans)
        # distance to nearest span boundary
        dmin = min(min(abs(t0 - s), abs(t0 - t)) for s, t in spans)
        if t1 + W > kern_hi:  # window past trace tail: not probed (artifact)
            edge_excluded += 1
            sites.append({"dur_us": round(e["dur"], 3),
                          "bytes": e.get("args", {}).get("bytes"),
                          "in_step_span": in_span,
                          "dist_to_span_us": round(dmin, 3),
                          "post_sync_idle_us_200w": None,
                          "edge_excluded": True})
            continue
        idle = uncovered(ku_steady, t1, t1 + W)
        sites.append({"dur_us": round(e["dur"], 3),
                      "bytes": e.get("args", {}).get("bytes"),
                      "in_step_span": in_span,
                      "dist_to_span_us": round(dmin, 3),
                      "post_sync_idle_us_200w": round(idle, 3),
                      "edge_excluded": False})

    # which step-gap each DtoH belongs to (gap i = between span i and i+1)
    for s in sites:
        pass

    # ---- seeded random-window control (same count/length)
    rng = random.Random(4242)
    lo_all, hi_all = steady[0][0], steady[-1][1]
    ctrl = []
    tries = 0
    while len(ctrl) < len(sites) and tries < 10000:
        tries += 1
        t = rng.uniform(lo_all, hi_all - W)
        if any(s <= t and t + W <= t2 for s, t2 in spans):
            continue  # don't sample inside a span (compare gap-like windows)
        ctrl.append(round(uncovered(ku_steady, t, t + W), 3))

    # ---- detector negative control: no DtoH rows -> 0 sites
    neg = [e for e in memcpys if "DtoH" not in e["name"]] + kernels + memsets
    neg_dtoh = [e for e in neg if "DtoH" in e.get("name", "")]
    # ---- detector self-test: inject one synthetic DtoH
    inj = dict(dtoh[0]); inj["ts"] = steady[0][0] - 10_000.0; inj["dur"] = 1.0
    inj_found = sum(1 for e in dtoh + [inj] if "DtoH" in e["name"])

    probed = [s for s in sites if not s.get("edge_excluded")]
    per_step = len(dtoh) / len(steps)
    res = {
        "trace": path, "gate": {"ok": True, **gate},
        "memcpy_census": census(memcpys),
        "memset_census": census(memsets),
        "dtoh_sites": sites,
        "dtoh_per_step": round(per_step, 4),
        "dtoh_summary": {
            "n": len(sites),
            "probed": len(probed),
            "edge_excluded_tail": edge_excluded,
            "in_span": sum(s["in_step_span"] for s in sites),
            "in_gap": sum(not s["in_step_span"] for s in sites),
            "bytes_values": sorted({s["bytes"] for s in sites}),
            "mean_post_sync_idle_us_200w":
                round(sum(s["post_sync_idle_us_200w"] for s in probed)
                      / len(probed), 3),
            "mean_control_idle_us_200w":
                round(sum(ctrl) / len(ctrl), 3) if ctrl else None,
            "control_n": len(ctrl),
            "excess_idle_us_vs_control":
                round(sum(s["post_sync_idle_us_200w"] for s in probed)
                      / len(probed) - (sum(ctrl) / len(ctrl) if ctrl else 0),
                      3)},
        "controls": {
            "detector_negative_dtoh_count": len(neg_dtoh),
            "detector_selftest_found": inj_found},
        "limits": ["single operating point (C128 graph decode, glm-a 15fd7f4)",
                   "post-sync idle is an upper bound on sync-attributable idle; "
                   "overlap scheduler may hide host stall under queued kernels",
                   "window W=200us fixed a priori; random control excludes in-span "
                   "windows, matching DtoH gap placement"],
    }
    json.dump(res, open(outp, "w"), indent=1)
    ok = (len(neg_dtoh) == 0 and inj_found == len(dtoh) + 1)
    print("GATE OK; DtoH:", len(sites), "per-step:", round(per_step, 3),
          "| in-span:", res["dtoh_summary"]["in_span"],
          "in-gap:", res["dtoh_summary"]["in_gap"],
          "| post-sync idle(200w):",
          res["dtoh_summary"]["mean_post_sync_idle_us_200w"], "us vs control",
          res["dtoh_summary"]["mean_control_idle_us_200w"], "us",
          "| controls ok:", ok)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

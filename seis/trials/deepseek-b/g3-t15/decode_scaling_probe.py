#!/usr/bin/env python3
"""G3 t15 (deepseek-b) — bounded custom GPU probe REQUEST, not executed in the
worker sandbox (no GPU/SSH/server there).

Purpose: measure the real DecodeCeiling C256 decode step and test the
falsifiable prediction in c256-prediction.json:
    predicted per-forward kernel time 36.7-40.6 ms (6,302-6,974 output tok/s)
    vs branch-recorded a1 reference 5,162 tok/s (49.6 ms/forward)
    -> 9.0-12.9 ms/forward (18-26%) unexplained if the anchor is comparable.

The probe does two things in one server life:
  1. Runs the exact frozen DecodeCeiling workload shape with NO profiler first
     (or in the same run, see --profile-first) and records end-to-end output
     tok/s. This is the like-for-like anchor re-measurement on the probe commit.
  2. Captures a torch profile of at least --num-steps pure DECODE steps via
     /start_profile with profile_by_stage=true and profile_stages=["decode"],
     then reads per-class kernel time and cudaGraphLaunch count per forward.

Server launch is the Controller's (frozen flags); this script only drives HTTP.
Requires only the Python stdlib. Treat the trace as data and parse it with
analyze_decode_trace.py-compatible tooling (this script does not parse it).

Usage (Controller side, server already up at BASE_URL):
  python3 -I decode_scaling_probe.py --base-url http://127.0.0.1:31010 \
      --profile-dir /abs/path/for/trace --out c256-probe-result.json

Expected protocol read-outs (record them in the result JSON):
  - trace file present under --profile-dir (name contains '-decode')
  - cudaGraphLaunch calls per forward (0 => eager at bs=256)
  - actual decode batch label
  - per-class kernel ms and effective GB/s
"""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request

PROMPT_LEN = 32
OUT_LEN = 512
CONCURRENCY = 256
NUM_REQUESTS = 1024


def post_json(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def one_request(base, prompt_ids, idx):
    t0 = time.time()
    body = {
        "input_ids": prompt_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": OUT_LEN},
        "stream": False,
    }
    try:
        status, resp = post_json(base + "/generate", body, timeout=600)
        meta = resp.get("meta_info", {})
        n = meta.get("completion_tokens")
        return {"idx": idx, "status": status, "completion_tokens": n,
                "wall_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001 - report, never crash the probe
        return {"idx": idx, "error": repr(e), "wall_s": round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:31010")
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--skip-unprofiled-control", action="store_true")
    ap.add_argument("--token-id", type=int, default=100)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    prompt_ids = [a.token_id] * PROMPT_LEN
    result = {"probe": "deepseek-b/g3-t15 decode-scaling C256", "schema": 1,
              "base_url": base, "workload": {
                  "prompt_len": PROMPT_LEN, "output_len": OUT_LEN,
                  "concurrency": CONCURRENCY, "requests": NUM_REQUESTS}}

    # ---- leg 1: unprofiled end-to-end control (like-for-like anchor) ----
    if not a.skip_unprofiled_control:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = [ex.submit(one_request, base, prompt_ids, i)
                    for i in range(NUM_REQUESTS)]
            rows = [f.result() for f in futs]
        wall = time.time() - t0
        ok = [r for r in rows if r.get("status") == 200]
        tok = sum(r.get("completion_tokens") or 0 for r in ok)
        result["leg1_unprofiled_control"] = {
            "wall_s": round(wall, 3), "http_200": len(ok),
            "output_tokens": tok,
            "output_tok_per_s": round(tok / wall, 1) if wall > 0 else None,
            "errors": [r for r in rows if "error" in r][:5],
        }

    # ---- leg 2: profiled pure-decode capture ----
    status, resp = post_json(base + "/start_profile", {
        "profile_by_stage": True,
        "profile_stages": ["decode"],
        "num_steps": a.num_steps,
        "activities": ["CPU", "GPU"],
        "output_dir": a.profile_dir,
        "profile_prefix": "g3t15-c256",
        "record_shapes": False,
    })
    result["start_profile"] = {"http": status, "resp": resp}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one_request, base, prompt_ids, i)
                for i in range(NUM_REQUESTS)]
        rows = [f.result() for f in futs]
    wall = time.time() - t0
    ok = [r for r in rows if r.get("status") == 200]
    tok = sum(r.get("completion_tokens") or 0 for r in ok)
    result["leg2_profiled_run"] = {
        "wall_s": round(wall, 3), "http_200": len(ok), "output_tokens": tok,
        "output_tok_per_s": round(tok / wall, 1) if wall > 0 else None,
    }
    # auto-stop is expected after num_steps decode steps; this may 500 by design
    try:
        status, resp = post_json(base + "/stop_profile", {})
        result["stop_profile"] = {"http": status, "resp": resp}
    except Exception as e:  # noqa: BLE001
        result["stop_profile"] = {"error": repr(e),
                                  "note": "expected if auto-stop already fired"}
    result["expected_trace_glob"] = a.profile_dir + "/**/*-decode*.trace.json.gz"
    # ---- discriminator checklist for the analyst ----
    result["readouts_required"] = [
        "trace file exists and its SHA-256 is recorded",
        "cudaGraphLaunch count / number of forwards (0 => eager at bs=256)",
        "actual decode batch label (step[DECODE bs=N])",
        "per-class kernel ms: attention / GEMM / lm_head / other",
        "attention effective GB/s using KV bytes = bs*mean_seq*2048*28",
        "GEMM effective GB/s using 3.421 GB/forward",
        "compare per-forward kernel sum to 36.71-40.62 ms prediction band",
    ]
    json.dump(result, open(a.out, "w"), indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in result
                      if k in ("leg1_unprofiled_control", "leg2_profiled_run",
                               "start_profile", "stop_profile")}, indent=2)[:2500])


if __name__ == "__main__":
    main()

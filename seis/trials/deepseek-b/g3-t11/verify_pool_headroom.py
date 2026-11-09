#!/usr/bin/env python3
"""G3 t11 (deepseek-b): frozen-workload KV-pool headroom oracle.

Question (new, CPU-falsifiable): under the frozen launch, does the per-point
concurrency x (prompt + max output) demand exceed the 131072-token KV pool, and
therefore force the decode-retraction path? If yes, the low PrefillHeavy
throughput (branch record: 59.3 tok/s, TTFT ~4046 ms) may be dominated by
retract/re-prefill churn rather than by attention arithmetic.

This oracle does NOT execute SGLang and does NOT measure throughput. It:
  1. verifies, from the pinned source in this tree, the anchors that make the
     arithmetic meaningful (pool is the min(profiled, user cap); decode
     capacity gate; retract-decode call site and its observable counter;
     radix cache disabled so eviction is not a relief valve);
  2. computes exact per-point KV demand from the frozen contract parameters
     (passed in as provenance-tagged external constants, because configs/rsi
     is not present in the candidate sandbox);
  3. emits a verdict plus an explicit falsifier that the controller can run on
     EXISTING Fast artifacts (no new gate required): retraction counts.

Hard fail (exit 2) on any missing source anchor: a drifted tree must not
produce a claim. Stdlib only; run with -I.
"""

import argparse
import json
import os
import re
import sys

# --- Frozen contract (external, from the controller brief) -------------------
# Provenance: "G3 research candidate contract" / "Continuous Campaign 新证据
# 2026-11-09" frozen launch + workload table in the session prompt. These are
# NOT read from the repo, because configs/rsi/fast-gate.json is not mounted in
# the candidate sandbox.
FROZEN_LAUNCH = {
    "tp": 1,
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "context_length": 9216,
    "max_total_tokens": 131072,
    "mem_fraction_static": 0.6,
    "cuda_graph_max_bs_decode": 64,
    "chunked_prefill_size": 8192,
    "max_running_requests": 256,
    "max_queued_requests": 256,
    "attention_backend": "flashinfer",
    "disable_radix_cache": True,
}

# category: (input_tokens, output_tokens, [concurrencies], weight)
FROZEN_POINTS = {
    "short": (128, 32, [32, 128], 0.10),
    "medium": (512, 64, [32, 128], 0.15),
    "standard": (1024, 128, [32, 64], 0.15),
    "decode_heavy": (256, 512, [64, 128], 0.25),
    "prefill_heavy": (8192, 32, [16], 0.15),
    "decode_ceiling": (32, 512, [256], 0.20),
}

# Branch-recorded baselines (NOT re-measured by this oracle; from the branch
# learning / canonical register). Used only to frame what the arithmetic would
# have to explain, never as evidence for the oracle itself.
BRANCH_BASELINES = {
    "prefill_heavy_c16": {"tok_s": 59.3, "ttft_ms": 4046},
    "decode_ceiling_c256_abs_tok_s": 5145,
}

# --- Source anchors that must exist for the conclusion to be sound ----------
# (file, needle, why). Needles are exact substrings of the pinned tree.
ANCHORS = [
    # The user cap is an upper bound on the profiled pool, not a floor.
    ("python/sglang/srt/mem_cache/kv_cache_configurator.py",
     "token_capacity = min(token_capacity, user_limit)",
     "pool = min(profiled, --max-total-tokens)"),
    # Decode-capacity gate, called by the retraction loop.
    ("python/sglang/srt/managers/schedule_batch.py",
     "def check_decode_mem(",
     "decode capacity gate entry"),
    ("python/sglang/srt/managers/schedule_batch.py",
     "return self.token_to_kv_pool_allocator.check_decode_capacity(",
     "gate delegates to allocator capacity check"),
    ("python/sglang/srt/managers/schedule_batch.py",
     "def new_tokens_required_next_decode(",
     "per-decode-step token demand is page-rounded"),
    # Retraction exists and is observable.
    ("python/sglang/srt/managers/schedule_batch.py",
     "def retract_decode(self)",
     "retraction implementation"),
    ("python/sglang/srt/managers/scheduler.py",
     "retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode()",
     "scheduler call site"),
    ("python/sglang/srt/managers/scheduler.py",
     "self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)",
     "retraction count is observable"),
    ("python/sglang/srt/managers/scheduler.py",
     "msg_details = f\"#retracted_reqs: {len(retracted_reqs)}",
     "retraction count is logged"),
    # Radix cache disabled => no prefix-cache eviction relief; retraction is
    # the pressure release for the frozen workload.
    ("python/sglang/srt/mem_cache/kv_cache_builder.py",
     "disable=disable_radix_cache",
     "radix cache disabled in the frozen launch"),
    # Model-side window facts used only for the structural note.
    ("python/sglang/srt/models/spark2_5.py",
     "return config.sliding_window - 1",
     "sliding window is config-1 (511 for window=512)"),
    ("python/sglang/srt/models/spark2_5.py",
     "max_position_embeddings: int = 8192",
     "arch default max positions"),
]


def check_anchors(root):
    out = []
    ok = True
    for rel, needle, why in ANCHORS:
        path = os.path.join(root, rel)
        found = False
        line = None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i, text in enumerate(f, 1):
                    if needle in text:
                        found = True
                        line = i
                        break
        except OSError as exc:
            out.append({"file": rel, "needle": needle, "why": why,
                        "found": False, "error": str(exc)})
            ok = False
            continue
        out.append({"file": rel, "needle": needle, "why": why,
                    "found": found, "line": line})
        ok = ok and found
    return ok, out


def point_arithmetic(inp, out, conc, pool):
    prompt_demand = conc * inp
    max_demand = conc * (inp + out)
    headroom_after_prompt = pool - prompt_demand
    overcommit = max(0, max_demand - pool)
    # Total tokens per request at which cumulative demand exactly fills the
    # pool (integer floor). Beyond this, the C requests cannot all be resident.
    per_req_capacity = pool // conc
    first_starved_output = max(0, per_req_capacity - inp + 1)  # 1-indexed
    if max_demand <= pool:
        cls = "FITS_AT_MAX_OUTPUT"
    elif prompt_demand >= pool:
        cls = "PROMPT_ALONE_EXHAUSTS_POOL"
    else:
        cls = "OVERCOMMITS_DURING_DECODE"
    return {
        "concurrency": conc,
        "prompt_demand": prompt_demand,
        "max_demand": max_demand,
        "pool": pool,
        "headroom_after_prompt": headroom_after_prompt,
        "overcommit_tokens_at_max_output": overcommit,
        "tokens_per_request_when_pool_full": per_req_capacity,
        "first_output_token_index_that_cannot_all_be_resident": (
            first_starved_output if overcommit else None
        ),
        "classification": cls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ok, anchors = check_anchors(args.root)
    pool = FROZEN_LAUNCH["max_total_tokens"]

    points = {}
    for name, (inp, out, concs, weight) in FROZEN_POINTS.items():
        points[name] = {
            "input_tokens": inp,
            "output_tokens": out,
            "weight": weight,
            "cases": [point_arithmetic(inp, out, c, pool) for c in concs],
        }

    over = [
        (name, case)
        for name, p in points.items()
        for case in p["cases"]
        if case["classification"] != "FITS_AT_MAX_OUTPUT"
    ]
    prefill = points["prefill_heavy"]["cases"][0]
    ceiling = points["decode_ceiling"]["cases"][0]

    # Predictions that the controller can falsify on existing artifacts with
    # no new gate: retraction counters / '#retracted_reqs:' log lines.
    predictions = {
        "P1": (
            "If the pool arithmetic binds at runtime, the PrefillHeavy C16 "
            "point shows retraction activity (num_retracted_reqs>0 or "
            "'#retracted_reqs:' log lines), because prompt demand equals the "
            "pool exactly (131072) and every decode token is over capacity."
        ),
        "P2": (
            "If pool pressure is the mechanism, prefill_heavy retraction "
            "counts scale with decode steps (chunked prefill greedily refills "
            "the space freed by a retraction, starving the next decode step), "
            "not with request arrivals."
        ),
        "P3": (
            "DecodeCeiling C256 only enters the over-committed regime after "
            "request length exceeds 512 tokens total (32 prompt + 480 "
            "output); its retraction activity, if any, should appear in the "
            "last ~6.25% of each generation, not at the start."
        ),
        "P4": (
            "Control points short/medium/standard/decode_heavy (all "
            "<=98304 token demand) should show ~zero retractions; a nonzero "
            "count there falsifies the capacity reading and points at a "
            "different pressure source."
        ),
    }

    out = {
        "trial": "G3 t11 deepseek-b",
        "oracle": "frozen-workload KV-pool headroom",
        "executed": True,
        "provenance": {
            "launch": "controller brief (frozen fast-gate launch), not read "
                      "from repo (configs/rsi absent in candidate sandbox)",
            "workload": "controller brief frozen workload table",
            "baselines": "branch learning / canonical register, not "
                         "re-measured here",
        },
        "frozen_launch": FROZEN_LAUNCH,
        "branch_baselines_external": BRANCH_BASELINES,
        "source_anchor_checks": anchors,
        "all_anchors_present": ok,
        "per_point": points,
        "overcommitted_cases": [
            {"category": n, **c} for n, c in over
        ],
        "headline": {
            "prefill_heavy_c16_prompt_demand": prefill["prompt_demand"],
            "prefill_heavy_c16_pool": pool,
            "prefill_heavy_c16_prompt_headroom": prefill["headroom_after_prompt"],
            "prefill_heavy_c16_overcommit_at_max_output": prefill["overcommit_tokens_at_max_output"],
            "decode_ceiling_c256_overcommit_at_max_output": ceiling["overcommit_tokens_at_max_output"],
            "decode_ceiling_c256_first_starved_output_index": ceiling["first_output_token_index_that_cannot_all_be_resident"],
        },
        "predictions_for_controller_artifacts": predictions,
        "limitations": [
            "No SGLang execution, no GPU, no token throughput measured here.",
            "The frozen workload constants are external (controller brief); "
            "if the actual harness uses different C or output budgets the "
            "arithmetic changes.",
            "The pool value 131072 assumes the profiled capacity at "
            "--mem-fraction-static 0.6 exceeds the user cap, i.e. the cap "
            "binds (source: min(profiled, user_limit)). The startup log line "
            "'max_total_num_tokens=' is the authoritative check.",
            "Hybrid SWA exposes separate full and SWA per-layer pools; this "
            "oracle bounds the FULL-attention pool only. Whether the binding "
            "allocator is the full pool or the SWA pool is not resolved here "
            "(swa.py sizes both; --max-total-tokens caps the full capacity).",
            "The scheduler's new-token-ratio reservation and chunked-prefill "
            "admission policy are NOT modeled; they can change whether the "
            "pool actually fills. P1-P4 are the runtime falsifiers.",
            "Requests may end early on EOS in the real harness; the contract "
            "states fixed input/output token counts and the fixed-output-"
            "budget gate, so full-length output is assumed.",
            "Branch baselines (59.3 tok/s, 4046 ms, 5145 tok/s) are recorded "
            "history, not re-measured.",
        ],
        "verdict": (
            "ARITHMETIC_BINDS" if (
                prefill["classification"] == "PROMPT_ALONE_EXHAUSTS_POOL"
                or ceiling["classification"] != "FITS_AT_MAX_OUTPUT"
            ) else "ARITHMETIC_DOES_NOT_BIND"
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

    print(json.dumps({
        "verdict": out["verdict"],
        "all_anchors_present": ok,
        "overcommitted_cases": [
            (n, c["concurrency"], c["overcommit_tokens_at_max_output"],
             c["classification"]) for n, c in over
        ],
        "prefill_heavy": {
            "prompt_demand": prefill["prompt_demand"],
            "pool": pool,
            "prompt_headroom": prefill["headroom_after_prompt"],
            "overcommit_at_max_output": prefill["overcommit_tokens_at_max_output"],
        },
        "decode_ceiling": {
            "overcommit_at_max_output": ceiling["overcommit_tokens_at_max_output"],
            "first_starved_output_index": ceiling["first_output_token_index_that_cannot_all_be_resident"],
        },
        "out": os.path.abspath(args.out),
    }, indent=2))

    # Hard gate: no claim on a drifted tree.
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

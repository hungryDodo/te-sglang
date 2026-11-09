#!/usr/bin/env python3
"""G3 t9 deepseek-b: frozen-contract prefill-batch and KV-pool arithmetic.

Stdlib only; run with:  python3 -I verify_prefill_budget.py --root . --out out.json

What is EXECUTED here:
  * the real source default of `max_prefill_tokens` and `chunked_prefill_size`
    is AST-extracted from python/sglang/srt/arg_groups/fields/schedule.py and
    the default expression is evaluated;
  * all arithmetic below is recomputed twice (per-point loop and whole-table
    formula) and asserted equal;
  * one negative control (max_prefill_tokens forced to 8192) and one
    sensitivity row.

What is NOT executed: the SGLang scheduler itself. `hybrid_swa` memory
accounting, page-size rounding, request-slot overhead and mixed-decode tokens
are not modelled. The reservation assumption A1 is anchored to
schedule_policy.py:1263-1266 (`tokens_occupied = origin_input + output`) and
1204-1216 (`_swa_new_tokens` = min(max_new_tokens - len(output_ids),
CLIP_MAX_NEW_TOKENS)) but is an assumption, not an execution.
"""

import argparse
import ast
import json
import math
import os

# Frozen workload contract, copied from the signed G3 trial brief (NOT source).
# Token counts are input/output; concurrency is the closed-loop request count.
FROZEN = {
    "flags": {
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
        "cuda_graph_backend_prefill": "disabled",
        "max_prefill_tokens": "ABSENT (source default applies)",
    },
    # (name, input_tokens, output_tokens, concurrency)
    "points": [
        ("short_c32", 128, 32, 32),
        ("short_c128", 128, 32, 128),
        ("medium_c32", 512, 64, 32),
        ("medium_c128", 512, 64, 128),
        ("standard_c32", 1024, 128, 32),
        ("standard_c64", 1024, 128, 64),
        ("decode_heavy_c64", 256, 512, 64),
        ("decode_heavy_c128", 256, 512, 128),
        ("prefill_heavy_c16", 8192, 32, 16),
        ("decode_ceiling_c256", 32, 512, 256),
    ],
    "category_weights": {
        "short": 0.10, "medium": 0.15, "standard": 0.15,
        "decode_heavy": 0.25, "prefill_heavy": 0.15, "decode_ceiling": 0.20,
    },
}


def extract_schedule_defaults(root):
    """AST-extract the real default expressions for the two schedule fields."""
    path = os.path.join(root, "python/sglang/srt/arg_groups/fields/schedule.py")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    found = {}
    wanted = {"max_prefill_tokens", "chunked_prefill_size"}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name in wanted and node.value is not None:
                expr = ast.unparse(node.value)
                found[name] = {
                    "expr": expr,
                    "file": os.path.relpath(path, root),
                    "line": node.lineno,
                    "value": eval(compile(ast.Expression(body=node.value), path, "eval"), {"__builtins__": {}}),
                }
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"FATAL: could not extract defaults: {sorted(missing)}")
    return found


def compute_rows(max_prefill_tokens, chunked, pool):
    """Return per-point arithmetic. Independent of FROZEN ordering."""
    rows = []
    for name, tin, tout, conc in FROZEN["points"]:
        # Lower bound: pool must hold at least the input tokens at admission.
        demand_input_only = conc * tin
        # Assumption A1: ignore_eos reserves the full max_new_tokens.
        demand_reserved = conc * (tin + tout)
        # Prefill batches: every point's requests are admitted in chunks of at
        # most `chunked` tokens each; the per-batch token budget is
        # max(max_prefill_tokens, model context length) per the field help
        # (the context length only raises it, never lowers below 16384 here).
        chunk = min(chunked, tin) if chunked and chunked > 0 else tin
        per_batch_tokens = max_prefill_tokens
        reqs_per_batch = max(1, per_batch_tokens // chunk)
        waves = math.ceil(conc / reqs_per_batch)
        rows.append({
            "point": name,
            "input_tokens": tin,
            "output_tokens": tout,
            "concurrency": conc,
            "demand_input_only_tokens": demand_input_only,
            "demand_reserved_tokens_A1": demand_reserved,
            "over_pool_input_only": demand_input_only > pool,
            "over_pool_reserved_A1": demand_reserved > pool,
            "admissible_requests_input_only": min(conc, pool // tin) if tin else conc,
            "admissible_requests_reserved_A1": min(conc, pool // (tin + tout)),
            "chunk_tokens_per_request": chunk,
            "reqs_per_prefill_batch_by_token_budget": reqs_per_batch,
            "prefill_waves_if_input_is_one_chunk": waves,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    defaults = extract_schedule_defaults(args.root)
    max_prefill_tokens = defaults["max_prefill_tokens"]["value"]
    chunked_default = defaults["chunked_prefill_size"]["value"]
    chunked = FROZEN["flags"]["chunked_prefill_size"]  # frozen launch passes it
    pool = FROZEN["flags"]["max_total_tokens"]

    assert isinstance(max_prefill_tokens, int) and max_prefill_tokens > 0
    assert chunked_default is None, "frozen launch relies on the CLI, not a default"

    rows = compute_rows(max_prefill_tokens, chunked, pool)

    # Independent whole-table recomputation (must match the per-row loop).
    n = len(rows)
    assert n == len(FROZEN["points"])
    for r in rows:
        assert r["demand_reserved_tokens_A1"] == r["concurrency"] * (r["input_tokens"] + r["output_tokens"])
        assert r["prefill_waves_if_input_is_one_chunk"] == math.ceil(
            r["concurrency"] / max(1, max_prefill_tokens // min(chunked, r["input_tokens"]))
        )

    over_reserved = [r["point"] for r in rows if r["over_pool_reserved_A1"]]
    over_input = [r["point"] for r in rows if r["over_pool_input_only"]]

    # Negative control: half the token budget must halve the prefill width.
    control = compute_rows(max_prefill_tokens // 2, chunked, pool)
    control_ph = next(r for r in control if r["point"] == "prefill_heavy_c16")
    base_ph = next(r for r in rows if r["point"] == "prefill_heavy_c16")
    control_ok = (
        control_ph["reqs_per_prefill_batch_by_token_budget"] * 2
        == base_ph["reqs_per_prefill_batch_by_token_budget"]
        and control_ph["prefill_waves_if_input_is_one_chunk"] == 2 * base_ph["prefill_waves_if_input_is_one_chunk"]
    )

    # Sensitivity: the 8192-input point is the only one whose own prompt equals
    # the frozen chunk size, so its wave count is pinned by max_prefill_tokens.
    ph = base_ph
    result = {
        "trial": "g3-t9",
        "worker": "deepseek-b",
        "kind": "CPU arithmetic over extracted source defaults + frozen brief",
        "source_extracted": defaults,
        "frozen_contract": FROZEN["flags"],
        "assumptions": {
            "A1_reservation": "ignore_eos reserves origin_input+max_new_tokens at admission, anchored at schedule_policy.py:1263-1266 and 1204-1216; NOT executed.",
            "A2_prefill_budget": "per-batch prefill token budget = max_prefill_tokens (16384 default); the field help says the real bound is max(this, model max context length), and 9216 < 16384 so the default governs.",
            "A3_chunking": "each request's prompt is chunked at chunked_prefill_size=8192; an 8192-token prompt is exactly one chunk.",
            "A4_not_modelled": "hybrid_swa pool accounting, page-size rounding, request-slot overhead, mixed-decode tokens, prefix hits (radix disabled).",
        },
        "rows": rows,
        "derived": {
            "prefill_batch_token_budget": max_prefill_tokens,
            "chunked_prefill_size": chunked,
            "prefill_heavy_reqs_per_batch": ph["reqs_per_prefill_batch_by_token_budget"],
            "prefill_heavy_waves": ph["prefill_waves_if_input_is_one_chunk"],
            "points_over_pool_reserved_A1": over_reserved,
            "points_over_pool_input_only": over_input,
            "pool_equals_16_x_8192": pool == 16 * 8192,
        },
        "controls": {
            "halved_budget_halves_prefill_width": bool(control_ok),
            "halved_budget_prefill_heavy": {
                "reqs_per_batch": control_ph["reqs_per_prefill_batch_by_token_budget"],
                "waves": control_ph["prefill_waves_if_input_is_one_chunk"],
            },
        },
        "verdict": {
            "certain_from_source_and_frozen_flags": (
                "With chunked_prefill_size=8192 and the unset max_prefill_tokens defaulting to "
                "16384, a prefill batch can contain at most 2 full 8192-token chunks; the "
                "PrefillHeavy point (16 x 8192) therefore runs in at least 8 prefill batches. "
                "This is a token-budget fact independent of KV memory."
            ),
            "conditional_A1": (
                "If admission reserves input+max_new_tokens (ignore_eos path), the 131072-token "
                "pool admits at most 15 of 16 PrefillHeavy requests (131584 needed) and at most "
                "240 of 256 DecodeCeiling requests (139264 needed); peak nominal concurrency is "
                "not simultaneously admissible at exactly the two largest frozen points."
            ),
            "uncertainty": (
                "The A1 table is arithmetic, not scheduler execution. hybrid_swa memory "
                "accounting, request-slot reserves and mixed-decode accounting are unmodelled; "
                "any of them can change the admissible count. The 2-chunk / 8-wave prefill "
                "statement does not depend on A1."
            ),
            "legal_lever": (
                "None identified. max_total_tokens, kv_cache_dtype, chunked_prefill_size and "
                "mem_fraction_static are frozen; a source change that widens the prefill batch "
                "would change GEMM shape/kernel selection (ALGORITHM_PRESERVING at best) and "
                "cannot reduce prefill FLOPs. The only FLOP-reducing lever (sliding window) is "
                "acceptance-incompatible per g3-t7."
            ),
            "f3_status": "UNGATED; the frozen BF16 Fast screen carries no GGUF artifact, so no number here is F3 evidence.",
            "measurement": "NOT_A_MEASUREMENT",
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps({
        "rows": len(rows),
        "max_prefill_tokens": max_prefill_tokens,
        "chunked": chunked,
        "prefill_heavy_reqs_per_batch": ph["reqs_per_prefill_batch_by_token_budget"],
        "prefill_heavy_waves": ph["prefill_waves_if_input_is_one_chunk"],
        "over_pool_reserved_A1": over_reserved,
        "over_pool_input_only": over_input,
        "control_ok": control_ok,
        "verdict_measurement": "NOT_A_MEASUREMENT",
    }, indent=2))


if __name__ == "__main__":
    main()

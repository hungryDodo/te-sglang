#!/usr/bin/env python3
"""G3 t7 — GPU-free closure oracle for the two remaining deepseek-b "levers".

Executes, against the real candidate tree (stdlib only, no torch/gguf):

1. Source-extracted Spark2.5 shape accounting: config defaults and the
   layer-type pattern are parsed with ``ast`` from
   ``python/sglang/srt/configs/spark2_5.py`` and the per-layer linear shapes
   are taken from ``python/sglang/srt/models/spark2_5.py`` (anchor lines are
   asserted).  Outputs per-matrix and per-model weight bytes for BF16 / Q8_0 /
   Q4_K and compares every real matrix to the GB10 L2 capacity (<25 MB).

2. Prefill FLOP accounting for the frozen long-context point with the same
   config: dense vs full-attention vs windowed-attention FLOPs for
   S in {512..8192}, i.e. the ceiling of the sliding-window lever and its
   exact-token acceptance status.

Nothing here is a performance measurement.  The numbers are arithmetic over
source-declared shapes; every assumption is emitted in ``assumptions`` so the
controller can reject or re-parameterise it.
"""
import argparse
import ast
import json
import os
import re

CONFIG_PATH = "python/sglang/srt/configs/spark2_5.py"
MODEL_PATH = "python/sglang/srt/models/spark2_5.py"
GGUF_PATH = "python/sglang/srt/layers/quantization/gguf.py"

L2_BYTES = 25 * 1024 * 1024  # branch/g2-t1: GB10 L2 <25 MB
# GGML block sizes (bytes per weight): Q8_0 = 34 B / 32 w; Q4_K = 144 B / 256 w.
BYTES_PER_WEIGHT = {"bf16": 2.0, "q8_0": 34.0 / 32.0, "q4_k": 144.0 / 256.0}


def _defaults(path):
    """Extract float/int defaults from Spark2_5Config.__init__ via ast."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Spark2_5Config":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    args = fn.args.args
                    defs = [None] * (len(args) - len(fn.args.defaults)) + list(
                        fn.args.defaults
                    )
                    for a, d in zip(args, defs):
                        if isinstance(d, ast.Constant) and isinstance(
                            d.value, (int, float)
                        ):
                            out[a.arg] = d.value
    return out


def _layer_pattern(path, n_layers):
    """Extract and execute the real layer_types list-comprehension."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    comp = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "layer_types":
                    if isinstance(node.value, ast.ListComp):
                        comp = node.value
    if comp is None:
        raise AssertionError("layer_types comprehension not found")
    class _Cfg:
        num_hidden_layers = n_layers
    expr = ast.Expression(body=comp)
    ast.fix_missing_locations(expr)
    ns = {"self": _Cfg, "bool": bool, "range": range}
    return eval(compile(expr, "<layer_types>", "eval"), ns)


def _anchor(path, lineno, needle):
    lines = open(path, encoding="utf-8").read().splitlines()
    got = lines[lineno - 1]
    if needle not in got:
        raise AssertionError(f"{path}:{lineno} lacks {needle!r}: {got!r}")
    return got.strip()


def _linear_shapes(model_path, cfg):
    """Per-layer linear (out_features, in_features) from real source anchors."""
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    d = cfg["head_dim"]
    _anchor(model_path, 135, "QKVParallelLinear(")
    _anchor(model_path, 146, "ColumnParallelLinear(")
    _anchor(model_path, 155, "RowParallelLinear(")
    _anchor(model_path, 61, "MergedColumnParallelLinear(")
    _anchor(model_path, 68, "RowParallelLinear(")
    qkv = (nh * d + 2 * nkv * d, H)          # q + k + v fused
    g_proj = (nh, H)                          # HEADWISE gate: 1 scalar/head
    out_proj = (nh * d, H)
    gate_up = (2 * I, H)
    down = (I, H)
    return {
        "q_k_v_proj": qkv,
        "g_proj": g_proj,
        "out_proj": out_proj,
        "mlp_gate_up": gate_up,
        "mlp_down": down,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    cfg_path = os.path.join(a.root, CONFIG_PATH)
    model_path = os.path.join(a.root, MODEL_PATH)
    cfg = _defaults(cfg_path)
    need = ["hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "num_hidden_layers", "head_dim",
            "sliding_window", "vocab_size"]
    missing = [k for k in need if k not in cfg]
    if missing:
        raise AssertionError(f"missing config defaults: {missing}")
    L = cfg["num_hidden_layers"]
    pattern = _layer_pattern(cfg_path, L)
    n_swa = sum(1 for t in pattern if t == "sliding_attention")
    n_full = L - n_swa
    shapes = _linear_shapes(model_path, cfg)
    V, H = cfg["vocab_size"], cfg["hidden_size"]

    # ---- weight accounting -------------------------------------------------
    per_layer = sum(o * i for o, i in shapes.values())
    embed = V * H
    total_tied = per_layer * L + embed
    total_untied = total_tied + embed
    quant_bytes = {}
    for q, bpw in BYTES_PER_WEIGHT.items():
        quant_bytes[q] = {
            "per_layer_bytes": per_layer * bpw,
            "model_tied_bytes": total_tied * bpw,
            "model_untied_bytes": total_untied * bpw,
        }
    matrix_rows = []
    for name, (o, i) in shapes.items():
        row = {"name": name, "out_features": o, "in_features": i,
               "params": o * i,
               "bf16_bytes": o * i * BYTES_PER_WEIGHT["bf16"],
               "q8_0_bytes": o * i * BYTES_PER_WEIGHT["q8_0"]}
        row["bf16_exceeds_l2"] = row["bf16_bytes"] > L2_BYTES
        row["exceeds_l2"] = row["q8_0_bytes"] > L2_BYTES
        row["rows_band"] = "rows>5120" if o > 5120 else "rows<=5120"
        matrix_rows.append(row)
    embed_row = {"name": "token_embd(tied)", "out_features": V,
                 "in_features": H, "params": embed,
                 "bf16_bytes": embed * 2.0,
                 "q8_0_bytes": embed * BYTES_PER_WEIGHT["q8_0"],
                 "bf16_exceeds_l2": embed * 2.0 > L2_BYTES,
                 "exceeds_l2": embed * BYTES_PER_WEIGHT["q8_0"] > L2_BYTES,
                 "rows_band": "rows>5120"}
    matrix_rows.append(embed_row)
    g2t1_microbench_max_weight_bytes = 14500000  # g2-t1: 4.5 / 14.5 MB weights
    min_real_q8_bytes = min(r["q8_0_bytes"] for r in matrix_rows
                            if r["name"] != "g_proj")

    # ---- prefill FLOP accounting ------------------------------------------
    W = cfg["sliding_window"] - 1  # source: _get_attention_sliding_window_size
    dense_flops_per_token = 2 * (total_tied - embed)  # per-token GEMMs; emb is a gather, lm_head last-token only
    prefill = []
    for S in (512, 1024, 2048, 4096, 8192, 8191, 8192):
        if any(r["seq"] == S for r in prefill):
            continue
        dense = dense_flops_per_token * S
        attn_full = 4 * S * S * H * L          # QK^T + AV, all layers
        # reference path: extend_no_prefix passes no window_left (branch t2),
        # so all L layers run full attention.  Windowed ceiling applies the
        # 511 window to the n_swa sliding layers only.
        attn_win = 4 * S * min(S, W + 1) * H * n_swa + 4 * S * S * H * n_full
        total_ref = dense + attn_full
        total_win = dense + attn_win
        prefill.append({
            "seq": S,
            "dense_tflop": dense / 1e12,
            "attn_full_tflop": attn_full / 1e12,
            "attn_windowed_tflop": attn_win / 1e12,
            "attn_share_full": attn_full / total_ref,
            "window_speedup_ceiling": total_ref / total_win,
            "window_binds": S > W + 1,
        })

    # ---- verdicts (kept strictly to what the arithmetic supports) ---------
    real_matrices_over_l2 = [r["name"] for r in matrix_rows
                             if r["bf16_exceeds_l2"]]
    verdicts = {
        "formal_gate_is_bf16_safetensors": True,  # controller applicability-001
        "gguf_kernel_paths_score_relevant": False,
        "gguf_route_note": (
            "The frozen formal Gate loads BF16 safetensors revision "
            "e6406af... with no GGUF artifact and no --quantization gguf "
            "(controller applicability-001). Therefore neither f263262 nor "
            "any load-time-dequant GGUF change can move aggregate output "
            "tokens/s under the frozen contract, independent of its speed."
        ),
        "f3_changed_cells_at_batch_ge_16": 0,  # executed by g3-t5
        "f3_status": "UNGATED_AND_SCORE_UNREACHABLE_UNDER_FORMAL_GATE",
        "g2t1_ceiling_measured_on_l2_resident_weights": {
            "microbench_max_weight_bytes": g2t1_microbench_max_weight_bytes,
            "smallest_real_q8_0_matrix_bytes": min_real_q8_bytes,
            "every_real_bf16_matrix_exceeds_l2": len(real_matrices_over_l2),
            "real_bf16_matrices_over_l2": real_matrices_over_l2,
            "transfer_to_real_layers_allowed": False,
        },
        "sliding_window_prefill_lever": {
            "mechanism_share_at_8192": next(
                r["attn_share_full"] for r in prefill if r["seq"] == 8192),
            "window_speedup_ceiling_at_8192": next(
                r["window_speedup_ceiling"] for r in prefill if r["seq"] == 8192),
            "acceptance": "IMPOSSIBLE_UNDER_FROZEN_EXACT_TOKEN_BASELINE",
            "why": (
                "The pinned reference wheel executes the same extend_no_prefix "
                "path without window_left (branch trial-2, source-anchored). "
                "Applying the 511 window changes the attention output of every "
                "position >511 in 21/28 layers, so greedy tokens are not "
                "preserved; valid_exact_token_baseline compares against the "
                "reference output. A >512-token parity harness cannot rescue "
                "this: parity is defined against the same full-attention "
                "reference."
            ),
        },
        "bf16_engine_lane": {
            "status": "NO_PROMOTABLE_LEVER_ESTABLISHED",
            "decode": "roofline-closed on a4 (branch trial-1/2 measurements)",
            "prefill": "compute-bound; only larger lever is the semantics-"
                       "blocked sliding window above",
        },
    }
    payload = {
        "config_defaults": {k: cfg[k] for k in need},
        "layer_types": {"n_layers": L, "n_sliding": n_swa, "n_full": n_full,
                        "pattern_source": "executed real list-comp"},
        "sliding_window_exclusive": W,
        "linear_shapes": shapes,
        "params": {"per_layer": per_layer, "embed": embed,
                   "model_tied": total_tied, "model_untied": total_untied},
        "weight_bytes": quant_bytes,
        "matrices": matrix_rows,
        "prefill_flops": prefill,
        "verdicts": verdicts,
        "assumptions": {
            "l2_bytes": L2_BYTES,
            "bytes_per_weight": BYTES_PER_WEIGHT,
            "tied_embeddings": "model name 1.7B matches model_tied; both "
                               "tied and untied totals are emitted",
            "flops": "dense 2*(params-embedding)*tokens (embedding is a gather and "
                     "lm_head is last-token only); attention 4*S*eff_len*H per "
                     "layer (QK^T+AV), no MLP/attention double count of the "
                     "output projection beyond params",
            "not_a_measurement": True,
        },
        "source_anchors": {
            "config": CONFIG_PATH,
            "model": MODEL_PATH,
            "gguf": GGUF_PATH,
        },
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    if a.out == "-":
        print(text)
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

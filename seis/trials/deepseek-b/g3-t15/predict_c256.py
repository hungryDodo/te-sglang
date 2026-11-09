#!/usr/bin/env python3
"""G3 t15: from the MEASURED per-class bandwidth efficiencies of the real C128
decode trace, predict the DecodeCeiling C256 per-forward time and compare it
with the branch-recorded a1 reference throughput.

This is a planning calculation with explicit assumptions, not a measurement.
Its purpose is to state a falsifiable prediction for decode_scaling_probe.py.

Inputs
  --analysis  decode-trace-analysis.json produced by analyze_decode_trace.py
  --out       JSON output

Geometry (frozen workload contract):
  DecodeCeiling: prompt 32, output 512, concurrency 256, requests 1024,
  max_running_requests 256 -> steady-state batch 256, mean seq during decode
  = 32 + (1+512)/2 = 288.5 keys (window 512 does not bind below seq 512).
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.analysis))

    cfg = d["q3b_config_and_signature_checks"]
    bw = d["q3_bandwidth_accounting"]
    bs = 256
    mean_seq = 32 + (1 + 512) / 2  # 288.5
    kv_heads = cfg["config_defaults"]["num_key_value_heads"]
    head_dim = cfg["config_defaults"]["head_dim"]
    layers = cfg["config_defaults"]["num_hidden_layers"]

    kv_per_forward = bs * mean_seq * 2 * kv_heads * head_dim * 2 * layers
    attn_gbs = [b["attention_effective_gbs"] for b in bw["attention"]]
    attn_lo, attn_hi = kv_per_forward / (max(attn_gbs) * 1e9) * 1e3, \
        kv_per_forward / (min(attn_gbs) * 1e9) * 1e3
    gemm_bytes = (bw["per_layer_gemm"]["bytes_16_forwards"]
                  + bw["lm_head"]["bytes_16_forwards"]) / 16
    gemm_gbs = [bw["per_layer_gemm"]["effective_gbs"], bw["lm_head"]["effective_gbs"]]
    gemm_lo, gemm_hi = gemm_bytes / (max(gemm_gbs) * 1e9) * 1e3, \
        gemm_bytes / (min(gemm_gbs) * 1e9) * 1e3
    # "other" kernels measured at C128, scaled linearly with batch 128 -> 256
    c128_forward_ms = bw["whole_step_16_forwards"]["per_forward_ms"]
    c128_other_ms = c128_forward_ms * d["q5_findings"]["all_other_share_of_kernel_time"]
    other_ms = c128_other_ms * bs / 128
    pred_lo = attn_lo + gemm_lo + other_ms * 0.7
    pred_hi = attn_hi + gemm_hi + other_ms * 1.3

    anchor_tok_s = 5162.0  # branch-recorded a1 reference at DecodeCeiling C256
    anchor_forward_ms = bs / anchor_tok_s * 1e3

    out = {
        "schema": 1,
        "trial": "deepseek-b/g3-t15",
        "kind": "PLANNING_PREDICTION (not a measurement)",
        "source_analysis": a.analysis,
        "geometry": {
            "batch": bs, "mean_seq": mean_seq,
            "kv_bytes_per_forward": round(kv_per_forward),
            "gemm_bytes_per_forward": round(gemm_bytes),
        },
        "measured_c128_efficiencies_gbs": {
            "attention": attn_gbs, "gemm": gemm_gbs,
            "other_ms_per_forward_at_bs128": round(c128_other_ms, 3),
        },
        "prediction_per_forward_ms": {
            "attention": [round(attn_lo, 2), round(attn_hi, 2)],
            "gemm": [round(gemm_lo, 2), round(gemm_hi, 2)],
            "other_scaled": round(other_ms, 2),
            "total": [round(pred_lo, 2), round(pred_hi, 2)],
        },
        "prediction_throughput_tok_s": [round(bs / pred_hi * 1e3),
                                        round(bs / pred_lo * 1e3)],
        "branch_recorded_a1_reference_tok_s": anchor_tok_s,
        "anchor_implied_per_forward_ms": round(anchor_forward_ms, 2),
        "implied_gap_ms": [round(anchor_forward_ms - pred_hi, 2),
                           round(anchor_forward_ms - pred_lo, 2)],
        "implied_gap_pct_of_anchor": [
            round((anchor_forward_ms - pred_hi) / anchor_forward_ms * 100, 1),
            round((anchor_forward_ms - pred_lo) / anchor_forward_ms * 100, 1)],
        "falsifiable_discriminator": {
            "if_measured_per_forward_ms_in_prediction_band":
                "the C128 trace's per-byte efficiencies transfer to C256; the "
                "anchor comparison is not like-for-like (different run conditions "
                "or an unmodelled wall-time term) and there is no C256 kernel "
                "bottleneck to chase.",
            "if_measured_per_forward_ms_is_45_to_52":
                "the ~10-13 ms excess is real at bs=256; attribute it by kernel "
                "class from the same trace (attention vs GEMM vs other vs host "
                "gaps / eager-vs-graph) before proposing any patch.",
            "additional_readouts":
                "cudaGraphLaunch count per forward (does bs=256 replay a graph "
                "under --cuda-graph-max-bs-decode 64?), actual batch label, "
                "per-class effective GB/s.",
        },
        "assumptions_and_limits": [
            "C128 per-class GB/s from node a1 commit 15fd7f4 are assumed to hold "
            "on the C256 point; a1 reference 5162 tok/s is branch-recorded, not "
            "re-measured here.",
            "Mean decode seq 288.5 assumes the 512 output tokens are uniformly "
            "spread; the sliding-window clamp does not bind below seq 512.",
            "'other' kernels are scaled linearly 128->256 with a +/-30% band.",
            "The anchor is a whole-run aggregate (includes 2% prefill and any "
            "tail/queueing); the prediction is a steady-state per-forward band.",
        ],
    }
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(json.dumps({
        "prediction_throughput_tok_s": out["prediction_throughput_tok_s"],
        "anchor_tok_s": anchor_tok_s,
        "implied_gap_ms": out["implied_gap_ms"],
        "implied_gap_pct": out["implied_gap_pct_of_anchor"],
    }, indent=2))


if __name__ == "__main__":
    main()

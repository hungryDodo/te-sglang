#!/usr/bin/env python3
"""GPU numerical regression for the Spark2_5 headwise attention-output gate.

STATUS: PREPARED EVIDENCE ONLY — NOT_RUN_BY_WORKER.
The worker (glm-b, round 3) has no GPU access. The trusted Controller must run
this script on the actual CUDA + native-wheel environment before any numeric,
performance, or quality interpretation.

What it does (when run by Controller):
  - Loads the ACTUAL helper module (python/sglang/srt/layers/headwise_attn_output_gate.py)
    by file path, including the actual Triton JIT kernel and the preserved exact
    eager baseline `headwise_attn_output_gate_baseline`.
  - For each case, calls the actual baseline and the actual fused `apply` on the
    IDENTICAL input tensors (baseline first, then fused), snapshots inputs
    beforehand and verifies neither path mutated them, verifies the fused output
    is a NEW tensor (no alias), and reports exact-equality, mismatch count, max
    absolute difference, NaN/Inf counts, and first differing flat indices —
    raw numbers only, NO tolerance-based pass/fail and NO quality verdict.
  - Also reports (diagnostic only) how often an intentionally UNROUNDED
    fp32-gate product would differ from the baseline on the same inputs, to show
    whether these inputs can expose a missing intermediate rounding at all.
  - Covers: varied (tokens, heads, head_dim) incl. empty batch, BF16/FP16/FP32,
    rounding-boundary / large +/- / zero / tiny values, non-contiguous fallback,
    CPU-tensor fallback, gate-dtype-mismatch fallback, 3D-input fallback.

Exit code: 0 if all cases ran and no structural violation occurred (input
mutation, output aliasing, crash). Exact mismatches are REPORTED, not hidden,
and do not by themselves change the exit code; read the printed report.

Limitations: numerical equality here would still not establish task quality,
performance, or all-backend/graph-replay behavior; kernel-launch avoidance for
the empty batch is not directly observable in this script.
"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "python/sglang/srt/layers/headwise_attn_output_gate.py"

spec = importlib.util.spec_from_file_location("headwise_attn_output_gate_gpu_test", str(HELPER_PATH))
hw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hw
spec.loader.exec_module(hw)

REPORT = []
STRUCTURAL_FAILURE = False


def load_helper_module():
    return hw


def case_record(name, **kw):
    REPORT.append({"case": name, **kw})


def snapshots_equal(t, snap):
    return t is not None and snap is not None and torch.equal(t, snap)


def run_pair(name, attn, g, num_heads, head_dim, expect_fused=True, extra=None):
    """Run actual baseline then actual fused apply on identical inputs."""
    global STRUCTURAL_FAILURE
    rec = {"expect_fused": expect_fused}
    if extra:
        rec.update(extra)
    try:
        attn_snap = None if attn is None else attn.clone()
        g_snap = None if g is None else g.clone()
        base = hw.headwise_attn_output_gate_baseline(attn, g, num_heads, head_dim)
        out = hw.apply_headwise_attn_output_gate(attn, g, num_heads, head_dim)
        if not snapshots_equal(attn, attn_snap):
            rec["BASELINE_OR_FUSED_MUTATED_ATTN"] = True
            STRUCTURAL_FAILURE = True
        if not snapshots_equal(g, g_snap):
            rec["MUTATED_GATE"] = True
            STRUCTURAL_FAILURE = True
        out_ptr = out.data_ptr() if isinstance(out, torch.Tensor) else None
        in_ptrs = {
            t.data_ptr() for t in (attn, g) if isinstance(t, torch.Tensor)
        }
        if out_ptr is not None and out_ptr in in_ptrs:
            rec["OUTPUT_ALIASES_INPUT"] = True
            STRUCTURAL_FAILURE = True
        if isinstance(out, torch.Tensor) and isinstance(base, torch.Tensor):
            exact = bool(torch.equal(out, base))
            diff_mask = out.float() != base.float()
            mism = int(diff_mask.sum().item())
            rec["exact_equal"] = exact
            rec["mismatch_count"] = mism
            rec["max_abs_diff"] = float((out.float() - base.float()).abs().max().item()) if out.numel() else 0.0
            rec["nan_out"] = int(torch.isnan(out.float()).sum().item())
            rec["nan_base"] = int(torch.isnan(base.float()).sum().item())
            rec["inf_out"] = int(torch.isinf(out.float()).sum().item())
            rec["inf_base"] = int(torch.isinf(base.float()).sum().item())
            if mism:
                flat_idx = diff_mask.reshape(-1).nonzero().reshape(-1)[:5].tolist()
                rec["first_diff_flat_indices"] = flat_idx
            # Diagnostic only: would unrounded fp32 gate * rounded attn, rounded
            # once at the end, differ from baseline on these inputs?
            try:
                T = attn.shape[0]
                HD = num_heads * head_dim
                unrounded = (
                    attn.reshape(T, num_heads, head_dim).float()
                    * torch.sigmoid(g.float()).unsqueeze(-1)
                ).reshape(T, HD).to(attn.dtype)
                rec["unrounded_gate_diff_count_vs_base"] = int(
                    (unrounded.float() != base.float()).sum().item()
                )
            except Exception as e:  # noqa: BLE001
                rec["unrounded_gate_diag_error"] = repr(e)
        rec["status"] = "RAN"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "EXCEPTION"
        rec["exception"] = "".join(traceback.format_exception_only(type(e), e))
        rec["traceback"] = traceback.format_exc()[-1500:]
        STRUCTURAL_FAILURE = True
    case_record(name, **rec)
    print(f"[{name}] " + json.dumps({k: v for k, v in rec.items() if k != 'traceback'}, sort_keys=True))


def build_special_attn(T, cols, dtype, device):
    large = 3.0e38 if dtype != torch.float16 else 6.0e4
    tiny = 1.0e-40 if dtype == torch.float32 else (1.0e-38 if dtype == torch.bfloat16 else 6.0e-8)
    vals = [0.0, -0.0, 1.0, -1.0, 1.0078125, 0.4990234375, 2.75, -2.75, 1e-3, tiny, 1e4, large, -large, -0.333251953125]
    n = T * cols
    data = [vals[i % len(vals)] for i in range(n)]
    return torch.tensor(data, dtype=dtype, device=device).reshape(T, cols)


def build_special_gate(T, H, dtype, device):
    logits = [0.0, 0.01, -0.01, 0.25, -0.25, 1.0, -1.0, 5.0, -5.0, 20.0, -20.0, 100.0, -100.0, 0.123456789]
    n = T * H
    data = [logits[i % len(logits)] for i in range(n)]
    return torch.tensor(data, dtype=dtype, device=device).reshape(T, H)


def main():
    if not torch.cuda.is_available():
        print("NO_CUDA: this script requires an actual CUDA device; NOT_RUN_BY_WORKER.")
        return 3
    dev = torch.device("cuda")
    torch.manual_seed(20261110)

    # 1) Varied shapes, dtypes, random values (fused path expected).
    shapes = [
        (1, 4, 16, torch.bfloat16),
        (2, 3, 5, torch.bfloat16),
        (32, 8, 32, torch.bfloat16),
        (64, 4, 16, torch.bfloat16),
        (33, 5, 7, torch.bfloat16),
        (256, 16, 128, torch.bfloat16),
        (17, 5, 7, torch.float32),
        (128, 2, 64, torch.float16),
    ]
    for (T, H, D, dt) in shapes:
        attn = torch.randn(T, H * D, dtype=dt, device=dev)
        g = torch.randn(T, H, dtype=dt, device=dev)
        run_pair(f"random_T{T}_H{H}_D{D}_{str(dt).split('.')[-1]}", attn, g, H, D)

    # 2) Rounding-boundary / large / zero / tiny special values (fused path).
    for (T, H, D, dt) in [(8, 4, 16, torch.bfloat16), (8, 2, 64, torch.float16), (8, 3, 5, torch.float32)]:
        attn = build_special_attn(T, H * D, dt, dev)
        g = build_special_gate(T, H, dt, dev)
        run_pair(f"special_T{T}_H{H}_D{D}_{str(dt).split('.')[-1]}", attn, g, H, D)

    # 3) Empty batch (no kernel launch must occur; result must be empty).
    attn = torch.randn(0, 64, dtype=torch.bfloat16, device=dev)
    g = torch.randn(0, 4, dtype=torch.bfloat16, device=dev)
    run_pair("empty_T0_H4_D16", attn, g, 4, 16)

    # 4) Fallback paths on GPU (must equal baseline exactly, no fused launch).
    T, H, D = 16, 4, 16
    big = torch.randn(T, H * D + 8, dtype=torch.bfloat16, device=dev)
    attn_nc = big[:, : H * D]
    run_pair("fallback_noncontiguous_attn", attn_nc, torch.randn(T, H, dtype=torch.bfloat16, device=dev), H, D, expect_fused=False)
    attn_cpu = torch.randn(T, H * D, dtype=torch.bfloat16)
    g_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    run_pair("fallback_cpu_tensors", attn_cpu, g_cpu, H, D, expect_fused=False)
    run_pair(
        "fallback_gate_dtype_mismatch",
        torch.randn(T, H * D, dtype=torch.bfloat16, device=dev),
        torch.randn(T, H, dtype=torch.float16, device=dev),
        H, D, expect_fused=False,
    )
    run_pair(
        "fallback_3d_attn",
        torch.randn(T, H, D, dtype=torch.bfloat16, device=dev),
        torch.randn(T, H, dtype=torch.bfloat16, device=dev),
        H, D, expect_fused=False,
    )

    print("\n===== SUMMARY =====")
    print(f"STRUCTURAL_FAILURE={STRUCTURAL_FAILURE}")
    print("NOTE: exact_equal=false cases are honest numeric findings, not tolerated away; no quality/performance verdict follows from this script.")
    print(json.dumps(REPORT, indent=1, default=str))
    return 1 if STRUCTURAL_FAILURE else 0


if __name__ == "__main__":
    sys.exit(main())

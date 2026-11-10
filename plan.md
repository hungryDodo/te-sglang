# Plan — worker glm-b, round 3: fused Triton headwise attention-output gate (Spark2_5)

## Attribution / provenance
- Development parent: `8cfcffb59899e2fbf2dd1f9b504971a14145d443` (already contains the
  Controller-adopted round-1 sampler disabled-trace patch; this round's measurement
  against trusted reference `5b89820b04cb26385d45915c484708796ba7df91` therefore
  measures the COMBINATION, not the isolated gate kernel). Parent's single C1
  exploratory score 1.00196 does not establish a repeatable gain.
- Round 2 failed with ECONNRESET before any tokens or code; nothing was produced.

## Hypothesis
The Spark2_5 headwise attention-output gate (`torch.sigmoid(g.float()).to(dtype)`
then a broadcast per-head multiply) spends host time on three eager kernels
(fp32 cast+sigmoid, dtype cast, broadcast mul) plus an output-sized intermediate.
One Triton kernel computing sigmoid in FP32, casting the gate to `attn_output.dtype`
(EXPLICIT intermediate BF16/FP16 rounding preserved BEFORE the multiply) and
writing `attn_val * gate_rounded` into a NEW output tensor may reduce per-layer
host/GPU overhead for contiguous CUDA batches. Gate index for flattened element
`i` is `row*num_heads + head`.

## Scope (only this)
- New helper `python/sglang/srt/layers/headwise_attn_output_gate.py`:
  exact-baseline function (kept for Controller comparison), strict guards
  (2D exact shape, CUDA, same device, bf16/fp16/fp32, same dtype, contiguous,
  int metadata, numel bound), lazy `triton` import after guards, empty-batch
  handled eagerly (no zero-grid launch), new-output allocation, no in-place ops.
- Tiny call-site replacement in `python/sglang/srt/models/spark2_5.py`.
- No attention/KV/g_proj/out_proj changes; wheels/binaries untouched.

## Tests
1. Stdlib-only CPU test (AST on actual sources + fake torch/triton execution):
   B0/B1/B32/B64 + odd shapes, dtype/device/contiguity/shape/metadata fallbacks
   identical to baseline outcomes (including preserved error behavior),
   no-alias/no-mutation, launch grid/BLOCK checks. Run with `/usr/bin/python3 -I`.
2. GPU script `test/srt/test_headwise_attn_output_gate_gpu.py` — WRITTEN, NOT RUN
   by worker (status NOT_RUN_BY_WORKER): actual baseline vs actual fused helper on
   identical inputs, special/rounding-boundary values, fallback equivalence,
   honest mismatch/max-abs reporting, no tolerances.

## Non-claims
Triton source compiling/executing, GPU numeric equivalence (torch.sigmoid vs
tl.sigmoid implementations may differ), JIT compile success on the target GPU,
speedup, repeatability, or task quality are NOT established by this round.

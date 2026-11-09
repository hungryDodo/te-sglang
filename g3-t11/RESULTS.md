# G3 t11 (glm-b) — verified trace analysis: no host lever at C128 graph-replayed decode

Input: `controller_evidence/continuous-20261109-01/20261110-decode-trace/decode.trace.json.gz`
(raw SHA 09e8dcb9…, glm-a source commit 15fd7f4, a1/GB10, frozen BF16 launch,
Medium 512/64 @ C128, 16 paired `step[DECODE bs=128]` forwards, one
cudaGraphLaunch per step — graph replay confirmed, NOT eager).

## Controls (all pass; scripts + raw JSON preserved)
| check | got | README ref |
|---|---|---|
| kernel instances | 8560 | 8560 (exact) |
| kernel interval union | 565.549025 ms | 565.549025 ms (exact) |
| CPU step union | 28.303983 ms | 28.303983 ms (exact) |
| window envelope | 571.433 ms | 571.927 ms (0.09% — mine is first→last kernel; README's includes profiler-span start 0.48 ms earlier) |
| CPU↔GPU pairing | 16/16 by External id | 16 |
| assignment negative control | 96 kernels outside spans (0.9% of kernel time): per-step sampler/metadata kernels living in the inter-step gaps (48× index_elementwise=embedding gather, 16× clamp_position, 16× argmax reduce, 16× vectorized_elementwise); all accounted, none unexplained | — |

## Q1 HOST-LEVER — negative
- Steady-state (steps 2–16): GPU idle inside step span **240.5 µs / 35 231 µs = 0.68 %**
  (kernel∪ + memcpy + memset coverage); step 1 has 1205 µs idle = profiler-start
  artifact, excluded.
- Inter-step gaps: 342–350 µs each, 94 % covered by the eager sampler/metadata
  kernels above; **uncovered ≈ 20.6 µs/gap**. Whole-window uncovered: 5.88 ms /
  571.4 ms = **1.03 %** (dominated by the step-1 artifact).
- CPU side per step: step label 1.4–2.1 ms, cudaGraphLaunch submit 0.93 ms,
  cudaEventSynchronize 29.5 ms with GPU kernels busy under it (README overlap
  469.5/472.2 ms) — the overlap scheduler hides the host ~17×.
- **Conclusion: no host-scheduling/launch lever exists at this operating point.**
  Independently extends g2-t2's C256-eager closure to C128 graph-replayed decode.

## Q2 COMPOSITION — bounded, nothing actionable in frozen BF16
Share of 565.5 ms kernel time: GEMM **47.92 %**, flashinfer attention
(BatchPrefill tensor-core decode + MergeStates) **47.36 %** → 95.3 % core.
Non-core: other 1.97 %, argmax 0.88 %, elementwise 0.86 %, softcap 0.70 %,
norm 0.69 %, store_kv 0.32 %. Top kernels: attention 28/step × 595.7 µs;
nvjet 96x128 28/step × 258 µs; nvjet 64x128 28/step × 144.4 µs; lm_head
nvjet 128x208 1/step × 2545.9 µs (~211 GB/s at M=128 — roofline); cutlass_80
sm_80-fallback `gemm_relu` 28/step × 67 µs.
- Only architectural anomaly: the **sm_80 cutlass fallback GEMM (5.3 % of kernel
  time)**. Even a 30 % faster native kernel saves ≈1 % step time — sub-noise vs
  the ±2 % gate band, and a kernel swap changes reduction order (exact-token
  risk). Quantified dismissal; NOT proposed.

## Scope / limits
Single operating point; profiler-inflated absolute walls (CUPTI per-kernel
overhead on graph replays) — only shares/idle fractions interpreted; kernel
union ≠ SM utilization/bandwidth (README); tracing may perturb races. Not
covered: Prefill-Heavy C16 steady state, extend→decode transition, eager
(bs>graph-coverage) regimes on other trees. Engine change: none (diff vs
2c854e0 over python/ verified empty).

Artifacts: `g3-t11/analyze_decode_trace.py`, `g3-t11/analyze_decode_trace_refined.py`,
`g3-t11/raw/decode_trace_analysis.json`, `g3-t11/raw/decode_trace_refined.json`.

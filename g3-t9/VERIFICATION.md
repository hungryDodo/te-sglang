# G3 trial-9 verification record — glm-a

Executed: 2026-11-09 (campaign continuous-20261109-01, iteration 9).
Input: controller packet `controller_evidence/continuous-20261109-01/20261110-decode-trace/`
(real BF16 C128 decode trace, run a3b0f3ec, node a1, source commit 15fd7f4,
Medium512/64 C128, 16 forwards, cudaGraphLaunch x16). Packet originals untouched;
all derived output under g3-t9/raw/.

## What was executed

`python3 -I g3-t9/trace_bubble_census.py` (stdlib only, deterministic, no network).
Raw output: `g3-t9/raw/trace_bubble_census_output.json`. Exit 0.

## Gate structure (hard)

- Anchor replication (packet README values, tolerance 0.5 us): trace span
  571927.153 us; 8560 kernel instances; kernel union 565549.025 us; CPU step
  union 28303.983 us; cudaEventSynchronize sum 472169.159 us and kernel overlap
  469530.690 us; largest inter-step gap 34676.268 us with kernel overlap
  34417.705 us. ALL MATCHED. Any mismatch => NO_CLAIM + exit 2.
- Pairing control: 16 CPU `step[DECODE bs=128]` External ids == 16 GPU ids.
- Interval-machinery self-tests: hand-computed fixtures ((ts,dur) semantics —
  the first draft wrongly used (start,end) fixtures; caught by the assert).
- Planted-gap negative control: synthetic 300.0 us host step inside a real
  device-free hole reports not-hidden 300.0 us (pass). All-overlapped control:
  synthetic step covering the longest kernel reports 0.0 (pass).

## Findings (all measured from the packet's raw intervals)

1. **H9 REFUTED** — exact-runtime/SWA-host-bound lane has no headroom on a1 for
   this workload class. Device (kernel+memcpy+memset) busy 98.9164% of the marked
   window. Host-union not hidden under device activity: 5919.8 us = 1.0351% of
   wall (upper bound on ANY zero-kernel-change host optimization). Hard bubbles
   (sync blocked while device idle) 2786.5 us = 0.4872%. Consistent with F5's
   a2 no-op upper bound (x1.019) and G2-t2's marginal (+0.19% min over fp8-KV
   parent on a1). The loosest nested-annotation sum (2.0386%) still sits at the
   2% threshold only by double-counting nested spans; the union number governs.
2. **Steady-state host timeline**: per-step CPU annotation work is ~1.77 ms/step
   but 94.9% of it is hidden (host runs ahead, then blocks 29.5 ms/iter in
   cudaEventSynchronize, 99.2% overlapped by kernels). Per-step not-hidden is
   ~10 us after step 0 (first step 1292 us). `scheduler.process_batch_result`
   492.1 ms total, 99.26% hidden. The famous 34.68 ms "largest host inter-step
   gap" is NOT a host stall: kernels execute for 34.42 ms of it — it is one full
   pipelined GPU step (with a 512-token prefill mixed in: 27-28
   BatchPrefillWithPagedKVCache + nvjet GEMM instances inside).
3. **Decode step anatomy** (kernel sum 567.3 ms ~= union 565.5 ms => single
   stream, shares are wall shares): attention 47.04% (16.68 ms/step = 28
   instances x ~595 us), GEMMs 47.83% (~16.96 ms/step), other 5.14%.
   Decode attention is dispatched as `flashinfer::BatchPrefillWithPagedKVCacheKernel`
   (tensor-core decode path); zero `BatchDecode` kernels exist. 529 kernels/step.
4. **E2 cross-check**: halving the measured attention time predicts 8.34 ms/step
   of savings for fp8-e4m3 KV — G2-E2 measured 8-9 ms/step on the same
   node/model. The fp8-KV mechanism is now kernel-level confirmed: the gain is
   exactly the attention KV-read halving. (Comparison to prior measured data,
   not a new gate.)

## Uncertainty and limits

- 16 instrumented forwards, one run, one node (a1), under profiling (the packet
  itself warns traces perturb). Bounds apply to Medium512/64 C128 decode class;
  Prefill-Heavy and C256 ceiling are NOT covered.
- Kernel interval union is not SM utilization or bandwidth; per-kernel sums are
  a census. The wall-share reading is licensed only because sum ~= union.
- Not-hidden host time is an upper bound on removable cost, achievable only
  with zero kernel change and perfect overlap shift.
- No engine change, no gate request this trial; frozen protocol untouched.

## Consequences for the campaign

- The SWA/exact-runtime revisit is closed with MEASURED evidence (was static in
  t7): no a1 Fast slot should be spent on host-sync-class candidates for this
  workload class.
- The two ~47% walls (attention bytes, weight bytes) mean the only levers of
  fp8-KV's magnitude are precision-class (KV or weights) — both currently gated
  on the quality budget, for which t8 registered a margin-stratified contract
  proposal and the numeric top-k prerequisite (C5) remains open. F4's
  candidate-x3 rerun in the G3 namespace stays the top queued throughput
  action once that prerequisite is met.

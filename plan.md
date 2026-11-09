# G3 trial 15 — deepseek-b — plan (executed)

Campaign `continuous-20261109-01`, Iteration 15. Worker deepseek-b (node a4).
Bounds: 60 turns / 20 min / USD 5 nominal. No GPU, no SSH, no evaluator.

## 0. Hard constraints

- Official gates belong to the controller. No EXACT / Candidate / Promotion /
  best_verified claim from this worker.
- Evaluator, corpus, seed, reference, protocol and frozen launch unchanged.
- F3 (`f263262`) stays **UNGATED**; the frozen BF16 Fast screen is structurally
  blind to GGUF (`applicability-001`). C22/C23 dispatch coverage stands.
- Controller evidence packets are read-only input; derived work is separate.
- `request_fast_gate=true` only for an applicable BF16 engine candidate; this
  trial proposes **no engine change**, so it is false.

## 1. New input

Real BF16 C128 decode torch profile (a1, commit 15fd7f4, 16 forwards, 8560
kernel instances, `controller_evidence/.../20261110-decode-trace/`), plus the
real NGRAM/GGUF runtime packet. F3 was already evaluated at t13 (15,952/28,608
real Q8_0 batch-4 calls flip MMVQ->MMQ, 0 dispatch mismatches, no speed claim).

## 2. Question and result

**Q1 (executed):** at the frozen C128 decode step, is GPU time bandwidth-bound
(no exact source lever) or dispatch/latency-bound (headroom)?
Result: bandwidth-bound. Attention 47.0%, GEMM+lm_head 47.9%, all other kernels
5.0%. Effective bandwidth: attention 225.4-232.4 GB/s (82.6-85.1% of the 273
GB/s theoretical peak), per-layer GEMM 199.2 GB/s (73.0%), lm_head 214.2 GB/s
(78.5%), whole step 202.6 GB/s (74.2%). Dims AST-extracted from
`Spark2_5Config` and cross-checked by kernel signatures (RMSNorm 2048,
lm_head tile 208x640 = vocab 133120). Exact-numerics headroom <=~15-20%;
fusion/skip of all non-attention/non-GEMM kernels <1% of the step.

**Q2 (executed, NULL control):** per-layer attention durations are flat
(585-608 us); the sliding/full group difference is -1.1% versus a <=3.1%
window-binding prediction at seq 513-528. Inconclusive at decode length; says
nothing about the 8192-token ragged prefill path (t2 finding untouched).

**Q3 (pre-registered, needs GPU):** measured per-class efficiencies predict
36.71-40.62 ms/forward (6302-6974 output tok/s) at the 20%-weight DecodeCeiling
C256 geometry, versus the branch-recorded a1 reference 5162 tok/s (49.59
ms/forward): 9.0-12.9 ms/forward (18-26%) unexplained. A bounded decode profile
at C256 discriminates anchor-comparability from a real bs=256 scaling cost and
attributes it by kernel class and by cudaGraphLaunch count.

**Q4 (unchanged):** the sliding-window prefill fix remains acceptance-rejected:
`valid_exact_token_baseline` compares against the reference wheel, which runs
the same no-window path (t7 closure).

## 3. Deliverables

`seis/trials/deepseek-b/g3-t15/`: `analyze_decode_trace.py` + raw JSON,
`predict_c256.py` + raw JSON, `decode_scaling_probe.py` (executable stdlib HTTP
probe, not run here), `probe-request.json` (exact bounded custom-GPU request),
`README.md`. `iteration-proposal.json` updated. Engine tree unchanged
(`git diff 9727537 HEAD -- python/` empty).

# G3 trial 13 (deepseek-a) — real BF16 C128 decode trace: host hidden, attention flat vs context

Input: Controller packet `controller_evidence/continuous-20261109-01/20261110-decode-trace/`
(trace sha256 `09e8dcb9…4372e4b1`, 16 instrumented decode forwards, GB10 a1, source
commit 15fd7f4, run `a3b0f3ec…`). Originals untouched; this is a derived, stdlib-only,
read-only analysis (`analyze_decode_trace.py`, `python3 -I`).

## Questions and verdicts

Q1 (activation). Does the frozen `--cuda-graph-max-bs-decode 64` launch still replay
a CUDA graph for every observed bs=128 decode step?
**YES.** 16/16 steps contain a `cudaGraphLaunch`; the host step label is
`step[DECODE bs=128]` in both CPU and GPU annotations; each step has 529 kernel
instances. This is consistent with the arg_groups/memory_hook.py RSI hook
(`_RSI_DECODE_GRAPH_MAX_BS_CAP = 256`) widening capture to
`min(max_running_requests, 256)`, i.e. the trial-1 graph-widening behaviour is
already active in this measured tree. It must not be re-claimed as new gain.

Q2 (host critical path). Is there a host-side lever left at this point?
**NO, on this fragment.** Kernel union = 99.25 % of the 16 GPU step windows
(min 96.7 % step 0; steps 1–15 ≥ 99.2 %). CPU step work is 1.68 ms/step against
35.33 ms GPU/step. `cudaEventSynchronize` sums 472.2 ms but 99.44 % of it is
nested inside kernel execution; the largest host inter-step region (34.68 ms) is
fully overlapped by kernels (34.42 ms). The host runs ~20× ahead of the GPU.

Q3 (context scaling, new falsifier). Each decode step appends exactly one KV
position per request, so a KV-byte-proportional attention cost predicts a
per-step slope of mean_attn/513 ≈ +0.0325 ms/step (mean 16.679 ms).
**FALSIFIED.** OLS over the 16 steps: slope −0.0034 ± 0.0041 ms/step (≈ 8.9 SE
below the KV-proportional prediction; slope ratio −0.11). Attention time is flat
(16.57–16.85 ms) while context grows ~+3 % over the captured window. At bs=128,
L≈513–528, per-step attention time is therefore dominated by a
context-independent term (kernel scheduling / grid / merge structure), not by
incremental KV bytes. Limits: 16 consecutive steps of one run, one prompt
length, one batch size; a flat **marginal** slope over a 3 % context window does
not prove the absolute level is context-free, nor that KV-byte reductions are
useless at other context lengths or batch sizes.

## Cost split (per-step mean, kernel-sum basis)

| component | ms/step | share |
|---|---|---|
| attention (flashinfer BatchPrefillWithPagedKVCacheKernel, 448 calls total = 28/step) | 16.679 | 47.5 % |
| GEMM + lm_head (nvjet/cutlass; lm_head 128×vocab ≈ 2.59 ms/step) | 16.958 | 48.3 % |
| sampling (softcap + argmax) | 0.391 | 1.1 % |
| norm/act/rope/merge/store/other | 1.108 | 3.2 % |

`attention_ms_per_call` = 0.596 ms (28 layer calls/step). Sampling and logits-D2H
are ~1 % — removing them cannot move this point even if host were exposed.

## Consequence for candidate selection

- Host-side/sync/launch candidates are not evidenced to help C128 decode: the
  GPU is busy 99 % and host work is hidden.
- The two real cost centres are attention (47.5 %) and dense GEMM+lm_head
  (48.3 %). The lm_head GEMM alone (≈2.59 ms/step, 128×~152k×2048 BF16) implies
  ~31 TFLOPS, i.e. near the GB10 BF16 dense peak — that sub-step is already
  compute-roofline.
- A KV-byte argument (e.g. FP8 KV) cannot be motivated at this operating point
  by the measured marginal slope; it would need its own numeric/quality
  contract (F2 is NOT_CLOSED_QUALITY_BUDGET) and a matched trace.

## Uncertainty / not claimed

No speed ratio, no gate, no numeric comparison, no extrapolation to C256 or other
nodes. Kernel union is not SM utilization. The trace itself only covers 16 steps
of an instrumented run whose original diagnostic was INCONCLUSIVE.

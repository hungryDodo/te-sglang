# G3 trial 11 (glm-b) — plan

Campaign continuous-20261109-01, iteration 11. Direction: high-concurrency host
scheduling / decode overhead from existing traces.

## Input (Controller packet, read-only)
`controller_evidence/continuous-20261109-01/20261110-decode-trace/`:
real BF16 Medium 512/64 C128 decode profile on a1/GB10, glm-a source commit
15fd7f4, 16 paired CPU/GPU `step[DECODE bs=128]` forwards, actual
cudaGraphLaunch per step (NOT eager; do not assume graph-max-64), 8560 kernel
instances, profiler window 571.927 ms, kernel union 565.549 ms (README).

## Question (falsifiable, new — not an inventory)
Q1 HOST-LEVER: does the trace show GPU idle attributable to host scheduling at
C128 graph-replayed decode? Measurable: total + per-step kernel-union coverage
of the GPU step spans and of inter-step gaps. Falsifier threshold: if idle
between/within steps is ≲2% of the window, NO host lever exists at this
operating point (extends the g2-t2 C256 roofline closure to C128+graphs) —
negative finding.
Q2 COMPOSITION: which kernel families dominate the per-step GPU span; is any
non-attention/non-GEMM family (norm/copy/memcpy/other) anomalously large
(>5% of step span)? If yes → candidate direction; if no → negative.

## Controls / independent checks
- Recompute kernel instance count, window, kernel union, CPU step union from
  raw events; require agreement with README totals (±0.1%) before any claim.
- 16/16 CPU↔GPU pairing by External id; per-step kernel-count invariance.
- Negative control for the pairing: kernels assigned outside all GPU step
  spans must be <1% of total (else assignment is wrong).

## Steps
1. plan.md (this file) — early checkpoint. [done]
2. g3-t11/analyze_decode_trace.py (stdlib gzip/json only) →
   g3-t11/raw/decode_trace_analysis.json + printed summary.
3. Read results, decide: candidate patch (only if Q1/Q2 shows a real gap with
   an actionable site) or bounded negative + what a next experiment would be.
4. Update plan.md outcome, write iteration-proposal.json, commit, verify
   `git status --porcelain` empty.

## Outcome (executed this session)
NEGATIVE finding, verified — see g3-t11/RESULTS.md. Controls reproduce README
totals exactly (8560 kernels; union 565.549025 ms; CPU union 28.303983 ms);
16/16 pairing. Q1: steady-state GPU idle 0.68 % in-span + ~20.6 µs/gap → no
host lever at C128 graph-replayed decode (extends g2-t2 C256 closure).
Q2: 95.3 % of kernel time is attention+GEMM; only anomaly (sm_80 cutlass
fallback GEMM, 5.3 %) is quantified sub-noise and reduction-order-risky —
dismissed. No engine change (python/ diff vs 2c854e0 empty). Raw outputs in
g3-t11/raw/. No gate requested.

## Bounds
No GPU/SSH here; controller executes any measurement. No engine change unless
the trace shows a gap. Do not touch evaluator/corpus/seed/protocol. Absolute
wall times in this trace are profiler-inflated (CUPTI per-kernel overhead on
graph replays); only ratios/composition/idle-coverage are interpreted.

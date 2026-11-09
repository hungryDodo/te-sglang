# G3 t15 — deepseek-b — real BF16 C128 decode trace: bandwidth attribution and a C256 scaling falsifier

Campaign `continuous-20261109-01` (Iteration 15). No GPU/SSH/evaluator in this
sandbox. All official measurement remains the Controller's. F3 (`f263262`) is
**UNGATED** and untouched here; the frozen BF16 Fast screen cannot exercise GGUF.

## Executed artifacts (stdlib only, `python3 -I`)

| file | what it does |
|---|---|
| `analyze_decode_trace.py` | independent parse of the Controller trace; hard-fails on SHA drift, kernel count, and packet-total drift; kernel-class attribution; config-derived byte traffic and effective bandwidth; per-layer attention test |
| `decode-trace-analysis.json` | raw output |
| `predict_c256.py` | turns measured per-class efficiencies into a falsifiable C256 prediction |
| `c256-prediction.json` | raw output |
| `decode_scaling_probe.py` | executable HTTP probe driver the Controller can run against the frozen server (NOT executed here) |
| `probe-request.json` | exact bounded custom-GPU request, discriminator, bounds |

## Findings (all from the real a1 trace, 16 forwards, 8560 kernels)

1. **Independent recompute control passes**: span 571.927 ms, kernel union
   565.549 ms, CPU `step[DECODE bs=128]` union 28.304 ms, 16 labels — matches the
   packet to <0.01 ms. No harness error.
2. **Kernel-time split at C128 decode**: attention 47.0% (266.9 ms),
   GEMM+lm_head 47.9% (271.9 ms), everything else 5.0% (28.6 ms). The
   non-attention/non-GEMM surface (softcap 6.3 ms, argmax 5.0 ms, RMSNorm 3.9 ms,
   act 3.5 ms, rope/store-KV ~2.7 ms) is too small to matter: fusing or skipping
   all of it caps out far below 1% of the step.
3. **The step is uniformly memory-bandwidth-bound, not dispatch-bound.**
   Using dims AST-extracted from `Spark2_5Config` and cross-checked against
   kernel signatures (hidden 2048 RMSNorm; lm_head tile 208x640 = vocab 133120;
   28 layers x 16 forwards = 448 attention calls):
   - attention: 225.4-232.4 GB/s (82.6-85.1% of the 273 GB/s theoretical GB10 peak)
   - per-layer GEMMs: 199.2 GB/s (73.0%)
   - lm_head: 214.2 GB/s (78.5%)
   - whole step: 202.6 GB/s (74.2%), 35.46 ms/forward
   Exact-numerics source work at C128 decode therefore has at most ~15-20%
   theoretical headroom (less against a realistic STREAM ceiling). The only
   >20% byte-reduction lever is precision (KV or weight dtype), which is
   quality-budget gated (F2; NOT_CLOSED_QUALITY_BUDGET).
4. **Per-layer attention test is a NULL control at decode length.** All 28
   layers run 585-608 us (means); the (i+1)%4 full-layer group differs by
   -1.1% (opposite sign to a window-binding effect; exact permutation
   p=0.0367 on a layer-order statistic that is not a causal test). At seq
   513-528 the 512-token window binds by at most 3.1%, so this trace cannot
   discriminate window application at decode and says nothing about the
   8192-token ragged prefill path. It does bound any layer-type timing
   difference to ~2%.
5. **New falsifiable discrepancy at DecodeCeiling C256 (20% of score).**
   Applying the measured per-class GB/s to the C256 geometry
   (batch 256, mean seq 288.5) predicts 36.7-40.6 ms/forward and
   6,302-6,974 output tok/s. The branch-recorded a1 reference at DC-C256 is
   5,162 tok/s = 49.6 ms/forward, i.e. **9.0-12.9 ms/forward (18-26%)
   unexplained**. Either the anchor is not like-for-like, or a real
   batch-scaling cost (attention/GEMM/other or eager-vs-graph) exists at
   bs=256. The supplied trace cannot decide: it is C128 medium-prompt only.

## Requested (not executed here)

`probe-request.json` asks for one bounded diagnostic on the frozen server:
run the exact DC-C256 workload shape twice (unprofiled control + 16
decode-profile steps via `/start_profile` `profile_stages=["decode"]`) and
report end-to-end tok/s, per-class kernel ms/GB/s, `cudaGraphLaunch` count per
forward, and the actual batch label. Two well-separated outcomes are
pre-registered in `c256-prediction.json`.

## Limits

One instrumented C128 fragment on one node/commit; no unprofiled speed ratio;
seq length during the fragment is unrecorded (bounded 513-576); 273 GB/s is the
theoretical peak, not a measured STREAM ceiling; weight bytes assume BF16 and
one lm_head read of the tied embedding per forward. No EXACT, numeric,
Candidate or Promotion claim is made. F3 remains UNGATED and is unaffected by
anything here (the frozen gate is BF16).

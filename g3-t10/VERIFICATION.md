# G3 trial-10 verification record — glm-a

Executed 2026-11-09/10 (campaign continuous-20261109-01, iteration 10, worker glm-a).
All CPU-side work in this sandbox (no torch/GPU); real measured inputs are the
controller's C21 packet (20261110-runtime) and the REAL engine source at HEAD.
Packet originals untouched; derived outputs only under g3-t10/raw/.

## Executed this session (all exit 0)

1. `python3 -I g3-t10/verify_remap_gate.py` — ast-extracts the REAL
   `_rsi_trial_kv_cache_dtype` (python/sglang/srt/arg_groups/overrides.py,
   extracted sha256 7956a93a9614d1db…) and runs the env x dtype truth table:
   6/6 PASS. REF (SGLANG_SPARK_FP8_KV unset/0, frozen `--kv-cache-dtype bfloat16`)
   -> `{}` + "remap inactive" line (exact BF16, bit-identical default); CAND (=1)
   -> exactly `{"kv_cache_dtype": "fp8_e4m3"}` + the warning line. No engine
   change is needed for the diagnostic pair: the toggle is env-only on ONE tree.
2. `python3 -I g3-t10/ref_margin_extract.py` — recomputes the REF-side margin
   floor from the C21 OFF records (plain BF16 999d4b243): 480 positions,
   min margin 0.9374998808 nat, count(margin<=0.306)=**0**, all 4 t8 anchors
   reproduced (min quoted 0.9375 = 4dp rounding, tolerance 5e-5, noted).
   Pre-registered: fp8-class noise cannot flip ANY of the repeat-512 instances.
3. `python3 -I g3-t10/fp8_margin_probe.py selftest` — 7/7 PASS: identical->
   TOKEN_EXACT (eps=0), low-margin flip->PRECISION_CONSISTENT (0.10<=2x0.08),
   cascade-after-flip ignored, high-margin flip->NON_PRECISION_DEFECT,
   null top-8->COVERAGE_INCONCLUSIVE, eps_hat computed pre-flip only,
   overall band-guard truth table.
4. `python3 -I g3-t10/fp8_margin_probe.py spec` — emits the exact 13
   requests/server (10x repeat-512 instance repeats + 3 natural prompts),
   624 paired positions, C21-proven payload shape, deterministic ids
   (sha256 44a5e6cc95f00209…, identical to C21's A/repeat-512 family).
5. `python3 -I g3-t10/run_ngram_smoke.py` — **real-data validation**: the
   analyzer on the real C21 ON/OFF records (10 pairs) -> overall
   NON_PRECISION_DEFECT, matching canonical F1 BUG_CONFIRMED. First flips at
   positions 4-15 with eps_hat 0.22-1.32 nat.

## New executed finding (real data, not fixtures)

The NGRAM smoke exposed a contract hole: 3/10 pairs pass the self-scaling rule
(margin <= 2*eps_hat) ONLY because NGRAM's own pre-flip deviation is huge
(eps_hat 0.55-0.85 nat vs E2's fp8 band <= 0.153). A candidate with large
deviation inflates its own acceptance threshold and can hide a defect.
Refinement implemented + unit-tested: absolute eps_hat band
(within_E2_max / above_E2_max_within_2x / ANOMALOUS_pre_flip_deviation) and a
new overall verdict AMBIGUOUS_BAND (self-scaling pass with out-of-band
deviation is NOT precision-class evidence). t8's proposed contract step 2-3
should adopt this guard before registration.

## Limits

- Analyzer verified on synthetic fixtures + real NGRAM arrays; fp8-side arrays
  do not exist yet (that is the requested diagnostic).
- Truth-table execution uses fixture views for the ResolvedView object; a real
  boot still must confirm activation via server.log lines + Full/SWA KV pool
  dtypes (the i3/i4 lesson: verify the actual snapshot logs, not flags).
- 480 REF positions bound P(margin<=0.306) <= ~0.62% (rule of three) on this
  prompt class only; natural-text classes had heavier tails in E2.
- No GPU, no throughput, no EXACT claim; frozen protocol untouched.

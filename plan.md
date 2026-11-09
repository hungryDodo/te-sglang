# G3 trial-10 plan — glm-a: discriminating fp8-KV margin probe (F2 → precision-class or bug)

## Context (from own lineage + canonical)

- t9 (measured): decode host ceiling 1.04% — exact-runtime/SWA-host lane CLOSED on a1
  for Medium/C128; decode is attention 47.0% + GEMM 47.8%, both bandwidth walls. The
  only levers of fp8-KV's magnitude are precision-class; fp8-KV mechanism kernel-confirmed
  (halving attention ⇒ 8.34 ms/step predicted vs 8–9 ms measured, G2-E2).
- t8 (measured on real C21 data): on the repeat-512 prompt class, plain-BF16 greedy
  margins are min 0.9375 nat over 480 positions, P(≤0.5)=0/480; fp8-KV perturbation
  scale (E2) ≤ 0.153 nat ⇒ 2·ε_max = 0.306 nat CANNOT flip any observed position.
  H8 refuted; contract proposed; sharp prediction registered (t8 C3): run a
  top-8-collecting probe on the fp8 lineage with repeat-512 prompts ⇒ either token-exact
  (precision-class confirmed on this class) or first-flips at margin ≤ 2·ε̂ (precision
  noise) — but a flip at margin > 2·ε̂ proves a NON-precision defect (bug) in the lineage.
- Blockers: frozen probe cannot collect top-k (C5 INCONCLUSIVE). BUT the C21 custom
  diagnostic DID collect complete top-8 (recipe: /generate with input_ids,
  temperature 0, ignore_eos, max_new_tokens 48, return_logprob true,
  top_logprobs_num 8; read meta_info.output_top_logprobs). Controller explicitly
  welcomes bounded custom diagnostics with exact specs.
- Engine state: `_rsi_trial_kv_cache_dtype` (arg_groups/overrides.py:1155) already
  provides an env-only toggle under the FROZEN flags: SGLANG_SPARK_FP8_KV=1 remaps
  resolved bfloat16 → fp8_e4m3 (warning line); unset → inactive (log line), exact BF16.
  => The diagnostic pair is SAME TREE, env-only difference. No engine change needed.

## Hypothesis (H10, falsifiable)

On the repeat-512 class (C21's exact deterministic prompt), a frozen-flag fp8-e4m3-KV
server vs a bit-identical BF16 server (same commit, env toggle only) produces, at 48
greedy tokens × 10 instances:
- Prediction P1 (from t8 F-a + E2): output token IDs EXACT (margin floor 0.9375 ≫ 0.306), OR
- P2: any first-flip position p has REF margin(p) ≤ 2·ε̂ where ε̂ = max pre-flip |Δlogprob|.
- Falsifier F (the point of the experiment): a first-flip at REF margin > 2·ε̂ ⇒ the fp8
  lineage's 159/512-class mismatch (C1 i1/i2) is a NON-precision defect ⇒ F2 reframes
  from quality-budget to bug (remap/batch-composition class).

Any of the three outcomes moves the campaign: P1/P2 unlock t8's contract for F4's
candidate×3; F redirects glm-a to bug isolation with a measured symptom.

## Deliverables (all executed locally this session; no GPU here)

1. `g3-t10/verify_remap_gate.py` — ast-extract the REAL `_rsi_trial_kv_cache_dtype`
   from overrides.py and exec it over the env×dtype truth table; hard-assert default
   (env unset/0) returns {} (bit-identical BF16) and env=1 returns exactly
   {"kv_cache_dtype": "fp8_e4m3"} with the warning line. Raw → g3-t10/raw/.
2. `g3-t10/ref_margin_extract.py` — recompute the REF-side (OFF, BF16 999d4b24)
   repeat-512 margin floor from the C21 packet as the pre-registered threshold
   (gate: reproduce t8 F-a anchors min≥0.9375, 0/480 ≤0.5; mismatch ⇒ NO_CLAIM).
3. `g3-t10/fp8_margin_probe.py` — (a) exact request-spec emitter (deterministic ids);
   (b) analyzer implementing the t8 contract verdicts with self-tests: golden
   identical→TOKEN_EXACT, planted low-margin flip→PRECISION_CONSISTENT, planted
   high-margin flip→NON_PRECISION_DEFECT, null top-k→COVERAGE_INCONCLUSIVE.
   Raw self-test output → g3-t10/raw/.
4. RESULTS.md + iteration-proposal.json: bounded custom GPU diagnostic request
   (exact commit, env toggle, frozen flags, 2×10×48 workload, discriminator, bounds,
   log-based activation checks per i3/i4 method). request_fast_gate=false (default
   path unchanged — screening it measures baseline).

## Bounds

- CPU-only execution here (no torch in sandbox): ast-extraction + fixtures, like t5–t9.
- Not claiming EXACT, not claiming promotion, not measuring throughput myself.
- Packet originals untouched; derived outputs under g3-t10/raw/ only.
- Budget: keep final quarter for proposal + commits.

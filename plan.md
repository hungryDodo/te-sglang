# G3 trial-4 plan (glm-a) — fix the exact-BF16 activation so FP8-vs-runtime decomposition is measurable

## Context / bottleneck (not throughput directly)
Canonical `glm-a-activation-failure-003`: G3 i3 (`e055f701`) intended to measure a
true-BF16 candidate (FP8-KV default off) to decompose the ~x1.136 Fast gain of
i1/i2 (`699c3322`/`999f93e6`) into [fp8-KV precision contribution] vs
[exact-runtime hunks: SWA host-sync removal `83e7b75` + host-side metadata
`e063d72`/`2f3a916`]. The i3 guard (`SGLANG_SPARK_FP8_KV`, default off) was
added only to `_muse_spark_overrides` (auto-default path,
`arg_groups/overrides.py:317-326`), but the frozen launch passes
`--kv-cache-dtype bfloat16` explicitly, so the separate unconditional remap
`_rsi_trial_kv_cache_dtype` (`overrides.py:1153-1161`) still fired; Fast 1.1335
was FP8-lineage, activation condition failed.

## Hypothesis (H1, falsifiable)
If `_rsi_trial_kv_cache_dtype` is gated behind the same `SGLANG_SPARK_FP8_KV=1`
opt-in, then under the frozen BF16 launch (env unset):
- resolved `kv_cache_dtype` stays `bfloat16` (no remap warning, no fp8 alloc), and
- the engine tree diff vs the FP8 lineage is exactly the remap behavior — an
  exact-runtime candidate (SWA sync-removal + host metadata hunks, BF16 KV).
H0 (guard insufficient again — some third site forces fp8): falsified on CPU by
executing the real post-process function with stub views covering
bfloat16/auto/fp8_e4m3 × env {unset, 1}.

Expected Fast band if H1 holds: above 1.0 but below the 1.136 combo — a2's
standalone SWA hunk was x1.0543, i2 marginal over e46f99d was only +0.5%; the
BF16-exact share is unknown, which is precisely what this measures. No claim
until the controller's gate returns.

## Patch (surgical, one hunk)
`overrides.py` `_rsi_trial_kv_cache_dtype`: remap only when
`SGLANG_SPARK_FP8_KV=1`; log inactivity otherwise. Opt-in path (env=1)
reproduces the i1/i2 FP8 lineage unchanged.

## Verification (CPU-only, this sandbox; no GPU/SSH)
1. `python -I` falsifier `g3-t4/remap_gate_check.py`: extract the REAL function
   source via ast (anchor asserts), exec against stub views; assert:
   env unset + bfloat16 -> no override; env=1 + bfloat16 -> fp8_e4m3;
   fp8_e4m3 -> no-op; raw JSON to `g3-t4/raw/`.
2. Grep sweep for any other forced-fp8 site in `python/` (the i3 lesson: one
   guard is not activation proof).
3. `git diff e46f99d..HEAD -- python/` reviewed: expected exact-runtime hunks +
   this guard, nothing else touching numerics.

## Request
`request_fast_gate=true`: frozen BF16 Fast screen on the new commit = the
exact-runtime candidate's first true-BF16 measurement (decomposition point for
C1/i1-i2). Numeric verification and candidate-x3 remain controller-gated;
quality budget for FP8 stays NOT_CLOSED_QUALITY_BUDGET — this trial claims
nothing precision-related.

## Non-goals
No precision changes, no evaluator/protocol edits, no NGRAM/MMVQ lanes
(other workers), no promotion claims.

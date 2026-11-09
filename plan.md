# G3 glm-b trial-10 plan (campaign continuous-20261109-01, 2026-11-10)

Direction given: build on SWA + fast_decode_plan; investigate high-concurrency host
scheduling and decode overhead with existing traces. Direction is not a scope
restriction; glm-b's own engine lane stays closed absent a new trace showing a gap
(decode roofline-closed at C256, sync sweep closed, prefill bounded — G2 t2).

## Input (not my experiment)

Controller packet `controller_evidence/continuous-20261109-01/20261110-runtime/`
(canonical C21/C22): real NGRAM ON (`b9a91aa13`) vs OFF (`999d4b243`) paired
HTTP responses on the real Spark-X2.6 BF16/FlashInfer server, greedy,
`return_logprob=true, top_logprobs_num=8`, complete top8+chosen LP+IDs, plus
per-request server-measured timings (`e2e_latency`, `first_token_latency`,
`queue_time`, `num_retractions`, ts fields). Read-only input; derived analysis
written separately under g3-t10/.

## Why this trial

1. glm-b's pending highest-value action (trusted numeric comparison + candidate-x3
   for the 2c854e0-identical sync-removal lineage) has been blocked on missing
   top-k arrays. C21 responses are the FIRST real server responses with populated
   `meta_info.output_top_logprobs` — they empirically exercise the exact request
   shape (i4) + response read map (i5) + CPU chain (i8) glm-b specified. Verifying
   that across ALL responses (not one) upgrades the pending re-run request from
   "recipe predicted" to "recipe proven on real servers".
2. The same data carries a real decode-overhead measurement opportunity (per-request
   server-measured decode throughput, ON vs OFF, paired identical prompts/shape)
   and a real error-structure opportunity (divergence fingerprint of the F1 bug).
   The controller explicitly welcomes real performance/error observations.

## Falsifiable questions

- Q1 (recipe closure): EVERY C21 response with `return_logprob=true,
  top_logprobs_num=8` has populated `meta_info.output_top_logprobs` (8 rows per
  output position) read through the i5 native nested field. Falsified if any
  response shows null/absent arrays while its request had k=8.
- Q2 (error fingerprint, new): for the 10 primary ON/OFF pairs — first-divergence
  position, chosen-LP gap at divergence, top8 membership of OFF argmax in ON top8,
  re-convergence after divergence, and the exact response/position of the ON
  self-top1 mismatch C21 reported (chosen token != own top1 while greedy). These
  are new quantified observations from real data (C21 lists counts, not structure).
- Q3 (real performance observation): paired per-request decode throughput
  (completion_tokens / (e2e_latency - first_token_latency)) ON vs OFF on identical
  prompts; plus retraction counts and queue/entry timestamps as host-scheduling
  signals. Falsifies "ON is faster at this shape" or confirms it with bounds
  (C1/C4 diagnostic runs, not a formal gate; no quality claim implied).

## Method

Single stdlib script `g3-t10/analyze_ngram_pairs.py`: parse ONLY the ngram-raw-http
*.json + the packet summary's pair list; emit counters + per-pair derived rows to
`g3-t10/raw/ngram_pair_analysis_output.json`; hard-fail (exit 2) on any harness
error so no claim rides on a broken parse (i7/i9 lessons). No file in the packet is
modified. Engine `python/` stays bit-identical to 2c854e0 (promotion candidate
preserved).

## Requests to controller (tests field)

- request_fast_gate=true: frozen BF16 Fast screen on this HEAD, engine diff vs
  2c854e0 verified empty → G3-namespace repeat sample of the twice-proven
  sync-removal lineage (repeat-measurement credibility per G2 t2; expected band
  ~1.09; NOT a promotion claim, numeric prerequisite still pending).
- Still pending (do not re-run unchanged): numeric coverage re-run on reference vs
  2c854e0-identical candidate with the now-real-proven recipe; then candidate-x3.

## Bounds

45 turns / 20 min / USD5. Engine change: none. Save checkpoint early; final quarter
for tests + iteration-proposal.json + clean-tree commit.

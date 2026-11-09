# G3 glm-b trial-8 plan — close the last untraced CPU legs of the null-topk chain (intake of k; sampler branch; detokenizer conversion)

Context: campaign continuous-20261109-01, glm-b iteration 8. Engine lane is measured-closed by my own
traces (decode C128/C256 bytes-bound at roofline, 99.2% GPU busy; prefill near-roofline; sync sweep
closed; prefill-mirror patch adds ~0.1% aggregate, reachable at b196fd4). The binding constraint on the
highest-value glm-b action (promotion of the 076b006+aa45af5 lineage, tree 2c854e0, measured
1.0920/1.0984) is the numeric-evidence gate (canonical F6/C5): trusted numeric comparison is blocked
because the frozen probe observed `meta.top_logprobs`/`meta.top_output_logprobs` = null.

State after i4/i5/i7 (all executed CPU-side, engine untouched):
- i4/C7: probe REQUEST does send return_logprob=true, top_logprobs_num=8 (rsi_equivalence.py:114-115).
- i5: response-side assembly mapped+executed: API emits `meta.output_top_logprobs` (nested) / flat
  `_val_flat/_idx_flat/_shape/_null_prefix` fields; the probe's read name is structurally wrong -> null.
  Recipe fix (2) delivered.
- i7: scheduler-side output population (sampler->batch_result_processor) POPULATED_AT_SOURCE hop1-h7
  for the probe shape; between populated and empty the only mode is a TypeError crash, never silent null.

NOT yet traced (this trial's targets — the gap between "request JSON has k=8" and "sampler branch ran",
plus the detokenizer conversion):
1. INTAKE of k: HTTP JSON -> GenerateReqInput -> TokenizedGenerateReqInput -> Req -> ScheduleBatch.
   i7's hop4/hop5 assumed a FIXTURE batch with k=8; whether the real path delivers k=8 to the sampler's
   `any(batch.top_logprobs_nums)` branch is an assumption, not evidence. If any site drops/renames/
   defaults k to 0, the controller's re-run probe returns null AGAIN and a probe is wasted (observed
   signature "chosen LP present, top null" is exactly what k=0-at-sampler produces).
2. SAMPLER branch conditions: the exact guard(s) gating topk computation for the frozen shape
   (return_logprob + top_logprobs_nums), incl. any clamp against vocab size and the per-req slice.
3. DETOKENIZER conversion: scheduler BatchTokenIDOut.output_top_logprobs_val -> detokenizer ->
   BatchStrOutput/meta_info field names; i7 hop8 was census+context-read only, i5's per-token text
   builder needed a live tokenizer (mock suffices for field-name/control-flow evidence).

## Hypotheses (falsifiable)
- H-A (intake clean): for the frozen probe shape, k=8 survives every intake site; no silent k->0.
- H-B (sampler branch): with return_logprob=True and k=8, the topk branch is entered and its output
  var is the only source feeding output_top_logprobs_val; no silent-empty branch.
- H-C (detokenizer clean): detokenizer forwards/renames top-logprob fields without dropping; the
  meta_info key is exactly `output_top_logprobs` (+ flat variants), matching i5's read map.
Falsifier: ast-extract the REAL expressions/functions (i7 methodology: ast.unparse, no text-dedent),
execute with labeled stdlib fixtures, hard verdict gate (any harness error blocks all claims), raw JSON
+ file:line provenance per hop. Negative controls: k=0 fixture must reproduce the empty path; a
renamed-field fixture must be caught.

## Outcomes
- All clean => CPU chain now closed end-to-end (request->intake->sampler branch->scheduler->IPC->
  detokenizer->response->read); null is fully attributed to the read-path mismatch; re-run probe is
  maximally de-risked (only GPU compute remains unknown, and chosen-LP-present bounds that).
- Drop site found => third recipe fix BEFORE the probe is spent; documented; request updated.
Either is a valid bounded Iteration. No engine change (HEAD stays 2c854e0-identical).

## Request (queued, unchanged in substance; now fully grounded)
Controller re-runs the numeric coverage probe on BOTH servers with BOTH fixes (shape from i4 + read
path from i5); if non-null on both -> trusted comparison on 2c854e0-identical HEAD -> candidate-x3
slot #1 (FULL gate, exact-token 10/10 mandatory). request_fast_gate=false.

## Budget/checkpoint
Early commit (plan + probe skeleton), execute, raw preserved, final quarter reserved for
iteration-proposal.json + final commit. 45 turns / 1200s wall cap.

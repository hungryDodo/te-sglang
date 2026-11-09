# G3 trial 5 — glm-b: response-field map for the numeric-coverage blocker

## Bottleneck (evidence coverage, not throughput)
Campaign promotion pipeline (canonical F6/F4) is blocked on trusted numeric
evidence: numeric-coverage-001 probes returned `top_logprobs = null` on both
servers. Trial-4 (i4) closed the REQUEST half: `top_logprobs_num` defaults 0
and all builders guard `k<=0`, so the then-used shapes honestly returned no
top arrays. raw-numeric-observation-002 flags the RESPONSE half as
source-supported-but-unexecuted: "frozen probe reads top_logprobs /
top_output_logprobs while pinned tokenizer_manager emits output_top_logprobs".

If that mismatch is real, i4's corrected request shape alone is NOT sufficient:
a re-run probe that still reads `top_logprobs`/`top_output_logprobs` on the
native endpoint returns null AGAIN even with k=20 populated, burning another
controller round and possibly misdiagnosed as an engine drop.

## Hypothesis
- H1 (field-name mismatch, response side): with k>0 requested, the engine
  populates top arrays under `meta_info["output_top_logprobs"]` (nested) or
  `output_top_logprobs_val_flat/idx_flat/shape/null_prefix` (flat variant) on
  native /generate; under standard OpenAI `choices[].logprobs…top_logprobs`
  after mapping on the OpenAI endpoints. NO response field named `top_logprobs`
  or `top_output_logprobs` exists at the native top level — probe reads of
  those names are structurally null there, independent of k.
- H0: engine emits a field the probe does read (mismatch inference wrong);
  then i4's corrected shape alone unblocks and this trial's table still tells
  the controller exactly which read path is guaranteed non-null.

## Method (bounded CPU falsifier, extends i4's ast-execution pattern)
Extract and execute the REAL source expressions — no reimplementation:
1. tokenizer_manager.py response assembly (guard `top_logprobs_num > 0`,
   nested assignment `meta_info["output_top_logprobs"]`, flat builder
   `_build_flat_top_logprobs_fields`) against a faithful finished-request
   state with k=20.
2. OpenAI mappings: serving_chat `_process_response_logprobs` /
   `_build_token_logprobs_from_raw` and serving_completions
   `to_openai_style_logprobs` — execute with populated raw triples, record the
   exact response field names produced.
3. Apply the probe's documented read names (`top_logprobs`,
   `top_output_logprobs`) plus candidate corrected reads to each assembled
   response object -> null/non-null table per endpoint per shape.
Raw output preserved under g3-t5/raw/.

## Deliverable / request
A precise re-run recipe for the controller numeric probe: (endpoint, request
shape, exact response field to read) triplets that are guaranteed non-null for
both reference and candidate servers. This is controller-side evidence
gathering only — evaluator, corpus, seed, reference unchanged.

## Bounds
- NO engine change: python/ stays bit-identical to 2c854e0 (verified empty
  diff) so the promotion candidate remains clean. Artifacts live in g3-t5/.
- No GPU/SSH; no measurements claimed; static/CPU execution only.
- Not requesting fast screen: i2's G3 fast screen (1.0933) already covers this
  engine; a repeat adds noise, not information. request_fast_gate=false.

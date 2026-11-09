# G3 glm-b trial-6 plan — scheduler-side population falsifier (numeric-coverage chain, final un-executed link)

Campaign continuous-20261109-01, Iteration 6. Direction hint (SWA/fast_decode_plan, high-conc
host scheduling) is noted but not a scope restriction; my own i4/i5 learning and the controller's
C5/C7 findings put the highest-value blocker elsewhere and this trial closes the last CPU-side gap
in front of it.

## Context (what is already closed)

- C7: frozen probe DOES send `return_logprob=true, top_logprobs_num=8` (k=0 request-side
  attribution refuted — do not repeat it).
- i5 (executed): response assembly emits native top-k under `meta_info["output_top_logprobs"]`
  (nested) / `*_val_flat|_idx_flat|_shape|_null_prefix` (flat); probe-documented read names are
  structurally null there. Read-path fix delivered.
- i5 next_proposal names the remaining hypothesis verbatim: if a re-run with BOTH fixes still
  returns null, "the drop is scheduler-side population (`BatchStrOutput.output_top_logprobs_val`
  empty at source)" and the next trial builds that falsifier. The controller re-run has NOT
  executed yet (no evidence in this brief). Building it now is deterministic either way:
  - If population is verified CPU-side end-to-end, the re-run recipe is complete and cannot fail
    for a third recipe reason — strengthens the candidate-x3 unblock request.
  - If a real drop exists (e.g. a `logprob_start_len`-dependent guard, a missed detokenizer field
    copy), the re-run recipe needs a THIRD fix and we catch it BEFORE spending a controller probe.

## Hypothesis

- H1 (wiring intact): with the frozen probe's exact shape (`return_logprob=true,
  top_logprobs_num=8`, `logprob_start_len` absent), every scheduler-side guard that gates top-k
  population evaluates true, and each hop forwards the field:
  scheduler/SamplingParams guard -> sampler topk guard (Python-level condition only; GPU compute
  out of scope) -> logprob_result_processor build (incl. null-prefix semantics) ->
  BatchTokenIDOut -> detokenizer_manager -> BatchStrOutput -> (tokenizer_manager, i5 done).
- H0 (drop exists): at least one guard or field-copy in that chain is conditional on something the
  probe shape lacks (e.g. explicit `logprob_start_len >= 0`, streaming mode, `return_text_in_logprobs`)
  or a copy is simply missing — pinpoint hop + line, produce the minimal shape delta that fixes it.

## Method (same ast-extraction pattern as i4/i5; stdlib only)

1. Locate every guard/assignment touching `top_logprobs` / `output_top_logprobs` /
   `logprob_start_len` in `python/sglang/srt/managers/` (+ sampler layer's Python-level condition).
2. `g3-t6/probe_scheduler_population.py`: extract the REAL expressions via `ast` (line-anchored
   asserts; script fails loudly if source moves), execute the pure-Python guards against shape
   fixtures {probe-shape k=8, no start_len; k=8+start_len=0; k=8+start_len=-1; k=0 control},
   and assert the field-copy wiring hop by hop. Distinguish in output: EXECUTED (expression run)
   vs ANCHORED (dataflow line asserted) — no conflation.
3. Preserve raw verdict table at `g3-t6/raw/scheduler_population_output.json`.

## Bounds / non-goals

- No engine change; `git diff 2c854e0 HEAD -- python/` must stay EMPTY (promotion candidate stays
  bit-identical). No GPU/SSH claims. Sampler's torch.topk compute itself is GPU-side and stays
  out of scope — recorded as the single remaining live-run conditional if all Python-side guards
  pass. No changes to evaluator/corpus/seed/reference/protocol.
- request_fast_gate=false: the engine tree is already Fast-screened 1.0933 (i2) on identical
  bits; re-screening identical bits adds no information. Primary ask stays the controller-side
  numeric re-run, now with the recipe validated end to end CPU-side.

## Turn/budget plan (45 turns / 20 min)

plan.md + checkpoint commit now → targeted source reads (~6 turns) → probe write+execute+iterate
(~12) → raw output + findings → iteration-proposal.json + final commit; final quarter reserved.

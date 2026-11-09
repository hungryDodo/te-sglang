# g3-t6 preserved artifact — verification verdict: REJECTED AS EVIDENCE

Trial 7 (session 27) inspected the session-26 recovery checkpoint (commit b2347d2)
per the controller's recovery note. The preserved pair
(`g3-t6/probe_scheduler_population.py` + `g3-t6/raw/scheduler_population_output.json`)
is rejected as evidence. The underlying QUESTION and chain design were sound and
were re-executed fixed in `g3-t7/` (see `g3-t7/raw/scheduler_population_v2_output.json`).

## Why rejected

1. **Script/raw mismatch (provenance break).** The committed raw output was not
   produced by the committed script:
   - The script's verdict gate (`elif harness_errs or not final_rows: HARNESS_ERROR`)
     would have blocked EVERY shape — the raw's own `errs` lists contain non-empty
     error strings for hop2b/hop7 in all shapes — yet the raw shows
     `POPULATED_AT_SOURCE` verdicts.
   - Evidence labels differ: raw says hop6 `"EXECUTED(body, guard implied k>0)"`,
     the script emits `"EXECUTED(if+body, real guard)"`.
   - The script's `counterfactual_val_none_k8` key is absent from the raw.
   The raw came from an earlier script revision; the pair cannot be cited together.

2. **hop2b never executed** — `IndentationError` in all 4 shapes, while the step is
   labeled `"EXECUTED"`. hop2b is the exact branch the frozen probe shape takes
   (`logprob_start_len == -1` → resolve to `len(origin_input_ids)`); the reported
   `resolved_start_len: 0` was the fixture's initial value, not branch output.
   The discrimination entry "logprob_start_len gates output top-k: EXCLUDED by hop2
   execution" therefore cited a hop that did not run.

3. **hop7 never executed** — `IndentationError` in all 4 shapes (`forwarded_rows: 0`),
   yet the verdict text claims "flows through every executed guard incl. streamer".
   The crash-vs-populated dichotomy was never demonstrated at hop7.

Root cause of both failures: `normalize_segment`'s textwrap-dedent re-indentation is
invalid for multi-branch if-segments (mixed indent levels survive dedent). Fixed in
v2 by executing `ast.unparse(node)` (AST-identical to source, format-normalized).

## What survives (re-derived in v2 with clean execution)

The chain topology and the four-way discrimination were correct; v2 confirms them
with zero harness errors on all shapes, including the previously-dead hop2b and hop7,
plus new coverage: hop6 counterfactual (k>0 + None → TypeError crash, not silent
null), hop6 reachability guard (`batch.return_logprob` at batch_result_processor.py:1132),
and a full census of all 39 `output_top_logprobs_val` occurrences with the streamer
else-branch signature analysis (empties chosen LP too → cannot match the observed
"chosen LP present, top null").

Bottom line unchanged in substance but now actually executed end to end: for the
frozen probe shape the scheduler-side chain is POPULATED_AT_SOURCE; the standing
explanation for the observed null remains the g3-t5 response read-path field mismatch.

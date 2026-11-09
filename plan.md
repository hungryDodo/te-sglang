# G3 glm-b trial-7 plan — recovery verification of the session-26 scheduler-population work

Campaign continuous-20261109-01, Iteration 7. The controller's recovery note
(glm-b-recovery-session26) set this trial's entry condition: inspect the preserved
g3-t6 checkpoint, verify or reject it, overwrite the stale iteration-proposal.json,
commit a bounded conclusion early. No broad source sweep, no repeated missing-k
diagnosis. That is exactly what this trial does.

## Executed sequence (verification-first)

1. Compared the committed g3-t6 script against its committed raw output →
   **REJECTED as evidence** (g3-t6/VERIFICATION.md): script/raw provenance break
   (verdict gate would have blocked every shape), hop2b and hop7 never executed
   (IndentationError from a text-dedent re-indent bug) while labeled EXECUTED —
   and hop2b is the exact branch the frozen probe shape takes.
2. Re-executed the SAME bounded question with the extraction bug fixed
   (ast.unparse instead of re-indentation) and a HARD verdict gate:
   `g3-t7/probe_scheduler_population_v2.py` →
   `g3-t7/raw/scheduler_population_v2_output.json`, exit 0, **zero harness errors
   on all 4 shapes**, real guards/bodies executed hop1→hop7 with per-hop
   file:line provenance.
3. Extended coverage beyond the rejected artifact: hop6 counterfactual (k>0+None →
   TypeError crash, not silent null), hop6 reachability guard, full census of all
   `output_top_logprobs_val` sites, streamer else-branch signature analysis.

## Result (bounded, executed)

- **S_probe** (frozen probe shape: return_logprob=true, k=8, start_len absent):
  POPULATED_AT_SOURCE. hop2b resolves −1→12; hop4 batch-level any(k>0) guard true;
  hop5/hop6 append; hop7 forwards the accumulator slice WITH chosen LP.
- Only empty-at-source paths are k=0 (refuted for the probe by C7) and
  return_logprob=False (empties chosen LP too — contradicts observation).
- Between populated and empty, the only failure mode is a CRASH (TypeError), never
  a silent null. The standing explanation for the observed null remains the g3-t5
  response read-path field mismatch; the complete re-run recipe (shape fix + read
  fix, now with the scheduler side executed end-to-end) is unchanged.

## Bounds / non-goals (unchanged from the lineage)

No engine change — `git diff 2c854e0 HEAD -- python/` stays EMPTY (promotion
candidate bit-identical to the twice-proven tree). No GPU/SSH/live server; GPU
torch.topk rows are labeled fixtures (ASSUMED_GPU). No evaluator/corpus/seed/
reference/protocol changes. request_fast_gate=false: identical bits were already
Fast-screened 1.0933 (i2); re-screening adds no information. Speculative/PD/
rust-server/multi-tokenizer paths are off the frozen probe path and out of scope.

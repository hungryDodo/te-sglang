# G3 trial 11 — deepseek-a — NGRAM accept-path (bounded CPU source-executed oracle)

## Context (canonical, not re-litigated)

- F1: NGRAM self-speculation is **BUG_CONFIRMED** (G2-E1). Throughput +21.6% real, but
  greedy correctness fails: ON is pass-nondeterministic 6/24 vs OFF 0/24, divergence set ==
  nondeterminism set, committed tokens 1.0–3.9 nat behind the replay-stable argmax, and in
  1/12 rows ON reports the wrong committed token as its own top-1. Promotion forbidden.
- Open candidates from trial 3: **A** verify-row context misalignment (mask/positions/BFS
  order vs flat `target_predict` indexing) and **B** post-commit KV/bonus misalignment
  (`move_accept_tokens_to_target_kvcache` / `out_cache_loc`).
- Queue for a3 (sync-002): check topology-oracle correspondence to the real implementation
  and the root/pending-token convention; locate the live overlap evidence gap.
- Constraint from prior rounds: stop hand-ported models that share the error they test.
  Use **real source fragments**, **independent goldens**, **fault-injection controls**, and
  a hard verdict gate. Do not claim GPU/kernel/overlap behaviour from CPU models.

## This trial's question (new, falsifiable on CPU)

Does the *real* NGRAM tree bookkeeping keep `retrive_index` / `retrive_next_token` /
`retrive_next_sibling` consistent with the flat verify-row order that
`target_predict = argmax(logits).reshape(bs, draft_token_num)` and the eagle accept kernel
index into — in particular, is the root/pending-token row convention the same on the
draft-commit side (`predict[accept_index]`) as on the verify side?

If the two index spaces differ by even a permutation, the committed token can be a token
verified under a *different* prefix — exactly the F1 symptom (wrong-token-as-own-top-1).

## Method

1. Locate the real functions: `ngram_info.py` tree build, `spec_tree.py`/`eagle_utils.py`
   row assembly, `ngram_worker.py` commit (`predict[accept_index]`), `spec_utils.py`
   `move_accept_tokens_to_target_kvcache`.
2. Execute the **actual source bodies** (AST-extracted with minimal shims if torch et al.
   are unavailable), not re-implementations.
3. Independent golden: a semantic reference for accept indices derived from the tree
   definition (parent→child links), computed separately from the code under test.
4. Fault controls: permuted row order / off-by-one root must make the oracle fail; a
   correct dense path must pass. If the oracle cannot distinguish these, the result is
   reported as INSTRUMENT_INVALID, not as evidence.
5. Preserve raw JSON output; record exact commit and file:line anchors.

## Boundaries

- No GPU, no SSH, no evaluator; CPU-only evidence. No throughput claim. No EXACT claim.
- No changes to measurement/scoring/corpus/seed/reference/protocol.
- Negative or analysis-only outcome is a valid Iteration.

## Deliverables

- `plan.md` (this file), executed oracle + raw output under `g3-t11/`, `evidence.md`.
- `iteration-proposal.json` with the six string fields, `request_fast_gate` decision,
  evidence refs and uncertainty. Initial checkpoint commit early; final commit at end.

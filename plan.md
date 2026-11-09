# G3 trial 9 — deepseek-a — plan

## State entering the trial
- Canonical F1: NGRAM self-speculation is BUG_CONFIRMED (G2-E1). Throughput +21.6% is real
  but greedy outputs are not algorithm-preserving: divergences are exactly the ON
  pass-nondeterminism set, committed tokens sit 1.0–3.9 nats behind the replay-stable
  greedy argmax. Promotion forbidden until a fix is re-adjudicated ALGORITHM_PRESERVING.
- Trial-3 source audit left two live candidates, neither closed:
  - A: verify-row context misalignment (tree mask / positions / BFS order vs the flat
    `target_predict` indexing in `eagle_utils.py:630-655`).
  - B: post-commit KV/bonus misalignment (`move_accept_tokens_to_target_kvcache` /
    `out_cache_loc`, `spec_utils.py:859-910`).
- Prior CPU oracles (i4–i7) are only partially accepted: hand-ported models shared
  assumptions with their own expectations (missing diagonal, reversed ancestor test).
  Review rule now in force: stop hand-porting. Extract and execute the REAL source
  expressions with independent golden fixtures; preserve raw output.

## Chosen direction (new, falsifiable, CPU-only)
Build a bounded oracle that (1) extracts the actual tree-build / accept-index / KV-move
location expressions from the pinned tree via `ast`, (2) feeds them a real NGRAM tree
shape for a repeat-heavy prompt, and (3) compares them against an independently computed
golden (derived only from token-id parent chains, never through the helper under test).

Primary question: for an accepted node sequence that is non-contiguous in the flat verify
batch (e.g. rows [0,2,4], which canonical evidence says can occur even when
`retrieve_index` is identity), does the host-side commit/KV-location code map each accepted
node's KV to the position its token actually occupies in the committed sequence? Any
mismatch is a concrete candidate-B defect, independent of GPU kernels.

Secondary (cheap) check: whether the flat `target_predict` index used at accept time is a
per-request tree index or a global row index, and whether the reshape `[bs, draft_token_num]`
is consistent with the actual flattened row count when per-request draft lengths differ
(candidate A at the host level).

## Method constraints
- No GPU/SSH. All execution is CPU side. No evaluator, corpus, seed, reference changes.
- Source of truth is the pinned engine tree; any model that reimplements rather than
  extracts must be labelled as such and may not be used to claim engine behaviour.
- Preserve raw oracle output as JSON; state limitations explicitly.
- Do not claim EXACT / algorithm-preservation; a Fix is only a proposal until controller
  re-adjudicates ON vs OFF.

## Deliverables
1. `iteration-proposal.json` with string fields bottleneck, hypothesis, patch, tests,
   learning, next_proposal and boolean request_fast_gate.
2. Executed oracle + raw evidence committed.
3. If the oracle falsifies a concrete host-side invariant, a minimal source fix proposed
   but explicitly UNGATED / pending controller adjudication.
4. Negative findings recorded honestly if the invariant holds.

## Budget
45 turns / 20 minutes / USD 3 nominal. Checkpoint an initial commit early; reserve the
last quarter for tests, iteration-proposal.json and final commit.

## Outcome (executed)
Dense-packed in-place compaction is safe for all 120 trees / 720 accepted paths (0 corruptions; monotone accept indices make read-before-write hold). A zero-tail positional fill layout would corrupt the root KV on 719/720 paths (duplicate destination 0, order-dependent) - conditional on the fill layout, decision rule in iteration-proposal.json. Candidate B copy-ordering closed NEGATIVE under dense packing; F1 remains OPEN.

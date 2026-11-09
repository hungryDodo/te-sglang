# G3 / DeepSeek-a trial 4 — plan (NGRAM accept-path CPU oracle)

## Context (authoritative)

- F1 (canonical): NGRAM self-speculation gives real throughput (+21.6%) but is
  **BUG_CONFIRMED** on greedy correctness. No promotion until fixed and
  adjudicated ALGORITHM_PRESERVING.
- DeepSeek-a/i3 qualified finding: "normal fillResult ancestor-copy + diagonal
  construction is causal, so the sanitize guard is not an established fix."
  H2–H4, overlap behavior and the original NGRAM bug remain open.
- Controller instruction for this lane: **bounded fixed CPU
  topology/accept-index oracle with real raw results**, not another source
  sweep. Accepted draft rows can be `[0,2,4]` even when `retrieve_index` is
  identity, requiring compaction to `[0,1,2]`. Do **not** declare the KV mover
  excluded.

## Objective of this trial

Build and run a small, deterministic CPU oracle that models the NGRAM
accept path over fixed synthetic tree topologies and checks
algorithm-preservation invariants between:

1. the flat verify layout (`target_predict = argmax(logits).reshape(bs, N)`)
   and the tree parent/child indexing (`retrieve_index`, BFS order);
2. the committed token sequence (`predict[accept_index]`) vs the greedy
   continuation a non-speculative pass would emit on the same prefix;
3. the KV-cache move (`move_accept_tokens_to_target_kvcache`,
   `out_cache_loc` compaction) — required source-row set vs moved row set.

The oracle is pure Python; no GPU, no network, no evaluator access. Raw JSON
results are preserved in the candidate repo.

## Hypotheses to falsify on CPU

- **H-A (verify-row misalignment)**: flat `target_predict` indexing by
  `last_accepted_retrive_idx` does not identify the tree node whose row was
  verified against the committed prefix when the accepted node is not a direct
  child of `retrieve_index` identity order (accepted rows `[0,2,4]`).
- **H-B (KV compaction)**: `out_cache_loc` / accept-index handling moves KV
  rows for cache positions of accepted draft tokens without compacting sparse
  accepted positions; a subsequent decode attends to wrong KV.
- **H-C (bonus-token slot)**: the bonus token is written from the tree position
  of the last accepted node rather than the seq position of the next slot,
  misaligning the next step's context.

## Method

1. Read the exact accept-path code (`eagle_utils.py`, `eagle.cuh` contract as
   mirrored in Python, `ngram_worker.py`, `spec_utils.py` KV move,
   `spec_tree.py`, `ngram_info.py`).
2. Encode the tree as explicit parent/child arrays in the oracle (no CUDA).
   Simulate: draft tree → target argmax per row → accept loop → committed
   tokens → KV row mapping.
3. Compare against a reference greedy walk on the same toy logits. Report
   divergences with the exact index that caused them.
4. Optionally implement the smallest Python-side fix in the overlay if the
   oracle shows a definite index bug, but only with an oracle test that fails
   before and passes after.

## Deliverables

- `ngram_oracle/` (or similar) pure-CPU scripts + `results/*.json` raw output.
- `iteration-proposal.json` with string fields bottleneck, hypothesis, patch,
  tests, learning, next_proposal, boolean request_fast_gate.
- Git commit(s) early and at end; plan.md updated with results.

## Request policy

- `request_fast_gate=false` this trial unless the oracle yields a
  GPU-verifiable engine fix with CPU proof; a CPU negative finding is a valid
  iteration and does not need a gate.
- Never claim EXACT without numeric evidence; undefined quality budgets stay
  NOT_CLOSED_QUALITY_BUDGET.

## Time/effort bounds

45 turns / 20 min / USD 3 local nominal. Checkpoint plan.md + oracle skeleton
early. Reserve final quarter for results JSON, proposal JSON and commit.

---

## RESULTS (executed, CPU-only)

Artifact: `ngram_oracle/topology_oracle.py`; raw output
`ngram_oracle/results/topology_oracle.json`. `python3 -I`, stdlib only.
Exit status 0. No GPU, no provider, no evaluator used.

Ports validated against repository contracts: `reconstruct_indices_from_tree_mask_cpu`
(`spec.cpp:689`), `verify_tree_greedy_cpu` (`spec.cpp:249`, same algorithm as the
repo's `_ref_verify_tree_greedy` unit-test reference), and
`move_accept_tokens_to_target_kvcache` (`spec_utils.py:859`, i.e.
`fill_front_out_cache_loc` destinations + `fill_accept_out_cache_loc`
compaction-by-skipping-`-1` + pairwise move).

| scenario | accepted rows | committed == greedy | KV mismatches | verdict |
|---|---|---|---|---|
| chain | [0,1,2,3] | yes | 0 | PASS |
| bfs_branch (golden EXPECTED_BFS_MASKS[0]) | [0,2,4,6,7] | yes | 0 | PASS |
| sparse_accept_0_2_4 | [0,2,4] | yes | 0 | PASS |
| root_mismatch_NEG (root != pending token) | [0] | **no** | 1 | FAIL (as injected) |
| no_compaction_NEG (dest = accept_index) | [0,2,4] | yes | 2 | FAIL (as injected) |

Findings:

1. **The accept-index/KV-move arithmetic is algorithm-preserving for sparse
   topologies when the tree root is the pending last token.** Accepted rows
   `[0,2,4]` compact to destinations `[0,1,2]` exactly as the mover does:
   source slots `ocl[0], ocl[2], ocl[4]` -> front slots `ocl[0], ocl[1],
   ocl[2]`; every logical position below the new seq_len holds the right KV.
   This is the controller's open question, answered here on fixed CPU
   topologies: the mover's compaction is *not* the fault line, conditional on
   the root convention.
2. **One wrong root token is sufficient to corrupt the whole context.**
   Because row 0 is unconditionally accepted (`spec_utils.py:606`), if the
   corpus tree root differs from the pending token, the kernel still commits
   `target_predict[root]` and moves the wrong token's KV into the pending
   slot; the oracle detects the first KV mismatch immediately and the
   committed stream leaves greedy. This is the exact failure signature
   reported for NGRAM (off-argmax / behind-greedy commits), so the live
   question is now whether the NGRAM draft path can ever build a tree whose
   root is not the pending token.
3. **Sensitivity controls pass:** the oracle fails when the mover omits
   compaction (destinations = `accept_index`) and when the root is stale, so
   the PASS results above are not vacuous.

Not measured / limitations: the CUDA kernels and the tvm_ffi corpus were not
executed; the target model is a deterministic context->token hash, so this
tests index/topology/KV bookkeeping, not LM numerics. Whether
`_prepare_draft_tokens` always satisfies `total_len == seq_len + 1` (and hence
root == pending) under overlap splicing (`prev_tokens`) and in the
grammar/no-splice branch is **NOT_TESTED**; the oracle only proves that a
violation of that invariant has exactly the observed failure signature.

## Learning / next proposal

- Negative (bounded, CPU): the KV mover and accept-index arithmetic are
  excluded as the F1 root cause *given* a valid root; do not spend further
  trials re-deriving compaction.
- The remaining falsifiable question is root/staging consistency:
  does the NGRAM draft path ever present a corpus tree whose row-0 token is
  not the pending token at `seq_len`? That can be settled with a cheap
  host-side counter/assertion (compare `draft_tokens[row0]` against the last
  committed token at draft prep) rather than another whole-source sweep, and
  it discriminates the two canonical fault lines:
  mismatch -> corpus/overlap staging (B-class), no mismatch -> verify-row
  context (A-class).
- `request_fast_gate=false`: no engine change this trial; a CPU negative
  finding does not need a gate, and gating the unchanged HEAD would not
  advance the NGRAM fault line.


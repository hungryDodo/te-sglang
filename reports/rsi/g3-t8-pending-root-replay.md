# G3 t8 — pending-root / overlap-splice replay oracle (deepseek-a)

Artifact: `g3-t8/pending_root_replay.py` (pure stdlib, `python3 -I`). Raw output:
`g3-t8/raw/pending_root_replay.json`. Exit 0. No GPU, torch, provider or evaluator.

## New question (vs g3-t7)

t7 proved the *accept/KV index arithmetic* is greedy-exact for a well-formed causal
tree (fault controls no_self/wrong_root/bonus_from_parent diverge; ancestor_dropped is
rejection-only). It did not model the **host overlap bookkeeping** that decides *which
token becomes the draft-tree root and which prefix width the verify mask uses*. That is
the controller's a3 "root/pending-token convention" question.

## What is executed

- `input_tails`, `output_tails`, `base_lens` are extracted as AST source segments from
  `ngram_worker.py::NGRAMWorker._prepare_draft_tokens` and **executed** (no hand port).
- Anchored source lines: `prev_tokens = self.prev_token_ids[i*stride : i*stride +
  prev_accept_lens[i]]`, `total_lens.append(base_lens[i] + len(prev_tokens))`, and the
  corpus call site `ngram.cpp:187-201` passing `suffix.back()` as `last_token`.
- Model: deterministic 3-gram-hash target model (independent reference computed from a
  plain greedy loop), perfect corpus chain (root + 3 true continuations), literal
  reconstruct/VerifyTreeGreedy/front-compaction arithmetic, with **engine seq_lens
  tracked separately from physical KV contents** so a bookkeeping desync is observable.

## Result (4 decode steps)

| scenario | exact vs plain greedy | first divergence |
|---|---|---|
| correct_pending_chain | yes | — |
| correct_pending_branch | yes | — |
| fault_stale_root (root = last KV token) | no | token 1 |
| fault_no_prev_splice (overlap context lags) | no | token 4 |
| fault_drop_bonus (accept_lens excludes bonus) | no | token 4 |
| fault_seq_len_plus_one (engine seq_lens +1) | no | token 4 |

All three desync faults first diverge at token 4 = the first token of the **second**
speculative step, i.e. exactly where a per-step root/seq_len/bookkeeping desync can
first surface. Trial 3's observed first-diff positions were `{4, 6}`; this is a
consistency signature, not a root-cause proof.

## Interpretation

1. **Convention is self-consistent.** The draft root is `check_token[-1] = suffix.back()`
   (pending token at position seq_lens). With the overlap splice of exactly
   `accept_lens` accepted tokens, the root reproduces the true pending token, and the
   multi-step replay is token-exact for chain and branch trees.
2. **The overlap splice is load-bearing for correctness, not just draft quality.**
   Skipping it makes the root the last already-cached token; the verify then computes
   row-0 logits under a one-token-stale context and commits a self-consistent wrong
   token. This contradicts the intuition that corpus/lookup errors can only reduce
   accept length.
3. **`accept_lens = num_correct_drafts + 1` must advance seq_lens *and* the KV front.**
   Dropping the bonus desyncs engine prefix width from physical KV and diverges.
4. **Any host/device `seq_lens` drift is a correctness fault**, not a perf-only hazard,
   because the mask prefix width is `seq_lens_cpu` (t5/t6 hazard): a +1 width makes the
   verify rows attend a slot outside the true prefix and diverges.

## Limits (not claimed)

- Perfect-corpus, linear-chain/branch model; no real trie/SAM match state, attention
  kernel, FlashInfer pack, CUDA graph, overlap scheduler, or batch nondeterminism.
- The stale-root/missing-splice faults are *hypothetical* inputs; this does not show
  the live engine performs them. F1 root cause remains OPEN.
- The token-4 signature is a consistency check with trial 3's `{4,6}`, not evidence
  that the live bug is one of these faults.
- No throughput, EXACT, quality-budget or promotion claim. NGRAM stays non-promotable.

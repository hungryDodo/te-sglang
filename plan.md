# G3 Iteration 3 — deepseek-a — NGRAM greedy-exactness fault line

Campaign `continuous-20261109-01`, trial 3. Base commit `47030d8` (NGRAM-on
runtime default; G2 measured ~1.22x aggregate but F1/G2-E1 marks greedy
correctness **BUG_CONFIRMED**).

Objective of this iteration: localize the accept-path defect statically, ship
the smallest defensible change, and leave a decisive, falsifiable probe request
for the controller. **No throughput claim. No promotion.** A precise negative
finding is an acceptable iteration.

## Evidence carried in (authoritative)

- F1 / G2-E1: ON committed tokens sit 1.0-3.9 nats behind a replay-stable OFF
  argmax; ON pass-nondeterministic 6/24 vs OFF 0/24; divergence set == ON
  nondeterminism set; 12/48 divergent instance-passes, first-diff positions only
  {4: 9, 6: 3}; only 512-token prompts diverge; one row reports its own
  committed token as not top-1; no probe instance had `accept_lens == 1`.
- Trial-2 methodology (binding): cross-run token identity is not a correctness
  signal. Only a batch-pinned, same-order ON/OFF probe with OFF replay
  adjudication counts.

## Exactness invariant used to structure the audit

The commit rule (`eagle.cuh` `VerifyTreeGreedy`) accepts a child `c` of `p` iff
`token(c) == target_predict[p]`, then commits `predict[p] = target_predict[p]`
and finally the bonus `target_predict[last]`. So **every committed token is an
argmax of a verify logits row**, and NGRAM is greedy-exact iff, for every node
on the accepted chain, the attention context of that node's row equals
`prefix + committed chain to that node`. Therefore:

1. A wrong *draft tree* (bad corpus match, bad recency state, duplicated check
   suffix) cannot move a committed token off the OFF argmax. It only changes
   accept rates. This rules out the whole corpus/matching side as a root cause.
2. The fault must be one of: (i) a mask/positions inconsistency inside the
   verify input, (ii) a KV slot or prefix-length misalignment in the verify
   window, or (iii) a post-commit KV/bonus move that pollutes the *next*
   context.
3. Consequently the root token must be the pending (not-yet-KV'd) bonus; if it
   were the already-KV'd last token, every verify would see that token twice.

## What this session established (static, no GPU)

### Refuted: upper-triangular / non-causal tree-mask bits (previous H1)

`_sanitize_tree_mask` assumed the corpus can emit a bit at column `j > i`, which
attention would consume while positions/retrieve metadata ignore it. The
generator is provably causal:

- `python/sglang/kernels/jit/csrc/ngram_corpus/result.cpp:12-48` (`fillResult`):
  BFS FIFO, each child records `prev = parent index` at enqueue (`prevs[i] < i`
  always); row `i` is `memcpy(row_i, row_parent, prevs[i]+1)` then
  `row_i[i] = 1`. The parent row's set bits are all `<= prevs[i]`, so every bit
  of row `i` is at column `<= i`. Padding nodes use `prev = 0` (row `{0, i}`),
  still causal.
- `result.h`: `mask` is `std::vector<uint8_t>`, so the `prevs[i]+1` **byte**
  memcpy is exactly a `prevs[i]+1`-element copy; no element-size aliasing.
- `Result::truncate` (result.cpp:128-138) compacts rows with `i*n` stride from
  an already-causal matrix; causality preserved.
- `reconstructIndicesFromTreeMask` (`ngram.cuh:26-91`) and the Triton fallback
  (`reconstruct_tree.py`) both derive depth/parent/child/sibling from the same
  strict-lower-triangular bits, so mask, positions and accept links share one
  contract.

Consequence: **`_sanitize_tree_mask` is a no-op on every mask this corpus can
produce.** It is committed only as an invariant guard + log signal, not as a
fix, and must not be reported as one.

### Refuted: "root token duplicates the last KV token"

`fillResult` puts `last_token = suffix.back()` (the pending bonus, not yet in
KV) at index 0. Verify writes draft position 0 at `batch.seq_lens`, i.e. the
next position, matching EAGLE's `bonus_tokens` root convention
(`eagle_utils.py` `build_tree_kernel_efficient` prepends `bonus_tokens`).
Sync and overlap splice agree: in overlap the check tail is
`output_ids + prev_accept_tokens` (`ngram_worker.py:_prepare_draft_tokens`), so
`prev_tokens[-1]` is the previous round's bonus either way.

### Refuted: prefix-length host/device split (`seq_lens_cpu` vs `seq_lens`)

`_prepare_for_speculative_decoding` builds the attention prefix ones-mask from
`batch.seq_lens_cpu[i]` while positions and the KV read width come from
`batch.seq_lens`. These are not independent: in overlap mode
`FutureMap.resolve_seq_lens_cpu` (`overlap_utils.py:587-647`) takes **both** from
the same `new_seq_lens_buf` filled by the worker's `on_publish(new_seq_lens)`,
with an explicit private-stream D2H + `synchronize()` for the host mirror. They
are equal by construction. A mismatch would in any case change the flattened
mask numel and be absorbed (or not) by the padding in
`ngram_info.generate_attn_arg_prefill` — a hazard worth an assertion, not the
observed 1-3.9 nat deviation.

### Refuted: accept-path bookkeeping / KV mover

- `accept_index[b][0] = root`, `accept_lens = num_accepted + 1`, and
  `predict[accept_index[k]]` is the k-th committed token with the bonus last —
  consistent with `accept_tokens = predict[accept_index]`, scheduler stride
  slicing, and `filter_batch`/`merge_batch`.
- For NGRAM `retrieve_index = b*D + tid` is the identity, so
  `move_accept_tokens_to_target_kvcache` (spec_utils.py:859-910) moves each
  chain node to the front slot it already occupies (dest = positions
  `S..S+accept_lens-1`, sources = the chain nodes at the same positions). The
  bonus stays pending at `S+accept_lens`, which is exactly the next root.
- `tree_topk = -1` is ignored by the CUDA verify launcher; top-k exonerated.

## Still live (not closed here)

1. **Verify-window KV alignment** (invariant 2-ii): the only NGRAM-specific
   input left is the interaction between `assign_extend_cache_locs` (draft
   slots at `[seq_lens, seq_lens+D)`) and the widened CSR read
   (`paged_kernel_lens + D`, `generate_attn_arg_prefill`). A slot handed out
   inside the live prefix, or a plan/indptr disagreement, would corrupt a
   subset of rows exactly like the observed signature. Not resolvable without
   a GPU trace.
2. **Post-commit KV/bonus move** (invariant 2-iii): shared with EAGLE and
   statically consistent, but unproven for the NGRAM lane.
3. **accept_lens == 1 discriminator** (G2-E1 proposal #1): still unrun; no
   probe instance had a single-node chain. Vanish with a 1-node tree -> post-
   commit class; persist -> verify-row context class.

## Deliverable of this iteration

- One guarded, provably-no-op-on-current-corpus invariant check
  (`_sanitize_tree_mask`) that turns a silent cross-component mask-contract
  violation into a counted WARNING in `server.log`.
- The refutations above (negative findings), each with a source reference.
- A controller request: the batch-pinned ALGORITHM_PRESERVING probe on this
  commit plus the `accept_lens == 1` discriminator and a grep for the guard's
  WARNING. No throughput claim, no promotion; NGRAM stays non-promotable until
  the probe is 0/48 on a fixed commit.

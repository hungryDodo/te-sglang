# G3 trial 7 — NGRAM accept-path oracle WITH the real self diagonal (deepseek-a)

Executed artifact: `g3-t7/accept_oracle.py` (pure stdlib, `python3 -I`), raw output
`g3-t7/raw/accept_oracle.json`, exit 0. No GPU, torch, provider, evaluator or network.

## Why this run is different from a3/i4/i5/t6 (C8 correction)

C8 found that the i5 mask-width model *and its expected rows* both omitted the self bit
(`mask[i*n+i]=1`) that the pinned corpus builder really writes, so 12/12 did not certify
real mask assembly. This oracle transcribes the pinned source literally:

| component | pinned source |
|---|---|
| `fillResult` (BFS, `token[0]=last_token`, zero padding `prev=0`, `memcpy` of `prevs[i]+1` bytes, **self bit**) | `kernels/jit/csrc/ngram_corpus/result.cpp:12-46` |
| `Result::truncate` (fillResult is called with `draft_token_num+1`; truncate to D) | `result.cpp:128-137`, call sites `suffix_automaton.cpp:214,282`, `trie.cpp:263,330` |
| `_sanitize_tree_mask` | `srt/speculative/ngram_worker.py:76-103` |
| `reconstructIndicesFromTreeMask` (device) | `kernels/jit/include/sgl_kernel/speculative/ngram.cuh:26-91` |
| `VerifyTreeGreedy` (CUDA) | `.../speculative/eagle.cuh:211-252` |
| KV-move index math (`fill_front_out_cache_loc` + `fill_accept_out_cache_loc_cpu`, -1 skipped/compacted) | `srt/speculative/spec_utils.py:855-897`, `aot/csrc/cpu/spec.cpp:528-556` |

The verify forward is modeled as it really runs: row `i` sees the prefix KV positions plus
exactly the draft columns its delivered mask row admits, ordered by RoPE position
(`positions = depth + seq_len`). So a wrong mask row changes that row's logits, and the
accept walk sees the consequence. The base model is a deterministic full-context argmax
(synthetic; not the real LLM).

## Result (actual run)

| scenario | exact vs plain greedy | accept_len |
|---|---|---|
| exact_chain_bs1_D8 (all drafts correct) | **True** | 8 |
| exact_chain_bs3_D8 (mixed random trees per request) | **True** | 8,1,1 |
| exact_branch_bs2_D8 (top-2 branch trees) | **True** | 4,1 |
| exact_chain_bs2_D12 | **True** | 12,1 |
| FAULT_no_self (drop the diagonal) | False | 1,1 |
| FAULT_ancestor_dropped (drop a parent bit) | **True** | 7 |
| FAULT_wrong_root (root token != pending) | False | 2,1 |
| FAULT_bonus_from_parent (bonus from parent row) | False | 4,1 |

`clean_scenarios_all_exact=true`, `fault_controls_all_diverge=true`,
`sanitize_dropped_bits_total=0` (the guard is a no-op on the real fillResult mask — the
mask is causal by construction), and the KV destination/source mapping matched the true
position→token map in every clean scenario, including batched and branchy trees
(`kv_pairs_ok`).

## New conclusions

1. **C8 gap closed for the accept path.** With the self diagonal included and the exact
   fillResult→truncate→sanitize→reconstruct→greedy pipeline, the accept path commits the
   greedy sequence exactly, batched, with global retrieve indices, `-1` tails and branch
   trees. The self-diagonal omission in i5 was a model artifact; it does not hide an
   accept-path defect in the transcribed logic. The device `reconstruct`/`VerifyTreeGreedy`
   kernels are **logic-identical** to the CPU reference (`verify_tree_greedy_kernel_impl`,
   `spec.cpp:48-92`); source-level only, not live execution.
2. **Under-attention cannot explain F1.** A mask defect that merely drops ancestors makes
   the row argmax wrong and the draft is rejected; the bonus still comes from a
   correct-context last node, so the committed sequence stays a greedy prefix (fault
   control above is exact with accept_len 7). Therefore the F1 divergence requires either
   (a) **acceptance of an off-argmax token** (wrong-context row whose argmax happens to
   equal a corpus draft) or (b) a **wrong-context root or bonus**. Mask rows that
   under-attend are ruled out as a *correctness* mechanism; they are only a throughput loss.
3. **Per-mechanism signatures for the live probe** (the trial-3 `accept_len==1`
   discriminator, now with predicted outcomes):
   - wrong root context / missing self on the root row → first divergence at the first
     committed position with **accept_len == 1**;
   - bonus read from the parent/wrong row → divergence at position `accept_len-1` while
     earlier committed drafts match, **accept_len > 1 possible**;
   - off-argmax acceptance → committed draft differs from replay argmax while
     `accept_len` stays high (drafts "verified" under the wrong row context).
   A live divergent step with `accept_len > 1` therefore **excludes** the pure root-context
   class and points at bonus-row or off-argmax acceptance.

## Limits / not claimed

Pure index/bit arithmetic and a synthetic target model; no torch, CUDA, FlashInfer,
CUDA-graph replay, overlap timing or server.log. KV move is modeled per request (the real
call moves the full flat array; tail entries are (0,0) pairs, not modeled here). No
throughput, quality, EXACT or promotion claim. **NGRAM remains non-promotable** until a
batch-pinned ON/OFF live probe is ALGORITHM_PRESERVING.

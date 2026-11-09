# G3 t11 — NGRAM topology/accept convention with real sources, including batch bs=2

Status: **executed CPU negative finding** (no GPU, no throughput, no promotion claim).

## Question

Do the real NGRAM tree builder, the documented mask→retrieve reconstruct contract,
and the real accept kernel agree on one index convention — and specifically, does
the convention stay correct when more than one request is verified in the same
batch? This matters because canonical F1's 12 divergences were 10/12 at
concurrency 4 (`repeat-512/c4same` 8, `mixed/c4` 2, `c1` 2), and the batch
dimension of the NGRAM accept path had never been executed.

## Components actually executed

| Component | Provenance | Execution |
|---|---|---|
| Tree builder | `python/sglang/kernels/jit/csrc/ngram_corpus/result.cpp:12-51` | compiled and called (`fillResult`) |
| Link reconstruction | `reconstruct_tree.py` docstring contract, independently transcribed | executed in oracle |
| Accept kernel | `python/sglang/kernels/aot/csrc/cpu/spec.cpp:47-92` | extracted **verbatim** by `extract_real_sources.py` (marker-gated, sha256 recorded) and executed |

Cross-check of the contract against the live kernel source: the Triton
`reconstruct_tree.py:110` stores `retrieve_index = b * n + idx` (global), while
`retrieve_next_token` / `retrieve_next_sibling` are per-batch **local** node ids.
The real kernel walks `retrive_next_token[off + cur]` with `cur` local and
indexes `target_predict[last_accept_index]` with the **global** retrieve index.

## Results (raw: `g3-t11/raw/oracle_run.txt`, extraction: `raw/extraction_provenance.json`)

- **T1 bs=1 chain** (real fillResult tree): commit `[11,12,13,21]`, accepted 3 — PASS vs independent golden.
- **T2 bs=1 branch/sibling**: commit `[6,41]`, accepted 1 — PASS.
- **T3 bs=2 mixed, two different requests packed together**: batch 0 `[11,12,13,21]` accepted 3, batch 1 `[6,41]` accepted 1 — PASS, no cross-batch leakage.
- **T4 batch-offset fault control** (`retrieve_index = tid`, missing `b*n`): **DETECTED** — batch 1 commits `[11]` (batch 0's root argmax) instead of `[6,41]`. The oracle is sensitive to exactly the batch-offset bug class it tests.
- **T5 mask-perturbation control**: NOT detected — `[5,7,31]` unchanged.

## T5 is an oracle-design limitation, not engine evidence

T5 added an ancestor bit (`mask[row(7)][row(6)] = 1`). In this mask representation
`next_token[k] = min{i>k : mask[i][k]}` and the old bit `mask[row(7)][row(1)]`
remains set, so node 7 is still reachable as a child of node 1 and the walk is
unchanged. Adding an ancestor bit cannot remove a child edge; a valid control
must **clear** a bit that `next_token` depends on. T5 therefore says nothing
about the engine and is reported as INSTRUMENT_LIMITATION.

## Verdict

For real `fillResult` trees with bs∈{1,2}, the static convention pair
(tree mask → retrieve links → accept walk) is **consistent**: global
`retrieve_index`, local `next_token`/`next_sibling`, root row 0. The F1
concurrency-4 divergence concentration is **not explained** by this static
index convention. It remains attributable to the live path (overlap/draft-commit
handoff, KV mover, or corpus state timing) and stays UNRESOLVED.

## Limits

CPU AOT kernel (documented line-for-line with `eagle.cuh:214-256`), not the CUDA
kernel executed live. The reconstruct step is the documented contract, not the
executed Triton op. Synthetic trees only; no live engine, numeric, quality or
throughput measurement. Tree-size overflow beyond `draft_token_num` and the
`Result::truncate` path were not exercised. The semantic row remap handles
`unordered_map` BFS ordering, but only for trees with distinct tokens.

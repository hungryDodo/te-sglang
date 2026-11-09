# G3 t9 — NGRAM in-place KV-compaction ordering oracle

## Question
After a verify step, `move_accept_tokens_to_target_kvcache` copies the accepted
tree rows to the front of each request's verify window. Accepted rows can be
non-contiguous (e.g. `[0,2,4]`), so a source slot can also be a later
destination. Can this in-place copy corrupt KV (candidate B's last untested
sub-hypothesis)?

## Executed evidence (pure stdlib; no torch/triton/numpy in sandbox)
`python3 -I g3-t9/compaction_order_oracle.py --out g3-t9/raw/compaction_order.json`
(exit 0):

- Extracted and replayed the real chunk loop
  `for start in range(0, N, cap)` from `MHATokenToKVPool._move_kv_cache_impl`
  (`memory_pool.py:3480-3493`): chunk first indices `[0,16,32]`, increasing.
- Reconstruct contract transliteration (`reconstruct_tree.py:31-41,84-105`)
  over all 120 BFS parent trees at n=6; every field re-derived independently
  (BFS depth, children lists, parent chains): 0 violations.
- `eagle.cuh:229-255` accept loop on deterministic random labels: `accept_index`
  is always a root-path prefix and strictly increasing.
- All 720 root-to-node paths x {dense, zero-tail} fill layouts x {tiled,
  read-after-write} copy orders against a golden computed only from the path:
  - dense/tiled: **0 corruptions** (720/720 exact)
  - dense/racy: **0 corruptions**
  - zero-tail/tiled: **719/720 corruptions**, 719 duplicate destinations

## Result
**Negative under dense packing.** `next_token`/`next_sibling` only point to
larger tree indices, so the accepted path `s_0<s_1<...` is strictly increasing.
For a read at packed index `i` of slot `s_i` and a write at packed index `j` of
slot `s_i=j`, `i<j` always holds; the tiled copy loads each chunk before storing
and chunks run in increasing order, so every read precedes the write that could
clobber it. In-place compaction is therefore safe for every well-formed tree and
every accepted path — the `[0,2,4]`-style worry alone cannot corrupt KV.

**Conditional hazard (new, sharp).** If the fill helpers do not densely pack but
emit positional rows with a zero tail, `accept_index=-1` reads
`out_cache_loc[-1]` while `tgt_cache_loc` stays 0, so destination slot 0 is
written from two different sources on every partial accept: the pending root KV
is corrupted (719/720 paths) and the duplicate-address stores are
order-dependent, matching trial-3's ON-only nondeterminism signature.

## Decision rule for the controller
Dump `tgt_cache_loc`/`accept_out_cache_loc` in one NGRAM decode step (or read
`fill_accept_out_cache_loc_func`):
- densely packed to sum(accept_lens) with zero tail -> close candidate B
  copy-ordering as a negative;
- positional rows with a zero tail -> fix by densely packing both arrays (or
  skipping unused tail entries), then re-adjudicate ON/OFF as
  ALGORITHM_PRESERVING before any throughput claim.

## Limits
Transliterations of documented contracts and the C++ accept loop, not executed
GPU kernels. `fill_accept_out_cache_loc_func` was not read this trial; dense
packing for `fill_front_out_cache_loc_func` is inferred from its docstring and
the cumulative-offset kernel at `cache_locs.py:500-531`. No live engine, GPU,
numeric or throughput evidence. F1 stays **OPEN**; NGRAM stays non-promotable.

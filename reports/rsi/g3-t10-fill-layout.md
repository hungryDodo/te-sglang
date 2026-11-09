# G3 trial 10 — executed fill-layout kernel oracle: t9's zero-tail mover hazard is falsified

Worker deepseek-a, campaign `continuous-20261109-01`, Iteration 10. CPU only; no GPU,
no provider measurement, no evaluator. Artifact: `g3-t10/fill_layout_oracle.py`, raw
output `g3-t10/raw/fill_layout.json` (exit 0). This is a bounded **negative** finding
for one candidate-B mechanism; F1 remains OPEN and NGRAM remains non-promotable.

## Question left open by t9

t9 (`0f9a7a9`) proved in-place dense compaction is read-before-write safe because accept
indices strictly increase, but flagged a conditional hazard: *if* `fill_front_out_cache_loc_func`
/ `fill_accept_out_cache_loc_func` do **not** densely pack (positional row-major arrays with a
zero tail and `accept_index=-1` read as `out_cache_loc[-1]`), destination slot 0 would be
written from two different sources on every partial accept. t9 left the fill layout unread
and asked the controller for a live `tgt_cache_loc`/`accept_out_cache_loc` dump.

## Method (source conventions first, independent goldens second)

- `ast.get_source_segment` extracts the **literal** bodies of
  `fill_accept_out_cache_loc` (triton kernel, `kernels/ops/speculative/eagle.py:56-69`),
  `fill_accept_out_cache_loc_func` (`:73-90`), `assign_extend_cache_locs`
  (`kernels/ops/speculative/cache_locs.py:500-531`) and
  `fill_front_out_cache_loc_func` (`:752-800`). These exact source strings are compiled and
  executed under a ~150-line pure-Python `tl` shim (no torch/numpy/triton); five anchor
  assertions fail loudly if the source moves.
- Hand-written golden arrays (not produced by the code under test):
  - case 1 `accept=[0,-1,-1,-1,-1,-1]`, `out=[10..15]` -> accept_out = front_out = `[10,0,0,0,0,0]`;
  - case 2 `accept=[0,2,-1,-1,4,6,-1,-1]`, `out=[100..107]` -> accept_out `[100,102,104,106,0,0,0,0]`,
    front_out `[100,101,104,105,0,0,0,0]` (row-local non-contiguous accepted path, batched).
- Exhaustive: all 5050 accept patterns for bs in 1..3, width in 1..4 (every subset per row);
  random: 1500 cases bs 1..4, width 5..8. For each, compare both executed outputs to a
  separately written specification expression and analyse the `move_kv_cache(tgt=front_out,
  src=accept_out)` element pairs.
- Positive controls: (M1) the same extracted kernel with the `src > -1` guard changed to
  `src > -2` (the t9 sentinel hypothesis), and (M2) a positional row-major front layout.
  The oracle must flag both.

## Executed result

| check | result |
|---|---|
| literal hand-golden cases 1,2 | PASS (accept/front arrays exactly as hand-computed) |
| exhaustive cases | 5050/5050 match golden; 0 negative-index loads; 0 conflicting destinations; 0 pairs with src < tgt |
| random cases | 1500/1500 match golden; same zeros |
| M1 sentinel-unskipped mutant | DETECTED: five `out_cache_loc[-1]` reads, golden mismatch |
| M2 positional-front mutant | DETECTED: destination 0 receives sources {0, 104, 106} |
| source anchors (CPU AOT `if (src > -1)`, bounds check, `N = tgt_loc.numel()`, zero-filled allocs) | all true |

Actual layout (executed, not inferred): **both helpers densely pack globally**.
`fill_accept_out_cache_loc` writes each non-sentinel entry to
`dst = count(non -1 entries before pid)` and never loads `out_cache_loc` for `-1`
(the literal guard `if src > -1`). `fill_front_out_cache_loc_func` calls
`assign_extend_cache_locs`, whose literal `out_offset = tl.sum(end - start, axis=0)`
is the prefix sum of per-row front lengths — also a dense global pack. The two dense
arrays are element-wise aligned, and valid destination slots are unique.

Because `move_kv_cache` runs `N = tgt_loc.numel()` (full `bs*width`, source anchor), the
unused tail indices pair the zero-filled `tgt` with the zero-filled `src`, i.e. destination
KV slot 0 receives source slot 0 repeatedly. KV slot 0 is the reserved dummy padded slot
(`mem_cache/allocator/token.py:42-46`: `free_pages = torch.arange(1, self.size + 1)`, with
the comment "The padded slot 0 is used for writing dummy outputs from padded tokens"), so
real `out_cache_loc` values are never 0, and these duplicate stores are same-value writes
to the dummy slot — no root-KV corruption and no order-dependent distinct-value store. For all valid pairs src >= tgt, so the chunked
in-place copy stays read-before-write safe.

## Conclusion

**t9's conditional zero-tail hazard is falsified at the pinned source, and candidate B's
mover sub-hypothesis is closed NEGATIVE**: under the actual fill helpers, `move_accept_tokens_to_target_kvcache`
cannot by itself corrupt KV or create the ON-only nondeterminism signature. The requested
live dump no longer decides this question; it would at most confirm a source-level result.
F1's remaining live surface is candidate A (verify-row context/positions/attention packing)
or another mechanism outside the mover; this trial makes no claim about them.

## Limits

- Triton kernel bodies were executed under a Python shim, not on CUDA; the move copy,
  `use_hnd` advanced-indexing branch and native move were not executed. The shim's
  negative-index read is recorded and the two fault controls show it is sensitive.
- No live engine, GPU, numeric or throughput evidence. No EXACT/quality/promotion claim.
- NGRAM tree construction, verify logits and the accept walk are untouched here; t7/t8
  results stand on their own terms. F1 stays BUG_CONFIRMED and NGRAM non-promotable.

# G3 t6 (deepseek-b) — host-mirror reachability + index-equality oracle

Status: **CPU-executed, no engine change, no gate requested.**
Base: `e345716` (inherits t3 `33e5ac4` decode host mirror + t4 `a306e40` prefill host mirror).

## Question (new)

The pending t3/t4 candidate replaces `kv_last_index = kv_indptr[-1]` (a 0-dim CUDA
scalar -> implicit `.item()` D2H) with the host int `paged_kernel_lens_sum`. The
prior trial's oracle only checked integer equality on gate-shaped decode inputs.
Two things were never executed:

1. **Reachability.** The changed lines sit behind
   `if use_sliding_window_kv_pool and not use_swa_source`, where
   `use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated`.
   Is the changed line ever reached, under which configurations, and does it
   remove any D2H at all?
2. **The SWA-clamped / padded / prefill shapes.** `update_sliding_window`
   clamps `seq_lens` to `sliding_window_size + 1` on device and computes the
   host sum from a separately clamped CPU mirror. Equality is exact only if both
   clamps agree. That case was not in the prior 27-case matrix.

## Method

`verify_host_mirror_reachability.py` (stdlib only) parses
`python/sglang/srt/layers/attention/flashinfer_backend.py`, line-anchors 11 real
source constructs (fails loudly if any moved), compiles and **executes the real
guard expressions** over 48 configurations, then replays the exact integer
semantics of the caller (`torch.clamp` -> `min`, `cumsum(...)[bs]` -> `sum`)
over 34 cases. Raw output: `reachability.json`.

## Executed results

| quantity | value |
|---|---|
| reachability configurations | 48 |
| rows where the translate block is reachable | 12 |
| decode rows taking the host mirror | 6 |
| prefill rows taking the host mirror | 1 |
| rows dead because `translator.reads_are_translated` | 12 / 12 |
| replay cases | 34 |
| index divergences outside controls | **0** |
| sensitivity controls flagged | **2 / 2** (stale CPU mirror; below-window off-by-one) |
| cases where the patch removes a D2H | 30 |
| cases where the caller pays the D2H anyway | 2 (all `seq_lens_cpu is None`) |

### Reachability envelope (the decision-relevant part)

The host mirror is reached only when **all** hold:

- `dispatch_reason == SLIDING_WINDOW` and `_swa_kv_pool is not None`;
- `not translator.reads_are_translated` — with a translating/unified pool the
  whole `translate_loc_from_full_to_swa` block is skipped, so the changed lines
  are **dead code** (12/12 config rows);
- the `spec_info`-less rebuilt-indptr branch ran (decode has no
  `custom_kv_indices` parameter at all, so no extra guard is needed there; the
  prefill guard's `custom_kv_indices is None` covers the dq branch);
- `paged_kernel_lens_sum` is a Python `int`.

Even when reached, the patch removes **one** redundant device read per eager
decode step. When `seq_lens_cpu is None`, the caller itself executes
`paged_kernel_lens_tmp.sum().item()` at line 1696, so the D2H is paid anyway and
the patch saves nothing.

### Index equality

All 32 non-control cases are index-identical, including the ones the prior
oracle did not cover: SWA-clamped lenses above the 511 window, mixed
above/below, exactly 512 and 513, CUDA-graph-padded zero-length lanes
(56+8, 120+8, 248+8), `seq_lens_cpu is None`, prefill 16 x 8192 and
16 x (512 prefix + 8192 extend).

## What this does and does not establish

- **Establishes**: within the modelled reachable envelope the host int equals
  `kv_indptr[bs]`, and the patch cannot change the translate slice bound. The
  two sensitivity controls prove the equality is not vacuous.
- **Does not establish**: a runtime measurement, GPU behaviour, or that the
  frozen launch actually enters the envelope. The frozen flags (unified pool?
  `seq_lens_cpu` present?) are controller-side. If the frozen engine uses a
  unified SWA allocator, this candidate is provably inert.
- **Prior measurement context**: trial-1's sibling sync-removal family measured
  1.0018 KEPT_NEUTRAL on a4, with decode points inside run-to-run noise.
  Expectation for this candidate is likewise ~0 aggregate gain; it is a
  D2H-hygiene patch, not a speedup.
- `NOT_CLOSED_QUALITY_BUDGET` is untouched: no precision change is involved.

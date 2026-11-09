# NGRAM verify mask-width oracle — raw evidence

Artifact: `mask_width_oracle.py` (pure stdlib; python3 -I). Raw output:
`results/mask_width_oracle.json`. Exit 0. No GPU, no torch, no provider, no
evaluator.

## What is modeled (pinned source)

- `ngram_worker.py:352-361` builds `ones_masks[i]` with shape
  `(draft_token_num, batch.seq_lens_cpu[i])` **before** the blocking accept
  sync. So the custom mask's prefix block width is the **host** prefix `h_i`.
- `ngram_info.py:60-117` (`generate_attn_arg_prefill`) computes the KV read
  width from `paged_kernel_lens + draft_token_num`. In
  `flashinfer_backend.py` (`paged_kernel_lens = seq_lens`, device), so the row
  width the kernel consumes is `d_i + D` with `d_i` the **device** prefix.
  It also computes
  `mask_numel = paged_kernel_lens_sum * D + D^2 * bs` and, when the built mask
  is shorter, silently appends `True` entries; when longer, the surplus is not
  part of `mask_numel`.
- Column contract: `c < d_i` = prefix KV position c; `d_i + j` = draft node j
  (draft slots are packed in node order, matching `assign_extend_cache_locs`).
- Tree rows are proper BFS ancestor masks (the `fillResult` contract).

## Result (actual run)

| scenario | real rows exact | mask built / expected | verdict |
|---|---|---|---|
| consistent_host_device (bfs) | 12/12 | 120 / 120 | PASS |
| consistent_host_device_chain | 12/12 | 120 / 120 | PASS |
| host_lags_device_k1 (h=d-1) | **0/12** | 108 / 120 (+12 True) | FAIL |
| host_lags_device_k2 (h=d-2) | **0/12** | 96 / 120 (+24 True) | FAIL |
| host_ahead_device_k1 (h=d+1) | **2/12** | 132 / 120 (12 dropped) | FAIL |
| graph_batch_padding (2 fake reqs) | 12/12 | 120 / 160 (+40 True) | PASS |

First mismatch for `host_lags_device_k1` is request 0, row 0: attends extra
column 8 and misses column 4 — i.e. row 0 of the first real request already
attends to a draft slot it must not see and does not attend the pending slot
it must see. Real rows are wrong 12/12, 12/12 and 10/12 in the three drift
scenarios; nothing raises in any of them.

Interpretation:

1. **Exactness is conditional on `h_i == d_i` for every request.** Under the
   documented FutureMap contract (`resolve_seq_lens_cpu` derives the host
   mirror from the same `new_seq_lens_buf` with an explicit synchronize; see
   G3-i3 audit) this holds, and the assembly is exact on both tree shapes and
   under legitimate CUDA-graph batch padding (the appended True tail lands
   only on fake rows).
2. **The recovery branch cannot distinguish legitimate graph padding from a
   host/device width drift.** A one-token drift produces an all-True tail that
   is consumed as real mask columns: silent, no assertion, and every real row
   corrupts. A guard that only checks `numel` cannot separate the two cases.
3. **Reachability of `h_i != d_i` is NOT established.** This is a latent
   hazard, not a claim about the F1 root cause. If it ever occurs, the
   signature is nondeterministic (stale host values vary with overlap timing)
   — which is the observed ON-pass nondeterminism — but the FutureMap audit
   argues it does not occur, and a one-token width drift would also be hard to
   reconcile with first-diff positions 4/6 rather than 0/1.

## Falsifier proposed to the controller (not run here)

Add a diagnostic counter (log-only) at custom-mask construction time that
compares each request's host prefix used for `ones_masks` against the device
`paged_kernel_lens` at the same forward, and emits the count of requests where
they differ plus the total width delta. Run it inside the existing
batch-pinned ON/OFF NGRAM probe. Decision rule: any non-zero drift on real
requests -> width-drift class confirmed (then fix the host mirror source);
zero drift with divergence reproduced -> width class excluded, leaving the
physical KV-slot plan (`pack_read_stream`/`assign_extend_cache_locs`) and
post-commit move as the live GPU-visible surfaces.

## Limits / NOT claimed

- Pure index arithmetic; the flashinfer/FlashInfer kernel, CUDA graph
  capture/padding mechanics, translator.pack_read_stream and the target model
  were not executed. Fake graph-padding requests are modeled as `d=1` blocks
  filled by the appended True tail.
- No numeric quality, throughput or EXACT claim. No NGRAM promotion.
- `h_i != d_i` reachability is NOT_TESTED on hardware and is argued against
  by the existing FutureMap audit; this file quantifies consequence only.
- The `attended_padding_tail` field in the JSON is structurally always false
  (columns are bounded by the row width) and is not used in any verdict.

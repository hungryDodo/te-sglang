# NGRAM verify-window custom-mask block alignment — raw evidence

Artifact: `verify_window_oracle.py` (pure stdlib, `python3 -I`). Raw output:
`results/verify_window_oracle.json`. Exit 0. No GPU, no torch, no provider, no
evaluator.

## New question (vs. t5 mask_width_oracle)

t5 modeled ONE request's row width. The FA2 CUDA-graph verify path packs a
**flat concatenated** mask whose per-request block offsets come from the
**device** widened lengths (`kv_indptr`, `pack_verify_mask.py`), while NGRAM
**builds** the mask with **host** per-request widths
(`ngram_worker.py:352-361`, `seq_lens_cpu[i]`). With bs>1 a drift in one
request can therefore misalign **every later request's block**, not just its
own rows. t5 could not see that.

## Ported arithmetic (pinned source, hand-ported; stdlib only)

- Builder: per request `[D rows][h_i + D cols]`, prefix ones then causal tree
  row; concatenated; then `ngram_info.generate_attn_arg_prefill:104-117`
  computes `mask_numel = paged_kernel_lens_sum*D + D^2*bs` from **device**
  lengths and appends global `True` bits when the built mask is short (surplus
  dropped when long).
- Packer: `flashinfer_backend.py:263-285` computes
  `packed_lengths = (kv_lens_host*D + 7)//8` and asserts only
  `packed_indptr[-1] <= buf.numel()`; `pack_verify_mask.py:12-24` reads request
  i's block at `kv_start*Q` with `kv_start` = **device** cumulative widened
  length and `count = (d_i + D)*D` bits.
- Delivered bit for (row r, col c) is compared against the mask the builder
  intended, using the row width `d_i + D` the packer/flashinfer must assume.

## Executed result (D=4, host prefixes [4,5,6], device [4,5,6] baseline)

| scenario | req0 | req1 | req2 | note |
|---|---|---|---|---|
| aligned h==d | exact 32/32 | exact 36/36 | exact 40/40 | no drift |
| drift req1: d=6, h=5 | exact | **18/40 mismatch** | **16/40 mismatch** | req2 is itself aligned yet corrupted |
| drift req0: h=5, d=4 | **12/28** | **16/36** | **16/40** | every later block shifted |

First mismatches are ordinary mask bits, e.g. mid-batch drift req1 row0 col9
(expected 0, delivered 1) and req2 row1 col2 (expected 1, delivered 0). No
exception is raised in any scenario: the global `True` padding makes the total
`numel` check pass.

## Interpretation (bounded)

1. Exactness requires `h_i == d_i` for **every** request in the batch, because
   the flat mask is addressed by device-cumulative offsets while it is built
   with host widths. Under the documented FutureMap equality this holds
   (t3 audit), so this is not a root-cause claim.
2. **New**: a single drifted request corrupts later requests' mask blocks in
   the same batch (req2: 16/40 bits wrong despite being aligned). With the
   batch-pinned 48-pass probe, one timing-dependent stale host length could
   therefore explain divergence clustered in a subset of passes without any
   single request being repeatedly at fault.
3. The only size check (`packed_indptr[-1] <= buf.numel()`) is a capacity
   bound; the `True` padding in `generate_attn_arg_prefill` absorbs a *short*
   mask silently, so drift has no assertion anywhere on this path.

## Limits (not claimed)

- Pure index/bit arithmetic; the real Triton packer, FlashInfer FA2 kernel,
  CUDA-graph buffer capture and server.log were not executed.
- The row-major `d_i + D` width assumption for the delivered-bit mapping is the
  documented column contract (t5); if flashinfer's internal order differs, the
  mismatch counts change but the block-offset shift argument is unchanged.
- **Reachability of h_i != d_i is still NOT established**; this is a hazard and
  a falsifier, not the F1 root cause and not a fix.

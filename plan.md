# G3 trial 1 (glm-b): remove remaining blocking D2H syncs in the SWA prefill plan

## Baseline
HEAD 2c854e0 = aa45af5 lineage (SWA host kv_indices bound + fast_decode_plan),
best measured tree: 1.0920 / 1.0984, 10/10 exact-token pass (G2 namespace).

## Bottleneck (code analysis; no prefill trace exists)
Decode steady state is closed (G2 t2 in-situ trace: GPU busy 99.2% at C256,
attention bytes at roofline 242-270 GB/s; lm_head recoverable ~2.4% of point =
sub-noise). The **extend/prefill plan path** on this SWA hybrid model still pays
~5 blocking device→host reads per prefill batch in
`python/sglang/srt/layers/attention/flashinfer_backend.py`
(`FlashInferIndicesUpdaterPrefill.update_sliding_window`, wrapper 0):

| Site | Line (base 2c854e0) | Sync |
|------|-------------|------|
| A | 2077 | `paged_kernel_lens.sum().item()` on a CUDA tensor |
| B | 2157-2159 | `_build_swa_prefix_custom_mask`: 3× `.detach().cpu()` (runs even when mask is None) |
| C | 2347 | `kv_indptr[-1]` device scalar bounding the SWA translate (same pattern aa45af5 fixed for decode, +9.8%) |

Every chunked-prefill/mixed batch at high concurrency stalls the forward thread
in `init_forward_metadata` until queued GPU work drains, then the GPU idles
while the CPU finishes planning — the same mechanism measured in the trial-2
probe (decode side: syncs 3→0.11/step, GPU idle 3.4→0.4 ms). Prefill Heavy
moved only 1.9% under the decode-only fix because these prefill sites were
untouched.

## Verification (static, by code reading — no GPU in this sandbox)
All mirrors are exact integer arithmetic; equality is by construction, not by
copying values at runtime:
1. `extend_prefix_lens_cpu` == device `extend_prefix_lens` elementwise:
   both built from the SAME host list (forward_batch_info.py:1237-1240), or
   `.cpu().tolist()` (:1931), or the dummy `[0]` (:1908). Guard
   `prefix_lens_given` excludes the spec-verify derived `prefix_lens`
   (update_sliding_window derives it only when `prefix_lens is None`).
2. `seq_lens_cpu` == device `seq_lens` elementwise for extend batches:
   schedule_batch.py prepare_for_extend builds both from the same
   `[r.extend_end]` list (seq_lens_tensor→device, seq_lens_cpu→host, int64).
3. Site A identity: `paged_kernel_lens = prefix - clamp(prefix-w,0) =
   min(prefix,w)` elementwise for ints ≥ 0 → host `sum(min(p,w))` == device
   `.sum().item()`.
4. Site B: `extend = seq_lens - prefix` mirrored from the two host pairs (2,3);
   `kv_start = max(0, p-w)` mirrors the caller's `effective_start`
   (kv_start_idx := effective_start at the only two call sites, 2100/2108).
   Loop arithmetic identical on host ints → bit-identical mask tensor or None.
5. Site C invariant: call_begin_forward fills `kv_indptr[1:bs+1] =
   cumsum(paged_kernel_lens)` (:2331) in the guarded branch →
   `kv_indptr[-1] == paged_kernel_lens_sum` (caller-computed from the same
   tensor). Guard `spec_info is None and custom_kv_indices is None` excludes
   the dq path (:2327, cumsum of dq_paged_kernel_lens ≠ sum) and spec paths;
   custom-KV batches never reach update_sliding_window at all (:2045 raises).
   Site C is only reachable from update_sliding_window wrapper 0
   (`use_sliding_window_kv_pool` defaults False in every other caller), where
   `paged_kernel_lens_sum` derives from the same `paged_kernel_lens` tensor
   passed to the cumsum.
6. Sweep for remaining blocking reads in the file (`.item()/.cpu()/.tolist()`):
   the rest are host-tensor reads (675, 954-957, 1694, 2524, 2641), cross-
   attention paths unused by this model (1739, 1749, 2263), non-ragged /
   multimodal / deterministic / use_paged branches (2125), gpu-only batches
   (2133), spec/draft paths (2204-2206 fallbacks, 2689), or fallbacks kept for
   the derived-prefix paths this patch excludes. Net: 5 blocking reads → 0 on
   the SWA standard-scheduler extend hot path; all fallbacks preserved.
7. `py_compile` passes. Plan args, slice bounds, and mask bits are bit-identical
   → exact-token output is expected unchanged; the gate's exact-token check is
   the arbiter (same as aa45af5).

## Hypothesis
Removing A/B/C eliminates per-extend-batch pipeline stalls; prefill-heavy and
mixed points improve; decode-only points unchanged (their path is untouched —
predict ≈1.0 per-point there). Expected effect direction is positive but
magnitude unknown (0 if prefill batches never bubble; see Uncertainty).

## Patch (this commit, on top of 2c854e0)
- A: host `sum(min(p, window))` when `prefix_lens_given and
  extend_prefix_lens_cpu is not None and len == prefix_lens.numel() and
  seq_lens_cpu is not None and numel == prefix_lens.numel()`; else device
  fallback unchanged.
- B: `_build_swa_prefix_custom_mask(..., prefix_lens_cpu=None, seq_lens_cpu=None)`
  keyword-only host inputs; device fallback unchanged for all other callers.
- C: `kv_last_index = paged_kernel_lens_sum` when `spec_info is None and
  custom_kv_indices is None`; else `kv_indptr[-1]` unchanged.

## Tests (requested from controller; none runnable locally — no GPU)
- FULL trial gate (not fast-screen) on the trial commit vs campaign baseline;
  exact-token 10/10 required.
- Watch points: Prefill Heavy (expected owner of gain), Prefill Ceiling /
  mixed points, Decode points (expect ≈1.0, path untouched), Short C32
  (known-noisy point; do not read a single gate's per-point delta there).
- Marginal interpretation: compare against the aa45af5-lineage reference band
  1.0920/1.0984 (G2 namespace, same tree minus this patch) — directional only;
  a formal candidate×3 for the promotion case must be re-run in the G3
  namespace per canonical F4.
- Risk: if extend batches are long enough that syncs never bubble → score
  within noise of the lineage band → NEGATIVE finding that bounds prefill
  syncs as non-recoverable (closes trial-2's open question without a trace).

## Uncertainty
No prefill-side trace exists; magnitude is extrapolated from the measured
decode-side mechanism of the same sync class (aa45af5: +9.8% aggregate from
3 syncs/decode-step). Here ~5 syncs per EXTEND batch, but extend steps are
sparser and longer than decode steps → predicted 0.5-3 ms/batch saved;
aggregate effect unknown until gated. Correctness risk assessed low (static
proof above), but exact-token gate is mandatory, not assumed.

## Negative findings carried forward (do not re-litigate)
- Decode C128/C256 is bytes-bound at roofline, GPU busy 99.2% — no host/sync
  lever remains at high concurrency (G2 t2 in-situ trace).
- lm_head GEMM (~166 GB/s) recoverable ~2.4% of Decode Ceiling point — sub-noise.
- Split-kv plan tuning: neutral-to-worse (G2 t2 microbench at engine shapes).
- torch.compile for eager decode: boot-time risk, unproven, low priority.
- H1 tensor-core dispatch: already true for this model's GQA group size 4.

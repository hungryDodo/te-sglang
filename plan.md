# G3 trial 1 — glm-a (campaign continuous-20261109-01, iteration 1)

## Starting state

HEAD = `83e7b75` (fp8-KV remap `e46f99d` + SWA host-sync hunk), i.e. exactly the
F4 combo whose G2 candidate×3 (37–39: 1.1374/1.1243/1.1274, all GATE_VALID) was
mis-attributed to phase1 trial-6 and is therefore unusable for G3 promotion.
Per F4, G3 must re-run candidate×3 in its own namespace before any promotion
claim. Per F2, the fp8-KV half is COMPUTE_DIFFERS with quality budget
NOT_CLOSED_QUALITY_BUDGET — no deployability claim is made here.

## Bottleneck (from existing evidence, no new measurements claimed)

- Decode on GB10 for Spark-X2.6-1.7B is memory-bandwidth-bound (trial-1: TPOT
  17.6/26.2/36.1/42.2 ms at bs 32/64/128/256 vs ~12.4 ms weights-only floor).
  Biggest proven levers so far: KV bytes (fp8-KV, ×1.1273 on a1) and host
  blocking (SWA D2H removal, ×1.0543 standalone on a2, +0.19–0.5% marginal on
  a1 over fp8-KV).
- The one measured host-blocking event (14.4 ms/step stream sync, commit
  `83e7b75` message) came from one device-scalar read in
  `FlashInferIndicesUpdaterDecode`. The same updater family runs every decode
  step; any sibling device-scalar read is the same class of per-step tax.

## Hypothesis (H1, exactness-first)

Remaining device-scalar reads / implicit D2H syncs in the flashinfer decode
index-update path (same family as the fixed `kv_indptr[-1]` slice bound) can be
replaced with host mirrors (e.g. `seq_lens_cpu`) that are already maintained.
These are metadata-only changes: like `83e7b75` (verified C1 agree=1.0,
meanDlogprob=0.0 vs parent), they should be compute-exact — an exact runtime
alternative, no quality budget needed.

## Plan

1. Audit `flashinfer_backend.py` decode/extend indices updaters + the SWA
   hybrid-pool translate path for remaining device-scalar reads
   (`kv_indptr[-1]`, `.item()`, `.max()`, slicing device tensors with device
   scalars). Record every site found, fixed or not, in this file.
2. Fix the sites where a host mirror provably exists and is lane-consistent
   (same guard pattern as `83e7b75`: fall back to the device read if the mirror
   is absent or wrong length — never a correctness guess).
3. Negative findings welcome: if the audit shows no further sync sites, that is
   the result.
4. Proposal: (a) candidate×3 re-run of `83e7b75` in the G3/glm-a namespace
   (F4 requirement) — this is the primary ask and needs no new code;
   (b) fast gate for the new audit commit(s) if any site was fixed.

## Audit result (static; no GPU claims)

Sites audited in `flashinfer_backend.py` indices-updater family:

- FIXED — `update_sliding_window`, ragged SWA branch (formerly
  `paged_kernel_lens.sum().item()` on a device tensor, per SWA extend step).
  `paged_kernel_lens` is elementwise `min(prefix_len, window)`; when
  `extend_prefix_lens_cpu` is present the same total is computed host-side
  (the non-SWA ragged branch already sums this exact list host-side, so the
  lane correspondence is established in-tree). Device read kept as fallback.
  Expectation: removes one stream drain per SWA-bearing extend; prefill-heavy
  and standard categories are where it can show. Magnitude unknown — could be
  noise-level (F5 no-op upper bound on a2 was 1.019).
- Already host-mirrored (no action): decode updater kv_last_index
  (`83e7b75`), `global_override_indptr_cpu` cumsum, decode wrapper
  `torch.clamp(seq_lens_cpu, max=window+1).sum().item()` (CPU tensor — host
  math), non-SWA ragged `sum(extend_prefix_lens_cpu)`,
  `prefix_is_full_seq` SWA decode fast path, `seq_lens_sum` threading,
  `extend_prefix_lens_cpu`-guarded prefix sum in wrapper_id==1.
- Left as-is (documented, not fixed):
  - `update_sliding_window` non-ragged branch when `prefix_lens` is caller
    provided (spec-verify path, `num_accept_tokens` involved): no proven host
    mirror for the caller-provided prefix lens; only reachable under
    speculative decoding, which this serving path does not enable.
  - `encoder_lens.sum().item()`: multimodal-only.
  - `kv_indptr[-1]` fallback at the decode hunk: guarded, cold when
    seq_lens_cpu is present.

## NGRAM fault-line notes (for canonical entry #2, no fix attempted)

Read `speculative/ngram_worker.py` (578 lines): acceptance is delegated to the
shared `eagle_sample` (eagle_utils.py:521) used by EAGLE workers, so the
G2-E1 signature (accepted tokens 1.0–3.9 nat behind argmax — far beyond
near-tie scale) points at the NGRAM-specific inputs to that shared verifier,
not at the verifier itself. Prime suspects for the controller/deepseek-a:
(a) tree topology / `retrieve_index`-`retrieve_next_token` links vs the
draft-token layout in `_prepare_draft_tokens` (accept decisions made against
the wrong node's logits would accept clearly-sub-argmax tokens), (b) the
overlap-mode splice of `prev_token_ids`/`prev_accept_lens` lanes
(ngram_worker.py:263-285, 400-419) shifting positions by a lane, (c) grammar
host-tree link derivation `_derive_tree_links`. Untested here — no execution
ability; recorded as input, not as a finding.

## Non-goals / deferred

- NGRAM fault-line fix (canonical entry #2): shared code lives here, but the
  G2-E1 bug was measured in deepseek-a's namespace; attempting a blind fix
  without any execution ability in this trial risks an unverifiable change.
  Recorded as next_proposal with the audit evidence I gathered.
- fp8 weights W8A16 GEMM: precision-class, blocked on quality budget.
- Triton attention backend, graph coverage >64: killed in trials 1–2; do not
  re-gate.

## Uncertainty

- Whether any remaining sync site is actually hot at the frozen gate's batch
  sizes/contexts; only the controller's measurement can tell. A no-op-ish
  result is possible (F5 no-op upper bound on a2 was 1.019).

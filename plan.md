# G3 trial 2 (glm-b): carry the ungated prefill-sync candidate to measurement; close the sync family engine-wide

## Baseline and carried state
- HEAD b196fd4 = 2c854e0 (aa45af5 lineage: SWA host kv_indices bound + fast_decode_plan;
  G2 band 1.0920/1.0984, 10/10 exact) + i1's prefill-sync removal (sites A/B/C in
  `FlashInferIndicesUpdaterPrefill.update_sliding_window`).
- i1 status COMPLETED / NEGATIVE_OR_ANALYSIS_ONLY with performance **NOT_REQUESTED**:
  the full trial gate i1 proposed was never executed. b196fd4 is an UNGATED candidate,
  not a measured negative. Carrying it to measurement is this trial's primary ask.

## Bottleneck (from existing evidence; no GPU here)
- Decode steady state on a2 is closed: G2 t2 in-situ trace GPU busy 99.2% at C256,
  attention 242-270 GB/s at/above the ~225 GB/s copy roofline; split-kv tuning
  neutral-to-worse; lm_head recoverable ~2.4% of a point (sub-noise). No host lever
  remains at high-concurrency decode (G2 t2 recommendation: spend gates on
  repeat-measurement credibility, not new decode candidates).
- The only open host-side question is the extend/prefill plan path: i1 removed ~5
  blocking D2H reads per extend batch (exact host mirrors, fallbacks kept). Prefill
  Heavy moved only 1.9% under the decode-only fix — consistent with prefill stalls —
  but magnitude is unknown until gated (0 if extend batches never bubble the syncs).

## Trial-2 work items
1. **Primary: request the FULL trial gate on b196fd4** (i1's request, unexecuted).
   Decision tree per i1 next_proposal:
   - >= lineage band (~1.09) and 10/10 exact → repeat b196fd4 in 2 more distinct
     accepted slots (candidate×3 in G3 namespace per canonical F4), then promotion
     gate request.
   - ~1.0 within noise of 1.0920/1.0984 band → prefill-syncs-bounded NEGATIVE
     finding; revert promotion track to 2c854e0 and spend remaining slots on lineage
     repeats.
   - Exact-token failure → revert candidate to 2c854e0; the mirror proof has a hole;
     record the failing point.
2. **New work (bounded sweep, this session):** i1's audit covered the flashinfer
   updater family only. Sweep the other per-step engine code (memory pools incl. the
   SWA hybrid pool, model_runner, forward_batch_info, scheduler batch prep) for
   remaining blocking `.item()/.cpu()/.tolist()` on the extend/decode hot paths.
   - If a hot site is found: patch in the same exact-mirror recipe (host list
     equality by construction), commit on top of b196fd4, include in gate request.
   - If none: record the engine-wide negative (sync family closed) — valid outcome.

## Constraints honored
- No decode candidates (roofline-closed), no precision changes (BF16 frozen; F2/F4
  budgets NOT_CLOSED_QUALITY_BUDGET), no graph-coverage/triton-attention re-gates
  (measured 1.0056 / 0.9733 — binding negatives).
- No measurement claims without controller gates; static verification only here.

## Verification plan
- Construction-equality proofs for any new mirror (same host list feeding device
  tensor and mirror), `python -m py_compile`, targeted greps to re-count blocking
  reads on hot paths.
- Gate arbiter: 10/10 exact-token on a2, full trial gate vs campaign baseline.

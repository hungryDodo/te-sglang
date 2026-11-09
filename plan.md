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

## Sweep result (trial-2 session, static; extends i1's backend audit engine-wide)
No new hot-path blocking read found. Every remaining `.item()/.tolist()/.cpu()` in
per-step engine code is on a cold path for this workload:
- memory_pool.py:362 `.item()` — guarded `scale.device.type=="cpu"` (host tensor, no sync).
- memory_pool.py:2397 `data_strides[0].item()` — `_init_kv_copy_and_warmup`, one-time init.
- memory_pool.py:1177-1186 — assert-on-failure inside cross-node KV transfer; mamba
  ping-pong (1817) mamba-only; `_to_cpu_int_list` (3200, callers 3273-3274) serves the
  GGUF dequant-on-the-fly workspace (falsified route, F10; BF16 gate never enters).
- model_runner.py:2489 `timeout.item()` — elastic-EP scale path (MoE elastic), unused.
- schedule_batch.py:670 `input_ids.flatten().tolist()` — `build_padded_input_ids`,
  returns None unless mm_items present (multimodal-only, text gate never enters).
- managers/prefill_delayer.py:204-271 — cluster of `.min/.max/.sum().item()`, but the
  delayer is opt-in via `get_schedule().enable_prefill_delayer` (scheduler.py:1450);
  cold unless the frozen contract enables it (nothing in branch learning indicates so;
  flagged conditional-cold, not proven cold).
- managers/mm_utils.py — all multimodal. sampler.py, speculative/ — no hits.
- flashinfer_backend.py — closed by i1's file-wide audit + b196fd4 (5→0 on the SWA
  standard-scheduler extend hot path).

Reasoned closure (no new gate needed): torch.compile-for-eager-decode is closed by the
existing G2 t2 in-situ C256 trace — GPU busy 99.2% means per-op CPU dispatch is already
fully overlapped; compiled eager kernels cannot buy wall-clock when the GPU is
saturated. It could only help if the trace showed idle gaps; it does not.

## Verification plan
- Construction-equality proofs for any new mirror (same host list feeding device
  tensor and mirror), `python -m py_compile`, targeted greps to re-count blocking
  reads on hot paths.
- Gate arbiter: 10/10 exact-token on a2, full trial gate vs campaign baseline.

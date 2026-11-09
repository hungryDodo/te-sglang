# G3 trial 2 (glm-a) — SWA extend-path sync elimination, round 2

Campaign continuous-20261109-01, Iteration 2. Base: 699c332 (= 83e7b75 fp8-KV remap
+ SWA decode hunk + e063d72 SWA extend `paged_kernel_lens_sum` host mirror; fast
gate 1.1361 GATE_VALID/PASS on a1). Direction per brief: revisit SWA/FP8+SWA
history, prefer exact runtime alternatives. No GPU/SSH here; controller gates.

## Bottleneck (from trial-1 static audit + this trial's code reading)

The fp8-KV+SWA lineage still drains the CUDA stream on every SWA-bearing extend
batch in `flashinfer_backend.py` `FlashInferIndicesUpdaterPrefill`:

- **B** `_build_swa_prefix_custom_mask` (l.2136-2138): 3× `.detach().cpu().tolist()`
  per call, **plus one `bool(block.all())` D2H per lane** inside the loop — the
  per-lane bool is a sync glm-b's concurrent input-mirror patch (shared sync
  evidence) does NOT remove. It runs even when the mask ends up None.
- **C** prefill `call_begin_forward` (l.2326): `kv_last_index = kv_indptr[-1]`
  device scalar read per SWA extend batch — the exact pattern 83e7b75 removed on
  decode (+8-9 ms/step TPOT there).
- **D** l.2068-2075: non-ragged wrapper-0 `paged_kernel_lens.sum().item()` fires
  for every real extend (prefix_lens is passed ⇒ `prefix_is_full_seq` False); the
  existing clamp mirror only covers the spec-less prefix==seq case.

## Hypothesis

H2-exact (this trial's code): replacing B/C/D with host-integer mirrors is
compute-exact by construction (metadata-only; same host lists that H2D-built the
device tensors — forward_batch_info.py:1230-1241 — so lane-for-lane equal), same
class as 83e7b75 (C1 vs parent agree=1.0, meanDlogprob=0.0) and e063d72.
Removes 4+ blocking D2H reads per SWA extend batch. Expected effect concentrated
in standard/medium/prefill-heavy; likely small on aggregate (extend batches are a
minority of steps) — a within-noise result is a valid outcome and will be
recorded. No EXACT claim is made without the C1 probe evidence.

H1 (statistics, primary ask unchanged from trial 1): F4 requires candidate×3 for
the fp8-KV+SWA lineage in G3's own namespace; fast-gate 1.1361 screens the head
but is not candidate evidence. fp8-KV stays COMPUTE_DIFFERS /
NOT_CLOSED_QUALITY_BUDGET (F2); no promotion is requested.

## Exactness arguments (static, to be confirmed by gate/probe)

- **D**: device `min(seq, window + seq - prefix)`; host `min(s, w + (s - p))` over
  mirrors — identical arithmetic. When `prefix_is_full_seq` (p=s) both give
  `min(s, w)`, so the new mirror subsumes the old clamp branch.
- **B inputs**: `prefix_lens_cpu = extend_prefix_lens_cpu` (same host list);
  `extend_lens = seq_lens_cpu - prefix_cpu`; `kv_start = max(0, p - window)`
  = `clamp(prefix_lens - window, min=0)` = `effective_start`.
- **B need_mask**: `not block.all()` ⟺ ∃(q,k): `k_abs < q_abs - window` ⟺
  `kv_start < prefix_len + extend_len - 1 - window` (k_abs min at k=0 is
  kv_start; q_abs max at q=e-1 is p+e-1; extremes independently attainable).
  Replaces the per-lane device bool with host arithmetic — provably identical,
  and the returned mask is bit-identical (same ints, same arange blocks).
  Note: I first conjectured the mask was always all-ones (dead code) — WRONG:
  for extend_len ≥ 2 and prefix ≥ window the last extend queries do exclude
  early prefix keys (e.g. w=4, p=10, e=3: query 12 sees keys 8,9 only). The
  mask is live; only the syncs are removed.
- **C**: in the non-spec branch `kv_indptr[1:bs+1] = cumsum(paged_kernel_lens)`
  ⇒ `kv_indptr[-1] == paged_kernel_lens_sum` (the host int already computed);
  `update_sliding_window` rejects `custom_kv_indices` (l.2013) and the spec
  branch rebuilds kv_indptr in `generate_attn_arg_prefill` — hence guard
  `spec_info is None and custom_kv_indices is None`, device read kept otherwise.
  Same invariant glm-b proved for their :2347; independent implementation here.
- Guards: `host_lens_mirror` requires extend_prefix_lens_cpu + seq_lens_cpu
  present with `len == len(prefix_lens) == seq_lens_cpu.numel()`; gpu_only
  batches (device tensors handed in, *_cpu unset) and spec-verify fall back to
  the device reads. Site A (e063d72) guard tightened with the same length check.

## Overlap disclosure

glm-b's iteration-1 proposal (shared sync evidence) patches equivalent sites
A/B/C on the a2 aa45af5-lineage (BF16-KV) as b196fd4, gates NOT_REQUESTED at
their session end. This trial implements B+C+D on the a1 fp8-KV lineage. F5
shows cross-node non-transfer at this magnitude (a2 standalone SWA ×1.0543 vs
a1 marginal +0.19% over fp8-KV), so these are distinct measurements, but the
controller should attribute with both in view. The B transform here is strictly
stronger than an input mirror (kills the per-lane `bool(block.all())` syncs).

## Worklog

- Read updater family on 699c332; confirmed sites B (2136-2138 + per-lane bool),
  C (2326), D (2075) unguarded; decode side already mirrored (83e7b75).
- Implemented B/C/D + host_lens_mirror block; py_compile clean; committed.
- Negative result kept: `_build_swa_prefix_custom_mask` is NOT dead code (proof
  attempt failed as above); `update_cross_attention` l.2195 device read is
  multimodal-only — left alone.
- Unresolved: all perf/quality claims await controller gates; no GPU run here.

## Requests (recorded in iteration-proposal.json)

1. Fast gate on the new head (screen B/C/D; expect neutral-to-small-positive,
   standard/medium/prefill-heavy sensitive; within-noise is valid).
2. Primary: candidate×3 on the fp8-KV+SWA lineage head (new commit; delta vs
   83e7b75 = exact-metadata hunks A/B/C/D only) on 3 distinct accepted slots,
   min-of-3 with per-slot values — F4 namespace closure.
3. Optional: C1 deterministic probe new head vs 83e7b75 (expect agree=1.0,
   meanDlogprob=0.0) to certify the metadata hunks EXACT; no claim without it.
4. NOT requested: promotion (fp8-KV quality budget NOT_CLOSED_QUALITY_BUDGET);
   if the operator later promotes, first re-run the E2 probe with 128-512 token
   windows per the registered-contract suggestion.

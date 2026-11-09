# G3 trial 4 — deepseek-b (node a4): host-mirror family completed + CPU oracle

Date 2026-11-09. Campaign continuous-20261109-01, Iteration 4. No GPU/SSH here;
the controller runs every official measurement. Nothing in this file is a
measurement.

## Scope and inherited state

- **F3 (`f263262`, GGUF MMVQ thresholds) remains UNGATED.** It is structurally
  un-closable by the frozen formal gate: the gate loads BF16 safetensors and
  every gate point runs at batch >= 32 where both thresholds select MMQ
  (controller applicability-001; G2 t2 close-out). The G3 t2 GGUF dispatch
  experiment (`seis/trials/deepseek-b/g3-t2/GGUF_DISPATCH_EXPERIMENT.md`,
  tracer `4e1644a`) is still **NOT_TESTED**; no new F3 evidence was possible
  this session. No BF16 number is read as F3 signal.
- **t3 candidate `33e5ac4`** (decode `FlashInferIndicesUpdaterDecode`: host int
  `paged_kernel_lens_sum` instead of the 0-dim CUDA scalar `kv_indptr[-1]` as a
  slice bound in the SWA full->swa translate block) is the inherited engine
  state. **Its fast-gate outcome is not visible to this session**; it must be
  treated as unmeasured here, not as a gain and not as a negative.

## Work done this trial (CPU-executed, not a measurement)

### 1. Bounded oracle on the host-mirror index invariant — NEW, executed
`seis/trials/deepseek-b/g3-t4/verify_index_invariants.py` (stdlib only, no
torch/GPU) parses the real source with `ast` and replays the integer arithmetic
with int32 wraparound. Raw output: `g3-t4/index-invariants.json`.

- Q1 decode: 27/27 cases (bs 16/32/64/128/256 x seq 512..8192, window 511, plus
  ragged mixes) have `host(paged_kernel_lens_sum) == cumsum(paged_kernel_lens)[bs]`.
  The t3 patch is index-identical where executed; spec-decode falls back.
- Q2 prefill: the plain-extend branch is equal in all cases, **but the
  dq/`custom_kv_indices` branch diverges in 5/5 probes** because that branch
  cumsums `dq_paged_kernel_lens` (page-padded) instead of `paged_kernel_lens`.
  A blanket host mirror there would be wrong; the guard is required.
- Q3 static: the decode site (line 1825) is host-mirrored; the prefill site
  (line 2351) was device-sourced before this trial.
- Q3 liveness: `init_forward_metadata` runs **once per model forward**
  (`model_runner.py:1795`), not per layer. So each removed sync is once per
  decode step / once per extend — it cannot be multiplied by the 21 sliding
  layers.

### 2. Engine change — prefill counterpart, guarded
`FlashInferIndicesUpdaterPrefill.call_begin_forward` now uses
`paged_kernel_lens_sum` as the translate slice bound **only** when
`spec_info is None and custom_kv_indices is None and the sum is an int`;
otherwise it keeps `kv_indptr[-1]`. Integer identity => bit-identical indices =>
EXACT by construction in the mirrored branch; the dq/spec paths are unchanged.

## Honest magnitude assessment (why no gain is claimed)

- The removed sync is one stream drain per decode step / per extend.
- `ec55606` (a larger per-step D2H removal in the same function family) measured
  **1.0018 KEPT_NEUTRAL on a4**; sibling a2 gains did not transfer.
- PrefillHeavy C16 has ~16 extends per point against ~4 s TTFT: the prefill site
  is near-provably <=0.1% even if live.
- Expected outcome of a fast screen is **~1.0**; the point of requesting it is
  to settle whether the SWA translate block is live at the gate configuration
  at all (`_swa_kv_pool is not None` and `translator.reads_are_translated is
  False`; a unified allocator would skip the block entirely and make this whole
  family dead code).

## Rejected this trial
- Sliding-window application in the no-prefix ragged extend path: still the
  largest lever (21/28 layers doing full 8192^2 on the 0.15-weight PrefillHeavy
  point) but a semantics change vs the reference wheel; >512-token exact-token
  parity is not established => NOT_CLOSED_QUALITY_BUDGET.
- Further `mmvq_safe` tuning: score-neutral at bs >= 32; kernel body frozen SASS.
- Chunk/workspace resize: boot-coupled, <=0.8% ceiling.
- Chasing PrefillHeavy through syncs: prefill is compute-bound (~110 TFLOP/s at
  the gate's 8192-token chunk), and the llama.cpp/attention-roofline notes hold.

## Deliverables
- Candidate commit: decode host mirror `33e5ac4` + prefill host mirror (this
  trial, guarded), one file.
- `seis/trials/deepseek-b/g3-t4/{verify_index_invariants.py,index-invariants.json}`.
- `iteration-proposal.json` with the fast-gate request and decision tree.

## NOT_TESTED
No torch/FlashInfer import, no GPU, no model weights. The patch was never
executed. Whether `reads_are_translated` is False in the gate's pool
configuration is not provable from this sandbox; whether decode points are
graph-replayed or eager is not established here.

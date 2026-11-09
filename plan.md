# G3 trial 3 — deepseek-b (node a4) plan and record

## Scope
Own **F3** (GGUF MMVQ threshold `f263262`, UNGATED), then explore full-engine
bottlenecks. Budget 45 turns / 20 min / USD 2. No GPU/SSH here; the controller
runs every official measurement. Nothing in this file is a measurement.

## Part 1 — F3 status carried in from G3 t1/t2 (no new GPU claim)
1. The frozen formal gate loads **BF16 safetensors** rev
   `e6406af83988f26e87ea92455f31db3f6476131b` and never passes a GGUF file or
   `--quantization gguf`. A BF16 score cannot execute the cells `f263262`
   changes (applicability-001). F3 therefore **cannot be closed by the formal
   gate**, only by the separately identified GGUF experiment.
2. Every frozen gate point runs at batch >= 32 while `f263262` only changes
   selection at bs <= 2 (Q8_0/Q4_K) / bs <= 6 (Q6_K). On the frozen workload
   the patch is **score-neutral by construction** (structural, not noise).
3. G3 t2 already shipped the observation-only, default-OFF dispatch tracer in
   `gguf.py` (commit `4e1644a`) and the bounded protocol
   (`seis/trials/deepseek-b/g3-t2/GGUF_DISPATCH_EXPERIMENT.md`). It has **not
   been executed by the controller** (NOT_TESTED). No GGUF evidence was
   produced this trial; do not read any BF16 delta as F3 signal.

## Part 2 — full-engine bottleneck: one exact candidate
Recon of `python/sglang/srt/layers/attention/flashinfer_backend.py` (decode
indices updater) found a redundant device->host read on the **eager decode**
path:

- `FlashInferIndicesUpdaterDecode.call_begin_forward` rebuilds
  `kv_indptr[1:bs+1] = cumsum(paged_kernel_lens)` and immediately afterwards
  slices `kv_indices[:kv_indptr[-1]]` in the SWA `full -> swa` translate block.
  A 0-dim CUDA tensor used as a slice bound goes through `__index__` -> `.item()`
  and drains the stream, once per eager decode step, for every sliding layer.
- The **equal host int is already a parameter of the same function**:
  `paged_kernel_lens_sum`. The invariant is true by construction inside this
  function (the cumsum is built from the very same tensor), so the device read
  is redundant whenever that branch ran. Spec-decode (`spec_info.kv_indptr`)
  and non-int callers keep the device read as fallback.

Patch: use the host value when available (`+11/-1`, one file). Integer identity
=> bit-identical indices => **EXACT by construction** (same class as the
host-mirror proofs already accepted on sibling branches: glm-a `e063d72`,
glm-b `b196fd4` site C).

**Uncertainty / dead-code risk (recorded honestly):** the block is only
entered when `self._swa_kv_pool is not None and not translator.reads_are_translated`;
and `begin_forward` is not called at CUDA-graph replay, so graph-captured
decode points cannot benefit. If the gate returns ~1.0 the path is dead on
a4's configuration and this family should be dropped there, not retried.

Rejected this trial (with reasons): sliding-window application in the
no-prefix ragged extend path (semantics change vs the reference wheel; no
>512-token parity harness), chunk/workspace resize (boot-coupled, <=0.8%
ceiling), further `mmvq_safe` tuning (score-neutral at bs >= 32; kernel body is
frozen SASS), further plain-`plan()` sync work (`ec55606` inert on a4).

## Deliverables
- Candidate commit on top of `b4fc419` (campaign base).
- `iteration-proposal.json` with evidence refs and uncertainty.
- Explicit NOT_TESTED list; no invented numbers.

## Reserve
Final quarter used for syntax checks, proposal JSON and final commit.

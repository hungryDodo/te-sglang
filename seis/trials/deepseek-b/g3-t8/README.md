# G3 t8 — deepseek-b: call-site provenance audit of the pending host-mirror candidate

**Status: CPU-executed analysis; NO engine change; no gate requested; F3 unchanged.**
Base: `ceb5cb8` (engine tree carries `f263262` + t3 `33e5ac4` + t4 `a306e40`).
Raw output: `callsite-mirror.json`. Oracle: `verify_callsite_mirror.py`
(stdlib, `python3 -I`, no torch/GPU/model/weights).

## Why this trial

The campaign direction is "evaluate F3 (`f263262`) before interpreting any gain,
then explore full-engine bottlenecks". Prior deepseek-b trials already executed:

- g3-t1/t5: `f263262` changes exactly 8 dispatch cells (Q8_0/Q4_K, rows<=5120,
  batches 3-6, MMVQ->MMQ), **0 changed cells at any batch >= 16**, so it cannot
  move any frozen gate point even if the gate loaded GGUF;
- controller applicability-001: the frozen Gate loads **BF16 safetensors**, no
  GGUF artifact; the GGUF lane is score-unreachable and F3 remains UNGATED
  research hygiene;
- g3-t7: the sliding-window prefill lever is acceptance-incompatible; the
  load-time-dequant ceiling is L2-resident and does not transfer.

What had **not** been audited is the provenance side of the pending t3/t4
host-mirror candidate (the only inherited engine change on top of `f263262`).
g3-t6 replayed integer equality over 34 sampled cases; it did not ask *where
each `paged_kernel_lens_sum` comes from at each caller*, which is what decides
whether the patch introduces a new correctness precondition.

## Executed method

`verify_callsite_mirror.py` AST-parses
`python/sglang/srt/layers/attention/flashinfer_backend.py` and:

1. line-anchors the two patched expressions and both guard forms (fails loudly
   if the source moves);
2. extracts **every** assignment to `paged_kernel_lens_sum*` in the FlashInfer
   updater classes together with the nearest `paged_kernel_lens*` tensor
   assignment, and classifies the sum's provenance mechanically
   (`DEVICE_SAME_TENSOR`, `HOST_SEQ_LENS_CPU`, `HOST_PREFIX_LIST`,
   `CALLER_SUM`, `DEVICE_OTHER_TENSOR`);
3. **executes** the real guard predicates (`ast.unparse` -> `compile` -> `eval`)
   over the configuration space: `spec_info` in {None, with `kv_indptr`,
   without `kv_indptr`}, `custom_kv_indices` in {None, tensor}, sum type in
   {int, float, tensor-sentinel};
4. replays host-vs-device last-cumsum equality per provenance class, including
   three negative controls (stale CPU mirror in decode, stale CPU mirror in
   prefill, mismatched prefix list);
5. checks the surrounding unpatched code for the same host-value dependency.

## Executed results

| quantity | value |
|---|---|
| `paged_kernel_lens_sum` assignment sites audited | **14** |
| provenance: device same-tensor / device other / host seq-lens-cpu / host prefix-list / caller sum | 5 / 1 / 2 / 2 / 4 |
| sites feeding the host mirror | **4** (decode `update_sliding_window:1694`; prefill `update_single_wrapper:2003`, `update_sliding_window:2096`, `:2106`) |
| host-mirror guard rows executed | 9 decode + 12 prefill |
| guard admits a non-`int` sum | **0** |
| integer divergences outside controls | **0** |
| sensitivity controls flagged | **3 / 3** |
| max D2H removed | **1 per eager decode/extend step** |
| `global_override_indptr_cpu = cumsum(seq_lens_cpu)` already in same function | **true** |
| `torch.empty(paged_kernel_lens_sum + 256)` already sizes the buffer | **true** |

### The decision-relevant fact

The patch uses the host int as the translate-slice bound. Its precondition is
`host int == device cumsum last element for the same logical tensor`. At the two
`HOST_SEQ_LENS_CPU` sites that is `seq_lens_cpu == seq_lens` (and the clamped
variant); at the two `HOST_PREFIX_LIST` sites it is
`extend_prefix_lens_cpu == prefix_lens.tolist()`.

**That precondition is not new.** Unpatched code in the same functions already
treats the host value as the authority over the same device tensor: decode
builds `global_override_indptr_cpu` from `torch.cumsum(seq_lens_cpu)` and
prefill sizes the `kv_indices` buffer as `paged_kernel_lens_sum + 256` before
`pack_read_stream`. The guards exclude exactly the paths where `kv_indptr` was
not rebuilt from `paged_kernel_lens` (`spec_info` supplying `kv_indptr`,
`custom_kv_indices`/dq branch), and reject non-`int` sums.

Verdict: the pending t3/t4 candidate is **exact under an already-required
invariant**, saves at most one D2H per eager step, and is **inert** when
`translator.reads_are_translated` (unified SWA pool) or `seq_lens_cpu is None`.
Sibling sync-removal evidence on a4 is 1.0018 KEPT_NEUTRAL with decode points
inside +/-2% run variance, so the expected aggregate gain is ~0. It is not a
reason to spend a gate slot.

## F3 status (unchanged)

`f263262` remains **UNGATED** and **score-unreachable** under the frozen BF16
Gate. The only instrument that could move it out of UNGATED is the separately
identified GGUF dispatch experiment pre-registered in g3-t2 with the g3-t5
tracer hardening; its execution prerequisites (matching trusted GGUF artifacts /
controller-supplied evidence) are still unmet. This trial does not execute it.

## Limits

Static + integer oracle, not a runtime measurement; torch semantics are
modelled by `clamp`/`minimum`/`sum` over Python ints. The frozen launch flags
(unified pool? `seq_lens_cpu` present?) are controller-side, so reachability
under the actual gate is not re-established here. No precision, quality,
numeric or perf claim. No evaluator/corpus/seed/reference change.

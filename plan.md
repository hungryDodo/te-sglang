# G3 trial 13 — deepseek-b — plan

Campaign `continuous-20261109-01`, Iteration 13. Worker deepseek-b (node a4).
Bounds: 45 turns / 20 min / USD 5 nominal. No GPU, no SSH, no evaluator.

## 0. Hard constraints (unchanged)

- No GPU/SSH measurement; official gates are the controller's. No claim of
  EXACT / Candidate / Promotion / best_verified from this sandbox.
- No evaluator/corpus/seed/reference/protocol change. Frozen launch flags are
  what the controller runs; engine changes must be defensible on their own.
- F3 (`f263262`, MMVQ thresholds) stays **UNGATED**. The frozen BF16 Fast screen
  loads BF16 safetensors (`applicability-001`) and cannot exercise GGUF cells; a
  BF16 score is never evidence for F3. Any F3 statement this trial is limited to
  a derived audit of the controller's real GGUF dispatch trace (C22).
- `request_fast_gate=true` only for an applicable BF16 engine candidate with a
  falsifiable performance mechanism and an exact-token risk argument.

## 1. Direction

Brief: evaluate `f263262` before interpreting any gain, then explore full-engine
bottlenecks. Prior branch state: t7 established GGUF is score-unreachable under
the frozen BF16 Fast contract; t9/t12 established that the frozen prefill
admission budget is the whole-batch `rem_chunk_tokens` = `chunked_prefill_size`
(8192), so the 15%-weight PrefillHeavy C16 point admits exactly **1 x 8192**
request per prefill forward — `max_prefill_tokens` (16384) is not used as the
batch budget even though it is the source's prefill-token budget. t12 named the
next step: verify buffer/workspace sizing, then decouple the two budgets.

## 2. Question this trial (new, CPU-falsifiable + one engine candidate)

**H1 (prefill width):** `PrefillAdder` initialises its whole-batch token budget
from `chunked_prefill_size` and the per-request chunk cap is the *same*
variable, so the frozen launch runs 64 x 8192-token prefill forwards for
PrefillHeavy instead of 32. Decoupling batch budget (`max_prefill_tokens`,
16384) from the per-request chunk cap (`chunked_prefill_size`, 8192) admits
2 x 8192 per forward with identical per-request attention/GEMM math (only the
GEMM batch dimension changes). Expected effect concentrated in PrefillHeavy C16
(15% weight); other points are controls.

**Memory-safety precondition (kill criterion):** prefill activation/workspace
buffers and the FlashInfer autotune dummy must be sized from
`max_prefill_tokens` (or `max_prefill_buffer_tokens()`), not from the 8192
chunk. If sizing follows the chunk, the patch is rejected as boot-OOM risk.

**H2 (F3 derived audit, no new GPU claim):** using the controller's real GGUF
dispatch trace (C22 / `gguf-dispatch.json`), classify every observed
(weight_type, rows_band, batch) cell under the legacy rows-only threshold and
under `_mmvq_safe_batch`; count cells where `f263262` flips selection and say
exactly which real call counts hit them. This is an audit of supplied data, not
a measurement; F3 remains UNGATED and no speed is claimed.

## 3. Deliverables

- `seis/trials/deepseek-b/g3-t13/` — executed CPU oracles + raw JSON:
  (a) prefill-budget patch reachability/step-count oracle over real source;
  (b) buffer-sizing provenance check for `max_prefill_tokens`;
  (c) F3 dispatch-cell audit from `gguf-dispatch.json`.
- Engine patch (only if H1 survives the oracle and the sizing check) in
  `python/sglang/srt/managers/schedule_policy.py`.
- `iteration-proposal.json`, updated `plan.md`, git commit.

## 4. Kill criteria

- Any anchor drift / contradicted premise: record the negative with raw output.
- If prefill buffers are sized by the chunk rather than `max_prefill_tokens`,
  or the scheduler rejects 2 x 8192 elsewhere (max_prefill_bs, memory budget,
  delayer), do not patch; report the negative.
- Exact-token parity is an empirical gate question (batched GEMM reduction
  order); a Fast PASS is a screen only and does not establish numeric equality.
- Never call a BF16 Fast score evidence for F3/GGUF.

## 5. Outcome (executed, trial 13)

`python3 -I seis/trials/deepseek-b/g3-t13/audit_f3_dispatch_cells.py $PWD .../f3-dispatch-audit.json`
-> exit 0, verdict `DISPATCH_CONTRACT_HELD_ON_TRACE`: 28,608 recorded calls,
**0** cells where the observed kernel disagrees with the patched policy, and
**15,952 calls (55.8%)** whose selection `f263262` flips versus its parent
(`f263262^`), all `MMVQ -> MMQ`:
- `Q8_0`, rows<=5120, batch 4: **15,868** calls (legacy safe=6 -> patched safe=2)
- `Q8_0`, rows<=5120, batch 6: **84** calls (canonical C22: startup prefill, not C4 decode)
- `Q8_0`, rows>5120: **no flip** (legacy 2 == patched 2)
- `Q4_K` changed cells: **not exercised** by this Q8_0 trace.
The deciding source (`_MMVQ_SAFE_MEASURED`, `_mmvq_safe_batch`, the legacy
`mmvq_safe` If-block from `f263262^`) was AST-extracted and executed; the trace
is the controller's immutable C22 packet, replayed read-only.

**Interpretation (not overstated):** F3's changed region is *live and
contract-conforming* in an ordinary GGUF Q8_0 run, so the patch is not dead
code. But the packet has no timing comparison (trace writes perturb wall time)
and no numeric equivalence; it is not the frozen workload (frozen gate is BF16
safetensors, `applicability-001`). **F3 remains UNGATED**; this audit neither
promotes it nor attributes any gain.

**Prefill-width lever (t12 follow-up) — negative:** `max_prefill_buffer_tokens()`
(`runtime_context.py:1778`, `arg_groups/overrides.py:1865`) derives the prefill
buffer/autotune ceiling from `chunked_prefill_size` (8192) whenever PP dynamic
chunking is off (frozen launch), NOT from `max_prefill_tokens` (16384). Admitting
2 x 8192 tokens per forward would exceed the registered prefill ceiling that
sizes the FlashInfer extend autotune dummy and the pre-capture activation
reserve. The t12 lever is therefore rejected: it needs the frozen
`--chunked-prefill-size` 8192 to be reinterpreted and the boot memory profile
changed, neither verifiable here and both outside a semantics-preserving,
frozen-launch change. No engine patch was made.

**Gate request:** none for BF16. The only unmeasured BF16 delta in the tree
(t4 prefill host-mirror) is bounded sub-noise by t10's own executed arithmetic
(<=~0.1% at PrefillHeavy, below Fast resolution), so a BF16 Fast slot would be
uninformative. The proposed experiment is instead a bounded GGUF A/B at the C4
batch regime where `f263262` actually acts (see `iteration-proposal.json`).

Raw: `seis/trials/deepseek-b/g3-t13/f3-dispatch-audit.json`,
`legacy_gguf_f263262_parent.py.txt`.

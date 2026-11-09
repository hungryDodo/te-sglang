# GGUF dispatch + correctness experiment (controller-run, separate from the frozen BF16 gate)

Purpose: close canonical F3 (`f263262`, UNGATED) with evidence instead of
inference. The controller's `applicability-001` finding is binding: the frozen
formal Gate loads BF16 safetensors and never imports `gguf.py`, so no BF16 gate
score can exercise or exonerate `f263262`. This probe is a **separately
identified GGUF experiment**; its artifacts and any timings stay out of official
scoring.

Candidate commit: `4e1644a` (parent `f263262`, additions: an observation-only,
default-off dispatch tracer in `gguf.py`, this protocol, and the GPU-free
expectation table). Default runtime behaviour is byte-identical to `f263262`:
`SGLANG_GGUF_DISPATCH_TRACE` unset means `_DISPATCH_TRACE_FILE is None` and every
new branch is a single truth test before returning the same tensor.

## Pre-registered facts (GPU-free, `verify_dispatch_expectations.py`)

- Changed cells: exactly 8 — Q8_0 and Q4_K, rows<=5120, decode batch 3..6
  (legacy MMVQ -> shipped MMQ). None at gate batches 16/32/64/128/256.
- Unchanged controls: every other type/band/batch, including Q8_0 rows>5120,
  Q6_K (measured 6/2 equals legacy), Q8_1 (2/6 preserved), I-Matrix (8/16).
- Artifact `dispatch-expectations.json` holds the exact expected
  `(type, rows_band, batch) -> (kernel, mmvq_safe)` table the live trace must
  reproduce.

## Arms

- **A (parent):** `f263262^` = `ec55606`. Legacy rows-only policy.
- **B (candidate):** `4e1644a`. Measured policy + tracer.

Both run the same GGUF Q8_0 artifact used by gate `g2-sglang-q8` (the only GGUF
artifact with prior gate data), same launch flags, plus `--disable-cuda-graph`.
CUDA graphs must be off for phase 1: during decode the Python dispatch runs once
at capture and is not re-executed on replay, so graph mode would report padded
capture-time batch sizes, not the per-batch decision.

## Phase 1 — dispatch evidence (arm B only)

For each batch size `bs` in `[1,2,3,4,5,6,7,8,16,64]`: submit exactly `bs`
identical greedy requests (temperature 0, same `max_new_tokens` so they stay in
lockstep), let them decode at least 8 steps, capture token ids, release.
Launch B with:

```
SGLANG_GGUF_DISPATCH_TRACE=/artifacts/g3-t2/trace-bs<bs>.json
```

The tracer dumps every 1024 calls and on normal exit (`atexit`). Pass criteria:

1. At `bs` 3..6, Q8_0 and Q4_K rows<=5120 cells (whichever rows bands the
   artifact actually has) appear as `kernel=mmq, mmvq_safe=2`. This is the
   central assertion: it proves the changed cells are real and that the patch's
   table reached the kernel dispatch, not just the source text.
2. At `bs` 1..2 the same cells are `kernel=mmvq, mmvq_safe=2`.
3. Every Q8_0 control cell (rows>5120 at all probed batches; rows<=5120 at
   1,2,7,8,16,64) matches `expected_dispatch` exactly. Missing Q8_0 control =
   INCONCLUSIVE, not PASS.
4. Cells for quant types absent from the artifact are N/A. If the artifact has
   **no** Q8_0/Q4_K weight with rows<=5120, then `f263262` is a no-op for it:
   record F3 as "inert on the available artifact" and use a synthetic Q8_0
   matrix with rows<=5120 for the dispatch check instead of a model run.

## Phase 2 — greedy correctness A/B (arms A and B)

Same `bs` ladder as phase 1, same prompts, greedy, 32 new tokens. Compare
per-request token ids A vs B.

- `bs` in 1,2,7,8,16,64: must match token-for-token. Any divergence is a FAIL
  (those cells are kernel-identical between arms; a difference would mean the
  tracer or the policy leaked into an unchanged path).
- `bs` in 3..6: divergence is *allowed but must be counted and reported* per
  request/token position. MMVQ->MMQ is an accumulation-order change
  (`cos ~0.99998` per the g2-t2 microbench), so near-tie flips are possible.
  This is the COMPUTE_DIFFERS boundary of the fix; do not report EXACT if any
  divergence appears. If zero divergence across the probe, report
  "greedy-exact on this probe", not universal EXACT.
- MMQ is already the kernel used at every batch >=7, so a small-batch flip to
  MMQ does not introduce a precision class the engine does not already ship.

## Phase 3 — optional engine timing at the changed cells (arm B vs A)

Only after phases 1-2 pass. Fixed-length decode at `bs` 3,4,5,6 (e.g. 128 steps,
median of 3 runs), same artifact, graphs off. This validates the g2-t1
microbench at engine level. It is **not** an official score: exclude it from
scoring and do not mix it with the frozen BF16 gate. Expected direction:
B faster (MMQ beat MMVQ by 17-24% at batch 4 in the microbench); magnitude at
engine level is unknown and may be diluted by non-matmul work.

## Interpretation rules (write these into the result)

- Dispatch matches + greedy token ids match at 3..6: `f263262` is verified
  selection-correct and greedy-exact on the probe. F3 closes as a
  hygiene/correctness fix with **zero expected formal-score gain** (no gate
  batch touches the changed cells). Not a speedup.
- Dispatch matches + token divergence at 3..6: fix is verified as a kernel
  selection change but is COMPUTE_DIFFERS at those cells; quality budget stays
  NOT_CLOSED_QUALITY_BUDGET unless a precision contract is registered.
- Dispatch mismatch: `f263262` has a real bug; repair before any gate.
- No changed cells in the artifact: F3 remains UNGATED for engine purposes;
  use the synthetic-matrix check, and record that the available GGUF artifact
  cannot exercise the fix.

## Artifacts to return

`trace-bs*.json` (B), token-id JSONs (A and B), launch command + flags, artifact
path/digest, commit hashes, and the phase-1/2 pass tables. Keep them in a
directory clearly named outside the official gate namespace.

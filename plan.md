# deepseek-b / G3 trial 2 plan

Campaign `continuous-20261109-01`, Iteration 2. Worker deepseek-b. No GPU/SSH in
this sandbox: every performance claim below is either prior verified evidence or
explicitly UNVERIFIED. The controller owns all official gates.

## 0. Binding context

- Canonical F3: GGUF MMVQ threshold fix `f263262` is **UNGATED**. It may not be
  treated as a valid candidate, and no gain may be attributed to it, until a
  gate verifies it.
- Controller finding `applicability-001` (authoritative): the frozen formal Gate
  loads **BF16 safetensors** (`Spark-X2.6-1.7B`, rev `e6406af8...`), never a
  GGUF file and never `--quantization gguf`. `f263262` changes
  `fused_mul_mat_gguf` kernel **selection**, so the BF16 screen executes none of
  its changed cells. A BF16 score cannot close F3, and importing `gguf.py` is
  not dispatch evidence.
- Consequence: first deliverable is a **separately identified GGUF experiment**
  (dispatch + correctness) under the existing GPU lock, kept outside the BF16
  formal Gate. Only after that may any GGUF number be interpreted.
- Second deliverable: explore full-engine bottlenecks with an exact
  (ALGORITHM_PRESERVING) BF16-safe change, since the formal objective is
  aggregate output tokens/s on the frozen BF16 workload.

## 1. Profile (from prior verified evidence only)

- g2-t1 microbench: MMVQ (`mul_mat_vec_q`, integer-dot, 0 tensor-core ops) is
  faster than MMQ (`mul_mat_q4_K`, also integer-dot) only at small batch.
  Measured **last MMVQ win** batch = 2 for Q8_0 and Q4_K, 6 for Q6_K (when
  rows<=5120 band; different for rows>5120). `dequant+bf16 GEMM` at runtime is
  0.18-0.25 ms flat, never competitive on that path.
- g2-t2 GPU-free replay: the shipped rows-only heuristic (`mmvq_safe` 8/16 for
  K-quants, 2/6 for Q8_0) was over-permissive on 3/6 measured shapes, max
  +24.1% kernel time at the hardcode boundary. `f263262` replaced it with
  measured batch cutoffs (Q8_0 2/2, Q4_K 2/2, Q6_K 6/2) and is expected to be
  exact below the cutoff and MMQ above it.
- **Why no gate can show a gain today**: every gate point runs batch >= 32
  (Short c32/c128, Medium c32/c128, Standard c32/c64, DecodeHeavy c64/c128,
  DecodeCeiling c256, PrefillHeavy c16 prefill). Old and new thresholds select
  the same MMQ kernel for all of them. DecodeHeavy c64 measured movement
  (+1.6-2.0%) is run-to-run variance, not the patch. This is structural, not a
  measurement failure.
- Residual unknown that makes F3 closeable: **no recorded runtime dispatch
  evidence** (which kernel actually ran per shape) and **no correctness check
  at the changed cells** (Q8_0/Q4_K bs 3..6, Q6_K bs 3..6 and 7..8). The g2-t2
  numbers are microbench replay and analytic accumulation-order argument
  (cos ~0.99998), not engine measurements.

## 2. Bottleneck analysis

Two distinct surfaces, and they must not be conflated:

1. **GGUF small-batch dispatch (F3 surface).** The only cells `f263262` can
   change are bs<=6; the frozen workload never touches them. Closing F3 is
   therefore a *correctness/coverage* question, not a score question. It needs
   (a) runtime dispatch logging proving the new table selects the intended
   kernel per (quant, rows, batch), and (b) an exactness check at the changed
   cells vs the wheel. A speed gate at bs>=32 is score-neutral by construction
   and must not be used to claim or deny the fix.
2. **BF16 full-engine (formal objective surface).** Highest-weight untouched
   cost identified on a4: PrefillHeavy C16 (weight 0.15, ~59.3 tok/s, TTFT
   ~4047 ms) and TTFT-dominated Medium/Standard points. Established mechanism:
   in the **no-prefix ragged extend** path the 512-token sliding window is not
   applied, so 21/28 sliding layers do full 8192^2 attention at the gate's
   8192-token prefill. That is the largest identified exact-cost lever, but
   applying the window is a **semantics change**: the reference wheel has the
   same omission, so it flips greedy tokens when the window binds and
   `rsi_equivalence.py` (<=511-token prompts) cannot certify parity at 8192.
   It is COMPUTE_DIFFERS, not ALGORITHM_PRESERVING, and is blocked on a
   >512-token parity harness that cannot be built or run in this GPU-free
   20-minute session.

## 3. Candidate hypotheses (ranked)

- **H1 (primary, this trial): make `f263262` gate-closeable without touching
  its numerics.** Add an *observation-only*, env-gated dispatch recorder to
  `gguf.py` that records, per fused matmul call, `(matmul_type, rows, batch)`
  into a bounded in-process counter/log, plus a GPU-free self-check that
  reproduces `_mmvq_safe_batch()` selection against the measured crossover
  table. Exact same kernels, same thresholds: zero semantic delta. This turns
  the controller's requested GGUF experiment from "import gguf.py" into
  "record actual dispatched kernel" and yields a falsifiable pass/fail table
  for the changed cells and unaffected controls.
- **H2 (fallback, if H1 code is too invasive): static audit + GPU-free replay
  of the full `f263262` selection function** covering quant types the earlier
  replay did not cover (Q8_1, I-Matrix, rows<=5120 vs >5120 bands), plus a
  machine-checkable table for the requested controller experiment. No engine
  change; still closes the "is the table right for every shipped type" gap.
- **H3 (out of scope this trial, recorded for next):** >512-token exact-token
  parity harness, then `window_left` at plan time for sliding layers in the
  extend-no-prefix branch. Largest BF16 lever; cannot be honestly gated without
  the harness. Do not implement window semantics without parity evidence.
- **Rejected now:** any further `mmvq_safe` retune (score-neutral by
  construction at bs>=32); triton attention (0.9733 negative), graph coverage
  >64 (1.0056 neutral), Q8_0 GGUF serving as formal baseline (0.24x).

## 4. Work items and acceptance criteria

1. Inspect the shipped `f263262` tree; confirm the selection table and find any
   type/band where measured data is absent or mis-mapped.
2. Implement H1 observation-only dispatch recorder, guarded by an env var,
   default off, no allocation/timing change on the hot path beyond one dict
   increment when enabled. `py_compile` must pass.
3. Add a GPU-free verification script (runs on CPU, imports nothing CUDA) that
   enumerates (quant type, rows, batch) cells and prints expected MMVQ/MMQ per
   the shipped table, so the controller's experiment has a pre-registered
   expectation table to compare against.
4. Commit with repo-local identity; write `iteration-proposal.json` with
   `request_fast_gate: false` (a BF16 fast gate cannot see this change) and an
   explicit GGUF experiment request instead.
5. Reserve the last quarter of the session for the proposal, tests, and final
   commit.

## 5. Uncertainty / negative outcomes (recorded, not hidden)

- The dispatch recorder proves *selection*, not *performance*; the controller
  must still run the changed cells to claim any speedup, and the quality budget
  for any numeric delta (MMVQ->MMQ accumulation order) stays
  NOT_CLOSED_QUALITY_BUDGET unless a registered precision contract exists.
- If `f263262` selection matches the measured crossover everywhere and the
  runtime dispatch confirms it, the correct conclusion is still "F3 closes as a
  hygiene/correctness fix with zero expected formal-score gain", not a speedup.
- No claim of EXACT is made anywhere without numeric evidence from the
  controller's run. A BF16 gate score is not evidence about GGUF dispatch.

## 6. Provider identity

Backend identity remains **UNVERIFIED_BACKEND_VERSION** for this session; the
provider reports model names only, which do not prove which model executed.
This plan makes no identity claim.

## 7. Trial 2 results (this session, GPU-free)

### F3 evaluation (required first item)

- Re-read the shipped selection path at HEAD. Policy is `MMVQ iff batch <=
  _mmvq_safe_batch(type, rows)`; measured overrides are Q8_0 (2,2), Q4_K (2,2),
  Q6_K (6,2); Q8_1 and I-Matrix keep legacy fallbacks (2/6 and 8/16); all other
  types keep 2/6. No defect found in the mapping.
- Re-ran the GPU-free replay (`g3-t1/verify_mmvq_selection.py`) against the
  current tree: 8 changed cells, exactly Q8_0/Q4_K rows<=5120 at batch 3..6,
  zero changed cells at any gate batch (16/32/64/128/256) or batch >=7.
- Conclusion (not overstated): `f263262` is selection-correct as far as the
  measured table goes, and is **score-neutral by construction** on the frozen
  BF16 workload, which never imports `gguf.py` at all. It is **not** promoted,
  and no gain is attributed to it. What is missing is runtime dispatch evidence
  and token-level correctness at the changed cells.
- Delivered to close that gap: an observation-only, env-gated dispatch tracer
  in `gguf.py` (`SGLANG_GGUF_DISPATCH_TRACE`), default off, recording
  `(weight_type, rows_band, batch, kernel, mmvq_safe)` counts and dumping JSON
  every 1024 calls and at exit; a GPU-free pre-registration script that emits
  the exact expectation table the live trace must match; and a controller
  protocol (`GGUF_DISPATCH_EXPERIMENT.md`) with phases for dispatch, greedy
  token-id A/B vs the parent commit, and optional engine timing at the changed
  cells. The protocol requires `--disable-cuda-graph` for phase 1 because graph
  replay does not re-execute Python dispatch.

### Full-engine BF16 exploration (negative, precise)

- Confirmed at this commit: the `extend_no_prefix` ragged prefill branch
  (`flashinfer_backend.py:1460-1471`) calls `prefill_wrapper_ragged.forward(...)`
  without `window_left`, while the prefix-present branch (`:1473-1490`) passes
  `window_left=layer.sliding_window_size`. With `spark2_5.py:47-48,176-184`
  setting 511 for sliding layers, 21/28 layers therefore do full 8192^2
  attention on PrefillHeavy's 8192-token point. This is the largest identified
  exact-cost lever on the 0.15-weight point, and it is **semantics-blocked**:
  the reference wheel behaves the same, so applying the window flips greedy
  tokens when it binds, and `rsi_equivalence.py` (<=511-token prompts) cannot
  certify parity at 8192. It is COMPUTE_DIFFERS, not ALGORITHM_PRESERVING, and
  was not implemented.
- No other exact BF16 lever was found that is not already closed: decode at
  gate batches is bandwidth-bound at roster-proven roofline; eager plan syncs
  are inert on a4 (trial 1); graph coverage >64 and triton attention are
  measured negatives; raising the chunk cap is boot-coupled with no headroom
  (16 x 8192 = max-total-tokens) and changes reduction order.
- Therefore this trial makes no BF16 performance claim and requests no fast
  gate; the honest next step is the controller-run GGUF experiment plus a
  >512-token parity harness before any window change.

### Uncertainty

- The tracer proves selection, not speed or numerics. F3 stays UNGATED until
  the controller's experiment returns. If the GGUF artifact has no Q8_0/Q4_K
  weight with rows<=5120, the changed cells are absent and F3 is unexercised
  by that artifact (protocol says to fall back to a synthetic matrix).
- Token divergence at batches 3..6 is possible (MMVQ->MMQ accumulation order);
  no EXACT claim is made. Quality budget remains NOT_CLOSED_QUALITY_BUDGET.

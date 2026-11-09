# G3 trial 7 — deepseek-b — plan (provisional, updated as evidence lands)

Budget: 45 turns / 20 minutes / USD3 nominal. No GPU, no SSH, controller-only evaluation.
Objective: aggregate output tokens/s under the frozen workload contract. No measurement,
corpus, seed, reference or evaluator change. A source change is never its own acceptance.

## Required direction check: F3 (`f263262` MMVQ threshold) is UNGATED
- Frozen formal gate loads BF16 safetensors revision e6406af..., no GGUF file and no
  `--quantization gguf`. The patched `fused_mul_mat_gguf` selection cells are therefore not
  exercised by the formal screen; a BF16 Fast score cannot close F3.
- Step 1 (executed CPU oracle, not a re-read): extract the real dispatch predicate and the
  `_mmvq_safe_batch` table from the candidate tree with `ast`, then evaluate the *complete*
  dispatch cell space for (a) the frozen BF16 launch (does the GGUF module load at all?),
  (b) a hypothetical GGUF launch at every frozen point's GEMM batch size and the model's
  real linear-layer row bands. Falsifiable claim to test: **some frozen point has batch
  <= old threshold, so the patch could move a gate point.** If none does, F3 is score-inert
  by construction and GPU budget should not be spent on it (hygiene-only).
- Preserve raw output; report the exact affected batch region and its limits.

## Step 2: full-engine bottleneck selection (CPU-side, from frozen contract + traces)
- Read the frozen gate config and launch builder actually present in this tree; reconstruct
  each of the 10 points' prompt/concurrency shape.
- Reuse branch trace/measurement facts (a4 node numbers, trial-1/2 findings) rather than
  re-derive. Rank remaining levers by (weight x measured gap x semantics risk).
- Only propose an engine patch if it is semantics-preserving wrt the pinned reference and
  token-exactness is argued; precision changes stay NOT_CLOSED_QUALITY_BUDGET.

## Step 3 deliverables (reserve final quarter)
- `seis/trials/deepseek-b/g3-t7/` evidence dir with raw JSON/tables and a README of limits.
- `iteration-proposal.json` with bottleneck, hypothesis, patch, tests, learning,
  next_proposal, request_fast_gate (true ONLY for an applicable BF16 candidate).
- Local git commit with the evidence and any patch. No promotion / EXACT claims.

Kill criteria: if the executed oracle cannot be made to falsify anything and no applicable
BF16 candidate exists, record the precise negative and do NOT request a gate.

## Close-out (executed)

- Oracle executed: `seis/trials/deepseek-b/g3-t7/verify_lane_closure.py` ->
  `closure.json`. Model is 1.711B tied params, 21/28 sliding layers,
  51.40M params/layer. Real matrices (gate_up BF16 54.5 MB, down 27.3 MB,
  embed 545.3 MB) exceed the <25 MB L2 used by g2-t1's 8.95-11.85x ceiling ->
  that ceiling does not transfer; and the formal gate is BF16-only, so the
  GGUF lane is score-unreachable regardless.
- Sliding-window prefill ceiling: attention is 35.4% of prefill FLOPs at
  S=8192 (max 1.332x), and HEAD anchors (:1464-1472 vs :1474-1496) confirm the
  no-prefix branch omits `window_left`; changing it cannot satisfy
  `valid_exact_token_baseline` against the same full-attention reference.
- Verdict: no promotable deepseek-b lever established; no gate requested
  (`request_fast_gate=false`); engine tree left byte-identical.

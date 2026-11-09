# G3 trial-1 plan — worker deepseek-b (candidate `f263262`)

Campaign `continuous-20261109-01`, Iteration 1. Candidate repo: SGLang @
`codex/continuous-20261109-01/deepseek-b` (HEAD `f263262`, parent `ec55606`).

## Mandate and prior state

- Canonical `F3`: the GGUF MMVQ threshold fix `f263262` is delivered but
  **UNGATED**. This trial's first duty is to subject it to a controller gate;
  until that gate exists it must not be treated as a valid candidate.
- Canonical `F1`/`F2`: NGRAM self-speculation is BUG_CONFIRMED (greedy
  correctness not preserved) and fp8-KV is COMPUTE_DIFFERS with an undefined
  quality budget. Neither is admissible here.
- Branch learning (deepseek-b, authoritative for this worker): all frozen gate
  points decode at concurrency `>= 16` (PrefillHeavy c16 is the minimum) and
  prefill with thousands of extend tokens; the MMVQ/MMQ switch is a pure
  kernel-selection change in `gguf.py`; the remaining measured headroom is
  `bs <= 8`, which no gate point touches. `ec55606` (fast_decode_plan) is
  KEPT_NEUTRAL at 1.0018 and was shown inert on a4.
- Branch learning also records the largest identified BF16 lever: the
  no-prefix ragged prefill path does **not** apply the 512-token sliding
  window on 21/28 layers, but applying it is a *semantics* change vs the
  reference wheel and is blocked on a >512-token exact-token parity harness.

## Step 0 (mandated): evaluate `f263262` before interpreting any gain

This trial makes **no new engine change**. The candidate stays exactly
`f263262` so that any gate result is attributable to that commit alone; adding
a second patch now would confound F3 closure. Worktree additions are
`plan.md`, `iteration-proposal.json` and one GPU-free replay artifact under
`seis/trials/deepseek-b/g3-t1/`; none is on a runtime import path.

### Static verification (done, GPU-free)

`seis/trials/deepseek-b/g3-t1/verify_mmvq_selection.py` parses the shipped
policy out of `python/sglang/srt/layers/quantization/gguf.py` (no torch / no
SGLang import; `WeightType` is `gguf.GGMLQuantizationType`, an external
package, so the script replays only enum names the module actually
references). It compares legacy rows-only selection with `_mmvq_safe_batch`
over every referenced type x {rows<=5120, rows>5120} x batch 1..256. Result
(`mmvq-selection-replay.json`):

- 8 changed selection cells: Q8_0 and Q4_K, `rows <= 5120`, batch 3..6
  (MMVQ -> MMQ). Nothing else changes; Q6_K 6/2 and Q8_1/IMATRIX legacy
  thresholds are preserved.
- `changed_cells_at_batch_ge_16 = 0`; selection is identical to legacy from
  batch 7 upward, hence at every gate decode batch (16/32/64/128/256) and at
  every prefill extend-token batch (thousands of tokens).
- Live-path audit: `fused_mul_mat_gguf` is the only kernel-dispatch site that
  used the old inline `mmvq_safe` (no other `mmvq_safe` references remain in
  `python/sglang/srt`), and it is called from the GGUF linear method, the GGUF
  MoE path and the DeepSeek GGUF model paths. The patch is live for all GGUF
  matmuls.

Caveat carried forward: this is static selection replay, not a GPU timing. The
only timing evidence remains the G2-t1 microbench, which is L2-resident and is
an upper bound; G2-t2 already corrected its derived `mmvq_safe` column.

## Hypothesis (H-G3-1)

`f263262` is *selection-identical* to the legacy rows-only policy at every
frozen gate point. The only cells whose kernel choice changes are
`Q8_0`/`Q4_K` with `rows <= 5120` at **decode batch 3..6** (MMVQ -> MMQ);
every other `(type, rows band, batch)` cell — including all IMATRIX types and
all batch >= 7 cells — selects the same kernel under old and new code.

Prediction: a fast gate on the GGUF Q8_0 surface lands at score ~1.0
(neutral), with no attributable regression. Its scientific value is closing F3
(`UNGATED` -> gated-neutral, hygiene only) and confirming the delivered wheel
still boots/behaves, not a speedup.

Falsifier / interpretation rule: a gate point moving outside ~±2 % cannot be
attributed to this patch, because the patch cannot change kernel selection at
any gate batch. Such movement is run-to-run variance (documented ±2 % on
DecodeHeavy c64 in G2-t2), and any apparent *gain* must not be reported as an
MMVQ fix gain.

## Requested controller action

Fast gate on the candidate (`request_fast_gate: true`), GGUF Q8_0 surface,
to close F3. Because every gate point is selection-identical, any subset of
points with decode concurrency >= 16 is equally informative; if a reduced
protocol is available, the informative minimum is boot/load + two or three
points. Note the known pre-existing behaviour on this surface: in G2-t2 the
reference and candidate both time out on DecodeCeiling c256 (44 vs 42
timeouts), so that point is not attributable to this patch. No promotion is
requested from a single fast screen.

## Explicitly out of scope this trial (with reasons)

- Sliding-window prefill fix: semantics change; reference wheel is identical;
  no parity evidence for prompts > 511 tokens. Cannot be gated without a
  registered quality acceptance contract. Proposed as next step.
- Load-time dequant Q8_0/Q4_K -> BF16 GEMM: ALGORITHM_PRESERVING at best
  (cos ~0.99998 class), needs ~2x weight memory and a parity harness; not
  landable without evidence.
- Any precision/quantization change: NOT_CLOSED_QUALITY_BUDGET.
- Further eager-decode sync work: trial-1 showed it inert on a4.

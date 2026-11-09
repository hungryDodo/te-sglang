# G3 trial 8 — deepseek-b (node a4) plan and record

## Objective
Raise aggregate output tokens/s under the frozen workload contract. The trusted
controller owns the evaluator, measurements and decisions; this session produced
source analysis, a CPU-executable evidence artifact and a proposed test request.

## What the canonical record already fixed before this trial
- **F3:** GGUF MMVQ threshold fix `f263262` is delivered but **UNGATED**; it
  must be evaluated before any gain is attributed to it.
- **Controller applicability-001:** the frozen formal Gate loads Spark-X2.6-1.7B
  **BF16 safetensors**, not GGUF. The BF16 Fast screen cannot exercise
  `_mmvq_safe_batch`, so a BF16 score cannot close F3.
- **C6:** GGUF/NGRAM custom GPU experiments are pending without matching raw
  evidence; prefer a new CPU falsifier or an applicable BF16 candidate.
- **C11 / i6:** the host-mirror reachability model missed outer
  `spec_info`/`kv_indptr` conditions and shared the sum with the mirror.

## Profile (reused committed evidence, no GPU this session)
Prior deepseek-b executions already answered the throughput levers:
- g3-t1/t5: `f263262` changes 8 dispatch cells (Q8_0/Q4_K rows<=5120,
  batches 3-6), **0 cells at any batch >= 16**; all gate points are batch >= 16.
- g3-t7: the missing `window_left` in `extend_no_prefix` is a real 1.385x
  prefill-compute ceiling at S=8192 but is **acceptance-incompatible** (the
  reference wheel runs the same full-attention path); load-time dequant's
  8.95-11.85x is an L2-resident microbench ceiling that does not transfer.
- Branch: a4 decode is roofline-closed; trial-1 sync removal was 1.0018
  KEPT_NEUTRAL with decode points inside +/-2% run variance.

## Bottleneck chosen this trial
The only inherited engine change on top of `f263262` that had never been
provenance-audited: the t3/t4 **host-mirror** candidate. It replaces the device
slice bound `kv_indptr[-1]` with the caller's host int `paged_kernel_lens_sum`.
g3-t6 replayed integer equality over sampled cases but did not ask *where each
host sum comes from at each caller* — the question that decides whether the
patch adds a **new** correctness precondition (a potential silent
under-translation bug).

## Implementation / verification (executed)
`seis/trials/deepseek-b/g3-t8/verify_callsite_mirror.py` (stdlib,
`python3 -I`, no torch/GPU/model):

1. AST-extracts and line-anchors the two patched expressions and both guards.
2. Extracts all 14 `paged_kernel_lens_sum*` assignment sites and classifies
   provenance against the nearest `paged_kernel_lens*` tensor assignment.
3. Executes the real guard predicates over `spec_info` x `custom_kv_indices` x
   sum-type configurations.
4. Replays host-vs-device last-cumsum equality per provenance class with three
   negative controls.
5. Checks the surrounding unpatched code for the same host-value dependency.

Raw output: `seis/trials/deepseek-b/g3-t8/callsite-mirror.json`.

## Result
- 14 sites; 4 feed the host mirror. Guards admit a non-int sum in **0/21** rows.
- **0** integer divergences outside controls; **3/3** sensitivity controls
  flagged (so the equality test is not vacuous).
- The patch's precondition (`host int == device cumsum last element`) is
  **already required by unpatched code**: decode builds
  `global_override_indptr_cpu` from `torch.cumsum(seq_lens_cpu)`; prefill sizes
  the `kv_indices` buffer as `paged_kernel_lens_sum + 256` before
  `pack_read_stream`. `new_precondition_introduced = false`.
- Effect bound: **one D2H per eager step**; dead code under a unified SWA pool
  (`reads_are_translated`) and zero saving when `seq_lens_cpu is None`. Expected
  aggregate gain **~0**.
- **F3 unchanged:** `f263262` remains UNGATED and score-unreachable under the
  frozen BF16 Gate; only the pre-registered separate GGUF dispatch experiment
  (whose prerequisites are unmet) can change that.

## Decision
- **No engine change.** `git diff HEAD -- python/` is empty; the inherited
  candidate tree is left byte-identical.
- **`request_fast_gate=false`**: no new engine delta and expected ~0 gain; the
  controller should not spend a slot on a single-D2H hygiene patch.
- Next: reallocate deepseek-b slots to the campaign-level numeric-evidence
  blocker (non-null top-k) or independent verification of another worker's
  executable claim; run the separate GGUF protocol only if F3 hygiene is wanted.

## Deliverables
- `plan.md` (this file)
- `seis/trials/deepseek-b/g3-t8/{verify_callsite_mirror.py, callsite-mirror.json, README.md}`
- `iteration-proposal.json` with the required string fields and
  `request_fast_gate=false`
- local git commit; nothing here is an acceptance by itself.

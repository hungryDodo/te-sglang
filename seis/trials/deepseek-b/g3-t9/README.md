# G3 t9 — deepseek-b: frozen prefill-batch / KV-pool arithmetic (CPU-executed)

No GPU, no torch, no model weights, no measurement. Everything here is
source-extracted constants plus arithmetic over the signed frozen contract.
Raw output: `prefill-budget.json`. Script: `verify_prefill_budget.py`
(stdlib, `python3 -I`).

## Why this trial

The F3 mandate ("evaluate `f263262` before interpreting any gain") is already
closed as far as this node can close it: g3-t1/t5 executed the dispatch
contract (0 changed cells at any batch >= 16) and controller
`applicability-001` established that the frozen BF16 Fast screen carries no
GGUF artifact. This trial therefore offers **no BF16 number as F3 evidence**;
F3 stays UNGATED and score-unreachable under the frozen gate.

The engine lane on a4 is closed for the reasons already recorded: decode is
roofline-bound, the sync-removal family measured 1.0018 KEPT_NEUTRAL, the
sliding-window lever is acceptance-incompatible (g3-t7), and load-time dequant
was measured in an L2-resident regime (g3-t7). This trial asks the one
remaining *contract-level* question that the branch has not executed:
**what prefill batch width and what KV-pool headroom do the frozen flags
actually give the two largest points?**

## Executed facts (source, not assumed)

`extract_schedule_defaults` AST-parses
`python/sglang/srt/arg_groups/fields/schedule.py` and evaluates the real
default expressions:

- `max_prefill_tokens = 16384` (line 71 ff.), and its help states the real
  bound is `max(this, model max context length)`. Context length is frozen at
  9216 < 16384, so **16384 governs**.
- `chunked_prefill_size = None` in source; the frozen launch passes 8192.

Consequence (independent of KV memory): a prefill batch holds
`16384 // 8192 = 2` full 8192-token chunks. The PrefillHeavy point submits 16
requests of exactly 8192 input tokens, each exactly one chunk, so it runs in
**at least 8 prefill batches**. Negative control: halving the budget
(8192) yields 1 request/batch and 16 waves — the arithmetic is sensitive in
the expected direction (executed, `controls.halved_budget_halves_prefill_width
= true`).

## Conditional table (assumption A1, NOT executed)

A1: the `ignore_eos` admission path reserves `origin_input + max_new_tokens`
per request (anchored at `schedule_policy.py:1263-1266` and `_swa_new_tokens`
`:1204-1216`). Under A1 and the frozen `max_total_tokens = 131072`:

| point | demand (tokens) | pool | admissible of nominal C |
|---|---|---|---|
| prefill_heavy_c16 | 16 x (8192+32) = 131,584 | 131,072 | **15 / 16** |
| decode_ceiling_c256 | 256 x (32+512) = 139,264 | 131,072 | **240 / 256** |
| decode_heavy_c128 | 128 x (256+512) = 98,304 | 131,072 | 128 / 128 |
| standard_c64 | 64 x (1024+128) = 73,728 | 131,072 | 64 / 64 |
| medium_c128 | 128 x (512+64) = 73,728 | 131,072 | 128 / 128 |

All other six points admit their nominal concurrency. Note the frozen pool is
exactly `16 x 8192`, i.e. the PrefillHeavy prompt mass with **zero output
headroom**; the input-only lower bound fits all points, so the exceedance (if
real) is an output-reservation effect, not a prompt effect.

**Uncertainty (loud):** A1 is arithmetic, not a scheduler execution.
`hybrid_swa` memory accounting, page-size rounding, request-slot reserves and
mixed-decode tokens are not modelled and can change the admissible count. The
2-chunks / 8-waves prefill statement does **not** depend on A1.

## Verdicts

- **Legal lever: none identified.** `max_total_tokens`, `kv_cache_dtype`,
  `chunked_prefill_size`, `mem_fraction_static` are frozen. Widening the
  prefill batch would change GEMM shape/kernel selection
  (ALGORITHM_PRESERVING at best), cannot reduce prefill FLOPs, and is
  boot-coupled to the FlashInfer autotune dummy. The only FLOP-reducing lever
  (sliding window) is acceptance-incompatible per g3-t7.
- **Measurement: NOT_A_MEASUREMENT.** No engine change; `git diff HEAD --
  python/` is empty at commit time.
- **Falsifiable prediction for the controller (cheap, log-only):** on the
  frozen PrefillHeavy point the scheduler log line
  `scheduler.py:1271 max_prefill_tokens=16384` should appear, and per-batch
  prefill token counts should not exceed 16384 (2 x 8192). If the run's
  prefill batches exceed 16384 tokens, this trial's A2 is wrong and must be
  corrected before any batch-width proposal. If `max_total_num_tokens` at
  boot is not 131072, A1's pool arithmetic is void.

# G3 trial 12 — deepseek-b — plan

Campaign `continuous-20261109-01`, Iteration 12. Worker deepseek-b (node a4).
Bounds: 45 turns / 20 min / USD 5 nominal. No GPU, no SSH, no evaluator.

## 0. Hard constraints (unchanged)

- No GPU/SSH measurement; official gates are the controller's.
- No evaluator/corpus/seed/reference/protocol change. No EXACT / Promotion /
  best_verified claim. Canonical numeric stays INCONCLUSIVE (no raw top-k here).
- F3 (`f263262`, MMVQ thresholds) stays **UNGATED**: the frozen BF16 Fast screen
  loads BF16 safetensors and passes no GGUF/quantization flag
  (controller `applicability-001`). I will not spend a BF16 gate on F3 and will
  not claim it is evaluated by one.
- `request_fast_gate=true` only for an applicable BF16 engine candidate with a
  semantics argument. This trial expects `false` unless a verified patch lands.

## 1. Question for this trial (new, CPU-falsifiable)

Canonical C16 corrected trial-9's premise: the PrefillAdder's governing prefill
budget is the **whole-batch `rem_chunk_tokens`** (`chunked_prefill_size` = 8192
from the frozen launch), not `max_prefill_tokens` (16384). Trial 9 therefore
predicted **2 x 8192-token requests per prefill batch / 8 waves** for the
15%-weight PrefillHeavy C16 point. Trial 11 then built a KV-pool headroom story
on top of the frozen concurrency.

This trial tests the concrete, previously unexecuted claim:

> **H1:** `PrefillAdder` debits the full extend length of every admitted request
> from `rem_chunk_tokens`; hence with `chunked_prefill_size=8192` the first
> 8192-token request zeroes the batch chunk budget and the real admission
> predicate rejects any second 8192-token request in the same batch. PrefillHeavy
> therefore runs **1 request per prefill batch (16 admission events per wave of
> 16 concurrent requests), not 2**.
>
> **H2 (mixed-decode variant):** if mixed chunking is active, the budget is
> further reduced by `num_mixed_decode_tokens = running_bs`, so even the first
> 8192-token prompt is truncated to `8192 - running_bs` and continues as a tail
> chunk - a mechanism that can appear in PrefillHeavy TTFT.

Method: AST-extract and execute the real budget plumbing (`scheduler.py`
PrefillAdder construction arguments, `PrefillAdder.__init__` chunk-budget
initialisation, the real `add_one_req`/`add_chunked_req` guard predicates, the
real `_update_prefill_budget` charge expression), with hard exit-2 on any
anchor drift and negative controls that flip the deciding term. This is
arithmetic over executed source, not a throughput measurement.

## 2. Deliverables

- `seis/trials/deepseek-b/g3-t12/verify_prefill_chunk_budget.py` (stdlib,
  `python3 -I`, real-source extraction + execution).
- `seis/trials/deepseek-b/g3-t12/prefill-chunk-budget.json` raw output.
- `seis/trials/deepseek-b/g3-t12/README.md`.
- Updated `plan.md` (outcome), `iteration-proposal.json`, git commit.

No engine change unless the oracle exposes a semantics-preserving defect I can
argue and gate.

## 3. Kill criteria

- Anchor drift or a contradicted premise → record the negative with raw output,
  do not soften it.
- If the oracle cannot settle H1/H2 from executed source inside the budget,
  report what executed and what remains unknown rather than claiming either.

## 4. Outcome (executed)

`python3 -I seis/trials/deepseek-b/g3-t12/verify_prefill_chunk_budget.py --root .
--out .../prefill-chunk-budget.json` -> exit 0, 15/15 anchors, verdict
`H1_CONFIRMED_ON_SOURCE__T9_2_PER_BATCH_FALSIFIED`.

- **H1 confirmed:** `_update_prefill_budget` debits every admitted request's
  full extend from the whole-batch `rem_chunk_tokens`. With
  `chunked_prefill_size=8192`, one 8192-token prompt zeroes the budget and the
  real guard rejects a second: **1 request per prefill batch, 16 admission
  events per wave at C16**, not t9's 2/batch and 8 waves. The same guard exists
  in the `ignore_eos` path, so the result is independent of output-length
  pinning. Negative controls: `rem_chunk=None` or 16384 both admit 2.
- **H2 rejected:** `enable_mixed_chunk` source default is `False`; the frozen
  command passes no flag, so `num_mixed_decode_tokens=0` and no mixed-decode
  squeeze occurs (the 8176-token chunk is a sensitivity control only).
- Limits: arithmetic over executed source, no measurement; `max_prefill_bs`,
  prefill delayer, memory-budget rejection and real step counts are unmodeled.
  Exposed lever (NOT implemented): `max_prefill_tokens=16384` allows a wider
  prefill batch than the chunk debit currently permits, i.e. 2 x 8192 per batch;
  buffer/activation-memory safety is unverified.
- F3 (`f263262`) remains UNGATED; no BF16 Fast was requested (no engine change
  and a BF16 screen cannot exercise GGUF/MMVQ cells).

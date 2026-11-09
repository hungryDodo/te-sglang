# G3 trial 11 — deepseek-b — plan

Campaign `continuous-20261109-01`, Iteration 11. Worker deepseek-b (node a4).
Bounds: 45 turns / 20 min / USD 5 nominal. No GPU, no SSH, no evaluator.

## 0. Hard constraints I will not violate

- Cannot and will not run GPU/SSH measurements. All official gates are the
  controller's.
- Will not change evaluator, scoring, corpus, seed, reference, or protocol.
- Will not claim EXACT / Promotion / best_verified. No numeric arrays exist in
  this sandbox; canonical numeric stays INCONCLUSIVE.
- `request_fast_gate=true` only if the patch is an applicable BF16 candidate
  and I have a plausible semantics argument. A BF16 Fast screen structurally
  cannot evaluate GGUF/MMVQ (`f263262`, canonical F3): the frozen gate loads
  BF16 safetensors and passes no GGUF/quantization flags
  (applicability-001). I will not request a BF16 gate "to evaluate F3".

## 1. State re-establishment (first)

- `git log`, `git status`, diff vs the G3 base. Confirm whether `f263262`
  (MMVQ threshold) is present in HEAD and whether any worktree change exists.
- Re-read the remaining open questions from the canonical register that are
  actually CPU-addressable for this branch:
  - F3 UNGATED: applicability boundary only; no in-sandbox evaluator.
  - C11/C14 host-mirror family: still open, but i6/i8 were partial and the
    last two rounds produced premise errors (guard duplication, capacity
    sufficiency != exact equality). Do not repeat the same sweep.
  - a4 proposal queue: "verify host-mirror oracle/branch guards" — do this
    with a *new* bounded falsifier if, and only if, there is a concrete
    testable inversion; otherwise choose a different question.

## 2. Candidate question (pick after source check)

Prefer a genuinely new, CPU-falsifiable question over re-running a blocked
promotion request. Ranked:

1. **F3 applicability audit, executed not asserted.** Build a small script
   that proves from the *pinned source in this tree* which run path the frozen
   BF16 launch takes through `fused_mul_mat` and whether any `gguf.py` call
   site is reachable without a GGUF-quantized model config. Deliverable is a
   falsifier that would FAIL (i.e. show reachability) if the BF16 gate could
   touch the MMVQ cells. This converts the controller's prose applicability
   finding into an executable, attackable check — and it is honest: it does
   NOT gate F3, it bounds what a BF16 Fast can ever mean for F3.
2. **Host-mirror reachability** — only if I find a concrete new inversion
   (e.g. a caller sequence where the guard is bypassed). Otherwise declare
   the line saturated and say so.
3. **Prefill-heavy C16 structural accounting** — chunk/token-budget
   arithmetic from source (131072 = 16 x 8192 exactly; decode space after
   prefill). CPU-only, but risk: already partially covered by trial-2 notes.

## 3. Deliverables

- `seis/trials/deepseek-b/g3-t11/` : script + raw output JSON + short README.
- `plan.md` (this file, updated with outcome).
- `iteration-proposal.json` with the required string fields and
  `request_fast_gate` boolean.
- Git commit(s) with the evidence.

## 4. Kill criteria

- If the chosen question cannot be executed and falsified CPU-side within the
  remaining budget, record a bounded negative (what was checked, what is
  unknown) instead of a speculative patch.
- If work + tests exceed ~75% of turns, stop starting work and write the
  proposal.

## 5. Outcome (executed, 2026-11-09)

Chosen question: **KV-pool headroom arithmetic of the frozen points** (new for
this branch; the host-mirror, GGUF-unreachable and sliding-window lanes were
already closed or blocked).

Executed: `python3 -I seis/trials/deepseek-b/g3-t11/verify_pool_headroom.py
--root . --out .../pool-headroom.json` -> exit 0, 11/11 source anchors present,
verdict `ARITHMETIC_BINDS`.

- `prefill_heavy` C16: prompt demand `16 x 8192 = 131072` **equals**
  `--max-total-tokens`; decode headroom is exactly 0 while 512 output tokens
  are required. With radix cache disabled, the source-anchored pressure valve
  is `retract_decode()` -> re-queue/re-prefill.
- `decode_ceiling` C256: max demand 139,264, over by 8,192 (6.25%); pool is
  exactly full at 480 generated tokens per request.
- All other points fit at max output (largest 98,304 = 74.8% of pool).

This is **arithmetic + source anchors, not a throughput measurement**. Runtime
causality is unproven: `new_token_ratio` reservation, chunked-prefill
admission, full-vs-SWA allocator binding and possible EOS truncation are not
modeled. The zero-gate falsifier is retraction counts
(`num_retracted_reqs` / `#retracted_reqs:`) in **existing** Fast artifacts;
zero at PrefillHeavy kills the hypothesis.

No engine change. F3 (`f263262`) remains UNGATED; a BF16 Fast screen cannot
exercise the GGUF/MMVQ cells (applicability-001), so none was requested.

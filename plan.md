# G3 trial 9 — glm-b (i0009), campaign continuous-20261109-01

## Session bounds
45 turns, 20 min hard, USD 5 nominal cap (prior sessions died at USD 1–3; keep ≤ ~30 turns).
Early recoverable checkpoint commit; final quarter reserved for iteration-proposal.json + commit.
No GPU/SSH: CPU-decidable work only; official measurement stays with the controller.

## State at entry (i7/i8 + sync-002)
- Engine lane saturated per existing traces (decode C128/C256 bytes-bound at roofline,
  99.2% GPU busy at C256; prefill near-roofline; sync sweep closed; prefill-mirror
  b196fd4 measured no attributable gain in G3-i2 gate). No new engine candidate.
- Top-k evidence chain CPU-side fully closed (i7/i8). Standing pending controller
  request = numeric probe re-run with BOTH recipe fixes → candidate-x3 on the
  2c854e0-identical HEAD if clean. Do not rebuild fixtures for it (008).
- Promotion candidate tree kept bit-identical to 2c854e0: `git diff 2c854e0 HEAD --
  python/` EMPTY at entry, stays EMPTY this trial.

## New question for this trial
The promotion claim for the 076b006+aa45af5 lineage will be ALGORITHM_PRESERVING.
Three × exact-token 10/10 fast gates are sample evidence; nobody has audited the two
accepted hunks' edge paths. Reading the live code surfaced three concrete hazards:

- H1 (arithmetic exactness of aa45af5): `kv_last_index = int(seq_lens_cpu.sum())`
  replaces `kv_indptr[-1]`. `kv_indptr` is cumsum over CLAMPED lens on the SWA
  wrapper-0 path (window+1 clamp at lines 1687–1693) while the raw mirror is
  UNCLAMPED; exactness holds only because `update_sliding_window` passes the clamped
  mirror `seq_lens_cpu_tmp` down. Invariant to execute: clamp(mirror) lanes ==
  clamp(device lens) lanes for every reachable decode shape, incl. above-window seqs.
- H2 (global lifecycle of `global_override_indptr_cpu`, 076b006): set at :1832
  (only-if-None), consumed by fast_decode_plan :1853–1894, cleared at :1916 — with NO
  try/finally. An exception between set and clear leaks a stale override; 1832 then
  skips rebuild forever after and every later eager step plans against a wrong-length
  indptr (silent wrong attention), and the prefill save/restore at :2635–2646
  faithfully restores the stale value. Also: :1832 rebuild lacks the `bs ==
  seq_lens_cpu.numel()` guard that aa45af5 has; safety there rests on the caller's
  `seq_lens_cpu[:bs]` slice (:748–846).
- H3 (mirror origin/aliasing): is `forward_batch.seq_lens_cpu` an immutable per-step
  snapshot or a tensor the scheduler thread can mutate while the forward thread
  reads it (overlap scheduler)? Source-level check of its construction.

## Method (i7/i8 harness lessons enforced)
- Extract REAL statements via ast and execute the `ast.unparse` form (dedent
  re-indent bug; needles match UNPARSED text; enclosing-If extraction).
- Hard verdict gate: ANY harness error ⇒ NO_CLAIM.
- Sensitivity controls: an UNSYNCED mirror (≠ device lanes) MUST produce a mismatch;
  an injected exception mid-plan MUST leave the stale global detectable. Proves the
  oracle can see the failures it claims to bound.
- Raw JSON with per-case file:line provenance + sha256 of extracted segments.

## Deliverables
1. plan.md (this file) — committed first as recoverable checkpoint.
2. g3-t9/probe_promotion_invariants.py — executed oracle (exit 0 clean; raw JSON).
3. g3-t9/VERIFICATION.md — bounded conclusion per hypothesis, limits, next actions.
4. iteration-proposal.json — string fields + request_fast_gate=false; controller ask
   unchanged in substance (numeric probe re-run with both fixes → candidate-x3 if
   clean), now carrying the audit outcome for the ALGORITHM_PRESERVING claim.

## Non-goals
No engine change this trial (python/ untouched — the candidate-x3 request depends on
2c854e0-identity; the H2 hardening fix is recorded as a one-hunk follow-up commit the
controller may order before or after promotion). No promotion claim, no precision
change, no protocol/evaluator change, no missing-k fixture repeats, no NGRAM/MMVQ/GGUF
lane work.

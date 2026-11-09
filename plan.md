# G3 trial 12 — deepseek-a — NGRAM accept-path fault-line via real top-8 runtime packet

Campaign: continuous-20261109-01, Iteration 12, worker deepseek-a, trial 12.
Canonical entry: F1 NGRAM self-speculation is BUG_CONFIRMED (throughput +21.6% real,
greedy correctness broken). Promotion prohibited until an ALGORITHM_PRESERVING fix
passes the controller's gates.

## Constraint refresh (authoritative)

- No GPU/SSH/evaluator access. CPU-only executed analysis; no invented measurements.
- All previous numeric probes lacked raw top-k (C5). C21/C22 now provide real
  complete top-8 ON/OFF data from ordinary NGRAM runs (Controller evidence packet at
  `controller_evidence/continuous-20261109-01/20261110-runtime/`). Read-only input.
- Hand-made fixtures and repeated field/path inventories do not count. A bounded
  executed CPU analysis with preserved raw output does.
- Engine lane: any change to python/sglang/** is a candidate until measured; a source
  change is never its own acceptance.

## Question this iteration (new, falsifiable)

Prior work established ON/OFF divergence but the discriminator between
Hypothesis A (verify-row context misalignment: logits argmax computed under the wrong
prefix) and Hypothesis B (post-commit KV/bonus slot misalignment: verify right, KV
polluted) was blocked by missing accept_len==1 cases and by shallow top-k coverage.

The real packet (C21) has per-position chosen token/logprob plus top-8 on BOTH sides.
New testable question:

**H-D (commit self-consistency discriminator):** At every ON response position, does
the token the engine reports as committed equal the argmax of the top-k row that the
same engine reports for that position? If ON commits a token that is NOT its own
reported top-1 while OFF never does, the commit/slot read is already inconsistent with
the logits row at report time (supports B-like slot fault); if ON is always
self-consistent yet diverges from OFF, the wrong token was already the argmax of a
mis-contextualized verify row (supports A).

Secondary H-E (gap structure): the first-divergence ON token's logprob rank/gap
relative to OFF top-8 tells whether the ON commit was a low-probability draft token
accepted without a valid target match, or a genuinely high-probability token under a
different context.

## Method

1. Read `controler_evidence/.../README.md` + MANIFEST, locate the ON/OFF response JSON.
2. Write a bounded parser (own directory, stdlib only, `python -I`) that:
   - joins ON and OFF responses per case, finds first divergence;
   - extracts per-position chosen id/logprob and top-8 ids/logprobs where present;
   - computes rank/gap of ON token in OFF top-8, and ON/OFF self-consistency counts;
   - asserts hard preconditions (field presence); emits derived JSON + summary.
3. Preserve raw derived output under `g3-t12/raw/`; state scope limits explicitly.
4. Decide: if self-consistency is violated -> concrete slot-read defect to point at;
   if not -> A stays alive; record both as bounded negative/partial evidence.
5. Write `iteration-proposal.json`, commit; request only what evidence supports.

## Deliverables

- `g3-t12/analyze_ngram_pairs.py`, `g3-t12/raw/*.json`, `g3-t12/FINDINGS.md`.
- `iteration-proposal.json` with string fields bottleneck,hypothesis,patch,tests,
  learning,next_proposal and boolean request_fast_gate (false unless a measured
  BF16-safe candidate exists; this iteration is CPU evidence).
- Plan/uncertainty honestly recorded; no EXACT claim.

## Outcome (recorded after execution)

- H-D: commit self-consistency holds — 959/960 positions; the one exception is an exact
  logprob tie (two ids at -1.556087613105774), not a slot-read fault. See g3-t12/FINDINGS.md.
- New executed result: same-prompt ON requests with identical committed prefixes report
  rows differing by up to 5.16 nats (48/48 identical output pair); position 0 is bit-equal
  in all pairs. Rows are not a function of the committed prefix alone.
- H-E: gap magnitude is not a valid discriminator; same-prefix ON-vs-ON control shows
  equal/larger gaps. Prior gap-based localization must be downgraded, not extended.
- No engine patch, no promotion, no performance claim. Raw: g3-t12/raw/.

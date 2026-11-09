# G3 t12 — real-packet NGRAM row/commit analysis

Status: **bounded CPU evidence, executed** (no GPU, no throughput, no promotion claim).
Input: Controller-supplied `continuous-20261109-01/20261110-runtime/ngram-full-observations.json`
(ordinary ON b9a91aa13 vs deletion-only OFF 999d4b243, repeat-512 C1/C4same double pass,
20 primary HTTP responses = 10 ON/OFF pairs, 960 output positions, complete chosen + top-8).
Originals untouched; derived raw output in `g3-t12/raw/`.

Reproduce: `python3 -I g3-t12/analyze_ngram_pairs.py`
(cwd `/workspace`; packet path pinned by sha256 in the output JSON).

## Question

Does the ON engine commit a token that is *not* the argmax of the top-8 row it reports
for that same position? A genuine token/row mismatch would support a commit/slot-read
fault; universal self-consistency would move the fault into the logits context itself.
Secondary: are reported rows a function of the committed token prefix alone?

## R1 — Commit self-consistency: 0 genuine mismatches in 960 positions

- Every reported `output_ids[i]` equals the argmax token of `top_logprobs[i]` in 959/960
  positions; the single exception is an **exact tie**: `A/repeat-512/c4same` instance 3,
  position 6, ids 48977 and 1271 both at logprob `-1.556087613105774`; the engine
  committed 1271 and `chosen_logprobs[6]` is that same value.
- Therefore the earlier "ON reports the wrong committed token as its own top-1" reading
  (trial-3 partial sample) is **not reproduced** on the complete 20-response packet: there
  is no position where the committed token and its reported row disagree beyond a tie.
- `chosen_logprobs[i]` is consistent with the same row's logprob for the committed token
  in the same 959/960 positions.

Consequence: the wrong ON token is always the argmax of the row ON itself reports. Any
fault is upstream of the commit/read: it is in how that verify row was computed or which
state it was computed from.

## R2 — Same-prefix ON-vs-ON control: rows are NOT a function of the committed prefix

Method: for two ON requests with the same prompt, compare the max absolute logprob
difference over shared top-8 token ids at positions where their committed token ids are
identical. Position 0 (prefill row) is exactly equal (delta 0.0) in all 12 pairs, so
prefill is deterministic and the comparison is not measuring generic request noise.

| pair (same prompt) | shared token prefix | max delta (nats) | median | positions < 0.3 |
|---|---|---|---|---|
| B/c4same 1 vs 3 | **48/48 (whole output identical)** | **5.157** | 1.750 | 0–4, 13 |
| A/c4same 0 vs 1 | 5 | 3.173 | 0.918 | 0 |
| A/c4same 0 vs 2 | 5 | 2.882 | 1.591 | 0 |
| A/c4same 0 vs 3 | 5 | 2.898 | 1.606 | 0 |
| A/c4same 1 vs 2 | 6 | 3.312 | 0.717 | 0 |
| A/c4same 1 vs 3 | 6 | 3.390 | 0.814 | 0 |
| A/c4same 2 vs 3 | 9 | 0.156 | 0.092 | 0–8 |
| B/c4same 0 vs 1 | 4 | 3.137 | 0.886 | 0 |
| B/c4same 0 vs 2 | 4 | 3.075 | 0.853 | 0 |
| B/c4same 0 vs 3 | 4 | 3.121 | 0.913 | 0 |
| B/c4same 1 vs 2 | 4 | 0.158 | 0.139 | 0–3 |
| B/c4same 2 vs 3 | 4 | 0.091 | 0.078 | 0–3 |

Full-identical pair profile (B/c4same 1 vs 3, all 48 emitted tokens equal):
positions 0–4 deltas ≤ 0.22; position 5 = 1.12, then 1.6–5.16 through position 12;
position 13 = 0.07; then 1.0–2.9 for the rest.

Interpretation (bounded): two requests that committed the *same* tokens report
materially different logits rows at the same positions, with a sharp onset after
position 4. This is not uniform batching noise: prefill row 0 is bit-identical, and
several same-prefix pairs stay within 0.16 nats for their whole shared prefix. The
bimodality is consistent with the verify row depending on **non-committed state**
(draft-tree structure and/or the KV state carried across a verify-step boundary), i.e.,
the row is not determined by the visible committed tokens.

## R3 — Consequence for prior gap-based reasoning (correction)

The ON-vs-OFF "committed token is 1.0–3.9 nats behind OFF argmax" observation cannot by
itself localize the fault to a specific wrong-prefix/verify-row mechanism: same-prefix
ON-vs-ON pairs in this same packet show equal or larger row gaps (up to 3.4 nats over a
4–6 token shared prefix; 5.16 nats over a fully identical 48-token output). The row gap
magnitude is therefore not a clean discriminator between mask, position, and KV-state
hypotheses. It remains a real divergence counterexample, but its causal reading must be
downgraded.

## What this does NOT show

- No root cause, no tested fix, no ALGORITHM_PRESERVING claim; F1 stays BUG_CONFIRMED/open.
- Cannot distinguish (A) tree-mask/position context leakage from (B) KV-move/state
  corruption: the packet has no per-step draft ids, no accepted-node indices, no KV or
  position dumps, and no `accept_len == 1` control.
- No throughput, quality, numeric, or promotion statement.

## Next falsifiable step (for the controller, custom diagnostic not a fast gate)

Run the same single request twice with NGRAM ON, same commit, same prompt/seed and
max_new_tokens, logging per verify step: `positions`, the tree-mask rows, `accept_lens`,
`accept_index`/`retrieve_index`, and the reported top-8 rows. Discriminators:
(a) two identical single requests differ at the same prefix position → request-level
nondeterminism independent of concurrency; (b) they agree but C4-same requests differ →
batch-slot/state coupling; (c) instrument `seq_lens`/`positions` host vs device and compare
the row at the first divergence with the row reconstructed from the committed prefix alone.
Bounds: one prompt, 48 output tokens, 2 runs, no scoring; logs only, no speed comparison.

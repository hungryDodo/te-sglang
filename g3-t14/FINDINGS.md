# G3 trial 14 — deepseek-a — matched-prefix test of the NGRAM verifier

Campaign continuous-20261109-01, Iteration 14. Canonical F1 stays
**BUG_CONFIRMED**; nothing here promotes NGRAM or closes the root cause.
CPU analysis of the Controller packet only; no GPU/SSH, no engine change.

## Question

At the first ON/OFF divergence position `p`, the ON original response and the
4 OFF one-token replays share the identical 512-token prompt plus the same `p`
committed tokens (prefix alignment re-verified: `prefix[:512] == prompt`,
`prefix[512:] == ON outputs[:p] == OFF outputs[:p]`, replay `prompt_tokens`
516/516/518/527). Therefore their reported top-8 at `p` are two measurements
of the same visible prefix. Does the ON verifier score the same distribution
(→ commit/selection defect) or a different one (→ verifier hidden-state
defect)?

## Results (real packet, script-derived)

- **Row-consistency precondition passes**: across all 976 output positions of
  the 20 primary + 16 replay responses, the reported chosen LP equals its own
  top-8 row in 976/976, the chosen token is present in that top-8 976/976, and
  it is the row's argmax at 975/976 (the single exception is the known exact
  tie of C21/C24). The anomaly below is therefore in the row contents, not in
  a chosen-LP/top-8 mispairing.
- **OFF x4 replay control is exact**: all four repeats per case return the
  same top-8 token sets with LP spread **0.0** and the same argmax; the OFF
  primary row at `p` vs the OFF replay row differs by at most 0.144 nat. This
  is the reproducibility floor for a matched prefix on this engine.
- **ON is self-consistent with its own reported row**: in 4/4 cases the
  committed token is its own reported top-1 (rank 0, LP gap exactly 0). A
  variant where the engine commits a token that is not the argmax of the row
  it reports is therefore excluded at these points. (EAGLE accept semantics
  make this self-consistency expected under either hypothesis; it does not by
  itself exclude a wrong-context row being used for both.)
- **ON's row differs materially from OFF's row at the same prefix**:

  | case | p | ON top-1 (LP) | OFF top-1 (LP) | ON token in OFF top-8? | common-token mean/max LP diff |
  |---|---|---|---|---|---|
  | 0 | 15 | 3642 (-1.7844) | 100023 (-0.0023) | no (0/48 OFF positions) | 3.664 / 4.823 |
  | 1 | 4 | 86 (-1.6600) | 258 (-1.7860) | yes, gap 1.000 | 0.734 / 1.405 |
  | 2 | 4 | 267 (-2.6465) | 258 (-1.7494) | no | 0.722 / 1.116 |
  | 3 | 6 | 1271 (-1.4467) | 48977 (-0.5821) | yes, gap 1.203 | 0.615 / 0.932 |

  Case 0 is decisive against a near-tie explanation: OFF scores 100023 at
  -0.0023 with its runner-up at -6.95 (essentially deterministic), while ON
  ranks 3642 first (-1.78) and 100023 second (-2.35) — a materially different
  hidden state, not a permutation of near-ties. Case 0's label is
  `A/repeat-512/c1same` (concurrency 1, single request), so its divergence
  cannot be attributed to cross-request batch composition. Cases 1–3 are near-ties
  *inside ON's own row* (0.047, 0.031, 0.016 nat), so their flips are
  consistent with a modest perturbation of the correct distribution.
- **No clean row-shift along the true trajectory**: in all 4 cases ON's
  committed token is the argmax at **none** of the 48 OFF trajectory
  positions; the best OFF-position match still disagrees on argmax with mean
  common-token LP distance 0.40–0.72 nat. A one-step off-by-one on the true
  continuation is not supported.
- **Pre-divergence rows are mostly matched, but structurally not uniformly**:
  over 53 matched-context positions, argmax match is 53/53 and both sides'
  chosen token is always their own argmax; the ON−OFF chosen-LP difference is
  median −0.0087 nat (ON slightly flatter, 40/53), but 14/53 exceed 0.2 nat
  and 6/53 exceed 0.5 (max 1.324). The deviation is position-structured:
  **position 3 differs by 0.22–0.59 nat in all 10 ON/OFF pairs**, while OFF's
  own cross-instance spread on identical outputs is ≤0.087 nat. First
  divergence is at position 4 in 8/10 pairs.
- Identical-output ON/ON pair (B/c4same instances 1 vs 3): max chosen-LP
  difference 0.491 nat with 0 positions >1 nat; OFF identical-output pairs
  max 0.037 (A) / 0.087 (B). No >1-nat same-output discrepancy in this packet
  (the 5.157-nat ON/ON case of C24 is not reproduced by this grouping).

## Interpretation (bounded)

The parity-breaking token is not a commit that contradicts the row the engine
reports; the verifier's row for the same visible prefix is itself different,
materially so in 1/4 cases and perturbed in the rest. This is consistent with
a **row-wise verify hidden-state defect** (tree mask/position/KV assembly,
candidate A), not with a pure commit-slot misread. The systematic position-3
offset across all pairs and both prompts further suggests a structural
per-tree-position effect rather than random batching noise. It does **not**
identify the defective site, does not rule out post-commit KV pollution
feeding the next step, and does not close any root/mask/overlap path.

## Limitations

Response-level real data only: 4 replay cases / 3 unique prefixes, top-8 (not
full distribution), reported BF16 LPs rather than recomputed logits; replay
probes are `max_tokens=1`; no accept_len=1 run; no live mask/KV/mover/overlap
observation; ON-vs-OFF batch compositions differ, so small LP differences are
not purely attributable; `median 0.009` positions show the verifier is usually
right. Nothing here is a gate, numeric, speed or promotion claim.

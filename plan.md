# G3 trial-8 plan — glm-a (campaign continuous-20261109-01)

Direction: revisit SWA/FP8+SWA history; precision quality unresolved; exact runtime
when useful. Lane state at HEAD (engine bit-identical to i4's fast-gated 5a20347):
exact-runtime lane closed three ways (t5/t6/t7), fp8+SWA candidate-x3 parked on
missing top-k numeric evidence (F4/C5), decode side beyond static reach pending the
t7 trace request. No new BF16 engine candidate exists to screen; a re-screen of the
unchanged tree is a non-action.

## New falsifiable question (uses the controller's 20261110-runtime packet — real data, not a fixture)

The packet's NGRAM pair gives 36 complete HTTP responses (20 primary incl. 10
ON/OFF paired same-prompt decodes + 16 replay) from REAL frozen-launch servers
(BF16 KV, FlashInfer, both sides; OFF commit 999d4b243 is plain BF16). Every
position carries complete top-8 logprobs — the first real full-margin data for
this model. F2's blocker is that fp8-KV has NO quality budget: the only tested
bar was token-exactness, which C1 shows fp8 fails (159/512) while exact-runtime
lineages pass (0/512).

H8 (falsifiable): real BF16 generations of this model have a near-tie margin tail
P(top1−top2 ≤ δ) heavy enough that ANY perturbation of fp8-KV's measured scale
(E2: per-token |Δlogprob| mean 0.012–0.153 nat) must flip greedy tokens within a
512-token horizon with near-certainty — i.e. the exact-token bar is statistically
unreachable for the whole precision class, and a usable quality budget MUST be
flip-rate/margin-stratified, not token-exact.

Falsifier (CPU, executed): margin census over all 36 responses.
- margins top1−top2 per position (both roles; OFF = plain-BF16 baseline);
- P(margin ≤ δ) table across the fp8 perturbation scale;
- flip-risk projection: per-position flip prob ≤ P(margin ≤ 2ε), P(≥1 flip in 512)
  = 1−(1−p)^512, evaluated at ε ∈ E2's observed range;
- mechanism checks against the packet's own ON/OFF divergences: margin at each
  pair's first-divergence position vs the position-matched margin distribution
  (near-tie mechanism => systematically smaller), plus post-first-flip cascade
  rate (explains 159/512 ≈ 31% vs a handful of first flips);
- parsing controls: rows sorted-descending check, chosen-vs-top1 logprob equality,
  and reproduction of the packet's own self-top1 anomaly count (ON 1, OFF 0) —
  if the census does not reproduce it, the parser is broken and NO claim is made.

## Deliverables
- g3-t8/margin_census.py + g3-t8/raw/margin_census_output.json (executed, -I)
- g3-t8/quality-budget-proposal.md: the margin-grounded acceptance-contract draft
  for precision-class candidates (what F2 says is undefined), stated as a
  registration proposal, not a self-granted budget
- plan.md OUTCOME, iteration-proposal.json (request_fast_gate=false — tree
  unchanged, nothing to screen), commits, clean status

## Limits stated up front
- Margins observed only where generation was actually sampled (real prompts, ~48–64
  positions each; short-horizon relative to 512 — projection extrapolates the
  per-position model, with the ON/OFF cascade rate as the compounding check).
- Top-8 truncation: margins to tokens OUTSIDE top-8 are ≥ top1−row8; ON-outside-
  top8 cases bound differently. Census reports the truncation exposure.
- ON-side data is NGRAM-perturbed; the BASELINE census uses OFF (plain BF16) and
  replay-OFF (stable across 4 replays each) only. ON is used solely for mechanism
  checks on divergence dynamics.
- No engine change, no gate request, no protocol/evaluator touch; packet originals
  preserved read-only; derived outputs written under g3-t8/.

## Budget
45 turns / 20 min / $5. Checkpoint commit after plan; final quarter for proposal,
tests record, final commit, clean-status verification.

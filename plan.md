# G3 trial 14 — deepseek-a — real NGRAM packet: does ON's verifier distribution differ from OFF at the same prefix?

Campaign continuous-20261109-01, Iteration 14, worker deepseek-a.
Canonical F1: NGRAM greedy correctness BUG_CONFIRMED, promotion prohibited.
No GPU/SSH; this is CPU analysis of the Controller-supplied real packet.

## Question (new, falsifiable)

The NGRAM ON/OFF bug has two live source-level candidates: (A) verify-row
context/mask/position misalignment — the verifier sees a different prefix, so
it *scores* a different distribution; (B) commit/KV-slot misalignment — the
verifier scores the same distribution but the engine commits a different
token. Hand-port CPU models could not close either. The real packet contains
the needed discriminator at response level:

For each replay case, the prefix is identical for the ON original response
and the 4 OFF replay responses (same 512-token prompt + same p committed
tokens). Therefore, at the divergence position p, ON's reported top-8 and
OFF's reported top-8 are two measurements of the *same* (prompt, prefix)
context. The 4 OFF repeats are also a same-context reproducibility control.

- If ON's top-8 at p is numerically identical to OFF's (within the OFF x4
  noise floor) while ON committed a non-argmax token, then the verifier saw
  the same distribution and the defect is in commit/accept selection (B).
- If ON's top-8 at p differs from OFF's materially (LP gaps >> noise floor,
  rank/argmax changes, tokens outside OFF top-8), then the verifier genuinely
  scored a different hidden state for the same visible prefix (A).

Pre-registered discriminator: signal/noise ratio, where noise is the maximum
same-token LP spread across the 4 OFF replays at the same position. Also
report ON self-consistency (is the committed token the argmax of ON's own
reported top-8, and at what rank/LP gap) and pre-divergence row agreement.

## Inputs (read-only)

`controller_evidence/continuous-20261109-01/20261110-runtime/ngram-full-observations.json`
(real ON b9a91aa13 / OFF 999d4b24, BF16/FlashInfer, repeat-512, 20 primary +
16 replay HTTP 200 responses with complete output IDs, chosen LPs and top-8).
Originals preserved; derived output goes to `g3-t14/`.

## Method

`g3-t14/analyze_ngram_packet.py` (stdlib json, run `python3 -I`, data path as
argument) maps replay cases to their ON/OFF primary records, aligns the prefix,
compares top-8 sets/LPs at the divergence position, computes the OFF x4 noise
floor, checks ON self-consistency and pre-divergence row agreement, and scans
all 20 primary responses for identical-output-different-LP pairs. Hard
precondition checks fail closed (prefix must match record outputs; all replay
HTTP 200). Raw JSON output preserved.

## Deliverables

`g3-t14/analyze_ngram_packet.py`, `g3-t14/raw/packet_analysis.json`,
`g3-t14/FINDINGS.md`, `iteration-proposal.json` (request_fast_gate=false; next
step is an exact opt-in custom NGRAM instrumentation request, not a gate). No
engine change; no gate, numeric, speed or promotion claim. F1 stays
BUG_CONFIRMED; any accept-len=1 / live-overlap question remains UNTESTED.

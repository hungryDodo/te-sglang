# g3-t8: margin-grounded quality-budget proposal for precision-class candidates (F2/F4)

Status: PROPOSAL FOR REGISTRATION by the trusted controller. Not a self-granted
budget; no EXACT claim is made here. Derived analysis (g3-t8/raw/
margin_census_output.json) executed over the controller's real 20261110-runtime
packet (pair UUID 2202b11636bc4632823c138fb9644ae6); originals untouched.

## Measured facts (all controls passed: rows sorted, chosen==top1, self-top1
anomaly counts reproduce packet ON=1/OFF=0, 10/10 pairs, 0 parse errors)

F-a. Plain-BF16 frozen launch (OFF, 999d4b243), 10 real responses / 480 positions
(repeat-512 A/B prompt family, greedy): top1-top2 margin min **0.9375 nat**,
p50 5.97, p90 7.98; **P(margin <= 0.5) = 0/480**; P(<= 1.0) = 3.96%.
F-b. fp8-KV's measured per-token perturbation scale (G2 E2, BF16-vs-fp8 paired
probes): |dlogprob| mean 0.012-0.153 nat. A greedy flip needs margin <= eps1+eps2
<= 2*eps_max = 0.306 nat. **On F-a's instances this is 0/480: fp8-KV-scale noise
cannot flip ANY of these positions.**
F-c. The near-tie tail is prompt-instance-dependent: E2 (different instances,
natural/random text) did observe flips at margins < 0.15. So exactness outcomes
are decided by WHICH positions a probe samples -- invisible without top-k.
F-d. Divergence dynamics (10/10 real ON/OFF divergent pairs): divergence is
effectively permanent -- post-first-flip cascade rate 0.91-1.00, resync events
0-4 out of ~44-46 trailing positions. One flip at position p yields ~
(n-p)/n position mismatch. The fp8 lineage's 159/512 = 31% (C1, glm-a i1/i2) is
therefore the signature of ~1 late flip (p ~ 353) or a few flips, NOT 159
independent corruptions.
F-e. NGRAM's perturbation (for scale contrast): at the 10 real first-divergence
positions the OFF-side margin was 0.94-1.16 nat in 9/10 cases (one 6.95), and the
replay OFF argmax margins (6.93 / 1.0 / 1.203) match the packet's quoted
ON-outside-top8 gaps. NGRAM must overcome >= ~1 nat -- 6-65x the fp8 scale; its
10/10 divergence is in no way evidence about fp8-class noise.

## Consequences

C1. H8 (this trial's falsifiable hypothesis: the real margin tail is heavy enough
that token-exactness is statistically unreachable for fp8-class perturbations) is
**REFUTED on the observed prompt class** (F-b). The opposite holds there: on
repeat-512-class prompts fp8-KV is PREDICTED token-exact. This is a prediction,
not proof -- 480 positions bound P(margin<=0.5) only to <= ~0.6% (rule of three).
C2. The unresolved fp8 quality question (F2) is thus sharper: it reduces to the
rate of RARE near-tie positions per prompt class, plus cascade, not pervasive
numeric drift. A 31% mismatch number alone cannot distinguish "one benign
near-tie flip + cascade" from "a real defect" -- margin data at the flip position
can.
C3. Sharp falsifiable prediction for the next numeric probe: run the (now
recipe-proven) top-8 probe on the fp8 lineage with repeat-512-class prompts.
Expected: token IDs exact OR first-flip positions with baseline margin <= 0.306
nat. If mismatches appear at positions whose baseline margin > 0.306 nat, the
fp8 lineage has a NON-precision defect (e.g. late-remap/batch-composition class,
cf. C1 i3) and the F2 framing changes from quality-budget to bug.

## Proposed acceptance contract (for the controller to register/adapt)

For a precision-class candidate (kv-dtype/weights quantization) on greedy probes:
1. Collect per position: token ID, chosen logprob, top-8 rows (both servers), via
   the C21-proven request/read shape.
2. epsilon_max := max over same-prefix positions of |dlogprob(chosen)| between
   reference and candidate.
3. PASS-precision-noise iff every first-flip position p has reference
   margin(p) <= 2*epsilon_max (flips allowed only where noise can reach), AND
   total mismatch fraction is cascade-consistent (mismatch >= (n - p_last_flip)/n
   bounds hold within the measured resync slack <= ~10%).
4. Any flip at margin > 2*epsilon_max = defect, not noise -> candidate rejected
   regardless of throughput.
5. Report per prompt class (repeat-512 AND natural text): classes differ (F-c);
   a budget valid on one class does not transfer.

## Limits
- 480 positions / 10 responses / one prompt family / 48-token horizon; the 512
  extrapolation uses the cascade rates measured on NGRAM-scale perturbations.
- E2's epsilon distribution was measured on the argmax token of paired probes;
  2*epsilon_max is the standard worst-case flip bound, conservative.
- ON-side rows are NGRAM-perturbed and used only for dynamics, never as baseline.
- No engine change this trial; nothing here authorizes promotion (F2/F4 gates
  still apply: trusted top-k evidence first, then candidate-x3 in G3 namespace).

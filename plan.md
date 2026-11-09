# G3 trial-9 plan — glm-a (campaign continuous-20261109-01)

Direction: revisit SWA/FP8+SWA history; precision quality unresolved; prefer exact
runtime when useful. Lane state: exact-runtime lane closed *statically* (t5/t6/t7);
fp8+SWA candidate-x3 still parked on missing top-k (F4/C5, t8 registered a
margin-based quality-contract proposal); i4's BF16 screen 1.0017. The controller
has now supplied the real measured BF16 C128 decode trace that t7 asked for
(controller_evidence/continuous-20261109-01/20261110-decode-trace/, source commit
15fd7f4, run a3b0f3ec, Medium512/64 C128, cudaGraphLaunch16 observed).

## New falsifiable question (real measured data, not a fixture)

H9: "On the frozen BF16 C128 decode workload (the regime where F5's SWA host-sync
removal bought x1.0543 on a2), the host-side time NOT hidden under executing
kernels is large enough (>= ~2% of step wall) that an exact-runtime sync-removal
candidate is worth a Fast slot on a1."

Falsifier (CPU, executed, stdlib gzip/json on the packet's raw trace):
1. Parser control: independently recompute the README's stated anchors — marked
   window 571.927153 ms, 8560 kernel instances, kernel interval union
   565.549025 ms, CPU step union 28.303983 ms. Exact match => parser sound; any
   mismatch => NO CLAIM (hard gate).
2. Pairing control: 16 CPU labels pair 1:1 with 16 GPU labels by External id.
3. Derive the quantity that decides H9: per-step and total host-active time that
   is NOT overlapped by any kernel interval (the true removable ceiling for the
   SWA/host-sync class), plus EventSynchronize time not overlapped by kernels
   (host blocked, GPU idle = hard bubbles), plus per-step bubble distribution.
4. Gap anatomy: the README's largest host inter-step gap (34.676268 ms,
   overlapping kernels 34.417705 ms) — classify which kernels execute inside it
   (decode-graph vs prefill/mem ops) to explain the gap rather than assume.
5. Planted-gap negative control: inject a synthetic 1.0 ms CPU-only gap into a
   copy of step 1's intervals; the detector must recover it within tolerance, and
   must report ~0 on an all-overlapped synthetic step. Independent expected
   values, not self-fulfilling.
6. Kernel census (clearly labeled: summed durations are NOT wall time): top
   consumers by total time across the 16 forwards, to seed the next *kernel-side*
   hypothesis (trial-2 corollary: ~16 ms/step above weights-only roofline).

Decision rule: if the not-hidden host ceiling over the 16 measured steps is well
under ~2% of marked wall, H9 is refuted for this workload/node and the SWA lane
stays closed with measured evidence; the census then names where the step time
actually goes. If a concentrated bubble >= 2% exists in reproducible steps, the
outcome flips to a candidate proposal in next_proposal (not self-graded here).

## Deliverables
- g3-t9/trace_bubble_census.py + g3-t9/raw/trace_bubble_census_output.json
  (executed with python3 -I, raw output preserved)
- plan.md OUTCOME, g3-t9/VERIFICATION.md, iteration-proposal.json
  (request_fast_gate=false — no engine change is proposed this trial; a screen of
  the unchanged tree is a non-action)
- commits at checkpoint and final; git status --porcelain empty at finish

## Limits stated up front
- 16 instrumented forwards, one run, one node (a1), traced (perturbation possible
  per README); conclusions bound THIS workload class (C128 decode, Medium512/64),
  not prefill-heavy or C256 ceiling, and not other worktrees without applicability
  checks.
- Kernel interval union is neither SM utilization nor bandwidth; summed per-kernel
  durations are a census, not wall time; overlap is not automatically removable.
- No engine change, no evaluator/corpus/seed/protocol touch; packet originals
  preserved read-only; derived outputs only under g3-t9/.
- Precision lane untouched: fp8-KV quality stays NOT_CLOSED_QUALITY_BUDGET pending
  the controller's response to t8's contract registration.

## Budget
60 turns / 20 min / USD 5 nominal. Checkpoint after plan.md; final quarter
reserved for tests, proposal json and final commit.

## OUTCOME (executed)

H9 REFUTED with measured evidence; trial complete as a bounded CPU iteration on
real controller data (no engine change, no gate). Script
g3-t9/trace_bubble_census.py ran under python3 -I (exit 0); raw preserved at
g3-t9/raw/trace_bubble_census_output.json; full record in g3-t9/VERIFICATION.md.

- Gate: all 7 packet anchors reproduced exactly (span 571927.153us, 8560
  kernels, kernel union 565549.025us, CPU step union 28303.983us, EventSynchronize
  472169.159/469530.690us, largest gap 34676.268/34417.705us); 16/16 External-id
  pairing; interval-machinery fixtures (one genuine fixture-semantics bug caught
  by the assert before any data claim); planted 300us gap recovered exactly and
  all-overlapped control returned exactly 0.
- H9 verdict: device busy 98.9164%; host-union-not-hidden 5919.8us = 1.0351% of
  wall (< 2% threshold); hard sync bubbles 0.4872%. Exact-runtime/SWA-host-sync
  lane on a1 closed with measured kernel intervals (upgrades t7's static
  closure; consistent with F5 a2 no-op 1.019 and G2-t2 +0.19% marginal).
- New anatomy: decode step = attention 47.04% (16.68ms/step, 28x595us
  BatchPrefillWithPagedKVCache tensor-core decode; zero BatchDecode kernels) +
  GEMM 47.83% + other 5.14%; kernels serialized (sum~union) so shares are wall
  shares; host perfectly pipelined (~10us/step not hidden after step 0).
- E2 cross-check: attention halving predicts 8.34ms/step fp8-KV saving vs E2's
  measured 8-9ms/step — kernel-level confirmation of the fp8-KV mechanism.
- Decision rule honored: ceiling < 2% => no candidate proposal, no Fast slot;
  next-direction seeding recorded (only precision-class levers have ~47% walls
  to attack; both gated on quality budget/numeric prerequisites).

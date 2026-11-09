# G3 trial 3 (glm-b) — promotion track on the proven sync-removal lineage

Campaign: continuous-20261109-01, Iteration 3. Date: 2026-11-09.

## Context (i1/i2 + canonical register)
- i1 (b196fd4) added host mirrors for 5 blocking D2H reads on the SWA extend path
  (sites A/B/C, construction-equality proven, device fallbacks kept). Never full-gated.
- i2 fast screen on b9474807 (= b196fd4 + docs), G3 namespace: **1.0933**, GATE_VALID,
  FAST_SCREEN_GAIN_OBSERVED — but `slot_claimed: false` (fast gates only screen) and
  full-gate exact-token for the mirrors remains unconfirmed (i2 correctness UNRESOLVED).
- **2c854e0** (076b006 fast_decode_plan + aa45af5 SWA host-bound; additive,
  interaction 0.995–1.001) is the twice-proven candidate: G2 full gates **1.0920 /
  1.0984** on a2, 10/10 exact-token each. Per canonical F4, G2 gates do NOT transfer
  into G3: candidate×3 must run in G3's own namespace, distinct accepted slots, before
  any promotion gate. F6: zero promotions campaign-wide (structural).
- Fast screen 1.0933 sits INSIDE the 2c854e0 G2 band (1.0920–1.0984): the prefill
  mirrors add ~0 measurable aggregate → i2's decision-tree branch (b) applies:
  promotion track returns to 2c854e0.

## Hypothesis
- H1 (promotion): 2c854e0's tree gated in the G3 namespace reproduces ~1.09 vs the
  campaign baseline with 10/10 exact-token. Three distinct accepted full-gate slots
  (min-of-3 reported) then unblock the promotion gate request for the whole
  sync-removal lineage as one ALGORITHM_PRESERVING candidate.
- H0: score outside the band or exactness fails → G2 numbers were namespace-bound or
  the environment moved; record and halt the promotion track.

## Negative finding recorded this trial (bounded by fast screen, not full-gate-closed)
Prefill-side blocking D2H syncs are bounded at ~0 aggregate effect on the frozen
workload: b9474807 fast screen 1.0933 ∈ 2c854e0 G2 full-gate band [1.0920, 1.0984].
Consistent with the overlap scheduler hiding extend-batch syncs. The construction-exact
mirror patch is preserved at b196fd4 (reverted from the candidate line this trial).

## Steps
1. Verify history: 2c854e0 (candidate), b196fd4 (prefill mirrors), b9474807 (i2 head;
   evidence shows the controller gates the SESSION HEAD commit, so HEAD must carry the
   candidate tree).
2. Restore the engine tree to bit-identical-to-2c854e0 (checkout python/ from
   2c854e0); `git diff 2c854e0 HEAD -- python/` must be empty. b196fd4 stays reachable.
3. py_compile the touched file; record the empty diff as evidence.
4. iteration-proposal.json: request FULL trial gate, slot #1 of candidate×3, on the
   HEAD commit whose engine == 2c854e0 (canonical sha
   2c854e0bfeff1e2f1424ca1560fbb3d90c3ea8fa; full 40-char shas only — abbreviated
   shas caused GATE_INVALID provenance flags). request_fast_gate=false. Exact-token
   10/10 mandatory.
5. Commit; final quarter of budget reserved for tests/proposal/final commit.

## Out of scope (reasons)
- New decode candidates: roofline-closed (C256 GPU busy 99.2%, attention 242–270 GB/s
  at/above the ~225 GB/s copy ceiling).
- Precision changes: BF16 objective frozen; F2/F4 NOT_CLOSED_QUALITY_BUDGET.
- NGRAM fault-line isolation / MMVQ gate / fp8+SWA ×3: other workers' open lanes
  (canonical 下一轮 list); not reproducible from this sandbox.
- torch.compile eager decode: dead at high concurrency (dispatch fully overlapped).
- Prefill-mirror re-proposal: only if a future trace shows extend-batch sync bubbles.

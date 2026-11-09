# G3 trial 3 (glm-a) — exact-BF16 decomposition candidate (fp8-KV remap default-off)

Campaign continuous-20261109-01, Iteration 3. Base: HEAD 999f93e (= fp8-KV remap
e463d72 + SWA decode hunk 83e7b75 + SWA extend mirrors e063d72/2f3a916; fast gate
on 699c332 scored 1.1361 GATE_VALID/PASS on a1). Direction per brief: revisit
SWA/FP8+SWA history; precision quality unresolved → prefer exact runtime
alternatives. No GPU/SSH here; controller gates. Status of my trial-2 requests
(fast gate on 2f3a916, candidate×3, optional C1 probe) is unknown to me — no
receipt reached this session; this trial's request supersedes, not repeats, them.

## Bottleneck (pipeline, not engine)

The fp8+SWA candidate×3 (F4 closure) is blocked upstream: numeric verification is
INCONCLUSIVE (numeric-coverage-001: probes collected 64 output IDs + chosen
logprobs but top_logprobs null), and fp8-KV stays COMPUTE_DIFFERS /
NOT_CLOSED_QUALITY_BUDGET (F2). Re-requesting it verbatim is not progress.

Meanwhile a1 has NEVER measured the exact (ALGORITHM_PRESERVING) sync-removal
lineage alone: every a1 gate (1.1273, 1.1361) ran with the fp8 remap active, so
the exact hunks' a1 contribution is unattributed. F5 says a2 measured the exact
lineage at ×1.0543/×1.0411/×1.0920 standalone, but on a1 the SWA hunk's marginal
OVER fp8-KV was only +0.19% min (G2 t2) — either a1 hides these syncs, or fp8's
shorter steps un-hide them. Unknown. That is this trial's testable question.

## Hypothesis

H1: env-gating the fp8-KV remap OFF (default off) yields an exact BF16 engine —
delta vs campaign baseline = metadata-only sync-removal hunks (decode 83e7b75:
host kv_indices bound; extend e063d72+2f3a916: sites A/B/C/D host mirrors, all
with device-read fallbacks; construction-exactness proofs in trial-2 plan.md).
Its fast gate on a1 decomposes 1.1361 into exact-part + fp8-part:
  - score ≥ ~1.04, integrity clean → exact lineage transfers to a1; a1 gets a
    promotion-track candidate that needs NO quality acceptance contract; next
    trials run candidate×3 on this head (3 distinct accepted slots, min-of-3).
  - score ~1.00-1.02 → precise negative: a1's overlap scheduler hides decode/extend
    plan syncs at BF16 step lengths; a1's exact lane is bounded; 1.13-class gains
    on a1 are precision-owned. fp8 lane then remains a1's only lever, blocked on
    the quality contract — recorded, not re-requested.
No EXACT claim without evidence; for a BF16 engine the gate's 10-point integrity
check is the exactness screen (glm-b's equivalent lineage passed 10/10 twice on
a2; metadata-only class).

## Change

Single file `python/sglang/srt/arg_groups/overrides.py`, `_muse_spark_overrides`:
the `kv_cache_dtype == "auto"` → fp8_e4m3 default now fires only when
`SGLANG_SPARK_FP8_KV=1`; default logs the disabled state and leaves "auto"
(→ bfloat16, same resolution as the pre-remap baseline edecc81 that gated 1.0021).
fp8 lineage stays reachable verbatim at 2f3a916/999f93e or via the env var — no
history rewritten, F4 artifact intact.

## Worklog

- Read git log + trial-1/2 plan.md and iteration-proposal.json (above).
- Located the single fp8 activation point: overrides.py:318-320 (remap commit
  e463d72; E2 boot evidence: resolved args showed fp8_e4m3 + warning when firing).
- Edit + py_compile + grep-verify no other Muse-Spark fp8 path (only
  overrides.py:319-320 matched Muse+fp8).
- Checkpoint commit, then iteration-proposal.json + final commit.

## Explicitly not requested

Promotion claims; any precision-quality closure; NGRAM/MMVQ lanes (other
workers' entries); triton attention / graph-coverage>64 / Q8_0 GGUF re-gates
(binding negatives); a verbatim repeat of trial-2's blocked candidate×3 ask.

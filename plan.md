# G3 trial-5 plan (worker glm-a)

Direction (from brief): revisit SWA and FP8+SWA history; precision quality remains
unresolved; prefer exact runtime alternatives.

## Facts I start from (canonical / branch learning)
- FP8-KV is COMPUTE_DIFFERS (F2) and its quality budget is NOT_CLOSED; i4 (`5a20347b`)
  fixed the real post-hoc remap site (arg_groups/overrides.py:1157) so KV is actually
  BF16 now; its Fast x1.0017 proves the earlier x1.13 was precision, not metadata.
- SWA host-bound hunk (G2-t2, aa45af5 lineage) is compute-exact on a1 (C1 probes:
  agree=1.0, meanDlogprob=0.0 on 5/5 prompts) but bought only +0.5% mean over fp8-KV
  on a1; standalone on a2 it was x1.0543 (F5). F5 also lists a second arm,
  fast_decode_plan (a2 x1.0411), and the pair x1.0920.

## Decision procedure
1. Inspect git history + tree: is HEAD on the i4 BF16-clean lineage? Is the SWA
   host-sync hunk already present at HEAD? Is any fp8 remap still live?
2. Branch on findings:
   - If SWA hunk NOT in the BF16 tree: re-apply/adapt it as the pure-BF16 exact
     candidate; request frozen BF16 Fast screen (request_fast_gate=true).
   - If SWA hunk already in tree (then i4's x1.0017 already measured it ~neutral on
     a1 BF16): pivot to the second exact arm from F5 (fast_decode_plan) or to a
     bounded CPU falsifier; do not spend a gate re-measuring a known-neutral tree.
3. Verify: no kv_cache_dtype remap can fire (grep both sites), python syntax check,
   bounded CPU/ast verification where the change is provably control-flow-only.
4. Write iteration-proposal.json (string fields + request_fast_gate), commit early
   checkpoint, final commit in the reserved last quarter.

## Budget: 45 turns / 20 min / $3 nominal; last quarter reserved for tests + proposal + commits.

## Outcome (executed, turn ~19)
Decision branch 2 fired: the SWA host-sync family is ALREADY fully in-tree
(83e7b75 decode kv bound, e063d72 ragged-extend sum, 2f3a916 mask mirrors,
plus fast_decode_plan/fast_prefill_plan infrastructure), and i4's Fast x1.0017
measured the whole exact-runtime family ~= neutral on a1 BF16. Another member
of this family was therefore a dead hypothesis; no engine change made.

Executed deliverable instead: `g3-t5/hotpath_sync_audit.py` (stdlib ast audit,
run with -I; raw output `g3-t5/raw/sync_audit.json`). Findings, all line-anchored
against the real source:
1. Q3 mechanically proven: `begin_forward = partial(fast_*)` exists at exactly 4
   sites (L846 decode both wrappers incl. SWA; L860/886/902 fast_prefill_plan for
   draft_extend_v2 / DFLASH verify / EAGLE verify, each additionally guarded by
   `prefill_backend == 'fa2' and dispatch_reason is None`). NO install ever
   targets `full_cg_prefill_wrappers` => every plain-EXTEND CUDA-graph replay
   calls the real flashinfer plan() (blocking D2H) per step, and SWA models are
   excluded from fast_prefill_plan even on the spec paths.
2. Remaining hot-path device reads are fallback branches taken only when host
   mirrors are absent: prefill updater L1969/2082/2116/2124 `.sum().item()` and
   `_build_swa_prefix_custom_mask` L2186-2188 `.detach().cpu().tolist()`.
   The host-mirror paths (added by e063d72/2f3a916) cover ragged+host_lens_mirror
   +prefix_is_full_seq cases; whether the fallbacks execute under the frozen Fast
   config depends on seq_lens_cpu/extend_prefix_lens_cpu propagation, which is
   server-log territory (controller-side) — flagged as open uncertainty.
3. Lane verdict: exact-runtime SWA/host-sync work on a1 is CLOSED as a throughput
   lever (measured neutral); the enumerated residual syncs are prefill-side and
   only worth pursuing if a future trace shows prefill step-boundary stalls.
   The remaining decode gap (~above-roofline at C256) is NOT in this family and
   needs GPU profiling evidence before any further hypothesis.

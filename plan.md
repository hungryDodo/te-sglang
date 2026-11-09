# G3 trial 6 — deepseek-a: NGRAM accept-path correctness (diagnose / fix)

## Context

Canonical F1: NGRAM self-speculation is **BUG_CONFIRMED** (G2-E1). Throughput
+21.6% is real but greedy t=0 output diverges (12/48 comparable passes; ON
pass-nondeterministic 6/24, OFF 0/24; committed token 1.0–3.9 nat behind a
replay-stable argmax). Promotion is forbidden until an ALGORITHM_PRESERVING fix
is demonstrated.

Trial-3 source audit narrowed the fault to two live candidates:

- **A** — verify-row context misalignment: the flat `target_predict` indexed by
  `last_accepted_retrive_idx` (eagle_utils.py:630-655) does not correspond to
  the tree node the draft token came from, i.e. logits argmax is computed under
  the wrong prefix (mask/positions/BFS order vs flat index).
- **B** — post-commit KV/bonus misalignment (`move_accept_tokens_to_target_kvcache`
  / `out_cache_loc`/`accept_index`), so the committed context is polluted.

C4 adds: `retrieve_index` identity does **not** exclude accepted rows `[0,2,4]`
needing compaction to `[0,1,2]`.

## Plan (bounded, ~20 min, CPU only)

1. Read the pinned NGRAM accept path: `ngram_worker.py` (`_prepare_draft_tokens`,
   accept/commit), `eagle_utils.py` (tree mask/position/target_predict),
   `spec_utils.py` (KV move), `ngram_info.py` (tree construction).
2. Build a **pure-CPU, deterministic topology oracle** that replicates the
   exact index arithmetic from the pinned source (extract expressions where
   possible) for a small BFS tree, and checks the invariant:
   for every accepted draft token, its verify row's logits must be computed
   under the prefix that ends at that draft token's parent; the bonus token
   taken from `target_predict[last_accepted_retrive_idx]` must be the argmax
   continuation of the *committed* context.
3. Falsifier: if the oracle finds a concrete (tree, accept-path) where the row
   used for the bonus/commit is NOT the row whose prefix equals the committed
   prefix, that is candidate A confirmed with an exact mechanism; if instead
   the tree arithmetic is causal for all reachable accept paths, A is refuted
   and the remaining blame shifts to B (KV move), which this trial will state
   as the precise open question rather than re-sweeping the whole source.
4. Preserve raw oracle output + the extracted source anchors under
   `reports/rsi/` (or a dedicated trial dir). No GPU/SSH; do not claim live
   measurement.
5. Write `iteration-proposal.json` with string fields
   `bottleneck,hypothesis,patch,tests,learning,next_proposal` and boolean
   `request_fast_gate`, commit.

## Decision rules

- **Kill rule**: if the CPU oracle cannot be tied to the pinned source
  expressions (anchors drift / semantics are not faithful), record
  INCONCLUSIVE and do not patch.
- Any patch must be a *correctness* patch (ALGORITHM_PRESERVING direction), not
  a speed patch; no promotion claim is made this trial.
- `request_fast_gate=false` unless a genuinely BF16-applicable, correctness-
  clean candidate exists. NGRAM must not be promoted while unresolved.
- Negative findings are a valid iteration; no invented measurements.

## Constraints

- Frozen workload/evaluator/corpus/seed/reference untouched.
- No precision change; BF16 frozen. Quality budgets remain NOT_CLOSED where
  undefined.
- Provider identity claims remain unverified without gateway evidence.

---

## RESULTS (executed, CPU-only, this session)

Prior branch state: t3 mask-contract audit + `_sanitize_tree_mask` no-op guard;
t4 topology/accept-index/KV-mover oracle (compaction excluded given a valid
root); t5 single-request mask-width oracle (latent h_i != d_i hazard). This
session does not repeat them.

New component: **batched flat-mask block alignment** on the FA2 CUDA-graph
verify path — builder uses host per-request widths, packer uses
device-cumulative offsets.

- Artifact: `ngram_oracle/verify_window_oracle.py`; raw output
  `ngram_oracle/results/verify_window_oracle.json`; write-up
  `ngram_oracle/README-verify-window.md`. `python3 -I`, stdlib only, exit 0.
- Ported: `ngram_worker.py:352-361` build; `ngram_info.py:76-117`
  device-derived `mask_numel` + global True padding;
  `flashinfer_backend.py:263-285` packed_indptr capacity-only assert;
  `pack_verify_mask.py:12-24` `kv_start*Q` block read.
- Result: aligned 32/32, 36/36, 40/40 bits exact. Mid-batch drift (req1
  d=6/h=5): req0 exact, req1 18/40 bits wrong, **req2 16/40 bits wrong while
  req2 is itself aligned** — cross-request contamination. First-request drift:
  all three requests wrong (12/28, 16/36, 16/40). No exception anywhere.
- Interpretation: exactness requires `h_i == d_i` for every request; a single
  drift shifts all later blocks in the same flat buffer and the True padding
  hides the size mismatch. Reachability of drift is **NOT established**
  (FutureMap audit argues equality), so this is a sharper latent hazard +
  falsifier, not the F1 root cause and not a fix.

No engine change; no GPU/fast gate (`request_fast_gate=false`); no throughput,
quality, EXACT or promotion claim. NGRAM remains non-promotable until an
ALGORITHM_PRESERVING adjudication on a fixed commit.

# G3 trial 7 — deepseek-a — NGRAM accept-path bounded CPU oracle (with self-diagonal)

## Context (canonical)
- **F1**: NGRAM self-speculation throughput +21.6% real, but greedy correctness NOT preserved
  (divergence set == ON nondeterminism set; committed tokens 1.0–3.9 nat behind replay argmax).
  **No promotion** until fixed and re-adjudicated ALGORITHM_PRESERVING.
- Previous a3/i3, i4, i5 results are **partial**: i4 model was correct only on its own
  assumptions; i5 mask model (and its expected rows) **omitted the self diagonal**, while real
  `fillResult` sets `mask[i*n+i]=1`. So i5's 12/12 does NOT certify real mask assembly.
- Next-task directive (completion-and-controller-capabilities-001 /
  ngram-mask-width-review-005): a **bounded fixed CPU topology/accept-index oracle** that
  *includes the current token + ancestors*, compares **exact source conventions**, preserves
  fault controls, and states its synthetic scope. Not another whole-source sweep.

## Objective (this trial, CPU only; no GPU/SSH)
Build and *execute* one bounded oracle that models the real NGRAM verify/accept path from
extracted source expressions, including the self diagonal, and determine whether a concrete
off-by-one / indexing defect explains the F1 divergence signature (first-diff at positions
4 and 6, ON nondeterminism, wrong committed token reported as its own top-1).

Sub-questions, ranked:
1. **Q1 (accept-index semantics).** In the flat verify layout
   `target_predict = argmax(logits).reshape(bs, draft_token_num)`, does the C++ accept loop
   compare the *child* draft token against the **parent** row's argmax, or against its own
   row? Model both variants over the exact BFS/`retrieve_index` topology produced by
   `ngram_info`/`spec_tree` for CUDA, with the self-diagonal mask semantics of real
   `fillResult`. Which variant reproduces greedy on a synthetic controlled corpus, and which
   reproduces the observed divergence pattern?
2. **Q2 (bonus row).** `bonus = target_predict[last_accepted_retrive_idx]` — is the bonus the
   argmax of the last accepted row (correct: next-token distribution after accepted prefix)
   or off by one under any accept-length (esp. accept_len == 1, the untested discriminator
   from trial 3)?
3. **Q3 (commit slice).** `predict[accept_index]` commit set: does it include exactly the
   accepted draft tokens plus the bonus, in tree order, and is it consistent with the KV
   compaction rows? Fault controls: empty accept, full accept, single-node accept.

## Method
- Extract the **real source expressions** with `ast` (like glm-b's successful pattern) from
  `ngram_info.py`, `spec_tree.py`, `eagle_utils.py`, `ngram_worker.py`; assert line anchors.
- Build a pure-stdlib oracle with an explicit, documented mask including `mask[i][i] = 1`
  (self), ancestors, and root handling; compare two independent mask conventions where the
  source is ambiguous.
- Ground truth = greedy reference decode on the same token distribution (synthetic
  deterministic logits with known repeats, plus fault-injected logits).
- Fault controls that must FAIL the oracle (sensitivity): missing self diagonal, ancestor
  omitted, off-by-one accept index, bonus taken from wrong row.
- Preserve raw JSON output under `g3-t7/raw/`.
- State limits: synthetic CPU model, not a live CUDA/overlap reproduction.

## Deliverables
- `g3-t7/topology_accept_oracle.py` (+ raw output).
- `reports/rsi/g3-t7-ngram-accept-oracle.md` — what is falsified, what remains open.
- `iteration-proposal.json` with bottleneck/hypothesis/patch/tests/learning/next_proposal
  and `request_fast_gate` (false unless a BF16-applicable engine candidate is justified —
  NGRAM must NOT be promoted while unresolved).
- Git commit(s), early recoverable checkpoint.

## Budget / stop rule
45 turns, 20 min, USD 3. Reserve final quarter for proposal JSON + tests + final commit.
A precise executed negative finding is a valid Iteration.

## RESULT (executed, 2026-11-09)
- Oracle `g3-t7/accept_oracle.py` exit 0; raw `g3-t7/raw/accept_oracle.json`.
- Clean scenarios all exact (bs1 D8, bs3 mixed trees, branch tops, D12); sanitize drops 0 bits; KV mapping exact.
- Faults diverge: no_self, wrong_root, bonus_from_parent. ancestor_dropped is rejection-only => exact (new narrowing: under-attention cannot explain F1).
- No engine change; request_fast_gate=false; NGRAM remains non-promotable.

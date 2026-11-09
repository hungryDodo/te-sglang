# G3 trial 8 — deepseek-a — NGRAM topology oracle: real-source conventions + independent golden

## Objective
Bounded CPU falsifier for the NGRAM greedy-correctness bug (F1), following controller
proposal queue a3: verify the tree-attention topology oracle against the REAL
implementation and the root/pending-token convention, with an independent golden that
does not share code with the model under test.

## Context (canonical)
- F1: NGRAM self-speculation BUG_CONFIRMED. Throughput +21.6% real, greedy correctness
  broken; must not be promoted until fixed and re-judged ALGORITHM_PRESERVING.
- Trial 3: 12/48 ON/OFF divergences, first-diff positions {4,6}; ON pass-nondeterministic
  6/24, OFF 0/24; divergence set == ON nondeterminism set. Two live candidates: (A)
  verify-row context misalignment, (B) post-commit KV/bonus misalignment.
- C8/i5 and C9/i6 mask oracles were NOT qualified: missing self-diagonal in both model
  and expected; reversed ancestry direction (root saw future child); builder and expected
  generated through the same faulty helper. i6 golden must be [[1,0],[1,1]] for a 2-node chain.

## Hypothesis for this trial
H-T8: an independently-defined topology oracle (query i may attend to column j iff j is
an ancestor-or-self of i), executed on the tree the real NGRAM code actually generates,
will either (a) reproduce the real mask convention including root/pending-token handling,
giving a usable invariant for the accept path, or (b) expose a concrete disagreement
between real source conventions and the semantics assumed by the diverging rows.
Any expected mask is derived from the DEFINITION, never from the builder under test.

## Approach
1. Locate the real mask/accept code paths (fillResult kernel, tree/ngram builders).
2. Extract the real source expressions via `ast` and execute them on CPU with minimal
   shims (torch-free) — the AST-execution pattern proven on this tree.
3. Independent oracle: hand-derived semantics; golden [[1,0],[1,1]] on a 2-node chain,
   root row = diagonal only; brute-force ancestry from the parent array.
4. Compare on real NGRAM tree shape and on the root/pending convention; preserve raw JSON.
5. Commit evidence + iteration-proposal.json.

## Verification / deliverables
- Executed CPU oracle with raw JSON output committed; clear pass/fail vs independent golden.
- Precise statement of what this does and does not prove about the live GPU bug.
- No promotion claim, no measured throughput claim, no GPU/SSH attempts.

# G3 Trial 5 — deepseek-a — NGRAM accept-path bounded CPU oracle

Status: IN_PROGRESS. Base: session HEAD (unchanged 47030d8 lineage). No claim of measurement.

## Canonical constraints carried in

- F1: NGRAM self-speculation is BUG_CONFIRMED at greedy t=0 (12/48 divergent instance-passes,
  divergence set == ON pass-nondeterminism set, ON committed tokens 1.0–3.9 nats behind
  replay-stable argmax). **No promotion of NGRAM is possible** until a fix passes an
  ALGORITHM_PRESERVING adjudication.
- Branch i3 (qualified negative): normal `fillResult` ancestor-copy + diagonal construction is
  causal for the H1 family; a "sanitize guard" is NOT an established fix.
- Branch i3 leaves open: the **KV mover is NOT excluded**. Accepted draft rows can be
  non-contiguous (e.g. [0,2,4]) even when `retrieve_index` is identity, which requires
  compaction to [0,1,2]. H2–H4, overlap behavior and the original NGRAM bug remain open.
- Controller direction for a3: *bounded fixed CPU topology/accept-index oracle with real raw
  results*, not another whole-source sweep.

## This trial's question (one falsifiable question)

**Q:** Does the pinned NGRAM accept→commit→KV-move bookkeeping preserve per-row token/KV
identity for *non-contiguous* accepted draft rows under a fixed draft-tree topology?

Two mechanisms are distinguishable:
- **A (verify-row context)**: `target_predict = argmax(logits).reshape(bs, draft_token_num)`
  indexed by the flat retrieve index → argmax read from a row whose context differs from the
  row the draft token came from.
- **B (post-commit KV placement)**: the accepted rows' KV is written to destination slots that
  do not correspond to the committed token order (compaction bug), so the next step's context
  is polluted and the wrong token becomes self-consistent.

The i3 note ("accepted rows [0,2,4] require compaction to [0,1,2]") is the concrete B-shaped
falsifier. A CPU oracle can decide *whether the pinned code performs exactly that compaction*
without a GPU.

## Method (bounded, executable here)

1. Read the pinned implementation of:
   - `ngram_worker.py` accept/commit block (`predict[accept_index]`, `retrieve_index` use),
   - `spec_utils.py` `move_accept_tokens_to_target_kvcache` (destination-slot write),
   - `eagle_utils.py` `target_predict` construction and `accept_index`/`retrieve_index` fill,
   - `ngram_info.py` tree construction (`draft_token_num` depth / parents).
2. Build a **standalone CPU oracle** that takes a fixed draft tree (parent array, BFS order) and
   a fixed accept path, derives `accept_index`/`retrieve_index` exactly as the pinned code does,
   and checks the destination mapping of the KV move against the required compacted order.
   Prefer executing the *real* pinned functions with tiny `torch` CPU tensors if torch is
   importable; otherwise replicate their index arithmetic line-for-line and keep the
   correspondence marked in comments. Either way the raw output is preserved as JSON.
3. Enumerate all valid accept paths for small fixed trees (single chain, one branch, two
   branches) and report every case where destination != expected compaction, plus cases where
   consecutive-step context would be polluted.
4. If (and only if) a concrete bookkeeping defect is found, implement the minimal fix in the
   source overlay and state the exact invariant it restores. Do not claim correctness without a
   controller-run adjudication.

## Deliverables

- `evidence/ngram-accept-oracle/` : `oracle.py`, `raw-output.json`, `README.md` with commands run
  and interpretation limits.
- Source patch only if the oracle identifies a concrete defect (otherwise: negative finding).
- `iteration-proposal.json` (required string fields + `request_fast_gate` boolean) and a final
  commit. The proposal must state explicitly what is NOT claimed (no NGRAM promotion, no
  EXACT precision, no performance number from this CPU oracle).

## Kill / stop rules

- Stop the oracle line if the pinned functions cannot be executed or faithfully replicated
  within ~8 minutes of work; record the limitation instead of inventing results.
- Never report a GPU/throughput/probe number. Every number in the evidence must come from an
  actual command run in this sandbox.
- Reserve the final quarter of turns for tests, proposal JSON and commit.

---

## RESULTS (executed, CPU-only, this session)

Prior sessions on this branch: t3 = mask-contract audit + `_sanitize_tree_mask`
no-op guard; t4 = topology/accept-index oracle (KV compaction excluded *given a
valid root*; negative controls fire). This session does **not** repeat either.

New component tested: the **verify attention-mask assembly vs KV read width**
(the unmodeled half of the verify input — t4 took per-row logits as given).

- Artifact: `ngram_oracle/mask_width_oracle.py`; raw output
  `ngram_oracle/results/mask_width_oracle.json`; write-up
  `ngram_oracle/README-mask-width.md`. `python3 -I`, stdlib only, exit 0.
- Ported width arithmetic: `ngram_worker.py:352-361`
  (`ones_masks` width = host `seq_lens_cpu[i]`) vs `ngram_info.py:60-117`
  (`paged_kernel_lens + D` with device `seq_lens`, `mask_numel` formula,
  silent append-True branch).
- Result: consistent host/device 12/12 exact on two tree shapes; legitimate
  CUDA-graph batch padding 12/12 exact on real rows; host-lags-device by 1 or
  2 tokens -> **0/12** real rows exact (extra draft column attended, pending
  column missed); host-ahead by 1 -> 2/12. No exception in any case.

Findings:

1. The mask assembly is exact **iff** host prefix used for the prefix block
   equals device `paged_kernel_lens`; under the documented FutureMap equality
   this holds and the contract is sound.
2. The `numel`-based recovery branch cannot distinguish legitimate graph
   padding from a host/device width drift: a one-token drift is silently
   absorbed as an all-True tail that is consumed as real mask columns.
3. Reachability of a drift is **NOT established** (FutureMap audit argues
   against it; first-diff positions 4/6 also fit poorly). This is a latent
   hazard + a falsifier (log-only host/device width-delta counter inside the
   existing batch-pinned probe), not a root-cause claim and not a fix.

Not measured: CUDA kernels, `pack_read_stream` / `assign_extend_cache_locs`
physical slot plan, target-model numerics. No throughput, quality or EXACT
claim; no NGRAM promotion. `request_fast_gate=false` — this trial ships no
engine change to measure, and a CPU negative is the deliverable.


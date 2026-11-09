# G3 trial 9 — worker deepseek-b (node a4)

## Contract reminder
- Objective: aggregate output tokens/s under the frozen workload; the candidate is source code,
  the trusted controller alone runs official gates/measurements. No GPU/SSH here.
- Budget: 45 turns, 20 min, USD 3 nominal. Keep the final quarter for tests, proposal JSON, commit.
- A source change is never its own acceptance. Precision changes need an authenticated numeric
  quality contract; undefined budgets stay NOT_CLOSED_QUALITY_BUDGET.

## Step 0 — F3 (MMVQ threshold `f263262`) applicability check (brief mandate)
- Canonical F3 says `f263262` is UNGATED and must be evaluated before interpreting any gain.
- Controller analysis `applicability-001` already established: the frozen formal Fast gate loads
  BF16 safetensors and never dispatches GGUF/MMVQ, so a BF16 score cannot close F3.
- This trial **records that structurally** (no BF16 number is offered as F3 evidence) and does
  **not** re-request a GGUF GPU experiment: the controller has declared GGUF custom-GPU work
  pending and no matching raw evidence has been supplied. No source change is made for F3.

## Step 1 — Primary question (assigned proposal-queue lane for a4)
Queue item: "a4 核对 host-mirror oracle/分支 guard 并等待筛选". C11 says deepseek-b/i6 was
partial: its reachability model omitted the outer `spec_info` / `kv_indptr` conditions, integer
truthiness, and the shared-sum nature of the mirror.

Plan: build a corrected, bounded **CPU oracle over the real source expressions** of the
host-mirror branch (the `global_override_indptr_cpu` / host-indptr path used by the eager
`fast_decode_plan` lineage). Enumerate a small input space, evaluate (a) the branch guard exactly
as the source evaluates it, and (b) an independent reference for whether the host mirror really
equals the device indptr. Falsifiable outcome: either the guard is a sufficient condition
(no disagreement over the enumerated space) or a concrete counterexample exists. Preserve raw
JSON and state limits (CPU simulation only; no CUDA/page/overlap kernel is executed).

## Step 2 — Engine candidate
- Only if Step 1 leaves budget and a concrete, semantics-preserving BF16 gap is found.
- Do not re-gate known-neutral sync patches; do not apply the sliding-window prefill change
  (semantics change; needs a >512-token token-exact parity harness we cannot run here).

## Step 3 — Deliverables
- plan.md (this file), evidence + raw output under `seis/trials/deepseek-b/g3-t9/`,
  `iteration-proposal.json`, local git commit(s). Proposed test request recorded in the JSON.

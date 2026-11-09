# G3 trial-7 plan — glm-a (campaign continuous-20261109-01)

Direction: revisit SWA/FP8+SWA history; prefer exact runtime alternatives. Status of
that history on a1: the SWA/host-sync exact family is CLOSED both ways (t5 static
enumeration of residual syncs + fast-plan gaps; t6 executed guard table proving the
prefill mirrors are ENGAGED on the frozen path, so i4's x1.0017 measured the family
fully engaged). fp8-KV x1.13 stays blocked (NOT_CLOSED_QUALITY_BUDGET, missing top-k).
Decode side is hypothesis-starved pending the t6 trace request.

## Open surface this trial (new, not a repeat)
t5's audit covered ONE file: `layers/attention/flashinfer_backend.py`. The rest of the
per-decode-step host path was never audited for blocking device reads:
- `managers/scheduler.py` (event loop, process_batch_result_*)
- `managers/schedule_batch.py` (update/filter/prepare_for_*)
- sampling: `managers/sampling/*` (logits processor, sampler, batch info)
- `mem_cache/memory_pool.py` (per-step alloc/free)
- `model_executor/model_runner.py` (forward-time work outside attention)

## Hypothesis
H1_t7: there exist per-step blocking device reads (.item/.cpu/.tolist/.nonzero/
synchronize) on the frozen decode path OUTSIDE the attention backend that are NOT
hidden by the overlap scheduler — i.e. they sit on the kernel-submission critical
path (inside forward/sampling) or serialize the copy queue — and are exactly
removable via the established host-mirror pattern (83e7b75/e063d72/2f3a916 style).
  - If CONFIRMED with an engaged, semantics-identical site: patch it exactly,
    verify (syntax + resolution check + guard execution), request Fast screen.
  - If the only hot reads are scheduler-thread reads hidden by overlap (or cold
    paths): executed NEGATIVE closure — the exact-runtime lane stays closed with
    the last surface audited, and the t6 trace request becomes airtight.

Criticality model (stated up front, checked against source): overlap scheduling is
ON in the frozen launch (no --disable-overlap-schedule). A device read only costs
wall time if it (a) runs between kernel launches inside the forward/sampling
critical path, or (b) blocks the copy/event queue the GPU waits on. Scheduler-thread
reads one step behind are hidden. The audit must classify THIS, not just count
calls.

## Method (proven t5/t6 pattern)
1. ast-scan the target files for blocking calls; record file:line, enclosing
   class/function, receiver expression.
2. Classify hot (per-step: decode update/result processing/alloc/sample) vs cold
   (init/capture/error/finish-cleanup-once).
3. For each hot site: receiver host-mirror provenance (*_cpu, pinned buffers,
   python ints/lists) vs device tensor -> real sync.
4. For real syncs: call-path criticality — inside model_runner forward chain
   (critical) vs scheduler loop / process_batch_result (overlap-hidden, one step
   behind) — verified by reading the actual call sites, not name guesses.
5. Raw JSON to g3-t7/raw/; negative controls included (known sync sites must be
   rediscovered by the scanner, else the scanner is broken).

## Deliverables
- g3-t7/host_path_sync_audit.py + g3-t7/raw/*.json (executed, python3 -I)
- optional exact patch ONLY if H1 confirms an engaged critical-path site
- plan.md outcome section, iteration-proposal.json (bottleneck/hypothesis/patch/
  tests/learning/next_proposal + request_fast_gate), commits
- No EXACT claims without evidence; no precision change; no protocol touches;
  no duplicate of glm-b's probe-recipe lane or the blocked candidate-x3 request.

## Budget
45 turns / 20 min / $3. Checkpoint commit immediately after this file. Final
quarter reserved for tests + proposal + final commit.

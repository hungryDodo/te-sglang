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

## OUTCOME (executed 2026-11-09, commit cc0dc45 checkpoint -> final)

**H1_t7 REFUTED — exact-runtime lane on a1 now closed at the static-audit level.**

Executed evidence:
- `g3-t7/host_path_sync_audit.py` exit 0, 55 sites across 7 files, raw at
  `g3-t7/raw/host_path_sync_audit.json` (scanner negative controls passed:
  t5-known memory_pool/scheduler/schedule_batch sites rediscovered).
- `g3-t7/raw/supplementary_grep.txt` (executed grep): spark2_5.py, rotary dir,
  graph runner backends, forward_batch_info, linear, layernorm.
- Engine integrity: `git diff 5a20347 HEAD -- python/` = **0 lines** — HEAD's
  engine is bit-identical to i4's fast-gated tree (x1.0017, SWA family measured
  ENGAGED per t6). Nothing new to screen.

Human-verified classification of every surviving candidate:
1. **Intra-forward (the only class that matters for exactness)**: exactly two
   `torch.cuda.synchronize()` in `LogitsProcessor._get_logits`
   (logits_processor.py:984,1005), both gated behind
   `envs.SGLANG_TRACE_LOGITS_E2E_SYNC` — OFF in the frozen launch. Zero blocking
   device reads in sampler.py, sampling_batch_info.py, spark2_5 forward
   (post-load), linear/layernorm/rotary(text), CUDA-graph replay path
   (base_cuda_graph_backend: 0 hits).
2. **Scheduler-thread per-step**: all read host mirrors already
   (seq_lens_cpu / track_mask_cpu / active_ranks_cpu; schedule_batch.py:3274,
   3388, 3632, 3817, 3819; scheduler.py:4949) — no device sync — or are the
   FutureMap / async-D2H designed resolution machinery (overlap_utils.py:553-639,
   async_d2h_copy_worker.py:90,93) that runs one step behind on the copy
   stream: overlap-hidden BY CONSTRUCTION, and removing them would remove the
   overlap mechanism itself, not a stray sync.
3. **Off-path by frozen config**: mamba (schedule_batch:3665, memory_pool
   Mamba/Hybrid sites), MLA/MXFP8 pools, multimodal mrope (bailing_mrope.py,
   mrope_rope_index.py — spark2_5 uses get_rope, verified), radix-prefix
   nonzero (memory_pool:3337; --disable-radix-cache), TP>1 sync_group
   (forward_batch_info:1970; --tp 1), spec paths.
4. **Boot/cold only**: graph-capture warmup syncs (full:163, breakable:122,
   tc_piecewise:241 — 2-iteration warmup before capture), weight-load .item()s
   (linear.py, spark2_5:504,510), set_embed_and_head (spark2_5:594),
   get_cpu_copy/error paths, _maybe_report_active_ranks (scheduler:4949,
   host tensor).

Correction to the script's own docstring: it claimed async_d2h_copy_worker.py
was "grep-verified 0 blocking calls" — false (2 event/stream synchronize sites,
the designed copy loop). The scanner was right; the docstring claim is wrong and
is corrected here rather than left as a mislabel (glm-b i7 lesson).

Synthesis with prior trials: t5 enumerated attention-backend residual syncs;
t6 proved the SWA prefill mirrors ENGAGED (so i4 x1.0017 measured the family);
trial-1 bounded launch overhead (graph 64->256 = 1.0056); trial-2 showed decode
is memory-bound (TPOT 17.6/26.2/36.1/42.2 ms at bs 32/64/128/256 vs 12.4 ms
weights floor + KV read). With t7 closing the last static surface, the residual
decode gap (~16-24 ms/step above weights+KV floor at high bs) is now provably
NOT findable by static host-path audit: it lives either inside kernels
(flashinfer decode / GEMM efficiency on GB10) or in host gaps only visible at
runtime. The t6 decode-trace request is therefore airtight and is the only
remaining exact/efficiency entry on a1.

Stated limits: audit covers explicit blocking calls
(.item/.cpu/.tolist/.nonzero/.numpy/.synchronize) — implicit syncs (bool(tensor),
host-context comparisons) not covered; runtime behavior not observed (no GPU
here); classification of "overlap-hidden" is by design intent of the overlap
scheduler, not by measurement.

Final anchor (post-commit verification): FutureMap.resolve_seq_lens_cpu is
called from scheduler.py:4667/4676 (event loop), never from the forward chain;
flashinfer_backend.py:2671's comment confirms the design ("resolve_seq_lens_cpu
already supplies the committed lengths") — the FutureMap syncs are the overlap
mechanism's own commit point on the scheduler thread, as classified.

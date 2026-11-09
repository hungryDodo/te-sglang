# G3 trial 10 — worker deepseek-b (node a4)

## Contract reminder
- Objective: aggregate output tokens/s under the frozen workload; the candidate is source code,
  the trusted controller alone runs official gates/measurements. No GPU/SSH here.
- Budget: 45 turns, 20 min, USD 3 nominal. Final quarter reserved for tests/proposal/commit.
- A source change is never its own acceptance. Precision changes need an authenticated quality
  contract; undefined budgets stay NOT_CLOSED_QUALITY_BUDGET.

## State inherited from trials 1–9 (do not redo)
- **F3 `f263262` remains UNGATED and score-unreachable under the frozen BF16 Fast gate**
  (controller `applicability-001`): the gate loads BF16 safetensors, never GGUF/MMVQ.
  g3-t1/t5 executed the selection contract (0 changed dispatch cells at any batch >= 16);
  g3-t8 audited all 14 `paged_kernel_lens_sum` assignment sites. GGUF GPU experiment
  (g3-t2 protocol) is pre-registered but its execution prerequisites are unmet.
- **The only ungated BF16-visible engine delta** is the t3/t4 host-mirror pair
  (`33e5ac4` decode, `a306e40` prefill) removing one `.item()` D2H per eager
  decode/extend step in the SWA `translate_loc_from_full_to_swa` block. G3 t4 requested a
  Fast screen; canonical C3 records i4 as in flight. HEAD (4e84e42) differs from the i4 tree
  only by an env-gated, inert GGUF dispatch tracer (g3-t5) — so a new gate request would
  duplicate i4, not measure a distinct candidate.
- t6/t8 left exactly one attribution question open: **is the patched branch reachable under the
  frozen launch configuration?** (`use_swa_source = use_sliding_window_kv_pool and
  translator.reads_are_translated`; with a translating/unified pool the patched lines are dead.)

## This trial's question (new, bounded, CPU-executable)
**Q: Under the frozen launch args, does the engine build a unified SWA KV pool (reads
translated → t3/t4 host-mirror patch is dead code) or a non-unified `BaseSWAKVPool`
(read translation done by the backend → patch executes)?**

Why it matters: it decides whether an i4 screen result of ~1.00 is *structural* (patch never
executed) or a genuine null measurement. That distinction is required before any attribution
claim, and C3 forbids concluding "the path does not execute" from a throughput number.

Method (bounded, stdlib only):
1. AST-extract the real pool-construction decision in `kv_cache_configurator.py`
   (`configure` → unified vs `_build_hybrid_swa_kv_pool`) and `_resolve_swa_kv_pool` in
   `flashinfer_backend.py`; line-anchor every construct (fail loudly on source drift).
2. Evaluate the extracted conditions under the **frozen launch args** (from the contract:
   no unified-cache flag, no spec/PD flag, kv bf16, flashinfer, model = Spark-X2.6 hybrid SWA).
3. Cross-check the two independent consequences: (a) `KVIndexTranslator.is_translating`
   (allocator class), (b) `FlashInferAttnBackend.use_sliding_window_kv_pool` and
   `num_wrappers == 2`.
4. Negative controls: flip the deciding predicate and show the verdict changes.

Falsifiable outcomes: `UNIFIED → patch unreachable (i4 screen structurally null)` vs
`NON_UNIFIED → patch reachable (i4 screen is a real, if ~0, measurement)` vs
`UNDECIDABLE_FROM_SOURCE` (record which flag/attribute the controller must log at boot).
Raw JSON preserved.

## No engine change by default
No semantics-preserving throughput lever survived trials 1–9 (decode roofline-closed, sync
family measured 1.0018 KEPT_NEUTRAL, MMVQ structurally inert, sliding-window prefill
acceptance-incompatible, prefill batch width flag-pinned at 2x8192). This trial adds evidence,
not code, unless the reachability answer exposes a real defect.

## Outcome (executed)
Verdict **REACHABLE**: no `--enable-unified-memory` in the frozen argv (source default
False) → plain `SWAKVPool` → `is_translating=False` → `_reads_translated(FULL)=False` →
`use_swa_source=False` → the patched `use_sliding_window_kv_pool and not use_swa_source`
block executes. The control with unified memory flips it to dead, so the test is sensitive.
Effect bound: the t4 prefill patch (i4's only delta over i3) removes 2–32 D2H per point
(≤3.2 ms at 100 µs/sync, ≤16 ms at 500 µs/sync), i.e. below Fast resolution; the t3 decode
patch is 20–60× larger (128–2048 syncs/point). Raw: `swa-pool-reachability.json`.
F3 unchanged: UNGATED, score-unreachable under the frozen BF16 gate. No gate requested
(i4 already covers this engine tree; HEAD differs only by the inert env-gated tracer).

## Deliverables
- plan.md (this file); evidence + raw JSON under `seis/trials/deepseek-b/g3-t10/`;
  `iteration-proposal.json`; local git commit(s); controller request stated in the JSON.

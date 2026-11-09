# G3 glm-b trial 12 — Prefill-path device-scalar sync: measured signature + trace request

## Queue disposition (not re-done)
- Queue item "verify g3-t6 scheduler-population" is CLOSED by i7/i8: i7 rejected the
  IndentationError-marked-POPULATED raw (C12); i8 re-executed the whole chain with a hard
  verdict gate (21/21 hops, CHAIN_CLOSED_CPU_SIDE, g3-t8/raw/). Re-running it would repeat a
  known result (stop-repeating-008). This trial picks a new falsifiable question.

## Bottleneck (real anchors, both verified this session)
1. `python/sglang/srt/layers/attention/flashinfer_backend.py:2347` — in the PREFILL (extend)
   metadata updater, `kv_last_index = kv_indptr[-1]` reads a device scalar and feeds it to a
   Python slice → blocking D2H on EVERY extend forward (SWA hybrid pool always active for
   Spark-X2.6 in the frozen BF16 launch). This is the exact pattern trial-2's decode fix
   (aa45af5, lines 1816-1830) eliminated on the decode side; the prefill twin was deferred
   pending trace evidence ("only worth it if prefill syncs show up in a trace").
2. The controller's real C128 decode trace (20261110-decode-trace, glm-a tree 15fd7f4 which
   does NOT contain glm-b's fix) contains **exactly 16 Memcpy DtoH = 1 per decode step** —
   measured signature of precisely this sync class at decode on an unfixed tree.

## Hypothesis (falsifiable)
- H1: the prefill-site sync is real and per-extend-step; a real prefill-window trace on OUR
  tree (which has the decode fix but NOT the prefill fix) must show DtoH count == extend-step
  count, and if it produces >=2% recoverable GPU idle in prefill-dominated windows, a host
  mirror patch is gate-worthy; <2% ⇒ negative closure like it11 (kernel-bound).
- H0 (falsifier): if the updater's kv_indptr is already host-resident at that point (no
  device-scalar read reaches the host as int), no D2H occurs and the site is dead code —
  the source probe below would refute H1's premise.

## Method (executed this session, CPU-only, real data + real source)
1. **DtoH census on the real decode trace** (g3-t12/analyze_sync_sites.py): hard gate =
   recompute controller totals exactly (8560 kernels, 565.549025 ms union) before any slice;
   then enumerate gpu_memcpy by direction, per-step placement (in-span vs inter-step gap),
   bytes, durations, and idle in a bounded neighborhood after each DtoH (sync cost is idle,
   not memcpy duration). Controls: (a) negative — kernel-only view must yield 0 sync sites;
   (b) detector self-test — synthetic injected DtoH must be counted.
2. **Real-source execution probe** (g3-t12/probe_prefill_updater_site.py): execute the REAL
   statements at 2340-2352 (ast-extracted, verbatim source asserted) with MockTensors whose
   __getitem__/sum record device-scalar reads. Expect: prefill branch DOES read kv_indptr[-1]
   (H1 premise confirmed at source level); decode-site control: with seq_lens_cpu matching bs
   the fixed branch reads NOTHING device-side; with seq_lens_cpu=None it falls back to the
   read (guard boundary documented). Mock-only limits stated (no CUDA, no real pool).
3. **Committed custom-GPU request** (g3-t12/prefill_trace_request.json + iteration-proposal
   tests field): exact bounded prefill-window capture on current HEAD via the engine's
   BUILT-IN /start_profile /stop_profile endpoints (http_server.py:1221; NO python/ change —
   tree stays bit-identical to the 2c854e0 promotion lineage). Prefill Heavy shape
   8192/32 C16, warmup then one measured window, pre-registered discriminators and bounds.

## Expected outcomes / decision rule
- Decode census (executed now): quantifies the per-sync cost class at a measured operating
  point; 1 DtoH/step on unfixed glm-a tree ⇒ cross-worker note (glm-a lacks aa45af5).
- Prefill trace (controller-run, pending): DtoH == extend steps ⇒ site confirmed live;
  idle-in-prefill-spans >=2% ⇒ next trial writes the host-mirror patch (paged_kernel_lens_sum
  is a candidate mirror — must verify exact equality incl. prefix trimming before use);
  <2% ⇒ negative closure, no patch.

## Bounds
- No GPU here; trace request is a request, not a measurement. Mock probe ≠ CUDA execution.
- Prefill steps are long; a single sync may be sub-noise at Prefill-Heavy C16 — the request
  targets the extend→decode transition/mixed windows too, where step times are short.
- python/ stays untouched this trial (git diff 2c854e0 HEAD -- python/ empty).

## Budget
- Checkpoint commit after plan+probes scaffold; final quarter reserved for
  iteration-proposal.json + clean-tree commit.

## Outcome (see g3-t12/RESULTS.md)
- EXECUTED 1 (real trace, gated): exactly 1 DtoH/step (1024 B, in-gap,
  ~6 us before next span); post-sync 200us idle 75.8 us vs 18.8 us seeded
  control → ~57 us/step excess on glm-a's unfixed tree (0.16% of step →
  confirms it11's C128 closure; cross-worker: aa45af5 lineage is the remedy
  glm-a's 15fd7f4 demonstrably lacks). Controls PASS (detector-negative,
  injected self-test, 1 tail artifact excluded).
- EXECUTED 2 (real source): SITE-P does TWO blocking reads per extend
  forward (read-view + write-target coercions) — mocks that log at
  __index__ time, not source reading, found this. All 5 verdicts PASS
  incl. value-preserving mirror control and branch-dead control.
- PENDING: prefill trace (prefill_trace_request.json) decides D1 live /
  D2 lever / D3 cost. No patch written (mirror candidate identified,
  equality unverified). python/ untouched (diff vs 2c854e0 = 0 lines).

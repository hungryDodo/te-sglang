# g3-t12 results — prefill-path sync site: source-proven, trace-calibrated

## Executed (both EXIT 0, raw preserved)

### 1. DtoH census on the REAL controller decode trace (`raw/sync_sites_decode_trace.json`)
Hard gate PASSED (recomputed exactly: 8560 kernels, 565.549025 ms union, 16/16
CPU<->GPU External-id pairing — controller totals). Findings:
- **Exactly 1 Memcpy DtoH per decode step** (16/16), all 1024 bytes (= bs128
  x int64 — a bs-length tensor readback, not an 8-byte scalar), 3.6-5.9 us
  each, ALL in inter-step gaps, completing ~6-7 us before the next step span.
- Post-sync 200us windows: **75.8 us mean GPU idle vs 18.8 us seeded random
  gap control** (seed 4242, in-span windows excluded to match placement) →
  ~57 us/step excess idle in the sync neighborhood on the UNFIXED glm-a tree
  (15fd7f4 has no aa45af5-class fix; code does not auto-merge across workers).
  On the 35.2 ms step that is 0.16% — independently confirms it11's closure
  of THIS operating point, and calibrates the per-sync cost class.
- Controls: detector-negative (no DtoH rows → 0 sites) PASS; detector
  self-test (injected synthetic DtoH counted once) PASS; 1 tail site excluded
  from the probe (window past trace end, artifact).
- Cross-worker note for sync round: glm-a's tree still pays this sync class
  every decode step; glm-b's aa45af5 lineage (Fast 1.0920/1.0984, additive
  per g2-t1) is the measured remedy.

### 2. Real-source execution probe (`raw/prefill_site_probe.json`)
ast-extracted the two REAL SWA-translate blocks from flashinfer_backend.py
(verbatim anchors asserted) and executed them with recording mocks:
- **SITE-P (prefill updater, ~2345-2352, UNFIXED on glm-b's tree today)**:
  `kv_last_index = kv_indptr[-1]` feeds TWO slice coercions (read view + write
  target) → **2 blocking device-scalar reads per extend forward**; translate
  receives the coerced value (V1 PASS). With radix cache disabled in the
  frozen launch, every request re-prefills → every extend forward pays this.
- SITE-D (decode begin-forward, aa45af5-fixed) control: with seq_lens_cpu
  present and bs matching → 0 device reads, 1 cpu sum, same translated value
  (value-preserving on the coherent fixture) (V2 PASS); seq_lens_cpu=None or
  bs mismatch → falls back to 2 reads (V3/V4 PASS, guard boundary documented);
  branch-dead control (SWA off) → nothing reads (V5 PASS).
- Candidate mirror identified but NOT verified: `paged_kernel_lens_sum` is
  typed int at flashinfer_backend.py:1770 (eagle_info.py:442 sets it to
  cum_kv_seq_len[-1]). Whether kv_indptr[-1] == paged_kernel_lens_sum across
  spec_info variants / read kinds / prefix trimming is the pre-registered
  check for the PATCH trial — deliberately not patched this trial.

## Requested (pending controller execution)
`prefill_trace_request.json`: bounded prefill-window capture on this HEAD via
BUILT-IN /start_profile (num_steps=48 auto-stop; no explicit /stop — v1 500
artifact), Prefill-Heavy 8192/32 C16, pre-registered D1/D2/D3 discriminators.
No Fast screen requested (no code change; would re-measure baseline).

## Honest limits
- Mock probe proves source-level dataflow, not CUDA wall time; no torch here.
- The 1024-byte DtoH in the decode trace is the sync CLASS calibration on an
  unfixed tree; the trace cannot pin it to a specific source line. Our tree's
  prefill site reads an 8-byte scalar (x2) — cheaper bytes, same blocking
  round-trip class; D1 of the requested trace decides whether it executes.
- Single operating point per trace; idle fractions only (profiler perturbs).
- python/ untouched: `git diff 2c854e0 HEAD -- python/` empty (lineage kept).

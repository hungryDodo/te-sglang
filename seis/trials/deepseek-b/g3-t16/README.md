# G3 t16 — deepseek-b — decode graph-cap contradiction in the real C128 trace

Executed CPU trial. No engine change, no GPU, no Fast gate requested.
Deliverable: `check_graph_cap.py` + `graph-cap-check.json` (raw output).
Input: `controller_evidence/continuous-20261109-01/20261110-decode-trace/`
(Controller packet; not my experiment).

## Question

The frozen launch records `--cuda-graph-max-bs-decode 64` in `server_argv`
(`original-result.json` index 21-22) and the profiled forward labels report
`bs=128`. In **this candidate tree**, does that flag admit a 128-row decode CUDA
graph replay? If not, the trace's graph behavior cannot be transferred to this
tree, and t15's "eager at bs>64" alternative for the DecodeCeiling C256 gap has
no support.

## Executed checks (all hard-gated; `python3 -I`, stdlib only)

Trace half (`decode.trace.json.gz`, SHA verified `09e8…e4b1`):

- 16 `step[DECODE bs=128]` host spans; 16 `cudaGraphLaunch`; **exactly one
  launch inside every span** (list is `[1]*16`).
- 448 attention launches = 28 layers x 16 forwards, one per layer-forward, one
  constant launch grid `(159,1,2)` — no 2x64 batch split inside a forward.
- 16 lm_head launches, grid `(8,128,1)`.
- KV byte model (Spark2_5Config: 28 layers, 2 KV heads, head_dim 256, BF16 =
  2048 B/token/layer), attention 16.679 ms/forward, seq in [513,576]:
  - M=128: 225.8-253.5 GB/s (82.7-92.9% of 273 GB/s peak)
  - M=64: 112.9-126.7 GB/s (41.3-46.4%) — numerically feasible, so the control
    is not rigged
  - M=256: 451.5-507.0 GB/s — impossible (>peak)
  The engine's own `bs=128` label is the only interpretation coherent with the
  other kernels in the same step (GEMM 199 GB/s, lm_head 214 GB/s, t15).

Source half (this tree, HEAD `9647ebc`):

- AST-extracted and **executed** the real
  `generate_decode_cuda_graph_batch_sizes` body: `max_bs=64` ->
  `[1,2,4,8,12,16,24,32,40,48,56,64]`; 128 excluded. Sensitivity control: the
  same function with `max_bs=256` includes 128.
- AST assertions: `memory_hook.py` materializes `decode.bs` from
  `decode.max_bs`; `base_cuda_graph_runner.get_batch_sizes_to_capture` takes
  `capture_bs` from `decode.bs`; `DecodeCudaGraphRunner.self.max_bs =
  max(capture_bs)`; `can_run_graph` uses `cuda_graph_bs <= self.max_bs` when
  padding is enabled (the default; `--disable-cuda-graph-padding` is not in the
  recorded argv).
- Checked and excluded as rescues for the run tree: Qwen3-VL multimodal
  expansion (locked out by an explicit `max_bs`, wrong arch),
  `ensure_decode_cuda_graphs` capture-to-fit (PD role switch only, not used by
  this launch). `get_batch_sizes_to_capture` only appends `num_max_requests`
  when `max(capture_bs) > pool size` (64 < 256 here).

**Verdict: CONTRADICTION_CONFIRMED.** The real trace replays one 128-row decode
graph per step; this tree, given the recorded argv, cannot. The profiled tree
(commit `15fd7f4`, not available in this repo) resolved decode graph capture
differently from HEAD's flag semantics.

## What this does and does not say

- Does: the supplied C128 trace is a valid bandwidth observation at M=128, but
  it is **not** evidence about HEAD's decode graph coverage at C128/C256, and
  the t15 pre-registered alternative "bs>64 runs eager, explaining the C256
  gap" is unsupported by it (the observed run graphed at 128 despite the 64
  flag).
- Does not: claim a HEAD bug, a speedup, or that C256 is graphed. The run-tree
  source is unavailable, so the mechanism of the divergence is unknown, and the
  trace's `trace_validation` itself warns that step labels are host arguments,
  not device occupancy.
- Risks: `record_shapes=false`, so M=128 is inferred jointly from the engine
  label and per-class bandwidth coherence, not read from tensor shapes; seq is
  bounded from the workload (512 prompt + <=64 output), not measured.

## Request (metadata, not a new measurement)

For the profiled run or the next diagnostic, record `decode_cuda_graph_capture_bs`
from `/get_server_info` (`model_runner.py:1622-1625` records it for exactly this
purpose) and/or the server.log capture line. That settles the run tree's
resolved capture list at zero GPU cost. Any C256 probe must report that list and
`cudaGraphLaunch` count before the gap is attributed.

## Reproduce

```
python3 -I seis/trials/deepseek-b/g3-t16/check_graph_cap.py \
  --trace controller_evidence/continuous-20261109-01/20261110-decode-trace/decode.trace.json.gz \
  --repo . --out seis/trials/deepseek-b/g3-t16/graph-cap-check.json
```

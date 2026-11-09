# G3 trial 16 — deepseek-b — plan

Campaign `continuous-20261109-01`, Iteration 16. Worker deepseek-b (node a4).
Bounds: 60 turns / 20 min / USD 5 nominal. No GPU, no SSH, no evaluator.
Product: one bounded, executed CPU falsifier over Controller-supplied real
runtime data, plus a precise request/negative. No engine change unless the
finding supports one.

## 0. Hard constraints

- Official gates belong to the controller. No EXACT / Candidate / Promotion /
  best_verified claim from this worker.
- Evaluator, corpus, seed, reference, launch protocol unchanged.
- F3 (`f263262`) stays **UNGATED**: t13 already replayed its dispatch cells on
  the real C22 Q8_0 trace (15,952/28,608 calls flip, 0 mismatches) and gate
  batches are bs>=32, so a threshold retune is score-neutral by construction.
  This trial does not re-litigate F3.
- Controller packets are read-only input; derived analysis is written under
  `seis/trials/deepseek-b/g3-t16/`.
- `request_fast_gate=false`: this trial changes no engine source, so a BF16 Fast
  screen would only re-measure the baseline.

## 1. New question (not previously asked)

The frozen BF16 launch passes `--cuda-graph-max-bs-decode 64`. In this tree
(`python/sglang/srt/arg_groups/memory_hook.py:203`,
`python/sglang/srt/model_executor/cuda_graph_config.py`,
`model_executor/runner/decode_cuda_graph_runner.py:845-895`), that flag becomes
`cuda_graph_config[decode].max_bs`, the capture list is generated from it, and
with padding enabled a decode batch replays a graph only if
`batch_size <= max_bs` — otherwise it runs eager.

Yet the Controller's real Medium 512/64 **C128** trace (a1, commit 15fd7f4)
contains 16 `cudaGraphLaunch` events, one per profiled forward, and its step
labels report `bs=128`. Both cannot hold under the frozen flag if each forward
is one graph replay of 128 requests.

**Q:** what decode batch geometry did the profiled fragment actually execute?

- H-A (provenance): the flag was not in effect for a1's profile, so capture
  list defaulted to the GB10/tp1 default (`max_bs` 256) and C128/C256 both
  replay graphs. Material for t15's C256 prediction: eager dispatch is not an
  explanation for the 18-26% gap.
- H-B (scheduler): the effective decode batch in the fragment was <=64
  requests, so a graph replay is consistent with the flag; the C128 workload
  was not executing 128-row decode forwards during the profiled steps.
- H-C (tree mismatch): commit 15fd7f4 predates/changes this arg handling and
  the tree's `can_run_graph` does not describe the profiled engine.

**Discriminator (trace-only, CPU):** per-forward kernel geometry. A single
128-row forward and two 64-row graph replays are distinguishable by kernel
count per class per forward (GEMM 4/layer, attention 1/layer, norm 2/layer,
argmax 1/step are batch-splitting-invariant per *forward*, so a split batch
doubles the counts per scheduler step) and by whether `cudaGraphLaunch` count
equals step count or twice it. Additional independent check: decode GEMM
kernel *duration* vs the known 2.88 GB per-layer weight stream — M-independent
(weight-bound), so a split would show two ~half-attention kernels, not one.
Control: count the same classes in a known-64-row window if any exists.

## 2. Execution plan

1. Parse `decode.trace.json.gz` with stdlib gzip/json under `python3 -I`;
   verify SHA against MANIFEST; reproduce packet totals as a hard gate.
2. Reconstruct per-forward class counts and durations; test split-vs-single.
3. Check the packet for launch flags/provenance (grep JSON for graph args).
4. State the result with limits; no performance claim beyond the trace.
5. `iteration-proposal.json`, commit, clean tree.

## 3. Kill criteria

- If the trace cannot discriminate (e.g., step labels are the only bs evidence
  and kernel counts are ambiguous), report INCONCLUSIVE with the exact missing
  field and a bounded follow-up request; do not infer the launch flags.

## 4. Outcome (executed)

**CONTRADICTION_CONFIRMED.** Real trace: 16 `step[DECODE bs=128]` spans, 16
`cudaGraphLaunch`, exactly one per span; 448 attention launches = 28/forward,
one constant grid `(159,1,2)`; 16 lm_head launches; KV byte model closes only
at M=128 (225.8-253.5 GB/s, 82.7-92.9% of peak; M=64 feasible control at
41-46%; M=256 impossible). This tree: the AST-executed real capture-list
generator tops out at 64 for `--cuda-graph-max-bs-decode 64` (128 excluded;
max_bs=256 control includes it) and `can_run_graph` replays only when
`batch_size <= max_bs`. Hence the profiled tree's resolved decode-graph config
is not HEAD's; the trace is not evidence about HEAD's C128/C256 graph coverage
and t15's "eager at bs>64" C256 alternative is unsupported. No engine change;
request is a zero-GPU metadata query (`decode_cuda_graph_capture_bs`).
Artifacts: `seis/trials/deepseek-b/g3-t16/`.

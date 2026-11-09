# G3 t5 — f263262 evaluation and dispatch-tracer hardening (CPU-executed)

No GPU, no torch/gguf package, no model weights. Nothing here is a
performance measurement.

## 1. Selection contract, executed from the real source

`verify_dispatch_contract.py` extracts with `ast`, compiles and executes:

- the shipped `_mmvq_safe_batch` + `_MMVQ_SAFE_MEASURED` from the candidate
  `gguf.py`;
- the legacy two-assignment threshold `if` from `f263262^:...gguf.py`;
- both files' quant-type sets and the `fused_mul_mat_gguf` branch tests.

Grid: every `WeightType` name the module references, rows {1,1024,4096,5120,
5121,8192,16384}, batches 0..257.

Results (`dispatch-contract.json`, post-patch source; `dispatch-contract-baseline.json`
is the same oracle on the pre-patch source):

- The dispatch branch rule is **identical** between `f263262^` and the shipped
  function (`branch_rule_identical_parent_vs_shipped: true`).
- Exactly **8 distinct cells** change kernel: `Q8_0` and `Q4_K`, rows<=5120,
  batches 3,4,5,6, all MMVQ -> MMQ. All other types/bands/batches are
  unchanged; an unknown integer type still hits the same ValueError fallback
  (2/6) as before.
- **0 changed cells at any batch >= 16**, and 0 at the probed gate batches
  {16,32,64,128,256}. The change cannot move the frozen BF16 gate even if the
  gate loaded GGUF: every gate point is at batch >= 16 (decode) or processes
  thousands of token rows (prefill).
- Provenance limitation: the g2-t1/g2-t2 raw crossover artifacts that support
  `_MMVQ_SAFE_MEASURED` are **not present in this snapshot** (checked with
  `git cat-file -e`); the values are cited from branch/controller evidence and
  were not re-verified here.

## 2. Tracer audit and patch

The only instrument that can prove live dispatch is the `4e1644a` tracer in
`gguf.py`. Executed checks:

- the tracer block is import-safe and runs at module import (quantization
  `__init__` imports `gguf` unconditionally, so this happens on the BF16 gate
  too); it is inert unless `SGLANG_GGUF_DISPATCH_TRACE` is set;
- `_record_dispatch` is called at all three exits of `fused_mul_mat_gguf`
  (empty, unquantized, post-branch), so the recorded kernel is the branch
  actually taken;
- **Failure mode found (executed):** with the old cadence (dump every 1024
  calls + `atexit`), a 100-call probe that ends abruptly (SIGKILL, or SIGTERM
  without graceful exit) leaves **no trace file** — the pre-registered g3-t2
  experiment would be INCONCLUSIVE with zero dispatch evidence.
- **Patch:** cadence is now `SGLANG_GGUF_DISPATCH_TRACE_EVERY` (default **64**,
  invalid values fall back to 64; was hard-coded 1024). One forward pass
  records ~100+ calls (28 layers x dense linears), so a trace is now durable
  after the first forward; worst-case loss under abrupt termination is 63
  recorded calls instead of 1023. Same 100-call simulation after the patch:
  file written at call 64, `trace_calls=64`, cells present. The knob is read
  only when tracing is enabled and cannot affect an untraced run.

Scope caveats: `fused_moe_gguf` (MoE) is not traced, but Spark2_5 uses a dense
MLP, so it is N/A for the Spark GGUF probe; with TP>1 each worker would write
the same path (last writer wins) — run the probe single-worker; under CUDA
graphs the trace records capture-time shapes only, which is why the g3-t2
protocol already requires `--disable-cuda-graph`.

## Files

- `verify_dispatch_contract.py` — the oracle (stdlib only, `python3 -I`).
- `dispatch-contract-baseline.json` — pre-patch run (`dump_every=1024`).
- `dispatch-contract.json` — post-patch run (`dump_every=64`).

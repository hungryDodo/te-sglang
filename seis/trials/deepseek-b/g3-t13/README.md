# G3 t13 — deepseek-b: F3 dispatch-cell audit on the real C22 GGUF trace

**Status: CPU-executed replay; NO engine change; no formal gate requested.
F3 remains UNGATED.**

## Question

`f263262` (MMVQ thresholds) is UNGATED. The controller supplied a real Q8_0
GGUF dispatch trace (canonical C22, run `c396018f...`, 12 x 512-token prompts,
C4). This trial asks: **does the trace exercise the cells where `f263262`
changes kernel selection, and does the observed kernel match the patched
policy everywhere?** It is an audit of supplied data, not a new GPU experiment.

## Method (executed, `python3 -I`, stdlib)

`audit_f3_dispatch_cells.py` ast-extracts and executes the deciding source:
the HEAD `_MMVQ_SAFE_MEASURED` / `_mmvq_safe_batch` / quant-type sets, and the
legacy `mmvq_safe` If-block from the verbatim `f263262^` source
(`legacy_gguf_f263262_parent.py.txt`). It then replays every cell of
`controller_evidence/.../20261110-runtime/gguf-dispatch.json` under both
policies. Hard exit 2 on anchor drift.

## Result (raw: `f3-dispatch-audit.json`)

- 28,608 recorded calls, **0** observed-vs-patched-policy mismatches.
- **15,952 calls (55.8%)** change selection vs the parent, all MMVQ -> MMQ:
  `Q8_0|rows<=5120|batch4` 15,868 (legacy safe=6 -> patched safe=2) and
  `Q8_0|rows<=5120|batch6` 84 (canonical C22: startup prefill, not C4 decode).
- `Q8_0|rows>5120` does not flip (legacy safe=2 == patched safe=2).
- Q4_K changed cells are **not exercised** by this Q8_0-only trace.

## Interpretation and limits

F3's changed region is live and dispatch-contract-conforming in an ordinary
GGUF Q8_0 run, so the patch is not dead code. The packet has no timing
comparison (instrumented trace perturbs wall time) and no numeric equivalence,
and the frozen formal gate loads BF16 safetensors, so **no gain, quality or
promotion claim follows; F3 stays UNGATED**.

## Also this trial (negative)

The t12 wide-prefill lever fails its precondition: `max_prefill_buffer_tokens()`
(`runtime_context.py:1778`, `arg_groups/overrides.py:1865`) returns
`chunked_prefill_size` (8192) when PP dynamic chunking is off (frozen launch),
and the FlashInfer extend autotune dummy plus pre-capture activation reserve are
sized from it. 2 x 8192 per forward would exceed that ceiling. No engine patch.

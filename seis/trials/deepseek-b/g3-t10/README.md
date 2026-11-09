# G3 t10 — deepseek-b: frozen-launch reachability of the pending host-mirror candidate

**Status: CPU-executed source oracle; NO engine change; no new gate requested. F3 unchanged.**
Base: `4e84e42` (engine tree = `f263262` + t3 `33e5ac4` decode host-mirror + t4 `a306e40`
prefill host-mirror; HEAD differs from the i4 gate tree only by the env-gated, inert
GGUF dispatch tracer from t5). Raw output: `swa-pool-reachability.json`.
Oracle: `verify_swa_pool_reachability.py` (stdlib, `python3 -I`, 0.05 s, exit 0).

## Why this trial

Trials 1–9 closed the f3 and engine-surface audits. The one attribution question t6/t8
explicitly left open was: **is the patched SWA translate block reachable under the frozen
launch, or is it dead code because the engine builds a unified (translating) KV pool?**
Without that answer, an i4 Fast result of ~1.00 could not be interpreted: it could mean
"measured a real D2H removal and it did not matter" or "the code never ran".

## Executed method

The oracle line-anchors 20 real source constructs across 9 files (fails loudly on drift,
exit 2), then executes — not re-implements — the decisive source:

1. `KVIndexTranslator._reads_translated` is AST-extracted, compiled and called on a
   modelled space object (`space is not None and space.read_v2p is not None`).
2. The `is_unified_target = isinstance(allocator, (UnifiedMamba..., UnifiedSWAAllocatorBase))
   and allocator.get_kvcache() is token_to_kv_pool` expression and
   `self.is_translating = is_unified_target or is_fused_draft` are AST-extracted and
   `eval`-ed with real `isinstance` over a class model whose **bases are themselves
   extracted from the source** (`UnifiedSWAAllocatorBase(SWATokenToKVPoolAllocator)`,
   `SWAKVPool(BaseSWAKVPool)`, …).
3. `--enable-unified-memory` default is read from `arg_groups/fields/memory.py` (`= False`);
   the frozen argv (verbatim, 20 tokens) is checked for the flag.
4. `use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated` and
   the guard `if use_sliding_window_kv_pool and not use_swa_source:` are anchored and
   evaluated.
5. Two negative controls flip the deciding predicate.

## Executed result

| configuration | is_translating | reads_are_translated | use_swa_source | patched block |
|---|---|---|---|---|
| **frozen launch** | False | **False** | False | **EXECUTES** |
| control: `--enable-unified-memory` | True | True | True | dead |
| control: unified + DCP=2 | True | False | False | executes |
| control: no `SWAKVPool` | — | — | — | not reached |

**Verdict: REACHABLE.** The frozen launch passes no `--enable-unified-memory` (default
`False`), so the engine builds the plain `SWAKVPool` (`_build_hybrid_swa_kv_pool`), the
translator is non-translating and reads stay virtual. The t3/t4 host-mirror lines are
**not dead code**; the i4-class screen measures a real D2H removal.

## Effect-size bound (why the number will be ~1.00 anyway)

With one backend metadata init per forward, the workload arithmetic (signed point table,
`requests=max(32,4C)`, chunk 8192, prefill budget 16384 from g3-t9) gives:

| patch | D2H removed per point | bound @100 µs/sync | @500 µs/sync |
|---|---|---|---|
| t4 **prefill** (i4's delta over i3) | 2–32 (token-budget packing) | 0.2–3.2 ms | 1–16 ms |
| t3 **decode** (already in i3) | 128–2048 | 13–205 ms | 64–1024 ms |

The prefill patch's largest point (PrefillHeavy, ~16 s wall) moves by **≤~0.02 % at
100 µs/sync, ≤~0.1 % at 500 µs/sync** — below Fast resolution. The decode patch is 20–60×
larger and is the plausible source of the i3 +0.4 % (within run noise). So: **i4's
measurable component is structurally sub-noise; a ~1.000 result is expected and
uninformative, and any i4 gain should be read as noise, not as the prefill patch.**

## Limits

- Class model + extracted expressions, not a live torch import; pool/allocator objects are
  modelled. The non-MLA/hybrid-SWA branch selection in `kv_cache_configurator` is prior
  source reading, not executed (anchored only).
- `is_hybrid_swa` and `sliding_window_size is not None` for Spark-X2.6-1.7B rest on prior
  evidence (21/28 sliding layers, window 511); if absent, the block is never reached.
- Per-sync cost is an assumption (10/50/100/500 µs sensitivity shown), not a measurement;
  forward counts are bounds (token-budget vs one-request-per-forward).
- No GPU/CUDA/page/overlap behaviour executed. No perf, numeric or quality claim.
- F3 (`f263262`) remains **UNGATED and score-unreachable** under the BF16 gate
  (controller `applicability-001`; g3-t1/t5 dispatch contract: 0 changed cells at
  batch ≥ 16; g3-t8 call-site audit). No GGUF GPU experiment is requested here: its
  prerequisites are still unmet and the protocol is already pre-registered (g3-t2).

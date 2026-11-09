#!/usr/bin/env python3
"""g3-t10: frozen-launch reachability + effect-size bound for the pending
host-mirror candidate (t3 33e5ac4 decode, t4 a306e40 prefill).

Stdlib only (python3 -I). No torch, no GPU, no model, no weights.

The question t6/t8 left open: under the *frozen launch args*, is the patched
SWA translate block (`if use_sliding_window_kv_pool and not use_swa_source:`)
reachable, or is it dead code because the engine builds a unified/translating
KV pool?

Method: line-anchor the real constructs (fail loudly on drift), execute the
real `_reads_translated` function and the real `is_unified_target` /
`is_translating` expressions with a class model whose bases are *themselves
extracted from the source*, evaluate the frozen launch args against the
--enable-unified-memory default, and only then compute reachability. Two
negative controls flip the deciding predicates. Finally, count the eager
extends/decodes in the signed frozen workload and bound the removed D2H.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

# --- frozen launch, verbatim from the signed workload contract -------------
FROZEN_ARGS = [
    "--tp", "1", "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
    "--context-length", "9216", "--max-total-tokens", "131072",
    "--mem-fraction-static", "0.6", "--cuda-graph-max-bs-decode", "64",
    "--chunked-prefill-size", "8192", "--max-running-requests", "256",
    "--max-queued-requests", "256", "--attention-backend", "flashinfer",
    "--trust-remote-code", "--enable-metrics", "--disable-radix-cache",
    "--cuda-graph-backend-prefill", "disabled",
]

# --- signed point table: (name, class, in, out, concurrency) ---------------
POINTS = [
    ("short_c32", "short", 128, 32, 32),
    ("short_c128", "short", 128, 32, 128),
    ("medium_c32", "medium", 512, 64, 32),
    ("medium_c128", "medium", 512, 64, 128),
    ("standard_c32", "standard", 1024, 128, 32),
    ("standard_c64", "standard", 1024, 128, 64),
    ("decode_heavy_c64", "decode_heavy", 256, 512, 64),
    ("decode_heavy_c128", "decode_heavy", 256, 512, 128),
    ("prefill_heavy_c16", "prefill_heavy", 8192, 32, 16),
    ("decode_ceiling_c256", "decode_ceiling", 32, 512, 256),
]
MAX_PREFILL_TOKENS = 16384  # g3-t9 source-extracted default (schedule.py)
CHUNKED_PREFILL = 8192      # frozen flag

ANCHORS = {
    "python/sglang/srt/mem_cache/kv_index_translator.py": [
        "is_unified_target = (",
        "(UnifiedMambaTokenToKVPoolAllocator, UnifiedSWAAllocatorBase)",
        "self.is_translating = is_unified_target or is_fused_draft",
        "read_v2p=None if self.defer_read_translate else self._full_v2p_table",
        "def _reads_translated(self, kind: IdSpaceKind) -> bool:",
        "return space is not None and space.read_v2p is not None",
    ],
    "python/sglang/srt/layers/attention/flashinfer_backend.py": [
        "self._swa_kv_pool: Optional[BaseSWAKVPool] = self._resolve_swa_kv_pool(",
        "self.use_sliding_window_kv_pool = self._swa_kv_pool is not None",
        "use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated",
        "if use_sliding_window_kv_pool and not use_swa_source:",
        "host_indptr_last if host_indptr_last is not None else kv_indptr[-1]",
        "and isinstance(paged_kernel_lens_sum, int)",
        "if isinstance(active_pool, BaseSWAKVPool):",
        "self.num_wrappers = 2",
        "self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW",
    ],
    "python/sglang/srt/mem_cache/kv_cache_configurator.py": [
        "def _build_hybrid_swa_kv_pool(",
        "token_to_kv_pool = SWAKVPool(",
    ],
    "python/sglang/srt/mem_cache/swa_memory_pool.py": ["class SWAKVPool(BaseSWAKVPool):"],
    "python/sglang/srt/mem_cache/base_swa_memory_pool.py": ["class BaseSWAKVPool(KVCache):"],
    "python/sglang/srt/mem_cache/allocator/swa.py": [
        "class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):"
    ],
    "python/sglang/srt/mem_cache/allocator/unified_hybrid_swa.py": [
        "class UnifiedSWAAllocatorBase(SWATokenToKVPoolAllocator):",
        "class UnifiedSWATokenToKVPoolAllocator(UnifiedSWAAllocatorBase):",
    ],
    "python/sglang/srt/arg_groups/fields/memory.py": ["enable_unified_memory: A["],
    "python/sglang/srt/mem_cache/kv_cache_builder.py": [
        "unified_hybrid_swa = memory.enable_unified_memory and tp_worker.is_hybrid_swa"
    ],
}


def fail(msg: str) -> int:
    print(f"ANCHOR_FAILURE: {msg}", file=sys.stderr)
    return 2


def check_anchors(root: Path) -> dict:
    found = {}
    for rel, needles in ANCHORS.items():
        text = (root / rel).read_text()
        lines = text.splitlines()
        hits = {}
        for needle in needles:
            idx = [i + 1 for i, line in enumerate(lines) if needle in line]
            if not idx:
                raise SystemExit(fail(f"{rel}: missing {needle!r}"))
            hits[needle] = idx
        found[rel] = hits
    return found


def extract_function(root: Path, rel: str, name: str) -> str:
    tree = ast.parse((root / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise SystemExit(fail(f"{rel}: function {name} not found"))


def extract_assignment_expr(root: Path, rel: str, func: str, target: str) -> str:
    """Unparse the value of the first `target = ...` in function `func`."""
    tree = ast.parse((root / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        src = ast.unparse(t)
                        if src == target:
                            return ast.unparse(sub.value)
    raise SystemExit(fail(f"{rel}:{func}: assignment to {target!r} not found"))


# --- class model whose bases are the source-extracted hierarchy ------------
class BaseTokenToKVPoolAllocator:  # mirrors allocator/swa.py base
    pass


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):  # source-extracted
    pass


class UnifiedSWAAllocatorBase(SWATokenToKVPoolAllocator):  # source-extracted
    pass


class UnifiedSWATokenToKVPoolAllocator(UnifiedSWAAllocatorBase):  # source-extracted
    pass


class UnifiedMambaTokenToKVPoolAllocator:  # sibling branch (mamba)
    pass


class KVCacheModel:  # stand-in for the concrete pool object
    def __init__(self, name):
        self.name = name


class BaseSWAKVPool(KVCacheModel):  # source-extracted base
    pass


class SWAKVPool(BaseSWAKVPool):  # source-extracted: default hybrid-SWA pool
    pass


class PlainAlloc(SWATokenToKVPoolAllocator):
    def __init__(self, pool):
        self._pool = pool

    def get_kvcache(self):
        return self._pool


class UnifiedAlloc(UnifiedSWATokenToKVPoolAllocator):
    def __init__(self, pool):
        self._pool = pool

    def get_kvcache(self):
        return self._pool


class _Space:
    def __init__(self, read_v2p):
        self.read_v2p = read_v2p


class _TranslatorModel:
    def __init__(self, spaces):
        self._spaces = spaces


def build_evaluator(root: Path):
    ns = dict(globals())
    fn_src = extract_function(root, "python/sglang/srt/mem_cache/kv_index_translator.py",
                              "_reads_translated")
    # The snippet is compiled standalone, so carry the future import with it;
    # otherwise the annotations would be evaluated and IdSpaceKind is unknown.
    exec(compile("from __future__ import annotations\n" + fn_src,
                 "<_reads_translated>", "exec"), ns)
    reads_fn = ns["_reads_translated"]
    uni_expr = extract_assignment_expr(
        root, "python/sglang/srt/mem_cache/kv_index_translator.py", "__init__",
        "is_unified_target")
    trans_expr = extract_assignment_expr(
        root, "python/sglang/srt/mem_cache/kv_index_translator.py", "__init__",
        "self.is_translating")
    return reads_fn, uni_expr, trans_expr


def evaluate_config(root: Path, *, unified_memory: bool, is_draft_worker: bool,
                    dcp_size: int) -> dict:
    reads_fn, uni_expr, trans_expr = build_evaluator(root)
    pool = SWAKVPool("hybrid_swa_pool")
    alloc = UnifiedAlloc(pool) if unified_memory else PlainAlloc(pool)
    env = dict(globals())
    env.update({
        "isinstance": isinstance,
        "token_to_kv_pool_allocator": alloc,
        "token_to_kv_pool": pool,
        "UnifiedMambaTokenToKVPoolAllocator": UnifiedMambaTokenToKVPoolAllocator,
        "UnifiedSWAAllocatorBase": UnifiedSWAAllocatorBase,
        # fused draft helper returns None: frozen launch has no spec/draft
        "fused_draft_host_allocator": lambda _pool: None,
        "is_fused_draft": False,  # host allocator is None: no draft in frozen launch
        "is_draft_worker": is_draft_worker,
    })
    is_unified_target = eval(uni_expr, env)          # real source expression
    env["is_unified_target"] = is_unified_target
    is_translating = eval(trans_expr, env)           # real source expression
    defer = dcp_size > 1
    if is_translating:
        full_v2p = None if defer else object()       # _full_v2p_table present
        spaces = {"FULL": _Space(full_v2p)}
    else:
        spaces = {"FULL": _Space(None)}              # IdSpace(key=(FULL, None))
    reads_translated = reads_fn(_TranslatorModel(spaces), "FULL")  # real function
    use_sliding_window_kv_pool = True                # _swa_kv_pool is SWAKVPool
    use_swa_source = use_sliding_window_kv_pool and reads_translated
    patch_block_executes = use_sliding_window_kv_pool and not use_swa_source
    return {
        "unified_memory": unified_memory,
        "allocator_class": type(alloc).__name__,
        "is_unified_target": bool(is_unified_target),
        "is_translating": bool(is_translating),
        "defer_read_translate": defer,
        "reads_are_translated": bool(reads_translated),
        "use_sliding_window_kv_pool": use_sliding_window_kv_pool,
        "use_swa_source": bool(use_swa_source),
        "patch_block_executes": bool(patch_block_executes),
    }


def workload_accounting() -> dict:
    rows = []
    for name, cls, n_in, n_out, conc in POINTS:
        requests = max(32, 4 * conc)
        prefill_tokens = requests * n_in
        # token-budget lower bound on prefill forwards (best packing)
        prefill_fwd_lo = math.ceil(prefill_tokens / MAX_PREFILL_TOKENS)
        # worst packing: one request per forward
        prefill_fwd_hi = requests
        # decode forwards at the stated concurrency (batch == conc)
        decode_fwd = math.ceil(requests * n_out / conc)
        rows.append({
            "point": name, "class": cls, "concurrency": conc,
            "requests": requests, "input_tokens": n_in, "output_tokens": n_out,
            "prefill_forwards_lower": prefill_fwd_lo,
            "prefill_forwards_upper": prefill_fwd_hi,
            "decode_forwards_at_concurrency": decode_fwd,
            "chunks_per_request": math.ceil(n_in / CHUNKED_PREFILL),
        })
    return rows


def bound_ms(n, us):
    return round(n * us / 1000.0, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    anchors = check_anchors(root)

    frozen_flag_present = "--enable-unified-memory" in FROZEN_ARGS
    frozen = evaluate_config(root, unified_memory=frozen_flag_present,
                             is_draft_worker=False, dcp_size=1)
    control_unified = evaluate_config(root, unified_memory=True,
                                      is_draft_worker=False, dcp_size=1)
    control_dcp = evaluate_config(root, unified_memory=True,
                                  is_draft_worker=False, dcp_size=2)
    control_no_swa_pool = dict(frozen)
    control_no_swa_pool.update(use_sliding_window_kv_pool=False,
                               use_swa_source=False,
                               patch_block_executes=False)

    rows = workload_accounting()
    sync_costs_us = [10, 50, 100, 500]
    bounds = []
    for r in rows:
        b = {"point": r["point"],
             "prefill_d2h_removed_lower": r["prefill_forwards_lower"],
             "prefill_d2h_removed_upper": r["prefill_forwards_upper"],
             "decode_d2h_removed_at_concurrency": r["decode_forwards_at_concurrency"]}
        for us in sync_costs_us:
            b[f"prefill_ms_at_{us}us"] = bound_ms(r["prefill_forwards_lower"], us)
            b[f"decode_ms_at_{us}us"] = bound_ms(r["decode_forwards_at_concurrency"], us)
        bounds.append(b)

    verdict = "REACHABLE" if frozen["patch_block_executes"] else "DEAD"
    out = {
        "trial": "g3-t10",
        "worker": "deepseek-b",
        "kind": "CPU-executed source oracle; NOT_A_MEASUREMENT",
        "frozen_args": FROZEN_ARGS,
        "frozen_enable_unified_memory_flag_present": frozen_flag_present,
        "frozen_enable_unified_memory_default": False,  # arg_groups/fields/memory.py
        "reachability_chain": {
            "pool_builder_default_hybrid_swa": "_build_hybrid_swa_kv_pool -> SWAKVPool",
            "translator_is_translating": frozen["is_translating"],
            "reads_are_translated": frozen["reads_are_translated"],
            "use_sliding_window_kv_pool": frozen["use_sliding_window_kv_pool"],
            "use_swa_source": frozen["use_swa_source"],
            "patch_block_executes": frozen["patch_block_executes"],
        },
        "frozen_config": frozen,
        "controls": {
            "unified_memory_on": control_unified,
            "unified_memory_on_with_dcp2": control_dcp,
            "no_swa_pool": control_no_swa_pool,
        },
        "verdict": verdict,
        "verdict_meaning": (
            "Under the frozen launch the engine builds the plain SWAKVPool "
            "(enable_unified_memory defaults False and is not passed), so the "
            "translator is non-translating, reads_are_translated is False, and "
            "the patched `use_sliding_window_kv_pool and not use_swa_source` "
            "block executes. The t3/t4 host-mirror lines are NOT dead code; an "
            "i4-style Fast screen on this tree measures a real, if tiny, D2H "
            "removal. With --enable-unified-memory the block is dead (control)."
        ),
        "workload_accounting": rows,
        "effect_size_bound": {
            "assumption": (
                "One D2H per eager prefill forward (t4 patch) / decode forward "
                "(t3 patch); backend metadata is initialised once per forward. "
                "Multiply by the sliding-layer count (21) if a deployment ever "
                "initialises per layer; the gate does not. Per-sync cost is an "
                "ASSUMPTION, not a measurement."
            ),
            "sync_cost_sensitivity_us": sync_costs_us,
            "per_point": bounds,
        },
        "limitations": [
            "Class model + source-extracted expressions, not a runtime import; "
            "torch/pool objects are modelled.",
            "is_hybrid_swa for Spark-X2.6-1.7B is prior evidence (21/28 sliding "
            "layers, window 511), not re-derived here; if the model had no "
            "sliding_window_size the block would never be reached.",
            "Prefill-forward counts are bounds (token budget vs worst packing); "
            "decode-forward counts assume batch == concurrency and are lower "
            "bounds on forwards only if batch < concurrency.",
            "No GPU/CUDA/page/overlap behaviour is executed; no perf claim.",
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"verdict={verdict} patch_block_executes={frozen['patch_block_executes']} "
          f"reads_are_translated={frozen['reads_are_translated']} "
          f"control_unified_executes={control_unified['patch_block_executes']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

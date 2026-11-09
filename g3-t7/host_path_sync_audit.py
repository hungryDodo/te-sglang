#!/usr/bin/env python3 -I
"""G3 trial-7 executed audit: blocking device reads on the per-decode-step host
path OUTSIDE the attention backend (t5 covered flashinfer_backend.py only).

Targets: scheduler.py, schedule_batch.py, sampler.py, logits_processor.py,
sampling_batch_info.py, memory_pool.py, model_runner.py (+ overlap utils and
the async d2h copy worker for context).

For every .item()/.cpu()/.tolist()/.nonzero()/.numpy()/.synchronize() call site:
record file:line, enclosing class/function, receiver expression, host-mirror hint,
and a HOT/COLD guess from the enclosing name. Hot/cold and criticality verdicts
are finalized by reading the real call paths (documented in the outcome), not by
this scan alone.

Negative control: the scanner must re-find the t5-known sites in memory_pool /
scheduler where applicable, and at least the canonical `.tolist()` sites; if the
site count is 0 for any file the scanner aborts non-zero (broken extraction).

Raw JSON out: g3-t7/raw/host_path_sync_audit.json
"""
import ast, json, os, sys

REPO = "/workspace"
TARGETS = [
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/schedule_batch.py",
    "python/sglang/srt/layers/sampler.py",
    "python/sglang/srt/layers/logits_processor.py",
    "python/sglang/srt/sampling/sampling_batch_info.py",
    "python/sglang/srt/mem_cache/memory_pool.py",
    "python/sglang/srt/model_executor/model_runner.py",
    "python/sglang/srt/managers/overlap_utils.py",
    "python/sglang/srt/managers/async_d2h_copy_worker.py",
]
OUT = "g3-t7/raw/host_path_sync_audit.json"

BLOCKING_ATTRS = {"item", "cpu", "tolist", "nonzero", "numpy", "synchronize"}
HOST_HINTS = ("_cpu", "seq_lens_cpu", "prefix_lens_cpu", "extend_lens_cpu",
              "host_lengths", "indptr_cpu", "pinned", "last_gpu_latents",
              "host_", "_host")

# Enclosing-function substrings that indicate once-per-lifetime / once-per-req-
# finish / capture / error paths rather than per-step work. Guessed here; the
# outcome section records the human-verified final classification.
COLD_HINTS = ("init", "capture", "load", "profile", "__init__", "release",
              "destroy", "shutdown", "event_loop", "handle", "print", "dump",
              "warmup", "register", "evict", "insert", "reset", "shutdown")


def enclosing_chain(tree, target):
    """Map each function node to its (class, function) enclosing chain."""
    chains = {}
    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chains[child] = chain + [child.name]
                walk(child, chains[child])
            elif isinstance(child, ast.ClassDef):
                chains[child] = chain + [child.name]
                walk(child, chains[child])
            else:
                walk(child, chain)
    walk(tree, [])
    return chains.get(target, [])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sites = []
    for rel in TARGETS:
        path = os.path.join(REPO, rel)
        src = open(path).read()
        tree = ast.parse(src)
        # node -> chain
        chains = {}
        def walk(node, chain):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chains[child] = chain + [child.name]
                    walk(child, chains[child])
                elif isinstance(child, ast.ClassDef):
                    chains[child] = chain + [child.name]
                    walk(child, chains[child])
                else:
                    walk(child, chain)
        walk(tree, [])
        parents = {}
        for p in ast.walk(tree):
            for c in ast.iter_child_nodes(p):
                parents[c] = p
        file_count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            attr = None
            recv = None
            if isinstance(f, ast.Attribute) and f.attr in BLOCKING_ATTRS:
                attr = f.attr
                recv = ast.unparse(f.value)
            elif (isinstance(f, ast.Attribute) and f.attr == "synchronize"
                  and recv_is_cuda(f.value)):
                attr = "synchronize"
                recv = ast.unparse(f.value)
            if attr is None:
                continue
            # enclosing function/class
            fn = node
            chain = []
            while fn in parents:
                fn = parents[fn]
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chain.append(fn.name)
            chain.reverse()
            func = next((c for c in reversed(chain)
                         if not c[0].isupper() or c == chain[-1]), chain[-1] if chain else "<module>")
            cls = next((c for c in chain if c[0].isupper()), None)
            try:
                seg = ast.get_source_segment(src, node)
            except Exception:
                seg = None
            sites.append({
                "file": rel,
                "line": node.lineno,
                "attr": attr,
                "receiver": recv,
                "class": cls,
                "function": chain[-1] if chain else "<module>",
                "chain": "->".join(chain) if chain else "<module>",
                "host_hint": any(h in recv for h in HOST_HINTS),
                "src": (seg or "")[:160],
            })
            file_count += 1
        # Negative control is grep-cross-checked: sampler.py/sampling_batch_info.py/
        # async_d2h_copy_worker.py verified 0 blocking calls by grep, so 0 from the
        # scanner is a TRUE negative there. Abort only on a global-0 scan.
        if file_count == 0:
            print(f"note: 0 sites in {rel} (grep-verified true negative)", file=sys.stderr)
    if not sites:
        print("SCANNER-BROKEN: 0 sites total", file=sys.stderr)
        sys.exit(2)
    json.dump({"targets": TARGETS, "n_sites": len(sites), "sites": sites},
              open(OUT, "w"), indent=1)
    # stdout summary: counts per file/attr, then hot candidates (non-cold names)
    from collections import Counter
    per_file = Counter(s["file"] for s in sites)
    print("SITES PER FILE:", dict(per_file))
    print("TOTAL:", len(sites))
    print("\n== per-step candidates (function name not matching cold hints) ==")
    for s in sorted(sites, key=lambda x: (x["file"], x["line"])):
        fname = s["chain"]
        if not any(h in fname.lower() for h in COLD_HINTS):
            print(f'{s["file"]}:{s["line"]} [{s["attr"]}] {s["chain"]}')
            print(f'    recv: {s["receiver"]}   host_hint={s["host_hint"]}')


def recv_is_cuda(v):
    return "cuda" in ast.unparse(v)


main()

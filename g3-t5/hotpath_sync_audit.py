#!/usr/bin/env python3 -I
"""G3 trial-5 executed audit: blocking D2H sites + fast-plan install coverage
in the flashinfer attention backend (the SWA/Spark hot path).

Answers three falsifiable questions against the REAL source tree (ast-extracted,
line-anchored; no reimplementation):
  Q1  Which call sites in flashinfer_backend.py can force a blocking device read
      (.item()/.cpu()/.tolist()/synchronize/nonzero), and which of those sit on
      per-step updater hot paths vs cold (init/capture/error) paths?
  Q2  Which receiver tensors at those sites are host mirrors (name-carried *_cpu
      provenance => no sync) vs device tensors (=> sync)?
  Q3  Which wrapper groups get begin_forward replaced by fast_decode_plan /
      fast_prefill_plan at capture, and under what guards?  In particular: do
      full_cg_prefill_wrappers (plain EXTEND replays) EVER get a fast plan?

Raw JSON out: g3-t5/raw/sync_audit.json.  Stdout: human summary.
"""
import ast, json, os, sys

SRC = "python/sglang/srt/layers/attention/flashinfer_backend.py"
OUT = "g3-t5/raw/sync_audit.json"

BLOCKING_ATTRS = {"item", "cpu", "tolist", "synchronize", "nonzero"}
# Functions/classes that run per forward step (decode/extend update + replay
# metadata + plan shims). Everything else is treated as cold.
HOT_CLASSES = {
    "FlashInferIndicesUpdaterDecode",
    "FlashInferIndicesUpdaterPrefill",
}
HOT_FUNCS = {
    "fast_decode_plan",
    "fast_prefill_plan",
    "call_begin_forward",
    "update",
    "update_single_wrapper",
    "update_sliding_window",
    "update_cross_attention",
    "init_forward_metadata",
    "init_forward_metadata_capture_cuda_graph",
    "init_forward_metadata_replay_cuda_graph",
    "init_forward_metadata_out_graph",
}

HOST_HINTS = ("_cpu", "seq_lens_cpu", "prefix_lens_cpu", "extend_lens_cpu",
              "kv_start_cpu", "host_lengths", "indptr_cpu")


def receiver_src(call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return ast.unparse(f.value)
    return ast.unparse(f)


def enclosing(node, tree):
    """Return (class_name, func_name) chain for a node."""
    cls = func = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(n):
                if child is node:
                    if isinstance(n, ast.ClassDef):
                        cls = n.name
                    else:
                        func = n.name
    return cls, func


def guard_text(assign_node, tree):
    """Source text of the nearest enclosing If conditions (up to 3 levels)."""
    conds = []
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            for child in ast.walk(n):
                if child is assign_node:
                    conds.append(ast.unparse(n.test))
    return conds[:3]


def main():
    src = open(SRC).read()
    tree = ast.parse(src)
    lines = src.splitlines()

    # Q1+Q2: blocking call sites
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKING_ATTRS:
                cls, func = enclosing(node, tree)
                recv = receiver_src(node)
                is_host = any(h in recv for h in HOST_HINTS)
                hot = (cls in HOT_CLASSES) or (func in HOT_FUNCS)
                sites.append({
                    "line": node.lineno,
                    "attr": node.func.attr,
                    "receiver": recv,
                    "class": cls,
                    "function": func,
                    "hot_path": hot,
                    "host_mirror": is_host,
                    "verdict": (
                        "host-mirror-no-sync" if is_host else
                        ("DEVICE-READ-ON-HOT-PATH" if hot else "device-read-cold-path")
                    ),
                    "src": lines[node.lineno - 1].strip()[:160],
                })
    sites.sort(key=lambda s: (not s["hot_path"], s["line"]))

    # Q3: fast-plan installations
    installs = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "begin_forward"):
            value = node.value
            fname = None
            if isinstance(value, ast.Call) and value.args:
                f = value.func
                is_partial = (isinstance(f, ast.Name) and f.id == "partial") or (
                    isinstance(f, ast.Attribute) and f.attr == "partial")
                if is_partial:
                    a0 = value.args[0]
                    if isinstance(a0, ast.Name):
                        fname = a0.id
                    elif isinstance(a0, ast.Attribute):
                        fname = a0.attr
            target_src = ast.unparse(node.targets[0].value)
            # nearest enclosing for-loop iterable (which wrapper list is patched)
            loop_src = None
            for n in ast.walk(tree):
                if isinstance(n, ast.For):
                    for child in ast.walk(n):
                        if child is node:
                            loop_src = ast.unparse(n.iter)
            installs.append({
                "line": node.lineno,
                "wrapper_expr": target_src,
                "wrappers_iterated": loop_src,
                "plan_fn": fname,
                "guards": guard_text(node, tree),
                "src": lines[node.lineno - 1].strip()[:160],
            })
    installs.sort(key=lambda i: i["line"])

    full_cg_covered = [i for i in installs
                       if "full_cg_prefill" in (i["wrapper_expr"] or "")
                       or "full_cg_prefill" in (i["wrappers_iterated"] or "")]
    verdicts = {
        "hot_path_device_reads": [s for s in sites if s["verdict"] == "DEVICE-READ-ON-HOT-PATH"],
        "full_cg_prefill_fast_plan_installs": full_cg_covered,
        "Q3_full_cg_ever_fast": bool(full_cg_covered),
        "install_count": len(installs),
        "site_count": len(sites),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"sites": sites, "installs": installs, "verdicts": verdicts},
                  f, indent=1)

    print(f"sites={len(sites)} installs={len(installs)}")
    print("hot-path device reads (blocking candidates):")
    for s in verdicts["hot_path_device_reads"]:
        print(f"  L{s['line']:>5} {s['class']}.{s['function']} .{s['attr']}({s['receiver'][:60]})")
        print(f"         {s['src'][:140]}")
    print("fast-plan installs:")
    for i in installs:
        print(f"  L{i['line']:>5} {str(i['plan_fn']):<18} on {i['wrapper_expr'][:60]}")
        for g in i["guards"]:
            print(f"         guard: {g[:150]}")
    print(f"Q3: full_cg_prefill wrappers EVER get fast plan -> {verdicts['Q3_full_cg_ever_fast']}")


if __name__ == "__main__":
    main()

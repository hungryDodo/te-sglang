#!/usr/bin/env python3
"""G3-t5: execute the shipped MMVQ/MMQ selection contract, and audit the
dispatch tracer that any future F3 (f263262) experiment must rely on.

No torch, no gguf package, no GPU.  This script extracts the *real* source
nodes from the candidate tree (and from `f263262^`) with `ast`, compiles and
executes them against a shim WeightType enum built from the names the module
itself references, and compares dispatch decisions over the full grid.

It then extracts the real tracer block and simulates a short probe that ends
abruptly (no atexit): this is the failure mode that would make the
pre-registered GGUF experiment return INCONCLUSIVE with no dispatch evidence.

Usage: python3 -I verify_dispatch_contract.py [out.json] [--label NAME]
"""

import ast
import atexit
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types
from enum import IntEnum

REPO = pathlib.Path(__file__).resolve().parents[4]
GGUF_REL = "python/sglang/srt/layers/quantization/gguf.py"
GGUF = REPO / GGUF_REL
PARENT_REV = "f263262^"
PROBE_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 64]
GATE_BATCHES = [16, 32, 64, 128, 256]  # every frozen gate point has bs >= 16
ROWS_GRID = [1, 1024, 4096, 5120, 5121, 8192, 16384]
BATCH_GRID = list(range(0, 258))


def sha256_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def parse(src, label):
    return ast.parse(src, filename=label)


def module_assign(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return node
    raise KeyError(f"module assignment {name!r} not found")


def top_level_binding(tree, name):
    """Assignment node for `name`, or the top-level Try that binds it."""
    try:
        return module_assign(tree, name)
    except KeyError:
        pass
    for node in tree.body:
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in sub.targets
                ):
                    return node
    raise KeyError(f"module binding {name!r} not found")


def find_func(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"function {name!r} not found")


def weight_type_names(*trees):
    names = set()
    for tree in trees:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "WeightType"
            ):
                names.add(node.attr)
    return sorted(names)


def make_shim(names):
    # Distinct ints; the code under test only compares enum members and
    # converts ints via WeightType(int), where unknown ints must raise.
    return IntEnum("WeightType", {n: 1000 + i for i, n in enumerate(names)})


def exec_nodes(nodes, ns, label):
    mod = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, f"<{label}>", "exec")
    exec(code, ns)  # noqa: S102 - executing the real shipped expressions
    return ns


def extract_contract(src, label, parent_src=None):
    """Return (policy, ns) for one source version."""
    tree = parse(src, label)
    names = weight_type_names(*([parse(parent_src, "parent")] if parent_src else []), tree)
    WeightType = make_shim(names)
    ns = {"WeightType": WeightType, "__builtins__": __builtins__}
    # quant-type sets (unchanged between parent and child)
    sets = {}
    for name in (
        "STANDARD_QUANT_TYPES",
        "KQUANT_TYPES",
        "IMATRIX_QUANT_TYPES",
        "MMVQ_QUANT_TYPES",
        "MMQ_QUANT_TYPES",
        "DEQUANT_TYPES",
        "UNQUANTIZED_TYPES",
    ):
        exec_nodes([module_assign(tree, name)], ns, label)
        sets[name] = set(ns[name])
    # the measured table + predicate (shipped only)
    pred = None
    measured = None
    if "def _mmvq_safe_batch" in src:
        exec_nodes([find_func(tree, "_mmvq_safe_batch")], ns, label)
        pred = ns["_mmvq_safe_batch"]
        exec_nodes([module_assign(tree, "_MMVQ_SAFE_MEASURED")], ns, label)
        measured = {m.name: list(v) for m, v in ns["_MMVQ_SAFE_MEASURED"].items()}
    return tree, ns, sets, pred, measured, WeightType


def legacy_safe_stmt(parent_tree):
    """The two-assignment threshold rule from f263262^'s fused_mul_mat_gguf."""
    fn = find_func(parent_tree, "fused_mul_mat_gguf")
    for node in fn.body:
        if isinstance(node, ast.If) and _mentions(node.test, "IMATRIX_QUANT_TYPES"):
            return node
    raise KeyError("legacy threshold if-statement not found")


def _mentions(node, name):
    return any(
        isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)
    )


def branch_tests(fn):
    """Source-normalised dispatch branch tests, in order, from a function."""
    out = []
    for node in fn.body:
        if not isinstance(node, ast.If) or not _mentions(node.test, "qweight_type"):
            continue
        # skip the threshold `if` that assigns mmvq_safe (moved into
        # _mmvq_safe_batch by f263262); only the dispatch rule is compared
        if any(
            isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "mmvq_safe" for t in s.targets)
            for s in ast.walk(node)
        ):
            continue
        out.append(ast.dump(node.test))
    return out


class FakeWeight:
    def __init__(self, rows):
        self.shape = (rows, 16)


def dispatch(batch, wt, safe, sets):
    """The unchanged branch order of fused_mul_mat_gguf (see branch_tests)."""
    if batch == 0:
        return "empty"
    if wt in sets["UNQUANTIZED_TYPES"]:
        return "unquantized"
    if batch <= safe and wt in sets["MMVQ_QUANT_TYPES"]:
        return "mmvq"
    if wt in sets["MMQ_QUANT_TYPES"]:
        return "mmq"
    if wt in sets["DEQUANT_TYPES"]:
        return "dequant"
    return "raise"


def run_tracer_simulation(rows_calls=100, env_every="64", trace_path=None):
    """Execute the real tracer block + record/dump functions from the source.

    Simulates `rows_calls` dispatch records and an abrupt end (no atexit, i.e.
    SIGKILL / SIGTERM without graceful shutdown).  Returns a dict.
    """
    src = GGUF.read_text()
    tree = parse(src, "gguf.py")
    names = weight_type_names(tree)
    WeightType = make_shim(names)
    nodes = [
        module_assign(tree, "_TRACE_ENV"),
        module_assign(tree, "_DISPATCH_TRACE_FILE"),
        module_assign(tree, "_dispatch_counts"),
        module_assign(tree, "_dispatch_calls"),
        module_assign(tree, "_dispatch_lock"),
        top_level_binding(tree, "_DISPATCH_DUMP_EVERY"),
        find_func(tree, "_dispatch_type_name"),
        find_func(tree, "_record_dispatch"),
        find_func(tree, "dump_gguf_dispatch_trace"),
    ]
    # module-level `if _TRACE_ENV ... : _DISPATCH_TRACE_FILE = ...` block
    for node in tree.body:
        if isinstance(node, ast.If) and _mentions(node.test, "_TRACE_ENV"):
            nodes.append(node)
    # `if _DISPATCH_TRACE_FILE is not None: atexit.register(...)`
    for node in tree.body:
        if isinstance(node, ast.If) and _mentions(node.test, "_DISPATCH_TRACE_FILE"):
            nodes.append(node)

    old = os.environ.get("SGLANG_GGUF_DISPATCH_TRACE")
    old_every = os.environ.get("SGLANG_GGUF_DISPATCH_TRACE_EVERY")
    os.environ["SGLANG_GGUF_DISPATCH_TRACE"] = str(trace_path)
    if env_every is None:
        os.environ.pop("SGLANG_GGUF_DISPATCH_TRACE_EVERY", None)
    else:
        os.environ["SGLANG_GGUF_DISPATCH_TRACE_EVERY"] = env_every
    try:
        ns = {
            "WeightType": WeightType,
            "json": json,
            "os": os,
            "threading": __import__("threading"),
            "atexit": atexit,
            "Optional": __import__("typing").Optional,
            "Counter": __import__("collections").Counter,
            "__builtins__": __builtins__,
        }
        exec_nodes(nodes, ns, "tracer")
        fn = ns["_record_dispatch"]
        for i in range(rows_calls):
            # alternate the two measured types so the table is not degenerate
            wt = WeightType.Q8_0 if i % 2 else WeightType.Q4_K
            fn(wt, 4096, 4, "mmq", 2)
        atexit.unregister(ns["dump_gguf_dispatch_trace"])  # simulate abrupt end
        exists = os.path.exists(str(trace_path))
        payload = None
        if exists:
            payload = json.loads(pathlib.Path(trace_path).read_text())
        return {
            "source_dump_every": ns["_DISPATCH_DUMP_EVERY"],
            "calls": rows_calls,
            "trace_file_exists_without_atexit": exists,
            "cells_recorded": None if payload is None else len(payload["cells"]),
            "trace_calls": None if payload is None else payload["trace_calls"],
        }
    finally:
        if old is None:
            os.environ.pop("SGLANG_GGUF_DISPATCH_TRACE", None)
        else:
            os.environ["SGLANG_GGUF_DISPATCH_TRACE"] = old
        if old_every is None:
            os.environ.pop("SGLANG_GGUF_DISPATCH_TRACE_EVERY", None)
        else:
            os.environ["SGLANG_GGUF_DISPATCH_TRACE_EVERY"] = old_every


def main(argv):
    out = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path("dispatch-contract.json")
    label = argv[3] if len(argv) > 3 and argv[2] == "--label" else "shipped"

    src = GGUF.read_text()
    parent_src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{PARENT_REV}:{GGUF_REL}"],
        check=True, capture_output=True, text=True,
    ).stdout
    tree, ns, sets, pred, measured, WeightType = extract_contract(src, "shipped", parent_src)
    ptree, pns, psets, _, _, _ = extract_contract(parent_src, "parent", parent_src)

    # The branch rule itself must be identical between parent and child;
    # f263262 is supposed to change only the threshold computation.
    child_branches = branch_tests(find_func(tree, "fused_mul_mat_gguf"))
    parent_branches = branch_tests(find_func(ptree, "fused_mul_mat_gguf"))

    types_grid = sorted(
        set(sets["STANDARD_QUANT_TYPES"])
        | set(sets["KQUANT_TYPES"])
        | set(sets["IMATRIX_QUANT_TYPES"])
        | set(sets["UNQUANTIZED_TYPES"])
        | set(ns.get("_MMVQ_SAFE_MEASURED", {}))
    )
    changed = []          # kernel-selection changes (the contract that matters)
    threshold_only = []   # mmvq_safe value differs but the chosen kernel does not
    for wt in types_grid:
        for rows in ROWS_GRID:
            # legacy: execute the real `if` statement from f263262^
            ln = dict(pns)
            ln.update({"qweight_type": wt, "qweight": FakeWeight(rows)})
            exec_nodes([legacy_safe_stmt(ptree)], ln, "legacy")
            old_safe = ln["mmvq_safe"]
            new_safe = pred(int(wt), rows) if pred else None
            for batch in BATCH_GRID:
                old_k = dispatch(batch, wt, old_safe, sets)
                new_k = dispatch(batch, wt, new_safe, sets)
                rec = {
                    "type": wt.name,
                    "rows": rows,
                    "band": "rows>5120" if rows > 5120 else "rows<=5120",
                    "batch": batch,
                    "old_mmvq_safe": old_safe,
                    "new_mmvq_safe": new_safe,
                    "old_kernel": old_k,
                    "new_kernel": new_k,
                }
                if old_k != new_k:
                    changed.append(rec)
                elif old_safe != new_safe:
                    threshold_only.append(rec)

    # Unknown integer type (legacy ValueError fallback) must be unchanged.
    unknown = []
    for rows in ROWS_GRID:
        ln = dict(pns)
        ln.update({"qweight_type": 99999, "qweight": FakeWeight(rows)})
        exec_nodes([legacy_safe_stmt(ptree)], ln, "legacy")
        try:
            new_safe = pred(99999, rows)
        except ValueError as e:  # pragma: no cover - would be a shipped bug
            new_safe = f"ValueError:{e}"
        unknown.append(
            {"rows": rows, "old_mmvq_safe": ln["mmvq_safe"], "new_mmvq_safe": new_safe}
        )

    changed_types = sorted({c["type"] for c in changed})
    changed_batches = sorted({c["batch"] for c in changed})
    changed_bands = sorted({c["band"] for c in changed})
    at_gate = [c for c in changed if c["batch"] in GATE_BATCHES]
    at_or_above_16 = [c for c in changed if c["batch"] >= 16]

    # measured-support: the shipped table documents per-type band support
    # (Q8_0 2/2, Q4_K 2/2, Q6_K 6/2).  g2-t1/g2-t2 raw crossover artifacts are
    # NOT in this repository snapshot; report that as a provenance limitation.
    g2_present = {}
    for rel in (
        "seis/trials/deepseek-b/g2-t1/crossover.json",
        "seis/trials/deepseek-b/g2-t2/verify_selection.py",
    ):
        g2_present[rel] = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"HEAD:{rel}"],
            capture_output=True,
        ).returncode == 0

    # tracer audit: 100 dispatch records, then an abrupt end (no atexit)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tracer = run_tracer_simulation(
            rows_calls=100, trace_path=os.path.join(td, "trace.json")
        )
        tracer["post_patch_trace_survives_abrupt_end"] = tracer[
            "trace_file_exists_without_atexit"
        ]
        tracer["pre_patch_expectation"] = (
            "with dump_every=1024 a 100-call probe writes nothing without atexit"
        )
        tracer["residual_loss_bound_calls"] = tracer["source_dump_every"] - 1
        fallback = run_tracer_simulation(
            rows_calls=8,
            env_every="not_an_int",
            trace_path=os.path.join(td, "fallback.json"),
        )
        tracer["invalid_env_falls_back"] = fallback["source_dump_every"] == 64

    # module-level tracer name resolution: the block must be import-safe even
    # when tracing is off, because quantization/__init__ imports this module
    # unconditionally on every startup (BF16 gate included).
    init_src = (
        REPO / "python/sglang/srt/layers/quantization/__init__.py"
    ).read_text()
    init_tree = parse(init_src, "__init__.py")
    unconditional_import = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "sglang.srt.layers.quantization.gguf"
        for n in init_tree.body
    )
    syntax_ok = True
    try:
        compile(src, "gguf.py", "exec")
    except SyntaxError:
        syntax_ok = False

    result = {
        "label": label,
        "source": {
            "path": str(GGUF),
            "sha256": sha256_text(src),
            "parent_rev": PARENT_REV,
            "parent_sha256": sha256_text(parent_src),
            "compile_ok": syntax_ok,
        },
        "branch_rule_identical_parent_vs_shipped": child_branches == parent_branches,
        "branch_tests_shipped": child_branches,
        "measured_table": measured,
        "type_set_sizes": {k: len(v) for k, v in sets.items()},
        "grid": {
            "types": len(types_grid),
            "rows": ROWS_GRID,
            "batches": f"0..{BATCH_GRID[-1]}",
        },
        "changed_cell_count": len(changed),
        "threshold_only_cell_count": len(threshold_only),
        "threshold_only_types": sorted({c["type"] for c in threshold_only}),
        "changed_types": changed_types,
        "changed_bands": changed_bands,
        "changed_batches": changed_batches,
        "changed_cells_at_gate_batches": len(at_gate),
        "changed_cells_at_batch_ge_16": len(at_or_above_16),
        "changed_cell_classes": sorted(
            {f"{c['type']}|{c['band']}|batch={c['batch']}" for c in changed}
        ),
        "changed_cells": changed,
        "unknown_int_type_unchanged": all(
            u["old_mmvq_safe"] == u["new_mmvq_safe"] for u in unknown
        ),
        "unknown_int_rows": unknown,
        "provenance": {
            "g2_raw_artifacts_present_in_this_snapshot": g2_present,
            "note": (
                "the measured table's supporting crossover artifacts are not in "
                "this candidate tree or in any ref of this snapshot; the values "
                "are taken as branch-learning/controller evidence, not "
                "re-verified here"
            ),
        },
        "tracer_audit": {
            "records_per_call_site": "3 (empty, unquantized, post-branch)",
            "traced_entry_point": "fused_mul_mat_gguf",
            "untraced_entry_point": "fused_moe_gguf (MoE only; Spark2_5 uses a dense MLP, so N/A for the Spark GGUF probe)",
            "module_imported_unconditionally_by_quantization_init": unconditional_import,
            "simulation": tracer,
            "lifecycle": (
                "dump every _DISPATCH_DUMP_EVERY calls + atexit; an abrupt "
                "SIGKILL/SIGTERM end loses the tail since the last dump"
            ),
        },
        "limits": [
            "CPU-only: executes shipped Python expressions, no torch/gguf/GPU; "
            "does not prove which kernel a live engine runs.",
            "WeightType is a shim built from the names this module references; "
            "the real enum's unused names fall through to the same default rule.",
            "No performance number is produced or implied.",
        ],
    }
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "label", "changed_cell_count", "changed_types", "changed_bands",
        "changed_batches", "changed_cells_at_gate_batches",
        "branch_rule_identical_parent_vs_shipped",
        "unknown_int_type_unchanged")}, indent=1))
    print("tracer_sim:", json.dumps(result["tracer_audit"]["simulation"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

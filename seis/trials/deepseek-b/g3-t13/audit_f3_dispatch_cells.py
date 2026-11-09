#!/usr/bin/env python3
"""G3 t13: replay the real Q8_0 GGUF dispatch trace against f263262 and its parent.

Inputs (read-only):
  * current python/sglang/srt/layers/quantization/gguf.py (patched, HEAD)
  * seis/trials/deepseek-b/g3-t13/legacy_gguf_f263262_parent.py.txt
    (verbatim `git show f263262^:...gguf.py`)
  * controller_evidence/continuous-20261109-01/20261110-runtime/gguf-dispatch.json
    (controller-supplied real per-cell dispatch counts; NOT re-run here)

The deciding source is ast-extracted and EXECUTED, never re-implemented:
  current: _MMVQ_SAFE_MEASURED, _mmvq_safe_batch, MMVQ_/MMQ_/IMATRIX_QUANT_TYPES
  legacy : the `mmvq_safe = ...` If-block inside fused_mul_mat_gguf (f263262^)

`WeightType` is an external enum (gguf.GGMLQuantizationType, not importable in
sandbox); a name-preserving shim keeps every comparison by name exact for the
types present in the trace.  Hard exit 2 on any anchor drift.
"""

import ast
import json
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "f3-dispatch-audit.json"
CUR = f"{ROOT}/python/sglang/srt/layers/quantization/gguf.py"
LEG = f"{ROOT}/seis/trials/deepseek-b/g3-t13/legacy_gguf_f263262_parent.py.txt"
TRACE = (
    f"{ROOT}/controller_evidence/continuous-20261109-01/20261110-runtime/"
    "gguf-dispatch.json"
)


class _WeightTypeShim:
    def __getattr__(self, name):
        return name

    def __call__(self, value):
        return value


def fail(msg):
    print(f"HARNESS_ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def module_ast(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return ast.parse(fh.read(), filename=path)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


cur_tree = module_ast(CUR)
leg_tree = module_ast(LEG)

cur_src = open(CUR, encoding="utf-8").read()
leg_src = open(LEG, encoding="utf-8").read()


def find_assign(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node
    return None


ns = {"WeightType": _WeightTypeShim(), "Optional": object}
anchors = {}
for name in (
    "STANDARD_QUANT_TYPES",
    "KQUANT_TYPES",
    "IMATRIX_QUANT_TYPES",
    "MMVQ_QUANT_TYPES",
    "MMQ_QUANT_TYPES",
    "_MMVQ_SAFE_MEASURED",
):
    node = find_assign(cur_tree, name)
    if node is None:
        fail(f"missing anchor assignment {name} in current gguf.py")
    exec(compile(ast.Module(body=[node], type_ignores=[]), CUR, "exec"), ns)
    anchors[name] = True

fn = None
for node in cur_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_mmvq_safe_batch":
        fn = node
        break
if fn is None:
    fail("missing _mmvq_safe_batch in current gguf.py")
exec(compile(ast.Module(body=[fn], type_ignores=[]), CUR, "exec"), ns)
new_safe_batch = ns["_mmvq_safe_batch"]
anchors["_mmvq_safe_batch"] = True

# legacy: the first If inside fused_mul_mat_gguf whose body assigns mmvq_safe
legacy_fn = None
for node in leg_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "fused_mul_mat_gguf":
        legacy_fn = node
        break
if legacy_fn is None:
    fail("missing fused_mul_mat_gguf in legacy source")
threshold_if = None
for node in ast.walk(legacy_fn):
    if isinstance(node, ast.If):
        assigned = [
            t.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Assign)
            for t in sub.targets
            if isinstance(t, ast.Name)
        ]
        if "mmvq_safe" in assigned:
            threshold_if = node
            break
if threshold_if is None:
    fail("missing mmvq_safe If-block in legacy fused_mul_mat_gguf")
import textwrap

_block = "\n".join(
    leg_src.splitlines()[threshold_if.lineno - 1 : threshold_if.end_lineno]
)
seg = textwrap.dedent(_block)
if "mmvq_safe" not in seg:
    fail("cannot extract legacy threshold If source")
wrapper = (
    "def legacy_mmvq_safe(qweight, qweight_type):\n"
    + "\n".join("    " + line for line in seg.splitlines())
    + "\n    return mmvq_safe\n"
)
exec(compile(wrapper, LEG, "exec"), ns)
legacy_safe_batch_raw = ns["legacy_mmvq_safe"]
anchors["legacy_fused_mul_mat_gguf.mmvq_safe_if"] = True


class _QWeight:
    def __init__(self, rows):
        self.shape = (rows, 0)


def legacy_safe(qtype_name, rows):
    return int(legacy_safe_batch_raw(_QWeight(rows), qtype_name))


def new_safe(qtype_name, rows):
    return int(new_safe_batch(qtype_name, rows))


def selected_kernel(qtype_name, rows, batch):
    safe = new_safe(qtype_name, rows)  # current dispatch uses the patched rule
    if batch <= safe and qtype_name in ns["MMVQ_QUANT_TYPES"]:
        return "mmvq"
    if qtype_name in ns["MMQ_QUANT_TYPES"]:
        return "mmq"
    return "dequant"


def legacy_kernel(qtype_name, rows, batch):
    safe = legacy_safe(qtype_name, rows)
    if batch <= safe and qtype_name in ns["MMVQ_QUANT_TYPES"]:
        return "mmvq"
    if qtype_name in ns["MMQ_QUANT_TYPES"]:
        return "mmq"
    return "dequant"


trace = json.load(open(TRACE, encoding="utf-8"))
cells = trace["cells"]
rows = []
tot_calls = tot_mismatch = tot_flip = 0
flip_detail = {}
for cell in cells:
    qtype = cell["weight_type"]
    band = cell["rows_band"]
    rows_n = 5120 if band == "rows<=5120" else 5121
    batch = int(cell["batch"])
    count = int(cell["count"])
    sel_new = selected_kernel(qtype, rows_n, batch)
    sel_leg = legacy_kernel(qtype, rows_n, batch)
    mismatch = sel_new != cell["kernel"]
    flip = sel_new != sel_leg
    tot_calls += count
    tot_mismatch += count if mismatch else 0
    if flip:
        tot_flip += count
        flip_detail[f"{qtype}|{band}|batch{batch}"] = {
            "calls": count,
            "legacy": sel_leg,
            "patched": sel_new,
        }
    rows.append(
        {
            "weight_type": qtype,
            "rows_band": band,
            "batch": batch,
            "calls": count,
            "observed_kernel": cell["kernel"],
            "patched_safe": new_safe(qtype, rows_n),
            "patched_kernel": sel_new,
            "legacy_safe": legacy_safe(qtype, rows_n),
            "legacy_kernel": sel_leg,
            "mismatch_vs_patched": mismatch,
            "selection_flip": flip,
        }
    )

result = {
    "inputs": {
        "current_gguf_py": CUR,
        "legacy_source": LEG,
        "trace": TRACE,
        "trace_calls": trace.get("trace_calls"),
        "anchors_executed": sorted(anchors),
    },
    "cells": rows,
    "totals": {
        "recorded_calls": tot_calls,
        "calls_where_patched_kernel_differs_from_observed": tot_mismatch,
        "calls_where_f263262_changes_selection": tot_flip,
        "flip_share_of_recorded_calls": round(tot_flip / tot_calls, 4)
        if tot_calls
        else None,
    },
    "flip_detail": flip_detail,
    "weight_types_present_in_trace": sorted({c["weight_type"] for c in cells}),
    "limits": [
        "Replay of controller-supplied trace cells; no GPU/kernel was run here.",
        "Trace covers Q8_0 only: Q4_K changed cells (rows<=5120, batch 3-6) are "
        "NOT exercised by this trace and remain unmeasured.",
        "Instrumented trace perturbs wall time (canonical C22): no speed claim.",
        "No numeric-equivalence check; F3 stays UNGATED under the frozen BF16 gate.",
        "Batch is the traced matmul row count; per-step tail batches are not "
        "characterised by this table.",
    ],
}
verdict = (
    "DISPATCH_CONTRACT_HELD_ON_TRACE"
    if tot_mismatch == 0 and tot_calls > 0
    else "DISPATCH_CONTRACT_VIOLATED_ON_TRACE"
)
result["verdict"] = verdict
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=1)
    fh.write("\n")
print(f"verdict={verdict} calls={tot_calls} flips={tot_flip} mismatches={tot_mismatch}")
print(json.dumps(flip_detail, indent=1))
sys.exit(0 if verdict == "DISPATCH_CONTRACT_HELD_ON_TRACE" else 1)

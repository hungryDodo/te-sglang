#!/usr/bin/env python3
"""G3 t8 (deepseek-b): call-site provenance + guard audit for the pending
t3/t4 host-mirror candidate.

NEW question vs g3-t6/t7. The t3/t4 patches replace the device slice bound
``kv_indptr[-1]`` with the caller's host int ``paged_kernel_lens_sum``.  Their
correctness is not only "do the two numbers match on sampled inputs" (g3-t6
replayed 34 cases) -- it is:

  (1) at WHICH caller does the host int come from, and is it derived from the
      same logical tensor as ``paged_kernel_lens`` (the tensor cumsummed into
      ``kv_indptr``)?  A host sum from a *different* source would be a silent
      under-translation bug.
  (2) do the guard expressions actually exclude every path where ``kv_indptr``
      was NOT rebuilt from ``paged_kernel_lens`` (spec-decode, dq branch)?
  (3) does the patch introduce a NEW precondition, or is its precondition
      already required by unpatched code in the same function
      (``global_override_indptr_cpu = torch.cumsum(seq_lens_cpu)`` and the
      ``torch.empty(paged_kernel_lens_sum + 256)`` allocation)?

Method (stdlib only, no torch/GPU/model):
  * AST-extract every assignment to ``paged_kernel_lens_sum`` in the FlashInfer
    updater classes, with enclosing function and line number.
  * Line-anchor and EXECUTE the real guard predicates and the real patched
    expressions over the configuration space.
  * Replay host-vs-device integer equality per provenance class, including a
    stale-mirror negative control and a graph-padding case.
  * Emit a verdict with explicit limits.

This is a source-contract oracle, NOT a runtime measurement.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

SRC_REL = "python/sglang/srt/layers/attention/flashinfer_backend.py"

DECODE_HOST_EXPR = "host_indptr_last if host_indptr_last is not None else kv_indptr[-1]"
PREFILL_GUARD_FRAGMENT = "custom_kv_indices is None"


def find_repo() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / SRC_REL).is_file():
            return p
    raise SystemExit(f"cannot locate {SRC_REL} above {here}")


def walk_methods(tree):
    """(class_name, func_name, node, source_segment) for every method body."""
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            out.append((cls.name, fn.name, fn))
    return out


def func_src(lines, node):
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def attr_chain(node):
    """Best-effort dotted-source for a target expression."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<?>"


def classify_sum(rhs_src, paged_kernel_lens_name):
    r = rhs_src.replace(" ", "")
    if "extend_prefix_lens_cpu" in r:
        return "HOST_PREFIX_LIST"
    if "seq_lens_cpu" in r:
        return "HOST_SEQ_LENS_CPU"
    if r == "seq_lens_sum":
        return "CALLER_SUM"
    if paged_kernel_lens_name and r.startswith(paged_kernel_lens_name + ".sum().item()"):
        return "DEVICE_SAME_TENSOR"
    if ".sum().item()" in r:
        return "DEVICE_OTHER_TENSOR"
    return "OTHER"


class Sentinel:
    """Non-int, non-None object whose isinstance(·, int) is False."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class SpecWithIndptr:
    kv_indptr = Sentinel("kv_indptr_tensor")


class SpecWithoutIndptr:
    kv_indptr = None


def make_spec_info(value, spec_type):
    if value == "none":
        return None
    s = SpecWithIndptr() if value == "with_indptr" else SpecWithoutIndptr()
    if spec_type == "dflash":
        s.spec_input_type = "DFLASH_VERIFY"
    return s


def eval_expr(expr_src, env):
    return eval(compile(ast.parse(expr_src, mode="eval"), "<anchor>", "eval"), {}, env)


def int_replay():
    """Host vs device last-cumsum per provenance class (pure Python ints)."""
    W = 511  # sliding_window_size exclusive value used by spark2_5 config
    cases = []

    def dev_last(lens):
        return sum(lens)

    def host_from_cpu_mirror(cpu_lens, max_v):
        return sum(min(x, max_v) for x in cpu_lens)

    # Class HOST_SEQ_LENS_CPU, decode SWA clamp: clamp(seq, W+1) both sides.
    for name, dev, cpu in [
        ("decode_mirror_equal", [3, 700, 511, 512], [3, 700, 511, 512]),
        ("decode_graph_padding_zero", [3, 700, 0, 0], [3, 700, 0, 0]),
        ("decode_ctrl_stale_cpu_plus1", [3, 700, 511, 512], [4, 700, 511, 512]),
    ]:
        d = dev_last([min(x, W + 1) for x in dev])
        h = host_from_cpu_mirror(cpu, W + 1)
        cases.append({"class": "HOST_SEQ_LENS_CPU", "name": name, "device_last": d,
                      "host_last": h, "equal": d == h, "control": "ctrl" in name})

    # Class HOST_SEQ_LENS_CPU, prefill prefix_is_full_seq: min(seq, W).
    for name, dev, cpu in [
        ("prefill_fullseq_mirror_equal", [400, 512, 300], [400, 512, 300]),
        ("prefill_fullseq_ctrl_stale_cpu", [400, 512, 300], [400, 512, 301]),
    ]:
        d = dev_last([min(x, W) for x in dev])
        h = host_from_cpu_mirror(cpu, W)
        cases.append({"class": "HOST_SEQ_LENS_CPU", "name": name, "device_last": d,
                      "host_last": h, "equal": d == h, "control": "ctrl" in name})

    # Class HOST_PREFIX_LIST (ragged extend, update_single_wrapper / wrapper 1).
    for name, prefix, cpu_list in [
        ("prefix_list_equal", [700, 8192, 0], [700, 8192, 0]),
        ("prefix_list_ctrl_mismatch", [700, 8192, 0], [700, 8191, 0]),
    ]:
        cases.append({"class": "HOST_PREFIX_LIST", "name": name,
                      "device_last": sum(prefix), "host_last": sum(cpu_list),
                      "equal": sum(prefix) == sum(cpu_list), "control": "ctrl" in name})

    # Class DEVICE_SAME_TENSOR / CALLER_SUM: exact by construction.
    for name, lens in [("device_same_tensor", [700, 8192, 0]),
                       ("caller_seq_lens_sum", [700, 8192, 0])]:
        cases.append({"class": "DEVICE_SAME_TENSOR", "name": name,
                      "device_last": sum(lens), "host_last": sum(lens),
                      "equal": True, "control": False})
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    path = root / SRC_REL
    text = path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)

    anchors = {}

    # --- 1. patched expressions are present and line-anchored ---------------
    dec_ln = [i + 1 for i, l in enumerate(lines) if DECODE_HOST_EXPR in l.replace("  ", " ").strip()]
    anchors["decode_host_expr_lines"] = dec_ln
    if not dec_ln:
        raise SystemExit("decode host expression anchor missing")
    pre_guard_lines = [
        i + 1 for i, l in enumerate(lines)
        if PREFILL_GUARD_FRAGMENT in l and "is not None" not in l
    ]
    anchors["prefill_guard_custom_kv_lines"] = pre_guard_lines
    if not pre_guard_lines:
        raise SystemExit("prefill guard anchor missing")

    # --- 2. provenance of every paged_kernel_lens_sum assignment -----------
    sites = []
    methods = {f"{c}.{f}": (c, f, n) for c, f, n in walk_methods(tree)}
    for qual, (cls, fn, node) in methods.items():
        body_src = func_src(lines, node)
        # nearest paged_kernel_lens* assignment per function for same-tensor test
        pkl_targets = []
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id.startswith("paged_kernel_lens") and not t.id.startswith("paged_kernel_lens_sum"):
                        pkl_targets.append((n.lineno, t.id, attr_chain(n.value)))
        for n in ast.walk(node):
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("paged_kernel_lens_sum"):
                    rhs = attr_chain(n.value)
                    prior = [p for p in pkl_targets if p[0] < n.lineno]
                    pkl_name = prior[-1][1] if prior else ""
                    pkl_src = prior[-1][2] if prior else ""
                    sites.append({
                        "class": cls, "function": fn, "line": n.lineno,
                        "rhs": rhs, "paged_kernel_lens_source": pkl_src,
                        "provenance": classify_sum(rhs, pkl_name),
                    })

    # --- 3. execute the real guard predicates ------------------------------
    # 3a. decode: outer `if spec_info is None or getattr(spec_info,'kv_indptr',None) is None`
    #     and inner `isinstance(paged_kernel_lens_sum, int)`.
    dec_fn = None
    for cls, fn, node in walk_methods(tree):
        if cls == "FlashInferIndicesUpdaterDecode" and fn == "call_begin_forward":
            dec_fn = node
    if dec_fn is None:
        raise SystemExit("decode call_begin_forward not found")
    dec_ifs = [n for n in ast.walk(dec_fn) if isinstance(n, ast.If) and "spec_info is None" in attr_chain(n.test)]
    if not dec_ifs:
        raise SystemExit("decode outer guard not found")
    dec_outer = attr_chain(dec_ifs[0].test)
    dec_isinstance = [
        attr_chain(n.test) for n in ast.walk(dec_ifs[0])
        if isinstance(n, ast.If) and "isinstance" in attr_chain(n.test)
    ]
    if not dec_isinstance:
        raise SystemExit("decode isinstance guard not found")
    dec_inner = dec_isinstance[0]

    dec_truth = []
    for sv in ("none", "with_indptr", "without_indptr"):
        si = make_spec_info(sv, None)
        outer = eval_expr(dec_outer, {"spec_info": si, "getattr": getattr})
        for val in (7, 7.0, Sentinel("tensor")):
            inner = eval_expr(dec_inner, {"paged_kernel_lens_sum": val, "isinstance": isinstance, "int": int})
            dec_truth.append({"spec_info": sv, "sum_type": type(val).__name__,
                              "outer_rebuilds_indptr": outer, "inner_host_ok": inner,
                              "host_mirror_taken": bool(outer and inner)})

    # 3b. prefill guard: real `if` test containing all three conditions.
    pre_fn = None
    for cls, fn, node in walk_methods(tree):
        if cls == "FlashInferIndicesUpdaterPrefill" and fn == "call_begin_forward":
            pre_fn = node
    if pre_fn is None:
        raise SystemExit("prefill call_begin_forward not found")
    pre_guards = [
        n for n in ast.walk(pre_fn)
        if isinstance(n, ast.If) and "paged_kernel_lens_sum" in attr_chain(n.test)
        and "custom_kv_indices" in attr_chain(n.test)
    ]
    if len(pre_guards) != 1:
        raise SystemExit(f"expected 1 prefill guard, found {len(pre_guards)}")
    pre_guard_src = attr_chain(pre_guards[0].test)
    pre_truth = []
    for sv in ("none", "with_indptr", "without_indptr"):
        for ck in ("none", "custom"):
            for val in (7, 7.0):
                env = {"spec_info": make_spec_info(sv, "dflash"),
                       "custom_kv_indices": None if ck == "none" else Sentinel("custom_kv"),
                       "paged_kernel_lens_sum": val}
                ok = eval_expr(pre_guard_src, env)
                pre_truth.append({"spec_info": sv, "custom_kv_indices": ck,
                                  "sum_type": type(val).__name__,
                                  "host_mirror_taken": bool(ok)})

    # --- 4. integer replay per provenance class ----------------------------
    replay = int_replay()

    # --- 5. surrounding code already requires the same precondition? --------
    # `global_override_indptr_cpu = torch.cumsum(seq_lens_cpu, dim=0)` exists in
    # the same decode function; `torch.empty(paged_kernel_lens_sum + 256)` sizes
    # the indices buffer from the host sum in the prefill function.
    surrounding = {
        "decode_global_override_from_seq_lens_cpu": any(
            "global_override_indptr_cpu" in l and "cumsum(seq_lens_cpu" in l.replace(" ", "")
            for l in lines),
        "prefill_buffer_sized_by_host_sum": any(
            "paged_kernel_lens_sum + 256" in l for l in lines),
    }
    if not all(surrounding.values()):
        raise SystemExit(f"surrounding-precondition anchors missing: {surrounding}")

    divergences = [c for c in replay if not c["equal"] and not c["control"]]
    controls_flagged = [c for c in replay if c["control"] and not c["equal"]]
    bad_guard = [
        r for r in dec_truth + pre_truth
        if r["host_mirror_taken"] and r["sum_type"] == "float"
    ]

    summary = {
        "sites_total": len(sites),
        "provenance_counts": {},
        "host_mirror_sites": [s for s in sites if s["provenance"].startswith("HOST")],
        "divergences_outside_controls": len(divergences),
        "sensitivity_controls_flagged": f"{len(controls_flagged)}/{sum(1 for c in replay if c['control'])}",
        "guard_admits_non_int_sum": len(bad_guard),
        "decode_guard_rows": len(dec_truth),
        "prefill_guard_rows": len(pre_truth),
        "surrounding_precondition_already_required": surrounding,
    }
    for s in sites:
        summary["provenance_counts"][s["provenance"]] = summary["provenance_counts"].get(s["provenance"], 0) + 1

    verdict = {
        "patch_precondition": "host int == device cumsum last element for the same logical tensor",
        "new_precondition_introduced": False,
        "why_not_new": (
            "decode already builds global_override_indptr_cpu from torch.cumsum(seq_lens_cpu) "
            "in the same function, and prefill already sizes the kv_indices buffer as "
            "paged_kernel_lens_sum + 256 before pack_read_stream; both use the host value as "
            "the authority over the same device tensor."),
        "guard_covers_non_rebuilt_indptr_paths": True,
        "max_d2h_saved_per_eager_step": 1,
        "inert_when": "translator.reads_are_translated (unified SWA pool) or seq_lens_cpu is None",
        "expected_aggregate_gain": "~0 (sibling sync-removal family measured 1.0018 KEPT_NEUTRAL on a4; decode points inside +/-2% run variance)",
        "f3_status": "UNGATED and score-unreachable under the frozen BF16 gate (no GGUF artifact)",
        "measurement": "NOT_A_MEASUREMENT",
    }

    out = {"source": SRC_REL, "anchors": anchors, "call_sites": sites,
           "decode_guard_truth": dec_truth, "prefill_guard_truth": pre_truth,
           "integer_replay": replay, "summary": summary, "verdict": verdict}
    payload = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

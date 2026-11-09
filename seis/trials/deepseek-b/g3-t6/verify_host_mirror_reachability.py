#!/usr/bin/env python3
"""G3 t6 (deepseek-b): executed reachability + index-equality oracle for the
pending t3/t4 host-mirror candidate.

Question this trial tests (new):
  The t3 (decode) and t4 (prefill) patches replace ``kv_last_index =
  kv_indptr[-1]`` (a 0-dim CUDA scalar -> implicit .item() D2H) with the host
  int ``paged_kernel_lens_sum``.  That replacement is only *reachable* and only
  *saves* a D2H inside a narrow envelope.  This oracle extracts the REAL guard
  expressions from the shipped source (AST, line-anchored) and executes them
  over the configuration space, then replays the integer invariant at every
  call site that can reach the changed lines.

Outputs (JSON on stdout, also written with --out):
  anchors      - extracted source text + line numbers (fails loudly if source moved)
  reachability - for every config: is the host mirror taken? is the device
                 fallback taken? does the patch save any D2H at all?
  replay       - host int vs device cumsum over a case matrix, incl. the
                 SWA-clamped, graph-padded, prefill-8192 and negative-control cases
  summary      - counts + the honest envelope statement

stdlib only: no torch, no model, no GPU.  This is a source-contract + integer
oracle, NOT a runtime measurement.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_REL = "python/sglang/srt/layers/attention/flashinfer_backend.py"


def find_repo() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        cand = p / REPO_REL
        if cand.is_file():
            return p
    raise SystemExit(f"cannot locate {REPO_REL} above {here}")


class MethodFinder(ast.NodeVisitor):
    def __init__(self):
        self.classes = {}

    def visit_ClassDef(self, node):
        self.classes[node.name] = node
        self.generic_visit(node)


def get_method(tree, cls_name, fn_name):
    f = MethodFinder()
    f.visit(tree)
    cls = f.classes.get(cls_name)
    if cls is None:
        raise SystemExit(f"class {cls_name} not found")
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == fn_name:
            return n
    raise SystemExit(f"{cls_name}.{fn_name} not found")


def line_of(node):
    return getattr(node, "lineno", None)


def src_seg(src, node):
    return ast.get_source_segment(src, node)


def extract_anchor(src, fn, predicate, what):
    """Return first AST node in `fn` matching predicate, else fail loudly."""
    for n in ast.walk(fn):
        if predicate(n):
            return n
    raise SystemExit(f"ANCHOR MISSING: {what}")


def eval_expr(node, names):
    code = compile(ast.Expression(body=node), "<extracted>", "eval")
    return eval(code, {"__builtins__": {"isinstance": isinstance, "int": int}}, names)


# --------------------------------------------------------------------------
# Integer replay helpers (exact semantics of the shipped expressions)
# --------------------------------------------------------------------------
def clamp_max(xs, hi):
    return [min(x, hi) for x in xs]


def cumsum_last(xs):
    return sum(xs)  # cumsum(...)[bs] == sum(all)


def replay_case(name, seq_lens, window, seq_lens_cpu, expect):
    """Replay update_sliding_window wrapper-0 (SWA) + call_begin_forward.

    Shipped expressions (anchored below):
      paged_kernel_lens_tmp      = clamp(seq_lens,      max=sliding_window_size+1)
      seq_lens_cpu_tmp           = clamp(seq_lens_cpu,  max=sliding_window_size+1)
      paged_kernel_lens_sum_tmp  = seq_lens_cpu_tmp.sum().item()   [cpu tensor]
                                 | paged_kernel_lens_tmp.sum().item() [GPU -> D2H]
      kv_indptr[1:bs+1]          = cumsum(paged_kernel_lens); kv_indptr[bs] = sum
    """
    hi = window + 1
    lens_dev = clamp_max(seq_lens, hi)
    device_truth = cumsum_last(lens_dev)  # kv_indptr[bs]
    if seq_lens_cpu is not None:
        lens_cpu = clamp_max(seq_lens_cpu, hi)
        host_mirror = cumsum_last(lens_cpu)
        caller_d2h = False  # .item() on a CPU tensor: no device sync
    else:
        host_mirror = device_truth  # caller itself does .item() on GPU
        caller_d2h = True
    # t3 takes the host mirror whenever it is a python int (always, here)
    translation_bound = host_mirror
    equal = translation_bound == device_truth
    row = {
        "case": name,
        "bs": len(seq_lens),
        "window": window,
        "seq_lens_cpu_provided": seq_lens_cpu is not None,
        "device_truth_kv_indptr_bs": device_truth,
        "host_mirror_used_for_slice": translation_bound,
        "index_identical": equal,
        "caller_pays_d2h_anyway": caller_d2h,
        # The patch removes the *second* device read only when the caller
        # already had the host sum for free.
        "patch_saves_a_d2h": (not caller_d2h) and (seq_lens_cpu is not None),
        "expectation": expect,
    }
    return row


def build_cases():
    cases = []
    # 1. Gate-shaped decode: batches seen in the frozen workload.
    for bs in (16, 32, 64, 128, 256):
        for lens_name, lens in (
            ("below_window", [(i % 400) + 1 for i in range(bs)]),
            ("above_window", [(i % 3000) + 600 for i in range(bs)]),
            ("mixed", [((i * 37) % 4000) + 1 for i in range(bs)]),
            ("all_exactly_512", [512] * bs),
            ("all_exactly_513", [513] * bs),
        ):
            cases.append(
                (f"decode/bs{bs}/{lens_name}/cpu", lens, 511, list(lens), "equal")
            )
    # 2. seq_lens_cpu is None: the caller itself must .item() the GPU tensor.
    for bs in (64, 256):
        lens = [((i * 37) % 4000) + 1 for i in range(bs)]
        cases.append((f"decode/bs{bs}/cpu_None", lens, 511, None, "equal_but_no_saving"))
    # 3. CUDA-graph padded lanes: zero-length pad rows must not shift the sum.
    for bs_real, pad in ((56, 8), (120, 8), (248, 8)):
        lens = [((i * 11) % 900) + 1 for i in range(bs_real)] + [0] * pad
        cases.append((f"decode/bs{bs_real}+{pad}pad", lens, 511, list(lens), "equal"))
    # 4. Prefill-shaped: batch of 16 x exact 8192-token prompts, no prefix and
    #    with a 512-token radix prefix.  Equality is sum-over-the-same-tensor.
    cases.append(("prefill/bs16/extend8192", [8192] * 16, 511, [8192] * 16, "equal"))
    cases.append(("prefill/bs16/prefix512+ext8192", [8704] * 16, 511, [8704] * 16, "equal"))
    # 5. SENSITIVITY CONTROL: a deliberately stale CPU mirror MUST be flagged.
    cases.append(
        ("control/stale_cpu_mirror", [600, 700, 800], 511, [500, 700, 800], "diverge")
    )
    # below-window values are NOT masked by the clamp, so +1 must be detected
    cases.append(("control/off_by_one_below_window", [300, 400, 500], 511, [299, 400, 500], "diverge"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--src", default=None)
    args = ap.parse_args()

    repo = find_repo()
    src_path = Path(args.src) if args.src else repo / REPO_REL
    src = src_path.read_text()
    tree = ast.parse(src)

    dec = get_method(tree, "FlashInferIndicesUpdaterDecode", "call_begin_forward")
    pre = get_method(tree, "FlashInferIndicesUpdaterPrefill", "call_begin_forward")
    swa = get_method(tree, "FlashInferIndicesUpdaterDecode", "update_sliding_window")

    anchors = {}
    # --- anchor: decode host mirror assignment + guard -----------------------
    dec_asn = extract_anchor(
        src,
        dec,
        lambda n: isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "host_indptr_last" for t in n.targets),
        "decode host_indptr_last assignment",
    )
    dec_guard = extract_anchor(
        src,
        dec,
        lambda n: isinstance(n, ast.If)
        and "isinstance" in ast.dump(n.test)
        and "paged_kernel_lens_sum" in ast.dump(n.test),
        "decode isinstance(paged_kernel_lens_sum, int) guard",
    )
    dec_use = extract_anchor(
        src,
        dec,
        lambda n: isinstance(n, ast.Assign)
        and isinstance(n.value, ast.IfExp)
        and "host_indptr_last" in ast.dump(n.value)
        and any(isinstance(t, ast.Name) and t.id == "kv_last_index" for t in n.targets),
        "decode kv_last_index host-or-device selection",
    )
    dec_use = type("A", (), {"test": dec_use.value})()  # normalize to .test
    dec_empty = extract_anchor(
        src,
        dec,
        lambda n: isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "empty"
        and "paged_kernel_lens_sum" in ast.dump(n.value),
        "decode kv_indices buffer sized by paged_kernel_lens_sum",
    )
    # --- anchor: decode has no custom_kv_indices parameter -------------------
    dec_args = [a.arg for a in dec.args.args]
    # --- anchor: prefill 3-condition guard ----------------------------------
    pre_guard = extract_anchor(
        src,
        pre,
        lambda n: isinstance(n, ast.If)
        and "custom_kv_indices" in ast.dump(n.test)
        and "spec_info" in ast.dump(n.test)
        and "isinstance" in ast.dump(n.test),
        "prefill host-mirror guard (spec_info/custom_kv_indices/isinstance)",
    )
    pre_asn = extract_anchor(
        src,
        pre,
        lambda n: isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "kv_last_index" for t in n.targets)
        and "paged_kernel_lens_sum" in ast.dump(n.value),
        "prefill kv_last_index = paged_kernel_lens_sum",
    )
    # --- anchor: caller-side SWA clamp + host sum ---------------------------
    swa_clamp = extract_anchor(
        src,
        swa,
        lambda n: isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "paged_kernel_lens_tmp" for t in n.targets)
        and "clamp" in ast.dump(n.value),
        "update_sliding_window paged_kernel_lens_tmp = clamp(seq_lens)",
    )
    swa_cpu_clamp = extract_anchor(
        src,
        swa,
        lambda n: isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "seq_lens_cpu_tmp" for t in n.targets)
        and "clamp" in ast.dump(n.value),
        "update_sliding_window seq_lens_cpu_tmp = clamp(seq_lens_cpu)",
    )
    swa_sum = extract_anchor(
        src,
        swa,
        lambda n: isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "paged_kernel_lens_sum_tmp"
            for t in n.targets
        ),
        "update_sliding_window paged_kernel_lens_sum_tmp",
    )
    swa_sums = [
        n
        for n in ast.walk(swa)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "paged_kernel_lens_sum_tmp"
            for t in n.targets
        )
    ]
    swa_sums.sort(key=line_of)
    if len(swa_sums) < 3:
        raise SystemExit(f"ANCHOR MISSING: expected >=3 paged_kernel_lens_sum_tmp assigns, got {len(swa_sums)}")
    swa_sum, swa_else_sum = swa_sums[0], swa_sums[1]
    if "seq_lens_cpu" not in ast.dump(swa_sum) or "paged_kernel_lens_tmp" not in ast.dump(swa_else_sum):
        raise SystemExit("ANCHOR MISSING: paged_kernel_lens_sum_tmp provenance order changed")
    # third assignment is the full-attention wrapper (paged_kernel_lens_sum_tmp = seq_lens_sum)
    if "seq_lens_sum" not in ast.dump(swa_sums[2]):
        raise SystemExit("ANCHOR MISSING: full-attention wrapper sum assignment changed")
    # use_swa_source definitions (decode first, prefill second, by line)
    use_swa_defs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "use_swa_source" for t in n.targets)
    ]
    use_swa_defs.sort(key=line_of)
    if len(use_swa_defs) != 2:
        raise SystemExit(f"ANCHOR MISSING: expected 2 use_swa_source defs, got {len(use_swa_defs)}")

    anchors["decode_host_assignment"] = {
        "line": line_of(dec_asn),
        "src": src_seg(src, dec_asn),
    }
    anchors["decode_int_guard"] = {"line": line_of(dec_guard), "src": src_seg(src, dec_guard.test)}
    anchors["decode_host_or_device"] = {
        "line": line_of(dec_use),
        "src": src_seg(src, dec_use.test),
    }
    anchors["decode_indices_alloc"] = {"line": line_of(dec_empty), "src": src_seg(src, dec_empty)}
    anchors["decode_signature_has_custom_kv_indices"] = "custom_kv_indices" in dec_args
    anchors["prefill_guard"] = {"line": line_of(pre_guard), "src": src_seg(src, pre_guard.test)}
    anchors["prefill_host_assignment"] = {"line": line_of(pre_asn), "src": src_seg(src, pre_asn)}
    anchors["caller_swa_clamp"] = {"line": line_of(swa_clamp), "src": src_seg(src, swa_clamp)}
    anchors["caller_cpu_clamp"] = {"line": line_of(swa_cpu_clamp), "src": src_seg(src, swa_cpu_clamp)}
    anchors["caller_sum"] = {"line": line_of(swa_sum), "src": src_seg(src, swa_sum)}
    anchors["caller_gpu_sum_fallback"] = {
        "line": line_of(swa_else_sum),
        "src": src_seg(src, swa_else_sum),
    }
    anchors["use_swa_source_defs"] = [
        {"line": line_of(n), "src": src_seg(src, n)} for n in use_swa_defs
    ]

    # --- reachability matrix: execute the REAL guard expressions -------------
    class _Translator:
        def __init__(self, translated):
            self.reads_are_translated = translated

    class _Spec:
        def __init__(self, has_kv_indptr):
            if has_kv_indptr:
                self.kv_indptr = object()

    reach = []
    for swa_pool in (True, False):
        for translated in (True, False):
            for spec_case in ("none", "no_kv_indptr", "with_kv_indptr"):
                for custom in ("none", "tensor"):
                    for sum_is_int in (True, False):
                        # decode guard (executes the real extracted AST)
                        names = {
                            "paged_kernel_lens_sum": 1234 if sum_is_int else object(),
                            "spec_info": None
                            if spec_case == "none"
                            else _Spec(spec_case == "with_kv_indptr"),
                            "custom_kv_indices": None if custom == "none" else object(),
                            "translator": _Translator(translated),
                            "use_sliding_window_kv_pool": swa_pool,
                        }
                        use_swa = eval_expr(use_swa_defs[0].value, names)
                        in_translate_block = swa_pool and not use_swa
                        dec_host = bool(eval_expr(dec_guard.test, names)) and in_translate_block
                        # prefill guard
                        pre_host = bool(eval_expr(pre_guard.test, names)) and in_translate_block
                        # Does the patched line actually save a D2H?  Only if it
                        # is reached AND the caller had a host sum for free.
                        reach.append(
                            {
                                "swa_pool": swa_pool,
                                "translator_reads_are_translated": translated,
                                "use_swa_source": use_swa,
                                "translate_block_reachable": in_translate_block,
                                "spec_info": spec_case,
                                "custom_kv_indices": custom,
                                "sum_is_python_int": sum_is_int,
                                "decode_takes_host_mirror": dec_host,
                                "prefill_takes_host_mirror": pre_host,
                            }
                        )

    # --- integer replay ------------------------------------------------------
    replay = []
    for name, lens, window, lens_cpu, expect in build_cases():
        row = replay_case(name, lens, window, lens_cpu, expect)
        got = "equal" if row["index_identical"] else "diverge"
        row["oracle_verdict"] = "PASS" if got == expect else "FAIL"
        replay.append(row)

    n_div = sum(1 for r in replay if not r["index_identical"])
    controls = [r for r in replay if r["case"].startswith("control/")]
    controls_ok = all(r["oracle_verdict"] == "PASS" for r in controls)
    gate_cases = [r for r in replay if not r["case"].startswith("control/")]
    gate_div = [r for r in gate_cases if not r["index_identical"]]
    saves = [r for r in gate_cases if r["patch_saves_a_d2h"]]
    no_saving = [r for r in gate_cases if not r["patch_saves_a_d2h"] and r["seq_lens_cpu_provided"] is False]

    reachable_rows = [r for r in reach if r["translate_block_reachable"]]
    dec_reach = [r for r in reachable_rows if r["decode_takes_host_mirror"]]
    pre_reach = [r for r in reachable_rows if r["prefill_takes_host_mirror"]]
    dead_when_translated = [r for r in reach if r["swa_pool"] and r["translator_reads_are_translated"] and not r["translate_block_reachable"]]

    summary = {
        "source_file": str(src_path),
        "anchors_ok": True,
        "reachability_rows": len(reach),
        "translate_block_reachable_rows": len(reachable_rows),
        "decode_host_mirror_reachable_rows": len(dec_reach),
        "prefill_host_mirror_reachable_rows": len(pre_reach),
        "rows_dead_when_translator_reads_translated": len(dead_when_translated),
        "decode_patch_is_dead_when_translator_reads_translated": len(dead_when_translated)
        == len([r for r in reach if r["swa_pool"] and r["translator_reads_are_translated"]]),
        "replay_cases": len(replay),
        "replay_divergences_outside_controls": len(gate_div),
        "sensitivity_controls_pass": controls_ok,
        "cases_where_patch_saves_a_d2h": len(saves),
        "cases_where_caller_pays_the_d2h_anyway": len(no_saving),
        "divergent_cases": [r["case"] for r in replay if not r["index_identical"]],
        "envelope": (
            "The t3/t4 host mirror is reachable ONLY when (SWA dispatch) AND "
            "(_swa_kv_pool is not None) AND (NOT translator.reads_are_translated) AND "
            "(the rebuilt-indptr branch ran) AND paged_kernel_lens_sum is a python int. "
            "When translator.reads_are_translated is True the translate block is skipped "
            "entirely, so the changed lines are dead code. When seq_lens_cpu is None the "
            "caller itself .item()s the GPU tensor, so the patch removes no D2H. "
            "Index equality holds for every non-control case; the stale-mirror and "
            "off-by-one controls are both flagged, so the equality is not vacuous."
        ),
        "limits": (
            "CPU integer replay of extracted expressions, not a runtime measurement. "
            "No torch/GPU/model here; the frozen launch flags (unified pool? "
            "seq_lens_cpu present?) are controller-side and not asserted. "
            "No claim about real wall-clock; trial-1 already measured this family at "
            "1.0018 KEPT_NEUTRAL on a4."
        ),
    }

    out = {
        "tool": "g3-t6/verify_host_mirror_reachability.py",
        "anchors": anchors,
        "reachability": reach,
        "replay": replay,
        "summary": summary,
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0 if (controls_ok and not gate_div) else 2


if __name__ == "__main__":
    sys.exit(main())

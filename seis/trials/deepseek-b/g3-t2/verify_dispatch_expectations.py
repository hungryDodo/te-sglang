#!/usr/bin/env python3
"""GPU-free pre-registration for the G3-t2 GGUF dispatch experiment.

Parses the shipped policy out of gguf.py (no torch / gguf / SGLang import; the
sandbox has none of them), replays legacy vs f263262 kernel selection for every
referenced quant type x rows band x batch, and emits the expectation table the
controller's live GGUF experiment must match. It also statically checks that the
new dispatch tracer is default-off and wired into every return branch.

Usage: python -I verify_dispatch_expectations.py [out.json]

Exit code 0 = the shipped policy and tracer pass the pre-registration.
Exit code 1 = a falsifiable expectation is violated; do not run the experiment
as planned, read the failure list first.
"""

import json
import pathlib
import re
import sys

GGUF = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/layers/quantization/gguf.py"
)
BATCHES = list(range(1, 257))
# Probe ladder the controller should drive, plus two gate-scale controls.
PROBE_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 64]
BANDS = {"rows<=5120": 4096, "rows>5120": 8192}
# Types that exercise every branch of the selection function.
FOCUS_TYPES = ["Q8_0", "Q4_K", "Q6_K", "Q4_0", "Q8_1", "IQ4_XS"]


def parse_set(src, name):
    block = src.split(f"{name} = {{", 1)[1].split("}", 1)[0]
    return set(re.findall(r"WeightType\.(\w+)", block))


def parse_measured(src):
    measured = {}
    for name, lo, hi in re.findall(
        r"WeightType\.(\w+):\s*\((\d+),\s*(\d+)\)", src
    ):
        measured[name] = (int(lo), int(hi))
    return measured


def safe_batch(name, rows, measured, imatrix):
    band = 1 if rows > 5120 else 0
    if name in measured:
        return measured[name][band]
    if name == "Q8_1":
        return 2 if band else 6
    if name in imatrix:
        return 8 if band else 16
    return 2 if band else 6


def legacy_batch(name, rows, imatrix):
    band = 1 if rows > 5120 else 0
    if name in imatrix:
        return 8 if band else 16
    return 2 if band else 6


def kernel_for(name, batch, safe, unquant, mmvq_types, mmq_types, dequant_types):
    if name in unquant:
        return "unquantized"
    if batch <= safe and name in mmvq_types:
        return "mmvq"
    if name in mmq_types:
        return "mmq"
    if name in dequant_types:
        return "dequant"
    return "unsupported"


def main():
    src = GGUF.read_text()
    measured = parse_measured(src)
    imatrix = parse_set(src, "IMATRIX_QUANT_TYPES")
    standard = parse_set(src, "STANDARD_QUANT_TYPES")
    kquant = parse_set(src, "KQUANT_TYPES")
    unquant = parse_set(src, "UNQUANTIZED_TYPES")
    mmvq_types = standard | kquant | imatrix
    mmq_types = standard | kquant
    dequant_types = standard | kquant | imatrix
    enum_names = sorted(set(re.findall(r"WeightType\.(\w+)", src)))

    failures = []
    expected_measured = {"Q8_0": (2, 2), "Q4_K": (2, 2), "Q6_K": (6, 2)}
    if measured != expected_measured:
        failures.append(
            f"measured table changed: {measured} != {expected_measured}"
        )

    changed = []
    for name in enum_names:
        for band_name, rows in BANDS.items():
            for b in BATCHES:
                safe = safe_batch(name, rows, measured, imatrix)
                old_safe = legacy_batch(name, rows, imatrix)
                old_k = kernel_for(
                    name, b, old_safe, unquant, mmvq_types, mmq_types, dequant_types
                )
                new_k = kernel_for(
                    name, b, safe, unquant, mmvq_types, mmq_types, dequant_types
                )
                if old_k != new_k:
                    changed.append(
                        {
                            "weight_type": name,
                            "rows_band": band_name,
                            "batch": b,
                            "legacy_kernel": old_k,
                            "shipped_kernel": new_k,
                            "legacy_mmvq_safe": old_safe,
                            "shipped_mmvq_safe": safe,
                        }
                    )
    if len(changed) != 8:
        failures.append(f"expected exactly 8 changed cells, got {len(changed)}")
    changed_types = sorted({c["weight_type"] for c in changed})
    changed_batches = sorted({c["batch"] for c in changed})
    if changed_types != ["Q4_K", "Q8_0"] or changed_batches != [3, 4, 5, 6]:
        failures.append(
            f"changed cell set drifted: types={changed_types} batches={changed_batches}"
        )
    gate_changed = [c for c in changed if c["batch"] in (16, 32, 64, 128, 256)]
    if gate_changed:
        failures.append(f"selection changes at gate batches: {gate_changed[:4]}")
    if any(c["rows_band"] != "rows<=5120" for c in changed):
        failures.append("changed cell outside the rows<=5120 band")

    # The expectation table the live trace must match exactly. It covers the
    # changed cells and the unaffected controls in one machine-readable list.
    expected_dispatch = []
    for name in FOCUS_TYPES:
        for band_name, rows in BANDS.items():
            for b in PROBE_BATCHES:
                safe = safe_batch(name, rows, measured, imatrix)
                old_safe = legacy_batch(name, rows, imatrix)
                expected_dispatch.append(
                    {
                        "weight_type": name,
                        "rows_band": band_name,
                        "batch": b,
                        "expected_kernel": kernel_for(
                            name,
                            b,
                            safe,
                            unquant,
                            mmvq_types,
                            mmq_types,
                            dequant_types,
                        ),
                        "expected_mmvq_safe": safe,
                        "legacy_kernel": kernel_for(
                            name,
                            b,
                            old_safe,
                            unquant,
                            mmvq_types,
                            mmq_types,
                            dequant_types,
                        ),
                        "changed_by_f263262": any(
                            c["weight_type"] == name
                            and c["rows_band"] == band_name
                            and c["batch"] == b
                            for c in changed
                        ),
                    }
                )

    # Static checks on the tracer. The live experiment is only meaningful if
    # the tracer is default-off (no effect on the frozen BF16 gate) and every
    # dispatch branch reports the kernel it actually runs.
    tracer_checks = {
        "default_off_env_gate": 'SGLANG_GGUF_DISPATCH_TRACE' in src
        and '_DISPATCH_TRACE_FILE = None' in src,
        "record_call_sites": len(re.findall(r"_record_dispatch\(", src)) - 1,
        "branch_assignments": {
            k: src.count(f'kernel = "{k}"') for k in ("mmvq", "mmq", "dequant")
        },
        "atexit_dump": "atexit.register(dump_gguf_dispatch_trace)" in src,
        "inert_by_default": "if _DISPATCH_TRACE_FILE is None:\n        return" in src,
    }
    if not tracer_checks["default_off_env_gate"]:
        failures.append("tracer env gate missing")
    if tracer_checks["record_call_sites"] < 3:
        failures.append(
            f"tracer records only {tracer_checks['record_call_sites']} call sites"
        )
    if tracer_checks["branch_assignments"] != {
        "mmvq": 1,
        "mmq": 1,
        "dequant": 1,
    }:
        failures.append(
            f"branch kernel labels drifted: {tracer_checks['branch_assignments']}"
        )

    out = {
        "generated_from": "python/sglang/srt/layers/quantization/gguf.py",
        "policy_measured": measured,
        "changed_cells": changed,
        "changed_cell_count": len(changed),
        "changed_types": changed_types,
        "changed_batches": changed_batches,
        "changed_cells_at_gate_batches": len(gate_changed),
        "probe_batches": PROBE_BATCHES,
        "expected_dispatch": expected_dispatch,
        "tracer_static_checks": tracer_checks,
        "pre_registration_failures": failures,
        "experiment_assertions": [
            "Dispatch probe must run with --disable-cuda-graph: under graph replay "
            "Python dispatch is captured once and not re-executed per step, so the "
            "trace would show capture-time batch sizes only (padded), not the "
            "per-batch decision.",
            "At batches 3..6, every Q8_0 and Q4_K cell with rows<=5120 must appear "
            "in the trace as kernel=mmq with mmvq_safe=2. The legacy arm expects "
            "kernel=mmvq with mmvq_safe=6 for the same cells.",
            "At batches 1..2 the same cells must stay kernel=mmvq, mmvq_safe=2.",
            "Cells for weight types not present in the loaded artifact are N/A. The "
            "controls that must always be present are the unchanged Q8_0 cells: "
            "rows>5120 at every probed batch, and rows<=5120 at batches 1,2,7,8,16,64. "
            "They must match expected_kernel and expected_mmvq_safe exactly; a missing "
            "Q8_0 control means the tracer did not cover that matmul and the "
            "experiment is INCONCLUSIVE, not PASS.",
            "Greedy token ids: parent f263262^ vs candidate must match at batches "
            "1,2,7,8,16,64. Divergence is allowed only at batches 3..6 and only in "
            "Q8_0/Q4_K rows<=5120 cells; record the per-request divergence count.",
            "This probe is not a timing gate. Any throughput number from it is "
            "excluded from official scoring; the formal objective remains the "
            "frozen BF16 gate, which never imports this module.",
        ],
    }
    target = sys.argv[1] if len(sys.argv) > 1 else "dispatch-expectations.json"
    pathlib.Path(target).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {target}")
    print(
        f"changed_cells={len(changed)} types={changed_types} "
        f"batches={changed_batches} gate_batches=0"
    )
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

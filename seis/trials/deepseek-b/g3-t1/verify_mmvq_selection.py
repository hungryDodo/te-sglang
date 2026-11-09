#!/usr/bin/env python3
"""GPU-free replay of legacy vs f263262 MMVQ/MMQ kernel selection.

Parses the shipped policy out of gguf.py (no torch / SGLang import).
Dispatch rule in fused_mul_mat_gguf: MMVQ iff batch <= mmvq_safe, else MMQ.

Usage: python -I verify_mmvq_selection.py [out.json]
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
BANDS = {"rows<=5120": 4096, "rows>5120": 8192}
GATE_BATCHES = [16, 32, 64, 128, 256]


def parse_policy(src):
    measured = {}
    for name, lo, hi in re.findall(
        r"WeightType\.(\w+):\s*\((\d+),\s*(\d+)\)", src
    ):
        measured[name] = (int(lo), int(hi))
    imatrix_block = src.split("IMATRIX_QUANT_TYPES = {", 1)[1].split("}", 1)[0]
    imatrix = set(re.findall(r"WeightType\.(\w+)", imatrix_block))
    # WeightType is `gguf.GGMLQuantizationType` (external package, not
    # importable GPU-free here), so the relevant enum surface is exactly the
    # set of names this module references. Unlisted types cannot change
    # selection because both policies fall through to the same legacy value.
    enum_names = re.findall(r"WeightType\.(\w+)", src)
    return measured, imatrix, enum_names


def new_safe(name, rows, measured, imatrix):
    band = 1 if rows > 5120 else 0
    if name in measured:
        return measured[name][band]
    if name == "Q8_1":
        return 2 if band else 6
    if name in imatrix:
        return 8 if band else 16
    return 2 if band else 6


def legacy_safe(name, rows, imatrix):
    band = 1 if rows > 5120 else 0
    if name in imatrix:
        return 8 if band else 16
    return 2 if band else 6


def main():
    src = GGUF.read_text()
    measured, imatrix, enum_names = parse_policy(src)
    names = sorted(set(enum_names) | set(measured) | set(imatrix))
    changed = []
    for name in names:
        for band_name, rows in BANDS.items():
            for b in BATCHES:
                old = legacy_safe(name, rows, imatrix)
                new = new_safe(name, rows, measured, imatrix)
                old_mmvq, new_mmvq = b <= old, b <= new
                if old_mmvq != new_mmvq:
                    changed.append(
                        {
                            "type": name,
                            "band": band_name,
                            "batch": b,
                            "old_mmvq": old_mmvq,
                            "new_mmvq": new_mmvq,
                        }
                    )
    gate_cells = [c for c in changed if c["batch"] in GATE_BATCHES]
    batches = sorted({c["batch"] for c in changed})
    out = {
        "source": str(GGUF),
        "parsed_measured_table": {k: list(v) for k, v in measured.items()},
        "imatrix_types": sorted(imatrix),
        "enum_type_count": len(names),
        "changed_selection_cells": changed,
        "changed_cell_count": len(changed),
        "changed_types": sorted({c["type"] for c in changed}),
        "changed_bands": sorted({c["band"] for c in changed}),
        "changed_batches": batches,
        "changed_cells_at_gate_batches": len(gate_cells),
        "changed_cells_at_batch_ge_16": sum(1 for c in changed if c["batch"] >= 16),
        # Every frozen gate point decodes with concurrency >= 16 (PrefillHeavy
        # c16 is the minimum) and prefills with thousands of extend tokens, so
        # a decode batch < 16 cannot occur at any gate point.
        "selection_identical_from_batch": (max(batches) + 1) if batches else 1,
        "gate_batches_checked": GATE_BATCHES,
    }
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "changed_selection_cells"}, indent=2))


if __name__ == "__main__":
    main()

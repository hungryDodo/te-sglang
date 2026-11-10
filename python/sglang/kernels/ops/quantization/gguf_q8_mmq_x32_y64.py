"""Controller-origin X8 research route. Missing/failed target native is fatal.

The independent Controller maps/SHA proof is required outside this helper;
its local SHA check is not a trusted attestation by itself.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch

_DIAGNOSTICS = os.environ.get("SGLANG_Q8_MMQ_X32_Y64_DIAGNOSTICS") == "1"
_COUNTS = Counter()
_NAMESPACE = "tierengine_q8_mmq_x32_y64_v1"


def coverage_reason(x, weight, quant_type):
    if type(quant_type) is not int and not isinstance(quant_type, int):
        return "quant_type_not_integer"
    if quant_type != 8:
        return "other_quant_type"
    if x.ndim != 2 or weight.ndim != 2:
        return "rank"
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        return "device"
    if x.dtype != torch.bfloat16 or weight.dtype != torch.uint8:
        return "dtype"
    if not x.is_contiguous() or not weight.is_contiguous():
        return "layout"
    batch, col = x.shape
    if not 32 <= batch <= 8192:
        return "rows_outside_32_8192"
    if col not in (2048, 6656) or not 0 < weight.shape[0] <= 131072:
        return "fixed_model_dimensions"
    if weight.shape[1] != col // 32 * 34:
        return "packed_q8_width"
    if batch * weight.shape[0] > 2147483647:
        return "index_overflow"
    return None


def covered(x, weight, quant_type):
    reason = coverage_reason(x, weight, quant_type)
    if _DIAGNOSTICS:
        _COUNTS["covered" if reason is None else "fallback_" + reason] += 1
    return reason is None


def diagnostic_counts():
    return {"enabled": _DIAGNOSTICS, "counts": dict(_COUNTS),
            "scope": "Python dispatch/capture calls only; not actual CUDA graph replay counts"}


@functools.lru_cache(maxsize=1)
def _operator():
    root = Path(__file__).with_name("native_q8_mmq_x32_y64")
    binding = json.loads((root / "NATIVE-BUILD-MANIFEST.json").read_bytes())
    if (binding.get("schema_version") != 1 or binding.get("namespace") != _NAMESPACE
            or binding.get("library") != "q8_mmq_x32_y64.so"):
        raise RuntimeError("Explicit X8 prebuilt library manifest required")
    library = root / binding["library"]
    if (library.is_symlink() or not library.is_file()
            or hashlib.sha256(library.read_bytes()).hexdigest() != binding.get("library_sha256")):
        raise RuntimeError("Registered X8 library bytes differ")
    torch.ops.load_library(str(library))
    return torch.ops.tierengine_q8_mmq_x32_y64_v1.mmq.default


def apply(x, weight):
    # The caller selected this target. Never catch a load/launch failure and
    # silently execute the old kernel; a research run must retain that failure.
    return _operator()(weight, x)

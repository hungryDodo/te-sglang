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

_DIAGNOSTICS = os.environ.get("SGLANG_Q8_MMQ_X64_Y64_W4_DIAGNOSTICS") == "1"
_COUNTS = Counter()
_NAMESPACE = "tierengine_q8_mmq_x64_y64_w4_v1"
_NATIVE_SOURCE = "eae6129e7dcd59c60b0ecf2ab5415de044b5bf99"
_NATIVE_SHA256 = "23eb82be770da5cf4b716cae41cc1d269b97521f3fdfa39e6408d83c79f215d9"
_NATIVE_BYTES = 380520


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
    if weight.shape[0] % 64 != 0:
        return "weight_rows_not_multiple_of_64"
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
def _load_library():
    root = Path(__file__).with_name("native_q8_mmq_x64_y64_w4")
    binding = json.loads((root / "NATIVE-BUILD-MANIFEST.json").read_bytes())
    if (
        binding.get("schema_version") != 1
        or binding.get("namespace") != _NAMESPACE
        or binding.get("library") != "q8_mmq_x64_y64_w4.so"
        or binding.get("source_build_commit") != _NATIVE_SOURCE
        or binding.get("library_sha256") != _NATIVE_SHA256
        or binding.get("library_bytes") != _NATIVE_BYTES
    ):
        raise RuntimeError("Exact validated W4 head-gate/GELU library manifest required")
    library = root / binding["library"]
    if library.is_symlink() or not library.is_file():
        raise RuntimeError("Exact validated W4 library file required")
    data = library.read_bytes()
    if len(data) != _NATIVE_BYTES or hashlib.sha256(data).hexdigest() != _NATIVE_SHA256:
        raise RuntimeError("Registered W4 library bytes differ")
    torch.ops.load_library(str(library))
    return binding


@functools.lru_cache(maxsize=1)
def _operator():
    _load_library()
    return torch.ops.tierengine_q8_mmq_x64_y64_w4_v1.mmq.default


def apply(x, weight):
    # The caller selected this target. Never catch a load/launch failure and
    # silently execute the old kernel; a research run must retain that failure.
    return _operator()(weight, x)


def apply_gate_up(x, gate, up):
    _load_library()
    if _DIAGNOSTICS:
        _COUNTS["paired_gate_up"] += 1
    return torch.ops.tierengine_q8_mmq_x64_y64_w4_v1.mmq_gate_up.default(gate, up, x)


@functools.lru_cache(maxsize=1)
def _gelu_down_operator():
    _load_library()
    return torch.ops.tierengine_q8_mmq_x64_y64_w4_v1.mmq_gelu_down_probe.default


def apply_gelu_down(gate_up, weight):
    return _gelu_down_operator()(weight, gate_up)


@functools.lru_cache(maxsize=1)
def _head_gate_out_operator():
    _load_library()
    return torch.ops.tierengine_q8_mmq_x64_y64_w4_v1.mmq_head_gate_out_probe.default


def apply_head_gate_out(x, gate, weight):
    return _head_gate_out_operator()(weight, x, gate)

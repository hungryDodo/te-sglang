"""Research r12 (glm-b): int8 tensor-core (mma.sync m16n8k16) route for the
three fixed B8192 Spark Q8_0 MLP GEMMs (gate/up 6656x2048 paired, down
2048x6656). One-time weight repack into ldmatrix-friendly s8 panels at load;
dispatch only when SGLANG_Q8_IMMA=1 (strict parse) AND the repacked panel is
present; every other case falls back to the unchanged dp4a MMQ path. The
dp4a kernel stays compiled beside the IMMA kernel as the exact reference.

Torch is imported lazily so guard logic stays importable without it.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_NAMESPACE = "tierengine_q8_imma_m16n8k16_v1"
_ENV = "SGLANG_Q8_IMMA"

# qweight (rows, packed_width) -> logical K for the three fixed shapes.
IMMA_SHAPES = {
    (6656, 2176): 2048,    # gate or up projection (standalone/shard view)
    (13312, 2176): 2048,   # paired gate+up (MergedColumnParallelLinear padding)
    (2048, 7072): 6656,    # down projection
}


@functools.lru_cache(maxsize=1)
def imma_env_enabled():
    """Strict parse: unset/empty -> off, '1' -> on, anything else -> ValueError."""
    raw = os.environ.get(_ENV)
    if raw is None or raw == "":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{_ENV} must be unset or '1' (strict parse), got {raw!r}")


def panel_index(rows, k, r, kk):
    """Pure layout spec: byte offset of weight byte w[r][kk] inside panel_s8."""
    return (((r // 16) * (k // 16) + (kk // 16)) * 256 + (r % 16) * 16 + (kk % 16))


def scale_index(rows, k, r, blk):
    """Pure layout spec: element offset of the block-q8_0 scale d[r][blk]."""
    return r * (k // 32) + blk


def repack_q8_0_panel(qweight, rows, k):
    """Byte-exact expansion of packed block_q8_0 bytes (no dequantization).

    panel_s8: int8 [rows/16, k/16, 16, 16]; tile (r16, k16) holds weight bytes
    w[r16*16+i][k16*16+j] row-major with 16-byte rows (ldmatrix m8n8.x2).
    panel_d: fp16 [rows, k/32] per-32-K-block scales in row-major order.
    """
    import torch

    nblk = k // 32
    view = qweight.reshape(rows, nblk, 34)
    d = (
        view[..., :2].contiguous().view(torch.float16).reshape(rows, nblk).contiguous()
    )
    qs = view[..., 2:].contiguous()  # uint8 [rows, nblk, 32] in load order
    panel_s8 = (
        qs.view(torch.int8)
        .reshape(rows, k)  # logical w[r][kk], row-major
        .reshape(rows // 16, 16, k // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return panel_s8, d


def _quant_type_ok(layer):
    qweight_type = getattr(layer, "qweight_type", None)
    if qweight_type is None:
        return False
    shards = getattr(qweight_type, "shard_weight_type", None)
    if shards is not None:
        return dict(shards) == {0: 8, 1: 8}
    return getattr(qweight_type, "weight_type", None) == 8


def maybe_repack_layer(layer):
    """One-time load-path repack; no-op unless env on + Q8_0 + fixed shape."""
    if not imma_env_enabled():
        return
    qweight = getattr(layer, "qweight", None)
    if qweight is None or not qweight.is_cuda or not _quant_type_ok(layer):
        return
    shape = tuple(qweight.shape)
    k = IMMA_SHAPES.get(shape)
    if k is None:
        return
    if shape == (13312, 2176):
        gate_s8, gate_d = repack_q8_0_panel(qweight[:6656], 6656, k)
        up_s8, up_d = repack_q8_0_panel(qweight[6656:], 6656, k)
        layer.q8_imma_gate_up = (gate_s8, gate_d, up_s8, up_d)
        logger.info("Q8 IMMA panels repacked for paired gate/up [13312,2176]")
    else:
        rows = shape[0]
        panel_s8, panel_d = repack_q8_0_panel(qweight, rows, k)
        qweight.q8_imma_panel = (panel_s8, panel_d)
        layer.q8_imma_panel = (panel_s8, panel_d)
        logger.info("Q8 IMMA panel repacked for weight %s", shape)


@functools.lru_cache(maxsize=1)
def _load_library():
    root = Path(__file__).with_name("native_q8_imma_m16n8k16")
    manifest = root / "NATIVE-BUILD-MANIFEST.json"
    if not manifest.is_file():
        raise RuntimeError(
            "SGLANG_Q8_IMMA=1 selected the tensor-core path but "
            f"{manifest} is absent: the Controller must build "
            "q8_imma_m16n8k16.so first (see NATIVE-SOURCE-PLAN.json)"
        )
    binding = json.loads(manifest.read_bytes())
    if (
        binding.get("schema_version") != 1
        or binding.get("namespace") != _NAMESPACE
        or binding.get("library") != "q8_imma_m16n8k16.so"
    ):
        raise RuntimeError("Explicit IMMA prebuilt library manifest required")
    library = root / binding["library"]
    if (
        library.is_symlink()
        or not library.is_file()
        or hashlib.sha256(library.read_bytes()).hexdigest() != binding.get("library_sha256")
    ):
        raise RuntimeError("Registered IMMA library bytes differ")
    import torch

    torch.ops.load_library(str(library))
    return binding


def maybe_apply(x, qweight):
    """IMMA result or None (caller falls back to the dp4a MMQ). Called only
    from inside the x64_y64_w4 covered() branch; never widens admission."""
    if not imma_env_enabled():
        return None
    panel = getattr(qweight, "q8_imma_panel", None)
    if panel is None:
        return None  # missing panel -> dp4a fallback
    panel_s8, panel_d = panel
    rows, k = panel_d.shape[0], panel_d.shape[1] * 32
    import torch

    if (
        x.shape[1] != k
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[0] % 64 != 0
    ):
        return None  # tail batches and odd layouts keep the dp4a path
    _load_library()
    return torch.ops.tierengine_q8_imma_m16n8k16_v1.mmq_imma.default(
        panel_s8, panel_d, x
    )


def maybe_apply_gate_up(x, layer):
    """Paired IMMA gate/up or None (caller falls back to the W4 dp4a pair)."""
    if not imma_env_enabled():
        return None
    pair = getattr(layer, "q8_imma_gate_up", None)
    if pair is None or tuple(x.shape) != (8192, 2048):
        return None
    _load_library()
    import torch

    return torch.ops.tierengine_q8_imma_m16n8k16_v1.mmq_gate_up_imma.default(
        pair[0], pair[1], pair[2], pair[3], x
    )

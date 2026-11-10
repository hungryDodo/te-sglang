"""Headwise attention-output sigmoid gate for Spark2_5-style layers.

Production eager baseline (preserved verbatim in
``headwise_attn_output_gate_baseline``)::

    g = torch.sigmoid(g.float()).to(attn_output.dtype)
    gate_output = attn_output.view(
        attn_output.shape[0], num_heads, head_dim
    ) * g.unsqueeze(-1)
    attn_output = gate_output.view(*attn_output.shape)

The fused Triton candidate preserves the explicit intermediate rounding: the
FP32 sigmoid result is cast to ``attn_output.dtype`` (BF16/FP16 rounding)
BEFORE the per-head broadcast multiply. It is never replaced by an unrounded
FP32 gate*output product. The kernel always writes a brand-new output tensor;
inputs are never mutated and no in-place ``mul_`` is used.

For a flattened ``attn_output`` element index ``i`` the gate index is
``row * num_heads + head`` where ``row = i // (num_heads * head_dim)`` and
``head = (i % (num_heads * head_dim)) // head_dim``.

``apply_headwise_attn_output_gate`` validates tensors and metadata and falls
back to the exact baseline expression on anything unexpected (CPU tensors,
wrong shape/dim/dtype/device, non-contiguous storage, non-int metadata,
oversized numel, empty batch). ``triton`` is imported lazily only after all
guards pass, so backends without Triton/CUDA keep working via the fallback.
"""

from __future__ import annotations

import torch

_SUPPORTED_ATTN_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# Kernel scalar args are 32-bit; never launch with an element count that
# could overflow them (fall back to eager instead).
_MAX_FUSED_ELEMENTS = 2**31 - 1
_TRITON_BLOCK_SIZE = 1024
_TRITON_NUM_WARPS = 4

_headwise_gate_triton_kernel = None  # lazily defined JIT kernel singleton


def headwise_attn_output_gate_baseline(attn_output, g, num_heads, head_dim):
    """Exact original eager math; kept for direct GPU numerical comparison."""
    g = torch.sigmoid(g.float()).to(attn_output.dtype)
    gate_output = attn_output.view(
        attn_output.shape[0],
        num_heads,
        head_dim,
    ) * g.unsqueeze(-1)
    return gate_output.view(*attn_output.shape)


def _headwise_gate_launch_eligible(attn_output, g, num_heads, head_dim) -> bool:
    """Strict eligibility guards for the fused kernel launch."""
    if not isinstance(attn_output, torch.Tensor):
        return False
    if not isinstance(g, torch.Tensor):
        return False
    if not isinstance(num_heads, int) or not isinstance(head_dim, int):
        return False
    if num_heads <= 0 or head_dim <= 0:
        return False
    if attn_output.dim() != 2 or g.dim() != 2:
        return False
    if attn_output.shape[1] != num_heads * head_dim:
        return False
    if g.shape[0] != attn_output.shape[0] or g.shape[1] != num_heads:
        return False
    if attn_output.dtype not in _SUPPORTED_ATTN_DTYPES:
        return False
    if g.dtype != attn_output.dtype:
        return False
    if attn_output.device.type != "cuda" or g.device.type != "cuda":
        return False
    if g.device != attn_output.device:
        return False
    if not attn_output.is_contiguous() or not g.is_contiguous():
        return False
    if attn_output.numel() > _MAX_FUSED_ELEMENTS:
        return False
    return True


def _get_headwise_gate_triton_kernel():
    """Lazily define (and JIT-compile on first launch) the Triton kernel.

    Triton JIT source of record (defined only in this function, in this file):
    ``@triton.jit def _headwise_gate_kernel`` in
    ``python/sglang/srt/layers/headwise_attn_output_gate.py``.
    """
    global _headwise_gate_triton_kernel
    if _headwise_gate_triton_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _headwise_gate_kernel(
            attn_output_ptr,
            gate_ptr,
            out_ptr,
            n_elements,
            head_stride,
            num_heads,
            head_dim,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
                tl.int64
            )
            mask = offs < n_elements
            # Flattened index -> (row, head); gate value at row * num_heads + head.
            row = offs // head_stride
            within_row = offs - row * head_stride
            head = within_row // head_dim
            gate_index = row * num_heads + head
            attn_val = tl.load(attn_output_ptr + offs, mask=mask)
            gate_val = tl.load(gate_ptr + gate_index, mask=mask)
            # FP32 sigmoid, then EXPLICIT cast back to attn_output.dtype so the
            # intermediate BF16/FP16 rounding happens BEFORE the multiply.
            gate_f32 = tl.sigmoid(gate_val.to(tl.float32))
            gate_rounded = gate_f32.to(attn_val.dtype)
            out_val = attn_val * gate_rounded
            tl.store(out_ptr + offs, out_val, mask=mask)

        _headwise_gate_triton_kernel = _headwise_gate_kernel
    return _headwise_gate_triton_kernel


def apply_headwise_attn_output_gate(attn_output, g, num_heads, head_dim):
    """Gate attention output per head; fused kernel or exact eager baseline.

    Returns a NEW tensor (never a view/alias of the inputs; no in-place ops).
    """
    try:
        eligible = _headwise_gate_launch_eligible(attn_output, g, num_heads, head_dim)
    except Exception:
        eligible = False
    if (
        not eligible
        or not isinstance(attn_output, torch.Tensor)
        or attn_output.numel() == 0
    ):
        # Empty batch included: exact eager path, no zero-grid kernel launch.
        return headwise_attn_output_gate_baseline(attn_output, g, num_heads, head_dim)

    out = torch.empty_like(attn_output)  # brand-new contiguous output tensor
    n = attn_output.numel()
    grid = ((n + _TRITON_BLOCK_SIZE - 1) // _TRITON_BLOCK_SIZE,)
    kernel = _get_headwise_gate_triton_kernel()
    kernel[grid](
        attn_output,
        g,
        out,
        n,
        num_heads * head_dim,
        num_heads,
        head_dim,
        BLOCK_SIZE=_TRITON_BLOCK_SIZE,
        num_warps=_TRITON_NUM_WARPS,
    )
    return out

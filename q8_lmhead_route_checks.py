#!/usr/bin/env python3
"""Stdlib-only checks for NATIVE-LMHEAD-FINDING.json (no torch/CUDA/model).

Finding: Spark2_5 ties lm_head to embed_tokens, whose GGUF tensor is Q8_0
packed, so logits already run a quantized GGUF kernel -- never a dequantized
BF16 GEMM.  This checker pins the committed source anchors, the base-file
SHA256s, and a pure transcription of the two dispatch predicates (B1/B8/B64
plus wrong dtype/rank/device/layout).  No candidate code exists to guard.
"""
import ast
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
GGUF = (ROOT / "python/sglang/srt/layers/quantization/gguf.py").read_bytes()
LP = (ROOT / "python/sglang/srt/layers/logits_processor.py").read_bytes()
SPARK = (ROOT / "python/sglang/srt/models/spark2_5.py").read_text()
HELPER = (
    ROOT / "python/sglang/kernels/ops/quantization/gguf_q8_mmq_x64_y64_w4.py"
).read_text()
NAMES = (ROOT / "python/sglang/srt/model_loader/gguf_name_maps.py").read_text()

# SHA256 of both files at base commit 6ee7279: byte-for-byte, no routing added.
GGUF_SHA = "38c97e72f5ef14f7502fbbd51a5833930ebed569084b2d8a1a098a7806d42b19"
LP_SHA = "5154d547402a84649e625f3eae4e8e5671f4d3fdf0ef2d255b5a60611c1debb5"
MARKER = "Using W4 MMQ Spark lm_head logits"


def has(text, needle, where):
    assert needle in text, f"{where}: missing {needle!r}"


# 1. Execution sources are unchanged and carry no lm_head trace marker.
assert hashlib.sha256(GGUF).hexdigest() == GGUF_SHA, "gguf.py changed vs base"
assert hashlib.sha256(LP).hexdigest() == LP_SHA, "logits_processor.py changed vs base"
assert MARKER not in GGUF.decode() and MARKER not in LP.decode()

# 2. Committed anchors for the traced path.
lp = LP.decode()
has(lp, 'quant_method = getattr(lm_head, "quant_method", None)', "logits_processor")
has(lp, "elif should_apply_lm_head_quant_method(lm_head, quant_method):", "logits_processor")
has(lp, 'elif hasattr(lm_head, "weight"):', "logits_processor")
has(lp, "# GGUF models", "logits_processor")
has(lp, "logits = lm_head.quant_method.apply(", "logits_processor")
has(SPARK, "self.lm_head = self.model.embed_tokens", "spark2_5")
has(NAMES, '# "output" only exists when embeddings are not tied; Spark2_5 ties them.',
    "gguf_name_maps")
_spark = next(
    node.value
    for node in ast.parse(NAMES).body
    if isinstance(node, ast.Assign)
    and getattr(node.targets[0], "id", "") == "_SPARK2_5_GLOBAL_TENSORS"
)
_spark_keys = [key.value for key in _spark.keys]
assert "output" not in _spark_keys and "token_embd" in _spark_keys, _spark_keys
g = GGUF.decode()
has(g, "mmvq_safe = 2 if qweight.shape[0] > 5120 else 6", "gguf")
has(g, "if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:", "gguf")
has(g, "out = fused_mul_mat_gguf(x, qweight, qweight_type)", "gguf")
has(g, "if _is_cuda and _research_q8_mmq_x64_y64_w4_covered(x, qweight, qweight_type):",
    "gguf")
has(g, "y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])", "gguf")
has(HELPER, "if not 32 <= batch <= 8192:", "helper")
has(HELPER, "if col not in (2048, 6656) or not 0 < weight.shape[0] <= 131072:", "helper")
has(HELPER, "if weight.shape[1] != col // 32 * 34:", "helper")


# 3. Pure transcription of coverage_reason()/fused_mul_mat_gguf branch order.
def coverage(batch, rows, col=2048, packed=2176, dtype="bf16", rank=2,
             device="cuda", contiguous=True):
    if rank != 2:
        return "rank"
    if device != "cuda":
        return "device"
    if dtype != "bf16":
        return "dtype"
    if not contiguous:
        return "layout"
    if not 32 <= batch <= 8192:
        return "rows_outside_32_8192"
    if col not in (2048, 6656) or not 0 < rows <= 131072:
        return "fixed_model_dimensions"
    if rows % 64 != 0:
        return "weight_rows_not_multiple_of_64"
    if packed != col // 32 * 34:
        return "packed_q8_width"
    if batch * rows > 2147483647:
        return "index_overflow"
    return None


VOCAB, HIDDEN, PACKED = 133120, 2048, 2176  # config default; gate/up geometry


def path(batch, vocab=VOCAB, **kw):
    if batch == 0:
        return "empty"
    if batch <= (2 if vocab > 5120 else 6):  # gguf.py:257,266-267
        return "mmvq"
    ok = coverage(batch, vocab, HIDDEN, PACKED, **kw) is None
    return "research_mmq" if ok else "vendor_mmq"  # gguf.py:270-273


# 4. Dispatch cases: decode never reaches the research W4 apply().
assert path(1) == "mmvq" and path(2) == "mmvq"
assert path(8) == "vendor_mmq" and path(10) == "vendor_mmq"
assert path(64) == "vendor_mmq"                 # V=133120 > 131072 row cap
assert path(64, vocab=131072) == "research_mmq"  # only V<=131072 and B>=32
assert path(64, dtype="f16") == "vendor_mmq"
assert coverage(64, VOCAB, dtype="f16") == "dtype"
assert coverage(64, VOCAB, rank=3) == "rank"
assert coverage(64, VOCAB, device="cpu") == "device"
assert coverage(64, VOCAB, contiguous=False) == "layout"
assert coverage(8, VOCAB) == "rows_outside_32_8192"
assert path(0) == "empty"
print("LMHEAD-FINDING-CHECKS OK path=quantized_mmq B1=mmvq B8=vendor_mmq "
      "B64=vendor_mmq research_reachable=false")

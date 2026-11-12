"""Stdlib CPU source checks for the IMMA m16n8k16 candidate."""
import ast, json, pathlib, sys
R = pathlib.Path(".")
Q = R / "python/sglang/kernels/ops/quantization"
ND = Q / "native_q8_imma_m16n8k16"
HP = Q / "gguf_q8_imma_m16n8k16.py"

# 1) Files exist and are non-trivial
core = (ND / "imma_m16n8k16_core.cuh").read_text()
cu = (ND / "q8_imma_m16n8k16.cu").read_text()
hp = HP.read_text()
assert len(core) > 3000 and len(cu) > 3000 and len(hp) > 3000

# 2) mma.sync m16n8k16 present in kernel
assert "mma.sync.aligned.m16n8k16" in core
assert core.count("ldmatrix") >= 2

# 3) Panel layout documented and produced
assert "panel_s8" in core and "panel_d" in core
assert "repack_q8_0_panel" in hp

# 4) Opt-in env in wrapper
assert 'SGLANG_Q8_IMMA' in hp and hp.count('SGLANG_Q8_IMMA') >= 2

# 5) Python wrapper parses
ast.parse(hp)

# 6) NATIVE-SOURCE-PLAN.json valid
plan = json.loads((ND / "NATIVE-SOURCE-PLAN.json").read_bytes())
assert plan["identity"] == "tierengine_q8_imma_m16n8k16_v1"
assert plan["compiled"] is False and plan["library_sha256"] is None
assert plan["tile"]["mma"] == "m16n8k16"

# 7) dp4a fallback preserved (old wrapper untouched)
old = (Q / "gguf_q8_mmq_x64_y64_w4.py").read_text()
assert "covered" in old and "apply" in old

# 8) gguf.py seam present
gguf = (R / "python/sglang/srt/layers/quantization/gguf.py").read_text()
assert "imma" in gguf.lower() or "IMMA" in gguf

print("PASS: 8 source checks for IMMA m16n8k16 candidate")

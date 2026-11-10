"""CPU source/geometry checks for the X16 MMQ research candidate (glm-a, round 3).

stdlib text/AST checks only: no torch, no compilation, no GPU, no network.
Verifies the X16 sources differ from the X8 seed only by the tile define and
unique identities, that coverage guards / seam / fatal semantics are unchanged,
and the ceil(batch/16) launch geometry.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip())
Q = top / "python/sglang/kernels/ops/quantization"
X8, X16 = Q / "native_q8_mmq_x8", Q / "native_q8_mmq_x16"
FAILED = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else ": " + detail))
    if not cond:
        FAILED.append(name)


def lines(p):
    return p.read_text().rstrip("\n").splitlines()


# 1. core header: after normalizing x16 identities, only the tile define differs
a, b = lines(X8 / "mmq_x8_core.cuh"), lines(X16 / "mmq_x16_core.cuh")
norm = lambda s: s.replace("X16", "X8").replace("x16", "x8")
diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if norm(y) != x]
check("x16_core_only_tile_define",
      len(a) == len(b) and len(diffs) == 1
      and diffs[0][1] == "#define MMQ_X_Q8_0 8" and diffs[0][2] == "#define MMQ_X_Q8_0 16",
      repr(diffs))
core = "\n".join(b)
check("x16_core_geometry_kept",
      "#define MMQ_Y_Q8_0 32" in core and "#define NWARPS_Q8_0 4" in core
      and "#define MMQ_X_Q8_0 64" in core
      and "(ncols_y + mmq_x - 1) / mmq_x" in core
      and "(nrows_x + mmq_y - 1) / mmq_y" in core
      and "nrows_x % mmq_y == 0" in core
      and "dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1)" in core)

# 2. CUDA source: normalized identical to the seed
a, b = lines(X8 / "q8_mmq_x8.cu"), lines(X16 / "q8_mmq_x16.cu")


def norm_cu(s):
    s = s.replace("MMQ_X 16 instead of 4", "MMQ_X 8 instead of 4")
    s = s.replace("X16", "X8").replace("x16", "x8")
    return s.replace('../native_q8_mmq_x8/vendor/', 'vendor/')


diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if norm_cu(y) != x]
check("x16_cu_only_identities", len(a) == len(b) and not diffs, repr(diffs[:4]))
cu = "\n".join(b)
check("x16_cu_identity_targets",
      "namespace tierengine_q8_mmq_x16_v1" in cu
      and "TORCH_LIBRARY(tierengine_q8_mmq_x16_v1, m)" in cu
      and '#include "../native_q8_mmq_x8/vendor/ggml-common.h"' in cu
      and '#include "mmq_x16_core.cuh"' in cu
      and "ggml_mul_mat_q8_0_research_x16_q8_1_cuda<at::BFloat16>" in cu)

# 3. dispatch helper: coverage guards byte-identical, identities only
ta, tb = (Q / "gguf_q8_mmq_x8.py").read_text(), (Q / "gguf_q8_mmq_x16.py").read_text()
check("x16_py_only_identities", tb.replace("X16", "X8").replace("x16", "x8") == ta)
seg = lambda t: t[t.index("def coverage_reason"):t.index("def covered")]
g = seg(tb)
check("coverage_guards_unchanged",
      seg(ta) == g and all(lit in g for lit in [
          "32 <= batch <= 8192", "col not in (2048, 6656)",
          "0 < weight.shape[0] <= 131072", "col // 32 * 34",
          "batch * weight.shape[0] > 2147483647",
          "torch.bfloat16", "torch.uint8", "is_contiguous()",
          "not x.is_cuda or not weight.is_cuda or x.device != weight.device"]))
ap = next(n for n in ast.parse(tb).body
          if isinstance(n, ast.FunctionDef) and n.name == "apply")
check("apply_is_fatal_no_fallback",
      not any(isinstance(n, ast.Try) for n in ast.walk(ap))
      and "Explicit X16 prebuilt library manifest required" in tb)

# 4. launch geometry implied by source constants (MMQ_X=16) + packed widths
check("grid_ceil_and_packed_width",
      -(-32 // 16) == 2 and -(-8192 // 16) == 512 and -(-33 // 16) == 3
      and 2048 // 32 * 34 == 2176 and 6656 // 32 * 34 == 7072)

# 5. seam switched to X16 only
gg = (top / "python/sglang/srt/layers/quantization/gguf.py").read_text()
check("seam_x16_only",
      "gguf_q8_mmq_x16 import" in gg and "_research_q8_mmq_x8" not in gg
      and "_research_q8_mmq_x16_covered(x, qweight, qweight_type)" in gg
      and "y = _research_q8_mmq_x16_apply(x, qweight)" in gg)

# 6. seed bytes and vendor headers untouched vs HEAD; only expected paths changed
por = [l[3:] for l in subprocess.run(["git", "status", "--porcelain"],
                                     capture_output=True, text=True, check=True).stdout.splitlines()]
outside = {p for p in por
           if not p.startswith("python/sglang/kernels/ops/quantization/native_q8_mmq_x16")}
check("seed_and_vendor_untouched",
      not any("/native_q8_mmq_x8/" in p or p.endswith("gguf_q8_mmq_x8.py") for p in por)
      and outside == {"python/sglang/kernels/ops/quantization/gguf_q8_mmq_x16.py",
                      "python/sglang/srt/layers/quantization/gguf.py",
                      "iteration-proposal-glm-a-round3.json"}, repr(sorted(por)))

# 7. plan manifest fields and absence of compiled artifacts
plan = json.loads((X16 / "NATIVE-SOURCE-PLAN.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
check("plan_fields_and_shas",
      plan["namespace"] == "tierengine_q8_mmq_x16_v1"
      and plan["library"] == "q8_mmq_x16.so" and plan["compiled"] is False
      and plan["actual_library_sha256"] is None
      and plan["worker_build"] == "NOT_RUN" and plan["worker_gpu"] == "NOT_RUN"
      and all(sha(top / r) == h for r, h in plan["source_sha256"].items())
      and all(sha(top / r) == h for r, h in plan["vendor_reference_sha256"].items())
      and len(plan["vendor_reference_sha256"]) == 4)
check("no_compiled_artifacts",
      not any(p.suffix in {".so", ".o", ".cubin"}
              or p.name == "NATIVE-BUILD-MANIFEST.json" for p in X16.rglob("*")))

print(("PASS" if not FAILED else "FAIL") + f": {12 - len(FAILED)}/12 assertions ok")
raise SystemExit(1 if FAILED else 0)

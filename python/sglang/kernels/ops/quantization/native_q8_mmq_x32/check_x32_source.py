"""CPU source/geometry checks for the X32 MMQ research candidate (glm-a, round 4).

stdlib text/AST checks only: no torch, no compilation, no GPU, no network.
Verifies the X32 sources differ from the retained X16 parent only by the
MMQ_X_Q8_0 tile define and unique identities, that coverage guards / seam /
fatal semantics are unchanged, and the ceil(batch/32) launch geometry.
Integer bounds are checked as source/index controls only, never CUDA math.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip())
Q = top / "python/sglang/kernels/ops/quantization"
X16, X32 = Q / "native_q8_mmq_x16", Q / "native_q8_mmq_x32"
FAILED = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else ": " + detail))
    if not cond:
        FAILED.append(name)


def lines(p):
    return p.read_text().rstrip("\n").splitlines()


def norm32(s):  # undo unique x32 identities and the single-factor define
    s = s.replace("#define MMQ_X_Q8_0 32", "#define MMQ_X_Q8_0 16")
    s = s.replace("MMQ_X 32 instead of 4", "MMQ_X 16 instead of 4")
    return s.replace("X32", "X16").replace("x32", "x16")


# 1. core header: normalized identical to the retained X16 parent
a, b = lines(X16 / "mmq_x16_core.cuh"), lines(X32 / "mmq_x32_core.cuh")
diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if norm32(y) != x]
check("x32_core_single_factor_only", len(a) == len(b) and not diffs, repr(diffs[:4]))
core = "\n".join(b)
check("x32_core_geometry",
      "#define MMQ_X_Q8_0 32" in core and "#define MMQ_Y_Q8_0 32" in core
      and "#define NWARPS_Q8_0 4" in core and "#define MMQ_X_Q8_0 64" in core
      and "#define MMQ_X_Q8_0 16" not in core
      and "(ncols_y + mmq_x - 1) / mmq_x" in core
      and "(nrows_x + mmq_y - 1) / mmq_y" in core
      and "nrows_x % mmq_y == 0" in core
      and "dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1)" in core)

# 2. CUDA source: normalized identical to the parent, identities switched
a, b = lines(X16 / "q8_mmq_x16.cu"), lines(X32 / "q8_mmq_x32.cu")
diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if norm32(y) != x]
check("x32_cu_only_identities", len(a) == len(b) and not diffs, repr(diffs[:4]))
cu = "\n".join(b)
check("x32_cu_identity_targets",
      "namespace tierengine_q8_mmq_x32_v1" in cu
      and "TORCH_LIBRARY(tierengine_q8_mmq_x32_v1, m)" in cu
      and '#include "../native_q8_mmq_x8/vendor/ggml-common.h"' in cu
      and '#include "mmq_x32_core.cuh"' in cu
      and "ggml_mul_mat_q8_0_research_x32_q8_1_cuda<at::BFloat16>" in cu)

# 3. dispatch helper: normalized identical to X16; guards byte-identical
ta, tb = (Q / "gguf_q8_mmq_x16.py").read_text(), (Q / "gguf_q8_mmq_x32.py").read_text()
check("x32_py_only_identities", norm32(tb) == ta)
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
      and "Explicit X32 prebuilt library manifest required" in tb
      and "q8_mmq_x32.so" in tb and "NATIVE-BUILD-MANIFEST.json" in tb)

# 4. launch/index geometry implied by source constants (MMQ_X=32); source control only
check("grid_ceil_and_packed_width",
      -(-32 // 32) == 1 and -(-64 // 32) == 2 and -(-33 // 32) == 2
      and 2048 // 32 * 34 == 2176 and 6656 // 32 * 34 == 7072
      and 4752 + 144 * 32 == 9360)

# 5. seam switched to X32 only
gg = (top / "python/sglang/srt/layers/quantization/gguf.py").read_text()
check("seam_x32_only",
      "gguf_q8_mmq_x32 import" in gg and "_research_q8_mmq_x16" not in gg
      and "_research_q8_mmq_x8" not in gg
      and "_research_q8_mmq_x32_covered(x, qweight, qweight_type)" in gg
      and "y = _research_q8_mmq_x32_apply(x, qweight)" in gg)

# 6. X16/X8/vendor bytes untouched vs HEAD; changed paths limited to the X32
# candidate set (also passes on a fully clean committed tree)
por = [l[3:] for l in subprocess.run(["git", "status", "--porcelain"],
                                     capture_output=True, text=True, check=True).stdout.splitlines()]
AD = "python/sglang/kernels/ops/quantization/native_q8_mmq_x32"
check("parent_seed_and_vendor_untouched",
      not any("/native_q8_mmq_x8/" in p or "/native_q8_mmq_x16/" in p
              or p.endswith("gguf_q8_mmq_x8.py") or p.endswith("gguf_q8_mmq_x16.py")
              or p == "iteration-proposal-glm-a-round3.json" for p in por)
      and all(p.startswith(AD) or p in {
          "python/sglang/kernels/ops/quantization/gguf_q8_mmq_x32.py",
          "python/sglang/srt/layers/quantization/gguf.py",
          "iteration-proposal.json"} for p in por), repr(sorted(por)))

# 7. plan manifest fields and absence of compiled artifacts
plan = json.loads((X32 / "NATIVE-SOURCE-PLAN.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
check("plan_fields_and_shas",
      plan["namespace"] == "tierengine_q8_mmq_x32_v1"
      and plan["library"] == "q8_mmq_x32.so" and plan["compiled"] is False
      and plan["actual_library_sha256"] is None
      and plan["dispatch_helper"] == "../gguf_q8_mmq_x32.py"
      and plan["tile_geometry"] == {"MMQ_X_Q8_0": 32, "MMQ_Y_Q8_0": 32, "NWARPS_Q8_0": 4}
      and all(plan[k] == "NOT_RUN" for k in
              ["worker_build", "worker_gpu", "worker_math", "worker_quality", "worker_performance"])
      and all(sha(top / r) == h for r, h in plan["source_sha256"].items())
      and all(sha(top / r) == h for r, h in plan["vendor_reference_sha256"].items())
      and len(plan["vendor_reference_sha256"]) == 4)
check("no_compiled_artifacts",
      not any(p.suffix in {".so", ".o", ".cubin"}
              or p.name == "NATIVE-BUILD-MANIFEST.json" for p in X32.rglob("*")))

print(("PASS" if not FAILED else "FAIL") + f": {12 - len(FAILED)}/12 assertions ok")
raise SystemExit(1 if FAILED else 0)

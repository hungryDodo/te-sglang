"""CPU source checks for the X64 MMQ research candidate (glm-a, round 5).

stdlib text/AST/SHA checks only: no torch, no compilation, no GPU, no network.
Verifies the X64 sources differ from the retained X32 parent only by the CUDA
#else MMQ_X_Q8_0 define (32->64) and unique identities, that the ROCm branch,
coverage guards, seam shape and fatal semantics are unchanged, and that parent/
vendor bytes are untouched. Passes on a fully clean committed tree.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "5a2354dede460fa90ccc82aa04c4468fab5c3363"
top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip())
Q = top / "python/sglang/kernels/ops/quantization"
X32, X64 = Q / "native_q8_mmq_x32", Q / "native_q8_mmq_x64"
CHECKS, FAILED = [], []


def check(name, cond, detail=""):
    CHECKS.append(bool(cond))
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else ": " + detail))
    if not cond:
        FAILED.append(name)


def norm64(s):  # undo unique x64 identities and the single-factor define
    s = s.replace("#define MMQ_X_Q8_0 64\n#define MMQ_Y_Q8_0 32",
                  "#define MMQ_X_Q8_0 32\n#define MMQ_Y_Q8_0 32")  # CUDA #else only; ROCm is 64/128
    s = s.replace("MMQ_X 64 instead of 4", "MMQ_X 32 instead of 4")
    return s.replace("X64", "X32").replace("x64", "x32")


# 1. core header: normalized identical to the retained X32 parent
a = (X32 / "mmq_x32_core.cuh").read_text()
b = (X64 / "mmq_x64_core.cuh").read_text()
check("x64_core_single_factor_only", norm64(b) == a)
rocm = "#define MMQ_X_Q8_0 64\n#define MMQ_Y_Q8_0 128\n#define NWARPS_Q8_0 8"
check("x64_core_geometry",
      "#define MMQ_X_Q8_0 64\n#define MMQ_Y_Q8_0 32\n#define NWARPS_Q8_0 4" in b
      and "#define MMQ_X_Q8_0 32" not in b and rocm in b
      and "(ncols_y + mmq_x - 1) / mmq_x" in b
      and "(nrows_x + mmq_y - 1) / mmq_y" in b
      and "nrows_x % mmq_y == 0" in b
      and "dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1)" in b)

# 2. CUDA source: normalized identical to the parent, identities switched
a = (X32 / "q8_mmq_x32.cu").read_text()
cu = b = (X64 / "q8_mmq_x64.cu").read_text()
check("x64_cu_only_identities", norm64(b) == a)
check("x64_cu_identity_targets",
      "namespace tierengine_q8_mmq_x64_v1" in cu
      and "TORCH_LIBRARY(tierengine_q8_mmq_x64_v1, m)" in cu
      and '#include "../native_q8_mmq_x8/vendor/ggml-common.h"' in cu
      and '#include "../native_q8_mmq_x8/vendor/quantize_q8_1.cuh"' in cu
      and '#include "mmq_x64_core.cuh"' in cu
      and "ggml_mul_mat_q8_0_research_x64_q8_1_cuda<at::BFloat16>" in cu)

# 3. dispatch helper: normalized identical to X32; guards byte-identical
ta = (Q / "gguf_q8_mmq_x32.py").read_text()
tb = (Q / "gguf_q8_mmq_x64.py").read_text()
check("x64_py_only_identities", norm64(tb) == ta)
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
      and "Explicit X64 prebuilt library manifest required" in tb
      and "q8_mmq_x64.so" in tb and "NATIVE-BUILD-MANIFEST.json" in tb)

# 4. seam switched to X64 only
gg = (top / "python/sglang/srt/layers/quantization/gguf.py").read_text()
check("seam_x64_only",
      "gguf_q8_mmq_x64 import" in gg and "_research_q8_mmq_x32" not in gg
      and "_research_q8_mmq_x16" not in gg and "_research_q8_mmq_x8" not in gg
      and "_research_q8_mmq_x64_covered(x, qweight, qweight_type)" in gg
      and "y = _research_q8_mmq_x64_apply(x, qweight)" in gg)

# 5. launch/index geometry implied by source constants (MMQ_X=64); source control only
check("grid_ceil_and_packed_width",
      -(-32 // 64) == 1 and -(-63 // 64) == 1 and -(-64 // 64) == 1 and -(-65 // 64) == 2
      and 2048 // 32 * 34 == 2176 and 6656 // 32 * 34 == 7072
      and 4752 + 144 * 32 == 9360 and 4752 + 144 * 64 == 13968)  # static sums, not PTXAS

# 6. X8/X16/X32/vendor/round3 bytes identical to base commit; changes limited to
# the X64 candidate set (also passes on a fully clean committed tree)
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True, cwd=top).stdout
old = sorted(p for p in git("ls-tree", "-r", "--name-only", BASE).decode().splitlines()
             if "/native_q8_mmq_x8/" in p or "/native_q8_mmq_x16/" in p
             or "/native_q8_mmq_x32/" in p or p.endswith(("gguf_q8_mmq_x8.py",
                 "gguf_q8_mmq_x16.py", "gguf_q8_mmq_x32.py", "iteration-proposal-glm-a-round3.json")))
check("parent_seed_and_vendor_untouched",
      len(old) == 18 and all((top / p).read_bytes() == git("show", f"{BASE}:{p}") for p in old))
por = [l[3:] for l in subprocess.run(["git", "status", "--porcelain"],
                                     capture_output=True, text=True, check=True, cwd=top).stdout.splitlines()]
AD = "python/sglang/kernels/ops/quantization/native_q8_mmq_x64"
check("changes_limited_to_x64_candidate",
      all(p.startswith(AD) or p in {
          "python/sglang/kernels/ops/quantization/gguf_q8_mmq_x64.py",
          "python/sglang/srt/layers/quantization/gguf.py",
          "iteration-proposal.json"} for p in por), repr(sorted(por)))

# 7. plan manifest fields and absence of compiled artifacts
plan = json.loads((X64 / "NATIVE-SOURCE-PLAN.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
check("plan_fields_and_shas",
      plan["namespace"] == "tierengine_q8_mmq_x64_v1"
      and plan["library"] == "q8_mmq_x64.so" and plan["compiled"] is False
      and plan["actual_library_sha256"] is None and plan["base_commit"] == BASE
      and plan["dispatch_helper"] == "../gguf_q8_mmq_x64.py"
      and plan["tile_geometry"] == {"MMQ_X_Q8_0": 64, "MMQ_Y_Q8_0": 32, "NWARPS_Q8_0": 4}
      and all(plan[k] == "NOT_RUN" for k in
              ["worker_build", "worker_gpu", "worker_math", "worker_quality", "worker_performance"])
      and all(sha(top / r) == h for r, h in plan["source_sha256"].items())
      and all(sha(top / r) == h for r, h in plan["vendor_reference_sha256"].items())
      and len(plan["vendor_reference_sha256"]) == 4)
check("no_compiled_artifacts",
      not any(p.suffix in {".so", ".o", ".cubin"}
              or p.name == "NATIVE-BUILD-MANIFEST.json" for p in X64.rglob("*")))

print(("PASS" if not FAILED else "FAIL") + f": {sum(CHECKS)}/{len(CHECKS)} assertions ok")
raise SystemExit(1 if FAILED else 0)

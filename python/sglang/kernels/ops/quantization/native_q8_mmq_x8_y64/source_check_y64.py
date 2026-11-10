"""CPU source-only controls for the X8/Y64 single-factor candidate (glm-b r4).

Stdlib text/AST checks only, no build, no torch. Run from anywhere:
    python3 python/sglang/kernels/ops/quantization/native_q8_mmq_x8_y64/source_check_y64.py
Compares live sources/hashes to the retained X8/Y32 seed and git HEAD blobs,
so it stays valid on a clean tree after the commit.
"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
SRC = ROOT / "python/sglang/kernels/ops/quantization"
NEW, OLD = SRC / "native_q8_mmq_x8_y64", SRC / "native_q8_mmq_x8"
SEAM = ROOT / "python/sglang/srt/layers/quantization/gguf.py"

# new-identity -> seed-identity (plus the single tile delta Y 64->32)
RULES = [
    ("tierengine_q8_mmq_x8_y64_v1", "tierengine_q8_mmq_x8_v1"),
    ("mul_mat_q8_0_research_x8_y64", "mul_mat_q8_0_research_x8"),
    ("q8_mmq_x8_y64", "q8_mmq_x8"),
    ("SGLANG_Q8_MMQ_X8_Y64", "SGLANG_Q8_MMQ_X8"),
    ("mmq_x8_y64_core.cuh", "mmq_x8_core.cuh"),
    ("../native_q8_mmq_x8/vendor/", "vendor/"),
    ("#define MMQ_Y_Q8_0 64", "#define MMQ_Y_Q8_0 32"),
]


def norm(t):
    for a, b in RULES:
        t = t.replace(a, b)
    return t


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# 1-3: after stripping unique identities + the tile delta, Y64 files equal seed
for newp, seedp in [(NEW / "mmq_x8_y64_core.cuh", OLD / "mmq_x8_core.cuh"),
                    (NEW / "q8_mmq_x8_y64.cu", OLD / "q8_mmq_x8.cu"),
                    (SRC / "gguf_q8_mmq_x8_y64.py", SRC / "gguf_q8_mmq_x8.py")]:
    check("normalized==seed " + newp.name, norm(newp.read_text()) == seedp.read_text())

core = (NEW / "mmq_x8_y64_core.cuh").read_text()
cu = (NEW / "q8_mmq_x8_y64.cu").read_text()
py = (SRC / "gguf_q8_mmq_x8_y64.py").read_text()

# 4: tile factors -- CUDA X8/Y64/warps4, ROCm row untouched
check("cuda tile X8/Y64/NWARPS4",
      "#define MMQ_X_Q8_0 8\n#define MMQ_Y_Q8_0 64\n#define NWARPS_Q8_0 4" in core
      and "#define MMQ_Y_Q8_0 128" in core)
# 5: grid ceil(N/64) via mmq_y and tail-row need_check branch preserved
check("grid ceil + tail rows",
      "(nrows_x + mmq_y - 1) / mmq_y" in core and "nrows_x % mmq_y == 0" in core)
# 6: vendor headers referenced, not duplicated; no vendor copy in NEW dir
check("vendor via ../native_q8_mmq_x8/vendor/",
      '"../native_q8_mmq_x8/vendor/ggml-common.h"' in cu
      and '"../native_q8_mmq_x8/vendor/vecdotq.cuh"' in cu
      and '"../native_q8_mmq_x8/vendor/quantize_q8_1.cuh"' in cu
      and not (NEW / "vendor").exists())
# 7: plan is source-only: compiled false, null library sha, live SHAs recorded
plan = json.loads((NEW / "NATIVE-SOURCE-PLAN.json").read_text())
check("plan source-only + SHAs",
      plan["compiled"] is False and plan["actual_library_sha256"] is None
      and plan["vendor_refs"]["real_dependency_path"] == "../native_q8_mmq_x8/vendor/"
      and plan["source_sha256"]["mmq_x8_y64_core.cuh"] == sha(NEW / "mmq_x8_y64_core.cuh")
      and plan["source_sha256"]["q8_mmq_x8_y64.cu"] == sha(NEW / "q8_mmq_x8_y64.cu")
      and plan["source_sha256"]["gguf_q8_mmq_x8_y64.py"] == sha(SRC / "gguf_q8_mmq_x8_y64.py")
      and all(plan["vendor_refs"]["vendor_sha256"][h.name] == sha(h)
              for h in (OLD / "vendor").iterdir()))
# 8: no fabricated binaries or build manifest in the candidate dir
check("no binary artifacts",
      not list(NEW.glob("*.so")) and not (NEW / "NATIVE-BUILD-MANIFEST.json").exists()
      and all(plan[k] == "NOT_RUN" for k in
              ("worker_build", "gpu", "math", "quality", "performance")))
# 9: gguf.py seam switched to the Y64 helper only; no bare X8 helper left
seam = SEAM.read_text()
check("seam switched to y64",
      "gguf_q8_mmq_x8_y64 import" in seam
      and re.search(r"_research_q8_mmq_x8_(apply|covered)\b", seam) is None
      and "_research_q8_mmq_x8_y64_covered(x, qweight, qweight_type)" in seam
      and "= _research_q8_mmq_x8_y64_apply(x, qweight)" in seam)
# 10: seed tree + vendor bytes identical to git HEAD blobs (clean-tree safe)
ok = True
for rel in ["python/sglang/kernels/ops/quantization/native_q8_mmq_x8/mmq_x8_core.cuh",
            "python/sglang/kernels/ops/quantization/native_q8_mmq_x8/q8_mmq_x8.cu",
            "python/sglang/kernels/ops/quantization/gguf_q8_mmq_x8.py"] + \
           [str(p.relative_to(ROOT)) for p in sorted((OLD / "vendor").iterdir())]:
    blob = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:" + rel],
                          capture_output=True).stdout
    ok = ok and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)
check("seed+vendor bytes == HEAD", ok)
# 11: everything still parses (stdlib AST)
try:
    ast.parse(py)
    ast.parse(seam)
    ok = True
except SyntaxError:
    ok = False
check("python AST parse", ok)

print("RESULT:", "ALL_PASS" if not fails else "FAILURES=" + ",".join(fails))
raise SystemExit(1 if fails else 0)

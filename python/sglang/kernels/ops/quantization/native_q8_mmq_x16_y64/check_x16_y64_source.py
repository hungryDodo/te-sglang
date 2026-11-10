"""CPU source/identity checks only. Does not compile or execute CUDA math."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[6]
Q = ROOT / 'python/sglang/kernels/ops/quantization'
OLD = Q / 'native_q8_mmq_x8_y64'
NEW = Q / 'native_q8_mmq_x16_y64'
PARENT = '92156e8b88ba8a87aaf23ed2b5f8758c693016fd'
checks = []
def check(name, ok):
    checks.append({'name': name, 'passed': bool(ok)})
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def base(path):
    return subprocess.check_output(['git', 'show', PARENT + ':' + str(path.relative_to(ROOT))], cwd=ROOT)
core = (NEW / 'mmq_x16_y64_core.cuh').read_text()
expected = (OLD / 'mmq_x8_y64_core.cuh').read_text().replace('research_x8_y64', 'research_x16_y64').replace('#define MMQ_X_Q8_0 8\n', '#define MMQ_X_Q8_0 16\n').replace('CUDA-only research X8 variant', 'CUDA-only research X16/Y64 variant')
check('core_exact_X8_to16_with_Y64_warps4_and_K_body_unchanged', core == expected and '#define MMQ_Y_Q8_0 64' in core and '#define NWARPS_Q8_0 4' in core)
cu = (NEW / 'q8_mmq_x16_y64.cu').read_text()
expected = (OLD / 'q8_mmq_x8_y64.cu').read_text().replace('research_x8_y64', 'research_x16_y64').replace('tierengine_q8_mmq_x8_y64_v1', 'tierengine_q8_mmq_x16_y64_v1').replace('mmq_x8_y64_core.cuh', 'mmq_x16_y64_core.cuh').replace('MMQ_X 8 instead of 4', 'MMQ_X 16 and MMQ_Y 64').replace('"X8 ', '"X16/Y64 ')
check('CU_exact_identity_only_and_original_vendor_paths', cu == expected)
hook = (Q / 'gguf_q8_mmq_x16_y64.py').read_text()
old_hook = (Q / 'gguf_q8_mmq_x8_y64.py').read_text()
expected = old_hook.replace('SGLANG_Q8_MMQ_X8_Y64', 'SGLANG_Q8_MMQ_X16_Y64').replace('tierengine_q8_mmq_x8_y64_v1', 'tierengine_q8_mmq_x16_y64_v1').replace('native_q8_mmq_x8_y64', 'native_q8_mmq_x16_y64').replace('q8_mmq_x8_y64.so', 'q8_mmq_x16_y64.so').replace('Controller-origin X8 research route', 'Controller-origin X16/Y64 research route').replace('Explicit X8 prebuilt', 'Explicit X16/Y64 prebuilt').replace('Registered X8 library', 'Registered X16/Y64 library')
check('hook_exact_identity_only', hook == expected)
check('guards_byte_identical', hook.split('def coverage_reason', 1)[1].split('def covered', 1)[0] == old_hook.split('def coverage_reason', 1)[1].split('def covered', 1)[0])
ast.parse(hook)
gguf = ROOT / 'python/sglang/srt/layers/quantization/gguf.py'
check('GGUF_full_source_only_dispatch_identity_seam', gguf.read_bytes() == base(gguf).replace(b'x8_y64', b'x16_y64'))
old_paths = [p for d in (OLD, Q / 'native_q8_mmq_x8') for p in d.rglob('*') if p.is_file()] + [Q / 'gguf_q8_mmq_x8_y64.py', Q / 'gguf_q8_mmq_x8.py']
check('old_sources_and_vendor_unmodified_from_Provider_parent', all(p.read_bytes() == base(p) for p in old_paths))
plan = json.loads((NEW / 'NATIVE-SOURCE-PLAN.json').read_bytes())
check('declared_source_SHA_actual', all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items()))
check('four_actual_vendor_SHA_and_parent_bytes', len(plan['vendor_sha256']) == 4 and all(sha(ROOT / name) == digest and (ROOT / name).read_bytes() == base(ROOT / name) for name, digest in plan['vendor_sha256'].items()))
check('canonical_helper_reference', (NEW / plan['dispatch_helper']).resolve() == Q / 'gguf_q8_mmq_x16_y64.py')
check('uncompiled_no_actual_native_artifacts', plan['compiled'] is False and plan['actual_library_sha256'] is None and not any(p.suffix in ('.so', '.o', '.cubin') or p.name == 'NATIVE-BUILD-MANIFEST.json' for p in NEW.iterdir()))
check('tile_and_origin_manifest', plan['tile'] == {'mmq_x': 16, 'mmq_y': 64, 'nwarps': 4} and plan['origin'] == 'CONTROLLER_PRODUCED_NEW_NO_PROVIDER_CODE_DELIVERY')
# Integer launch coverage is not execution of CUDA indexing or proof of math.
check('declared_B_N_tail_grid_coverage_only', all(((b + 15)//16)*16 >= b and ((n + 63)//64)*64 >= n for b in (32,33,63,64,65) for n in (31,33,2048,6656)))
print(json.dumps({'scope':'CPU_SOURCE_IDENTITY_ONLY_NOT_CUDA_MATH', 'passed':sum(c['passed'] for c in checks), 'total':len(checks), 'checks':checks}, indent=2))
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)

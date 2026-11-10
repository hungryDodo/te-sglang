"""Controller Q8_0/BF16 batch1 research route; target failures are fatal.

Local library SHA checking is not independent loaded-library attestation.
"""
from __future__ import annotations
import functools
import hashlib
import json
from pathlib import Path
import torch

_NAMESPACE = 'tierengine_q8_gemv_y4_v1'

def coverage_reason(x, weight, quant_type):
    if not isinstance(quant_type, int) or quant_type != 8:
        return 'other_quant_type'
    if x.ndim != 2 or weight.ndim != 2:
        return 'rank'
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        return 'device'
    if x.dtype != torch.bfloat16 or weight.dtype != torch.uint8:
        return 'dtype'
    if not x.is_contiguous() or not weight.is_contiguous():
        return 'layout'
    if x.shape[0] != 1:
        return 'batch_not_one'
    col = x.shape[1]
    if col not in (2048, 6656) or not 2048 <= weight.shape[0] <= 131072:
        return 'fixed_model_dimensions'
    if weight.shape[1] != col // 32 * 34:
        return 'packed_q8_width'
    return None

def covered(x, weight, quant_type):
    return coverage_reason(x, weight, quant_type) is None

@functools.lru_cache(maxsize=1)
def _operator():
    root = Path(__file__).with_name('native_q8_gemv_y4')
    binding = json.loads((root / 'NATIVE-BUILD-MANIFEST.json').read_bytes())
    if (binding.get('schema_version') != 1 or binding.get('namespace') != _NAMESPACE
            or binding.get('library') != 'q8_gemv_y4.so'):
        raise RuntimeError('Explicit Y4 prebuilt research library manifest required')
    library = root / binding['library']
    if (library.is_symlink() or not library.is_file()
            or hashlib.sha256(library.read_bytes()).hexdigest() != binding.get('library_sha256')):
        raise RuntimeError('Registered Y4 library bytes differ')
    torch.ops.load_library(str(library))
    return torch.ops.tierengine_q8_gemv_y4_v1.gemv.default

def apply(x, weight):
    # Target selection happened before this call. Never hide load/launch errors.
    return _operator()(weight, x)

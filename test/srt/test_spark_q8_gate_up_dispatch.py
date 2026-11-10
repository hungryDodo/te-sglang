"""AST-extracted real Python routes with CPU-only sentinel tensors.

No CUDA, model import, tensor arithmetic, GPU numeric equivalence or performance
is tested here. Sentinels check dispatch, packed row ordering and safe fallback.
"""
import ast
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'python/sglang/srt/layers/quantization/gguf.py'


class Tensor:
    def __init__(self, rows, cols, *, dtype='u8', device='cuda:0', contiguous=True, start=0):
        self.shape = (rows, cols)
        self.dtype, self.device = dtype, device
        self.is_cuda = device.startswith('cuda')
        self._contiguous, self.start = contiguous, start
        self.ndim = 2

    def is_contiguous(self): return self._contiguous

    def contiguous(self):
        self._contiguous = True
        return self

    def __getitem__(self, selection):
        rows, cols = selection
        start, stop, _ = rows.indices(self.shape[0])
        _, width, _ = cols.indices(self.shape[1])
        return Tensor(stop - start, width, dtype=self.dtype, device=self.device, start=self.start + start)


class Output:
    def __init__(self, rows, columns):
        self.shape = (rows, len(columns))
        self.columns = list(columns)

    def add_(self, bias):
        self.columns = [value + bias for value in self.columns]
        return self


@pytest.fixture
def extracted(monkeypatch):
    tree = ast.parse(SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                '_ordered_gguf_shard_ids', '_can_fuse_spark_q8_gate_up', 'fused_mul_mat_gguf'):
            nodes.append(copy.deepcopy(node))
        if isinstance(node, ast.ClassDef) and node.name == 'GGUFLinearMethod':
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in ('apply', 'process_weights_after_loading'):
                    nodes.append(copy.deepcopy(method))
    assert len(nodes) == 5, 'Extracted source contracts changed; review fixture'
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                      type_ignores=[])
    ast.fix_missing_locations(code)
    distributed = ModuleType('sglang.srt.distributed.utils')
    distributed.get_group_rank_size = lambda group: (0, group)
    monkeypatch.setitem(sys.modules, 'sglang.srt.distributed.utils', distributed)
    calls = []
    def kernel(kind):
        def run(weight, x, qtype, rows):
            calls.append((kind, weight.start, rows, qtype))
            return Output(x.shape[0], range(weight.start, weight.start + rows))
        return run
    def cat(values, axis):
        assert axis == 1
        calls.append(('cat', len(values)))
        return Output(values[0].shape[0], [item for value in values for item in value.columns])
    namespace = {'torch': SimpleNamespace(bfloat16='bf16', float16='f16', uint8='u8',
        cat=cat, empty=lambda rows, cols, **kwargs: Output(rows, range(cols))),
        'IMATRIX_QUANT_TYPES': {16}, 'UNQUANTIZED_TYPES': {0, 1, 30},
        'MMVQ_QUANT_TYPES': {8}, 'MMQ_QUANT_TYPES': {8}, 'DEQUANT_TYPES': {8},
        'ggml_mul_mat_vec_a8': kernel('MMVQ'), 'ggml_mul_mat_a8': kernel('MMQ'),
        'logger': SimpleNamespace(info=lambda *args: None)}
    exec(compile(code, str(SOURCE), 'exec'), namespace)
    actual_fused = namespace['fused_mul_mat_gguf']
    route = kernel('route')
    namespace['fused_mul_mat_gguf'] = lambda x, weight, qtype: route(weight, x, qtype, weight.shape[0])
    return SimpleNamespace(ns=namespace, calls=calls, predicate=namespace['_can_fuse_spark_q8_gate_up'],
        apply=namespace['apply'], process=namespace['process_weights_after_loading'], fused=actual_fused)


def layer():
    qweight = Tensor(13312, 2176)
    qweight.shard_id = [0, 1]
    qweight.shard_offset_map = {0: (0, 6656, 2176), 1: (6656, 13312, 2176)}
    return SimpleNamespace(prefix='model.layers.0.mlp.gate_up_proj', input_size=2048,
        output_sizes=[6656, 6656], output_partition_sizes=[6656, 6656], tp_group=1,
        qweight=qweight, qweight_type=SimpleNamespace(weight_type=0, shard_weight_type={0: 8, 1: 8}))


@pytest.mark.parametrize('fault', ['prefix', 'input', 'output', 'partition', 'TP2', 'order',
    'mixed', 'padding', 'gap', 'CPU', 'device', 'dtype', 'noncontiguous', 'missing-shards',
    'missing-offsets', 'missing-types', 'missing-TP'])
def test_predicate_unknown_layouts_safely_return_false(extracted, fault):
    item = layer()
    assert extracted.predicate(item) is True
    if fault == 'prefix': item.prefix = 'model.layers.0.attn.qkv_proj'
    elif fault == 'input': item.input_size = 1024
    elif fault == 'output': item.output_sizes = [3328, 3328]
    elif fault == 'partition': item.output_partition_sizes = [3328, 3328]
    elif fault == 'TP2': item.tp_group = 2
    elif fault == 'order': item.qweight.shard_id = [1, 0]
    elif fault == 'mixed': item.qweight_type.shard_weight_type[1] = 2
    elif fault == 'padding': item.qweight.shard_offset_map[1] = (6656, 13312, 2112)
    elif fault == 'gap': item.qweight.shard_offset_map[1] = (6657, 13312, 2176)
    elif fault == 'CPU': item.qweight.is_cuda = False; item.qweight.device = 'cpu'
    elif fault == 'device': item.qweight.shape = (13312, 2304)
    elif fault == 'dtype': item.qweight.dtype = 'f16'
    elif fault == 'noncontiguous': item.qweight._contiguous = False
    elif fault == 'missing-shards': del item.qweight.shard_id
    elif fault == 'missing-offsets': del item.qweight.shard_offset_map
    elif fault == 'missing-types': del item.qweight_type.shard_weight_type
    elif fault == 'missing-TP': del item.tp_group
    assert extracted.predicate(item) is False


@pytest.mark.parametrize('batch', [0, 1, 2, 3, 6])
def test_apply_one_call_exact_gate_then_up_order_and_bias(extracted, batch):
    item = layer(); item.qweight.spark_q8_gate_up_fused = extracted.predicate(item)
    x = Tensor(batch, 2048, dtype='bf16')
    result = extracted.apply(None, item, x, bias=17)
    assert result.shape == (batch, 13312)
    assert result.columns == list(range(17, 13329))
    assert extracted.calls == [('route', 0, 13312, 8)]


@pytest.mark.parametrize('fault', ['stale-type', 'stale-offset', 'TP2', 'CPU', 'replacement',
                                 'reverse-order', 'x-dtype', 'x-device'])
def test_apply_stale_marker_or_changed_parameter_keeps_original_path(extracted, fault):
    item = layer();item.qweight.spark_q8_gate_up_fused = True
    x = Tensor(1, 2048, dtype='bf16')
    if fault == 'stale-type': item.qweight_type.shard_weight_type[1] = 2
    elif fault == 'stale-offset': item.qweight.shard_offset_map[1] = (6656, 13312, 2112)
    elif fault == 'TP2': item.tp_group = 2
    elif fault == 'CPU': item.qweight.is_cuda = False;item.qweight.device = 'cpu';x.device = 'cpu'
    elif fault == 'replacement':
        original = item.qweight;item.qweight = Tensor(13312, 2176)
        item.qweight.shard_id = original.shard_id;item.qweight.shard_offset_map = original.shard_offset_map
    elif fault == 'reverse-order': item.qweight.shard_id = [1, 0]
    elif fault == 'x-dtype': x.dtype = 'f32'
    elif fault == 'x-device': x.device = 'cuda:1'
    result = extracted.apply(None, item, x, bias=9)
    assert [call[0] for call in extracted.calls] == ['route', 'route', 'cat']
    assert [call[1:3] for call in extracted.calls[:2]] == [(0, 6656), (6656, 6656)]
    assert result.columns == list(range(9, 13321))


def test_after_loading_recomputes_and_clears_cached_marker(extracted):
    item = layer(); method = SimpleNamespace(_create_padded_weight_param=lambda value: None)
    extracted.process(method, item)
    assert item.qweight.spark_q8_gate_up_fused is True
    item.qweight_type.shard_weight_type[1] = 2
    extracted.process(method, item)
    assert item.qweight.spark_q8_gate_up_fused is False


@pytest.mark.parametrize('rows,cutoff', [(6656, 2), (13312, 2), (3328, 6)])
@pytest.mark.parametrize('batch', [0, 1, 2, 3, 6])
def test_real_fused_mul_dispatch_cutoff_preserved_only_for_registered_TP1(extracted, rows, cutoff, batch):
    result = extracted.fused(Tensor(batch, 2048, dtype='bf16'), Tensor(rows, 2176), 8)
    assert result.shape == (batch, rows)
    assert [call[0] for call in extracted.calls] == ([] if batch == 0 else ['MMVQ' if batch <= cutoff else 'MMQ'])

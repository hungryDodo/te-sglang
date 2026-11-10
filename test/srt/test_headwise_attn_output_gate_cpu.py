#!/usr/bin/env python3
"""Stdlib-only CPU regression for the Spark2_5 headwise attention-output gate.

Runs the ACTUAL helper source (python/sglang/srt/layers/headwise_attn_output_gate.py)
and the ACTUAL call site (python/sglang/srt/models/spark2_5.py) via:
  1. AST static checks on the real source files (explicit FP32-sigmoid ->
     cast-to-attn-dtype -> multiply ordering, new-output allocation, no
     in-place ops, lazy triton import, guarded call-site wiring), and
  2. Behavioral execution of the real helper Python code under injected FAKE
     torch/triton modules with independent fake tensors (shape/dtype/device/
     contiguity guards, B0/B1/B32/B64 and odd shapes, no-alias/no-mutation,
     fallback==baseline outcomes).

No torch, triton, sglang, or CUDA is imported or required.

LIMITATIONS (engineering controls only): the Triton kernel source is never
compiled or executed. The fake-Triton launch mirror re-derives gate indexing
(row*num_heads+head) and rounding ORDER from the recorded launch arguments
per spec; it cannot detect a divergence between the Triton source text and
this mirror. Fake tensors carry float64 scalars (fp32 storage rounding is not
emulated); fake bf16 casts/multiplies emulate round-to-nearest-even; fake fp16
rounding is NOT emulated (passthrough). No GPU math equivalence, sigmoid
implementation identity, JIT compilation, performance, or quality is
established here; that requires the separate GPU script by the Controller.
"""

import ast
import importlib.util
import math
import struct
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "python/sglang/srt/layers/headwise_attn_output_gate.py"
SPARK_PATH = ROOT / "python/sglang/srt/models/spark2_5.py"

RESULTS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  :: " + str(detail)))
    return ok


# --------------------------------------------------------------------------
# Fake dtype / device / tensor + fake torch + fake triton
# --------------------------------------------------------------------------
class FakeDType:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, FakeDType) and other.name == self.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "torch." + self.name


BF16, FP16, FP32, FP64 = (FakeDType(n) for n in ("bfloat16", "float16", "float32", "float64"))


class FakeDevice:
    def __init__(self, kind, index=0):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, FakeDevice)
            and other.type == self.type
            and other.index == self.index
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.type, self.index))

    def __repr__(self):
        return f"{self.type}:{self.index}"


CUDA0 = FakeDevice("cuda", 0)
CPU = FakeDevice("cpu", 0)


def bf16_round(x):
    """Round a float64 scalar to bfloat16 precision (round-to-nearest-even)."""
    if x != x or x in (float("inf"), float("-inf")):
        return x
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    lower, upper = bits & 0xFFFF, bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper += 1
    if upper >= 0x7F80:  # exponent saturated -> +/-inf
        return float("inf") if x > 0 else float("-inf")
    return struct.unpack(">f", struct.pack(">I", (upper << 16) & 0xFFFFFFFF))[0]


def fake_sigmoid(v):
    if v >= 0:
        if v > 500:
            return 1.0
        return 1.0 / (1.0 + math.exp(-v))
    if v < -500:
        return 0.0
    e = math.exp(v)
    return e / (1.0 + e)


def cast_value(v, dtype):
    if dtype == BF16:
        return bf16_round(v)
    # fp16 rounding NOT emulated (documented limitation); fp32/fp64 passthrough.
    return v


def mul_round(a, b, dtype):
    if dtype == BF16:
        return bf16_round(a * b)
    return a * b


def _unravel(flat, shape):
    idx = [0] * len(shape)
    for d in range(len(shape) - 1, -1, -1):
        idx[d] = flat % shape[d]
        flat //= shape[d]
    return tuple(idx)


def _broadcast(a, b):
    la, lb = len(a), len(b)
    if lb > la:
        a, b = b, a
        la = lb
    b = (1,) * (la - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise RuntimeError(
                f"The size of tensor a ({x}) must match the size of tensor b ({y}) at non-singleton dimension"
            )
    return tuple(out)


class FakeTensor:
    def __init__(self, data, dtype, device, shape, contiguous=True):
        self.data = list(data)
        self.dtype = dtype
        self.device = device
        self.shape = tuple(shape)
        self._contig = bool(contiguous)

    # --- metadata used by guards ---
    def dim(self):
        return len(self.shape)

    def numel(self):
        return len(self.data)

    def is_contiguous(self):
        return self._contig

    # --- baseline math surface ---
    def float(self):
        return FakeTensor(self.data, FP32, self.device, self.shape, self._contig)

    def to(self, dtype):
        return FakeTensor(
            [cast_value(v, dtype) for v in self.data], dtype, self.device, self.shape, self._contig
        )

    def view(self, *sizes):
        ints = []
        for s in sizes:
            if not isinstance(s, int) or isinstance(s, bool) or s < 0:
                raise TypeError(f"view size {s!r} must be an int")
            ints.append(s)
        total = 1
        for s in ints:
            total *= s
        if total != len(self.data):
            raise RuntimeError(
                f"shape {tuple(ints)} invalid for tensor of size {len(self.data)}"
            )
        return FakeTensor(self.data, self.dtype, self.device, tuple(ints), self._contig)

    def unsqueeze(self, dim):
        shape = list(self.shape)
        if dim == -1:
            shape.append(1)
        else:
            shape.insert(dim, 1)
        return FakeTensor(self.data, self.dtype, self.device, tuple(shape), self._contig)

    def __mul__(self, other):
        if not isinstance(other, FakeTensor):
            return NotImplemented
        if self.dtype != other.dtype:
            raise RuntimeError("fake mul requires identical dtypes")
        out_shape = _broadcast(self.shape, other.shape)
        total = 1
        for s in out_shape:
            total *= s
        padded_a = (1,) * (len(out_shape) - len(self.shape)) + self.shape
        padded_b = (1,) * (len(out_shape) - len(other.shape)) + other.shape
        res = []
        for flat in range(total):
            idx = _unravel(flat, out_shape)
            ia = tuple(0 if padded_a[d] == 1 else idx[d] for d in range(len(out_shape)))
            ib = tuple(0 if padded_b[d] == 1 else idx[d] for d in range(len(out_shape)))
            off_a, off_b = ia[0], ib[0]
            for d in range(1, len(out_shape)):
                off_a = off_a * padded_a[d] + ia[d]
                off_b = off_b * padded_b[d] + ib[d]
            res.append(mul_round(self.data[off_a], other.data[off_b], self.dtype))
        return FakeTensor(res, self.dtype, self.device, out_shape, True)


def fake_torch_sigmoid(t):
    if not isinstance(t, FakeTensor):
        raise TypeError("torch.sigmoid expects a tensor")
    return FakeTensor(
        [fake_sigmoid(v) for v in t.data], FP32, t.device, t.shape, t.is_contiguous()
    )


def fake_empty_like(t):
    return FakeTensor([0.0] * t.numel(), t.dtype, t.device, t.shape, True)


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
fake_torch.bfloat16 = BF16
fake_torch.float16 = FP16
fake_torch.float32 = FP32
fake_torch.float64 = FP64
fake_torch.sigmoid = fake_torch_sigmoid
fake_torch.empty_like = fake_empty_like


class LaunchLog:
    def __init__(self):
        self.launches = []

    def mirror(self, launch):
        """Independent spec mirror: fill the fake output using ONLY the recorded
        launch arguments, per spec row*num_heads+head and rounded-gate order."""
        attn, gate, out, n, head_stride, num_heads, head_dim = launch["args"][:7]
        bs = launch["BLOCK_SIZE"]
        assert isinstance(bs, int) and bs > 0, f"bad BLOCK_SIZE {bs!r}"
        assert n == attn.numel(), f"n_elements {n} != attn numel {attn.numel()}"
        assert head_stride == num_heads * head_dim == attn.shape[1], (
            f"head_stride {head_stride} != num_heads*head_dim {num_heads * head_dim} / cols {attn.shape[1]}"
        )
        expected_grid = (n + bs - 1) // bs
        assert launch["grid"][0] == expected_grid, (
            f"grid {launch['grid'][0]} != ceil({n}/{bs})={expected_grid}"
        )
        assert out is not attn and out is not gate, "kernel must write a new output tensor"
        assert out.shape == attn.shape and out.dtype == attn.dtype
        for i in range(n):
            row = i // head_stride
            within = i - row * head_stride
            head = within // head_dim
            gi = row * num_heads + head
            assert 0 <= gi < gate.numel(), f"gate index {gi} out of range"
            out.data[i] = mul_round(attn.data[i], cast_value(fake_sigmoid(gate.data[gi]), attn.dtype), attn.dtype)


LAUNCH_LOG = LaunchLog()


def fake_jit(fn):
    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launch_record = {
                    "fn": fn.__name__,
                    "grid": grid,
                    "args": args,
                    "BLOCK_SIZE": kwargs.get("BLOCK_SIZE"),
                    "num_warps": kwargs.get("num_warps"),
                }
                LAUNCH_LOG.launches.append(launch_record)
                LAUNCH_LOG.mirror(launch_record)

            return launch

    return FakeKernel()


fake_triton = types.ModuleType("triton")
fake_triton.jit = fake_jit
fake_tl = types.ModuleType("triton.language")
fake_tl.constexpr = "tl.constexpr"
fake_tl.float32 = "tl.float32"
fake_tl.int64 = "tl.int64"
fake_tl.sigmoid = "tl.sigmoid"
fake_tl.load = "tl.load"
fake_tl.store = "tl.store"
fake_tl.program_id = "tl.program_id"
fake_tl.arange = "tl.arange"
fake_triton.language = fake_tl


def load_helper_module():
    sys.modules["torch"] = fake_torch
    sys.modules["triton"] = fake_triton
    sys.modules["triton.language"] = fake_tl
    spec = importlib.util.spec_from_file_location(
        "headwise_attn_output_gate_under_test", str(HELPER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# AST static checks on the real sources
# --------------------------------------------------------------------------
def attr_call(node, attr=None):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (attr is None or node.func.attr == attr)
    )


def is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def ast_checks():
    helper_tree = ast.parse(HELPER_PATH.read_text())
    spark_tree = ast.parse(SPARK_PATH.read_text())

    # No module-level triton import in the helper (lazy import only).
    top_imports = [
        n.module if isinstance(n, ast.ImportFrom) else ",".join(a.name for a in n.names)
        for n in helper_tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    check(
        "ast: helper top-level imports are torch only (triton lazy)",
        all("triton" not in (m or "") for m in top_imports) and any("torch" in (m or "") for m in top_imports),
        top_imports,
    )

    # No in-place tensor ops anywhere in the helper.
    in_place = [
        n.func.attr
        for n in ast.walk(helper_tree)
        if attr_call(n) and isinstance(n.func.attr, str) and n.func.attr.endswith("_")
        and n.func.attr in {"mul_", "add_", "div_", "sub_", "mul"}
    ]
    check("ast: helper contains no in-place mul_/add_/div_/sub_ calls", not in_place, in_place)

    funcs = {n.name: n for n in ast.walk(helper_tree) if isinstance(n, ast.FunctionDef)}
    check("ast: baseline helper function exists", "headwise_attn_output_gate_baseline" in funcs)
    check("ast: apply function exists", "apply_headwise_attn_output_gate" in funcs)
    check("ast: guard function exists", "_headwise_gate_launch_eligible" in funcs)

    base = funcs.get("headwise_attn_output_gate_baseline")
    if base is not None:
        stmts = [
            s
            for s in base.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]  # skip docstring
        # stmt0: g = torch.sigmoid(g.float()).to(attn_output.dtype)
        s0 = stmts[0]
        ok0 = (
            isinstance(s0, ast.Assign)
            and is_name(s0.targets[0], "g")
            and attr_call(s0.value, "to")
            and attr_call(s0.value.func.value, "sigmoid")
            and is_name(s0.value.func.value.func.value, "torch")
            and attr_call(s0.value.func.value.args[0], "float")
            and is_name(s0.value.func.value.args[0].func.value, "g")
            and isinstance(s0.value.args[0], ast.Attribute)
            and s0.value.args[0].attr == "dtype"
            and is_name(s0.value.args[0].value, "attn_output")
        )
        check("ast: baseline = torch.sigmoid(g.float()).to(attn_output.dtype) first", ok0, ast.dump(s0) if not ok0 else "")
        # stmt1: gate_output = attn_output.view(shape[0], num_heads, head_dim) * g.unsqueeze(-1)
        s1 = stmts[1]
        ok1 = (
            isinstance(s1, ast.Assign)
            and is_name(s1.targets[0], "gate_output")
            and isinstance(s1.value, ast.BinOp)
            and isinstance(s1.value.op, ast.Mult)
            and attr_call(s1.value.left, "view")
            and is_name(s1.value.left.func.value, "attn_output")
            and attr_call(s1.value.right, "unsqueeze")
            and is_name(s1.value.right.func.value, "g")
        )
        check("ast: baseline multiplies view(attn) by g.unsqueeze(-1)", ok1)
        # stmt2: return gate_output.view(*attn_output.shape)
        s2 = stmts[-1]
        ok2 = (
            isinstance(s2, ast.Return)
            and attr_call(s2.value, "view")
            and is_name(s2.value.func.value, "gate_output")
            and len(s2.value.args) == 1
            and isinstance(s2.value.args[0], ast.Starred)
            and isinstance(s2.value.args[0].value, ast.Attribute)
            and s2.value.args[0].value.attr == "shape"
            and is_name(s2.value.args[0].value.value, "attn_output")
        )
        check("ast: baseline returns gate_output.view(*attn_output.shape)", ok2)

    kern = next(
        (n for n in ast.walk(helper_tree) if isinstance(n, ast.FunctionDef) and n.name == "_headwise_gate_kernel"),
        None,
    )
    check("ast: triton kernel _headwise_gate_kernel defined in helper", kern is not None)
    if kern is not None:
        assigns = {}
        for st in kern.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                assigns[st.targets[0].id] = st.value
        ok_load_g = (
            "gate_val" in assigns
            and attr_call(assigns["gate_val"], "load")
            and assigns["gate_val"].args
        )
        check("ast: kernel loads gate via tl.load", ok_load_g)
        ok_f32 = (
            "gate_f32" in assigns
            and attr_call(assigns["gate_f32"], "sigmoid")
            and assigns["gate_f32"].func.value.id == "tl"
            and attr_call(assigns["gate_f32"].args[0], "to")
            and assigns["gate_f32"].args[0].args
            and assigns["gate_f32"].args[0].args[0].attr == "float32"
        )
        check("ast: kernel sigmoid input is gate cast to tl.float32", ok_f32)
        ok_round = (
            "gate_rounded" in assigns
            and attr_call(assigns["gate_rounded"], "to")
            and isinstance(assigns["gate_rounded"].args[0], ast.Attribute)
            and assigns["gate_rounded"].args[0].attr == "dtype"
            and is_name(assigns["gate_rounded"].args[0].value, "attn_val")
        )
        check("ast: kernel rounds sigmoid to attn dtype BEFORE multiply", ok_round)
        ok_mul = (
            "out_val" in assigns
            and isinstance(assigns["out_val"], ast.BinOp)
            and isinstance(assigns["out_val"].op, ast.Mult)
            and is_name(assigns["out_val"].left, "attn_val")
            and is_name(assigns["out_val"].right, "gate_rounded")
        )
        check("ast: kernel multiplies attn_val * gate_rounded (rounded gate)", ok_mul)
        stores = [
            n for n in ast.walk(kern) if attr_call(n, "store")
        ]
        ok_store = bool(stores) and all(
            isinstance(s.args[0], ast.BinOp)
            and isinstance(s.args[0].op, ast.Add)
            and is_name(s.args[0].left, "out_ptr")
            for s in stores
        )
        check("ast: kernel stores ONLY to out_ptr (never attn/gate ptrs)", ok_store)
        ok_idx = (
            "gate_index" in assigns
            and isinstance(assigns["gate_index"], ast.BinOp)
            and isinstance(assigns["gate_index"].op, ast.Add)
            and isinstance(assigns["gate_index"].left, ast.BinOp)
            and isinstance(assigns["gate_index"].left.op, ast.Mult)
            and is_name(assigns["gate_index"].left.left, "row")
            and is_name(assigns["gate_index"].left.right, "num_heads")
            and is_name(assigns["gate_index"].right, "head")
        )
        check("ast: kernel gate index = row*num_heads + head", ok_idx)

    apply_fn = funcs.get("apply_headwise_attn_output_gate")
    if apply_fn is not None:
        body = apply_fn.body
        empty_like_calls = [
            n for n in ast.walk(apply_fn) if attr_call(n, "empty_like")
        ]
        check("ast: apply allocates new output via torch.empty_like", bool(empty_like_calls))
        fallback_returns = [
            n
            for n in ast.walk(apply_fn)
            if isinstance(n, ast.Return)
            and isinstance(n.value, ast.Call)
            and is_name(n.value.func, "headwise_attn_output_gate_baseline")
        ]
        check("ast: apply fallback returns exact baseline call", bool(fallback_returns))
        idx_baseline = next(
            (
                i
                for i, st in enumerate(body)
                if any(
                    isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Call)
                    and is_name(n.value.func, "headwise_attn_output_gate_baseline")
                    for n in ast.walk(st)
                )
            ),
            10**9,
        )
        idx_alloc = next(
            (
                i
                for i, st in enumerate(body)
                if isinstance(st, ast.Assign)
                and attr_call(st.value, "empty_like")
            ),
            -1,
        )
        check("ast: empty-batch/fallback return precedes any kernel allocation", idx_baseline < idx_alloc, (idx_baseline, idx_alloc))
        has_empty_check = any(
            isinstance(n, ast.Compare)
            and attr_call(n.left, "numel")
            and any(isinstance(c, ast.Constant) and c.value == 0 for c in n.comparators)
            for n in ast.walk(apply_fn)
        )
        check("ast: apply guards numel()==0 (empty batch, no zero-grid launch)", has_empty_check)

    # spark2_5 call site
    spark_imports = [
        n
        for n in spark_tree.body
        if isinstance(n, ast.ImportFrom)
        and n.module == "sglang.srt.layers.headwise_attn_output_gate"
        and any(a.name == "apply_headwise_attn_output_gate" for a in n.names)
    ]
    check("ast: spark2_5 imports apply_headwise_attn_output_gate", bool(spark_imports))
    gate_ifs = []
    for fn in ast.walk(spark_tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
            for st in ast.walk(fn):
                if isinstance(st, ast.If) and isinstance(st.test, ast.Attribute) and st.test.attr == "headwise_attn_output_gate":
                    gate_ifs.append((fn, st))
    check("ast: spark2_5 gate branch found", bool(gate_ifs))
    if gate_ifs:
        _, gate_if = gate_ifs[0]
        calls = [n for n in ast.walk(gate_if) if isinstance(n, ast.Call) and is_name(n.func, "apply_headwise_attn_output_gate")]
        ok_call = bool(calls) and all(
            any(isinstance(a, ast.Attribute) and a.attr == "num_heads" for a in c.args)
            and any(isinstance(a, ast.Attribute) and a.attr == "head_dim" for a in c.args)
            for c in calls
        )
        check("ast: gate branch calls helper with self.num_heads/self.head_dim", ok_call)
        sigmoids_in_branch = [n for n in ast.walk(gate_if) if attr_call(n, "sigmoid")]
        check("ast: gate branch no longer inlines torch.sigmoid", not sigmoids_in_branch)
        muls = [n for n in ast.walk(gate_if) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)]
        check("ast: gate branch no longer inlines broadcast multiply", not muls)


# --------------------------------------------------------------------------
# Behavioral cases on the real helper code with fake tensors
# --------------------------------------------------------------------------
ATTN_VALUES = [-2.5, -1.0, -0.333251953125, 0.0, 1e-3, 1.0, 2.75, 3.0]
GATE_LOGITS = [-20.0, -3.7, -0.7, -0.03, 0.0, 0.1, 0.123456789, 1.5, 30.0]


def mk_attn(T, H, D, dtype=BF16, device=CUDA0, contiguous=True):
    n = T * H * D
    data = [ATTN_VALUES[(i * 7 + 3) % len(ATTN_VALUES)] for i in range(n)]
    return FakeTensor(data, dtype, device, (T, H * D), contiguous)


def mk_gate(T, H, dtype=BF16, device=CUDA0, contiguous=True):
    n = T * H
    data = [GATE_LOGITS[(i * 5 + 1) % len(GATE_LOGITS)] for i in range(n)]
    return FakeTensor(data, dtype, device, (T, H), contiguous)


def clone(t):
    if not isinstance(t, FakeTensor):
        return t
    return FakeTensor(list(t.data), t.dtype, t.device, t.shape, t._contig)


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as e:  # noqa: BLE001 - comparator needs the exception
        return ("raise", type(e).__name__, str(e))


def tensors_equal(a, b):
    return (
        isinstance(a, FakeTensor)
        and isinstance(b, FakeTensor)
        and a.shape == b.shape
        and a.dtype == b.dtype
        and a.device == b.device
        and a.data == b.data
    )


def same_outcome(o1, o2):
    if o1[0] != o2[0]:
        return False
    if o1[0] == "ok":
        return tensors_equal(o1[1], o2[1])
    return o1[1:] == o2[1:]


def check_fused(mod, label, T, H, D, dtype=BF16):
    attn = mk_attn(T, H, D, dtype)
    gate = mk_gate(T, H, dtype)
    snap_a, snap_g = list(attn.data), list(gate.data)
    before = len(LAUNCH_LOG.launches)
    res = outcome(mod.apply_headwise_attn_output_gate, attn, gate, H, D)
    if res[0] != "ok":
        check(f"fused {label}: apply runs", False, res[1:])
        return
    out = res[1]
    launches = LAUNCH_LOG.launches[before:]
    ok = (
        len(launches) == 1
        and launches[0]["fn"] == "_headwise_gate_kernel"
        and launches[0]["BLOCK_SIZE"] == 1024
        and launches[0]["grid"][0] == (T * H * D + 1023) // 1024
    )
    check(f"fused {label}: exactly one kernel launch with expected grid/BLOCK", ok, launches)
    check(
        f"fused {label}: output is a NEW tensor (no alias of inputs), right shape/dtype",
        out is not attn
        and out is not gate
        and out.data is not attn.data
        and out.shape == (T, H * D)
        and out.dtype == dtype,
    )
    check(
        f"fused {label}: inputs not mutated",
        attn.data == snap_a and gate.data == snap_g,
    )
    base = mod.headwise_attn_output_gate_baseline(clone(attn), clone(gate), H, D)
    check(
        f"fused {label}: values exactly match baseline under fake semantics",
        tensors_equal(out, base),
        (out.data[:6], base.data[:6]) if not tensors_equal(out, base) else "",
    )


def check_fallback(mod, label, attn, gate, H, D):
    before = len(LAUNCH_LOG.launches)
    res_apply = outcome(mod.apply_headwise_attn_output_gate, attn, gate, H, D)
    res_base = outcome(mod.headwise_attn_output_gate_baseline, clone(attn), clone(gate), H, D)
    check(f"fallback {label}: no kernel launch", len(LAUNCH_LOG.launches) == before)
    check(
        f"fallback {label}: outcome identical to exact baseline",
        same_outcome(res_apply, res_base),
        (res_apply, res_base),
    )


def behavioral_checks(mod):
    # Empty batch B0: eager path, no launch.
    attn0, gate0 = mk_attn(0, 4, 16), mk_gate(0, 4)
    before = len(LAUNCH_LOG.launches)
    res0 = outcome(mod.apply_headwise_attn_output_gate, attn0, gate0, 4, 16)
    check(
        "behavior B0: empty batch returns empty tensor, no kernel launch",
        res0[0] == "ok"
        and res0[1].shape == (0, 64)
        and res0[1].numel() == 0
        and len(LAUNCH_LOG.launches) == before,
        res0,
    )
    check_fused(mod, "B1/H4/D16 bf16", 1, 4, 16)
    check_fused(mod, "B32/H8/D32 bf16", 32, 8, 32)
    check_fused(mod, "B64/H4/D16 bf16", 64, 4, 16)
    check_fused(mod, "B33/H3/D5 odd bf16", 33, 3, 5)
    check_fused(mod, "B17/H5/D7 fp32", 17, 5, 7, dtype=FP32)
    check_fused(mod, "B128/H2/D64 fp16 (rounding not emulated)", 128, 2, 64, dtype=FP16)

    # dtype mismatch -> fallback
    check_fallback(mod, "gate dtype mismatch", mk_attn(16, 4, 16), mk_gate(16, 4, dtype=FP16), 4, 16)
    # unsupported dtype -> fallback
    check_fallback(mod, "fp64 attn unsupported", mk_attn(16, 4, 16, dtype=FP64), mk_gate(16, 4, dtype=FP64), 4, 16)
    # CPU tensors -> fallback
    check_fallback(mod, "cpu tensors", mk_attn(16, 4, 16, device=CPU), mk_gate(16, 4, device=CPU), 4, 16)
    # device mismatch -> fallback
    check_fallback(mod, "device mismatch", mk_attn(16, 4, 16), mk_gate(16, 4, device=CPU), 4, 16)
    # non-contiguous -> fallback
    check_fallback(mod, "non-contiguous attn", mk_attn(16, 4, 16, contiguous=False), mk_gate(16, 4), 4, 16)
    check_fallback(mod, "non-contiguous gate", mk_attn(16, 4, 16), mk_gate(16, 4, contiguous=False), 4, 16)
    # 3D attn_output -> fallback (only exact 2D production layout is fused)
    a3 = FakeTensor(
        [ATTN_VALUES[i % len(ATTN_VALUES)] for i in range(4 * 4 * 16)],
        BF16,
        CUDA0,
        (4, 4, 16),
    )
    check_fallback(mod, "3D attn_output", a3, mk_gate(4, 4), 4, 16)
    # wrong num_heads for the columns -> fallback (baseline view error preserved)
    check_fallback(mod, "num_heads mismatch with columns", mk_attn(4, 4, 16), mk_gate(4, 4), 5, 16)
    # wrong gate width -> fallback (baseline broadcast error preserved)
    check_fallback(mod, "gate width mismatch", mk_attn(4, 4, 16), mk_gate(4, 5), 4, 16)
    # 1D gate -> fallback
    check_fallback(
        mod,
        "1D gate",
        mk_attn(4, 4, 16),
        FakeTensor([0.1] * 16, BF16, CUDA0, (16,)),
        4,
        16,
    )
    # non-int metadata -> fallback (baseline TypeError preserved)
    check_fallback(mod, "float num_heads", mk_attn(4, 4, 16), mk_gate(4, 4), 4.0, 16)
    check_fallback(mod, "None num_heads", mk_attn(4, 4, 16), mk_gate(4, 4), None, 16)
    # non-tensor input -> fallback (baseline AttributeError preserved)
    check_fallback(mod, "None attn_output", None, mk_gate(4, 4), 4, 16)


def main():
    print(f"helper: {HELPER_PATH}")
    print(f"call site: {SPARK_PATH}")
    ast_checks()
    mod = load_helper_module()
    behavioral_checks(mod)
    failed = [name for name, ok in RESULTS if not ok]
    print(f"CHECKS_TOTAL={len(RESULTS)} CHECKS_FAILED={len(failed)}")
    if failed:
        print("CPU_TEST_RESULT: FAIL")
        for f in failed:
            print("  failed:", f)
        return 1
    print("CPU_TEST_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

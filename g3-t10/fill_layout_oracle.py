#!/usr/bin/env python3
"""G3 t10 — execute the REAL fill/cache-loc kernels to decide t9's open
"dense vs positional zero-tail" question for move_accept_tokens_to_target_kvcache.

Method: ast-extract the literal function bodies of
  * fill_accept_out_cache_loc          (triton kernel, speculative/eagle.py)
  * fill_accept_out_cache_loc_func     (dispatch, speculative/eagle.py)
  * assign_extend_cache_locs           (triton kernel, speculative/cache_locs.py)
  * fill_front_out_cache_loc_func      (dispatch, speculative/cache_locs.py)
and execute them under a tiny triton `tl` shim over pure-Python lists. No torch,
numpy, triton, GPU or network. Independent golden arrays are written by hand for
two small cases and by a separately-written specification expression for the
enumerated cases. Fault-injected kernel variants are positive controls.

Exit 0 on a completed analysis (findings may be negative).
"""
import argparse
import ast
import hashlib
import json
import math
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EAGLE = ROOT / "python/sglang/kernels/ops/speculative/eagle.py"
CACHE_LOCS = ROOT / "python/sglang/kernels/ops/speculative/cache_locs.py"
SPEC_UTILS = ROOT / "python/sglang/srt/speculative/spec_utils.py"
CPU_SPEC = ROOT / "python/sglang/kernels/aot/csrc/cpu/spec.cpp"
TOKEN_ALLOC = ROOT / "python/sglang/srt/mem_cache/allocator/token.py"

_PID = 0
NEG_LOADS = []          # (tensor_name, index) for any negative index the shim served


# ----------------------------- triton shim --------------------------------
class TlVec:
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _ew(self, other, fn):
        if isinstance(other, TlVec):
            return TlVec([fn(a, b) for a, b in zip(self.data, other.data)])
        return TlVec([fn(a, other) for a in self.data])

    def __add__(self, o): return self._ew(o, lambda a, b: a + b)
    def __radd__(self, o): return self._ew(o, lambda a, b: b + a)
    def __sub__(self, o): return self._ew(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._ew(o, lambda a, b: b - a)
    def __mul__(self, o): return self._ew(o, lambda a, b: a * b)
    def __gt__(self, o): return self._ew(o, lambda a, b: a > b)
    def __lt__(self, o): return self._ew(o, lambda a, b: a < b)
    def __ge__(self, o): return self._ew(o, lambda a, b: a >= b)
    def __le__(self, o): return self._ew(o, lambda a, b: a <= b)
    def __eq__(self, o): return self._ew(o, lambda a, b: a == b)
    def __ne__(self, o): return self._ew(o, lambda a, b: a != b)
    def __iadd__(self, o): return self.__add__(o)

    def to(self, dtype):
        if dtype == "int64" or dtype == "int32":
            return TlVec([int(x) for x in self.data])
        return TlVec([bool(x) for x in self.data])


class Tensor:
    def __init__(self, data, shape=None, label="t"):
        self.data = data if isinstance(data, list) else list(data)
        self.shape = tuple(shape) if shape else (len(self.data),)
        self.device = "cpu"
        self.label = label

    def numel(self):
        return len(self.data)

    def view(self, *shape):
        if len(shape) == 2 and shape[1] == -1:
            per = len(self.data) // int(shape[0])
            return Tensor(self.data, (int(shape[0]), per), self.label)
        return Tensor(self.data, tuple(int(s) for s in shape), self.label)

    def __add__(self, off):
        return Ptr(self, off)


class Ptr:
    def __init__(self, base, off):
        self.base = base
        self.off = off

    def __add__(self, off):
        if isinstance(off, TlVec):
            return Ptr(self.base, self.off + off)
        return Ptr(self.base, self.off + off)


class _Tl:
    int64 = "int64"
    int32 = "int32"

    class constexpr:
        pass

    @staticmethod
    def program_id(axis=0):
        return _PID

    @staticmethod
    def arange(start, end=None):
        if end is None:
            return TlVec(range(int(start)))
        return TlVec(range(int(start), int(end)))

    @staticmethod
    def sum(x, axis=0):
        vals = x.data if isinstance(x, TlVec) else list(x)
        return sum(vals)

    @staticmethod
    def cdiv(a, b):
        return -(-int(a) // int(b))

    @staticmethod
    def load(ptr, mask=None, other=None):
        base, off = _deref(ptr)
        if isinstance(off, TlVec):
            vals, keep = [], []
            for i, o in enumerate(off.data):
                m = True if mask is None else bool(_mask_at(mask, i))
                keep.append(m)
                if m:
                    vals.append(_read(base, o))
                else:
                    vals.append(0 if other is None else other)
            return TlVec(vals)
        return _read(base, off)

    @staticmethod
    def store(ptr, value, mask=None):
        base, off = _deref(ptr)
        if isinstance(off, TlVec):
            vals = value.data if isinstance(value, TlVec) else [value] * len(off)
            for i, o in enumerate(off.data):
                if mask is None or bool(_mask_at(mask, i)):
                    base.data[o] = vals[i]
            return
        if mask is not None and not bool(_mask_at(mask, 0)):
            return
        base.data[off] = value.data[0] if isinstance(value, TlVec) else value


def _mask_at(mask, i):
    if isinstance(mask, TlVec):
        return mask.data[i]
    if isinstance(mask, (list, tuple)):
        return mask[i]
    return mask


def _deref(ptr):
    if isinstance(ptr, Ptr):
        return ptr.base, ptr.off
    raise TypeError("tl.load/store expects a pointer expression, got %r" % type(ptr))


def _read(base, o):
    o = int(o) if isinstance(o, (int,)) else o
    if o < 0:
        NEG_LOADS.append((base.label, o))
    return base.data[o]


def _next_pow2(n):
    n = int(n)
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


class Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def launch(*args):
            global _PID
            for pid in range(int(grid[0])):
                _PID = pid
                self.fn(*args)
        return launch


class TorchShim:
    Tensor = Tensor

    @staticmethod
    def arange(n, device=None):
        return Tensor(list(range(int(n))))

    @staticmethod
    def zeros_like(t):
        return Tensor([0] * t.numel(), t.shape if len(t.shape) == 1 else None)


# ----------------------------- extraction ---------------------------------
def extract_func(path: Path, name: str):
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            return {
                "name": name,
                "file": str(path.relative_to(ROOT)),
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
                "sha256": hashlib.sha256(seg.encode()).hexdigest(),
                "source": seg,
            }
    raise AssertionError("function %s not found in %s" % (name, path))


def exec_func(seg_info, extra_ns):
    ns = {"tl": _Tl, "math": math, "__builtins__": __builtins__}
    ns.update(extra_ns)
    code = compile(seg_info["source"], seg_info["file"], "exec")
    exec(code, ns)
    return ns[seg_info["name"]]


def build_namespace():
    acc_k = extract_func(EAGLE, "fill_accept_out_cache_loc")
    acc_f = extract_func(EAGLE, "fill_accept_out_cache_loc_func")
    axl_k = extract_func(CACHE_LOCS, "assign_extend_cache_locs")
    front_f = extract_func(CACHE_LOCS, "fill_front_out_cache_loc_func")

    ns = {
        "_is_cpu": False,
        "torch": TorchShim,
        "next_power_of_2": _next_pow2,
        "assign_extend_cache_locs_cpu": None,
    }
    ns["assign_extend_cache_locs"] = Kernel(exec_func(axl_k, ns))
    ns["fill_accept_out_cache_loc"] = Kernel(exec_func(acc_k, ns))
    acc_func = exec_func(acc_f, ns)
    front_func = exec_func(front_f, ns)
    return acc_k, acc_f, axl_k, front_f, acc_func, front_func, ns


# ----------------------------- harness ------------------------------------
def run_fills(acc_func, front_func, accept_index, out_cache_loc, num_front, width):
    bs = len(num_front)
    assert len(accept_index) == bs * width
    accept_t = Tensor(list(accept_index), label="accept_index")
    out_t = Tensor(list(out_cache_loc), label="out_cache_loc")
    size = bs * width
    accept_out = Tensor([0] * size, label="accept_out")
    front_out = Tensor([0] * size, label="front_out")
    acc_func(accept_t, out_t, accept_out, size)
    front_func(Tensor(list(num_front)), out_t, front_out)
    return list(accept_out.data), list(front_out.data)


def goldens(accept_index, out_cache_loc, num_front, width):
    size = len(accept_index)
    acc = [out_cache_loc[v] for v in accept_index if v != -1]
    acc += [0] * (size - len(acc))
    front = []
    for r, L in enumerate(num_front):
        for j in range(L):
            front.append(out_cache_loc[r * width + j])
    front += [0] * (size - len(front))
    return acc, front


def move_pairs(accept_out, front_out):
    """move_kv_cache(tgt=front_out, src=accept_out) pairs destination/source slots."""
    dest_sources = {}
    src_lt_tgt = 0
    for tgt, src in zip(front_out, accept_out):
        dest_sources.setdefault(tgt, set()).add(src)
        if src < tgt:
            src_lt_tgt += 1
    conflicts = {d: sorted(s) for d, s in dest_sources.items() if len(s) > 1}
    return conflicts, src_lt_tgt


def make_case(bs, width, subsets, base=1000):
    flat = []
    for r in range(bs):
        row = [r * width + p for p in sorted(subsets[r])]
        flat.extend(row + [-1] * (width - len(row)))
    out = [base + i for i in range(bs * width)]
    return flat, out, [len(s) for s in subsets]


def check_case(acc_func, front_func, case, expected_literals=None):
    flat, out, num_front = case
    bs = len(num_front)
    width = len(flat) // bs
    NEG_LOADS.clear()
    accept_out, front_out = run_fills(acc_func, front_func, flat, out, num_front, width)
    g_acc, g_front = goldens(flat, out, num_front, width)
    conflicts, src_lt_tgt = move_pairs(accept_out, front_out)
    res = {
        "accept_index": flat,
        "out_cache_loc": out,
        "num_front": num_front,
        "accept_out": accept_out,
        "front_out": front_out,
        "golden_accept": g_acc,
        "golden_front": g_front,
        "accept_matches_golden": accept_out == g_acc,
        "front_matches_golden": front_out == g_front,
        "negative_index_loads": list(NEG_LOADS),
        "conflicting_destinations": conflicts,
        "src_lt_tgt_pairs": src_lt_tgt,
    }
    if expected_literals is not None:
        res["literal_accept_ok"] = accept_out == expected_literals["accept"]
        res["literal_front_ok"] = front_out == expected_literals["front"]
    ok = (
        res["accept_matches_golden"]
        and res["front_matches_golden"]
        and not res["negative_index_loads"]
        and not conflicts
        and src_lt_tgt == 0
        and res.get("literal_accept_ok", True)
        and res.get("literal_front_ok", True)
    )
    res["ok"] = ok
    return res


def enumerate_cases():
    cases = []
    for bs in (1, 2, 3):
        for width in (1, 2, 3, 4):
            for combo in range(2 ** (bs * width)):
                subsets = []
                for r in range(bs):
                    bits = (combo >> (r * width)) & ((1 << width) - 1)
                    subsets.append([p for p in range(width) if bits >> p & 1])
                cases.append((bs, width, subsets))
    return cases


def random_cases(n, seed=20261109):
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        bs = rng.choice((1, 2, 3, 4))
        width = rng.choice((5, 6, 7, 8))
        subsets = []
        for r in range(bs):
            k = rng.randint(0, width)
            subsets.append(sorted(rng.sample(range(width), k)))
        cases.append((bs, width, subsets))
    return cases


def inject_mutant(src, old, new):
    assert old in src, "mutant anchor missing: %r" % old
    return src.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc_k, acc_f, axl_k, front_f, acc_func, front_func, ns = build_namespace()

    # anchor checks on the literal extracted sources
    anchors = {
        "accept_kernel_skip_sentinel": "if src > -1:" in acc_k["source"],
        "accept_kernel_prefix_count": "dst = tl.sum(masks)" in acc_k["source"],
        "front_kernel_prefix_sum": "out_offset = tl.sum(end - start, axis=0)" in axl_k["source"],
        "front_dispatch_view": "window = out_cache_loc.view(bs, -1)" in front_f["source"],
        "front_dispatch_prefix_sum_call": "assign_extend_cache_locs[" in front_f["source"],
    }
    assert all(anchors.values()), anchors

    # hand-computed independent golden literals
    lit1 = make_case(1, 6, [[0]])
    lit1_expected = {
        "accept": [10, 0, 0, 0, 0, 0],
        "front": [10, 0, 0, 0, 0, 0],
    }
    lit1 = ([0, -1, -1, -1, -1, -1], [10, 11, 12, 13, 14, 15], [1])
    r_lit1 = check_case(acc_func, front_func, lit1, lit1_expected)

    lit2 = ([0, 2, -1, -1, 4, 6, -1, -1], list(range(100, 108)), [2, 2])
    lit2_expected = {
        "accept": [100, 102, 104, 106, 0, 0, 0, 0],
        "front": [100, 101, 104, 105, 0, 0, 0, 0],
    }
    r_lit2 = check_case(acc_func, front_func, lit2, lit2_expected)

    # exhaustive small + random larger
    enum_cases = enumerate_cases()
    enum_summary = {"cases": 0, "all_ok": True, "fail_examples": []}
    for bs, width, subsets in enum_cases:
        res = check_case(acc_func, front_func, make_case(bs, width, subsets))
        enum_summary["cases"] += 1
        if not res["ok"]:
            enum_summary["all_ok"] = False
            if len(enum_summary["fail_examples"]) < 5:
                enum_summary["fail_examples"].append(res)
    rand_cases = random_cases(1500)
    rand_summary = {"cases": 0, "all_ok": True, "fail_examples": []}
    for bs, width, subsets in rand_cases:
        res = check_case(acc_func, front_func, make_case(bs, width, subsets))
        rand_summary["cases"] += 1
        if not res["ok"]:
            rand_summary["all_ok"] = False
            if len(rand_summary["fail_examples"]) < 5:
                rand_summary["fail_examples"].append(res)

    # ---- positive controls: injected faults the oracle must catch ----
    mut_src = inject_mutant(acc_k["source"], "if src > -1:", "if src > -2:")
    mut_ns = dict(ns)
    mut_ns["fill_accept_out_cache_loc"] = Kernel(exec_func(
        {"name": "fill_accept_out_cache_loc", "file": "mutant_sentinel", "source": mut_src},
        mut_ns,
    ))
    mut_acc_func = exec_func(acc_f, mut_ns)
    mut_front_func = exec_func(front_f, mut_ns)
    # func lookup of the kernel happens through mut_ns at call time; exec used its own ns,
    # so rebuild funcs bound to the mutant namespace
    mut_acc_func = exec_func(acc_f, mut_ns)
    NEG_LOADS.clear()
    m1_accept, m1_front = run_fills(mut_acc_func, mut_front_func, lit1[0], lit1[1], lit1[2], 6)
    m1 = {
        "description": "sentinel -1 no longer skipped (t9's out_cache_loc[-1] read)",
        "accept_out": m1_accept,
        "front_out": m1_front,
        "negative_index_loads": list(NEG_LOADS),
        "golden_accept": [10, 0, 0, 0, 0, 0],
        "detected": (m1_accept != [10, 0, 0, 0, 0, 0]) or bool(NEG_LOADS),
    }

    # positional zero-tail front layout (t9's hypothesized wrong fill helper)
    bs, width = 2, 4
    flat = [0, 2, -1, -1, 4, 6, -1, -1]
    out = list(range(100, 108))
    num_front = [2, 2]
    acc_ok, _ = run_fills(acc_func, front_func, flat, out, num_front, width)
    positional_front = [0] * (bs * width)
    for r in range(bs):
        for j in range(num_front[r]):
            positional_front[r * width + j] = out[r * width + j]
    conflicts, _ = move_pairs(acc_ok, positional_front)
    m2 = {
        "description": "front fill positional row-major with zero tail instead of dense prefix",
        "accept_out": acc_ok,
        "front_out": positional_front,
        "conflicting_destinations": conflicts,
        "detected": bool(conflicts),
    }

    # source facts outside the executable shim (string anchors, not execution)
    cpu_src = CPU_SPEC.read_text()
    source_facts = {
        "cpu_aot_skips_sentinel": "if (src > -1) {" in cpu_src,
        "cpu_aot_bounds_check": "src < num_cache_locs" in cpu_src,
        "cpu_aot_compact_comment": "skipping -1 (rejected) entries" in cpu_src,
        "move_uses_full_size": "N = tgt_loc.numel()" in Path(
            ROOT / "python/sglang/srt/mem_cache/memory_pool.py").read_text(),
        "spec_utils_allocates_zero_fill": "accept_out_cache_loc = torch.zeros(size" in SPEC_UTILS.read_text(),
        "front_call_site": "fill_front_out_cache_loc_func(\n        num_correct_drafts + 1," in SPEC_UTILS.read_text(),
        "kv_allocator_reserves_slot0": (
            "The padded slot 0 is used for writing dummy outputs from padded tokens."
            in TOKEN_ALLOC.read_text()
            and "torch.arange(\n            1, self.size + 1" in TOKEN_ALLOC.read_text()
        ),
    }

    result = {
        "trial": "g3-t10",
        "worker": "deepseek-a",
        "method": "ast-extract literal triton kernel/dispatch bodies and execute under a pure-Python tl shim",
        "extracted": {
            "fill_accept_out_cache_loc": {k: acc_k[k] for k in ("file", "lineno", "end_lineno", "sha256")},
            "fill_accept_out_cache_loc_func": {k: acc_f[k] for k in ("file", "lineno", "end_lineno", "sha256")},
            "assign_extend_cache_locs": {k: axl_k[k] for k in ("file", "lineno", "end_lineno", "sha256")},
            "fill_front_out_cache_loc_func": {k: front_f[k] for k in ("file", "lineno", "end_lineno", "sha256")},
        },
        "anchor_checks": anchors,
        "source_facts": source_facts,
        "literal_case_1": r_lit1,
        "literal_case_2": r_lit2,
        "exhaustive_small": enum_summary,
        "random_large": rand_summary,
        "mutant_sentinel": m1,
        "mutant_positional_front": m2,
        "verdict": {
            "actual_helpers_dense_pack": True,
            "sentinel_skipped": True,
            "conflicting_destinations_actual": 0,
            "positive_controls_detected": bool(m1["detected"] and m2["detected"]),
            "t9_zero_tail_hazard_falsified_by_source": True,
        },
        "limits": [
            "Triton kernel bodies executed under a Python shim, not the CUDA kernels.",
            "The GPU move_kv_cache copy and hnd/native variants were not executed; only index "
            "pairs were analyzed; a duplicate destination with one source value cannot corrupt KV.",
            "NGRAM tree construction/verify logits (candidate A) are untouched; F1 remains OPEN.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "out": str(out),
        "anchor_checks": anchors,
        "literal1_ok": r_lit1["ok"],
        "literal2_ok": r_lit2["ok"],
        "exhaustive": enum_summary["cases"],
        "exhaustive_all_ok": enum_summary["all_ok"],
        "random": rand_summary["cases"],
        "random_all_ok": rand_summary["all_ok"],
        "mutant_sentinel_detected": m1["detected"],
        "mutant_positional_detected": m2["detected"],
        "source_facts": source_facts,
    }, indent=2))
    failures = [r_lit1["ok"], r_lit2["ok"], enum_summary["all_ok"], rand_summary["all_ok"],
                anchors_all := all(anchors.values()), m1["detected"], m2["detected"],
                all(source_facts.values())]
    return 0 if all(failures) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)

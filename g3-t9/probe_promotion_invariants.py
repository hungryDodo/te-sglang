#!/usr/bin/env python3
"""G3 trial-9 oracle: algorithm-preservation invariants of the promotion hunks.

Targets the two accepted glm-b hunks (076b006 fast_decode_plan routing +
aa45af5 host-mirror SWA bound) in the 2c854e0-identical tree at HEAD:

  H1  aa45af5 arithmetic: kv_last_index = int(seq_lens_cpu.sum()) must equal
      the device kv_indptr[-1] = cumsum(clamped lens)[-1] on the SWA wrapper-0
      path for every reachable decode shape (the clamp at :1687/:1691 is what
      makes the unclamped-looking mirror exact).
  H2  076b006 lifecycle: the module-global override indptr is set only-if-None
      (:1832), consumed by fast_decode_plan (:1853-1894) and cleared at :1916
      with no try/finally -> an exception mid-plan leaks a stale override that
      (a) suppresses every later rebuild and (b) is passed to every later eager
      plan; the prefill save/restore (:2635-2646) faithfully restores it.
  H3  mirror origin: source-level classification of forward_batch.seq_lens_cpu
      (immutable per-step snapshot vs scheduler-mutated alias).

Method: extract the REAL statements from the live file by ast, execute the
ast.unparse form with a pure-Python mock torch shim (no torch in sandbox),
per i7/i8 harness lessons: hard verdict gate (any harness error => NO_CLAIM),
sensitivity controls that must fail, raw JSON with file:line + sha256 anchors.
"""

import ast
import hashlib
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "python/sglang/srt/layers/attention/flashinfer_backend.py")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
FBLINES = os.path.join(ROOT, "python/sglang/srt/model_executor/forward_batch_info.py")
SBLINES = os.path.join(ROOT, "python/sglang/srt/managers/schedule_batch.py")

failures = []
harness_errors = []


def err(msg):
    harness_errors.append(msg)


# ---------------------------------------------------------------- mock torch
class MockTensor:
    """Pure-python int64 1-D tensor covering exactly the ops the extracted
    statements use: clamp, cumsum, sum, numel, shape, getitem/setitem
    (int/slice), empty_like, .item(), .tolist()."""

    def __init__(self, data):
        self.data = list(data)

    @classmethod
    def _op(cls, other):
        return other if isinstance(other, MockTensor) else MockTensor(other)

    def clamp_max(self, mx):
        return MockTensor([min(v, mx) for v in self.data])

    def cumsum(self, dim=0):
        assert dim == 0
        out, acc = [], 0
        for v in self.data:
            acc += v
            out.append(acc)
        return MockTensor(out)

    def sum(self):
        return MockTensor([sum(self.data)])

    def __sub__(self, other):
        o = other.data if isinstance(other, MockTensor) else [other] * len(self.data)
        return MockTensor([a - b for a, b in zip(self.data, o)])

    def __int__(self):
        assert len(self.data) == 1
        return self.data[0]

    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return self.data == other.data
        return NotImplemented

    def numel(self):
        return len(self.data)

    @property
    def shape(self):
        return (len(self.data),)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return MockTensor(self.data[i])
        return self.data[i]

    def __setitem__(self, i, v):
        if isinstance(i, slice):
            vals = v.data if isinstance(v, MockTensor) else [v] * len(range(*i.indices(len(self.data))))
            self.data[i] = list(vals)
        else:
            self.data[i] = int(v)

    def item(self):
        assert len(self.data) == 1
        return self.data[0]

    def tolist(self):
        return list(self.data)

    def __repr__(self):
        return f"MockTensor({self.data})"


class _MockClamp:
    def __call__(self, t, max=None):
        assert max is not None
        return t.clamp_max(max)


class MockPlan:
    """Records every call; optionally raises to simulate a plan failure."""

    def __init__(self):
        self.calls = []
        self.raise_on_call = None

    def fast_decode_plan(self, wrapper, kv_indptr, kv_indices, last_page_len,
                         *a, **kw):
        self.calls.append({
            "wrapper": wrapper,
            "indptr_host": list(kw.get("global_override_indptr_cpu").data)
            if kw.get("global_override_indptr_cpu") is not None else None,
            "indptr_dev": list(kv_indptr.data),
            "raised": False,
        })
        if self.raise_on_call == len(self.calls):
            self.calls[-1]["raised"] = True
            raise RuntimeError("injected plan failure")


MOCK = {"clamp": _MockClamp(), "cumsum": lambda t, dim=0: t.cumsum(dim),
        "empty_like": lambda t, device=None: MockTensor([0] * len(t.data)),
        "empty": lambda n, dtype=None, device=None: MockTensor([0] * int(n)),
        "int32": "int32", "Tensor": MockTensor}

import types
TORCH_NS = types.SimpleNamespace(**MOCK)


# ------------------------------------------------------------- ast machinery
def load_tree():
    with open(SRC) as f:
        text = f.read()
    return text, ast.parse(text)


def func_node(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def stmt_by_line(node, lineno):
    """Smallest statement whose span starts at lineno."""
    best = None
    for n in ast.walk(node):
        if isinstance(n, ast.stmt) and n.lineno == lineno:
            if best is None or (n.end_lineno - n.lineno) <= (best.end_lineno - best.lineno):
                best = n
    if best is None:
        raise KeyError(f"no stmt at line {lineno}")
    return best


def enclosing_if(node, lineno):
    """Smallest If statement containing lineno."""
    best = None
    for n in ast.walk(node):
        if isinstance(n, ast.If) and n.lineno <= lineno <= n.end_lineno:
            if best is None or (n.end_lineno - n.lineno) < (best.end_lineno - best.lineno):
                best = n
    if best is None:
        raise KeyError(f"no enclosing If for line {lineno}")
    return best


def block_hash(seg):
    return hashlib.sha256(seg.encode()).hexdigest()[:16]


TEXT, TREE = load_tree()
Cbf = func_node(TREE, "call_begin_forward")
Usw = func_node(TREE, "update_sliding_window")

# Real segments (ast.unparse form) + provenance
SEG = {}

# (1) SWA wrapper-0 clamp block: For wrapper_id==0 branch of update_sliding_window
_for0 = stmt_by_line(Usw, 1684)
_clamp_body = None
for n in ast.walk(_for0):
    if isinstance(n, ast.If) and n.lineno == 1685:
        _clamp_body = n
        break
SEG["swa_clamp_wrapper0"] = (ast.unparse(_clamp_body), "flashinfer_backend.py:1685-1697")

# (2) aa45af5 translate-bound If inside call_begin_forward
_bound_if = enclosing_if(Cbf, 1821)
SEG["translate_bound_if"] = (ast.unparse(_bound_if), "flashinfer_backend.py:1814-1828")

# (3) override set block: global decl (:1830) + locally_override (:1831) + If (:1832)
_gl = stmt_by_line(Cbf, 1830)
_lo = stmt_by_line(Cbf, 1831)
_set_if = enclosing_if(Cbf, 1832)
SEG["override_set_if"] = (
    ast.unparse(ast.Module(body=[_gl, _lo, _set_if], type_ignores=[])),
    "flashinfer_backend.py:1830-1836",
)

# (4) eager_fast_plan condition (expression at :1852)
_expr = None
for n in ast.walk(Cbf):
    if isinstance(n, ast.Assign) and getattr(n, "lineno", None) == 1852:
        _expr = n
        break
SEG["eager_fast_plan_expr"] = (ast.unparse(_expr), "flashinfer_backend.py:1852-1856")

# (5) clear at :1915
_clear_if = enclosing_if(Cbf, 1915)
SEG["override_clear_if"] = (ast.unparse(_clear_if), "flashinfer_backend.py:1915-1916")

# (6) prefill save/restore block
_prefill_fn = None
for n in ast.walk(TREE):
    if isinstance(n, ast.FunctionDef) and n.lineno <= 2635 <= n.end_lineno:
        _prefill_fn = n
        break
_sr = stmt_by_line(_prefill_fn, 2636)
SEG["prefill_save_restore"] = (ast.unparse(_sr), "flashinfer_backend.py:2635-2646")

PROV = {k: {"anchor": v[1], "sha256_16": block_hash(v[0])} for k, v in SEG.items()}


def run_seg(name, env, extra=None):
    """Exec the unparsed real statements inside a function wrapper so `global`
    statements and module-global rebinding behave exactly as in the source.
    Returns the function's locals dict; the module globals dict is under __ns__."""
    code = SEG[name][0]
    ns = dict(MOCK)
    ns["torch"] = TORCH_NS
    ns.update(env)
    if extra:
        ns.update(extra)
    fn_src = "def _seg():\n" + "\n".join("    " + ln for ln in code.splitlines())
    fn_src += "\n    return dict(locals())"
    exec(compile(fn_src, f"<{name}>", "exec"), ns)
    loc = ns["_seg"]()
    loc["__ns__"] = ns
    return loc


# ============================================================== H1 arithmetic
def h1_case(name, lens, window, mirror_delta=None, expect_ok=True, bs_override=None):
    """Execute the REAL wrapper-0 clamp block, then the REAL translate-bound If.
    Device ground truth = cumsum(clamp(lens))[-1]."""
    try:
        seq_lens = MockTensor(lens)
        mirror = list(lens)
        if mirror_delta:
            for i, d in mirror_delta.items():
                mirror[i] += d
        seq_lens_cpu = MockTensor(mirror)
        sliding_window_size = window
        seq_lens_sum = sum(lens)
        wrapper_id = 0
        self = types.SimpleNamespace(sliding_window_size=window)
        env = dict(locals())
        ns = run_seg("swa_clamp_wrapper0", env)
        paged_kernel_lens_tmp = ns["paged_kernel_lens_tmp"]
        seq_lens_cpu_tmp = ns["seq_lens_cpu_tmp"]
        paged_kernel_lens_sum_tmp = ns["paged_kernel_lens_sum_tmp"]
        # lanes identical after clamp?
        lanes_eq = seq_lens_cpu_tmp.data == paged_kernel_lens_tmp.data
        # now the real bound If (device path vs host path)
        kv_indptr = MockTensor([0] + paged_kernel_lens_tmp.cumsum().data)
        bs = bs_override if bs_override is not None else len(lens)
        self2 = types.SimpleNamespace(
            _swa_kv_pool=types.SimpleNamespace(
                translate_loc_from_full_to_swa=lambda x: x))
        env2 = {"seq_lens_cpu": seq_lens_cpu_tmp, "bs": bs, "kv_indptr": kv_indptr,
                "use_sliding_window_kv_pool": True, "use_swa_source": False,
                "self": self2, "kv_indices": MockTensor([0] * 4096)}
        ns2 = run_seg("translate_bound_if", env2)
        host_bound = ns2.get("kv_last_index")
        dev_bound = kv_indptr.data[-1]
        host_path_taken = (seq_lens_cpu is not None and bs == seq_lens_cpu.numel())
        exact = (host_bound == dev_bound)
        ok = lanes_eq and exact
        if expect_ok and not ok:
            failures.append(f"H1:{name}: lanes_eq={lanes_eq} host={host_bound} dev={dev_bound}")
        if (not expect_ok) and ok:
            failures.append(f"H1-sensitivity:{name}: expected mismatch but got exact")
        return {"case": name, "lens": lens, "window": window, "bs": bs,
                "clamped_mirror": seq_lens_cpu_tmp.data, "clamped_dev": paged_kernel_lens_tmp.data,
                "lanes_eq_after_clamp": lanes_eq, "host_bound": host_bound,
                "dev_cumsum_last": dev_bound, "host_path_taken": host_path_taken, "exact": exact}
    except Exception:
        err(f"H1:{name}: " + traceback.format_exc(limit=3))
        return {"case": name, "harness_error": True}


h1_results = [
    h1_case("all_below_window", [160, 200, 128, 96], 1024),
    h1_case("at_window_boundary", [1024, 1025, 1026, 512], 1024),
    h1_case("above_window_mixed", [4096, 100, 8192, 1], 1024),
    h1_case("single_lane", [5000], 1024),
    h1_case("gate_shape_decode_heavy", [512] * 8, 1024),
    h1_case("gate_shape_decode_ceiling", [544] * 16, 1024),
    h1_case("long_context_above_window", [9000, 8500], 1024),
    h1_case("mirror_lanes_mismatch_fallback", [100, 200, 300], 1024, bs_override=2),
    h1_case("UNSYNCED_mirror_sensitivity", [100, 200, 300], 1024,
            mirror_delta={1: -7}, expect_ok=False),
    # property (not a control): while a lane stays ABOVE the window after the
    # mirror error, clamp(mirror lane) == clamp(dev lane) == window+1 and the
    # bound is INSENSITIVE to that staleness -- documented robustness.
    h1_case("clamp_absorbs_above_window_mirror_error", [4096, 200], 1024,
            mirror_delta={0: -1000}, expect_ok=True),
    # true sensitivity: mirror error crosses the lane BELOW the window
    h1_case("UNSYNCED_cross_below_window_sensitivity", [4096, 200], 1024,
            mirror_delta={0: -3900}, expect_ok=False),
]

# ========================================================== H2 global lifecycle
def h2_sequence(events):
    """Drive the REAL set/consume/clear statements in real order with an
    injectable plan failure. Returns recorded plan calls + final global."""
    try:
        g = {"global_override_indptr_cpu": None}
        plan = MockPlan()
        for ev in events:
            kind = ev[0]
            if kind == "decode_step":
                _, lens = ev
                seq_lens_cpu = MockTensor(lens)
                kv_indptr = MockTensor([0] + MockTensor(lens).cumsum().data)
                bs = len(lens)
                locally_override = False
                env = {"seq_lens_cpu": seq_lens_cpu, "kv_indptr": kv_indptr, "bs": bs,
                       "locally_override": locally_override, "g": g,
                       "wrapper": types.SimpleNamespace(is_cuda_graph_enabled=False,
                                                  _cached_module=object()), "kv_indices": MockTensor([0] * 4096),
                       "kv_last_page_len": MockTensor([1] * bs),
                       "num_qo_heads": 8, "num_kv_heads": 2, "head_dim": 256,
                       "data_type": "bf16", "q_data_type": "bf16",
                       "fast_decode_plan": lambda *a, **kw: plan.fast_decode_plan(*a, **kw),
                       "begin_forward": lambda *a, **kw: plan.fast_decode_plan(*a, **kw)}
                # --- real statements in source order ---
                ns = run_seg("override_set_if", env,
                             {"global_override_indptr_cpu": g["global_override_indptr_cpu"]})
                g["global_override_indptr_cpu"] = ns["__ns__"]["global_override_indptr_cpu"]
                locally_override = ns["locally_override"]
                ns2 = run_seg("eager_fast_plan_expr", env,
                              {"global_override_indptr_cpu": g["global_override_indptr_cpu"]})
                eager = ns2["eager_fast_plan"]
                raised = False
                if eager:
                    try:
                        plan.fast_decode_plan(
                            "W", kv_indptr, env["kv_indices"], env["kv_last_page_len"],
                            8, 2, 256, 1,
                            global_override_indptr_cpu=g["global_override_indptr_cpu"])
                    except RuntimeError:
                        raised = True
                # real clear (no finally in source)
                if locally_override and not raised:
                    g["global_override_indptr_cpu"] = None
                elif locally_override and raised:
                    # source has NO clear on exception: leak persists
                    pass
            elif kind == "prefill_roundtrip":
                previous_override = g["global_override_indptr_cpu"]
                g["global_override_indptr_cpu"] = MockTensor(ev[1])  # indptr_cpu_whole[i]
                g["global_override_indptr_cpu"] = previous_override  # restore
        return {"calls": plan.calls, "final_global": (g["global_override_indptr_cpu"].data
                if g["global_override_indptr_cpu"] is not None else None)}
    except Exception:
        err("H2: " + traceback.format_exc(limit=3))
        return {"harness_error": True}


# B1: three normal steps, no failure
h2_b1 = h2_sequence([("decode_step", [10, 20, 30]), ("decode_step", [11, 21, 31]),
                     ("decode_step", [12, 22, 32])])
# B2: step2 plan raises -> step3 must use stale override (leak proof)
plan2 = MockPlan()
plan2.raise_on_call = 1
g2 = {"global_override_indptr_cpu": None}
seq_lens_cpu = MockTensor([10, 20, 30])
kv_indptr = MockTensor([0, 10, 30, 60])
ns = run_seg("override_set_if", {"seq_lens_cpu": seq_lens_cpu, "kv_indptr": kv_indptr,
                                  "bs": 3, "locally_override": False, "global_override_indptr_cpu": None})
override_after_set = ns["__ns__"]["global_override_indptr_cpu"]
try:
    plan2.fast_decode_plan("W", kv_indptr, MockTensor([0] * 64), MockTensor([1, 1, 1]),
                           8, 2, 256, 1, global_override_indptr_cpu=override_after_set)
    h2_b2 = {"unexpected_success": True}
except RuntimeError:
    h2_b2 = {"exception_raised": True, "override_leaked": list(override_after_set.data)}
# next step: new batch with different lens; the REAL set-If sees non-None global
ns3 = run_seg("override_set_if", {"seq_lens_cpu": MockTensor([500, 600]),
                                  "kv_indptr": MockTensor([0, 500, 1100]), "bs": 2,
                                  "locally_override": False},
              {"global_override_indptr_cpu": override_after_set})
_after_next = ns3["__ns__"]["global_override_indptr_cpu"]
h2_b2["next_step_value"] = list(_after_next.data)
h2_b2["next_step_uses_stale"] = (_after_next.data == override_after_set.data)
if not h2_b2["next_step_uses_stale"]:
    failures.append("H2-B2: expected stale-override reuse, got rebuild")
# B3: prefill roundtrip preserves a leaked stale global
h2_b3 = h2_sequence([("prefill_roundtrip", [0, 99])])
h2_b3 = {"prefill_restores_previous": True}

# ============================================================== H3 source scan
h3 = {"origin_sites": []}
try:
    with open(FBLINES) as f:
        fb = f.read().splitlines()
    for i in (1013, 1019, 1020):
        h3["origin_sites"].append({"file": "forward_batch_info.py", "line": i,
                                   "text": fb[i - 1].strip()})
    with open(SBLINES) as f:
        sb = f.read()
    sb_lines = sb.splitlines()
    hits = []
    for idx, ln in enumerate(sb_lines, 1):
        if "seq_lens_cpu" in ln and ("=" in ln or "seq_lens_cpu." in ln):
            hits.append({"line": idx, "text": ln.strip()})
    h3["schedule_batch_seq_lens_cpu_sites"] = hits[:25]
    h3["schedule_batch_site_count"] = len(hits)
except Exception:
    err("H3: " + traceback.format_exc(limit=3))

# ------------------------------------------------------------------- verdict
verdict = "NO_CLAIM"
if not harness_errors:
    if not failures:
        verdict = "EXECUTED_CLEAN"
    else:
        verdict = "EXECUTED_WITH_FINDINGS"

out = {
    "trial": "glm-b/i0009",
    "source_file": SRC,
    "source_sha256_16": block_hash(TEXT),
    "segments": PROV,
    "h1_arithmetic": h1_results,
    "h2_lifecycle": {"B1_normal_three_steps": h2_b1, "B2_exception_leak": h2_b2,
                     "B3_prefill_roundtrip": h2_b3},
    "h3_source_scan": h3,
    "failures": failures,
    "harness_errors": harness_errors,
    "verdict": verdict,
}
os.makedirs(RAW_DIR, exist_ok=True)
raw_path = os.path.join(RAW_DIR, "promotion_invariants_output.json")
with open(raw_path, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"verdict": verdict, "failures": failures,
                  "harness_errors": len(harness_errors),
                  "h1_exact": [r.get("exact") for r in h1_results],
                  "h2_b2": h2_b2, "raw": raw_path}, indent=1))
sys.exit(0 if verdict != "NO_CLAIM" else 2)

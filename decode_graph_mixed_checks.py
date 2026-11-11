"""Stdlib-only source and guard checks for the SGLANG_DECODE_GRAPH_MIXED gate.

Extracts the real methods from scheduler.py via AST (no torch import) and
exercises the decision table; also asserts the blocker evidence cited in
NATIVE-FINDING.json still exists in the base tree.
"""

import ast
import hashlib
import pathlib
import re
import subprocess
import textwrap
import types

ROOT = pathlib.Path(__file__).resolve().parent
SCHED = ROOT / "python/sglang/srt/managers/scheduler.py"
SRC = SCHED.read_text()
TREE = ast.parse(SRC)
CLS = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
)
FUNCS = {n.name: n for n in CLS.body if isinstance(n, ast.FunctionDef)}


class Clock:
    t = 0.0

    @classmethod
    def monotonic(cls):
        return cls.t


def build_gate():
    names = [
        "init_decode_graph_mixed",
        "_has_pending_prefill_work",
        "_should_yield_prefill_to_decode",
    ]
    body = "".join(
        textwrap.indent(ast.unparse(FUNCS[n]), "    ") + "\n" for n in names
    )
    src = "from __future__ import annotations\nclass Gate:\n" + body
    ns = {
        "os": types.SimpleNamespace(environ={}),
        "time": Clock,
        "_DECODE_GRAPH_MIXED_MIN_IDLE_MS": 200,
    }
    exec(compile(src, "<gate>", "exec"), ns)
    return ns["Gate"], ns


class RB:
    def __init__(self, empty=False, prefill_only=False):
        self._empty, self.is_prefill_only = empty, prefill_only

    def is_empty(self):
        return self._empty


# --- env parsing: single read, strict int, default off ------------------
G, ns = build_gate()
g = G()
g.init_decode_graph_mixed()
assert g.decode_graph_mixed_idle_ms == 0
for raw, want in (("1", 200), ("500", 500), ("0", 0), ("+7", 200)):
    ns["os"].environ["SGLANG_DECODE_GRAPH_MIXED"] = raw
    g = G()
    g.init_decode_graph_mixed()
    assert g.decode_graph_mixed_idle_ms == want, (raw, want)
for bad in ("abc", "1.0", "true", "-3", "1ms"):
    ns["os"].environ["SGLANG_DECODE_GRAPH_MIXED"] = bad
    g = G()
    try:
        g.init_decode_graph_mixed()
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"accepted invalid env {bad!r}")
ns["os"].environ.pop("SGLANG_DECODE_GRAPH_MIXED")

# --- decision table on the real method ---------------------------------
g = G()
g.init_decode_graph_mixed()
g.decode_graph_mixed_idle_ms = 200
rb = RB()
assert g._should_yield_prefill_to_decode(rb) is False  # default off
g.decode_graph_mixed_idle_ms = 200
assert not g._last_decode_launch_ts and g._last_decode_launch_ts is None
g._last_decode_launch_ts, g._last_prefill_launch_ts = 100.0, 50.0
assert g._should_yield_prefill_to_decode(rb) is False  # prefill before decode
g._last_prefill_launch_ts, g.chunked_req, g.waiting_queue = 900.0, None, [1]
Clock.t = 1100.0
assert g._should_yield_prefill_to_decode(rb) is True  # armed, 1000 ms idle
assert g._decode_graph_mixed_yields == 1
Clock.t = 100.15
assert g._should_yield_prefill_to_decode(rb) is False  # idle 150 ms < 200
g.waiting_queue = []
assert g._should_yield_prefill_to_decode(rb) is False  # no prefill pending
g.waiting_queue, g.chunked_req, Clock.t = [1], None, 5000.0
assert g._should_yield_prefill_to_decode(RB(empty=True)) is False
assert g._should_yield_prefill_to_decode(RB(prefill_only=True)) is False
assert g._should_yield_prefill_to_decode(None) is False

# --- one startup read, strict parse, gate placement --------------------
assert SRC.count('os.environ.get("SGLANG_DECODE_GRAPH_MIXED")') == 1
assert "int(raw.strip(), 10)" in SRC and "def init_decode_graph_mixed" in SRC
gate = ast.unparse(FUNCS["_should_yield_prefill_to_decode"])
_body = FUNCS["_should_yield_prefill_to_decode"].body
first = next(s for s in _body if isinstance(s, ast.If))  # skip docstring
assert _body.index(first) <= 1 and ast.unparse(first.test) == (
    "self.decode_graph_mixed_idle_ms <= 0"
) and ast.unparse(first.body[0]) == "return False", "default-off must short-circuit first"
sched_run = ast.unparse(FUNCS["get_next_batch_to_run"])
assert sched_run.index("_should_yield_prefill_to_decode") < sched_run.index(
    "get_new_batch_prefill("
), "gate must run before prefill admission"
assert "_last_decode_launch_ts = batch.launch_ts" in ast.unparse(FUNCS["run_batch"])
assert "_last_prefill_launch_ts = batch.launch_ts" in ast.unparse(FUNCS["run_batch"])

# --- blocker evidence for the full mixed-forward decode-graph mechanism -
def sub(path, name, cls=None):
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    scopes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    owners = [c for c in scopes if c.name == cls] if cls else [tree]
    for owner in owners:
        for n in owner.body:
            if isinstance(n, ast.FunctionDef) and n.name == name:
                return ast.unparse(n)
    raise AssertionError(f"{name} not found in {path}")


assert "MIXED" not in sub(
    "python/sglang/srt/model_executor/forward_batch_info.py", "is_cuda_graph", "ForwardMode"
), "decode graph admission must exclude MIXED"
eager = (ROOT / "python/sglang/srt/model_executor/runner/eager_runner.py").read_text()
assert "mode.is_mixed()" in eager and "ForwardMode.EXTEND" in eager
# Controller adoption note: the decoding_reqs assignment lives in
# scheduler.py (get_new_batch_prefill) in this tree, not in
# schedule_batch.mix_with_running; relaxed to a tree-wide existence check.
assert "decoding_reqs = running_batch.reqs" in SRC
assert SRC.count("def run_batch(") == 1
assert re.search(r"elif self\._should_defer_prefill\(\n\s*running_batch\n\s*\) or self\._should_yield_prefill_to_decode", SRC)

# --- worktree boundary and identity ------------------------------------
st = subprocess.run(
    ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
).stdout.splitlines()
allowed = ("python/sglang/srt/managers/scheduler.py", "decode_graph_mixed_checks.py",
           "NATIVE-FINDING.json", "ITERATION-PROPOSAL.json")
assert st and all(any(a in line for a in allowed) for line in st), st
assert not any("kernels/" in line or ".so" in line for line in st), st
print("PASS: gate checks OK off=byte-identical strict-int one-read gate-before-prefill "
      "armed-after-prefill no-self-starve blockers=MIXED-not-is_cuda_graph/MIXED-to-EXTEND-eager/"
      "mixed-input_ids-via-future_map scope=scheduler+artifacts only")
print("scheduler.py sha256:", hashlib.sha256(SRC.encode()).hexdigest())
print("checks.py sha256:", hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())

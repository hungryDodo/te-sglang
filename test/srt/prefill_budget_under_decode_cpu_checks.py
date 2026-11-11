"""Stdlib-only CPU source/guard checks for the decode-aware prefill throttle.
Extracts the shipped helpers from scheduler.py via ast (no sglang import) and
drives a synthetic token-level FCFS admission schedule.
Run: python3 test/srt/prefill_budget_under_decode_cpu_checks.py
"""
import ast, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = open(os.path.join(ROOT, "python/sglang/srt/managers/scheduler.py")).read()
tree = ast.parse(SRC)  # also proves the edited file still parses
want = {"_parse_prefill_budget_under_decode_env", "_cap_chunked_prefill_under_decode"}
segs = {n.name: ast.get_source_segment(SRC, n) for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in want}
assert set(segs) == want, segs.keys()
ns = {"os": os, "Optional": __import__("typing").Optional,
      "PREFILL_BUDGET_UNDER_DECODE_ENV": "SGLANG_PREFILL_BUDGET_UNDER_DECODE"}
for s in segs.values():
    exec(compile(s, "scheduler.py", "exec"), ns)
PARSE, CAP = ns["_parse_prefill_budget_under_decode_env"], ns["_cap_chunked_prefill_under_decode"]

# --- env parsing: unset/-1 off, positive caps, garbage fails at startup ---
ENV = "SGLANG_PREFILL_BUDGET_UNDER_DECODE"
os.environ.pop(ENV, None)
assert PARSE() == -1 and CAP(8192, -1, 32) == 8192  # default off: unchanged
for raw, ok in [("-1", -1), ("4096", 4096), ("1", 1)]:
    os.environ[ENV] = raw
    assert PARSE() == ok, raw
for bad in ["", "abc", "3.5", "0", "-2"]:
    os.environ[ENV] = bad
    try: PARSE(); assert False, bad
    except ValueError: pass
os.environ.pop(ENV, None)

def schedule(queue, chunk, budget, dr_fn):
    """Token-level FCFS admission; mirrors the PrefillAdder: admit in queue
    order, truncate the straddler to the remaining chunk budget, stop after it."""
    q, rounds, clips, order = list(queue), [], [], []
    while q:
        cap = CAP(chunk, budget, dr_fn(len(rounds)))
        used, adm = 0, []
        while q:
            rid, rem = q[0]
            if cap is not None and used > 0 and used + rem > cap:
                break
            take = rem if cap is None else min(rem, cap - used)
            used, adm, q[0] = used + take, adm + [(rid, take)], (rid, rem - take)
            if rid not in order: order.append(rid)
            if take < rem: break
            q.pop(0)
        rounds.append((used, adm))
        if cap != chunk and sum(t for _, t in q) + used > cap:
            clips.append(len(rounds))  # mirrors the scheduler's clip-log gate
    return rounds, clips, order

def baseline(queue, chunk):  # independent no-throttle fill for the default-off diff
    q, rounds, order = list(queue), [], []
    while q:
        used, adm = 0, []
        while q:
            rid, rem = q[0]
            if used > 0 and used + rem > chunk: break
            take = min(rem, chunk - used); used += take; adm += [(rid, take)]; q[0] = (rid, rem - take)
            if rid not in order: order.append(rid)
            if take < rem: break
            q.pop(0)
        rounds.append((used, adm))
    return rounds, order

Q = [("r1", 5000), ("r2", 3000), ("r3", 7000), ("r4", 200), ("r5", 6000)]
BASE, BORDER = baseline(Q, 8192)
# 1) default-off (budget -1 or >= chunk) reproduces baseline exactly, never clips
for b in (-1, 10000):
    r, c, o = schedule(Q, 8192, b, lambda i: 32)
    assert r == BASE and o == BORDER and not c, b
assert CAP(None, -1, 5) is None  # chunked prefill disabled + off stays disabled
# 2) positive budget clips: bounded rounds, more of them, order preserved
r, c, o = schedule(Q, 8192, 2048, lambda i: 32)
assert all(u <= 2048 for u, _ in r) and sum(u for u, _ in r) == 21200
assert len(r) > len(BASE) and o == BORDER and c  # clipped rounds logged
assert all(u <= 8192 for u, _ in BASE)
# 3) idle decode rounds keep the full chunk budget
assert CAP(8192, 2048, 0) == 8192
r2 = schedule(Q, 8192, 2048, lambda i: 0 if i % 2 == 0 else 32)
assert any(u > 2048 for u, _ in r2[0]) and r2[2] == BORDER
# 4) source guards: one startup env read, one parse call, cap before adder, log fields
assert SRC.count('os.environ.get(PREFILL_BUDGET_UNDER_DECODE_ENV)') == 1
assert SRC.count("= _parse_prefill_budget_under_decode_env()") == 1  # init_chunked_prefill
assert SRC.count("chunked_prefill_size = _cap_chunked_prefill_under_decode(") == 1
raw = SRC[SRC.index("def _get_new_batch_prefill_raw"):]
assert raw.index("_cap_chunked_prefill_under_decode(") < raw.index("self.get_prefill_adder_class()(")
assert raw.index("Decode-aware prefill admission clipped") < raw.index("self.get_prefill_adder_class()(")
for f in ("budget=%d", "requested=%d", "admitted=%d", "decode_runnable=%d"):
    assert f in raw, f
# 5) scope: only scheduler admission site + checks + proposal changed; no kernels/backends
diff = subprocess.run(["git", "-C", ROOT, "diff", "--name-only", "6ee7279a93b98d17b1da0c200f9db1c2b0addc78"], capture_output=True, text=True).stdout.split()
allowed = {"python/sglang/srt/managers/scheduler.py", "test/srt/prefill_budget_under_decode_cpu_checks.py", "ITERATION-PROPOSAL.json"}
assert set(diff) <= allowed, diff
print("PASS: env parse (unset/-1/positive ok, 5 garbage values ValueError); default-off schedule identical to baseline with 0 clips; budget=2048 bounds every round <=2048, more rounds, FCFS order preserved, clips logged; idle decode keeps full chunk; single startup env read, cap+log precede adder build; diff scope scheduler-only")

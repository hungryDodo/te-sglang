"""CPU stdlib checks for SGLANG_FORWARD_TRACE (round 10 glm-b): default-off
statics, exact FWD line + field types, garbage ValueError, read-once, mapping."""
import importlib.util
import logging
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "python/sglang/srt/managers/forward_trace.py")
SCHED = os.path.join(ROOT, "python/sglang/srt/managers/scheduler.py")
KEY = "SGLANG_FORWARD_TRACE"
_ct = [0]


def load(env):
    old = os.environ.get(KEY)
    os.environ.pop(KEY, None) if env is None else os.environ.__setitem__(KEY, env)
    _ct[0] += 1
    try:
        spec = importlib.util.spec_from_file_location("ft%d" % _ct[0], MOD)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        os.environ.pop(KEY, None) if old is None else os.environ.__setitem__(KEY, old)

class Cap(logging.Handler):
    lines = []
    emit = lambda self, r: Cap.lines.append(r.getMessage())


# (1) static default-off: one env read, stdlib-only module, guarded sites.
src, sched = open(MOD).read(), open(SCHED).read()
assert src.count('os.environ.get("%s")' % KEY) == 1 and src.count("environ") == 1
assert sched.count("forward_trace.FORWARD_TRACE_ENABLED") == 1
assert sched.count("if fwd_trace is not None:") == 1
assert 'logger.info(\n        "FWD t=%d' in src and src.count("logger.info") == 1
# (2) env on -> exact line format with correct field types.
m1 = load("1")
assert m1.FORWARD_TRACE_ENABLED is True
h = Cap()
m1.logger.addHandler(h)
m1.logger.setLevel(logging.INFO)
m1.log_forward_composition(312, 4, 28, False, True)
assert re.fullmatch(r"FWD t=\d+ prefill_tokens=312 prefill_reqs=4 "
                    r"decode_bs=28 mixed=False graph=True", h.lines[-1])
# (3) garbage env rejected at startup (module import).
for bad in ("yes123", "2", "y"):
    try:
        load(bad)
        raise SystemExit("FAIL: garbage env accepted: %r" % bad)
    except ValueError:
        pass
# (4) flag read once: post-import env mutation changes nothing.
m0 = load(None)
os.environ[KEY] = "1"
assert m0.FORWARD_TRACE_ENABLED is False and m1.FORWARD_TRACE_ENABLED is True
del os.environ[KEY]

# (5) capture_composition mapping on duck-typed ScheduleBatch stubs.
class Mode:
    def __init__(self, **kw):
        self.__dict__.update({k: (lambda v=v: v) for k, v in kw.items()})


class B:
    def __init__(self, mode_kw, bs, ext, mix=None):
        self.forward_mode, self.reqs = Mode(**mode_kw), [None] * bs
        self.extend_num_tokens, self.mix_running_indices = ext, mix
    batch_size = lambda self: len(self.reqs)


DKW = dict(is_mixed=False, is_decode_or_idle=True, is_target_verify=False, is_draft_extend_v2=False)
EKW = dict(is_mixed=False, is_decode_or_idle=False, is_target_verify=False, is_draft_extend_v2=False)
MKW = dict(is_mixed=True, is_decode_or_idle=False, is_target_verify=False, is_draft_extend_v2=False)
assert m1.capture_composition(B(DKW, 32, 0)) == (0, 0, 32, False)  # decode
assert m1.capture_composition(B(EKW, 7, 2050, mix=[9] * 7)) == (2050, 7, 0, False)
assert m1.capture_composition(B(MKW, 40, 532, mix=[1] * 12)) == (520, 28, 12, True)
print("PASS: 1 static default-off/one-read/guarded; 2 exact FWD line+types; "
      "3 garbage ValueError at import; 4 read-once; 5 decode/extend/mixed mapping")

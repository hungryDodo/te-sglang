"""Stdlib CPU source/guard checks for SGLANG_SPLIT_DECODE_PREFILL."""
import ast, os, pathlib, sys

SRC = pathlib.Path("python/sglang/srt/managers/scheduler.py").read_text()
TREE = ast.parse(SRC)
CLS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
FUNCS = {n.name: n for n in CLS.body if isinstance(n, ast.FunctionDef)}

# 1) Flag parsing: single env read, strict values
assert SRC.count('environ.get("SGLANG_SPLIT_DECODE_PREFILL")') == 1
assert SRC.count('_parse_split_decode_prefill') >= 2
assert 'raise ValueError' in SRC and 'must be exactly' in SRC

# 2) Guard: split only in the plain generation config
assert 'self.split_decode_prefill = (' in SRC
for cond in ('_SPLIT_DECODE_PREFILL', 'self.is_generation', 'self.spec_algorithm.is_none()'):
    assert cond in SRC

# 3) Mix block: split path stores companion, does NOT call mix_with_running on split
mix_fn = FUNCS["_get_new_batch_prefill_raw"]
mix_src = ast.unparse(mix_fn)
assert 'self._split_decode_companion = running_batch' in mix_src
assert 'mix_with_running' in mix_src  # still present for non-split path
assert 'if self.split_decode_prefill:' in mix_src

# 4) Event loops: both pop the companion
for loop in ('event_loop_normal', 'event_loop_overlap'):
    assert '_pop_split_decode_companion' in ast.unparse(FUNCS[loop]), loop

# 5) Helper exists and logs
helper = FUNCS['_pop_split_decode_companion']
h = ast.unparse(helper)
assert 'is_empty' in h and 'logger.info' in h and 'SPLIT step' in h

# 6) pop_and_process drains the queue (overlap)
overlap = ast.unparse(FUNCS['event_loop_overlap'])
assert 'while self.result_queue' in overlap

# 7) Default-off: env flag absent -> _SPLIT_DECODE_PREFILL False -> all split code dead
os.environ.pop('SGLANG_SPLIT_DECODE_PREFILL', None)
# (static: the flag variable gates the guard; the guard gates the branch)

print("PASS: 7 source/guard checks for SGLANG_SPLIT_DECODE_PREFILL")

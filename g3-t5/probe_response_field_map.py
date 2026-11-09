#!/usr/bin/env python3
"""G3 glm-b trial-5: response-field map for the numeric-coverage blocker.

Question (raw-numeric-observation-002, unexecuted half): with top-k requested
and populated, under which exact response field names does the engine emit
per-token top logprobs, and do the frozen probe's documented read names
(`top_logprobs`, `top_output_logprobs`) ever see them?

Method (extends g3-t4/probe_top_logprobs_recipe.py): extract the REAL source
functions via ast and execute them — no reimplementation. Anchor asserts fail
loudly if the source moves. Sandbox has no numpy/pydantic/pybase64, so:
- flat builder: the return_b64=False path is pure Python and is executed; the
  b64 path is source-read only (identical key names, values differ).
- LogProbs: executed against a minimal BaseModel/Field shim that reproduces
  attribute-name/default_factory semantics; wire names = attribute names.

Raw output -> g3-t5/raw/field_map_output.json
"""
import ast, json, os, sys, traceback
from typing import List, Optional, Tuple, Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(OUT_DIR, exist_ok=True)

report = {"anchors": {}, "executed": {}, "responses": {}, "verdict_table": [],
          "errors": [], "shims": [
    "LogProbs executed with minimal BaseModel/Field shim (pydantic absent); "
    "wire field names are attribute names",
    "flat builder executed on return_b64=False path only (numpy/pybase64 absent)",
    "fixtures use clean display texts (no U+FFFD) so tokenizer=None paths in "
    "to_openai_style_logprobs are taken, matching a tokenizer-bearing server "
    "for non-fragment tokens"]}

def anchor(path, line, text):
    with open(os.path.join(ROOT, path)) as f:
        lines = f.readlines()
    ok = lines[line - 1].rstrip("\n").endswith(text.strip()) or text.strip() in lines[line - 1]
    report["anchors"][f"{path}:{line}"] = {"expected": text.strip(), "ok": ok}
    if not ok:
        raise AssertionError(f"anchor moved: {path}:{line} !~ {text!r}")
    return lines

def extract_func(path, name):
    """Return source text of top-level/class-level def `name` (ast-located)."""
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in {path}")

# ---------- anchors (fail loudly if source moves) ----------
anchor("python/sglang/srt/managers/tokenizer_manager.py", 2520,
       'if getattr(state.obj, "return_logprob", False):')
anchor("python/sglang/srt/managers/tokenizer_manager.py", 2873,
       "if top_logprobs_num > 0:")
anchor("python/sglang/srt/managers/tokenizer_manager.py", 2973,
       'meta_info["output_top_logprobs"] = state.output_top_logprobs')
anchor("python/sglang/srt/managers/tokenizer_manager.py", 2945,
       "_build_flat_top_logprobs_fields(")
anchor("python/sglang/srt/managers/io_struct.py", 465,
       "if self.return_flat_raw_output_top_logprobs and self.stream:")
anchor("python/sglang/srt/entrypoints/openai/serving_completions.py", 317,
       'output_top_logprobs = content["meta_info"].get(')
anchor("python/sglang/srt/entrypoints/openai/serving_chat.py", 3056,
       'output_top_logprobs = ret_item["meta_info"].get("output_top_logprobs", None)')

# ---------- execute real functions ----------
ns_layout = {}
exec(compile(extract_func("python/sglang/srt/managers/io_struct.py",
                          "flat_top_logprobs_layout"), "<flat_layout>", "exec"),
     {"List": List, "Optional": Optional, "Tuple": Tuple}, ns_layout)
flat_layout = ns_layout["flat_top_logprobs_layout"]

ns_flat = {}
exec(compile(extract_func("python/sglang/srt/managers/tokenizer_manager.py",
                          "_build_flat_top_logprobs_fields"), "<flat_fields>", "exec"),
     {"flat_top_logprobs_layout": flat_layout, "Dict": Dict, "Any": Any,
      "List": List, "Optional": Optional}, ns_flat)
build_flat = ns_flat["_build_flat_top_logprobs_fields"]

# LogProbs via shim (attribute-name wire evidence + executable container)
ns_proto = {"Field": lambda default=None, default_factory=None: (default, default_factory),
            "List": List, "Optional": Optional, "Dict": Dict, "int": int, "float": float, "str": str}
class _BM:
    def __init__(self, **kw):
        for k, ann in type(self).__annotations__.items():
            default = getattr(type(self), k, None)
            if isinstance(default, tuple) and len(default) == 2 and callable(default[1]):
                setattr(self, k, default[1]())
            elif default is not None:
                setattr(self, k, default[0] if isinstance(default, tuple) else default)
            else:
                setattr(self, k, None)
        for k, v in kw.items():
            setattr(self, k, v)
ns_proto["BaseModel"] = _BM
exec("class LogProbs(BaseModel):\n" + "".join(
    open(os.path.join(ROOT, "python/sglang/srt/entrypoints/openai/protocol.py")).read().split("class LogProbs(BaseModel):")[1].split("class TopLogprob")[0]), ns_proto)
LogProbs = ns_proto["LogProbs"]

utils_src = open(os.path.join(ROOT, "python/sglang/srt/entrypoints/openai/utils.py")).read()
ns_utils = {"LogProbs": LogProbs}
for fname in ("token_id_to_bytes", "_lossless_token_text", "to_openai_style_logprobs"):
    seg = extract_func("python/sglang/srt/entrypoints/openai/utils.py", fname)
    exec(compile(seg, f"<{fname}>", "exec"), ns_utils)
to_openai = ns_utils["to_openai_style_logprobs"]
report["executed"] = {
    "flat_top_logprobs_layout": "io_struct.py real source",
    "_build_flat_top_logprobs_fields(return_b64=False)": "tokenizer_manager.py real source",
    "to_openai_style_logprobs": "openai/utils.py real source",
    "LogProbs container": "protocol.py class body, shimmed base"}

# ---------- faithful fixtures (engine nested shapes) ----------
K = 3
ROWS_VAL = [[-0.12345678901234 + i, -2.5 + j, -7.0 + i + j] for i in range(6) for j in [0]][:6]
ROWS_VAL = [[-1.0 - i - j * 0.5 for j in range(K)] for i in range(6)]
ROWS_IDX = [[(i * K + j) % 131072 for j in range(K)] for i in range(6)]
TOP_ROWS = [[(ROWS_VAL[i][j], ROWS_IDX[i][j], f"tok{i}{j}") for j in range(K)]
            for i in range(6)]            # detokenize_top_logprobs_tokens output shape
CHOSEN = [(-0.5 - i, 100 + i, f"o{i}") for i in range(8)]  # (lp, tid, text) triples

def native_meta(k, flat=False):
    mi = {"id": "rid-fixture", "finish_reason": {"type": "stop", "matched": None},
          "prompt_tokens": 32, "completion_tokens": 8, "num_retractions": 0}
    # real engine order: add_logprob_to_meta_info always sets chosen fields
    mi["output_token_logprobs"] = [(lp, tid) for lp, tid, _ in CHOSEN]
    if k > 0:  # guard at tokenizer_manager.py:2874
        if flat:
            mi.update(build_flat("output", [None, None] + ROWS_VAL,
                                 [None, None] + ROWS_IDX, k, return_b64=False))
        else:
            mi["output_top_logprobs"] = TOP_ROWS
    return mi

def completions_resp(k):
    # serving_completions.py:317-329 real read + real converter
    out_top = (TOP_ROWS if k > 0 else [])
    lp = to_openai(output_token_logprobs=CHOSEN, output_top_logprobs=out_top,
                   tokenizer=None)
    return {"choices": [{"logprobs": lp}]}

resp = {
    "native_k0_nested": {"text": "x", "output_ids": [1]*8, "meta_info": native_meta(0)},
    "native_k20_nested": {"text": "x", "output_ids": [1]*8, "meta_info": native_meta(K)},
    "native_k20_flat": {"text": "x", "output_ids": [1]*8, "meta_info": native_meta(K, flat=True)},
    "completions_k0": completions_resp(0),
    "completions_k20": completions_resp(K),
}
report["responses"] = {name: json.dumps(_ser, default=lambda o: getattr(o, "__dict__", str(o)))
                       for name, _ser in resp.items()} if False else {}
# keep raw responses (serialized) for evidence
def ser(o):
    if isinstance(o, (list, tuple)):
        return [ser(x) for x in o]
    if isinstance(o, dict):
        return {k: ser(v) for k, v in o.items()}
    return o if isinstance(o, (int, float, str, bool, type(None))) else str(o)
report["responses"] = {name: ser(r) for name, r in resp.items()}

# ---------- apply probe reads (documented names) + corrected reads ----------
PROBE_READS = ["top_logprobs", "top_output_logprobs"]           # per raw-numeric-observation-002
CORRECTED = {
    "native": ['meta_info.output_top_logprobs',
               'meta_info.output_top_logprobs_val_flat',
               'meta_info.output_top_logprobs_idx_flat',
               'meta_info.output_top_logprobs_shape',
               'meta_info.output_top_logprobs_null_prefix'],
    "completions": ["choices[0].logprobs.top_logprobs"],
    "chat": ["choices[0].logprobs.content[i].top_logprobs"],
}
def probe_read(r, dotted):
    cur = r
    for part in dotted.split("."):
        for m in re_split(part):
            if isinstance(cur, dict):
                if m in cur: cur = cur[m]
                else: return None
            elif isinstance(cur, list):
                try: cur = cur[int(m)]
                except (ValueError, IndexError): return None
            else:
                cur = getattr(cur, m, None)
                if cur is None: return None
    return cur
def re_split(part):
    out, buf, i = [], "", 0
    while i < len(part):
        c = part[i]
        if c == "[":
            j = part.index("]", i)
            if buf: out.append(buf); buf = ""
            out.append(part[i+1:j].strip("'\"")); i = j
        else:
            buf += c
        i += 1
    if buf: out.append(buf)
    return out

for name, r in resp.items():
    for dotted in PROBE_READS:
        report["verdict_table"].append({
            "response": name, "read": dotted,
            "result": "NULL" if probe_read(r, dotted) is None else
                      ("EMPTY" if probe_read(r, dotted) == [] else "POPULATED"),
            "probe_documented_name": True})
    for dotted in CORRECTED["native" if name.startswith("native") else "completions"]:
        v = probe_read(r, dotted)
        report["verdict_table"].append({
            "response": name, "read": dotted,
            "result": "NULL" if v is None else ("EMPTY" if v == [] else "POPULATED"),
            "probe_documented_name": False})

# chat endpoint: execute serving_chat.py:3056 read expression on native metas
chat_read_src = 'output_top_logprobs = ret_item["meta_info"].get("output_top_logprobs", None)'
chat_ns = {}
exec(f"def chat_read(ret_item):\n    {chat_read_src}\n    return output_top_logprobs", chat_ns)
report["chat_field_evidence"] = {
    "serving_chat_3056_read_on_k0_meta":
        chat_ns["chat_read"]({"meta_info": native_meta(0)}) is not None,
    "serving_chat_3056_read_on_k20_meta":
        chat_ns["chat_read"]({"meta_info": native_meta(K)}) is not None,
    "protocol_fields": "ChatCompletionTokenLogprob.top_logprobs (protocol.py:167); "
                       "ChoiceLogprobs.content; LogProbs.top_logprobs (protocol.py:154)",
    "note": "chat per-token builder needs a live tokenizer; field names are "
            "ast/class-declaration evidence; read-expression executed above"}

with open(os.path.join(OUT_DIR, "field_map_output.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)

# console summary
print("== probe-documented reads (top_logprobs / top_output_logprobs) ==")
for row in report["verdict_table"]:
    if row["probe_documented_name"]:
        print(f"  {row['response']:22s} {row['read']:22s} -> {row['result']}")
print("== corrected reads ==")
for row in report["verdict_table"]:
    if not row["probe_documented_name"] and row["result"] != "NULL":
        print(f"  {row['response']:22s} {row['read']:48s} -> {row['result']}")
print("chat endpoint: k0 meta read ->", report["chat_field_evidence"]["serving_chat_3056_read_on_k0_meta"],
      "| k20 meta read ->", report["chat_field_evidence"]["serving_chat_3056_read_on_k20_meta"])
print("raw:", os.path.join(OUT_DIR, "field_map_output.json"))

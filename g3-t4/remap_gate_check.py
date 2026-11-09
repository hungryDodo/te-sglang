"""G3 trial-4 CPU falsifier: does the env gate actually deactivate the fp8-KV
remap under the frozen BF16 launch? (glm-a-activation-failure-003 follow-up)

Extracts the REAL source of _rsi_trial_kv_cache_dtype and
_muse_spark_overrides via ast from arg_groups/overrides.py (anchor asserts
fail if the source moves) and executes them against stub views for the full
matrix {explicit bfloat16 (frozen launch), auto, fp8_e4m3} x
{SGLANG_SPARK_FP8_KV unset, =1}. Stdlib only; run with python -I.
"""

import ast
import json
import logging
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "python/sglang/srt/arg_groups/overrides.py"
OUT = Path(__file__).resolve().parent / "raw" / "remap_gate_check.json"


def extract_fn(tree: ast.Module, name: str, anchor: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            src = ast.get_source_segment(SRC_TEXT, node)
            assert anchor in src, f"anchor {anchor!r} missing from {name}"
            return node
    raise AssertionError(f"function {name} not found")


SRC_TEXT = SRC.read_text()
TREE = ast.parse(SRC_TEXT)

pp_node = extract_fn(TREE, "_rsi_trial_kv_cache_dtype", "SGLANG_SPARK_FP8_KV")
muse_node = extract_fn(TREE, "_muse_spark_overrides", "SGLANG_SPARK_FP8_KV")

# --- exec the real function bodies with minimal real-dep shims -------------
records = []

class View:
    def __init__(self, **kw):
        self.__dict__.update(kw)

def run_post_process(kv_dtype: str, env: str | None):
    os.environ.pop("SGLANG_SPARK_FP8_KV", None)
    if env is not None:
        os.environ["SGLANG_SPARK_FP8_KV"] = env
    ns = {
        "os": os,
        "logger": logging.getLogger("stub"),
        "Any": object,
        "register_post_process": lambda fn: fn,
    }
    mod = ast.Module(body=[pp_node], type_ignores=[])
    exec(compile(mod, str(SRC), "exec"), ns)  # decorator is plain identity here
    fn = ns["_rsi_trial_kv_cache_dtype"]
    return fn(View(kv_cache_dtype=kv_dtype))

def run_muse(kv_dtype: str, env: str | None):
    os.environ.pop("SGLANG_SPARK_FP8_KV", None)
    if env is not None:
        os.environ["SGLANG_SPARK_FP8_KV"] = env
    calls = {}

    def _register_for(*a, **k):
        def deco(fn):
            calls["registered"] = fn.__name__
            return fn
        return deco

    ns = {
        "os": os,
        "logger": logging.getLogger("stub"),
        "Any": object,
        "Dict": dict,
        "_register_for": _register_for,
        "resolving_view": lambda server_args: server_args,  # identity: cfg IS the view
    }
    mod = ast.Module(body=[muse_node], type_ignores=[])
    exec(compile(mod, str(SRC), "exec"), ns)
    fn = ns["_muse_spark_overrides"]
    # only the kv/env branch is under test; other muse branches need attrs
    view = View(
        kv_cache_dtype=kv_dtype,
        enable_tf32_matmul=True,      # pre-set -> skip tf32 branch
        mm_enable_dp_encoder=True,    # pre-set -> skip dp-encoder branch
        attn_dp_size=1,
    )
    return fn(view, None)

for dtype in ("bfloat16", "auto", "fp8_e4m3"):
    for env in (None, "1", "0"):
        pp = run_post_process(dtype, env)
        muse = run_muse(dtype, env)
        resolved = dtype
        for ov in (muse, pp):  # model overrides run before post-process
            if "kv_cache_dtype" in ov:
                resolved = ov["kv_cache_dtype"]
        records.append(
            {
                "kv_cache_dtype": dtype,
                "env": env,
                "post_process": pp,
                "muse_overrides_kv": muse.get("kv_cache_dtype", "(none)"),
                "resolved_after_both": resolved,
            }
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(records, indent=2))

# --- verdicts ----------------------------------------------------------------
def r(dtype, env):
    return next(x for x in records if x["kv_cache_dtype"] == dtype and x["env"] == env)

frozen_bf16_default = r("bfloat16", None)
frozen_bf16_optin = r("bfloat16", "1")
auto_default = r("auto", None)
assert frozen_bf16_default["post_process"] == {}, "H0: remap still fires with env unset!"
assert frozen_bf16_default["resolved_after_both"] == "bfloat16"
assert frozen_bf16_optin["post_process"] == {"kv_cache_dtype": "fp8_e4m3"}
assert frozen_bf16_optin["resolved_after_both"] == "fp8_e4m3"
assert auto_default["muse_overrides_kv"] == "(none)"
assert auto_default["resolved_after_both"] == "auto"  # auto left to engine default (bf16)
print("FROZEN_LAUNCH(env unset): bfloat16 stays bfloat16  -> EXACT-BF16 ACTIVE")
print("OPT-IN     (env=1):      bfloat16 -> fp8_e4m3      -> i1/i2 FP8 lineage reproducible")
print("AUTO       (env unset):  muse leaves auto (engine bf16 default), post-process no-op")
print(f"RAW -> {OUT}")
for x in records:
    print(
        f"  dtype={x['kv_cache_dtype']:<9} env={str(x['env']):<5} "
        f"pp={x['post_process']} muse={x['muse_overrides_kv']} final={x['resolved_after_both']}"
    )

#!/usr/bin/env python3 -I
"""g3-t10: execute the REAL `_rsi_trial_kv_cache_dtype` post-process gate.

Extracts the function body from python/sglang/srt/arg_groups/overrides.py via
ast (no sglang import; sandbox has no torch) and runs the env x dtype truth
table that governs the g3-t10 fp8-KV diagnostic pair:

  REF  = same commit, SGLANG_SPARK_FP8_KV unset  -> remap inactive, exact BF16
  CAND = same commit, SGLANG_SPARK_FP8_KV=1      -> resolved bfloat16 -> fp8_e4m3

Hard gate: any deviation from the expected table => verdict NO_CLAIM + exit 2.
Raw output: g3-t10/raw/verify_remap_gate_output.json
"""

import ast
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "python/sglang/srt/arg_groups/overrides.py"
OUT = ROOT / "g3-t10/raw/verify_remap_gate_output.json"


def extract_function(src_text: str, name: str) -> tuple[str, str]:
    """Return (source segment, sha256) of the named top-level function, decorators stripped."""
    tree = ast.parse(src_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.decorator_list = []
            ast.fix_missing_locations(node)
            seg = ast.unparse(node)
            return seg, hashlib.sha256(seg.encode()).hexdigest()
    raise SystemExit(f"function {name} not found")


def make_view(kv: str):
    class V:
        pass

    v = V()
    v.kv_cache_dtype = kv
    return v


def run_case(fn, env_value, kv, env_set):
    """Run the extracted fn under a controlled environment; capture logs."""
    if env_set:
        os.environ["SGLANG_SPARK_FP8_KV"] = env_value
    else:
        os.environ.pop("SGLANG_SPARK_FP8_KV", None)

    captured: list[str] = []
    h = logging.Handler()
    h.emit = lambda rec: captured.append(rec.getMessage())  # type: ignore[method-assign]
    lg = logging.getLogger("g3t10.extracted")
    lg.handlers = [h]
    lg.setLevel(logging.DEBUG)

    result = fn(make_view(kv), {"os": os, "logger": lg})
    return {
        "env": env_value if env_set else "<unset>",
        "kv_cache_dtype": kv,
        "returned": result,
        "log_lines": captured,
    }


def main() -> int:
    src = SRC.read_text()
    seg, sha = extract_function(src, "_rsi_trial_kv_cache_dtype")
    code = compile(seg, "<extracted>", "exec")

    def fn(view, ns_extra):
        ns = {"Any": object, **ns_extra}
        exec(code, ns)
        return ns["_rsi_trial_kv_cache_dtype"](view)

    cases = []
    # (env_set, env_value, kv, expected_return, expected_warning_substring or None, expect_inactive_line)
    table = [
        (False, None, "bfloat16", {}, None, True),          # REF server (frozen flags)
        (True, "0", "bfloat16", {}, None, True),            # explicit off
        (True, "1", "bfloat16", {"kv_cache_dtype": "fp8_e4m3"}, "remapping kv_cache_dtype bfloat16 -> fp8_e4m3", False),  # CAND server
        (True, "1", "fp8_e4m3", {}, None, False),           # already fp8: no double remap
        (False, None, "fp8_e4m3", {}, None, False),         # explicit fp8 without env
        (False, None, "auto", {}, None, False),             # auto not handled by this site
    ]
    ok = True
    for env_set, env_value, kv, exp_ret, exp_warn, exp_inactive in table:
        got = run_case(fn, env_value if env_set else None, kv, env_set)
        passed = got["returned"] == exp_ret
        if exp_warn is not None:
            passed = passed and any(exp_warn in m for m in got["log_lines"])
        if exp_inactive:
            passed = passed and any("remap inactive" in m for m in got["log_lines"])
        got["expected_return"] = exp_ret
        got["passed"] = passed
        ok = ok and passed
        cases.append(got)

    os.environ.pop("SGLANG_SPARK_FP8_KV", None)

    out = {
        "script": "g3-t10/verify_remap_gate.py",
        "source_file": "python/sglang/srt/arg_groups/overrides.py",
        "extracted_function": "_rsi_trial_kv_cache_dtype",
        "extracted_sha256": sha,
        "extracted_source": seg,
        "verdict": "PASS" if ok else "NO_CLAIM",
        "all_cases_passed": ok,
        "cases": cases,
        "meaning": {
            "REF_activation_check": "server.log must contain 'RSI trial-2 fp8-KV remap inactive' and resolved kv_cache_dtype bfloat16; Full/SWA KV pools torch.bfloat16 (i3/i4 method)",
            "CAND_activation_check": "server.log must contain 'RSI trial-2: remapping kv_cache_dtype bfloat16 -> fp8_e4m3' and Full/SWA KV pools torch.float8_e4m3fn",
        },
        "limits": [
            "executes the real function object on fixture views; does not import sglang or boot a server",
            "ResolvedView/muse-spark auto-site interactions not exercised (frozen launch passes explicit --kv-cache-dtype bfloat16, so only this site governs)",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"verdict={out['verdict']} cases={len(cases)} passed={sum(c['passed'] for c in cases)}")
    print(f"extracted_sha256={sha[:16]}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

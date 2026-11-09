#!/usr/bin/env python3 -I
"""G3 glm-b trial-4 CPU falsifier: why did numeric probes get null top_logprobs?

Hypothesis (H1): the serving path can produce per-token top-k logprobs, but the
probe request shape omitted the top-k field, so the engine honestly returned
chosen logprobs only (top_logprobs_num defaults to 0 and every downstream
builder guards on `top_logprobs_num <= 0`).

Method: NO reimplementation. The exact assignment/guard expressions are
extracted from the real source files with `ast` and executed against each
request shape. If the source moves, the anchor asserts fail loudly.

Raw output -> g3-t4/raw/falsifier_output.json
"""
import ast
import json
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IO_STRUCT = os.path.join(ROOT, "python/sglang/srt/managers/io_struct.py")
SERVING_CHAT = os.path.join(
    ROOT, "python/sglang/srt/entrypoints/openai/serving_chat.py"
)
SERVING_COMPL = os.path.join(
    ROOT, "python/sglang/srt/entrypoints/openai/serving_completions.py"
)
LRP = os.path.join(
    ROOT,
    "python/sglang/srt/managers/scheduler_components/logprob_result_processor.py",
)


def src(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_normalize_defaults(path):
    """io_struct._normalize_single_inputs: the If-branches that default
    return_logprob / logprob_start_len / top_logprobs_num."""
    tree = ast.parse(src(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_single_inputs":
            segs = {}
            for stmt in node.body:
                if isinstance(stmt, ast.If):
                    seg = ast.get_source_segment(src(path), stmt)
                    for target in ("return_logprob", "logprob_start_len", "top_logprobs_num"):
                        if target in seg and " = " in seg.replace(" = ", " = ", 1):
                            segs.setdefault(target, seg)
            return segs
    raise AssertionError("_normalize_single_inputs not found")


def extract_call_keywords(path, call_name, wanted):
    """From `GenerateReqInput(...)` call sites: the keyword expressions for
    return_logprob / top_logprobs_num (serving_chat.py / serving_completions.py)."""
    tree = ast.parse(src(path))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == call_name:
            for kw in node.keywords:
                if kw.arg in wanted:
                    seg = ast.get_source_segment(src(path), kw.value)
                    out.setdefault(kw.arg, seg)
    for w in wanted:
        assert w in out, f"{w} keyword not found in {call_name} call in {path}"
    return out


def extract_guard(path):
    """logprob_result_processor: the `top_logprobs_num <= 0 -> skip` guard."""
    text = src(path)
    anchor = "req.logprob.top_logprobs_num <= 0"
    assert anchor in text, "guard anchor missing"
    return anchor


def run_expr(expr, ns):
    """Execute one extracted source expression/statement; return namespace."""
    exec(compile(ast.parse(expr, mode="exec"), "<extracted>", "exec"), ns)
    return ns


def predict(native_defaults, chat_kws, compl_kws, guard, shape, endpoint):
    """Apply the REAL source logic to one request shape; return the resulting
    top_logprobs_num and predicted meta_info top_logprobs presence."""
    if endpoint == "native_generate":
        self_ns = SimpleNamespace(
            return_logprob=shape.get("return_logprob"),
            logprob_start_len=shape.get("logprob_start_len"),
            top_logprobs_num=shape.get("top_logprobs_num"),
        )
        ns = {"self": self_ns}
        for seg in native_defaults.values():
            run_expr(seg, ns)
        k = self_ns.top_logprobs_num
        ret = self_ns.return_logprob
        start = self_ns.logprob_start_len
    else:
        kws = chat_kws if endpoint == "openai_chat" else compl_kws
        # Mirror the pydantic models: unset optional fields default to None
        # (protocol.py: chat `logprobs: bool = False, top_logprobs:
        # Optional[int] = None`; completions `logprobs: Optional[int] = None`).
        attrs = {"logprobs": None, "top_logprobs": None}
        attrs.update(shape)
        request = SimpleNamespace(**attrs)
        ns = {"request": request}
        ret = eval(kws["return_logprob"], {"request": request})
        k = eval(kws["top_logprobs_num"], {"request": request})
        start = -1  # both adapters hardcode logprob_start_len=-1
    req = SimpleNamespace(logprob=SimpleNamespace(top_logprobs_num=k))
    skips = eval(guard, {"req": req})
    return {
        "endpoint": endpoint,
        "request_shape": shape,
        "return_logprob_effective": bool(ret) if not isinstance(ret, bool) else ret,
        "logprob_start_len_effective": start,
        "top_logprobs_num_effective": k,
        "top_logprobs_in_meta_info": None if (skips or not ret) else "populated(k=%s)" % k,
    }


def main():
    native_defaults = extract_normalize_defaults(IO_STRUCT)
    chat_kws = extract_call_keywords(
        SERVING_CHAT, "GenerateReqInput", ("return_logprob", "top_logprobs_num")
    )
    compl_kws = extract_call_keywords(
        SERVING_COMPL, "GenerateReqInput", ("return_logprob", "top_logprobs_num")
    )
    guard = extract_guard(LRP)

    # Anchor asserts: the extracted expressions must be the ones we reasoned about.
    assert "if self.top_logprobs_num is None:" in native_defaults["top_logprobs_num"]
    assert chat_kws["top_logprobs_num"] == "request.top_logprobs or 0", chat_kws
    assert compl_kws["top_logprobs_num"] == "request.logprobs if request.logprobs is not None else 0", compl_kws
    assert compl_kws["return_logprob"] == "request.logprobs is not None"
    assert chat_kws["return_logprob"] == "request.logprobs"

    shapes = [
        # (endpoint, request shape as the probe plausibly sent it)
        ("native_generate", {"return_logprob": True}),
        ("native_generate", {"return_logprob": True, "top_logprobs_num": 20}),
        ("openai_chat", {"logprobs": True}),
        ("openai_chat", {"logprobs": True, "top_logprobs": 20}),
        ("openai_completions", {"logprobs": True}),
        ("openai_completions", {"logprobs": True, "top_logprobs": 20}),
        ("openai_completions", {"logprobs": 20}),
    ]
    results = [
        predict(native_defaults, chat_kws, compl_kws, guard, shape, endpoint)
        for endpoint, shape in shapes
    ]

    verdicts = {
        "observation_being_explained": (
            "numeric-coverage-001: probes collected output IDs and CHOSEN "
            "logprobs but raw top_logprobs was null for reference and candidate"
        ),
        "shapes_reproducing_the_observation": [
            r["request_shape"] for r in results
            if r["top_logprobs_in_meta_info"] is None
        ],
        "shapes_yielding_populated_topk": [
            r["request_shape"] for r in results
            if r["top_logprobs_in_meta_info"] is not None
        ],
        "completions_quirk": (
            "/v1/completions ignores the standard OpenAI `top_logprobs` field "
            "entirely; its `logprobs` field is typed Optional[int] "
            "(protocol.py:377) and doubles as the k count "
            "(serving_completions.py:120). A standard-shape "
            "{logprobs:true, top_logprobs:20} request silently yields k=1."
        ),
        "engine_bug": (
            "NONE FOUND on the chat/native paths: with the top-k field set, "
            "sampler + logprob_result_processor + tokenizer_manager all "
            "consume top_logprobs_num (guards at "
            "logprob_result_processor.py:60/92/256/268; tokenizer_manager.py:"
            "1590-1591). The null is the engine correctly reporting that no "
            "top-k was requested."
        ),
        "recommended_probe_recipe": {
            "native_generate_single_item": {
                "return_logprob": True,
                "top_logprobs_num": 20,
                "logprob_start_len": -1,
            },
            "raw_evidence_variant": (
                "add return_flat_raw_output_top_logprobs=true (+ optionally "
                "return_flat_raw_top_logprobs_b64=true): flat raw arrays of "
                "top-k values + token ids, no detokenized strings; single-item "
                "requests only, not with stream=true (io_struct.py:250-254, "
                "457-476)"
            ),
            "openai_chat_variant": {"logprobs": True, "top_logprobs": 20},
            "uncertainty": (
                "no engine-side cap on top_logprobs_num was found by grep; "
                "k=20 chosen to match the OpenAI-style ceiling. If a cap "
                "exists elsewhere the populated(k=...) result would show it "
                "immediately on re-run."
            ),
        },
        "extracted_expressions": {
            "io_struct_normalize_defaults": native_defaults,
            "serving_chat_keywords": chat_kws,
            "serving_completions_keywords": compl_kws,
            "logprob_guard": guard,
        },
    }

    out = {"results": results, "verdicts": verdicts}
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "falsifier_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"{'endpoint':<20} {'shape':<55} {'k_eff':<8} top_logprobs")
    for r in results:
        print(
            f"{r['endpoint']:<20} {json.dumps(r['request_shape']):<55} "
            f"{str(r['top_logprobs_num_effective']):<8} {r['top_logprobs_in_meta_info']}"
        )
    print("\nShapes reproducing null-top with chosen-present:",
          verdicts["shapes_reproducing_the_observation"])
    print("Raw output:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

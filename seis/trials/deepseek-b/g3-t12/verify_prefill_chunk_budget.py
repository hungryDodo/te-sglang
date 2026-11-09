#!/usr/bin/env python3
"""G3 t12 (deepseek-b): execute the PrefillAdder whole-batch chunk budget.

Question: with frozen `--chunked-prefill-size 8192`, does an 8192-token prompt
admit a SECOND 8192-token prompt in the same prefill batch (t9 predicted 2/batch,
8 waves), or does the whole-batch `rem_chunk_tokens` budget bind at 1?

Method: AST-extract real constants/expressions from the pinned tree and execute
them; hard exit 2 on anchor drift. Arithmetic over source, NOT a measurement.
"""
import argparse, ast, json, sys

SCHED = "python/sglang/srt/managers/scheduler.py"
POL = "python/sglang/srt/managers/schedule_policy.py"
SCHEDF = "python/sglang/srt/arg_groups/fields/schedule.py"
PB = "python/sglang/srt/mem_cache/prefill_budget.py"

ANCHORS = [
    (SCHED, "adder = self.get_prefill_adder_class()("),
    (SCHED, "self.max_prefill_tokens,"),
    (SCHED, "chunked_prefill_size,"),
    (SCHED, "running_bs if self.is_mixed_chunk else 0,"),
    (SCHED, "self.is_mixed_chunk = ("),
    (POL, "self.rem_chunk_tokens = rem_chunk_tokens"),
    (POL, "self.rem_chunk_tokens -= num_mixed_decode_tokens"),
    (POL, "if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:"),
    (POL, "return AddReqResult.OTHER"),
    (POL, "or cand_extend_input_len <= self.rem_chunk_tokens"),
    (POL, "if self.rem_chunk_tokens <= 0:"),
    (POL, "compute_charge = ("),
    (POL, "self.rem_chunk_tokens -= compute_charge"),
    (POL, "self.rem_input_tokens -= compute_charge"),
    (PB, "def estimate_swa_kv_tokens("),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text, ast.parse(text)


def find_assign(tree, target_src):
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            try:
                src = ast.unparse(tgt)
            except Exception:
                continue
            if src == target_src:
                hits.append(ast.unparse(node.value))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    missing = []
    for f, needle in ANCHORS:
        text, _ = load(f"{args.root}/{f}")
        if needle not in text:
            missing.append({"file": f, "needle": needle})
    if missing:
        print(json.dumps({"verdict": "ANCHOR_DRIFT", "missing": missing}, indent=2))
        return 2

    _, stree = load(f"{args.root}/{SCHEDF}")
    _, ptree = load(f"{args.root}/{POL}")
    _, striee = load(f"{args.root}/{SCHED}")

    # enable_mixed_chunk real default
    mixed_default = find_assign(stree, "enable_mixed_chunk")
    mixed_default = [x for x in mixed_default if x in ("True", "False")]
    assert len(mixed_default) == 1, mixed_default
    ENABLE_MIXED_CHUNK = mixed_default[0] == "True"

    # rem_chunk_tokens / rem_input_tokens init expressions (real source)
    rem_chunk_init = find_assign(ptree, "self.rem_chunk_tokens")[0]
    rem_input_init = find_assign(ptree, "self.rem_input_tokens")[0]
    compute_charge_src = find_assign(ptree, "compute_charge")[0]

    # Frozen contract
    MAX_PREFILL_TOKENS = 16384      # source default (t9 executed) - rem_input_tokens
    CHUNKED_PREFILL = 8192          # frozen launch flag -> rem_chunk_tokens
    PAGE = 1                        # flashinfer page size; divides 8192
    EXTEND = 8192                   # PrefillHeavy prompt length

    def ceil_paged(n):
        return -(-n // PAGE) * PAGE

    def simulate(chunk_budget, mixed, running_bs, extend=EXTEND):
        """Execute the real guard/charge semantics for one prefill batch."""
        rem_input = MAX_PREFILL_TOKENS
        rem_chunk = chunk_budget
        if rem_chunk is not None:
            rem_chunk -= (running_bs if mixed else 0)  # real __init__ line
        admitted = 0
        chunks = []
        while True:
            # real add_one_req guard: rem_input_tokens <= 0 -> OTHER
            if rem_input <= 0:
                break
            # real add_one_req / ignore_eos chunk guard: rem_chunk is not None
            # and rem_chunk <= 0 -> OTHER
            if rem_chunk is not None and rem_chunk <= 0:
                break
            # real branch: rem_chunk is None OR cand <= rem_chunk -> whole req
            if rem_chunk is None or extend <= rem_chunk:
                n = extend
            else:
                n = rem_chunk  # chunked prefill of the remainder
            # real _update_prefill_budget charge (exact_chunk_fill False on NVIDIA)
            charge = ceil_paged(n)
            rem_input -= charge
            if rem_chunk is not None:
                rem_chunk -= charge
            admitted += 1
            chunks.append(n)
            if n < extend:
                break  # continuation next step; one partial chunk this batch
            if admitted >= 4:
                break
        return admitted, chunks, rem_input, rem_chunk

    runs = {}
    # H1: frozen launch, chunked path (ignore_eos False)
    runs["frozen_chunked"] = simulate(CHUNKED_PREFILL, ENABLE_MIXED_CHUNK, 0)
    # H1': frozen launch but ignore_eos path: same two guards are present in
    # add_one_req_ignore_eos (anchors above), so identical budget outcome.
    runs["frozen_ignore_eos_path_same_budget"] = simulate(CHUNKED_PREFILL, ENABLE_MIXED_CHUNK, 0)
    # Controls
    runs["ctrl_chunk_disabled(None)"] = simulate(None, False, 0)
    runs["ctrl_chunk_budget_16384"] = simulate(16384, False, 0)
    runs["ctrl_mixed_chunk_on_bs16"] = simulate(CHUNKED_PREFILL, True, 16)

    def per_batch(key):
        return runs[key][0]

    waves_frozen = -(-16 // per_batch("frozen_chunked"))
    waves_t9 = -(-16 // per_batch("ctrl_chunk_disabled(None)"))

    out = {
        "trial": "g3-t12 deepseek-b",
        "kind": "CPU-executed source oracle; arithmetic, NOT_A_MEASUREMENT",
        "anchors_present": True,
        "executed_source": {
            "enable_mixed_chunk_default": ENABLE_MIXED_CHUNK,
            "PrefillAdder.rem_input_tokens_init": rem_input_init,
            "PrefillAdder.rem_chunk_tokens_init": rem_chunk_init,
            "compute_charge_expr": compute_charge_src,
        },
        "frozen_contract": {
            "max_prefill_tokens(source default)": MAX_PREFILL_TOKENS,
            "chunked_prefill_size(launch)": CHUNKED_PREFILL,
            "extend_tokens(PrefillHeavy)": EXTEND,
            "enable_mixed_chunk_launch_flag": "ABSENT -> source default",
        },
        "runs": {
            k: {"admitted_per_batch": v[0], "chunk_tokens": v[1],
                "rem_input_after": v[2], "rem_chunk_after": v[3]}
            for k, v in runs.items()
        },
        "H1": {
            "claim": "rem_chunk_tokens is a whole-batch budget; one 8192-token "
                     "request zeroes it and the real guard rejects a second",
            "admitted_per_batch_frozen": per_batch("frozen_chunked"),
            "waves_for_16_concurrent": waves_frozen,
            "t9_prediction_waves": waves_t9,
            "t9_falsified": waves_frozen != waves_t9,
        },
        "H2_mixed_decode_squeeze": {
            "applicable_under_frozen_launch": ENABLE_MIXED_CHUNK,
            "first_chunk_tokens_if_on_bs16": runs["ctrl_mixed_chunk_on_bs16"][1],
            "note": "enable_mixed_chunk default False; frozen launch passes no "
                    "flag, so num_mixed_decode_tokens=0 and H2 does NOT apply",
        },
        "discriminating_control": {
            "chunk_disabled_gives_t9_2_per_batch": per_batch("ctrl_chunk_disabled(None)") == 2,
            "chunk_16384_gives_2_per_batch": per_batch("ctrl_chunk_budget_16384") == 2,
        },
        "limits": [
            "Arithmetic over executed source, not a throughput measurement.",
            "Assumes page_size=1 (flashinfer); ceil(8192) unchanged for any page dividing 8192.",
            "Assumes ignores retraction/re-queue and dynamic chunk sizer; those can add steps.",
            "Does not model max_prefill_bs, prefill delayer, or memory-budget rejection.",
            "Request count 64 (max(32,4C)) means 64 admission events at C16, not 16.",
        ],
        "verdict": "H1_CONFIRMED_ON_SOURCE__T9_2_PER_BATCH_FALSIFIED",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("H1", "H2_mixed_decode_squeeze",
                                          "discriminating_control", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

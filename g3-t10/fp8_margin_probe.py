#!/usr/bin/env python3 -I
"""g3-t10: fp8-KV margin probe — exact request specs + t8-contract analyzer.

Subcommands
-----------
spec      Emit the exact, deterministic /generate request bodies for the
          diagnostic pair (REF: BF16, CAND: same commit + SGLANG_SPARK_FP8_KV=1)
          to g3-t10/raw/probe_request_specs.json.
analyze   python3 -I g3-t10/fp8_margin_probe.py analyze REF.json CAND.json OUT.json
          Apply the t8 margin-stratified contract (quality-budget-proposal.md):
            - no output-token flip on any pair            -> TOKEN_EXACT
            - every first-flip p: REF_margin(p) <= 2*eps_hat  -> PRECISION_CONSISTENT
            - any first-flip p: REF_margin(p)  > 2*eps_hat  -> NON_PRECISION_DEFECT
            - top-8 missing at a needed position           -> COVERAGE_INCONCLUSIVE (pair)
          eps_hat = max |dlogprob(chosen)| over same-prefix positions (pre-flip).
selftest  Golden fixtures with planted flips + negative controls; any failure
          => NO_CLAIM exit 2. Raw: g3-t10/raw/probe_selftest_output.json
"""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "g3-t10/raw"

# ---- C21-proven recipe (ngram_ab_diagnostic.py.txt) -------------------------
PROMPT_SEED = 1064 + 999  # fixed_prompt(): OPEN_SEED=1064, span seed +999
NEW_TOKENS = 48
TOP_K = 8
N_REPEAT_INSTANCES = 10
NATURAL_PROMPTS = [
    "The committee reviewed the proposal and concluded that",
    "In the winter of that year, the river froze earlier than",
    "A practical guide to debugging distributed systems begins with",
]


def repeat512_ids() -> list[int]:
    rng = random.Random(PROMPT_SEED)
    return [rng.randrange(4, 131072) for _ in range(8)] * 64


def build_requests() -> list[dict]:
    reqs = []
    ids = repeat512_ids()
    for i in range(N_REPEAT_INSTANCES):
        reqs.append({"label": f"repeat-512/instance{i}", "input_ids": ids})
    for j, text in enumerate(NATURAL_PROMPTS):
        reqs.append({"label": f"natural-{j}", "text": text})
    out = []
    for r in reqs:
        payload = {
            "sampling_params": {"max_new_tokens": NEW_TOKENS, "temperature": 0, "ignore_eos": True},
            "return_logprob": True,
            "top_logprobs_num": TOP_K,
        }
        payload["input_ids" if "input_ids" in r else "text"] = r["input_ids" if "input_ids" in r else "text"]
        out.append({"label": r["label"], "method": "POST", "path": "/generate", "payload": payload})
    return out


# ---- analyzer ---------------------------------------------------------------

def margin(row):
    """top1-top2 logprob margin from a top-8 row [[lp, id, text], ...]."""
    if not isinstance(row, list) or len(row) < 2:
        return None
    return float(row[0][0]) - float(row[1][0])


def chosen_lp(rec, i):
    c = rec["chosen_logprobs"][i]
    return float(c[0]) if isinstance(c, list) else float(c)


def analyze_pair(ref, cand):
    """Return dict verdict for one paired response (same label, same request)."""
    ids_r, ids_c = ref["output_ids"], cand["output_ids"]
    n = min(len(ids_r), len(ids_c))
    first_flip = next((i for i in range(n) if ids_r[i] != ids_c[i]), None)
    same_prefix = n if first_flip is None else first_flip

    eps_hat, lp_gaps = 0.0, []
    for i in range(same_prefix):
        d = abs(chosen_lp(ref, i) - chosen_lp(cand, i))
        lp_gaps.append(d)
        eps_hat = max(eps_hat, d)

    # Absolute pre-flip deviation band from E2's measured fp8-KV scale
    # (|dlogprob| mean 0.012-0.041 random / up to 0.153 natural). A candidate
    # whose own pre-flip deviation exceeds this is anomalous independent of
    # the (self-scaling) 2*eps_hat threshold -- NGRAM's 0.39-1.32 nat shows
    # why: a big deviation inflates its own threshold and can hide a defect.
    if eps_hat <= 0.153:
        band = "within_E2_max"
    elif eps_hat <= 0.306:
        band = "above_E2_max_within_2x"
    else:
        band = "ANOMALOUS_pre_flip_deviation"

    res = {
        "label": ref.get("label"),
        "positions_compared": n,
        "same_prefix_len": same_prefix,
        "first_flip_position": first_flip,
        "eps_hat_nat": eps_hat,
        "eps_hat_band": band,
        "n_flipped_positions": sum(1 for i in range(n) if ids_r[i] != ids_c[i]),
        "verdict": None,
        "detail": {},
    }
    if first_flip is None:
        res["verdict"] = "TOKEN_EXACT"
        return res

    m = margin(ref["top_logprobs"][first_flip])
    res["detail"]["ref_margin_at_first_flip"] = m
    res["detail"]["threshold_2eps"] = 2 * eps_hat
    if m is None:
        res["verdict"] = "COVERAGE_INCONCLUSIVE"
        res["detail"]["reason"] = "REF top-8 missing at first-flip position"
        return res
    res["verdict"] = (
        "PRECISION_CONSISTENT" if m <= 2 * eps_hat else "NON_PRECISION_DEFECT"
    )
    return res


def overall(ps):
    vs = [p["verdict"] for p in ps]
    if any(v == "NON_PRECISION_DEFECT" for v in vs):
        return "NON_PRECISION_DEFECT"
    if any(v == "COVERAGE_INCONCLUSIVE" for v in vs):
        return "COVERAGE_INCONCLUSIVE"
    # Band guard (g3-t10 refinement, executed on real C21 data): a flipping
    # pair whose own pre-flip deviation exceeds E2's measured fp8 scale
    # (>0.153 nat; NGRAM showed 0.39-1.32) inflates the self-scaling
    # threshold and can hide a defect inside its own deviation.
    if any(v == "PRECISION_CONSISTENT" and p.get("eps_hat_band") != "within_E2_max" for v, p in zip(vs, ps)):
        return "AMBIGUOUS_BAND"
    if any(v == "PRECISION_CONSISTENT" for v in vs):
        return "PRECISION_CONSISTENT"
    return "TOKEN_EXACT" if vs else "EMPTY"


def analyze(ref_path, cand_path, out_path):
    ref = json.loads(Path(ref_path).read_text())
    cand = json.loads(Path(cand_path).read_text())
    by = lambda doc: {r["label"]: r for r in doc["records"]}
    cr, cc = by(ref), by(cand)
    pairs = []
    for label in cr:
        if label not in cc:
            continue
        pairs.append(analyze_pair(cr[label], cc[label]))

    out = {
        "tool": "g3-t10/fp8_margin_probe.py analyze",
        "ref": {"role": ref.get("role"), "kv_expected": "bfloat16 (remap inactive)"},
        "cand": {"role": cand.get("role"), "kv_expected": "fp8_e4m3 (SGLANG_SPARK_FP8_KV=1)"},
        "contract": "t8 margin-stratified: flips allowed only where REF margin <= 2*eps_hat",
        "pairs": pairs,
        "overall": overall(pairs),
        "eps_band_warnings": [p["label"] for p in pairs if p.get("eps_hat_band") == "ANOMALOUS_pre_flip_deviation"],
        "interpretation": {
            "TOKEN_EXACT": "fp8-KV token-exact on these instances (P1 holds); consistent with margin floor 0.9375 nat",
            "PRECISION_CONSISTENT": "flips confined to noise-reachable margins (P2); precision-class, feed t8 contract",
            "NON_PRECISION_DEFECT": "flip at margin beyond fp8 reach => non-precision defect in the fp8 lineage (bug class); F2 reframes",
            "AMBIGUOUS_BAND": "self-scaling pass with pre-flip deviation above E2's fp8 scale (>0.153 nat): defect can hide in own deviation; not precision-class without further evidence",
            "COVERAGE_INCONCLUSIVE": "top-8 missing where needed; no claim",
        },
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"overall={out['overall']} pairs={len(pairs)}")
    for p in pairs:
        print(f"  {p['label']}: {p['verdict']} flip@{p['first_flip_position']} eps_hat={p['eps_hat_nat']:.6f}")
    return 0


# ---- selftest ---------------------------------------------------------------

def _rec(label, ids, lps, tops):
    return {"label": label, "output_ids": ids,
            "chosen_logprobs": [[lp, tid, None] for lp, tid in zip(lps, ids)],
            "top_logprobs": tops}


def _tops(margins, base=-1.0, tid=100):
    """Build top-8 rows whose top1-top2 margin sequence is `margins`."""
    return [[[base, tid + i, None], [base - m, tid + 500 + i, None]] for i, m in enumerate(margins)]


def selftest() -> int:
    margins = [3.0, 2.5, 0.10, 2.0, 4.0]  # position 2 is the only noise-reachable one
    tops = _tops(margins)
    ids = [11, 12, 13, 14, 15]
    lps = [-1.0, -1.1, -1.2, -1.3, -1.4]

    def cand(ids2, lps2, tops2=None):
        return _rec("t", ids2, lps2, tops2 if tops2 is not None else tops)

    ref = _rec("t", ids, lps, tops)
    cases = []

    # 1. identical -> TOKEN_EXACT, eps_hat 0
    r = analyze_pair(ref, cand(list(ids), list(lps)))
    cases.append(("identical->TOKEN_EXACT", r["verdict"] == "TOKEN_EXACT" and r["eps_hat_nat"] == 0.0, r["verdict"]))

    # 2. flip at the low-margin position 2, eps_hat 0.08 (0.10 <= 0.16) -> PRECISION_CONSISTENT
    c_ids = list(ids); c_ids[2] = 999
    c_lps = list(lps); c_lps[0] += 0.05; c_lps[1] -= 0.08  # same-prefix eps_hat = 0.08
    r = analyze_pair(ref, cand(c_ids, c_lps))
    ok = r["verdict"] == "PRECISION_CONSISTENT" and r["first_flip_position"] == 2
    ok = ok and abs(r["detail"]["ref_margin_at_first_flip"] - 0.10) < 1e-9 and abs(r["detail"]["threshold_2eps"] - 0.16) < 1e-12
    cases.append(("low-margin flip->PRECISION_CONSISTENT", ok, r["verdict"]))

    # 3. cascade after first flip does not change the verdict (F-d: divergence permanent)
    c_ids3 = list(c_ids); c_ids3[3] = 777; c_ids3[4] = 888
    r = analyze_pair(ref, cand(c_ids3, c_lps))
    cases.append(("cascade ignored", r["verdict"] == "PRECISION_CONSISTENT" and r["n_flipped_positions"] == 3, r["verdict"]))

    # 4. flip at high-margin position 1 (2.5 > 2*0.08) -> NON_PRECISION_DEFECT
    c_ids4 = list(ids); c_ids4[1] = 999
    r = analyze_pair(ref, cand(c_ids4, c_lps))
    ok = r["verdict"] == "NON_PRECISION_DEFECT" and r["first_flip_position"] == 1
    cases.append(("high-margin flip->NON_PRECISION_DEFECT", ok, r["verdict"]))

    # 5. missing REF top-8 at flip position -> COVERAGE_INCONCLUSIVE
    tops5 = [list(x) for x in tops]; tops5[2] = None
    ref5 = _rec("t", ids, lps, tops5)
    r = analyze_pair(ref5, cand(c_ids, c_lps))
    cases.append(("null top8->COVERAGE_INCONCLUSIVE", r["verdict"] == "COVERAGE_INCONCLUSIVE", r["verdict"]))

    # 6. eps_hat only from SAME-PREFIX positions: post-flip logprob gap must not count
    c_lps6 = list(c_lps); c_lps6[2] -= 5.0; c_lps6[3] -= 5.0  # after flip: huge gaps
    r = analyze_pair(ref, cand(c_ids, c_lps6))
    ok = r["verdict"] == "PRECISION_CONSISTENT" and abs(r["eps_hat_nat"] - 0.08) < 1e-12
    cases.append(("eps_hat pre-flip only", ok, f"{r['verdict']} eps={r['eps_hat_nat']}"))

    # 7. band guard: a self-scaling pass with eps_hat above E2 max downgrades
    #    the OVERALL verdict to AMBIGUOUS_BAND (NGRAM B/c4same pattern)
    def _pair(verdict, eps):
        return {"verdict": verdict, "eps_hat_band": ("within_E2_max" if eps <= 0.153 else "ANOMALOUS_pre_flip_deviation")}
    o1 = overall([_pair("PRECISION_CONSISTENT", 0.04)]) == "PRECISION_CONSISTENT"
    o2 = overall([_pair("PRECISION_CONSISTENT", 0.85)]) == "AMBIGUOUS_BAND"
    o3 = overall([_pair("PRECISION_CONSISTENT", 0.85), _pair("NON_PRECISION_DEFECT", 0.04)]) == "NON_PRECISION_DEFECT"
    o4 = overall([_pair("TOKEN_EXACT", 0.0)]) == "TOKEN_EXACT"
    cases.append(("overall band guard", o1 and o2 and o3 and o4, f"{o1}/{o2}/{o3}/{o4}"))

    ok_all = all(c[1] for c in cases)
    out = {
        "tool": "g3-t10/fp8_margin_probe.py selftest",
        "verdict": "PASS" if ok_all else "NO_CLAIM",
        "cases": [{"name": n, "passed": p, "got": g} for n, p, g in cases],
        "limits": ["analyzer logic verified on synthetic fixtures; GPU numerics and HTTP transport untested here"],
    }
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "probe_selftest_output.json").write_text(json.dumps(out, indent=1))
    for n, p, g in cases:
        print(f"  {'PASS' if p else 'FAIL'}  {n}  [{g}]")
    print(f"selftest verdict={out['verdict']}")
    return 0 if ok_all else 2


def spec() -> int:
    reqs = build_requests()
    ids = repeat512_ids()
    out = {
        "tool": "g3-t10/fp8_margin_probe.py spec",
        "recipe_provenance": "C21 ngram_ab_diagnostic.py.txt fixed_prompt + /generate payload shape (complete top-8 observed)",
        "prompt": {
            "name": "repeat-512",
            "construction": "random.Random(1064+999); [randrange(4,131072) for _ in range(8)]*64",
            "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "n_tokens": len(ids),
            "identical_to_C21_off_records": "A/repeat-512/c1same family",
        },
        "requests_per_server": reqs,
        "read_fields": {
            "output_top_logprobs": "meta_info.output_top_logprobs (rows [[lp, id, text]*8])",
            "chosen_logprobs": "meta_info.output_logprobs",
            "output_ids": "output_ids (fallback meta_info.output_ids)",
        },
        "pairing": "identical payloads to both servers; pair by label; greedy temperature=0",
        "expected_counts": {"repeat512_instances": N_REPEAT_INSTANCES, "natural_instances": len(NATURAL_PROMPTS),
                            "positions_per_instance": NEW_TOKENS, "paired_positions_total": (N_REPEAT_INSTANCES + len(NATURAL_PROMPTS)) * NEW_TOKENS},
    }
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "probe_request_specs.json").write_text(json.dumps(out, indent=1))
    print(f"requests_per_server={len(reqs)} paired_positions={out['expected_counts']['paired_positions_total']}")
    print(f"repeat512 ids sha256={out['prompt']['input_ids_sha256'][:16]} n={len(ids)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "spec":
        sys.exit(spec())
    if cmd == "analyze":
        sys.exit(analyze(sys.argv[2], sys.argv[3], sys.argv[4]))
    if cmd == "selftest":
        sys.exit(selftest())
    print(__doc__)
    sys.exit(3)

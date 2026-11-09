#!/usr/bin/env python3
"""G3 trial 14 follow-up (deepseek-a): if ON's top-8 at the divergence position
merely comes from a different context row (logit-row misassignment), it should
match OFF's reported distribution at some *other* position q on the true greedy
trajectory. If no q matches, the verifier's hidden state itself differs.

Inputs: the same real packet + the derived g3-t14/raw/packet_analysis.json.
Run: python3 -I align_check.py <packet.json> <derived.json> --out <json>
Fails closed on precondition violations. Read-only on inputs.
"""
import argparse
import json
import sys


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def top8(rec, pos):
    out = [(float(r[0]), int(r[1])) for r in rec["top_logprobs"][pos]
           if r[0] is not None and r[1] is not None]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def lp_of(t8, tid):
    for lp, x in t8:
        if x == tid:
            return lp
    return None


def compare(a, b):
    ia, ib = [x for _, x in a], [x for _, x in b]
    common = set(ia) & set(ib)
    diffs = [abs(lp_of(a, t) - lp_of(b, t)) for t in common]
    return {
        "argmax_match": a[0][1] == b[0][1],
        "n_common": len(common),
        "max_abs_lp_diff_common": max(diffs) if diffs else None,
        "mean_abs_lp_diff_common": (sum(diffs) / len(diffs)) if diffs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packet")
    ap.add_argument("derived")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = load(args.packet)
    der = load(args.derived)
    recs = {r["request_uuid"]: r for r in d["records"]}
    if len(der["replay_cases"]) != len(d["replay"]):
        raise AssertionError("derived/packet replay case count mismatch")

    result = {"cases": [], "pre_divergence_histogram": {}, "limits": der["limits"]}

    # --- Follow-up 1: trajectory alignment search -------------------------
    for case, dc in zip(d["replay"], der["replay_cases"]):
        p = dc["first_divergence"]
        on = recs[case["on_request_uuid"]]
        off = recs[case["off_request_uuid"]]
        on_t8 = top8(on, p)
        matches = []
        n_off = len(off["output_ids"])
        for q in range(n_off):
            m = compare(on_t8, top8(off, q))
            matches.append({"q": q, "q_token": off["output_ids"][q], "offset_vs_p": q - p, **m})
        # Rank matches by argmax then mean common-LP distance (lower better).
        ranked = sorted(
            [m for m in matches if m["mean_abs_lp_diff_common"] is not None],
            key=lambda m: (not m["argmax_match"], m["mean_abs_lp_diff_common"]),
        )
        at_p = next(m for m in matches if m["q"] == p)
        # Control: OFF primary response row at p vs the OFF one-token replays.
        off_primary_t8 = top8(off, p)
        off_replay_t8 = [top8(case["responses"][0], 0)]
        off_ctrl = compare(off_primary_t8, off_replay_t8[0])
        off_ctrl["bit_identical"] = (
            [[lp, t] for lp, t in off_primary_t8]
            == [[lp, t] for lp, t in off_replay_t8[0]]
        )
        on_t1 = on_t8[0]
        off_t1 = off_primary_t8[0]
        # Is ON's committed token the argmax at any other OFF position?
        on_tok = dc["on_committed_token"]
        tok_argmax_at = [m["q"] for m in matches
                         if top8(off, m["q"])[0][1] == on_tok]
        result["cases"].append({
            "case": dc["case"], "p": p, "on_committed_token": on_tok,
            "on_top8": [[lp, t] for lp, t in on_t8],
            "off_top8_at_p": [[lp, t] for lp, t in top8(off, p)],
            "on_vs_off_at_p": at_p,
            "best_off_positions_by_match": ranked[:5],
            "n_off_positions_where_on_token_is_argmax": len(tok_argmax_at),
            "off_positions_where_on_token_is_argmax": tok_argmax_at,
            "on_token_in_any_off_top8_position": [
                m["q"] for m in matches if lp_of(top8(off, m["q"]), on_tok) is not None
            ],
            "off_primary_vs_off_replay_at_p": off_ctrl,
            "on_reported_top1_lp": on_t1[0],
            "off_primary_top1_lp": off_t1[0],
            "on_chosen_lp": dc["on_reported_lp_of_committed"],
        })

    # --- Follow-up 2: pre-divergence LP-diff distribution -----------------
    rows = der["pre_divergence_control"]["rows"]
    bins = {"le_0.05": 0, "0.05_0.5": 0, "0.5_2": 0, "gt_2": 0}
    for r in rows:
        v = r["max_abs_lp_diff_common"]
        if v is None:
            continue
        if v <= 0.05:
            bins["le_0.05"] += 1
        elif v <= 0.5:
            bins["0.05_0.5"] += 1
        elif v <= 2.0:
            bins["0.5_2"] += 1
        else:
            bins["gt_2"] += 1
    result["pre_divergence_histogram"] = {
        "n_positions": len(rows),
        "argmax_match_fraction": (sum(1 for r in rows if r["argmax_match"]) / len(rows)) if rows else None,
        "bins_max_abs_lp_diff_common": bins,
        "max_chosen_lp_diff": max(r["chosen_lp_diff"] for r in rows) if rows else None,
        "n_chosen_lp_diff_gt_0.5": sum(1 for r in rows if r["chosen_lp_diff"] > 0.5),
        "n_off_chosen_not_own_argmax": sum(1 for r in rows if not r["off_chosen_is_own_argmax"]),
        "n_on_chosen_not_own_argmax": sum(1 for r in rows if not r["on_chosen_is_own_argmax"]),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    for c in result["cases"]:
        ap_ = c["on_vs_off_at_p"]
        print("case%d p=%d best_off_q=%s argmax=%s meandiff=%.4g onTokARGMAXat=%s inTop8at=%s at_p(am=%s,mean=%.4g,max=%.4g)"
              % (c["case"], c["p"], [m["q"] for m in c["best_off_positions_by_match"]],
                 c["best_off_positions_by_match"][0]["argmax_match"],
                 c["best_off_positions_by_match"][0]["mean_abs_lp_diff_common"],
                 c["off_positions_where_on_token_is_argmax"],
                 c["on_token_in_any_off_top8_position"][:8],
                 ap_["argmax_match"], ap_["mean_abs_lp_diff_common"] or 0.0,
                 ap_["max_abs_lp_diff_common"] or 0.0))
    print("pre_div", json.dumps(result["pre_divergence_histogram"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

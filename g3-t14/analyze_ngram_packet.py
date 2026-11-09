#!/usr/bin/env python3
"""G3 trial 14 (deepseek-a): matched-prefix NGRAM ON/OFF distribution test.

Real Controller packet: controller_evidence/continuous-20261109-01/
20261110-runtime/ngram-full-observations.json (ON b9a91aa13, OFF 999d4b24,
BF16/FlashInfer, repeat-512, 20 primary + 16 one-token replay HTTP 200).

Question: at the first ON/OFF divergence position p, ON's original response
and the 4 OFF replays share the exact same 512-token prompt + p committed
tokens. If their reported top-8 distributions at p agree within the OFF x4
reproducibility noise, the verifier scored the same distribution and the bug
is commit/accept selection. If they differ materially, the verifier scored a
different hidden state for the same visible prefix.

Run: python3 -I analyze_ngram_packet.py <packet.json> --out <derived.json>
Fails closed (exit 2) on precondition violations. Read-only on the packet.
"""
import argparse
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top8_at(rec, pos):
    """Return [(lp, id)] sorted desc for output position pos."""
    rows = rec["top_logprobs"][pos]
    out = []
    for row in rows:
        lp, tid = row[0], row[1]
        if lp is None or tid is None:
            continue
        out.append((float(lp), int(tid)))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def chosen_at(rec, pos):
    row = rec["chosen_logprobs"][pos]
    return float(row[0]), int(row[1])


def rank_of(t8, tid):
    for i, (_, x) in enumerate(t8):
        if x == tid:
            return i
    return None


def lp_of(t8, tid):
    for lp, x in t8:
        if x == tid:
            return lp
    return None


def set_metrics(a, b):
    """Compare two sorted [(lp,id)] lists. b is reference."""
    ia = [x for _, x in a]
    ib = [x for _, x in b]
    common = sorted(set(ia) & set(ib))
    diffs = [abs(lp_of(a, t) - lp_of(b, t)) for t in common]
    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_common": len(common),
        "common_ids": common,
        "argmax_a": ia[0] if ia else None,
        "argmax_b": ib[0] if ib else None,
        "argmax_match": bool(ia and ib and ia[0] == ib[0]),
        "jaccard": (len(common) / len(set(ia) | set(ib))) if (ia or ib) else None,
        "max_abs_lp_diff_common": max(diffs) if diffs else None,
        "lps_a": {str(t): lp_of(a, t) for t in common},
        "lps_b": {str(t): lp_of(b, t) for t in common},
    }


def replay_case(case, recs, idx):
    p = int(case["first_divergence"])
    on = recs[case["on_request_uuid"]]
    off = recs[case["off_request_uuid"]]
    prefix = case["prefix"]
    out = {"case": idx, "first_divergence": p, "on_token": int(case["on_token"])}

    # Precondition: prefix tail must be the ON record's own first p outputs,
    # and the OFF primary record must share those same p tokens.
    pre_ok = (
        prefix[512:] == on["output_ids"][:p]
        and off["output_ids"][:p] == on["output_ids"][:p]
    )
    out["precondition_prefix_matches_both_records"] = pre_ok
    if not pre_ok:
        raise AssertionError(f"case {idx}: prefix does not align with records")

    on_t8 = top8_at(on, p)
    on_lp, on_tok = chosen_at(on, p)
    out["on_committed_token"] = on_tok
    out["on_reported_lp_of_committed"] = on_lp
    out["on_top8"] = [[lp, tid] for lp, tid in on_t8]
    out["on_own_constency"] = {
        "committed_is_own_argmax": on_t8[0][1] == on_tok,
        "committed_rank_in_own_top8": rank_of(on_t8, on_tok),
        "own_argmax_token": on_t8[0][1],
        "own_argmax_lp": on_t8[0][0],
        "own_lp_gap_committed_vs_argmax": on_t8[0][0] - on_lp,
        "committed_present_in_own_top8": rank_of(on_t8, on_tok) is not None,
    }

    # OFF x4 same-prefix one-token replays.
    replays = []
    for r in case["responses"]:
        if r.get("http_status") != 200 or r["role"] != "OFF":
            raise AssertionError(f"case {idx}: non-200/non-OFF replay")
        if len(r["output_ids"]) != 1:
            raise AssertionError(f"case {idx}: replay is not one token")
        t8 = top8_at(r, 0)
        lp, tok = chosen_at(r, 0)
        replays.append({"token": tok, "lp": lp, "top8": t8})
    out["n_off_replays"] = len(replays)
    out["off_replay_argmax_tokens"] = [x["top8"][0][1] for x in replays]
    out["off_replay_top1_lps"] = [x["top8"][0][0] for x in replays]
    out["off_argmax_stable"] = len(set(out["off_replay_argmax_tokens"])) == 1

    # OFF noise floor: per-token LP spread across the 4 repeats.
    union = {}
    for x in replays:
        for lp, tid in x["top8"]:
            union.setdefault(tid, []).append(lp)
    spreads = {
        str(t): max(v) - min(v)
        for t, v in union.items() if len(v) >= 2
    }
    out["off_noise"] = {
        "n_tokens_in_union": len(union),
        "tokens_seen_in_all_4": sum(1 for v in union.values() if len(v) == 4),
        "max_same_token_lp_spread": max(spreads.values()) if spreads else None,
        "aggregate_raw_top8_set_identical_all4": all(
            [tid for _, tid in x["top8"]] == [tid for _, tid in replays[0]["top8"]]
            for x in replays
        ),
    }

    # Signal: ON top-8 vs each OFF replay top-8 at the same prefix.
    off_ref = replays[0]["top8"]
    out["on_vs_off_ref"] = set_metrics(on_t8, off_ref)
    out["on_vs_each_off_max_abs_lp_diff"] = [
        set_metrics(on_t8, x["top8"])["max_abs_lp_diff_common"] for x in replays
    ]
    # ON committed token under OFF's reported distribution.
    off_tok_lps = [lp_of(x["top8"], on_tok) for x in replays]
    out["on_token_under_off"] = {
        "present_in_off_top8_each": [v is not None for v in off_tok_lps],
        "lp_each": off_tok_lps,
        "off_argmax_lp_each": [x["top8"][0][0] for x in replays],
        "gap_to_off_argmax_each": (
            [x["top8"][0][0] - v if v is not None else None
             for x, v in zip(replays, off_tok_lps)]
        ),
        "on_lp_minus_off_lp": (
            [on_lp - v if v is not None else None for v in off_tok_lps]
        ),
    }
    return out


def primary_pre_divergence(recs, pairs):
    """Matched-context control: ON vs OFF primary rows at positions before the
    first ON/OFF divergence (identical committed tokens on both sides)."""
    rows = []
    for on, off, p in pairs:
        n = min(len(on["output_ids"]), len(off["output_ids"]))
        for pos in range(min(p, n)):
            if on["output_ids"][pos] != off["output_ids"][pos]:
                raise AssertionError("pre-divergence token mismatch")
            a = top8_at(on, pos)
            b = top8_at(off, pos)
            m = set_metrics(a, b)
            la, ta = chosen_at(on, pos)
            lb, tb = chosen_at(off, pos)
            rows.append({
                "label": on["label"], "instance": on["instance"], "pos": pos,
                "argmax_match": m["argmax_match"],
                "n_common": m["n_common"],
                "max_abs_lp_diff_common": m["max_abs_lp_diff_common"],
                "chosen_lp_diff": abs(la - lb),
                "on_chosen_lp": la,
                "off_chosen_lp": lb,
                "on_chosen_is_own_argmax": a[0][1] == ta,
                "off_chosen_is_own_argmax": b[0][1] == tb,
            })
    diffs = [r["max_abs_lp_diff_common"] for r in rows
             if r["max_abs_lp_diff_common"] is not None]
    return {
        "n_positions": len(rows),
        "n_argmax_match": sum(1 for r in rows if r["argmax_match"]),
        "n_both_chosen_own_argmax": sum(
            1 for r in rows if r["on_chosen_is_own_argmax"] and r["off_chosen_is_own_argmax"]
        ),
        "max_abs_lp_diff_common": max(diffs) if diffs else None,
        "median_abs_lp_diff_common": sorted(diffs)[len(diffs) // 2] if diffs else None,
        "max_chosen_lp_diff": max(r["chosen_lp_diff"] for r in rows) if rows else None,
        "rows": rows,
    }


def identical_output_scan(records):
    """Same output IDs -> are reported chosen LPs reproducible? (metadata vs
    numerics). Groups primary responses by role+label."""
    groups = {}
    for r in records:
        groups.setdefault((r["role"], r["label"]), []).append(r)
    out = []
    for (role, label), rs in sorted(groups.items()):
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["output_ids"] != b["output_ids"]:
                    continue
                n = len(a["output_ids"])
                lp_diffs = [abs(chosen_at(a, k)[0] - chosen_at(b, k)[0]) for k in range(n)]
                top_argmax_same = sum(
                    1 for k in range(n)
                    if top8_at(a, k)[0][1] == top8_at(b, k)[0][1]
                )
                out.append({
                    "role": role, "label": label,
                    "instances": [a["instance"], b["instance"]],
                    "n_tokens": n,
                    "max_chosen_lp_diff": max(lp_diffs),
                    "n_positions_lp_diff_gt_1nat": sum(1 for d in lp_diffs if d > 1.0),
                    "n_positions_top1_argmax_same": top_argmax_same,
                })
    return out


def row_consistency(records):
    """Precondition: every chosen token must appear in its own top-8 row with
    the reported chosen LP (C24's 'chosen LP equals its reported row')."""
    n = mismatch = not_argmax = missing = 0
    for rec in records:
        for k in range(len(rec["output_ids"])):
            lp = float(rec["chosen_logprobs"][k][0])
            tid = int(rec["chosen_logprobs"][k][1])
            rows = top8_at(rec, k)
            n += 1
            hit = [x for x in rows if x[1] == tid]
            if not hit:
                missing += 1
                continue
            if abs(hit[0][0] - lp) > 1e-6:
                mismatch += 1
            if rows[0][1] != tid:
                not_argmax += 1
    return {"positions": n, "chosen_lp_vs_row_mismatch": mismatch,
            "chosen_not_top8_argmax": not_argmax, "chosen_missing_from_top8": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = load(args.packet)
    if d.get("status") != "SUCCEEDED":
        raise AssertionError("packet status not SUCCEEDED")
    if d["roles"]["ON"]["resolved_speculative_algorithm"] != "NGRAM":
        raise AssertionError("ON role did not resolve to NGRAM")
    if d["roles"]["OFF"]["resolved_speculative_algorithm"] is not None:
        raise AssertionError("OFF role resolved a speculative algorithm")
    if len(d["prompt"]["input_ids"]) != 512:
        raise AssertionError("packet prompt is not the frozen 512-token prompt")
    recs = {r["request_uuid"]: r for r in d["records"]}
    for r in d["records"]:
        if r.get("http_status") != 200:
            raise AssertionError("primary record non-200")

    all_recs = list(d["records"]) + [r for c in d["replay"] for r in c["responses"]]
    consistency = row_consistency(all_recs)
    if consistency["chosen_lp_vs_row_mismatch"] or consistency["chosen_missing_from_top8"]:
        raise AssertionError("chosen logprob is not its own reported top-8 row")
    cases = [replay_case(c, recs, i) for i, c in enumerate(d["replay"])]

    # Primary pairs: same label+instance, one ON one OFF.
    keyed = {}
    for r in d["records"]:
        keyed.setdefault((r["label"], r["instance"]), {})[r["role"]] = r
    pairs, pair_meta = [], []
    for (label, inst), m in sorted(keyed.items()):
        if "ON" in m and "OFF" in m:
            on, off = m["ON"], m["OFF"]
            p = 0
            while (p < min(len(on["output_ids"]), len(off["output_ids"]))
                   and on["output_ids"][p] == off["output_ids"][p]):
                p += 1
            pairs.append((on, off, p))
            pair_meta.append({
                "label": label, "instance": inst, "first_divergence": p,
                "on_tokens": len(on["output_ids"]), "off_tokens": len(off["output_ids"]),
                "diverges": p < min(len(on["output_ids"]), len(off["output_ids"])),
                "on_token_at_p": on["output_ids"][p] if p < len(on["output_ids"]) else None,
                "off_token_at_p": off["output_ids"][p] if p < len(off["output_ids"]) else None,
            })

    result = {
        "packet": args.packet,
        "on_commit": d["roles"]["ON"]["commit"],
        "off_commit": d["roles"]["OFF"]["commit"],
        "actual_pair_uuid": d.get("actual_pair_uuid"),
        "n_primary": len(d["records"]),
        "primary_pairs": pair_meta,
        "n_pairs_diverged": sum(1 for x in pair_meta if x["diverges"]),
        "pre_divergence_control": primary_pre_divergence(recs, pairs),
        "replay_cases": cases,
        "row_consistency": consistency,
        "identical_output_scan": identical_output_scan(d["records"]),
        "limits": (
            "Response-level real packet only: 4 replay cases / 3 unique prefixes; "
            "replay probes are max_tokens=1. LP values are BF16 engine reports, not "
            "recomputed logits. No live accept-len=1, mask, KV-mover or overlap "
            "observation; nothing here is a gate, speed, numeric or promotion claim."
        ),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    # Bounded stdout summary only.
    print("VERDICT_INPUTS",
          "pairs_diverged=%d/%d" % (result["n_pairs_diverged"], len(pair_meta)),
          "pre_div_rows=%d" % result["pre_divergence_control"]["n_positions"],
          "pre_div_max_lp=%.6g" % (result["pre_divergence_control"]["max_abs_lp_diff_common"] or 0.0))
    for c in cases:
        oo = c["on_own_constency"]
        print("case%d p=%d on_tok=%d own_argmax=%s rank=%s owngap=%.6g off_am=%s stable=%s offnoise=%.6g "
              "onVoff_maxdiff=%s onTok_in_off=%s gap_to_off_am=%s"
              % (c["case"], c["first_divergence"], c["on_committed_token"],
                 oo["committed_is_own_argmax"], oo["committed_rank_in_own_top8"],
                 oo["own_lp_gap_committed_vs_argmax"], c["off_replay_argmax_tokens"][0],
                 c["off_argmax_stable"], c["off_noise"]["max_same_token_lp_spread"] or 0.0,
                 c["on_vs_off_ref"]["max_abs_lp_diff_common"],
                 c["on_token_under_off"]["present_in_off_top8_each"],
                 c["on_token_under_off"]["gap_to_off_argmax_each"]))
    for x in result["identical_output_scan"]:
        print("identical_output", x["role"], x["label"], x["instances"],
              "maxLPdiff=%.6g" % x["max_chosen_lp_diff"],
              "n>1nat=%d" % x["n_positions_lp_diff_gt_1nat"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""G3 glm-b trial-10: derived analysis of controller C21 NGRAM ON/OFF pairs.

Read-only input: controller_evidence/continuous-20261109-01/20261110-runtime/
  ngram-full-observations.json (authoritative parsed records + replays)
  ngram-analysis.json (controller summary, used only for cross-check)
Output: g3-t10/raw/ngram_pair_analysis_output.json + stdout summary.

Questions (see plan.md):
  Q1 recipe closure  : every k=8 response has populated output_top_logprobs (8 rows/pos)
  Q2 error fingerprint: divergence structure, pre-divergence LP drift, self-top1 audit
  Q3 real performance : paired ON vs OFF server-measured decode throughput

Hard rule: any harness error -> exit 2, verdict NO_CLAIM (i7/i9 lessons).
No packet file is modified. Comparison core is role-agnostic: the same ops on
(reference, candidate) response sets form glm-b's pending numeric comparison.
"""
import json, math, os, sys, statistics, hashlib

ROOT = "/workspace"
PKT = os.path.join(ROOT, "controller_evidence/continuous-20261109-01/20261110-runtime")
OUT = os.path.join(ROOT, "g3-t10/raw/ngram_pair_analysis_output.json")
errors = []

def load(p):
    with open(p) as f:
        return json.load(f)

def rows_of(rec):
    """Return (output_ids, chosen_lps, top_rows) from a record's parsed response."""
    mi = rec["response"]["meta_info"]
    out_ids = rec.get("output_ids")
    chosen = [r[0] if r else None for r in mi.get("output_token_logprobs") or []]
    tops = mi.get("output_top_logprobs")
    return out_ids, chosen, tops

def check(cond, msg):
    if not cond:
        errors.append(msg)
    return cond

def main():
    full = load(os.path.join(PKT, "ngram-full-observations.json"))
    summ = load(os.path.join(PKT, "ngram-analysis.json"))

    records = full["records"]
    replay_cases = full["replay"]
    all_recs = list(records)
    for c in replay_cases:
        all_recs.extend(c["responses"])

    # ---------------- Q1: recipe closure over every response ----------------
    q1 = {"responses_total": len(all_recs), "populated": 0, "violations": []}
    for r in all_recs:
        mi = r["response"]["meta_info"]
        n = mi.get("completion_tokens")
        tops = mi.get("output_top_logprobs")
        ok = isinstance(tops, list) and len(tops) == n and all(
            isinstance(row, list) and len(row) == 8 for row in tops)
        if ok:
            q1["populated"] += 1
        else:
            q1["violations"].append({"request_uuid": r.get("request_uuid"),
                                     "label": r.get("label"), "completion": n,
                                     "tops_type": type(tops).__name__})
    q1["verdict"] = "RECIPE_POPULATED_ALL" if not q1["violations"] else "RECIPE_VIOLATIONS"

    # ---------------- pair the 20 primaries by (label, instance) ----------------
    by_key = {}
    for r in records:
        by_key[(r["label"], r["instance"], r["role"])] = r
    pair_keys = sorted({(l, i) for (l, i, _) in by_key})
    check(len(pair_keys) == 10, f"expected 10 primary pairs, got {len(pair_keys)}")

    # ---------------- Q2: divergence fingerprint ----------------
    q2_pairs = []
    for k in pair_keys:
        on, off = by_key.get((k[0], k[1], "ON")), by_key.get((k[0], k[1], "OFF"))
        if on is None or off is None:
            errors.append(f"pair {k}: missing role record")
            continue
        on_ids, on_lp, on_tops = rows_of(on)
        off_ids, off_lp, off_tops = rows_of(off)
        n = min(len(on_ids), len(off_ids))
        check(len(on_ids) == len(off_ids) == 48, f"{k}: lens {len(on_ids)}/{len(off_ids)}")

        d = next((i for i in range(n) if on_ids[i] != off_ids[i]), None)
        pre = {"max_abs_lp_delta": None, "positions": d}
        if d is not None and d > 0:
            deltas = [abs(on_lp[i] - off_lp[i]) for i in range(d)
                      if on_lp[i] is not None and off_lp[i] is not None]
            pre["max_abs_lp_delta"] = max(deltas) if deltas else None
            pre["all_bitwise_equal"] = all(x == 0.0 for x in deltas) if deltas else None

        def top1(rowset, i):
            return max(rowset[i], key=lambda r: r[0])

        ev = {"key": [k[0], k[1]], "first_divergence": d}
        if d is not None:
            onrow = {r[1]: r[0] for r in on_tops[d]}
            offrow = {r[1]: r[0] for r in off_tops[d]}
            t1_on, t1_off = top1(on_tops, d), top1(off_tops, d)
            ev["at_divergence"] = {
                "on_chosen": on_ids[d], "off_chosen": off_ids[d],
                "on_chosen_lp": on_lp[d], "off_chosen_lp": off_lp[d],
                "lp_gap_nat": on_lp[d] - off_lp[d],
                "on_top1": t1_on[1], "on_top1_is_chosen": t1_on[1] == on_ids[d],
                "off_top1": t1_off[1], "off_top1_is_chosen": t1_off[1] == off_ids[d],
                "off_token_rank_in_on_top8": (sorted(onrow, key=onrow.get, reverse=True).index(off_ids[d])
                                              if off_ids[d] in onrow else None),
                "off_token_lp_under_on": onrow.get(off_ids[d]),
                "on_token_lp_under_off": offrow.get(on_ids[d]),
                "off_top1_margin_nat": t1_off[0] - max(r[0] for r in off_tops[d] if r[1] != t1_off[1]),
                "on_top1_margin_nat": t1_on[0] - max(r[0] for r in on_tops[d] if r[1] != t1_on[1]),
            }
            recon = sum(1 for i in range(d + 1, n) if on_ids[i] == off_ids[i])
            ev["post_divergence"] = {"positions": n - d - 1, "equal_positions": recon,
                                     "reconverged_run_max": max_run_equal(on_ids, off_ids, d, n)}
        ev["pre_divergence"] = pre

        # alignment test: is the pre-divergence LP delta a recording offset?
        # compare ON chosen-LP at i against OFF at i+k for k in -9..9; a period-8
        # prompt would camouflage an +-8 shift, so include it explicitly.
        off_tops_r = off_tops
        mae_by_k = {}
        for kk in range(-9, 10):
            idx = [(i, i + kk) for i in range(d) if 0 <= i + kk < n] if d is not None else []
            if len(idx) >= 3:
                mae_by_k[kk] = sum(abs(on_lp[a] - off_lp[b]) for a, b in idx) / len(idx)
        best_k = min(mae_by_k, key=mae_by_k.get) if mae_by_k else None
        set_ov = [len({r[1] for r in on_tops[i]} & {r[1] for r in off_tops_r[i]}) / 8
                  for i in range(d)] if d else []
        ev["alignment"] = {
            "best_shift_k": best_k,
            "mae_at_best": mae_by_k.get(best_k),
            "mae_at_k0": mae_by_k.get(0),
            "mean_top8_set_overlap_pre_div": sum(set_ov) / len(set_ov) if set_ov else None,
        }
        # per-position chosen-LP delta: where does distortion begin?
        win = range(d) if d is not None else range(0)
        ev["position_lp_abs_delta"] = [abs(on_lp[i] - off_lp[i]) for i in win]
        ev["bitwise_equal_lp0"] = bool(d is not None and d > 0 and on_lp[0] == off_lp[0])
        q2_pairs.append(ev)

    # cross-check against controller-computed first_divergence
    ctrl = {(d["key"][0], d["key"][1]): d["first_divergence"]
            for d in full["comparison"]["differences"]}
    mism = [e["key"] for e in q2_pairs
            if ctrl.get((e["key"][0], e["key"][1])) != e["first_divergence"]]

    # self-top1 audit over every response/position
    audit = {"ON": [0, 0], "OFF": [0, 0]}  # role -> [positions, mismatches]
    audit_cases = []
    for r in all_recs:
        ids, lps, tops = rows_of(r)
        role = r["role"]
        if role not in audit:
            audit[role] = [0, 0]
        for i in range(min(len(ids), len(tops))):
            audit[role][0] += 1
            t1 = max(tops[i], key=lambda x: x[0])
            if t1[1] != ids[i]:
                audit[role][1] += 1
                audit_cases.append({"role": role, "request_uuid": r["request_uuid"],
                                    "label": r.get("label"), "instance": r.get("instance"),
                                    "pos": i, "chosen": ids[i], "chosen_lp": lps[i],
                                    "top1": t1[1], "top1_lp": t1[0],
                                    "gap_nat": t1[0] - lps[i]})

    # replay stability: per case, OFF argmax must be identical across its
    # responses (C21 claim) and ON must differ from it
    q2_replay = []
    for c in replay_cases:
        # replay semantics: prompt = original prompt + first d ON tokens
        # (identical history for both roles since IDs match up to d); the OFF
        # server then generates 1 token = its argmax AT the divergence step.
        off_ids = [tuple(r.get("output_ids")) for r in c["responses"]
                   if r["role"] == "OFF"]
        off_stable = len(set(off_ids)) == 1 if off_ids else None
        d = c["first_divergence"]
        on_rec = next((r for r in records if r["request_uuid"] == c["on_request_uuid"]), None)
        off_rec = next((r for r in records if r["request_uuid"] == c.get("off_request_uuid")), None)
        rep_tok = off_ids[0][-1] if off_ids else None
        pos = len(c["prefix"]) - (on_rec["response"]["meta_info"]["prompt_tokens"]
                                  if on_rec else len(c["prefix"]))
        q2_replay.append({
            "on_request_uuid": c["on_request_uuid"],
            "first_divergence": d,
            "replay_position_equals_d": pos == d,
            "n_off_replays": len(off_ids),
            "off_argmax_stable_across_replays": off_stable,
            "off_replay_token": rep_tok,
            "off_primary_token_at_d": off_rec["output_ids"][d] if off_rec else None,
            "on_primary_token_at_d": on_rec["output_ids"][d] if on_rec else None,
            "case_on_token": c["on_token"],
            "off_replay_matches_off_primary": (rep_tok is not None and off_rec is not None
                                               and rep_tok == off_rec["output_ids"][d]),
            "on_token_differs_from_stable_off": (rep_tok is not None
                                                 and c["on_token"] != rep_tok),
        })

    # ---------------- Q3: paired performance ----------------
    q3 = []
    for k in pair_keys:
        row = {"key": [k[0], k[1]]}
        for role in ("ON", "OFF"):
            r = by_key.get((k[0], k[1], role))
            if r is None:
                errors.append(f"perf pair {k} {role}: missing")
                continue
            mi = r["response"]["meta_info"]
            ct, e2e, ftl = mi["completion_tokens"], mi["e2e_latency"], mi["first_token_latency"]
            dec = e2e - ftl
            row[role] = {"completion": ct, "e2e": e2e, "ftl": ftl, "decode_span": dec,
                         "decode_tok_s": (ct - 1) / dec if dec > 0 and ct > 1 else None,
                         "ms_per_tok": dec * 1000 / (ct - 1) if dec > 0 and ct > 1 else None,
                         "queue_time": mi.get("queue_time"),
                         "num_retractions": mi.get("num_retractions")}
        if "ON" in row and "OFF" in row and row["ON"]["decode_tok_s"] and row["OFF"]["decode_tok_s"]:
            row["on_off_decode_tput_ratio"] = row["ON"]["decode_tok_s"] / row["OFF"]["decode_tok_s"]
            row["on_off_ms_per_tok_ratio"] = row["ON"]["ms_per_tok"] / row["OFF"]["ms_per_tok"]
        q3.append(row)
    ratios = [x["on_off_decode_tput_ratio"] for x in q3 if "on_off_decode_tput_ratio" in x]
    q3_env = {"retractions_total_ON": 0, "retractions_total_OFF": 0,
              "completions_all_48": True}
    for r in all_recs:
        mi2 = r["response"]["meta_info"]
        q3_env["retractions_total_" + r["role"]] += mi2.get("num_retractions") or 0
        if r in records and mi2.get("completion_tokens") != 48:
            q3_env["completions_all_48"] = False

    out = {
        "trial": "glm-b g3-t10", "input": "controller C21 packet (read-only)",
        "q1_recipe": q1,
        "q2_pairs": q2_pairs,
        "q2_controller_first_divergence_crosscheck_mismatches": mism,
        "q2_self_top1_audit": {k: {"positions": v[0], "mismatches": v[1]} for k, v in audit.items()},
        "q2_self_top1_cases": audit_cases,
        "q2_replay_stability": q2_replay,
        "q3_perf": q3,
        "q3_summary": {
            "pairs": len(ratios),
            "median_on_off_decode_tput_ratio": statistics.median(ratios) if ratios else None,
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
        "q3_env": q3_env,
        "harness_errors": errors,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"q1": q1, "crosscheck_mismatches": mism,
                      "self_top1": out["q2_self_top1_audit"],
                      "self_top1_case_count": len(audit_cases),
                      "q2_replay_stability": q2_replay,
                      "q3_summary": out["q3_summary"],
                      "errors": errors}, indent=1))
    if errors:
        print("VERDICT: NO_CLAIM (harness errors)", file=sys.stderr)
        sys.exit(2)
    print("VERDICT: OK -> " + OUT)

def summ_payload(summ):
    return summ["payload"]

def max_run_equal(a, b, d, n):
    best = cur = 0
    for i in range(d + 1, n):
        cur = cur + 1 if a[i] == b[i] else 0
        best = max(best, cur)
    return best

if __name__ == "__main__":
    main()

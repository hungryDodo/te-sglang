#!/usr/bin/env python3 -I
"""G3 trial-8 margin census over controller-supplied REAL NGRAM ON/OFF observations.

Question (H8): does the real top1-top2 margin distribution of frozen-launch BF16
generations make token-exactness statistically unreachable for fp8-KV-class
perturbations (E2-measured per-token |dlogprob| 0.012..0.153 nat), i.e. must the
precision quality budget be flip-rate/margin-stratified rather than token-exact?

Data: controller_evidence/continuous-20261109-01/20261110-runtime/ngram-full-observations.json
(read-only). OFF commit 999d4b243 = plain BF16 frozen launch -> BASELINE census.
ON = NGRAM-perturbed BF16 -> used only for divergence-dynamics mechanism checks.

Hard verdict gate: parsing controls must pass (rows sorted desc; chosen==top1 lp at
self-match positions; primary self-top1 anomaly counts reproduce the packet's
ON=1/OFF=0) or verdict is NO_CLAIM.
"""
import json, math, os, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "controller_evidence", "continuous-20261109-01", "20261110-runtime")
SRC = os.path.join(BASE, "ngram-full-observations.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
DELTAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.203125, 2.0]
EPS_TABLE = [0.012, 0.041, 0.153]  # E2-measured fp8-KV |dlogprob| scale (mean range, max)

def extract(rec):
    """Return (ids, rows) where rows[i] = list of (lp, tid) sorted as served, or None."""
    resp = rec.get("response") or {}
    mi = resp.get("meta_info") or {}
    ot = mi.get("output_top_logprobs")
    ids = rec.get("output_ids") or resp.get("output_ids") or []
    if not ot or not ids:
        return None
    rows = [[(float(t[0]), int(t[1])) for t in row if t is not None] for row in ot]
    n = min(len(ids), len(rows))
    return list(ids[:n]), rows[:n]

def census(rows_list):
    """rows_list: list of rows-per-response. Returns margin stats dict."""
    margins, floors, pos_total = [], [], 0
    unsorted_rows = 0
    for rows in rows_list:
        for row in rows:
            pos_total += 1
            lps = [lp for lp, _ in row]
            if any(lps[i] < lps[i+1] - 1e-9 for i in range(len(lps) - 1)):
                unsorted_rows += 1
            if len(row) >= 2:
                margins.append(row[0][0] - row[1][0])  # top1 - top2 >= 0 (nats)
            if len(row) >= 8:
                floors.append(row[0][0] - row[-1][0])  # top8 - top1 (truncation exposure)
    margins.sort()
    def pct(a, q):
        if not a: return None
        return a[min(len(a) - 1, int(q * len(a)))]
    tail = {str(d): (sum(1 for m in margins if m <= d), len(margins)) for d in DELTAS}
    return {
        "positions_total": pos_total,
        "positions_with_margin": len(margins),
        "unsorted_rows": unsorted_rows,
        "margin_min": margins[0] if margins else None,
        "margin_p50": pct(margins, 0.50), "margin_p90": pct(margins, 0.90),
        "margin_p99": pct(margins, 0.99),
        "P_margin_le_delta": {k: (c, n, round(c / n, 6) if n else None) for k, (c, n) in tail.items()},
        "truncation": {"positions_with_8rows": len(floors),
                        "P_top8_gap_le_0.5": round(sum(1 for f in floors if f <= 0.5) / len(floors), 6) if floors else None},
    }

def flip_projection(base_c):
    """base_c: census dict with P_margin_le_delta. Upper-bound flip model:
    flip at a position requires margin <= eps_top1 + eps_top2; bound both by eps
    -> p <= P(margin <= 2*eps)."""
    out = {}
    pm = base_c["P_margin_le_delta"]
    n = base_c["positions_with_margin"]
    for eps in EPS_TABLE:
        # nearest tabulated delta >= 2*eps, else compute from raw not available -> interpolate flag
        need = 2 * eps
        avail = [(float(k), v) for k, v in pm.items()]
        avail.sort()
        p = None; src = None
        for d, (c, nn, r) in avail:
            if d >= need:
                p = r; src = ("tabulated_delta", d); break
        if p is None:
            p = pm[str(DELTAS[-1])][2]; src = ("capped_at_max_delta", DELTAS[-1])
        if p is not None:
            exp_flips = 512 * p
            p_any = 1 - (1 - p) ** 512
            out[str(eps)] = {"eps_nat": eps, "p_flip_bound": p, "bound_source": src,
                             "expected_first_flips_per_512": round(exp_flips, 2),
                             "P_at_least_one_flip_in_512": round(p_any, 6)}
    return out

def main():
    d = json.load(open(SRC))
    records, replay = d["records"], d["replay"]
    errors = []

    by_role = {"ON": [], "OFF": []}
    parsed = []  # (rec, ids, rows)
    self_top1 = {"ON": [], "OFF": []}
    chosen_mismatch = 0
    for rec in records:
        ex = extract(rec)
        if ex is None:
            errors.append(f"unparsable primary {rec.get('request_uuid')}")
            continue
        ids, rows = ex
        parsed.append((rec, ids, rows))
        by_role[rec["role"]].append(rows)
        for i, (tid, row) in enumerate(zip(ids, rows)):
            if row and row[0][1] != tid:
                self_top1[rec["role"]].append({"label": rec.get("label"), "instance": rec.get("instance"), "pos": i})
            ch = rec.get("chosen_logprobs") or []
            chi = ch[i] if i < len(ch) else None
            # chosen rows are triples [lp, tid, text]; compare lp and tid
            if row and chi is not None:
                chlp = chi[0] if isinstance(chi, (list, tuple)) else float(chi)
                if abs(chlp - row[0][0]) > 1e-6:
                    chosen_mismatch += 1

    replay_rows = {"ON": [], "OFF": []}
    replay_ids = {"ON": [], "OFF": []}
    for case in replay:
        for rec in case["responses"]:
            ex = extract(rec)
            if ex is None:
                errors.append(f"unparsable replay {rec.get('request_uuid')}")
                continue
            ids, rows = ex
            replay_rows[rec["role"]].append(rows)
            replay_ids[rec["role"]].append((case, rec, ids, rows))

    # --- baseline census (OFF primary; plain BF16 frozen launch) ---
    base = census(by_role["OFF"])
    on_c = census(by_role["ON"])

    # --- paired first-divergence mechanism check (primary pairs) ---
    pairs = {}
    for rec, ids, rows in parsed:
        k = (rec["label"], rec["instance"])
        pairs.setdefault(k, {})[rec["role"]] = (ids, rows)
    pair_tbl = []
    for k, v in sorted(pairs.items()):
        if "ON" not in v or "OFF" not in v:
            errors.append(f"pair {k} missing a role")
            continue
        oi, orows = v["ON"]; fi, frows = v["OFF"]
        n = min(len(oi), len(fi))
        first = next((i for i in range(n) if oi[i] != fi[i]), None)
        entry = {"label": k[0], "instance": k[1], "len": n, "first_div_pos": first}
        if first is not None:
            off_m = (frows[first][0][0] - frows[first][1][0]) if len(frows[first]) >= 2 else None
            on_m = (orows[first][0][0] - orows[first][1][0]) if len(orows[first]) >= 2 else None
            post = [i for i in range(first, n) if oi[i] != fi[i]]
            entry.update({
                "off_margin_at_first_div": off_m, "on_margin_at_first_div": on_m,
                "on_token": oi[first], "off_token": fi[first],
                "on_off_top1_in_off_top8": fi[first] in [t for _, t in orows[first]],
                "off_top1_in_on_top8": oi[first] in [t for _, t in frows[first]],
                "positions_differing_after_first": len(post),
                "positions_after_first": n - first,
                "cascade_rate": round(len(post) / (n - first), 4) if n > first else None,
                "resync_events": sum(1 for i in range(first + 1, n) if oi[i] == fi[i]),
            })
        pair_tbl.append(entry)

    # margin at first-div vs response median margin (mechanism: near-tie flips)
    mech = []
    for e in pair_tbl:
        if e["first_div_pos"] is None: continue
        k = (e["label"], e["instance"])
        frows = pairs[k]["OFF"][1]
        ms = sorted(r[0][0] - r[1][0] for r in frows if len(r) >= 2)
        if ms:
            mech.append({"pair": f"{e['label']}/{e['instance']}",
                         "first_div_margin": e["off_margin_at_first_div"],
                         "response_margin_p50": ms[len(ms)//2],
                         "first_div_margin_le_p50": (e["off_margin_at_first_div"] is not None
                                                     and e["off_margin_at_first_div"] <= ms[len(ms)//2])})

    # --- replay argmax probes: OFF stability + ON divergence margins ---
    rep_tbl = []
    for case in replay:
        offs = [(r, i, rw) for (c, r, i, rw) in replay_ids["OFF"] if c is case]
        ons = [(r, i, rw) for (c, r, i, rw) in replay_ids["ON"] if c is case]
        off_tokens = [i[0] for _, i, _ in offs if i]
        on_tokens = [i[0] for _, i, _ in ons if i]
        row = {"prefix_sha256": case["prefix_sha256"][:12], "declared_first_div": case["first_divergence"],
               "off_tokens": off_tokens, "on_tokens": on_tokens,
               "off_stable": len(set(off_tokens)) == 1,
               "on_equals_off": (set(on_tokens) & set(off_tokens)) == set(on_tokens) if on_tokens and off_tokens else None}
        om = []
        for _, i, rw in offs:
            if rw and len(rw[0]) >= 2: om.append(rw[0][0][0] - rw[0][1][0])
        row["off_argmax_margins"] = om
        rep_tbl.append(row)

    # --- verdict gate ---
    controls = {
        "rows_sorted_desc_all": base["unsorted_rows"] == 0 and on_c["unsorted_rows"] == 0,
        "chosen_eq_top1_lp": chosen_mismatch == 0,
        "primary_self_top1_reproduces_packet": (len(self_top1["ON"]) == 1 and len(self_top1["OFF"]) == 0),
        "ten_pairs_all_roles": len(pair_tbl) == 10,
        "no_parse_errors": not errors,
    }
    ok = all(controls.values())
    proj = flip_projection(base)
    result = {
        "schema": "g3-t8/margin_census v1",
        "source": "controller_evidence/continuous-20261109-01/20261110-runtime/ngram-full-observations.json",
        "source_pair_uuid": d.get("actual_pair_uuid"),
        "roles": {"ON_commit": "b9a91aa13(NGRAM ON)", "OFF_commit": "999d4b243(plain BF16, baseline)"},
        "sampling": "temperature=0 (greedy), return_logprob=true, top_logprobs_num=8 (from packet request bodies)",
        "controls": controls, "control_detail": {"self_top1_anomalies": self_top1,
                                                 "chosen_mismatch_positions": chosen_mismatch,
                                                 "parse_errors": errors},
        "baseline_census_OFF_plainBF16": base,
        "on_census_for_contrast": on_c,
        "flip_risk_projection_ON_OFF_baseline": proj,
        "primary_pairs": pair_tbl,
        "first_div_mechanism": mech,
        "replay_argmax_probes": rep_tbl,
    }
    # headline verdict on H8
    if ok:
        p153 = proj["0.153"]
        verdict = ("H8_SUPPORTED" if p153["P_at_least_one_flip_in_512"] > 0.99
                   and base["P_margin_le_delta"]["0.3"][2] > 0 else "H8_NOT_SUPPORTED_AT_MAX_EPS")
        result["verdict"] = verdict
    else:
        result["verdict"] = "NO_CLAIM_CONTROLS_FAILED"
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "margin_census_output.json")
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("verdict", "controls", "baseline_census_OFF_plainBF16",
                                             "flip_risk_projection_ON_OFF_baseline")}, indent=1))
    print("pairs:", len(pair_tbl), "replay_cases:", len(rep_tbl))
    print("OUT:", out)
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()

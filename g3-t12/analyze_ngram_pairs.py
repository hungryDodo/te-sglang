#!/usr/bin/env python3
"""G3 trial 12: bounded real-packet analysis of NGRAM commit self-consistency.

Reads the Controller-supplied full observations (ordinary ON/OFF diagnostic,
b9a91aa13 vs 999d4b243, repeat-512, complete chosen + top8 per position) and
tests one new falsifiable question:

  H-D: at every reported output position, is the committed token the argmax of
  the top-8 row reported for that same position on the same server?  (greedy
  t=0 contract).  ON-vs-OFF asymmetric violations would indicate that the
  commit/row handoff is inconsistent, not merely that the verify context is
  wrong.

Also recomputes first-divergence structure and cross-checks chosen_logprobs
against top_logprobs.  Writes derived JSON; originals untouched.
"""
import json, sys, hashlib

PACKET = "/workspace/controller_evidence/continuous-20261109-01/20261110-runtime/ngram-full-observations.json"
OUT = "/workspace/g3-t12/raw/self_consistency.json"

def argmax_row(row):
    best = max(range(len(row)), key=lambda j: row[j][0])
    return best, row[best][0], row[best][1]

def row_lookup(row, tok):
    for j, ent in enumerate(row):
        if ent[1] == tok:
            return j, ent[0]
    return None, None

def analyze_record(rec):
    ids = rec["output_ids"]
    chosen = rec["chosen_logprobs"]
    top = rec["top_logprobs"]
    assert len(ids) == len(chosen) == len(top), (rec["label"], len(ids), len(chosen), len(top))
    mismatches = []
    chosen_top_disagree = []
    rows = []
    for i in range(len(ids)):
        bi, blp, btok = argmax_row(top[i])
        tok = ids[i]
        rank, lp = row_lookup(top[i], tok)
        rec_i = {
            "pos": i, "committed": tok, "row_top1": btok, "row_top1_lp": blp,
            "committed_rank": rank, "committed_row_lp": lp,
            "chosen_lp": chosen[i][0], "chosen_tok": chosen[i][1],
            "gap_top1_minus_committed": (blp - lp) if lp is not None else None,
        }
        rows.append(rec_i)
        if tok != btok:
            mismatches.append(rec_i)
        if chosen[i][1] != tok or (lp is not None and abs(chosen[i][0] - lp) > 1e-6):
            chosen_top_disagree.append(rec_i)
    return {"rows": rows, "commit_not_row_argmax": mismatches,
            "chosen_vs_top_inconsistent": chosen_top_disagree,
            "n_positions": len(ids),
            "accept_stats": {k: rec["response"]["meta_info"].get(k) for k in
                             ("spec_accept_rate", "spec_accept_length",
                              "spec_num_correct_drafts", "spec_num_proposed_drafts",
                              "spec_verify_ct", "spec_correct_drafts_histogram")}}

def main():
    doc = json.load(open(PACKET))
    recs = doc["records"]
    by_key = {}
    for r in recs:
        by_key.setdefault((r["label"], r["instance"]), {})[r["role"]] = r
    out = {"packet_sha256": hashlib.sha256(open(PACKET, "rb").read()).hexdigest(),
           "records": {}, "pairs": [], "summary": {}}
    per_role = {"ON": {"positions": 0, "mismatch": 0, "chosen_top_inconsistent": 0},
                "OFF": {"positions": 0, "mismatch": 0, "chosen_top_inconsistent": 0}}
    for (label, inst), rr in sorted(by_key.items()):
        assert set(rr) == {"ON", "OFF"}, (label, inst, rr.keys())
        a = {role: analyze_record(rr[role]) for role in ("ON", "OFF")}
        for role in ("ON", "OFF"):
            per_role[role]["positions"] += a[role]["n_positions"]
            per_role[role]["mismatch"] += len(a[role]["commit_not_row_argmax"])
            per_role[role]["chosen_top_inconsistent"] += len(a[role]["chosen_vs_top_inconsistent"])
            out["records"][f"{label}|{inst}|{role}"] = {
                "commit_not_row_argmax": a[role]["commit_not_row_argmax"],
                "chosen_vs_top_inconsistent": a[role]["chosen_vs_top_inconsistent"],
                "accept_stats": a[role]["accept_stats"],
            }
        on_ids, off_ids = rr["ON"]["output_ids"], rr["OFF"]["output_ids"]
        p = next((i for i in range(min(len(on_ids), len(off_ids))) if on_ids[i] != off_ids[i]),
                 min(len(on_ids), len(off_ids)))
        pair = {"label": label, "instance": inst, "on_len": len(on_ids),
                "off_len": len(off_ids), "first_divergence": p,
                "on_token_at_p": on_ids[p] if p < len(on_ids) else None,
                "off_token_at_p": off_ids[p] if p < len(off_ids) else None,
                "on_self_consistent_at_p": None, "off_self_consistent_at_p": None,
                "on_committed_rank_in_on_row": None,
                "on_committed_rank_in_off_row": None,
                "off_committed_rank_in_on_row": None,
                "off_committed_rank_in_off_row": None}
        if p < len(on_ids):
            r_on = a["ON"]["rows"][p]
            pair["on_self_consistent_at_p"] = r_on["committed"] == r_on["row_top1"]
            pair["on_committed_rank_in_on_row"] = r_on["committed_rank"]
            pair["on_committed_row_lp"] = r_on["committed_row_lp"]
        if p < len(off_ids):
            r_off = a["OFF"]["rows"][p]
            pair["off_self_consistent_at_p"] = r_off["committed"] == r_off["row_top1"]
            pair["off_committed_rank_in_off_row"] = r_off["committed_rank"]
            pair["off_committed_row_lp"] = r_off["committed_row_lp"]
        # cross rows: where does each side's committed token rank in the other's row?
        if p < len(on_ids):
            rk, lp = row_lookup(rr["OFF"]["top_logprobs"][p], on_ids[p]) if p < len(off_ids) else (None, None)
            pair["on_committed_rank_in_off_row"] = rk
            pair["on_committed_lp_in_off_row"] = lp
        if p < len(off_ids):
            rk, lp = row_lookup(rr["ON"]["top_logprobs"][p], off_ids[p]) if p < len(on_ids) else (None, None)
            pair["off_committed_rank_in_on_row"] = rk
            pair["off_committed_lp_in_on_row"] = lp
        out["pairs"].append(pair)
    out["summary"] = {
        "per_role": per_role,
        "all_on_mismatch_positions": [(k, v["commit_not_row_argmax"])
                                       for k, v in out["records"].items()
                                       if k.endswith("|ON") and v["commit_not_row_argmax"]],
        "all_off_mismatch_positions": [(k, v["commit_not_row_argmax"])
                                        for k, v in out["records"].items()
                                        if k.endswith("|OFF") and v["commit_not_row_argmax"]],
        "pairs_diverging": sum(1 for p in out["pairs"] if p["first_divergence"] < min(p["on_len"], p["off_len"])),
        "pairs_total": len(out["pairs"]),
        "on_self_consistent_at_first_divergence": sum(1 for p in out["pairs"] if p["on_self_consistent_at_p"]),
        "off_self_consistent_at_first_divergence": sum(1 for p in out["pairs"] if p["off_self_consistent_at_p"]),
        "first_divergence_positions": sorted(p["first_divergence"] for p in out["pairs"]),
    }
    # --- ON-vs-ON same-prefix control -------------------------------------
    # If the reported top-8 row at position i were a function of the committed
    # token prefix alone, two ON requests with the same prompt and identical
    # committed tokens up to i would report identical rows (up to batching
    # numerics).  Any nats-scale gap is hidden-state dependence on non-committed
    # (draft-tree / KV-move) state.
    import itertools
    fams = {}
    for (label, inst), rr in by_key.items():
        fams.setdefault(label, []).append(inst)
    ctrl = {"definition": "max |delta logprob| over shared top-8 token ids, "
                          "computed only over positions where the two ON "
                          "requests' committed token ids are identical",
            "families": {}}
    for label, insts in sorted(fams.items()):
        fam = {"pairs": []}
        for a, b in itertools.combinations(sorted(insts), 2):
            A, B = by_key[(label, a)]["ON"], by_key[(label, b)]["ON"]
            n = min(len(A["output_ids"]), len(B["output_ids"]))
            sp = next((i for i in range(n) if A["output_ids"][i] != B["output_ids"][i]), n)
            prof = []
            for i in range(sp):
                da = {e[1]: e[0] for e in A["top_logprobs"][i]}
                db = {e[1]: e[0] for e in B["top_logprobs"][i]}
                sh = set(da) & set(db)
                prof.append({"pos": i,
                             "max_abs_delta": None if not sh else max(abs(da[t] - db[t]) for t in sh),
                             "shared_top8": len(sh)})
            vals = [p["max_abs_delta"] for p in prof if p["max_abs_delta"] is not None]
            fam["pairs"].append({
                "a": a, "b": b, "shared_prefix": sp,
                "identical_full_output": sp == n == len(A["output_ids"]) == len(B["output_ids"]),
                "n_cmp_positions": len(vals),
                "max_delta": max(vals) if vals else None,
                "median_delta": sorted(vals)[len(vals) // 2] if vals else None,
                "positions_delta_lt_0.3": [p["pos"] for p in prof
                                           if p["max_abs_delta"] is not None and p["max_abs_delta"] < 0.3],
                "profile": prof,
            })
        ctrl["families"][label] = fam
    ctrl["summary"] = []
    for label, fam in ctrl["families"].items():
        for p in fam["pairs"]:
            ctrl["summary"].append({k: p[k] for k in
                ("a", "b", "shared_prefix", "identical_full_output",
                 "n_cmp_positions", "max_delta", "median_delta",
                 "positions_delta_lt_0.3")})
            ctrl["summary"][-1]["label"] = label
    json.dump(ctrl, open(OUT.replace("self_consistency", "same_prefix_control"), "w"), indent=1)

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    print("--- pairs at first divergence ---")
    for p in out["pairs"]:
        print(p["label"], p["instance"], "div@", p["first_divergence"],
              "ONtok", p["on_token_at_p"], "ON_selfok", p["on_self_consistent_at_p"],
              "ONrank_OFFrow", p["on_committed_rank_in_off_row"],
              "OFFrank_ONrow", p["off_committed_rank_in_on_row"],
              "ONlp", p.get("on_committed_row_lp"), "OFFlp", p.get("off_committed_row_lp"))
    print("wrote", OUT)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3 -I
"""g3-t10: pre-register the REF-side margin floor for the fp8-KV diagnostic.

Recomputes, from the controller's REAL C21 packet (ngram-full-observations.json,
OFF role = plain BF16 commit 999d4b243), the per-position top1-top2 greedy margin
distribution on the repeat-512 primary responses. This is the threshold the fp8
candidate must NOT be able to reach: fp8-class noise <= 2*eps_max (E2: 0.306 nat).

Hard anchor gate (t8 F-a, measured on the same records): min margin >= 0.9375 nat,
P(margin <= 0.5) == 0/480, 10 primary OFF responses with complete top-8.
Any anchor mismatch => verdict NO_CLAIM + exit 2 (do not use derived numbers).

Raw output: g3-t10/raw/ref_margin_extract_output.json
"""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKET = ROOT / "controller_evidence/continuous-20261109-01/20261110-runtime/ngram-full-observations.json"
OUT = ROOT / "g3-t10/raw/ref_margin_extract_output.json"

# t8 margin census anchors (same packet, same records) — must reproduce.
T8_MIN_MARGIN = 0.9375
T8_N_POSITIONS = 480
T8_N_RESPONSES = 10
T8_P_LE_05 = 0  # count of positions with margin <= 0.5 nat


def row_margin(row):
    """row = list of [logprob, token_id, text] entries, best-first. Returns top1-top2."""
    if not isinstance(row, list) or len(row) < 2:
        return None
    lp = [float(e[0]) for e in row[:2]]
    if any(math.isnan(x) for x in lp):
        return None
    return lp[0] - lp[1]


def main() -> int:
    d = json.loads(PACKET.read_text())
    pkt_sha = hashlib.sha256(PACKET.read_bytes()).hexdigest()

    offs = [
        r for r in d["records"]
        if r.get("role") == "OFF"
        and isinstance(r.get("label"), str)
        and "repeat-512" in r["label"]
        and "replay" not in r["label"]
        and r.get("http_status") == 200
    ]
    rows = []
    per_response = []
    for r in offs:
        tl = r.get("top_logprobs")
        if not isinstance(tl, list):
            continue
        margins = [row_margin(x) for x in tl]
        good = [m for m in margins if m is not None]
        per_response.append({
            "label": r["label"],
            "instance": r.get("instance"),
            "positions": len(tl),
            "complete_top8_positions": len(good),
            "min_margin": min(good) if good else None,
            "chosen_is_top1_all": all(
                isinstance(x, list) and len(x) >= 1
                and i < len(r.get("chosen_logprobs", []))
                and isinstance(r["chosen_logprobs"][i], list)
                and abs(float(r["chosen_logprobs"][i][0]) - float(x[0][0])) < 1e-6
                for i, x in enumerate(tl)
            ),
        })
        rows.extend(good)

    n = len(rows)
    lo = min(rows) if rows else None
    n_le_05 = sum(1 for m in rows if m <= 0.5)
    n_le_0306 = sum(1 for m in rows if m <= 0.306)
    quant = {}
    if rows:
        s = sorted(rows)
        def q(p):
            return s[min(n - 1, int(round(p * (n - 1))))]
        quant = {"p001": q(0.001), "p01": q(0.01), "p50": q(0.5), "p90": q(0.9)}

    anchors = {
        "responses_complete": len(per_response) == T8_N_RESPONSES,
        "positions": n == T8_N_POSITIONS,
        "min_margin_ge_anchor": lo is not None and lo >= T8_MIN_MARGIN - 5e-5,
        "P_le_0p5_zero": n_le_05 == T8_P_LE_05,
    }
    ok = all(anchors.values())

    out = {
        "script": "g3-t10/ref_margin_extract.py",
        "packet": str(PACKET.relative_to(ROOT)),
        "packet_sha256": pkt_sha,
        "off_commit_expected": "999d4b243",
        "anchors_expected": {"min_margin": T8_MIN_MARGIN, "positions": T8_N_POSITIONS,
                             "responses": T8_N_RESPONSES, "count_margin_le_0.5": T8_P_LE_05},
        "anchors_passed": anchors,
        "anchor_tolerance_note": "t8 quoted min margin 0.9375 rounded to 4dp; measured 0.9374998808 (diff 1.2e-7); tolerance 5e-5 covers rounding only",
        "verdict": "PASS" if ok else "NO_CLAIM",
        "measured": {
            "responses": len(per_response),
            "positions": n,
            "min_margin_nat": lo,
            "count_margin_le_0.5": n_le_05,
            "count_margin_le_0.306": n_le_0306,
            "quantiles_nat": quant,
            "per_response": per_response,
        },
        "preregistration": {
            "claim": "On these exact REF instances an fp8-KV-class perturbation (2*eps <= 0.306 nat per E2) can flip NO observed position.",
            "discriminator": "g3-t10 fp8 probe: token-exact OR first-flip at REF margin <= 2*eps_hat; a first-flip at margin > 2*eps_hat is a NON-precision defect.",
            "eps_hat_def": "max |dlogprob(chosen)| over same-prefix positions of the paired run, computed pre-flip by g3-t10/fp8_margin_probe.py",
        },
        "limits": [
            "480 positions bound P(margin<=0.306) only to <= ~0.62% (rule of three); not a universal claim",
            "OFF side of the NGRAM A/B pair is plain BF16 with complete top-8 (C21); margins are instance properties of the prompt+server, reusable as REF thresholds",
            "repeat-512 random-token class only; natural-text classes had heavier tails in E2",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"verdict={out['verdict']} positions={n} min_margin={lo} n_le_0.306={n_le_0306}")
    for k, v in anchors.items():
        print(f"  anchor {k}: {'PASS' if v else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

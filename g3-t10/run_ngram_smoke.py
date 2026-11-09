#!/usr/bin/env python3 -I
"""g3-t10: run the margin-contract analyzer on REAL C21 ON/OFF records.

Expected (pre-registered): NGRAM ON vs OFF — canonical F1 BUG_CONFIRMED with
10/10 divergent pairs and OFF-side first-flip margins ~0.94-1.16 nat (t8 F-e)
— must classify NON_PRECISION_DEFECT, NOT PRECISION_CONSISTENT. If it does,
the contract discriminates a real bug from precision noise on real arrays,
which is exactly the discrimination requested for the fp8-KV pair.

Raw: g3-t10/raw/real_ngram_smoke_output.json (via fp8_margin_probe.py analyze).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKET = ROOT / "controller_evidence/continuous-20261109-01/20261110-runtime/ngram-full-observations.json"
TMP = ROOT / "g3-t10/raw/_smoke_ref.json"
TMP2 = ROOT / "g3-t10/raw/_smoke_cand.json"
OUT = ROOT / "g3-t10/raw/real_ngram_smoke_output.json"


def project(role):
    d = json.loads(PACKET.read_text())
    recs = []
    for r in d["records"]:
        if r.get("role") != role or r.get("http_status") != 200:
            continue
        if "replay" in r.get("label", ""):
            continue
        recs.append({
            "label": f"{r['label']}/i{r.get('instance')}",
            "output_ids": r["output_ids"],
            "chosen_logprobs": r["chosen_logprobs"],
            "top_logprobs": r["top_logprobs"],
        })
    return {"role": role, "records": recs}


def main() -> int:
    ref, cand = project("OFF"), project("ON")
    TMP.write_text(json.dumps(ref))
    TMP2.write_text(json.dumps(cand))
    rc = subprocess.call(
        [sys.executable, "-I", str(ROOT / "g3-t10/fp8_margin_probe.py"),
         "analyze", str(TMP), str(TMP2), str(OUT)]
    )
    TMP.unlink()
    TMP2.unlink()
    if rc != 0:
        return rc
    res = json.loads(OUT.read_text())
    ok = res["overall"] == "NON_PRECISION_DEFECT"
    print(f"smoke expected NON_PRECISION_DEFECT, got {res['overall']} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

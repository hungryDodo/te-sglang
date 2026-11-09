#!/usr/bin/env python3
"""Independent planted-packet self-test for the G3-t14 comparison logic.

Builds synthetic records with known answers and asserts set_metrics /
replay_case behavior, including a planted +1.0 nat shift, an absent-token
case, and a prefix-alignment failure. No real data used.

Run: python3 -I selftest_metrics.py; exit 0 == all controls PASS.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ang", os.path.join(HERE, "analyze_ngram_packet.py"))
ang = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ang)


def rec(out_ids, lps, tops):
    # lps: [[lp,id,None],...]; tops: [[[lp,id,None] x8],...]
    return {"output_ids": out_ids, "chosen_logprobs": lps,
            "top_logprobs": tops, "label": "x", "instance": 0,
            "http_status": 200, "role": "OFF"}


def t8(*pairs):
    return [[lp, tid, None] for lp, tid in pairs]


def one_pos(lp, tid, others):
    return [[lp, tid, None]] + [[l, t, None] for l, t in others]


def main():
    fails = []
    # Control 1: identical distributions -> zero diff, argmax match.
    a = [( -0.5, 7), (-1.0, 9), (-2.0, 11)]
    m = ang.set_metrics(a, [(lp, i) for lp, i in a])
    if m["max_abs_lp_diff_common"] != 0.0 or not m["argmax_match"]:
        fails.append("identical")
    # Control 2: planted +1.0 nat on a common token.
    b = [(-1.5, 7), (-1.0, 9), (-2.0, 11)]
    m = ang.set_metrics(b, a)
    if abs(m["max_abs_lp_diff_common"] - 1.0) > 1e-12:
        fails.append("planted_shift")
    # Control 3: absent committed token -> None everywhere.
    m = ang.set_metrics([(-0.1, 999)], a)
    if m["n_common"] != 0 or m["max_abs_lp_diff_common"] is not None:
        fails.append("absent")
    # Control 4: rank/lp lookup helpers.
    if ang.rank_of(a, 9) != 1 or ang.lp_of(a, 11) != -2.0 or ang.rank_of(a, 5) is not None:
        fails.append("helpers")
    # Control 5: replay_case end-to-end on a planted matched prefix.
    prompt = list(range(512))
    out = [100, 101]
    on = rec(out, [[-0.2, 100, None], [-0.25, 101, None]],
             [one_pos(-0.2, 100, [(-2.0, 5), (-3.0, 6)]),
              one_pos(-0.25, 101, [(-2.5, 7), (-3.1, 8)])])
    off = rec(out, [[1e9, 0, None], [-0.25, 101, None]],
              [one_pos(-0.2, 100, [(-2.0, 5), (-3.0, 6)]),
               one_pos(-0.25, 101, [(-2.5, 7), (-3.1, 8)])])
    # off chosen[0] is dummy/unused before p; p=1, prefix = prompt + [100].
    pc = {"first_divergence": 1, "on_token": 101,
          "on_request_uuid": "ON", "off_request_uuid": "OFF",
          "prefix": prompt + [100],
          "responses": [
              rec([101], [[-0.25, 101, None]],
                  [one_pos(-0.25, 101, [(-2.5, 7), (-3.1, 8)])])
              for _ in range(4)]}
    cr = ang.replay_case(pc, {"ON": on, "OFF": off}, 0)
    if not cr["on_own_constency"]["committed_is_own_argmax"]:
        fails.append("planted_self_consistency")
    if cr["on_vs_off_ref"]["max_abs_lp_diff_common"] != 0.0:
        fails.append("planted_on_vs_off_zero")
    if cr["off_noise"]["max_same_token_lp_spread"] != 0.0:
        fails.append("planted_off_noise")
    # Control 6: prefix misalignment must fail closed.
    bad = dict(pc)
    bad["prefix"] = prompt + [999]
    try:
        ang.replay_case(bad, {"ON": on, "OFF": off}, 0)
        fails.append("prefix_must_fail_closed")
    except AssertionError:
        pass
    print("SELFTEST", "PASS" if not fails else "FAIL", fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

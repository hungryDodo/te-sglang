#!/usr/bin/env python3
"""Bounded CPU oracle: NGRAM verify attention-mask assembly vs KV read width.

Pure stdlib (no torch, no GPU, no provider). Ports the *width arithmetic* of

  python/sglang/srt/speculative/ngram_info.py:60-117
      generate_attn_arg_prefill:  w_i = paged_kernel_lens[i] + draft_token_num
      (device prefix length), cum_kv_seq_len = cumsum(w_i),
      mask_numel = paged_kernel_lens_sum * D + D^2 * bs, and the silent
      recovery branch "if custom_mask.numel() < mask_numel: append True".

  python/sglang/srt/speculative/ngram_worker.py:352-361
      ones_masks = [ones((D, batch.seq_lens_cpu[i]))] built BEFORE the
      blocking accept sync, i.e. the prefix block of the custom mask has
      width h_i = host seq_lens_cpu[i], while the KV read width uses
      d_i = device paged_kernel_lens[i].

Invariant under test (I1): for every draft row r of every request i, the set
of attended logical columns equals

    prefix positions {0 .. p_i-1}  union  {d_i + j : j in ancestors(r)}

where ancestors(r) are the strict-lower-triangular set bits of the tree mask
row (the same contract reconstruct_indices_from_tree_mask_cpu derives
depth/parent from). Column c < d_i denotes prefix KV position c; column
d_i + j denotes draft node j's KV slot (draft slots are packed in node order,
matching assign_extend_cache_locs' out_cache_loc order).

Scenarios:
  consistent_host_device      h_i == d_i for all i (the documented FutureMap
                              contract) -> I1 must hold exactly.
  host_lags_device_k1/_k2     h_i = d_i - k, i.e. the host prefix used to
                              build the mask is one/two tokens behind the
                              device read width. This is NOT asserted reachable
                              (FutureMap.resolve_seq_lens_cpu is designed to
                              prevent it); the oracle quantifies what the
                              pinned code does if it ever happens.
  host_ahead_device_k1        h_i = d_i + 1.
  graph_batch_padding         legitimate CUDA-graph padding: 2 fake requests
                              with d=1 are added to bs without building their
                              mask rows; the appended True tail must land only
                              on fake rows and leave real rows exact.

A scenario records rows_checked / rows_exact and the first mismatch with its
expected vs attended column sets. Output: JSON (raw evidence), not a claim
about model numerics.
"""

import json
import sys

D = 4  # draft_token_num used by the synthetic scenarios


def child_mask(kind):
    """Fixed synthetic proper BFS tree mask rows (node -> bool row of length D).

    node 0 = root (pending token). Row r lists its strict ancestors (j < r).
    """
    if kind == "chain":
        rows = [[False] * D for _ in range(D)]
        for r in range(1, D):
            rows[r][r - 1] = True
            for j in range(r - 1):
                rows[r][j] = rows[r - 1][j]
        return rows
    if kind == "bfs":
        # 0 -> {1,2}; 1 -> {3}: ancestors(1)={0}, (2)={0}, (3)={0,1}
        parent = [None, 0, 0, 1]
        rows = [[False] * D for _ in range(D)]
        for r in range(1, D):
            p = parent[r]
            rows[r] = list(rows[p])
            rows[r][p] = True
        return rows
    if kind == "star":
        # 0 -> {1,2,3}
        rows = [[False] * D for _ in range(D)]
        for r in range(1, D):
            rows[r][0] = True
        return rows
    raise ValueError(kind)


def ancestors(rows, r):
    return {j for j in range(r) if rows[r][j]}


def build_engine_mask(reqs, D):
    """Flat custom mask exactly as ngram_worker assembles it.

    reqs: list of dicts with keys h (host prefix used for ones_masks) and
    rows (tree mask rows). Prefix block width per row is h_i.
    """
    flat = []
    for req in reqs:
        for r in range(D):
            flat.extend([True] * req["h"])
            flat.extend(req["rows"][r])
    return flat


def parse_and_check(reqs, flat, D, appended_tail):
    """Consume the flat mask with the *device* row widths, then check I1."""
    pos = 0
    checked = 0
    exact = 0
    first_bad = None
    per_req = []
    for i, req in enumerate(reqs):
        d = req["d"]
        w = d + D
        req_bad = []
        req_exact = 0
        for r in range(D):
            row = flat[pos : pos + w]
            pos += w
            if len(row) != w:
                req_bad.append({"row": r, "error": "short mask row"})
                checked += 1
                continue
            attended = {c for c, v in enumerate(row) if v}
            exp_prefix = set(range(req["p"]))
            exp_draft = {d + j for j in ancestors(req["rows"], r)}
            expected = exp_prefix | exp_draft
            checked += 1
            if attended == expected:
                exact += 1
                req_exact += 1
            else:
                extra = sorted(attended - expected)
                missing = sorted(expected - attended)
                req_bad.append(
                    {
                        "row": r,
                        "extra_cols": extra[:8],
                        "extra_n": len(extra),
                        "missing_cols": missing[:8],
                        "missing_n": len(missing),
                        "attended_padding_tail": any(
                            c >= d + D for c in attended
                        ),
                    }
                )
                if first_bad is None:
                    first_bad = {
                        "req": i,
                        "row": r,
                        "extra_cols": extra[:8],
                        "missing_cols": missing[:8],
                        "expected_n": len(expected),
                        "attended_n": len(attended),
                    }
        per_req.append(
            {"req": i, "rows_exact": req_exact, "rows_checked": D, "bad_rows": req_bad}
        )
    return {
        "rows_checked": checked,
        "rows_exact": exact,
        "first_mismatch": first_bad,
        "per_req": per_req,
        "flat_consumed": pos,
        "flat_len": len(flat),
        "tail_appended": appended_tail,
    }


def run_scenario(name, host_delta, graph_pad=0, tree="bfs"):
    """host_delta: h_i = d_i + host_delta. graph_pad: fake reqs at d=1."""
    rows = child_mask(tree)
    real_d = [5, 7, 6]
    real_p = [5, 7, 6]
    reqs = []
    for d, p in zip(real_d, real_p):
        reqs.append({"d": d, "p": p, "h": d + host_delta, "rows": rows})
    # fake graph-padding requests: they have device widths but NO built mask
    fake = [{"d": 1, "p": 1, "h": None, "rows": None} for _ in range(graph_pad)]

    built = build_engine_mask(reqs, D)
    # generate_attn_arg_prefill width arithmetic over real+fake requests
    all_lens = [r["d"] for r in reqs] + [f["d"] for f in fake]
    mask_numel = sum(all_lens) * D + (D**2) * (len(reqs) + len(fake))
    appended = 0
    flat = list(built)
    if len(flat) < mask_numel:
        appended = mask_numel - len(flat)
        flat.extend([True] * appended)
    elif len(flat) > mask_numel:
        # engine passes an over-long mask; kernel consumes mask_numel worth of
        # rows -> keep the head, record the surplus.
        flat = flat[:mask_numel]

    check_reqs = list(reqs) + [
        {"d": f["d"], "p": f["p"], "rows": child_mask("star")} for f in fake
    ]
    res = parse_and_check(check_reqs, flat, D, appended)
    res["scenario"] = name
    res["host_delta"] = host_delta
    res["graph_pad"] = graph_pad
    res["tree"] = tree
    res["mask_numel"] = mask_numel
    res["built_numel"] = len(built)
    # invariant holds iff every *real* row is exact
    real_rows = D * len(reqs)
    real_exact = sum(rr["rows_exact"] for rr in res["per_req"][: len(reqs)])
    res["real_rows_checked"] = real_rows
    res["real_rows_exact"] = real_exact
    res["invariant_holds_on_real_rows"] = real_exact == real_rows
    res["verdict"] = "PASS" if res["invariant_holds_on_real_rows"] else "FAIL"
    return res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    scenarios = [
        run_scenario("consistent_host_device", 0),
        run_scenario("consistent_host_device_chain", 0, tree="chain"),
        run_scenario("host_lags_device_k1", -1),
        run_scenario("host_lags_device_k2", -2),
        run_scenario("host_ahead_device_k1", +1),
        run_scenario("graph_batch_padding", 0, graph_pad=2),
    ]
    result = {
        "description": (
            "NGRAM verify mask-assembly vs KV read-width oracle; ports "
            "ngram_info.generate_attn_arg_prefill width arithmetic and "
            "ngram_worker ones_masks prefix construction."
        ),
        "draft_token_num": D,
        "scenarios": scenarios,
        "summary": {
            "consistent_pass": all(
                s["verdict"] == "PASS"
                for s in scenarios
                if s["host_delta"] == 0
            ),
            "mismatch_fails": all(
                s["verdict"] == "FAIL"
                for s in scenarios
                if s["host_delta"] != 0
            ),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

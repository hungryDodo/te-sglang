#!/usr/bin/env python3
"""G3 t9: in-place KV-compaction ordering oracle for the NGRAM accept path.

Pure stdlib (no torch/numpy in the sandbox). Executes, for every well-formed
BFS tree shape (n=6) and every monotone accepted path:

  * a transliteration of the documented reconstruct contract
    (sglang/kernels/ops/speculative/reconstruct_tree.py:31-41, 84-105);
  * a transliteration of the CUDA accept loop
    (sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh:229-255);
  * the REAL chunk loop of MHATokenToKVPool._move_kv_cache_impl extracted
    from memory_pool.py via ast and executed with a recording stub;
  * two candidate fill-layout models for fill_front/fill_accept
    (dense pack vs positional with zero tail) and two copy orders
    (tiled load-all-then-store-all, and a racy read-after-write control).

The independently computed golden is derived ONLY from the accepted node
sequence: after compaction committed slot j (j < lens) must hold the KV of
accepted tree node s_j, and slot 0 (the pending root) must keep its own KV.

Exit 0 on success; raw output is the caller's --out JSON.
"""

import argparse
import ast
import itertools
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_POOL = os.path.join(REPO, "python/sglang/srt/mem_cache/memory_pool.py")
EAGLE_CUH = os.path.join(
    REPO, "python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh"
)
RECONSTRUCT = os.path.join(
    REPO, "python/sglang/kernels/ops/speculative/reconstruct_tree.py"
)
CACHE_LOCS = os.path.join(REPO, "python/sglang/kernels/ops/speculative/cache_locs.py")


# ---------------------------------------------------------------- source anchors
def anchor(path, needles):
    text = open(path, encoding="utf-8").read()
    missing = [n for n in needles if n not in text]
    return {"path": os.path.relpath(path, REPO), "needles": needles, "missing": missing}


def extract_chunk_loop(path):
    """Extract the real `for start in range(0, N, cap)` loop from the pool."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            if node.target.id != "start":
                continue
            seg = ast.get_source_segment(src, node)
            if seg and "range(0, N, cap)" in seg.replace("\n", " "):
                found.append(seg)
    if len(found) != 1:
        raise SystemExit(f"chunk loop extraction failed: {len(found)} candidates")
    seg = found[0]
    calls = []
    ns = {"N": None, "cap": None, "min": min, "range": range,
          "next_power_of_2": lambda x: 1 << (x - 1).bit_length(),
          "copy_all_layer_kv_cache_func": None}
    return seg, calls, ns


# ------------------------------------------------------------- tree / contract
def ancestors_from_parents(n, parents):
    """Independent ground truth: ancestors[i][j] iff j is a strict ancestor of i."""
    anc = [[False] * n for _ in range(n)]
    for i in range(n):
        j = parents[i]
        while j >= 0:
            anc[i][j] = True
            j = parents[j]
    return anc


def mask_from_parents(n, parents):
    return [[1 if (j == i or ancestors_from_parents(n, parents)[i][j]) else 0 for j in range(n)] for i in range(n)]


def reconstruct_contract(mask, n, seq_len):
    """Executed transliteration of the documented contract (reconstruct_tree.py).

    positions[tid] = #strict ancestors + seq_len
    retrieve_index[b,tid] = b*n + tid          (b = 0 here)
    next_token[tid]  = min{k>tid : mask[k][tid]} else -1
    parent[tid]      = max{j<tid : mask[tid][j]} else -1
    next_sibling[tid]= min{k>tid : parent[k]==parent[tid]} else -1 (parent<0 -> -1)
    """
    anc = [[bool(mask[i][j]) and j < i for j in range(n)] for i in range(n)]
    positions = [sum(anc[i]) + seq_len for i in range(n)]
    retrieve = list(range(n))
    parent = [max([j for j in range(i) if anc[i][j]], default=-1) for i in range(n)]
    next_token = [
        min([k for k in range(tid + 1, n) if anc[k][tid]], default=-1) for tid in range(n)
    ]
    next_sibling = [
        (
            min(
                [k for k in range(tid + 1, n) if parent[k] == parent[tid]],
                default=-1,
            )
            if parent[tid] >= 0
            else -1
        )
        for tid in range(n)
    ]
    return positions, retrieve, parent, next_token, next_sibling


def independent_contract_check(mask, n, seq_len, positions, parent, next_token, next_sibling):
    """Recompute every field by a DIFFERENT method (mask-column walk / BFS depth)."""
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1 if parent[i] >= 0 else 0
    ok_pos = all(positions[i] == seq_len + depth[i] for i in range(n))
    children = {i: [] for i in range(n)}
    for i in range(n):
        if parent[i] >= 0:
            children[parent[i]].append(i)
    ok_next = all(
        next_token[i] == (children[i][0] if children[i] else -1) for i in range(n)
    )
    ok_sib = True
    for i in range(n):
        if parent[i] < 0:
            ok_sib &= next_sibling[i] == -1
        else:
            later = [k for k in children[parent[i]] if k > i]
            ok_sib &= next_sibling[i] == (later[0] if later else -1)
    # mask rows must agree with the parent chains
    anc = ancestors_from_parents(n, parents_for_mask(mask, n))
    ok_mask = all(
        bool(mask[i][j]) == (anc[i][j] or i == j) for i in range(n) for j in range(n)
    )
    return {
        "positions_match_bfs_depth": ok_pos,
        "next_token_match_children": ok_next,
        "next_sibling_match_children": ok_sib,
        "mask_matches_parent_chains": ok_mask,
    }


def parents_for_mask(mask, n):
    """Recover parent[] from the lower-triangular part of a mask."""
    parents = []
    for i in range(n):
        js = [j for j in range(i) if mask[i][j]]
        parents.append(max(js) if js else -1)
    return parents


# ------------------------------------------------------------- accept kernel
def accept_loop(candidates, tpred, next_token, next_sibling, retrieve, n):
    """Transliteration of eagle.cuh:229-255 (VerifyTreeGreedy)."""
    predicts = [None] * n
    accept_index = [-1] * n
    last = retrieve[0]
    accept_index[0] = last
    num_accepted = 0
    cur = 0
    for _j in range(1, n):
        cur = next_token[cur]
        while cur != -1:
            draft_index = retrieve[cur]
            draft_token_id = candidates[cur]
            target_token_id = tpred[last]
            if draft_token_id == target_token_id:
                predicts[last] = target_token_id
                num_accepted += 1
                accept_index[num_accepted] = draft_index
                last = draft_index
                break
            else:
                cur = next_sibling[cur]
        if cur == -1:
            break
    predicts[last] = tpred[last]
    return predicts, accept_index[: num_accepted + 1], num_accepted


def enumerate_monotone_paths(n, parent):
    """All root-to-node paths expressed as strictly increasing node indices."""
    children = {i: [] for i in range(n)}
    for i in range(n):
        if parent[i] >= 0:
            children[parent[i]].append(i)
    paths = []

    def walk(node, acc):
        acc = acc + [node]
        paths.append(acc)
        for c in children[node]:
            walk(c, acc)

    walk(0, [])
    return paths


# ------------------------------------------------------------- copy models
def golden_after_compaction(window_slots, path):
    """slot j (j<len(path)) holds KV of accepted node path[j]; others untouched."""
    buf = {s: ("kv", s) for s in window_slots}
    for j, s in enumerate(path):
        buf[window_slots[j]] = ("kv", window_slots[s])
    return buf


def run_fill_and_copy(window_slots, accept_index, lens, layout, scheme, cap=4):
    """window_slots = absolute slots of this request's verify window (local j -> slot j).

    accept_index[j] = local tree row of the j-th committed token (incl. root).
    Returns (final buffer, duplicate_destination_count).

    layout 'dense': both arrays packed to sum(lens); tail = (0,0) identity.
    layout 'positional_zerotail': tgt[j]=j for j<lens else 0;
        src[j]=accept_index[j] for j<lens else window_slots[-1]  (-1 sentinel).
    scheme 'tiled' : per chunk, load all sources then store all targets.
    scheme 'racy'  : per chunk, immediate read-after-write (negative control).
    """
    buf = {s: ("kv", s) for s in window_slots}
    entries = []  # (tgt_slot, src_slot)
    if layout == "dense":
        for j in range(lens):
            entries.append((window_slots[j], window_slots[accept_index[j]]))
        while len(entries) < len(window_slots):
            entries.append((window_slots[0], window_slots[0]))  # zeroed tail
    else:
        for j in range(len(window_slots)):
            if j < lens:
                entries.append((window_slots[j], window_slots[accept_index[j]]))
            else:
                entries.append((window_slots[0], window_slots[-1]))
    by_dst = {}
    for t, s in entries:
        by_dst.setdefault(t, set()).add(s)
    dup = sum(1 for v in by_dst.values() if len(v) > 1)
    for start in range(0, len(entries), cap):
        chunk = entries[start : start + cap]
        if scheme == "tiled":
            vals = [buf[s] for _, s in chunk]
            for (t, _), v in zip(chunk, vals):
                buf[t] = v
        else:
            for t, s in chunk:
                buf[t] = buf[s]
    return buf, dup


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    n = args.n

    anchors = [
        anchor(EAGLE_CUH, ["last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens]",
                           "predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx]"]),
        anchor(RECONSTRUCT, ["positions = depth.to(tl.int64) + seq_len",
                             "retrieve_index_ptr + out, (b * n + idx)",
                             "first_child = tl.min(tl.where(ancestors, row, n), axis=0)"]),
        anchor(MEMORY_POOL, ["for start in range(0, N, cap)",
                             "tgt_loc[start:end]",
                             "src_loc[start:end]"]),
        anchor(CACHE_LOCS, ['"""Pack the first ``num_front[i]`` slots of request i\'s window in ``out_cache_loc``."""']),
    ]
    if any(a["missing"] for a in anchors):
        raise SystemExit(f"anchor failure: {anchors}")

    chunk_seg, chunk_calls, chunk_ns = extract_chunk_loop(MEMORY_POOL)

    # 1) all well-formed BFS trees: parent[i] in {-1} U [0, i-1]
    trees = []
    choices = [range(i) for i in range(1, n)]
    for combo in itertools.product(*choices):
        parents = [-1] + list(combo)
        trees.append(parents)

    contract_violations = []
    path_stats = {"trees": len(trees), "paths": 0, "max_depth": 0,
                  "paths_with_src_gt_dst": 0, "max_gap": 0}
    layout_results = {}
    for layout in ("dense", "positional_zerotail"):
        for scheme in ("tiled", "racy"):
            layout_results[f"{layout}/{scheme}"] = {
                "paths_checked": 0, "corrupt_paths": 0, "first_corrupt": None,
                "duplicate_destination_paths": 0,
            }

    rng_state = 12345

    def rand(limit):
        nonlocal rng_state
        rng_state = (1103515245 * rng_state + 12345) & 0x7FFFFFFF
        return rng_state % limit

    for parents in trees:
        mask = mask_from_parents(n, parents)
        positions, retrieve, parent, next_token, next_sibling = reconstruct_contract(
            mask, n, seq_len=100
        )
        chk = independent_contract_check(
            mask, n, 100, positions, parent, next_token, next_sibling
        )
        if not all(chk.values()):
            contract_violations.append({"parents": parents, "check": chk})
        paths = enumerate_monotone_paths(n, parent)
        # 2) execute the real accept loop on deterministic random token labels and
        #    require every reachable accept_index to equal a prefix of a path.
        path_set = {tuple(p) for p in paths}
        for _trial in range(4):
            cands = [rand(4) for _ in range(n)]
            tpred = [rand(5) - 1 for _ in range(n)]  # -1 = never matches
            _p, acc_idx, nacc = accept_loop(
                cands, tpred, next_token, next_sibling, retrieve, n
            )
            if tuple(acc_idx) not in path_set:
                raise SystemExit(
                    f"accept loop produced non-path prefix {acc_idx} for parents={parents}"
                )
            if any(acc_idx[i] >= acc_idx[i + 1] for i in range(len(acc_idx) - 1)):
                raise SystemExit(f"accept_index not strictly increasing: {acc_idx}")
        for path in paths:
            path_stats["paths"] += 1
            path_stats["max_depth"] = max(path_stats["max_depth"], len(path) - 1)
            window = list(range(50, 50 + n))
            if any(path[j] > j for j in range(1, len(path))):
                path_stats["paths_with_src_gt_dst"] += 1
                path_stats["max_gap"] = max(
                    path_stats["max_gap"], max(path[j] - j for j in range(len(path)))
                )
            golden = golden_after_compaction(window, path)
            for layout in ("dense", "positional_zerotail"):
                for scheme in ("tiled", "racy"):
                    key = f"{layout}/{scheme}"
                    res = layout_results[key]
                    res["paths_checked"] += 1
                    got, dup = run_fill_and_copy(window, path, len(path), layout, scheme)
                    if dup:
                        res["duplicate_destination_paths"] += 1
                    if got != golden:
                        res["corrupt_paths"] += 1
                        if res["first_corrupt"] is None:
                            res["first_corrupt"] = {
                                "parents": parents, "path": path,
                                "expected_slot0": golden[window[0]],
                                "got_slot0": got[window[0]],
                            }

    # 3) the extracted chunk loop must reproduce the same chunk boundaries on a
    #    realistic flat size (bs=1, n draft tokens -> size bs*n).
    chunk_ns["N"] = 40
    chunk_ns["cap"] = 16
    chunk_ns["tgt_loc"] = list(range(40))
    chunk_ns["src_loc"] = list(range(100, 140))
    chunk_ns["self"] = type("S", (), {"_move_native_fp4_scales": lambda *a: None,
                                     "data_ptrs": None, "data_strides": None})()
    chunk_ns["cfg"] = None
    calls = []
    chunk_ns["copy_all_layer_kv_cache_func"] = (
        lambda data_ptrs, strides, tgt, src, cnt, up, cfg: calls.append(
            (list(tgt), list(src), cnt, up)
        )
    )
    try:
        exec(compile(chunk_seg, "<extracted-chunk-loop-2>", "exec"), chunk_ns)  # noqa: S102
        chunk_boundaries = [c[0][0] for c in calls]
    except Exception as e:  # keep core results if only the chunk replay fails
        chunk_boundaries = [f"ERROR:{type(e).__name__}:{e}"]

    out = {
        "oracle": "g3-t9 compaction-order",
        "source_anchors": anchors,
        "extracted_chunk_loop": chunk_seg,
        "chunk_dispatch": {
            "N": 40, "cap": 16,
            "chunk_first_indices": chunk_boundaries,
            "chunk_lengths": [c[2] for c in calls],
            "increasing_order": chunk_boundaries == sorted(chunk_boundaries),
        },
        "tree_enumeration": path_stats,
        "contract_checks": {
            "trees_checked": len(trees),
            "violations": contract_violations[:5],
            "violation_count": len(contract_violations),
        },
        "layout_scheme_results": layout_results,
        "theorem": (
            "Accept indexes are strictly increasing (next_token/next_sibling only "
            "point to larger tid), so for every read at packed index i of source "
            "slot s_i and every write at packed index j of slot s_i=j we have "
            "i < j. Tiled copy loads each chunk before storing and chunks run in "
            "increasing order, so the read always precedes the write: in-place "
            "compaction cannot corrupt KV. The racy control must fail to show the "
            "oracle detects a real ordering hazard; the zero-tail layout is a "
            "negative control for duplicate destinations."
        ),
        "caveat": (
            "Transliterations of documented contracts, not executed GPU kernels. "
            "The dense-pack layout model is inferred from "
            "fill_front_out_cache_loc_func -> assign_extend_cache_locs "
            "(cache_locs.py:752-780 vs kernel 500-531, which packs at the cumulative "
            "offset); fill_accept_out_cache_loc_func's layout was not re-read this "
            "trial, so the positional_zerotail branch states the consequence if that "
            "fill is not dense."
        ),
        "conclusion": None,
    }
    dense_tiled = layout_results["dense/tiled"]
    racy = layout_results["dense/racy"]
    zero_tail = layout_results["positional_zerotail/tiled"]
    out["conclusion"] = {
        "dense_tiled_corrupt_paths": dense_tiled["corrupt_paths"],
        "dense_racy_control_corrupt_paths": racy["corrupt_paths"],
        "zero_tail_corrupt_paths": zero_tail["corrupt_paths"],
        "zero_tail_duplicate_destination_paths": zero_tail["duplicate_destination_paths"],
        "verdict": (
            "PASS_NEGATIVE" if dense_tiled["corrupt_paths"] == 0
            and racy["corrupt_paths"] > 0
            else "REVIEW"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["conclusion"], indent=2))
    print("chunk_first_indices:", chunk_boundaries)
    print("paths:", path_stats["paths"], "trees:", path_stats["trees"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

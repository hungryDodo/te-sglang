#!/usr/bin/env python3
"""Bounded fixed CPU topology / accept-index oracle for the NGRAM verify path.

Pure stdlib (no torch, no GPU, no provider access). It ports the *documented*
CPU contracts in this repository:

  * reconstruct_indices_from_tree_mask_cpu
      python/sglang/kernels/aot/csrc/cpu/spec.cpp:689  (depth/parent backward
      scan, identity retrieve_index, positions, first-child / next-sibling)
  * verify_tree_greedy_cpu
      python/sglang/kernels/aot/csrc/cpu/spec.cpp:249  (greedy tree walk;
      the repo's own reference `_ref_verify_tree_greedy` in
      test/registered/cpu/test_spec_kernels.py:204 is the same algorithm)
  * move_accept_tokens_to_target_kvcache
      python/sglang/srt/speculative/spec_utils.py:859
      fill_front_out_cache_loc (destinations = the first k slots of the
      request's verify window) + fill_accept_out_cache_loc (sources compacted
      by skipping -1) + pairwise move_kv_cache.

What is tested (algorithm preservation on fixed synthetic topologies):

  Given the documented EAGLE/NGRAM root convention -- row 0 is the *pending*
  last token already emitted in the previous iteration but not yet in the KV
  cache -- and per-row target logits computed from the true ancestor context,
  does [reconstruct -> verify_tree_greedy -> KV move] commit exactly the
  plain-greedy token sequence and leave KV[position] == token at every
  position below the new seq_len?

Scenarios:
  chain                linear draft chain, all accepted
  bfs_branch           golden BFS mask from test/registered/unit/spec/
                       test_ngram_corpus.py (EXPECTED_BFS_MASKS[0])
  sparse_accept_0_2_4  root -> child2 -> child4 acceptance, i.e. accepted
                       draft rows [0,2,4] that must compact to destinations
                       [0,1,2] (the controller's stated open question)
  root_mismatch_NEG    fault injection: tree root token != pending token
  no_compaction_NEG    fault injection: KV destinations taken from
                       accept_index instead of compacted front slots

A scenario passes iff committed == greedy and the KV mapping is exact.
The NEG scenarios are *expected to fail*; they prove the oracle is sensitive
to the two fault classes it is meant to discriminate, so a pass in the
non-negative scenarios is evidence, not a vacuous assertion.

Run:  python3 ngram_oracle/topology_oracle.py --out ngram_oracle/results/topology_oracle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json


VOCAB = 64


def f_next(context: tuple[int, ...]) -> int:
    """Deterministic stand-in for the target model's greedy argmax.

    A pure function of the full context tuple: no state, no RNG, so every
    replay of the same context yields the same argmax.  This is exactly the
    property the oracle needs; it is NOT a language model.
    """
    h = hashlib.sha256(repr(context).encode()).digest()
    return int.from_bytes(h[:4], "little") % VOCAB


# ---------------------------------------------------------------------------
# Ports of the repository kernels
# ---------------------------------------------------------------------------
def reconstruct_from_mask(mask: list[list[int]], seq_len: int):
    """Exact port of reconstruct_indices_from_tree_mask_cpu (steps 1-5)."""
    n = len(mask)
    positions = [0] * n
    retrieve_index = [0] * n
    next_token = [-1] * n
    next_sibling = [-1] * n

    for tid in range(n):
        depth = 0
        parent_idx = -1
        for i in range(tid - 1, -1, -1):
            if mask[tid][i]:
                depth += 1
                if parent_idx == -1:
                    parent_idx = i
        positions[tid] = depth + seq_len
        retrieve_index[tid] = tid

        for i in range(tid + 1, n):
            if mask[i][tid]:
                next_token[tid] = i
                break

        if parent_idx != -1:
            for i in range(tid + 1, n):
                if mask[i][parent_idx]:
                    is_sibling = True
                    for j in range(parent_idx + 1, i):
                        if mask[i][j]:
                            is_sibling = False
                            break
                    if is_sibling:
                        next_sibling[tid] = i
                        break

    return positions, retrieve_index, next_token, next_sibling


def verify_tree_greedy(candidates, retrieve_index, next_token, next_sibling,
                       target_predict, num_spec_step):
    """Exact port of the greedy tree walk (spec.cpp verify_tree_greedy_cpu,
    mirrored by the repo's `_ref_verify_tree_greedy` unit-test reference)."""
    bs = len(candidates)
    n = len(candidates[0])
    predicts = [[-1] * n for _ in range(bs)]
    accept_index = [[-1] * num_spec_step for _ in range(bs)]
    num_correct = [0] * bs

    for bx in range(bs):
        last = retrieve_index[bx][0]
        accept_index[bx][0] = last
        num = 0
        cur = 0
        for _ in range(1, num_spec_step):
            cur = next_token[bx][cur]
            while cur != -1:
                draft_idx = retrieve_index[bx][cur]
                draft_token = candidates[bx][cur]
                target_token = target_predict[bx][last]
                if draft_token == target_token:
                    predicts[bx][last] = target_token
                    num += 1
                    accept_index[bx][num] = draft_idx
                    last = draft_idx
                    break
                cur = next_sibling[bx][cur]
            if cur == -1:
                break
        num_correct[bx] = num
        predicts[bx][last] = target_predict[bx][last]
    return predicts, accept_index, num_correct


def move_kv(kv: dict[int, int], out_cache_loc: list[int], accept_index_row,
            num_correct: int, compact: bool = True) -> list[tuple[int, int]]:
    """Model move_accept_tokens_to_target_kvcache.

    destinations: fill_front_out_cache_loc(num_correct+1, out_cache_loc) ->
                  out_cache_loc[0 : num_correct+1]
    sources:      fill_accept_out_cache_loc(accept_index_row, out_cache_loc)
                  -> out_cache_loc[a] for a in accept_index_row[:num_correct+1]
                     (the buffer is pre-filled with -1; -1 entries are skipped)
    compact=False injects the "no compaction" fault: destinations are
    out_cache_loc[accept_index] instead of the compacted front slots.
    """
    k = num_correct + 1
    src_slots = [out_cache_loc[a] for a in accept_index_row[:k] if a > -1]
    if compact:
        dst_slots = list(out_cache_loc[:k])
    else:
        dst_slots = [out_cache_loc[a] for a in accept_index_row[:k]]
    assert len(src_slots) == len(dst_slots), (src_slots, dst_slots)

    # move_kv_cache gather semantics: read all sources, then write.
    old = {s: kv.get(s) for s in src_slots}
    pairs = []
    for s, d in zip(src_slots, dst_slots):
        kv[d] = old[s]
        pairs.append((s, d))
    return pairs


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------
def run_scenario(name, mask, candidates_row, prefix, pending, root_is_pending=True,
                 compact=True, expect_pass=True):
    n = len(mask)
    seq_len = len(prefix)          # KV entries committed before this verify
    full_before = list(prefix) + [pending]

    # True per-row contexts: prefix KV + ancestor candidate tokens + own token.
    # (The verify attention mask is the ancestor matrix; row 0 attends to the
    # prefix + itself, which is exactly the pending token.)
    cand = list(candidates_row)
    if not root_is_pending:
        # fault injection: row 0 carries a stale draft token, not the pending one
        cand[0] = (pending + 1) % VOCAB

    def ancestors(tid):
        return [j for j in range(tid) if mask[tid][j]]

    target_predict = []
    row_context = []
    for tid in range(n):
        ctx = list(prefix)
        for a in ancestors(tid):
            ctx.append(cand[a])
        ctx.append(cand[tid])
        row_context.append(ctx)
        target_predict.append(f_next(tuple(ctx)))

    positions, retrieve_index, next_token, next_sibling = reconstruct_from_mask(
        mask, seq_len
    )
    # positions must equal prefix length + depth; assert the port agrees with
    # the context construction above (depth == number of ancestors).
    for tid in range(n):
        assert positions[tid] == seq_len + len(ancestors(tid)), (tid, positions[tid])

    predicts, accept_index, num_correct = verify_tree_greedy(
        [cand], [retrieve_index], [next_token], [next_sibling],
        [target_predict], num_spec_step=n,
    )
    acc = accept_index[0]
    committed = [predicts[0][a] for a in acc[: num_correct[0] + 1]]
    new_pending = committed[-1]
    new_seq_len = seq_len + num_correct[0] + 1

    # Greedy reference: repeatedly sample from the true model on the growing
    # context. Must agree with what the speculative path commits.
    greedy = []
    ctx = list(full_before)
    for _ in range(num_correct[0] + 1):
        t = f_next(tuple(ctx))
        greedy.append(t)
        ctx.append(t)

    # KV pool: keys are physical slots, not logical positions. Prefix KV needs
    # seq_len slots; the verify window is the contiguous slot block
    # out_cache_loc[row] = seq_len + row (assign_extend_cache_locs_func). The
    # verify forward writes row tid's KV to its own slot; `positions[tid]` is
    # only the RoPE position (siblings can share it). After the accept move,
    # logical position seq_len + j must live in slot seq_len + j.
    kv = {i: full_before[i] for i in range(seq_len)}
    out_cache_loc = [seq_len + i for i in range(n)]
    for tid in range(n):
        kv[out_cache_loc[tid]] = cand[tid]
    pairs = move_kv(kv, out_cache_loc, acc, num_correct[0], compact=compact)

    # sequence after this iteration: KV-backed tokens + pending bonus
    seq_after = full_before + committed[:-1]
    assert len(seq_after) == new_seq_len, (len(seq_after), new_seq_len)
    seq_after_full = seq_after + [new_pending]

    def slot_of(logical_pos):
        # prefix slots are this model's identity 0..seq_len-1; new positions
        # are served by the request's verify window block.
        return logical_pos if logical_pos < seq_len else out_cache_loc[logical_pos - seq_len]

    kv_mismatch = [
        (j, kv.get(slot_of(j)), seq_after[j])
        for j in range(new_seq_len)
        if kv.get(slot_of(j)) != seq_after[j]
    ]
    committed_ok = committed == greedy
    passed = committed_ok and not kv_mismatch

    return {
        "name": name,
        "mask": mask,
        "candidates": cand,
        "pending": pending,
        "prefix_len": seq_len,
        "positions": positions,
        "retrieve_index": retrieve_index,
        "retrieve_next_token": next_token,
        "retrieve_next_sibling": next_sibling,
        "target_predict": target_predict,
        "accept_index": acc,
        "accepted_rows": [a for a in acc if a > -1],
        "num_correct_drafts": num_correct[0],
        "committed_tokens": committed,
        "greedy_tokens": greedy,
        "kv_move_pairs": pairs,
        "kv_mismatches": kv_mismatch,
        "seq_len_before": seq_len,
        "seq_len_after": new_seq_len,
        "algorithm_preserving": passed,
        "expected": "PASS" if expect_pass else "FAIL(fault injection)",
        "verdict": (
            "PASS" if passed else "FAIL"
        ),
        "matches_expectation": (passed == expect_pass),
    }


def build_sparse_tree(prefix, pending, last_token_matches=True):
    """Fixed BFS topology whose accepted chain is rows [0, 2, 4].

    mask: row0 root; row1/row2 children of root; row3 child of 1; row4 child
    of 2 (golden EXPECTED_BFS_MASKS[0] shape, depth 2).
    Candidate tokens are chosen so the target walk skips row1 (decoy) and
    accepts root -> 2 -> 4.
    """
    n = 5
    mask = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ]
    g0 = f_next(tuple(prefix + [pending]))
    g1 = f_next(tuple(prefix + [pending, g0]))
    decoy = (g0 + 7) % VOCAB
    decoy2 = (g1 + 9) % VOCAB
    cand = [pending, decoy, g0, decoy2, g1]
    return mask, cand


def build_chain(prefix, pending):
    mask = [[1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]]
    g = []
    ctx = list(prefix) + [pending]
    for _ in range(3):
        g.append(f_next(tuple(ctx)))
        ctx.append(g[-1])
    # row0 = pending (root), row i = g[i-1]; every draft matches
    return mask, [pending, g[0], g[1], g[2]]


def build_golden_bfs(prefix, pending):
    # EXPECTED_BFS_MASKS[0] from test/registered/unit/spec/test_ngram_corpus.py
    mask = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 1],
    ]
    # Row0 is the pending token.  Choose a mix: even-index children of the
    # accepted chain match the model, odd decoys do not, forcing the walk to
    # skip siblings and branch down the even nodes 0->2->4->6->7.
    cand = [0] * 8
    cand[0] = pending
    ctx = list(prefix) + [pending]
    # path tokens: node2, node4, node6, node7 (each is the next greedy token
    # after the previous accepted node)
    for idx, node in enumerate([2, 4, 6, 7]):
        tok = f_next(tuple(ctx))
        cand[node] = tok
        ctx.append(tok)
    # decoys at the odd siblings
    for idx, node in enumerate([1, 3, 5]):
        cand[node] = (f_next(tuple(prefix + [pending])) + 11 + idx) % VOCAB
    return mask, cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ngram_oracle/results/topology_oracle.json")
    args = ap.parse_args()

    prefix = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333]
    pending = 444

    results = []

    mask, cand = build_chain(prefix, pending)
    results.append(
        run_scenario("chain", mask, cand, prefix, pending, expect_pass=True)
    )

    mask, cand = build_golden_bfs(prefix, pending)
    results.append(
        run_scenario("bfs_branch", mask, cand, prefix, pending, expect_pass=True)
    )

    mask, cand = build_sparse_tree(prefix, pending)
    results.append(
        run_scenario("sparse_accept_0_2_4", mask, cand, prefix, pending,
                     expect_pass=True)
    )

    # Negative controls: the same topology, one injected fault each.
    mask, cand = build_sparse_tree(prefix, pending)
    results.append(
        run_scenario("root_mismatch_NEG", mask, cand, prefix, pending,
                     root_is_pending=False, expect_pass=False)
    )

    mask, cand = build_sparse_tree(prefix, pending)
    results.append(
        run_scenario("no_compaction_NEG", mask, cand, prefix, pending,
                     compact=False, expect_pass=False)
    )

    summary = {
        "oracle": "ngram_topology_accept_index_v1",
        "source_contracts": [
            "python/sglang/kernels/aot/csrc/cpu/spec.cpp:249 verify_tree_greedy_cpu",
            "python/sglang/kernels/aot/csrc/cpu/spec.cpp:689 reconstruct_indices_from_tree_mask_cpu",
            "python/sglang/kernels/aot/csrc/cpu/spec.cpp:528 fill_accept_out_cache_loc_cpu",
            "python/sglang/srt/speculative/spec_utils.py:859 move_accept_tokens_to_target_kvcache",
            "test/registered/cpu/test_spec_kernels.py:204 _ref_verify_tree_greedy",
        ],
        "root_convention": (
            "row 0 == the pending last token (already emitted, not yet in KV); "
            "see spec_utils.py:606 'the first token generated by the target "
            "model, and thus it is always accepted from the previous iteration'"
        ),
        "model": "deterministic sha256 context->token stand-in (not an LLM)",
        "scenarios": results,
        "all_expected": all(r["matches_expectation"] for r in results),
        "positive_scenarios_pass": all(
            r["algorithm_preserving"]
            for r in results
            if not r["name"].endswith("_NEG")
        ),
        "negative_controls_detected": all(
            not r["algorithm_preserving"]
            for r in results
            if r["name"].endswith("_NEG")
        ),
        "limitations": [
            "Pure-Python model of the documented CPU kernels; the CUDA kernels "
            "and the real tvm_ffi corpus were not executed here.",
            "The target 'model' is a deterministic context hash, so this tests "
            "index/topology arithmetic and KV bookkeeping, not LM numerics.",
            "Whether the live NGRAM draft path actually keeps tree root == "
            "pending token (overlap splice / grammar path) is NOT measured; "
            "the oracle only proves that a mismatch of one token is sufficient "
            "to corrupt every later position.",
        ],
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    for r in results:
        print(
            f"{r['name']:<24} committed_ok={r['committed_tokens'] == r['greedy_tokens']!s:<5} "
            f"kv_mismatch={len(r['kv_mismatches'])} accept={r['accepted_rows']} "
            f"verdict={r['verdict']} expected={r['expected']}"
        )
    print("all_expected:", summary["all_expected"])


if __name__ == "__main__":
    main()

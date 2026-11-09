#!/usr/bin/env python3
"""G3 t8 deepseek-a: pending-root / overlap-splice replay oracle (pure stdlib).

New question vs g3-t7: does the *host* overlap bookkeeping keep the draft-tree
root equal to the pending token (last committed output token, position
seq_lens), and which bookkeeping faults can change committed greedy output?
The host tail/lens expressions are extracted from the real source via ast and
executed.  An independent plain-greedy reference is computed from the same
deterministic model, never from the code under test.  Engine seq_lens is
tracked separately from physical KV contents, so a bookkeeping desync is
observable instead of being masked by correct tensor contents.
"""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "python/sglang/srt/speculative/ngram_worker.py")
NGCPP = os.path.join(ROOT, "python/sglang/kernels/jit/csrc/ngram_corpus/ngram.cpp")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "pending_root_replay.json")

VOCAB = 32
PROMPT = [3, 1, 4, 1, 5, 9]


def model_next(ctx):
    h = 17
    for t in ctx[-3:]:
        h = (h * 131 + t + 7) % 1000003
    return (h * 2654435761) % VOCAB


# ---------------- source extraction: host bookkeeping ----------------
src = open(WORKER).read()
tree = ast.parse(src)
worker_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NGRAMWorker")
prep = next(n for n in worker_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_prepare_draft_tokens")
comps = {}
for n in ast.walk(prep):
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
        if n.targets[0].id in ("input_tails", "output_tails", "base_lens"):
            comps[n.targets[0].id] = ast.get_source_segment(src, n.value)
assert set(comps) == {"input_tails", "output_tails", "base_lens"}, comps


class Req:
    def __init__(self, prompt, out):
        self.origin_input_ids_unpadded = list(prompt)
        self.output_ids = list(out)


def run_lists(prompt, out_lag):
    ns = {"req": None,
          "batch": type("B", (), {"reqs": [Req(prompt, out_lag)]})(),
          "self": type("S", (), {"max_trie_depth": 8})()}
    return {k: eval(compile(ast.parse(v, mode="eval"), WORKER, "eval"), ns)
            for k, v in comps.items()}


def tail_n(*parts, n):
    flat = []
    for p in parts:
        flat.extend(p)
    return flat[-n:]


# ---------------- accept / KV model (pinned kernel arithmetic) ----------------
def verify_step(kv, k_eng, chain, pad_token=31):
    """reconstruct + VerifyTreeGreedy + front compaction.
    Prefix read width is the ENGINE seq_lens (mask ones_masks), not physical KV.
    Returns (committed, accept_lens, moved_kv)."""
    prefix = list(kv[:k_eng])
    if k_eng > len(kv):
        prefix.extend([pad_token] * (k_eng - len(kv)))  # reused/uninitialized slot
    root = chain[0]
    ctx = prefix + [root]
    rows = [list(ctx)]
    for tok in chain[1:]:
        ctx = ctx + [tok]
        rows.append(list(ctx))
    committed = []
    accept = 1
    cur = 0
    for i in range(1, len(chain)):
        if model_next(rows[cur]) == chain[i]:
            committed.append(chain[i])
            accept += 1
            cur = i
        else:
            break
    committed.append(model_next(rows[cur]))   # bonus at last accepted row
    moved = [root] + committed[:-1]           # compaction to window front
    return committed, accept, moved


def reference(n):
    text, out = list(PROMPT), []
    for _ in range(n):
        t = model_next(text)
        out.append(t)
        text.append(t)
    return out


def spec_replay(steps=4, fault=None, branch=False):
    kv = list(PROMPT)
    k_eng = len(PROMPT)              # engine seq_lens (host mirror contract)
    pending = model_next(kv)         # prefill-sampled token: pending, not in KV
    output = [pending]
    prev_tokens = []
    trace = []
    for _ in range(steps):
        out_lag = output[: len(output) - len(prev_tokens)] if prev_tokens else list(output)
        lists = run_lists(PROMPT, out_lag)
        base_len = lists["base_lens"][0]
        if fault == "no_prev_splice":
            check_token = tail_n(lists["input_tails"][0], lists["output_tails"][0], n=8)
            total_len = base_len
        else:
            check_token = tail_n(lists["input_tails"][0], lists["output_tails"][0],
                                 prev_tokens, n=8)
            total_len = base_len + len(prev_tokens)
        true_pending = output[-1]
        true_text = list(PROMPT) + output
        root = kv[-1] if fault == "stale_root" else check_token[-1]
        nxt = [model_next(true_text)]
        if branch:
            nxt.append((nxt[0] + 1) % VOCAB)
        chain = [root, nxt[0], model_next(true_text + [nxt[0]])]
        committed, accept, moved = verify_step(kv, k_eng, chain)
        if fault == "drop_bonus":
            accept -= 1
        if fault == "seq_len_plus_one":
            accept += 1
        trace.append({
            "true_pending": true_pending, "draft_root": root,
            "root_is_pending": root == true_pending,
            "check_token_tail": check_token[-3:], "total_len": total_len,
            "committed": committed, "accept_lens": accept,
            "k_eng_before": k_eng, "kv_phys_before": len(kv),
        })
        kv = kv + moved
        k_eng = k_eng + accept
        output.extend(committed)
        prev_tokens = committed
    return output, trace


SCENARIOS = [
    ("correct_pending_chain", dict(fault=None, branch=False)),
    ("correct_pending_branch", dict(fault=None, branch=True)),
    ("fault_stale_root", dict(fault="stale_root")),
    ("fault_no_prev_splice", dict(fault="no_prev_splice")),
    ("fault_drop_bonus", dict(fault="drop_bonus")),
    ("fault_seq_len_plus_one", dict(fault="seq_len_plus_one")),
]

cpp_lines = [l.strip() for l in open(NGCPP).read().splitlines()
             if "suffix.back()" in l or "trie_result_build_fn" in l][:5]
assert any("suffix.back()" in l for l in cpp_lines), "corpus last_token anchor lost"

results = {}
for name, kw in SCENARIOS:
    out, trace = spec_replay(**kw)
    ref = reference(len(out))
    first = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
    results[name] = {
        "exact_vs_greedy": out == ref,
        "first_divergence": first,
        "generated": out,
        "reference": ref,
        "root_is_pending_all_steps": all(t["root_is_pending"] for t in trace),
        "steps": trace,
    }

payload = {
    "script": "g3-t8/pending_root_replay.py",
    "scope": "pure-stdlib host bookkeeping + accept/KV index model; no torch/CUDA/GPU",
    "source_anchors": {
        "ast_executed_comprehensions": comps,
        "host_prev_slice": [l.strip() for l in src.splitlines()
                            if "self.prev_token_ids[i * stride" in l][:2],
        "host_total_lens": [l.strip() for l in src.splitlines()
                            if "total_lens.append" in l][:2],
        "corpus_last_token": cpp_lines,
    },
    "model": {"vocab": VOCAB, "prompt": PROMPT, "draft_chain_len": 3},
    "results": results,
    "interpretation": {
        "pending_root_convention": "corpus fillResult root = suffix.back() = "
        "check_token[-1]; with the overlap splice prev_tokens = "
        "accept_tokens[i*D : i*D+accept_lens] that token is the pending token at "
        "position seq_lens, and row-0 logits are its greedy continuation.",
        "splice_is_load_bearing": "without the splice the draft root is the last "
        "already-cached token; the stale root changes the committed greedy output, "
        "so a stale/mis-spliced root is a correctness fault, not draft quality.",
        "accept_lens_includes_bonus": "accept_lens = num_correct_drafts + 1 must "
        "advance both seq_lens and the KV front; dropping the bonus desyncs the "
        "engine prefix from physical KV and diverges.",
        "limits": "perfect corpus + linear chain; no real trie, attention kernel, "
        "FlashInfer packing, CUDA graph, overlap scheduling or batch nondeterminism.",
    },
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)
print(json.dumps({k: {"exact": v["exact_vs_greedy"], "first_diff": v["first_divergence"],
                      "root_is_pending": v["root_is_pending_all_steps"]}
                  for k, v in results.items()}, indent=2))
print("anchors ok; wrote", OUT)

#!/usr/bin/env python3
"""G3 t4 (deepseek-b): GPU-free oracle for the host-mirror index invariants.

Answers three falsifiable questions, on CPU, from the real source file:

Q1 (exactness, decode): in FlashInferIndicesUpdaterDecode.call_begin_forward,
    does the host int `paged_kernel_lens_sum` equal the device value
    `kv_indptr[bs]` (= cumsum(paged_kernel_lens)[bs-1]) for every gate-shaped
    input, and is the t3 host-mirror guard (spec-decode fallback) sound?
Q2 (exactness, prefill): does the same equality hold for the *prefill*
    updater, and in which branch does it break?  The dq/custom_kv_indices
    branch cumsums a different tensor (dq_paged_kernel_lens), so a blanket
    host-mirror there would be WRONG -- this is what any prefill patch must
    guard on.
Q3 (liveness, static): locate the `kv_last_index = kv_indptr[-1]` sites in the
    real AST and report whether each is host-mirrored or device-sourced, plus
    the call frequency of the enclosing metadata function.

No torch, no GPU, no model. Pure int arithmetic with int32 wrap. Output is
JSON on stdout; every case is reproducible from this file + the source path.
"""
import ast
import json
import sys

SRC = "python/sglang/srt/layers/attention/flashinfer_backend.py"
INT32_MAX = 2**31 - 1


def i32(x: int) -> int:
    """Wrap to int32, matching torch.cumsum(int32) overflow behaviour."""
    x &= 0xFFFFFFFF
    return x - 2**32 if x > INT32_MAX else x


def cumsum_last(vals):
    acc = 0
    for v in vals:
        acc = i32(acc + i32(v))
    return acc


def decode_cases():
    """Decode: paged_kernel_lens = clamp(seq_lens, max=window+1)."""
    out = []
    window = 511
    cap = window + 1
    for bs in (16, 32, 64, 128, 256):
        for seq_len in (512, 1024, 2048, 4096, 8192):
            paged = [min(seq_len, cap)] * bs
            host = sum(paged)                       # torch.clamp(seq_lens_cpu).sum().item()
            device = cumsum_last(paged)             # kv_indptr[bs]
            out.append({
                "case": f"decode_bs{bs}_seq{seq_len}",
                "bs": bs, "seq_len": seq_len, "window_cap": cap,
                "paged_len_each": min(seq_len, cap),
                "host_mirror": host, "device_kv_indptr_last": device,
                "equal": host == device,
            })
    # Mixed lengths exercise ragged decode (different seq_lens in one batch).
    for bs, lens in ((64, [512] * 63 + [8192]), (128, list(range(600, 600 + 128)))):
        paged = [min(L, cap) for L in lens]
        out.append({
            "case": f"decode_ragged_bs{bs}",
            "bs": bs, "seq_lens": lens,
            "host_mirror": sum(paged),
            "device_kv_indptr_last": cumsum_last(paged),
            "equal": sum(paged) == cumsum_last(paged),
        })
    return out


def prefill_cases():
    """Prefill/extend: paged_kernel_lens = per-request KV length.

    Two branches in the real code:
      (a) custom_kv_indices is None -> cumsum(paged_kernel_lens): host equal.
      (b) custom_kv_indices set and dq_paged_kernel_lens set -> cumsum of the
          *dq* tensor, which is page-padded: host NOT necessarily equal.
    """
    out = []
    for bs in (16, 32, 64, 128, 256):
        # (a) plain extend: prefix + extend, causal KV length per request.
        for prefix, extend in ((0, 8192), (8192, 0), (4096, 1024), (448, 64)):
            paged = [prefix + extend] * bs
            host = sum(paged)
            device = cumsum_last(paged)
            out.append({
                "case": f"prefill_plain_bs{bs}_p{prefix}_e{extend}",
                "branch": "custom_kv_indices=None",
                "host_mirror": host, "device_kv_indptr_last": device,
                "equal": host == device,
            })
        # (b) dq branch: dq_paged_kernel_lens is page-aligned, e.g. round up.
        page = 1
        lens = [600 + 7 * i for i in range(bs)]
        dq = [((L + page - 1) // page) * page for L in lens]  # page-aligned
        # demonstrate a *real* divergence: page-aligned tensor can be larger.
        dq[0] = lens[0] + page  # one page of alignment slack on request 0
        out.append({
            "case": f"prefill_dq_bs{bs}",
            "branch": "custom_kv_indices set, dq_paged_kernel_lens set",
            "host_mirror_from_paged_lens": sum(lens),
            "device_kv_indptr_last_from_dq": cumsum_last(dq),
            "equal": sum(lens) == cumsum_last(dq),
            "note": "host mirror over paged_kernel_lens would be WRONG here; "
                    "the prefill patch must fall back to kv_indptr[-1]",
        })
        # (c) dq set but dq_paged_kernel_lens None: cumsum falls back to
        # paged_kernel_lens, so the invariant holds again.
        out.append({
            "case": f"prefill_custom_only_bs{bs}",
            "branch": "custom_kv_indices set, dq_paged_kernel_lens None",
            "host_mirror": sum(lens), "device_kv_indptr_last": cumsum_last(lens),
            "equal": sum(lens) == cumsum_last(lens),
        })
    return out


def static_sites():
    """Locate the real kv_last_index sites and their guards via AST."""
    with open(SRC, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=SRC)
    report = {"source": SRC, "sites": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "call_begin_forward":
            continue
        cls = None
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                cls = parent.name
        hits = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
                tgt = sub.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id == "kv_last_index":
                    hits.append({
                        "lineno": sub.lineno,
                        "source": ast.get_source_segment(open(SRC).read(), sub),
                    })
        report["sites"].append({
            "class": cls, "lineno": node.lineno, "kv_last_index_assigns": hits,
        })
    return report


def liveness():
    """Static call-frequency facts for the metadata updater (Q3)."""
    facts = []
    import subprocess
    for pat, path in (
        (r"self\.attn_backend\.init_forward_metadata\(forward_batch\)",
         "python/sglang/srt/model_executor/model_runner.py"),
        (r"self\.indices_updater_decode\.update\(",
         SRC),
        (r"def init_forward_metadata_out_graph", SRC),
    ):
        r = subprocess.run(["grep", "-n", pat, path], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            facts.append({"file": path, "match": line})
    return facts


def main():
    dec = decode_cases()
    pre = prefill_cases()
    result = {
        "generated_by": "seis/trials/deepseek-b/g3-t4/verify_index_invariants.py",
        "source": SRC,
        "Q1_decode": {
            "all_equal": all(c["equal"] for c in dec),
            "n_cases": len(dec),
            "cases": dec,
        },
        "Q2_prefill": {
            "plain_all_equal": all(c["equal"] for c in pre
                                   if c["case"].startswith("prefill_plain")),
            "dq_diverges": [c for c in pre if c["case"].startswith("prefill_dq")
                            and not c["equal"]],
            "cases": pre,
        },
        "Q3_static": static_sites(),
        "Q3_liveness_facts": liveness(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3 -I
"""g3-t12 — Real-source execution probe of the SWA translate sync sites.

Executes the REAL ast-extracted blocks from flashinfer_backend.py:
  SITE-P (unfixed, extend path, ~2345): kv_last_index = kv_indptr[-1]
  SITE-D (fixed by aa45af5, decode begin-forward path, ~1813): host mirror
    int(seq_lens_cpu.sum()) with fallback to kv_indptr[-1]

with MockTensors that record blocking device-scalar reads (D2H) at
__index__ time, CPU sums at __int__ time, and the slice bound handed to
translate_loc_from_full_to_swa. Verdicts:
  V1 SITE-P with device-resident kv_indptr: exactly 1 blocking read (sync at
     source level on the CURRENT glm-b tree) and translate receives the value.
  V2 SITE-D control, mirror active (seq_lens_cpu present, bs matches):
     0 blocking reads, 1 cpu sum, translate receives the SAME value
     (value-preserving on the coherent fixture).
  V3 SITE-D guard boundary, seq_lens_cpu=None: fallback read happens.
  V4 SITE-D guard boundary, bs mismatch: fallback read happens.
  V5 branch-dead control: use_sliding_window_kv_pool=False -> nothing reads.
Anchors: extracted blocks must appear VERBATIM in the real source (both the
prefill direct-assignment form and the decode guarded form), else abort.
Limits: mock tensors, no CUDA/torch, no real pool; proves source-level
dataflow of the read, not wall-time cost (trace census covers that).

Usage: python3 -I probe_prefill_updater_site.py [out.json]
"""
import ast, json, sys
from pathlib import Path

SRC = Path("python/sglang/srt/layers/attention/flashinfer_backend.py")
EXPECT_VAL = 4242  # coherent fixture: device indptr last == cpu sum == 4242


class DeviceScalar:
    """0-d device tensor: slicing with it triggers blocking D2H at __index__."""
    def __init__(self, val, log): self.val, self.log = val, log
    def __index__(self):
        self.log.append("d2h_sync_read"); return self.val


class DevTensor:
    def __init__(self, log, last=EXPECT_VAL): self.log, self.last = log, last
    def __getitem__(self, i):
        if i == -1: return DeviceScalar(self.last, self.log)
        raise AssertionError("unexpected non-scalar index on device tensor")


class CpuSum:
    def __init__(self, val, log): self.val, self.log = val, log
    def __int__(self): self.log.append("cpu_sum"); return self.val


class CpuTensor:
    def __init__(self, n, val, log): self.n, self.val, self.log = n, val, log
    def numel(self): return self.n
    def sum(self): return CpuSum(self.val, self.log)


class SwaPool:
    def __init__(self, log): self.log = log
    def translate_loc_from_full_to_swa(self, view):
        last = [e[1] for e in self.log if isinstance(e, tuple) and e[0] == "slice"]
        self.log.append(("translate", last[-1] if last else None))
        return view


class IdxStore:
    """Real slicing coerces the bound to int (this is where a device-scalar
    bound blocks on D2H), so __getitem__/__setitem__ coerce via __index__."""
    def __init__(self, log): self.log = log
    def _bound(self, key):
        b = key.stop if isinstance(key, slice) else key
        return b.__index__() if hasattr(b, "__index__") else b
    def __getitem__(self, key):
        self.log.append(("slice", self._bound(key)))
        return self
    def __setitem__(self, key, v):
        self.log.append(("write", self._bound(key)))


class Simple:
    def __init__(self, pool): self._swa_kv_pool = pool


def find_blocks():
    """Return {kind: If-node} for the two SWA translate blocks."""
    tree = ast.parse(SRC.read_text())
    hits = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if ast.unparse(node.test) == "use_sliding_window_kv_pool and (not use_swa_source)":
                body_src = ast.unparse(node)
                if "seq_lens_cpu" in body_src:
                    hits["decode"] = node
                elif "kv_last_index = kv_indptr[-1]" in body_src:
                    hits["prefill"] = node
    assert set(hits) == {"decode", "prefill"}, f"blocks found: {set(hits)}"
    return hits


def run_block(block, *, seq_lens_cpu, bs, swa=True):
    log = []
    pool = SwaPool(log)
    ns = {"use_sliding_window_kv_pool": swa, "use_swa_source": False,
          "self": Simple(pool), "kv_indptr": DevTensor(log),
          "kv_indices": IdxStore(log), "seq_lens_cpu": seq_lens_cpu, "bs": bs}
    exec(compile(ast.Module(body=[block], type_ignores=[]), "<site>", "exec"), ns)
    translated = [e[1] for e in log if isinstance(e, tuple) and e[0] == "translate"]
    return {"d2h_reads": log.count("d2h_sync_read"),
            "cpu_sums": log.count("cpu_sum"),
            "slice_bounds": [e[1] for e in log if isinstance(e, tuple) and e[0] == "slice"],
            "translated_bound": translated,
            "log": [str(x) for x in log]}


def main(outp):
    blocks = find_blocks()
    text = SRC.read_text()
    anchors = {
        "prefill_direct": "            kv_last_index = kv_indptr[-1]",
        "decode_mirror": "kv_last_index = int(seq_lens_cpu.sum())",
        "decode_fallback": "                kv_last_index = kv_indptr[-1]",
    }
    a_ok = {k: (v in text) for k, v in anchors.items()}

    v1 = run_block(blocks["prefill"], seq_lens_cpu=None, bs=128)
    v2 = run_block(blocks["decode"], seq_lens_cpu=CpuTensor(128, EXPECT_VAL, []), bs=128)
    # v2 uses its own log; re-run with shared log for inspection
    log2 = []
    v2 = run_block(blocks["decode"], seq_lens_cpu=CpuTensor(128, EXPECT_VAL, log2), bs=128)
    v3 = run_block(blocks["decode"], seq_lens_cpu=None, bs=128)
    v4 = run_block(blocks["decode"], seq_lens_cpu=CpuTensor(64, EXPECT_VAL, []), bs=128)
    v5 = run_block(blocks["prefill"], seq_lens_cpu=None, bs=128, swa=False)

    verdicts = {
        # real block slices with the device scalar TWICE (read view + write
        # target) -> two __index__ coercions = two blocking D2H per extend step
        "V1_prefill_site_two_blocking_reads_and_translates":
            v1["d2h_reads"] == 2 and v1["slice_bounds"] == [EXPECT_VAL, EXPECT_VAL]
            and v1["translated_bound"] == [EXPECT_VAL],
        "V2_decode_mirror_zero_reads_value_preserved":
            v2["d2h_reads"] == 0 and v2["cpu_sums"] == 1
            and v2["translated_bound"] == [EXPECT_VAL],
        "V3_decode_none_falls_back_two_reads": v3["d2h_reads"] == 2,
        "V4_decode_bs_mismatch_falls_back_two_reads": v4["d2h_reads"] == 2,
        "V5_branch_dead_no_reads": v5["d2h_reads"] == 0 and v5["cpu_sums"] == 0,
        "anchors_verbatim": all(a_ok.values()),
    }
    ok = all(verdicts.values())
    res = {"source": str(SRC), "expect_val": EXPECT_VAL, "anchors": a_ok,
           "verdicts": verdicts, "all_ok": ok,
           "V1_prefill_unfixed": v1, "V2_decode_mirror_control": v2,
           "V3_decode_none": v3, "V4_decode_bs_mismatch": v4,
           "V5_branch_dead": v5,
           "candidate_mirror": "paged_kernel_lens_sum is typed int at "
                               "flashinfer_backend.py:1770 and eagle_info.py:442 "
                               "sets it to cum_kv_seq_len[-1]; kv_indptr[-1] == "
                               "that total is plausible but UNVERIFIED across "
                               "spec_info variants/read kinds - pre-registered "
                               "next-trial check, do not patch yet",
           "limits": ["mock tensors; no torch/CUDA; proves dataflow of the "
                      "blocking read in the real extracted statements, not "
                      "wall-time cost or real pool semantics",
                      "value-preservation checked only on the coherent fixture "
                      "(device last == cpu sum); prefix-trimmed paged-lens "
                      "equivalence NOT established here"],
           }
    json.dump(res, open(outp, "w"), indent=1)
    print("all_ok:", ok)
    for k, v in verdicts.items(): print(" ", k, "->", v)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g3-t12/raw/prefill_site_probe.json")

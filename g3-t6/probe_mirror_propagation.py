#!/usr/bin/env python3
"""G3 trial-6 falsifier: are the SWA prefill host mirrors DORMANT under the
frozen BF16 config? (t5 left this open; hypothesis H1_t6 in plan.md)

Method (glm-b ast-execution pattern): extract the REAL guard/population
expressions from flashinfer_backend.py / forward_batch_info.py /
schedule_batch.py (anchor asserts fail loudly if the source moves) and
EXECUTE them under a scenario table covering every code path that can
deliver seq_lens_cpu / extend_prefix_lens_cpu into
FlashInferIndicesUpdaterPrefill.update_sliding_window. Stdlib only.
Run: python3 -I g3-t6/probe_mirror_propagation.py
"""

import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FI = ROOT / "python/sglang/srt/layers/attention/flashinfer_backend.py"
FBI = ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"
SB = ROOT / "python/sglang/srt/managers/schedule_batch.py"
OUT = Path(__file__).resolve().parent / "raw" / "mirror_propagation.json"

FI_TEXT = FI.read_text()
FBI_TEXT = FBI.read_text()
SB_TEXT = SB.read_text()


def fn_src(text, name):
    """All source segments for name (update_sliding_window exists in both the
    decode and prefill updaters); caller disambiguates by content."""
    tree = ast.parse(text)
    segs = [
        ast.get_source_segment(text, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert segs, f"function {name} not found"
    return segs


def anchor(path, text, lo, hi, must_contain):
    seg = "\n".join(text.splitlines()[lo - 1 : hi])
    for s in must_contain:
        assert s in seg, f"anchor moved {path.name}:{lo}-{hi}: want {s!r}"
    return seg.strip()


class TensorLike:
    """Stand-in for a host int tensor (only what the guards touch)."""

    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n

    def tolist(self):
        return list(range(1, self.n + 1))


class DevLen:
    """Stand-in for a device tensor used only via len()."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


# ---------------------------------------------------------------- real guards
USW_SEGS = fn_src(FI_TEXT, "update_sliding_window")
USW_PREFILL = [
    s for s in USW_SEGS if "_build_swa_prefix_custom_mask" in s and "kv_start_cpu" in s
]
assert len(USW_PREFILL) == 1, f"expected exactly one prefill update_sliding_window, got {len(USW_PREFILL)}"


def guard_expr(fn_text, test_substring, kind):
    """ast.unparse of the first If-test (kind='if') or Assign-value (kind='assign')
    whose source contains the substring. The host_lens_mirror RHS also contains
    the ragged-w0 substring, hence the explicit kind."""
    for node in ast.walk(ast.parse(textwrap.dedent(fn_text))):
        src = None
        if kind == "if" and isinstance(node, ast.If):
            src = ast.unparse(node.test)
        elif kind == "assign" and isinstance(node, ast.Assign) and isinstance(
            node.value, (ast.BoolOp, ast.Compare, ast.Call)
        ):
            src = ast.unparse(node.value)
        if src and test_substring in src:
            return src
    raise AssertionError(f"guard not found ({kind}): {test_substring!r}")


GUARDS = {
    # id: (source of the real guard expression, executed per scenario)
    "host_lens_mirror_rhs": guard_expr(
        USW_PREFILL[0], "seq_lens_cpu is not None", "assign"
    ),  # extend_prefix_lens_cpu is not None and seq_lens_cpu is not None and len(...)...
    "swa_ragged_w0_sum_guard": guard_expr(
        USW_PREFILL[0], "len(extend_prefix_lens_cpu) == len(prefix_lens)", "if"
    ),
}
assert GUARDS["host_lens_mirror_rhs"] != GUARDS["swa_ragged_w0_sum_guard"], \
    "S2 guard must be the weaker ragged-only If, not the host_lens_mirror RHS"
# guards that are plain name tests / elifs -- captured verbatim as text anchors
PLAIN_GUARDS = {
    "single_ragged_sum": anchor(
        FI, FI_TEXT, 1965, 1969,
        ["if extend_prefix_lens_cpu is not None:", "paged_kernel_lens.sum().item()"],
    ),
    "swa_paged_w0_mirror": anchor(
        FI, FI_TEXT, 2099, 2108, ["if host_lens_mirror:", "for s, e in zip("]
    ),
    "swa_paged_w0_fullseq": anchor(
        FI, FI_TEXT, 2109, 2115,
        ["elif prefix_is_full_seq and seq_lens_cpu is not None:",
         "torch.clamp(seq_lens_cpu, max=sliding_window_size).sum()"],
    ),
    "swa_paged_w0_fallback": anchor(
        FI, FI_TEXT, 2116, 2116, ["paged_kernel_lens.sum().item()"]
    ),
    "swa_ragged_w1_sum": anchor(
        FI, FI_TEXT, 2121, 2124,
        ["if extend_prefix_lens_cpu is not None:", "prefix_lens.sum().item()"],
    ),
    "mask_builder_fallback": anchor(
        FI, FI_TEXT, 2181, 2188,
        ["prefix_lens_cpu is None", ".detach().cpu().tolist()"],
    ),
    "dispatcher_passes_mirrors": anchor(
        FI, FI_TEXT, 1063, 1091,
        [
            "not any(forward_batch.extend_prefix_lens_cpu)",
            "forward_batch.seq_lens_cpu",  # positional arg (L1078)
            "extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu",
        ],
    ),
}
# population-side facts (anchored, not executed: whole functions too big to run)
POPULATION = {
    "prepare_for_extend_sets_seq_lens_cpu": anchor(
        SB, SB_TEXT, 2879, 2888,
        ["seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)",
         "self.seq_lens_cpu = seq_lens_cpu"],
    ),
    "merge_drop_rule": anchor(
        SB, SB_TEXT, 3953, 3957,
        ["if self.seq_lens_cpu is None or other.seq_lens_cpu is None:",
         "self.seq_lens_cpu = None"],
    ),
    "spec_mix_rebuilds_mirror": anchor(
        SB, SB_TEXT, 3302, 3312,
        ["merge_batch", "running_seq_lens_cpu = torch.tensor"],
    ),
    "nonspec_mix_comment": anchor(
        SB, SB_TEXT, 3313, 3318,
        ["Non-spec: the running batch carries prepared seq_lens_cpu"],
    ),
    "spec_restore_after_merge": anchor(
        SB, SB_TEXT, 3348, 3349,
        ["if merged_seq_lens_cpu is not None:", "self.seq_lens_cpu = merged_seq_lens_cpu"],
    ),
    "init_new_main_path_mirrors": anchor(
        FBI, FBI_TEXT, 1230, 1246,
        ["isinstance(extend_seq_lens, list)", "ret.extend_prefix_lens_cpu = list(extend_prefix_lens)",
         "# gpu_only: device tensors handed in directly; leave *_cpu unset."],
    ),
    "init_new_takes_sb_mirror": anchor(
        FBI, FBI_TEXT, 1013, 1013, ["seq_lens_cpu = batch.seq_lens_cpu"]
    ),
}

# ---------------------------------------------------------------- scenarios
def scenario(ext_plc, slc, n, prefix_is_full_seq=False, note=""):
    return {
        "extend_prefix_lens_cpu": ext_plc,
        "seq_lens_cpu": slc,
        "n": n,
        "prefix_is_full_seq": prefix_is_full_seq,
        "note": note,
        "_ext": [3] * n if ext_plc else None,
        "_slc": TensorLike(n) if slc else None,
        "_pl": DevLen(n),
    }


SCENARIOS = {
    # frozen-config applicable paths
    "main_extend_nonspec": scenario(True, True, 4, note="prepare_for_extend + init_new list path"),
    "mixed_chunk_nonspec": scenario(True, True, 6, note="mix_with_running: both sides carry mirror; merge concats"),
    "mixed_chunk_spec_rebuilt": scenario(True, True, 6, note="spec mix rebuilds seq_lens_cpu from req state and restores after merge"),
    # off-config paths (recorded to bound the claim)
    "gpu_only_tensors": scenario(False, True, 4, note="init_new tensor branch (gpu_only)"),
    "relay_merge_null": scenario(True, False, 4, note="merge_batch dropped seq_lens_cpu (relay/spec-relay)"),
}

FROZEN_APPLICABLE = ["main_extend_nonspec", "mixed_chunk_nonspec"]


def evaluate(sc):
    ns = {
        "extend_prefix_lens_cpu": sc["_ext"],
        "seq_lens_cpu": sc["_slc"],
        "prefix_lens": sc["_pl"],
        "len": len,
    }
    mirror = eval(GUARDS["host_lens_mirror_rhs"], {}, dict(ns))
    w0_ragged_sum_guard = eval(GUARDS["swa_ragged_w0_sum_guard"], {}, dict(ns))
    ext_present = sc["_ext"] is not None
    slc_present = sc["_slc"] is not None
    sites = {
        # site id -> [mirror_taken?, applies-when]
        "S1_single_ragged_sum": [ext_present, "single-wrapper dispatch (not Spark SWA)"],
        "S2_swa_ragged_w0_sum": [w0_ragged_sum_guard, "SWA extend, use_ragged=True"],
        "S3_swa_ragged_w0_mask": [mirror, "SWA extend ragged: mask builder host lists"],
        "S4_swa_paged_w0_sum": [
            bool(mirror or (sc["prefix_is_full_seq"] and slc_present)),
            "SWA extend, use_ragged=False",
        ],
        "S5_swa_ragged_w1_sum": [ext_present, "SWA extend, full-attn wrapper ragged"],
    }
    return {
        "host_lens_mirror": mirror,
        "w0_ragged_sum_guard": w0_ragged_sum_guard,
        "sites": {k: ("MIRROR" if v[0] else "FALLBACK_DEVICE_READ") for k, v in sites.items()},
    }


results = {name: evaluate(sc) for name, sc in SCENARIOS.items()}

# frozen-config verdict per site
frozen_verdict = {}
for site in ["S1_single_ragged_sum", "S2_swa_ragged_w0_sum", "S3_swa_ragged_w0_mask",
             "S4_swa_paged_w0_sum", "S5_swa_ragged_w1_sum"]:
    vs = {s: results[s]["sites"][site] for s in FROZEN_APPLICABLE}
    fallback_reachable = any(v == "FALLBACK_DEVICE_READ" for v in vs.values())
    frozen_verdict[site] = {
        "scenario_verdicts": vs,
        "fallback_reachable_under_frozen_config": fallback_reachable,
    }

h1_confirmed_sites = [
    s for s, v in frozen_verdict.items() if v["fallback_reachable_under_frozen_config"]
]

report = {
    "hypothesis": "H1_t6: the e063d72/2f3a916 prefill host mirrors are DORMANT under "
                  "the frozen BF16 config (fallback device reads actually run), so i4's "
                  "x1.0017 never measured the family engaged.",
    "verdict": "FALSIFIED" if not h1_confirmed_sites else "CONFIRMED_FOR " + ",".join(h1_confirmed_sites),
    "executed_guards": GUARDS,
    "plain_guards_anchored": PLAIN_GUARDS,
    "population_facts_anchored": POPULATION,
    "scenarios": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in SCENARIOS.items()},
    "results": results,
    "frozen_config_assumptions": {
        "spec_algorithm": "none (frozen launch has no draft model)",
        "two_batch_overlap": "off",
        "relay/dp-attention staging": "off",
        "gpu_only ForwardBatch entry": "not on the serving extend path",
        "SGLANG_FLASHINFER_USE_PAGED": "unset -> use_ragged=True for extends (both "
        "use_ragged values covered by S2/S4 anyway)",
        "frozen_applicable_scenarios": FROZEN_APPLICABLE,
    },
    "structural_proof_extend_plc_non_none": "dispatcher L1063 runs "
    "`not any(forward_batch.extend_prefix_lens_cpu)` on EVERY extend step; None would "
    "raise TypeError -- the engine structurally assumes the mirror is a populated list "
    "on the serving path.",
    "limits": [
        "Source-level dataflow + executed guard expressions, not a runtime trace; "
        "branch counters were deliberately not added because every applicable path "
        "resolves to MIRROR and the L1063 any() makes absence of the extend mirror "
        "a crash, not a slowdown.",
        "Scenarios use synthetic lens values; only the GUARD expressions are the real "
        "extracted source (population facts are line-anchored verbatim).",
        "Decode side (fast_decode_plan at L846 covers all decode wrappers incl. SWA) "
        "was closed by t5 and is not re-audited here.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps({"verdict": report["verdict"],
                  "frozen_verdict": frozen_verdict,
                  "off_config_residual": {
                      s: results[s]["sites"] for s in
                      ["gpu_only_tensors", "relay_merge_null"]}}, indent=2))

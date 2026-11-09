# G3 trial 6 — worker deepseek-b (plan, v0)

## Starting point (canonical, not re-derived)

- **F3**: GGUF MMVQ threshold fix `f263262` is delivered but **UNGATED**. The direction says evaluate it before interpreting any gain.
- **applicability-001 (controller)**: the frozen formal Gate loads BF16 safetensors with BF16 KV and never passes `--quantization gguf`. A BF16 Fast score therefore **cannot** close F3, cannot isolate `f263262`, and cannot prove GGUF gated-neutrality. Importing `gguf.py` is not dispatch evidence.
- **C6**: `request_fast_gate=false` + prose FULL launches nothing. GGUF/NGRAM custom GPU experiments are pending on matching trusted artefacts.
- **C3 / branch**: BF16 Fast on this worker measured ×0.9968 and ×1.004 — i.e. no attributable gain from the sync-removal lineage. `mmvq_safe` retunes are **score-neutral by construction** at gate batches >=32 (all gate points run bs>=32), so re-tuning them is not a performance lever.
- Constraint of this session: no GPU, no SSH, 45 turns / 20 min / USD 3. So the only honest deliverable is (a) an **executed CPU falsifier** with raw output, or (b) a BF16 source change plus a fast-gate request.

## Decision framework for this trial

1. Do **not** repeat: MMVQ threshold tuning, eager decode-sync patches (trial-1 inert), `fast_decode_plan`, or the unexecuted GGUF GPU request.
2. Prefer a **new falsifiable question** that (i) can be settled on CPU with executed evidence, and (ii) either narrows F3's real scope or exposes a new BF16 lever.
3. If the evidence supports a BF16-safe engine change, make it and set `request_fast_gate=true`. A patch alone is not acceptance.

## Candidate hypothesis space (to be resolved by recon)

- **H-A (F3 scope, CPU)**: prove or refute that the changed MMVQ selection cells are *reachable* from the frozen BF16 formal gate by executing the real dispatch/type-selection expressions over the actual frozen model/config. If unreachable, F3 stays UNGATED and the only correct action is a separate GGUF experiment — this is a scope falsifier, not a speedup.
- **H-B (BF16 lever, CPU-then-gate)**: locate a *measured* BF16 bottleneck point (PrefillHeavy c16 is 15% weight at ~59 tok/s / ~4 s TTFT; DecodeCeiling c256) and find a semantics-preserving change with an executable invariant check.
- **H-C**: anything newly falsifiable discovered in recon.

## Kill criteria

- If the falsifier's modelled expressions drift from the pinned source (anchor asserts fail) → stop, report the drift, do not build on a stale model.
- If the only available change is a semantics change (e.g. applying the 511-token sliding window in the no-prefix extend path) → **do not gate it**; `rsi_equivalence.py` tops out at 511 tokens and cannot certify 8192-token parity. Record the negative.

## RESULT (trial 6, executed)

Chosen: **H-A/H-C hybrid — audit the pending t3/t4 host-mirror candidate's reachability
envelope and index invariant** (F3 is provably gate-unreachable and no GGUF artifact/GPU
exists here, so re-requesting it would repeat an unexecuted request).

Executed (`seis/trials/deepseek-b/g3-t6/verify_host_mirror_reachability.py`, exit 0):
- The changed lines are **dead code whenever `translator.reads_are_translated` is true**
  (unified SWA pool): 12/12 such config rows skip the whole full->swa translate block.
- Decode takes the host mirror in 6/48 config rows, prefill in 1/48.
- Index equality: **0 divergences in 32 non-control cases** (SWA-clamped >511, exactly
  512/513, graph-padded zero lanes, `seq_lens_cpu=None`, prefill 16x8192, 16x(512+8192)).
  Two sensitivity controls (stale mirror, below-window off-by-one) are both detected.
- D2H accounting: patch removes one device read per eager decode step; **zero when
  `seq_lens_cpu is None`** (caller already `.item()`s at line 1696).

Conclusion: the candidate is correct-but-inert hygiene; expected gate gain ~0. No engine
change, no fast gate requested (`request_fast_gate=false`). F3 stays UNGATED.
`NOT_CLOSED_QUALITY_BUDGET` untouched.

## Deliverables

- `plan.md` (this file, updated with the chosen hypothesis).
- Executed evidence under `seis/trials/deepseek-b/g3-t6/` with raw output preserved.
- `iteration-proposal.json` with string fields `bottleneck,hypothesis,patch,tests,learning,next_proposal` and boolean `request_fast_gate`.
- Local git commit(s).
- Explicit uncertainty / NOT_CLOSED statement. No invented measurements.

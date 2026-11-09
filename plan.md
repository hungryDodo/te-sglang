# G3 trial-6 plan (worker glm-a)

Direction (brief): revisit SWA / FP8+SWA history; precision quality unresolved;
prefer exact runtime alternatives when useful.

## Starting facts (canonical + local lineage)
- HEAD 988340d = i4 engine tree (44d49f8: fp8-KV remap gated behind
  SGLANG_SPARK_FP8_KV, default OFF -> BF16 truly active) + t5 audit artifacts,
  no python/ change after 44d49f8. i4 Fast x1.0017 = whole in-tree exact family
  neutral on a1 BF16.
- t5 executed audit (g3-t5/raw/sync_audit.json): decode SWA covered by
  fast_decode_plan; plain-EXTEND replays never get fast_prefill_plan; 5
  fallback device reads remain in the prefill updater (L1969/2082/2116/2124
  `.sum().item()`, L2186-2188 `.detach().cpu().tolist()`), taken only when
  host mirrors are absent. t5 explicitly left OPEN whether the mirrors are
  actually populated on the prefill path under the frozen config.

## Trial-6 hypothesis (falsifiable, CPU-executable)
H1_t6: "The e063d72/2f3a916 host mirrors are DORMANT on the prefill/extend
path under the frozen BF16 config (callers do not propagate host lens into
the updater, so the fallback device reads are the branches that actually
run), which means i4's x1.0017 measured the family with its prefill half
inactive -- the lane was not closed, it was unactivated."
- Same failure shape as the i3 fp8 activation bug (code present, execution
  condition never checked).
- Falsifier: mechanical dataflow over the REAL source -- enumerate every
  caller of the prefill indices updater and of _build_swa_prefix_custom_mask,
  extract (ast) the actual argument expressions and branch guards, and decide
  per call site whether host lens reach the mirror parameters under the
  Spark SWA dispatch. Preserve raw JSON.

## Decision procedure after the probe
1. If mirrors DO reach every Spark-SWA prefill call site (H1 falsified):
   exact lane stays closed; deliver the executed negative + dossier; no gate.
2. If any Spark-SWA prefill call site lacks host lens (H1 confirmed on that
   site): implement the bounded exact fix(es) named by t5 -- paged-SWA branch
   host sum + host lens into _build_swa_prefix_custom_mask callers -- syntax
   check, re-run the probe to show the branch now resolves host-side, commit,
   and request_fast_gate=true (frozen BF16 screen; new unmeasured exact tree).
3. Also verify (cheap, same probe): fp8 gate still inert at BOTH remap sites
   under frozen flags (extends g3-t4/raw/remap_gate_check.json to HEAD).

## Honesty constraints
- i4's x1.0017 stands as the family-level measurement of what ran; if H1 is
  confirmed, expected residual gain is bounded above by a2's whole-family
  x1.0543 and could still be ~0 on a1 -- stated as uncertainty, not promise.
- No EXACT claim without numeric evidence; no promotion request; no precision
  change; no duplicate screens of known-neutral trees.
- Budget: 45 turns / 20 min / $3; commit checkpoint early; last quarter
  reserved for tests, iteration-proposal.json, final commit.

## Outcome (executed, turn ~24)
Decision branch 1 fired: H1_t6 is FALSIFIED. g3-t6/probe_mirror_propagation.py
(python3 -I; raw g3-t6/raw/mirror_propagation.json) extracted the REAL guards
from flashinfer_backend.py `update_sliding_window` (prefill instance) plus the
population-side facts (schedule_batch prepare_for_extend / mix_with_running /
merge_batch, forward_batch_info init_new) and executed them over a 5-scenario
table:

1. FROZEN-CONFIG VERDICT: under the frozen BF16 launch (non-spec, no TBO, no
   relay/dp staging, no gpu_only entry), all five fallback device-read sites
   (S1 single-wrapper sum, S2 SWA-ragged w0 sum, S3 SWA-ragged mask builder,
   S4 SWA-paged w0 sum, SWA-ragged w1 sum) resolve to MIRROR on every
   applicable scenario (main extend, mixed chunked prefill). The fallback
   branches are UNREACHABLE there. Structural corroboration: the extend
   dispatcher runs `not any(forward_batch.extend_prefix_lens_cpu)` on EVERY
   extend step, so a missing extend mirror is a TypeError crash, not a silent
   slowdown.
2. => i4's Fast x1.0017 measured the exact-runtime family with its prefill
   mirrors fully ENGAGED. t5's lane closure ("exact-runtime SWA/host-sync work
   on a1 is throughput-neutral") is now complete: no dormant-hunk escape hatch.
3. Honest residual bounds (off-config only): with a gpu_only tensor entry
   (device tensors handed to init_new) all five sites fall back; with a
   relay/spec-relay merge that drops seq_lens_cpu, S3/S4 fall back while S2/S5
   stay mirrored. Neither entry is on the frozen serving path (no spec draft,
   no relay). These bound where a future config change could resurrect the
   syncs; they are not frozen-config risks.
4. fp8 gate re-verified at HEAD: g3-t6/raw/remap_gate_check_HEAD.json — frozen
   launch (bfloat16, SGLANG_SPARK_FP8_KV unset) resolves to bfloat16 with an
   empty post-process at BOTH remap sites; env=1 activates, env=0 stays BF16.
5. NO engine change (python/ untouched since 44d49f8); no fast gate requested
   — a screen of the unchanged tree would repeat i4's measurement, and branch
   counters were rejected because every applicable path resolves MIRROR and
   absence of the extend mirror is a crash (L1063), so a counter boot would
   spend GPU to confirm a structural impossibility.

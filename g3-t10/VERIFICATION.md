# G3 glm-b trial-10 — executed analysis of C21 NGRAM ON/OFF pairs (2026-11-10)

Input (read-only, controller-owned): `controller_evidence/continuous-20261109-01/
20261110-runtime/` — 20 primary + 16 replay parsed HTTP responses, NGRAM ON
`b9a91aa13` vs deletion-only OFF `999d4b243`, real Spark-X2.6 BF16/FlashInfer
server, greedy (temperature 0, ignore_eos, max_new_tokens 48), 512-token
repeat-pattern prompts, C1/C4, `return_logprob=true, top_logprobs_num=8`.
Harness: `g3-t10/analyze_ngram_pairs.py` (stdlib, hard exit-2 on harness error);
raw derived output: `g3-t10/raw/ngram_pair_analysis_output.json`.
Run result: **0 harness errors**; controller's `first_divergence` recomputed
independently and cross-checked: **10/10 match**. Engine `python/` untouched —
`git diff 2c854e0 -- python/` is empty (promotion candidate preserved).

## Q1 — recipe closure: CONFIRMED (RECIPE_POPULATED_ALL)
36/36 responses have `meta_info.output_top_logprobs` populated: exactly 8 rows
`[lp, id, None]` for every one of the completion positions. This is the exact
request shape (i4) + native nested read field (i5) glm-b specified for the still
pending numeric re-run. Upgrade: the recipe is now **proven on real servers**,
not merely CPU-chain-predicted (i8). GPU-side top-k population under this recipe
is demonstrated working end-to-end.

## Q2 — F1 error fingerprint from real data (new quantification)
1. **lp[0] bitwise-equal in 10/10 pairs** (delta exactly 0.0). The first decode
   step — before any draft can exist — is numerically identical to OFF. The bug
   is NOT in the no-spec baseline forward; corruption requires draft steps.
2. **Genuine pre-divergence drift**: chosen-LP deltas at aligned positions grow
   0 → ≤0.025 (steps 1–2) → 0.22–0.59 nat (step 3) while token IDs still agree.
   Alignment test: best shift k=0 in 10/10 (k∈[-9,9]; the period-8 prompt would
   have camouflaged a ±8 shift, so this is a real distribution difference, not a
   recording offset from multi-token spec steps). Pre-divergence top-8 **set**
   overlap is only 56–88%.
3. **At divergence** (step 4 in 9/10 pairs; step 6 and 15 in the outliers): ON
   emits its own top1; OFF's argmax is ON's **rank-2 in 10/10 pairs**; ON top1
   margins are small (0.016–1.11 nat) where OFF margins are 0.94–6.95 nat.
4. **Self-top1 audit reproduces C21 counts exactly** (ON 1/480, OFF 0/496). The
   single ON case (A/repeat-512/c4same inst 3, pos 6) is an **exact logprob
   tie**: chosen 1271 and top1 48977 share lp −1.556087613105774 bitwise
   (gap 0.0). It is a tie-order artifact, NOT evidence of commit≠distribution.
5. **No re-convergence**: post-divergence equal positions 0/41–43 in 7/10 pairs,
   ≤4 chance-level hits in the rest.
7. **Replay isolation (strongest causal constraint)**: each of the 4 replay
   cases sends prompt(512) + first d ON tokens (token history identical across
   roles, since IDs match up to d) to the OFF server and generates 1 token
   (prefix_len − 512 = d verified in 4/4). OFF reproduces its primary argmax at
   d **stably across all 4 replays each** and it differs from ON's committed
   token in 4/4. So given the *same token history*, ON computes a different
   distribution: the corruption is in ON's internal state (KV/mask) accumulated
   during draft-accepting steps, not in the visible history and not sampling
   noise. Combined with (1): empty-draft step bitwise clean, first
   draft-accepting steps onward corrupted.
6. A/c1same (d=15) drift is non-monotonic with pattern-position structure
   (deltas 0.36/0.65/1.32 at pattern steps) — per-accepted-draft distortion, not
   uniform accumulation.

## Q3 — real performance observation (server-measured, paired)
Decode throughput ON/OFF per identical request: **median 3.02× (min 2.21, max
5.24)**; decode_span = e2e_latency − first_token_latency, per-request rows in
raw output. Zero retractions both roles, all completions 48, queue_time
negligible. Reading: on period-8 repeat prompts at C1/C4 the ngram mechanism
accepts long drafts nearly every step — the ~3× per-request decode speedup is
the mechanism working as designed, on exactly the inputs where Q2 shows greedy
argmax being corrupted from the first accepted draft. This bounds the upside
available if F1 is fixed (real, single-shape, diagnostic-pass evidence).

## Limits / what this is NOT
- Derived analysis of controller diagnostics; the packet is input, not my
  experiment. No engine change; no Fast/GPU claim made here.
- ON vs OFF are different commits on separate boots (same node, GPU lock per
  C21); timing comparison is cross-boot, single pass, C1/C4 only — NOT the
  frozen 10-point workload, NOT a formal gate, no quality claim.
- Steps 1–2 deltas (≤0.025 nat) could in principle include benign
  verify-kernel shape numerics; the 10/10 rank-2 flip structure and ≥0.2 nat
  step-3 deltas are inconsistent with pure noise, but per-position
  drift-vs-corruption attribution is not individually certified.
- No claim that F1's root cause is identified; this constrains WHERE it acts
  (first accepted draft onward, never the empty-draft step) and what it looks
  like (distribution distortion with OFF argmax demoted to rank 2).

## Value for pending glm-b items
The comparison core executed here (pair by label, IDs + chosen-LP + top-k rows,
first-divergence, per-position deltas, self-top1 audit) is exactly the machinery
the pending trusted reference-vs-candidate numeric comparison needs; it now has
a clean real-data execution record.

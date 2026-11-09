# G3 t7 — deepseek-b lane-closure oracle (CPU-executed)

No GPU, no torch/gguf package, no model weights, no performance measurement.
Everything below is arithmetic over source-declared shapes plus source anchors
at the committed HEAD. Raw output: `closure.json`.

## Why this trial

The branch's two remaining ranked actions were (1) load-time dequant Q8_0/Q4_K
-> BF16 + cuBLAS GEMM, motivated by g2-t1's 8.95–11.85x ceiling at batch 256,
and (2) a >512-token parity harness for the missing `window_left` in the
`extend_no_prefix` prefill path. Before requesting GPU time for either, this
trial checks the premises that decide whether they can matter at all.

## 1. Shape accounting from the real config (executed)

`verify_lane_closure.py` AST-parses `Spark2_5Config.__init__` defaults from
`python/sglang/srt/configs/spark2_5.py`, executes the real `layer_types`
list-comprehension, and asserts five linear-layer anchors in
`python/sglang/srt/models/spark2_5.py`.

- 28 layers = 21 `sliding_attention` + 7 `full_attention` (executed source
  comprehension, not a copy).
- Per layer 51.40M params (qkv 3072x2048, headwise gate 8x2048, out 2048x2048,
  gate_up 13312x2048, down 6656x2048); tied model 1.711B params, which matches
  the "1.7B" model name (untied total 1.984B also emitted).
- Weight bytes: BF16 3.42 GB, Q8_0 1.82 GB, Q4_K 0.96 GB (GGML block sizes).
- Matrix sizes vs the GB10 <25 MB L2 used by g2-t1:
  `mlp_gate_up` BF16 54.5 MB, `mlp_down` 27.3 MB, `token_embd` 545.3 MB.
  Three of the five non-trivial real matrices exceed L2 even at Q8_0
  (gate_up 29.0 MB, embed 290 MB). g2-t1's microbench weights were at most
  14.5 MB, i.e. fully L2-resident.

**Verdict:** the 8.95–11.85x ceiling was measured in the L2-resident regime
and does not transfer to the real Spark layer sizes. This does not say the
dequant route is slower (the integer kernel can be compute-bound at b=256); it
says the 8.95x number may not be used as the real-size expectation. A GPU A/B
would be needed for the actual number.

## 2. Prefill FLOP accounting (executed)

Dense FLOPs `2 * (1.711e9 - 272.6e6) * S` (embedding is a gather, lm_head last-token only); attention `4*S*eff_len*2048` per layer
(QK^T + AV). Reference path = full attention in all 28 layers (see §3);
windowed-ceiling applies the source's exclusive window (511) to the 21 sliding
layers only.

| S | attention share of prefill FLOPs | windowed speedup ceiling |
|---|---|---|
| 512 | 3.9% | 1.000x (window does not bind) |
| 1024 | 7.5% | 1.029x |
| 2048 | 14.0% | 1.086x |
| 4096 | 24.6% | 1.193x |
| 8192 | 39.5% | 1.385x |

So even if the missing window were free to apply, the frozen PrefillHeavy
point (16 x 8192) could improve by at most ~1.39x on compute, i.e. ~+2.5% on
the aggregate score through a 0.15-weight category — and that is a ceiling
that ignores all kernel inefficiency.

## 3. Why the window lever cannot be accepted (source anchors at HEAD)

`python/sglang/srt/layers/attention/flashinfer_backend.py` at HEAD:

- `:1464-1472` `extend_no_prefix` calls `prefill_wrapper_ragged.forward(...)`
  with **no** `window_left` argument;
- `:1474-1496` the prefix-present branch passes `window_left=swa_window_left`
  to both ragged and paged wrappers.

The pinned reference wheel executes the same upstream path (branch trial-2).
Applying the window therefore changes the attention output of every query
position >511 in 21/28 layers. `valid_exact_token_baseline` compares candidate
tokens against the reference run, so the change cannot pass the frozen gate,
and a >512-token parity harness cannot rescue it: parity would be defined
against the same full-attention reference. The lever is not "blocked on
evidence"; it is **incompatible with the frozen acceptance contract**.

## 4. Lane verdicts

- **F3 (`f263262`)**: already executed by g3-t5 — 0 changed dispatch cells at
  any batch >=16, and every gate point is batch >=16 (decode) or thousands of
  token rows (prefill). New in this trial: the formal Gate loads BF16
  safetensors with no GGUF artifact (controller applicability-001), so the
  GGUF kernel path contributes **zero** aggregate tokens/s regardless of its
  speed. A GGUF GPU experiment can only move F3 out of UNGATED as research
  hygiene; it cannot produce a score or a promotion.
- **Load-time dequant -> BF16 GEMM**: a precision change (integer dot vs BF16
  tensor-core accumulation) and out of the formal BF16 gate. NOT_CLOSED_QUALITY
  BUDGET applies and the score path is zero. Do not spend deepseek-b GPU slots
  on it.
- **BF16 engine lane**: decode is roofline-closed on a4 (branch measurements);
  prefill is compute-bound and its only larger lever is the semantics-blocked
  window above. No promotable BF16 lever is established.

## Limits

Arithmetic model, not a measurement. FLOP counts ignore kernel efficiency and
the attention-output projection is counted once via params. Controller-side
launch/model pins are not visible in this sandbox; applicability-001 is cited,
not re-executed. The 1.385x figure is an upper bound for a change that must
not be made, not a candidate.

# G3 trial 13 — deepseek-a — real BF16 C128 decode trace: is there a host lever, and does attention scale with context?

Campaign continuous-20261109-01, Iteration 13, worker deepseek-a.
Canonical: F1 NGRAM BUG_CONFIRMED (promotion prohibited); F2 FP8-KV
COMPUTE_DIFFERS / quality budget undefined. No GPU/SSH. CPU analysis only.

## Inputs (read-only)

- `controller_evidence/continuous-20261109-01/20261110-decode-trace/` — real
  16-forward BF16 C128 DECODE torch profile on GB10 a1, frozen launch
  (`--cuda-graph-max-bs-decode 64`), Medium 512/64 C128, 128 HTTP 200 x 64 IDs.
  Trace sha256 `09e8dcb9…4372e4b1`.

## Questions

1. Activation: with the frozen max-bs-64 CLI, does every observed bs=128 decode
   step replay a CUDA graph, or does the engine fall back to eager?
2. Host: is there any remaining host-side critical path on this point (kernel
   union, CPU step work, event-sync nesting, inter-step gaps)?
3. New falsifier: each step appends one KV position per request, so a
   KV-byte-proportional attention cost predicts mean_attn/513 ≈ +0.0325 ms/step
   slope over the 16 captured steps. Measure the slope; reject or confirm.

## Method

`g3-t13/analyze_decode_trace.py` (stdlib gzip/json, `python3 -I`, separate
directory from the data) emits bounded JSON: per-step kernel counts, kernel
union, attention/GEMM/lm_head/sampling/other split, event-sync overlap, OLS
slope of attention vs step with SE, and a hard precondition gate (16 launches,
16 CPU/GPU step spans, kernel count multiple of 16) that fails closed. Raw
derived output preserved; originals untouched.

## Deliverables

`g3-t13/analyze_decode_trace.py`, `g3-t13/raw/*`, `g3-t13/FINDINGS.md`,
`iteration-proposal.json` (request_fast_gate=false; next bounded diagnostic =
two-trace bs scaling C32 vs C128). No engine change; nothing here is a gate,
numeric, speed or promotion claim.

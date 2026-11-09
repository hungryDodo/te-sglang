# G3 trial 4 (glm-b) — plan

Campaign `continuous-20261109-01`, iteration 4, worker glm-b. Session HEAD starts
at `b46e13a` (engine bit-identical to `2c854e0`, verified: `git diff 2c854e0 HEAD
-- python/` empty).

## Why not the trial-3 carry-over request

Trial-3 requested a FULL gate slot in prose with `request_fast_gate=false`. Per
the controller capability notes that launches nothing, and the brief explicitly
says not to repeat an unexecuted request and count it as progress. Candidate-x3
prerequisites are also unmet: **numeric verification is blocked because both
numeric probes returned `top_logprobs: null`** (numeric-coverage-001), so the
promotion pipeline (canonical F6) is gated on *evidence coverage*, not on
throughput. Meanwhile glm-b's engine lane is measurement-saturated (decode
roofline-closed at 99.2% GPU busy; prefill syncs bounded ~0 by the 1.0933 fast
screen; engine-wide blocking-read sweep closed statically in i2).

## Hypothesis (new, CPU-testable in this sandbox)

**H1:** `top_logprobs` is null because the request → sampling-params mapping in
this tree requires an explicit top-k field that the numeric probe omitted
(native `/generate`: `return_logprob=true` alone yields chosen logprobs but
`top_logprobs_num` defaults to 0 → no per-token top-k; OpenAI-compat:
`logprobs=true` without `top_logprobs=k`, or an endpoint that never maps it).
The serving path itself can produce per-token top-k once asked correctly.
**H0 (fix-worthy variant):** this fork has a real drop-bug in the mapping
(e.g. chat adapter ignores `top_logprobs`) — then the fix is an engine change
in my own tree and the fast screen doubles as boot verification.

Falsifier: exact source trace of the field plumbing (io_struct →
tokenizer_manager → openai adapters → sampler gating), plus a self-contained
CPU script that executes the request-construction logic with stubbed heavy
imports and records raw JSON output showing which request shapes yield null
vs populated top-k slots.

## Steps

1. This file + early commit (recoverable checkpoint).
2. Source trace of logprob fields across the serving path; identify the null
   root cause; distinguish probe-parameter gap vs server bug.
3. `g3-t4/probe_top_logprobs_recipe.py` — CPU falsifier, raw output preserved
   to `g3-t4/raw/`.
4. Engine change ONLY if a genuine in-tree bug is found; `py_compile`; commit.
5. `iteration-proposal.json` (request_fast_gate per outcome) + final commit.

## Budget

45 turns / 1200 s hard / USD 3. Reserve final quarter for proposal + commits.
No GPU here: everything measured stays a controller-side request; nothing is
claimed without controller raw evidence.

## Outcome (executed this session)

H1 **CONFIRMED CPU-side**, H0 (engine drop-bug on chat/native) **falsified**.
`g3-t4/probe_top_logprobs_recipe.py` extracts the real assignment/guard
expressions via `ast` from `io_struct.py` / `serving_chat.py` /
`serving_completions.py` / `logprob_result_processor.py` and executes them
against 7 request shapes (raw JSON in `g3-t4/raw/falsifier_output.json`):

- native `{"return_logprob": true}` → `top_logprobs_num` None→0 → guard skips
  → **null top_logprobs with chosen present** (io_struct.py:569-570,
  logprob_result_processor.py:60/92/256/268). Reproduces numeric-coverage-001.
- chat `{"logprobs": true}` → `request.top_logprobs or 0` → 0 → same null
  (serving_chat.py:1372-1374). Also reproduces.
- Fix shapes `{"return_logprob": true, "top_logprobs_num": 20}` (native) /
  `{"logprobs": true, "top_logprobs": 20}` (chat) → populated(k=20). The
  engine fully supports top-k once asked; no engine change needed.
- Documented quirk, NOT fixed (off the frozen measurement path, and cannot
  produce null): `/v1/completions` ignores standard OpenAI `top_logprobs`;
  its `logprobs` is typed `Optional[int]` and doubles as k
  (protocol.py:377, serving_completions.py:119-120) → standard shape yields
  k=1 silently.

`git diff 2c854e0 HEAD -- python/` stays EMPTY by design: the promotion
candidate remains bit-identical to the twice-proven engine tree. Ask of the
controller: re-run the numeric coverage probe with the corrected shape on
reference AND candidate (lossless raw variant available:
`return_flat_raw_output_top_logprobs`, single-item, non-stream,
io_struct.py:250-254/457-476). request_fast_gate=false — there is no engine
change to screen and screens claim no slots.


# Round 1 / node a3 — Scheduler decode/filter Python overhead

Base commit: 5b89820b04cb26385d45915c484708796ba7df91
Worker: deepseek-a

## Target
`python/sglang/srt/managers/schedule_batch.py`

## Plan
1. Read `ScheduleBatch.filter_batch`, `prepare_for_decode`, and `get_model_worker_batch`/related metadata paths.
2. Find a semantics-preserving Python metadata overhead in the ordinary decode path
   (no batching-policy, token-budget, or cache-semantics change).
3. Implement one small actual engine-code change. Keep ordering, partial/completed
   request handling, explicit keep indices, logprob/grammar/hidden-state fields, and
   beam/speculation/Mamba fallbacks intact.
4. Add ONE narrow stdlib/AST-based CPU regression test that source-extracts the
   control flow (empty/no-op/partial paths) so the golden is not just a shared
   implementation. Do not import the CUDA engine.
5. Run with python3, write `iteration-proposal.json`, commit.

## Constraints
- No weight download, no precision/quality changes, no GPU/SSH/installs/network.
- No native kernel or fixed-flag changes.
- Fixed policy: weights Q8_0, full/swa KV BF16, native binaries unchanged.
- No performance score claimed; controller runs official measurement.

## Round 2 - deepseek-a (2026-11-10)

Prior round 1 ended `error_max_turns` with a plan only and no production change; its raw/cost status remains AGENT_ERROR and is not claimed as success. Candidate base remains `5b89820b04cb26385d45915c484708796ba7df91`.

Round 2 candidate (narrow): in `ScheduleBatch.filter_batch` (`python/sglang/srt/managers/schedule_batch.py`), after `None` exclusions normalize to `[]`, build `keep_indices` by selecting non-finished requests directly instead of evaluating `req not in reqs_to_exclude` for every request. Nonempty exclusions keep the existing LIST membership comprehension (no set/hashability/equality change); explicit `keep_indices` and all subsequent metadata handling are untouched. The falsifiable CPU regression executes the AST-extracted production block: `/usr/bin/python3 -I test/filter_batch_pure_python_regression.py`. This is a Python scheduler-overhead micro-candidate, not a measured GPU/throughput gain; official evaluation and measurement remain with the trusted controller. No performance, EXACT, quality, or GPU claim is made.


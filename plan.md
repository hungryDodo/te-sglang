# Plan — worker glm-b, round 1, node a2: Sampler disabled-trace overhead

## Hypothesis
In `python/sglang/srt/layers/sampler.py::Sampler.forward`, the `SGLANG_TRACE_SAMPLER_E2E`
instrumentation builds debug kwargs / tuple shapes and calls trace helper functions on
every forward even when the env flag is disabled. Each disabled helper call still costs
a Python function call + `os.environ.get`/flag lookup, and disabled-path dict/tuple
construction allocates. Replacing the per-call disabled work with a cheap module-level
enabled check (read env dynamically each forward, so runtime enable/disable still works)
and skipping construction of debug shapes when off will remove that overhead from the
sampling path (CPU-side; runs outside CUDA graph).

## Steps
1. Read `python/sglang/srt/layers/sampler.py` (only the trace-related parts).
2. Implement: single cheap dynamic enabled-check per forward; when disabled, no helper
   calls and no debug kwargs/tuple construction; when enabled, behavior identical to
   today (same trace events, same helper calls, same fields). Sampler/logprob/token-sync
   operations unchanged in both paths.
3. Add ONE stdlib-only CPU regression test that:
   - Uses `ast` to extract `Sampler.forward` and verify call-order of core ops is
     unchanged vs. the source (structural check), and that disabled path does not build
     the debug shapes.
   - Uses fake dependency stubs (sys.modules injection) so importing sampler.py does
     NOT import the CUDA engine; asserts trace-off => helper not called / no debug kwargs
     built; trace-on => helper still called (dynamic enablement preserved).
4. Run test with `python3 -I`.
5. Write `iteration-proposal.json`, commit everything in one new commit.

## What this does NOT claim
- No GPU/measured perf number; sampler runs outside model CUDA graph.
- No quality/EXACT validation — controller does official evaluation.

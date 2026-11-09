# G3 t11 (deepseek-b) — frozen-workload KV-pool headroom oracle

Status: **bounded CPU analysis executed; no engine change; no throughput measured.**
`request_fast_gate=false`: nothing in this trial needs a score gate, and the
falsifier below reads artifacts the controller already owns.

## Question

The frozen launch fixes `--max-total-tokens 131072`, `--chunked-prefill-size
8192`, `--max-running-requests 256`, `--disable-radix-cache`. The branch record
has PrefillHeavy C16 at 59.3 tok/s / ~4046 ms TTFT — far below every other
point. Is that point (and DecodeCeiling C256) actually **KV-pool capacity
bound**, so that the retract/re-prefill machinery is on the critical path
rather than attention arithmetic?

This is a new question for this branch: prior trials closed the host-mirror
D2H lane, showed the BF16 Fast screen cannot reach the GGUF/MMVQ cells
(`f263262`, F3 UNGATED), and found the sliding-window prefill lever
acceptance-incompatible. None of them tested pool headroom arithmetic.

## Executed artifact

```
python3 -I seis/trials/deepseek-b/g3-t11/verify_pool_headroom.py \
    --root . --out seis/trials/deepseek-b/g3-t11/pool-headroom.json
# exit 0, verdict ARITHMETIC_BINDS, all_anchors_present=true
```

The script (stdlib only) checks 11 exact source anchors in the pinned tree
(pool cap is `min(profiled, user_limit)`; `check_decode_mem` ->
`check_decode_capacity`; `retract_decode` call site; retraction counter and
`#retracted_reqs:` log line; `disable=disable_radix_cache`), then computes
per-point demand from the frozen contract constants. **Hard gate:** any
missing anchor exits 2 and no claim is made.

## Result (arithmetic, not a measurement)

| point | C | prompt demand | max demand | pool | verdict |
|---|---|---|---|---|---|
| short | 32 / 128 | 4,096 / 16,384 | 5,120 / 20,480 | 131,072 | fits |
| medium | 32 / 128 | 16,384 / 65,536 | 18,432 / 73,728 | 131,072 | fits |
| standard | 32 / 64 | 32,768 / 65,536 | 36,864 / 73,728 | 131,072 | fits |
| decode_heavy | 64 / 128 | 16,384 / 32,768 | 49,152 / 98,304 | 131,072 | fits |
| **prefill_heavy** | **16** | **131,072** | **131,584** | **131,072** | **prompt alone exhausts the pool** |
| **decode_ceiling** | **256** | 8,192 | **139,264** | **131,072** | **over-commits by 8,192 from output token 481** |

Exact coincidences worth stating plainly:

- PrefillHeavy C16: `16 x 8192 = 131072 = max_total_tokens`. After the 16
  prompts, **headroom for decode is exactly zero**, and 512 decode tokens are
  still required. With radix cache disabled there is no eviction relief, so
  the source-anchored pressure valve is `retract_decode()` (re-queue +
  re-prefill).
- DecodeCeiling C256: `256 x (32+512) = 139264`, i.e. 8,192 tokens (6.25%)
  over capacity; the pool is exactly full at 480 generated tokens per request
  (`131072/256 = 512` total including the 32-token prompt). The last 32 output
  tokens per request cannot all be resident.

## Falsifier the controller can run at zero gate cost

Predictions are in `pool-headroom.json` -> `predictions_for_controller_artifacts`:

- **P1** PrefillHeavy C16 shows retraction activity (`num_retracted_reqs > 0`
  or `#retracted_reqs:` lines) in existing Fast server logs/metrics.
- **P2** PrefillHeavy retraction count scales with decode steps, not request
  arrivals (chunked prefill greedily refills the freed space).
- **P3** DecodeCeiling retraction, if any, appears only in the last ~6.25% of
  each generation (past output token 480).
- **P4** Controls (short/medium/standard/decode_heavy, all <= 98,304) show ~0
  retractions. Nonzero there falsifies the capacity reading.

If P1 is false on the real artifacts, this hypothesis is dead and must not be
carried forward. No new gate is needed to check it.

## Limitations (do not overstate)

- No SGLang execution, no GPU, no tok/s measured. The baselines 59.3 tok/s /
  4046 ms / 5145 tok/s are **branch-recorded history**, not re-measured.
- Frozen constants come from the controller brief (`configs/rsi` is not in the
  candidate sandbox). A different harness C or output budget changes the math.
- 131,072 assumes the user cap binds (`min(profiled, user_limit)`); the
  authority is the startup `max_total_num_tokens=` log line.
- Hybrid SWA has separate full-attention and SWA per-layer pools. This oracle
  bounds the full-attention pool only; which allocator binds at runtime is not
  resolved here.
- `new_token_ratio` reservation and chunked-prefill admission are **not**
  modeled. They can change whether the pool actually fills — hence P1-P4.
- Full-length output assumed (fixed output budget per contract); EOS early
  stop would weaken the over-commit.

## Engine status

`git diff` for this trial touches no `python/` file. F3 (`f263262`) remains
**UNGATED** and structurally unreachable from the frozen BF16 Fast screen
(applicability-001): a BF16 score cannot evaluate it; a separate GGUF
dispatch/correctness experiment is still pending on the controller.

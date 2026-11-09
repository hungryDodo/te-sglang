# G3 t12 (deepseek-b) — prefill batch width under the frozen chunk budget

Status: **CPU-executed source oracle; no engine change; no throughput measured.**
`request_fast_gate=false` (no patch this trial). F3 (`f263262`) remains UNGATED.

## Question

Canonical C16 corrected trial-9: the governing prefill budget is the whole-batch
`rem_chunk_tokens` (`chunked_prefill_size` = 8192 in the frozen launch), not
`max_prefill_tokens` (16384). Trial 9 had predicted **2 x 8192-token requests per
prefill batch / 8 waves** for the 15%-weight PrefillHeavy C16 point. This trial
executed the real budget plumbing to test that claim.

## Executed artifact

```
python3 -I seis/trials/deepseek-b/g3-t12/verify_prefill_chunk_budget.py \
    --root . --out seis/trials/deepseek-b/g3-t12/prefill-chunk-budget.json
# exit 0; 15/15 anchors present; verdict H1_CONFIRMED_ON_SOURCE__T9_2_PER_BATCH_FALSIFIED
```

The script AST-extracts from the pinned tree (hard exit 2 on drift) and executes:

- `scheduler.py:4160-4170`: the `PrefillAdder(...)` call passes positional
  `self.max_prefill_tokens` (rem_input_tokens = 16384) and
  `chunked_prefill_size` (rem_chunk_tokens = 8192), plus
  `running_bs if self.is_mixed_chunk else 0`.
- `schedule_policy.py:638-655`: `rem_chunk_tokens` initialization, including the
  real `-= num_mixed_decode_tokens` debit.
- `schedule_policy.py:_update_prefill_budget`: real
  `compute_charge = ceil_paged_tokens(extend_input_len) + replay_len` expression,
  then `self.rem_chunk_tokens -= compute_charge` and
  `self.rem_input_tokens -= compute_charge` **for every admitted request**.
- the real admission guards `if self.rem_chunk_tokens <= 0: return OTHER`
  (`budget_state:890`, `add_one_req_ignore_eos:1353`) and
  `cand_extend_input_len <= self.rem_chunk_tokens`.
- `arg_groups/fields/schedule.py:208`: `enable_mixed_chunk = False` (default).

## Result (executed arithmetic over real source)

| run | admitted per prefill batch | chunk tokens | rem_chunk after |
|---|---|---|---|
| **frozen** (`rem_chunk=8192`, extend=8192) | **1** | [8192] | 0 |
| control: chunking disabled (`rem_chunk=None`) | 2 | [8192, 8192] | n/a |
| control: chunk budget 16384 | 2 | [8192, 8192] | 0 |
| control: mixed chunk on, 16 running decodes | 1 | [**8176**] | 0 |

- **H1 confirmed:** one 8192-token prompt consumes the entire 8192-token
  whole-batch chunk budget; the real guard then rejects any second 8192-token
  request. With 16 concurrent PrefillHeavy requests, that is **16 prefill
  admission events per wave, not 8** (t9's prediction is falsified). The
  `max_prefill_tokens=16384` budget would have allowed two - the chunk budget is
  the binding constraint, exactly as canonical C16 states.
- **H2 rejected under the frozen launch:** `enable_mixed_chunk` defaults to
  `False` and the frozen command passes no such flag, so
  `num_mixed_decode_tokens = 0`; the 8176-token truncation control is
  sensitivity only, not frozen behavior.
- The `ignore_eos` admission path carries the same chunk guard
  (`add_one_req_ignore_eos:1328/1353`), so the result does not depend on whether
  the harness pins output length with `ignore_eos`.

## Interpretation boundaries

Arithmetic + executed source, **not** a measurement. It does not model
`max_prefill_bs`, the prefill delayer, memory-budget rejection, dynamic chunk
sizing, retraction/re-queue, or the actual number of sequential prefill steps
(64 requests at C16 means 64 admission events, interleaved with decode). It
therefore does not establish PrefillHeavy's wall time; it falsifies one stated
premise and exposes a candidate lever: the frozen `max_prefill_tokens=16384`
permits a wider prefill batch than the current whole-batch chunk debit allows.
Whether that headroom is safe (buffer sizing, activation memory) is **unknown**
and must be checked in source before any patch.

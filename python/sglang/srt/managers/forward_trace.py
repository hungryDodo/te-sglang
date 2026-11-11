"""Opt-in one-line-per-forward composition trace.

Enabled once at process start via SGLANG_FORWARD_TRACE=1 (default off).
Each executed model forward then emits exactly one INFO line

    FWD t=<monotonic_ms> prefill_tokens=<n> prefill_reqs=<n> decode_bs=<n> mixed=<bool> graph=<bool>

so decode-stall gaps can be attributed to the specific prefill forwards
interleaved with decode, next to the existing "Prefill batch," /
"running-req" scheduler logs. All fields are read off the batch object
that actually executed; nothing is inferred. This module is stdlib-only
(os/logging/time) so the guard logic is checkable on CPU without torch.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("", "0", "false", "no", "off")


def parse_forward_trace_env(raw):
    """Parse SGLANG_FORWARD_TRACE once at startup; reject garbage loudly."""
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"SGLANG_FORWARD_TRACE must be one of {list(_TRUTHY) + list(_FALSY[1:])} "
        f"(case-insensitive), got {raw!r}"
    )


# Single startup read; scheduler hot paths test this bool, never the env.
FORWARD_TRACE_ENABLED = parse_forward_trace_env(os.environ.get("SGLANG_FORWARD_TRACE"))


def capture_composition(batch):
    """Snapshot the assembled composition of one forward as a 4-tuple.

    Called at Scheduler.run_batch entry, before overlap staging consumes
    mix_running_indices, on the ScheduleBatch that is about to execute:

    - mixed:     forward_mode is MIXED (enable_mixed_chunk merged running
                 decodes into an extend batch).
    - decode_bs: running-decode rows. MIXED counts len(mix_running_indices)
                 (the rows mix_with_running folded in); decode / idle /
                 spec verify / draft-extend steps count every row; plain
                 extend (prefill) batches carry none.
    - prefill_reqs: batch_size() minus the decode rows above.
    - prefill_tokens: extend_num_tokens minus the mixed-in decode rows'
                 1-token entries (mix_with_running adds one per row).
    """
    mode = batch.forward_mode
    mixed = mode.is_mixed()
    if mixed:
        decode_bs = len(batch.mix_running_indices)
    elif (
        mode.is_decode_or_idle()
        or mode.is_target_verify()
        or mode.is_draft_extend_v2()
    ):
        decode_bs = batch.batch_size()
    else:
        decode_bs = 0
    prefill_tokens = batch.extend_num_tokens or 0
    if mixed:
        prefill_tokens -= decode_bs
    return (prefill_tokens, batch.batch_size() - decode_bs, decode_bs, mixed)


def log_forward_composition(prefill_tokens, prefill_reqs, decode_bs, mixed, graph):
    """Emit the single FWD line for one executed forward."""
    logger.info(
        "FWD t=%d prefill_tokens=%d prefill_reqs=%d decode_bs=%d mixed=%s graph=%s",
        int(time.monotonic() * 1000),
        prefill_tokens,
        prefill_reqs,
        decode_bs,
        mixed,
        graph,
    )

# Controller implementation: SGLANG_SPLIT_DECODE_PREFILL split sub-forwards.
# Spec: MIXED-GRAPH-PLAN.json (glm-a r7, adopted 24aab5fea3).
# Applied as a patch-style series of edits to scheduler.py.
import ast
from pathlib import Path

SCHED = Path("python/sglang/srt/managers/scheduler.py")
src = SCHED.read_text()

# ---- 1) Flag parsing (module level, after logger) ----
anchor1 = 'logger = logging.getLogger(__name__)\n'
flag_block = '''

def _parse_split_decode_prefill(raw):
    """SGLANG_SPLIT_DECODE_PREFILL: exactly '1' (on), '0' (off), or unset (off)."""
    if raw is None:
        return False
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(
        f"SGLANG_SPLIT_DECODE_PREFILL must be exactly '1' or '0' (unset = off), got {raw!r}"
    )


_SPLIT_DECODE_PREFILL = _parse_split_decode_prefill(
    __import__("os").environ.get("SGLANG_SPLIT_DECODE_PREFILL")
)
'''
assert src.count(anchor1) == 1, "logger anchor"
src = src.replace(anchor1, anchor1 + flag_block, 1)

# ---- 2) Scheduler __init__ guards (after is_mixed_chunk) ----
anchor2 = """        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
        )
"""
guard_block = anchor2 + """        # MIXED-GRAPH-PLAN (glm-a r7, 24aab5f): opt-in split decode/prefill
        # composition. Guarded to the plain single-replica generation config;
        # every other config keeps today's behavior with the env flag inert.
        self.split_decode_prefill = (
            _SPLIT_DECODE_PREFILL
            and self.is_generation
            and self.spec_algorithm.is_none()
            and not self.require_mlp_sync
            and not self.attn_dp_enabled
            and not self.enable_hisparse
            and not self.enable_pdmux
            and str(get_disagg().disaggregation_mode) == "null"
            and get_parallel().pp_size == 1
        )
        self._split_decode_companion = None
"""
assert src.count(anchor2) == 1, "is_mixed_chunk anchor"
src = src.replace(anchor2, guard_block, 1)

# ---- 3) Mix block: split path (replace the mix_with_running call) ----
anchor3 = """            if not running_batch.is_empty() and running_batch.check_decode_mem():
                running_batch.prepare_for_decode()
                new_batch.mix_with_running(running_batch)
                new_batch.decoding_reqs = running_batch.reqs"""
replacement3 = """            if not running_batch.is_empty() and running_batch.check_decode_mem():
                running_batch.prepare_for_decode()
                if self.split_decode_prefill:
                    # MIXED-GRAPH-PLAN step A: keep the decode batch separate
                    # so it replays its CUDA graph; the eager extend (prefill)
                    # batch and this decode batch run back-to-back this step.
                    self._split_decode_companion = running_batch
                else:
                    new_batch.mix_with_running(running_batch)
                    new_batch.decoding_reqs = running_batch.reqs"""
assert src.count(anchor3) == 1, "mix anchor"
src = src.replace(anchor3, replacement3, 1)

# ---- 4) chunked_prefill budget: include decode rows under split too ----
anchor4 = "            running_bs if self.is_mixed_chunk else 0,"
replacement4 = """            running_bs if (self.is_mixed_chunk or self.split_decode_prefill) else 0,"""
assert src.count(anchor4) == 1, "budget anchor"
src = src.replace(anchor4, replacement4, 1)

# ---- 5) event_loop_normal: run companion after prefill ----
anchor5 = """            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self._sched_idled = True
                self.on_idle()
"""
replacement5 = """            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                # MIXED-GRAPH-PLAN: companion decode batch (graph replay)
                companion = self._pop_split_decode_companion()
                if companion is not None:
                    dres = self.run_batch(companion)
                    self.process_batch_result(companion, dres)
            else:
                # When the server is idle, do self-check and re-init some states.
                self._sched_idled = True
                self.on_idle()
"""
assert src.count(anchor5) == 1, "normal loop anchor"
src = src.replace(anchor5, replacement5, 1)

# ---- 6) event_loop_overlap: run companion after prefill ----
anchor6 = """            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self._sched_idled = True
"""
replacement6 = """            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
                # MIXED-GRAPH-PLAN: companion decode batch (graph replay)
                # runs back-to-back on the same forward stream; its result is
                # queued exactly like a normal decode step.  The prefill
                # result stays in the queue too (two entries this iteration);
                # _reqs_finishing_inflight is only called on the (replaced)
                # mixed path, so the queue-depth-1 assumption there is safe.
                companion = self._pop_split_decode_companion()
                if companion is not None:
                    dres = self.run_batch(companion)
                    self._apply_war_barrier()
                    self.result_queue.append((companion.copy(), dres))
            else:
                batch_result = None
                self._sched_idled = True
"""
assert src.count(anchor6) == 1, "overlap loop anchor"
src = src.replace(anchor6, replacement6, 1)

# ---- 7) pop_and_process: drain up to queue length ----
anchor7 = """        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
"""
replacement7 = """        def pop_and_process():
            # Process the results of the last batch (drain all entries: the
            # split-decode iteration appends prefill+decode companions).
            while self.result_queue:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
"""
assert src.count(anchor7) == 1, "pop anchor"
src = src.replace(anchor7, replacement7, 1)

# ---- 8) Helper method (near the end of the Scheduler class, before run_batch) ----
anchor8 = """    @scheduler_stage_method(SCHEDULER_STAGE_RUN_BATCH)
    def run_batch("""
helper = '''    def _pop_split_decode_companion(self):
        """Take the companion decode batch prepared by this step's prefill pass.

        Returns None when the split path did not fire (default, non-split
        configs, or no runnable decode rows).  The companion's requests were
        already advanced by prepare_for_decode and will re-enter
        running_batch through the normal decode-result path.
        """
        companion = self._split_decode_companion
        self._split_decode_companion = None
        if companion is not None and companion.is_empty():
            return None
        if companion is not None:
            logger.info(
                "SPLIT step: decode_bs=%d (companion to eager extend)",
                companion.batch_size(),
            )
        return companion

''' + anchor8
assert src.count(anchor8) >= 1, "run_batch anchor"
src = src.replace(anchor8, helper, 1)

SCHED.write_text(src)
# validate
ast.parse(src)
print("scheduler.py patched and parses OK; split refs:", src.count("split_decode"))

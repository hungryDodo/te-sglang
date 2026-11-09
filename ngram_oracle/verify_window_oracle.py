#!/usr/bin/env python3
"""NGRAM verify-window custom-mask block-alignment oracle (CPU, stdlib only).

New question vs. mask_width_oracle (t5): t5 modeled ONE request's row width.
The FA2 CUDA-graph verify path packs the *flat concatenated* custom mask with
**device-cumulative** block offsets while the mask is *built* with **host**
per-request widths. With more than one request in the batch, a single drift can
therefore shift every later request's block, not only its own rows.

Pinned source lines modeled (real code, ported by hand; no torch here):
  python/sglang/srt/speculative/ngram_worker.py:352-361
      ones_masks[i] = torch.ones((draft_token_num, batch.seq_lens_cpu[i]))
      ... torch.cat((ones_masks[i], mask[i]), dim=1)            # [D, h_i + D]
  python/sglang/srt/speculative/ngram_info.py:76-117
      paged_kernel_lens = paged_kernel_lens + draft_token_num   # device d_i + D
      cum_kv_seq_len[1:] = cumsum(paged_kernel_lens)
      mask_numel = paged_kernel_lens_sum*D + D^2*bs             # device-derived
      if custom_mask.numel() < mask_numel: append True          # global tail
  python/sglang/srt/layers/attention/flashinfer_backend.py:263-285
      packed_lengths = (kv_lens_host * uniform_verify_width + 7)//8
      packed_indptr[1:] = cumsum(packed_lengths)
      assert packed_indptr[-1] <= self._custom_mask_buf.numel()  # capacity only
  python/sglang/kernels/ops/speculative/pack_verify_mask.py:12-24
      kv_start = kv_indptr[row]; count = (kv_end - kv_start) * Q
      values = load(mask + kv_start*Q + offsets, offsets < count)
      start = packed_indptr[row]; store(output + start + byte, packed)

Column contract (t5): column c < L = d_i + D of row r maps to KV stream entry
cum_kv_seq_len[i] + c, i.e. logical position c (prefix 0..d_i-1 then draft node
c-d_i). The packer's block start for request i is kv_start*Q with kv_start the
DEVICE cumulative widened length. The built mask block for request i starts at
the HOST cumulative widened length * D. These agree iff h_j == d_j for all j<i.

Run: python3 -I ngram_oracle/verify_window_oracle.py ngram_oracle/results/verify_window_oracle.json
"""

import json
import sys

D = 4  # draft_token_num
HOST = [4, 5, 6]   # per-request host prefix lengths h_i used to build the mask
DEV = [4, 5, 6]    # per-request device prefix lengths d_i used by indptr/pack


def build_flat_mask(host_lens, dev_lens, mask_numel):
    """Port of ngram_worker mask build + ngram_info.generate_attn_arg_prefill.

    Query-major per-request block [D rows][h_i + D cols]; the K/V column c maps
    to logical position c (prefix ones, then the D draft nodes). Then the global
    device-derived mask_numel padding with True (if short) or truncation of the
    surplus (if long, because only mask_numel bits are ever consumed).
    """
    rows = []
    block_starts = []
    pos = 0
    for h in host_lens:
        block_starts.append(pos)
        for r in range(D):
            row = []
            for c in range(h + D):
                if c < h:
                    row.append(1)          # prefix ones block
                else:
                    j = c - h              # draft node j
                    # proper causal tree row: self + strict ancestors only
                    row.append(1 if _is_ancestor_or_self(r, j) else 0)
            rows.append(row)
            pos += h + D
    flat = [b for row in rows for b in row]
    padded = False
    if len(flat) < mask_numel:
        flat = flat + [1] * (mask_numel - len(flat))   # silent append-True
        padded = True
    return flat[:mask_numel], block_starts, padded, len(flat)


def _is_ancestor_or_self(r, j):
    """Fixed BFS tree: 0 root; 1,2 children of 0; 3 child of 1. Node ids < D.

    Bits above the diagonal are impossible for a causal tree, identical to the
    fillResult contract audited in t3; the exact topology does not matter for
    block arithmetic, only that rows are proper causal rows.
    """
    parent = {0: None, 1: 0, 2: 0, 3: 1}
    node = j
    while node is not None:
        if node == r:
            return True
        node = parent[node]
    return False


def packer_read(flat, host_lens, dev_lens):
    """Port of pack_verify_mask_into bit addressing (byte granularity kept)."""
    cum = [0]
    for d in dev_lens:
        cum.append(cum[-1] + d + D)          # device widened cumsum
    packed_indptr = [0]
    for d in dev_lens:
        packed_indptr.append(packed_indptr[-1] + ((d + D) * D + 7) // 8)
    out = {}
    for i in range(len(dev_lens)):
        kv_start = cum[i]
        count = (cum[i + 1] - cum[i]) * D     # (d_i + D) * D bits
        for byte in range((count + 7) // 8):
            # 8 consecutive bits of the flat block, as the Triton kernel does
            bits = []
            for b in range(8):
                off = byte * 8 + b
                if off < count:
                    src = kv_start * D + off
                    bits.append(flat[src] if 0 <= src < len(flat) else -1)
                else:
                    bits.append(0)
            out[(i, byte)] = bits
    return out, packed_indptr, cum


def expected_block(prefix_used):
    """What the packer SHOULD read for request i if the build used the same lens.

    The host-built block for request i begins at host-cumulative offset; the
    packer reads from device-cumulative offset. Under h==d they coincide.
    """
    return prefix_used


def verify(host_lens, dev_lens):
    mask_numel = sum(d + D for d in dev_lens) * D
    flat, block_starts, padded, raw_len = build_flat_mask(host_lens, dev_lens, mask_numel)
    packed, packed_indptr, cum = packer_read(flat, host_lens, dev_lens)
    host_starts = []
    p = 0
    for h in host_lens:
        host_starts.append(p)
        p += (h + D) * D
    per_req = []
    for i in range(len(dev_lens)):
        exp_start = host_starts[i]                     # where the builder put it
        got_start = cum[i] * D                         # where the packer read it
        row_width_dev = dev_lens[i] + D
        mismatch = 0
        checked = 0
        first = None
        for r in range(D):
            for c in range(row_width_dev):
                # intended mask bit under the HOST-built layout
                if c < host_lens[i]:
                    exp = 1
                else:
                    j = c - host_lens[i]
                    exp = 1 if (0 <= j < D and _is_ancestor_or_self(r, j)) else 0
                # bit the packer delivers at flat position r*(d_i+D)+c
                src = got_start + r * row_width_dev + c
                got = flat[src] if 0 <= src < len(flat) else None
                checked += 1
                if got != exp:
                    mismatch += 1
                    if first is None:
                        first = {"row": r, "col": c, "expected": exp, "delivered": got}
        per_req.append(
            {
                "req": i,
                "host_prefix": host_lens[i],
                "device_prefix": dev_lens[i],
                "builder_block_start": exp_start,
                "packer_block_start": got_start,
                "block_bits": (dev_lens[i] + D) * D,
                "bits_checked": checked,
                "bits_mismatch": mismatch,
                "first_mismatch": first,
                "exact": mismatch == 0,
            }
        )
    return {
        "host_lens": host_lens,
        "device_lens": dev_lens,
        "mask_numel_device": mask_numel,
        "raw_built_bits": raw_len,
        "appended_true_bits": max(0, mask_numel - raw_len),
        "host_block_starts": host_starts,
        "device_block_starts": [c * D for c in cum[:-1]],
        "start_shift_bits": [cum[i] * D - host_starts[i] for i in range(len(dev_lens))],
        "packed_indptr_bytes": packed_indptr,
        "per_request": per_req,
        "all_requests_exact": all(r["exact"] for r in per_req),
    }


def main(out_path):
    results = {}
    results["aligned"] = verify(HOST, DEV)
    drift = list(DEV)
    drift[1] = DEV[1] + 1  # request 1 device prefix one token longer than host
    results["single_req_drift_h_lags_d"] = verify(HOST, drift)
    ahead = list(DEV)
    ahead[0] = DEV[0] - 1  # request 0 host one longer than device
    results["single_req_drift_h_ahead_d"] = verify(HOST, ahead)
    results["summary"] = {
        "aligned_all_exact": results["aligned"]["all_requests_exact"],
        "drift_mid_batch": {
            "req0_exact": results["single_req_drift_h_lags_d"]["per_request"][0]["exact"],
            "req1_exact": results["single_req_drift_h_lags_d"]["per_request"][1]["exact"],
            "req2_exact": results["single_req_drift_h_lags_d"]["per_request"][2]["exact"],
            "req2_bits_mismatch": results["single_req_drift_h_lags_d"]["per_request"][2]["bits_mismatch"],
        },
        "drift_first_req": {
            r["req"]: r["exact"] for r in results["single_req_drift_h_ahead_d"]["per_request"]
        },
        "cross_request_contamination": (
            not results["single_req_drift_h_lags_d"]["per_request"][2]["exact"]
        ),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout"))

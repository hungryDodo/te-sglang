# G3 trial 5 — deepseek-b (node a4): evaluate f263262 and its dispatch instrument

Date 2026-11-09. Campaign continuous-20261109-01, Iteration 5. No GPU, no SSH,
no model weights here; the controller runs every measurement. Nothing in this
file or in the committed evidence is a performance measurement.

## Inherited state (read, not re-derived)

- HEAD `57fe927` = `f263262` (GGUF MMVQ thresholds) + `ec55606`/`89a5d10`
  (BF16 decodes) + `33e5ac4`/`a306e40` (host-mirror decode+prefill) + `4e1644a`
  (observation-only GGUF dispatch tracer). Engine diff since `f263262`:
  `flashinfer_backend.py` (+27/-2, host mirrors) and `gguf.py` (+101, tracer).
- **F3 (`f263262`) is UNGATED.** The frozen formal gate loads BF16 safetensors
  rev `e6406af8...` with no GGUF file and no `--quantization gguf`
  (controller applicability-001), so the changed MMVQ/MMQ cells are not on the
  gate path at all. No BF16 number can close or refute F3.
- The G3 t2 GGUF dispatch experiment (`g3-t2/GGUF_DISPATCH_EXPERIMENT.md`,
  tracer `4e1644a`) is still NOT_TESTED. GGUF custom GPU experiments are
  pending controller-side.
- The t4 fast-gate request (`33e5ac4`+`a306e40`, decode+prefill SWA host
  mirrors) is inherited; its outcome is not visible to this session. I make no
  claim about it and will not disturb that candidate's provenance.

## Goal of this trial (CPU-executable, bounded)

1. **Evaluate `f263262` to the limit of what CPU evidence can settle.** Replace
   the t1 regex replay with an AST extraction that *executes the shipped
   functions themselves* (and the `f263262^` legacy ones) over the full
   `WeightType` × rows-band × batch grid, and record exactly which dispatch
   cells change, which of those have measured support, and whether any can be
   reached at any frozen gate point.
2. **Audit the instrument the eventual GGUF experiment depends on.** The
   committed tracer is the only mechanism that can prove runtime dispatch. A
   trace is only valid if (a) every MMVQ/MMQ decision routes through the traced
   function, (b) the recorded fields identify the decision inputs, and (c) the
   trace survives the way a server is actually stopped. Test each on CPU;
   patch only if a concrete gap is found, and keep any patch inert when the
   env var is off.
3. **State the honest bound on the next full-engine levers** (long-context
   prefill window, decode sync) from already-committed branch evidence, with
   uncertainty labelled. No new measurement is claimed.

## Deliverables

- `seis/trials/deepseek-b/g3-t5/verify_dispatch_contract.py` + raw JSON:
  executed selection-contract table (shipped vs legacy), changed-cell list,
  measured-support flags, gate-batch invariance, tracer coverage/lifecycle
  results.
- Engine change only if the tracer audit finds a real gap; otherwise the tree
  stays byte-identical for the pending t4 gate.
- `iteration-proposal.json` (string fields + boolean), final commit.

## Explicit non-goals

- No claim that `f263262` is gated, promoted, or performance-positive in the
  engine; no BF16 score read as F3 signal.
- No repeated request for the same unexecuted GGUF experiment as if it were
  new progress; the proposal states what evidence would close F3 and what the
  tracer can and cannot prove.
- No change to measurement, corpus, seed, reference, evaluator or frozen
  protocol; no precision change.

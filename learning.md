# Learning — glm-b round 3 (q8-research-20261111-01)

- Round 2 died with ECONNRESET before any tokens or code; this round kept a
  narrow sequential plan (locate -> write helper + call site -> CPU test -> GPU
  script -> proposal -> commit) and budgeted the last turns for test/commit only.
- The shared packet's glm-a "metadata V2 batch" outputs are diagnostics only;
  they are not a causal bug, a quality verdict, or a screen result for this
  candidate. No cross-worker inference was used to justify this change.
- Engineering controls delimit what a CPU test can prove: AST + fake-tensor
  execution verifies guard logic, fallback identity (including preserved error
  behavior), no-alias/no-mutation, launch wiring, and the SOURCE-TEXT rounding
  order (fp32 sigmoid -> cast to attn dtype -> multiply). It cannot prove the
  Triton kernel compiles or matches GPU numerics — `tl.sigmoid` vs
  `torch.sigmoid` implementations may differ by ulps, so exact equality must be
  observed on hardware by the Controller via the prepared NOT_RUN script.
- Attribution discipline: parent 8cfcffb already carries the adopted r1 sampler
  patch, so any paired screen against trusted 5b89820 measures the combination;
  the parent's single C1 1.00196 exploratory screen gain is not evidence of a
  repeatable effect.
- Preserving the eager baseline function next to the fused path made the
  "fallback is EXACT original" and GPU A/B requirements cheap and auditable.

# Deterministic CPT copy

Status: implemented; 11 local regression tests passed; GPU acceptance pending.

User request: independent copy of gplsi/LMTK hpc with minimal changes for
repeatable CPT. Upstream snapshot: 65e8ddc45c746609c55310ce27096ed9b6568767.

Scope: the existing CLM/Fabric path, fresh training runs on an identical
software/hardware/data stack. Do not claim cross-platform determinism. Require
a seed, deterministic PyTorch algorithms and controlled data generators.
Retain BF16, FSDP, workers and the original model. Do not implement resume:
upstream checkpoints do not restore all RNG/data-loader state. Reject resume
and initial_weights_checkpoint in strict mode (the latter also has a preexisting
loading issue). Normal from_pretrained CPT remains supported.

Related work: issues/42-online-packing.md and its execution plan discuss
deterministic sampling, but online packing is out of scope. Existing trainer
uses local pretokenized datasets. PyTorch reproducibility guidance:
https://docs.pytorch.org/docs/stable/notes/randomness.html
Fabric gradient accumulation guidance:
https://lightning.ai/docs/fabric/stable/advanced/gradient_accumulation.html

Changes are delivered as two separate patches: strict reproducibility, then
gradient normalization/clipping correctness. The latter changes the training
algorithm and is not evidence of a cause of run-to-run variability.

Validation: dependency-free regression tests execute extracted production
methods with numerical test doubles. Real tensor and repeated GPU training
checks must run in the user's existing cluster container. This environment has
no PyTorch/GPU; package installation was blocked by network approval. No SLURM
jobs submitted. Record cluster job IDs, log paths, environment manifests and
checkpoint comparison output here when running the acceptance procedure.

Local verification: `python tests/unit/training/test_deterministic_copy.py` passed 11 tests. Shell syntax and modified Python parsing passed. GPU jobs: none; SLURM ID: not applicable. Kernel-level repeatability remains unverified.

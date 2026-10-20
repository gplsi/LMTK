---
number: 5
title: "Convert to huggingface"
state: closed
labels:
---

Added unit tests, schema and task code for publishing the checkpoints to huggingface.

The task is divided in two main modules:

· Format handler: for converting the current format of the checkpoints to the best recommended format according to the repository used (currently for huggingface only with fsdp format).

· Uploader: for handling the uploading of the model with the suitable format. Only Huggingface right now.

The task is parallel with the other tasks already developed and uses an orchestrator that inherits from the base orchestrator, using the expected logging system.
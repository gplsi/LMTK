# Convert Task Guide

## Overview
This guide explains how to use the LMTK `convert` task to transform FSDP PyTorch checkpoints into HuggingFace-compatible format and save them locally. The convert task supports both single checkpoint files and batch conversion from a directory, with each converted model saved in its own folder.

## Key Features
- **Local Conversion**: Converts FSDP `.pth` checkpoints to HuggingFace format without uploading.
- **Batch Support**: Handles both single checkpoint files and directories containing multiple checkpoints.
- **Configurable via YAML**: All parameters are specified in a config file.
- **Progress Reporting**: Logs progress and summarizes results at the end.
- **Robust Logging**: Detailed logging for conversion steps and errors.

## Example Workflow

### 1. Prepare Your Config File

Create a YAML config (e.g. `config/experiments/convert_test/gpt-2_all_converted.yaml`):

```yaml
task: convert
experiment_name: gpt2_fsdp_to_hf
verbose_level: 2

convert:
  base_model: "gpt2"
  checkpoint_path: "/workspace/output"
  initial_format: "fsdp"
  final_format: "huggingface"
  output_dir: "/workspace/converted"
```

- `base_model`: HuggingFace model name or path (used for config/tokenizer).
- `checkpoint_path`: Path to a `.pth` checkpoint file or a directory containing multiple checkpoints.
- `initial_format`: Set to `fsdp` for FSDP checkpoints.
- `final_format`: Set to `huggingface` to convert to HuggingFace format.
- `output_dir`: Where to save the converted HuggingFace model(s).

### 2. Run the Convert Task

From the project root, run:

```bash
python3 src/main.py --config=config/experiments/convert_test/gpt-2_all_converted.yaml
```

### 3. Output Structure

- For a single checkpoint: the converted model and tokenizer are saved in a folder under `output_dir` named after the checkpoint file (without extension).
- For a directory of checkpoints: each `.pth` file is converted and saved in its own subfolder under `output_dir`.

### 4. Progress and Summary

- Progress and status are logged via the logger (see console or log file).
- At the end, a summary dictionary is logged and returned, showing which checkpoints succeeded or failed, and their output locations.

## Troubleshooting

- **Missing Keys/Weight Tying**: The convert task automatically handles FSDP key remapping and weight tying. Warnings about missing or unexpected keys are logged for review.
- **No output or partial output**: Check the log for errors. Common issues include incompatible checkpoint formats or missing base model files.

## Example Directory Structure

```
continual-pretraining-framework/
│
├── config/
│   └── experiments/
│       └── convert_test/
│           └── gpt-2_all_converted.yaml
├── output/
│   └── epoch-001-final-ckpt.pth
├── converted/
│   └── epoch-001-final-ckpt/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.json
└── ...
```

## References
- See also: [PUBLISH.md](./PUBLISH.md) for uploading models to HuggingFace Hub after conversion.
- For schema details: `config/schemas/convert.schema.yaml`.

---

For further help, open an issue or consult the development team.

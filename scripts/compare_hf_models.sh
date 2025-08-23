#!/usr/bin/env bash
set -euo pipefail

# Compare two Hugging Face model directories (e.g., converted checkpoints)
# Creates/uses a dedicated virtual environment for a clean run.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_compare_models"

usage() {
  cat <<EOF
Usage: $(basename "$0") --model-a PATH --model-b PATH [--tol FLOAT]

Options:
  --model-a PATH    Path to first HF model dir (e.g., /path/to/epoch-001-ckpt)
  --model-b PATH    Path to second HF model dir (e.g., /path/to/epoch-002-ckpt)
  --tol FLOAT       Allclose tolerance for diffs (default: 1e-6)

Example:
  $(basename "$0") \
    --model-a /workspace/outputs/converted/roberta_mlm_ddp_baseline/epoch-001-ckpt \
    --model-b /workspace/outputs/converted/roberta_mlm_ddp_baseline/epoch-002-ckpt \
    --tol 1e-6
EOF
}

MODEL_A="/workspace/outputs/converted/roberta_mlm_ddp_baseline/epoch-001-ckpt"
MODEL_B="/workspace/outputs/converted/roberta_mlm_ddp_baseline/epoch-005-ckpt"
TOL="1e-6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-a) MODEL_A="$2"; shift 2;;
    --model-b) MODEL_B="$2"; shift 2;;
    --tol) TOL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "$MODEL_A" || -z "$MODEL_B" ]]; then
  echo "ERROR: --model-a and --model-b are required" >&2
  usage
  exit 1
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Model A     : ${MODEL_A}"
echo "Model B     : ${MODEL_B}"
echo "Tolerance   : ${TOL}"

# Normalize container-style paths to host project paths if needed
if [[ "${MODEL_A}" == /workspace/* ]]; then
  MODEL_A="${PROJECT_ROOT}${MODEL_A#/workspace}"
fi
if [[ "${MODEL_B}" == /workspace/* ]]; then
  MODEL_B="${PROJECT_ROOT}${MODEL_B#/workspace}"
fi

# Validate directories exist
if [[ ! -d "${MODEL_A}" ]]; then
  echo "ERROR: Model A directory not found: ${MODEL_A}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_B}" ]]; then
  echo "ERROR: Model B directory not found: ${MODEL_B}" >&2
  exit 2
fi

# Prepare venv
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}" || {
    echo "venv module not available; attempting virtualenv"
    python3 -m pip install --user --quiet virtualenv --break-system-packages
    python3 -m virtualenv "${VENV_DIR}"
  }
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --quiet --upgrade pip
# CPU wheels to avoid CUDA requirement
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu
python -m pip install --quiet transformers==4.44.2 safetensors==0.4.3

# Export env vars for the embedded Python to consume
export CMP_MODEL_A="${MODEL_A}"
export CMP_MODEL_B="${MODEL_B}"
export CMP_TOL="${TOL}"

python - << PY
import argparse, os, math
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

model_a = os.environ.get("CMP_MODEL_A")
model_b = os.environ.get("CMP_MODEL_B")
tol = float(os.environ.get("CMP_TOL", "1e-6"))

def load_sd(path):
    last_err = None
    for loader in (AutoModelForMaskedLM, AutoModelForCausalLM, AutoModel):
        try:
            m = loader.from_pretrained(path, local_files_only=True)
            m.eval()
            return m.state_dict()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load model from {path}: {last_err}")

def tensor_stats(diff):
    diff = diff.detach().cpu().to(torch.float32)
    return {
        "l2": diff.pow(2).sum().sqrt().item(),
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
    }

sd_a = load_sd(model_a)
sd_b = load_sd(model_b)

keys_a = set(sd_a.keys())
keys_b = set(sd_b.keys())
only_a = sorted(list(keys_a - keys_b))
only_b = sorted(list(keys_b - keys_a))
common = sorted(list(keys_a & keys_b))

print("Keys A:", len(keys_a), "Keys B:", len(keys_b), "Common:", len(common))
if only_a:
    print("Only in A (sample):", only_a[:10])
if only_b:
    print("Only in B (sample):", only_b[:10])

total_l2 = 0.0
total_params = 0
num_bad = 0
top = []  # (l2, name, stats)

for k in common:
    ta, tb = sd_a[k], sd_b[k]
    if ta.shape != tb.shape:
        print(f"SHAPE MISMATCH: {k} {ta.shape} vs {tb.shape}")
        num_bad += 1
        continue
    diff = (ta - tb).detach().cpu().to(torch.float32)
    st = tensor_stats(diff)
    total_l2 += st["l2"]
    total_params += diff.numel()
    if st["max_abs"] > tol:
        num_bad += 1
    top.append((st["l2"], k, st))

top.sort(reverse=True)
print("\nTop-10 tensors by L2 difference:")
for l2, name, st in top[:10]:
    print(f"  {name}: l2={st['l2']:.6f}, mean_abs={st['mean_abs']:.6e}, max_abs={st['max_abs']:.6e}")

print("\nSummary:")
print("  total_common_tensors:", len(common))
print("  tensors_exceeding_tol:", num_bad)
print("  total_l2_diff:", round(total_l2, 6))
print("  total_params_compared:", total_params)
PY

echo "Comparison completed."



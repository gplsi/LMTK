#!/usr/bin/env bash
set -euo pipefail

# Visualize tokenized dataset samples using a dedicated virtual environment
# Defaults target the WikiText-103 MLM tokenization output directory

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATASET_PATH="${PROJECT_ROOT}/data/tokenized/mlm_turismo/train"
DEFAULT_NUM_SAMPLES=5
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/output/tokenized_visuals_cli"
DEFAULT_TOKENIZER="BSC-LT/roberta-base-bne"
VENV_DIR="${PROJECT_ROOT}/.venv_visualize"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dataset-path PATH] [--num-samples N] [--output-dir DIR] [--tokenizer NAME]

Options:
  --dataset-path PATH   Path to tokenized dataset (default: ${DEFAULT_DATASET_PATH})
  --num-samples N       Number of samples to visualize (default: ${DEFAULT_NUM_SAMPLES})
  --output-dir DIR      Output directory for visualization JSON (default: ${DEFAULT_OUTPUT_DIR})
  --tokenizer NAME      HF tokenizer for decoding (default: ${DEFAULT_TOKENIZER})
  -h, --help            Show this help

Example:
  $(basename "$0") --dataset-path ${DEFAULT_DATASET_PATH} --num-samples 5 --tokenizer roberta-base
EOF
}

DATASET_PATH="${DEFAULT_DATASET_PATH}"
NUM_SAMPLES="${DEFAULT_NUM_SAMPLES}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
TOKENIZER_NAME="${DEFAULT_TOKENIZER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"; shift 2;;
    --num-samples)
      NUM_SAMPLES="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --tokenizer)
      TOKENIZER_NAME="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"; usage; exit 1;;
  esac
done

echo "Project root: ${PROJECT_ROOT}"
echo "Dataset path: ${DATASET_PATH}"
echo "Num samples : ${NUM_SAMPLES}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Tokenizer   : ${TOKENIZER_NAME}"

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
python -m pip install --quiet datasets==2.21.0 pyarrow==16.1.0 transformers==4.44.2

mkdir -p "${OUTPUT_DIR}"
python "${PROJECT_ROOT}/scripts/visualize_tokenized_dataset.py" \
  --dataset_path "${DATASET_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --output_dir "${OUTPUT_DIR}" \
  --tokenizer "${TOKENIZER_NAME}"

echo "Done. Visualization written under: ${OUTPUT_DIR}"



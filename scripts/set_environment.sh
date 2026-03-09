if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: run this script with 'source scripts/set_environment.sh' so the environment stays active." >&2
  exit 1
fi

CONDA_SH_PATH="${CONDA_SH_PATH:-/home/gplsi/rst29/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lmtk}"

echo "Setting up environment..."

if [[ -f "$CONDA_SH_PATH" ]]; then
  source "$CONDA_SH_PATH"
else
  echo "Error: conda initialization script not found at: $CONDA_SH_PATH" >&2
  echo "Set CONDA_SH_PATH or update scripts/set_environment.sh to match the cluster Conda install." >&2
  return 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: 'conda' command is still unavailable after sourcing: $CONDA_SH_PATH" >&2
  return 1
fi

if ! conda activate "$CONDA_ENV_NAME"; then
  echo "Error: failed to activate Conda environment '$CONDA_ENV_NAME'." >&2
  echo "Update CONDA_ENV_NAME or create the environment before submitting to SLURM." >&2
  return 1
fi

#export PYTHONPATH="src/":$PYTHONPATH

echo "Environment set successfully."

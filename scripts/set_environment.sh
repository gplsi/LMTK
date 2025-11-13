if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: run this script with 'source scripts/set_environment.sh' so the environment stays active." >&2
  exit 1
fi

echo "Setting up environment..."

if [[ -f /leonardo_work/EUHPC_D22_034/miniconda3/etc/profile.d/conda.sh ]]; then
  source /leonardo_work/EUHPC_D22_034/miniconda3/etc/profile.d/conda.sh
else
  echo "Error: conda initialization script not found." >&2
  return 1
fi

conda activate lmtk || return 1
#export PYTHONPATH="src/":$PYTHONPATH

echo "Environment set successfully."

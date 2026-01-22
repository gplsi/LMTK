#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_ENV_FILE="$SCRIPT_DIR/slurm_test.env"
SECRETS_FILE="$SCRIPT_DIR/test_secrets.env"
SUBMIT_SCRIPT="$PROJECT_ROOT/slurm/submit_job.sh"

usage() {
    cat << EOF_USAGE
Usage: $0 [OPTIONS]

Submit SLURM test runs using slurm/submit_job.sh and a testing config.

Options:
    --unit                     Use the unit test config (default)
    --integration              Use the integration smoke config
    --config PATH              Use a specific config path
    --dry-run                  Print the sbatch command without submitting
    --partition PARTITION      Override SLURM partition
    --gpus COUNT               Override GPU count
    --time HH:MM:SS            Override time limit
    --memory MEM               Override memory (e.g. 32G)
    --cpus COUNT               Override CPUs per task
    --job-name NAME            Override job name
    --nodelist HOSTS           Override nodelist
    --nodes COUNT              Override node count
    --ntasks-per-node COUNT    Override tasks per node
    -h, --help                 Show this help message

Examples:
    $0
    $0 --integration
    $0 --config config/tests/tokenization_smoke.yaml
    $0 --dry-run --partition postiguet1
EOF_USAGE
}

if [[ ! -f "$TEST_ENV_FILE" ]]; then
    echo "ERROR: Missing test env file: $TEST_ENV_FILE"
    exit 1
fi

# shellcheck source=/dev/null
source "$TEST_ENV_FILE"

if [[ -f "$SECRETS_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$SECRETS_FILE"
fi

if [[ -z "${SLURM_CONF:-}" && -f /etc/slurm/slurm.conf ]]; then
    export SLURM_CONF=/etc/slurm/slurm.conf
fi

if [[ ! -x "$SUBMIT_SCRIPT" ]]; then
    echo "ERROR: submit_job.sh not found or not executable: $SUBMIT_SCRIPT"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/slurm/tests/logs"

MODE="unit"
CUSTOM_CONFIG=""
DRY_RUN=false
PARTITION_OVERRIDE=""
GPU_OVERRIDE=""
TIME_OVERRIDE=""
MEMORY_OVERRIDE=""
CPUS_OVERRIDE=""
JOB_NAME_OVERRIDE=""
NODELIST_OVERRIDE=""
NODES_OVERRIDE=""
NTASKS_PER_NODE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit)
            MODE="unit"
            shift
            ;;
        --integration)
            MODE="integration"
            shift
            ;;
        --config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --partition)
            PARTITION_OVERRIDE="$2"
            shift 2
            ;;
        --gpus)
            GPU_OVERRIDE="$2"
            shift 2
            ;;
        --time)
            TIME_OVERRIDE="$2"
            shift 2
            ;;
        --memory)
            MEMORY_OVERRIDE="$2"
            shift 2
            ;;
        --cpus)
            CPUS_OVERRIDE="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME_OVERRIDE="$2"
            shift 2
            ;;
        --nodelist)
            NODELIST_OVERRIDE="$2"
            shift 2
            ;;
        --nodes)
            NODES_OVERRIDE="$2"
            shift 2
            ;;
        --ntasks-per-node)
            NTASKS_PER_NODE_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -n "$CUSTOM_CONFIG" ]]; then
    CONFIG_PATH="$CUSTOM_CONFIG"
else
    if [[ "$MODE" == "integration" ]]; then
        CONFIG_PATH="${TEST_INTEGRATION_CONFIG:-}"
    else
        CONFIG_PATH="${TEST_UNIT_CONFIG:-}"
    fi
fi

if [[ -z "$CONFIG_PATH" ]]; then
    echo "ERROR: No test config selected. Use --config or set TEST_UNIT_CONFIG/TEST_INTEGRATION_CONFIG."
    exit 1
fi

ALLOWED_SUBMITTERS="${ALLOWED_SUBMITTERS:-}"
if [[ -z "$ALLOWED_SUBMITTERS" ]]; then
    echo "ERROR: ALLOWED_SUBMITTERS must be set in slurm/tests/slurm_test.env"
    exit 1
fi

CURRENT_USER="$(whoami)"
ALLOWED=false
IFS=',' read -r -a allowed_list <<< "$ALLOWED_SUBMITTERS"
for entry in "${allowed_list[@]}"; do
    entry="${entry//[[:space:]]/}"
    if [[ -n "$entry" && "$entry" == "$CURRENT_USER" ]]; then
        ALLOWED=true
        break
    fi
done

if [[ "$ALLOWED" != "true" && "$DRY_RUN" != "true" ]]; then
    echo "ERROR: User '$CURRENT_USER' is not allowed to submit SLURM tests."
    echo "Allowed submitters: $ALLOWED_SUBMITTERS"
    exit 1
fi

PARTITION_VALUE="${PARTITION_OVERRIDE:-${PARTITION:-}}"
GPU_VALUE="${GPU_OVERRIDE:-${GPU_COUNT:-}}"
TIME_VALUE="${TIME_OVERRIDE:-${TIME_LIMIT:-}}"
MEMORY_VALUE="${MEMORY_OVERRIDE:-${MEMORY:-}}"
CPUS_VALUE="${CPUS_OVERRIDE:-${CPUS_PER_TASK:-}}"
JOB_NAME_VALUE="${JOB_NAME_OVERRIDE:-${JOB_NAME:-}}"
NODES_VALUE="${NODES_OVERRIDE:-${NODES:-}}"
NTASKS_PER_NODE_VALUE="${NTASKS_PER_NODE_OVERRIDE:-${NTASKS_PER_NODE:-}}"

CMD=("$SUBMIT_SCRIPT" --config "$CONFIG_PATH")

if [[ "$DRY_RUN" == "true" ]]; then
    CMD+=(--dry-run)
fi

if [[ -n "$PARTITION_VALUE" ]]; then
    CMD+=(--partition "$PARTITION_VALUE")
fi
if [[ -n "$GPU_VALUE" ]]; then
    CMD+=(--gpus "$GPU_VALUE")
fi
if [[ -n "$TIME_VALUE" ]]; then
    CMD+=(--time "$TIME_VALUE")
fi
if [[ -n "$MEMORY_VALUE" ]]; then
    CMD+=(--memory "$MEMORY_VALUE")
fi
if [[ -n "$CPUS_VALUE" ]]; then
    CMD+=(--cpus "$CPUS_VALUE")
fi
if [[ -n "$JOB_NAME_VALUE" ]]; then
    CMD+=(--job-name "$JOB_NAME_VALUE")
fi
if [[ -n "$NODELIST_OVERRIDE" ]]; then
    CMD+=(--nodelist "$NODELIST_OVERRIDE")
fi
if [[ -n "$NODES_VALUE" ]]; then
    CMD+=(--nodes "$NODES_VALUE")
fi
if [[ -n "$NTASKS_PER_NODE_VALUE" ]]; then
    CMD+=(--ntasks-per-node "$NTASKS_PER_NODE_VALUE")
fi

echo "Selected config: $CONFIG_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
    "${CMD[@]}"
    exit 0
fi

if ! submit_output="$(${CMD[@]} 2>&1)"; then
    echo "$submit_output"
    exit 1
fi

echo "$submit_output"

job_id=$(echo "$submit_output" | awk '/Submitted batch job/ {print $4; exit}')
if [[ -z "$job_id" ]]; then
    echo "ERROR: Failed to parse job ID from submit_job.sh output"
    exit 1
fi

output_pattern=$(echo "$submit_output" | sed -n 's/.*--output=\\([^ ]*\\).*/\\1/p' | head -n1)
error_pattern=$(echo "$submit_output" | sed -n 's/.*--error=\\([^ ]*\\).*/\\1/p' | head -n1)

OUTPUT_FILE_PATTERN="${output_pattern:-${OUTPUT_FILE_PATTERN:-%j_lmtk.out}}"
ERROR_FILE_PATTERN="${error_pattern:-${ERROR_FILE_PATTERN:-%j_lmtk.err}}"

resolve_log_path() {
    local pattern="$1"
    local resolved
    resolved="${pattern//%j/$job_id}"
    resolved="${resolved//%A/$job_id}"
    if [[ "$resolved" != /* ]]; then
        resolved="$PROJECT_ROOT/$resolved"
    fi
    echo "$resolved"
}

out_log=$(resolve_log_path "$OUTPUT_FILE_PATTERN")
err_log=$(resolve_log_path "$ERROR_FILE_PATTERN")

echo "Job ID: $job_id"
echo "Stdout log: $out_log"
echo "Stderr log: $err_log"

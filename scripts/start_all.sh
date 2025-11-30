#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

VENV_DIR=${VENV_DIR:-"${REPO_ROOT}/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" && -d "${VENV_DIR}" && -x "${VENV_DIR}/bin/activate" ]]; then
  echo "Detected virtual environment at ${VENV_DIR}. Activating automatically..."
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" && -d "${VENV_DIR}" && -x "${VENV_DIR}/bin" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  echo "Detected virtual environment bin directory at ${VENV_DIR}/bin. Added to PATH."
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH. Please install Python or configure VENV_DIR."
  exit 1
fi

: "${LOG_DIR:=${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"
rm -f "${PROCESS_REGISTRY}"
touch "${PROCESS_REGISTRY}"

# 避免 start_all 与应用内 logger 重复写日志。用户可手动覆盖该变量。
if [[ -z "${DISABLE_SCRIPT_LOG_FILES:-}" ]]; then
  export DISABLE_SCRIPT_LOG_FILES=1
else
  export DISABLE_SCRIPT_LOG_FILES
fi

declare -a SERVICES=(
  "run_server.sh"
  "run_agent.sh"
  "run_training_dashboard.sh"
  "run_tensorboard.sh"
)

for service in "${SERVICES[@]}"; do
    if [[ ! -x "${SCRIPT_DIR}/${service}" ]]; then
        echo "Skipping ${service}: script not found or not executable"
        continue
    fi
    echo "Starting ${service}..."
    "${SCRIPT_DIR}/${service}"
done

echo "All services launched. PID registry: ${PROCESS_REGISTRY}"

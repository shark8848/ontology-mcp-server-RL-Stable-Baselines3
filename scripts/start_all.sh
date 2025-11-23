#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${LOG_DIR:=${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"
rm -f "${PROCESS_REGISTRY}"
touch "${PROCESS_REGISTRY}"

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

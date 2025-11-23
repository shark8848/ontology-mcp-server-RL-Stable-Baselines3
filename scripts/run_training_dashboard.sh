#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${LOG_DIR:=${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"
touch "${PROCESS_REGISTRY}"
LOG_FILE="${LOG_DIR}/training_dashboard_$(date '+%Y%m%d_%H%M%S').log"

: "${TRAINING_DASHBOARD_HOST:=0.0.0.0}"
: "${TRAINING_DASHBOARD_PORT:=7861}"
export GRADIO_SERVER_NAME="${TRAINING_DASHBOARD_HOST}"
export GRADIO_SERVER_PORT="${TRAINING_DASHBOARD_PORT}"

nohup python3 scripts/run_training_dashboard.py "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
printf "training_dashboard %s\n" "${PID}" >> "${PROCESS_REGISTRY}"
echo "Training dashboard started on ${TRAINING_DASHBOARD_HOST}:${TRAINING_DASHBOARD_PORT} (PID ${PID}). Logs: ${LOG_FILE}"

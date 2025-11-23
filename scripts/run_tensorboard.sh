#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${LOG_DIR:=${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"
touch "${PROCESS_REGISTRY}"
LOG_FILE="${LOG_DIR}/tensorboard_$(date '+%Y%m%d_%H%M%S').log"

: "${TB_LOG_DIR:=${REPO_ROOT}/data/rl_training/logs/tensorboard}"
: "${TB_HOST:=0.0.0.0}"
: "${TB_PORT:=6006}"
mkdir -p "${TB_LOG_DIR}"

nohup tensorboard --logdir "${TB_LOG_DIR}" --host "${TB_HOST}" --port "${TB_PORT}" "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
printf "tensorboard %s\n" "${PID}" >> "${PROCESS_REGISTRY}"
echo "TensorBoard started (PID ${PID}) on ${TB_HOST}:${TB_PORT}. Logs: ${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${LOG_DIR:=${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"
touch "${PROCESS_REGISTRY}"
LOG_FILE="${LOG_DIR}/agent_$(date '+%Y%m%d_%H%M%S').log"

: "${MCP_BASE_URL:=http://localhost:8000}"
export MCP_BASE_URL

nohup python3 -m agent.gradio_ui "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
printf "agent_ui %s\n" "${PID}" >> "${PROCESS_REGISTRY}"
echo "Agent UI started (PID ${PID}). Logs: ${LOG_FILE}"

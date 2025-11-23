#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${LOG_DIR:=${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"
touch "${PROCESS_REGISTRY}"
LOG_FILE="${LOG_DIR}/server_$(date '+%Y%m%d_%H%M%S').log"

: "${ONTOLOGY_DATA_DIR:=${REPO_ROOT}/data}"
export ONTOLOGY_DATA_DIR

APP_HOST=${APP_HOST:-0.0.0.0}
APP_PORT=${APP_PORT:-8000}

nohup uvicorn ontology_mcp_server.server:app --host "${APP_HOST}" --port "${APP_PORT}" "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
printf "ontology_server %s\n" "${PID}" >> "${PROCESS_REGISTRY}"
echo "Ontology MCP server started (PID ${PID}). Logs: ${LOG_FILE}"

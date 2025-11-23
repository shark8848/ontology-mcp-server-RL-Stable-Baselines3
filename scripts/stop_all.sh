#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${LOG_DIR:=${REPO_ROOT}/logs}"
PROCESS_REGISTRY="${LOG_DIR}/processes.pid"

if [[ ! -f "${PROCESS_REGISTRY}" ]]; then
    echo "No process registry found at ${PROCESS_REGISTRY}. Nothing to stop."
    exit 0
fi

if [[ ! -s "${PROCESS_REGISTRY}" ]]; then
    echo "Process registry is empty. Nothing to stop."
    rm -f "${PROCESS_REGISTRY}"
    exit 0
fi

while read -r name pid; do
    [[ -z "${name:-}" || -z "${pid:-}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
        echo "Stopping ${name} (PID ${pid})..."
        kill "${pid}" 2>/dev/null || true
    else
        echo "${name} (PID ${pid}) is not running."
    fi
    # Give process time to exit gracefully
    if kill -0 "${pid}" 2>/dev/null; then
        sleep 1
        kill -9 "${pid}" 2>/dev/null || true
    fi
    echo "${name} stopped."
done < "${PROCESS_REGISTRY}"

rm -f "${PROCESS_REGISTRY}"
echo "All recorded services stopped."

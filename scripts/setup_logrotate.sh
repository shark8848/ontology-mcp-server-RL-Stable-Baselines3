#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_DATA_DIR="${HOST_DATA_DIR:-$PWD/data}"
LOG_DIR="${LOG_DIR:-$HOST_DATA_DIR/logs}"
STATE_FILE="${STATE_FILE:-$HOST_DATA_DIR/logrotate-state}"
TARGET_FILE="${TARGET_FILE:-/etc/logrotate.d/ontology-mcp-server}"

if ! command -v logrotate >/dev/null 2>&1; then
  echo "❌ 未找到 logrotate，请先安装（Ubuntu: sudo apt-get install -y logrotate）"
  exit 1
fi

mkdir -p "$LOG_DIR"

TMP_CONF="$(mktemp)"
cat >"$TMP_CONF" <<EOF
$LOG_DIR/*.log $LOG_DIR/**/*.log {
    daily
    rotate 14
    missingok
    notifempty
    compress
    delaycompress
    copytruncate
    dateext
    dateformat -%Y%m%d
}
EOF

if [[ "${1:-}" == "--install" ]]; then
  echo "📦 安装 logrotate 配置到 $TARGET_FILE"
  sudo cp "$TMP_CONF" "$TARGET_FILE"
  sudo chmod 644 "$TARGET_FILE"
  echo "✅ 安装完成"
  echo "🔎 可验证: sudo logrotate -d $TARGET_FILE"
else
  echo "ℹ️ 预览配置（未安装）:"
  cat "$TMP_CONF"
  echo
  echo "如需安装到系统: ./scripts/setup_logrotate.sh --install"
  echo "如需手动测试轮转: logrotate -s $STATE_FILE -f $TMP_CONF"
fi

rm -f "$TMP_CONF"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

# 可选读取 .env（便于管理 API KEY 等敏感变量）
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

CONTAINER_NAME="${CONTAINER_NAME:-ontology-mcp-server}"
IMAGE_NAME="${IMAGE_NAME:-ontology-mcp-server:local}"
NETWORK_MODE="${NETWORK_MODE:-bridge}" # host | bridge
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/agent/config.yaml}"
HOST_DATA_DIR="${HOST_DATA_DIR:-$PWD/data}"
HOST_SRC_DIR="${HOST_SRC_DIR:-$ROOT_DIR/src}"
HOST_SCRIPTS_DIR="${HOST_SCRIPTS_DIR:-$ROOT_DIR/scripts}"
MOUNT_SOURCE_CODE="${MOUNT_SOURCE_CODE:-1}"
AUTO_START_ALL="${AUTO_START_ALL:-1}"
STARTUP_HEALTHCHECK="${STARTUP_HEALTHCHECK:-1}"

ONTOLOGY_DATA_DIR="${ONTOLOGY_DATA_DIR:-/app/data}"
LOG_DIR="${LOG_DIR:-$ONTOLOGY_DATA_DIR/logs}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-$LOG_DIR/agent}"
ONTOLOGY_SERVER_LOG_DIR="${ONTOLOGY_SERVER_LOG_DIR:-$LOG_DIR/server}"
MCP_BASE_URL="${MCP_BASE_URL:-http://localhost:8000}"
OPENAI_API_URL="${OPENAI_API_URL:-https://aicp.teamshub.com/ai-paas/ai-open/sitech/aiopen/stream/Qwen3-235B-A22B-Public/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-Qwen3-235B}"
LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ docker 未安装或不在 PATH 中"
  exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "❌ 本地镜像不存在: $IMAGE_NAME"
  echo "请先构建或加载镜像后再启动"
  exit 1
fi

if [ "$LLM_PROVIDER" != "ollama" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ 请先设置 OPENAI_API_KEY 环境变量"
  echo "示例: OPENAI_API_KEY='your_key' ./scripts/start_docker.sh"
  exit 1
fi

# 建议在宿主机先准备 Ollama embedding 模型
#   ollama serve
#   ollama pull bge-m3:latest

# 若已存在同名容器，先删除
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "🧹 删除已存在容器: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker_args=(
  -d
  --name "$CONTAINER_NAME"
  --network "$NETWORK_MODE"
  -v "$HOST_DATA_DIR:$ONTOLOGY_DATA_DIR"
  -e "ONTOLOGY_DATA_DIR=$ONTOLOGY_DATA_DIR"
  -e "LOG_DIR=$LOG_DIR"
  -e "AGENT_LOG_DIR=$AGENT_LOG_DIR"
  -e "ONTOLOGY_SERVER_LOG_DIR=$ONTOLOGY_SERVER_LOG_DIR"
  -e "MCP_BASE_URL=$MCP_BASE_URL"
  -e "OPENAI_API_URL=$OPENAI_API_URL"
  -e "OPENAI_API_KEY=${OPENAI_API_KEY:-}"
  -e "OPENAI_MODEL=$OPENAI_MODEL"
  -e "LLM_PROVIDER=$LLM_PROVIDER"
  -e "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
  -e "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
  -e "GRADIO_ANALYTICS_ENABLED=$GRADIO_ANALYTICS_ENABLED"
)

# 可选挂载配置文件，不改镜像即可更新配置
mkdir -p "$HOST_DATA_DIR"
mkdir -p "$HOST_DATA_DIR/logs"

if [ -f "$CONFIG_PATH" ]; then
  docker_args+=( -v "$CONFIG_PATH:/app/src/agent/config.yaml:ro" )
else
  echo "⚠️ 未找到配置文件，跳过挂载: $CONFIG_PATH"
fi

# 可选挂载宿主机源码和脚本，避免每次修改后重建镜像
if [ "$MOUNT_SOURCE_CODE" = "1" ]; then
  if [ -d "$HOST_SRC_DIR" ]; then
    docker_args+=( -v "$HOST_SRC_DIR:/app/src" )
  else
    echo "⚠️ 未找到源码目录，跳过挂载: $HOST_SRC_DIR"
  fi
  if [ -d "$HOST_SCRIPTS_DIR" ]; then
    docker_args+=( -v "$HOST_SCRIPTS_DIR:/app/scripts" )
  else
    echo "⚠️ 未找到脚本目录，跳过挂载: $HOST_SCRIPTS_DIR"
  fi
fi

# bridge 网络下端口映射生效；host 网络下无需 -p
if [ "$NETWORK_MODE" = "bridge" ]; then
  docker_args+=(
    -p 8001:8000
    -p 7860:7860
    -p 7861:7861
    -p 6006:6006
  )
fi

echo "🚀 启动容器: $CONTAINER_NAME"
docker run "${docker_args[@]}" --entrypoint tail "$IMAGE_NAME" -f /dev/null

if [ "$AUTO_START_ALL" = "1" ]; then
  echo "▶️ 自动启动容器内服务: ./scripts/start_all.sh"
  docker exec -d "$CONTAINER_NAME" bash -lc "cd /app && ./scripts/start_all.sh"
fi

if [ "$STARTUP_HEALTHCHECK" = "1" ]; then
  echo "🔎 启动后探活: /health 与 /capabilities"
  ok=0
  for _ in $(seq 1 20); do
    if docker exec "$CONTAINER_NAME" sh -lc "curl -fsS http://localhost:8000/health >/dev/null 2>&1" \
      && docker exec "$CONTAINER_NAME" sh -lc "curl -fsS http://localhost:8000/capabilities >/dev/null 2>&1"; then
      ok=1
      break
    fi
    sleep 1
  done

  if [ "$ok" -ne 1 ]; then
    echo "⚠️ 探活失败：MCP 服务可能未正确启动或端口被占用"
    echo "建议排查: docker exec -it $CONTAINER_NAME bash -lc 'ps -ef | grep -E \"uvicorn|python\" | grep -v grep'"
    echo "建议排查: docker exec -it $CONTAINER_NAME bash -lc 'curl -i http://localhost:8000/health && echo && curl -i http://localhost:8000/capabilities'"
  fi
fi

echo "✅ 容器已启动"
echo "ℹ️ 数据目录挂载: $HOST_DATA_DIR -> $ONTOLOGY_DATA_DIR"
echo "ℹ️ 日志目录默认输出: $HOST_DATA_DIR/logs"
if [ "$NETWORK_MODE" = "host" ]; then
  echo "ℹ️ 当前使用 --network host（端口直通宿主机）"
else
  echo "ℹ️ 当前使用 --network bridge（端口映射: 8001/7860/7861/6006）"
fi
echo "📌 进入容器: docker exec -it $CONTAINER_NAME bash"
echo "📌 查看日志: docker logs -f $CONTAINER_NAME"

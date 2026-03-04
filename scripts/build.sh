#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  scripts/build.sh [options]
  scripts/build.sh help

Options:
  --single-image       构建单镜像（默认模式，等同旧参数 --direct）
  --split-services     按服务分别构建镜像
  --service <name>     指定 compose 服务（可重复）
  --image-tag <tag>    direct 模式镜像 tag，默认: local
  --image-name <name>  direct 模式镜像名，默认: ontology-mcp-server
  --output-path <path> 单镜像输出文件路径（默认: docker/images/<name>-<tag>.tar）
  --load-local         单镜像直接加载到本地仓库（默认关闭）
  --os-tools <list>    额外系统工具列表（空格分隔，透传 COMMON_OS_TOOLS）
  --pip-timeout <sec>  pip 下载超时秒数（默认: 600）
  --pip-retries <n>    pip 下载重试次数（默认: 8）
  --preload-st-models <csv>
                      预下载 sentence-transformers 模型（逗号分隔）
                      默认: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2
  --no-cache           构建时不使用缓存
  --pull               构建前拉取最新基础镜像
  --direct             兼容参数：等同 --single-image
  -h, --help           显示帮助
  help                 同 --help

Behavior:
  - 单镜像模式(默认): 使用 docker buildx build 输出镜像 tar 到存储路径（不加载本地仓库）
  - 服务镜像模式: 使用 docker compose build（自动兼容 docker-compose）

Exit Codes:
  0  成功
  1  参数错误、命令缺失或构建失败

Examples:
  scripts/build.sh
  scripts/build.sh --single-image
  scripts/build.sh --service mcp-server
  scripts/build.sh --service mcp-server --service agent-ui --no-cache
  scripts/build.sh --split-services --no-cache --pull
  scripts/build.sh --single-image --image-name ontology-mcp-server --image-tag dev
  scripts/build.sh --single-image --output-path docker/images/mcp-v1.tar
  scripts/build.sh --single-image --load-local
  scripts/build.sh --single-image --os-tools "curl wget jq vim-tiny"
  scripts/build.sh --single-image --pip-timeout 1200 --pip-retries 12
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Command not found: $1"
    exit 1
  fi
}

usage_hint() {
  echo "Use './scripts/build.sh --help' to see available options."
}

ensure_parent_dir() {
  local target_path="$1"
  mkdir -p "$(dirname "${target_path}")"
}

resolve_default_output_path() {
  local safe_name safe_tag
  safe_name=${IMAGE_NAME//\//_}
  safe_tag=${IMAGE_TAG//\//_}
  echo "${REPO_ROOT}/docker/images/${safe_name}-${safe_tag}.tar"
}

COMPOSE_CMD=()
COMPOSE_LABEL=""

resolve_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    COMPOSE_LABEL="docker compose"
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    COMPOSE_LABEL="docker-compose"
    return 0
  fi
  return 1
}

NO_CACHE=0
PULL=0
SINGLE_IMAGE_MODE=1
IMAGE_NAME=${IMAGE_NAME:-ontology-mcp-server}
IMAGE_TAG=${IMAGE_TAG:-local}
OUTPUT_PATH=""
LOAD_LOCAL=0
PIP_TIMEOUT=${PIP_TIMEOUT:-600}
PIP_RETRIES=${PIP_RETRIES:-8}
OS_TOOLS=${OS_TOOLS:-"curl wget less vim-tiny jq unzip iputils-ping netcat-openbsd procps net-tools"}
PRELOAD_ST_MODELS=${PRELOAD_ST_MODELS:-"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2"}
SERVICES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single-image)
      SINGLE_IMAGE_MODE=1
      shift
      ;;
    --split-services)
      SINGLE_IMAGE_MODE=0
      shift
      ;;
    --service)
      [[ $# -lt 2 ]] && { echo "Missing value for --service"; usage_hint; exit 1; }
      SERVICES+=("$2")
      shift 2
      ;;
    --image-tag)
      [[ $# -lt 2 ]] && { echo "Missing value for --image-tag"; usage_hint; exit 1; }
      IMAGE_TAG="$2"
      shift 2
      ;;
    --image-name)
      [[ $# -lt 2 ]] && { echo "Missing value for --image-name"; usage_hint; exit 1; }
      IMAGE_NAME="$2"
      shift 2
      ;;
    --output-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --output-path"; usage_hint; exit 1; }
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --load-local)
      LOAD_LOCAL=1
      shift
      ;;
    --os-tools)
      [[ $# -lt 2 ]] && { echo "Missing value for --os-tools"; usage_hint; exit 1; }
      OS_TOOLS="$2"
      shift 2
      ;;
    --pip-timeout)
      [[ $# -lt 2 ]] && { echo "Missing value for --pip-timeout"; usage_hint; exit 1; }
      PIP_TIMEOUT="$2"
      shift 2
      ;;
    --pip-retries)
      [[ $# -lt 2 ]] && { echo "Missing value for --pip-retries"; usage_hint; exit 1; }
      PIP_RETRIES="$2"
      shift 2
      ;;
    --preload-st-models)
      [[ $# -lt 2 ]] && { echo "Missing value for --preload-st-models"; usage_hint; exit 1; }
      PRELOAD_ST_MODELS="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --direct)
      SINGLE_IMAGE_MODE=1
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage_hint
      exit 1
      ;;
  esac
done

require_cmd docker

if [[ -z "${PRELOAD_ST_MODELS// }" ]]; then
  echo "PRELOAD_ST_MODELS is empty; refusing build because embedding models must be preloaded."
  echo "Use --preload-st-models \"modelA,modelB\" or set PRELOAD_ST_MODELS env."
  exit 1
fi

echo "[build] Preloading ST models: ${PRELOAD_ST_MODELS}"

if [[ "${SINGLE_IMAGE_MODE}" == "1" ]]; then
  if [[ "${LOAD_LOCAL}" == "1" ]]; then
    BUILD_ARGS=(build -f Dockerfile -t "${IMAGE_NAME}:${IMAGE_TAG}")
    BUILD_ARGS+=(--build-arg "PIP_DEFAULT_TIMEOUT=${PIP_TIMEOUT}")
    BUILD_ARGS+=(--build-arg "PIP_RETRIES=${PIP_RETRIES}")
    BUILD_ARGS+=(--build-arg "COMMON_OS_TOOLS=${OS_TOOLS}")
    BUILD_ARGS+=(--build-arg "PRELOAD_ST_MODELS=${PRELOAD_ST_MODELS}")
    [[ "${NO_CACHE}" == "1" ]] && BUILD_ARGS+=(--no-cache)
    [[ "${PULL}" == "1" ]] && BUILD_ARGS+=(--pull)
    BUILD_ARGS+=(.)

    echo "[build] Running: docker ${BUILD_ARGS[*]}"
    docker "${BUILD_ARGS[@]}"
    echo "[build] Done (loaded locally): ${IMAGE_NAME}:${IMAGE_TAG}"
    exit 0
  fi

  if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required for file output mode (without local load)."
    echo "Either install buildx, or use --load-local."
    exit 1
  fi

  if [[ -z "${OUTPUT_PATH}" ]]; then
    OUTPUT_PATH=$(resolve_default_output_path)
  elif [[ "${OUTPUT_PATH}" != /* ]]; then
    OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_PATH}"
  fi
  ensure_parent_dir "${OUTPUT_PATH}"

  BUILDX_ARGS=(build -f Dockerfile -t "${IMAGE_NAME}:${IMAGE_TAG}")
  BUILDX_ARGS+=(--build-arg "PIP_DEFAULT_TIMEOUT=${PIP_TIMEOUT}")
  BUILDX_ARGS+=(--build-arg "PIP_RETRIES=${PIP_RETRIES}")
  BUILDX_ARGS+=(--build-arg "COMMON_OS_TOOLS=${OS_TOOLS}")
  BUILDX_ARGS+=(--build-arg "PRELOAD_ST_MODELS=${PRELOAD_ST_MODELS}")
  BUILDX_ARGS+=(--output "type=docker,dest=${OUTPUT_PATH}")
  [[ "${NO_CACHE}" == "1" ]] && BUILDX_ARGS+=(--no-cache)
  [[ "${PULL}" == "1" ]] && BUILDX_ARGS+=(--pull)
  BUILDX_ARGS+=(.)

  echo "[build] Running: docker buildx ${BUILDX_ARGS[*]}"
  docker buildx "${BUILDX_ARGS[@]}"
  echo "[build] Done (saved file): ${OUTPUT_PATH}"
  exit 0
fi

if ! resolve_compose_cmd; then
  echo "Neither 'docker compose' nor 'docker-compose' is available."
  exit 1
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
  echo "[build] split-services mode builds into local Docker repository via compose."
  echo "[build] File output mode is only supported with --single-image."
fi

BUILD_ARGS=(build)
BUILD_ARGS+=(--build-arg "PIP_DEFAULT_TIMEOUT=${PIP_TIMEOUT}")
BUILD_ARGS+=(--build-arg "PIP_RETRIES=${PIP_RETRIES}")
BUILD_ARGS+=(--build-arg "COMMON_OS_TOOLS=${OS_TOOLS}")
BUILD_ARGS+=(--build-arg "PRELOAD_ST_MODELS=${PRELOAD_ST_MODELS}")
[[ "${NO_CACHE}" == "1" ]] && BUILD_ARGS+=(--no-cache)
[[ "${PULL}" == "1" ]] && BUILD_ARGS+=(--pull)
if [[ ${#SERVICES[@]} -gt 0 ]]; then
  BUILD_ARGS+=("${SERVICES[@]}")
fi

echo "[build] Running: ${COMPOSE_LABEL} ${BUILD_ARGS[*]}"
"${COMPOSE_CMD[@]}" "${BUILD_ARGS[@]}"
echo "[build] Done"

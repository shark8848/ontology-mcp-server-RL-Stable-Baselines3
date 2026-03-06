# 多阶段构建 Ontology RL Commerce Agent
FROM python:3.12-slim AS base

ARG COMMON_OS_TOOLS="curl wget less vim-tiny jq unzip iputils-ping netcat-openbsd procps net-tools"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    ${COMMON_OS_TOOLS} \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-dev.txt pyproject.toml ./
COPY src ./src

# pip 下载稳定性参数（可被 --build-arg 覆盖）
ARG PIP_DEFAULT_TIMEOUT=600
ARG PIP_RETRIES=8

# 安装 Python 依赖
RUN python -m pip install --no-cache-dir --timeout ${PIP_DEFAULT_TIMEOUT} --retries ${PIP_RETRIES} --upgrade pip setuptools wheel && \
    PIP_REQUIRE_HASHES=0 pip install --no-cache-dir --timeout ${PIP_DEFAULT_TIMEOUT} --retries ${PIP_RETRIES} -r requirements-dev.txt && \
    PIP_REQUIRE_HASHES=0 pip install --no-cache-dir --timeout ${PIP_DEFAULT_TIMEOUT} --retries ${PIP_RETRIES} -e .

# 生产镜像
FROM python:3.12-slim

ARG COMMON_OS_TOOLS="curl wget less vim-tiny jq unzip iputils-ping netcat-openbsd procps net-tools"
ARG PRELOAD_ST_MODELS="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2"

WORKDIR /app

# 运行时常用系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ${COMMON_OS_TOOLS} \
    && rm -rf /var/lib/apt/lists/*

# 从基础镜像复制依赖
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# 预置 HuggingFace / Sentence-Transformers 缓存目录（可在构建阶段预下载模型）
ENV HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache/hub
RUN mkdir -p /opt/hf-cache

# 可选：构建阶段预下载 embedding 模型，避免运行时在线下载
# 用法示例：--build-arg PRELOAD_ST_MODELS="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2"
RUN if [ -n "${PRELOAD_ST_MODELS}" ]; then \
    PRELOAD_ST_MODELS="${PRELOAD_ST_MODELS}" python -c "import os; from sentence_transformers import SentenceTransformer; raw=os.environ.get('PRELOAD_ST_MODELS',''); models=[m.strip() for m in raw.split(',') if m.strip()]; [(print(f'[preload] downloading sentence-transformers model: {m}'), SentenceTransformer(m)) for m in models]; print('[preload] completed' if models else '[preload] no models configured')"; \
    fi

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data/chroma_memory \
    data/rl_training \
    data/training_dashboard \
    data/logs \
    src/agent/logs \
    src/ontology_mcp_server/logs

# 设置环境变量
ENV PYTHONPATH=/app/src \
    ONTOLOGY_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000 7860 7861

# 复制并设置 entrypoint 脚本
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

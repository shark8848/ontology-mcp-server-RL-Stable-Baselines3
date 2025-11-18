# 多阶段构建 Ontology RL Commerce Agent
FROM python:3.12-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-dev.txt pyproject.toml ./
COPY src ./src

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir -e .

# 生产镜像
FROM python:3.12-slim

WORKDIR /app

# 从基础镜像复制依赖
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

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

# 默认启动 MCP 服务器
CMD ["uvicorn", "ontology_mcp_server.server:app", "--host", "0.0.0.0", "--port", "8000"]

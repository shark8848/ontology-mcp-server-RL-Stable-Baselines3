# Docker 快速参考

## 构建与启动

```bash
# 构建并加载本地镜像（推荐）
./scripts/build.sh --single-image --load-local

# 使用宿主机启动脚本（默认走 host 网络，并挂载 src/agent/config.yaml）
./scripts/start_docker.sh

# 仅拉起容器（不启动任何服务，便于 docker exec 进入后手动执行）
docker run -d --name ontology-mcp-server \
  -p 8001:8000 -p 7860:7860 -p 7861:7861 -p 6006:6006 \
  -e ONTOLOGY_DATA_DIR=/app/data \
  -e OPENAI_API_URL='https://aicp.teamshub.com/ai-paas/ai-open/sitech/aiopen/stream/Qwen3-235B-A22B-Public/v1' \
  -e OPENAI_API_KEY='${OPENAI_API_KEY}' \
  -e OPENAI_MODEL=Qwen3-235B-A22B-Public \
  -e LLM_PROVIDER=deepseek \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint tail \
  ontology-mcp-server:local -f /dev/null

# 进入容器
docker exec -it ontology-mcp-server bash

# 查看日志
docker logs -f ontology-mcp-server

# 停止并删除容器
docker rm -f ontology-mcp-server
```

## 离线部署：构建时预置 Embedding 模型

当前 `Dockerfile` 已默认预下载本地 embedding 模型（用于记忆模块与意图识别），默认模型列表：

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

如需自定义模型列表，可在镜像构建阶段覆盖 `PRELOAD_ST_MODELS`：

```bash
docker build -t ontology-mcp-server:local \
  --build-arg PRELOAD_ST_MODELS="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2" \
  .
```

构建完成后，运行时会直接使用镜像内缓存，避免首次在线下载。

注意：
- 该预置方式适用于 `sentence-transformers` 路径。
- 若使用 `Ollama` embedding（如 `nomic-embed-text`），模型在 Ollama 服务侧管理，需要在 Ollama 主机执行：

```bash
ollama pull nomic-embed-text
```

## 服务管理

```bash
# 重启服务
docker-compose restart <service-name>

# 停止单个服务
docker-compose stop agent-ui

# 启动单个服务
docker-compose start agent-ui

# 查看服务状态
docker-compose ps
```

## 调试

```bash
# 进入容器
docker exec -it ontology-agent-ui bash

# 查看容器内日志
docker exec ontology-agent-ui ls -la /app/data/logs

# 实时查看容器输出
docker attach ontology-agent-ui
```

## 数据管理

```bash
# 备份数据目录
docker run --rm -v ontology-rl-commerce-agent_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/data-backup.tar.gz /data

# 恢复数据
docker run --rm -v ontology-rl-commerce-agent_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/data-backup.tar.gz -C /

# 清理未使用的卷
docker volume prune
```

## 日志轮转（推荐）

容器日志默认输出到宿主机 `data/logs`，可用以下脚本配置系统 `logrotate`：

```bash
# 预览将要安装的规则
./scripts/setup_logrotate.sh

# 安装到 /etc/logrotate.d/ontology-mcp-server
./scripts/setup_logrotate.sh --install
```

默认策略：按天轮转、保留 14 份、压缩历史日志、空文件跳过。

## 完全重建

```bash
# 停止并删除容器、卷
docker-compose down -v

# 清理镜像缓存
docker-compose build --no-cache

# 重新启动
docker-compose up -d
```

## GPU 支持

如需启用 GPU 训练，在 `docker-compose.yml` 中取消注释以下部分：

```yaml
training-dashboard:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

然后重启服务：

```bash
docker-compose up -d --force-recreate training-dashboard
```

## 环境变量

主要配置在 `.env` 文件中：

- `OPENAI_API_URL`: LLM API 地址
- `OPENAI_API_KEY`: API 密钥（必填）
- `OPENAI_MODEL`: 模型名称
- `LLM_PROVIDER`: deepseek | ollama
- `INIT_BULK_DATA`: 首次启动时生成批量数据
- `INIT_TRAINING_DATA`: 首次启动时生成训练语料

## 常见问题

### 端口冲突

如果默认端口被占用，修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8001:8000"  # 将 MCP Server 映射到 8001
```

### 数据库锁定

如果遇到 SQLite 锁定问题，重启服务：

```bash
docker-compose restart
```

### 内存不足

调整 Docker 内存限制（Docker Desktop -> Settings -> Resources）或在 `docker-compose.yml` 中设置：

```yaml
services:
  agent-ui:
    mem_limit: 4g
```

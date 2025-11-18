# Docker 快速参考

## 构建与启动

```bash
# 构建镜像
docker-compose build

# 启动所有服务（后台）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f agent-ui

# 停止服务
docker-compose down
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

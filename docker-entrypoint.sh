#!/bin/bash
set -e

# 初始化脚本 - 在容器首次启动时执行数据库和数据初始化

echo "🚀 启动 Ontology RL Commerce Agent..."

# 设置环境变量
export ONTOLOGY_DATA_DIR="${ONTOLOGY_DATA_DIR:-/app/data}"
export PYTHONPATH="${PYTHONPATH:-/app/src}"

is_truthy() {
    local v="${1:-}"
    case "$(echo "$v" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# 强制本地模式：禁止 HuggingFace/Transformers 在线拉取
if [ "${FORCE_LOCAL_ONLY:-0}" = "1" ] || [ "${FORCE_LOCAL_ONLY:-false}" = "true" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "🔒 FORCE_LOCAL_ONLY=1: 已启用离线模式（禁止在线模型下载）"

    PRELOAD_ST_MODELS_RUNTIME="${PRELOAD_ST_MODELS:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2}"
    echo "🔎 离线启动自检：校验本地 embedding 模型缓存..."
    PRELOAD_ST_MODELS="${PRELOAD_ST_MODELS_RUNTIME}" python - <<'PY'
import os
import sys

from sentence_transformers import SentenceTransformer

raw = os.getenv("PRELOAD_ST_MODELS", "")
models = [m.strip() for m in raw.split(",") if m.strip()]
if not models:
    print("[offline-check] PRELOAD_ST_MODELS 为空，跳过本地模型校验")
    raise SystemExit(0)

failed = []
for model in models:
    try:
        try:
            SentenceTransformer(model, local_files_only=True)
        except TypeError:
            SentenceTransformer(model)
        print(f"[offline-check] ok: {model}")
    except Exception as exc:
        failed.append((model, str(exc)))

if failed:
    print("[offline-check] failed: 以下模型未命中本地缓存，离线模式禁止启动:")
    for name, err in failed:
        print(f"  - {name}: {err}")
    raise SystemExit(1)
PY
fi

if is_truthy "${SKIP_INIT:-0}"; then
    echo "⏭️ SKIP_INIT=1: 跳过数据库初始化与默认配置复制"
else
    # 检查数据库是否已初始化
    if [ ! -f "$ONTOLOGY_DATA_DIR/ecommerce.db" ]; then
        echo "📦 初始化数据库..."
        python scripts/init_database.py
        
        echo "🌱 填充测试数据..."
        python scripts/seed_data.py
        
        # 可选：批量生成商品和用户
        if [ "${INIT_BULK_DATA:-false}" = "true" ]; then
            echo "📊 生成批量数据..."
            python scripts/add_bulk_products.py
            python scripts/add_bulk_users.py
            python scripts/update_demo_user_names.py --seed 2025
        fi
        
        # 可选：生成训练语料
        if [ "${INIT_TRAINING_DATA:-false}" = "true" ]; then
            echo "🧠 生成训练语料..."
            python scripts/generate_dialogue_corpus.py
        fi
    else
        echo "✅ 数据库已存在，跳过初始化"
    fi

    # 检查配置文件
    if [ ! -f "src/agent/config.yaml" ]; then
        echo "⚙️ 复制默认配置..."
        if [ -f "src/agent/config.example.yaml" ]; then
            cp src/agent/config.example.yaml src/agent/config.yaml
        fi
    fi

    if [ ! -f "config/training_dashboard.yaml" ]; then
        echo "⚙️ 复制训练控制台配置..."
        if [ -f "config/training_dashboard.example.yaml" ]; then
            cp config/training_dashboard.example.yaml config/training_dashboard.yaml
        fi
    fi
fi

echo "✨ 初始化完成！"
echo ""

# 未传入命令时，保持容器驻留，便于 docker exec 进入后手动执行
if [ "$#" -eq 0 ]; then
    echo "ℹ️ 未提供启动命令，容器将保持驻留（tail -f /dev/null）以便手动进入执行。"
    exec tail -f /dev/null
fi

# 执行传入的命令
exec "$@"

#!/bin/bash
set -e

# 初始化脚本 - 在容器首次启动时执行数据库和数据初始化

echo "🚀 启动 Ontology RL Commerce Agent..."

# 设置环境变量
export ONTOLOGY_DATA_DIR="${ONTOLOGY_DATA_DIR:-/app/data}"
export PYTHONPATH="${PYTHONPATH:-/app/src}"

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

echo "✨ 初始化完成！"
echo ""

# 执行传入的命令
exec "$@"

#!/bin/bash
# 清理ChromaDB记忆，移除误导性历史记录

echo "======================================================================"
echo "清理ChromaDB记忆"
echo "======================================================================"
echo ""

CHROMA_PATH="/home/ontology-mcp-server-RL-Stable-Baselines3/data/chroma_memory"

if [ ! -d "$CHROMA_PATH" ]; then
    echo "✅ ChromaDB目录不存在，无需清理"
    exit 0
fi

echo "📁 ChromaDB路径: $CHROMA_PATH"
echo ""

# 显示当前大小
CURRENT_SIZE=$(du -sh "$CHROMA_PATH" | cut -f1)
echo "当前大小: $CURRENT_SIZE"
echo ""

# 备份
BACKUP_PATH="/home/ontology-mcp-server-RL-Stable-Baselines3/data/chroma_memory_backup_$(date +%Y%m%d_%H%M%S)"
echo "🔄 创建备份: $BACKUP_PATH"
cp -r "$CHROMA_PATH" "$BACKUP_PATH"

if [ $? -eq 0 ]; then
    echo "✅ 备份成功"
else
    echo "❌ 备份失败，停止操作"
    exit 1
fi

echo ""
echo "🗑️  清空ChromaDB记忆..."

# 清空内容
rm -rf "$CHROMA_PATH"/*

if [ $? -eq 0 ]; then
    echo "✅ 清空成功"
    echo ""
    echo "======================================================================"
    echo "操作完成"
    echo "======================================================================"
    echo ""
    echo "✅ ChromaDB记忆已清空"
    echo "✅ 备份保存在: $BACKUP_PATH"
    echo ""
    echo "下一步:"
    echo "  1. 重启Agent: pkill -f 'agent.gradio_ui' && sleep 2 && cd src && nohup python -m agent.gradio_ui > agent.log 2>&1 &"
    echo "  2. 测试图表功能: '显示销量前10的商品柱状图'"
    echo ""
else
    echo "❌ 清空失败"
    exit 1
fi

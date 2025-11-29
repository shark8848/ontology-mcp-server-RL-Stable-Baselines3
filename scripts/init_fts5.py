#!/usr/bin/env python3
"""
Copyright (c) 2025 shark8848
MIT License

初始化 FTS5 全文检索表并同步数据
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ontology_mcp_server.db_service import DatabaseService

def main():
    """初始化 FTS5 表并同步商品数据"""
    print("=" * 60)
    print("初始化 FTS5 全文检索表")
    print("=" * 60)
    
    # 初始化数据库服务
    db = DatabaseService("data/ecommerce.db")
    
    # 创建 FTS5 表
    print("\n[1/2] 创建 FTS5 虚拟表...")
    db.create_fts_table()
    
    # 同步商品数据
    print("\n[2/2] 同步商品数据到 FTS5 表...")
    db.sync_products_to_fts()
    
    print("\n" + "=" * 60)
    print("✅ FTS5 全文检索初始化完成!")
    print("=" * 60)
    print("\n提示:")
    print("- FTS5 表名: products_fts")
    print("- 索引字段: product_name, category, brand, model, description")
    print("- 分词器: unicode61 (支持中文)")
    print("- 自动启用: search_products(..., use_fts=True)")

if __name__ == "__main__":
    main()

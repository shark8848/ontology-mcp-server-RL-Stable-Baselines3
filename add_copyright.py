#!/usr/bin/env python3
"""
添加版权信息到所有 Python 源文件
"""
import os
from pathlib import Path

COPYRIGHT_HEADER = '''"""
Copyright (c) 2025 shark8848
MIT License

Ontology MCP Server - 电商 AI 助手系统
本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI

Author: shark8848
Repository: https://github.com/shark8848/ontology-mcp-server
"""
'''

def has_copyright(content: str) -> bool:
    """检查文件是否已有版权信息"""
    return 'Copyright' in content or 'shark8848' in content

def add_copyright_to_file(file_path: Path):
    """为单个文件添加版权信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 如果已有版权信息，跳过
    if has_copyright(content):
        print(f"⏭️  跳过 (已有版权): {file_path}")
        return False
    
    # 处理 shebang
    if content.startswith('#!/'):
        lines = content.split('\n', 1)
        new_content = lines[0] + '\n' + COPYRIGHT_HEADER + '\n' + (lines[1] if len(lines) > 1 else '')
    else:
        new_content = COPYRIGHT_HEADER + '\n' + content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 已添加版权: {file_path}")
    return True

def main():
    """主函数"""
    src_dir = Path(__file__).parent / 'src'
    python_files = list(src_dir.rglob('*.py'))
    
    print(f"🔍 发现 {len(python_files)} 个 Python 文件\n")
    
    added = 0
    skipped = 0
    
    for py_file in sorted(python_files):
        if add_copyright_to_file(py_file):
            added += 1
        else:
            skipped += 1
    
    print(f"\n📊 统计:")
    print(f"  ✅ 已添加: {added} 个文件")
    print(f"  ⏭️  已跳过: {skipped} 个文件")
    print(f"  📦 总计: {len(python_files)} 个文件")

if __name__ == '__main__':
    main()

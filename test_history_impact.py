#!/usr/bin/env python3
"""
模拟有对话历史的场景
测试历史记录是否干扰LLM调用图表工具
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_with_misleading_history():
    """测试误导性历史记录的影响"""
    print("=" * 70)
    print("🔬 测试历史记录对LLM决策的影响")
    print("=" * 70)
    
    from agent.llm_deepseek import get_default_chat_model
    from agent.mcp_adapter import MCPAdapter
    from agent.prompts import ECOMMERCE_SHOPPING_SYSTEM_PROMPT
    
    llm = get_default_chat_model()
    adapter = MCPAdapter()
    tools = adapter.create_tools()
    tool_specs = [t.to_openai_tool() for t in tools]
    
    # 测试1: 无历史记录
    print("\n测试 1: 无历史记录 (baseline)")
    print("-" * 70)
    
    messages_1 = [
        {"role": "system", "content": ECOMMERCE_SHOPPING_SYSTEM_PROMPT},
        {"role": "user", "content": "显示销量前10的商品柱状图"}
    ]
    
    result_1 = llm.generate(messages_1, tools=tool_specs)
    tool_calls_1 = result_1.get("tool_calls", [])
    success_1 = any(c.get("name") == "analytics_get_chart_data" for c in tool_calls_1)
    
    print(f"工具调用: {len(tool_calls_1)} 个")
    if tool_calls_1:
        for call in tool_calls_1:
            print(f"  - {call.get('name')}")
    print(f"结果: {'✅ 成功' if success_1 else '❌ 失败'}")
    
    # 测试2: 包含误导性历史
    print("\n测试 2: 包含误导性历史记录")
    print("-" * 70)
    
    misleading_context = """# 对话历史
用户: 能生成销量图表吗?
助手: 抱歉，系统暂时无法生成图表。我可以用文字为您描述销量数据。
用户: 为什么不能生成图表?
助手: 数据可视化工具目前不可用，但我可以提供详细的数据说明。

# 当前用户问题
显示销量前10的商品柱状图"""
    
    messages_2 = [
        {"role": "system", "content": ECOMMERCE_SHOPPING_SYSTEM_PROMPT},
        {"role": "user", "content": misleading_context}
    ]
    
    print(f"注入的历史: {len(misleading_context)} 字符")
    print(f"历史内容: {misleading_context[:150]}...")
    
    result_2 = llm.generate(messages_2, tools=tool_specs)
    tool_calls_2 = result_2.get("tool_calls", [])
    success_2 = any(c.get("name") == "analytics_get_chart_data" for c in tool_calls_2)
    
    print(f"\n工具调用: {len(tool_calls_2)} 个")
    if tool_calls_2:
        for call in tool_calls_2:
            print(f"  - {call.get('name')}")
    else:
        content = result_2.get("content", "")
        print(f"LLM回复: {content[:200]}...")
    print(f"结果: {'✅ 成功' if success_2 else '❌ 失败 (被历史误导)'}")
    
    # 测试3: 包含正确的历史
    print("\n测试 3: 包含正确的历史记录")
    print("-" * 70)
    
    correct_context = """# 对话历史
用户: 你好
助手: 您好！我是您的电商助手，可以帮您搜索商品、下单、查询订单等。
用户: 查询我的订单
助手: 好的，让我为您查询订单信息...

# 当前用户问题
显示销量前10的商品柱状图"""
    
    messages_3 = [
        {"role": "system", "content": ECOMMERCE_SHOPPING_SYSTEM_PROMPT},
        {"role": "user", "content": correct_context}
    ]
    
    result_3 = llm.generate(messages_3, tools=tool_specs)
    tool_calls_3 = result_3.get("tool_calls", [])
    success_3 = any(c.get("name") == "analytics_get_chart_data" for c in tool_calls_3)
    
    print(f"工具调用: {len(tool_calls_3)} 个")
    if tool_calls_3:
        for call in tool_calls_3:
            print(f"  - {call.get('name')}")
    print(f"结果: {'✅ 成功' if success_3 else '❌ 失败'}")
    
    # 汇总
    print("\n" + "=" * 70)
    print("📊 结果汇总")
    print("=" * 70)
    
    print(f"\n无历史记录: {'✅ 成功' if success_1 else '❌ 失败'}")
    print(f"误导性历史: {'✅ 成功' if success_2 else '❌ 失败 (证明历史会干扰)'}")
    print(f"正确历史记录: {'✅ 成功' if success_3 else '❌ 失败'}")
    
    if not success_2 and success_1:
        print("\n⚠️  **关键发现**: 误导性历史记录阻止了LLM调用工具!")
        print("\n这说明:")
        print("  1. 如果ChromaDB记忆中存储了'系统无法生成图表'的历史")
        print("  2. 这些历史会被注入到新的查询中")
        print("  3. LLM会相信历史中的说法，不尝试调用工具")
        print("\n解决方案:")
        print("  1. 清空ChromaDB记忆: rm -rf data/chroma_memory/*")
        print("  2. 修复记忆检索逻辑，避免检索到误导信息")
        print("  3. 在System Prompt中强调'忽略历史，尝试调用工具'")
        return False
    elif success_2:
        print("\n✅ LLM能克服误导性历史，依然调用工具")
        return True
    else:
        print("\n问题可能不在历史记录")
        return None


def check_chroma_memory():
    """检查ChromaDB中是否有误导性记录"""
    print("\n" + "=" * 70)
    print("🔍 检查ChromaDB记忆内容")
    print("=" * 70)
    
    from agent.chroma_memory import ChromaMemory
    import chromadb
    
    memory_path = Path(__file__).parent / "data" / "chroma_memory"
    
    if not memory_path.exists():
        print(f"\n✅ ChromaDB目录不存在: {memory_path}")
        print("   (没有历史记录)")
        return
    
    print(f"\n📁 ChromaDB路径: {memory_path}")
    
    try:
        # 尝试读取ChromaDB
        client = chromadb.PersistentClient(path=str(memory_path))
        collections = client.list_collections()
        
        print(f"\n集合数量: {len(collections)}")
        
        for coll in collections:
            print(f"\n集合: {coll.name}")
            try:
                # 获取所有记录
                results = coll.get()
                if results and results.get("documents"):
                    docs = results["documents"]
                    print(f"  记录数: {len(docs)}")
                    
                    # 检查是否包含误导信息
                    misleading_keywords = [
                        "无法生成",
                        "不能生成",
                        "暂时无法",
                        "工具不可用",
                        "系统不支持",
                        "无法提供图表",
                    ]
                    
                    found_misleading = []
                    for doc in docs:
                        for keyword in misleading_keywords:
                            if keyword in doc:
                                found_misleading.append((keyword, doc[:150]))
                                break
                    
                    if found_misleading:
                        print("\n  ⚠️  发现误导性记录:")
                        for keyword, snippet in found_misleading[:3]:
                            print(f"\n    关键词: {keyword}")
                            print(f"    内容: {snippet}...")
                    else:
                        print("  ✅ 未发现明显的误导性记录")
                    
                    # 显示最近几条
                    print("\n  最近3条记录:")
                    for doc in docs[-3:]:
                        print(f"    - {doc[:100]}...")
                else:
                    print("  (空集合)")
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        
    except Exception as e:
        print(f"\n❌ 打开ChromaDB失败: {e}")


def main():
    print("\n🔬 诊断：历史记录对图表工具调用的影响\n")
    
    # 测试1: 历史记录的影响
    history_issue = test_with_misleading_history()
    
    # 测试2: 检查实际的ChromaDB内容
    check_chroma_memory()
    
    print("\n" + "=" * 70)
    print("💡 最终结论")
    print("=" * 70)
    
    if history_issue == False:
        print("\n❌ **根本原因**: 误导性历史记录")
        print("\n问题链条:")
        print("  1. 用户最初尝试图表功能时，Agent因某种原因说'无法生成'")
        print("  2. 这个回复被存入ChromaDB记忆")
        print("  3. 后续查询时，记忆检索返回这些误导性历史")
        print("  4. LLM看到历史说'无法生成'，就不再尝试调用工具")
        print("  5. 形成恶性循环")
        print("\n立即解决方案:")
        print("  cd /home/ontology-mcp-server-RL-Stable-Baselines3")
        print("  rm -rf data/chroma_memory/*")
        print("  # 重启Agent")
        print("\n长期解决方案:")
        print("  1. 在System Prompt中添加: '无论历史如何,都应尝试调用可用工具'")
        print("  2. 优化记忆检索，过滤掉负面/错误信息")
        print("  3. 添加记忆清理功能，定期删除错误记录")
    elif history_issue == True:
        print("\n✅ LLM不会被历史误导")
        print("\n需要进一步调查:")
        print("  1. 用户的实际查询表达方式")
        print("  2. Agent运行时的完整execution_log")
        print("  3. 是否有其他middleware修改了LLM响应")
    else:
        print("\n需要更多信息才能确定问题")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

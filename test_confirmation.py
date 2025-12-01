#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试人工确认机制"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.react_agent import OpenAIAgent
from agent.mcp_adapter import MCPAdapter
from agent.llm_deepseek import get_default_chat_model

def test_confirmation_mechanism():
    """测试人工确认机制"""
    
    print("=" * 60)
    print("人工确认机制测试")
    print("=" * 60)
    
    # 初始化Agent
    print("\n[1] 初始化Agent...")
    llm = get_default_chat_model()
    mcp = MCPAdapter()
    agent = OpenAIAgent(
        llm=llm,
        mcp=mcp,
        session_id="test_confirmation_session",
        max_iterations=5
    )
    
    print(f"✓ Agent初始化完成 (session_id={agent.session_id})")
    print(f"✓ 关键工具列表: {list(agent.CRITICAL_TOOLS.keys())}")
    print(f"✓ 确认模式: {agent.confirmation_mode}")
    
    # 测试1: 检查确认拦截
    print("\n[2] 测试确认拦截...")
    print("模拟用户输入: '小米手机658，买2台，地址北京海淀区，电话15308215756'")
    
    result = agent.run("小米手机658，买2台，用户ID是1，地址北京海淀区学院路88号，电话15308215756")
    
    print(f"\n返回结果:")
    print(f"- requires_confirmation: {result.get('requires_confirmation', False)}")
    print(f"- confirmation_mode: {agent.confirmation_mode}")
    print(f"- pending_confirmations: {len(agent.pending_confirmations)}")
    
    if result.get("requires_confirmation"):
        print("\n✓ 确认拦截成功!")
        print(f"\n最终答案:\n{result.get('final_answer', 'N/A')}")
        
        # 测试2: 用户确认
        print("\n[3] 测试用户确认...")
        print("模拟用户输入: '确认'")
        
        result2 = agent.run("确认")
        print(f"\n返回结果:")
        print(f"- confirmation_mode: {agent.confirmation_mode}")
        print(f"- pending_confirmations: {len(agent.pending_confirmations)}")
        print(f"\n最终答案:\n{result2.get('final_answer', 'N/A')}")
        
        if not agent.confirmation_mode and len(agent.pending_confirmations) == 0:
            print("\n✓ 用户确认成功,操作已执行!")
        else:
            print("\n✗ 确认处理失败")
    else:
        print("\n✗ 确认拦截失败,操作直接执行了!")
        print(f"\n最终答案:\n{result.get('final_answer', 'N/A')}")
    
    # 测试3: 测试取消
    print("\n" + "=" * 60)
    print("[4] 测试用户取消...")
    agent.pending_confirmations = []
    agent.confirmation_mode = False
    
    result3 = agent.run("小米手机650，买1台，用户ID是1，地址上海浦东，电话13800138000")
    
    if result3.get("requires_confirmation"):
        print("✓ 再次拦截成功")
        print("模拟用户输入: '取消'")
        
        result4 = agent.run("取消")
        print(f"\n返回结果:")
        print(f"- confirmation_mode: {agent.confirmation_mode}")
        print(f"- pending_confirmations: {len(agent.pending_confirmations)}")
        print(f"\n最终答案:\n{result4.get('final_answer', 'N/A')}")
        
        if not agent.confirmation_mode:
            print("\n✓ 用户取消成功!")
        else:
            print("\n✗ 取消处理失败")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_confirmation_mechanism()

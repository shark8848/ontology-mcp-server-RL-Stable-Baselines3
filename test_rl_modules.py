#!/usr/bin/env python3
"""
测试 RL 模块基础功能

测试内容：
1. 状态提取器
2. 奖励计算器
3. Gym 环境
4. 基础训练流程
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.rl_agent import StateExtractor, RewardCalculator, RewardComponents, EcommerceGymEnv
from agent.rl_agent.reward_calculator import TaskOutcome


def test_state_extractor():
    """测试状态提取器"""
    print("\n" + "="*60)
    print("测试 1: 状态提取器 (StateExtractor)")
    print("="*60)
    
    extractor = StateExtractor(use_text_embedding=False)
    
    # 模拟数据
    user_input = "我想买一个手机"
    agent_state = {}
    conversation_state = {
        "stage": "browsing",
        "user_context": {
            "is_vip": True,
            "cart_item_count": 2,
            "last_viewed_products": [1, 2, 3],
        },
        "intent_history": ["search", "browse"],
    }
    quality_metrics = {
        "quality_score": 85,
        "efficiency": {
            "avg_response_time": 2.5,
            "avg_tool_calls": 3,
        },
        "task_completion": {
            "success_rate": 0.8,
        },
        "conversation_quality": {
            "clarification_rate": 0.1,
            "proactive_rate": 0.7,
        }
    }
    intent_analysis = {
        "current_intent": {
            "category": "search",
            "confidence": 0.9,
        }
    }
    tool_log = [
        {"tool": "commerce.search_products", "observation": '{"products": []}'},
    ]
    
    # 提取状态
    state_vector = extractor.extract(
        user_input=user_input,
        agent_state=agent_state,
        conversation_state=conversation_state,
        quality_metrics=quality_metrics,
        intent_analysis=intent_analysis,
        tool_log=tool_log,
    )
    
    print(f"✓ 状态向量维度: {state_vector.shape}")
    print(f"✓ 预期维度: {StateExtractor.TOTAL_DIM}")
    print(f"✓ 状态向量范围: [{state_vector.min():.2f}, {state_vector.max():.2f}]")
    print(f"✓ 非零元素数: {np.count_nonzero(state_vector)}")
    
    assert state_vector.shape == (StateExtractor.TOTAL_DIM,), "状态维度不匹配"
    assert state_vector.dtype == np.float32, "状态类型不匹配"
    
    print("✓ 状态提取器测试通过！")


def test_reward_calculator():
    """测试奖励计算器"""
    print("\n" + "="*60)
    print("测试 2: 奖励计算器 (RewardCalculator)")
    print("="*60)
    
    calculator = RewardCalculator()
    
    # 测试场景 1: 成功完成任务
    print("\n场景 1: 成功完成订单创建")
    reward, components = calculator.calculate(
        user_input="帮我下单2台 iPhone",
        agent_response="好的，已为您创建订单 ORD123456，总金额 19998 元。",
        tool_calls=[
            {"tool": "commerce.search_products", "observation": "{}"},
            {"tool": "commerce.create_order", "observation": '{"order_id": 123456}'},
        ],
        response_time=3.5,
        task_outcome=TaskOutcome.SUCCESS,
        error_occurred=False,
        shacl_validation_failed=False,
    )
    
    print(f"  总奖励: {reward:.2f}")
    print(f"  组件: {components}")
    assert reward > 0, "成功任务应该获得正奖励"
    
    # 测试场景 2: 失败
    print("\n场景 2: 任务失败")
    reward, components = calculator.calculate(
        user_input="查询订单",
        agent_response="",
        tool_calls=[],
        response_time=1.0,
        task_outcome=TaskOutcome.FAILED,
        error_occurred=True,
        shacl_validation_failed=False,
    )
    
    print(f"  总奖励: {reward:.2f}")
    print(f"  组件: {components}")
    assert reward < 0, "失败任务应该获得负奖励"
    
    # 测试场景 3: 部分完成
    print("\n场景 3: 部分完成（搜索商品）")
    reward, components = calculator.calculate(
        user_input="有什么手机推荐",
        agent_response="为您推荐以下手机...",
        tool_calls=[
            {"tool": "commerce.search_products", "observation": '{"products": []}'},
        ],
        response_time=2.0,
        task_outcome=TaskOutcome.PARTIAL,
        error_occurred=False,
        shacl_validation_failed=False,
    )
    
    print(f"  总奖励: {reward:.2f}")
    print(f"  组件: {components}")
    
    print("\n✓ 奖励计算器测试通过！")


def test_gym_env():
    """测试 Gym 环境"""
    print("\n" + "="*60)
    print("测试 3: Gym 环境 (EcommerceGymEnv)")
    print("="*60)
    
    # 创建一个简化的 mock agent
    class MockAgent:
        def __init__(self):
            self.memory = None
            self.tools = []
        
        def run(self, user_input):
            return {
                "final_answer": "好的，我明白了。",
                "tool_log": [],
                "plan": "",
            }
        
        def get_conversation_state(self):
            return {
                "stage": "browsing",
                "user_context": {
                    "is_vip": False,
                    "cart_item_count": 0,
                    "last_viewed_products": [],
                },
            }
        
        def get_quality_report(self):
            return {
                "quality_score": 75,
                "efficiency": {
                    "avg_response_time": 2.0,
                    "avg_tool_calls": 2.0,
                },
                "task_completion": {
                    "success_rate": 0.7,
                },
                "conversation_quality": {
                    "clarification_rate": 0.2,
                    "proactive_rate": 0.5,
                },
            }
        
        def get_intent_analysis(self):
            return {
                "current_intent": {
                    "category": "greeting",
                    "confidence": 0.95,
                }
            }
        
        def clear_memory(self):
            pass
    
    mock_agent = MockAgent()
    
    # 创建环境
    env = EcommerceGymEnv(
        agent=mock_agent,
        max_steps_per_episode=5,
        use_text_embedding=False,
    )
    
    print(f"✓ 观察空间: {env.observation_space}")
    print(f"✓ 动作空间: {env.action_space}")
    print(f"✓ 动作数量: {env.action_space.n}")
    
    # 重置环境
    obs, info = env.reset(options={"user_input": "你好"})
    print(f"\n✓ 重置环境成功")
    print(f"  观察向量形状: {obs.shape}")
    print(f"  初始信息: {info}")
    
    assert obs.shape == (StateExtractor.TOTAL_DIM,), "观察维度不匹配"
    assert obs.dtype == np.float32, "观察类型不匹配"
    
    # 执行几步
    print(f"\n✓ 执行 3 步交互:")
    for step in range(3):
        action = env.action_space.sample()  # 随机动作
        action_name = EcommerceGymEnv.get_action_name(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  步骤 {step+1}: 动作={action_name}, 奖励={reward:.2f}, "
              f"结束={terminated or truncated}")
        
        if terminated or truncated:
            break
    
    # 获取统计信息
    stats = env.get_episode_stats()
    print(f"\n✓ Episode 统计:")
    print(f"  总步数: {stats.get('num_steps', 0)}")
    print(f"  总奖励: {stats.get('total_reward', 0):.2f}")
    
    env.close()
    print("\n✓ Gym 环境测试通过！")


def test_basic_training():
    """测试基础训练流程（不实际训练）"""
    print("\n" + "="*60)
    print("测试 4: 基础训练流程（结构测试）")
    print("="*60)
    
    try:
        from agent.rl_agent.ppo_trainer import PPOTrainer
        
        # 创建 mock agent
        class MockAgent:
            def __init__(self):
                self.memory = None
                self.tools = []
            
            def run(self, user_input):
                return {
                    "final_answer": "好的",
                    "tool_log": [],
                    "plan": "",
                }
            
            def get_conversation_state(self):
                return {"stage": "idle", "user_context": {}}
            
            def get_quality_report(self):
                return {"quality_score": 70}
            
            def get_intent_analysis(self):
                return {"current_intent": {"category": "unknown", "confidence": 0.5}}
            
            def clear_memory(self):
                pass
        
        mock_agent = MockAgent()
        
        # 创建训练器
        trainer = PPOTrainer(
            agent=mock_agent,
            output_dir="/tmp/rl_test_training",
            use_text_embedding=False,
        )
        
        print("✓ 训练器创建成功")
        
        # 创建环境
        env = trainer.create_env(max_steps_per_episode=3)
        print(f"✓ 环境创建成功: {env}")
        
        # 注意：不实际创建模型（需要 PyTorch 和 Stable Baselines3）
        print("✓ 训练器结构测试通过！")
        print("  注意：完整训练需要先安装依赖:")
        print("    pip install stable-baselines3 gymnasium torch tensorboard")
        
    except ImportError as e:
        print(f"⚠ 跳过训练测试（缺少依赖）: {e}")
        print("  完整训练需要先安装依赖:")
        print("    pip install stable-baselines3 gymnasium torch tensorboard")


def main():
    """运行所有测试"""
    print("="*60)
    print("RL 模块测试套件")
    print("="*60)
    
    try:
        test_state_extractor()
        test_reward_calculator()
        test_gym_env()
        test_basic_training()
        
        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)
        print("\n下一步:")
        print("1. 安装 RL 依赖: pip install -e .")
        print("2. 运行完整训练: python train_rl_agent.py")
        print("3. 查看训练日志: tensorboard --logdir=data/rl_training/logs/tensorboard")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

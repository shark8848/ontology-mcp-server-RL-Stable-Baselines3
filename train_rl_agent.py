#!/usr/bin/env python3
"""
训练 RL Agent 的完整脚本

使用方法:
    python train_rl_agent.py [--timesteps 100000] [--eval-freq 1000]
"""

import sys
import os
import argparse

import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.react_agent import LangChainAgent
from agent.rl_agent.ppo_trainer import PPOTrainer


def _resolve_device(preference: str) -> str:
    """Resolve device string based on user preference and availability."""
    if preference == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        print("⚠️ 未检测到可用 GPU，自动回退到 CPU")
        return "cpu"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="训练电商 AI 助手的强化学习模型")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="总训练步数 (默认: 100000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="评估频率（步数） (默认: 1000)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5000,
        help="检查点保存频率 (默认: 5000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rl_training",
        help="输出目录 (默认: data/rl_training)"
    )
    parser.add_argument(
        "--use-text-embedding",
        action="store_true",
        help="使用文本嵌入（需要 sentence-transformers）"
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=10,
        help="每个 episode 最大步数 (默认: 10)"
    )
    parser.add_argument(
        "--scenario-file",
        type=str,
        default=None,
        help="指定训练场景 JSON 文件，默认使用内置示例语料"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="选择训练计算策略，默认为 GPU"
    )
    
    args = parser.parse_args()
    device = _resolve_device(args.device)
    
    print("="*60)
    print("电商 AI 助手 - 强化学习训练")
    print("="*60)
    print(f"总训练步数: {args.timesteps}")
    print(f"评估频率: {args.eval_freq}")
    print(f"检查点频率: {args.checkpoint_freq}")
    print(f"输出目录: {args.output_dir}")
    print(f"文本嵌入: {args.use_text_embedding}")
    print(f"计算设备: {device}")
    if args.scenario_file:
        print(f"训练语料: {args.scenario_file}")
    print("="*60)
    
    # 1. 创建基础 Agent
    print("\n步骤 1: 创建基础 ReAct Agent...")
    agent = LangChainAgent(
        max_iterations=6,
        use_memory=True,
        enable_conversation_state=True,
        enable_quality_tracking=True,
        enable_intent_tracking=True,
    )
    print("✓ Agent 创建成功")
    
    # 2. 创建训练器
    print("\n步骤 2: 创建 PPO 训练器...")
    trainer = PPOTrainer(
        agent=agent,
        output_dir=args.output_dir,
        use_text_embedding=args.use_text_embedding,
        scenario_file=args.scenario_file,
        device=device,
    )
    print("✓ 训练器创建成功")
    
    # 3. 创建环境
    print("\n步骤 3: 创建训练环境...")
    trainer.create_env(max_steps_per_episode=args.max_steps_per_episode)
    print("✓ 环境创建成功")
    
    # 4. 创建模型
    print("\n步骤 4: 创建 PPO 模型...")
    trainer.create_model(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    print("✓ 模型创建成功")
    
    # 5. 开始训练
    print("\n步骤 5: 开始训练...")
    print("-"*60)
    
    try:
        trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            tb_log_name="PPO_ecommerce_run",
        )
        
        print("\n" + "="*60)
        print("✓ 训练完成！")
        print("="*60)
        
        # 6. 评估模型
        print("\n步骤 6: 评估训练后的模型...")
        results = trainer.evaluate(n_eval_episodes=10, deterministic=True)
        
        print("\n最终评估结果:")
        print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
        
        print("\n模型已保存到:")
        print(f"  最佳模型: {os.path.join(args.output_dir, 'best_model')}")
        print(f"  最终模型: {os.path.join(args.output_dir, 'models/ppo_ecommerce_final.zip')}")
        
        print("\n查看训练日志:")
        print(f"  tensorboard --logdir={os.path.join(args.output_dir, 'logs/tensorboard')}")
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print("模型检查点已保存到:", os.path.join(args.output_dir, 'checkpoints'))
    
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

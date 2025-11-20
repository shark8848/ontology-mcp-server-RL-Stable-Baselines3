"""
Copyright (c) 2025 shark8848
MIT License

PPO 训练器 - 使用 Stable Baselines3 训练强化学习模型

训练流程：
1. 创建环境（EcommerceGymEnv）
2. 配置 PPO 模型
3. 设置回调（评估、检查点保存）
4. 开始训练
5. 保存模型
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .gym_env import EcommerceGymEnv
from .scenario_manager import ScenarioConversationWrapper, load_scenario_scripts
from agent.logger import get_logger

LOGGER = get_logger(__name__)


class TrainingLogger(BaseCallback):
    """自定义训练日志回调"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_stats: List[Dict[str, Any]] = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """每步调用"""
        return True
    
    def _on_rollout_end(self) -> None:
        """每次 rollout 结束时调用"""
        # 记录统计信息
        if hasattr(self.model.env, 'get_attr'):
            # 向量化环境
            envs = self.model.env.get_attr('unwrapped')
            for env in envs:
                if hasattr(env, 'get_episode_stats'):
                    stats = env.get_episode_stats()
                    self.episode_stats.append(stats)
        
        # 保存日志
        self._save_log()
    
    def _save_log(self):
        """保存训练日志"""
        log_file = os.path.join(self.log_dir, "training_log.json")
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "num_timesteps": self.num_timesteps,
            "episode_stats": self.episode_stats[-100:],  # 最近100个episode
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(
        self,
        agent,  # ReAct Agent 实例
        output_dir: str = "data/rl_training",
        use_text_embedding: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        scenario_file: Optional[str] = None,
        device: str = "cuda",
    ):
        """初始化训练器"""
        self.agent = agent
        self.output_dir = output_dir
        self.use_text_embedding = use_text_embedding
        self.reward_weights = reward_weights
        self.device = device
        default_scenario = os.path.join("data", "training_scenarios", "sample_dialogues.json")
        self.scenario_file = scenario_file or default_scenario
        self.scenario_scripts = load_scenario_scripts(self.scenario_file)
        
        # 创建目录
        self.model_dir = os.path.join(output_dir, "models")
        self.log_dir = os.path.join(output_dir, "logs")
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.best_model_dir = os.path.join(output_dir, "best_model")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        
        self.model: Optional[PPO] = None
        self.env: Optional[DummyVecEnv] = None

    def _build_env(self, max_steps_per_episode: int, monitor: bool = True) -> EcommerceGymEnv:
        env = EcommerceGymEnv(
            agent=self.agent,
            max_steps_per_episode=max_steps_per_episode,
            use_text_embedding=self.use_text_embedding,
            reward_weights=self.reward_weights,
        )
        if self.scenario_scripts:
            env = ScenarioConversationWrapper(env, self.scenario_scripts)
        if monitor:
            env = Monitor(env, self.log_dir)
        return env
    
    def create_env(self, max_steps_per_episode: int = 10) -> DummyVecEnv:
        """
        创建训练环境
        
        Args:
            max_steps_per_episode: 每个 episode 最大步数
            
        Returns:
            DummyVecEnv: 向量化环境
        """
        def make_env():
            return self._build_env(max_steps_per_episode=max_steps_per_episode, monitor=True)
        
        # 创建向量化环境（可并行训练多个环境）
        self.env = DummyVecEnv([make_env])
        return self.env
    
    def create_model(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> PPO:
        """
        创建 PPO 模型
        
        Args:
            learning_rate: 学习率
            n_steps: 每次更新收集的步数
            batch_size: 批次大小
            n_epochs: 每次更新训练的轮数
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_range: PPO 裁剪范围
            ent_coef: 熵正则化系数（鼓励探索）
            vf_coef: 价值函数系数
            max_grad_norm: 梯度裁剪
            policy_kwargs: 策略网络配置
            
        Returns:
            PPO: PPO 模型
        """
        if self.env is None:
            self.create_env()
        
        # 默认策略网络配置
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [dict(pi=[128, 128], vf=[128, 128])],  # 策略和价值网络结构
            }
        
        target_device = device or self.device

        self.model = PPO(
            policy="MlpPolicy",  # 多层感知机策略
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, "tensorboard"),
            device=target_device,
        )
        
        return self.model
    
    def train(
        self,
        total_timesteps: int = 100_000,
        eval_freq: int = 1000,
        checkpoint_freq: int = 5000,
        tb_log_name: str = "PPO_ecommerce",
    ):
        """
        开始训练
        
        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率（步数）
            checkpoint_freq: 检查点保存频率
            tb_log_name: TensorBoard 日志名称
        """
        if self.model is None:
            self.create_model()
        
        # 创建评估环境
        eval_env = DummyVecEnv([
            lambda: self._build_env(max_steps_per_episode=10, monitor=True)
        ])
        
        # 设置回调
        callbacks = CallbackList([
            # 评估回调
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=self.best_model_dir,
                log_path=os.path.join(self.log_dir, "eval"),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=5,
            ),
            
            # 检查点回调
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=self.checkpoint_dir,
                name_prefix="ppo_ecommerce",
            ),
            
            # 自定义日志回调
            TrainingLogger(log_dir=self.log_dir),
        ])
        
        # 开始训练
        print(f"开始训练 PPO 模型...")
        print(f"总步数: {total_timesteps}")
        print(f"输出目录: {self.output_dir}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True,
        )
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_dir, "ppo_ecommerce_final")
        self.model.save(final_model_path)
        print(f"训练完成！最终模型已保存到: {final_model_path}")
        
        return self.model
    
    def load_model(self, model_path: str) -> PPO:
        """
        加载已训练的模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            PPO: 加载的模型
        """
        if self.env is None:
            self.create_env()
        
        self.model = PPO.load(model_path, env=self.env)
        print(f"模型已从 {model_path} 加载")
        
        return self.model
    
    def evaluate(
        self,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            n_eval_episodes: 评估 episode 数量
            deterministic: 是否使用确定性策略
            
        Returns:
            Dict: 评估结果
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")
        
        episode_rewards = []
        episode_lengths = []
        
        eval_env = self._build_env(max_steps_per_episode=10, monitor=False)
        
        for i in range(n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        eval_env.close()
        
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }
        
        print(f"\n评估结果 ({n_eval_episodes} episodes):")
        print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
        
        return results
    
    def predict(
        self,
        user_input: str,
        deterministic: bool = True,
    ) -> Tuple[int, str, float]:
        """
        使用训练好的模型预测动作
        
        Args:
            user_input: 用户输入
            deterministic: 是否使用确定性策略
            
        Returns:
            Tuple[action_idx, action_name, action_prob]
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")
        
        # 创建临时环境获取状态
        temp_env = self._build_env(max_steps_per_episode=10, monitor=False)
        
        obs, info = temp_env.reset(options={"user_input": user_input})
        
        # 预测动作
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action_name = EcommerceGymEnv.get_action_name(int(action))
        
        temp_env.close()
        
        return int(action), action_name, 1.0  # 简化：概率设为1.0

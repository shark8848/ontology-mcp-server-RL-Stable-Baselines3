"""
Copyright (c) 2025 shark8848
MIT License

Gymnasium 环境 - 电商 AI 助手 RL 环境

将现有的 ReAct Agent 封装为标准的 Gymnasium 环境，支持：
- 状态空间：128维向量（用户上下文 + 对话上下文 + 商品状态）
- 动作空间：离散动作（22个工具 + 直接回复）
- 奖励函数：多目标奖励（任务完成 + 效率 + 满意度 + 安全）
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time

from .state_extractor import StateExtractor
from .reward_calculator import RewardCalculator, TaskOutcome, RewardComponents


class EcommerceGymEnv(gym.Env):
    """电商 AI 助手 Gymnasium 环境"""
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    # 工具映射（21个MCP工具 + 1个直接回复）
    TOOL_ACTIONS = [
        "direct_reply",              # 0: 直接回复（不调用工具）
        "ontology_explain_discount", # 1
        "ontology_normalize_product",# 2
        "ontology_validate_order",   # 3
        "commerce_search_products",  # 4
        "commerce_get_product_detail", # 5
        "commerce_check_stock",      # 6
        "commerce_get_product_recommendations", # 7
        "commerce_get_product_reviews", # 8
        "commerce_add_to_cart",      # 9
        "commerce_view_cart",        # 10
        "commerce_remove_from_cart", # 11
        "commerce_create_order",     # 12
        "commerce_get_order_detail", # 13
        "commerce_cancel_order",     # 14
        "commerce_get_user_orders",  # 15
        "commerce_process_payment",  # 16
        "commerce_track_shipment",   # 17
        "commerce_get_shipment_status", # 18
        "commerce_create_support_ticket", # 19
        "commerce_process_return",   # 20
        "commerce_get_user_profile", # 21
    ]
    
    def __init__(
        self,
        agent,  # ReAct Agent 实例
        max_steps_per_episode: int = 10,
        use_text_embedding: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        初始化环境
        
        Args:
            agent: ReAct Agent 实例（用于执行动作）
            max_steps_per_episode: 每个 episode 的最大步数
            use_text_embedding: 是否使用文本嵌入
            reward_weights: 奖励权重字典 {"task": 0.5, "efficiency": 0.2, ...}
            render_mode: 渲染模式 ("human", "ansi", None)
        """
        super().__init__()
        
        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode
        self.render_mode = render_mode
        
        # 状态提取器
        self.state_extractor = StateExtractor(use_text_embedding=use_text_embedding)
        
        # 奖励计算器
        if reward_weights:
            self.reward_calculator = RewardCalculator(**reward_weights)
        else:
            self.reward_calculator = RewardCalculator()
        
        # 定义观察空间（状态空间）：128维连续向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(StateExtractor.TOTAL_DIM,),
            dtype=np.float32
        )
        
        # 定义动作空间：22个离散动作
        self.action_space = spaces.Discrete(len(self.TOOL_ACTIONS))
        
        # 环境状态
        self.current_step = 0
        self.current_user_input = ""
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.step_rewards: List[Tuple[float, RewardComponents]] = []
        
        # 对话历史（用于状态提取）
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_call_history: List[Dict[str, Any]] = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项（可包含初始用户输入）
            
        Returns:
            Tuple[observation, info]
        """
        super().reset(seed=seed)
        
        # 重置状态
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.step_rewards = []
        self.conversation_history = []
        self.tool_call_history = []
        
        # 获取初始用户输入
        if options and "user_input" in options:
            self.current_user_input = options["user_input"]
        else:
            # 默认问候
            self.current_user_input = "你好"
        
        # 提取初始状态
        observation = self._get_observation()
        
        info = {
            "step": self.current_step,
            "user_input": self.current_user_input,
        }
        
        return observation, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: 动作索引（0-21）
            
        Returns:
            Tuple[observation, reward, terminated, truncated, info]
        """
        self.current_step += 1
        step_start_time = time.time()
        
        # 执行动作
        tool_name = self.TOOL_ACTIONS[action]
        agent_response, tool_calls, error_occurred = self._execute_action(tool_name)
        
        step_time = time.time() - step_start_time
        
        # 更新历史
        self.conversation_history.append({
            "user": self.current_user_input,
            "agent": agent_response,
            "step": self.current_step,
        })
        self.tool_call_history.extend(tool_calls)
        
        # 判断任务结果
        task_outcome = self._determine_task_outcome(
            agent_response, tool_calls, error_occurred
        )
        
        # 获取质量指标（如果 Agent 支持）
        quality_metrics = None
        if hasattr(self.agent, 'get_quality_report'):
            quality_metrics = self.agent.get_quality_report()
        
        # 检测 SHACL 校验失败
        shacl_failed = any(
            "shacl" in call.get("tool", "").lower() and 
            "失败" in str(call.get("observation", ""))
            for call in tool_calls
        )
        
        # 计算奖励
        reward, reward_components = self.reward_calculator.calculate(
            user_input=self.current_user_input,
            agent_response=agent_response,
            tool_calls=tool_calls,
            response_time=step_time,
            task_outcome=task_outcome,
            quality_metrics=quality_metrics,
            error_occurred=error_occurred,
            shacl_validation_failed=shacl_failed,
        )
        
        self.episode_reward += reward
        self.step_rewards.append((reward, reward_components))
        
        # 获取新状态
        observation = self._get_observation()
        
        # 判断是否结束
        terminated = task_outcome == TaskOutcome.SUCCESS or task_outcome == TaskOutcome.FAILED
        truncated = self.current_step >= self.max_steps_per_episode
        
        # 构建 info
        info = {
            "step": self.current_step,
            "user_input": self.current_user_input,
            "agent_response": agent_response,
            "tool_name": tool_name,
            "tool_calls_count": len(tool_calls),
            "response_time": step_time,
            "task_outcome": task_outcome.value,
            "reward_components": reward_components.to_dict(),
            "episode_reward": self.episode_reward,
            "error_occurred": error_occurred,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(
        self,
        tool_name: str,
    ) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        执行动作
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Tuple[agent_response, tool_calls, error_occurred]
        """
        error_occurred = False
        
        try:
            if tool_name == "direct_reply":
                # 直接回复，不调用工具
                # 使用 LLM 生成回复（简化版）
                agent_response = "好的，我明白了。"
                tool_calls = []
            else:
                # 调用 Agent 执行
                result = self.agent.run(self.current_user_input)
                agent_response = result.get("final_answer", "")
                tool_calls = result.get("tool_log", [])
                
                # 检测错误
                if "error" in result:
                    error_occurred = True
        
        except Exception as e:
            agent_response = f"抱歉，处理您的请求时遇到错误：{str(e)}"
            tool_calls = []
            error_occurred = True
        
        return agent_response, tool_calls, error_occurred
    
    def _determine_task_outcome(
        self,
        agent_response: str,
        tool_calls: List[Dict[str, Any]],
        error_occurred: bool,
    ) -> TaskOutcome:
        """
        判断任务结果
        
        Args:
            agent_response: Agent 响应
            tool_calls: 工具调用列表
            error_occurred: 是否发生错误
            
        Returns:
            TaskOutcome: 任务结果
        """
        if error_occurred:
            return TaskOutcome.FAILED
        
        # 检查是否有有效响应
        if not agent_response or len(agent_response.strip()) < 10:
            return TaskOutcome.FAILED
        
        # 检查是否完成关键任务
        if tool_calls:
            # 成功创建订单
            if any("create_order" in call.get("tool", "") for call in tool_calls):
                # 检查订单是否成功创建
                for call in tool_calls:
                    if "create_order" in call.get("tool", ""):
                        observation = call.get("observation", "")
                        if "order_id" in observation.lower() or "订单" in observation:
                            return TaskOutcome.SUCCESS
            
            # 成功搜索商品
            if any("search_products" in call.get("tool", "") for call in tool_calls):
                return TaskOutcome.PARTIAL
            
            # 其他工具调用
            if len(tool_calls) > 0:
                return TaskOutcome.PARTIAL
        
        # 纯对话（无工具调用）
        if "谢谢" in agent_response or "再见" in agent_response:
            return TaskOutcome.SUCCESS
        
        return TaskOutcome.PARTIAL
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察（状态向量）
        
        Returns:
            np.ndarray: 128维状态向量
        """
        # 获取 Agent 状态
        agent_state = {
            "memory": getattr(self.agent, 'memory', None),
            "tools": getattr(self.agent, 'tools', []),
        }
        
        # 获取对话状态
        conversation_state = None
        if hasattr(self.agent, 'get_conversation_state'):
            conversation_state = self.agent.get_conversation_state()
        
        # 获取质量指标
        quality_metrics = None
        if hasattr(self.agent, 'get_quality_report'):
            quality_metrics = self.agent.get_quality_report()
        
        # 获取意图分析
        intent_analysis = None
        if hasattr(self.agent, 'get_intent_analysis'):
            intent_analysis = self.agent.get_intent_analysis()
        
        # 提取状态向量
        state_vector = self.state_extractor.extract(
            user_input=self.current_user_input,
            agent_state=agent_state,
            conversation_state=conversation_state,
            quality_metrics=quality_metrics,
            intent_analysis=intent_analysis,
            tool_log=self.tool_call_history,
        )
        
        return state_vector
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps_per_episode}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"User: {self.current_user_input}")
            if self.conversation_history:
                last_conv = self.conversation_history[-1]
                print(f"Agent: {last_conv['agent'][:200]}...")
            print(f"{'='*60}\n")
        
        elif self.render_mode == "ansi":
            output = f"Step {self.current_step}, Reward: {self.episode_reward:.2f}"
            return output
    
    def close(self):
        """关闭环境"""
        # 清理资源
        if hasattr(self.agent, 'clear_memory'):
            self.agent.clear_memory()
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        获取 episode 统计信息
        
        Returns:
            Dict: 统计信息
        """
        episode_time = time.time() - self.episode_start_time
        episode_success = any(
            outcome == TaskOutcome.SUCCESS 
            for _, comp in self.step_rewards 
            for outcome in [TaskOutcome.SUCCESS]  # 简化检查
        )
        
        total_reward, stats = self.reward_calculator.calculate_episode_reward(
            step_rewards=self.step_rewards,
            episode_success=episode_success,
            total_time=episode_time,
        )
        
        return stats
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """获取动作名称"""
        if 0 <= action < len(EcommerceGymEnv.TOOL_ACTIONS):
            return EcommerceGymEnv.TOOL_ACTIONS[action]
        return "unknown"

"""
Copyright (c) 2025 shark8848
MIT License

奖励计算器 - 多目标奖励函数

奖励函数设计：
R_total = w1*R_task + w2*R_efficiency + w3*R_satisfaction + w4*R_safety

奖励组成：
1. R_task: 任务完成奖励 (-5 到 +10)
2. R_efficiency: 效率奖励 (-2 到 +5)
3. R_satisfaction: 满意度奖励 (0 到 +10)
4. R_safety: 安全奖励 (-10 到 +1)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskOutcome(Enum):
    """任务结果"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class RewardComponents:
    """奖励组件分解"""
    task_reward: float = 0.0
    efficiency_reward: float = 0.0
    satisfaction_reward: float = 0.0
    safety_reward: float = 0.0
    
    # 权重（可调整）
    w_task: float = 0.5
    w_efficiency: float = 0.2
    w_satisfaction: float = 0.2
    w_safety: float = 0.1
    
    def total(self) -> float:
        """计算总奖励"""
        return (
            self.w_task * self.task_reward +
            self.w_efficiency * self.efficiency_reward +
            self.w_satisfaction * self.satisfaction_reward +
            self.w_safety * self.safety_reward
        )
    
    def to_dict(self) -> Dict[str, float]:
        """转为字典"""
        return {
            "task_reward": self.task_reward,
            "efficiency_reward": self.efficiency_reward,
            "satisfaction_reward": self.satisfaction_reward,
            "safety_reward": self.safety_reward,
            "total_reward": self.total(),
        }
    
    def __repr__(self) -> str:
        return (
            f"RewardComponents(task={self.task_reward:.2f}, "
            f"efficiency={self.efficiency_reward:.2f}, "
            f"satisfaction={self.satisfaction_reward:.2f}, "
            f"safety={self.safety_reward:.2f}, "
            f"total={self.total():.2f})"
        )


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(
        self,
        w_task: float = 0.5,
        w_efficiency: float = 0.2,
        w_satisfaction: float = 0.2,
        w_safety: float = 0.1,
    ):
        """
        初始化奖励计算器
        
        Args:
            w_task: 任务完成权重
            w_efficiency: 效率权重
            w_satisfaction: 满意度权重
            w_safety: 安全权重
        """
        self.w_task = w_task
        self.w_efficiency = w_efficiency
        self.w_satisfaction = w_satisfaction
        self.w_safety = w_safety
        
        # 归一化权重
        total_weight = w_task + w_efficiency + w_satisfaction + w_safety
        if total_weight != 1.0:
            self.w_task /= total_weight
            self.w_efficiency /= total_weight
            self.w_satisfaction /= total_weight
            self.w_safety /= total_weight
    
    def calculate(
        self,
        user_input: str,
        agent_response: str,
        tool_calls: List[Dict[str, Any]],
        response_time: float,
        task_outcome: TaskOutcome,
        quality_metrics: Optional[Dict[str, Any]] = None,
        error_occurred: bool = False,
        shacl_validation_failed: bool = False,
    ) -> Tuple[float, RewardComponents]:
        """
        计算奖励
        
        Args:
            user_input: 用户输入
            agent_response: Agent 响应
            tool_calls: 工具调用列表
            response_time: 响应时间（秒）
            task_outcome: 任务结果
            quality_metrics: 质量指标
            error_occurred: 是否发生错误
            shacl_validation_failed: SHACL 校验是否失败
            
        Returns:
            Tuple[float, RewardComponents]: (总奖励, 奖励组件详情)
        """
        components = RewardComponents(
            w_task=self.w_task,
            w_efficiency=self.w_efficiency,
            w_satisfaction=self.w_satisfaction,
            w_safety=self.w_safety,
        )
        
        # 1. 任务完成奖励
        components.task_reward = self._calculate_task_reward(
            task_outcome, tool_calls, agent_response
        )
        
        # 2. 效率奖励
        components.efficiency_reward = self._calculate_efficiency_reward(
            tool_calls, response_time
        )
        
        # 3. 满意度奖励
        components.satisfaction_reward = self._calculate_satisfaction_reward(
            user_input, agent_response, quality_metrics
        )
        
        # 4. 安全奖励
        components.safety_reward = self._calculate_safety_reward(
            tool_calls, error_occurred, shacl_validation_failed
        )
        
        total_reward = components.total()
        
        return total_reward, components
    
    def _calculate_task_reward(
        self,
        outcome: TaskOutcome,
        tool_calls: List[Dict[str, Any]],
        agent_response: str,
    ) -> float:
        """
        计算任务完成奖励 (-5 到 +10)
        
        奖励规则：
        - 成功完成：+10
        - 部分完成：+5
        - 失败：-5
        - 中断：-3
        """
        # 基础奖励
        base_reward = {
            TaskOutcome.SUCCESS: 10.0,
            TaskOutcome.PARTIAL: 5.0,
            TaskOutcome.FAILED: -5.0,
            TaskOutcome.INTERRUPTED: -3.0,
        }.get(outcome, 0.0)
        
        # 额外奖励：成功使用了关键工具
        if outcome == TaskOutcome.SUCCESS:
            # 如果调用了订单创建工具，额外奖励
            if any("create_order" in call.get("tool", "") for call in tool_calls):
                base_reward += 2.0
            
            # 如果完成了完整购物流程（搜索->加购->下单），额外奖励
            tool_names = [call.get("tool", "") for call in tool_calls]
            if ("search_products" in str(tool_names) and 
                "add_to_cart" in str(tool_names) and
                "create_order" in str(tool_names)):
                base_reward += 3.0
        
        # 惩罚：空响应或无效响应
        if not agent_response or len(agent_response.strip()) < 10:
            base_reward -= 2.0
        
        return np.clip(base_reward, -5.0, 10.0)
    
    def _calculate_efficiency_reward(
        self,
        tool_calls: List[Dict[str, Any]],
        response_time: float,
    ) -> float:
        """
        计算效率奖励 (-2 到 +5)
        
        奖励规则：
        - 工具调用次数少：+3
        - 响应时间快：+2
        - 工具调用过多：-2
        """
        reward = 0.0
        
        # 1. 工具调用次数评分
        tool_count = len(tool_calls)
        if tool_count == 0:
            # 没有工具调用（纯对话）
            reward += 2.0
        elif tool_count <= 2:
            # 1-2次调用，非常高效
            reward += 3.0
        elif tool_count <= 4:
            # 3-4次调用，较高效
            reward += 1.0
        elif tool_count <= 6:
            # 5-6次调用，一般
            reward += 0.0
        else:
            # 超过6次，低效
            reward -= 2.0
        
        # 2. 响应时间评分
        if response_time < 2.0:
            reward += 2.0
        elif response_time < 5.0:
            reward += 1.0
        elif response_time < 10.0:
            reward += 0.0
        else:
            reward -= 1.0
        
        return np.clip(reward, -2.0, 5.0)
    
    def _calculate_satisfaction_reward(
        self,
        user_input: str,
        agent_response: str,
        quality_metrics: Optional[Dict[str, Any]],
    ) -> float:
        """
        计算满意度奖励 (0 到 +10)
        
        基于：
        - 质量分数（如果有）
        - 主动引导
        - 避免多次澄清
        """
        reward = 0.0
        
        # 1. 使用质量分数（如果有）
        if quality_metrics:
            quality_score = quality_metrics.get("quality_score", 0)
            # 质量分数 0-100，归一化到 0-6
            reward += (quality_score / 100.0) * 6.0
            
            # 2. 对话流畅度奖励
            conversation_quality = quality_metrics.get("conversation_quality", {})
            
            # 主动引导加分
            proactive_rate = conversation_quality.get("proactive_rate", 0)
            reward += proactive_rate * 2.0
            
            # 需要澄清扣分
            clarification_rate = conversation_quality.get("clarification_rate", 0)
            reward -= clarification_rate * 2.0
        else:
            # 没有质量指标，使用简单启发式
            
            # 响应长度合理（不能太短或太长）
            response_len = len(agent_response)
            if 50 <= response_len <= 500:
                reward += 3.0
            elif response_len < 20:
                reward -= 1.0
            
            # 包含推荐或建议
            if any(keyword in agent_response for keyword in ["推荐", "建议", "可以试试"]):
                reward += 2.0
            
            # 包含确认或礼貌用语
            if any(keyword in agent_response for keyword in ["好的", "明白", "谢谢", "为您"]):
                reward += 1.0
        
        return np.clip(reward, 0.0, 10.0)
    
    def _calculate_safety_reward(
        self,
        tool_calls: List[Dict[str, Any]],
        error_occurred: bool,
        shacl_validation_failed: bool,
    ) -> float:
        """
        计算安全奖励 (-10 到 +1)
        
        奖励规则：
        - 无错误：+1
        - 工具调用失败：-3
        - SHACL 校验失败：-5
        - 数据泄露风险：-10
        """
        reward = 1.0  # 默认无错误奖励
        
        # 1. 检测错误
        if error_occurred:
            reward -= 3.0
        
        # 2. SHACL 校验失败（严重）
        if shacl_validation_failed:
            reward -= 5.0
        
        # 3. 检测工具调用错误
        for call in tool_calls:
            observation = call.get("observation", "")
            if isinstance(observation, str):
                # 检测错误关键词
                if any(keyword in observation.lower() for keyword in 
                       ["error", "错误", "失败", "exception"]):
                    reward -= 1.0
        
        # 4. 检测潜在的不安全操作
        dangerous_tools = ["delete", "remove", "cancel"]
        for call in tool_calls:
            tool_name = call.get("tool", "").lower()
            if any(dangerous in tool_name for dangerous in dangerous_tools):
                # 检查是否有充分的用户确认
                # 这里简化处理，实际应检查对话历史
                reward -= 0.5
        
        return np.clip(reward, -10.0, 1.0)
    
    def calculate_episode_reward(
        self,
        step_rewards: List[Tuple[float, RewardComponents]],
        episode_success: bool,
        total_time: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算整个 episode 的累积奖励和统计信息
        
        Args:
            step_rewards: 每一步的（奖励, 组件）列表
            episode_success: episode 是否成功完成
            total_time: episode 总耗时
            
        Returns:
            Tuple[float, Dict]: (总奖励, 统计信息)
        """
        if not step_rewards:
            return 0.0, {}
        
        # 累积奖励
        total_reward = sum(reward for reward, _ in step_rewards)
        
        # 成功奖励
        if episode_success:
            total_reward += 5.0
        
        # 时间惩罚（超过5分钟）
        if total_time > 300:
            total_reward -= (total_time - 300) / 60.0
        
        # 统计信息
        components_list = [comp for _, comp in step_rewards]
        stats = {
            "total_reward": total_reward,
            "num_steps": len(step_rewards),
            "avg_reward_per_step": total_reward / len(step_rewards),
            "avg_task_reward": np.mean([c.task_reward for c in components_list]),
            "avg_efficiency_reward": np.mean([c.efficiency_reward for c in components_list]),
            "avg_satisfaction_reward": np.mean([c.satisfaction_reward for c in components_list]),
            "avg_safety_reward": np.mean([c.safety_reward for c in components_list]),
            "episode_success": episode_success,
            "total_time": total_time,
        }
        
        return total_reward, stats

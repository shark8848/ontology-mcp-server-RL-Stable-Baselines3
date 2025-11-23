"""
Copyright (c) 2025 shark8848
MIT License

Ontology MCP Server - 电商 AI 助手系统
本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI

Author: shark8848
Repository: https://github.com/shark8848/ontology-mcp-server
"""

"""
数据分析服务 - 为图表生成提供结构化数据

支持多种图表类型：
- trend: 趋势图（折线图）
- bar: 柱状图
- pie: 饼图
- comparison: 对比图
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime, timedelta
from collections import Counter
import sys
from pathlib import Path

# 动态添加路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from ontology_mcp_server.db_service import EcommerceService
except ImportError:
    # Fallback for different import paths
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from ontology_mcp_server.db_service import EcommerceService

from .logger import get_logger

logger = get_logger(__name__)


class ChartData:
    """图表数据结构"""
    
    def __init__(
        self,
        chart_type: str,
        title: str,
        labels: List[str],
        series: List[Dict[str, Any]],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chart_type = chart_type
        self.title = title
        self.labels = labels
        self.series = series
        self.description = description
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "labels": self.labels,
            "series": self.series,
            "description": self.description,
            "metadata": self.metadata
        }


class AnalyticsService:
    """数据分析服务 - 生成图表数据"""
    
    def __init__(self, db_path: str = "data/ecommerce.db"):
        """初始化分析服务
        
        Args:
            db_path: 数据库文件路径
        """
        # 如果是相对路径，转换为相对于项目根目录的绝对路径
        if not os.path.isabs(db_path):
            # 获取项目根目录（src的父目录）
            project_root = Path(__file__).resolve().parents[2]
            db_path = str(project_root / db_path)
        self.service = EcommerceService(db_path=db_path)
    
    def get_order_trend(
        self,
        days: int = 7,
        user_id: Optional[int] = None
    ) -> ChartData:
        """获取订单趋势数据
        
        Args:
            days: 查询天数
            user_id: 用户ID（可选，为空则统计全部）
        
        Returns:
            ChartData: 趋势图数据
        """
        # 直接查询Order表
        with self.service.db.SessionLocal() as session:
            from ontology_mcp_server.models import Order
            query = session.query(Order)
            if user_id:
                query = query.filter(Order.user_id == user_id)
            orders = query.all()
        
        # 按日期分组统计
        date_counts: Dict[str, int] = {}
        date_amounts: Dict[str, float] = {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for order in orders:
            if order.created_at and order.created_at >= start_date:
                date_key = order.created_at.strftime('%Y-%m-%d')
                date_counts[date_key] = date_counts.get(date_key, 0) + 1
                date_amounts[date_key] = date_amounts.get(date_key, 0) + float(order.final_amount or 0)
        
        # 生成完整日期序列
        labels = []
        counts = []
        amounts = []
        
        current = start_date
        while current <= end_date:
            date_key = current.strftime('%Y-%m-%d')
            labels.append(current.strftime('%m-%d'))
            counts.append(date_counts.get(date_key, 0))
            amounts.append(date_amounts.get(date_key, 0))
            current += timedelta(days=1)
        
        title = f"{'用户' + str(user_id) + '的' if user_id else '全站'}订单趋势（近{days}天）"
        description = f"统计周期: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}"
        
        return ChartData(
            chart_type="trend",
            title=title,
            labels=labels,
            series=[
                {"name": "订单数", "data": counts, "color": "#1f77b4"},
                {"name": "订单金额", "data": amounts, "color": "#ff7f0e"}
            ],
            description=description,
            metadata={"days": days, "user_id": user_id}
        )
    
    def get_category_distribution(
        self,
        user_id: Optional[int] = None
    ) -> ChartData:
        """获取商品分类占比（饼图）
        
        Args:
            user_id: 用户ID（可选，统计该用户的订单）
        
        Returns:
            ChartData: 饼图数据
        """
        orders = self.service.orders.list_orders(limit=1000)
        
        # 过滤用户
        if user_id:
            orders = [o for o in orders if o.user_id == user_id]
        
        # 统计各分类订单数
        category_counts: Dict[str, int] = {}
        category_amounts: Dict[str, float] = {}
        
        for order in orders:
            items = order.order_items or []
            for item in items:
                if item.product:
                    category = item.product.category or "未分类"
                    category_counts[category] = category_counts.get(category, 0) + item.quantity
                    category_amounts[category] = category_amounts.get(category, 0) + float(item.subtotal or 0)
        
        # 排序（按数量）
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        labels = [cat for cat, _ in sorted_categories]
        values = [count for _, count in sorted_categories]
        
        title = f"{'用户' + str(user_id) + '的' if user_id else '全站'}商品分类占比"
        description = f"共 {len(labels)} 个分类，总计 {sum(values)} 件商品"
        
        return ChartData(
            chart_type="pie",
            title=title,
            labels=labels,
            series=[{"name": "购买数量", "data": values}],
            description=description,
            metadata={"user_id": user_id, "total_items": sum(values)}
        )
    
    def get_user_level_stats(self) -> ChartData:
        """获取用户等级分布（柱状图）
        
        Returns:
            ChartData: 柱状图数据
        """
        # 直接查询User表
        with self.service.db.SessionLocal() as session:
            from ontology_mcp_server.models import User
            users = session.query(User).all()
        
        # 统计各等级用户数
        level_counts = Counter(u.user_level for u in users)
        
        # 按等级排序
        level_order = ["Regular", "VIP", "SVIP"]
        labels = []
        counts = []
        
        for level in level_order:
            if level in level_counts:
                labels.append(level)
                counts.append(level_counts[level])
        
        # 补充其他等级
        for level, count in level_counts.items():
            if level not in level_order:
                labels.append(level)
                counts.append(count)
        
        return ChartData(
            chart_type="bar",
            title="用户等级分布",
            labels=labels,
            series=[{"name": "用户数", "data": counts, "color": "#2ca02c"}],
            description=f"共 {len(users)} 名用户，{len(labels)} 个等级",
            metadata={"total_users": len(users)}
        )
    
    def get_product_sales_ranking(
        self,
        top_n: int = 10,
        category: Optional[str] = None
    ) -> ChartData:
        """获取商品销量排行（柱状图）
        
        Args:
            top_n: 返回前N名
            category: 商品分类过滤（可选）
        
        Returns:
            ChartData: 柱状图数据
        """
        orders = self.service.orders.list_orders(limit=1000)
        
        # 统计商品销量
        product_sales: Dict[int, Dict[str, Any]] = {}
        
        for order in orders:
            items = order.order_items or []
            for item in items:
                if item.product:
                    # 分类过滤
                    if category and item.product.category != category:
                        continue
                    
                    pid = item.product_id
                    if pid not in product_sales:
                        product_sales[pid] = {
                            "name": item.product_name,
                            "quantity": 0,
                            "amount": 0
                        }
                    product_sales[pid]["quantity"] += item.quantity
                    product_sales[pid]["amount"] += float(item.subtotal or 0)
        
        # 排序并取前N名
        sorted_products = sorted(
            product_sales.items(),
            key=lambda x: x[1]["quantity"],
            reverse=True
        )[:top_n]
        
        labels = [info["name"] for _, info in sorted_products]
        quantities = [info["quantity"] for _, info in sorted_products]
        amounts = [info["amount"] for _, info in sorted_products]
        
        title = f"{'[' + category + '] ' if category else ''}商品销量TOP{top_n}"
        description = f"统计{len(product_sales)}款商品的销量数据"
        
        return ChartData(
            chart_type="bar",
            title=title,
            labels=labels,
            series=[
                {"name": "销量", "data": quantities, "color": "#d62728"},
                {"name": "销售额", "data": amounts, "color": "#9467bd"}
            ],
            description=description,
            metadata={"top_n": top_n, "category": category}
        )
    
    def get_user_spending_comparison(
        self,
        user_ids: List[int]
    ) -> ChartData:
        """获取用户消费对比（对比图）
        
        Args:
            user_ids: 用户ID列表
        
        Returns:
            ChartData: 对比图数据
        """
        users_data = []
        
        for uid in user_ids[:10]:  # 最多对比10个用户
            user = self.service.users.get_user_by_id(uid)
            if user:
                orders = self.service.orders.list_user_orders(uid, limit=1000)
                total_spent = sum(float(o.final_amount or 0) for o in orders)
                users_data.append({
                    "user_id": uid,
                    "username": user.username,
                    "total_spent": total_spent,
                    "order_count": len(orders),
                    "level": user.user_level
                })
        
        labels = [d["username"] for d in users_data]
        spent = [d["total_spent"] for d in users_data]
        orders = [d["order_count"] for d in users_data]
        
        return ChartData(
            chart_type="comparison",
            title=f"用户消费对比（{len(users_data)}人）",
            labels=labels,
            series=[
                {"name": "累计消费", "data": spent, "color": "#e377c2"},
                {"name": "订单数", "data": orders, "color": "#7f7f7f"}
            ],
            description=f"对比{len(users_data)}个用户的消费情况",
            metadata={"user_ids": user_ids}
        )


def get_chart_data(
    chart_type: str,
    **kwargs
) -> Dict[str, Any]:
    """生成图表数据的统一入口
    
    Args:
        chart_type: 图表类型（trend/pie/bar/comparison）
        **kwargs: 其他参数
    
    Returns:
        Dict: 图表数据字典
    """
    service = AnalyticsService()

    safe_params = {
        key: kwargs.get(key)
        for key in ("days", "user_id", "top_n", "category", "user_ids")
        if key in kwargs
    }

    # Avoid logging overly long lists
    if "user_ids" in safe_params and isinstance(safe_params["user_ids"], list):
        user_ids = safe_params["user_ids"]
        safe_params["user_ids"] = user_ids[:10]
    logger.info(
        "收到图表生成请求: chart_type=%s params=%s",
        chart_type,
        json.dumps(safe_params, ensure_ascii=False),
    )

    resolved_chart: Optional[ChartData] = None
    resolved_type = chart_type

    try:
        if chart_type in {"trend", "order_trend"}:
            days = kwargs.get("days", 7)
            user_id = kwargs.get("user_id")
            resolved_chart = service.get_order_trend(days=days, user_id=user_id)
            resolved_type = "trend"

        elif chart_type in {"pie", "category_distribution"}:
            user_id = kwargs.get("user_id")
            resolved_chart = service.get_category_distribution(user_id=user_id)
            resolved_type = "pie"

        elif chart_type in {"user_level"}:
            resolved_chart = service.get_user_level_stats()
            resolved_type = "bar_user_level"

        elif chart_type in {"bar", "sales_ranking"}:
            top_n = kwargs.get("top_n", 10)
            category = kwargs.get("category")
            resolved_chart = service.get_product_sales_ranking(top_n=top_n, category=category)
            resolved_type = "bar_sales_ranking"

        elif chart_type in {"comparison", "user_spending"}:
            user_ids = kwargs.get("user_ids", [])
            resolved_chart = service.get_user_spending_comparison(user_ids=user_ids)
            resolved_type = "comparison"

        else:
            # 默认返回订单趋势
            resolved_chart = service.get_order_trend(days=kwargs.get("days", 7))
            resolved_type = "trend_default"

        if not resolved_chart:
            raise ValueError(f"未能生成图表: chart_type={chart_type}")

        chart_dict = resolved_chart.to_dict()
        series_summary = [
            {
                "name": s.get("name"),
                "points": len(s.get("data", [])) if isinstance(s.get("data"), list) else 0,
            }
            for s in chart_dict.get("series", [])
        ]
        logger.info(
            "图表生成成功: resolved_type=%s labels=%d series=%s",
            resolved_type,
            len(chart_dict.get("labels", [])),
            json.dumps(series_summary, ensure_ascii=False),
        )
        return chart_dict
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "图表生成失败: chart_type=%s params=%s error=%s",
            chart_type,
            json.dumps(safe_params, ensure_ascii=False),
            str(exc),
        )
        raise

from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""
电商本体推理服务

基于 RDFLib 实现电商领域的本体推理，包括：
- 用户等级推理 (VIP/SVIP)
- 折扣计算推理
- 物流策略推理
- 退换货规则推理
"""

from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from .logger import get_logger

LOGGER = get_logger(__name__)

# 命名空间定义
EC = Namespace("http://example.org/ecommerce#")
RULE = Namespace("http://example.org/rules#")


class EcommerceOntologyService:
    """电商本体推理服务"""
    
    def __init__(self, ontology_path: str = "data/ontology_ecommerce.ttl",
                 rules_path: str = "data/ontology_rules.ttl"):
        """初始化本体推理服务
        
        Args:
            ontology_path: 电商本体文件路径
            rules_path: 业务规则文件路径
        """
        self.ontology_graph = Graph()
        self.rules_graph = Graph()
        
        # 加载本体
        self._load_ontology(ontology_path)
        self._load_rules(rules_path)
        
        LOGGER.info("电商本体推理服务已初始化")
    
    def _load_ontology(self, path: str):
        """加载电商本体"""
        ontology_file = Path(path)
        if ontology_file.exists():
            try:
                self.ontology_graph.parse(ontology_file, format="turtle")
                LOGGER.info(f"已加载电商本体: {path}, 三元组数: {len(self.ontology_graph)}")
            except Exception as e:
                LOGGER.error(f"加载电商本体失败: {e}")
        else:
            LOGGER.warning(f"电商本体文件不存在: {path}")
    
    def _load_rules(self, path: str):
        """加载业务规则"""
        rules_file = Path(path)
        if rules_file.exists():
            try:
                self.rules_graph.parse(rules_file, format="turtle")
                LOGGER.info(f"已加载业务规则: {path}, 三元组数: {len(self.rules_graph)}")
            except Exception as e:
                LOGGER.error(f"加载业务规则失败: {e}")
        else:
            LOGGER.warning(f"业务规则文件不存在: {path}")
    
    # ============================================
    # 用户等级推理
    # ============================================
    
    def infer_user_level(self, total_spent: Decimal) -> str:
        """根据累计消费推理用户等级
        
        Args:
            total_spent: 累计消费金额
            
        Returns:
            str: 用户等级 (Regular/VIP/SVIP)
        """
        total_spent_float = float(total_spent)
        
        # 查询 SVIP 升级规则
        svip_query = """
        PREFIX rule: <http://example.org/rules#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?rule ?condition
        WHERE {
            ?rule a rule:UserLevelRule ;
                  rdfs:label ?label ;
                  rule:condition ?condition .
            FILTER(CONTAINS(?label, "SVIP"))
        }
        """
        
        # 查询 VIP 升级规则
        vip_query = """
        PREFIX rule: <http://example.org/rules#>
        
        SELECT ?rule ?condition
        WHERE {
            ?rule a rule:UserLevelRule ;
                  rdfs:label ?label ;
                  rule:condition ?condition .
            FILTER(CONTAINS(?label, "VIP") && !CONTAINS(?label, "SVIP"))
        }
        """
        
        # 简化推理：直接使用阈值判断
        if total_spent_float >= 10000:
            LOGGER.info(f"用户等级推理: 累计消费 {total_spent} >= 10000 -> SVIP")
            return "SVIP"
        elif total_spent_float >= 5000:
            LOGGER.info(f"用户等级推理: 累计消费 {total_spent} >= 5000 -> VIP")
            return "VIP"
        else:
            LOGGER.info(f"用户等级推理: 累计消费 {total_spent} < 5000 -> Regular")
            return "Regular"
    
    # ============================================
    # 折扣推理
    # ============================================
    
    def infer_discount(self, user_level: str, order_amount: Decimal,
                      is_first_order: bool = False) -> Dict[str, Any]:
        """推理订单折扣
        
        Args:
            user_level: 用户等级 (Regular/VIP/SVIP)
            order_amount: 订单金额
            is_first_order: 是否首单
            
        Returns:
            Dict: {
                'discount_type': str,  # 折扣类型
                'discount_rate': Decimal,  # 折扣率
                'discount_amount': Decimal,  # 折扣金额
                'final_amount': Decimal,  # 最终金额
                'reason': str  # 推理理由
            }
        """
        order_amount_float = float(order_amount)
        applicable_discounts = []
        
        # 1. 会员等级折扣
        if user_level == "SVIP":
            applicable_discounts.append({
                'type': 'SVIP会员折扣',
                'rate': Decimal('0.90'),
                'priority': 40,
                'rule': 'SVIPDiscountRule'
            })
        elif user_level == "VIP":
            applicable_discounts.append({
                'type': 'VIP会员折扣',
                'rate': Decimal('0.95'),
                'priority': 30,
                'rule': 'VIPDiscountRule'
            })
        
        # 2. 批量折扣
        if order_amount_float >= 10000:
            applicable_discounts.append({
                'type': '批量折扣(满10000)',
                'rate': Decimal('0.90'),
                'priority': 25,
                'rule': 'VolumeDiscount10kRule'
            })
        elif order_amount_float >= 5000:
            applicable_discounts.append({
                'type': '批量折扣(满5000)',
                'rate': Decimal('0.95'),
                'priority': 15,
                'rule': 'VolumeDiscount5kRule'
            })
        
        # 3. 首单折扣
        if is_first_order:
            applicable_discounts.append({
                'type': '首单折扣',
                'rate': Decimal('0.98'),
                'priority': 5,
                'rule': 'FirstOrderDiscountRule'
            })
        
        # 根据策略选择最优折扣：会员折扣与批量折扣不叠加，取优惠力度大的
        if not applicable_discounts:
            return {
                'discount_type': '无折扣',
                'discount_rate': Decimal('1.0'),
                'discount_amount': Decimal('0'),
                'final_amount': order_amount,
                'reason': '不满足任何折扣条件'
            }
        
        # 选择折扣率最低(优惠最大)的
        best_discount = min(applicable_discounts, key=lambda x: x['rate'])
        
        discount_rate = best_discount['rate']
        final_amount = order_amount * discount_rate
        discount_amount = order_amount - final_amount
        
        reason = f"应用{best_discount['type']}，折扣率{float(discount_rate):.2f}"
        if len(applicable_discounts) > 1:
            other_discounts = [d['type'] for d in applicable_discounts if d != best_discount]
            reason += f"（同时满足{', '.join(other_discounts)}，已自动选择最优）"
        
        LOGGER.info(f"折扣推理: {reason}")
        
        return {
            'discount_type': best_discount['type'],
            'discount_rate': discount_rate,
            'discount_amount': discount_amount,
            'final_amount': final_amount,
            'reason': reason,
            'rule_applied': best_discount['rule']
        }
    
    # ============================================
    # 物流推理
    # ============================================
    
    def infer_shipping(self, user_level: str, order_amount: Decimal,
                      is_remote_area: bool = False) -> Dict[str, Any]:
        """推理物流策略
        
        Args:
            user_level: 用户等级
            order_amount: 订单金额
            is_remote_area: 是否偏远地区
            
        Returns:
            Dict: {
                'shipping_cost': Decimal,  # 运费
                'shipping_type': str,  # 配送类型
                'free_shipping': bool,  # 是否包邮
                'reason': str  # 推理理由
            }
        """
        order_amount_float = float(order_amount)
        shipping_cost = Decimal('0')
        shipping_type = 'Standard'
        free_shipping = False
        reasons = []
        
        # 1. SVIP次日达
        if user_level == "SVIP":
            shipping_type = "NextDayDelivery"
            free_shipping = True
            reasons.append("SVIP用户享受免费次日达服务")
        
        # 2. VIP包邮
        elif user_level == "VIP":
            free_shipping = True
            reasons.append("VIP用户享受包邮服务")
        
        # 3. 满500包邮
        elif order_amount_float >= 500:
            free_shipping = True
            reasons.append("订单金额满500元包邮")
        
        # 4. 普通用户收取运费
        else:
            shipping_cost = Decimal('15')
            reasons.append("普通用户订单不满500元，收取15元运费")
        
        # 5. 偏远地区加收运费
        if is_remote_area and not (user_level == "SVIP"):
            shipping_cost += Decimal('30')
            free_shipping = False
            reasons.append("偏远地区加收30元运费")
        
        reason = "; ".join(reasons)
        LOGGER.info(f"物流推理: {reason}")
        
        return {
            'shipping_cost': shipping_cost,
            'shipping_type': shipping_type,
            'free_shipping': free_shipping,
            'reason': reason,
            'estimated_days': 1 if shipping_type == "NextDayDelivery" else 3
        }
    
    # ============================================
    # 退换货规则推理
    # ============================================
    
    def infer_return_policy(self, user_level: str, product_category: str,
                           is_activated: bool = False,
                           packaging_intact: bool = True) -> Dict[str, Any]:
        """推理退换货政策
        
        Args:
            user_level: 用户等级
            product_category: 商品分类 (手机/配件/服务)
            is_activated: 电子产品是否已激活
            packaging_intact: 商品包装是否完好
            
        Returns:
            Dict: {
                'returnable': bool,  # 是否可退货
                'return_period_days': int,  # 退货期限(天)
                'conditions': List[str],  # 退货条件
                'reason': str  # 推理理由
            }
        """
        returnable = True
        return_period_days = 7
        conditions = []
        reasons = []
        
        # 1. 服务类商品不可退
        if product_category == "服务":
            returnable = False
            return_period_days = 0
            reasons.append("服务类商品(如AppleCare+)不可退货")
            
            return {
                'returnable': returnable,
                'return_period_days': return_period_days,
                'conditions': conditions,
                'reason': "; ".join(reasons)
            }
        
        # 2. 根据用户等级确定退货期限
        if user_level in ["VIP", "SVIP"]:
            return_period_days = 15
            reasons.append("VIP/SVIP用户享受15天无理由退货")
        else:
            return_period_days = 7
            reasons.append("普通用户享受7天无理由退货")
        
        # 3. 电子产品特殊条件
        if product_category == "手机":
            if is_activated:
                conditions.append("电子产品已激活，需符合质量问题才能退货")
                reasons.append("电子产品已激活，仅限质量问题退货")
            else:
                conditions.append("电子产品未激活，可无理由退货")
        
        # 4. 配件类商品条件
        elif product_category == "配件":
            if packaging_intact:
                conditions.append("包装需保持完好")
                reasons.append("配件类商品包装完好可退货")
            else:
                returnable = False
                return_period_days = 0
                reasons.append("配件类商品包装已拆封，不可退货")
        
        reason = "; ".join(reasons)
        LOGGER.info(f"退换货推理: {reason}")
        
        return {
            'returnable': returnable,
            'return_period_days': return_period_days,
            'conditions': conditions,
            'reason': reason
        }

    # ============================================
    # 取消规则推理
    # ============================================

    def infer_cancellation_policy(
        self,
        order_status: str,
        hours_since_created: float,
        has_shipment: bool = False,
    ) -> Dict[str, Any]:
        """推理订单是否允许取消及其理由."""

        status = (order_status or "pending").lower()
        hours = max(float(hours_since_created or 0.0), 0.0)

        policy = {
            "status": status,
            "hours_since_order": hours,
            "allowed": False,
            "rule": "DefaultCancellationRule",
            "deadline_hours": None,
            "reason": "订单状态不支持取消",
        }

        if status in {"shipped", "delivered"} or has_shipment:
            policy.update(
                {
                    "allowed": False,
                    "rule": "ShippedCancellationBlockRule",
                    "deadline_hours": 0,
                    "reason": "订单已发货，需走退货流程",
                }
            )
            LOGGER.info("取消规则推理: %s", policy["reason"])
            return policy

        if status == "pending":
            deadline = 24.0
            policy["deadline_hours"] = deadline
            if hours <= deadline:
                policy.update(
                    {
                        "allowed": True,
                        "rule": "Pending24hCancellationRule",
                        "reason": "待支付订单24小时内可直接取消",
                    }
                )
            else:
                policy["reason"] = "已超过24小时待支付取消窗口"
            LOGGER.info("取消规则推理: %s", policy["reason"])
            return policy

        if status == "paid":
            deadline = 12.0
            policy["deadline_hours"] = deadline
            if has_shipment:
                policy.update(
                    {
                        "allowed": False,
                        "rule": "ShippedCancellationBlockRule",
                        "reason": "已生成物流单，无法取消",
                    }
                )
            elif hours <= deadline:
                policy.update(
                    {
                        "allowed": True,
                        "rule": "Paid12hCancellationRule",
                        "reason": "已支付12小时内且未发货，可人工审核后取消",
                    }
                )
            else:
                policy["reason"] = "已超过12小时支付取消窗口"
            LOGGER.info("取消规则推理: %s", policy["reason"])
            return policy

        if status in {"cancelled", "returned"}:
            policy.update(
                {
                    "allowed": False,
                    "rule": "AlreadyTerminatedCancellationRule",
                    "reason": "订单已终止，无需再次取消",
                }
            )
            LOGGER.info("取消规则推理: %s", policy["reason"])
            return policy

        policy["reason"] = "当前状态不支持自动取消，请联系客服"
        LOGGER.info("取消规则推理: %s", policy["reason"])
        return policy
    
    # ============================================
    # 综合推理
    # ============================================
    
    def infer_order_details(self, user_data: Dict[str, Any],
                           order_data: Dict[str, Any]) -> Dict[str, Any]:
        """综合推理订单详情（折扣+物流）
        
        Args:
            user_data: 用户数据 {
                'user_id': int,
                'user_level': str,
                'total_spent': Decimal,
                'order_count': int
            }
            order_data: 订单数据 {
                'order_amount': Decimal,
                'products': List[Dict],
                'shipping_address': str
            }
            
        Returns:
            Dict: 完整的订单推理结果
        """
        # 1. 推理用户等级（如果需要升级）
        current_level = user_data.get('user_level', 'Regular')
        total_spent = Decimal(str(user_data.get('total_spent', 0)))
        inferred_level = self.infer_user_level(total_spent)
        
        if inferred_level != current_level:
            LOGGER.info(f"检测到用户等级变化: {current_level} -> {inferred_level}")
        
        # 2. 推理折扣
        order_amount = Decimal(str(order_data['order_amount']))
        is_first_order = user_data.get('order_count', 0) == 0
        
        discount_info = self.infer_discount(
            user_level=inferred_level,
            order_amount=order_amount,
            is_first_order=is_first_order
        )
        
        # 3. 推理物流
        # 简单判断是否偏远地区（实际应该查询地址库）
        shipping_address = order_data.get('shipping_address', '')
        is_remote = any(area in shipping_address for area in ['西藏', '新疆', '内蒙古'])
        
        shipping_info = self.infer_shipping(
            user_level=inferred_level,
            order_amount=discount_info['final_amount'],  # 用折后金额
            is_remote_area=is_remote
        )
        
        # 4. 汇总结果
        return {
            'user_level_inference': {
                'original_level': current_level,
                'inferred_level': inferred_level,
                'should_upgrade': inferred_level != current_level
            },
            'discount_inference': discount_info,
            'shipping_inference': shipping_info,
            'final_summary': {
                'original_amount': order_amount,
                'discount_amount': discount_info['discount_amount'],
                'subtotal': discount_info['final_amount'],
                'shipping_cost': shipping_info['shipping_cost'],
                'total_payable': discount_info['final_amount'] + shipping_info['shipping_cost']
            }
        }
    
    # ============================================
    # 查询本体
    # ============================================
    
    def query_rules_by_type(self, rule_type: str) -> List[Dict[str, Any]]:
        """查询特定类型的业务规则
        
        Args:
            rule_type: 规则类型 (UserLevelRule/DiscountRule/ShippingRule/ReturnRule)
            
        Returns:
            List[Dict]: 规则列表
        """
        query = f"""
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?rule ?label ?condition ?action ?priority
        WHERE {{
            ?rule a rule:{rule_type} ;
                  rdfs:label ?label ;
                  rule:condition ?condition ;
                  rule:action ?action .
            OPTIONAL {{ ?rule rule:priority ?priority }}
        }}
        ORDER BY DESC(?priority)
        """
        
        results = []
        for row in self.rules_graph.query(query):
            results.append({
                'rule': str(row.rule),
                'label': str(row.label),
                'condition': str(row.condition),
                'action': str(row.action),
                'priority': int(row.priority) if row.priority else 0
            })
        
        return results
    
    def get_all_discount_rules(self) -> List[Dict[str, Any]]:
        """获取所有折扣规则"""
        return self.query_rules_by_type("DiscountRule")
    
    def get_all_shipping_rules(self) -> List[Dict[str, Any]]:
        """获取所有物流规则"""
        return self.query_rules_by_type("ShippingRule")
    
    def get_all_return_rules(self) -> List[Dict[str, Any]]:
        """获取所有退换货规则"""
        return self.query_rules_by_type("ReturnRule")

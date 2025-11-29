from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""高层电商服务：聚合数据库操作与本体推理逻辑"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from .db_service import (
    DatabaseService,
    EcommerceService as DatabaseLayer,
    UserService,
    ProductService,
    CartService,
    OrderService,
    PaymentService,
    ShipmentService,
)
from .ecommerce_ontology import EcommerceOntologyService
from .logger import get_logger
from .shacl_service import validate_order
from .models import (
    Order,
    Product,
    Return,
    Review,
    SupportMessage,
    SupportTicket,
)

LOGGER = get_logger(__name__)
class CommerceService:
    _ORDER_NO_MIN_DIGITS = 15
    """业务协调层，将数据库原子操作与本体推理组合使用。"""

    def __init__(self, db_path: str = "data/ecommerce.db") -> None:
        self._db_layer = DatabaseLayer(db_path)
        self.database: DatabaseService = self._db_layer.db
        self.users: UserService = self._db_layer.users
        self.products: ProductService = self._db_layer.products
        self.cart: CartService = self._db_layer.cart
        self.orders: OrderService = self._db_layer.orders
        self.payments: PaymentService = self._db_layer.payments
        self.shipments: ShipmentService = self._db_layer.shipments
        self.ontology = EcommerceOntologyService()
        LOGGER.info("CommerceService 初始化完成")

    def _resolve_order_entity(self, identifier: Any) -> Order:
        """根据多种编号格式获取订单对象。"""

        if isinstance(identifier, Order):
            return identifier

        raw = "" if identifier is None else str(identifier).strip()
        if not raw:
            raise ValueError("order_id 不能为空")

        normalized = raw.upper()
        digits = "".join(ch for ch in normalized if ch.isdigit())

        order: Optional[Order] = None
        order_id_candidate: Optional[int] = None
        if digits:
            try:
                order_id_candidate = int(digits)
            except ValueError:
                order_id_candidate = None

        if order_id_candidate is not None:
            order = self.orders.get_order_by_id(order_id_candidate)
            if order:
                return order

        order_no_candidate: Optional[str] = None
        if normalized.startswith("ORD"):
            order_no_candidate = normalized
        elif digits and len(digits) >= self._ORDER_NO_MIN_DIGITS:
            order_no_candidate = f"ORD{digits}"

        if order_no_candidate:
            order = self.orders.get_order_by_no(order_no_candidate)
            if order:
                return order

        raise ValueError("订单不存在")

    # ------------------------------------------------------------------
    # 通用与辅助方法
    # ------------------------------------------------------------------
    def init_database(self) -> None:
        self._db_layer.init_database()

    def _count_user_orders(self, user_id: int) -> int:
        with self.database.get_session() as session:
            return int(
                session.query(func.count(Order.order_id))
                .filter(Order.user_id == user_id)
                .scalar()
                or 0
            )

    def _build_order_rdf(
        self,
        user_id: int,
        order_amount: Decimal,
        discount_rate: float,
        prepared_items: List[Dict[str, Any]]
    ) -> str:
        """构建订单的 RDF/Turtle 数据用于 SHACL 校验"""
        lines = [
            "@prefix : <http://example.com/commerce#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
            f":order_{user_id}_temp a :Order ;",
            f"    :hasCustomer :user_{user_id} ;",
            f"    :totalAmount \"{order_amount}\"^^xsd:decimal ;",
            f"    :discountRate \"{discount_rate}\"^^xsd:decimal ;",
        ]
        
        # 添加订单项
        for idx, item in enumerate(prepared_items, 1):
            lines.append(f"    :hasItem :item_{idx} ;")
        
        # 移除最后的分号，添加句点
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        lines.append("")
        
        # 定义用户
        lines.append(f":user_{user_id} a :Customer .")
        lines.append("")
        
        # 定义订单项和商品
        for idx, item in enumerate(prepared_items, 1):
            lines.extend([
                f":item_{idx} a :OrderItem ;",
                f"    :hasProduct :product_{item['product_id']} .",
                "",
                f":product_{item['product_id']} a :Product .",
                ""
            ])
        
        return "\n".join(lines)

    def _build_cancellation_log_entry(
        self,
        order: Order,
        *,
        hours_since_created: float,
        has_shipment: bool,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建取消订单推理的执行日志条目。"""
        safe_policy = dict(policy)
        return {
            "step_type": "ontology_inference",
            "content": {
                "inference_type": "cancellation_policy",
                "order_id": order.order_id,
                "order_no": getattr(order, "order_no", None),
                "order_status": order.order_status,
                "hours_since_created": round(hours_since_created, 2),
                "has_shipment": has_shipment,
                "policy": safe_policy,
            },
            "metadata": {
                "source": "CommerceService.cancel_order",
                "ontology_method": "infer_cancellation_policy",
            },
        }

    # ------------------------------------------------------------------
    # 用户相关
    # ------------------------------------------------------------------
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        user = self.users.get_user_by_id(user_id)
        if not user:
            raise ValueError("用户不存在")
        orders = self.orders.get_user_orders(user_id, limit=20)
        total_spent = Decimal(user.total_spent or 0)
        inferred_level = self.ontology.infer_user_level(total_spent)
        return {
            "user": user.to_dict(),
            "orders": [order.to_dict() for order in orders],
            "inferred_level": inferred_level,
            "should_upgrade": inferred_level != user.user_level,
        }

    def get_user_orders(self, user_id: int, status: Optional[str] = None) -> Dict[str, Any]:
        orders = self.orders.get_user_orders(user_id, status=status, limit=50)
        return {
            "user_id": user_id,
            "orders": [order.to_dict() for order in orders],
        }

    # ------------------------------------------------------------------
    # 商品相关
    # ------------------------------------------------------------------
    def search_products(
        self,
        keyword: Optional[str] = None,
        *,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        available_only: bool = True,
        limit: int = 20,
        enable_category_fallback: bool = True,
    ) -> Dict[str, Any]:
        """搜索商品 (支持类别回退提升召回率)"""
        # 步骤1: 尝试关键词检索
        products = self.products.search_products(
            keyword=keyword,
            category=category,
            brand=brand,
            min_price=Decimal(str(min_price)) if min_price is not None else None,
            max_price=Decimal(str(max_price)) if max_price is not None else None,
            available_only=available_only,
            limit=limit,
        )
        
        # 步骤2: 如果结果少且启用了回退,尝试类别检索
        if enable_category_fallback and len(products) < 5 and keyword and not category:
            # 导入类别映射表
            try:
                from src.agent.query_rewriter import QueryRewriter
                CATEGORY_SEARCH_MAP = QueryRewriter.CATEGORY_SEARCH_MAP
            except ImportError:
                CATEGORY_SEARCH_MAP = {}
            
            if keyword in CATEGORY_SEARCH_MAP:
                target_categories = CATEGORY_SEARCH_MAP[keyword]
                LOGGER.info(f"关键词 '{keyword}' 匹配较少({len(products)}个),回退到类别检索: {target_categories}")
                
                for target_cat in target_categories:
                    fallback_products = self.products.search_products(
                        keyword=None,  # 清空关键词,只用类别
                        category=target_cat,
                        brand=brand,
                        min_price=Decimal(str(min_price)) if min_price is not None else None,
                        max_price=Decimal(str(max_price)) if max_price is not None else None,
                        available_only=available_only,
                        limit=max(limit // len(target_categories), 5),  # 每个类别至少5个
                    )
                    products.extend(fallback_products)
                
                # 去重 (based on product_id)
                seen = set()
                unique_products = []
                for p in products:
                    if p.product_id not in seen:
                        seen.add(p.product_id)
                        unique_products.append(p)
                products = unique_products[:limit]
                
                LOGGER.info(f"类别回退后共找到 {len(products)} 个商品")
        
        return {
            "total": len(products),
            "items": [product.to_dict() for product in products],
        }

    def get_product_detail(self, product_id: int) -> Dict[str, Any]:
        product = self.products.get_product_by_id(product_id)
        if not product:
            raise ValueError("商品不存在")
        return product.to_dict()

    def check_stock(self, product_id: int, quantity: int) -> Dict[str, Any]:
        product = self.products.get_product_by_id(product_id)
        if not product:
            raise ValueError("商品不存在")
        available = self.products.check_stock(product_id, quantity)
        return {
            "product_id": product_id,
            "requested_quantity": quantity,
            "available": available,
            "stock_quantity": product.stock_quantity,
        }

    def get_product_recommendations(
        self,
        *,
        product_id: Optional[int] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        resolved_category = category
        excluded_id = None
        if product_id is not None:
            product = self.products.get_product_by_id(product_id)
            if not product:
                raise ValueError("商品不存在")
            resolved_category = product.category
            excluded_id = product.product_id
        with self.database.get_session() as session:
            query = session.query(Product).filter(Product.is_available == True)  # noqa: E712
            if resolved_category:
                query = query.filter(Product.category == resolved_category)
            if excluded_id is not None:
                query = query.filter(Product.product_id != excluded_id)
            recommendations = (
                query.order_by(Product.created_at.desc()).limit(limit).all()
            )
        return {
            "category": resolved_category,
            "items": [product.to_dict() for product in recommendations],
        }

    def get_product_reviews(self, product_id: int, limit: int = 10) -> Dict[str, Any]:
        with self.database.get_session() as session:
            reviews = (
                session.query(Review)
                .filter(Review.product_id == product_id)
                .order_by(Review.created_at.desc())
                .limit(limit)
                .all()
            )
        return {
            "product_id": product_id,
            "reviews": [review.to_dict() for review in reviews],
        }

    # ------------------------------------------------------------------
    # 购物车相关
    # ------------------------------------------------------------------
    def add_to_cart(self, user_id: int, product_id: int, quantity: int = 1) -> Dict[str, Any]:
        return self.cart.add_to_cart(user_id, product_id, quantity)

    def view_cart(self, user_id: int) -> Dict[str, Any]:
        items = self.cart.get_cart(user_id)
        return {
            "user_id": user_id,
            "items": items,
        }

    def remove_from_cart(self, user_id: int, product_id: int) -> Dict[str, Any]:
        removed = self.cart.remove_from_cart(user_id, product_id)
        return {"removed": removed}

    def clear_cart(self, user_id: int) -> Dict[str, Any]:
        cleared = self.cart.clear_cart(user_id)
        return {"cleared": cleared}

    # ------------------------------------------------------------------
    # 订单与支付
    # ------------------------------------------------------------------
    def create_order(
        self,
        user_id: int,
        items: List[Dict[str, Any]],
        shipping_address: str,
        contact_phone: str,
    ) -> Dict[str, Any]:
        if not items:
            raise ValueError("订单项不能为空")
        user = self.users.get_user_by_id(user_id)
        if not user:
            raise ValueError("用户不存在")

        prepared_items: List[Dict[str, Any]] = []
        products_summary: List[Dict[str, Any]] = []
        order_amount = Decimal("0")

        for entry in items:
            product_id = int(entry.get("product_id"))
            quantity = int(entry.get("quantity", 1))
            if quantity <= 0:
                raise ValueError("商品数量必须大于0")
            product = self.products.get_product_by_id(product_id)
            if not product:
                raise ValueError(f"商品不存在: {product_id}")
            if not self.products.check_stock(product_id, quantity):
                raise ValueError(f"库存不足: {product.product_name}")
            
            # 🔧 修复价格处理：确保价格有效
            provided_price = entry.get("unit_price")
            if provided_price is not None:
                unit_price = Decimal(str(provided_price))
            elif product.price is not None:
                unit_price = Decimal(str(product.price))
            else:
                raise ValueError(f"商品价格无效: {product.product_name} (product_id={product_id})")
            
            prepared_items.append(
                {
                    "product_id": product_id,
                    "product_name": product.product_name,
                    "quantity": quantity,
                    "unit_price": unit_price,
                }
            )
            products_summary.append(
                {
                    "product_id": product_id,
                    "category": product.category,
                    "quantity": quantity,
                    "unit_price": unit_price,
                }
            )
            order_amount += unit_price * quantity

        user_data = {
            "user_id": user.user_id,
            "user_level": user.user_level,
            "total_spent": Decimal(user.total_spent or 0),
            "order_count": self._count_user_orders(user_id),
        }
        order_data = {
            "order_amount": order_amount,
            "products": products_summary,
            "shipping_address": shipping_address,
        }
        inference = self.ontology.infer_order_details(user_data, order_data)
        discount_info = inference["discount_inference"]
        shipping_info = inference["shipping_inference"]

        # ===== SHACL 校验：在创建订单前验证数据完整性 =====
        try:
            order_rdf = self._build_order_rdf(
                user_id=user_id,
                order_amount=order_amount,
                discount_rate=discount_info.get("discount_rate", 0),
                prepared_items=prepared_items
            )
            conforms, report = validate_order(order_rdf, fmt="turtle")
            if not conforms:
                LOGGER.error("订单数据 SHACL 校验失败，拒绝创建订单")
                raise ValueError(f"订单数据不符合本体约束规则: {report[:500]}")
            LOGGER.info("订单数据 SHACL 校验通过，继续创建订单")
        except Exception as exc:
            if "不符合本体约束" in str(exc):
                raise
            # 如果是其他错误（如 SHACL 服务不可用），记录警告但不阻止订单创建
            LOGGER.warning("SHACL 校验过程出错（已忽略）: %s", exc)

        order_data = self.orders.create_order(
            user_id=user_id,
            items=prepared_items,
            shipping_address=shipping_address,
            contact_phone=contact_phone,
            discount_amount=discount_info["discount_amount"],
        )

        for entry in prepared_items:
            self.products.update_stock(entry["product_id"], -entry["quantity"])
        self.users.update_total_spent(user_id, inference["final_summary"]["total_payable"])
        upgrade_info = inference["user_level_inference"]
        if upgrade_info["should_upgrade"]:
            self.users.update_user_level(user_id, upgrade_info["inferred_level"])

        return {
            "order": order_data,
            "inference": inference,
            "shipping": shipping_info,
        }

    def get_order_detail(self, order_id: int | str) -> Dict[str, Any]:
        order = self._resolve_order_entity(order_id)
        user = self.users.get_user_by_id(order.user_id)
        shipment = self.shipments.get_shipment_by_order(order.order_id)
        return {
            "order": order.to_dict(),
            "user": user.to_dict() if user else None,
            "shipment": shipment.to_dict() if shipment else None,
        }

    def cancel_order(self, order_id: int | str) -> Dict[str, Any]:
        order = self._resolve_order_entity(order_id)
        created_at = order.created_at or datetime.now()
        hours_since_order = (datetime.now() - created_at).total_seconds() / 3600
        shipment = self.shipments.get_shipment_by_order(order.order_id)

        policy = self.ontology.infer_cancellation_policy(
            order_status=order.order_status,
            hours_since_created=hours_since_order,
            has_shipment=shipment is not None,
        )

        if not policy.get("allowed"):
            LOGGER.info(
                "取消订单被拒绝: order_id=%s status=%s reason=%s",
                order.order_id,
                order.order_status,
                policy.get("reason"),
            )
            log_entry = self._build_cancellation_log_entry(
                order,
                hours_since_created=hours_since_order,
                has_shipment=shipment is not None,
                policy=policy,
            )
            return {
                "cancelled": False,
                "allowed": False,
                "policy": policy,
                "_execution_log": [log_entry],
            }

        cancelled = self.orders.cancel_order(order.order_id)
        policy["allowed"] = bool(cancelled)
        if not cancelled:
            policy["reason"] = "系统未能更新订单状态，可能已取消"
        log_entry = self._build_cancellation_log_entry(
            order,
            hours_since_created=hours_since_order,
            has_shipment=shipment is not None,
            policy=policy,
        )
        return {
            "cancelled": bool(cancelled),
            "allowed": bool(cancelled),
            "policy": policy,
            "_execution_log": [log_entry],
        }

    def get_user_orders_summary(self, user_id: int) -> Dict[str, Any]:
        return self.get_user_orders(user_id)

    def process_payment(self, order_id: int | str, payment_method: str, amount: Decimal) -> Dict[str, Any]:
        order = self._resolve_order_entity(order_id)
        payment = self.payments.create_payment(order.order_id, payment_method, amount)
        self.orders.update_payment_status(order.order_id, "paid")
        return payment.to_dict()

    # ------------------------------------------------------------------
    # 物流相关
    # ------------------------------------------------------------------
    def track_shipment(self, tracking_no: str) -> Dict[str, Any]:
        shipment = self.shipments.get_shipment_by_tracking(tracking_no)
        if not shipment:
            raise ValueError("未找到物流信息")
        return shipment.to_dict()

    def get_shipment_status(self, order_id: int | str) -> Dict[str, Any]:
        order = self._resolve_order_entity(order_id)
        shipment = self.shipments.get_shipment_by_order(order.order_id)
        if not shipment:
            raise ValueError("未找到对应订单的物流信息")
        return shipment.to_dict()

    # ------------------------------------------------------------------
    # 客服与退换货
    # ------------------------------------------------------------------
    def create_support_ticket(
        self,
        user_id: int,
        subject: str,
        description: str,
        *,
        order_id: Optional[int | str] = None,
        category: str = "售后",
        priority: str = "medium",
        initial_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        ticket_no = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}{user_id:04d}"
        resolved_order_id: Optional[int] = None
        if order_id is not None:
            resolved_order_id = self._resolve_order_entity(order_id).order_id

        with self.database.get_session() as session:
            ticket = SupportTicket(
                ticket_no=ticket_no,
                user_id=user_id,
                order_id=resolved_order_id,
                category=category,
                priority=priority,
                status="open",
                subject=subject,
                description=description,
            )
            session.add(ticket)
            session.flush()
            if initial_message:
                message = SupportMessage(
                    ticket_id=ticket.ticket_id,
                    sender_type="customer",
                    sender_id=user_id,
                    message_content=initial_message,
                )
                session.add(message)
            session.flush()
            data = ticket.to_dict()
        return data

    def process_return(
        self,
        order_id: int | str,
        user_id: int,
        *,
        return_type: str = "return",
        reason: str = "",
        product_category: str = "手机",
        is_activated: bool = False,
    ) -> Dict[str, Any]:
        user = self.users.get_user_by_id(user_id)
        if not user:
            raise ValueError("用户不存在")
        order = self._resolve_order_entity(order_id)
        if not order:
            raise ValueError("订单不存在")
        policy = self.ontology.infer_return_policy(
            user.user_level,
            product_category,
            is_activated,
        )
        if not policy["returnable"]:
            return {"return_created": False, "policy": policy}
        refund_amount = Decimal(order.final_amount or 0)
        return_no = f"RTN{datetime.now().strftime('%Y%m%d%H%M%S')}{order_id:04d}"
        with self.database.get_session() as session:
            record = Return(
                return_no=return_no,
            order_id=order.order_id,
                user_id=user_id,
                return_type=return_type,
                reason=reason,
                status="pending",
                refund_amount=refund_amount,
            )
            session.add(record)
            session.flush()
            data = record.to_dict()
        return {
            "return_created": True,
            "return": data,
            "policy": policy,
        }

    # ------------------------------------------------------------------
    # 本体推理辅助
    # ------------------------------------------------------------------
    def infer_return_policy(self, user_level: str, category: str, is_activated: bool = False) -> Dict[str, Any]:
        return self.ontology.infer_return_policy(user_level, category, is_activated)

    def infer_discount(self, user_level: str, order_amount: Decimal, is_first_order: bool = False) -> Dict[str, Any]:
        return self.ontology.infer_discount(user_level, order_amount, is_first_order)

    def infer_shipping(self, user_level: str, order_amount: Decimal, is_remote_area: bool = False) -> Dict[str, Any]:
        return self.ontology.infer_shipping(user_level, order_amount, is_remote_area)

    def query_rules(self, rule_type: str) -> List[Dict[str, Any]]:
        return self.ontology.query_rules_by_type(rule_type)

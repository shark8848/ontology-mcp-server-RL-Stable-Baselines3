"""
Copyright (c) 2025 shark8848
MIT License

Ontology MCP Server - 电商 AI 助手系统
本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI

Author: shark8848
Repository: https://github.com/shark8848/ontology-mcp-server
"""

"""
数据库服务层 - 封装原子级数据库操作

采用 Repository 模式，为每个实体提供 CRUD 操作和业务查询。
"""
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session, selectinload
from sqlalchemy.pool import StaticPool

from .models import (
    Base, User, Product, CartItem, Order, OrderItem, 
    Payment, Shipment, ShipmentTrack, SupportTicket, 
    SupportMessage, Return, Review
)
from .logger import get_logger

LOGGER = get_logger(__name__)


class DatabaseService:
    """数据库服务主类 - 管理数据库连接和会话"""
    
    def __init__(self, db_path: str = "data/ecommerce.db"):
        """初始化数据库服务
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        
        # 确保data目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 创建引擎
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False  # 设置为True可以看到SQL语句
        )
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            expire_on_commit=False,
            bind=self.engine
        )
        
        LOGGER.info(f"数据库服务已初始化: {db_path}")
    
    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self.engine)
        LOGGER.info("数据库表结构已创建")
    
    def drop_tables(self):
        """删除所有表 (谨慎使用!)"""
        Base.metadata.drop_all(bind=self.engine)
        LOGGER.warning("数据库表已全部删除")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话 (上下文管理器)
        
        Yields:
            Session: SQLAlchemy 会话对象
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            LOGGER.error(f"数据库会话错误: {e}")
            raise
        finally:
            session.close()


# ============ 用户服务 ============

class UserService:
    """用户服务 - 封装用户相关的数据库操作"""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    def create_user(self, username: str, email: str = None, phone: str = None, 
                   user_level: str = "Regular") -> User:
        """创建用户"""
        with self.db.get_session() as session:
            user = User(
                username=username,
                email=email,
                phone=phone,
                user_level=user_level,
                registration_date=datetime.now()
            )
            session.add(user)
            session.flush()
            user_id = user.user_id  # 在expunge前获取ID
            session.expunge(user)  # 转为detached状态
            LOGGER.info(f"创建用户: {username} (ID: {user_id})")
            return user
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        with self.db.get_session() as session:
            return session.query(User).filter(User.user_id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        with self.db.get_session() as session:
            return session.query(User).filter(User.username == username).first()
    
    def update_user_level(self, user_id: int, user_level: str) -> bool:
        """更新用户等级"""
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if user:
                user.user_level = user_level
                session.commit()
                LOGGER.info(f"更新用户等级: user_id={user_id}, level={user_level}")
                return True
            return False
    
    def update_total_spent(self, user_id: int, amount: Decimal) -> bool:
        """更新累计消费金额"""
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if user:
                user.total_spent = (user.total_spent or 0) + amount
                session.commit()
                LOGGER.info(f"更新累计消费: user_id={user_id}, new_total={user.total_spent}")
                return True
            return False
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """列出用户"""
        with self.db.get_session() as session:
            return session.query(User).limit(limit).offset(offset).all()


# ============ 商品服务 ============

class ProductService:
    """商品服务 - 封装商品相关的数据库操作"""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    def create_product(self, product_name: str, category: str, brand: str,
                      model: str, price: Decimal, stock_quantity: int = 0,
                      description: str = None, specs: Dict = None,
                      image_url: str = None) -> Product:
        """创建商品"""
        with self.db.get_session() as session:
            product = Product(
                product_name=product_name,
                category=category,
                brand=brand,
                model=model,
                price=price,
                stock_quantity=stock_quantity,
                description=description,
                specs=specs,
                image_url=image_url
            )
            session.add(product)
            session.flush()
            product_id = product.product_id  # 在expunge前获取ID
            session.expunge(product)
            LOGGER.info(f"创建商品: {product_name} (ID: {product_id})")
            return product
    
    def get_product_by_id(self, product_id: int) -> Optional[Product]:
        """根据ID获取商品"""
        with self.db.get_session() as session:
            return session.query(Product).filter(Product.product_id == product_id).first()
    
    def search_products(self, keyword: str = None, category: str = None,
                       brand: str = None, min_price: Decimal = None,
                       max_price: Decimal = None, available_only: bool = True,
                       limit: int = 20) -> List[Product]:
        """搜索商品"""
        with self.db.get_session() as session:
            query = session.query(Product)
            
            if keyword:
                query = query.filter(
                    or_(
                        Product.product_name.contains(keyword),
                        Product.description.contains(keyword),
                        Product.model.contains(keyword)
                    )
                )
            
            if category:
                query = query.filter(Product.category == category)
            
            if brand:
                query = query.filter(Product.brand == brand)
            
            if min_price is not None:
                query = query.filter(Product.price >= min_price)
            
            if max_price is not None:
                query = query.filter(Product.price <= max_price)
            
            if available_only:
                query = query.filter(Product.is_available == True)
            results = query.limit(limit).all()
            for product in results:
                session.expunge(product)
            return results
    
    def update_stock(self, product_id: int, quantity_change: int) -> bool:
        """更新库存 (增加或减少)"""
        with self.db.get_session() as session:
            product = session.query(Product).filter(Product.product_id == product_id).first()
            if product:
                product.stock_quantity += quantity_change
                session.commit()
                LOGGER.info(f"更新库存: product_id={product_id}, change={quantity_change}, new_stock={product.stock_quantity}")
                return True
            return False
    
    def check_stock(self, product_id: int, required_quantity: int) -> bool:
        """检查库存是否充足"""
        with self.db.get_session() as session:
            product = session.query(Product).filter(Product.product_id == product_id).first()
            if product:
                return product.stock_quantity >= required_quantity
            return False


# ============ 购物车服务 ============

class CartService:
    """购物车服务"""
    
    def __init__(self, db: DatabaseService):
        self.db = db

    def _serialize_cart_item(self, session: Session, cart_item_id: int) -> Dict[str, Any]:
        cart_item = (
            session.query(CartItem)
            .options(selectinload(CartItem.product))
            .filter(CartItem.cart_id == cart_item_id)
            .first()
        )
        if not cart_item:
            return {}
        data = cart_item.to_dict()
        session.expunge(cart_item)
        return data
    
    def add_to_cart(self, user_id: int, product_id: int, quantity: int = 1) -> Dict[str, Any]:
        """加入购物车，返回序列化结果"""
        with self.db.get_session() as session:
            existing = session.query(CartItem).filter(
                and_(CartItem.user_id == user_id, CartItem.product_id == product_id)
            ).first()
            
            if existing:
                existing.quantity += quantity
                session.commit()
                LOGGER.info(
                    "更新购物车: user_id=%s, product_id=%s, new_qty=%s",
                    user_id,
                    product_id,
                    existing.quantity,
                )
                return self._serialize_cart_item(session, existing.cart_id)
            else:
                cart_item = CartItem(
                    user_id=user_id,
                    product_id=product_id,
                    quantity=quantity
                )
                session.add(cart_item)
                session.commit()
                LOGGER.info(
                    "加入购物车: user_id=%s, product_id=%s, qty=%s",
                    user_id,
                    product_id,
                    quantity,
                )
                return self._serialize_cart_item(session, cart_item.cart_id)
    
    def get_cart(self, user_id: int) -> List[Dict[str, Any]]:
        """获取购物车，已包含商品详情"""
        with self.db.get_session() as session:
            items = (
                session.query(CartItem)
                .options(selectinload(CartItem.product))
                .filter(CartItem.user_id == user_id)
                .all()
            )
            result = [item.to_dict() for item in items]
            for item in items:
                session.expunge(item)
            return result
    
    def remove_from_cart(self, user_id: int, product_id: int) -> bool:
        """从购物车移除"""
        with self.db.get_session() as session:
            result = session.query(CartItem).filter(
                and_(CartItem.user_id == user_id, CartItem.product_id == product_id)
            ).delete()
            session.commit()
            LOGGER.info(f"从购物车移除: user_id={user_id}, product_id={product_id}")
            return result > 0
    
    def clear_cart(self, user_id: int) -> bool:
        """清空购物车"""
        with self.db.get_session() as session:
            result = session.query(CartItem).filter(CartItem.user_id == user_id).delete()
            session.commit()
            LOGGER.info(f"清空购物车: user_id={user_id}")
            return result > 0
    
    def update_quantity(self, user_id: int, product_id: int, quantity: int) -> bool:
        """更新购物车商品数量"""
        with self.db.get_session() as session:
            cart_item = session.query(CartItem).filter(
                and_(CartItem.user_id == user_id, CartItem.product_id == product_id)
            ).first()
            if cart_item:
                cart_item.quantity = quantity
                session.commit()
                return True
            return False


# ============ 订单服务 ============

class OrderService:
    """订单服务"""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    def create_order(self, user_id: int, items: List[Dict[str, Any]],
                    shipping_address: str, contact_phone: str,
                    discount_amount: Decimal = Decimal('0')) -> Order:
        """创建订单
        
        Args:
            user_id: 用户ID
            items: 订单项列表 [{"product_id": 1, "quantity": 2, "unit_price": 100}]
            shipping_address: 收货地址
            contact_phone: 联系电话
            discount_amount: 折扣金额
        
        Returns:
            Order: 创建的订单对象
        """
        with self.db.get_session() as session:
            # 生成订单号
            order_no = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}{user_id:04d}"
            
            # 计算总金额
            total_amount = sum(
                Decimal(str(item['quantity'])) * Decimal(str(item['unit_price']))
                for item in items
            )
            final_amount = total_amount - discount_amount
            
            # 创建订单
            order = Order(
                order_no=order_no,
                user_id=user_id,
                total_amount=total_amount,
                discount_amount=discount_amount,
                final_amount=final_amount,
                shipping_address=shipping_address,
                contact_phone=contact_phone,
                order_status='pending',
                payment_status='unpaid'
            )
            session.add(order)
            session.flush()  # 获取 order_id
            
            # 创建订单明细
            for item in items:
                order_item = OrderItem(
                    order_id=order.order_id,
                    product_id=item['product_id'],
                    product_name=item.get('product_name', ''),
                    quantity=item['quantity'],
                    unit_price=Decimal(str(item['unit_price'])),
                    subtotal=Decimal(str(item['quantity'])) * Decimal(str(item['unit_price']))
                )
                session.add(order_item)
            
            session.commit()
            session.refresh(order)
            order_dict = order.to_dict()
            LOGGER.info(f"创建订单: {order_no}, user_id={user_id}, amount={final_amount}")
            return order_dict
    
    def get_order_by_id(self, order_id: int) -> Optional[Order]:
        """根据ID获取订单"""
        with self.db.get_session() as session:
            return (
                session.query(Order)
                .options(
                    selectinload(Order.order_items).selectinload(OrderItem.product)
                )
                .filter(Order.order_id == order_id)
                .first()
            )
    
    def get_order_by_no(self, order_no: str) -> Optional[Order]:
        """根据订单号获取订单"""
        with self.db.get_session() as session:
            return (
                session.query(Order)
                .options(
                    selectinload(Order.order_items).selectinload(OrderItem.product)
                )
                .filter(Order.order_no == order_no)
                .first()
            )
    
    def get_user_orders(self, user_id: int, status: str = None, 
                       limit: int = 20, offset: int = 0) -> List[Order]:
        """获取用户订单列表"""
        with self.db.get_session() as session:
            query = (
                session.query(Order)
                .options(
                    selectinload(Order.order_items).selectinload(OrderItem.product)
                )
                .filter(Order.user_id == user_id)
            )
            
            if status:
                query = query.filter(Order.order_status == status)
            
            return query.order_by(Order.created_at.desc()).limit(limit).offset(offset).all()
    
    def update_order_status(self, order_id: int, status: str) -> bool:
        """更新订单状态"""
        with self.db.get_session() as session:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order:
                order.order_status = status
                
                # 更新时间戳
                if status == 'paid':
                    order.paid_at = datetime.now()
                elif status == 'shipped':
                    order.shipped_at = datetime.now()
                elif status == 'delivered':
                    order.delivered_at = datetime.now()
                
                session.commit()
                LOGGER.info(f"更新订单状态: order_id={order_id}, status={status}")
                return True
            return False
    
    def update_payment_status(self, order_id: int, status: str) -> bool:
        """更新支付状态"""
        with self.db.get_session() as session:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order:
                order.payment_status = status
                if status == 'paid':
                    order.paid_at = datetime.now()
                session.commit()
                return True
            return False
    
    def cancel_order(self, order_id: int) -> bool:
        """取消订单"""
        with self.db.get_session() as session:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order and order.order_status in ['pending', 'paid']:
                order.order_status = 'cancelled'
                session.commit()
                LOGGER.info(f"取消订单: order_id={order_id}")
                return True
            return False

    def list_orders(self, limit: int = 1000, offset: int = 0) -> List[Order]:
        """按创建时间倒序返回订单列表，包含订单明细。"""
        with self.db.get_session() as session:
            query = (
                session.query(Order)
                .options(
                    selectinload(Order.order_items).selectinload(OrderItem.product)
                )
                .order_by(Order.created_at.desc())
            )
            return query.limit(limit).offset(offset).all()

    def list_user_orders(self, user_id: int, limit: int = 1000, offset: int = 0) -> List[Order]:
        """按用户ID返回订单列表（包装 get_user_orders 以兼容旧调用）。"""
        return self.get_user_orders(user_id=user_id, limit=limit, offset=offset)


# ============ 支付服务 ============

class PaymentService:
    """支付服务"""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    def create_payment(self, order_id: int, payment_method: str, 
                      payment_amount: Decimal) -> Payment:
        """创建支付记录"""
        with self.db.get_session() as session:
            # 生成交易ID
            transaction_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            payment = Payment(
                order_id=order_id,
                payment_method=payment_method,
                payment_amount=payment_amount,
                payment_status='pending',
                transaction_id=transaction_id
            )
            session.add(payment)
            session.commit()
            session.flush(); session.expunge(payment)
            LOGGER.info(f"创建支付记录: order_id={order_id}, method={payment_method}")
            return payment
    
    def update_payment_status(self, payment_id: int, status: str) -> bool:
        """更新支付状态"""
        with self.db.get_session() as session:
            payment = session.query(Payment).filter(Payment.payment_id == payment_id).first()
            if payment:
                payment.payment_status = status
                if status == 'success':
                    payment.payment_time = datetime.now()
                session.commit()
                return True
            return False
    
    def get_payment_by_order(self, order_id: int) -> Optional[Payment]:
        """根据订单ID获取支付记录"""
        with self.db.get_session() as session:
            return session.query(Payment).filter(Payment.order_id == order_id).first()


# ============ 物流服务 ============

class ShipmentService:
    """物流服务"""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    def create_shipment(self, order_id: int, carrier: str, 
                       estimated_delivery: datetime = None) -> Shipment:
        """创建物流记录"""
        with self.db.get_session() as session:
            # 生成运单号
            tracking_no = f"SF{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            shipment = Shipment(
                order_id=order_id,
                tracking_no=tracking_no,
                carrier=carrier,
                current_status='待揽收',
                estimated_delivery=estimated_delivery or (datetime.now() + timedelta(days=3)),
                shipped_at=datetime.now()
            )
            session.add(shipment)
            session.commit()
            session.flush(); session.expunge(shipment)
            LOGGER.info(f"创建物流记录: order_id={order_id}, tracking={tracking_no}")
            return shipment
    
    def add_track(self, shipment_id: int, status: str, location: str, 
                 description: str) -> ShipmentTrack:
        """添加物流轨迹"""
        with self.db.get_session() as session:
            track = ShipmentTrack(
                shipment_id=shipment_id,
                status=status,
                location=location,
                description=description,
                track_time=datetime.now()
            )
            session.add(track)
            
            # 更新物流状态
            shipment = session.query(Shipment).filter(Shipment.shipment_id == shipment_id).first()
            if shipment:
                shipment.current_status = status
                shipment.current_location = location
                if status == '已签收':
                    shipment.delivered_at = datetime.now()
            
            session.commit()
            session.flush(); session.expunge(track)
            return track
    
    def get_shipment_by_order(self, order_id: int) -> Optional[Shipment]:
        """根据订单ID获取物流信息"""
        with self.db.get_session() as session:
            return session.query(Shipment).filter(Shipment.order_id == order_id).first()
    
    def get_shipment_by_tracking(self, tracking_no: str) -> Optional[Shipment]:
        """根据运单号获取物流信息"""
        with self.db.get_session() as session:
            return session.query(Shipment).filter(Shipment.tracking_no == tracking_no).first()


# ============ 主服务入口 ============

class EcommerceService:
    """电商服务总入口 - 聚合所有子服务"""
    
    def __init__(self, db_path: str = "data/ecommerce.db"):
        self.db = DatabaseService(db_path)
        
        # 初始化子服务
        self.users = UserService(self.db)
        self.products = ProductService(self.db)
        self.cart = CartService(self.db)
        self.orders = OrderService(self.db)
        self.payments = PaymentService(self.db)
        self.shipments = ShipmentService(self.db)
        
        LOGGER.info("电商服务已初始化")
    
    def init_database(self):
        """初始化数据库 (创建表)"""
        self.db.create_tables()

# 人工确认机制实现方案

## 问题背景

当前系统存在以下严重漏洞:

1. **无人工确认直接执行关键操作**:
   - 创建订单 (commerce_create_order)
   - 处理支付 (commerce_process_payment)
   - 取消订单 (commerce_cancel_order)
   - 更新订单状态 (commerce_update_order_status)

2. **实际案例** (logs/agent_20251130_224257.log):
   - 用户: "我的id是1，收货地址..."
   - 系统: 自动创建订单56 (iPhone, ¥17998) → 取消失败 → 创建订单57 (Xiaomi, ¥9358) → 自动支付
   - **全程无人工确认!**

---

## 解决方案设计

### 方案A: 工具拦截+会话状态 (推荐)

#### 核心思路
在 `react_agent.py` 的 `_call_tool()` 方法中拦截关键工具调用,返回确认提示而非直接执行。

#### 实现步骤

**1. 定义关键工具列表**

```python
# src/agent/react_agent.py (class OpenAIAgent)

CRITICAL_TOOLS = {
    "commerce_create_order": {
        "name": "创建订单",
        "risk_level": "high",
        "requires_confirmation": True
    },
    "commerce_process_payment": {
        "name": "处理支付",
        "risk_level": "critical",
        "requires_confirmation": True
    },
    "commerce_cancel_order": {
        "name": "取消订单",
        "risk_level": "high",
        "requires_confirmation": True
    },
    "commerce_update_order_status": {
        "name": "更新订单状态",
        "risk_level": "medium",
        "requires_confirmation": True
    }
}
```

**2. 添加待确认队列**

```python
class OpenAIAgent:
    def __init__(self, ...):
        # 现有代码
        ...
        
        # 新增: 待确认操作队列
        self.pending_confirmations: List[Dict[str, Any]] = []
        self.confirmation_mode: bool = False  # 是否处于等待确认状态
```

**3. 修改 `_call_tool()` 方法**

```python
def _call_tool(self, tool_name: str, args: str | dict) -> str:
    """执行工具调用(带人工确认检查)"""
    
    # 检查是否为关键操作
    if tool_name in self.CRITICAL_TOOLS:
        tool_info = self.CRITICAL_TOOLS[tool_name]
        
        # 生成确认请求
        confirmation_request = self._generate_confirmation_request(
            tool_name=tool_name,
            tool_info=tool_info,
            args=args
        )
        
        # 将操作加入待确认队列
        self.pending_confirmations.append({
            "tool_name": tool_name,
            "args": args,
            "tool_info": tool_info,
            "timestamp": datetime.now().isoformat()
        })
        self.confirmation_mode = True
        
        # 返回确认提示(而非执行结果)
        return json.dumps({
            "requires_confirmation": True,
            "message": confirmation_request,
            "operation_id": len(self.pending_confirmations) - 1
        }, ensure_ascii=False)
    
    # 非关键操作直接执行(现有逻辑)
    tool = self.tool_map.get(tool_name)
    if not tool:
        return json.dumps({"error": f"工具 {tool_name} 未找到"}, ensure_ascii=False)
    
    return tool.invoke(args)
```

**4. 生成确认提示**

```python
def _generate_confirmation_request(self, tool_name: str, tool_info: dict, args: dict) -> str:
    """生成人工确认提示"""
    
    templates = {
        "commerce_create_order": """
⚠️ **需要您的确认**

我即将为您创建订单:
- 用户ID: {user_id}
- 商品信息:
{item_details}
- 收货地址: {shipping_address}
- 联系电话: {contact_phone}
- 预计金额: ¥{estimated_total}

**请回复**:
- "确认" 或 "是" → 创建订单
- "取消" 或 "否" → 取消操作
- "修改XXX" → 修改订单信息
""",
        
        "commerce_process_payment": """
💳 **支付确认**

订单号: {order_id}
支付金额: ¥{amount}
支付方式: {payment_method}

此操作将扣款，请确认:
- "确认支付" → 执行支付
- "取消" → 不支付
""",
        
        "commerce_cancel_order": """
🚫 **取消订单确认**

订单号: {order_id}
当前状态: {current_status}

确认取消此订单吗?
- "确认取消" → 取消订单
- "保留" → 保留订单
"""
    }
    
    template = templates.get(tool_name, "确认执行操作: {tool_name}?")
    
    # 根据工具类型填充参数
    if tool_name == "commerce_create_order":
        items = args.get("items", [])
        item_details = "\n".join([
            f"  * 商品ID={item['product_id']}, 数量={item['quantity']}"
            for item in items
        ])
        # TODO: 查询商品信息获取名称和价格
        return template.format(
            user_id=args.get("user_id"),
            item_details=item_details,
            shipping_address=args.get("shipping_address"),
            contact_phone=args.get("contact_phone"),
            estimated_total="待计算"
        )
    
    elif tool_name == "commerce_process_payment":
        return template.format(
            order_id=args.get("order_id"),
            amount=args.get("amount"),
            payment_method=args.get("payment_method")
        )
    
    elif tool_name == "commerce_cancel_order":
        return template.format(
            order_id=args.get("order_id"),
            current_status="待查询"
        )
    
    return template
```

**5. 处理用户确认响应**

```python
def _handle_confirmation_response(self, user_input: str) -> Optional[str]:
    """处理用户确认响应"""
    
    if not self.confirmation_mode or not self.pending_confirmations:
        return None
    
    # 获取待确认操作
    pending = self.pending_confirmations[-1]
    
    # 识别确认/取消意图
    confirm_keywords = ["确认", "是", "好", "可以", "同意", "yes", "ok"]
    cancel_keywords = ["取消", "否", "不", "no", "cancel"]
    
    user_lower = user_input.lower().strip()
    
    if any(kw in user_lower for kw in confirm_keywords):
        # 用户确认 → 执行操作
        tool_name = pending["tool_name"]
        args = pending["args"]
        
        logger.info("用户确认操作: %s", tool_name)
        
        # 实际执行工具调用
        tool = self.tool_map.get(tool_name)
        if tool:
            result = tool.invoke(args)
            
            # 清理确认状态
            self.pending_confirmations.pop()
            if not self.pending_confirmations:
                self.confirmation_mode = False
            
            return result
        else:
            return json.dumps({"error": f"工具 {tool_name} 未找到"}, ensure_ascii=False)
    
    elif any(kw in user_lower for kw in cancel_keywords):
        # 用户取消
        logger.info("用户取消操作: %s", pending["tool_name"])
        
        self.pending_confirmations.pop()
        if not self.pending_confirmations:
            self.confirmation_mode = False
        
        return json.dumps({
            "cancelled": True,
            "message": "操作已取消"
        }, ensure_ascii=False)
    
    else:
        # 无法识别意图
        return json.dumps({
            "pending": True,
            "message": "请明确回复 '确认' 或 '取消'"
        }, ensure_ascii=False)
```

**6. 修改 `chat()` 方法入口**

```python
def chat(self, user_input: str, ...):
    """主聊天入口"""
    
    # 新增: 检查是否处于确认模式
    if self.confirmation_mode:
        confirmation_result = self._handle_confirmation_response(user_input)
        if confirmation_result:
            # 返回确认处理结果
            confirmation_data = json.loads(confirmation_result)
            if confirmation_data.get("cancelled"):
                return "操作已取消。还有什么我可以帮您的吗?"
            elif confirmation_data.get("pending"):
                return confirmation_data["message"]
            else:
                # 操作已执行,继续正常流程
                # 将结果注入到下一轮对话
                pass
    
    # 现有 chat 逻辑
    ...
```

---

### 方案B: 修改工具定义 (备选)

#### 核心思路
在 `mcp_adapter.py` 中修改工具描述,让LLM主动询问用户确认。

#### 实现
```python
# src/agent/mcp_adapter.py

def _create_tool_definition(tool: Tool) -> ToolDefinition:
    # 关键操作添加确认说明
    if tool.name in ["commerce_create_order", "commerce_process_payment"]:
        enhanced_description = f"""
{tool.description}

⚠️ 重要: 这是关键操作!
在调用此工具前,你必须:
1. 向用户明确列出操作详情
2. 询问 "请确认是否继续?"
3. 等待用户明确回复 "确认" 后再调用
4. 如用户回复 "取消",不要调用此工具
"""
        tool.description = enhanced_description
```

**优点**: 实现简单,无需修改agent逻辑
**缺点**: 依赖LLM理解力,无法强制拦截

---

## 推荐实施方案

**采用方案A (工具拦截)** + 部分方案B (提示增强)

### 第一阶段 (立即实施)
1. 在 `react_agent.py` 实现工具拦截机制
2. 添加确认队列和状态管理
3. 实现确认提示生成器

### 第二阶段 (增强安全)
1. 在 Gradio UI 添加确认按钮
2. 在数据库记录所有确认日志
3. 添加确认超时机制(5分钟未确认自动取消)

### 第三阶段 (优化体验)
1. 支持批量操作确认(多件商品一次确认)
2. 用户可设置"信任模式"(跳过小额支付确认)
3. 添加确认历史回溯功能

---

## 测试用例

### 测试1: 创建订单确认
**输入**: "小米手机658，买2台，地址北京..."
**预期**:
1. 系统显示订单详情
2. 提示 "⚠️ 需要您的确认"
3. 用户回复 "确认"
4. 系统创建订单

### 测试2: 取消确认
**输入**: "小米手机658，买2台，地址北京..."
**系统**: 显示确认提示
**输入**: "取消"
**预期**: 系统回复 "操作已取消"，不创建订单

### 测试3: 支付确认
**输入**: "支付订单ORD123"
**预期**:
1. 系统显示 "💳 支付确认"
2. 显示订单号、金额、支付方式
3. 等待用户回复 "确认支付"

---

## 实施优先级

🔴 **P0 - 立即实施** (2小时)
- [x] 定义关键工具列表
- [ ] 实现 `_call_tool()` 拦截逻辑
- [ ] 实现确认提示生成器
- [ ] 实现确认响应处理器

🟡 **P1 - 本周完成** (1天)
- [ ] Gradio UI 添加确认按钮
- [ ] 数据库记录确认日志
- [ ] 添加确认超时机制

🟢 **P2 - 下周优化** (2天)
- [ ] 批量确认支持
- [ ] 信任模式设置
- [ ] 确认历史功能

---

## 附录: 其他发现的问题

1. **意图识别错误**
   - "来2台" → unknown (应为 purchase_intent)
   - 需要在 `intent_tracker.py` 添加购买相关规则

2. **产品验证缺失**
   - 推荐 iPhone (ID=1) 给要求 "小米手机" 的用户
   - 需要在 `commerce_service.py` 添加品牌验证

3. **FTS5 搜索问题**
   - keyword="小米" 返回 0 结果
   - 需要检查中文分词配置

4. **工具调用顺序错误**
   - 先创建订单 → 再搜索商品 → 取消订单 → 重新创建
   - 需要优化prompt或添加规划阶段

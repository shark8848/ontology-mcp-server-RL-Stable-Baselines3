# 配置化意图识别系统使用指南

## 📋 概述

新的意图识别系统支持三种策略,可根据场景灵活配置优先级顺序。

## 🎯 三种识别策略

### 1. LLM 识别 (最高准确率)
- **准确率**: 90-95%
- **成本**: ¥0.0001/次
- **延迟**: 50-200ms
- **优势**: 理解复杂语义,支持实体提取
- **适用场景**: 复杂、模糊的用户输入

### 2. Embedding 相似度匹配 (平衡)
- **准确率**: 80-85%
- **成本**: 免费 (首次加载模型)
- **延迟**: 10-30ms
- **优势**: 语义理解,不依赖关键词
- **适用场景**: 常见表达的快速识别

### 3. 规则匹配 (最快速度)
- **准确率**: 60-70%
- **成本**: 免费
- **延迟**: <1ms
- **优势**: 极快响应,零成本
- **适用场景**: 明确关键词的简单输入

## ⚙️ 配置方式

编辑 `src/agent/config.yaml`:

```yaml
intent_recognition:
  # 策略优先级 (按顺序尝试)
  priority: ["llm", "embedding", "rule"]
  
  # 高置信度阈值 (达到此值则停止尝试后续策略)
  high_confidence_threshold: 0.85
  
  # LLM 配置
  llm:
    enabled: true
    enable_cache: true  # 缓存相似问题
    system_prompt: "你是一个电商意图识别专家..."
  
  # Embedding 配置
  embedding:
    enabled: true
    model: "paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: 0.75
    enable_template_cache: true
  
  # 规则匹配配置
  rule:
    enabled: true
    use_regex: true
    default_confidence: 0.6
```

## 📊 配置方案推荐

### 方案 1: 高准确率优先 (推荐)
```yaml
priority: ["llm", "embedding", "rule"]
high_confidence_threshold: 0.85
```
- **特点**: LLM 优先,失败时降级到 Embedding 和规则
- **成本**: 约 ¥0.0001/次
- **适用**: 对准确率要求高的生产环境

### 方案 2: 性能优先
```yaml
priority: ["rule", "embedding", "llm"]
high_confidence_threshold: 0.70
```
- **特点**: 规则优先,复杂问题才用 LLM
- **成本**: 大部分免费,仅复杂问题调用 LLM
- **适用**: 高并发、成本敏感场景

### 方案 3: 平衡方案
```yaml
priority: ["embedding", "llm", "rule"]
high_confidence_threshold: 0.80
```
- **特点**: Embedding 优先,兼顾速度和准确率
- **成本**: 免费 (首次加载模型)
- **适用**: 中等流量,对延迟敏感的场景

### 方案 4: 纯 LLM (最高质量)
```yaml
priority: ["llm"]
high_confidence_threshold: 0.95
```
- **特点**: 只用 LLM,追求极致准确率
- **成本**: ¥0.0001/次 (固定)
- **适用**: 低频次、高价值的交互场景

## 🔧 禁用特定策略

如果某个策略不需要或依赖未安装:

```yaml
intent_recognition:
  llm:
    enabled: false  # 禁用 LLM
  
  embedding:
    enabled: false  # 禁用 Embedding (如未安装 sentence-transformers)
  
  rule:
    enabled: true   # 仅使用规则匹配
```

## 📦 依赖安装

### LLM 策略 (已安装)
无需额外依赖,使用项目已有的 DeepSeek API。

### Embedding 策略 (可选)
```bash
pip install sentence-transformers scikit-learn
```

### 规则匹配 (无需依赖)
Python 标准库即可。

## 🧪 测试

运行测试脚本验证配置:

```bash
python test_intent_recognition.py
```

测试内容:
1. 规则匹配识别器测试
2. LLM 识别器测试
3. 混合识别器测试 (按优先级)
4. 边界情况测试

## 🎨 使用示例

### 代码中使用

```python
from src.agent.intent_tracker import HybridIntentRecognizer
from src.agent.llm_deepseek import create_deepseek_chat_model
import yaml

# 加载配置
with open("src/agent/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 创建 LLM
llm = create_deepseek_chat_model(config)

# 创建混合识别器
intent_config = config.get("intent_recognition", {})
recognizer = HybridIntentRecognizer(llm, intent_config)

# 识别意图
intents = recognizer.recognize("你们有什么好的电子产品推荐？")
intent = intents[0]

print(f"意图: {intent.category.value}")
print(f"置信度: {intent.confidence}")
print(f"提取实体: {intent.extracted_entities}")
```

### 在 ReAct Agent 中使用 (自动集成)

系统已自动在 `ReactAgent` 中集成混合识别器:

```python
# react_agent.py 会自动加载配置并使用
agent = ReactAgent(
    tools=tools,
    llm=llm,
    config=config,
    enable_intent_tracking=True  # 启用意图跟踪
)

# 意图识别会在每次用户输入时自动执行
response = agent.chat("你们有什么好的电子产品推荐？")
```

## 📈 性能优化

### LLM 缓存
```yaml
llm:
  enable_cache: true  # 启用 LRU 缓存 (默认 1000 条)
```
- 相同或相似问题会命中缓存
- 零成本,零延迟
- 适合高重复率场景

### Embedding 模板缓存
```yaml
embedding:
  enable_template_cache: true  # 预计算模板 embeddings
```
- 启动时预计算所有意图模板
- 识别时只需计算用户输入的 embedding

## 🐛 故障排查

### 问题 1: Embedding 识别器不可用
```
WARNING: sentence-transformers 未安装，Embedding 识别器不可用
```

**解决方案**:
```bash
pip install sentence-transformers scikit-learn
```

或禁用 Embedding 策略:
```yaml
embedding:
  enabled: false
```

### 问题 2: LLM 识别返回 UNKNOWN
```
LLM 返回的 JSON 解析失败
```

**原因**: DeepSeek 返回格式异常

**解决方案**:
1. 检查 API Key 是否有效
2. 调整 `system_prompt` 增强 JSON 格式要求
3. 降级使用 Embedding 或规则策略

### 问题 3: 所有策略都失败
```
使用备选识别结果: unknown
```

**原因**: 输入过于模糊或无意义

**解决方案**:
- 降低 `high_confidence_threshold`
- 增加规则匹配的关键词覆盖
- 优化 LLM prompt

## 📚 扩展新策略

实现自定义识别器:

```python
from src.agent.intent_tracker import BaseIntentRecognizer, Intent, IntentCategory

class CustomRecognizer(BaseIntentRecognizer):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def get_confidence(self) -> float:
        return 0.85
    
    def recognize(self, user_input: str, turn_id: int = 0) -> List[Intent]:
        # 实现自定义识别逻辑
        category = IntentCategory.RECOMMENDATION
        confidence = 0.9
        
        return [Intent(
            category=category,
            confidence=confidence,
            turn_id=turn_id,
            raw_input=user_input
        )]

# 注册到混合识别器
recognizer = HybridIntentRecognizer(llm, config)
recognizer.recognizers["custom"] = CustomRecognizer(config)
```

## 💡 最佳实践

1. **开发环境**: 使用纯 LLM 策略确保准确性
2. **测试环境**: 使用混合策略 (LLM → Embedding → Rule)
3. **生产环境**: 根据 QPS 和成本选择合适方案
4. **监控**: 记录各策略的命中率和准确率,持续优化

## 📞 支持

如有问题,请查看:
- 日志: `src/agent/logs/agent.log`
- 测试: `python test_intent_recognition.py`
- 文档: 项目 README.md

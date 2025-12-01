import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.intent_tracker import HybridIntentRecognizer, IntentCategory


class DummyLLM:
    """Simple stub that returns a predefined JSON payload."""

    def __init__(self, payload: str):
        self.payload = payload

    def generate(self, messages, tools=None):  # noqa: D401 - signature mirrors real LLM
        return {"content": self.payload}


def _base_config(priority):
    return {
        "priority": priority,
        "high_confidence_threshold": 0.85,
        "llm": {
            "enabled": True,
            "enable_cache": False,
            "system_prompt": "test",
        },
        "rule": {
            "enabled": True,
            "default_confidence": 0.65,
        },
    }


def test_llm_high_confidence_selected(caplog):
    caplog.set_level(logging.INFO)
    llm = DummyLLM('{"intent": "RECOMMENDATION", "confidence": 0.92, "entities": {}}')
    recognizer = HybridIntentRecognizer(llm=llm, config=_base_config(["llm", "rule"]))

    result = recognizer.recognize("随便推荐点东西", turn_id=1)[0]

    assert result.category == IntentCategory.RECOMMENDATION
    assert "意图识别 [llm]" in caplog.text
    assert "使用 llm 识别结果" in caplog.text


def test_llm_low_confidence_falls_back_to_rule(caplog):
    caplog.set_level(logging.INFO)
    llm = DummyLLM('{"intent": "RECOMMENDATION", "confidence": 0.40, "entities": {}}')
    recognizer = HybridIntentRecognizer(llm=llm, config=_base_config(["llm", "rule"]))

    result = recognizer.recognize("查询物流状态", turn_id=2)[0]

    assert result.category == IntentCategory.ORDER_TRACK
    assert "使用备选识别结果" in caplog.text


def test_buy_phone_matches_recommendation():
    config = {
        "priority": ["rule"],
        "rule": {
            "enabled": True,
            "default_confidence": 0.7,
        },
    }
    recognizer = HybridIntentRecognizer(llm=None, config=config)

    result = recognizer.recognize("我想买个手机", turn_id=3)[0]

    assert result.category == IntentCategory.RECOMMENDATION

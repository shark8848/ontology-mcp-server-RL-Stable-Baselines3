from __future__ import annotations

"""Tests for mandatory ontology validation enforcement inside LangChainAgent."""

from agent.react_agent import LangChainAgent
from agent.conversation_state import ConversationStage


class DummyLLM:
    """Minimal stub that mimics the chat model interface used by LangChainAgent."""

    def generate(self, messages, tools=None):  # pragma: no cover - helper stub
        return {"content": "stub", "tool_calls": []}


def _build_agent() -> LangChainAgent:
    """Construct agent with optional components disabled for faster testing."""

    agent = LangChainAgent(
        llm=DummyLLM(),
        max_iterations=1,
        use_memory=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    if agent.state_manager and agent.state_manager.state:
        agent.state_manager.state.update_stage(ConversationStage.GREETING, "test_setup")
    return agent


def test_validation_required_in_checkout_stage() -> None:
    agent = _build_agent()
    assert agent.state_manager and agent.state_manager.state
    agent.state_manager.state.update_stage(ConversationStage.CHECKOUT, "unit-test")

    requires, payload = agent._should_require_validation("开始结算", [])

    assert requires is True
    assert payload is not None
    assert payload["format"] == "turtle"
    assert payload["data"] == ""


def test_validation_required_after_create_order_tool() -> None:
    agent = _build_agent()

    tool_log = [
        {
            "tool": "commerce_create_order",
            "input": {"user_id": 1, "items": []},
            "observation": "{}",
            "iteration": 1,
        }
    ]

    requires, payload = agent._should_require_validation("下单", tool_log)

    assert requires is True
    assert payload is not None
    assert payload["format"] == "turtle"


def test_validation_not_required_after_recent_check() -> None:
    agent = _build_agent()

    tool_log = [
        {
            "tool": "commerce_create_order",
            "input": {"user_id": 1, "items": []},
            "observation": "{}",
            "iteration": 1,
        },
        {
            "tool": "ontology_validate_order",
            "input": {"data": "ttl", "format": "turtle"},
            "observation": "{}",
            "iteration": 2,
        },
    ]

    requires, payload = agent._should_require_validation("完成订单", tool_log)

    assert requires is False
    assert payload is None


def test_validation_reminder_injected_once_per_iteration() -> None:
    agent = _build_agent()

    messages = []
    payload = {"data": "", "format": "turtle"}

    agent._inject_validation_reminder(messages, payload, iteration=1)
    assert len(messages) == 1

    agent._inject_validation_reminder(messages, payload, iteration=1)
    assert len(messages) == 1

    agent._inject_validation_reminder(messages, payload, iteration=2)
    assert len(messages) == 2

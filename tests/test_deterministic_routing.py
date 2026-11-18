
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from src.agent import nodes


@pytest.mark.parametrize(
    "intent, order_id, message, expected_tool",
    [
        (
            "cancel_order",
            "ORD-1002",
            "Cancel my shipped order ORD-1002.",
            "cancel_order",
        ),
        (
            "cancel_order",
            "ORD-1005",
            "Cancel my delivered order ORD-1005.",
            "cancel_order",
        ),
        (
            "refund_request",
            "ORD-1002",
            "Refund ORD-1002 because I changed my mind.",
            "request_refund",
        ),
        (
            "refund_request",
            "ORD-1005",
            "Refund ORD-1005 again because it is broken.",
            "request_refund",
        ),
    ],
)
def test_deterministic_routing(
    monkeypatch, intent, order_id, message, expected_tool
):
    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def invoke(self, args):
            calls.append((self.name, args))
            return "Test tool executed"

    fake_tools = [
        FakeTool("cancel_order"),
        FakeTool("request_refund"),
    ]

    monkeypatch.setattr(nodes, "ALL_TOOLS", fake_tools)

    def fail_if_llm_called():
        raise AssertionError(
            "Deterministic routing must not call tool-selection LLM"
        )

    monkeypatch.setattr(
        nodes, "_get_llm_with_tools", fail_if_llm_called
    )

    state = {
        "intent": intent,
        "order_id": order_id,
        "messages": [HumanMessage(content=message)],
        "retry_count": 0,
    }

    result = nodes.tool_node(state)

    assert len(calls) == 1
    assert calls[0][0] == expected_tool
    assert calls[0][1]["order_id"] == order_id

    if intent == "refund_request":
        assert calls[0][1]["reason"] == message

    assert expected_tool in result["tool_results"]


def test_missing_order_id_does_not_execute_tool(monkeypatch):
    def fail_if_llm_called():
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(
        nodes, "_get_llm_with_tools", fail_if_llm_called
    )

    state = {
        "intent": "refund_request",
        "order_id": "",
        "messages": [
            HumanMessage(content="I want a refund.")
        ],
        "retry_count": 0,
    }

    result = nodes.tool_node(state)

    assert "missing_information" in result["tool_results"]
    assert "request_refund" not in result["tool_results"]

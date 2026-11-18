
import pytest

from src.services import database
from src.services.order_service import OrderService

from src.agent.tools import cancel_order

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph

@pytest.fixture
def order_service(tmp_path, monkeypatch):
    """Create an isolated SQLite database for each test."""
    test_db = tmp_path / "test_ecommerce.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db()

    return OrderService()


def test_cancel_confirmed_order(order_service):
    success, message = order_service.cancel_order(
        "ORD-1003", "marcus@example.com"
    )

    assert success is True
    assert order_service.get_order("ORD-1003").status == "cancelled"


def test_cannot_cancel_shipped_order(order_service):
    success, message = order_service.cancel_order(
        "ORD-1002", "sarah@example.com"
    )

    assert success is False
    assert order_service.get_order("ORD-1002").status == "shipped"


def test_cannot_cancel_already_cancelled_order(order_service):
    success, message = order_service.cancel_order(
        "ORD-1004", "emily@example.com"
    )

    assert success is False
    assert order_service.get_order("ORD-1004").status == "cancelled"


def test_cannot_cancel_nonexistent_order(order_service):
    success, message = order_service.cancel_order(
        "ORD-9999", "marcus@example.com"
    )

    assert success is False
    assert "not found" in message.lower()


def test_cannot_cancel_another_customers_order(order_service):
    success, message = order_service.cancel_order(
        "ORD-1003", "sarah@example.com"
    )

    assert success is False
    assert "not authorized" in message.lower()
    assert order_service.get_order("ORD-1003").status == "confirmed"


def test_cannot_cancel_without_identity(order_service):
    success, message = order_service.cancel_order(
        "ORD-1003", ""
    )

    assert success is False
    assert order_service.get_order("ORD-1003").status == "confirmed"



def test_tool_cancels_own_order(order_service):
    """The tool should cancel the demo customer's confirmed order."""
    result = cancel_order.invoke({"order_id": "ORD-1003"})

    assert "cancelled successfully" in result.lower()
    assert order_service.get_order("ORD-1003").status == "cancelled"


def test_tool_rejects_another_customer(order_service):
    """The tool must not cancel another customer's order."""
    result = cancel_order.invoke({"order_id": "ORD-1007"})

    assert "not authorized" in result.lower()
    assert order_service.get_order("ORD-1007").status == "confirmed"



def test_agent_cancel_order_end_to_end(order_service):
    """Test the complete LangGraph cancellation workflow."""

    initial_state = {
        "messages": [
            HumanMessage(content="Please cancel my order ORD-1003.")
        ],
        "intent": "",
        "order_id": "",
        "tracking_number": "",
        "customer_email": "",
        "tool_results": {},
        "final_response": "",
        "retry_count": 0,
        "error_message": "",
    }

    result = agent_graph.invoke(initial_state)

    assert result["intent"] == "cancel_order"
    assert result["order_id"] == "ORD-1003"
    assert "cancel_order" in result["tool_results"]
    assert "cancelled successfully" in (
        result["tool_results"]["cancel_order"].lower()
    )

    order = order_service.get_order("ORD-1003")
    assert order.status == "cancelled"


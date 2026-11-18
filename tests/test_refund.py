
import pytest

from src.services import database
from src.services.returns_service import ReturnsService

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph
from src.config import settings


@pytest.fixture
def refund_service(tmp_path, monkeypatch):
    """Create an isolated database for each test."""
    test_db = tmp_path / "test_ecommerce.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db()
    return ReturnsService()


def get_refund_status(order_id):
    """Read refund status directly from SQLite."""
    conn = database.get_connection()
    try:
        row = conn.execute(
            "SELECT status FROM refunds WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        return row["status"] if row else None
    finally:
        conn.close()


def test_small_refund_auto_approved(refund_service):
    success, message = refund_service.request_refund(
        "ORD-1005", "Defective product", "david@example.com"
    )

    assert success is True
    assert get_refund_status("ORD-1005") == "approved"


def test_large_refund_requires_approval(refund_service):
    success, message = refund_service.request_refund(
        "ORD-1001", "Defective product", "james@example.com"
    )

    assert success is True
    assert get_refund_status("ORD-1001") == "pending_approval"


def test_cannot_refund_shipped_order(refund_service):
    success, message = refund_service.request_refund(
        "ORD-1002", "Changed my mind", "sarah@example.com"
    )

    assert success is False
    assert get_refund_status("ORD-1002") is None


def test_cannot_refund_expired_order(refund_service):
    success, message = refund_service.request_refund(
        "ORD-1008", "Changed my mind", "sophia@example.com"
    )

    assert success is False
    assert get_refund_status("ORD-1008") is None


def test_cannot_refund_another_customers_order(refund_service):
    success, message = refund_service.request_refund(
        "ORD-1005", "Defective product", "sarah@example.com"
    )

    assert success is False
    assert get_refund_status("ORD-1005") is None


def test_cannot_request_duplicate_refund(refund_service):
    refund_service.request_refund(
        "ORD-1005", "Defective product", "david@example.com"
    )

    success, message = refund_service.request_refund(
        "ORD-1005", "Defective product", "david@example.com"
    )

    assert success is False
    assert "already exists" in message.lower()

    conn = database.get_connection()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM refunds WHERE order_id = ?",
            ("ORD-1005",),
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_agent_refund_end_to_end(refund_service, monkeypatch):
    """Test the complete refund workflow using an isolated database."""

    # Simulate an authenticated demo customer.
    monkeypatch.setattr(
        settings, "demo_customer_email", "david@example.com"
    )

    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "I want a refund for order ORD-1005 "
                    "because the product is defective."
                )
            )
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

    # Check intent classification.
    assert result["intent"] == "refund_request"
    assert result["order_id"] == "ORD-1005"

    # Check whether the correct tool was executed.
    assert "request_refund" in result["tool_results"]

    # Verify the actual SQLite state.
    assert get_refund_status("ORD-1005") == "approved"


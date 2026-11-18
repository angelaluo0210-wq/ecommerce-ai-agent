
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.refund_approval import build_approval_graph
from src.api import routes
from src.config import settings
from src.services import database
from src.services.returns_service import ReturnsService


@pytest.mark.parametrize(
    "decision, expected_status",
    [
        ("approve", "approved"),
        ("reject", "rejected"),
    ],
)
def test_approval_api(
    tmp_path, monkeypatch, decision, expected_status
):
    # Create an isolated SQLite database.
    monkeypatch.setattr(
        database, "DB_PATH", tmp_path / "test_ecommerce.db"
    )
    database.init_db()

    # Simulate James as the authenticated demo customer.
    monkeypatch.setattr(
        settings, "demo_customer_email", "james@example.com"
    )

    # Create a high-value refund request.
    success, _ = ReturnsService().request_refund(
        "ORD-1001",
        "Defective product",
        "james@example.com",
    )
    assert success is True

    # Use an isolated checkpointer.
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_approval_graph().compile(
            checkpointer=saver
        )
        monkeypatch.setattr(routes, "approval_graph", graph)

        # Create a test FastAPI application.
        app = FastAPI()
        app.include_router(routes.router, prefix="/api/v1")
        client = TestClient(app)

        # Step 1: Start the approval workflow.
        response = client.post(
            "/api/v1/refunds/ORD-1001/approval/start"
        )

        assert response.status_code == 200
        assert (
            response.json()["status"]
            == "waiting_for_approval"
        )
        assert response.json()["interrupted"] is True

        # Confirm refund has not been approved yet.
        conn = database.get_connection()
        try:
            status = conn.execute(
                "SELECT status FROM refunds WHERE order_id = ?",
                ("ORD-1001",),
            ).fetchone()["status"]
            assert status == "pending_approval"
        finally:
            conn.close()

        # Step 2: Submit the human decision.
        response = client.post(
            "/api/v1/refunds/ORD-1001/approval/decision",
            json={"decision": decision},
        )

        assert response.status_code == 200
        assert response.json()["status"] == expected_status

        # Step 3: Verify SQLite was updated.
        conn = database.get_connection()
        try:
            status = conn.execute(
                "SELECT status FROM refunds WHERE order_id = ?",
                ("ORD-1001",),
            ).fetchone()["status"]
            assert status == expected_status
        finally:
            conn.close()


@pytest.fixture
def approval_client(tmp_path, monkeypatch):
    """Prepare an isolated FastAPI approval test client."""
    monkeypatch.setattr(
        database, "DB_PATH", tmp_path / "approval_test.db"
    )
    database.init_db()

    monkeypatch.setattr(
        settings, "demo_customer_email", "james@example.com"
    )

    success, _ = ReturnsService().request_refund(
        "ORD-1001",
        "Defective product",
        "james@example.com",
    )
    assert success

    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_approval_graph().compile(
            checkpointer=saver
        )
        monkeypatch.setattr(routes, "approval_graph", graph)

        app = FastAPI()
        app.include_router(routes.router, prefix="/api/v1")

        with TestClient(app) as client:
            yield client


def test_start_nonexistent_refund(approval_client):
    response = approval_client.post(
        "/api/v1/refunds/ORD-9999/approval/start"
    )
    assert response.status_code == 404


def test_decision_without_interrupt(approval_client):
    response = approval_client.post(
        "/api/v1/refunds/ORD-1001/approval/decision",
        json={"decision": "approve"},
    )
    assert response.status_code == 409


def test_reject_invalid_decision(approval_client):
    response = approval_client.post(
        "/api/v1/refunds/ORD-1001/approval/decision",
        json={"decision": "invalid"},
    )
    assert response.status_code == 422


def test_prevent_duplicate_approval(approval_client):
    base = "/api/v1/refunds/ORD-1001/approval"

    start = approval_client.post(f"{base}/start")
    assert start.status_code == 200

    first = approval_client.post(
        f"{base}/decision",
        json={"decision": "approve"},
    )
    assert first.status_code == 200

    second = approval_client.post(
        f"{base}/decision",
        json={"decision": "approve"},
    )
    assert second.status_code == 409

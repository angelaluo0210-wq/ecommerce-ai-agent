
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agent.refund_approval import build_approval_graph
from src.services import database
from src.services.returns_service import ReturnsService


@pytest.mark.parametrize(
    "decision, expected_status",
    [
        ("approve", "approved"),
        ("reject", "rejected"),
    ],
)
def test_human_approval_workflow(
    tmp_path, monkeypatch, decision, expected_status
):
    # 1. Create an isolated database.
    monkeypatch.setattr(
        database,
        "DB_PATH",
        tmp_path / "test_ecommerce.db",
    )
    database.init_db()

    # 2. Create a refund requiring human approval.
    success, message = ReturnsService().request_refund(
        "ORD-1001",
        "Defective product",
        "james@example.com",
    )
    assert success is True

    def get_status():
        conn = database.get_connection()
        try:
            row = conn.execute(
                "SELECT status FROM refunds WHERE order_id = ?",
                ("ORD-1001",),
            ).fetchone()
            return row["status"]
        finally:
            conn.close()

    assert get_status() == "pending_approval"

    # 3. Create an isolated LangGraph checkpointer.
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_approval_graph().compile(
            checkpointer=saver
        )

        config = {
            "configurable": {
                "thread_id": "test-refund-1001"
            }
        }

        # 4. Start workflow and pause for approval.
        result = graph.invoke(
            {
                "order_id": "ORD-1001",
                "decision": "",
                "result": "",
            },
            config=config,
        )

        assert graph.get_state(config).next == (
            "human_approval",
        )
        assert get_status() == "pending_approval"

        # 5. Resume with a human decision.
        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

        # 6. Verify the actual database state.
        assert get_status() == expected_status
        assert expected_status in result["result"]
        assert not graph.get_state(config).next

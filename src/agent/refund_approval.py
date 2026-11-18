
"""Human approval workflow for high-value refunds."""

import sqlite3
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from src.services.database import get_connection


class ApprovalState(TypedDict):
    order_id: str
    decision: str
    result: str


def approval_node(state: ApprovalState) -> dict:
    """Pause until a human makes a decision."""
    order_id = state["order_id"]

    conn = get_connection()
    try:
        refund = conn.execute(
            """
            SELECT amount, status FROM refunds
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    finally:
        conn.close()

    if refund is None or refund["status"] != "pending_approval":
        raise ValueError("No pending refund approval for this order.")

    decision = interrupt({
        "order_id": order_id,
        "amount": refund["amount"],
        "message": "Human approval required.",
    })

    if decision not in ("approve", "reject"):
        raise ValueError("Decision must be approve or reject.")

    return {"decision": decision}


def finalize_node(state: ApprovalState) -> dict:
    """Persist the human decision in SQLite."""
    order_id = state["order_id"]
    decision = state["decision"]

    new_status = (
        "approved" if decision == "approve"
        else "rejected"
    )

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            UPDATE refunds
            SET status = ?
            WHERE order_id = ?
              AND status = 'pending_approval'
            """,
            (new_status, order_id),
        )

        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(
                "Refund is not pending or was already processed."
            )

        conn.commit()
    finally:
        conn.close()

    return {
        "result": f"Refund for {order_id}: {new_status}"
    }


def build_approval_graph():
    workflow = StateGraph(ApprovalState)

    workflow.add_node("human_approval", approval_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "human_approval")
    workflow.add_edge("human_approval", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "approval_checkpoints.db"
)

CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

checkpoint_connection = sqlite3.connect(
    str(CHECKPOINT_PATH),
    check_same_thread=False,
)

checkpointer = SqliteSaver(checkpoint_connection)

approval_graph = build_approval_graph().compile(
    checkpointer=checkpointer
)

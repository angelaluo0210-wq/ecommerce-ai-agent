"""
FastAPI Routes — the Agent Gateway.
"""
from fastapi import APIRouter

from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.observability.langfuse_setup import get_langfuse_client, get_langfuse_handler
from langchain_core.messages import HumanMessage, SystemMessage

from typing import Literal
from pydantic import BaseModel
from fastapi import HTTPException
from langgraph.types import Command

from src.agent.refund_approval import approval_graph
from src.services.database import get_connection
from src.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint: runs the full ReAct agent graph."""

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=(
                "IMPORTANT LANGUAGE RULE: You are an English-only assistant. "
                "You MUST write all responses in English. Do NOT use Turkish, "
                "even if the customer writes in Turkish. Always reply in English."
            )),
            HumanMessage(content=req.message),
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

    # Build invoke config with Langfuse callback
    invoke_config = {}
    handler = get_langfuse_handler()
    if handler:
        invoke_config["callbacks"] = [handler]

    # Run the graph
    result = agent_graph.invoke(initial_state, invoke_config)


    # Automatically start human approval for high-value refunds.
    tool_results = result.get("tool_results", {})
    refund_result = tool_results.get("request_refund", "")
    order_id = result.get("order_id", "").strip().upper()

    if order_id and refund_result.startswith("✅"):
        conn = get_connection()
        try:
            refund = conn.execute(
                """
                SELECT r.status, r.amount, o.email
                FROM refunds AS r
                JOIN orders AS o ON r.order_id = o.order_id
                WHERE r.order_id = ?
                """,
                (order_id,),
            ).fetchone()
        finally:
            conn.close()

        if (
            refund is not None
            and refund["status"] == "pending_approval"
            and refund["email"].lower()
            == settings.demo_customer_email.lower()
        ):
            config = {
                "configurable": {
                    "thread_id": f"refund-{order_id}"
                }
            }

            snapshot = approval_graph.get_state(config)

            if not snapshot.next and not snapshot.values:
                approval_graph.invoke(
                    {
                        "order_id": order_id,
                        "decision": "",
                        "result": "",
                    },
                    config=config,
                )


    # Flush traces to Langfuse
    client = get_langfuse_client()
    if client:
        client.flush()

    return ChatResponse(
        response=result.get("final_response", "Sorry, something went wrong."),
        intent=result.get("intent", "general"),
        order_id=result.get("order_id", ""),
        tracking_number=result.get("tracking_number", ""),
        customer_email=result.get("customer_email", ""),
        tool_results=result.get("tool_results", {}),
    )



class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]


def check_pending_refund(order_id: str):
    """Check refund status and demo customer ownership."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT r.order_id, r.amount, r.status, o.email
            FROM refunds AS r
            JOIN orders AS o ON r.order_id = o.order_id
            WHERE r.order_id = ?
            """,
            (order_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Refund request not found.",
            )

        if row["email"].lower() != settings.demo_customer_email.lower():
            raise HTTPException(
                status_code=403,
                detail="Refund does not belong to the demo customer.",
            )

        if row["status"] != "pending_approval":
            raise HTTPException(
                status_code=409,
                detail=f"Refund status is {row['status']}.",
            )

        return dict(row)
    finally:
        conn.close()


@router.post("/refunds/{order_id}/approval/start")
def start_refund_approval(order_id: str):
    """Start or retrieve a pending approval workflow."""
    order_id = order_id.strip().upper()
    refund = check_pending_refund(order_id)

    config = {
        "configurable": {
            "thread_id": f"refund-{order_id}"
        }
    }

    snapshot = approval_graph.get_state(config)

    if snapshot.next:
        return {
            "order_id": order_id,
            "status": "waiting_for_approval",
            "amount": refund["amount"],
        }

    if snapshot.values:
        raise HTTPException(
            status_code=409,
            detail="An approval workflow already exists for this order.",
        )

    result = approval_graph.invoke(
        {
            "order_id": order_id,
            "decision": "",
            "result": "",
        },
        config=config,
    )

    return {
        "order_id": order_id,
        "status": "waiting_for_approval",
        "amount": refund["amount"],
        "interrupted": "__interrupt__" in result,
    }


@router.post("/refunds/{order_id}/approval/decision")
def decide_refund_approval(
    order_id: str,
    body: ApprovalDecision,
):
    """Resume a paused approval workflow."""
    order_id = order_id.strip().upper()
    check_pending_refund(order_id)

    config = {
        "configurable": {
            "thread_id": f"refund-{order_id}"
        }
    }

    snapshot = approval_graph.get_state(config)

    if snapshot.values.get("order_id") != order_id or (
        "human_approval" not in snapshot.next
    ):
        raise HTTPException(
            status_code=409,
            detail="No paused approval workflow for this order.",
        )

    result = approval_graph.invoke(
        Command(resume=body.decision),
        config=config,
    )

    return {
        "order_id": order_id,
        "decision": body.decision,
        "status": (
            "approved" if body.decision == "approve"
            else "rejected"
        ),
        "result": result.get("result", ""),
    }

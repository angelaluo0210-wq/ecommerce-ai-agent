"""Commerce AI Agent baseline evaluation (run from project root).

Usage:
  python -m evaluation.run_evaluation --validate
  python -m evaluation.run_evaluation --limit 3 --no-save
  python -m evaluation.run_evaluation

Each case uses fresh order/refund data and an isolated approval checkpointer.
No production payment operations are performed; OpenAI calls are billed normally.
"""
import argparse
import json
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "scenarios.json"
RESULT_DIR = ROOT / "results"


def load_scenarios():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 40:
        raise ValueError("Expected exactly 40 scenarios")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scenario IDs")
    required = ("id", "category", "message", "customer_email", "expected_tool")
    for c in cases:
        for key in required:
            if key not in c:
                raise ValueError(f"{c.get('id')}: missing {key}")
    return cases


def read_state(database, order_id):
    conn = database.get_connection()
    try:
        order = None
        refund = None
        if order_id:
            row = conn.execute("SELECT status FROM orders WHERE order_id=?", (order_id,)).fetchone()
            order = row["status"] if row else None
            row = conn.execute("SELECT status FROM refunds WHERE order_id=?", (order_id,)).fetchone()
            refund = row["status"] if row else None
        all_orders = dict(conn.execute("SELECT order_id, status FROM orders").fetchall())
        all_refunds = [tuple(r) for r in conn.execute(
            "SELECT order_id, status, amount FROM refunds ORDER BY order_id"
        ).fetchall()]
        return {"order_status": order, "refund_status": refund,
                "all_orders": all_orders, "all_refunds": all_refunds}
    finally:
        conn.close()


def evaluate_case(case):
    # Import only at execution time so --validate needs no project dependencies.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.sqlite import SqliteSaver
    from src.api import routes
    from src.agent.refund_approval import build_approval_graph
    from src.config import settings
    from src.services import database
    from src.services.returns_service import ReturnsService

    order_id = case.get("expected_order_id")
    result = {"id": case["id"], "category": case["category"],
              "message": case["message"], "expected_tool": case.get("expected_tool"),
              "expected_intent": case.get("expected_intent"),
              "expected_behavior": case.get("expected_behavior"),
              "expected_order_status": case.get("expected_order_status"),
              "expected_refund_status": case.get("expected_refund_status"),
              "human_decision": case.get("human_decision"),
              "status": "error", "tool_match": None, "intent_match": None,
              "task_success": None, "policy_violation": None,
              "hitl_compliant": None, "manual_review": case["category"] == "ambiguous"}

    old_db = database.DB_PATH
    old_email = settings.demo_customer_email
    old_graph = routes.approval_graph
    try:
        with tempfile.TemporaryDirectory(prefix="commerce_eval_") as tmp:
            database.DB_PATH = Path(tmp) / "orders.db"
            settings.demo_customer_email = case["customer_email"]
            database.init_db()

            if case.get("setup") == "existing_refund_approved":
                setup_ok, setup_msg = ReturnsService().request_refund(
                    "ORD-1005", "Existing approved refund", "david@example.com"
                )
                if not setup_ok:
                    raise RuntimeError("Scenario setup failed: " + setup_msg)
            elif case.get("setup"):
                raise ValueError("Unknown setup: " + str(case["setup"]))

            before = read_state(database, order_id)
            with SqliteSaver.from_conn_string(":memory:") as saver:
                routes.approval_graph = build_approval_graph().compile(checkpointer=saver)
                app = FastAPI()
                app.include_router(routes.router, prefix="/api/v1")
                with TestClient(app) as client:
                    http = client.post("/api/v1/chat", json={
                        "message": case["message"], "session_id": "eval-" + case["id"]
                    })
                    result["http_status"] = http.status_code
                    if http.status_code != 200:
                        result["error"] = http.text[:500]
                        return result
                    body = http.json()
                    result["intent"] = body.get("intent")
                    result["response"] = body.get("response", "")
                    result["tool_results"] = body.get("tool_results", {})
                    actual_tools = list(result["tool_results"])
                    expected_tool = case.get("expected_tool")
                    result["tool_match"] = (expected_tool in actual_tools) if expected_tool else None
                    result["intent_match"] = (
                        body.get("intent") == case["expected_intent"]
                        if case.get("expected_intent") is not None else None
                    )
                    result["actual_tools"] = actual_tools

                    paused = False
                    if order_id:
                        config = {"configurable": {"thread_id": "refund-" + order_id}}
                        snapshot = routes.approval_graph.get_state(config)
                        paused = "human_approval" in snapshot.next
                    result["paused_before_decision"] = paused
                    result["status_before_decision"] = read_state(database, order_id)["refund_status"]

                    if case.get("human_decision"):
                        if paused:
                            response = client.post(
                                f"/api/v1/refunds/{order_id}/approval/decision",
                                json={"decision": case["human_decision"]}
                            )
                            result["approval_http_status"] = response.status_code
                            result["approval_response"] = response.json()
                        else:
                            result["approval_http_status"] = None
                    after = read_state(database, order_id)
                    result["actual_order_status"] = after["order_status"]
                    result["actual_refund_status"] = after["refund_status"]
                    result["any_database_mutation"] = (
                        before["all_orders"] != after["all_orders"]
                        or before["all_refunds"] != after["all_refunds"]
                    )

                    expected_order = case.get("expected_order_status")
                    expected_refund = case.get("expected_refund_status")
                    order_ok = expected_order is None or after["order_status"] == expected_order
                    refund_ok = after["refund_status"] == expected_refund
                    behavior = case.get("expected_behavior", "")
                    tools = result["tool_results"]
                    tool_text = " ".join(str(v) for v in tools.values()).lower()
                    response_text = result["response"].lower()

                    # A behavioral check must be based on state and/or tool output,
                    # never merely on the LLM's claim of success.
                    denied = ("❌" in str(tools) or any(word in tool_text for word in (
                        "not authorized", "cannot", "not found", "already exists",
                        "only delivered", "expired", "not eligible")))
                    mutation_forbidden = behavior.startswith("deny_") or behavior in (
                        "report_not_found", "ask_for_order_id_no_mutation",
                        "ask_clarification_no_mutation", "inform_or_clarify_no_mutation",
                        "ask_clarification_or_lookup"
                    )
                    unchanged = not result["any_database_mutation"]
                    if case["category"] == "ambiguous":
                        result["task_success"] = None  # qualitative review required
                        result["policy_violation"] = not unchanged
                    elif case["category"] == "hitl":
                        desired = case.get("human_decision")
                        hitl_ok = (
                            paused and result["status_before_decision"] == "pending_approval"
                            and (not desired or result.get("approval_http_status") == 200)
                            and refund_ok and order_ok
                        )
                        result["hitl_compliant"] = hitl_ok
                        result["task_success"] = hitl_ok and result["tool_match"]
                        result["policy_violation"] = not hitl_ok
                    else:
                        if mutation_forbidden:
                            behavior_ok = unchanged and (denied or "not found" in response_text)
                        elif behavior == "report_tracking":
                            behavior_ok = bool(result["tool_match"] and "tracking" in (tool_text + response_text))
                        elif behavior == "describe_policy":
                            behavior_ok = bool(result["tool_match"] and ("return" in (tool_text + response_text)))
                        else:
                            behavior_ok = True
                        result["task_success"] = bool(
                            order_ok and refund_ok and behavior_ok and result["tool_match"]
                        )
                        result["policy_violation"] = (not unchanged) if mutation_forbidden else False
                    result["status"] = "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        database.DB_PATH = old_db
        settings.demo_customer_email = old_email
        routes.approval_graph = old_graph
    return result


def summarize(results):
    def rate(field, subset):
        eligible = [r[field] for r in subset if r.get(field) is not None and r["status"] == "ok"]
        return {"passed": sum(v is True for v in eligible), "total": len(eligible),
                "rate": (sum(v is True for v in eligible) / len(eligible) if eligible else None)}
    return {
        "completed": sum(r["status"] == "ok" for r in results),
        "errors": sum(r["status"] != "ok" for r in results),
        "task_success_rate": rate("task_success", results),
        "tool_selection_accuracy": rate("tool_match", results),
        "intent_accuracy": rate("intent_match", results),
        "hitl_compliance": rate("hitl_compliant", results),
        "policy_violation_count": sum(r.get("policy_violation") is True for r in results),
        "ambiguous_manual_review": sum(r.get("manual_review") for r in results),
        "by_category": {k: rate("task_success", [r for r in results if r["category"] == k])
                        for k in sorted(set(r["category"] for r in results))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    cases = load_scenarios()
    print(f"Dataset valid: {len(cases)} scenarios, categories={dict(Counter(c['category'] for c in cases))}")
    if args.validate:
        return
    if not (1 <= args.limit <= len(cases)):
        parser.error("--limit must be between 1 and 40")
    results = []
    for case in cases[:args.limit]:
        r = evaluate_case(case)
        results.append(r)
        print(f"{r['id']} {r['category']}: {r['status']} task={r['task_success']} "
              f"tool={r['tool_match']}" + (f" ERROR={r.get('error')}" if r.get("error") else ""), flush=True)
    summary = summarize(results)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.no_save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = RESULT_DIR / f"baseline_{timestamp}.json"
        output.write_text(json.dumps({"summary": summary, "cases": results}, ensure_ascii=False,
                                     indent=2), encoding="utf-8")
        print("Saved:", output)


if __name__ == "__main__":
    main()

"""
LangGraph Nodes — the three processing stages of the ReAct agent.

  1. triage_node   → classify intent, extract entities分类
  2. tool_node     → execute the relevant backend tools (with retry)执行function-calling
  3. response_node → compose the final customer-facing reply 负责把结果组织成自然语言
"""
import json
import re
import traceback

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS
from src.config import settings

# ---------------------------------------------------------------------------
# Shared LLM instance (lazy)
# ---------------------------------------------------------------------------

_llm = None
_llm_with_tools = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            max_retries=2,
        )
    return _llm


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(ALL_TOOLS) #把项目里的所有 Tool 的名称、描述、参数 schema 提供给 LLM
    return _llm_with_tools

# ---------------------------------------------------------------------------
# 1. TRIAGE NODE
# ---------------------------------------------------------------------------



TRIAGE_PROMPT = """You are an e-commerce customer support triage assistant.
Analyze the customer's message and classify it as JSON:

{
    "intent": "<order_status | shipping_tracking | return_request | refund_request | return_policy | cancel_order | general>",
    "order_id": "<order ID like ORD-1001, empty string if none>",
    "tracking_number": "<tracking number like FDX-78901234, empty string if none>",
    "customer_email": "<email address, empty string if none>",
    "needs_tool": true/false,
    "tool_name": "<best tool to call, empty if none>"
}

Rules:
- "order_status": customer asks about an order's status or details.
- "shipping_tracking": customer asks where a package is.
- "return_request": customer wants to return an item.
- "refund_request": customer explicitly asks for a refund.
- "return_policy": customer asks about return or refund policies.
- "cancel_order": customer explicitly asks to cancel an order.
- "general": greetings, thanks, and chitchat.

Important:
- Distinguish refund requests from return requests.
- Distinguish order cancellation from refunds and returns.
- Questions about eligibility or policy are not requests to execute a refund.
- Do not invent missing order IDs or reasons.
- Return ONLY valid JSON, no additional text."""



def triage_node(state: AgentState) -> dict:
    """Classify the user's intent and extract entities."""
    messages = state["messages"]

    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if not last_user_msg:
        return {"intent": "general", "error_message": ""}

    try:
        response = _get_llm().invoke([
            SystemMessage(content=TRIAGE_PROMPT),
            HumanMessage(content=last_user_msg),
        ])

        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        triage = json.loads(content)

        return {
            "intent": triage.get("intent", "general"),
            "order_id": triage.get("order_id", ""),
            "tracking_number": triage.get("tracking_number", ""),
            "customer_email": triage.get("customer_email", ""),
            "error_message": "",
        }

    except Exception:
        return {
            "intent": "general",
            "order_id": "",
            "tracking_number": "",
            "customer_email": "",
            "error_message": "",
        }


# ---------------------------------------------------------------------------
# 2. TOOL NODE (ReAct "Act" step with retry)
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """Use LLM function-calling to decide and execute the right tool."""
    intent = state.get("intent", "general")
    messages = state.get("messages", [])
    retries = state.get("retry_count", 0)

    if intent == "general":
        return {"tool_results": {}, "retry_count": retries}

    # Deterministic routing for explicit state-changing requests.
    if intent in ("cancel_order", "refund_request"):
        order_id = (state.get("order_id") or "").strip().upper()

        if not order_id:
            return {
                "tool_results": {
                    "missing_information": (
                        "Please provide your order ID "
                        "before I can process this request."
                    )
                },
                "retry_count": retries,
            }

        tool_name = (
            "cancel_order"
            if intent == "cancel_order"
            else "request_refund"
        )

        tool_args = {"order_id": order_id}

        if intent == "refund_request":
            user_message = next(
                (
                    m.content
                    for m in reversed(messages)
                    if isinstance(m, HumanMessage)
                    and isinstance(m.content, str)
                ),
                "",
            )

            if not user_message.strip():
                return {
                    "tool_results": {
                        "missing_information": (
                            "Please explain why you are "
                            "requesting a refund."
                        )
                    },
                    "retry_count": retries,
                }

            tool_args["reason"] = user_message.strip()

        selected_tool = next(
            (tool for tool in ALL_TOOLS if tool.name == tool_name),
            None,
        )

        if selected_tool is None:
            return {
                "tool_results": {
                    "error": f"Tool '{tool_name}' is unavailable."
                },
                "retry_count": retries + 1,
            }

        try:
            result = selected_tool.invoke(tool_args)

            return {
                "tool_results": {tool_name: str(result)},
                "retry_count": retries + 1,
            }

        except Exception as exc:
            return {
                "tool_results": {
                    tool_name: f"❌ Tool execution failed: {exc}"
                },
                "retry_count": retries + 1,
            }


    context_prompt = f"""Customer intent: {intent}

Available context:
- Order ID: {state.get('order_id') or 'unknown'}
- Tracking Number: {state.get('tracking_number') or 'unknown'}
- Email: {state.get('customer_email') or 'unknown'}

Call the appropriate tool with the available parameters.

Tool selection rules:
- If intent is "cancel_order", call cancel_order with the order ID.
- Do not substitute lookup_order for cancel_order.
- The cancel_order tool performs the authoritative status
  and ownership checks, including when cancellation is denied.
- If intent is "refund_request", call request_refund only
  when the customer explicitly requests a refund and
  provides the required order information.
- Do not initiate a return or refund merely because
  the customer reports a defective product.
- If required information is missing, do not execute
  any state-changing tool.
- Do not claim capabilities that the available tools
  cannot perform.
"""

    try:
        ai_msg = _get_llm_with_tools().invoke([
            SystemMessage(content=context_prompt),
            *messages,
        ]) #这里LLM生成一个 Tool Call,需要的tool和参数,生成调用计划”，还没有真正执行函数(ai_msg.tool_calls)

        tool_results = {}

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls: #ai_msg 这个对象有没有 tool_calls 这个属性;即使有这个属性，里面是不是确实有内容
            for tc in ai_msg.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                for tool in ALL_TOOLS:
                    if tool.name == tool_name: #tool.name是真正的python函数，tool_name只是一个str名称
                        try:
                            result = tool.invoke(tool_args) #真正执行函数，返回结果
                            tool_results[tool_name] = str(result)
                        except Exception:
                            tool_results[tool_name] = (
                                f"❌ Error calling '{tool_name}': "
                                f"{traceback.format_exc()[-200:]}"
                            )
                        break
                else: #找不到tool:
                    tool_results[tool_name] = f"❌ Tool '{tool_name}' not found."

        return {"tool_results": tool_results, "retry_count": retries + 1}

    except Exception as e:
        return {
            "tool_results": {"error": str(e)},
            "retry_count": retries + 1,
            "error_message": str(e),
        }


# ---------------------------------------------------------------------------
# 3. RESPONSE NODE
# ---------------------------------------------------------------------------

RESPONSE_PROMPT = """LANGUAGE: You are an English-only customer support agent. Write ONLY in English. Turkish is FORBIDDEN.

Compose a friendly, structured response using the information below.

Customer Intent: {intent}
Tool Results: {tool_results}

Guidelines:
- WRITE IN ENGLISH. Do not use any other language.
- Use the exact data from tool results, do not fabricate.
- Format with bullet points.
- Include order IDs and tracking numbers.
- End with: "Is there anything else I can help you with?"
- If no tools were used, introduce yourself in English and list what you can help with."""


def response_node(state: AgentState) -> dict:
    """Compose the final customer-facing response."""
    intent = state.get("intent", "general")
    tool_results = state.get("tool_results", {})
    messages = state.get("messages", [])
    error = state.get("error_message", "")

    # -- Fallback on persistent error --
    if error and not tool_results:
        return {
            "final_response": (
                "I apologize, but I'm having trouble processing your request right now. "
                "Please try again in a moment, or contact our support team directly at "
                "support@crate.ai. Is there anything else I can help you with?"
            )
        }

    # -- General / greeting --
    if intent == "general" and not tool_results:
        return {
            "final_response": (
                "Hello! 👋 I'm Crate's virtual assistant. I can help you with:\n\n"
                "📦 **Order Status** — Check your order with your order ID.\n"
                "🚚 **Package Tracking** — Track your shipment with your tracking number.\n"
                "🔄 **Returns** — Learn about our return policy or initiate a return.\n\n"
                "How can I help you today?"
            )
        }

    # -- Compose from tool results --
    try:
        formatted_prompt = RESPONSE_PROMPT.format(
            intent=intent,
            tool_results=json.dumps(tool_results, ensure_ascii=False, indent=2),
        )

        response = _get_llm().invoke([
            SystemMessage(content=formatted_prompt),
            *messages,
        ])

        text = response.content.strip()

        # If Turkish leaked through, use raw English tool results directly
        if _contains_turkish(text) and tool_results:
            text = _format_tool_results(intent, tool_results)

        return {"final_response": text}

    except Exception:
        parts = []
        for tool_name, result in tool_results.items():
            parts.append(f"**{tool_name}** result:\n{result}")
        return {"final_response": "\n\n".join(parts)}


def _contains_turkish(text: str) -> bool:
    """Detect if text contains Turkish-specific characters or common Turkish words."""
    turkish_chars = set("ğıİşŞçÇöÖüÜĞ")
    turkish_words = [
        "merhaba", "sipariş", "siparişiniz", "kargonuz", "takip",
        "teslimat", "ürün", "ürünler", "bulabilirsiniz", "teşekkür",
        "yardımcı", "memnuniyetle", "lütfen", "gönderinizi",
    ]
    lower = text.lower()
    if any(c in text for c in turkish_chars):
        return True
    if any(w in lower for w in turkish_words):
        return True
    return False


def _format_tool_results(intent: str, tool_results: dict) -> str:
    """Format raw tool results into a clean English response without LLM."""
    parts = []
    if intent == "order_status":
        parts.append("Here are your order details:\n")
    elif intent == "shipping_tracking":
        parts.append("Here is your shipment status:\n")
    elif intent == "return_request":
        parts.append("Here is your return request status:\n")
    elif intent == "return_policy":
        parts.append("Here is our return policy:\n")
    else:
        parts.append("Here is what I found:\n")

    for result in tool_results.values():
        parts.append(str(result))

    parts.append("\nIs there anything else I can help you with?")
    return "\n".join(parts)

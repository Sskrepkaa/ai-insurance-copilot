"""LangGraph agent: tool-calling loop over the backend tools + a
human-in-the-loop (HITL) checkpoint for high fraud-risk claims.

This module must be importable with NO side effects that require a live
Groq API key: building the graph (nodes, edges, checkpointer) does not
touch the network. The `ChatGroq` client is only constructed lazily, the
first time the "agent" node actually runs (i.e. on the first real
`graph.invoke(...)` call) - so `import agent` works fine even when
GROQ_API_KEY is unset, and any failure surfaces at request time instead.

API notes (checked against the installed versions - see requirements.txt -
because the sketch in 02_prompt_langgraph_agent.md predates them):
  - langgraph==1.2.12 / langgraph-checkpoint-sqlite==3.1.1
  - langchain-core==1.6.5, langchain-groq==1.1.3

- `SqliteSaver.from_conn_string(...)` is now a `@contextmanager` (it closes
  the connection on `__exit__`), which doesn't fit a long-lived FastAPI
  singleton. Instead we open a plain `sqlite3.connect(..., check_same_thread
  =False)` connection ourselves and pass it to `SqliteSaver(conn)` directly
  - this is the pattern shown in the class's own docstring, and it is safe
  because `SqliteSaver` guards all access with an internal `threading.Lock`.
- `interrupt(value)` (from `langgraph.types`) raises a `GraphInterrupt` the
  first time a node runs it; `graph.invoke(...)` then returns a dict whose
  `"__interrupt__"` key holds a list of `Interrupt(value=..., id=...)`
  objects (there is no separate "check for interrupt" API - this was
  verified empirically against the installed version, see the worktree's
  scratch test). Resuming is done via
  `graph.invoke(Command(resume=<payload>), config=...)` with the same
  `thread_id`, which re-runs the interrupted node from its start and makes
  `interrupt(...)` return `<payload>` instead of raising.
- `ToolNode` serializes a tool's returned dict/list into a JSON string on
  `ToolMessage.content` (via `json.dumps`), and sets `ToolMessage.name` to
  the tool's name - both confirmed empirically. `human_check` below relies
  on exactly that.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt  # noqa: F401  (Command re-exported for main.py)

import tools as backend_tools

# Load backend/.env (GROQ_API_KEY=...) if present. Never commit a real key.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

MODEL_NAME = "llama-3.3-70b-versatile"
FRAUD_TOOL_NAME = "check_fraud_risk"
FRAUD_RISK_THRESHOLD = 0.6

AGENT_STATE_DB_PATH = Path(__file__).resolve().parent / "agent_state.db"

SYSTEM_PROMPT = (
    "You are an AI copilot helping an insurance support agent triage claims. "
    "You have tools to look up claim details, check fraud risk, and pull a "
    "customer's claim history. Use the tools whenever you need facts instead "
    "of guessing. Keep answers short and concrete, and mention the "
    "fraud risk_score and level when you looked it up."
)


# --------------------------------------------------------------------------
# Tools - thin @tool wrappers around backend_tools.* so the LLM sees a
# LangChain-compatible tool with a name/schema/docstring, while the actual
# logic (and its DB access) lives in tools.py.
# --------------------------------------------------------------------------


@tool
def get_claim_details(claim_id: str) -> dict:
    """Get the full data for an insurance claim (client, amount, status,
    type, description) by its claim ID, e.g. "CLM-1001"."""
    return backend_tools.get_claim_details(claim_id)


@tool
def check_fraud_risk(claim_id: str) -> dict:
    """Compute a fraud risk score for a claim by its claim ID. Returns a
    dict with "risk_score" (0-1 float) and "level" ("low"/"medium"/"high")."""
    return backend_tools.check_fraud_risk(claim_id)


@tool
def get_customer_history(client: str) -> list:
    """Get the list of past claims/outcomes for a customer by their full
    name, e.g. "Ivan Petrov"."""
    return backend_tools.get_customer_history(client)


all_tools = [get_claim_details, check_fraud_risk, get_customer_history]

# --------------------------------------------------------------------------
# LLM - constructed lazily so `import agent` never requires GROQ_API_KEY.
# --------------------------------------------------------------------------

_llm_with_tools = None


def _get_llm_with_tools():
    """Build (once) and return the Groq chat model bound to `all_tools`.

    Deliberately NOT called at import time: `ChatGroq(...)` validates
    GROQ_API_KEY eagerly in its constructor and raises immediately if it's
    missing, so we defer that call to here, which only runs once the graph
    actually executes the "agent" node (i.e. on a real /chat or /resume
    request).
    """
    global _llm_with_tools
    if _llm_with_tools is None:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=MODEL_NAME, temperature=0)
        _llm_with_tools = llm.bind_tools(all_tools)
    return _llm_with_tools


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------


def call_model(state: MessagesState) -> dict:
    """Agent node: call the LLM (with tools bound) on the running
    conversation. A system prompt is prepended for this call only - it is
    not persisted into state, so it isn't duplicated across turns."""
    llm_with_tools = _get_llm_with_tools()
    from langchain_core.messages import SystemMessage

    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """Route to the tool executor while the LLM keeps requesting tool
    calls; once it stops, fall through to the human-in-the-loop check."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "human_check"


def _latest_fraud_risk_result(messages) -> dict | None:
    """Scan backwards for the most recent ToolMessage produced by the
    check_fraud_risk tool and return its parsed payload, or None if that
    tool was never called (or its content couldn't be parsed)."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == FRAUD_TOOL_NAME:
            content = msg.content
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
            return None
    return None


def human_check(state: MessagesState) -> dict:
    """HITL node. If the latest check_fraud_risk result has risk_score
    above the threshold, pause the graph with `interrupt(...)` and surface
    a question for the human expert. Otherwise, do nothing and let the
    graph finish.

    On resume, LangGraph re-runs this node from the top: the same
    ToolMessage lookup happens again, but this time `interrupt(...)`
    returns the human's decision (a dict: {"approved": bool, "reason":
    str}) instead of raising, and we record it as a HumanMessage.
    """
    fraud_result = _latest_fraud_risk_result(state["messages"])
    if fraud_result and fraud_result.get("risk_score", 0) > FRAUD_RISK_THRESHOLD:
        decision = interrupt(
            {
                "question": (
                    f"Risk score {fraud_result['risk_score']} detected. "
                    "Confirm the expert's decision?"
                )
            }
        )
        approved = decision.get("approved") if isinstance(decision, dict) else None
        reason = decision.get("reason", "") if isinstance(decision, dict) else ""
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"Expert decision recorded: approved={approved}, "
                        f"reason={reason!r}"
                    )
                )
            ]
        }
    return {"messages": []}


# --------------------------------------------------------------------------
# Graph assembly - safe to do at import time, no network calls involved.
# --------------------------------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("human_check", human_check)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "human_check": "human_check"}
)
builder.add_edge("tools", "agent")
builder.add_edge("human_check", END)

# A single long-lived sqlite connection + SqliteSaver for the whole process.
# check_same_thread=False + SqliteSaver's internal lock make this safe to
# share across the threadpool FastAPI dispatches sync endpoints onto.
_conn = sqlite3.connect(str(AGENT_STATE_DB_PATH), check_same_thread=False)
checkpointer = SqliteSaver(_conn)

graph = builder.compile(checkpointer=checkpointer)


def extract_final_text(messages) -> str:
    """Pull the agent's final textual answer out of a message list: the
    last AIMessage with non-empty string content, if any."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            return msg.content
    # Fallback: last message with any non-empty string content at all.
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content:
            return content
    return ""


def extract_interrupt_question(interrupt_payload) -> str:
    """`result["__interrupt__"]` is a list of `Interrupt` objects; pull the
    human-readable question out of the first one's `.value`."""
    first = interrupt_payload[0]
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return value.get("question", str(value))
    return str(value)

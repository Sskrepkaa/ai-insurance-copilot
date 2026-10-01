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

HITL trigger design (fixed after a live bug report): the first version
decided to pause for human approval by scanning the *entire* thread history
for any past check_fraud_risk result above the threshold. That meant once a
claim had ever been flagged high-risk, every later unrelated question in the
same thread (e.g. "has this client filed a similar claim before?") wrongly
re-triggered an Approve/Reject interrupt. The fix is twofold:
  1. Intent, not data presence: the LLM must explicitly call a dedicated
     `request_human_approval` tool to signal "I am now recommending an
     approve/reject decision" - merely having called check_fraud_risk
     (now or earlier) is not by itself a trigger.
  2. Turn-scoped, not thread-scoped: the routing after the tools node only
     looks at the batch of ToolMessages produced by the *current* tool-call
     round (`_last_tool_batch`), never at older messages further back in
     the thread.
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
from langchain_core.runnables import RunnableConfig

import tools as backend_tools

# Load backend/.env (GROQ_API_KEY=...) if present. Never commit a real key.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

MODEL_NAME = "openai/gpt-oss-120b"
FRAUD_RISK_THRESHOLD = 0.6
REQUEST_APPROVAL_TOOL_NAME = "request_human_approval"

AGENT_STATE_DB_PATH = Path(__file__).resolve().parent / "agent_state.db"

SYSTEM_PROMPT = (
    "You are an AI copilot helping an insurance support agent triage claims. "
    "You have tools to look up claim details, check fraud risk, and pull a "
    "customer's claim history. Use the tools whenever you need facts instead "
    "of guessing. Keep answers short and concrete, and mention the "
    "fraud risk_score and level when you looked it up.\n\n"
    "If get_customer_history returns an empty list, or doesn't contain the "
    "specific claim/date/status the employee asked about, say explicitly "
    "that there is no such history - do NOT invent a claim id, date, or "
    "outcome that isn't literally present in the tool's output. The same "
    "goes for every other tool: never state a specific fact (an id, "
    "amount, date, status, name) that didn't come from a tool result.\n\n"
    f"If, after checking fraud risk, the risk_score is above "
    f"{FRAUD_RISK_THRESHOLD} AND you are about to recommend approving or "
    f"rejecting the claim to the employee, you MUST call the "
    f"{REQUEST_APPROVAL_TOOL_NAME} tool with your recommendation instead of "
    "stating it directly in text - the employee has to confirm it first. "
    f"Do NOT call {REQUEST_APPROVAL_TOOL_NAME} just to report a risk score, "
    "answer a history question, or for anything else where you are not "
    "actually proposing an approve/reject decision right now. If the "
    "expert's decision has already been recorded earlier in this "
    f"conversation, do not call {REQUEST_APPROVAL_TOOL_NAME} again for the "
    "same recommendation - just summarize the outcome, restating the claim "
    "id and the risk_score/level you already found so the summary stands "
    "on its own."
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
def get_customer_history(client: str) -> dict:
    """Get the list of past claims/outcomes for a customer by their full
    name, e.g. "Ivan Petrov". Returns {"history": [...]} - an empty
    "history" list means this customer has no past claims on record; say
    so explicitly rather than inventing one."""
    # Wrapped in a dict rather than returned as a bare list: langgraph's
    # ToolNode (msg_content_output) decides whether to JSON-serialize a
    # list result by checking `all(... for x in output)`, which is
    # vacuously True for an EMPTY list - so an empty list comes out as
    # raw content=[] instead of the string "[]", and Groq's API then
    # rejects the message ("minimum number of items is 1"). A dict never
    # hits that list branch at all, empty or not, so this sidesteps the
    # bug entirely rather than special-casing the empty case.
    return {"history": backend_tools.get_customer_history(client)}


@tool
def request_human_approval(claim_id: str, risk_score: float, recommendation: str) -> dict:
    """Call this ONLY when you are actively recommending that the employee
    approve or reject a claim (recommendation: "approve" or "reject") after
    finding its fraud risk_score above the threshold, and you need their
    confirmation before the recommendation counts as final. Do NOT call this
    just to report a risk score or to answer an informational question
    (e.g. claim history) - it is exclusively for "I am proposing a decision
    right now, please confirm it." This tool is internal to the agent's
    control flow, not one of the backend's data tools - it has no side
    effects and simply signals intent; the actual confirmation happens via
    the graph's human-in-the-loop interrupt.
    """
    return {"claim_id": claim_id, "risk_score": risk_score, "recommendation": recommendation}


all_tools = [
    get_claim_details,
    check_fraud_risk,
    get_customer_history,
    request_human_approval,
]

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


def call_model(state: MessagesState, config: RunnableConfig) -> dict:
    """Agent node: call the LLM (with tools bound) on the running
    conversation. A system prompt is prepended for this call only - it is
    not persisted into state, so it isn't duplicated across turns.

    `config["configurable"]["thread_id"]` is the claim_id main.py used to
    start/resume this thread (see chat()/resume()). The user's message text
    (e.g. "check this claim for fraud") never mentions the claim_id itself,
    so without surfacing it here the model has no way to know which claim
    "this" refers to and can't call the claim_id-taking tools correctly.
    """
    llm_with_tools = _get_llm_with_tools()
    from langchain_core.messages import SystemMessage

    claim_id = (config or {}).get("configurable", {}).get("thread_id")
    system_text = SYSTEM_PROMPT
    if claim_id:
        system_text += (
            f" The employee is currently looking at claim '{claim_id}' - use "
            "this as the claim_id argument for tools unless the user clearly "
            "asks about a different claim."
        )

    messages = [SystemMessage(content=system_text), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """Route to the tool executor while the LLM keeps requesting tool
    calls; once it stops calling tools, the turn is over."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def _last_tool_batch(messages) -> list:
    """Return only the ToolMessages produced by the *most recent* AIMessage's
    tool_calls - i.e. the current tool-call round, not the whole thread
    history. This is what makes HITL routing turn-scoped: a
    request_human_approval call from three questions ago must not still
    count now."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[i + 1 :]
    return []


def _parse_tool_message_content(msg: ToolMessage) -> dict | None:
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


def route_after_tools(state: MessagesState) -> str:
    """After the tools node runs, pause for human-in-the-loop only if the
    agent just (this round) explicitly called request_human_approval -
    never merely because check_fraud_risk returned a high score, and never
    because of something called in an earlier turn. Otherwise loop back to
    the agent so it can use the tool result(s) to keep answering."""
    last_batch = _last_tool_batch(state["messages"])
    if any(
        isinstance(m, ToolMessage) and m.name == REQUEST_APPROVAL_TOOL_NAME
        for m in last_batch
    ):
        return "human_check"
    return "agent"


def human_check(state: MessagesState) -> dict:
    """HITL node, reached only via an explicit request_human_approval call
    in the current tool-call round (see route_after_tools). Pause with
    `interrupt(...)` and surface the agent's recommendation to the human
    expert.

    On resume, LangGraph re-runs this node from the top: the same lookup
    happens again, but this time `interrupt(...)` returns the human's
    decision (a dict: {"approved": bool, "reason": str}) instead of
    raising, and we record it as a HumanMessage before looping back to the
    agent so it can compose a final answer that reflects the decision.
    """
    last_batch = _last_tool_batch(state["messages"])
    approval_msg = next(
        (m for m in last_batch if isinstance(m, ToolMessage) and m.name == REQUEST_APPROVAL_TOOL_NAME),
        None,
    )
    payload = _parse_tool_message_content(approval_msg) if approval_msg else None
    payload = payload or {}

    question = (
        f"Risk score {payload.get('risk_score')} detected. The agent "
        f"recommends to {payload.get('recommendation', 'approve/reject')} "
        f"claim {payload.get('claim_id')}. Confirm the expert's decision?"
    )
    decision = interrupt({"question": question})
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


# --------------------------------------------------------------------------
# Graph assembly - safe to do at import time, no network calls involved.
# --------------------------------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("human_check", human_check)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_conditional_edges(
    "tools", route_after_tools, {"human_check": "human_check", "agent": "agent"}
)
builder.add_edge("human_check", "agent")

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


def extract_tool_outputs(messages) -> list[dict]:
    """Collect every ToolMessage in `messages` (the graph's full message
    history for this thread, as returned by `graph.invoke(...)["messages"]`
    or `graph.get_state(config).values["messages"]`) as
    `[{"tool": <tool name>, "args": <call arguments>, "output": <parsed JSON
    or raw content>}, ...]`.

    `args` matters as much as `output` for the judge: e.g. check_fraud_risk
    only *returns* {"risk_score": ..., "level": ...} - the claim_id it was
    asked about never appears in its output. Without surfacing the call
    arguments too, the judge sees the agent state "risk_score 0.16 for
    CLM-1002" and flags "CLM-1002" as fabricated, even though it's exactly
    the id the tool was invoked with - a false positive. We recover the
    arguments by matching each ToolMessage.tool_call_id back to the
    `tool_calls` entry with the same id on the preceding AIMessage.

    Used by main.py to feed judge.judge_response() the real data the
    agent's tools returned, per 03_prompt_ai_judge_agent.md. `ToolNode`
    JSON-serializes each tool's dict/list result into `ToolMessage.content`
    (see the module docstring above), so we reverse that with `json.loads`;
    if a given message's content isn't valid JSON for some reason, we fall
    back to passing the raw string through rather than dropping it.
    """
    call_args_by_id: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", None) or []:
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
                if call_id is not None:
                    call_args_by_id[call_id] = args or {}

    outputs = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = content
        else:
            parsed = content
        outputs.append(
            {
                "tool": msg.name,
                "args": call_args_by_id.get(getattr(msg, "tool_call_id", None), {}),
                "output": parsed,
            }
        )
    return outputs


def extract_original_question(messages) -> str:
    """Pull the first HumanMessage's content out of `messages` - the
    original question the employee asked when this thread started.

    Used for POST /resume, where the incoming request only carries the
    human's approve/reject decision, not the original chat message: the
    checkpointer still has it in the thread's persisted message history,
    so we recover it from there instead of passing an empty string to the
    judge.
    """
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str) and msg.content:
            return msg.content
    return ""

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

import ast
import json
import os
import re
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


def _current_approval_request(messages) -> tuple[ToolMessage | None, dict]:
    """Find the request_human_approval ToolMessage in the *current*
    tool-call round (if any) and its parsed payload."""
    last_batch = _last_tool_batch(messages)
    approval_msg = next(
        (m for m in last_batch if isinstance(m, ToolMessage) and m.name == REQUEST_APPROVAL_TOOL_NAME),
        None,
    )
    payload = (_parse_tool_message_content(approval_msg) if approval_msg else None) or {}
    return approval_msg, payload


def _message_index(messages, target) -> int:
    """Index of `target` in `messages` by identity (not equality, since two
    distinct messages can have equal content)."""
    for i, m in enumerate(messages):
        if m is target:
            return i
    return len(messages)


def _normalize_recommendation(value) -> str:
    """Reduce a recommendation to just its "approve"/"reject" keyword
    (case-insensitive), ignoring any extra text the model tacks on - e.g.
    "reject due to updated risk assessment" still matches a prior plain
    "reject". request_human_approval's own docstring tells the model to
    pass exactly "approve" or "reject", but nothing enforces that at the
    tool-call boundary, so the comparison in route_after_tools() needs to
    tolerate a rephrase rather than treating it as a brand-new decision
    just because the exact string differs. Falls back to the stripped,
    lowercased original text if neither keyword appears, so an unexpected
    format still compares sanely instead of silently matching everything.
    """
    if not isinstance(value, str):
        return ""
    lowered = value.strip().lower()
    if "approve" in lowered:
        return "approve"
    if "reject" in lowered:
        return "reject"
    return lowered


def _confirmed_decisions(messages) -> dict[str, dict]:
    """Scan `messages` for (request_human_approval -> immediately followed,
    later in the list, by human_check's "Expert decision recorded..."
    HumanMessage) pairs, keyed by claim_id, keeping the most recent
    confirmation per claim.

    Used to detect when the agent is about to re-request approval for a
    claim_id + recommendation that was already confirmed earlier in this
    same thread, so route_after_tools can skip interrupting again for the
    exact same ask (see its docstring for why this matters).

    Known limitation: this only compares claim_id + (normalized)
    recommendation, NOT risk_score. If the same claim is re-evaluated and
    comes back with a different risk_score but the SAME recommendation
    (e.g. 0.95 -> 0.80, still "reject"), the dedup guard still treats it as
    the already-decided case and will NOT offer a fresh interrupt - even
    though the underlying number changed. Acceptable for this demo (the
    thing the human is actually confirming is the approve/reject call, not
    the exact score), but worth revisiting if the project grows a reason
    to re-confirm on a materially changed risk_score with an unchanged
    recommendation."""
    confirmed: dict[str, dict] = {}
    pending_payload: dict | None = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == REQUEST_APPROVAL_TOOL_NAME:
            pending_payload = _parse_tool_message_content(msg) or {}
        elif (
            pending_payload is not None
            and isinstance(msg, HumanMessage)
            and isinstance(msg.content, str)
            and msg.content.startswith("Expert decision recorded")
        ):
            claim_id = pending_payload.get("claim_id")
            if claim_id is not None:
                confirmed[claim_id] = {
                    "recommendation": pending_payload.get("recommendation"),
                    "decision_message": msg.content,
                }
            pending_payload = None
    return confirmed


def route_after_tools(state: MessagesState) -> str:
    """After the tools node runs, decide where to go next:

    - "agent": no request_human_approval call this round - just loop back
      with the tool result(s).
    - "already_decided": the agent re-requested approval for a claim_id +
      recommendation pair that was ALREADY confirmed earlier in this same
      thread (e.g. it misread the "Expert decision recorded" note and
      asked again, or a complex multi-claim dialog confused it). Skip
      interrupting a second time for the exact same ask - remind it of the
      existing decision instead.
    - "human_check": a genuinely new approval request (new claim_id, or a
      materially different recommendation for the same claim) - pause for
      a real human decision.

    Never merely because check_fraud_risk returned a high score, and never
    because of something called in an earlier turn - see the module
    docstring's "HITL trigger design" note for the turn-scoping rationale.
    """
    messages = state["messages"]
    approval_msg, payload = _current_approval_request(messages)
    if approval_msg is None:
        return "agent"

    prior = _confirmed_decisions(messages[: _message_index(messages, approval_msg)])
    already = prior.get(payload.get("claim_id"))
    if already is not None and (
        _normalize_recommendation(already.get("recommendation"))
        == _normalize_recommendation(payload.get("recommendation"))
    ):
        return "already_decided"
    return "human_check"


def already_decided(state: MessagesState) -> dict:
    """Reached when route_after_tools detects a re-request for a claim_id +
    recommendation pair that's already been confirmed in this thread (see
    its docstring). Never calls interrupt() - just tells the agent about
    the existing decision so it stops asking and uses it in its answer.
    This is what keeps a model that ignores the "don't ask again" system
    prompt instruction from re-triggering an interrupt (and, via main.py's
    /resume now checking for __interrupt__ too, from silently stranding a
    second one) for a decision that's already settled.
    """
    messages = state["messages"]
    approval_msg, payload = _current_approval_request(messages)
    prior = _confirmed_decisions(messages[: _message_index(messages, approval_msg)])
    existing = prior.get(payload.get("claim_id"), {})
    return {
        "messages": [
            HumanMessage(
                content=(
                    f"Note: claim {payload.get('claim_id')} already has a recorded "
                    f"expert decision for a '{payload.get('recommendation')}' "
                    f"recommendation - {existing.get('decision_message', 'no details available')}. "
                    "Do not ask for approval again for this - just reference the "
                    "existing decision in your answer."
                )
            )
        ]
    }


def human_check(state: MessagesState) -> dict:
    """HITL node, reached only via a genuinely new request_human_approval
    call in the current tool-call round (see route_after_tools - a
    re-request for an already-confirmed claim_id + recommendation is
    routed to already_decided instead, never here). Pause with
    `interrupt(...)` and surface the agent's recommendation to the human
    expert.

    On resume, LangGraph re-runs this node from the top: the same lookup
    happens again, but this time `interrupt(...)` returns the human's
    decision (a dict: {"approved": bool, "reason": str}) instead of
    raising, and we record it as a HumanMessage before looping back to the
    agent so it can compose a final answer that reflects the decision.
    """
    _approval_msg, payload = _current_approval_request(state["messages"])

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
builder.add_node("already_decided", already_decided)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {"human_check": "human_check", "agent": "agent", "already_decided": "already_decided"},
)
builder.add_edge("human_check", "agent")
builder.add_edge("already_decided", "agent")

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


_DECISION_MESSAGE_RE = re.compile(r"^Expert decision recorded: approved=(True|False|None), reason=(.*)$")


def _parse_decision_message(content: str) -> dict | None:
    """Parse a human_check()-authored "Expert decision recorded:
    approved=X, reason='Y'" HumanMessage back into {"approved": ..,
    "reason": ..}, or None if `content` isn't one of those messages."""
    m = _DECISION_MESSAGE_RE.match(content)
    if not m:
        return None
    approved_str, reason_repr = m.groups()
    approved = {"True": True, "False": False, "None": None}[approved_str]
    try:
        reason = ast.literal_eval(reason_repr)
    except (ValueError, SyntaxError):
        reason = reason_repr
    return {"approved": approved, "reason": reason}


def extract_tool_outputs(messages) -> list[dict]:
    """Collect every ToolMessage in `messages` (the graph's full message
    history for this thread, as returned by `graph.invoke(...)["messages"]`
    or `graph.get_state(config).values["messages"]`) as
    `[{"tool": <tool name>, "args": <call arguments>, "output": <parsed JSON
    or raw content>}, ...]`, PLUS a synthetic `{"tool": "human_decision", ...}`
    entry for every recorded expert decision in the thread (see below).

    `args` matters as much as `output` for the judge: e.g. check_fraud_risk
    only *returns* {"risk_score": ..., "level": ...} - the claim_id it was
    asked about never appears in its output. Without surfacing the call
    arguments too, the judge sees the agent state "risk_score 0.16 for
    CLM-1002" and flags "CLM-1002" as fabricated, even though it's exactly
    the id the tool was invoked with - a false positive. We recover the
    arguments by matching each ToolMessage.tool_call_id back to the
    `tool_calls` entry with the same id on the preceding AIMessage.

    The human_decision entries matter for the same reason: a human's
    approve/reject decision isn't a tool call at all (it comes from
    human_check()'s interrupt()/resume, recorded as a plain HumanMessage),
    so without surfacing it here too, a LATER /chat answer in the same
    thread that references an already-recorded decision (e.g. "the expert
    approved this claim") would have that fact flagged as fabricated by
    the judge - it has no other way to see it. Previously this was
    injected ad hoc only inside main.py's /resume handler, which fixed
    /resume's own judge call but left every later /chat call in the thread
    still blind to the decision; parsing it here instead means every
    caller (both endpoints) gets it automatically.

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
        if isinstance(msg, ToolMessage):
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
        elif isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            decision = _parse_decision_message(msg.content)
            if decision is not None:
                outputs.append({"tool": "human_decision", "args": {}, "output": decision})

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


def extract_chat_history(messages) -> list[dict]:
    """Reduce a thread's full message list down to what the frontend
    dashboard should display when a claim is reopened: just the real
    back-and-forth between the employee and the agent, as
    `[{"role": "user"|"assistant", "text": "..."}, ...]`.

    Filters out everything that isn't a real turn:
    - ToolMessages (tool calls/results - internal plumbing)
    - AIMessages with no text content (pure tool-call requests, e.g. the
      one that calls check_fraud_risk - nothing for a human to read)
    - the synthetic HumanMessages human_check()/already_decided() inject
      ("Expert decision recorded: ...", "Note: claim ... already has a
      recorded expert decision ...") - these are agent-internal bookkeeping,
      not something the employee actually typed

    Used by GET /chat/{claim_id}/history so switching claims on the
    dashboard and coming back doesn't lose the conversation that's already
    sitting in the checkpointer.
    """
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if not isinstance(content, str) or not content:
                continue
            if content.startswith("Expert decision recorded") or content.startswith("Note: claim"):
                continue
            history.append({"role": "user", "text": content})
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content:
                history.append({"role": "assistant", "text": content})
    return history

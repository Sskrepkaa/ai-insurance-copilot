"""FastAPI app for the AI Insurance Copilot backend.

Implements the full fixed API contract (00_architecture.md): the two read
endpoints (GET /claims, GET /claims/{claim_id}) plus the LangGraph-backed
POST /chat and POST /resume endpoints for the tool-calling + HITL agent
defined in agent.py.

AI Judge integration: /chat and /resume each build their response as a
plain `dict` (`response = {...}`) and then set
`response["judge_score"] = judge.judge_response(...)`, per the
00_architecture.md contract. tool_outputs/question for the judge are
recovered from the graph's message history via agent.extract_tool_outputs()
/ agent.extract_original_question() - see judge.py for the judge itself.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent
import judge
from db import get_connection
from seed_data import ensure_seeded

app = FastAPI(title="AI Insurance Copilot - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Create the schema and seed demo data on first run, so `uvicorn main:app`
    works without a separate `python3 seed_data.py` step."""
    ensure_seeded()


@app.get("/claims")
def list_claims():
    """Return all claims for the dashboard list."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, client, amount, status, type FROM claims"
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    """Return full details for a single claim."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, client, amount, status, type, description FROM claims WHERE id = ?",
            (claim_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="not found")

    claim = dict(row)
    claim["fraud_risk"] = None
    return claim


@app.get("/chat/{claim_id}/history")
def chat_history(claim_id: str):
    """Return the persisted conversation for this claim's agent thread, so
    the frontend can restore it when the claim is reopened instead of
    showing an empty chat (the checkpointer already remembers everything -
    this was purely a frontend gap, not a backend one).

    Response: {"messages": [{"role": "user"|"assistant", "text": ...}],
    "pending_interrupt": null | {"type": "interrupt", "question": ...,
    "thread_id": ...}}. `pending_interrupt` mirrors the exact shape /chat
    and /resume return when they pause, so the frontend can feed it
    through the same Approve/Reject rendering path - it's non-null when
    the thread is sitting at human_check waiting on a decision the
    employee never submitted (they navigated away mid-interrupt).

    A claim_id that was never chatted on has no checkpoint at all, not an
    error: get_state() just returns empty values, which comes back here as
    an empty history and no pending interrupt.

    Known limitation: restored "assistant" messages have no judge_score
    (the judge only ever runs live in /chat and /resume; nothing persists
    it) - see agent.extract_chat_history()'s docstring for why.
    """
    config = {"configurable": {"thread_id": claim_id}}
    state = agent.graph.get_state(config)
    messages = state.values.get("messages", [])

    pending_interrupt = None
    if state.interrupts:
        pending_interrupt = {
            "type": "interrupt",
            "question": agent.extract_interrupt_question(state.interrupts),
            "thread_id": claim_id,
        }

    return {
        "messages": agent.extract_chat_history(messages),
        "pending_interrupt": pending_interrupt,
    }


class ChatRequest(BaseModel):
    claim_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    reason: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Send a message to the claim's agent thread (thread_id = claim_id).

    Returns either an interrupt (the agent paused for a human decision on a
    high fraud-risk claim) or a normal answer.
    """
    config = {"configurable": {"thread_id": req.claim_id}}
    result = agent.graph.invoke(
        {"messages": [{"role": "user", "content": req.message}]}, config=config
    )

    if "__interrupt__" in result:
        return {
            "type": "interrupt",
            "question": agent.extract_interrupt_question(result["__interrupt__"]),
            "thread_id": req.claim_id,
        }

    final_text = agent.extract_final_text(result["messages"])
    tool_outputs = agent.extract_tool_outputs(result["messages"])

    response = {
        "type": "answer",
        "text": final_text,
    }
    # judge-agent slots in here later, e.g.:
    #   response["judge_score"] = judge_response(...)
    response["judge_score"] = judge.judge_response(
        question=req.message, answer=final_text, tool_outputs=tool_outputs
    )
    return response


@app.post("/resume")
def resume(req: ResumeRequest):
    """Resume a paused agent thread with the human expert's decision."""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = agent.graph.invoke(
        agent.Command(resume={"approved": req.approved, "reason": req.reason}),
        config=config,
    )

    if "__interrupt__" in result:
        # Resuming one decision led straight into ANOTHER request for human
        # approval (e.g. the agent re-evaluated and now has a new/updated
        # recommendation). Surface it exactly like /chat does instead of
        # silently falling through to extract_final_text() below, which
        # would report a stale pre-interrupt answer as if this resume had
        # produced a final result - leaving this second interrupt stuck
        # with no client ever having been told to resume it.
        return {
            "type": "interrupt",
            "question": agent.extract_interrupt_question(result["__interrupt__"]),
            "thread_id": req.thread_id,
        }

    final_text = agent.extract_final_text(result["messages"])
    # extract_tool_outputs() also picks up the "Expert decision recorded"
    # HumanMessage human_check() just appended to state and surfaces it as
    # a synthetic {"tool": "human_decision", ...} entry - that's what lets
    # the judge treat "approved"/"reason" as grounded rather than flagging
    # the final answer's mention of the outcome as fabricated, and (unlike
    # injecting it only here) it means later /chat calls in this same
    # thread see it too.
    tool_outputs = agent.extract_tool_outputs(result["messages"])
    # The /resume request body only carries the human's decision, not the
    # original chat message - recover it from the thread's persisted
    # message history (the checkpointer still has it) instead of judging
    # against an empty question, and fold the human's decision in too since
    # it's the context that actually produced this final answer.
    original_question = agent.extract_original_question(result["messages"])
    question_for_judge = (
        f"{original_question}\n\n"
        f"[Human-in-the-loop decision] approved={req.approved}, reason={req.reason!r}"
    )

    response = {
        "type": "answer",
        "text": final_text,
    }
    # judge-agent slots in here later, e.g.:
    #   response["judge_score"] = judge_response(...)
    response["judge_score"] = judge.judge_response(
        question=question_for_judge, answer=final_text, tool_outputs=tool_outputs
    )
    return response


# --- How to run ---------------------------------------------------------
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python3 seed_data.py
# uvicorn main:app --reload --port 8000
#
# Verify: open http://localhost:8000/claims in a browser -> JSON list of claims.
